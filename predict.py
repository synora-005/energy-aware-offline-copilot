"""Predict -> apply -> verify loop.

  python predict.py predict background     store a prediction BEFORE applying a change
  (apply the change: python harness.py run background --repeats 2 --seed 110)
  python predict.py verify background      compare with the measured saving, recalibrate
  python predict.py verify background --retrospective
                                           use runs that already exist (labelled so)
  python predict.py demo                   predict + verify both scenarios, show recalibration
  python predict.py show                   latest prediction per scenario
  python predict.py recalibrate            recompute the factor (verifications with >= 3 pairs only)
  python predict.py reset --yes            delete predictions and the calibration factor

Method (state in the report)
  Power model : average power P (W) = a + b x mean CPU% , fitted by least squares on
                MEASURED runs (battery capacity drop / duration) of the OTHER scenarios.
                The scenario being predicted is excluded, so the check is out-of-sample.
  Change      : the planned reduction in CPU busy time comes from the workload DESIGN
                (burst period, setup cost x tasks), never from the optimized runs.
  Prediction  : saving_Wh = b x (planned CPU drop in percentage points) x window_s / 3600,
                interval = +- t(95%, n-2) x SE(b) x same factor (slope uncertainty only).
  Verify      : measured saving = mean paired difference (baseline - optimized).
  Recalibrate : factor = mean(measured / raw prediction) over verified predictions;
                it scales later predictions. With few scenarios this is indicative only.
"""
import datetime
import json
import math
import statistics
import sys

import sqlite3

import psutil

from harness import t_crit
from monitor import db, ensure_schema, get_assumption, now_iso, set_assumption
from outbox import enqueue

MIN_PAIRS = 3     # a verification with fewer paired repeats is too noisy to retune the model

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    pred_id           TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    scenario          TEXT NOT NULL,
    change_desc       TEXT,
    model_json        TEXT,
    planned_drop_pts  REAL,
    window_s          REAL,
    raw_saving_wh     REAL,
    calibration       REAL,
    predicted_wh      REAL,
    interval_wh       REAL,
    status            TEXT NOT NULL,      -- predicted | verified
    measured_wh       REAL,
    measured_ci_wh    REAL,               -- half-width of 95% CI of the measured saving
    error_wh          REAL,               -- measured - predicted
    within_interval   INTEGER,
    n_pairs           INTEGER,
    retrospective     INTEGER,
    verified_at       TEXT
)"""


def init():
    ensure_schema()
    with db() as conn:
        conn.execute(SCHEMA)
        try:                                   # databases made by an older version
            conn.execute("ALTER TABLE predictions ADD COLUMN measured_ci_wh REAL")
        except sqlite3.OperationalError:
            pass
        old = conn.execute("SELECT pred_id, scenario, created_at, retrospective, measured_wh, n_pairs"
                           " FROM predictions WHERE status='verified' AND measured_ci_wh IS NULL"
                           ).fetchall()
    for pid, sc, created, retro, m0, n0 in old:      # same data only, never newer runs
        m, n, half = measured_saving(sc, None if retro else created)
        if half is not None and n == n0 and abs(m - m0) < 1e-6:
            with db() as conn:
                conn.execute("UPDATE predictions SET measured_ci_wh=? WHERE pred_id=?", (half, pid))


# ----------------------------------------------------------- power model
def fit(exclude):
    """Least-squares P = a + b*cpu on measured runs of all scenarios but `exclude`."""
    with db() as conn:
        rows = conn.execute(
            "SELECT r.scenario, e.energy_wh, e.duration_s,"
            " (SELECT AVG(cpu_pct) FROM readings d WHERE d.run_id = r.run_id)"
            " FROM runs r JOIN energy_summary e ON e.run_id = r.run_id"
            " WHERE r.scenario NOT IN ('live', ?) AND e.energy_source = 'measured'",
            (exclude,)).fetchall()
    pts = [(c, wh * 3600 / du) for _, wh, du, c in rows if c is not None and du]
    if len(pts) < 4:
        sys.exit("Not enough measured runs to fit the power model.")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    se = math.sqrt(sum(r * r for r in res) / (len(xs) - 2) / sxx)
    r2 = 1 - sum(r * r for r in res) / sum((y - my) ** 2 for y in ys)
    return {"a": a, "b": b, "se_b": se, "r2": r2, "n": len(xs),
            "fitted_without": exclude}


# ------------------------------------------------------------ the change
def logical_cores():
    """Cores of the machine that RECORDED the runs (CPU% is system-wide).
    Stored in the database so demo data stays consistent on another machine."""
    stored = get_assumption("cpu_logical_cores")
    if stored:
        return int(float(stored))
    n = psutil.cpu_count()
    set_assumption("cpu_logical_cores", n, "logical cores of the measurement machine")
    return n


def plan(scenario):
    """Planned CPU-busy reduction from the workload DESIGN."""
    cores = logical_cores()
    with db() as conn:
        row = conn.execute(
            "SELECT workload_params FROM runs WHERE scenario = ? "
            "ORDER BY started_at DESC LIMIT 1", (scenario,)).fetchone()
    p = json.loads(row[0]) if row and row[0] else {}
    window = p.get("window_s", 300)
    if scenario == "background":
        busy, base_period, opt_period = 80, 200, 2000      # mirrors harness.py
        core_fraction = busy / base_period - busy / opt_period
        desc = (f"throttle a chatty background app from {busy} ms every "
                f"{base_period} ms to {busy} ms every {opt_period} ms")
    elif scenario == "batching":
        n, setup_s = p.get("tasks", 120), p.get("setup_s", 1.0)
        core_fraction = (n - 1) * setup_s / window      # setup paid once, not N times
        desc = f"batch {n} tasks into one process (setup paid once, not {n} times)"
    else:
        sys.exit("Known scenarios: background, batching")
    return {"desc": desc, "window_s": window, "cores": cores,
            "drop_pts": core_fraction / cores * 100}


# ---------------------------------------------------------------- predict
def predict(scenario, factor=None):
    init()
    p, m = plan(scenario), fit(scenario)
    scale = p["window_s"] / 3600
    raw = m["b"] * p["drop_pts"] * scale
    half = t_crit(m["n"] - 2) * m["se_b"] * p["drop_pts"] * scale
    if factor is None:
        factor = float(get_assumption("prediction_calibration") or 1.0)
    pred_id = f"pred:{scenario}:{datetime.datetime.now(datetime.timezone.utc):%Y%m%d%H%M%S%f}"
    row = dict(pred_id=pred_id, created_at=now_iso(), scenario=scenario,
               change_desc=p["desc"], model_json=json.dumps({**m, "cores": p["cores"]}),
               planned_drop_pts=p["drop_pts"], window_s=p["window_s"], raw_saving_wh=raw,
               calibration=factor, predicted_wh=raw * factor, interval_wh=half * factor)
    with db() as conn:
        conn.execute("INSERT INTO predictions (" + ",".join(row) + ",status)"
                     " VALUES (" + ",".join("?" * len(row)) + ",'predicted')",
                     list(row.values()))
    enqueue("prediction", row, key=pred_id)
    print(f"[predict] {scenario}: {p['desc']}")
    print(f"  model  : P = {m['a']:.2f} + {m['b']:.3f} x CPU%  "
          f"(R2 {m['r2']:.2f}, n={m['n']} runs, fitted WITHOUT {scenario})")
    print(f"  change : CPU busy drops by {p['drop_pts']:.2f} percentage points "
          f"({p['cores']} logical cores)")
    print(f"  PREDICTED saving: {raw * factor:.3f} +- {half * factor:.3f} Wh per "
          f"{p['window_s']:.0f} s window (calibration x{factor:.2f})")
    return pred_id


# ----------------------------------------------------------------- verify
def measured_saving(scenario, since):
    with db() as conn:
        q = ("SELECT r.seed, r.variant, e.energy_wh FROM runs r JOIN energy_summary e"
             " ON e.run_id = r.run_id WHERE r.scenario = ?")
        args = [scenario]
        if since:
            q += " AND r.started_at > ?"
            args.append(since)
        rows = conn.execute(q, args).fetchall()
    by = {}
    for seed, variant, wh in rows:
        by.setdefault(seed, {}).setdefault(variant, []).append(wh)
    diffs = [statistics.mean(v["baseline"]) - statistics.mean(v["optimized"])
             for v in by.values() if "baseline" in v and "optimized" in v]
    if not diffs:
        return None, 0, None
    half = (t_crit(len(diffs) - 1) * statistics.stdev(diffs) / math.sqrt(len(diffs))
            if len(diffs) > 1 else None)
    return statistics.mean(diffs), len(diffs), half


def recalibrate():
    with db() as conn:
        rows = conn.execute(
            "SELECT scenario, measured_wh, raw_saving_wh FROM predictions "
            "WHERE status = 'verified' AND n_pairs >= ? ORDER BY verified_at",
            (MIN_PAIRS,)).fetchall()
    latest = {s: (m, r) for s, m, r in rows if r and r > 0}     # newest per scenario
    ratios = [m / r for m, r in latest.values()]
    if not ratios:
        print(f"  (no verification with >= {MIN_PAIRS} paired repeats yet: calibration unchanged)")
        return None
    factor = statistics.mean(ratios)
    set_assumption("prediction_calibration", round(factor, 4),
                   f"mean(measured/raw predicted saving) over {len(ratios)} verified scenario(s)")
    return factor, len(ratios)


def verify(scenario, retrospective=False):
    init()
    with db() as conn:
        row = conn.execute(
            "SELECT pred_id, created_at, predicted_wh, interval_wh FROM predictions"
            " WHERE scenario = ? AND status = 'predicted' ORDER BY created_at DESC LIMIT 1",
            (scenario,)).fetchone()
    if not row:
        sys.exit(f"No open prediction for {scenario}. Run: python predict.py predict {scenario}")
    pred_id, created, pred, half = row
    meas, n, mci = measured_saving(scenario, created)
    retro = 0
    if meas is None:
        if not retrospective:
            sys.exit("No paired runs after the prediction yet. Apply the change first:\n"
                     f"  python harness.py run {scenario} --repeats 2 --seed 110\n"
                     "or use --retrospective to compare with runs that already exist.")
        meas, n, mci = measured_saving(scenario, None)
        retro = 1
    err = meas - pred
    inside = int(abs(err) <= half)
    with db() as conn:
        conn.execute("UPDATE predictions SET status='verified', measured_wh=?, error_wh=?,"
                     " within_interval=?, n_pairs=?, retrospective=?, verified_at=?,"
                     " measured_ci_wh=? WHERE pred_id=?",
                     (meas, err, inside, n, retro, now_iso(), mci, pred_id))
    enqueue("verification", {"pred_id": pred_id, "scenario": scenario, "predicted_wh": pred,
                             "measured_wh": meas, "error_wh": err, "within_interval": inside,
                             "n_pairs": n, "retrospective": retro}, key=f"verify:{pred_id}")
    mtxt = f" +- {mci:.3f}" if mci is not None else ""
    print(f"[verify] {scenario}: predicted {pred:.3f} +- {half:.3f} Wh, "
          f"measured {meas:.3f}{mtxt} Wh ({n} paired repeats"
          f"{', RETROSPECTIVE: runs already existed' if retro else ''})")
    print(f"  error {err:+.3f} Wh ({100 * err / meas:+.0f}% of measured) - "
          f"{'inside' if inside else 'OUTSIDE'} the prediction interval")
    if mci is not None:
        print(f"  measured 95% CI = [{meas - mci:.3f}, {meas + mci:.3f}] Wh: prediction is "
              f"{'INSIDE' if abs(err) <= mci else 'OUTSIDE'} it "
              f"(with few repeats the measurement itself is uncertain)")
    cal = recalibrate()
    if cal:
        print(f"  recalibrated: factor x{cal[0]:.2f} from {cal[1]} verified scenario(s)")
    return pred_id


# ------------------------------------------------------------- show/demo
def show():
    init()
    with db() as conn:
        rows = conn.execute(
            "SELECT scenario, status, predicted_wh, interval_wh, measured_wh, error_wh,"
            " within_interval, calibration, retrospective, n_pairs, measured_ci_wh"
            " FROM predictions p WHERE created_at ="
            " (SELECT MAX(created_at) FROM predictions WHERE scenario = p.scenario)").fetchall()
    if not rows:
        print("No predictions yet.")
    for s, st, p, h, m, e, w, c, r, n, ci in rows:
        line = f"{s:11} {st:9} predicted {p:.3f} +- {h:.3f} Wh (x{c:.2f})"
        if st == "verified":
            ci_t = f" +- {ci:.3f}" if ci is not None else ""
            line += (f" | measured {m:.3f}{ci_t} ({n} pairs) | error {e:+.3f} | "
                     f"{'in' if w else 'OUT of'} interval")
            line += " | retrospective" if r else ""
        print(line)


def demo():
    scenarios = ("background", "batching")
    print("=== 1. Predict (uncalibrated, fitted without the scenario) ===")
    for s in scenarios:
        predict(s, factor=1.0)
    print("\n=== 2. Verify against measured paired runs ===")
    for s in scenarios:
        verify(s, retrospective=True)
    print("\n=== 3. Does recalibration help? (learn on ONE scenario, test on the other) ===")
    with db() as conn:
        rows = {s: (raw, m) for s, raw, m in conn.execute(
            "SELECT scenario, raw_saving_wh, measured_wh FROM predictions p "
            "WHERE status='verified' AND created_at = (SELECT MAX(created_at) "
            "FROM predictions WHERE scenario = p.scenario)").fetchall()}
    for s in scenarios:
        other = [x for x in scenarios if x != s][0]
        f = rows[other][1] / rows[other][0]
        raw, meas = rows[s]
        print(f"  {s:11} raw error {meas - raw:+.3f} Wh -> after x{f:.2f} learned "
              f"from {other}: error {meas - raw * f:+.3f} Wh")
    print("  (2 scenarios only: indicative, not proof)")


def reset():
    if "--yes" not in sys.argv:
        sys.exit("This deletes all predictions. Add --yes to confirm.")
    init()
    with db() as conn:
        conn.execute("DELETE FROM predictions")
        conn.execute("DELETE FROM settings WHERE key = 'prediction_calibration'")
    print("predictions and calibration cleared")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "predict" and len(sys.argv) > 2:
        predict(sys.argv[2])
    elif cmd == "verify" and len(sys.argv) > 2:
        verify(sys.argv[2], "--retrospective" in sys.argv)
    elif cmd == "demo":
        demo()
    elif cmd == "show":
        show()
    elif cmd == "recalibrate":
        init()
        cal = recalibrate()
        print(f"calibration factor x{cal[0]:.2f} from {cal[1]} scenario(s)" if cal else "unchanged")
    elif cmd == "reset":
        reset()
    else:
        print(__doc__)