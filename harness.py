"""Experiment harness: reproducible baseline vs optimized runs.

  python harness.py run background [--repeats 3] [--window 300]
  python harness.py run batching   [--repeats 3] [--window 300] [--tasks 120]
  python harness.py summary
  python harness.py reset --yes     delete all experiment runs (e.g. after smoke tests)

Design (state in the report):
  * Same seed for the baseline/optimized pair -> identical work.
  * ABBA ordering (baseline first on even repeats, optimized first on odd) so
    slow drift (battery level, temperature) does not favour one variant.
  * Equal-length windows; rest period between runs.
  * Energy per run comes from monitor.RunRecorder (battery capacity drop when
    unplugged = measured; otherwise a model = estimated).
  * 'summary' reports mean +- SD, the difference, and whether it exceeds
    2 standard errors; otherwise it says "not distinguishable from noise".

Scenarios
  background : SYNTHETIC chatty background app (busy 80 ms every 200 ms) vs
               throttled/coalesced (80 ms every 2000 ms). Simulated stand-in.
  batching   : N tasks, each needing setup. Baseline = one process per task
               spread evenly over the window; optimized = one batched process
               that pays setup once, then idles. Checksums prove same work.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time

import psutil

from monitor import (RunRecorder, db, ensure_schema, get_assumption,
                     set_assumption)
from workloads import calibrate

WORKLOADS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workloads.py")
MIN_BATTERY_PCT = 35


def py(*args):
    return subprocess.run([sys.executable, WORKLOADS, *map(str, args)],
                          capture_output=True, text=True)


def units_per_second(recalibrate=False):
    ups = None if recalibrate else get_assumption("units_per_s")
    if ups is None:
        ups = calibrate()
        set_assumption("units_per_s", ups,
                       "calibrated single-core SHA-256 units/s (workloads.py)")
        print(f"[harness] calibrated: {ups} units/s")
    return int(ups)


# ---------------------------------------------------------------- scenarios
def background(variant, a, ups, seed):
    busy, period = (80, 200) if variant == "baseline" else (80, 2000)
    out = py("noise", a.window, busy, period, ups).stdout
    bursts = int(out.split("bursts:")[1]) if "bursts:" in out else None
    return bursts, None


def batching(variant, a, ups, seed):
    setup_units, task_units = int(a.setup_s * ups), int(a.task_s * ups)
    start, lines = time.time(), []
    if variant == "baseline":
        for i in range(a.tasks):
            wait = start + i * a.window / a.tasks - time.time()
            if wait > 0:
                time.sleep(wait)
            lines += py("batch", seed, i, 1, setup_units, task_units).stdout.split()
    else:
        lines += py("batch", seed, 0, a.tasks, setup_units, task_units).stdout.split()
    lines.sort(key=lambda s: int(s.split(":")[0]))
    checksum = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]
    remaining = start + a.window - time.time()
    if remaining > 0:
        time.sleep(remaining)
    elif -remaining > 5:
        print(f"[harness] WARNING: work overran the window by {-remaining:.0f}s "
              "- reduce --tasks or increase --window")
    return len(lines), checksum


SCENARIOS = {"background": background, "batching": batching}


# --------------------------------------------------------------------- run
def battery_ok(a):
    batt = psutil.sensors_battery()
    if batt is None or batt.power_plugged:
        if a.allow_plugged:
            return True
        print("Unplug the charger: measured runs need battery power "
              "(or pass --allow-plugged for model-based estimates).")
        return False
    if batt.percent < MIN_BATTERY_PCT:
        print(f"Battery at {batt.percent:.0f}% (< {MIN_BATTERY_PCT}%). "
              "Charge, unplug and run again; finished runs are saved.")
        return False
    return True


def run(a):
    ensure_schema()
    ups = units_per_second(a.recalibrate)
    print("Checklist: unplugged, other apps closed, Ollama model unloaded "
          "(ollama stop <model>), same brightness/Wi-Fi as before, "
          "do not touch the laptop during runs.")
    if not a.yes:
        input("Press Enter to start...")
    fn = SCENARIOS[a.scenario]
    for r in range(a.repeats):
        seed = a.seed + r
        order = ["baseline", "optimized"] if r % 2 == 0 else ["optimized", "baseline"]
        for variant in order:
            if not battery_ok(a):
                return
            params = {"window_s": a.window, "units_per_s": ups, "seed": seed}
            if a.scenario == "batching":
                params.update(tasks=a.tasks, task_s=a.task_s, setup_s=a.setup_s)
            rec = RunRecorder(a.scenario, variant, seed=seed, repeat_index=r,
                              params=params, sampling_s=1.0)
            rec.start()
            try:
                count, checksum = fn(variant, a, ups, seed)
            finally:
                result = rec.stop()
            params["checksum"] = checksum
            with db() as conn:
                conn.execute("UPDATE energy_summary SET task_count=? WHERE run_id=?",
                             (count, rec.run_id))
                conn.execute("UPDATE runs SET workload_params=? WHERE run_id=?",
                             (json.dumps(params), rec.run_id))
            print(f"[harness] {a.scenario}/{variant} repeat {r}: "
                  f"{result['energy_wh']} Wh ({result['source']}), "
                  f"tasks/bursts={count}, checksum={checksum}")
            time.sleep(a.rest)
    print("[harness] done. Run: python harness.py summary")


# ----------------------------------------------------------------- summary
def mean_sd(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


# two-sided 95% critical values of Student's t (small samples need >> 2.0)
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
          6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t_crit(df):
    if df <= 10:
        return T_CRIT[df]
    return 2.201 if df < 15 else 2.131 if df < 20 else 2.086 if df < 30 else 2.042


def summary(_a):
    ensure_schema()
    tariff, grid = get_assumption("tariff_per_kwh"), get_assumption("grid_g_per_kwh")
    with db() as conn:
        rows = conn.execute(
            "SELECT r.scenario, r.variant, r.seed, r.workload_params, e.energy_wh,"
            " e.energy_source, e.duration_s, e.energy_cap_wh, e.energy_rate_wh"
            " FROM runs r JOIN energy_summary e ON e.run_id = r.run_id"
            " WHERE r.scenario IN ('background','batching')"
            " ORDER BY r.started_at").fetchall()
    for scenario in ("background", "batching"):
        sub = [r for r in rows if r[0] == scenario]
        if not sub:
            continue
        print(f"\n=== {scenario} === (energy source: "
              f"{', '.join(sorted({r[5] for r in sub}))})")
        for v in ("baseline", "optimized"):
            xs = [r[4] for r in sub if r[1] == v]
            if xs:
                m, sd = mean_sd(xs)
                print(f"  {v:9}: n={len(xs)}  {m:.3f} +- {sd:.3f} Wh"
                      f"  (runs: {', '.join(f'{x:.3f}' for x in xs)})")

        # Paired analysis: baseline and optimized share a seed (identical work)
        by_seed = {}
        for r in sub:
            by_seed.setdefault(r[2], {}).setdefault(r[1], []).append(r[4])
        pairs = [(statistics.mean(v["baseline"]), statistics.mean(v["optimized"]))
                 for v in by_seed.values() if "baseline" in v and "optimized" in v]
        diffs = [b - o for b, o in pairs]
        if diffs:
            mean_d = statistics.mean(diffs)
            base_mean = statistics.mean(b for b, _ in pairs)
            print(f"  saving   : {mean_d:.3f} Wh per window "
                  f"({100 * mean_d / base_mean:.1f}% of baseline), "
                  f"{sum(d > 0 for d in diffs)} of {len(diffs)} pairs favour optimized")
            if tariff is not None:
                print(f"             cost = {mean_d:.3f}/1000 x {tariff} = "
                      f"{mean_d / 1000 * tariff:.5f} (currency units)")
            if grid is not None:
                print(f"             CO2e = {mean_d:.3f}/1000 x {grid} g/kWh = "
                      f"{mean_d / 1000 * grid:.4f} g")
        if len(diffs) < 2:
            print("  verdict  : need >= 2 paired repeats to judge noise")
        else:
            se = statistics.stdev(diffs) / math.sqrt(len(diffs))
            df = len(diffs) - 1
            crit = t_crit(df)
            t = mean_d / se if se > 0 else float("inf")
            lo, hi = mean_d - crit * se, mean_d + crit * se
            print(f"  paired t-test: t = {t:.2f} (df={df}, needs > {crit}), "
                  f"95% CI of saving = [{lo:.3f}, {hi:.3f}] Wh")
            if abs(t) > crit:
                print("  verdict  : statistically distinguishable from noise (95%)")
            else:
                print("  verdict  : NOT yet significant - report as indicative only "
                      "and collect more paired repeats")

        chk = [(r[7], r[8]) for r in sub if r[7] is not None and r[7] >= 0.15 and r[8]]
        if chk:
            cap, rate = sum(c for c, _ in chk), sum(x for _, x in chk)
            worst = max(abs(x - c) / c for c, x in chk)
            print(f"  cross-check: battery capacity {cap:.2f} Wh vs rate integral "
                  f"{rate:.2f} Wh in total ({100 * (rate - cap) / cap:+.1f}%); "
                  f"worst single run differs {100 * worst:.0f}%")
        if scenario == "batching":
            cs_by_seed = {}
            for r in sub:
                cs = json.loads(r[3] or "{}").get("checksum")
                cs_by_seed.setdefault(r[2], set()).add(cs)
            ok = all(len(c) == 1 for c in cs_by_seed.values())
            print(f"  equivalent work (matching checksums per seed): "
                  f"{'YES' if ok else 'NO - investigate'}")


def reset(a):
    if not a.yes:
        sys.exit("This deletes all background/batching runs. Add --yes to confirm.")
    ensure_schema()
    cond = "SELECT run_id FROM runs WHERE scenario IN ('background','batching')"
    with db() as conn:
        for table in ("readings", "energy_summary"):
            conn.execute(f"DELETE FROM {table} WHERE run_id IN ({cond})")
        n = conn.execute("DELETE FROM runs WHERE scenario IN "
                         "('background','batching')").rowcount
    print(f"Deleted {n} experiment run(s).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("scenario", choices=SCENARIOS)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--window", type=float, default=300)
    r.add_argument("--tasks", type=int, default=120)
    r.add_argument("--task-s", dest="task_s", type=float, default=1.0)
    r.add_argument("--setup-s", dest="setup_s", type=float, default=1.0)
    r.add_argument("--rest", type=float, default=30)
    r.add_argument("--seed", type=int, default=100)
    r.add_argument("--yes", action="store_true")
    r.add_argument("--allow-plugged", dest="allow_plugged", action="store_true")
    r.add_argument("--recalibrate", action="store_true")
    r.set_defaults(fn=run)
    s = sub.add_parser("summary")
    s.set_defaults(fn=summary)
    x = sub.add_parser("reset")
    x.add_argument("--yes", action="store_true")
    x.set_defaults(fn=reset)
    args = p.parse_args()
    args.fn(args)