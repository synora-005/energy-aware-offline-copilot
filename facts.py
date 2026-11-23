"""Build the grounded facts JSON for the SLM. All numbers are computed here."""
import json
import math
import statistics

from monitor import db, ensure_schema, get_assumption
from harness import t_crit

NOTES = {
    "background": "SYNTHETIC chatty background app (simulated workload); "
                  "energy is measured from the battery",
    "batching": "one process per task vs one batched process; same work "
                "proven by checksums; energy is measured from the battery",
}

# Plain-words actions written by code (no numbers, so the verifier stays strict)
ACTIONS = {
    "background": "Throttle and coalesce the chatty background app's frequent "
                  "wake-ups so the CPU can idle for longer.",
    "batching": "Run the tasks in one batched process so the setup cost is "
                "paid once, then let the CPU idle.",
}


def build():
    ensure_schema()
    tariff = get_assumption("tariff_per_kwh")
    grid = get_assumption("grid_g_per_kwh")
    with db() as conn:
        rows = conn.execute(
            "SELECT r.scenario, r.variant, r.seed, r.workload_params, e.energy_wh"
            " FROM runs r JOIN energy_summary e ON e.run_id = r.run_id"
            " WHERE r.scenario IN ('background','batching')"
            " ORDER BY r.started_at").fetchall()
    facts = {"assumptions": {"tariff_per_kwh": tariff, "grid_g_per_kwh": grid,
                             "energy_method": "battery capacity drop (measured)"},
             "scenarios": []}
    for sc in ("background", "batching"):
        sub = [r for r in rows if r[0] == sc]
        by = {}
        for r in sub:
            by.setdefault(r[2], {}).setdefault(r[1], []).append(r[4])
        pairs = [(statistics.mean(v["baseline"]), statistics.mean(v["optimized"]))
                 for v in by.values() if "baseline" in v and "optimized" in v]
        if len(pairs) < 2:
            continue
        diffs = [b - o for b, o in pairs]
        md = statistics.mean(diffs)
        se = statistics.stdev(diffs) / math.sqrt(len(diffs))
        crit = t_crit(len(diffs) - 1)
        t = md / se if se > 0 else float("inf")
        base = statistics.mean(b for b, _ in pairs)
        opt = statistics.mean(o for _, o in pairs)
        window = json.loads(sub[0][3] or "{}").get("window_s")
        facts["scenarios"].append({
            "scenario": sc,
            "action": ACTIONS[sc],
            "description": NOTES[sc],
            "window_seconds": window,
            "paired_repeats": len(pairs),
            "baseline_wh": round(base, 3),
            "optimized_wh": round(opt, 3),
            "saving_wh": round(md, 3),
            "saving_percent": round(100 * md / base, 1),
            "ci95_wh": [round(md - crit * se, 3), round(md + crit * se, 3)],
            "pairs_favouring_optimized": sum(d > 0 for d in diffs),
            "statistically_significant": abs(t) > crit,
            "cost_saving": round(md / 1000 * tariff, 5) if tariff else None,
            "co2e_saving_g": round(md / 1000 * grid, 4) if grid else None,
        })
    return facts


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))