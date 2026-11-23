"""Net savings = optimization saving minus the copilot's own energy per explanation."""
import glob
import json
import os
import statistics

from facts import build

LABEL = "Q4_K_M"          # production model: qwen2.5:1.5b


def latest_rows():
    files = sorted(glob.glob(os.path.join("slm_benchmark_results", "energy_bench_*.jsonl")))
    if not files:
        raise SystemExit("No benchmark file in slm_benchmark_results/")
    with open(files[-1], encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return files[-1], [r for r in rows if r.get("phase") == "energy"]


def net_wh_per_call(rows, label):
    out = []
    for rep in sorted({r["rep"] for r in rows}):
        idle = next((r for r in rows if r["variant"] == "idle" and r["rep"] == rep), None)
        w = next((r for r in rows if r["variant"] == label and r["rep"] == rep), None)
        if not idle or not w or not w["calls"]:
            continue
        idle_w = idle["energy_wh"] * 3600 / idle["duration_s"]
        net = w["energy_wh"] - idle_w * w["duration_s"] / 3600
        out.append(net / w["calls"])
    return out


path, rows = latest_rows()
xs = net_wh_per_call(rows, LABEL)
m = statistics.mean(xs)
sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
print(f"Copilot cost ({LABEL}, n={len(xs)}, {os.path.basename(path)}): "
      f"{m:.4f} +- {sd:.4f} Wh per explanation (whole system, net of idle, measured)")
for s in build()["scenarios"]:
    save, win = s["saving_wh"], s["window_seconds"]
    print(f"\n{s['scenario']}:")
    print(f"  gross saving per {win} s window : {save} Wh")
    print(f"  one explanation costs          : {m:.4f} Wh ({100 * m / save:.1f}% of it)")
    print(f"  net, one explanation per window: {save - m:.3f} Wh")
    print(f"  break-even                     : {m / save * win:.0f} s "
          f"(worst case, mean+1SD: {(m + sd) / save * win:.0f} s)")
print("\nIndicative only: 3 repeats, large idle noise. Background scenario is synthetic.")