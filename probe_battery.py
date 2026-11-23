"""Probe how the Windows battery counters behave (idle vs CPU load).

Usage:  python probe_battery.py [phase_seconds]     (default 90)
Run UNPLUGGED, battery above ~40%, with other apps closed and the model
unloaded (ollama stop qwen2.5:1.5b). Do not use the laptop while it runs.

Answers: does DischargeRate update? Does it react to load? Does the
integrated rate agree with the drop in RemainingCapacity?
Writes probe_battery.csv for later use as evidence.
"""
import csv
import json
import multiprocessing as mp
import statistics
import subprocess
import sys
import time

import psutil

CMD = [
    "powershell", "-NoProfile", "-Command",
    "Get-CimInstance -Namespace root\\WMI -ClassName BatteryStatus | "
    "Select-Object DischargeRate,RemainingCapacity | ConvertTo-Json",
]
INTERVAL_S = 2.0


def read_battery():
    out = subprocess.run(CMD, capture_output=True, text=True, timeout=20)
    data = json.loads(out.stdout)
    if isinstance(data, list):
        data = data[0]
    return data.get("DischargeRate") or 0, data.get("RemainingCapacity") or 0


def burn(stop_at):
    while time.time() < stop_at:
        pass


def run_phase(name, seconds, load, writer):
    end = time.time() + seconds
    procs = []
    if load:
        procs = [mp.Process(target=burn, args=(end,))
                 for _ in range(psutil.cpu_count())]
        for p in procs:
            p.start()
    psutil.cpu_percent(interval=None)  # prime the counter
    rows, t0 = [], time.time()
    while time.time() < end:
        t = time.time()
        rate, cap = read_battery()
        cpu = psutil.cpu_percent(interval=None)
        rows.append((t - t0, rate, cap, cpu))
        writer.writerow([name, f"{t - t0:.1f}", rate, cap, cpu])
        time.sleep(max(0, INTERVAL_S - (time.time() - t)))
    for p in procs:
        p.join()

    rates = [r[1] for r in rows]
    duration = rows[-1][0] - rows[0][0]
    mean_w = statistics.mean(rates) / 1000
    cap_drop_wh = (rows[0][2] - rows[-1][2]) / 1000
    integrated_wh = mean_w * duration / 3600
    print(f"\n[{name}] {len(rows)} samples over {duration:.0f}s")
    print(f"  mean CPU: {statistics.mean(r[3] for r in rows):.0f}%")
    print(f"  discharge rate: mean {mean_w:.2f} W | "
          f"distinct values: {len(set(rates))} | "
          f"min/max {min(rates) / 1000:.2f}/{max(rates) / 1000:.2f} W")
    print(f"  energy from rate x time: {integrated_wh:.4f} Wh")
    print(f"  energy from capacity drop: {cap_drop_wh:.4f} Wh "
          f"(capacity steps seen: {len(set(r[2] for r in rows))})")
    return mean_w, len(set(rates))


def main():
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    batt = psutil.sensors_battery()
    if batt is None or batt.power_plugged:
        sys.exit("Unplug the charger first (battery must be discharging).")
    if batt.percent < 30:
        sys.exit("Battery below 30% - charge it first, then unplug again.")

    with open("probe_battery.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "t_s", "discharge_mw", "capacity_mwh", "cpu_pct"])
        idle_w, idle_distinct = run_phase("idle", seconds, False, writer)
        load_w, load_distinct = run_phase("load", seconds, True, writer)

    print("\n=== INTERPRETATION ===")
    print(f"Load vs idle power: {load_w:.2f} W vs {idle_w:.2f} W "
          f"(+{load_w - idle_w:.2f} W)")
    if idle_distinct <= 3 and load_distinct <= 3:
        print("Rate barely changes -> heavily smoothed or slow-updating; "
              "prefer capacity drop over long runs.")
    else:
        print("Rate updates during the run -> usable as a time series.")
    print("Saved samples to probe_battery.csv. Paste this output back.")


if __name__ == "__main__":
    main()