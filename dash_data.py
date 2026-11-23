"""Data access for dashboard.py. No Streamlit here, so it can be tested alone."""
import json
import glob
import os
import statistics
import subprocess
import psutil

from monitor import db, ensure_schema
from outbox import is_online, status

SCENARIO_LABEL = {
    "background": "SYNTHETIC background app (simulated workload, energy measured)",
    "batching": "Real batching workload (energy measured)",
}


def online():
    return is_online()


def queue_counts():
    counts = status()
    return {k: counts.get(k, 0) for k in ("pending", "in_flight", "done", "failed")}


def recent_outbox(limit=15):
    ensure_schema()
    with db() as conn:
        rows = conn.execute(
            "SELECT id, idempotency_key, kind, status, attempts, last_error,"
            " created_at, delivered_at FROM outbox ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
    cols = ("id", "key", "kind", "status", "attempts", "last_error",
            "created_at", "delivered_at")
    return [dict(zip(cols, r)) for r in rows]


def insights():
    """Saved SLM explanations, newest first, with their queue status."""
    ensure_schema()
    with db() as conn:
        rows = conn.execute(
            "SELECT rec_id, created_at, scenario, explanation, verifier_passed,"
            " status FROM recommendations ORDER BY created_at DESC").fetchall()
        qstatus = dict(conn.execute(
            "SELECT idempotency_key, status FROM outbox WHERE kind='insight'"
        ).fetchall())
    out = []
    for rec_id, created, scenario, expl, passed, st in rows:
        e = json.loads(expl or "{}")
        out.append({"rec_id": rec_id, "created_at": created, "scenario": scenario,
                    "verifier_passed": bool(passed), "status": st,
                    "queue": qstatus.get(rec_id, "not queued"), **e})
    return out


def run_energies():
    """One row per measured run: scenario, variant, seed, energy_wh, source."""
    ensure_schema()
    with db() as conn:
        rows = conn.execute(
            "SELECT r.scenario, r.variant, r.seed, e.energy_wh, e.energy_source,"
            " e.duration_s FROM runs r JOIN energy_summary e ON e.run_id=r.run_id"
            " WHERE r.scenario IN ('background','batching')"
            " ORDER BY r.started_at").fetchall()
    cols = ("scenario", "variant", "seed", "energy_wh", "source", "duration_s")
    return [dict(zip(cols, r)) for r in rows]

def inference_summary():
    """Scenario 3 table from the latest benchmark file (measured, idle-subtracted)."""
    files = sorted(glob.glob(os.path.join("slm_benchmark_results",
                                          "energy_bench_*.jsonl")))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    quality = [r for r in recs if r.get("phase") == "quality"]
    energy = [r for r in recs if r.get("phase") == "energy"]

    def ms(xs, nd):
        xs = [x for x in xs if x is not None]
        if not xs:
            return "n/a"
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return f"{statistics.mean(xs):.{nd}f} +- {sd:.{nd}f}"

    idle_w = []
    for r in energy:
        if r["variant"] == "idle":
            idle_w.append(r["energy_wh"] * 3600 / r["duration_s"])
    table, copilot = [], (None, None)
    for label in ("Q2_K", "Q4_K_M", "Q8_0"):
        q = [r for r in quality if r["model"] == label]
        if not q:
            continue
        accept = sum(bool(r["accepted"]) for r in q) / len(q)
        wh_call, tps, lat = [], [], []
        for rep in sorted({r["rep"] for r in energy}):
            idle = next((r for r in energy if r["variant"] == "idle"
                         and r["rep"] == rep), None)
            w = next((r for r in energy if r["variant"] == label
                      and r["rep"] == rep), None)
            if not idle or not w or not w["calls"]:
                continue
            iw = idle["energy_wh"] * 3600 / idle["duration_s"]
            net = w["energy_wh"] - iw * w["duration_s"] / 3600
            wh_call.append(net / w["calls"])
            lat.append(w["mean_wall_s"])
            if w["gen_s"] > 0:
                tps.append(w["tokens"] / w["gen_s"])
        wpa = (f"{statistics.mean(wh_call) / accept:.4f}"
               if wh_call and accept > 0 else "n/a (0 usable answers)")
        table.append({"Model (quantization)": label,
                      "Accepted first try": f"{accept:.0%} of {len(q)}",
                      "Tokens/s": ms(tps, 1), "Latency (s)": ms(lat, 1),
                      "Net Wh per explanation": ms(wh_call, 4),
                      "Wh per accepted": wpa})
        if label == "Q4_K_M" and wh_call:
            copilot = (statistics.mean(wh_call),
                       statistics.stdev(wh_call) if len(wh_call) > 1 else 0.0)
    return {"table": table, "file": os.path.basename(files[-1]),
            "copilot_wh": copilot[0], "copilot_sd": copilot[1],
            "idle_w": idle_w, "reps": len(idle_w)}

_BATT_CMD = [
    "powershell", "-NoProfile", "-Command",
    "Get-CimInstance -Namespace root\\WMI -ClassName BatteryStatus | "
    "Select-Object DischargeRate | ConvertTo-Json"]


def live_power():
    """Live readings. Battery watts only when unplugged (measured); else None."""
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    b = psutil.sensors_battery()
    plugged = b is None or b.power_plugged
    watts = None
    if not plugged:
        try:
            out = subprocess.run(_BATT_CMD, capture_output=True, text=True,
                                 timeout=10)
            data = json.loads(out.stdout)
            if isinstance(data, list):
                data = data[0]
            watts = (data.get("DischargeRate") or 0) / 1000
        except Exception:  # noqa: BLE001
            watts = None
    return {"cpu": cpu, "ram": ram, "plugged": plugged, "watts": watts,
            "battery_pct": None if b is None else b.percent}