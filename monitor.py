"""Monitoring engine for the Energy-Aware Offline Copilot.

Records one "run": samples CPU/RAM every second plus battery discharge power
(unplugged) into SQLite, then computes energy for the run.

Energy rule (documented for the report):
  * Unplugged: ground truth = drop in battery RemainingCapacity (mWh) over the
    run. Cross-check = integral of the (slow, smoothed) discharge rate.
    Whole-system measurement (screen, Wi-Fi etc. included) -> compare runs by
    DIFFERENCE under identical conditions.
  * Plugged in: falls back to a utilisation power model (ESTIMATED).
  * If the capacity drop is below MIN_CAP_DROP_WH (counter resolution), the
    rate integral is used instead and the run is noted as low-resolution.

CLI:
  python monitor.py demo [seconds]          record a short live run
  python monitor.py set <key> <value> [note]   store an assumption, e.g.
      python monitor.py set tariff_per_kwh 8.0 "my electricity bill"
      python monitor.py set grid_g_per_kwh <value> "CEA CO2 Baseline v22"
"""
import datetime as dt
import json
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager

import psutil

from init_db import SCHEMA

DB_PATH = "copilot.db"
# PRELIMINARY model constants from probe_battery.py / check_env.py on this
# laptop. Whole-system watts. Refit from real data before relying on them.
MODEL_IDLE_W = 8.0
MODEL_LOAD_W = 23.0
MIN_CAP_DROP_WH = 0.15

# One long-lived PowerShell process (much lighter than spawning one per sample)
PS_LOOP = (
    "while ($true) { "
    "$b = Get-CimInstance -Namespace root\\WMI -ClassName BatteryStatus "
    "| Select-Object -First 1; "
    "[Console]::Out.WriteLine($b.DischargeRate.ToString() + ',' + "
    "$b.RemainingCapacity.ToString()); "
    "[Console]::Out.Flush(); Start-Sleep -Seconds 3 }"
)


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


@contextmanager
def db(path=DB_PATH):
    conn = sqlite3.connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def ensure_schema(path=DB_PATH):
    with db(path) as conn:
        conn.executescript(SCHEMA)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(energy_summary)")}
        for col in ("energy_rate_wh", "energy_cap_wh"):
            if col not in cols:
                conn.execute(f"ALTER TABLE energy_summary ADD COLUMN {col} REAL")


def set_assumption(key, value, note="", path=DB_PATH):
    ensure_schema(path)
    with db(path) as conn:
        conn.execute(
            "INSERT INTO settings(key, value, note) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, note=excluded.note",
            (key, str(value), note))


def get_assumption(key, path=DB_PATH):
    with db(path) as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return float(row[0]) if row else None


class BatteryReader:
    """Background reader: latest discharge rate (W) and capacity (mWh)."""

    def __init__(self):
        self.rate_w = None
        self.cap_mwh = None
        self._proc = None

    def start(self, timeout=15):
        try:
            self._proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-Command", PS_LOOP],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (FileNotFoundError, OSError):
            return False
        threading.Thread(target=self._read, daemon=True).start()
        deadline = time.time() + timeout
        while self.cap_mwh is None and time.time() < deadline:
            time.sleep(0.2)
        return self.cap_mwh is not None

    def _read(self):
        for line in self._proc.stdout:
            try:
                rate, cap = line.strip().split(",")
                self.rate_w = int(rate) / 1000.0
                self.cap_mwh = int(cap)
            except ValueError:
                continue

    def stop(self):
        if self._proc:
            self._proc.terminate()


class RunRecorder:
    def __init__(self, scenario, variant, seed=None, repeat_index=0,
                 params=None, sampling_s=1.0, notes="", db_path=DB_PATH):
        self.scenario, self.variant = scenario, variant
        self.seed, self.repeat_index = seed, repeat_index
        self.params, self.sampling_s = params or {}, sampling_s
        self.notes, self.db_path = notes, db_path
        self.run_id = f"{scenario}-{variant}-{uuid.uuid4().hex[:8]}"
        self._samples = []          # (unix_time, watts)
        self._stop = threading.Event()

    def start(self):
        ensure_schema(self.db_path)
        batt = psutil.sensors_battery()
        self.on_battery = bool(batt and not batt.power_plugged)
        self.battery = BatteryReader()
        self.battery_ok = self.battery.start() if self.on_battery else False
        self.cap_start = self.battery.cap_mwh if self.battery_ok else None
        self.t_start = time.time()
        with db(self.db_path) as conn:
            conn.execute(
                "INSERT INTO runs(run_id, scenario, variant, seed, repeat_index,"
                " workload_params, sampling_s, started_at, on_battery, notes)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (self.run_id, self.scenario, self.variant, self.seed,
                 self.repeat_index, json.dumps(self.params), self.sampling_s,
                 now_iso(), int(self.on_battery), self.notes))
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        mode = "battery (measured)" if self.battery_ok else "model (estimated)"
        print(f"[monitor] run {self.run_id} started, power source: {mode}")
        return self.run_id

    def _loop(self):
        conn = sqlite3.connect(self.db_path)
        psutil.cpu_percent(interval=None)
        next_t = time.time()
        while not self._stop.is_set():
            cpu = psutil.cpu_percent(interval=None)
            ram_mb = psutil.virtual_memory().used / 2**20
            if self.battery_ok and self.battery.rate_w is not None:
                power, src, measured = self.battery.rate_w, "battery", 1
            else:
                power = MODEL_IDLE_W + (MODEL_LOAD_W - MODEL_IDLE_W) * cpu / 100
                src, measured = "model", 0
            t = time.time()
            self._samples.append((t, power))
            conn.execute(
                "INSERT INTO readings(run_id, ts, cpu_pct, ram_mb, power_w,"
                " power_source, is_measured) VALUES (?,?,?,?,?,?,?)",
                (self.run_id, now_iso(), cpu, ram_mb, power, src, measured))
            conn.commit()
            next_t += self.sampling_s
            self._stop.wait(max(0, next_t - time.time()))
        conn.close()

    def stop(self):
        self._stop.set()
        self._thread.join()
        t_end = time.time()
        cap_end = self.battery.cap_mwh if self.battery_ok else None
        if self.battery_ok:
            self.battery.stop()

        pts = self._samples
        e_rate = sum((pts[i][1] + pts[i + 1][1]) / 2 * (pts[i + 1][0] - pts[i][0])
                     for i in range(len(pts) - 1)) / 3600
        e_cap = None
        if self.cap_start is not None and cap_end is not None:
            e_cap = (self.cap_start - cap_end) / 1000.0

        note = ""
        if e_cap is not None and e_cap >= MIN_CAP_DROP_WH:
            energy, source = e_cap, "measured"
        elif self.battery_ok:
            energy, source = e_rate, "measured"
            note = "low-resolution: capacity drop below threshold, used rate integral"
        else:
            energy, source = e_rate, "estimated"
            note = "model-based estimate (plugged in or no battery signal)"

        duration = t_end - self.t_start
        tariff = get_assumption("tariff_per_kwh", self.db_path)
        grid = get_assumption("grid_g_per_kwh", self.db_path)
        cost = energy / 1000 * tariff if tariff is not None else None
        co2 = energy / 1000 * grid if grid is not None else None

        with db(self.db_path) as conn:
            conn.execute(
                "INSERT INTO energy_summary(run_id, duration_s, energy_wh,"
                " energy_source, tariff_per_kwh, grid_g_per_kwh, cost, co2e_g,"
                " energy_rate_wh, energy_cap_wh) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (self.run_id, duration, energy, source, tariff, grid, cost,
                 co2, e_rate, e_cap))
            conn.execute(
                "UPDATE runs SET ended_at=?, notes=TRIM(COALESCE(notes,'')||' '||?)"
                " WHERE run_id=?", (now_iso(), note, self.run_id))
        result = {"run_id": self.run_id, "duration_s": round(duration, 1),
                  "energy_wh": round(energy, 4), "source": source,
                  "rate_integral_wh": round(e_rate, 4),
                  "capacity_drop_wh": None if e_cap is None else round(e_cap, 4),
                  "cost": cost, "co2e_g": co2, "note": note}
        print("[monitor] run finished:", json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "set":
        set_assumption(sys.argv[2], sys.argv[3],
                       sys.argv[4] if len(sys.argv) > 4 else "")
        print("Saved assumption:", sys.argv[2], "=", sys.argv[3])
    else:
        secs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        rec = RunRecorder("live", "live", notes="demo run")
        rec.start()
        time.sleep(secs)
        rec.stop()