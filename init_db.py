"""Create the local SQLite database for the Energy-Aware Offline Copilot.

Usage:  python init_db.py [path_to_db]   (default: copilot.db)
Safe to run repeatedly (CREATE TABLE IF NOT EXISTS).
"""
import sqlite3
import sys

SCHEMA = """
PRAGMA journal_mode = WAL;      -- survives crashes better, allows concurrent reads
PRAGMA foreign_keys = ON;

-- One experiment run (baseline or optimized) or a live monitoring session
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    scenario        TEXT NOT NULL,       -- idle | batching | inference | live
    variant         TEXT NOT NULL,       -- baseline | optimized | live
    seed            INTEGER,
    repeat_index    INTEGER DEFAULT 0,
    workload_params TEXT,                -- JSON
    sampling_s      REAL NOT NULL,       -- sampling interval in seconds
    started_at      TEXT NOT NULL,       -- ISO-8601 UTC
    ended_at        TEXT,
    on_battery      INTEGER,             -- 1/0, runs must be comparable
    notes           TEXT
);

-- Raw resource/power samples
CREATE TABLE IF NOT EXISTS readings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    ts           TEXT NOT NULL,
    cpu_pct      REAL,
    ram_mb       REAL,
    proc_cpu_pct REAL,                   -- workload process only
    power_w      REAL,                   -- value used for energy calc
    power_source TEXT NOT NULL,          -- model | battery | libre_hw | rapl
    is_measured  INTEGER NOT NULL        -- 1 = measured, 0 = model estimate
);
CREATE INDEX IF NOT EXISTS idx_readings_run ON readings(run_id, ts);

-- Per-run totals with formula inputs kept for evidence
CREATE TABLE IF NOT EXISTS energy_summary (
    run_id         TEXT PRIMARY KEY REFERENCES runs(run_id),
    duration_s     REAL NOT NULL,
    energy_wh      REAL NOT NULL,        -- sum(P * dt) / 3600
    energy_source  TEXT NOT NULL,        -- measured | estimated
    tariff_per_kwh REAL,                 -- assumption, stored with the result
    grid_g_per_kwh REAL,                 -- assumption, stored with the result
    cost           REAL,
    co2e_g         REAL,
    task_count     INTEGER,              -- ensures equivalent work compared
    quality_score  REAL,                 -- for inference-settings scenario
    latency_s      REAL
);

-- Recommendation = reason + prediction + verification (the project promise)
CREATE TABLE IF NOT EXISTS recommendations (
    rec_id            TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    scenario          TEXT NOT NULL,
    trigger_metrics   TEXT NOT NULL,     -- JSON of readings that caused it
    explanation       TEXT,              -- SLM text
    verifier_passed   INTEGER,           -- numbers in text match source data
    predicted_saving_wh REAL,
    baseline_run_id   TEXT REFERENCES runs(run_id),
    optimized_run_id  TEXT REFERENCES runs(run_id),
    measured_saving_wh REAL,
    prediction_error_wh REAL,
    evidence_kind     TEXT,              -- measured | model-based
    status            TEXT DEFAULT 'proposed' -- proposed|accepted|rejected|verified
);

CREATE TABLE IF NOT EXISTS reports (
    report_id  TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    body_md    TEXT NOT NULL,
    run_ids    TEXT                      -- JSON list
);

-- Durable action/insight queue (outbox pattern)
CREATE TABLE IF NOT EXISTS outbox (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL UNIQUE, -- prevents duplicate enqueue/processing
    kind            TEXT NOT NULL,        -- insight | action | report | decision
    payload         TEXT NOT NULL,        -- JSON
    status          TEXT NOT NULL DEFAULT 'pending',
                    -- pending | in_flight | done | failed
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT,                 -- backoff schedule
    last_error      TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    delivered_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox(status, next_attempt_at);

-- Assumptions that must appear in every report
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    note  TEXT                            -- source / citation for the value
);
"""

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "copilot.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    print(f"Database ready at {path}")
    print("Tables:", ", ".join(tables))