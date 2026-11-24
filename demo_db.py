"""Export / load the clean demo database.

  python demo_db.py export    (author, once) copy copilot.db -> demo_data/copilot_demo.db
                              keeping measured runs, readings, energy and the
                              newest insight per scenario; dropping test queue
                              items, live-demo runs and superseded insights
  python demo_db.py load      (anyone, after cloning) copy the demo database to
                              copilot.db so the dashboard shows the measured results
  python demo_db.py load --force   replace an existing copilot.db (old one is
                              kept as copilot.db.bak-<time>)
"""
import os
import shutil
import sqlite3
import sys
import time

LIVE = "copilot.db"
DEMO = os.path.join("demo_data", "copilot_demo.db")


def counts(path):
    c = sqlite3.connect(path)
    out = {
        "runs": c.execute("SELECT scenario, COUNT(*) FROM runs GROUP BY 1 ORDER BY 1").fetchall(),
        "insights": c.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0],
        "outbox": c.execute("SELECT status, COUNT(*) FROM outbox GROUP BY 1").fetchall(),
        "readings": c.execute("SELECT COUNT(*) FROM readings").fetchone()[0],
    }
    c.close()
    return out


def export():
    if not os.path.exists(LIVE):
        sys.exit(f"{LIVE} not found. Run this from the project folder.")
    os.makedirs("demo_data", exist_ok=True)
    if os.path.exists(DEMO):
        os.remove(DEMO)
    src = sqlite3.connect(LIVE)
    dst = sqlite3.connect(DEMO)
    src.backup(dst)                      # consistent copy even with WAL
    src.close()

    # 1. live-demo runs are not experiment evidence
    live = "SELECT run_id FROM runs WHERE scenario = 'live'"
    for table in ("readings", "energy_summary"):
        dst.execute(f"DELETE FROM {table} WHERE run_id IN ({live})")
    dst.execute("DELETE FROM runs WHERE scenario = 'live'")

    # 2. keep only the newest insight per scenario
    dst.execute("DELETE FROM recommendations WHERE rec_id NOT IN ("
                " SELECT rec_id FROM recommendations r WHERE created_at = ("
                "  SELECT MAX(created_at) FROM recommendations WHERE scenario = r.scenario))")

    # 3. outbox: drop test items and items of removed insights
    dst.execute("DELETE FROM outbox WHERE idempotency_key LIKE 'test:%'")
    dst.execute("DELETE FROM outbox WHERE kind = 'insight' AND idempotency_key NOT IN"
                " (SELECT rec_id FROM recommendations)")
    dst.commit()
    dst.execute("PRAGMA journal_mode = DELETE")   # single file, no -wal sidecar
    dst.execute("VACUUM")
    dst.close()
    print("exported ->", DEMO, f"({os.path.getsize(DEMO) / 1e6:.2f} MB)")
    for k, v in counts(DEMO).items():
        print(f"  {k}: {v}")


def load(force):
    if not os.path.exists(DEMO):
        sys.exit(f"{DEMO} not found. Are you in the project folder?")
    if os.path.exists(LIVE):
        if not force:
            sys.exit(f"{LIVE} already exists. Use --force to replace it "
                     "(the old one is kept as a backup).")
        bak = f"{LIVE}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(LIVE, bak)
        print("existing database kept as", bak)
    for ext in ("-wal", "-shm"):
        if os.path.exists(LIVE + ext):
            os.remove(LIVE + ext)
    shutil.copyfile(DEMO, LIVE)
    print("loaded demo data into", LIVE)
    for k, v in counts(LIVE).items():
        print(f"  {k}: {v}")
    print("next: streamlit run dashboard.py")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "export":
        export()
    elif cmd == "load":
        load("--force" in sys.argv)
    else:
        print(__doc__)