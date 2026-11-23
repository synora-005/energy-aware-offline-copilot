"""Durable outbox queue + sync worker.

Guarantees (state these in the report):
  * Persistent: items live in SQLite, so they survive restarts and crashes.
  * At-least-once delivery: a crash between "sent" and "marked done" causes a
    re-send after restart. The receiver dedupes by idempotency key, so the
    EFFECT is processed once (effectively-once, not magic exactly-once).
  * No duplicate enqueue: the idempotency key is UNIQUE in the outbox.
  * Offline-safe: when the receiver is unreachable nothing is lost and
    attempts are not burned; delivery resumes automatically on reconnect.
  * Retry with capped exponential backoff for transient failures (network,
    HTTP 5xx). HTTP 4xx is a permanent failure and is marked 'failed'.

CLI:
  python outbox.py enqueue [N]            add N test insights (keys test:0..N-1)
  python outbox.py worker [--crash-after-send]   run the sync worker
  python outbox.py status                 counts by status
"""
import datetime as dt
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

from monitor import DB_PATH, db, ensure_schema, now_iso

CLOUD_URL = os.environ.get("CLOUD_URL", "http://127.0.0.1:8000")
MAX_BACKOFF_S = 60
SEND_TIMEOUT_S = 3


def _in_seconds(seconds):
    t = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=seconds)
    return t.isoformat(timespec="seconds")


def enqueue(kind, payload, key=None, path=DB_PATH):
    """Add an item. Returns True if inserted, False if the key already exists."""
    ensure_schema(path)
    key = key or f"{kind}:{uuid.uuid4().hex}"
    now = now_iso()
    with db(path) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO outbox(idempotency_key, kind, payload, status,"
            " attempts, next_attempt_at, created_at, updated_at)"
            " VALUES (?,?,?,'pending',0,?,?,?)",
            (key, kind, json.dumps(payload), now, now, now))
        return cur.rowcount == 1


def recover_in_flight(path=DB_PATH):
    """After a crash, items stuck 'in_flight' go back to 'pending'."""
    ensure_schema(path)
    now = now_iso()
    with db(path) as conn:
        cur = conn.execute(
            "UPDATE outbox SET status='pending', next_attempt_at=?, updated_at=?"
            " WHERE status='in_flight'", (now, now))
        return cur.rowcount


def claim_due(limit=10, path=DB_PATH):
    now = now_iso()
    with db(path) as conn:
        conn.execute("BEGIN IMMEDIATE")      # claim atomically
        rows = conn.execute(
            "SELECT id, idempotency_key, kind, payload, attempts FROM outbox"
            " WHERE status='pending' AND (next_attempt_at IS NULL"
            " OR next_attempt_at <= ?) ORDER BY id LIMIT ?", (now, limit)).fetchall()
        for row in rows:
            conn.execute("UPDATE outbox SET status='in_flight',"
                         " attempts=attempts+1, updated_at=? WHERE id=?",
                         (now, row[0]))
        return rows


def mark_done(item_id, path=DB_PATH):
    now = now_iso()
    with db(path) as conn:
        conn.execute("UPDATE outbox SET status='done', delivered_at=?,"
                     " updated_at=?, last_error=NULL WHERE id=?",
                     (now, now, item_id))


def mark_retry(item_id, attempts, error, path=DB_PATH):
    delay = min(MAX_BACKOFF_S, 2 ** attempts) + random.uniform(0, 1)
    with db(path) as conn:
        conn.execute("UPDATE outbox SET status='pending', next_attempt_at=?,"
                     " last_error=?, updated_at=? WHERE id=?",
                     (_in_seconds(delay), error, now_iso(), item_id))
    return delay


def mark_failed(item_id, error, path=DB_PATH):
    with db(path) as conn:
        conn.execute("UPDATE outbox SET status='failed', last_error=?,"
                     " updated_at=? WHERE id=?", (error, now_iso(), item_id))


def status(path=DB_PATH):
    ensure_schema(path)
    with db(path) as conn:
        return dict(conn.execute(
            "SELECT status, COUNT(*) FROM outbox GROUP BY status").fetchall())


def is_online():
    try:
        with urllib.request.urlopen(CLOUD_URL + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def send(key, kind, payload_json):
    """Returns (outcome, info): outcome in ok | retry | permanent."""
    body = json.dumps({"idempotency_key": key, "kind": kind,
                       "payload": json.loads(payload_json)}).encode()
    req = urllib.request.Request(
        CLOUD_URL + "/sync", data=body, method="POST",
        headers={"Content-Type": "application/json", "Idempotency-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=SEND_TIMEOUT_S) as r:
            return "ok", json.loads(r.read().decode()).get("status")
    except urllib.error.HTTPError as e:
        return ("permanent" if 400 <= e.code < 500 else "retry"), f"HTTP {e.code}"
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return "retry", str(e)


class SyncWorker:
    def __init__(self, path=DB_PATH, poll_s=1.0, crash_after_send=False):
        self.path, self.poll_s = path, poll_s
        self.crash_after_send = crash_after_send
        self.online = None
        self._stop = threading.Event()

    def run_once(self):
        online = is_online()
        if online != self.online:
            print(f"[sync] {'ONLINE' if online else 'OFFLINE'} - "
                  f"queue: {status(self.path)}")
            self.online = online
        if not online:
            return 0                  # keep items pending, do not burn attempts
        sent = 0
        for item_id, key, kind, payload, attempts in claim_due(path=self.path):
            outcome, info = send(key, kind, payload)
            if outcome == "ok":
                if self.crash_after_send:
                    print(f"[sync] CHAOS: crashing after sending {key} "
                          "but before marking it done")
                    os._exit(1)
                mark_done(item_id, self.path)
                sent += 1
                print(f"[sync] delivered {key} ({info})")
            elif outcome == "retry":
                delay = mark_retry(item_id, attempts + 1, info, self.path)
                print(f"[sync] retry {key} in {delay:.0f}s ({info})")
            else:
                mark_failed(item_id, info, self.path)
                print(f"[sync] FAILED permanently {key} ({info})")
        return sent

    def loop(self):
        recovered = recover_in_flight(self.path)
        if recovered:
            print(f"[sync] recovered {recovered} in-flight item(s) after restart")
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.poll_s)

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self):
        self._stop.set()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "enqueue":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        added = sum(enqueue("insight", {"n": i, "text": f"test insight {i}"},
                            key=f"test:{i}") for i in range(n))
        print(f"inserted {added}, ignored {n - added} duplicate key(s)")
        print("queue:", status())
    elif cmd == "worker":
        worker = SyncWorker(crash_after_send="--crash-after-send" in sys.argv)
        print(f"[sync] worker started -> {CLOUD_URL}  (Ctrl+C to stop)")
        try:
            worker.loop()
        except KeyboardInterrupt:
            print("\n[sync] stopped; queue:", status())
    else:
        print("queue:", status())