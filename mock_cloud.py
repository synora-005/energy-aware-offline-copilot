"""Mock cloud receiver for the sync demo (standard library only).

Run:  python mock_cloud.py [--reset]      (listens on 127.0.0.1:8000)

Endpoints
  POST /sync            {idempotency_key, kind, payload} -> processed | duplicate
  GET  /health          liveness probe used by the client to detect connectivity
  GET  /stats           {processed, duplicates_received}
  POST /admin/fail?n=3  make the next n /sync calls return HTTP 503 (flaky net)

Stopping this program = "network down" for the client. Starting it again =
"connectivity returned". The receiver stores every idempotency key, so a
retried delivery is acknowledged but NEVER processed twice.
"""
import json
import os
import sqlite3
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

CLOUD_DB = "cloud.db"
PORT = 8000
_fail_lock = threading.Lock()
_fail_remaining = 0


def init_db():
    with sqlite3.connect(CLOUD_DB) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS processed ("
            " idempotency_key TEXT PRIMARY KEY, kind TEXT, payload TEXT,"
            " received_at REAL, duplicate_count INTEGER DEFAULT 0)")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):   # silence default access log
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, {"ok": True})
        elif path == "/stats":
            conn = sqlite3.connect(CLOUD_DB)
            n, dups = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(duplicate_count),0) FROM processed"
            ).fetchone()
            conn.close()
            self._send(200, {"processed": n, "duplicates_received": dups})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        global _fail_remaining
        parsed = urlparse(self.path)
        if parsed.path == "/admin/fail":
            n = int(parse_qs(parsed.query).get("n", ["1"])[0])
            with _fail_lock:
                _fail_remaining = n
            print(f"[cloud] next {n} /sync calls will fail with 503")
            return self._send(200, {"will_fail": n})
        if parsed.path != "/sync":
            return self._send(404, {"error": "not found"})

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length).decode())
            key = body["idempotency_key"]
        except (ValueError, KeyError):
            return self._send(400, {"error": "bad request"})

        with _fail_lock:
            if _fail_remaining > 0:
                _fail_remaining -= 1
                print(f"[cloud] injected failure for {key}")
                return self._send(503, {"error": "injected failure"})

        conn = sqlite3.connect(CLOUD_DB)
        try:
            cur = conn.execute(
                "INSERT OR IGNORE INTO processed(idempotency_key, kind, payload,"
                " received_at) VALUES (?,?,?,?)",
                (key, body.get("kind"), json.dumps(body.get("payload")), time.time()))
            if cur.rowcount == 0:
                conn.execute("UPDATE processed SET duplicate_count="
                             "duplicate_count+1 WHERE idempotency_key=?", (key,))
                status = "duplicate"
            else:
                status = "processed"
            conn.commit()
        finally:
            conn.close()
        print(f"[cloud] {status}: {key}")
        self._send(200, {"status": status})


if __name__ == "__main__":
    if "--reset" in sys.argv and os.path.exists(CLOUD_DB):
        os.remove(CLOUD_DB)
    init_db()
    print(f"[cloud] listening on http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    try:
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n[cloud] stopped")