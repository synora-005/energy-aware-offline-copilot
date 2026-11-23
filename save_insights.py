"""Save SLM explanations to SQLite and queue them for sync.

  python save_insights.py          generate explanations (local SLM), save them,
                                   and put each one in the outbox queue
  python save_insights.py show     print saved insights + queue status
                                   (use this after a restart to prove retention)

Idempotency: the key is  insight:<scenario>:<hash of the facts>.
Same data  -> same key -> re-running never creates a duplicate.
New data   -> new key  -> a new insight is saved and queued.
"""
import hashlib
import json
import sys
import time

from monitor import db, ensure_schema, now_iso
from outbox import enqueue, status


def facts_hash(scenario_facts):
    blob = json.dumps(scenario_facts, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def save(facts, results, force_new=False):
    """Store each explanation in `recommendations` and enqueue it.
    Returns (new_rows, new_queue_items)."""
    ensure_schema()
    by_name = {s["scenario"]: s for s in facts["scenarios"]}
    new_rows = new_queued = 0
    for r in results:
        s = by_name[r["scenario"]]
        key = f"insight:{s['scenario']}:{facts_hash(s)}"
        if force_new:                       # demo: same data, fresh queue item
            key += f":{time.time_ns()}"
        explanation = {k: r.get(k) for k in
                       ("summary", "recommendation", "reason", "evidence")}
        explanation["source"] = r["source"]              # slm | template-fallback
        with db() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO recommendations(rec_id, created_at, scenario,"
                " trigger_metrics, explanation, verifier_passed, evidence_kind, status)"
                " VALUES (?,?,?,?,?,?,?,'proposed')",
                (key, now_iso(), s["scenario"], json.dumps(s),
                 json.dumps(explanation, ensure_ascii=False),
                 1 if r["source"] == "slm" else 0, "measured"))
            new_rows += cur.rowcount
        payload = {"rec_id": key, "scenario": s["scenario"],
                   "explanation": explanation, "metrics": s}
        new_queued += enqueue("insight", payload, key=key)
    return new_rows, new_queued


def show():
    ensure_schema()
    with db() as conn:
        rows = conn.execute(
            "SELECT rec_id, created_at, scenario, explanation, verifier_passed, status"
            " FROM recommendations ORDER BY created_at").fetchall()
        sent = dict(conn.execute(
            "SELECT idempotency_key, status FROM outbox WHERE kind='insight'"
        ).fetchall())
    print(f"{len(rows)} saved insight(s):")
    for rec_id, created, scenario, expl, passed, st in rows:
        e = json.loads(expl)
        print(f"\n  [{scenario}] {rec_id}")
        print(f"    created : {created}   verifier_passed={passed}   status={st}")
        print(f"    source  : {e.get('source')}")
        print(f"    summary : {e.get('summary')}")
        print(f"    queue   : {sent.get(rec_id, 'not queued')}")
    print("\nqueue:", status())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show()
    else:
        from explainer import explain_all      # imported late: needs Ollama
        from facts import build
        force = len(sys.argv) > 1 and sys.argv[1] == "new"
        facts = build()
        results = explain_all(facts)
        rows, queued = save(facts, results, force_new=force)
        print(f"saved {rows} new insight(s), queued {queued} new item(s)")
        show()