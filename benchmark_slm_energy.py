"""SLM quantization benchmark v2: latency, tokens/s, quality AND energy.

Does NOT modify explainer.py (it only imports its prompt and verifier), and
does not touch background/batching experiment data. Results are stored as
scenario 'inference' runs in copilot.db plus CSV/JSONL in slm_benchmark_results/.

  python benchmark_slm_energy.py                 3 repetitions, 90 s windows
  python benchmark_slm_energy.py --reps 2        quicker (about 20 min)
  python benchmark_slm_energy.py --allow-plugged model-based estimates only

Method (state this in the report)
  Quality  : each model answers the SAME 6 fact sets (the real scenarios from
             facts.py + synthetic ones) with the production prompt. Pass =
             the production verifier accepts it on the FIRST attempt.
  Energy   : per repetition every variant runs one equal-length window of
             back-to-back explanations; 'idle' (no model loaded) is one more
             variant. Energy = battery capacity drop (whole system, measured).
             Net energy = window energy - idle power x window duration.
             Variant order is rotated each repetition to cancel drift.
  Cost     : Wh per ACCEPTED explanation = net Wh per call / first-attempt
             pass rate (a rejected answer costs a retry in explainer.py).
"""
import argparse
import csv
import json
import random
import statistics
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import psutil

import explainer                      # imported only; never modified
from facts import build
from monitor import RunRecorder, db

OLLAMA = "http://localhost:11434"
MODELS = [("Q2_K", "qwen2.5:1.5b-instruct-q2_K"),
          ("Q4_K_M", "qwen2.5:1.5b"),
          ("Q8_0", "qwen2.5:1.5b-instruct-q8_0")]
OUT = Path(__file__).parent / "slm_benchmark_results"
MIN_BATTERY_PCT = 40
KEYS = ("summary", "recommendation", "reason")


# ------------------------------------------------------------------ Ollama
def post(path, body, timeout=300):
    req = urllib.request.Request(OLLAMA + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def unload_all():
    """Unload every model so each variant starts from the same state."""
    try:
        with urllib.request.urlopen(OLLAMA + "/api/ps", timeout=10) as r:
            loaded = json.loads(r.read()).get("models", [])
        for m in loaded:
            post("/api/generate", {"model": m["name"], "keep_alive": 0}, 60)
    except Exception as e:  # noqa: BLE001
        print("[bench] warning: could not unload models:", e)
    time.sleep(3)


def call(model, slim):
    """One production-style explanation request. Never raises."""
    facts = {"scenarios": [slim]}
    body = {"model": model, "stream": False, "format": "json",
            "keep_alive": "10m",
            "options": {"temperature": 0, "num_predict": 300},
            "messages": [
                {"role": "system", "content": explainer.SYSTEM},
                {"role": "user", "content": "FACTS:\n" + json.dumps(facts)}]}
    t0 = time.perf_counter()
    try:
        data = post("/api/chat", body)
        raw = data.get("message", {}).get("content", "")
        return {"wall_s": time.perf_counter() - t0, "raw": raw,
                "tokens": data.get("eval_count") or 0,
                "eval_s": (data.get("eval_duration") or 0) / 1e9,
                "load_s": (data.get("load_duration") or 0) / 1e9}
    except Exception as e:  # noqa: BLE001
        return {"wall_s": time.perf_counter() - t0, "raw": "", "tokens": 0,
                "eval_s": 0.0, "load_s": 0.0, "error": str(e)}


# ----------------------------------------------------------------- quality
def fact_sets():
    sets = []
    try:
        for s in build()["scenarios"]:
            sets.append({k: s[k] for k in explainer.KEEP})
    except Exception as e:  # noqa: BLE001
        print("[bench] could not build real facts:", e)
    rng = random.Random(7)
    names = ["Background workload optimization", "Batching optimization",
             "Inference settings", "Idle reduction"]
    for i, name in enumerate(names):
        base = round(rng.uniform(0.8, 1.4), 3)
        saving = round(base * rng.uniform(0.05, 0.20), 3)
        full = {"scenario": name,
                "action": "Apply the optimization described by the scenario name.",
                "window_seconds": rng.choice([120, 300, 600]),
                "saving_percent": round(100 * saving / base, 1),
                "statistically_significant": i % 2 == 0}
        sets.append({k: full[k] for k in explainer.KEEP})
    return sets


def grade(raw, slim):
    """Mirror explainer.explain(): accepted = keys present and no verifier
    or wording problems. Also records strict schema validity."""
    try:
        out = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"valid_json": False, "schema_ok": False, "accepted": False,
                "problems": ["not valid JSON"]}
    if not isinstance(out, dict):
        return {"valid_json": True, "schema_ok": False, "accepted": False,
                "problems": ["JSON is not an object"]}
    schema_ok = (set(out) == set(KEYS)
                 and all(isinstance(out[k], str) and out[k].strip() for k in KEYS))
    text = " ".join(str(out.get(k, "")) for k in KEYS)
    facts = {"scenarios": [slim]}
    problems = explainer.verify(text, facts) + explainer.semantic(text, facts)
    accepted = all(k in out for k in KEYS) and not problems
    return {"valid_json": True, "schema_ok": schema_ok, "accepted": accepted,
            "problems": problems}


def quality_phase(label, model, sets, log):
    """Unmeasured: same prompts for every model. Doubles as warm-up."""
    graded = []
    for i, s in enumerate(sets):
        r = call(model, s)
        g = grade(r["raw"], s) if not r.get("error") else {
            "valid_json": False, "schema_ok": False, "accepted": False,
            "problems": [r["error"]]}
        graded.append({**r, **g})
        log({"phase": "quality", "model": label, "set": i,
             "scenario": s["scenario"], **graded[-1]})
        print(f"  [{label}] set {i} ({s['scenario'][:22]}): "
              f"accepted={g['accepted']} schema={g['schema_ok']} "
              f"{r['wall_s']:.1f}s {g['problems'] if g['problems'] else ''}")
    return graded


# ------------------------------------------------------------------ energy
def battery_ok(allow_plugged):
    b = psutil.sensors_battery()
    if b is None or b.power_plugged:
        if allow_plugged:
            return True
        print("Unplug the charger (or pass --allow-plugged for estimates).")
        return False
    if b.percent < MIN_BATTERY_PCT:
        print(f"Battery at {b.percent:.0f}% (< {MIN_BATTERY_PCT}%). "
              "Charge, unplug and rerun.")
        return False
    return True


def energy_window(label, model, sets, window_s, rep, log):
    rec = RunRecorder("inference", label, seed=rep, repeat_index=rep,
                      params={"window_s": window_s, "model": model or "none",
                              "num_predict": 300}, sampling_s=1.0)
    rec.start()
    calls, i, t0 = [], 0, time.time()
    try:
        if model:
            while time.time() - t0 < window_s:
                calls.append(call(model, sets[i % len(sets)]))
                i += 1
        else:
            time.sleep(window_s)
    finally:
        res = rec.stop()
    toks = sum(c["tokens"] for c in calls)
    mean_wall = statistics.mean(c["wall_s"] for c in calls) if calls else None
    with db() as conn:
        conn.execute("UPDATE energy_summary SET task_count=?, latency_s=?"
                     " WHERE run_id=?", (len(calls), mean_wall, rec.run_id))
    row = {"phase": "energy", "variant": label, "rep": rep,
           "run_id": rec.run_id, "calls": len(calls), "tokens": toks,
           "gen_s": sum(c["eval_s"] for c in calls),
           "mean_wall_s": mean_wall, **{k: res[k] for k in (
               "duration_s", "energy_wh", "source")}}
    log(row)
    return row


# ----------------------------------------------------------------- summary
def ms(xs, d=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    sd = statistics.stdev(xs) if len(xs) > 1 else 0
    return f"{statistics.mean(xs):.{d}f} +- {sd:.{d}f}"


def summarize(quality, windows, out_csv):
    print("\n" + "=" * 70)
    print("RESULTS (energy = whole-system battery capacity drop, idle-subtracted)")
    print("=" * 70)
    reps = sorted({w["rep"] for w in windows})
    table = []
    for label, _ in MODELS:
        q = quality.get(label, [])
        accept = (sum(g["accepted"] for g in q) / len(q)) if q else 0
        schema = (sum(g["schema_ok"] for g in q) / len(q)) if q else 0
        j_tok, wh_call, tps, wall = [], [], [], []
        src = set()
        for r in reps:
            idle = next((w for w in windows
                         if w["variant"] == "idle" and w["rep"] == r), None)
            w = next((w for w in windows
                      if w["variant"] == label and w["rep"] == r), None)
            if not idle or not w or not w["calls"] or not w["tokens"]:
                continue
            src.add(w["source"])
            idle_w = idle["energy_wh"] * 3600 / idle["duration_s"]
            net = w["energy_wh"] - idle_w * w["duration_s"] / 3600
            wh_call.append(net / w["calls"])
            j_tok.append(net * 3600 / w["tokens"])
            if w["gen_s"] > 0:
                tps.append(w["tokens"] / w["gen_s"])
            wall.append(w["mean_wall_s"])
        wpa = (statistics.mean(wh_call) / accept
               if wh_call and accept > 0 else None)
        table.append({"model": label, "accept_rate": accept, "schema_rate": schema,
                      "tokens_per_s": ms(tps), "latency_s": ms(wall),
                      "net_wh_per_call": ms(wh_call, 4), "joules_per_token": ms(j_tok, 2),
                      "wh_per_accepted": wpa,
                      "energy_source": ",".join(sorted(src)) or "n/a"})
        print(f"\n{label}  (energy source: {table[-1]['energy_source']})")
        print(f"  quality      : accepted first try {accept:.0%}, "
              f"strict schema {schema:.0%}  (n={len(q)} prompts)")
        print(f"  speed        : {ms(tps)} tokens/s, {ms(wall)} s per explanation")
        print(f"  energy       : {ms(wh_call, 4)} Wh per explanation, "
              f"{ms(j_tok, 2)} J per token  (net of idle)")
        if wpa is not None:
            print(f"  Wh/accepted  : {wpa:.4f}  (net Wh per call / pass rate)")
    ok = [t for t in table if t["wh_per_accepted"] is not None]
    if ok:
        best = min(ok, key=lambda t: t["wh_per_accepted"])
        print(f"\nLowest energy per ACCEPTED explanation: {best['model']} "
              f"({best['wh_per_accepted']:.4f} Wh)")
    else:
        print("\nNo model produced an accepted explanation or energy data.")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]) if table else ["none"])
        w.writeheader()
        w.writerows(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--window", type=float, default=90)
    ap.add_argument("--rest", type=float, default=15)
    ap.add_argument("--allow-plugged", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jf = (OUT / f"energy_bench_{stamp}.jsonl").open("w", encoding="utf-8")

    def log(rec):
        jf.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        jf.flush()

    sets = fact_sets()
    print(f"[bench] {len(sets)} fact sets, {a.reps} repetitions, "
          f"{a.window:.0f}s windows")
    print("Checklist: unplugged, other apps closed, do not touch the laptop.")

    print("\n--- Phase A: quality on identical prompts (unmeasured) ---")
    quality = {}
    for label, model in MODELS:
        unload_all()
        quality[label] = quality_phase(label, model, sets, log)

    print("\n--- Phase B: energy windows ---")
    variants = [("idle", None)] + MODELS
    windows = []
    for rep in range(a.reps):
        k = rep % len(variants)
        for label, model in variants[k:] + variants[:k]:      # rotate order
            if not battery_ok(a.allow_plugged):
                print("[bench] stopping early; finished windows are kept.")
                break
            unload_all()
            if model:
                call(model, sets[0])          # warm-up: load model, unmeasured
            time.sleep(a.rest)
            row = energy_window(label, model, sets, a.window, rep, log)
            windows.append(row)
            print(f"[bench] rep {rep} {label:7}: {row['energy_wh']} Wh "
                  f"({row['source']}), {row['calls']} calls, "
                  f"{row['tokens']} tokens")
        else:
            continue
        break
    unload_all()
    jf.close()
    summarize(quality, windows, OUT / f"energy_bench_{stamp}.csv")
    print(f"\nRaw data: {OUT}")


if __name__ == "__main__":
    main()