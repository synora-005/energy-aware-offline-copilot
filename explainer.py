"""Local SLM explainer + number verifier.

The model only PHRASES numbers computed in facts.py. It sees a slim set of
facts (no tiny decimals it could rescale). Every number in its reply is checked
against those facts, plus a few wording checks. On failure we retry with a hint,
then fall back to a code-written template. The code adds an "evidence" line with
the CI, cost, CO2e and the synthetic/measured note, so the model never touches
those figures.

Standard library only (talks to Ollama's HTTP API)."""
import json
import re
import urllib.request

from facts import build

MODEL = "qwen2.5:1.5b"
URL = "http://localhost:11434/api/chat"
EXTRA_OK = {95, 1000}          # 95% confidence level, kWh<->Wh conversion

# Fields the model is allowed to see. Everything else is added by code.
# The model sees NO Wh figures (it kept rescaling 0.16 to 16); code writes them.
KEEP = ("scenario", "action", "window_seconds", "saving_percent",
        "statistically_significant")

SYSTEM = (
    "You are an energy copilot. Reply in ENGLISH only. Return JSON with "
    "exactly three string keys: summary, recommendation, reason. "
    "NUMBER RULES: copy numbers exactly as written in the FACTS JSON. Do NOT "
    "convert units, multiply, divide, round, or count things (never write "
    "'two scenarios'). Describe saving_percent as an ENERGY saving only. "
    "Do not mention CO2, "
    "carbon, cost or confidence intervals. "
    "If statistically_significant is true, say the saving is statistically "
    "distinguishable from noise; if false, say it is only indicative. "
    "Never say 'proven'. "
    "Do NOT write the Wh saving yourself; code adds it. Mention only "
    "saving_percent. The recommendation MUST be based on the 'action' field; "
    "do not invent other actions. saving_percent is the ENERGY saving, not a "
    "change in wake-ups, frequency or idle time. Never write the words true "
    "or false. Pattern for the summary: 'The change cuts energy use by "
    "<saving_percent>%, which is statistically distinguishable from noise.'"
)


# ------------------------------------------------------------- verification
def flatten(x):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        yield float(x)
    elif isinstance(x, dict):
        for v in x.values():
            yield from flatten(v)
    elif isinstance(x, list):
        for v in x:
            yield from flatten(v)


def verify(text, facts):
    """Return numbers in `text` that do not appear in `facts` (unit-tolerant)."""
    allowed = set(flatten(facts)) | EXTRA_OK
    allowed |= {a * 1000 for a in allowed} | {a / 1000 for a in allowed}
    bad = []
    # "CO2e" is a chemical name, not a number
    text = re.sub(r"CO\s*[2\u2082]\s*e?", "CO-two-e", text, flags=re.I)
    for tok in re.findall(r"\d+(?:\.\d+)?", text):
        n = float(tok)
        if not any(abs(n - a) <= max(0.0006, 0.01 * abs(a)) for a in allowed):
            bad.append(tok)
    return bad


def semantic(text, facts):
    """Cheap wording checks the number verifier cannot do."""
    t, problems = text.lower(), []
    if re.search(r"co\s*2[^.]*%", t):
        problems.append("CO2e must be grams, never a percentage")
    if "due to chance" in t or "proven" in t:
        problems.append("say 'distinguishable from noise at 95%', not proven")
    if re.search(r"\b(true|false)\b", t):
        problems.append("do not write true/false; say 'statistically "
                        "distinguishable from noise' or 'only indicative'")
    pct = r"\d+(?:\.\d+)?\s*%"
    for sent in re.split(r"(?<=[.!?])\s+", t):
        if not re.search(pct, sent):
            continue
        if not re.search(r"energy|power", sent):
            problems.append("describe the percentage as an ENERGY saving")
        if (re.search(r"(wake-?up frequency|idle time|cpu idle)[^.]*?" + pct, sent)
                or re.search(pct + r"\s*(reduction|decrease)?\s*in\s+[^,.]*?"
                             r"(wake|idle|frequency|cpu)", sent)):
            problems.append("the percentage is an ENERGY saving, not a change "
                            "in wake-ups, frequency or idle time")
    return problems


# --------------------------------------------------------------- model call
def ask(facts, hint=None):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "FACTS:\n" + json.dumps(facts)}]
    if hint:
        msgs.append({"role": "user", "content":
                     f"Your last answer was rejected: {hint}. Rewrite it using "
                     "only numbers copied exactly from FACTS, with no unit "
                     "conversion, no counting, and no mention of CO2."})
    body = {"model": MODEL, "stream": False, "format": "json",
            "options": {"temperature": 0, "num_predict": 300},
            "messages": msgs}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    out = json.loads(data["message"]["content"])
    return out, data.get("eval_count"), data.get("eval_duration")


def template(facts):
    """Code-written fallback: always correct, never uses the model."""
    parts = []
    for s in facts["scenarios"]:
        sig = ("statistically significant" if s["statistically_significant"]
               else "only indicative")
        parts.append(f"{s['scenario']}: saves {s['saving_wh']} Wh per "
                     f"{s['window_seconds']} s window ({s['saving_percent']}%), "
                     f"{sig}.")
    return {"summary": " ".join(parts),
            "recommendation": "Keep the optimized settings.",
            "reason": "Measured from battery capacity drop over paired repeats."}


def explain(facts, retries=2, fallback_facts=None):
    hint = None
    for attempt in range(1, retries + 2):
        try:
            out, tokens, ns = ask(facts, hint)
            text = " ".join(str(out.get(k, "")) for k in
                            ("summary", "recommendation", "reason"))
            bad = verify(text, facts) + semantic(text, facts)
            if not bad and all(k in out for k in
                               ("summary", "recommendation", "reason")):
                return {"source": "slm", "attempts": attempt, "tokens": tokens,
                        "eval_s": ns / 1e9 if ns else None, **out}
            print(f"[verify] attempt {attempt}: rejected -> {bad}")
            hint = sorted(set(bad))
        except Exception as e:
            print(f"[slm] attempt {attempt} failed: {e}")
    return {"source": "template-fallback", "attempts": retries + 1,
            **template(fallback_facts or facts)}


def explain_all(facts):
    """One model call per scenario; code appends the evidence line."""
    results = []
    for s in facts["scenarios"]:
        slim = {k: s[k] for k in KEEP}          # model sees only these
        r = explain({"scenarios": [slim]}, fallback_facts={"scenarios": [s]})
        r["scenario"] = s["scenario"]
        r["evidence"] = (
            f"Saving: {s['saving_wh']} Wh per {s['window_seconds']} s window "
            f"({s['saving_percent']}%). "
            f"Measured over {s['paired_repeats']} paired repeats; 95% CI of "
            f"saving {s['ci95_wh'][0]} to {s['ci95_wh'][1]} Wh; cost saving "
            f"{s['cost_saving']} (currency units); CO2e saving "
            f"{s['co2e_saving_g']} g. {s['description']}.")
        results.append(r)
    return results


if __name__ == "__main__":
    print(json.dumps(explain_all(build()), indent=2, ensure_ascii=False))

CHAT_SYSTEM = (
    "You are an energy copilot answering a question about the user's measured "
    "results. Reply in ENGLISH in 2-4 sentences, plain text. Use ONLY numbers "
    "copied exactly from the FACTS JSON; never convert, compute or round them. "
    "If the answer is not in FACTS, say you do not have that data. The "
    "background scenario is SYNTHETIC. Never say 'proven'.")


def chat_fallback(facts):
    lines = []
    for s in facts["scenarios"]:
        tag = " (SYNTHETIC)" if s["scenario"] == "background" else ""
        lines.append(
            f"{s['scenario'].capitalize()}{tag}: saves {s['saving_wh']} Wh per "
            f"{s['window_seconds']:.0f} s window ({s['saving_percent']}%), 95% CI "
            f"{s['ci95_wh'][0]} to {s['ci95_wh'][1]} Wh, cost saving "
            f"{s['cost_saving']}, CO2e saving {s['co2e_saving_g']} g.")
    return "I could not produce a verified answer, so here are the measured " \
           "numbers:\n\n" + "\n\n".join(lines)


def chat_answer(question, facts, retries=1):
    hint = None
    for attempt in range(retries + 1):
        msgs = [{"role": "system", "content": CHAT_SYSTEM},
                {"role": "user", "content": "FACTS:\n" + json.dumps(facts)
                 + "\n\nQUESTION: " + question}]
        if hint:
            msgs.append({"role": "user", "content":
                         f"Your last answer was rejected: {hint}. Answer again "
                         "using only numbers copied exactly from FACTS."})
        body = {"model": MODEL, "stream": False,
                "options": {"temperature": 0, "num_predict": 200},
                "messages": msgs}
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                text = json.loads(r.read())["message"]["content"].strip()
            bad = verify(text, facts)
            if "proven" in text.lower():
                bad.append("do not say proven")
            if text and not bad:
                return {"source": "slm", "text": text}
            hint = sorted(set(bad))
        except Exception as e:
            print(f"[chat] attempt {attempt + 1} failed: {e}")
    return {"source": "fallback", "text": chat_fallback(facts)}