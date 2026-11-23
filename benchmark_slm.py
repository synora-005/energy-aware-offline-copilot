
"""
Controlled benchmark for Qwen2.5 1.5B quantizations.
Uses Ollama's local API. Does not modify existing project data.
"""

import csv
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

MODELS = [
    ("Q2_K", "qwen2.5:1.5b-instruct-q2_K"),
    ("Q4_K_M", "qwen2.5:1.5b"),
    ("Q8_0", "qwen2.5:1.5b-instruct-q8_0"),
]

URL = "http://localhost:11434/api/chat"
OUTPUT_DIR = Path(__file__).parent / "slm_benchmark_results"

SYSTEM_PROMPT = """
You are an energy efficiency copilot.

Return ONLY valid JSON with exactly these three keys:
summary, recommendation, reason.

All three values MUST be strings.
Do not add any extra keys.

STRICT FACTUAL RULES:
1. Copy numerical values exactly from the supplied FACTS.
2. The energy saving is 0.160 Wh, NOT 0.160 kWh.
3. The energy saving percentage is 14.5%.
4. Never change, convert, round, or reinterpret units.
5. Do not invent measurements or additional numerical values.
6. If statistically_significant is true, describe the result as
   statistically distinguishable from noise.
7. Make a practical recommendation based on the supplied facts.

Return only the JSON object. No markdown or additional text.
"""

USER_PROMPT = """
FACTS:
{
  "scenario": "Background workload optimization",
  "window_seconds": 300,
  "baseline_wh": 1.100,
  "optimized_wh": 0.940,
  "saving_wh": 0.160,
  "saving_percent": 14.5,
  "statistically_significant": true
}

Explain these results clearly to a user monitoring energy consumption.
Return the required JSON.
"""

OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "num_predict": 160,
    "num_ctx": 2048,
}

REQUIRED_KEYS = {"summary", "recommendation", "reason"}

def check_factual_accuracy(parsed):
    if not isinstance(parsed, dict):
        return False, ["Invalid response structure"]

    text = " ".join(
        str(parsed.get(key, ""))
        for key in REQUIRED_KEYS
    ).lower()

    problems = []

    if "0.160" not in text:
        problems.append("Missing correct saving value: 0.160")

    if "14.5" not in text:
        problems.append("Missing correct saving percentage: 14.5")

    if "kwh" in text:
        problems.append("Incorrect unit: kWh instead of Wh")

    if "wh" not in text:
        problems.append("Missing Wh unit")

    if "0.060" in text:
        problems.append("Incorrect saving value: 0.060")

    return len(problems) == 0, problems

def call_model(model_name):
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "stream": False,
        "format": "json",
        "options": OPTIONS,
        "keep_alive": "5m",
    }

    request = urllib.request.Request(
        URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))

    wall_seconds = time.perf_counter() - start

    message = result.get("message", {}).get("content", "")
    try:
        parsed = json.loads(message)
        valid_json = isinstance(parsed, dict)
        valid_schema = (
            valid_json
            and set(parsed.keys()) == REQUIRED_KEYS
            and all(isinstance(parsed[k], str) for k in REQUIRED_KEYS)
            and all(parsed[k].strip() for k in REQUIRED_KEYS)
        )
    except (json.JSONDecodeError, TypeError):
        parsed = {}
        valid_json = False
        valid_schema = False

    if valid_schema:
        factual_valid, factual_problems = check_factual_accuracy(parsed)
    else:
        factual_valid = False
        factual_problems = ["Schema validation failed"]

    eval_count = result.get("eval_count", 0) or 0
    eval_duration = result.get("eval_duration", 0) or 0
    prompt_count = result.get("prompt_eval_count", 0) or 0

    eval_seconds = eval_duration / 1e9
    tokens_per_second = (
        eval_count / eval_seconds if eval_seconds > 0 else None
    )

    return {
        "wall_seconds": round(wall_seconds, 4),
        "total_duration_s": round(
            (result.get("total_duration", 0) or 0) / 1e9, 4
        ),
        "load_duration_s": round(
            (result.get("load_duration", 0) or 0) / 1e9, 4
        ),
        "prompt_eval_duration_s": round(
            (result.get("prompt_eval_duration", 0) or 0) / 1e9, 4
        ),
        "eval_duration_s": round(eval_seconds, 4),
        "prompt_tokens": prompt_count,
        "generated_tokens": eval_count,
        "tokens_per_second": (
            round(tokens_per_second, 3)
            if tokens_per_second is not None else None
        ),
        "valid_json": valid_json,
        "valid_schema": valid_schema,
        "factual_valid": factual_valid,
        "factual_problems": factual_problems,
        "response": message,
    }


def main(repeats=3):
    OUTPUT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = OUTPUT_DIR / f"benchmark_{timestamp}.jsonl"
    csv_path = OUTPUT_DIR / f"benchmark_{timestamp}.csv"

    rows = []

    print("\nQwen2.5 1.5B Quantization Benchmark")
    print("Models:", len(MODELS))
    print("Repetitions per model:", repeats)
    print("Total measured calls:", len(MODELS) * repeats)

    # Rotate model order between repetitions to reduce order bias.
    for repetition in range(1, repeats + 1):
        ordered_models = (
            MODELS[repetition - 1:] + MODELS[:repetition - 1]
        )

        for label, model_name in ordered_models:
            print(
                f"\nRun {repetition}/{repeats}: "
                f"{label} ({model_name})"
            )

            try:
                metrics = call_model(model_name)

                row = {
                    "timestamp": datetime.now().isoformat(),
                    "model_label": label,
                    "model_name": model_name,
                    "repetition": repetition,
                    **metrics,
                }

                rows.append(row)

                print("Wall time:", metrics["wall_seconds"], "seconds")
                print("Generated tokens:", metrics["generated_tokens"])
                print("Tokens/second:", metrics["tokens_per_second"])
                print("Valid JSON schema:", metrics["valid_schema"])

            except Exception as error:
                print("FAILED:", str(error))
                rows.append({
                    "timestamp": datetime.now().isoformat(),
                    "model_label": label,
                    "model_name": model_name,
                    "repetition": repetition,
                    "error": str(error),
                })

    # Preserve full raw responses in JSONL.
    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save a convenient tabular copy.
    columns = [
        "timestamp", "model_label", "model_name", "repetition",
        "wall_seconds", "total_duration_s", "load_duration_s",
        "prompt_eval_duration_s", "eval_duration_s",
        "prompt_tokens", "generated_tokens", "tokens_per_second",
        "valid_json", "valid_schema", "factual_valid", "factual_problems", "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: row.get(key, "") for key in columns
            })

    print("\nBenchmark finished.")
    print("CSV:", csv_path)
    print("Raw JSONL:", jsonl_path)


if __name__ == "__main__":
    main(repeats=1)
