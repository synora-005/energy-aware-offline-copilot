"""Go/no-go environment check for the Energy-Aware Offline Copilot.

Usage:  python check_env.py [ollama_model_name]
Default model: qwen2.5:1.5b

Checks: system info, battery discharge rate (WMI), LibreHardwareMonitor
package power, Ollama + local SLM speed, and structured JSON output.
Run once plugged in and once on battery (battery reading needs battery).
"""
import json
import platform
import subprocess
import sys
import time
import urllib.request

OLLAMA = "http://localhost:11434"
LHM = "http://localhost:8085/data.json"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:1.5b"
results = {}


def http_json(url, payload=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def section(title):
    print(f"\n=== {title} ===")


def check_system():
    section("System")
    print("Python:", sys.version.split()[0], "|", platform.platform())
    try:
        import psutil
    except ImportError:
        print("psutil is missing -> run: pip install psutil")
        results["psutil"] = False
        return
    results["psutil"] = True
    print("CPU cores (physical/logical):",
          psutil.cpu_count(logical=False), "/", psutil.cpu_count())
    freq = psutil.cpu_freq()
    if freq:
        print(f"CPU freq now/max: {freq.current:.0f} / {freq.max:.0f} MHz")
    vm = psutil.virtual_memory()
    print(f"RAM total/available: {vm.total / 2**30:.2f} / "
          f"{vm.available / 2**30:.2f} GiB")
    batt = psutil.sensors_battery()
    if batt:
        print(f"Battery: {batt.percent:.0f}% | plugged in: {batt.power_plugged}")
    else:
        print("No battery detected")


def check_battery_wmi():
    section("Battery discharge rate (Windows WMI)")
    if platform.system() != "Windows":
        print("Skipped (not Windows)")
        results["battery_wmi"] = None
        return
    cmd = [
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance -Namespace root\\WMI -ClassName BatteryStatus | "
        "Select-Object DischargeRate,ChargeRate,PowerOnline,Discharging,"
        "RemainingCapacity | ConvertTo-Json",
    ]
    rates = []
    for i in range(5):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if out.returncode != 0 or not out.stdout.strip():
                print("No data. stderr:", out.stderr.strip()[:200])
                results["battery_wmi"] = False
                return
            data = json.loads(out.stdout)
            if isinstance(data, list):
                data = data[0]
            rate = data.get("DischargeRate") or 0
            rates.append(rate)
            print(f"sample {i + 1}: discharge={rate} mW, "
                  f"plugged={data.get('PowerOnline')}, "
                  f"capacity={data.get('RemainingCapacity')}")
        except Exception as e:  # noqa: BLE001
            print("Error:", e)
            results["battery_wmi"] = False
            return
        time.sleep(1)
    ok = any(r > 0 for r in rates)
    if not ok:
        print("Rate is 0 -> unplug the charger and run again.")
    results["battery_wmi"] = ok


def find_power(node, found):
    text, val = node.get("Text", ""), node.get("Value", "")
    if isinstance(val, str) and val.strip().endswith("W") and "Package" in text:
        found.append((text, val))
    for child in node.get("Children", []):
        find_power(child, found)


def check_lhm():
    section("LibreHardwareMonitor (CPU package power)")
    try:
        tree = http_json(LHM, timeout=5)
    except Exception:
        print("Not reachable at", LHM)
        print("Setup: run LibreHardwareMonitor as admin -> Options -> "
              "Remote Web Server -> Run (port 8085).")
        results["lhm"] = False
        return
    found = []
    find_power(tree, found)
    for name, val in found:
        print(f"{name}: {val}")
    results["lhm"] = bool(found)
    if not found:
        print("Reachable, but no 'Package' power sensor was exposed.")


def check_ollama():
    section(f"Ollama + local SLM ({MODEL})")
    try:
        tags = http_json(f"{OLLAMA}/api/tags", timeout=5)
    except Exception:
        print("Ollama not reachable. Install it and make sure it is running.")
        results["ollama"] = False
        return
    names = [m["name"] for m in tags.get("models", [])]
    print("Installed models:", names)
    if not any(n.startswith(MODEL) for n in names):
        print(f"Model not found. Run: ollama pull {MODEL}")
        results["ollama"] = False
        return
    try:
        t0 = time.time()
        r = http_json(f"{OLLAMA}/api/generate", {
            "model": MODEL,
            "prompt": "In two sentences, explain why reducing idle "
                      "background processes saves energy.",
            "stream": False,
            "options": {"num_predict": 80, "temperature": 0},
        })
        wall = time.time() - t0
        tps = r["eval_count"] / (r["eval_duration"] / 1e9) if r.get("eval_duration") else 0
        print(f"Load time: {r.get('load_duration', 0) / 1e9:.1f}s | "
              f"generated {r['eval_count']} tokens | {tps:.1f} tok/s | "
              f"wall {wall:.1f}s")
        print("Sample:", r["response"].strip()[:200])
        results["ollama"] = True
        results["tokens_per_s"] = tps
    except Exception as e:  # noqa: BLE001
        print("Generation failed:", e)
        results["ollama"] = False
        return

    # Structured output test (JSON schema via 'format')
    try:
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "energy_wh": {"type": "number"},
            },
            "required": ["summary", "energy_wh"],
        }
        r = http_json(f"{OLLAMA}/api/generate", {
            "model": MODEL,
            "prompt": "Metrics: CPU 78%, energy 45 Wh. Return JSON with a "
                      "one-sentence summary and the energy_wh value.",
            "stream": False,
            "format": schema,
            "options": {"num_predict": 120, "temperature": 0},
        })
        parsed = json.loads(r["response"])
        print("Structured output OK:", parsed)
        results["structured"] = True
    except Exception as e:  # noqa: BLE001
        print("Structured output failed (update Ollama?):", e)
        results["structured"] = False


def summary():
    section("SUMMARY")
    for k, v in results.items():
        print(f"{k:14}: {v}")
    tps = results.get("tokens_per_s", 0)
    if results.get("ollama") and tps:
        verdict = "OK" if tps >= 3 else "SLOW (use short outputs / smaller model)"
        print(f"\nModel speed verdict: {verdict}")
    sources = [k for k in ("battery_wmi", "lhm") if results.get(k)]
    if sources:
        print("Independent energy signal available via:", ", ".join(sources))
    else:
        print("No independent energy signal -> label results as model-based.")


if __name__ == "__main__":
    check_system()
    check_battery_wmi()
    check_lhm()
    check_ollama()
    summary()