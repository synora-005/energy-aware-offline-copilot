# Energy-Aware Offline Copilot

**PS 04: Energy-Aware AI: Offline-First Copilot**

An offline-first energy copilot. It measures the energy of computing workloads, uses a local small language model to explain the results in plain English, proves savings with a reproducible baseline-vs-optimized experiment, and keeps insights in a durable queue that syncs safely when the network returns.


## At a glance

| Item | Value |
|---|---|
| Local SLM | **Qwen2.5-1.5B-Instruct** (about 1.5 billion parameters, below the 4B limit) |
| Quantization used | **Q4_K_M** (about 986 MB), served by **Ollama** |
| License of the model | Apache-2.0 |
| Energy source | Battery capacity drop via Windows WMI (**measured**); discharge-rate integral as cross-check |
| Sampling interval | 1 s |
| Storage | SQLite (WAL mode), fully local |
| Sync | SQLite outbox, idempotency keys, retry with backoff, crash recovery |
| Dashboard | Streamlit |
| Tested on | One Windows laptop (see Limitations) |

## Measured results (this laptop, unplugged)

Every run is a 300 s window; baseline and optimized share a seed, so they do identical work. Order is ABBA to cancel drift. Statistics are a paired t-test over 5 pairs (background) and 7 pairs (batching). All 12 pairs favour the optimized variant.

| Scenario | Baseline | Optimized | Saving | 95% CI of saving | Verdict |
|---|---|---|---|---|---|
| Idle/background (**synthetic** chatty app) | 1.100 Wh | 0.940 Wh | 0.160 Wh (14.5%) | 0.061 to 0.258 Wh | distinguishable from noise |
| Batching (one process per task vs one batched process, 7 pairs) | 1.218 Wh | 0.990 Wh | 0.228 Wh (18.7%) | 0.131 to 0.326 Wh | distinguishable from noise |

Batching runs prove equivalent work: baseline and optimized produce matching SHA-256 checksums for every seed.

**Formulas**

- Energy (Wh) = drop in battery capacity over the run (whole system)
- Cost = kWh x tariff (assumed 6.71 currency units per kWh)
- CO2e = kWh x grid factor (assumed 727 g CO2e per kWh)

Background: 0.160 / 1000 x 6.71 = 0.00107 currency units and 0.160 / 1000 x 727 = 0.1162 g CO2e per 300 s window. Batching: 0.228 Wh gives 0.00153 currency units and 0.1660 g CO2e.

Raw evidence: `evidence_summary.txt`.

### Scenario 3: model inference settings

Same 6 fact sets, same prompt, three quantizations of the same model. "Verified" means the production number verifier accepted the answer on the first attempt. Energy is whole-system battery drop minus idle power (net), measured over 3 repetitions.

| Quantization | Verified | Tokens/s | Latency | Net Wh per call |
|---|---|---|---|---|
| Q2_K | 0% | 26.1 | 5.0 s | 0.0126 |
| **Q4_K_M (chosen)** | **100%** | 18.3 | 7.0 s | 0.0230 |
| Q8_0 | 83% | 11.7 | 9.4 s | 0.0284 |

Q2_K is cheapest but produced no usable output. Q4_K_M gave 100% verified output, about 1.6x the speed of Q8_0 and about 19% less energy. Raw data: `slm_benchmark_results/`.

### Net savings (the copilot's own cost)

One explanation costs about 0.023 Wh, which is 12 to 14% of the gross saving per window. Break-even is 36 to 43 s of optimized running. This is indicative only (3 repeats, large idle noise). See `net_savings.txt`.

### Predict, apply, verify

`predict.py` stores a saving prediction before a change is applied, then compares it with the measured saving and recalibrates.

- **Model:** average power = a + b x CPU%, fitted on measured runs of the *other* scenarios (the scenario being predicted is excluded). The planned CPU reduction comes from the workload design, not from the optimized runs.
- **Result (retrospective: the runs already existed, but the model never saw them):**

| Scenario | Predicted | Measured (5 pairs) | Error |
|---|---|---|---|
| Background | 0.103 +- 0.023 Wh | 0.160 Wh | +0.057 Wh |
| Batching | 0.130 +- 0.036 Wh | 0.194 Wh | +0.064 Wh |

- **Finding:** the model underpredicts by about 1.5x because it counts CPU work but not wake-up and process-start overhead. Both measured savings lie inside the prediction's own uncertainty only once the measurement noise is included. A correction factor learned on one scenario cut the other's error to under 0.01 Wh. This uses 2 scenarios, so it is indicative only.
- **Not yet shown:** a fully prospective test with enough repeats. Two later batching pairs (seeds 110 and 111) were too few to confirm or refute the prediction.

```
python predict.py demo      # predict + verify both scenarios, show recalibration
python predict.py show
```

## How it works

```
Seeded workloads -> Monitor (1 s) -> Battery energy -> Paired statistics
                                  |
                           Local SQLite (readings, runs, energy, insights, outbox)
                                  |
Facts engine (all numbers in code) -> Local SLM (Ollama) -> Verifier -> Dashboard
                                                                |
                                      Outbox queue -> Sync worker -> Cloud receiver (dedupes by key)
```

- **Trust layer.** All numbers are computed in code (`facts.py`). The SLM only phrases them. `explainer.py` checks every number and some wording rules in the reply; on failure it retries with a hint, then falls back to a code-written text. The verifier checks that numbers exist in the data. It does not prove the reasoning is sound.
- **Durable sync.** Delivery is at-least-once. The receiver dedupes by idempotency key, so processing is effectively once. Items in flight at a crash are re-queued on restart. Being offline does not use up retry attempts.
- **Offline-first.** Measurements, insights and the queue live in SQLite. The only network use is the optional sync to the mock cloud receiver.

## Setup

Requirements: Windows 10/11 laptop with a battery (for measured runs), Python 3.11, [Ollama](https://ollama.com).

```
git clone https://github.com/synora-005/energy-aware-offline-copilot.git
cd energy-aware-offline-copilot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python demo_db.py load
```

`demo_db.py load` copies the measured demo data into `copilot.db` so the dashboard shows results immediately. Add `--force` to replace an existing database.

Download the model once (needs internet; after this everything works offline):

```
ollama pull qwen2.5:1.5b
ollama show qwen2.5:1.5b
python check_env.py qwen2.5:1.5b
```

For the quantization comparison only:

```
ollama pull qwen2.5:1.5b-instruct-q2_K
ollama pull qwen2.5:1.5b-instruct-q8_0
```

## Run the demo

Use three terminals, each with the virtual environment active.

```
# Terminal 1: dashboard
streamlit run dashboard.py

# Terminal 2: mock cloud receiver (stop it to simulate going offline)
python mock_cloud.py

# Terminal 3: sync worker
python outbox.py worker
```

Demo checklist (in the order the PS asks):

1. **SLM explanation from readings.** In the dashboard, open the AI insights tab and generate explanations. Or run `python save_insights.py`.
2. **Retention across restart.** Close everything, reopen, run `python save_insights.py show`. Insights and queue state are still there.
3. **Automatic sync on reconnect.** Stop `mock_cloud.py`, queue items, restart it. The worker delivers them with no action from you.
4. **Crash recovery.** `python outbox.py worker --crash-after-send` crashes right after a send. Restart the worker normally: it recovers the in-flight items and the receiver ignores the duplicate.
5. **Baseline vs optimized.** The Experiments tab, or `python harness.py summary`.

## Reproduce the experiments

Run unplugged, battery above 40%, other apps closed, and the model unloaded (`ollama stop qwen2.5:1.5b`). Each repeat is a baseline run and an optimized run of 300 s.

```
python harness.py run background --repeats 5
python harness.py run batching --repeats 5
python harness.py summary
python benchmark_slm_energy.py
python net_savings.py
```

`python harness.py run <scenario> --allow-plugged` produces model-based estimates only, and they are labelled as such.

## Files

| File | Purpose |
|---|---|
| `monitor.py` | Records a run: samples CPU/RAM and battery, computes energy, stores in SQLite |
| `workloads.py` | Seeded deterministic workloads (SHA-256 loops) |
| `harness.py` | Baseline vs optimized runs, ABBA order, paired t-test summary |
| `facts.py` | Builds the grounded facts JSON from stored runs |
| `explainer.py` | Local SLM explanations, number verifier, fallback |
| `save_insights.py` | Stores insights and queues them for sync |
| `outbox.py` | Durable queue and sync worker |
| `mock_cloud.py` | Mock receiver that dedupes by idempotency key |
| `benchmark_slm_energy.py` | Scenario 3: quantization quality, latency and energy |
| `net_savings.py` | Saving minus the copilot's own energy |
| `predict.py` | Predict, apply, verify loop with recalibration |
| `demo_db.py` | Export and load the clean demo database |
| `dashboard.py`, `dash_data.py` | Streamlit dashboard |
| `check_env.py`, `probe_battery.py` | Environment and battery counter checks |

## Measurement method and limitations

- **Measured vs simulated.** Energy is measured from the battery capacity drop. The **background workload is a synthetic stand-in** for a chatty app; its energy is still measured. Runs on mains power fall back to a model-based estimate and are labelled as such.
- **Counters disagree in single runs.** The discharge-rate integral can differ from the capacity drop by 30 to 36% in a single run. Over the full set the totals agree within 2%. Totals use the capacity drop.
- **Battery resolution.** Very short runs fall below the capacity resolution, so runs are 300 s.
- **One machine.** Results come from one Windows laptop and do not generalize to other hardware.
- **Whole-system energy.** The battery measures the whole laptop, so base load dominates and savings are a modest share of the total.
- **Small samples.** 5 paired repeats for scenarios 1 and 2, 3 repeats for scenario 3 and net savings. The latter is indicative.
- **Embodied carbon** of the device is not counted.
- **Assumptions.** Tariff 6.71 per kWh (BESCOM residential, excluding tax and fixed charges); grid factor 727 g CO2e/kWh (CEA CO2 Baseline Database v20.0). Both are stored in the database; change them with `set_assumption` in `monitor.py`.

## Not built yet

A fully prospective predict-apply-verify test with enough repeats, tariff and carbon-aware scheduling, a self-calibrating power model, testing on other machines.