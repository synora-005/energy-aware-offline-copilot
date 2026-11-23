"""Seeded, deterministic workloads used by harness.py.

Run directly (the harness does this for you):
  python workloads.py calibrate
  python workloads.py batch <seed> <first_index> <count> <setup_units> <task_units>
  python workloads.py noise <seconds> <busy_ms> <period_ms> <units_per_s>

'units' = iterations of a SHA-256 loop. Same seed -> identical digests, so two
variants can be PROVEN to have done equivalent work (matching checksum).
"""
import hashlib
import random
import sys
import time


def burn(units, seed):
    """Deterministic single-core CPU work. Returns a hex digest."""
    data = random.Random(seed).randbytes(4096)
    for _ in range(units):
        data = hashlib.sha256(data).digest() * 128      # keep 4096 bytes
    return hashlib.sha256(data).hexdigest()


def calibrate(target_s=0.5):
    """Units per second of single-core work on this machine."""
    units = 2000
    while True:
        t0 = time.perf_counter()
        burn(units, 0)
        dt = time.perf_counter() - t0
        if dt >= target_s:
            return int(units / dt)
        units *= 2


def run_batch(seed, first, count, setup_units, task_units):
    """One 'process invocation': pay setup cost ONCE, then do `count` tasks."""
    burn(setup_units, seed)                  # e.g. opening a connection/model
    for i in range(first, first + count):
        print(f"{i}:{burn(task_units, seed * 100000 + i)}", flush=True)


def run_noise(seconds, busy_ms, period_ms, units_per_s):
    """Synthetic 'chatty background app': busy_ms of work every period_ms."""
    end = time.time() + seconds
    units = max(1, int(units_per_s * busy_ms / 1000))
    bursts = 0
    while time.time() < end:
        t0 = time.perf_counter()
        burn(units, bursts)
        bursts += 1
        rest = period_ms / 1000 - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(min(rest, max(0.0, end - time.time())))
    print(f"bursts:{bursts}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "calibrate":
        print(calibrate())
    elif cmd == "batch":
        seed, first, count, setup_u, task_u = map(int, sys.argv[2:7])
        run_batch(seed, first, count, setup_u, task_u)
    elif cmd == "noise":
        run_noise(float(sys.argv[2]), float(sys.argv[3]),
                  float(sys.argv[4]), int(sys.argv[5]))