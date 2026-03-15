"""
Multi-architecture performance benchmark for NV Maser Digital Twin.
Benchmarks CNN, MLP, and LSTM controllers + RL environment step latency.
Sprint 4 spec: B=1 median inference < 1.0 ms per architecture.
"""
import time
import statistics

import numpy as np
import torch

from nv_maser.config import (
    CoilConfig,
    ModelArchitecture,
    ModelConfig,
    SimConfig,
)
from nv_maser.model.controller import build_controller
from nv_maser.rl.env import ShimmingEnv

# ── constants ────────────────────────────────────────────────────────────────
GRID_SIZE = 64
NUM_COILS = 8
N_WARMUP_B1 = 20
N_REPS_B1 = 300
N_REPS_B32 = 75
BATCH_SIZE = 32
SPEC_MS = 1.0          # Sprint 4 design spec: B=1 median < 1.0 ms
RL_STEPS = 100


# ── helpers ───────────────────────────────────────────────────────────────────

def _percentile(data: list, p: float) -> float:
    """Return the p-th percentile (0–100) of *data* (sorted copy)."""
    s = sorted(data)
    idx = int(p / 100 * len(s))
    idx = min(idx, len(s) - 1)
    return s[idx]


def _bench_architecture(
    arch: ModelArchitecture,
    grid_size: int,
    num_coils: int,
    n_warmup: int,
    n_b1: int,
    n_b32: int,
) -> dict:
    """Run B=1 and B=32 latency benchmarks for one architecture."""
    model_cfg = ModelConfig(architecture=arch)
    coil_cfg = CoilConfig(num_coils=num_coils)
    model = build_controller(grid_size, model_cfg, coil_cfg)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())

    # ── Warm-up (B=1) ────────────────────────────────────────────────────────
    dummy = torch.randn(1, 1, grid_size, grid_size)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

    # ── B=1 timing ───────────────────────────────────────────────────────────
    times_b1 = []
    for _ in range(n_b1):
        x = torch.randn(1, 1, grid_size, grid_size)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        times_b1.append((time.perf_counter() - t0) * 1000)

    # ── B=32 timing ──────────────────────────────────────────────────────────
    times_b32 = []
    for _ in range(n_b32):
        x = torch.randn(BATCH_SIZE, 1, grid_size, grid_size)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        times_b32.append((time.perf_counter() - t0) * 1000)

    b32_mean = statistics.mean(times_b32)

    return {
        "arch": arch.value.upper(),
        "params": param_count,
        "b1_mean": statistics.mean(times_b1),
        "b1_median": statistics.median(times_b1),
        "b1_p95": _percentile(times_b1, 95),
        "b1_p99": _percentile(times_b1, 99),
        "b32_mean": b32_mean,
        "b32_ms_per_sample": b32_mean / BATCH_SIZE,
        "spec_pass": statistics.median(times_b1) < SPEC_MS,
    }


def _bench_rl_env(n_steps: int = RL_STEPS) -> dict:
    """Benchmark ShimmingEnv.step() latency."""
    config = SimConfig()
    env = ShimmingEnv(config)
    env.reset()

    action = np.zeros(config.coils.num_coils, dtype=np.float32)
    step_times = []

    for _ in range(n_steps):
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        step_times.append((time.perf_counter() - t0) * 1000)
        if terminated or truncated:
            env.reset()

    return {
        "mean_ms": statistics.mean(step_times),
        "p95_ms": _percentile(step_times, 95),
        "n_steps": n_steps,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> dict:
    print("\n=== NV Maser Architecture Benchmark ===")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}  |  Coils: {NUM_COILS}  |  Reps: {N_REPS_B1}\n")

    architectures = [ModelArchitecture.CNN, ModelArchitecture.MLP, ModelArchitecture.LSTM]
    results = []

    for arch in architectures:
        r = _bench_architecture(
            arch,
            grid_size=GRID_SIZE,
            num_coils=NUM_COILS,
            n_warmup=N_WARMUP_B1,
            n_b1=N_REPS_B1,
            n_b32=N_REPS_B32,
        )
        results.append(r)

    # ── comparison table ─────────────────────────────────────────────────────
    col_w = [13, 11, 11, 9, 10, 10, 5]
    header = (
        f"{'Architecture':<{col_w[0]}} | "
        f"{'Params':<{col_w[1]}} | "
        f"{'B=1 Median':<{col_w[2]}} | "
        f"{'B=1 P95':<{col_w[3]}} | "
        f"{'B=32 mean':<{col_w[4]}} | "
        f"{'ms/sample':<{col_w[5]}} | "
        f"{'Spec':<{col_w[6]}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for r in results:
        spec_label = "PASS" if r["spec_pass"] else "FAIL"
        row = (
            f"{r['arch']:<{col_w[0]}} | "
            f"{r['params']:>{col_w[1]},} | "
            f"{r['b1_median']:>8.3f} ms | "
            f"{r['b1_p95']:>6.3f} ms | "
            f"{r['b32_mean']:>7.3f} ms  | "
            f"{r['b32_ms_per_sample']:>7.3f} ms | "
            f"{spec_label:<{col_w[6]}}"
        )
        print(row)

    print(sep)

    # ── per-architecture detail ───────────────────────────────────────────────
    print("\n--- Detailed B=1 Latency ---")
    for r in results:
        print(
            f"  {r['arch']:<4}: mean={r['b1_mean']:.3f} ms  "
            f"median={r['b1_median']:.3f} ms  "
            f"P95={r['b1_p95']:.3f} ms  "
            f"P99={r['b1_p99']:.3f} ms"
        )

    # ── RL environment benchmark ──────────────────────────────────────────────
    print(f"\n--- RL Environment Step ({RL_STEPS} steps) ---")
    rl = _bench_rl_env(RL_STEPS)
    print(f"  ShimmingEnv.step(): mean={rl['mean_ms']:.4f} ms  P95={rl['p95_ms']:.4f} ms")

    # ── sprint spec summary ───────────────────────────────────────────────────
    passing = sum(1 for r in results if r["spec_pass"])
    total = len(results)
    print(f"\nSprint 4 spec compliance: {passing}/{total} architectures within {SPEC_MS} ms median")

    return {
        "architectures": results,
        "rl_env": rl,
        "spec_compliance": f"{passing}/{total}",
    }


if __name__ == "__main__":
    main()
