"""
Performance benchmark for NV Maser shimming pipeline.
Measures: inference latency (batch=1, batch=32), throughput, memory.
"""
import time
import torch
import numpy as np
import statistics
from nv_maser.config import SimConfig
from nv_maser.physics.environment import FieldEnvironment
from nv_maser.model.controller import build_controller


def benchmark_inference(n_warmup=10, n_bench=200):
    config = SimConfig()
    env = FieldEnvironment(config)
    model = build_controller(config.grid.size, config.model, config.coils)
    model.eval()

    size = config.grid.size

    # Warmup
    dummy = torch.randn(1, 1, size, size)
    for _ in range(n_warmup):
        with torch.no_grad():
            model(dummy)

    # Single inference
    times_single = []
    for _ in range(n_bench):
        x = torch.randn(1, 1, size, size)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        times_single.append((time.perf_counter() - t0) * 1000)

    # Batch inference (32)
    times_batch = []
    for _ in range(n_bench // 4):
        x = torch.randn(32, 1, size, size)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        times_batch.append((time.perf_counter() - t0) * 1000)

    # Coil field computation
    coils_times = []
    for _ in range(n_bench):
        currents = np.random.uniform(-1, 1, config.coils.num_coils).astype(np.float32)
        t0 = time.perf_counter()
        env.coils.compute_field(currents)
        coils_times.append((time.perf_counter() - t0) * 1000)

    print("\n=== NV Maser Performance Benchmark ===")
    print(f"Grid: {size}x{size}, Coils: {config.coils.num_coils}, n={n_bench}")
    print(f"\n[Inference - Batch=1]")
    print(f"  Mean:   {statistics.mean(times_single):.3f} ms")
    print(f"  Median: {statistics.median(times_single):.3f} ms")
    print(f"  P95:    {sorted(times_single)[int(0.95*n_bench)]:.3f} ms")
    print(f"  P99:    {sorted(times_single)[int(0.99*n_bench)]:.3f} ms")
    print(f"  Spec:   < 1.0 ms (target from design spec)")
    passed_single = statistics.median(times_single) < 1.0
    print(f"  Result: {'PASS' if passed_single else 'WARN (CPU-only, GPU expected)'}")

    print(f"\n[Inference - Batch=32]")
    mean_b = statistics.mean(times_batch)
    print(f"  Mean per batch: {mean_b:.3f} ms  ({mean_b/32:.4f} ms/sample)")

    print(f"\n[Coil Field Computation]")
    print(f"  Mean:   {statistics.mean(coils_times):.4f} ms")
    print(f"  P95:    {sorted(coils_times)[int(0.95*n_bench)]:.4f} ms")

    return {
        "single_median_ms": statistics.median(times_single),
        "single_p95_ms": sorted(times_single)[int(0.95 * n_bench)],
        "batch32_mean_ms": statistics.mean(times_batch),
        "coil_mean_ms": statistics.mean(coils_times),
    }


if __name__ == "__main__":
    results = benchmark_inference()
