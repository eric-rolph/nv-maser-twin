"""
Controller comparison: least-squares baseline vs trained NN vs PPO.

The coil field is linear in the currents, so the ridge least-squares
solution is the OPTIMAL open-loop correction — no learned controller can
beat it on field variance. This script reports where every method stands
relative to that bound and to the absolute masing tolerance, in both:

  open loop    controller sees the exact distorted field, currents applied
               perfectly (the supervised-training task), and
  closed loop  sensor noise, DAC quantisation, coil L/R settling, and
               computation latency in the loop (the deployment task).

Usage:
    python scripts/eval_baseline.py
    python scripts/eval_baseline.py --samples 1000 --closed-loop-us 50000
    python scripts/eval_baseline.py --checkpoint checkpoints/best.pt \
        --ppo-checkpoint checkpoints/best_ppo.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nv_maser.config import SimConfig
from nv_maser.model.lstsq_baseline import LeastSquaresShimmer
from nv_maser.physics.closed_loop import ClosedLoopSimulator
from nv_maser.physics.environment import FieldEnvironment
from nv_maser.physics.maser_gain import max_tolerable_b_std


def _load_controllers(config: SimConfig, args) -> dict:
    """Build the controller_fn dict: name → (callable or None, note)."""
    env = FieldEnvironment(config)
    lstsq = LeastSquaresShimmer.from_environment(env, ridge=0.0)

    controllers: dict[str, tuple] = {
        "zero (no shim)": (
            lambda f: np.zeros(config.coils.num_coils, dtype=np.float32),
            "",
        ),
        "least-squares": (lstsq, "optimal linear bound"),
    }

    from nv_maser.rl.bridge import (
        load_ppo_controller,
        load_supervised_controller,
    )

    if Path(args.checkpoint).exists():
        try:
            controllers["NN (supervised)"] = (
                load_supervised_controller(args.checkpoint, config),
                args.checkpoint,
            )
        except Exception as exc:  # noqa: BLE001 — report and continue
            controllers["NN (supervised)"] = (None, f"load failed: {exc}")
    else:
        controllers["NN (supervised)"] = (
            None,
            f"checkpoint not found: {args.checkpoint}",
        )

    if Path(args.ppo_checkpoint).exists():
        try:
            controllers["PPO (RL)"] = (
                load_ppo_controller(args.ppo_checkpoint, config),
                args.ppo_checkpoint,
            )
        except Exception as exc:  # noqa: BLE001
            controllers["PPO (RL)"] = (None, f"load failed: {exc}")
    else:
        controllers["PPO (RL)"] = (
            None,
            f"checkpoint not found: {args.ppo_checkpoint}",
        )

    return controllers


def run_open_loop(config: SimConfig, controllers: dict, n_samples: int):
    env = FieldEnvironment(config)
    mask = env.grid.active_zone_mask
    tolerable = max_tolerable_b_std(config.nv, config.maser)

    print(f"\n=== Open loop ({n_samples} disturbance samples) ===")
    print(f"masing tolerance: field std <= {tolerable*1e6:.2f} uT")

    distorted, _ = env.generate_training_data(n_samples)
    before = np.std(distorted[:, mask], axis=1)
    print(f"uncorrected: mean std {np.mean(before)*1e6:.1f} uT")
    header = (
        f"{'method':<18} {'mean std (uT)':>14} {'improvement':>12} "
        f"{'masing %':>9} {'ms/solve':>9}"
    )
    print(header)
    print("-" * len(header))

    for name, (fn, note) in controllers.items():
        if fn is None:
            print(f"{name:<18} skipped - {note}")
            continue
        t0 = time.perf_counter()
        currents = np.stack(
            [fn(distorted[i]) for i in range(n_samples)]
        ).astype(np.float32)
        ms = (time.perf_counter() - t0) / n_samples * 1e3
        net = distorted + env.coils.compute_field(currents)
        after = np.std(net[:, mask], axis=1)
        improvement = float(np.mean(before) / np.mean(after))
        masing_pct = float(np.mean(after <= tolerable) * 100)
        print(
            f"{name:<18} {np.mean(after)*1e6:>14.1f} "
            f"{improvement:>11.2f}x {masing_pct:>8.1f}% {ms:>9.3f}"
        )


def run_closed_loop(config: SimConfig, controllers: dict, duration_us: float):
    # Same disturbance trajectory for every controller — otherwise each
    # simulator draws its own random sequence and the rows aren't comparable.
    config = config.model_copy(
        update={
            "disturbance": config.disturbance.model_copy(
                update={"seed": 42}
            )
        }
    )
    print(f"\n=== Closed loop ({duration_us/1e3:.0f} ms simulated, "
          "sensor noise + DAC + settling + latency) ===")
    header = (
        f"{'method':<18} {'mean std (uT)':>14} {'gain budget':>12} "
        f"{'masing %':>9}"
    )
    print(header)
    print("-" * len(header))

    for name, (fn, note) in controllers.items():
        if fn is None:
            print(f"{name:<18} skipped - {note}")
            continue
        sim = ClosedLoopSimulator(config, fn, seed=42)
        result = sim.run(duration_us)
        mean_std = float(
            np.mean([s.field_std for s in result.steps]) * 1e6
        )
        print(
            f"{name:<18} {mean_std:>14.1f} "
            f"{result.mean_gain_budget:>12.3f} "
            f"{result.masing_fraction*100:>8.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare shimming controllers against the LSQ baseline."
    )
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--closed-loop-us", type=float, default=50_000.0)
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt"
    )
    parser.add_argument(
        "--ppo-checkpoint", type=str, default="checkpoints/rl/best_ppo.pt"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="YAML config override"
    )
    args = parser.parse_args()

    config = SimConfig()
    if args.config:
        import yaml

        from nv_maser.main import _deep_merge_config

        with open(args.config, encoding="utf-8") as f:
            config = _deep_merge_config(config, yaml.safe_load(f))

    controllers = _load_controllers(config, args)
    run_open_loop(config, controllers, args.samples)
    run_closed_loop(config, controllers, args.closed_loop_us)

    print(
        "\nNote: with linear coils, least-squares is the optimal open-loop"
        "\ncorrection - learned controllers can match it, never beat it."
        "\nTheir value, if any, must show up in the closed-loop columns."
    )


if __name__ == "__main__":
    main()
