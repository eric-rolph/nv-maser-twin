"""
Hyperparameter sweep over learning-rate × architecture.

Usage:
    python scripts/run_sweep.py [--epochs 5] [--db experiments/runs.db]
                                [--archs cnn mlp lstm] [--lrs 1e-4 3e-4 1e-3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/run_sweep.py` from repo root
_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def run_sweep(
    lrs: list[float],
    archs: list[str],
    epochs: int,
    db: str,
) -> list[dict]:
    """
    Run a grid search over lrs × archs, logging every run to *db*.

    Returns a list of result dicts (arch, lr, best_val_loss).
    """
    from nv_maser.config import SimConfig, TrainingConfig, ModelConfig, ModelArchitecture
    from nv_maser.model.training import Trainer
    from nv_maser.tracking import ExperimentTracker

    tracker = ExperimentTracker(db_path=db)
    results: list[dict] = []

    total = len(archs) * len(lrs)
    combo_idx = 0

    for arch_str in archs:
        for lr in lrs:
            combo_idx += 1
            print(
                f"\n[Sweep {combo_idx}/{total}] arch={arch_str}  lr={lr:.2e}  epochs={epochs}"
            )
            try:
                config = SimConfig()
                config.training.learning_rate = lr
                config.training.epochs = epochs
                config.model.architecture = ModelArchitecture(arch_str)

                trainer = Trainer(config, tracker=tracker)
                trainer.train()
                best = trainer.best_val_loss
            except Exception as exc:  # noqa: BLE001
                print(f"  [Sweep] ERROR for arch={arch_str} lr={lr:.2e}: {exc}")
                best = None

            results.append({"arch": arch_str, "lr": lr, "best_val_loss": best})

    return results


def _print_summary(results: list[dict]) -> None:
    """Print a sorted summary table."""
    # Sort: runs with a real loss first (ascending), then failed runs
    def sort_key(r):
        v = r["best_val_loss"]
        return (0, v) if v is not None else (1, 0.0)

    sorted_results = sorted(results, key=sort_key)

    print("\n" + "=" * 52)
    print(f"{'ARCH':<8}  {'LR':>10}  {'BEST_VAL_LOSS':>18}")
    print("-" * 52)
    for r in sorted_results:
        bvl = f"{r['best_val_loss']:.6e}" if r["best_val_loss"] is not None else "FAILED"
        print(f"{r['arch']:<8}  {r['lr']:>10.2e}  {bvl:>18}")
    print("=" * 52)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep over lr × architecture"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epochs per run (default: 5 for fast sweep)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="experiments/runs.db",
        help="Path to the SQLite DB for experiment tracking",
    )
    parser.add_argument(
        "--archs",
        nargs="+",
        default=["cnn", "mlp", "lstm"],
        help="Architectures to sweep (default: cnn mlp lstm)",
    )
    parser.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        default=[1e-4, 3e-4, 1e-3],
        help="Learning rates to sweep (default: 1e-4 3e-4 1e-3)",
    )
    args = parser.parse_args()

    results = run_sweep(
        lrs=args.lrs,
        archs=args.archs,
        epochs=args.epochs,
        db=args.db,
    )
    _print_summary(results)


if __name__ == "__main__":
    main()
