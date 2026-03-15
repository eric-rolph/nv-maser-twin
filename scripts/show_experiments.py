"""
show_experiments.py — CLI to inspect the NV Maser experiment database.

Usage:
  # List all runs (most recent first):
  python scripts/show_experiments.py

  # Use a custom DB path:
  python scripts/show_experiments.py --db path/to/runs.db

  # Show detailed metrics for a specific run:
  python scripts/show_experiments.py --run-id 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nv_maser.tracking import ExperimentTracker


def _fmt(value: object, width: int = 12) -> str:
    """Right-align a value in a fixed-width column."""
    s = "" if value is None else str(value)
    return s[:width].rjust(width)


def print_runs(tracker: ExperimentTracker) -> None:
    header = (
        f"{'ID':>4}  {'Timestamp':>26}  {'Arch':>8}  "
        f"{'Epochs':>6}  {'Samples':>8}  {'BestVal':>10}  Notes"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    runs = tracker.list_runs(limit=100)
    if not runs:
        print("  (no runs recorded)")
        return
    for r in runs:
        best = f"{r['best_val_loss']:.6f}" if r["best_val_loss"] is not None else ""
        print(
            f"{r['id']:>4}  {r['timestamp']:>26}  {r['arch']:>8}  "
            f"{r['epochs'] or '':>6}  {r['num_samples'] or '':>8}  "
            f"{best:>10}  {r['notes'] or ''}"
        )


def print_run_detail(tracker: ExperimentTracker, run_id: int) -> None:
    runs = tracker.list_runs(limit=10_000)
    run = next((r for r in runs if r["id"] == run_id), None)
    if run is None:
        print(f"Run {run_id} not found.")
        sys.exit(1)

    print(f"\n=== Run {run_id} ===")
    for key, val in run.items():
        print(f"  {key:<20}: {val}")

    metrics = tracker.get_run_metrics(run_id)
    if not metrics:
        print("\n  (no epoch metrics recorded)")
        return

    print(f"\n  {'Epoch':>6}  {'TrainLoss':>12}  {'ValLoss':>12}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}")
    for m in metrics:
        tl = f"{m['train_loss']:.6f}" if m["train_loss"] is not None else ""
        vl = f"{m['val_loss']:.6f}" if m["val_loss"] is not None else ""
        print(f"  {m['epoch']:>6}  {tl:>12}  {vl:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the NV Maser experiment database."
    )
    parser.add_argument(
        "--db",
        default="experiments/runs.db",
        help="Path to the SQLite DB (default: experiments/runs.db)",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        metavar="N",
        help="Show detailed metrics for run N",
    )
    args = parser.parse_args()

    tracker = ExperimentTracker(db_path=args.db)

    if args.run_id is not None:
        print_run_detail(tracker, args.run_id)
    else:
        print_runs(tracker)


if __name__ == "__main__":
    main()
