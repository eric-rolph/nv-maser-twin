"""
Tests: ExperimentTracker integration with Trainer.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from nv_maser.config import SimConfig
from nv_maser.model.training import Trainer
from nv_maser.tracking import ExperimentTracker


def _minimal_config(tmp_path: Path) -> SimConfig:
    config = SimConfig()
    config.training.epochs = 1
    config.training.num_samples = 20
    config.training.batch_size = 10
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")
    return config


# ---------------------------------------------------------------------------
# Test 1: Trainer works without a tracker
# ---------------------------------------------------------------------------


def test_trainer_without_tracker(tmp_path):
    """Trainer must run without a tracker and not raise."""
    config = _minimal_config(tmp_path)
    trainer = Trainer(config)  # no tracker
    history = trainer.train()
    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) >= 1


# ---------------------------------------------------------------------------
# Test 2: Tracker records one run with the correct arch and best_val_loss
# ---------------------------------------------------------------------------


def test_trainer_with_tracker_logs_run(tmp_path):
    """A run must appear in tracker.list_runs() with arch and best_val_loss set."""
    db_path = tmp_path / "runs.db"
    config = _minimal_config(tmp_path)
    tracker = ExperimentTracker(db_path)

    trainer = Trainer(config, tracker=tracker)
    trainer.train()

    runs = tracker.list_runs()
    assert len(runs) == 1
    run = runs[0]

    assert run["arch"] == config.model.architecture.value
    assert run["best_val_loss"] is not None
    assert isinstance(run["best_val_loss"], float)


# ---------------------------------------------------------------------------
# Test 3: Tracker records per-epoch metrics with train_loss / val_loss
# ---------------------------------------------------------------------------


def test_trainer_with_tracker_logs_epochs(tmp_path):
    """get_run_metrics() must return >=1 row with float train_loss and val_loss."""
    db_path = tmp_path / "runs.db"
    config = _minimal_config(tmp_path)
    tracker = ExperimentTracker(db_path)

    trainer = Trainer(config, tracker=tracker)
    trainer.train()

    runs = tracker.list_runs()
    run_id = runs[0]["id"]
    metrics = tracker.get_run_metrics(run_id)

    assert len(metrics) >= 1
    row = metrics[0]
    assert isinstance(row["train_loss"], float)
    assert isinstance(row["val_loss"], float)


# ---------------------------------------------------------------------------
# Test 4: sweep script imports and exposes main() + run_sweep()
# ---------------------------------------------------------------------------


def test_sweep_script_imports():
    """scripts.run_sweep must be importable and expose main and run_sweep."""
    # Ensure repo root is on sys.path so `scripts` package is importable
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    mod = importlib.import_module("scripts.run_sweep")
    assert callable(getattr(mod, "main", None)), "run_sweep.main must be callable"
    assert callable(getattr(mod, "run_sweep", None)), "run_sweep.run_sweep must be callable"
