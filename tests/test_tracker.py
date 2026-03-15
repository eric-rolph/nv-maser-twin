"""
Tests for src/nv_maser/tracking/tracker.py

Run with:
  .venv/Scripts/python.exe -m pytest tests/test_tracker.py -v
"""
from __future__ import annotations

import pytest

from nv_maser.config import SimConfig
from nv_maser.tracking import ExperimentTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tracker(tmp_path) -> ExperimentTracker:
    return ExperimentTracker(db_path=tmp_path / "runs.db")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_create_tracker_empty_db(tmp_path):
    """Creating a tracker should produce a DB file on disk."""
    tracker = make_tracker(tmp_path)
    assert (tmp_path / "runs.db").exists()
    assert tracker.list_runs() == []


def test_start_run_returns_int(tmp_path):
    """start_run should return a positive integer run_id."""
    tracker = make_tracker(tmp_path)
    run_id = tracker.start_run(arch="cnn")
    assert isinstance(run_id, int)
    assert run_id > 0


def test_start_run_with_config(tmp_path):
    """start_run with a SimConfig should populate num_samples, epochs, config_hash."""
    tracker = make_tracker(tmp_path)
    cfg = SimConfig()
    run_id = tracker.start_run(arch="mlp", config=cfg, notes="smoke test")
    runs = tracker.list_runs()
    assert len(runs) == 1
    r = runs[0]
    assert r["id"] == run_id
    assert r["arch"] == "mlp"
    assert r["num_samples"] == cfg.training.num_samples
    assert r["epochs"] == cfg.training.epochs
    assert r["config_hash"] is not None
    assert r["notes"] == "smoke test"


def test_log_epoch_stored(tmp_path):
    """log_epoch should persist rows retrievable via get_run_metrics in order."""
    tracker = make_tracker(tmp_path)
    run_id = tracker.start_run(arch="cnn")
    for epoch in range(3):
        tracker.log_epoch(run_id, epoch, train_loss=1.0 - epoch * 0.1, val_loss=1.1 - epoch * 0.1)

    metrics = tracker.get_run_metrics(run_id)
    assert len(metrics) == 3
    assert [m["epoch"] for m in metrics] == [0, 1, 2]
    assert metrics[0]["train_loss"] == pytest.approx(1.0)
    assert metrics[2]["val_loss"] == pytest.approx(0.9)


def test_finish_run_updates_best_val_loss(tmp_path):
    """finish_run should set best_val_loss and end_timestamp on the run."""
    tracker = make_tracker(tmp_path)
    run_id = tracker.start_run(arch="lstm")
    tracker.finish_run(run_id, best_val_loss=0.042)

    runs = tracker.list_runs()
    r = runs[0]
    assert r["best_val_loss"] == pytest.approx(0.042)
    assert r["end_timestamp"] is not None


def test_list_runs_most_recent_first(tmp_path):
    """list_runs should return runs ordered by timestamp DESC."""
    import time

    tracker = make_tracker(tmp_path)
    id1 = tracker.start_run(arch="cnn", notes="first")
    time.sleep(0.01)  # ensure distinct timestamps
    id2 = tracker.start_run(arch="mlp", notes="second")

    runs = tracker.list_runs()
    assert len(runs) == 2
    # Most recently started run should come first
    assert runs[0]["id"] == id2
    assert runs[1]["id"] == id1


def test_delete_run_removes_metrics(tmp_path):
    """delete_run should remove the run row and all its metric rows."""
    tracker = make_tracker(tmp_path)
    run_id = tracker.start_run(arch="cnn")
    tracker.log_epoch(run_id, 0, 0.9, 0.95)
    tracker.log_epoch(run_id, 1, 0.8, 0.85)

    tracker.delete_run(run_id)

    assert tracker.list_runs() == []
    assert tracker.get_run_metrics(run_id) == []


def test_log_epoch_persists_physics_metrics(tmp_path):
    """Physics kwargs (gain_budget, cooperativity, etc.) must be persisted."""
    tracker = make_tracker(tmp_path)
    run_id = tracker.start_run(arch="cnn")
    tracker.log_epoch(
        run_id, 0, 0.5, 0.6,
        gain_budget=2.5, cooperativity=0.8,
        physics_penalty=0.01, field_variance=1e-5,
    )
    rows = tracker.get_run_metrics(run_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["gain_budget"] == pytest.approx(2.5)
    assert row["cooperativity"] == pytest.approx(0.8)
    assert row["physics_penalty"] == pytest.approx(0.01)
    assert row["field_variance"] == pytest.approx(1e-5)


def test_log_epoch_without_physics_stores_nulls(tmp_path):
    """When no physics kwargs are passed, columns should be NULL."""
    tracker = make_tracker(tmp_path)
    run_id = tracker.start_run(arch="cnn")
    tracker.log_epoch(run_id, 0, 0.5, 0.6)
    rows = tracker.get_run_metrics(run_id)
    row = rows[0]
    assert row["gain_budget"] is None
    assert row["cooperativity"] is None


def test_tracker_migrates_old_schema(tmp_path):
    """Opening a DB created with the old (4-column) schema should auto-migrate."""
    import sqlite3
    db_path = tmp_path / "old.db"
    # Create old-style DB without physics columns
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, arch TEXT, config_hash TEXT,
                num_samples INTEGER, epochs INTEGER, notes TEXT,
                best_val_loss REAL, end_timestamp TEXT
            );
            CREATE TABLE metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                train_loss REAL,
                val_loss REAL
            );
        """)
    # Opening with ExperimentTracker should add missing columns
    tracker = ExperimentTracker(db_path)
    run_id = tracker.start_run(arch="test")
    tracker.log_epoch(run_id, 0, 0.5, 0.6, gain_budget=1.23)
    rows = tracker.get_run_metrics(run_id)
    assert rows[0]["gain_budget"] == pytest.approx(1.23)
