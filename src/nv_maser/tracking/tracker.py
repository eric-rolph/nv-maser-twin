"""
Lightweight SQLite-based experiment tracker for NV Maser Digital Twin.

Uses only Python stdlib (sqlite3, hashlib, json, datetime) — no extra deps.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nv_maser.config import SimConfig

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    arch            TEXT NOT NULL,
    config_hash     TEXT,
    num_samples     INTEGER,
    epochs          INTEGER,
    notes           TEXT,
    best_val_loss   REAL,
    end_timestamp   TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    epoch       INTEGER NOT NULL,
    train_loss  REAL,
    val_loss    REAL
);
"""


class ExperimentTracker:
    """Persist training runs and per-epoch metrics to a local SQLite database."""

    def __init__(self, db_path: "str | Path" = "experiments/runs.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_DDL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(
        self,
        arch: str,
        config: "SimConfig | None" = None,
        notes: str = "",
    ) -> int:
        """Insert a new run row and return its run_id."""
        timestamp = datetime.utcnow().isoformat()
        config_hash = self._config_snapshot(config)
        num_samples: int | None = None
        epochs: int | None = None
        if config is not None:
            num_samples = config.training.num_samples
            epochs = config.training.epochs

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (timestamp, arch, config_hash, num_samples, epochs, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (timestamp, arch, config_hash, num_samples, epochs, notes),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def log_epoch(
        self,
        run_id: int,
        epoch: int,
        train_loss: float,
        val_loss: float,
    ) -> None:
        """Append one epoch's metrics for the given run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO metrics (run_id, epoch, train_loss, val_loss)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, epoch, train_loss, val_loss),
            )

    def finish_run(self, run_id: int, best_val_loss: float) -> None:
        """Mark a run as complete, storing best_val_loss and end_timestamp."""
        end_timestamp = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE runs
                SET best_val_loss = ?, end_timestamp = ?
                WHERE id = ?
                """,
                (best_val_loss, end_timestamp, run_id),
            )

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return up to *limit* runs ordered by timestamp DESC."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run_metrics(self, run_id: int) -> list[dict[str, Any]]:
        """Return all metric rows for *run_id* ordered by epoch."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM metrics WHERE run_id = ? ORDER BY epoch",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_run(self, run_id: int) -> None:
        """Delete a run and all its associated metrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _config_snapshot(self, config: "SimConfig | None") -> str | None:
        """Return SHA-256 hex digest of config's JSON representation, or None."""
        if config is None:
            return None
        payload = json.dumps(config.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()
