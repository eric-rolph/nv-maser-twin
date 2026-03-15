"""Tests for the REINFORCE RL training baseline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Ensure src/ is importable when run directly with pytest from the repo root.
# (The package is installed as an editable install, so this is usually redundant
# but keeps the test hermetic.)
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.train_rl import StochasticShimmingPolicy, _build_policy, train


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

GRID_SIZE = 64
NUM_COILS = 8


@pytest.fixture()
def cnn_policy() -> StochasticShimmingPolicy:
    return _build_policy("cnn", GRID_SIZE, NUM_COILS)


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_stochastic_policy_shape(cnn_policy: StochasticShimmingPolicy) -> None:
    """policy(obs) must return (action, log_prob) with correct batch shapes."""
    obs = torch.zeros(1, 1, GRID_SIZE, GRID_SIZE)
    action, log_prob = cnn_policy(obs)

    assert action.shape == (1, NUM_COILS), (
        f"Expected action shape (1, {NUM_COILS}), got {action.shape}"
    )
    assert log_prob.shape == (1,), (
        f"Expected log_prob shape (1,), got {log_prob.shape}"
    )


def test_train_runs_short() -> None:
    """train() completes without error and returns required keys."""
    result = train(
        episodes=5,
        steps_per_episode=10,
        arch="cnn",
        seed=0,
        lr=1e-3,
        log_interval=100,  # suppress output during test
    )
    assert isinstance(result, dict), "train() must return a dict"
    assert "final_mean_return" in result, (
        "result dict must contain 'final_mean_return'"
    )
    assert isinstance(result["final_mean_return"], float), (
        "final_mean_return must be a float"
    )
