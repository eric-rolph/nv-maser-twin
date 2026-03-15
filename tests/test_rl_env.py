"""Tests for the ShimmingEnv RL environment (Sprint 4)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nv_maser.rl.env import ShimmingEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def env() -> ShimmingEnv:
    return ShimmingEnv()


@pytest.fixture()
def reset_env(env: ShimmingEnv) -> tuple[ShimmingEnv, np.ndarray]:
    obs, _ = env.reset(seed=0)
    return env, obs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reset_returns_correct_shape(env: ShimmingEnv) -> None:
    obs, info = env.reset()
    assert obs.shape == (1, 64, 64), f"Expected (1,64,64), got {obs.shape}"
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_returns_correct_shape(reset_env: tuple[ShimmingEnv, np.ndarray]) -> None:
    env, _ = reset_env
    action = np.zeros(8, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (1, 64, 64), f"Expected (1,64,64), got {obs.shape}"
    assert obs.dtype == np.float32


def test_step_returns_float_reward(reset_env: tuple[ShimmingEnv, np.ndarray]) -> None:
    env, _ = reset_env
    action = np.zeros(8, dtype=np.float32)
    _, reward, _, _, _ = env.step(action)
    assert isinstance(reward, float)


def test_action_clipping(env: ShimmingEnv) -> None:
    """Large action values should be silently clipped — no crash, finite reward."""
    env.reset(seed=1)
    action = np.full(8, 999.0, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert math.isfinite(reward), "Reward should be finite after clipping"
    assert obs.shape == (1, 64, 64)


def test_episode_terminates(env: ShimmingEnv) -> None:
    """Environment should signal terminated=True exactly at max_steps."""
    env.reset(seed=2)
    action = np.zeros(env.num_coils, dtype=np.float32)
    terminated = False
    for step in range(env.max_steps):
        _, _, terminated, _, _ = env.step(action)
    assert terminated, "terminated should be True after max_steps steps"


def test_reset_with_seed(env: ShimmingEnv) -> None:
    """reset(seed=42) twice must produce identical initial observations."""
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    np.testing.assert_array_equal(
        obs1, obs2, err_msg="reset(seed=42) not reproducible"
    )


def test_observation_space_spec(env: ShimmingEnv) -> None:
    assert env.observation_space["shape"] == (1, 64, 64)
    assert env.action_space["shape"] == (8,)
    assert env.action_space["low"] == pytest.approx(-1.0)
    assert env.action_space["high"] == pytest.approx(1.0)
