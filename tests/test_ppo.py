"""Tests for the PPO RL training module."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from nv_maser.config import SimConfig
from nv_maser.rl.ppo import (
    ActorCritic,
    PPOConfig,
    PPOTrainer,
    RolloutBuffer,
    compute_gae,
)

GRID_SIZE = 64
NUM_COILS = 8


# ═══════════════════════════════════════════════════════════════════════
# ActorCritic
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture()
def actor_critic() -> ActorCritic:
    cfg = SimConfig()
    return ActorCritic(cfg.grid.size, cfg.model, cfg.coils)


def test_actor_critic_forward_shapes(actor_critic: ActorCritic) -> None:
    obs = torch.zeros(2, 1, GRID_SIZE, GRID_SIZE)
    action, log_prob, value = actor_critic(obs)
    assert action.shape == (2, NUM_COILS)
    assert log_prob.shape == (2,)
    assert value.shape == (2,)


def test_actor_critic_evaluate_shapes(actor_critic: ActorCritic) -> None:
    obs = torch.zeros(2, 1, GRID_SIZE, GRID_SIZE)
    actions = torch.zeros(2, NUM_COILS)
    log_prob, value, entropy = actor_critic.evaluate(obs, actions)
    assert log_prob.shape == (2,)
    assert value.shape == (2,)
    assert entropy.shape == (2,)


def test_actor_critic_actions_bounded(actor_critic: ActorCritic) -> None:
    """Actions should be clipped to [-max_current, max_current]."""
    obs = torch.randn(10, 1, GRID_SIZE, GRID_SIZE)
    action, _, _ = actor_critic(obs)
    max_c = actor_critic.max_current
    assert action.min() >= -max_c - 1e-6
    assert action.max() <= max_c + 1e-6


# ═══════════════════════════════════════════════════════════════════════
# GAE computation
# ═══════════════════════════════════════════════════════════════════════


def test_compute_gae_shapes() -> None:
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 1.0, 1.5]
    dones = [False, False, True]
    adv, ret = compute_gae(rewards, values, dones, last_value=0.0)
    assert adv.shape == (3,)
    assert ret.shape == (3,)


def test_compute_gae_terminal_episode() -> None:
    """When done=True at last step, last_value should not contribute."""
    rewards = [1.0, 1.0]
    values = [0.0, 0.0]
    dones = [False, True]
    adv, ret = compute_gae(rewards, values, dones, last_value=999.0, gamma=1.0, lam=1.0)
    # With gamma=1, lam=1: step 1 (done=True) → adv=reward=1.0
    # step 0 (done=False) → delta=1+1*0.0-0=1, gae=1+1*1*1=2
    assert adv[1] == pytest.approx(1.0)
    assert adv[0] == pytest.approx(2.0)


def test_compute_gae_no_discount() -> None:
    """With gamma=1, lam=1, GAE = Monte-Carlo returns minus values."""
    rewards = [1.0, 2.0, 3.0]
    values = [0.0, 0.0, 0.0]
    dones = [False, False, False]
    adv, ret = compute_gae(rewards, values, dones, last_value=0.0, gamma=1.0, lam=1.0)
    # Returns: [6, 5, 3], advantages same (values=0)
    np.testing.assert_allclose(ret, [6.0, 5.0, 3.0], atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Rollout buffer
# ═══════════════════════════════════════════════════════════════════════


def test_rollout_buffer_add_and_clear() -> None:
    buf = RolloutBuffer()
    buf.add(
        obs=np.zeros((1, 64, 64), dtype=np.float32),
        action=np.zeros(8, dtype=np.float32),
        log_prob=-1.0,
        reward=0.5,
        value=0.3,
        done=False,
    )
    assert len(buf) == 1
    buf.clear()
    assert len(buf) == 0


# ═══════════════════════════════════════════════════════════════════════
# PPOTrainer integration
# ═══════════════════════════════════════════════════════════════════════


def test_ppo_trainer_short_run() -> None:
    """PPOTrainer.train() completes and returns expected keys."""
    cfg = PPOConfig(
        total_timesteps=256,
        steps_per_rollout=128,
        num_epochs_per_update=2,
        batch_size=32,
        eval_episodes=2,
        log_interval=100,
    )
    trainer = PPOTrainer(ppo_config=cfg)
    result = trainer.train()

    assert "history" in result
    assert "best_eval_return" in result
    assert "final_eval" in result
    assert isinstance(result["best_eval_return"], float)
    assert len(result["history"]["rollout_returns"]) > 0


def test_ppo_collect_rollout() -> None:
    """collect_rollout fills the buffer with expected number of steps."""
    cfg = PPOConfig(steps_per_rollout=64, total_timesteps=64)
    trainer = PPOTrainer(ppo_config=cfg)
    trainer.collect_rollout()
    assert len(trainer.buffer) == 64


def test_ppo_evaluate() -> None:
    """evaluate() returns metrics dict."""
    cfg = PPOConfig(eval_episodes=2, total_timesteps=1)
    trainer = PPOTrainer(ppo_config=cfg)
    result = trainer.evaluate(num_episodes=2)
    assert "mean_return" in result
    assert "std_return" in result
    assert isinstance(result["mean_return"], float)


def test_ppo_with_tracker(tmp_path) -> None:
    """PPOTrainer logs runs and metrics to ExperimentTracker."""
    from nv_maser.tracking.tracker import ExperimentTracker

    tracker = ExperimentTracker(db_path=tmp_path / "test.db")
    cfg = PPOConfig(
        total_timesteps=128,
        steps_per_rollout=64,
        num_epochs_per_update=1,
        batch_size=32,
        eval_episodes=1,
        log_interval=100,
    )
    trainer = PPOTrainer(ppo_config=cfg, tracker=tracker)
    trainer.train()

    runs = tracker.list_runs()
    assert len(runs) == 1
    assert runs[0]["arch"].startswith("ppo_")
    metrics = tracker.get_run_metrics(runs[0]["id"])
    assert len(metrics) > 0


def test_ppo_checkpoint_saved(tmp_path) -> None:
    """PPOTrainer saves best_ppo.pt checkpoint."""
    cfg = PPOConfig(
        total_timesteps=128,
        steps_per_rollout=64,
        num_epochs_per_update=1,
        batch_size=32,
        eval_episodes=1,
        checkpoint_dir=str(tmp_path / "rl"),
        log_interval=100,
    )
    trainer = PPOTrainer(ppo_config=cfg)
    trainer.train()

    ckpt = tmp_path / "rl" / "best_ppo.pt"
    assert ckpt.exists()
    data = torch.load(ckpt, weights_only=False)
    assert "actor_critic_state" in data
    assert "eval_return" in data
