"""Tests for the RL → closed-loop bridge."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from nv_maser.config import SimConfig
from nv_maser.rl.bridge import (
    load_ppo_controller,
    load_supervised_controller,
    validate_policy_closed_loop,
)
from nv_maser.rl.ppo import ActorCritic, PPOConfig, PPOTrainer
from nv_maser.model.controller import build_controller


@pytest.fixture()
def ppo_checkpoint(tmp_path):
    """Train a tiny PPO run and return the checkpoint path."""
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
    return tmp_path / "rl" / "best_ppo.pt"


@pytest.fixture()
def supervised_checkpoint(tmp_path):
    """Save a dummy supervised checkpoint."""
    config = SimConfig()
    model = build_controller(config.grid.size, config.model, config.coils)
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": 0,
            "val_loss": 1.0,
        },
        tmp_path / "best.pt",
    )
    return tmp_path / "best.pt"


class TestLoadPPOController:
    def test_returns_callable(self, ppo_checkpoint) -> None:
        fn = load_ppo_controller(ppo_checkpoint)
        assert callable(fn)

    def test_output_shape(self, ppo_checkpoint) -> None:
        fn = load_ppo_controller(ppo_checkpoint)
        field = np.zeros((64, 64), dtype=np.float32)
        currents = fn(field)
        assert currents.shape == (8,)
        assert currents.dtype == np.float32

    def test_deterministic_mode(self, ppo_checkpoint) -> None:
        fn = load_ppo_controller(ppo_checkpoint, deterministic=True)
        field = np.random.randn(64, 64).astype(np.float32)
        c1 = fn(field)
        c2 = fn(field)
        np.testing.assert_array_equal(c1, c2)


class TestLoadSupervisedController:
    def test_returns_callable(self, supervised_checkpoint) -> None:
        config = SimConfig()
        fn = load_supervised_controller(supervised_checkpoint, config)
        assert callable(fn)

    def test_output_shape(self, supervised_checkpoint) -> None:
        config = SimConfig()
        fn = load_supervised_controller(supervised_checkpoint, config)
        field = np.zeros((64, 64), dtype=np.float32)
        currents = fn(field)
        assert currents.shape == (8,)


class TestValidatePolicyClosedLoop:
    def test_ppo_closed_loop(self, ppo_checkpoint) -> None:
        """Full pipeline: PPO checkpoint → closed-loop → summary."""
        summary = validate_policy_closed_loop(
            ppo_checkpoint,
            duration_us=5_000.0,
            policy_type="ppo",
        )
        assert "mean_variance" in summary
        assert "mean_gain_budget" in summary
        assert "masing_fraction" in summary
        assert summary["policy_type"] == "ppo"
        assert summary["num_steps"] > 0

    def test_supervised_closed_loop(self, supervised_checkpoint) -> None:
        """Full pipeline: supervised checkpoint → closed-loop → summary."""
        config = SimConfig()
        summary = validate_policy_closed_loop(
            supervised_checkpoint,
            config=config,
            duration_us=5_000.0,
            policy_type="supervised",
        )
        assert summary["policy_type"] == "supervised"
        assert summary["num_steps"] > 0
