"""
Bridge between trained RL policies and the closed-loop simulator.

Provides functions to load trained checkpoints (PPO or supervised) and wrap
them as ``controller_fn`` callables compatible with ``ClosedLoopSimulator``.

Usage:
    from nv_maser.rl.bridge import load_ppo_controller, load_supervised_controller

    # PPO checkpoint → closed-loop
    controller_fn = load_ppo_controller("checkpoints/rl/best_ppo.pt")
    sim = ClosedLoopSimulator(config, controller_fn)
    result = sim.run(duration_us=100_000)

    # Supervised checkpoint → closed-loop
    controller_fn = load_supervised_controller("checkpoints/best.pt", config)
    sim = ClosedLoopSimulator(config, controller_fn)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from numpy.typing import NDArray

from ..config import SimConfig, ModelConfig, ModelArchitecture, CoilConfig
from ..model.controller import build_controller
from .ppo import ActorCritic


def load_ppo_controller(
    checkpoint_path: str | Path,
    sim_config: SimConfig | None = None,
    deterministic: bool = True,
) -> Callable[[NDArray[np.float32]], NDArray[np.float32]]:
    """Load a PPO checkpoint and return a closed-loop-compatible controller_fn.

    Args:
        checkpoint_path: Path to ``best_ppo.pt``.
        sim_config: If ``None``, the config stored in the checkpoint is used.
        deterministic: If ``True``, use the policy mean (no sampling noise).

    Returns:
        Callable that maps ``(H, W) field → (num_coils,) currents``.
    """
    data = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    if sim_config is None:
        sim_config = SimConfig(**data["sim_config"])

    ac = ActorCritic(
        sim_config.grid.size,
        sim_config.model,
        sim_config.coils,
    )
    ac.load_state_dict(data["actor_critic_state"])
    ac.eval()

    @torch.no_grad()
    def controller_fn(field: NDArray[np.float32]) -> NDArray[np.float32]:
        obs = torch.from_numpy(field).unsqueeze(0).unsqueeze(0).float()
        if deterministic:
            features = ac._encode(obs)
            mean = ac.policy_head(features) * ac.max_current
            return mean.squeeze(0).numpy()
        else:
            action, _, _ = ac(obs)
            return action.squeeze(0).numpy()

    return controller_fn


def load_supervised_controller(
    checkpoint_path: str | Path,
    config: SimConfig,
) -> Callable[[NDArray[np.float32]], NDArray[np.float32]]:
    """Load a supervised-trained checkpoint (best.pt) as a controller_fn.

    Args:
        checkpoint_path: Path to ``best.pt`` from Trainer.
        config: Config matching the training setup.

    Returns:
        Callable that maps ``(H, W) field → (num_coils,) currents``.
    """
    data = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    model = build_controller(config.grid.size, config.model, config.coils)
    model.load_state_dict(data["model_state"])
    model.eval()

    @torch.no_grad()
    def controller_fn(field: NDArray[np.float32]) -> NDArray[np.float32]:
        obs = torch.from_numpy(field).unsqueeze(0).unsqueeze(0).float()
        currents = model(obs)
        return currents.squeeze(0).numpy()

    return controller_fn


def validate_policy_closed_loop(
    checkpoint_path: str | Path,
    config: SimConfig | None = None,
    duration_us: float = 100_000.0,
    policy_type: str = "ppo",
    seed: int | None = 42,
) -> dict:
    """Convenience: load checkpoint → run closed-loop → return summary.

    Args:
        checkpoint_path: Path to checkpoint file.
        config: SimConfig (overrides checkpoint's stored config if given).
        duration_us: Simulation duration in microseconds.
        policy_type: ``"ppo"`` or ``"supervised"``.
        seed: RNG seed for reproducibility.

    Returns:
        Dict from ``ClosedLoopResult.summary()`` plus ``policy_type``.
    """
    from ..physics.closed_loop import ClosedLoopSimulator

    if policy_type == "ppo":
        controller_fn = load_ppo_controller(checkpoint_path, config)
    elif policy_type == "supervised":
        if config is None:
            raise ValueError("config is required for supervised policy type")
        controller_fn = load_supervised_controller(checkpoint_path, config)
    else:
        raise ValueError(f"Unknown policy_type: {policy_type!r}")

    if config is None:
        data = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        config = SimConfig(**data["sim_config"])

    sim = ClosedLoopSimulator(config, controller_fn, seed=seed)
    result = sim.run(duration_us=duration_us)

    summary = result.summary()
    summary["policy_type"] = policy_type
    return summary
