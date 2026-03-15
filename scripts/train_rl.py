"""
RL training baseline: REINFORCE for magnetic field shimming.

Trains a stochastic policy (deterministic CNN/MLP/LSTM base + learned diagonal
Gaussian head) using episodic REINFORCE with return normalisation and gradient
clipping.

Usage:
    python scripts/train_rl.py [--episodes 500] [--steps 50] [--arch cnn] [--seed 42]
    python scripts/train_rl.py --episodes 200 --arch lstm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

# Make sure the installed package is found when running as a standalone script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nv_maser.config import (
    CoilConfig,
    ModelArchitecture,
    ModelConfig,
    SimConfig,
)
from nv_maser.model.controller import build_controller
from nv_maser.rl.env import ShimmingEnv


class StochasticShimmingPolicy(nn.Module):
    """
    Wraps a deterministic shimming controller with a learned diagonal Gaussian.

    The base controller outputs mean coil currents; a learned log_std parameter
    (one scalar per coil) defines the exploration noise.  During a forward pass,
    an action is sampled from Normal(mean, std) and its log-probability is
    returned for use in the REINFORCE loss.  Actions are clipped to the
    environment bounds *after* sampling; the log-prob is computed on the
    *pre-clip* sample so gradients flow cleanly through the distribution.
    """

    def __init__(self, base: nn.Module, num_coils: int) -> None:
        super().__init__()
        self.base = base
        # One learnable log-std per coil, initialised at 0 → std = 1
        self.log_std = nn.Parameter(torch.zeros(num_coils))

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, 1, H, W) observation tensor.

        Returns:
            action_sample: (batch, num_coils) clipped action.
            log_prob:      (batch,) sum of per-coil log-probs for the sample.
        """
        mean = self.base(obs)                          # (batch, num_coils)
        std = self.log_std.exp().clamp(min=1e-6)      # (num_coils,)
        dist = Normal(mean, std)
        raw_sample = dist.rsample()                    # (batch, num_coils), differentiable
        log_prob = dist.log_prob(raw_sample).sum(-1)   # (batch,)
        # Clip to [-max_current, max_current] — the base already constrains mean
        # via tanh, so we use the same scale for the clamp bounds
        max_c = getattr(self.base, "max_current", 1.0)
        action_sample = raw_sample.clamp(-max_c, max_c)
        return action_sample, log_prob


def _build_policy(
    arch: str, grid_size: int, num_coils: int
) -> StochasticShimmingPolicy:
    """Construct a StochasticShimmingPolicy for the requested architecture."""
    model_cfg = ModelConfig(architecture=ModelArchitecture(arch))
    coil_cfg = CoilConfig(num_coils=num_coils)
    base = build_controller(grid_size, model_cfg, coil_cfg)
    return StochasticShimmingPolicy(base, num_coils)


def train(
    episodes: int = 100,
    steps_per_episode: int = 50,
    arch: str = "cnn",
    seed: int = 42,
    lr: float = 3e-4,
    log_interval: int = 20,
) -> dict:
    """
    Run REINFORCE training on ShimmingEnv.

    Each episode rolls out *steps_per_episode* environment steps, collects
    (log_prob, reward) pairs, computes Monte-Carlo returns (no discounting for
    this single-step-reward baseline), and performs one Adam update with
    gradient clipping.

    Returns:
        dict with keys:
            final_mean_return  – mean episodic return over last log_interval eps
            episode_returns    – list of per-episode total returns
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = ShimmingEnv()
    grid_size = env.grid_size
    num_coils = env.num_coils

    policy = _build_policy(arch, grid_size, num_coils)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    episode_returns: list[float] = []
    recent_losses: list[float] = []

    for ep in range(1, episodes + 1):
        obs_np, _ = env.reset(seed=seed + ep)

        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []

        for _ in range(steps_per_episode):
            obs_t = torch.from_numpy(obs_np).unsqueeze(0)  # (1, 1, H, W)
            action_t, log_prob = policy(obs_t)              # action: (1, C), lp: (1,)

            action_np = action_t.squeeze(0).detach().cpu().numpy()
            obs_np, reward, terminated, truncated, _ = env.step(action_np)

            log_probs.append(log_prob.squeeze(0))   # scalar tensor
            rewards.append(reward)

            if terminated or truncated:
                break

        # ---------- REINFORCE update ----------
        # Compute Monte-Carlo returns (no discount — baseline keeps it simple)
        T = len(rewards)
        returns = torch.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            running += rewards[t]
            returns[t] = running

        # Normalise returns for variance reduction
        if T > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Stack log probs: each is a scalar tensor on the graph
        log_probs_t = torch.stack(log_probs)   # (T,)

        loss = -(log_probs_t * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        ep_return = float(sum(rewards))
        episode_returns.append(ep_return)
        recent_losses.append(loss.item())

        if ep % log_interval == 0:
            window = episode_returns[-log_interval:]
            mean_ret = float(np.mean(window))
            std_ret = float(np.std(window))
            mean_loss = float(np.mean(recent_losses[-log_interval:]))
            print(
                f"Episode {ep:4d}/{episodes} | "
                f"Mean Return: {mean_ret:+.4f} | "
                f"Std: {std_ret:.4f} | "
                f"Loss: {mean_loss:+.4f}"
            )

    final_window = episode_returns[-min(log_interval, episodes):]
    return {
        "final_mean_return": float(np.mean(final_window)),
        "episode_returns": episode_returns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="REINFORCE baseline for magnetic field shimming.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--arch",
        choices=["cnn", "mlp", "lstm"],
        default="cnn",
        help="Controller architecture.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-interval", type=int, default=20)
    args = parser.parse_args()

    results = train(
        episodes=args.episodes,
        steps_per_episode=args.steps,
        arch=args.arch,
        seed=args.seed,
        lr=args.lr,
        log_interval=args.log_interval,
    )
    print(f"\nTraining complete. Final mean return: {results['final_mean_return']:.4f}")


if __name__ == "__main__":
    main()
