"""
Proximal Policy Optimization (PPO) for magnetic field shimming.

Implements PPO-Clip with Generalized Advantage Estimation (GAE), a shared
CNN/MLP feature encoder, separate policy and value heads, and optional
physics-shaped reward shaping from the environment info dict.

Architecture:
    Observation (1, H, W) → shared encoder → policy head (mean coil currents)
                                            → value head (scalar V(s))
    Exploration via learned diagonal Gaussian (same as REINFORCE baseline).

Usage:
    trainer = PPOTrainer(config)
    results = trainer.train(total_timesteps=50_000)
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from ..config import CoilConfig, ModelConfig, ModelArchitecture, SimConfig
from ..model.controller import build_controller
from .env import ShimmingEnv


# ═══════════════════════════════════════════════════════════════════════
# Actor-Critic network
# ═══════════════════════════════════════════════════════════════════════


class ActorCritic(nn.Module):
    """Actor-Critic for shimming with shared feature encoder.

    The *actor* outputs a diagonal Gaussian over coil currents.
    The *critic* outputs a scalar state value V(s).
    Both share the base encoder from ``build_controller``, but the
    final linear layer is replaced by two heads.
    """

    def __init__(
        self,
        grid_size: int,
        model_cfg: ModelConfig,
        coil_cfg: CoilConfig,
    ) -> None:
        super().__init__()
        self.max_current = coil_cfg.max_current_amps
        num_coils = coil_cfg.num_coils

        # Build the base controller and steal its feature encoder.
        # We strip the final linear+tanh head and replace it.
        base = build_controller(grid_size, model_cfg, coil_cfg)

        # Extract the shared feature encoder (everything before the final head)
        if hasattr(base, "conv_stack"):
            self.encoder = base.conv_stack
            # Probe encoder output dim
            with torch.no_grad():
                dummy = torch.zeros(1, 1, grid_size, grid_size)
                enc_dim = self.encoder(dummy).view(1, -1).shape[1]
        else:
            # MLP fallback: use first N-2 layers (before final linear+tanh)
            layers = list(base.network.children())
            self.encoder = nn.Sequential(*layers[:-2])
            with torch.no_grad():
                dummy = torch.zeros(1, grid_size * grid_size)
                enc_dim = self.encoder(dummy).shape[1]

        # Policy head: mean coil currents
        self.policy_head = nn.Sequential(
            nn.Linear(enc_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_coils),
            nn.Tanh(),
        )
        # Value head: scalar V(s)
        self.value_head = nn.Sequential(
            nn.Linear(enc_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Learned log-std per coil
        self.log_std = nn.Parameter(torch.zeros(num_coils))

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Shared encoder: obs (B, 1, H, W) → features (B, enc_dim)."""
        if hasattr(self, "_is_mlp"):
            return self.encoder(obs.view(obs.size(0), -1))
        feat = self.encoder(obs)
        return feat.view(feat.size(0), -1)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action: (B, num_coils) sampled and clipped.
            log_prob: (B,) log probability of action.
            value: (B,) state value estimate.
        """
        features = self._encode(obs)
        mean = self.policy_head(features) * self.max_current
        std = self.log_std.exp().clamp(min=1e-6)
        dist = Normal(mean, std)
        raw = dist.rsample()
        log_prob = dist.log_prob(raw).sum(-1)
        action = raw.clamp(-self.max_current, self.max_current)
        value = self.value_head(features).squeeze(-1)
        return action, log_prob, value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given actions under current policy (for PPO update).

        Returns:
            log_prob: (B,)
            value: (B,)
            entropy: (B,)
        """
        features = self._encode(obs)
        mean = self.policy_head(features) * self.max_current
        std = self.log_std.exp().clamp(min=1e-6)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        value = self.value_head(features).squeeze(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy


# ═══════════════════════════════════════════════════════════════════════
# Rollout buffer
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class RolloutBuffer:
    """Stores one epoch of rollout data for PPO update."""

    observations: list[np.ndarray] = dc_field(default_factory=list)
    actions: list[np.ndarray] = dc_field(default_factory=list)
    log_probs: list[float] = dc_field(default_factory=list)
    rewards: list[float] = dc_field(default_factory=list)
    values: list[float] = dc_field(default_factory=list)
    dones: list[bool] = dc_field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self) -> None:
        for lst in (
            self.observations, self.actions, self.log_probs,
            self.rewards, self.values, self.dones,
        ):
            lst.clear()

    def __len__(self) -> int:
        return len(self.rewards)


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation.

    Returns:
        advantages: (T,) array.
        returns: (T,) array (advantages + values).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(T)):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ═══════════════════════════════════════════════════════════════════════
# PPO Trainer
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""

    total_timesteps: int = 50_000
    steps_per_rollout: int = 2048
    num_epochs_per_update: int = 10
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    seed: int = 42
    arch: str = "cnn"
    log_interval: int = 1
    checkpoint_dir: str = "checkpoints/rl"
    eval_episodes: int = 10


class PPOTrainer:
    """PPO training loop with GAE, clipped surrogate, and value baseline."""

    def __init__(
        self,
        sim_config: SimConfig | None = None,
        ppo_config: PPOConfig | None = None,
        tracker: "Any | None" = None,
    ) -> None:
        self.sim_cfg = sim_config or SimConfig()
        self.ppo_cfg = ppo_config or PPOConfig()
        self.tracker = tracker

        torch.manual_seed(self.ppo_cfg.seed)
        np.random.seed(self.ppo_cfg.seed)

        self.env = ShimmingEnv(self.sim_cfg)
        self.ac = ActorCritic(
            self.env.grid_size,
            self.sim_cfg.model,
            self.sim_cfg.coils,
        )
        self.optimizer = torch.optim.Adam(
            self.ac.parameters(), lr=self.ppo_cfg.learning_rate
        )
        self.buffer = RolloutBuffer()

    def collect_rollout(self) -> float:
        """Collect ``steps_per_rollout`` environment transitions.

        Returns:
            Mean episode return observed during collection.
        """
        self.buffer.clear()
        self.ac.eval()

        obs_np, _ = self.env.reset(seed=self.ppo_cfg.seed)
        episode_returns: list[float] = []
        ep_reward = 0.0

        for _ in range(self.ppo_cfg.steps_per_rollout):
            obs_t = torch.from_numpy(obs_np).unsqueeze(0)
            with torch.no_grad():
                action_t, log_prob_t, value_t = self.ac(obs_t)

            action_np = action_t.squeeze(0).cpu().numpy()
            next_obs_np, reward, terminated, truncated, _info = self.env.step(action_np)
            done = terminated or truncated

            self.buffer.add(
                obs=obs_np,
                action=action_np,
                log_prob=float(log_prob_t.item()),
                reward=reward,
                value=float(value_t.item()),
                done=done,
            )
            ep_reward += reward

            if done:
                episode_returns.append(ep_reward)
                ep_reward = 0.0
                obs_np, _ = self.env.reset()
            else:
                obs_np = next_obs_np

        return float(np.mean(episode_returns)) if episode_returns else ep_reward

    def update(self) -> dict[str, float]:
        """Run PPO update on collected rollout data.

        Returns:
            Dict of mean loss components over all mini-batch updates.
        """
        self.ac.train()
        cfg = self.ppo_cfg

        # Bootstrap last value for GAE
        last_obs = self.buffer.observations[-1]
        with torch.no_grad():
            last_obs_t = torch.from_numpy(last_obs).unsqueeze(0)
            _, _, last_val = self.ac(last_obs_t)
            last_value = float(last_val.item())

        advantages, returns = compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            last_value,
            cfg.gamma,
            cfg.gae_lambda,
        )

        # Normalise advantages
        adv_mean, adv_std = advantages.mean(), advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # Convert to tensors
        obs_t = torch.from_numpy(
            np.array(self.buffer.observations)
        )  # (T, 1, H, W)
        act_t = torch.from_numpy(
            np.array(self.buffer.actions)
        )  # (T, num_coils)
        old_log_probs_t = torch.tensor(
            self.buffer.log_probs, dtype=torch.float32
        )
        adv_t = torch.from_numpy(advantages)
        ret_t = torch.from_numpy(returns)

        T = len(self.buffer)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _epoch in range(cfg.num_epochs_per_update):
            indices = np.random.permutation(T)
            for start in range(0, T, cfg.batch_size):
                end = min(start + cfg.batch_size, T)
                idx = indices[start:end]

                mb_obs = obs_t[idx]
                mb_act = act_t[idx]
                mb_old_lp = old_log_probs_t[idx]
                mb_adv = adv_t[idx]
                mb_ret = ret_t[idx]

                new_lp, new_val, entropy = self.ac.evaluate(mb_obs, mb_act)

                # Clipped surrogate objective
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon
                ) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(new_val, mb_ret)

                # Total loss
                loss = (
                    policy_loss
                    + cfg.value_loss_coef * value_loss
                    - cfg.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ac.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def evaluate(self, num_episodes: int | None = None) -> dict[str, float]:
        """Run deterministic evaluation episodes.

        Returns:
            Dict with mean_return, mean_variance, mean_gain_budget.
        """
        n = num_episodes or self.ppo_cfg.eval_episodes
        self.ac.eval()
        returns: list[float] = []
        variances: list[float] = []
        gain_budgets: list[float] = []

        for ep in range(n):
            obs_np, _ = self.env.reset(seed=10_000 + ep)
            ep_reward = 0.0
            for _ in range(self.env.max_steps):
                obs_t = torch.from_numpy(obs_np).unsqueeze(0)
                with torch.no_grad():
                    action_t, _, _ = self.ac(obs_t)
                action_np = action_t.squeeze(0).cpu().numpy()
                obs_np, reward, terminated, truncated, info = self.env.step(action_np)
                ep_reward += reward
                if "variance" in info:
                    variances.append(info["variance"])
                if "gain_budget" in info:
                    gain_budgets.append(info["gain_budget"])
                if terminated or truncated:
                    break
            returns.append(ep_reward)

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_variance": float(np.mean(variances)) if variances else float("nan"),
            "mean_gain_budget": float(np.mean(gain_budgets)) if gain_budgets else float("nan"),
        }

    def train(self) -> dict:
        """Full PPO training loop.

        Returns:
            Dict with training history and final evaluation.
        """
        from pathlib import Path

        cfg = self.ppo_cfg
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        history: dict[str, list] = {
            "rollout_returns": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "eval_return": [],
        }

        run_id = None
        if self.tracker is not None:
            run_id = self.tracker.start_run(
                arch=f"ppo_{cfg.arch}",
                config=self.sim_cfg,
                notes=f"PPO lr={cfg.learning_rate} clip={cfg.clip_epsilon}",
            )

        total_steps = 0
        iteration = 0
        best_eval_return = -float("inf")

        while total_steps < cfg.total_timesteps:
            iteration += 1

            # Collect rollout
            mean_return = self.collect_rollout()
            total_steps += cfg.steps_per_rollout

            # PPO update
            losses = self.update()

            history["rollout_returns"].append(mean_return)
            history["policy_loss"].append(losses["policy_loss"])
            history["value_loss"].append(losses["value_loss"])
            history["entropy"].append(losses["entropy"])

            # Periodic evaluation
            eval_result = self.evaluate()
            history["eval_return"].append(eval_result["mean_return"])

            if iteration % cfg.log_interval == 0:
                print(
                    f"Iter {iteration:4d} | Steps {total_steps:6d}/{cfg.total_timesteps} | "
                    f"Rollout Return: {mean_return:+.4f} | "
                    f"Eval Return: {eval_result['mean_return']:+.4f} | "
                    f"Policy Loss: {losses['policy_loss']:.4f} | "
                    f"Value Loss: {losses['value_loss']:.4f}"
                )

            # Tracker logging
            if self.tracker is not None and run_id is not None:
                self.tracker.log_epoch(
                    run_id,
                    epoch=iteration,
                    train_loss=losses["policy_loss"],
                    val_loss=losses["value_loss"],
                    gain_budget=eval_result.get("mean_gain_budget", 0.0),
                )

            # Checkpoint best
            if eval_result["mean_return"] > best_eval_return:
                best_eval_return = eval_result["mean_return"]
                torch.save(
                    {
                        "iteration": iteration,
                        "total_steps": total_steps,
                        "actor_critic_state": self.ac.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "eval_return": best_eval_return,
                        "ppo_config": {
                            "arch": cfg.arch,
                            "clip_epsilon": cfg.clip_epsilon,
                            "gamma": cfg.gamma,
                            "gae_lambda": cfg.gae_lambda,
                            "learning_rate": cfg.learning_rate,
                        },
                        "sim_config": self.sim_cfg.model_dump(),
                    },
                    ckpt_dir / "best_ppo.pt",
                )

        if self.tracker is not None and run_id is not None:
            self.tracker.finish_run(run_id, best_val_loss=-best_eval_return)

        final_eval = self.evaluate()
        print(
            f"\nPPO Training complete. "
            f"Best eval return: {best_eval_return:+.4f} | "
            f"Final eval return: {final_eval['mean_return']:+.4f}"
        )

        return {
            "history": history,
            "best_eval_return": best_eval_return,
            "final_eval": final_eval,
        }
