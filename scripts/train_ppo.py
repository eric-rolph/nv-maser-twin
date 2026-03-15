"""
PPO training script for magnetic field shimming.

Usage:
    python scripts/train_ppo.py [--timesteps 50000] [--arch cnn] [--seed 42]
    python scripts/train_ppo.py --timesteps 100000 --arch mlp --eval-episodes 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nv_maser.config import SimConfig
from nv_maser.rl.ppo import PPOConfig, PPOTrainer
from nv_maser.tracking.tracker import ExperimentTracker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPO training for magnetic field shimming.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--steps-per-rollout", type=int, default=2048, dest="steps_per_rollout")
    parser.add_argument("--arch", choices=["cnn", "mlp", "lstm"], default="cnn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2, dest="clip_epsilon")
    parser.add_argument("--eval-episodes", type=int, default=10, dest="eval_episodes")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/rl", dest="checkpoint_dir")
    parser.add_argument("--no-tracker", action="store_true", dest="no_tracker")
    args = parser.parse_args()

    sim_cfg = SimConfig()
    ppo_cfg = PPOConfig(
        total_timesteps=args.timesteps,
        steps_per_rollout=args.steps_per_rollout,
        arch=args.arch,
        seed=args.seed,
        learning_rate=args.lr,
        clip_epsilon=args.clip_epsilon,
        eval_episodes=args.eval_episodes,
        checkpoint_dir=args.checkpoint_dir,
    )

    tracker = None if args.no_tracker else ExperimentTracker()

    trainer = PPOTrainer(sim_config=sim_cfg, ppo_config=ppo_cfg, tracker=tracker)
    results = trainer.train()

    print(f"\nFinal eval: {results['final_eval']}")


if __name__ == "__main__":
    main()
