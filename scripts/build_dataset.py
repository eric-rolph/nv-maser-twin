"""
Pre-build and cache a training dataset.

Usage:
    python scripts/build_dataset.py --samples 10000
    python scripts/build_dataset.py --samples 50000 --cache-dir datasets/ --force
"""
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

from nv_maser.config import SimConfig
from nv_maser.data.dataset import build_dataset


def main():
    parser = argparse.ArgumentParser(description="Build and cache shimming dataset")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--cache-dir", default="datasets/")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if cache exists")
    parser.add_argument("--arch", default="cnn", help="Architecture (affects nothing here, for logging)")
    args = parser.parse_args()

    config = SimConfig()
    dataset = build_dataset(config, args.samples, cache_dir=args.cache_dir, force_rebuild=args.force)
    print(f"Dataset ready: {len(dataset)} samples")


if __name__ == "__main__":
    main()
