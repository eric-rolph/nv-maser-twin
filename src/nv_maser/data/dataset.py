"""
Dataset builder for field shimming training data.

Generates (distorted_field, target_field) pairs and caches them to disk
with a content-addressed filename based on config hash.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from ..config import SimConfig
from ..physics.environment import FieldEnvironment
from ..physics.disturbance import DisturbanceGenerator

logger = logging.getLogger("nv_maser.data")


def _config_hash(config: SimConfig) -> str:
    """
    Compute a short hash of the config fields that affect dataset content.
    Only grid, disturbance, field, and coil configs matter — not training params.
    """
    relevant = {
        "grid": config.grid.model_dump(),
        "disturbance": config.disturbance.model_dump(),
        "field": config.field.model_dump(),
        "coils": config.coils.model_dump(),
    }
    # Deterministic JSON → SHA256 → first 12 hex chars
    blob = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def build_dataset(
    config: SimConfig,
    num_samples: int,
    cache_dir: Path | str = "datasets/",
    force_rebuild: bool = False,
) -> TensorDataset:
    """
    Build or load a cached (distorted_field, base_field) TensorDataset.

    Cache filename: datasets/shim_N{num_samples}_C{config_hash}.npz

    Args:
        config: simulation configuration
        num_samples: number of (distorted, base) pairs to generate
        cache_dir: directory to store .npz cache files
        force_rebuild: if True, regenerate even if cache exists

    Returns:
        TensorDataset of (distorted_field_tensor, base_field_tensor)
        where distorted shape is (N, 1, size, size) and base same shape.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    config_hash = _config_hash(config)
    cache_file = cache_dir / f"shim_N{num_samples}_C{config_hash}.npz"

    if cache_file.exists() and not force_rebuild:
        logger.info("Loading dataset from cache: %s", cache_file)
        data = np.load(cache_file)
        distorted = torch.from_numpy(data["distorted"])
        base = torch.from_numpy(data["base"])
        logger.info("Loaded %d samples from cache", len(distorted))
        return TensorDataset(distorted, base)

    logger.info("Building dataset: %d samples (hash=%s)", num_samples, config_hash)
    env = FieldEnvironment(config)
    disturbance_gen = DisturbanceGenerator(env.grid, config.disturbance)

    size = config.grid.size
    distorted_arr = np.empty((num_samples, 1, size, size), dtype=np.float32)
    base_arr = np.empty((num_samples, 1, size, size), dtype=np.float32)

    # Get base field once (it's deterministic)
    base_field = env.base_field.astype(np.float32)  # (size, size)

    log_every = max(1, num_samples // 10)
    for i in range(num_samples):
        disturbance_gen.randomize()
        disturbance = disturbance_gen.generate(t=0.0)
        distorted_arr[i, 0] = base_field + disturbance
        base_arr[i, 0] = base_field
        if (i + 1) % log_every == 0:
            logger.info(
                "  Generated %d/%d samples (%.0f%%)",
                i + 1,
                num_samples,
                100 * (i + 1) / num_samples,
            )

    logger.info("Saving dataset to cache: %s", cache_file)
    np.savez_compressed(cache_file, distorted=distorted_arr, base=base_arr)

    distorted = torch.from_numpy(distorted_arr)
    base = torch.from_numpy(base_arr)
    return TensorDataset(distorted, base)
