"""
Tests for src/nv_maser/data/dataset.py
"""
import pytest

from nv_maser.config import SimConfig, DisturbanceConfig
from nv_maser.data.dataset import build_dataset, _config_hash


def test_build_dataset_shape(tmp_path):
    """Dataset has correct length and tensor shapes."""
    config = SimConfig()
    dataset = build_dataset(config, num_samples=20, cache_dir=tmp_path)

    assert len(dataset) == 20

    distorted, base = dataset[0]
    assert distorted.shape == (1, 64, 64), f"Unexpected distorted shape: {distorted.shape}"
    assert base.shape == (1, 64, 64), f"Unexpected base shape: {base.shape}"


def test_dataset_cache_hit(tmp_path):
    """Second call with same cache_dir loads from .npz, returns same length."""
    config = SimConfig()

    ds1 = build_dataset(config, num_samples=20, cache_dir=tmp_path)

    # Verify the cache file was written
    cache_files = list(tmp_path.glob("*.npz"))
    assert len(cache_files) == 1

    ds2 = build_dataset(config, num_samples=20, cache_dir=tmp_path)

    assert len(ds1) == len(ds2) == 20
    # No new file should have been created
    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_config_hash_changes_with_params():
    """Different disturbance.num_modes produces different config hash."""
    config1 = SimConfig()
    config2 = SimConfig(disturbance=DisturbanceConfig(num_modes=10))

    assert config1.disturbance.num_modes != config2.disturbance.num_modes
    assert _config_hash(config1) != _config_hash(config2)


def test_force_rebuild(tmp_path):
    """force_rebuild=True regenerates dataset and overwrites cache."""
    config = SimConfig()

    ds1 = build_dataset(config, num_samples=20, cache_dir=tmp_path)
    cache_files_before = list(tmp_path.glob("*.npz"))
    mtime_before = cache_files_before[0].stat().st_mtime

    # Small sleep to ensure mtime would differ if file is rewritten
    import time
    time.sleep(0.05)

    ds2 = build_dataset(config, num_samples=20, cache_dir=tmp_path, force_rebuild=True)

    cache_files_after = list(tmp_path.glob("*.npz"))
    mtime_after = cache_files_after[0].stat().st_mtime

    assert len(ds2) == 20
    assert mtime_after > mtime_before, "Cache file should have been rewritten"
