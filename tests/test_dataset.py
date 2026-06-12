"""
Tests for src/nv_maser/data/dataset.py
"""

from nv_maser.config import DisturbanceConfig, SimConfig
from nv_maser.data.dataset import _config_hash, build_dataset


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


def test_config_hash_changes_with_halbach():
    """Halbach toggles change the base field, so they must change the hash."""
    config1 = SimConfig()
    config2 = SimConfig(halbach={"enabled": True})

    assert _config_hash(config1) != _config_hash(config2)


def test_config_hash_tracks_reference_map_content(tmp_path):
    """Cache key follows the reference map's CONTENT, not just its path."""
    import numpy as np

    from nv_maser.calibration import FieldMap, save_field_map

    ax = np.linspace(-5.0, 5.0, 64, dtype=np.float32)

    def _save(b0_value: float) -> None:
        fm = FieldMap(
            b_z=np.full((64, 64), b0_value, dtype=np.float32),
            x_mm=ax,
            y_mm=ax.copy(),
            b0_nominal_tesla=0.05,
            active_radius_mm=3.0,
        )
        save_field_map(tmp_path / "ref.npz", fm)

    config = SimConfig(
        calibration={"reference_map_path": str(tmp_path / "ref.npz")}
    )

    no_map_hash = _config_hash(SimConfig())

    _save(0.05)
    hash_a = _config_hash(config)

    _save(0.0501)  # re-measured map, same path
    hash_b = _config_hash(config)

    assert no_map_hash != hash_a, "Setting a reference map must change the hash"
    assert hash_a != hash_b, "Rewriting the map file must change the hash"


def test_force_rebuild(tmp_path):
    """force_rebuild=True regenerates dataset and overwrites cache."""
    config = SimConfig()

    build_dataset(config, num_samples=20, cache_dir=tmp_path)
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
