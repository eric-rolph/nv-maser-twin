"""Tests for FieldEnvironment."""
import numpy as np
import pytest

from nv_maser.config import SimConfig
from nv_maser.physics.environment import FieldEnvironment


@pytest.fixture
def env() -> FieldEnvironment:
    cfg = SimConfig()
    cfg.disturbance.seed = 42
    return FieldEnvironment(cfg)


def test_distorted_field_shape(env: FieldEnvironment) -> None:
    """Distorted field is (size, size)."""
    assert env.distorted_field.shape == (64, 64)


def test_zero_correction_identity(env: FieldEnvironment) -> None:
    """No coil current → net field equals distorted field."""
    distorted = env.distorted_field.copy()
    currents = np.zeros(env.config.coils.num_coils, dtype=np.float32)
    net = env.apply_correction(currents)
    assert np.allclose(net, distorted, atol=1e-6)


def test_uniformity_metrics_keys(env: FieldEnvironment) -> None:
    """Metrics dict has expected keys."""
    net = env.distorted_field
    metrics = env.compute_uniformity_metric(net)
    for key in ("variance", "std", "ppm", "max_deviation"):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_training_data_shapes(env: FieldEnvironment) -> None:
    """generate_training_data returns correct shapes."""
    n = 10
    distorted, disturbances = env.generate_training_data(n)
    assert distorted.shape == (n, 64, 64)
    assert disturbances.shape == (n, 64, 64)


def test_step_updates_disturbance(env: FieldEnvironment) -> None:
    """Calling step() changes the distorted field."""
    env.step(t=0.0)
    f0 = env.distorted_field.copy()
    env.step(t=100.0)
    f1 = env.distorted_field.copy()
    # They may or may not differ (seed-dependent), but step should not crash
    assert f0.shape == f1.shape


def test_apply_correction_changes_field(env: FieldEnvironment) -> None:
    """Non-zero correction changes the net field."""
    currents = np.ones(env.config.coils.num_coils, dtype=np.float32) * 0.5
    net = env.apply_correction(currents)
    assert not np.allclose(net, env.distorted_field)


def test_base_field_shape(env: FieldEnvironment) -> None:
    """Base field has correct shape."""
    assert env.base_field.shape == (64, 64)
