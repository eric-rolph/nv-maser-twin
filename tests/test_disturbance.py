"""Tests for the DisturbanceGenerator."""
import numpy as np
import pytest

from nv_maser.config import DisturbanceConfig, GridConfig
from nv_maser.physics.grid import SpatialGrid
from nv_maser.physics.disturbance import DisturbanceGenerator


@pytest.fixture
def grid() -> SpatialGrid:
    return SpatialGrid(GridConfig())


@pytest.fixture
def seeded_gen(grid: SpatialGrid) -> DisturbanceGenerator:
    cfg = DisturbanceConfig(seed=42)
    return DisturbanceGenerator(grid, cfg)


def test_output_shape(seeded_gen: DisturbanceGenerator) -> None:
    """Single disturbance is (size, size)."""
    d = seeded_gen.generate()
    assert d.shape == (64, 64)


def test_batch_shape(grid: SpatialGrid) -> None:
    """Batch disturbance is (batch, size, size)."""
    gen = DisturbanceGenerator(grid, DisturbanceConfig(seed=0))
    batch = gen.generate_batch(10)
    assert batch.shape == (10, 64, 64)


def test_deterministic_with_seed(grid: SpatialGrid) -> None:
    """Same seed → same output."""
    cfg = DisturbanceConfig(seed=99)
    g1 = DisturbanceGenerator(grid, cfg)
    g2 = DisturbanceGenerator(grid, cfg)
    assert np.allclose(g1.generate(), g2.generate())


def test_temporal_evolution(grid: SpatialGrid) -> None:
    """Different time → different output when drift_rate > 0."""
    cfg = DisturbanceConfig(seed=7, temporal_drift_rate=1.0)
    gen = DisturbanceGenerator(grid, cfg)
    d0 = gen.generate(t=0.0)
    d10 = gen.generate(t=10.0)
    assert not np.allclose(d0, d10)


def test_amplitude_bounds(grid: SpatialGrid) -> None:
    """Output magnitude bounded by theoretical max."""
    cfg = DisturbanceConfig(
        seed=0, num_modes=5, max_amplitude_tesla=0.005
    )
    gen = DisturbanceGenerator(grid, cfg)
    d = gen.generate()
    theoretical_max = cfg.max_amplitude_tesla * cfg.num_modes
    assert np.max(np.abs(d)) <= theoretical_max + 1e-6


def test_output_dtype(seeded_gen: DisturbanceGenerator) -> None:
    """Output is float32."""
    assert seeded_gen.generate().dtype == np.float32


def test_spatial_smoothness(grid: SpatialGrid) -> None:
    """Low-frequency content dominates (check via FFT power spectrum)."""
    cfg = DisturbanceConfig(
        seed=0, num_modes=5, min_spatial_freq=0.5, max_spatial_freq=4.0
    )
    gen = DisturbanceGenerator(grid, cfg)
    d = gen.generate()

    # 2D FFT — power spectrum
    fft = np.fft.fft2(d)
    power = np.abs(np.fft.fftshift(fft)) ** 2

    h, w = power.shape
    cy, cx = h // 2, w // 2
    low_freq_radius = h // 8  # inner region ≈ low freq

    # Mask: low-freq vs high-freq
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    low_mask = dist <= low_freq_radius
    high_mask = dist > h // 3

    low_power = float(power[low_mask].sum())
    high_power = float(power[high_mask].sum())

    # Low-frequency region should contain more power
    assert low_power > high_power


# ═══════════════════════════════════════════════════════════════════════
# Mains hum disturbance
# ═══════════════════════════════════════════════════════════════════════


def test_mains_hum_adds_oscillation(grid: SpatialGrid) -> None:
    """Mains hum should add a sinusoidal component at the configured frequency."""
    cfg = DisturbanceConfig(
        seed=0,
        num_modes=1,
        max_amplitude_tesla=0.0,  # disable spatial modes
        mains_hum_enabled=True,
        mains_hum_frequency_hz=50.0,
        mains_hum_amplitude_tesla=1e-4,
    )
    gen = DisturbanceGenerator(grid, cfg)
    # At t=0 → sin(0) = 0
    d0 = gen.generate(t=0.0)
    assert np.abs(d0).max() < 1e-6

    # At t = 1/(4*50) → sin(π/2) = 1 → peak = 1e-4
    d_quarter = gen.generate(t=1.0 / (4 * 50.0))
    assert np.isclose(d_quarter.mean(), 1e-4, rtol=0.01)


def test_mains_hum_disabled_by_default(grid: SpatialGrid) -> None:
    """Mains hum off by default — generate at different times should only vary by spatial drift."""
    cfg = DisturbanceConfig(seed=0, temporal_drift_rate=0.0)
    gen = DisturbanceGenerator(grid, cfg)
    d0 = gen.generate(t=0.0)
    d1 = gen.generate(t=0.005)  # 5ms =  quarter cycle of 50Hz
    np.testing.assert_array_equal(d0, d1)  # No temporal change


# ═══════════════════════════════════════════════════════════════════════
# DC drift disturbance
# ═══════════════════════════════════════════════════════════════════════


def test_dc_drift_grows_with_time(grid: SpatialGrid) -> None:
    """DC drift adds a linearly growing offset."""
    cfg = DisturbanceConfig(
        seed=0,
        num_modes=1,
        max_amplitude_tesla=0.0,
        temporal_drift_rate=0.0,
        dc_drift_enabled=True,
        dc_drift_rate_tesla_per_s=1e-5,
    )
    gen = DisturbanceGenerator(grid, cfg)
    d0 = gen.generate(t=0.0)
    d10 = gen.generate(t=10.0)
    offset = float(d10.mean() - d0.mean())
    expected = 1e-5 * 10.0
    assert np.isclose(offset, expected, rtol=0.01)


# ═══════════════════════════════════════════════════════════════════════
# Transient spikes
# ═══════════════════════════════════════════════════════════════════════


def test_transient_state_cleared_on_randomize(grid: SpatialGrid) -> None:
    """randomize() should clear active transients."""
    cfg = DisturbanceConfig(
        seed=0,
        transient_enabled=True,
        transient_rate_hz=1.0,
        transient_amplitude_tesla=1e-3,
    )
    gen = DisturbanceGenerator(grid, cfg)
    # Force a transient manually
    pattern = np.ones((64, 64), dtype=np.float32)
    gen._active_transients.append((0.0, 1e-3, pattern))
    assert len(gen._active_transients) == 1
    gen.randomize()
    assert len(gen._active_transients) == 0


def test_transient_disabled_by_default(grid: SpatialGrid) -> None:
    """With transient_enabled=False, no transients spawn."""
    cfg = DisturbanceConfig(seed=0, transient_enabled=False)
    gen = DisturbanceGenerator(grid, cfg)
    for _ in range(100):
        gen.generate(t=float(_) * 0.001)
    assert len(gen._active_transients) == 0
