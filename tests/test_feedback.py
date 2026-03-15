"""Tests for feedback hardware models: Hall sensors, DAC, coil dynamics."""
import numpy as np
import pytest

from nv_maser.config import FeedbackConfig, CoilConfig, GridConfig, SimConfig
from nv_maser.physics.grid import SpatialGrid
from nv_maser.physics.feedback import (
    HallSensorArray,
    quantize_currents,
    CoilDynamics,
)


@pytest.fixture
def grid() -> SpatialGrid:
    return SpatialGrid(GridConfig(size=32, physical_extent_mm=10.0, active_zone_fraction=0.6))


@pytest.fixture
def feedback_config() -> FeedbackConfig:
    return FeedbackConfig()


# ═══════════════════════════════════════════════════════════════════
#  Hall Sensor Array
# ═══════════════════════════════════════════════════════════════════


def test_sensor_count(grid: SpatialGrid) -> None:
    """Number of sensor positions matches config."""
    cfg = FeedbackConfig(num_sensors=9)
    sensors = HallSensorArray(grid, cfg, seed=0)
    assert len(sensors.sensor_x) == 9
    assert len(sensors.sensor_y) == 9
    assert len(sensors._indices) == 9


def test_sensor_positions_within_active_zone(grid: SpatialGrid) -> None:
    """All sensors placed inside the active zone."""
    cfg = FeedbackConfig(num_sensors=16)
    sensors = HallSensorArray(grid, cfg, seed=0)
    half_active = (grid.extent * grid.active_fraction) / 2.0
    assert np.all(np.abs(sensors.sensor_x) <= half_active)
    assert np.all(np.abs(sensors.sensor_y) <= half_active)


def test_measure_clean_exact(grid: SpatialGrid) -> None:
    """Clean measurement returns exact grid values at sensor locations."""
    cfg = FeedbackConfig(num_sensors=4)
    sensors = HallSensorArray(grid, cfg, seed=0)

    # Create a field with a known gradient
    field = np.linspace(0.04, 0.06, grid.size * grid.size).reshape(
        grid.size, grid.size
    ).astype(np.float32)

    clean = sensors.measure_clean(field)
    assert clean.shape == (4,)
    for k, (i, j) in enumerate(sensors._indices):
        assert clean[k] == pytest.approx(field[i, j], abs=1e-7)


def test_measure_adds_noise(grid: SpatialGrid) -> None:
    """Noisy measurement differs from clean by approximately sensor_noise_tesla."""
    noise_level = 1e-5  # exaggerate for statistical test
    cfg = FeedbackConfig(num_sensors=64, sensor_noise_tesla=noise_level)
    sensors = HallSensorArray(grid, cfg, seed=42)

    field = np.full((grid.size, grid.size), 0.05, dtype=np.float32)

    # Run multiple measurements to get good statistics
    diffs = []
    for _ in range(50):
        noisy = sensors.measure(field)
        clean = sensors.measure_clean(field)
        diffs.append(noisy - clean)
    all_diffs = np.concatenate(diffs)

    # RMS noise should be close to sensor_noise_tesla
    rms = float(np.sqrt(np.mean(all_diffs**2)))
    assert rms == pytest.approx(noise_level, rel=0.3)
    # Mean should be near zero (unbiased)
    assert abs(float(np.mean(all_diffs))) < noise_level * 0.3


def test_measure_zero_noise(grid: SpatialGrid) -> None:
    """With zero noise, measure == measure_clean."""
    cfg = FeedbackConfig(num_sensors=4, sensor_noise_tesla=0.0)
    sensors = HallSensorArray(grid, cfg, seed=0)

    field = np.full((grid.size, grid.size), 0.05, dtype=np.float32)
    noisy = sensors.measure(field)
    clean = sensors.measure_clean(field)
    np.testing.assert_array_equal(noisy, clean)


def test_measure_reproducible_with_seed(grid: SpatialGrid) -> None:
    """Same seed → same noise realization."""
    cfg = FeedbackConfig(num_sensors=4, sensor_noise_tesla=1e-6)
    field = np.full((grid.size, grid.size), 0.05, dtype=np.float32)

    s1 = HallSensorArray(grid, cfg, seed=99)
    s2 = HallSensorArray(grid, cfg, seed=99)
    np.testing.assert_array_equal(s1.measure(field), s2.measure(field))


# ═══════════════════════════════════════════════════════════════════
#  DAC Quantization
# ═══════════════════════════════════════════════════════════════════


def test_quantize_zero() -> None:
    """Zero current stays zero after quantization."""
    currents = np.array([0.0], dtype=np.float32)
    q = quantize_currents(currents, max_current=1.0, dac_bits=16)
    assert q[0] == pytest.approx(0.0, abs=1e-7)


def test_quantize_round_trip_small() -> None:
    """Values on exact DAC steps are unchanged."""
    max_curr = 1.0
    bits = 16
    step = 2 * max_curr / (2**bits)
    # Create an exact step value
    exact = np.array([5 * step], dtype=np.float32)
    q = quantize_currents(exact, max_curr, bits)
    assert q[0] == pytest.approx(exact[0], rel=1e-3)


def test_quantize_resolution() -> None:
    """Quantization error bounded by half LSB."""
    max_curr = 1.0
    bits = 12  # coarser for clear effect
    step = 2 * max_curr / (2**bits)

    rng = np.random.default_rng(42)
    currents = rng.uniform(-max_curr, max_curr, 1000).astype(np.float32)
    q = quantize_currents(currents, max_curr, bits)
    errors = np.abs(currents - q)
    assert np.all(errors <= step / 2 + 1e-7)


def test_quantize_clamps() -> None:
    """Currents exceeding max_current are clamped."""
    currents = np.array([-5.0, 5.0], dtype=np.float32)
    q = quantize_currents(currents, max_current=1.0, dac_bits=16)
    assert np.all(np.abs(q) <= 1.0)


def test_quantize_higher_bits_less_error() -> None:
    """Higher DAC resolution → smaller quantization error."""
    rng = np.random.default_rng(42)
    currents = rng.uniform(-1.0, 1.0, 500).astype(np.float32)

    q12 = quantize_currents(currents, 1.0, 12)
    q16 = quantize_currents(currents, 1.0, 16)

    err12 = float(np.sqrt(np.mean((currents - q12) ** 2)))
    err16 = float(np.sqrt(np.mean((currents - q16) ** 2)))
    assert err16 < err12


def test_quantize_preserves_shape() -> None:
    """Output shape matches input shape."""
    currents = np.zeros((3, 8), dtype=np.float32)
    q = quantize_currents(currents, 1.0, 16)
    assert q.shape == (3, 8)


# ═══════════════════════════════════════════════════════════════════
#  Coil L/R Dynamics
# ═══════════════════════════════════════════════════════════════════


def test_coil_dynamics_reset(feedback_config: FeedbackConfig) -> None:
    """After reset, currents are all zero."""
    cd = CoilDynamics(feedback_config)
    cd.reset(8)
    assert cd.current_state is not None
    np.testing.assert_array_equal(cd.current_state, np.zeros(8))


def test_coil_dynamics_step_toward_target(feedback_config: FeedbackConfig) -> None:
    """After one time constant, current reaches ~63.2% of target."""
    cd = CoilDynamics(feedback_config)
    cd.reset(1)
    tau = feedback_config.coil_time_constant_us  # 20 μs

    target = np.array([1.0], dtype=np.float32)
    actual = cd.step(target, dt_us=tau)
    # After 1τ: I = 1 - exp(-1) ≈ 0.632
    assert actual[0] == pytest.approx(1 - np.exp(-1), abs=0.01)


def test_coil_dynamics_converges(feedback_config: FeedbackConfig) -> None:
    """After many time constants, current ≈ target."""
    cd = CoilDynamics(feedback_config)
    cd.reset(1)
    tau = feedback_config.coil_time_constant_us

    target = np.array([0.5], dtype=np.float32)
    actual = cd.step(target, dt_us=10 * tau)  # 10τ → 99.995%
    assert actual[0] == pytest.approx(0.5, abs=1e-3)


def test_coil_dynamics_zero_inductance() -> None:
    """Zero inductance → instant settling."""
    cfg = FeedbackConfig(coil_inductance_uh=0.0)
    cd = CoilDynamics(cfg)
    cd.reset(4)
    target = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    actual = cd.step(target, dt_us=1.0)
    np.testing.assert_allclose(actual, target, atol=1e-7)


def test_coil_dynamics_multiple_steps(feedback_config: FeedbackConfig) -> None:
    """Multiple steps monotonically approach target."""
    cd = CoilDynamics(feedback_config)
    cd.reset(1)
    target = np.array([1.0], dtype=np.float32)

    prev = 0.0
    for _ in range(20):
        actual = cd.step(target, dt_us=5.0)
        assert actual[0] > prev  # strictly increasing toward target
        prev = actual[0]


def test_settling_fraction(feedback_config: FeedbackConfig) -> None:
    """settling_fraction at 0 is 0, at large t is ~1."""
    cd = CoilDynamics(feedback_config)
    assert cd.settling_fraction(0.0) == pytest.approx(0.0)
    assert cd.settling_fraction(1e6) == pytest.approx(1.0, abs=1e-6)
    tau = feedback_config.coil_time_constant_us
    assert cd.settling_fraction(tau) == pytest.approx(1 - np.exp(-1), abs=0.01)


def test_coil_time_constant() -> None:
    """L/R time constant = inductance / resistance."""
    cfg = FeedbackConfig(coil_inductance_uh=200.0, coil_resistance_ohm=10.0)
    assert cfg.coil_time_constant_us == pytest.approx(20.0)


def test_total_loop_latency() -> None:
    """Total latency = computation + DAC settling."""
    cfg = FeedbackConfig(computation_latency_us=100.0, dac_settling_time_us=10.0)
    assert cfg.total_loop_latency_us == pytest.approx(110.0)
