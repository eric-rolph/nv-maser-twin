"""Tests for closed-loop shimming simulation."""
import numpy as np
import pytest

from nv_maser.config import SimConfig
from nv_maser.physics.closed_loop import (
    ClosedLoopSimulator,
    ClosedLoopResult,
    LoopStepResult,
)


@pytest.fixture
def sim_config() -> SimConfig:
    """Small config for fast tests."""
    return SimConfig(
        grid={"size": 16, "physical_extent_mm": 10.0, "active_zone_fraction": 0.6},
        disturbance={"seed": 42, "num_modes": 3, "max_amplitude_tesla": 0.001},
        coils={"num_coils": 8, "max_current_amps": 1.0, "field_scale_factor": 0.005},
    )


def zero_controller(field: np.ndarray) -> np.ndarray:
    """No correction — outputs zero currents."""
    return np.zeros(8, dtype=np.float32)


def constant_controller(field: np.ndarray) -> np.ndarray:
    """Always output 0.1A on all coils."""
    return np.full(8, 0.1, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
#  Basic simulation runs
# ═══════════════════════════════════════════════════════════════════


def test_sim_runs_without_error(sim_config: SimConfig) -> None:
    """Simulator completes a short run."""
    sim = ClosedLoopSimulator(sim_config, zero_controller, seed=0)
    result = sim.run(duration_us=5000.0)
    assert result.num_steps == 5  # 5000/1000


def test_sim_returns_expected_steps(sim_config: SimConfig) -> None:
    """Number of steps matches duration / period."""
    sim = ClosedLoopSimulator(sim_config, zero_controller, seed=0)
    result = sim.run(duration_us=10_000.0)
    expected = int(10_000 / sim_config.feedback.control_loop_period_us)
    assert result.num_steps == expected


def test_step_result_fields(sim_config: SimConfig) -> None:
    """Each step has all expected data."""
    sim = ClosedLoopSimulator(sim_config, zero_controller, seed=0)
    result = sim.run(duration_us=1000.0)
    assert result.num_steps >= 1

    step = result.steps[0]
    assert isinstance(step, LoopStepResult)
    assert step.time_us == pytest.approx(0.0)
    assert step.commanded_currents.shape == (8,)
    assert step.quantized_currents.shape == (8,)
    assert step.actual_currents.shape == (8,)
    assert step.sensor_readings.shape == (sim_config.feedback.num_sensors,)
    assert step.field_variance >= 0
    assert step.field_std >= 0
    assert isinstance(step.gain_budget, float)
    assert isinstance(step.masing, bool)


# ═══════════════════════════════════════════════════════════════════
#  Zero controller baseline
# ═══════════════════════════════════════════════════════════════════


def test_zero_controller_all_zero_currents(sim_config: SimConfig) -> None:
    """Zero controller → commanded and quantized currents are zero."""
    sim = ClosedLoopSimulator(sim_config, zero_controller, seed=0)
    result = sim.run(duration_us=3000.0)
    for step in result.steps:
        np.testing.assert_array_equal(step.commanded_currents, 0.0)
        np.testing.assert_array_equal(step.quantized_currents, 0.0)


def test_zero_controller_quantization_error_zero(sim_config: SimConfig) -> None:
    """No commands → no quantization error."""
    sim = ClosedLoopSimulator(sim_config, zero_controller, seed=0)
    result = sim.run(duration_us=3000.0)
    assert result.current_quantization_error_rms == pytest.approx(0.0, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════
#  Constant controller
# ═══════════════════════════════════════════════════════════════════


def test_constant_controller_has_actual_currents(sim_config: SimConfig) -> None:
    """Constant controller → actual currents settle toward commanded."""
    sim = ClosedLoopSimulator(sim_config, constant_controller, seed=0)
    result = sim.run(duration_us=5000.0)
    # After several iterations, actual should approach 0.1
    last = result.steps[-1]
    assert np.all(np.abs(last.actual_currents - 0.1) < 0.05)


def test_settling_error_decreases_over_time() -> None:
    """Settling error should decrease as coils approach steady state."""
    # Use slow coils (high inductance) so settling error is visible
    cfg = SimConfig(
        grid={"size": 16, "physical_extent_mm": 10.0, "active_zone_fraction": 0.6},
        disturbance={"seed": 42, "num_modes": 3, "max_amplitude_tesla": 0.001},
        coils={"num_coils": 8, "max_current_amps": 1.0, "field_scale_factor": 0.005},
        feedback={"coil_inductance_uh": 5000.0, "coil_resistance_ohm": 5.0},
    )
    sim = ClosedLoopSimulator(cfg, constant_controller, seed=0)
    result = sim.run(duration_us=10_000.0)

    # First step settling error should be larger than last
    first_err = float(np.sqrt(np.mean(
        (result.steps[0].quantized_currents - result.steps[0].actual_currents) ** 2
    )))
    last_err = float(np.sqrt(np.mean(
        (result.steps[-1].quantized_currents - result.steps[-1].actual_currents) ** 2
    )))
    assert first_err > 0, "First step should have settling error with slow coils"
    assert last_err < first_err


# ═══════════════════════════════════════════════════════════════════
#  Aggregate metrics
# ═══════════════════════════════════════════════════════════════════


def test_summary_keys(sim_config: SimConfig) -> None:
    """summary() returns all expected keys."""
    sim = ClosedLoopSimulator(sim_config, zero_controller, seed=0)
    result = sim.run(duration_us=3000.0)
    s = result.summary()
    expected_keys = {
        "num_steps",
        "mean_variance",
        "mean_gain_budget",
        "masing_fraction",
        "quantization_error_rms_amps",
        "settling_error_rms_amps",
    }
    assert set(s.keys()) == expected_keys


def test_masing_fraction_range(sim_config: SimConfig) -> None:
    """Masing fraction is always in [0, 1]."""
    sim = ClosedLoopSimulator(sim_config, zero_controller, seed=0)
    result = sim.run(duration_us=5000.0)
    assert 0.0 <= result.masing_fraction <= 1.0


def test_mean_gain_budget_range(sim_config: SimConfig) -> None:
    """Gain budget is in (0, 1]."""
    sim = ClosedLoopSimulator(sim_config, zero_controller, seed=0)
    result = sim.run(duration_us=3000.0)
    assert 0.0 < result.mean_gain_budget <= 1.0


def test_mean_variance_positive(sim_config: SimConfig) -> None:
    """Variance is non-negative."""
    sim = ClosedLoopSimulator(sim_config, zero_controller, seed=0)
    result = sim.run(duration_us=3000.0)
    assert result.mean_variance >= 0.0


# ═══════════════════════════════════════════════════════════════════
#  Empty result edge cases
# ═══════════════════════════════════════════════════════════════════


def test_empty_result_defaults() -> None:
    """ClosedLoopResult with no steps returns safe defaults."""
    r = ClosedLoopResult()
    assert r.num_steps == 0
    assert r.mean_variance == float("inf")
    assert r.mean_gain_budget == 0.0
    assert r.masing_fraction == 0.0
    assert r.current_quantization_error_rms == 0.0
    assert r.current_settling_error_rms == 0.0


# ═══════════════════════════════════════════════════════════════════
#  Hardware degradation effects
# ═══════════════════════════════════════════════════════════════════


def test_more_sensor_noise_doesnt_crash(sim_config: SimConfig) -> None:
    """High sensor noise still produces valid results."""
    sim_config.feedback.sensor_noise_tesla = 1e-3  # extreme noise
    sim = ClosedLoopSimulator(sim_config, zero_controller, seed=0)
    result = sim.run(duration_us=3000.0)
    assert result.num_steps > 0
    assert np.isfinite(result.mean_variance)


def test_coarse_dac_produces_quantization_error(sim_config: SimConfig) -> None:
    """Low DAC bits → measurable quantization error with nonzero controller."""
    sim_config.feedback.dac_bits = 8  # very coarse
    sim = ClosedLoopSimulator(sim_config, constant_controller, seed=0)
    result = sim.run(duration_us=3000.0)
    # With 8-bit DAC at ±1A, step = 7.8 mA. 0.1A should get some error.
    assert result.current_quantization_error_rms > 0


def test_seed_reproducibility(sim_config: SimConfig) -> None:
    """Same seed → identical results."""
    sim1 = ClosedLoopSimulator(sim_config, zero_controller, seed=42)
    sim2 = ClosedLoopSimulator(sim_config, zero_controller, seed=42)
    r1 = sim1.run(duration_us=3000.0)
    r2 = sim2.run(duration_us=3000.0)
    assert r1.mean_variance == pytest.approx(r2.mean_variance)
    assert r1.masing_fraction == r2.masing_fraction
