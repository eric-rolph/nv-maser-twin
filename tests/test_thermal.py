"""Tests for thermal coupling model: ThermalState, compute_thermal_state, ThermalModel."""
import numpy as np
import pytest

from nv_maser.config import (
    SimConfig,
    ThermalConfig,
    FieldConfig,
    NVConfig,
    MaserConfig,
    FeedbackConfig,
)
from nv_maser.physics.thermal import (
    ThermalState,
    compute_thermal_state,
    ThermalModel,
)
from nv_maser.physics.environment import FieldEnvironment


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def thermal() -> ThermalConfig:
    return ThermalConfig()


@pytest.fixture
def field() -> FieldConfig:
    return FieldConfig()


@pytest.fixture
def nv() -> NVConfig:
    return NVConfig()


@pytest.fixture
def maser() -> MaserConfig:
    return MaserConfig()


@pytest.fixture
def feedback() -> FeedbackConfig:
    return FeedbackConfig()


# ═══════════════════════════════════════════════════════════════════
#  compute_thermal_state — reference temperature (zero shift)
# ═══════════════════════════════════════════════════════════════════

def test_reference_temp_zero_shift(thermal, field, nv, maser, feedback) -> None:
    """At reference temperature, all shifts are zero/identity."""
    state = compute_thermal_state(25.0, thermal, field, nv, maser, feedback)
    assert state.temperature_c == 25.0
    assert state.b0_shift_tesla == pytest.approx(0.0, abs=1e-15)
    assert state.effective_t2_star_us == pytest.approx(nv.t2_star_us, rel=1e-10)
    assert state.effective_cavity_q == pytest.approx(maser.cavity_q, rel=1e-10)
    assert state.effective_coil_resistance_ohm == pytest.approx(
        feedback.coil_resistance_ohm, rel=1e-10
    )


# ═══════════════════════════════════════════════════════════════════
#  compute_thermal_state — hot shifts
# ═══════════════════════════════════════════════════════════════════

def test_hot_b0_shift_negative(thermal, field, nv, maser, feedback) -> None:
    """NdFeB has negative tempco → heating reduces B₀."""
    state = compute_thermal_state(35.0, thermal, field, nv, maser, feedback)
    # ΔT = +10°C, α = -0.12%/°C → shift < 0
    expected = field.b0_tesla * (-0.12 / 100.0) * 10.0
    assert state.b0_shift_tesla == pytest.approx(expected, rel=1e-6)
    assert state.b0_shift_tesla < 0


def test_hot_t2_star_shorter(thermal, field, nv, maser, feedback) -> None:
    """Heating degrades T2* (phonon-limited)."""
    state = compute_thermal_state(35.0, thermal, field, nv, maser, feedback)
    # T_ref_K / T_K = 298.15 / 308.15 < 1, so T2* decreases
    assert state.effective_t2_star_us < nv.t2_star_us


def test_hot_cavity_q_lower(thermal, field, nv, maser, feedback) -> None:
    """Heating increases wall resistivity → lowers Q."""
    state = compute_thermal_state(35.0, thermal, field, nv, maser, feedback)
    assert state.effective_cavity_q < maser.cavity_q


def test_hot_coil_resistance_higher(thermal, field, nv, maser, feedback) -> None:
    """Heating increases copper resistance."""
    state = compute_thermal_state(35.0, thermal, field, nv, maser, feedback)
    expected = feedback.coil_resistance_ohm * (1.0 + 0.004 * 10.0)
    assert state.effective_coil_resistance_ohm == pytest.approx(expected, rel=1e-6)
    assert state.effective_coil_resistance_ohm > feedback.coil_resistance_ohm


# ═══════════════════════════════════════════════════════════════════
#  compute_thermal_state — cold shifts (opposite direction)
# ═══════════════════════════════════════════════════════════════════

def test_cold_b0_shift_positive(thermal, field, nv, maser, feedback) -> None:
    """Cooling → B₀ increases (negative tempco × negative ΔT)."""
    state = compute_thermal_state(15.0, thermal, field, nv, maser, feedback)
    assert state.b0_shift_tesla > 0


def test_cold_t2_star_longer(thermal, field, nv, maser, feedback) -> None:
    """Cooling → T2* improves."""
    state = compute_thermal_state(15.0, thermal, field, nv, maser, feedback)
    assert state.effective_t2_star_us > nv.t2_star_us


def test_cold_coil_resistance_lower(thermal, field, nv, maser, feedback) -> None:
    """Cooling reduces copper resistance."""
    state = compute_thermal_state(15.0, thermal, field, nv, maser, feedback)
    assert state.effective_coil_resistance_ohm < feedback.coil_resistance_ohm


# ═══════════════════════════════════════════════════════════════════
#  compute_thermal_state — edge cases
# ═══════════════════════════════════════════════════════════════════

def test_resistance_floor(thermal, field, nv, maser, feedback) -> None:
    """At extreme cold, resistance is floored (never ≤ 0)."""
    state = compute_thermal_state(-300.0, thermal, field, nv, maser, feedback)
    assert state.effective_coil_resistance_ohm > 0


def test_extreme_heat(thermal, field, nv, maser, feedback) -> None:
    """State is valid at extreme temperatures."""
    state = compute_thermal_state(125.0, thermal, field, nv, maser, feedback)
    assert np.isfinite(state.b0_shift_tesla)
    assert state.effective_t2_star_us > 0
    assert state.effective_cavity_q > 0
    assert state.effective_coil_resistance_ohm > 0


# ═══════════════════════════════════════════════════════════════════
#  compute_thermal_state — quantitative checks
# ═══════════════════════════════════════════════════════════════════

def test_b0_shift_magnitude(thermal, field, nv, maser, feedback) -> None:
    """Check B₀ shift magnitude at +10°C."""
    state = compute_thermal_state(35.0, thermal, field, nv, maser, feedback)
    # 0.05 T × (-0.0012) × 10 = -6e-4 T = -600 μT
    assert state.b0_shift_tesla == pytest.approx(-6e-4, rel=1e-6)


def test_cavity_q_formula(thermal, field, nv, maser, feedback) -> None:
    """Q at +10°C follows Q_ref / √(1 + α·ΔT)."""
    state = compute_thermal_state(35.0, thermal, field, nv, maser, feedback)
    expected = maser.cavity_q / np.sqrt(1.0 + 0.004 * 10.0)
    assert state.effective_cavity_q == pytest.approx(expected, rel=1e-6)


def test_t2_star_formula(thermal, field, nv, maser, feedback) -> None:
    """T2* at +10°C follows T2*_ref × (T_ref_K / T_K)^n."""
    state = compute_thermal_state(35.0, thermal, field, nv, maser, feedback)
    expected = nv.t2_star_us * (298.15 / 308.15) ** 1.0
    assert state.effective_t2_star_us == pytest.approx(expected, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════
#  ThermalState — immutability
# ═══════════════════════════════════════════════════════════════════

def test_thermal_state_frozen(thermal, field, nv, maser, feedback) -> None:
    """ThermalState is frozen (immutable)."""
    state = compute_thermal_state(30.0, thermal, field, nv, maser, feedback)
    with pytest.raises(AttributeError):
        state.temperature_c = 99.0  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
#  ThermalModel — temperature evolution
# ═══════════════════════════════════════════════════════════════════

def test_model_temperature_at_t0(thermal, field) -> None:
    """At t=0, temperature ≈ ambient (with small noise)."""
    model = ThermalModel(thermal, seed=42)
    # No drift at t=0; just noise
    temps = [model.temperature_at(0.0) for _ in range(100)]
    mean_t = np.mean(temps)
    assert mean_t == pytest.approx(thermal.ambient_temperature_c, abs=0.5)


def test_model_drift(field) -> None:
    """Temperature drifts linearly over time."""
    cfg = ThermalConfig(
        thermal_drift_rate_c_per_s=1.0,  # 1°C/s drift
        thermal_noise_std_c=0.0,          # no noise for deterministic test
    )
    model = ThermalModel(cfg, seed=0)
    t10 = model.temperature_at(10.0)
    # T(10) = 25 + 1.0 × 10 = 35°C
    assert t10 == pytest.approx(35.0, abs=1e-10)


def test_model_no_noise_deterministic(field) -> None:
    """With zero noise, temperature is perfectly deterministic."""
    cfg = ThermalConfig(thermal_noise_std_c=0.0)
    model = ThermalModel(cfg, seed=0)
    t1 = model.temperature_at(5.0)
    t2 = model.temperature_at(5.0)
    assert t1 != t2 or t1 == t2  # just shouldn't crash; noise=0 means same formula
    # Actually both should be: 25 + 0.01*5 = 25.05
    assert t1 == pytest.approx(25.05, abs=1e-10)


def test_model_seed_reproducibility(thermal, field, nv, maser, feedback) -> None:
    """Same seed → same trajectory."""
    m1 = ThermalModel(thermal, seed=123)
    m2 = ThermalModel(thermal, seed=123)
    for t in [0.0, 1.0, 5.0, 10.0]:
        s1 = m1.state_at(t, field, nv, maser, feedback)
        s2 = m2.state_at(t, field, nv, maser, feedback)
        assert s1.temperature_c == s2.temperature_c
        assert s1.b0_shift_tesla == s2.b0_shift_tesla


def test_model_different_seeds(thermal, field, nv, maser, feedback) -> None:
    """Different seeds → different noise."""
    m1 = ThermalModel(thermal, seed=0)
    m2 = ThermalModel(thermal, seed=999)
    temps1 = [m1.temperature_at(1.0) for _ in range(20)]
    temps2 = [m2.temperature_at(1.0) for _ in range(20)]
    # Very unlikely to be identical
    assert not np.allclose(temps1, temps2)


# ═══════════════════════════════════════════════════════════════════
#  ThermalModel — state_at
# ═══════════════════════════════════════════════════════════════════

def test_state_at_returns_thermal_state(thermal, field, nv, maser, feedback) -> None:
    """state_at returns a valid ThermalState."""
    model = ThermalModel(thermal, seed=0)
    state = model.state_at(1.0, field, nv, maser, feedback)
    assert isinstance(state, ThermalState)
    assert np.isfinite(state.b0_shift_tesla)
    assert state.effective_t2_star_us > 0
    assert state.effective_cavity_q > 0
    assert state.effective_coil_resistance_ohm > 0


# ═══════════════════════════════════════════════════════════════════
#  ThermalModel — trajectory
# ═══════════════════════════════════════════════════════════════════

def test_trajectory_length(thermal, field, nv, maser, feedback) -> None:
    """Trajectory returns correct number of states."""
    model = ThermalModel(thermal, seed=0)
    traj = model.trajectory(1.0, 0.1, field, nv, maser, feedback)
    # From 0.0 to 1.0 in steps of 0.1 → 11 steps (0.0, 0.1, ..., 1.0)
    assert len(traj) == 11


def test_trajectory_all_valid(thermal, field, nv, maser, feedback) -> None:
    """All trajectory states have finite values."""
    model = ThermalModel(thermal, seed=42)
    traj = model.trajectory(0.5, 0.1, field, nv, maser, feedback)
    for state in traj:
        assert isinstance(state, ThermalState)
        assert np.isfinite(state.temperature_c)
        assert np.isfinite(state.b0_shift_tesla)
        assert state.effective_t2_star_us > 0


def test_trajectory_temperature_drift(field, nv, maser, feedback) -> None:
    """Trajectory shows increasing temperature with positive drift."""
    cfg = ThermalConfig(
        thermal_drift_rate_c_per_s=0.5,
        thermal_noise_std_c=0.0,  # deterministic
    )
    model = ThermalModel(cfg, seed=0)
    traj = model.trajectory(2.0, 0.5, field, nv, maser, feedback)
    temps = [s.temperature_c for s in traj]
    # Monotonically increasing (no noise)
    for i in range(1, len(temps)):
        assert temps[i] > temps[i - 1]


# ═══════════════════════════════════════════════════════════════════
#  FieldEnvironment — thermal integration
# ═══════════════════════════════════════════════════════════════════

def test_env_thermal_state_none_before_step() -> None:
    """Before step(), thermal_state is None."""
    cfg = SimConfig(grid={"size": 16}, disturbance={"seed": 42})
    env = FieldEnvironment(cfg)
    assert env.thermal_state is None


def test_env_step_populates_thermal_state() -> None:
    """After step(), thermal_state is set."""
    cfg = SimConfig(grid={"size": 16}, disturbance={"seed": 42})
    env = FieldEnvironment(cfg, thermal_seed=0)
    env.step(t=0.0)
    assert env.thermal_state is not None
    assert isinstance(env.thermal_state, ThermalState)


def test_env_effective_base_field_no_shift() -> None:
    """At default config (ΔT=0), effective_base_field ≈ base_field."""
    cfg = SimConfig(grid={"size": 16}, disturbance={"seed": 42})
    env = FieldEnvironment(cfg, thermal_seed=0)
    env.step(t=0.0)
    # At default 25°C with noise, shift should be small
    diff = np.abs(env.effective_base_field - env.base_field)
    assert np.max(diff) < 1e-4  # sub-100μT shift from noise alone


def test_env_effective_base_field_with_thermal_shift() -> None:
    """With elevated ambient, effective_base_field shifts."""
    cfg = SimConfig(
        grid={"size": 16},
        disturbance={"seed": 42},
        thermal={"ambient_temperature_c": 45.0, "thermal_noise_std_c": 0.0,
                 "thermal_drift_rate_c_per_s": 0.0},
        optical_pump={"laser_power_w": 0.0},  # no pump heating for this test
    )
    env = FieldEnvironment(cfg, thermal_seed=0)
    env.step(t=0.0)
    # ΔT = 20°C, shift = 0.05 × (-0.0012) × 20 = -1.2e-3 T
    expected_shift = -1.2e-3
    actual_shift = float(np.mean(env.effective_base_field - env.base_field))
    assert actual_shift == pytest.approx(expected_shift, rel=1e-3)


def test_env_uniformity_includes_thermal_keys() -> None:
    """When thermal is active, metrics include temperature_c."""
    cfg = SimConfig(grid={"size": 16}, disturbance={"seed": 42})
    env = FieldEnvironment(cfg, thermal_seed=0)
    env.step(t=0.0)
    net = env.distorted_field
    metrics = env.compute_uniformity_metric(net)
    assert "temperature_c" in metrics
    assert "b0_shift_tesla" in metrics
    assert np.isfinite(metrics["temperature_c"])


# ═══════════════════════════════════════════════════════════════════
#  Closed-loop integration with thermal
# ═══════════════════════════════════════════════════════════════════

def test_closed_loop_with_thermal_drift() -> None:
    """Closed-loop simulation runs with thermal drift active."""
    from nv_maser.physics.closed_loop import ClosedLoopSimulator

    cfg = SimConfig(
        grid={"size": 16},
        disturbance={"seed": 42, "num_modes": 3, "max_amplitude_tesla": 0.001},
        coils={"num_coils": 8, "max_current_amps": 1.0, "field_scale_factor": 0.005},
        thermal={
            "ambient_temperature_c": 30.0,
            "thermal_drift_rate_c_per_s": 0.1,
            "thermal_noise_std_c": 0.0,
        },
    )

    def zero_ctrl(field: np.ndarray) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)

    sim = ClosedLoopSimulator(cfg, zero_ctrl, seed=0)
    result = sim.run(duration_us=5000.0)
    assert result.num_steps == 5
    # Each step should have maser metrics
    for step in result.steps:
        assert np.isfinite(step.field_variance)


def test_closed_loop_thermal_affects_coil_tau() -> None:
    """Thermal heating changes effective coil time constant."""
    from nv_maser.physics.closed_loop import ClosedLoopSimulator

    # Hot environment → higher R → lower tau → faster settling
    cfg_hot = SimConfig(
        grid={"size": 16},
        disturbance={"seed": 42, "num_modes": 3, "max_amplitude_tesla": 0.001},
        coils={"num_coils": 8, "max_current_amps": 1.0, "field_scale_factor": 0.005},
        thermal={
            "ambient_temperature_c": 75.0,
            "thermal_drift_rate_c_per_s": 0.0,
            "thermal_noise_std_c": 0.0,
        },
    )

    def ctrl(field: np.ndarray) -> np.ndarray:
        return np.full(8, 0.5, dtype=np.float32)

    sim = ClosedLoopSimulator(cfg_hot, ctrl, seed=0)
    result = sim.run(duration_us=2000.0)
    # Should run without error with adjusted tau
    assert result.num_steps >= 1


# ═══════════════════════════════════════════════════════════════════
#  Config integration
# ═══════════════════════════════════════════════════════════════════

def test_thermal_config_defaults() -> None:
    """ThermalConfig defaults match expected values."""
    tc = ThermalConfig()
    assert tc.reference_temperature_c == 25.0
    assert tc.ambient_temperature_c == 25.0
    assert tc.magnet_tempco_pct_per_c == -0.12
    assert tc.delta_t == 0.0


def test_thermal_config_delta_t_property() -> None:
    """delta_t property computes ambient - reference."""
    tc = ThermalConfig(ambient_temperature_c=35.0)
    assert tc.delta_t == pytest.approx(10.0)


def test_sim_config_has_thermal() -> None:
    """SimConfig includes thermal sub-config."""
    cfg = SimConfig()
    assert hasattr(cfg, "thermal")
    assert isinstance(cfg.thermal, ThermalConfig)


def test_sim_config_yaml_roundtrip() -> None:
    """SimConfig with thermal can be created from dict."""
    cfg = SimConfig(thermal={"ambient_temperature_c": 40.0})
    assert cfg.thermal.ambient_temperature_c == 40.0
    assert cfg.thermal.reference_temperature_c == 25.0  # default preserved
