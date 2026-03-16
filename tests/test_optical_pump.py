"""Tests for the 532 nm optical pump model."""
import math

import pytest

from nv_maser.config import NVConfig, OpticalPumpConfig
from nv_maser.physics.optical_pump import (
    PumpState,
    DepthResolvedPumpResult,
    compute_absorbed_power,
    compute_pump_rate,
    compute_pump_state,
    compute_depth_resolved_pump,
    _C,
    _HBAR,
)
from nv_maser.physics.environment import FieldEnvironment
from nv_maser.config import SimConfig


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def nv_cfg() -> NVConfig:
    return NVConfig()


@pytest.fixture
def pump_cfg() -> OpticalPumpConfig:
    return OpticalPumpConfig()


# ── OpticalPumpConfig ─────────────────────────────────────────────


class TestOpticalPumpConfig:
    def test_defaults(self) -> None:
        cfg = OpticalPumpConfig()
        assert cfg.laser_power_w == 2.0
        assert cfg.laser_wavelength_nm == 532.0
        assert cfg.beam_waist_mm == 1.5
        assert cfg.absorption_cross_section_m2 == 3.1e-21
        assert cfg.quantum_defect_fraction == 0.6
        assert cfg.spin_t1_ms == 5.0


# ── Pump rate ─────────────────────────────────────────────────────


class TestPumpRate:
    def test_positive(self, pump_cfg: OpticalPumpConfig) -> None:
        rate = compute_pump_rate(pump_cfg)
        assert rate > 0

    def test_formula(self, pump_cfg: OpticalPumpConfig) -> None:
        omega = 2 * math.pi * _C / (pump_cfg.laser_wavelength_nm * 1e-9)
        w0 = pump_cfg.beam_waist_mm * 1e-3
        i0 = 2 * pump_cfg.laser_power_w / (math.pi * w0**2)
        expected = pump_cfg.absorption_cross_section_m2 * i0 / (_HBAR * omega)
        actual = compute_pump_rate(pump_cfg)
        assert actual == pytest.approx(expected, rel=1e-10)

    def test_scales_with_power(self) -> None:
        low = OpticalPumpConfig(laser_power_w=1.0)
        high = OpticalPumpConfig(laser_power_w=2.0)
        assert compute_pump_rate(high) == pytest.approx(
            2.0 * compute_pump_rate(low), rel=1e-10
        )

    def test_smaller_waist_higher_rate(self) -> None:
        big = OpticalPumpConfig(beam_waist_mm=2.0)
        small = OpticalPumpConfig(beam_waist_mm=1.0)
        assert compute_pump_rate(small) > compute_pump_rate(big)

    def test_zero_power(self) -> None:
        cfg = OpticalPumpConfig(laser_power_w=0.0)
        assert compute_pump_rate(cfg) == 0.0


# ── Absorbed power ────────────────────────────────────────────────


class TestAbsorbedPower:
    def test_positive(self, pump_cfg: OpticalPumpConfig, nv_cfg: NVConfig) -> None:
        p = compute_absorbed_power(pump_cfg, nv_cfg)
        assert p > 0

    def test_less_than_incident(self, pump_cfg: OpticalPumpConfig,
                                 nv_cfg: NVConfig) -> None:
        p = compute_absorbed_power(pump_cfg, nv_cfg)
        assert p < pump_cfg.laser_power_w

    def test_beer_lambert(self, pump_cfg: OpticalPumpConfig,
                           nv_cfg: NVConfig) -> None:
        n_m3 = nv_cfg.nv_density_per_cm3 * 1e6
        alpha = n_m3 * pump_cfg.absorption_cross_section_m2
        d = nv_cfg.diamond_thickness_mm * 1e-3
        expected = pump_cfg.laser_power_w * (1.0 - math.exp(-alpha * d))
        assert compute_absorbed_power(pump_cfg, nv_cfg) == pytest.approx(
            expected, rel=1e-10
        )

    def test_thicker_diamond_absorbs_more(self, pump_cfg: OpticalPumpConfig) -> None:
        thin = NVConfig(diamond_thickness_mm=0.5)
        thick = NVConfig(diamond_thickness_mm=2.0)
        assert compute_absorbed_power(pump_cfg, thick) > compute_absorbed_power(
            pump_cfg, thin
        )

    def test_higher_density_absorbs_more(self, pump_cfg: OpticalPumpConfig) -> None:
        sparse = NVConfig(nv_density_per_cm3=1e16)
        dense = NVConfig(nv_density_per_cm3=1e18)
        assert compute_absorbed_power(pump_cfg, dense) > compute_absorbed_power(
            pump_cfg, sparse
        )


# ── Pump state ────────────────────────────────────────────────────


class TestPumpState:
    def test_returns_dataclass(self, pump_cfg: OpticalPumpConfig,
                                nv_cfg: NVConfig) -> None:
        state = compute_pump_state(pump_cfg, nv_cfg)
        assert isinstance(state, PumpState)

    def test_saturation_range(self, pump_cfg: OpticalPumpConfig,
                               nv_cfg: NVConfig) -> None:
        state = compute_pump_state(pump_cfg, nv_cfg)
        assert 0 <= state.pump_saturation <= 1

    def test_efficiency_range(self, pump_cfg: OpticalPumpConfig,
                               nv_cfg: NVConfig) -> None:
        state = compute_pump_state(pump_cfg, nv_cfg)
        # Max inversion is 2/3
        assert 0 <= state.effective_pump_efficiency <= 2.0 / 3.0 + 1e-10

    def test_thermal_load_positive(self, pump_cfg: OpticalPumpConfig,
                                    nv_cfg: NVConfig) -> None:
        state = compute_pump_state(pump_cfg, nv_cfg)
        assert state.thermal_load_w > 0

    def test_thermal_load_formula(self, pump_cfg: OpticalPumpConfig,
                                   nv_cfg: NVConfig) -> None:
        state = compute_pump_state(pump_cfg, nv_cfg)
        expected = state.absorbed_power_w * pump_cfg.quantum_defect_fraction
        assert state.thermal_load_w == pytest.approx(expected, rel=1e-10)

    def test_higher_power_higher_saturation(self, nv_cfg: NVConfig) -> None:
        low = OpticalPumpConfig(laser_power_w=0.01)
        high = OpticalPumpConfig(laser_power_w=10.0)
        s_lo = compute_pump_state(low, nv_cfg)
        s_hi = compute_pump_state(high, nv_cfg)
        assert s_hi.pump_saturation > s_lo.pump_saturation

    def test_zero_power_zero_saturation(self, nv_cfg: NVConfig) -> None:
        cfg = OpticalPumpConfig(laser_power_w=0.0)
        state = compute_pump_state(cfg, nv_cfg)
        assert state.pump_saturation == 0.0
        assert state.effective_pump_efficiency == 0.0
        assert state.thermal_load_w == 0.0

    def test_high_power_approaches_max(self, nv_cfg: NVConfig) -> None:
        """At very high pump power, saturation → 1, η → 2/3."""
        cfg = OpticalPumpConfig(laser_power_w=1000.0)
        state = compute_pump_state(cfg, nv_cfg)
        assert state.pump_saturation > 0.999
        assert state.effective_pump_efficiency == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_intensity_formula(self, pump_cfg: OpticalPumpConfig,
                                nv_cfg: NVConfig) -> None:
        state = compute_pump_state(pump_cfg, nv_cfg)
        w0 = pump_cfg.beam_waist_mm * 1e-3
        expected = 2.0 * pump_cfg.laser_power_w / (math.pi * w0**2)
        assert state.pump_intensity_w_per_m2 == pytest.approx(expected, rel=1e-10)


# ── Environment integration ──────────────────────────────────────


class TestEnvironmentPumpIntegration:
    def test_metrics_include_pump(self) -> None:
        config = SimConfig()
        env = FieldEnvironment(config)
        metrics = env.compute_uniformity_metric(env.distorted_field)
        assert "pump_rate_hz" in metrics
        assert "pump_saturation" in metrics
        assert "effective_pump_efficiency" in metrics
        assert "thermal_load_w" in metrics

    def test_metrics_include_cavity(self) -> None:
        config = SimConfig()
        env = FieldEnvironment(config)
        metrics = env.compute_uniformity_metric(env.distorted_field)
        assert "cooperativity" in metrics
        assert "threshold_margin" in metrics
        assert "n_effective" in metrics
        assert "ensemble_coupling_hz" in metrics


# ── Depth-resolved pump ──────────────────────────────────────────


class TestDepthResolvedPump:
    def test_returns_dataclass(self, pump_cfg: OpticalPumpConfig,
                                nv_cfg: NVConfig) -> None:
        result = compute_depth_resolved_pump(pump_cfg, nv_cfg, n_slices=10)
        assert isinstance(result, DepthResolvedPumpResult)

    def test_correct_number_of_slices(self, pump_cfg: OpticalPumpConfig,
                                       nv_cfg: NVConfig) -> None:
        for n in (1, 5, 20, 50):
            result = compute_depth_resolved_pump(pump_cfg, nv_cfg, n_slices=n)
            assert len(result.z_m) == n
            assert len(result.pump_rate_hz) == n
            assert len(result.saturation) == n
            assert len(result.inversion) == n

    def test_z_positions_span_diamond(self, pump_cfg: OpticalPumpConfig,
                                       nv_cfg: NVConfig) -> None:
        result = compute_depth_resolved_pump(pump_cfg, nv_cfg, n_slices=20)
        d = nv_cfg.diamond_thickness_mm * 1e-3
        assert result.z_m[0] > 0  # midpoint of first slice
        assert result.z_m[-1] < d  # midpoint of last slice

    def test_pump_rate_decreases_with_depth(self, pump_cfg: OpticalPumpConfig,
                                              nv_cfg: NVConfig) -> None:
        """Beer-Lambert attenuation → pump rate monotonically decreases."""
        result = compute_depth_resolved_pump(pump_cfg, nv_cfg, n_slices=20)
        for i in range(len(result.pump_rate_hz) - 1):
            assert result.pump_rate_hz[i] >= result.pump_rate_hz[i + 1]

    def test_inversion_decreases_with_depth(self, pump_cfg: OpticalPumpConfig,
                                              nv_cfg: NVConfig) -> None:
        result = compute_depth_resolved_pump(pump_cfg, nv_cfg, n_slices=20)
        for i in range(len(result.inversion) - 1):
            assert result.inversion[i] >= result.inversion[i + 1]

    def test_front_back_ratio_ge_one(self, pump_cfg: OpticalPumpConfig,
                                      nv_cfg: NVConfig) -> None:
        result = compute_depth_resolved_pump(pump_cfg, nv_cfg, n_slices=20)
        assert result.front_back_ratio >= 1.0

    def test_high_density_large_gradient(self) -> None:
        """At 12 ppm (~2.1e18/cm³), α×d ≈ 3.3 → 25× front-to-back ratio."""
        pump = OpticalPumpConfig()
        nv = NVConfig(nv_density_per_cm3=2.1e18, diamond_thickness_mm=0.5)
        result = compute_depth_resolved_pump(pump, nv, n_slices=50)
        assert result.front_back_ratio > 20.0
        # Front slice should be near full saturation at 2 W
        assert result.inversion[0] > result.inversion[-1]

    def test_low_density_nearly_uniform(self) -> None:
        """At 1e16/cm³ with 0.5 mm, α×d ≈ 0.015 → nearly uniform."""
        pump = OpticalPumpConfig()
        nv = NVConfig(nv_density_per_cm3=1e16, diamond_thickness_mm=0.5)
        result = compute_depth_resolved_pump(pump, nv, n_slices=20)
        assert result.front_back_ratio < 1.02
        # All slices should have nearly identical inversion
        inv = result.inversion
        assert max(inv) - min(inv) < 0.001

    def test_single_slice_matches_uniform(self, pump_cfg: OpticalPumpConfig,
                                           nv_cfg: NVConfig) -> None:
        """With 1 slice, depth-resolved should approximately match uniform model."""
        dr = compute_depth_resolved_pump(pump_cfg, nv_cfg, n_slices=1)
        uniform = compute_pump_state(pump_cfg, nv_cfg)
        # Single slice at midpoint ≈ uniform model (small difference due to
        # midpoint vs beam-center evaluation)
        assert dr.effective_pump_efficiency == pytest.approx(
            uniform.effective_pump_efficiency, rel=0.05
        )

    def test_converges_with_more_slices(self, pump_cfg: OpticalPumpConfig,
                                         nv_cfg: NVConfig) -> None:
        """Result should converge as n_slices increases."""
        eta_10 = compute_depth_resolved_pump(pump_cfg, nv_cfg, 10).effective_pump_efficiency
        eta_50 = compute_depth_resolved_pump(pump_cfg, nv_cfg, 50).effective_pump_efficiency
        eta_200 = compute_depth_resolved_pump(pump_cfg, nv_cfg, 200).effective_pump_efficiency
        # 50 and 200 should be very close
        assert abs(eta_50 - eta_200) < abs(eta_10 - eta_200)

    def test_absorbed_power_matches_uniform(self, pump_cfg: OpticalPumpConfig,
                                             nv_cfg: NVConfig) -> None:
        """Absorbed power is the same regardless of discretisation."""
        dr = compute_depth_resolved_pump(pump_cfg, nv_cfg, n_slices=20)
        uniform = compute_pump_state(pump_cfg, nv_cfg)
        assert dr.absorbed_power_w == pytest.approx(uniform.absorbed_power_w, rel=1e-10)

    def test_thermal_load_matches_uniform(self, pump_cfg: OpticalPumpConfig,
                                           nv_cfg: NVConfig) -> None:
        dr = compute_depth_resolved_pump(pump_cfg, nv_cfg, n_slices=20)
        uniform = compute_pump_state(pump_cfg, nv_cfg)
        assert dr.thermal_load_w == pytest.approx(uniform.thermal_load_w, rel=1e-10)

    def test_invalid_n_slices(self, pump_cfg: OpticalPumpConfig,
                               nv_cfg: NVConfig) -> None:
        with pytest.raises(ValueError):
            compute_depth_resolved_pump(pump_cfg, nv_cfg, n_slices=0)

    def test_depth_resolved_efficiency_less_than_uniform_high_density(self) -> None:
        """Depth-resolved η should be lower than uniform for high density
        because the back surface is poorly pumped."""
        pump = OpticalPumpConfig(laser_power_w=2.0)
        nv = NVConfig(nv_density_per_cm3=2.1e18, diamond_thickness_mm=0.5)
        dr = compute_depth_resolved_pump(pump, nv, n_slices=50)
        uniform = compute_pump_state(pump, nv)
        assert dr.effective_pump_efficiency < uniform.effective_pump_efficiency


class TestDepthResolvedEnvironmentIntegration:
    def test_depth_resolved_metrics(self) -> None:
        """When n_depth_slices > 1, environment returns front_back_ratio."""
        config = SimConfig(
            optical_pump=OpticalPumpConfig(n_depth_slices=20),
        )
        env = FieldEnvironment(config)
        metrics = env.compute_uniformity_metric(env.distorted_field)
        assert "pump_front_back_ratio" in metrics
        assert metrics["pump_front_back_ratio"] >= 1.0

    def test_uniform_no_front_back_ratio(self) -> None:
        """When n_depth_slices = 1, no front_back_ratio key."""
        config = SimConfig()  # default n_depth_slices=1
        env = FieldEnvironment(config)
        metrics = env.compute_uniformity_metric(env.distorted_field)
        assert "pump_front_back_ratio" not in metrics
