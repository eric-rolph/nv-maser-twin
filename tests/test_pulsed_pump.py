"""Tests for the pulsed optical pump model (Long et al. 2025)."""
import math

import numpy as np
import pytest

from nv_maser.config import NVConfig, OpticalPumpConfig, SimConfig
from nv_maser.physics.pulsed_pump import (
    PulsedPumpResult,
    pulsed_pump_rate,
    compute_pulsed_inversion,
    compute_equivalent_cw_power,
)
from nv_maser.physics.optical_pump import compute_pump_rate
from nv_maser.physics.environment import FieldEnvironment


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def nv_cfg() -> NVConfig:
    return NVConfig()


@pytest.fixture
def pump_cfg() -> OpticalPumpConfig:
    """Pulsed pump config: 100 µs pulse, 1000 µs period → 10% duty cycle."""
    return OpticalPumpConfig(
        pulsed=True,
        pulse_duration_us=100.0,
        pulse_period_us=1000.0,
    )


@pytest.fixture
def pump_cfg_50pct() -> OpticalPumpConfig:
    """50% duty cycle pulsed pump."""
    return OpticalPumpConfig(
        pulsed=True,
        pulse_duration_us=500.0,
        pulse_period_us=1000.0,
    )


# ── pulsed_pump_rate ──────────────────────────────────────────────


class TestPulsedPumpRate:
    def test_on_during_pulse(self) -> None:
        gamma = 1e6
        # t = 0 is the start of the pulse → ON
        assert pulsed_pump_rate(0.0, gamma, 100e-6, 1000e-6) == gamma

    def test_on_mid_pulse(self) -> None:
        gamma = 1e6
        assert pulsed_pump_rate(50e-6, gamma, 100e-6, 1000e-6) == gamma

    def test_off_after_pulse(self) -> None:
        gamma = 1e6
        assert pulsed_pump_rate(150e-6, gamma, 100e-6, 1000e-6) == 0.0

    def test_off_end_of_period(self) -> None:
        gamma = 1e6
        assert pulsed_pump_rate(999e-6, gamma, 100e-6, 1000e-6) == 0.0

    def test_on_in_second_cycle(self) -> None:
        gamma = 1e6
        # t = 1050 µs is 50 µs into the 2nd cycle → ON
        assert pulsed_pump_rate(1050e-6, gamma, 100e-6, 1000e-6) == gamma

    def test_off_in_second_cycle(self) -> None:
        gamma = 1e6
        # t = 1200 µs is 200 µs into the 2nd cycle → OFF
        assert pulsed_pump_rate(1200e-6, gamma, 100e-6, 1000e-6) == 0.0

    def test_cw_fallback_zero_period(self) -> None:
        gamma = 1e6
        assert pulsed_pump_rate(50e-6, gamma, 100e-6, 0.0) == gamma

    def test_100pct_duty(self) -> None:
        gamma = 1e6
        # pulse_duration == period → always ON
        assert pulsed_pump_rate(0.0, gamma, 100e-6, 100e-6) == gamma
        assert pulsed_pump_rate(50e-6, gamma, 100e-6, 100e-6) == gamma


# ── compute_pulsed_inversion ──────────────────────────────────────


class TestComputePulsedInversion:
    def test_returns_pulsed_result(self, pump_cfg, nv_cfg) -> None:
        result = compute_pulsed_inversion(pump_cfg, nv_cfg)
        assert isinstance(result, PulsedPumpResult)

    def test_duty_cycle(self, pump_cfg, nv_cfg) -> None:
        result = compute_pulsed_inversion(pump_cfg, nv_cfg)
        assert result.duty_cycle == pytest.approx(0.1, rel=1e-6)
        assert result.pump_on_fraction == result.duty_cycle

    def test_n_cycles(self, pump_cfg, nv_cfg) -> None:
        result = compute_pulsed_inversion(pump_cfg, nv_cfg, n_cycles=3)
        assert result.n_cycles == 3

    def test_time_array_length(self, pump_cfg, nv_cfg) -> None:
        result = compute_pulsed_inversion(pump_cfg, nv_cfg, n_cycles=5, points_per_cycle=200)
        assert len(result.time_s) == 1000
        assert len(result.inversion) == 1000

    def test_time_spans_all_cycles(self, pump_cfg, nv_cfg) -> None:
        result = compute_pulsed_inversion(pump_cfg, nv_cfg, n_cycles=5)
        tau_period = pump_cfg.pulse_period_us * 1e-6
        assert result.time_s[-1] == pytest.approx(5 * tau_period, rel=1e-4)

    def test_inversion_starts_at_zero(self, pump_cfg, nv_cfg) -> None:
        result = compute_pulsed_inversion(pump_cfg, nv_cfg)
        assert result.inversion[0] == pytest.approx(0.0, abs=1e-10)

    def test_peak_inversion_positive(self, pump_cfg, nv_cfg) -> None:
        result = compute_pulsed_inversion(pump_cfg, nv_cfg)
        assert result.peak_inversion > 0

    def test_peak_inversion_below_two_thirds(self, pump_cfg, nv_cfg) -> None:
        """Inversion can never exceed the 2/3 triplet limit."""
        result = compute_pulsed_inversion(pump_cfg, nv_cfg)
        assert result.peak_inversion < 2.0 / 3.0

    def test_mean_inversion_less_than_peak(self, pump_cfg, nv_cfg) -> None:
        result = compute_pulsed_inversion(pump_cfg, nv_cfg)
        assert result.mean_inversion < result.peak_inversion

    def test_mean_inversion_positive(self, pump_cfg, nv_cfg) -> None:
        result = compute_pulsed_inversion(pump_cfg, nv_cfg)
        assert result.mean_inversion > 0

    def test_inversion_builds_during_pulse(self, pump_cfg, nv_cfg) -> None:
        """During the first pump-ON phase, inversion should increase."""
        result = compute_pulsed_inversion(pump_cfg, nv_cfg, n_cycles=1, points_per_cycle=1000)
        tau_pulse = pump_cfg.pulse_duration_us * 1e-6
        # Find indices within the first pulse
        on_mask = result.time_s < tau_pulse
        on_inversion = result.inversion[on_mask]
        # Monotonically increasing from zero
        assert on_inversion[-1] > on_inversion[0]

    def test_inversion_decays_during_off(self, pump_cfg, nv_cfg) -> None:
        """During pump-OFF phase, inversion decays toward zero."""
        result = compute_pulsed_inversion(pump_cfg, nv_cfg, n_cycles=2, points_per_cycle=1000)
        tau_pulse = pump_cfg.pulse_duration_us * 1e-6
        tau_period = pump_cfg.pulse_period_us * 1e-6
        # Last quarter of the first OFF phase (well into decay)
        off_start = tau_pulse + 0.5 * (tau_period - tau_pulse)
        off_end = tau_period * 0.95
        off_mask = (result.time_s > off_start) & (result.time_s < off_end)
        off_inversion = result.inversion[off_mask]
        # Should be decaying: later values smaller
        assert off_inversion[-1] < off_inversion[0]

    def test_higher_duty_gives_higher_mean(self, nv_cfg) -> None:
        """Higher duty cycle → more average inversion."""
        low_duty = OpticalPumpConfig(
            pulsed=True, pulse_duration_us=50.0, pulse_period_us=1000.0
        )
        high_duty = OpticalPumpConfig(
            pulsed=True, pulse_duration_us=500.0, pulse_period_us=1000.0
        )
        r_low = compute_pulsed_inversion(low_duty, nv_cfg, n_cycles=10)
        r_high = compute_pulsed_inversion(high_duty, nv_cfg, n_cycles=10)
        assert r_high.mean_inversion > r_low.mean_inversion

    def test_more_cycles_converge(self, pump_cfg, nv_cfg) -> None:
        """After enough cycles, peak inversion should be roughly periodic."""
        result = compute_pulsed_inversion(pump_cfg, nv_cfg, n_cycles=20, points_per_cycle=200)
        tau_period = pump_cfg.pulse_period_us * 1e-6
        # Peak in cycle 15 vs cycle 20 should be close
        t15_start = 14 * tau_period
        t15_end = 15 * tau_period
        t20_start = 19 * tau_period
        t20_end = 20 * tau_period
        mask15 = (result.time_s >= t15_start) & (result.time_s < t15_end)
        mask20 = (result.time_s >= t20_start) & (result.time_s < t20_end)
        peak15 = float(np.max(result.inversion[mask15]))
        peak20 = float(np.max(result.inversion[mask20]))
        assert peak20 == pytest.approx(peak15, rel=0.01)

    def test_invalid_pulse_exceeds_period(self, nv_cfg) -> None:
        bad_cfg = OpticalPumpConfig(
            pulsed=True, pulse_duration_us=2000.0, pulse_period_us=1000.0
        )
        with pytest.raises(ValueError, match="cannot exceed"):
            compute_pulsed_inversion(bad_cfg, nv_cfg)

    def test_equivalent_cw_power_in_result(self, pump_cfg, nv_cfg) -> None:
        result = compute_pulsed_inversion(pump_cfg, nv_cfg)
        expected = pump_cfg.laser_power_w * (
            pump_cfg.pulse_duration_us / pump_cfg.pulse_period_us
        )
        assert result.equivalent_cw_power_w == pytest.approx(expected, rel=1e-6)

    def test_50pct_duty_vs_10pct(self, pump_cfg, pump_cfg_50pct, nv_cfg) -> None:
        """50% duty cycle should give higher equiv CW power than 10%."""
        r10 = compute_pulsed_inversion(pump_cfg, nv_cfg)
        r50 = compute_pulsed_inversion(pump_cfg_50pct, nv_cfg)
        assert r50.equivalent_cw_power_w > r10.equivalent_cw_power_w


# ── compute_equivalent_cw_power ───────────────────────────────────


class TestEquivalentCWPower:
    def test_basic_formula(self) -> None:
        cfg = OpticalPumpConfig(
            laser_power_w=10.0,
            pulse_duration_us=100.0,
            pulse_period_us=1000.0,
        )
        # 10 W × 0.1 = 1 W
        assert compute_equivalent_cw_power(cfg) == pytest.approx(1.0, rel=1e-6)

    def test_100pct_duty(self) -> None:
        cfg = OpticalPumpConfig(
            laser_power_w=5.0,
            pulse_duration_us=1000.0,
            pulse_period_us=1000.0,
        )
        assert compute_equivalent_cw_power(cfg) == pytest.approx(5.0, rel=1e-6)

    def test_low_duty(self) -> None:
        cfg = OpticalPumpConfig(
            laser_power_w=100.0,
            pulse_duration_us=7.0,
            pulse_period_us=1000.0,
        )
        # 100 W × 0.007 = 0.7 W
        assert compute_equivalent_cw_power(cfg) == pytest.approx(0.7, rel=1e-3)


# ── Environment integration ───────────────────────────────────────


class TestPulsedPumpEnvironmentIntegration:
    def test_pulsed_metrics_present_when_enabled(self) -> None:
        cfg = SimConfig(
            optical_pump=OpticalPumpConfig(
                pulsed=True,
                pulse_duration_us=100.0,
                pulse_period_us=1000.0,
            ),
        )
        env = FieldEnvironment(cfg)
        result = env.compute_uniformity_metric(env.distorted_field)
        assert "pulsed_peak_inversion" in result
        assert "pulsed_mean_inversion" in result
        assert "pulsed_duty_cycle" in result
        assert "pulsed_equivalent_cw_power_w" in result
        assert result["pulsed_duty_cycle"] == pytest.approx(0.1, rel=1e-6)

    def test_no_pulsed_metrics_when_disabled(self) -> None:
        cfg = SimConfig(
            optical_pump=OpticalPumpConfig(pulsed=False),
        )
        env = FieldEnvironment(cfg)
        result = env.compute_uniformity_metric(env.distorted_field)
        assert "pulsed_peak_inversion" not in result

    def test_pulsed_equivalent_cw_consistent(self) -> None:
        """Pulsed equiv CW power from environment matches standalone function."""
        cfg = SimConfig(
            optical_pump=OpticalPumpConfig(
                pulsed=True,
                laser_power_w=10.0,
                pulse_duration_us=200.0,
                pulse_period_us=1000.0,
            ),
        )
        env = FieldEnvironment(cfg)
        result = env.compute_uniformity_metric(env.distorted_field)
        expected = compute_equivalent_cw_power(cfg.optical_pump)
        assert result["pulsed_equivalent_cw_power_w"] == pytest.approx(expected, rel=1e-6)
