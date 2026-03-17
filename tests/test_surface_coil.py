"""Tests for the flat spiral surface coil model."""
import math

import numpy as np
import pytest

from nv_maser.config import SurfaceCoilConfig
from nv_maser.physics.surface_coil import (
    SurfaceCoil,
    CoilProperties,
    NoiseComponents,
    sensitivity_on_axis,
    sensitivity_off_axis,
    compute_coil_properties,
    compute_noise,
    snr_per_voxel,
    _MU0,
    _KB,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def default_config() -> SurfaceCoilConfig:
    return SurfaceCoilConfig()


@pytest.fixture
def coil(default_config: SurfaceCoilConfig) -> SurfaceCoil:
    return SurfaceCoil(default_config)


@pytest.fixture
def cooled_config() -> SurfaceCoilConfig:
    """Peltier-cooled coil at 200 K."""
    return SurfaceCoilConfig(temperature_k=200.0)


# ── Sensitivity ───────────────────────────────────────────────────


class TestSensitivity:
    def test_sensitivity_positive(self, default_config: SurfaceCoilConfig) -> None:
        depths = np.linspace(1.0, 30.0, 20)
        sens = sensitivity_on_axis(default_config, depths)
        assert np.all(sens > 0)

    def test_sensitivity_decreases_with_depth(self, default_config: SurfaceCoilConfig) -> None:
        depths = np.array([5.0, 10.0, 20.0, 30.0])
        sens = sensitivity_on_axis(default_config, depths)
        assert np.all(np.diff(sens) < 0), "Sensitivity must decrease with depth"

    def test_sensitivity_known_value(self) -> None:
        """Check against analytical formula: µ₀ N a² / [2(a²+d²)^(3/2)]."""
        cfg = SurfaceCoilConfig(coil_radius_mm=15.0, n_turns=5)
        d_mm = 20.0
        expected = _MU0 * 5 * 0.015**2 / (2 * (0.015**2 + 0.020**2) ** 1.5)
        actual = float(sensitivity_on_axis(cfg, np.array([d_mm]))[0])
        assert math.isclose(actual, expected, rel_tol=1e-10)

    def test_sensitivity_at_surface(self) -> None:
        """At depth=0, S = µ₀ N / (2a)."""
        cfg = SurfaceCoilConfig(coil_radius_mm=10.0, n_turns=3)
        a = 0.010
        expected = _MU0 * 3 / (2 * a)
        actual = float(sensitivity_on_axis(cfg, np.array([0.0]))[0])
        assert math.isclose(actual, expected, rel_tol=1e-6)

    def test_sensitivity_scales_with_turns(self) -> None:
        cfg1 = SurfaceCoilConfig(n_turns=1)
        cfg5 = SurfaceCoilConfig(n_turns=5)
        s1 = sensitivity_on_axis(cfg1, np.array([10.0]))
        s5 = sensitivity_on_axis(cfg5, np.array([10.0]))
        np.testing.assert_allclose(s5 / s1, 5.0, rtol=1e-10)


class TestOffAxisSensitivity:
    def test_on_axis_matches(self, default_config: SurfaceCoilConfig) -> None:
        """Off-axis at rho=0 should match on-axis."""
        s_on = float(sensitivity_on_axis(default_config, np.array([10.0]))[0])
        s_off = float(sensitivity_off_axis(default_config, np.array([0.0]), depth_mm=10.0)[0])
        assert math.isclose(s_off, s_on, rel_tol=1e-10)

    def test_decreases_off_axis(self, default_config: SurfaceCoilConfig) -> None:
        rho = np.array([0.0, 3.0, 6.0, 9.0])
        sens = sensitivity_off_axis(default_config, rho, depth_mm=10.0)
        assert np.all(np.diff(sens) <= 0)


# ── Coil properties ───────────────────────────────────────────────


class TestCoilProperties:
    def test_properties_returned(self, default_config: SurfaceCoilConfig) -> None:
        props = compute_coil_properties(default_config, frequency_hz=2.13e6)
        assert isinstance(props, CoilProperties)
        assert props.dc_resistance_ohm > 0
        assert props.ac_resistance_ohm >= props.dc_resistance_ohm
        assert props.inductance_h > 0

    def test_ac_resistance_higher_at_high_freq(self, default_config: SurfaceCoilConfig) -> None:
        props_low = compute_coil_properties(default_config, frequency_hz=1e5)
        props_high = compute_coil_properties(default_config, frequency_hz=100e6)
        assert props_high.ac_resistance_ohm >= props_low.ac_resistance_ohm

    def test_skin_depth_decreases_with_freq(self, default_config: SurfaceCoilConfig) -> None:
        props_low = compute_coil_properties(default_config, frequency_hz=1e5)
        props_high = compute_coil_properties(default_config, frequency_hz=100e6)
        assert props_high.skin_depth_mm < props_low.skin_depth_mm


# ── Noise ─────────────────────────────────────────────────────────


class TestNoise:
    def test_noise_positive(self, default_config: SurfaceCoilConfig) -> None:
        noise = compute_noise(default_config, 2.13e6, 10_000.0)
        assert isinstance(noise, NoiseComponents)
        assert noise.coil_thermal_v > 0
        assert noise.total_noise_v > 0

    def test_noise_scales_with_bandwidth(self, default_config: SurfaceCoilConfig) -> None:
        n1 = compute_noise(default_config, 2.13e6, 1_000.0)
        n10 = compute_noise(default_config, 2.13e6, 10_000.0)
        # V_noise ∝ √Δf, so ×10 BW → ×√10 noise
        ratio = n10.coil_thermal_v / n1.coil_thermal_v
        assert math.isclose(ratio, math.sqrt(10), rel_tol=0.05)

    def test_body_noise_small_at_low_freq(self, default_config: SurfaceCoilConfig) -> None:
        """At 2 MHz, body noise should be << coil noise for a surface coil."""
        noise = compute_noise(default_config, 2.13e6, 10_000.0)
        assert noise.body_noise_v < noise.coil_thermal_v * 0.1

    def test_cooled_coil_less_noise(self) -> None:
        cfg_rt = SurfaceCoilConfig(temperature_k=300.0)
        cfg_cool = SurfaceCoilConfig(temperature_k=200.0)
        n_rt = compute_noise(cfg_rt, 2.13e6, 10_000.0)
        n_cool = compute_noise(cfg_cool, 2.13e6, 10_000.0)
        assert n_cool.coil_thermal_v < n_rt.coil_thermal_v

    def test_noise_rss_correct(self, default_config: SurfaceCoilConfig) -> None:
        noise = compute_noise(default_config, 2.13e6, 10_000.0)
        expected = math.sqrt(noise.coil_thermal_v**2 + noise.body_noise_v**2)
        assert math.isclose(noise.total_noise_v, expected, rel_tol=1e-10)


# ── SNR ───────────────────────────────────────────────────────────


class TestSNR:
    def test_snr_positive(self, default_config: SurfaceCoilConfig) -> None:
        s = snr_per_voxel(
            default_config,
            depth_mm=10.0,
            voxel_size_mm=3.0,
            b0_tesla=0.05,
            frequency_hz=2.13e6,
            bandwidth_hz=10_000.0,
        )
        assert s > 0

    def test_snr_decreases_with_depth(self, default_config: SurfaceCoilConfig) -> None:
        snr_5 = snr_per_voxel(default_config, 5.0, 3.0, 0.05, 2.13e6, 10_000.0)
        snr_20 = snr_per_voxel(default_config, 20.0, 3.0, 0.05, 2.13e6, 10_000.0)
        assert snr_5 > snr_20

    def test_snr_scales_with_sqrt_averages(self, default_config: SurfaceCoilConfig) -> None:
        snr_1 = snr_per_voxel(default_config, 10.0, 3.0, 0.05, 2.13e6, 10_000.0, n_averages=1)
        snr_4 = snr_per_voxel(default_config, 10.0, 3.0, 0.05, 2.13e6, 10_000.0, n_averages=4)
        ratio = snr_4 / snr_1
        assert math.isclose(ratio, 2.0, rel_tol=1e-6)

    def test_snr_scales_with_voxel_volume(self, default_config: SurfaceCoilConfig) -> None:
        snr_2mm = snr_per_voxel(default_config, 10.0, 2.0, 0.05, 2.13e6, 10_000.0)
        snr_4mm = snr_per_voxel(default_config, 10.0, 4.0, 0.05, 2.13e6, 10_000.0)
        # SNR ∝ V ∝ L³ → ratio should be (4/2)³ = 8
        ratio = snr_4mm / snr_2mm
        assert math.isclose(ratio, 8.0, rel_tol=0.01)


# ── SurfaceCoil class ─────────────────────────────────────────────


class TestSurfaceCoilClass:
    def test_b1_per_amp(self, coil: SurfaceCoil) -> None:
        b1 = coil.b1_per_amp(np.array([5.0, 10.0, 20.0]))
        assert b1.shape == (3,)
        assert np.all(b1 > 0)

    def test_properties(self, coil: SurfaceCoil) -> None:
        props = coil.properties(2.13e6)
        assert isinstance(props, CoilProperties)

    def test_noise(self, coil: SurfaceCoil) -> None:
        noise = coil.noise(2.13e6, 10_000.0)
        assert isinstance(noise, NoiseComponents)

    def test_snr(self, coil: SurfaceCoil) -> None:
        s = coil.snr(10.0, 3.0, 0.05, 2.13e6, 10_000.0)
        assert s > 0
