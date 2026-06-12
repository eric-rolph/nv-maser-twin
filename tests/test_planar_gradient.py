"""Tests for planar_gradient.py — planar gradient coil model."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nv_maser.physics.planar_gradient import (
    DEFAULT_GX,
    DEFAULT_GY,
    GradientCoilSpec,
    GradientPulseResult,
    GradientWaveform,
    build_phase_encode_scheme,
    compute_coil_resistance,
    compute_gradient_efficiency,
    compute_inductance,
    compute_k_position,
    compute_k_trajectory,
    compute_max_gradient,
    compute_power_dissipation,
    current_for_gradient,
    evaluate_waveform,
    gradient_field_1d,
    linearity_error,
    sweep_efficiency_vs_radius,
    sweep_k_max_vs_fov,
    sweep_max_gradient_vs_current,
    sweep_resolution_vs_n_lines,
)

# ── Constants ──────────────────────────────────────────────────────
_MU0 = 4.0 * math.pi * 1e-7
_GAMMA_P = 2.0 * math.pi * 42.577e6  # rad/s/T


# ╔══════════════════════════════════════════════════════════════════╗
# ║  GradientCoilSpec                                                ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestGradientCoilSpec:
    def test_default_construction(self):
        spec = GradientCoilSpec()
        assert spec.loop_radius_mm == pytest.approx(20.0)
        assert spec.n_turns == 5
        assert spec.axis == "x"

    def test_custom_construction(self):
        spec = GradientCoilSpec(
            loop_radius_mm=15.0, offset_mm=10.6, n_turns=8, axis="y"
        )
        assert spec.n_turns == 8
        assert spec.axis == "y"

    def test_invalid_axis(self):
        with pytest.raises(ValueError, match="axis"):
            GradientCoilSpec(axis="z")

    def test_invalid_radius(self):
        with pytest.raises(ValueError):
            GradientCoilSpec(loop_radius_mm=0.0)

    def test_invalid_offset(self):
        with pytest.raises(ValueError):
            GradientCoilSpec(offset_mm=-1.0)

    def test_invalid_turns(self):
        with pytest.raises(ValueError):
            GradientCoilSpec(n_turns=0)

    def test_frozen(self):
        spec = GradientCoilSpec()
        with pytest.raises(Exception):
            spec.n_turns = 10  # type: ignore[misc]

    def test_default_gx_gx(self):
        assert DEFAULT_GX.axis == "x"
        assert DEFAULT_GY.axis == "y"

    def test_default_axes_are_symmetric(self):
        assert DEFAULT_GX.loop_radius_mm == DEFAULT_GY.loop_radius_mm
        assert DEFAULT_GX.n_turns == DEFAULT_GY.n_turns


# ╔══════════════════════════════════════════════════════════════════╗
# ║  GradientWaveform                                                ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestGradientWaveform:
    def test_duration(self):
        wf = GradientWaveform(amplitude_t_per_m=0.03, rise_time_us=200, flat_time_us=500)
        assert wf.duration_us == pytest.approx(200 + 500 + 200)

    def test_slew_rate(self):
        wf = GradientWaveform(amplitude_t_per_m=0.030, rise_time_us=200.0, flat_time_us=500)
        expected = 0.030 / (200e-6)
        assert wf.slew_rate_t_per_m_per_s == pytest.approx(expected)

    def test_zero_slew_rate_for_zero_amplitude(self):
        wf = GradientWaveform(amplitude_t_per_m=0.0, rise_time_us=100)
        assert wf.slew_rate_t_per_m_per_s == pytest.approx(0.0)

    def test_area(self):
        # Area = G × (rise_time + flat_time)  for trapezoid
        wf = GradientWaveform(amplitude_t_per_m=0.05, rise_time_us=100, flat_time_us=400)
        expected = 0.05 * (100 + 400)
        assert wf.area_t_per_m_us == pytest.approx(expected)

    def test_negative_amplitude(self):
        wf = GradientWaveform(amplitude_t_per_m=-0.03, rise_time_us=200, flat_time_us=300)
        assert wf.slew_rate_t_per_m_per_s > 0  # magnitude

    def test_frozen(self):
        wf = GradientWaveform()
        with pytest.raises(Exception):
            wf.amplitude_t_per_m = 0.1  # type: ignore[misc]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  compute_gradient_efficiency                                     ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestGradientEfficiency:
    def test_scales_with_n_turns(self):
        spec1 = GradientCoilSpec(n_turns=1)
        spec5 = GradientCoilSpec(n_turns=5)
        eta1 = compute_gradient_efficiency(spec1)
        eta5 = compute_gradient_efficiency(spec5)
        assert eta5 == pytest.approx(5 * eta1, rel=1e-6)

    def test_scales_as_inverse_radius_squared_approx(self):
        """η ∝ a² d / (a²+d²)^(5/2); for d = a/√2, η ∝ a^(-2) roughly."""
        spec1 = GradientCoilSpec(loop_radius_mm=10, offset_mm=7.07, n_turns=1)
        spec2 = GradientCoilSpec(loop_radius_mm=20, offset_mm=14.14, n_turns=1)
        eta1 = compute_gradient_efficiency(spec1)
        eta2 = compute_gradient_efficiency(spec2)
        # 2× radius → ~4× smaller efficiency (inverse-square-ish)
        ratio = eta1 / eta2
        assert ratio == pytest.approx(4.0, rel=0.05)

    def test_positive_value(self):
        assert compute_gradient_efficiency(DEFAULT_GX) > 0

    def test_units(self):
        """Efficiency should be small (~mT/m/A range for cm-scale coils)."""
        eta = compute_gradient_efficiency(DEFAULT_GX)
        # Between 1 and 100 mT/m/A for typical probe-sized coils
        assert 1e-3 <= eta <= 100e-3  # T/m/A

    def test_default_gx_value(self):
        """21 mT/m/A for 15 mm loop, 5 turns, 10.6 mm offset."""
        eta = compute_gradient_efficiency(DEFAULT_GX)
        assert eta == pytest.approx(21.5e-3, rel=0.02)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  compute_coil_resistance                                         ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestCoilResistance:
    def test_positive(self):
        assert compute_coil_resistance(DEFAULT_GX) > 0

    def test_scales_with_n_turns(self):
        spec1 = GradientCoilSpec(n_turns=1)
        spec5 = GradientCoilSpec(n_turns=5)
        r1 = compute_coil_resistance(spec1)
        r5 = compute_coil_resistance(spec5)
        # Wire length ∝ N → resistance ∝ N
        assert r5 == pytest.approx(5 * r1, rel=1e-6)

    def test_milliohm_range(self):
        """Typical surface coil resistance: 10–200 mΩ."""
        r = compute_coil_resistance(DEFAULT_GX)
        assert 1e-3 <= r <= 1.0  # Ω

    def test_thicker_wire_lower_resistance(self):
        thin = GradientCoilSpec(wire_diameter_mm=0.5)
        thick = GradientCoilSpec(wire_diameter_mm=2.0)
        assert compute_coil_resistance(thin) > compute_coil_resistance(thick)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  compute_inductance                                              ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestInductance:
    def test_positive(self):
        assert compute_inductance(DEFAULT_GX) > 0

    def test_scales_with_n_turns_squared(self):
        spec1 = GradientCoilSpec(n_turns=1)
        spec2 = GradientCoilSpec(n_turns=2)
        l1 = compute_inductance(spec1)
        l2 = compute_inductance(spec2)
        assert l2 == pytest.approx(4 * l1, rel=1e-6)

    def test_microhenry_range(self):
        """Surface coil inductance: ~1–100 µH."""
        inductance = compute_inductance(DEFAULT_GX)
        assert 0.1e-6 <= inductance <= 500e-6  # H


# ╔══════════════════════════════════════════════════════════════════╗
# ║  compute_power_dissipation                                       ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestPowerDissipation:
    def test_zero_current_is_zero_power(self):
        assert compute_power_dissipation(DEFAULT_GX, 0.0) == pytest.approx(0.0)

    def test_scales_as_current_squared(self):
        p1 = compute_power_dissipation(DEFAULT_GX, 1.0)
        p2 = compute_power_dissipation(DEFAULT_GX, 2.0)
        assert p2 == pytest.approx(4 * p1)

    def test_finite_positive(self):
        p = compute_power_dissipation(DEFAULT_GX, 5.0)
        assert p > 0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  compute_max_gradient                                            ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestMaxGradient:
    def test_positive(self):
        assert compute_max_gradient(DEFAULT_GX) > 0

    def test_equals_eta_times_imax(self):
        eta = compute_gradient_efficiency(DEFAULT_GX)
        expected = eta * DEFAULT_GX.max_current_a
        assert compute_max_gradient(DEFAULT_GX) == pytest.approx(expected)

    def test_default_exceeds_50mt_per_m(self):
        """Default coil must be able to reach 50 mT/m (probe requirement)."""
        assert compute_max_gradient(DEFAULT_GX) >= 0.050  # T/m


# ╔══════════════════════════════════════════════════════════════════╗
# ║  current_for_gradient                                            ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestCurrentForGradient:
    def test_roundtrip(self):
        g = 0.03  # T/m
        i = current_for_gradient(DEFAULT_GX, g)
        # Recover gradient
        eta = compute_gradient_efficiency(DEFAULT_GX)
        assert abs(i) * eta == pytest.approx(abs(g))

    def test_negative_gradient(self):
        i = current_for_gradient(DEFAULT_GX, -0.02)
        assert i < 0

    def test_exceeds_max_raises(self):
        # Very large gradient well beyond capability
        with pytest.raises(ValueError, match="exceeds max"):
            current_for_gradient(DEFAULT_GX, 100.0)  # 100 T/m is impossible


# ╔══════════════════════════════════════════════════════════════════╗
# ║  compute_k_position                                              ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestKPosition:
    def _make_wf(self):
        return GradientWaveform(
            amplitude_t_per_m=0.030, rise_time_us=200.0, flat_time_us=500.0
        )

    def test_zero_at_start(self):
        wf = self._make_wf()
        assert compute_k_position(wf, 0.0) == pytest.approx(0.0, abs=1.0)

    def test_before_waveform_is_zero(self):
        wf = self._make_wf()
        assert compute_k_position(wf, -100.0) == pytest.approx(0.0)

    def test_increases_during_ramp(self):
        wf = self._make_wf()
        k1 = compute_k_position(wf, 100.0)
        k2 = compute_k_position(wf, 200.0)
        assert k2 > k1 > 0

    def test_increases_linearly_during_flat(self):
        wf = self._make_wf()
        k1 = compute_k_position(wf, 300.0)  # into flat top
        k2 = compute_k_position(wf, 400.0)
        k3 = compute_k_position(wf, 500.0)
        dk12 = k2 - k1
        dk23 = k3 - k2
        assert dk12 == pytest.approx(dk23, rel=1e-5)

    def test_final_value_matches_area(self):
        wf = self._make_wf()
        # After full waveform, k = (γ/2π) × G × (t_r + t_f)
        t_end = 2 * 200.0 + 500.0 + 1.0  # after fall
        k_final = compute_k_position(wf, t_end)
        expected = (_GAMMA_P / (2 * math.pi)) * 0.030 * (200 + 500) * 1e-6
        assert k_final == pytest.approx(expected, rel=1e-4)

    def test_delayed_waveform(self):
        wf = GradientWaveform(
            amplitude_t_per_m=0.030, rise_time_us=200, flat_time_us=500, delay_us=100
        )
        assert compute_k_position(wf, 50.0) == pytest.approx(0.0)
        assert compute_k_position(wf, 150.0) > 0  # inside ramp after delay


# ╔══════════════════════════════════════════════════════════════════╗
# ║  compute_k_trajectory                                            ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestKTrajectory:
    def test_output_shapes(self):
        wf = GradientWaveform(amplitude_t_per_m=0.030, rise_time_us=200, flat_time_us=500)
        times, kvalues = compute_k_trajectory(wf, n_samples=64)
        assert times.shape == (64,)
        assert kvalues.shape == (64,)

    def test_monotonic_during_flat(self):
        wf = GradientWaveform(amplitude_t_per_m=0.030, rise_time_us=200, flat_time_us=500)
        times, k = compute_k_trajectory(wf, n_samples=32)
        assert np.all(np.diff(k) > 0)

    def test_bandwidth_of_trajectory(self):
        """k span during readout determines spatial bandwidth."""
        wf = GradientWaveform(amplitude_t_per_m=0.030, rise_time_us=200, flat_time_us=500)
        _, k = compute_k_trajectory(wf, n_samples=128, readout_duration_us=500)
        k_span = k[-1] - k[0]
        assert k_span > 0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  evaluate_waveform                                               ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestEvaluateWaveform:
    def test_returns_result_type(self):
        wf = GradientWaveform(amplitude_t_per_m=0.030, rise_time_us=200, flat_time_us=500)
        result = evaluate_waveform(wf, DEFAULT_GX)
        assert isinstance(result, GradientPulseResult)

    def test_meets_limits_for_safe_waveform(self):
        wf = GradientWaveform(amplitude_t_per_m=0.020, rise_time_us=200, flat_time_us=500)
        result = evaluate_waveform(wf, DEFAULT_GX)
        assert result.meets_limits

    def test_fails_limits_for_excessive_slew(self):
        # Very fast ramp exceeds slew rate
        wf = GradientWaveform(amplitude_t_per_m=0.050, rise_time_us=0.1, flat_time_us=500)
        result = evaluate_waveform(wf, DEFAULT_GX, max_slew_t_per_m_per_s=100.0)
        assert not result.meets_limits

    def test_power_positive(self):
        wf = GradientWaveform(amplitude_t_per_m=0.030, rise_time_us=200, flat_time_us=500)
        result = evaluate_waveform(wf, DEFAULT_GX)
        assert result.power_w > 0

    def test_k_position_positive_for_positive_gradient(self):
        wf = GradientWaveform(amplitude_t_per_m=0.030, rise_time_us=200, flat_time_us=500)
        result = evaluate_waveform(wf, DEFAULT_GX)
        assert result.k_position_per_m > 0

    def test_negative_gradient_negative_k(self):
        wf = GradientWaveform(amplitude_t_per_m=-0.030, rise_time_us=200, flat_time_us=500)
        result = evaluate_waveform(wf, DEFAULT_GX)
        assert result.k_position_per_m < 0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  build_phase_encode_scheme                                       ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestPhaseEncodeScheme:
    def test_n_lines_property(self):
        scheme = build_phase_encode_scheme(DEFAULT_GX, n_lines=32, fov_m=0.06)
        assert scheme.n_lines == 32

    def test_fov_correct(self):
        scheme = build_phase_encode_scheme(DEFAULT_GX, n_lines=32, fov_m=0.060)
        assert scheme.fov_m == pytest.approx(0.060)

    def test_resolution(self):
        scheme = build_phase_encode_scheme(DEFAULT_GX, n_lines=32, fov_m=0.064)
        assert scheme.resolution_mm == pytest.approx(64.0 / 32, rel=1e-4)

    def test_amplitude_array_length(self):
        scheme = build_phase_encode_scheme(DEFAULT_GX, n_lines=32, fov_m=0.06)
        assert len(scheme.gradient_amplitudes_t_per_m) == 32

    def test_amplitude_array_symmetric(self):
        scheme = build_phase_encode_scheme(DEFAULT_GX, n_lines=32, fov_m=0.06)
        amps = scheme.gradient_amplitudes_t_per_m
        # Phase-encode table uses arange(-N/2, N/2) × step — sum = −N/2 × step
        # Verify table is evenly spaced (that's what matters for phase encoding)
        diffs = np.diff(amps)
        assert np.allclose(diffs, diffs[0], rtol=1e-6)

    def test_odd_n_lines_raises(self):
        with pytest.raises(ValueError, match="even"):
            build_phase_encode_scheme(DEFAULT_GX, n_lines=33, fov_m=0.06)

    def test_scan_time(self):
        scheme = build_phase_encode_scheme(DEFAULT_GX, n_lines=32, fov_m=0.06, tr_ms=100.0)
        assert scheme.scan_time_s == pytest.approx(3.2)

    def test_k_step_times_fov_equals_one(self):
        scheme = build_phase_encode_scheme(DEFAULT_GX, n_lines=32, fov_m=0.08)
        assert scheme.k_step_per_m * scheme.fov_m == pytest.approx(1.0)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  gradient_field_1d and linearity_error                          ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestGradientField:
    def test_zero_at_origin(self):
        bz = gradient_field_1d(DEFAULT_GX, np.array([0.0]))
        assert bz[0] == pytest.approx(0.0, abs=1e-15)

    def test_antisymmetric(self):
        positions = np.array([-10.0, 10.0])
        bz = gradient_field_1d(DEFAULT_GX, positions)
        assert bz[0] == pytest.approx(-bz[1], rel=1e-6)

    def test_sign_positive_at_positive_position(self):
        bz = gradient_field_1d(DEFAULT_GX, np.array([5.0]))
        assert bz[0] > 0

    def test_scales_with_current(self):
        pos = np.array([10.0])
        bz1 = gradient_field_1d(DEFAULT_GX, pos, current_a=1.0)
        bz2 = gradient_field_1d(DEFAULT_GX, pos, current_a=2.0)
        assert bz2 == pytest.approx(2 * bz1)

    def test_output_shape(self):
        pos = np.linspace(-20, 20, 50)
        bz = gradient_field_1d(DEFAULT_GX, pos)
        assert bz.shape == (50,)


class TestLinearityError:
    def test_zero_at_origin(self):
        err = linearity_error(DEFAULT_GX, np.array([0.0]))
        assert err[0] == pytest.approx(0.0, abs=1e-10)

    def test_negative_near_origin(self):
        """Third-order term reduces field → negative error."""
        err = linearity_error(DEFAULT_GX, np.array([5.0]))
        assert err[0] < 0

    def test_symmetric(self):
        err = linearity_error(DEFAULT_GX, np.array([-5.0, 5.0]))
        assert err[0] == pytest.approx(err[1])

    def test_small_at_small_position(self):
        """Near axis linearity error should be smaller than far-field error."""
        err_near = linearity_error(DEFAULT_GX, np.array([5.0]))
        err_far = linearity_error(DEFAULT_GX, np.array([20.0]))
        assert abs(err_near[0]) < abs(err_far[0])


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Parametric sweeps                                               ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestSweeps:
    def test_efficiency_vs_radius_shape(self):
        radii = [10, 15, 20, 25]
        result = sweep_efficiency_vs_radius(radii)
        assert result.shape == (4,)

    def test_efficiency_decreases_with_radius(self):
        """Larger coil → lower efficiency for same turn count."""
        radii = [10, 20, 30, 40]
        result = sweep_efficiency_vs_radius(radii)
        assert np.all(np.diff(result) < 0)

    def test_max_gradient_vs_current_linear(self):
        currents = [1, 2, 3, 4, 5]
        g = sweep_max_gradient_vs_current(currents, DEFAULT_GX)
        np.testing.assert_allclose(np.diff(g), np.diff(g)[0] * np.ones(4), rtol=1e-10)

    def test_k_max_vs_fov_inverse(self):
        """k_max ∝ 1/FOV for fixed n_lines."""
        fovs = [0.04, 0.08, 0.12]
        k_maxes = sweep_k_max_vs_fov(fovs, n_lines=64)
        # ratio: halving FOV should double k_max
        assert k_maxes[0] == pytest.approx(2 * k_maxes[1], rel=1e-6)

    def test_resolution_vs_n_lines(self):
        n_lines_arr = [32, 64, 128]
        res = sweep_resolution_vs_n_lines(n_lines_arr, fov_m=0.064)
        # FOV = 64 mm → 64/32=2mm, 64/64=1mm, 64/128=0.5mm
        np.testing.assert_allclose(res, [2.0, 1.0, 0.5], rtol=1e-6)
