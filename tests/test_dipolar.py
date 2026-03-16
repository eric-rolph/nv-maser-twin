"""Tests for the dipolar spin-spin interaction module."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nv_maser.config import DipolarConfig, SpectralConfig
from nv_maser.physics.dipolar import (
    apply_dipolar_refilling,
    estimate_dipolar_coupling_hz,
    estimate_refilling_time_us,
    spectral_diffusion_step,
    stretched_exponential_refill,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def delta_hz() -> np.ndarray:
    return np.linspace(-50e6, 50e6, 201)


@pytest.fixture
def uniform_profile() -> np.ndarray:
    return np.full(201, 0.25)


@pytest.fixture
def holed_profile(delta_hz: np.ndarray) -> np.ndarray:
    """Profile with a spectral hole at centre."""
    p = np.full(201, 0.25)
    half_kappa = 5e4
    lorentzian = half_kappa**2 / (delta_hz**2 + half_kappa**2)
    return p * (1.0 - 0.8 * lorentzian)


# ── Stretched exponential refilling ──────────────────────────────


class TestStretchedExponentialRefill:
    def test_zero_dt_unchanged(self, holed_profile: np.ndarray, uniform_profile: np.ndarray) -> None:
        result = stretched_exponential_refill(
            holed_profile, uniform_profile, dt_s=0.0, refilling_time_s=11.6e-6,
        )
        np.testing.assert_array_equal(result, holed_profile)

    def test_long_time_converges_to_equilibrium(
        self, holed_profile: np.ndarray, uniform_profile: np.ndarray,
    ) -> None:
        result = stretched_exponential_refill(
            holed_profile, uniform_profile, dt_s=1.0, refilling_time_s=11.6e-6,
        )
        np.testing.assert_allclose(result, uniform_profile, atol=1e-10)

    def test_partial_refill(
        self, holed_profile: np.ndarray, uniform_profile: np.ndarray,
    ) -> None:
        result = stretched_exponential_refill(
            holed_profile, uniform_profile,
            dt_s=11.6e-6,  # dt = T_r
            refilling_time_s=11.6e-6,
            alpha=0.5,
        )
        mid = len(result) // 2
        # Should be between hole and equilibrium
        assert holed_profile[mid] < result[mid] < uniform_profile[mid]

    def test_alpha_one_is_exponential(
        self, holed_profile: np.ndarray, uniform_profile: np.ndarray,
    ) -> None:
        dt = 11.6e-6
        tr = 11.6e-6
        result = stretched_exponential_refill(
            holed_profile, uniform_profile, dt_s=dt, refilling_time_s=tr, alpha=1.0,
        )
        expected_decay = math.exp(-1.0)
        expected = uniform_profile - (uniform_profile - holed_profile) * expected_decay
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_alpha_half_less_decay_than_one(
        self, holed_profile: np.ndarray, uniform_profile: np.ndarray,
    ) -> None:
        """α=0.5 decays faster initially (for dt < T_r) than α=1."""
        dt = 5e-6  # less than T_r
        tr = 11.6e-6
        r05 = stretched_exponential_refill(
            holed_profile, uniform_profile, dt_s=dt, refilling_time_s=tr, alpha=0.5,
        )
        r10 = stretched_exponential_refill(
            holed_profile, uniform_profile, dt_s=dt, refilling_time_s=tr, alpha=1.0,
        )
        mid = len(r05) // 2
        # For dt/T_r < 1, exp(-(dt/T_r)^0.5) < exp(-dt/T_r)
        # so α=0.5 recovers MORE toward equilibrium
        assert r05[mid] > r10[mid]

    def test_returns_new_array(
        self, holed_profile: np.ndarray, uniform_profile: np.ndarray,
    ) -> None:
        result = stretched_exponential_refill(
            holed_profile, uniform_profile, dt_s=1e-6, refilling_time_s=11.6e-6,
        )
        assert result is not holed_profile


# ── Spectral diffusion step ──────────────────────────────────────


class TestSpectralDiffusionStep:
    def test_zero_diffusion_unchanged(self, uniform_profile: np.ndarray) -> None:
        result = spectral_diffusion_step(
            uniform_profile, d_freq_hz=5e5, diffusion_coeff_hz2_per_s=0.0, dt_s=1e-6,
        )
        np.testing.assert_array_equal(result, uniform_profile)

    def test_uniform_profile_no_change(self, uniform_profile: np.ndarray) -> None:
        """Diffusion of a flat profile should produce no change (Laplacian = 0)."""
        result = spectral_diffusion_step(
            uniform_profile, d_freq_hz=5e5, diffusion_coeff_hz2_per_s=1e12, dt_s=1e-9,
        )
        np.testing.assert_allclose(result, uniform_profile, atol=1e-12)

    def test_hole_fills_in(self, holed_profile: np.ndarray) -> None:
        """Diffusion should reduce the depth of a spectral hole."""
        mid = len(holed_profile) // 2
        before = holed_profile[mid]
        result = spectral_diffusion_step(
            holed_profile, d_freq_hz=5e5, diffusion_coeff_hz2_per_s=1e12, dt_s=1e-9,
        )
        after = result[mid]
        assert after > before

    def test_cfl_violation_raises(self, holed_profile: np.ndarray) -> None:
        with pytest.raises(ValueError, match="CFL"):
            spectral_diffusion_step(
                holed_profile, d_freq_hz=5e5,
                diffusion_coeff_hz2_per_s=1e15,
                dt_s=1.0,
            )

    def test_conserves_total_inversion(self, holed_profile: np.ndarray) -> None:
        """Diffusion with Neumann BCs conserves total integral."""
        d_freq = 5e5
        result = spectral_diffusion_step(
            holed_profile, d_freq_hz=d_freq,
            diffusion_coeff_hz2_per_s=1e12,
            dt_s=1e-9,
        )
        total_before = np.sum(holed_profile) * d_freq
        total_after = np.sum(result) * d_freq
        assert total_after == pytest.approx(total_before, rel=1e-6)


# ── Estimate dipolar coupling ───────────────────────────────────


class TestEstimateDipolarCoupling:
    def test_positive_for_nonzero_density(self) -> None:
        j = estimate_dipolar_coupling_hz(1.76e24)  # 10 ppm
        assert j > 0

    def test_zero_density_zero_coupling(self) -> None:
        assert estimate_dipolar_coupling_hz(0.0) == 0.0

    def test_order_of_magnitude(self) -> None:
        """At 10 ppm, nearest-neighbour coupling should be ~100 kHz."""
        j = estimate_dipolar_coupling_hz(1.76e24)
        assert 10e3 < j < 1e6  # 10 kHz to 1 MHz range

    def test_increases_with_density(self) -> None:
        j_low = estimate_dipolar_coupling_hz(1e23)
        j_high = estimate_dipolar_coupling_hz(1e24)
        assert j_high > j_low


# ── Estimate refilling time ──────────────────────────────────────


class TestEstimateRefillingTime:
    def test_reference_point(self) -> None:
        """11.6 μs at 10 ppm (calibration point from Kersten 2026)."""
        t_r = estimate_refilling_time_us(1.76e24)
        assert t_r == pytest.approx(11.6, rel=1e-3)

    def test_lower_density_slower(self) -> None:
        t_low = estimate_refilling_time_us(1e23)
        t_high = estimate_refilling_time_us(1e24)
        assert t_low > t_high

    def test_zero_density_infinite(self) -> None:
        assert estimate_refilling_time_us(0.0) == float("inf")


# ── apply_dipolar_refilling dispatcher ───────────────────────────


class TestApplyDipolarRefilling:
    def test_disabled_returns_copy(
        self, holed_profile: np.ndarray, uniform_profile: np.ndarray, delta_hz: np.ndarray,
    ) -> None:
        cfg = DipolarConfig(enable=False)
        result = apply_dipolar_refilling(
            holed_profile, uniform_profile, delta_hz, cfg, dt_s=1e-6,
        )
        np.testing.assert_array_equal(result, holed_profile)

    def test_stretched_exp_mode(
        self, holed_profile: np.ndarray, uniform_profile: np.ndarray, delta_hz: np.ndarray,
    ) -> None:
        cfg = DipolarConfig(enable=True, refilling_time_us=11.6, stretch_exponent=0.5)
        result = apply_dipolar_refilling(
            holed_profile, uniform_profile, delta_hz, cfg, dt_s=11.6e-6,
        )
        mid = len(result) // 2
        assert result[mid] > holed_profile[mid]

    def test_diffusion_mode(
        self, holed_profile: np.ndarray, uniform_profile: np.ndarray, delta_hz: np.ndarray,
    ) -> None:
        cfg = DipolarConfig(
            enable=True, diffusion_coefficient_mhz2_per_us=0.001,
        )
        result = apply_dipolar_refilling(
            holed_profile, uniform_profile, delta_hz, cfg, dt_s=1e-9,
        )
        mid = len(result) // 2
        assert result[mid] > holed_profile[mid]
