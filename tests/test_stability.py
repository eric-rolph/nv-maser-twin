"""Tests for the oscillator stability and Allan deviation module.

Physics covered:
- compute_white_fm_allan_deviation: σ_B(τ) = η_ST / √τ  [T]
- compute_allan_deviation_from_psd: numerical ADEV from S_φ(f)
- compute_oscillator_stability: full characterisation (both paths)
- OscillatorStabilityResult: unit conversions, slope, scalar summary

Key physical invariants verified:
- σ_B(τ=1 s) = η_ST = SensitivityResult.allan_deviation_1s_t
- White-FM averaging law: σ_B(4τ) = σ_B(τ) / 2  (doubling τ twice → 2× improvement)
- Exact formula: σ_B(τ) = η_ST / √τ for pure Schawlow-Townes noise
- Allan slope ≈ −0.5 for white FM noise throughout τ array
- Numerical PSD integral matches analytical floor to < 1 % for well-resolved grid
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nv_maser.config import MaserConfig, NVConfig, SignalChainConfig
from nv_maser.physics.quantum_noise import MaserNoiseResult, compute_phase_noise_spectrum
from nv_maser.physics.sensitivity import compute_sensitivity
from nv_maser.physics.signal_chain import compute_signal_chain_budget
from nv_maser.physics.stability import (
    CombinedADEVResult,
    NoiseProcessADEV,
    OscillatorStabilityResult,
    _compute_log_slope,
    compute_allan_deviation_from_psd,
    compute_combined_allan_deviation,
    compute_flicker_fm_adev,
    compute_oscillator_stability,
    compute_random_walk_fm_adev,
    compute_white_fm_allan_deviation,
    compute_white_pm_adev,
)

_TWO_PI = 2.0 * math.pi
_HBAR = 1.054571817e-34
_KB = 1.380649e-23


# ── Shared helpers ─────────────────────────────────────────────────────────


def _make_noise(
    n_sp: float = 1.5,
    kappa_hz: float = 147_000.0,
    nu0_hz: float = 1.47e9,
    n_bar: float = 200.0,
    output_power_w: float = 10.0e-12,
) -> MaserNoiseResult:
    """Build a MaserNoiseResult with fully controlled parameters."""
    t_noise = (_HBAR * _TWO_PI * nu0_hz * n_sp) / _KB
    st_lw = kappa_hz * n_sp / (2.0 * n_bar)
    rin = 2.0 * n_sp / n_bar
    return MaserNoiseResult(
        population_inversion_factor=n_sp,
        added_noise_number=n_sp,
        schawlow_townes_linewidth_hz=st_lw,
        noise_temperature_k=t_noise,
        steady_state_photons=n_bar,
        output_power_w=output_power_w,
        phase_noise_1hz_dbc_hz=10.0 * math.log10(st_lw / _TWO_PI),
        rin_floor_per_hz=rin,
        rin_floor_dbc_hz=10.0 * math.log10(rin),
        cavity_linewidth_hz=kappa_hz,
        cavity_frequency_hz=nu0_hz,
    )


@pytest.fixture
def nv_cfg() -> NVConfig:
    return NVConfig()


@pytest.fixture
def noise() -> MaserNoiseResult:
    """n_sp=1.5, κ=147 kHz, N̄=200, ν₀=1.47 GHz → Δν_ST=551.25 Hz."""
    return _make_noise()


@pytest.fixture
def tau_arr() -> np.ndarray:
    """Standard log-spaced τ array: 0.001 → 1000 s (50 points)."""
    return np.logspace(-3, 3, 50)


# ── Compute η_ST directly for assertions ──────────────────────────────────

def _eta_st(noise: MaserNoiseResult, nv_cfg: NVConfig) -> float:
    """Schawlow-Townes sensitivity η_ST = √(Δν_ST / 2π) / γ_e  [T/√Hz]."""
    gamma_e = nv_cfg.gamma_e_ghz_per_t * 1e9
    return math.sqrt(noise.schawlow_townes_linewidth_hz / _TWO_PI) / gamma_e


# ══ TestOscillatorStabilityResult ═════════════════════════════════════════


class TestOscillatorStabilityResult:
    """Structural and unit-conversion tests for the result dataclass."""

    def test_frozen_dataclass(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        result = compute_oscillator_stability(tau_arr, noise, nv_cfg)
        with pytest.raises((AttributeError, TypeError)):
            result.sigma_b_at_1s_t = 0.0  # type: ignore[misc]

    def test_unit_conversions_nt(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        result = compute_oscillator_stability(tau_arr, noise, nv_cfg)
        np.testing.assert_allclose(result.sigma_b_nt, result.sigma_b_t * 1.0e9)

    def test_unit_conversions_pt(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        result = compute_oscillator_stability(tau_arr, noise, nv_cfg)
        np.testing.assert_allclose(result.sigma_b_pt, result.sigma_b_t * 1.0e12)

    def test_array_lengths_match(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        result = compute_oscillator_stability(tau_arr, noise, nv_cfg)
        n = len(tau_arr)
        assert len(result.tau_s) == n
        assert len(result.sigma_b_t) == n
        assert len(result.sigma_b_nt) == n
        assert len(result.sigma_b_pt) == n
        assert len(result.sigma_y) == n
        assert len(result.sigma_b_st_floor_t) == n
        assert len(result.allan_slope) == n

    def test_sigma_y_consistent_with_sigma_b(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """σ_y = σ_B × γ_e / ν₀."""
        result = compute_oscillator_stability(tau_arr, noise, nv_cfg)
        gamma_e = nv_cfg.gamma_e_ghz_per_t * 1e9
        nu0 = noise.cavity_frequency_hz
        expected_sigma_y = result.sigma_b_t * gamma_e / nu0
        np.testing.assert_allclose(result.sigma_y, expected_sigma_y, rtol=1e-12)


# ══ TestComputeWhiteFmAllanDeviation ══════════════════════════════════════


class TestComputeWhiteFmAllanDeviation:
    """Unit tests for compute_white_fm_allan_deviation()."""

    def test_returns_array_same_length(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        result = compute_white_fm_allan_deviation(tau_arr, noise, nv_cfg)
        assert result.shape == tau_arr.shape

    def test_exact_formula_single_tau(
        self, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """σ_B(τ) = η_ST / √τ exactly — verified at five representative τ values."""
        eta = _eta_st(noise, nv_cfg)
        for tau_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
            tau = np.array([tau_val])
            result = compute_white_fm_allan_deviation(tau, noise, nv_cfg)
            expected = eta / math.sqrt(tau_val)
            assert result[0] == pytest.approx(expected, rel=1e-12)

    def test_at_1s_equals_eta_st(
        self, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """σ_B(τ=1 s) = η_ST."""
        eta = _eta_st(noise, nv_cfg)
        result = compute_white_fm_allan_deviation(np.array([1.0]), noise, nv_cfg)
        assert result[0] == pytest.approx(eta, rel=1e-12)

    def test_quadruple_tau_halves_sigma(
        self, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """White-FM averaging law: σ_B(4τ) = σ_B(τ) / 2."""
        tau_base = np.array([0.25, 1.0, 4.0, 16.0])
        tau_quad = tau_base * 4.0
        sig_base = compute_white_fm_allan_deviation(tau_base, noise, nv_cfg)
        sig_quad = compute_white_fm_allan_deviation(tau_quad, noise, nv_cfg)
        np.testing.assert_allclose(sig_quad, sig_base / 2.0, rtol=1e-12)

    def test_all_positive_finite(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        result = compute_white_fm_allan_deviation(tau_arr, noise, nv_cfg)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0.0)

    def test_sqrt_tau_scaling(
        self, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """σ_B ∝ τ^{-1/2}: ratios at τ and 100τ should be exactly 10."""
        tau = np.array([0.1, 10.0])
        result = compute_white_fm_allan_deviation(tau, noise, nv_cfg)
        assert result[0] / result[1] == pytest.approx(math.sqrt(10.0 / 0.1), rel=1e-12)

    def test_monotonically_decreasing(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """Pure white FM: σ_B(τ) strictly decreasing."""
        result = compute_white_fm_allan_deviation(tau_arr, noise, nv_cfg)
        assert np.all(np.diff(result) < 0.0)

    def test_larger_delta_nu_gives_larger_sigma(
        self, nv_cfg: NVConfig
    ) -> None:
        """Higher Δν_ST → larger ST linewidth → worse field stability."""
        tau = np.array([1.0])
        noise_lo = _make_noise(n_bar=1000.0)  # low Δν_ST
        noise_hi = _make_noise(n_bar=10.0)    # high Δν_ST
        sig_lo = compute_white_fm_allan_deviation(tau, noise_lo, nv_cfg)
        sig_hi = compute_white_fm_allan_deviation(tau, noise_hi, nv_cfg)
        assert sig_hi[0] > sig_lo[0]

    def test_error_on_non_positive_tau(
        self, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        with pytest.raises(ValueError, match="tau_s"):
            compute_white_fm_allan_deviation(np.array([0.0, 1.0]), noise, nv_cfg)

    def test_single_element_tau(
        self, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """Scalar-like input τ = [5.0] should work."""
        result = compute_white_fm_allan_deviation(np.array([5.0]), noise, nv_cfg)
        eta = _eta_st(noise, nv_cfg)
        assert result[0] == pytest.approx(eta / math.sqrt(5.0), rel=1e-12)


# ══ TestComputeAllanDeviationFromPsd ══════════════════════════════════════


class TestComputeAllanDeviationFromPsd:
    """Tests for compute_allan_deviation_from_psd()."""

    # Frequency grid fine enough for <1% accuracy at τ ∈ [0.1, 10] s.
    _FREQ = np.logspace(-2, 7, 20_000)

    @pytest.fixture
    def psd(self, noise: MaserNoiseResult) -> object:
        return compute_phase_noise_spectrum(
            noise.schawlow_townes_linewidth_hz,
            self._FREQ,
        )

    def test_returns_correct_shape(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, psd: object
    ) -> None:
        result = compute_allan_deviation_from_psd(
            tau_arr, psd, noise.cavity_frequency_hz,  # type: ignore[arg-type]
            noise.cavity_frequency_hz  # placeholder for gamma_e (just checks shape)
        )
        # We pass cavity_frequency_hz as gamma_e too just to check shape — not physics.
        assert result.shape == tau_arr.shape

    def test_st_psd_matches_analytical_within_1pct(
        self, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """For S_φ = Δν_ST/(πf²) the numerical ADEV matches the analytical floor to <1%."""
        tau = np.array([0.1, 1.0, 10.0])
        freq = np.logspace(-3, 7, 20_000)
        psd = compute_phase_noise_spectrum(noise.schawlow_townes_linewidth_hz, freq)
        gamma_e = nv_cfg.gamma_e_ghz_per_t * 1e9

        numerical = compute_allan_deviation_from_psd(
            tau, psd, noise.cavity_frequency_hz, gamma_e  # type: ignore[arg-type]
        )
        analytical = compute_white_fm_allan_deviation(tau, noise, nv_cfg)

        np.testing.assert_allclose(numerical, analytical, rtol=0.01)

    def test_all_positive_finite(
        self, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        tau = np.array([0.1, 1.0, 10.0])
        freq = np.logspace(-3, 7, 10_000)
        psd = compute_phase_noise_spectrum(noise.schawlow_townes_linewidth_hz, freq)
        gamma_e = nv_cfg.gamma_e_ghz_per_t * 1e9
        result = compute_allan_deviation_from_psd(
            tau, psd, noise.cavity_frequency_hz, gamma_e  # type: ignore[arg-type]
        )
        assert np.all(np.isfinite(result))
        assert np.all(result > 0.0)

    def test_larger_st_linewidth_increases_adev(
        self, nv_cfg: NVConfig
    ) -> None:
        """Doubling Δν_ST doubles S_y(f) → doubles σ_y² → multiplies σ_B by √2."""
        tau = np.array([1.0])
        freq = np.logspace(-3, 7, 10_000)
        noise_lo = _make_noise(n_bar=400.0)  # Δν_ST / 2
        noise_hi = _make_noise(n_bar=200.0)  # Δν_ST (baseline)
        gamma_e = nv_cfg.gamma_e_ghz_per_t * 1e9

        psd_lo = compute_phase_noise_spectrum(noise_lo.schawlow_townes_linewidth_hz, freq)
        psd_hi = compute_phase_noise_spectrum(noise_hi.schawlow_townes_linewidth_hz, freq)

        sig_lo = compute_allan_deviation_from_psd(
            tau, psd_lo, noise_lo.cavity_frequency_hz, gamma_e  # type: ignore[arg-type]
        )
        sig_hi = compute_allan_deviation_from_psd(
            tau, psd_hi, noise_hi.cavity_frequency_hz, gamma_e  # type: ignore[arg-type]
        )

        # Δν_ST(hi) = 2 × Δν_ST(lo) → σ_B(hi) ≈ √2 × σ_B(lo)
        assert sig_hi[0] / sig_lo[0] == pytest.approx(math.sqrt(2.0), rel=0.01)

    def test_error_on_non_positive_tau(
        self, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        freq = np.logspace(-2, 6, 100)
        psd = compute_phase_noise_spectrum(noise.schawlow_townes_linewidth_hz, freq)
        gamma_e = nv_cfg.gamma_e_ghz_per_t * 1e9
        with pytest.raises(ValueError, match="tau_s"):
            compute_allan_deviation_from_psd(
                np.array([-1.0, 1.0]), psd, noise.cavity_frequency_hz, gamma_e  # type: ignore[arg-type]
            )

    def test_error_on_non_positive_carrier_frequency(
        self, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        freq = np.logspace(-2, 6, 100)
        psd = compute_phase_noise_spectrum(noise.schawlow_townes_linewidth_hz, freq)
        gamma_e = nv_cfg.gamma_e_ghz_per_t * 1e9
        with pytest.raises(ValueError, match="carrier_frequency_hz"):
            compute_allan_deviation_from_psd(
                np.array([1.0]), psd, 0.0, gamma_e  # type: ignore[arg-type]
            )


# ══ TestComputeOscillatorStability ════════════════════════════════════════


class TestComputeOscillatorStability:
    """Integration tests for compute_oscillator_stability()."""

    def test_returns_oscillator_stability_result(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        result = compute_oscillator_stability(tau_arr, noise, nv_cfg)
        assert isinstance(result, OscillatorStabilityResult)

    def test_no_psd_uses_st_floor(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """Without phase_noise_spectrum, sigma_b_t == sigma_b_st_floor_t."""
        result = compute_oscillator_stability(tau_arr, noise, nv_cfg)
        np.testing.assert_array_equal(result.sigma_b_t, result.sigma_b_st_floor_t)

    def test_with_psd_argument_runs(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """Supplying a phase_noise_spectrum succeeds without error."""
        freq = np.logspace(-3, 7, 5_000)
        psd = compute_phase_noise_spectrum(noise.schawlow_townes_linewidth_hz, freq)
        result = compute_oscillator_stability(tau_arr, noise, nv_cfg, psd)
        assert isinstance(result, OscillatorStabilityResult)
        assert np.all(np.isfinite(result.sigma_b_t))

    def test_sigma_b_at_1s_matches_sensitivity_result(
        self, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """sigma_b_at_1s_t must equal SensitivityResult.allan_deviation_1s_t."""
        tau = np.logspace(-2, 2, 30)

        # Build a minimal signal chain budget to get a SensitivityResult.
        maser_cfg = MaserConfig()
        chain_cfg = SignalChainConfig()
        budget = compute_signal_chain_budget(nv_cfg, maser_cfg, chain_cfg, 1.0, noise)
        sens = compute_sensitivity(budget, noise, nv_cfg)

        stability = compute_oscillator_stability(tau, noise, nv_cfg)

        assert stability.sigma_b_at_1s_t == pytest.approx(
            sens.allan_deviation_1s_t, rel=1e-12
        )

    def test_sigma_b_at_1s_equals_eta_st(
        self, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """sigma_b_at_1s_t = η_ST exactly."""
        eta = _eta_st(noise, nv_cfg)
        tau = np.logspace(-2, 2, 20)
        result = compute_oscillator_stability(tau, noise, nv_cfg)
        assert result.sigma_b_at_1s_t == pytest.approx(eta, rel=1e-12)

    def test_provenance_linewidth_stored(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        result = compute_oscillator_stability(tau_arr, noise, nv_cfg)
        assert result.schawlow_townes_linewidth_hz == pytest.approx(
            noise.schawlow_townes_linewidth_hz, rel=1e-12
        )

    def test_carrier_frequency_stored(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        result = compute_oscillator_stability(tau_arr, noise, nv_cfg)
        assert result.carrier_frequency_hz == pytest.approx(
            noise.cavity_frequency_hz, rel=1e-12
        )

    def test_gamma_e_stored(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        result = compute_oscillator_stability(tau_arr, noise, nv_cfg)
        expected_gamma_e = nv_cfg.gamma_e_ghz_per_t * 1e9
        assert result.gamma_e_hz_per_t == pytest.approx(expected_gamma_e, rel=1e-12)

    def test_sigma_b_at_1s_finite_positive(
        self, tau_arr: np.ndarray, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        result = compute_oscillator_stability(tau_arr, noise, nv_cfg)
        assert math.isfinite(result.sigma_b_at_1s_t)
        assert result.sigma_b_at_1s_t > 0.0

    def test_st_floor_at_tau_1_equals_eta_st(
        self, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """sigma_b_st_floor_t at τ=1 s equals η_ST regardless of other input."""
        tau = np.array([1.0])
        result = compute_oscillator_stability(tau, noise, nv_cfg)
        eta = _eta_st(noise, nv_cfg)
        assert result.sigma_b_st_floor_t[0] == pytest.approx(eta, rel=1e-12)

    def test_error_on_non_positive_tau(
        self, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        with pytest.raises(ValueError):
            compute_oscillator_stability(np.array([0.0, 1.0]), noise, nv_cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-noise-process ADEV (SS28)
# ═══════════════════════════════════════════════════════════════════════════

# Shared test constants — NV maser carrier and gyromagnetic ratio
_CARRIER_HZ = 2.87e9  # NV zero-field splitting
_GAMMA_E = 28.024e9   # γ_e in Hz/T


class TestWhitePmAdev:
    """Tests for compute_white_pm_adev()."""

    @pytest.fixture
    def tau(self):
        return np.logspace(-2, 2, 50)

    def test_slope_minus_one(self, tau):
        """WPM Allan deviation has σ_y(τ) ∝ τ^{-1} → σ_B slope = -1."""
        sigma = compute_white_pm_adev(tau, h2=1e-20, f_h=1e6,
                                       carrier_frequency_hz=_CARRIER_HZ,
                                       gamma_e_hz_per_t=_GAMMA_E)
        slope = _compute_log_slope(tau, sigma)
        assert slope == pytest.approx(-1.0, abs=0.05)

    def test_doubling_tau_halves_sigma(self, tau):
        """σ_B,WPM(2τ) = σ_B,WPM(τ) / 2."""
        sigma = compute_white_pm_adev(tau, h2=1e-20, f_h=1e6,
                                       carrier_frequency_hz=_CARRIER_HZ,
                                       gamma_e_hz_per_t=_GAMMA_E)
        ratio = sigma[1:] / sigma[:-1]
        tau_ratio = tau[1:] / tau[:-1]
        expected_ratio = 1.0 / tau_ratio
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-10)

    def test_zero_h2_gives_zero(self, tau):
        sigma = compute_white_pm_adev(tau, h2=0.0, f_h=1e6,
                                       carrier_frequency_hz=_CARRIER_HZ,
                                       gamma_e_hz_per_t=_GAMMA_E)
        np.testing.assert_allclose(sigma, 0.0)

    def test_positive_output(self, tau):
        sigma = compute_white_pm_adev(tau, h2=1e-20, f_h=1e6,
                                       carrier_frequency_hz=_CARRIER_HZ,
                                       gamma_e_hz_per_t=_GAMMA_E)
        assert np.all(sigma > 0)


class TestFlickerFmAdev:
    """Tests for compute_flicker_fm_adev()."""

    @pytest.fixture
    def tau(self):
        return np.logspace(-2, 2, 50)

    def test_constant_across_tau(self, tau):
        """Flicker FM: σ_B is constant (independent of τ)."""
        sigma = compute_flicker_fm_adev(tau, h_minus1=1e-26,
                                         carrier_frequency_hz=_CARRIER_HZ,
                                         gamma_e_hz_per_t=_GAMMA_E)
        np.testing.assert_allclose(sigma, sigma[0], rtol=1e-12)

    def test_slope_zero(self, tau):
        sigma = compute_flicker_fm_adev(tau, h_minus1=1e-26,
                                         carrier_frequency_hz=_CARRIER_HZ,
                                         gamma_e_hz_per_t=_GAMMA_E)
        slope = _compute_log_slope(tau, sigma)
        assert np.all(np.abs(slope) < 0.01)

    def test_zero_h_minus1_gives_zero(self, tau):
        sigma = compute_flicker_fm_adev(tau, h_minus1=0.0,
                                         carrier_frequency_hz=_CARRIER_HZ,
                                         gamma_e_hz_per_t=_GAMMA_E)
        np.testing.assert_allclose(sigma, 0.0)


class TestRandomWalkFmAdev:
    """Tests for compute_random_walk_fm_adev()."""

    @pytest.fixture
    def tau(self):
        return np.logspace(-2, 2, 50)

    def test_slope_plus_half(self, tau):
        """RW FM: σ_y(τ) ∝ τ^{+0.5} → σ_B slope = +0.5."""
        sigma = compute_random_walk_fm_adev(tau, h_minus2=1e-30,
                                             carrier_frequency_hz=_CARRIER_HZ,
                                             gamma_e_hz_per_t=_GAMMA_E)
        slope = _compute_log_slope(tau, sigma)
        assert slope == pytest.approx(0.5, abs=0.05)

    def test_zero_h_minus2_gives_zero(self, tau):
        sigma = compute_random_walk_fm_adev(tau, h_minus2=0.0,
                                             carrier_frequency_hz=_CARRIER_HZ,
                                             gamma_e_hz_per_t=_GAMMA_E)
        np.testing.assert_allclose(sigma, 0.0)

    def test_positive_output(self, tau):
        sigma = compute_random_walk_fm_adev(tau, h_minus2=1e-30,
                                             carrier_frequency_hz=_CARRIER_HZ,
                                             gamma_e_hz_per_t=_GAMMA_E)
        assert np.all(sigma > 0)


class TestCombinedAllanDeviation:
    """Tests for compute_combined_allan_deviation()."""

    @pytest.fixture
    def tau(self):
        return np.logspace(-2, 3, 200)

    def test_returns_correct_type(self, tau):
        wpm = NoiseProcessADEV(
            noise_type="white_pm",
            tau_s=tau,
            sigma_b_t=compute_white_pm_adev(tau, 1e-20, 1e6, _CARRIER_HZ, _GAMMA_E),
            slope=-1.0,
        )
        result = compute_combined_allan_deviation(tau, [wpm])
        assert isinstance(result, CombinedADEVResult)

    def test_single_component_matches(self, tau):
        """With one component, combined σ_B = component σ_B."""
        wpm = NoiseProcessADEV(
            noise_type="white_pm",
            tau_s=tau,
            sigma_b_t=compute_white_pm_adev(tau, 1e-20, 1e6, _CARRIER_HZ, _GAMMA_E),
            slope=-1.0,
        )
        result = compute_combined_allan_deviation(tau, [wpm])
        np.testing.assert_allclose(result.sigma_b_t, wpm.sigma_b_t, rtol=1e-12)

    def test_bathtub_has_minimum(self, tau):
        """WPM (slope -1) + RW FM (slope +0.5) creates a bathtub minimum."""
        # Choose coefficients so the crossover falls well within τ range [0.01, 1000]
        sigma_wpm = compute_white_pm_adev(tau, 1e-20, 1e6, _CARRIER_HZ, _GAMMA_E)
        # Scale RW FM so it dominates at moderate τ
        sigma_rw = compute_random_walk_fm_adev(tau, 1e-20, _CARRIER_HZ, _GAMMA_E)
        wpm = NoiseProcessADEV("white_pm", tau, sigma_wpm, -1.0)
        rw = NoiseProcessADEV("random_walk_fm", tau, sigma_rw, 0.5)
        result = compute_combined_allan_deviation(tau, [wpm, rw])
        # The minimum should NOT be at the first or last τ
        idx_min = int(np.argmin(result.sigma_b_t))
        assert 0 < idx_min < len(tau) - 1

    def test_optimal_values_consistent(self, tau):
        """optimal_tau_s and optimal_sigma_b_t match the minimum of sigma_b_t."""
        sigma_wpm = compute_white_pm_adev(tau, 1e-20, 1e6, _CARRIER_HZ, _GAMMA_E)
        sigma_rw = compute_random_walk_fm_adev(tau, 1e-30, _CARRIER_HZ, _GAMMA_E)
        wpm = NoiseProcessADEV("white_pm", tau, sigma_wpm, -1.0)
        rw = NoiseProcessADEV("random_walk_fm", tau, sigma_rw, 0.5)
        result = compute_combined_allan_deviation(tau, [wpm, rw])
        assert result.optimal_sigma_b_t == pytest.approx(
            float(np.min(result.sigma_b_t)), rel=1e-12
        )
        assert result.optimal_tau_s == pytest.approx(
            float(tau[np.argmin(result.sigma_b_t)]), rel=1e-12
        )

    def test_rss_combination(self, tau):
        """Combined variance = sum of individual variances."""
        sigma_wpm = compute_white_pm_adev(tau, 1e-20, 1e6, _CARRIER_HZ, _GAMMA_E)
        sigma_ff = compute_flicker_fm_adev(tau, 1e-26, _CARRIER_HZ, _GAMMA_E)
        wpm = NoiseProcessADEV("white_pm", tau, sigma_wpm, -1.0)
        ff = NoiseProcessADEV("flicker_fm", tau, sigma_ff, 0.0)
        result = compute_combined_allan_deviation(tau, [wpm, ff])
        expected = np.sqrt(sigma_wpm**2 + sigma_ff**2)
        np.testing.assert_allclose(result.sigma_b_t, expected, rtol=1e-12)

    def test_unit_conversions(self, tau):
        """sigma_b_nt and sigma_b_pt are correct conversions."""
        sigma_wpm = compute_white_pm_adev(tau, 1e-20, 1e6, _CARRIER_HZ, _GAMMA_E)
        wpm = NoiseProcessADEV("white_pm", tau, sigma_wpm, -1.0)
        result = compute_combined_allan_deviation(tau, [wpm])
        np.testing.assert_allclose(result.sigma_b_nt, result.sigma_b_t * 1e9, rtol=1e-12)
        np.testing.assert_allclose(result.sigma_b_pt, result.sigma_b_t * 1e12, rtol=1e-12)

# ══ TestAllanSlope ════════════════════════════════════════════════════════


class TestAllanSlope:
    """Tests for the log-log slope field and _compute_log_slope helper."""

    def test_slope_minus_half_for_white_fm(
        self, noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """White FM noise gives slope ≈ −0.5 at all interior τ points."""
        tau = np.logspace(-2, 2, 200)
        result = compute_oscillator_stability(tau, noise, nv_cfg)
        # Check interior points only (edge points use one-sided differences).
        interior_slope = result.allan_slope[1:-1]
        np.testing.assert_allclose(interior_slope, -0.5, atol=1e-3)

    def test_slope_zero_for_constant_sigma(self) -> None:
        """Constant σ_B array (flicker-like) should give slope ≈ 0."""
        tau = np.logspace(-2, 2, 100)
        sigma = np.ones_like(tau) * 1.0e-9  # flat, σ ∝ τ^0
        slope = _compute_log_slope(tau, sigma)
        np.testing.assert_allclose(slope, 0.0, atol=1e-10)

    def test_slope_plus_half_for_random_walk(self) -> None:
        """σ_B ∝ τ^{+0.5} gives slope ≈ +0.5.  Simulates random-walk FM regime."""
        tau = np.logspace(-2, 2, 100)
        sigma = 1.0e-9 * np.sqrt(tau)  # σ ∝ τ^{+0.5}
        slope = _compute_log_slope(tau, sigma)
        # Interior points should be very close to +0.5; edges use one-sided differences.
        np.testing.assert_allclose(slope[1:-1], 0.5, atol=1e-3)
