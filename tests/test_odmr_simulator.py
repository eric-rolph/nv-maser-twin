"""Tests for the ODMR spectrum simulator (NVision-inspired)."""
from __future__ import annotations

import numpy as np
import pytest

from nv_maser.config import NVConfig
from nv_maser.physics.odmr_simulator import (
    ODMRResult,
    FitResult,
    CrossValidation,
    compute_odmr_spectrum,
    simulate_odmr_sweep,
    fit_odmr_spectrum,
    cross_validate_linewidth,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def nv_config() -> NVConfig:
    """Default NV config with 1 μs T2*."""
    return NVConfig()


@pytest.fixture
def freq_sweep() -> np.ndarray:
    """Frequency sweep around 2.87 GHz ± 50 MHz."""
    return np.linspace(2.82, 2.92, 501)


# ── Lineshape tests ──────────────────────────────────────────────────

class TestComputeODMRSpectrum:
    """Test the raw spectrum computation."""

    def test_background_far_from_resonance(self, freq_sweep):
        """Signal should be ~1.0 far from resonance."""
        signal = compute_odmr_spectrum(
            freq_sweep, center_freq_ghz=2.87,
            linewidth_ghz=0.001, contrast=0.03,
        )
        # Check edges (far from 2.87 GHz)
        assert signal[0] > 0.999
        assert signal[-1] > 0.999

    def test_dip_at_center(self, freq_sweep):
        """Signal should dip below background at resonance."""
        signal = compute_odmr_spectrum(
            freq_sweep, center_freq_ghz=2.87,
            linewidth_ghz=0.001, contrast=0.03,
        )
        center_idx = len(freq_sweep) // 2
        assert signal[center_idx] < 1.0

    def test_signal_shape(self, freq_sweep):
        """Output should have correct length and dtype."""
        signal = compute_odmr_spectrum(
            freq_sweep, center_freq_ghz=2.87,
            linewidth_ghz=0.001, contrast=0.03,
        )
        assert signal.shape == freq_sweep.shape
        assert signal.dtype == np.float64

    def test_hyperfine_splitting_creates_three_dips(self):
        """With splitting, there should be 3 dips."""
        freq = np.linspace(2.85, 2.89, 2001)
        signal = compute_odmr_spectrum(
            freq, center_freq_ghz=2.87,
            linewidth_ghz=0.0005,
            contrast=0.0001,
            splitting_ghz=0.005,
            k_np=3.0,
        )
        # Find local minima
        deriv = np.diff(np.sign(np.diff(signal)))
        local_min_count = np.sum(deriv > 0)
        assert local_min_count >= 3, f"Expected ≥3 dips, found {local_min_count}"

    def test_voigt_wider_than_lorentzian(self, freq_sweep):
        """Voigt profile should have broader wings than pure Lorentzian."""
        sig_lor = compute_odmr_spectrum(
            freq_sweep, center_freq_ghz=2.87,
            linewidth_ghz=0.001, contrast=0.03,
            profile="lorentzian",
        )
        sig_voigt = compute_odmr_spectrum(
            freq_sweep, center_freq_ghz=2.87,
            linewidth_ghz=0.001, contrast=0.03,
            profile="voigt", gaussian_fwhm_ghz=0.005,
        )
        # Voigt has broader wings: at an offset from center, the Voigt dip
        # is deeper (signal lower) than the Lorentzian.
        offset_idx = len(freq_sweep) // 2 + 25  # ~5 MHz offset
        assert sig_voigt[offset_idx] < sig_lor[offset_idx]

    def test_k_np_asymmetry(self):
        """k_np > 1 should make right peak deeper than left."""
        freq = np.linspace(2.85, 2.89, 2001)
        signal = compute_odmr_spectrum(
            freq, center_freq_ghz=2.87,
            linewidth_ghz=0.0005,
            contrast=0.0001,
            splitting_ghz=0.005,
            k_np=3.0,
        )
        # Left dip (A/k_np) should be shallower than right dip (A*k_np)
        left_idx = np.argmin(np.abs(freq - (2.87 - 0.005)))
        right_idx = np.argmin(np.abs(freq - (2.87 + 0.005)))
        left_depth = 1.0 - signal[left_idx]
        right_depth = 1.0 - signal[right_idx]
        assert right_depth > left_depth

    def test_zero_contrast_is_flat(self, freq_sweep):
        """Zero contrast should give flat background."""
        signal = compute_odmr_spectrum(
            freq_sweep, center_freq_ghz=2.87,
            linewidth_ghz=0.001, contrast=0.0,
        )
        np.testing.assert_allclose(signal, 1.0, atol=1e-12)


# ── Sweep simulation tests ──────────────────────────────────────────

class TestSimulateODMRSweep:
    """Test the full sweep simulation."""

    def test_basic_sweep(self, nv_config):
        """Should produce valid ODMRResult."""
        result = simulate_odmr_sweep(0.05, nv_config, n_points=201)
        assert isinstance(result, ODMRResult)
        assert len(result.frequencies_ghz) == 201
        assert len(result.signal) == 201
        assert result.noisy_signal is None  # no noise requested
        assert result.center_freq_ghz > 0
        assert result.linewidth_ghz > 0

    def test_center_frequency_tracks_field(self, nv_config):
        """Higher B-field → lower ν− (for |0⟩→|−1⟩)."""
        r1 = simulate_odmr_sweep(0.03, nv_config)
        r2 = simulate_odmr_sweep(0.06, nv_config)
        assert r2.center_freq_ghz < r1.center_freq_ghz

    def test_photon_noise_adds_variance(self, nv_config):
        """Poisson noise should increase signal variance."""
        clean = simulate_odmr_sweep(0.05, nv_config, n_points=501)
        noisy = simulate_odmr_sweep(
            0.05, nv_config, n_points=501,
            photon_count=1000.0, seed=42,
        )
        assert noisy.noisy_signal is not None
        # Noisy signal should differ from clean
        diff = np.abs(noisy.noisy_signal - noisy.signal)
        assert np.mean(diff) > 0

    def test_gaussian_noise(self, nv_config):
        """Gaussian noise should add scatter."""
        result = simulate_odmr_sweep(
            0.05, nv_config, n_points=501,
            noise_std=0.01, seed=42,
        )
        assert result.noisy_signal is not None
        noise = result.noisy_signal - result.signal
        # Noise std should be roughly what we asked for
        assert 0.005 < np.std(noise) < 0.02

    def test_drift_noise(self, nv_config):
        """Drift should create a trend in the signal."""
        result = simulate_odmr_sweep(
            0.05, nv_config, n_points=501,
            drift_per_point=0.0001, seed=42,
        )
        assert result.noisy_signal is not None
        drift_component = result.noisy_signal - result.signal
        # Drift should create a slope
        first_half_mean = np.mean(drift_component[:250])
        second_half_mean = np.mean(drift_component[250:])
        assert first_half_mean < second_half_mean

    def test_reproducible_with_seed(self, nv_config):
        """Same seed → same noise."""
        r1 = simulate_odmr_sweep(0.05, nv_config, noise_std=0.01, seed=42)
        r2 = simulate_odmr_sweep(0.05, nv_config, noise_std=0.01, seed=42)
        np.testing.assert_array_equal(r1.noisy_signal, r2.noisy_signal)


# ── Fitting tests ────────────────────────────────────────────────────

class TestFitODMRSpectrum:
    """Test Lorentzian spectral fitting."""

    def test_fit_clean_spectrum(self, nv_config):
        """Fit should recover parameters from clean spectrum."""
        result = simulate_odmr_sweep(0.05, nv_config, n_points=501)
        fit = fit_odmr_spectrum(
            result.frequencies_ghz, result.signal, fit_splitting=False,
        )
        assert isinstance(fit, FitResult)
        # Fitted frequency should be close to true
        assert abs(fit.center_freq_ghz - result.center_freq_ghz) < 0.001
        assert fit.linewidth_ghz > 0
        assert fit.residual_rms < 0.01

    def test_fit_noisy_spectrum(self, nv_config):
        """Fit should still work with moderate noise."""
        result = simulate_odmr_sweep(
            0.05, nv_config, n_points=501,
            noise_std=0.001, seed=42,
        )
        fit = fit_odmr_spectrum(
            result.frequencies_ghz, result.noisy_signal, fit_splitting=False,
        )
        # Frequency recovery within 5 MHz (shallow 1% dip + noise)
        assert abs(fit.center_freq_ghz - result.center_freq_ghz) < 0.005
        assert fit.linewidth_ghz > 0

    def test_fit_returns_all_params(self, nv_config):
        """Fit result should contain all named parameters."""
        result = simulate_odmr_sweep(0.05, nv_config, n_points=501)
        fit = fit_odmr_spectrum(result.frequencies_ghz, result.signal)
        for key in ["frequency", "linewidth", "splitting", "contrast", "k_np", "background"]:
            assert key in fit.params

    def test_fit_with_splitting(self, nv_config):
        """Fit with splitting enabled should find splitting > 0."""
        frequencies = np.linspace(2.45, 2.55, 2001)
        signal = compute_odmr_spectrum(
            frequencies, center_freq_ghz=2.50,
            linewidth_ghz=0.001,
            contrast=0.0001,
            splitting_ghz=0.005,
            k_np=2.0,
        )
        fit = fit_odmr_spectrum(
            frequencies, signal,
            initial_freq_ghz=2.50,
            fit_splitting=True,
        )
        # Should detect non-zero splitting
        assert fit.splitting_ghz > 0.001


# ── Cross-validation tests ──────────────────────────────────────────

class TestCrossValidateLinewidth:
    """Test analytical vs. fitted linewidth comparison."""

    def test_uniform_field_consistent(self, nv_config):
        """Uniform field should give consistent linewidths."""
        # Uniform field → zero inhomogeneous broadening
        b_field = np.full((10, 10), 0.05, dtype=np.float32)
        mask = np.ones((10, 10), dtype=bool)

        cv = cross_validate_linewidth(b_field, mask, nv_config)
        assert isinstance(cv, CrossValidation)
        assert cv.analytical_linewidth_ghz > 0
        assert cv.fitted_linewidth_ghz > 0
        assert cv.consistent

    def test_nonuniform_field(self, nv_config):
        """Non-uniform field should still produce valid comparison."""
        rng = np.random.default_rng(42)
        b_field = np.full((10, 10), 0.05, dtype=np.float32)
        b_field += rng.normal(0, 0.001, size=(10, 10)).astype(np.float32)
        mask = np.ones((10, 10), dtype=bool)

        cv = cross_validate_linewidth(b_field, mask, nv_config)
        assert cv.analytical_linewidth_ghz > 0
        assert cv.fitted_linewidth_ghz > 0
        # Both should be positive, even if not perfectly consistent
        assert cv.relative_error >= 0
