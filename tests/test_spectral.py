"""Tests for the spectral inversion profile module."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nv_maser.config import NVConfig, SpectralConfig
from nv_maser.physics.spectral import (
    build_detuning_grid,
    build_initial_inversion,
    burn_spectral_hole,
    compute_on_resonance_inversion,
    fwhm_to_sigma,
    q_gaussian,
    spectral_overlap_weights,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def spec_cfg() -> SpectralConfig:
    return SpectralConfig(
        n_freq_bins=201,
        freq_range_mhz=50.0,
        q_parameter=1.0,
        inhomogeneous_linewidth_mhz=8.65,
    )


@pytest.fixture
def nv_cfg() -> NVConfig:
    return NVConfig()


# ── Detuning grid ────────────────────────────────────────────────


class TestDetuningGrid:
    def test_shape(self, spec_cfg: SpectralConfig) -> None:
        grid = build_detuning_grid(spec_cfg)
        assert grid.shape == (201,)

    def test_symmetric(self, spec_cfg: SpectralConfig) -> None:
        grid = build_detuning_grid(spec_cfg)
        assert grid[0] == pytest.approx(-grid[-1])

    def test_zero_centred(self, spec_cfg: SpectralConfig) -> None:
        grid = build_detuning_grid(spec_cfg)
        mid = len(grid) // 2
        assert grid[mid] == pytest.approx(0.0)

    def test_range(self, spec_cfg: SpectralConfig) -> None:
        grid = build_detuning_grid(spec_cfg)
        assert grid[0] == pytest.approx(-50e6)
        assert grid[-1] == pytest.approx(50e6)


# ── q-Gaussian ───────────────────────────────────────────────────


class TestQGaussian:
    def test_gaussian_normalised(self) -> None:
        delta = np.linspace(-1e7, 1e7, 10001)
        pdf = q_gaussian(delta, sigma_hz=1e6, q=1.0)
        dx = delta[1] - delta[0]
        assert np.sum(pdf) * dx == pytest.approx(1.0, rel=1e-3)

    def test_lorentzian_like_normalised(self) -> None:
        delta = np.linspace(-1e8, 1e8, 100001)
        pdf = q_gaussian(delta, sigma_hz=1e6, q=1.9)
        dx = delta[1] - delta[0]
        assert np.sum(pdf) * dx == pytest.approx(1.0, rel=0.05)

    def test_peak_at_zero(self) -> None:
        delta = np.linspace(-1e7, 1e7, 10001)
        pdf = q_gaussian(delta, sigma_hz=1e6, q=1.0)
        mid = len(delta) // 2
        assert pdf[mid] == pdf.max()

    def test_wider_q_has_heavier_tails(self) -> None:
        delta = np.linspace(-1e7, 1e7, 10001)
        pdf_g = q_gaussian(delta, sigma_hz=1e6, q=1.0)
        pdf_l = q_gaussian(delta, sigma_hz=1e6, q=1.8)
        # At far detuning, the q=1.8 should have more weight
        wing = abs(delta) > 5e6
        assert np.mean(pdf_l[wing]) > np.mean(pdf_g[wing])

    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            q_gaussian(np.array([0.0]), sigma_hz=-1.0)


# ── FWHM conversion ─────────────────────────────────────────────


class TestFWHMToSigma:
    def test_gaussian_fwhm(self) -> None:
        fwhm = 1e6
        sigma = fwhm_to_sigma(fwhm, q=1.0)
        expected = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        assert sigma == pytest.approx(expected, rel=1e-10)

    def test_larger_q_gives_smaller_sigma(self) -> None:
        fwhm = 1e6
        s1 = fwhm_to_sigma(fwhm, q=1.0)
        s2 = fwhm_to_sigma(fwhm, q=1.5)
        # For the same FWHM, more Lorentzian shapes (heavier tails) need smaller σ
        assert s2 < s1

    def test_fwhm_roundtrip(self) -> None:
        """Check that σ from FWHM gives half-max at Δ = FWHM/2."""
        fwhm = 1e6
        sigma = fwhm_to_sigma(fwhm, q=1.0)
        delta = np.array([0.0, fwhm / 2.0])
        pdf = q_gaussian(delta, sigma, q=1.0)
        # pdf at FWHM/2 should be half the peak
        assert pdf[1] / pdf[0] == pytest.approx(0.5, rel=0.01)


# ── Initial inversion profile ───────────────────────────────────


class TestBuildInitialInversion:
    def test_peak_equals_sz0(self, nv_cfg: NVConfig, spec_cfg: SpectralConfig) -> None:
        delta, p = build_initial_inversion(nv_cfg, spec_cfg)
        sz0 = nv_cfg.pump_efficiency / 2.0
        assert p.max() == pytest.approx(sz0)

    def test_shape_matches_grid(self, nv_cfg: NVConfig, spec_cfg: SpectralConfig) -> None:
        delta, p = build_initial_inversion(nv_cfg, spec_cfg)
        assert delta.shape == p.shape == (spec_cfg.n_freq_bins,)

    def test_positive_everywhere(self, nv_cfg: NVConfig, spec_cfg: SpectralConfig) -> None:
        _, p = build_initial_inversion(nv_cfg, spec_cfg)
        assert np.all(p >= 0)

    def test_wings_smaller_than_centre(self, nv_cfg: NVConfig, spec_cfg: SpectralConfig) -> None:
        delta, p = build_initial_inversion(nv_cfg, spec_cfg)
        mid = len(delta) // 2
        assert p[0] < p[mid]
        assert p[-1] < p[mid]


# ── On-resonance inversion ──────────────────────────────────────


class TestOnResonanceInversion:
    def test_uniform_profile(self) -> None:
        delta = np.linspace(-1e6, 1e6, 101)
        p = np.full(101, 0.25)
        result = compute_on_resonance_inversion(p, delta, cavity_linewidth_hz=1e5)
        assert result == pytest.approx(0.25)

    def test_peaked_profile_higher(self) -> None:
        delta = np.linspace(-1e7, 1e7, 1001)
        sigma = 1e6
        p = 0.25 * np.exp(-0.5 * (delta / sigma) ** 2)
        result = compute_on_resonance_inversion(p, delta, cavity_linewidth_hz=1e5)
        # Should be close to peak (0.25) since cavity BW << sigma
        assert result > 0.2


# ── Spectral hole burning ───────────────────────────────────────


class TestBurnSpectralHole:
    def test_no_burn_unchanged(self) -> None:
        delta = np.linspace(-1e6, 1e6, 101)
        p = np.full(101, 0.25)
        p_new = burn_spectral_hole(p, delta, 1e5, burn_depth=0.0)
        np.testing.assert_array_equal(p_new, p)

    def test_full_burn_zero_at_centre(self) -> None:
        delta = np.linspace(-1e6, 1e6, 101)
        p = np.full(101, 0.25)
        p_new = burn_spectral_hole(p, delta, 1e5, burn_depth=1.0)
        mid = len(delta) // 2
        assert p_new[mid] == pytest.approx(0.0)

    def test_partial_burn_reduces_centre(self) -> None:
        delta = np.linspace(-1e6, 1e6, 101)
        p = np.full(101, 0.25)
        p_new = burn_spectral_hole(p, delta, 1e5, burn_depth=0.5)
        mid = len(delta) // 2
        assert 0.0 < p_new[mid] < 0.25

    def test_wings_less_affected(self) -> None:
        delta = np.linspace(-1e7, 1e7, 1001)
        p = np.full(1001, 0.25)
        p_new = burn_spectral_hole(p, delta, 1e5, burn_depth=0.8)
        # Wings (far from centre) should be nearly unchanged
        assert p_new[0] / p[0] > 0.99

    def test_does_not_modify_input(self) -> None:
        delta = np.linspace(-1e6, 1e6, 101)
        p = np.full(101, 0.25)
        p_orig = p.copy()
        burn_spectral_hole(p, delta, 1e5, burn_depth=0.5)
        np.testing.assert_array_equal(p, p_orig)


# ── Spectral overlap weights ────────────────────────────────────


class TestSpectralOverlapWeights:
    def test_peak_is_one(self) -> None:
        delta = np.linspace(-1e6, 1e6, 101)
        w = spectral_overlap_weights(delta, 1e5)
        mid = len(delta) // 2
        assert w[mid] == pytest.approx(1.0)

    def test_decays_with_detuning(self) -> None:
        delta = np.linspace(-1e6, 1e6, 101)
        w = spectral_overlap_weights(delta, 1e5)
        mid = len(delta) // 2
        assert w[0] < w[mid]
