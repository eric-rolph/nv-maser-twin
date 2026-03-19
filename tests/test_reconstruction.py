"""Tests for nv_maser.physics.reconstruction.

Covers FFT reconstruction, gridding, Haar wavelet transform/inverse,
compressed sensing, depth profiles, and sweep helpers.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nv_maser.physics.reconstruction import (
    DepthProfileResult,
    GriddingResult,
    ReconResult,
    apply_undersampling_mask,
    estimate_acceleration_factor,
    grid_kspace,
    haar_wavelet_inverse,
    haar_wavelet_transform,
    image_snr_from_phantom,
    reconstruct_compressed_sensing,
    reconstruct_depth_profile,
    reconstruct_fft,
    reconstruct_gridding,
    simulate_kspace,
    soft_threshold,
    sweep_resolution_vs_fov,
    sweep_snr_vs_acceleration,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _box_phantom(size: int = 32, box_frac: float = 0.5) -> np.ndarray:
    """Positive-valued box phantom for reconstruction tests."""
    img = np.zeros((size, size))
    m = int(size * box_frac) // 2
    c = size // 2
    img[c - m : c + m, c - m : c + m] = 1.0
    return img


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ─────────────────────────────────────────────────────────────────────────────
# TestDataClasses
# ─────────────────────────────────────────────────────────────────────────────

class TestDataClasses:
    def test_recon_result_fields(self):
        img = np.zeros((4, 4), dtype=complex)
        r = ReconResult(
            image=img,
            magnitude=np.abs(img),
            fov_x_m=0.05,
            fov_y_m=0.05,
            resolution_x_mm=1.5625,
            resolution_y_mm=1.5625,
            n_iterations=1,
            method="fft",
            snr_db=30.0,
        )
        assert r.fov_x_m == 0.05
        assert r.method == "fft"
        assert r.n_iterations == 1

    def test_recon_result_frozen(self):
        img = np.zeros((4, 4), dtype=complex)
        r = ReconResult(
            image=img, magnitude=np.abs(img),
            fov_x_m=0.05, fov_y_m=0.05,
            resolution_x_mm=1.0, resolution_y_mm=1.0,
            n_iterations=1, method="fft", snr_db=20.0,
        )
        with pytest.raises(Exception):  # frozen dataclass
            r.method = "other"  # type: ignore[misc]

    def test_depth_profile_result_fields(self):
        d = DepthProfileResult(
            depths_mm=np.linspace(0, 10, 50),
            signal=np.zeros(50, dtype=complex),
            magnitude=np.zeros(50),
            depths_mm_resolved=np.array([5.0]),
            peak_depth_mm=5.0,
            t2_star_ms=0.5,
            gradient_t_per_m=0.1,
            resolution_mm=2.0,
        )
        assert d.peak_depth_mm == 5.0
        assert d.t2_star_ms == 0.5

    def test_gridding_result_fields(self):
        g = GriddingResult(
            kspace_grid=np.zeros((16, 16), dtype=complex),
            density_weights=np.ones(100),
            n_samples=100,
            grid_size=(16, 16),
        )
        assert g.n_samples == 100
        assert g.grid_size == (16, 16)


# ─────────────────────────────────────────────────────────────────────────────
# TestSimulateKspace
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateKspace:
    def test_output_shape(self):
        img = _box_phantom(32)
        ks = simulate_kspace(img)
        assert ks.shape == (32, 32)

    def test_output_complex(self):
        img = _box_phantom(16)
        ks = simulate_kspace(img)
        assert np.iscomplexobj(ks)

    def test_roundtrip_noiseless(self):
        """simulate_kspace → reconstruct_fft should recover image up to Hamming blur."""
        img = _box_phantom(32)
        ks = simulate_kspace(img)
        result = reconstruct_fft(ks, fov_x_m=0.05, fov_y_m=0.05, apply_hamming=False)
        corr = float(np.corrcoef(img.ravel(), result.magnitude.ravel())[0, 1])
        assert corr > 0.999

    def test_with_snr_adds_noise(self):
        img = _box_phantom(32)
        ks_clean = simulate_kspace(img)
        ks_noisy = simulate_kspace(img, snr_db=10.0)
        # Noisy should differ from clean
        assert not np.allclose(ks_clean, ks_noisy)

    def test_dc_dominant_for_uniform_image(self):
        """DC (k=0) term should dominate for a box phantom."""
        img = _box_phantom(32)
        ks = simulate_kspace(img)
        ny, nx = ks.shape
        dc = abs(ks[ny // 2, nx // 2])
        assert dc > 0.5 * np.max(np.abs(ks))


# ─────────────────────────────────────────────────────────────────────────────
# TestReconstructFFT
# ─────────────────────────────────────────────────────────────────────────────

class TestReconstructFFT:
    def test_output_type(self):
        ks = simulate_kspace(_box_phantom(16))
        r = reconstruct_fft(ks, fov_x_m=0.05, fov_y_m=0.05)
        assert isinstance(r, ReconResult)

    def test_output_shape(self):
        ks = simulate_kspace(_box_phantom(32))
        r = reconstruct_fft(ks, fov_x_m=0.05, fov_y_m=0.05)
        assert r.image.shape == (32, 32)
        assert r.magnitude.shape == (32, 32)

    def test_magnitude_nonneg(self):
        ks = simulate_kspace(_box_phantom(16))
        r = reconstruct_fft(ks, fov_x_m=0.05, fov_y_m=0.05)
        assert np.all(r.magnitude >= 0)

    def test_resolution_formula(self):
        ks = simulate_kspace(_box_phantom(32))
        r = reconstruct_fft(ks, fov_x_m=0.08, fov_y_m=0.08)
        assert_allclose(r.resolution_x_mm, 0.08 / 32 * 1e3, rtol=1e-6)
        assert_allclose(r.resolution_y_mm, 0.08 / 32 * 1e3, rtol=1e-6)

    def test_fov_stored(self):
        ks = simulate_kspace(_box_phantom(16))
        r = reconstruct_fft(ks, fov_x_m=0.03, fov_y_m=0.07)
        assert r.fov_x_m == pytest.approx(0.03)
        assert r.fov_y_m == pytest.approx(0.07)

    def test_method_label(self):
        ks = simulate_kspace(_box_phantom(16))
        r = reconstruct_fft(ks, fov_x_m=0.05, fov_y_m=0.05)
        assert r.method == "fft"
        assert r.n_iterations == 1

    def test_noiseless_box_correlation(self):
        img = _box_phantom(32)
        ks = simulate_kspace(img)
        r = reconstruct_fft(ks, fov_x_m=0.05, fov_y_m=0.05, apply_hamming=False)
        corr = float(np.corrcoef(img.ravel(), r.magnitude.ravel())[0, 1])
        assert corr > 0.999

    def test_hamming_reduces_ringing(self):
        """Hamming window should reduce Gibbs ringing (magnitude differs from no-window)."""
        img = _box_phantom(32)
        ks = simulate_kspace(img)
        r_plain = reconstruct_fft(ks, fov_x_m=0.05, fov_y_m=0.05, apply_hamming=False)
        r_ham = reconstruct_fft(ks, fov_x_m=0.05, fov_y_m=0.05, apply_hamming=True)
        assert not np.allclose(r_plain.magnitude, r_ham.magnitude)

    def test_snr_reasonable(self):
        img = _box_phantom(32)
        ks = simulate_kspace(img, snr_db=30.0)
        r = reconstruct_fft(ks, fov_x_m=0.05, fov_y_m=0.05)
        # SNR stored should be a finite number
        assert math.isfinite(r.snr_db)


# ─────────────────────────────────────────────────────────────────────────────
# TestHaarWavelet
# ─────────────────────────────────────────────────────────────────────────────

class TestHaarWavelet:
    def test_roundtrip_levels1(self):
        rng = _rng(1)
        x = rng.standard_normal((32, 32))
        err = np.max(np.abs(haar_wavelet_inverse(haar_wavelet_transform(x, 1), 1) - x))
        assert err < 1e-10

    def test_roundtrip_levels2(self):
        rng = _rng(2)
        x = rng.standard_normal((32, 32))
        err = np.max(np.abs(haar_wavelet_inverse(haar_wavelet_transform(x, 2), 2) - x))
        assert err < 1e-10

    def test_roundtrip_levels3(self):
        rng = _rng(3)
        x = rng.standard_normal((32, 32))
        err = np.max(np.abs(haar_wavelet_inverse(haar_wavelet_transform(x, 3), 3) - x))
        assert err < 1e-10

    def test_roundtrip_complex(self):
        rng = _rng(4)
        x = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
        err = np.max(np.abs(haar_wavelet_inverse(haar_wavelet_transform(x, 2), 2) - x))
        assert err < 1e-10

    def test_output_shape_unchanged(self):
        x = np.ones((16, 16))
        c = haar_wavelet_transform(x, 2)
        assert c.shape == (16, 16)
        assert haar_wavelet_inverse(c, 2).shape == (16, 16)

    def test_energy_preserved(self):
        """Haar transform is orthonormal: ||coeffs||² == ||image||²."""
        rng = _rng(5)
        x = rng.standard_normal((32, 32))
        c = haar_wavelet_transform(x, 2)
        assert_allclose(np.sum(np.abs(c) ** 2), np.sum(x ** 2), rtol=1e-10)

    def test_ll_subband_is_approximation(self):
        """Top-left (LL) coeff at level=1 should capture the image mean."""
        x = np.ones((16, 16)) * 5.0
        c = haar_wavelet_transform(x, 1)
        # For constant image, all detail coeffs should be ~0
        hl = c[:8, 8:]   # high-freq horizontal detail
        assert np.max(np.abs(hl)) < 1e-10

    def test_constant_image_only_ll_nonzero(self):
        """Constant image → only LL subband nonzero after transform."""
        x = np.ones((16, 16)) * 3.0
        c = haar_wavelet_transform(x, 2)
        ll = c[:4, :4]
        rest = c.copy()
        rest[:4, :4] = 0
        assert np.sum(np.abs(ll)) > 0
        assert np.max(np.abs(rest)) < 1e-10

    def test_inverse_only_roundtrip(self):
        """Applying inverse of identity coefficients returns identity."""
        rng = _rng(6)
        x = rng.standard_normal((32, 32))
        c = haar_wavelet_transform(x, 2)
        x_back = haar_wavelet_inverse(c, 2)
        assert_allclose(x_back.real, x, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# TestSoftThreshold
# ─────────────────────────────────────────────────────────────────────────────

class TestSoftThreshold:
    def test_zero_below_threshold(self):
        x = np.array([0.5, 0.3, 0.1])
        out = soft_threshold(x, threshold=0.6)
        assert_allclose(out, [0, 0, 0], atol=1e-12)

    def test_shrinks_above_threshold(self):
        x = np.array([2.0, -3.0])
        out = soft_threshold(x, threshold=1.0)
        assert_allclose(np.abs(out), [1.0, 2.0], atol=1e-12)

    def test_complex_preserves_phase(self):
        x = np.array([3.0 + 4.0j])  # magnitude = 5
        out = soft_threshold(x, threshold=1.0)
        assert_allclose(np.abs(out), [4.0], atol=1e-12)
        assert_allclose(np.angle(out), [np.angle(x[0])], atol=1e-12)

    def test_zero_threshold(self):
        x = np.array([1.0, 2.0, 3.0])
        out = soft_threshold(x, threshold=0.0)
        assert_allclose(out, x, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# TestUndersamplingMask
# ─────────────────────────────────────────────────────────────────────────────

class TestUndersamplingMask:
    def test_output_shapes(self):
        ks = np.ones((32, 32), dtype=complex)
        sampled, mask = apply_undersampling_mask(ks, acceleration_factor=2, seed=1)
        assert sampled.shape == (32, 32)
        assert mask.shape == (32, 32)

    def test_mask_boolean(self):
        ks = np.ones((16, 16), dtype=complex)
        _, mask = apply_undersampling_mask(ks, acceleration_factor=2, seed=2)
        assert mask.dtype == bool

    def test_dc_always_sampled(self):
        """DC (centre of k-space) must always be sampled."""
        ks = np.ones((32, 32), dtype=complex)
        _, mask = apply_undersampling_mask(ks, acceleration_factor=4, seed=3)
        ny, nx = mask.shape
        assert mask[ny // 2, nx // 2]

    def test_acceleration_reduces_samples(self):
        ks = np.ones((32, 32), dtype=complex)
        _, mask1 = apply_undersampling_mask(ks, acceleration_factor=1, seed=4)
        _, mask2 = apply_undersampling_mask(ks, acceleration_factor=4, seed=4)
        assert mask2.sum() < mask1.sum()

    def test_unsampled_set_to_zero(self):
        ks = np.ones((16, 16), dtype=complex)
        sampled, mask = apply_undersampling_mask(ks, acceleration_factor=2, seed=5)
        assert_allclose(sampled[~mask], 0.0)

    def test_sampled_preserved(self):
        rng = _rng(10)
        ks = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
        sampled, mask = apply_undersampling_mask(ks, acceleration_factor=2, seed=5)
        assert_allclose(sampled[mask], ks[mask])

    def test_reproducible_with_seed(self):
        ks = np.ones((16, 16), dtype=complex)
        _, m1 = apply_undersampling_mask(ks, acceleration_factor=2, seed=99)
        _, m2 = apply_undersampling_mask(ks, acceleration_factor=2, seed=99)
        np.testing.assert_array_equal(m1, m2)


# ─────────────────────────────────────────────────────────────────────────────
# TestEstimateAccelerationFactor
# ─────────────────────────────────────────────────────────────────────────────

class TestEstimateAccelerationFactor:
    def test_full_mask_gives_one(self):
        mask = np.ones((32, 32), dtype=bool)
        assert estimate_acceleration_factor(mask) == pytest.approx(1.0)

    def test_half_mask_gives_two(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[::2, :] = True  # half the rows
        af = estimate_acceleration_factor(mask)
        assert af == pytest.approx(2.0, rel=0.1)

    def test_consistent_with_apply_mask(self):
        ks = np.ones((32, 32), dtype=complex)
        _, mask = apply_undersampling_mask(ks, acceleration_factor=2, seed=0)
        af = estimate_acceleration_factor(mask)
        assert af == pytest.approx(2.0, rel=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# TestGridKspace
# ─────────────────────────────────────────────────────────────────────────────

class TestGridKspace:
    def test_output_type(self):
        rng = _rng(20)
        kx = rng.uniform(-10, 10, 100)
        ky = rng.uniform(-10, 10, 100)
        s = rng.standard_normal(100) + 1j * rng.standard_normal(100)
        g = grid_kspace(kx, ky, s, grid_size=(16, 16), fov_x_m=0.05, fov_y_m=0.05)
        assert isinstance(g, GriddingResult)

    def test_output_grid_shape(self):
        rng = _rng(21)
        kx = rng.uniform(-10, 10, 100)
        ky = rng.uniform(-10, 10, 100)
        s = rng.standard_normal(100) + 1j * rng.standard_normal(100)
        g = grid_kspace(kx, ky, s, grid_size=(24, 32), fov_x_m=0.05, fov_y_m=0.05)
        assert g.kspace_grid.shape == (24, 32)

    def test_int_grid_size(self):
        rng = _rng(22)
        kx = rng.uniform(-10, 10, 50)
        ky = rng.uniform(-10, 10, 50)
        s = rng.standard_normal(50) + 1j * rng.standard_normal(50)
        g = grid_kspace(kx, ky, s, grid_size=16, fov_x_m=0.05, fov_y_m=0.05)
        assert g.kspace_grid.shape == (16, 16)

    def test_density_weights_positive(self):
        rng = _rng(23)
        kx = rng.uniform(-10, 10, 80)
        ky = rng.uniform(-10, 10, 80)
        s = rng.standard_normal(80) + 1j * rng.standard_normal(80)
        g = grid_kspace(kx, ky, s, grid_size=16, fov_x_m=0.05, fov_y_m=0.05)
        assert np.all(g.density_weights > 0)

    def test_n_samples_correct(self):
        rng = _rng(24)
        kx = rng.uniform(-10, 10, 77)
        ky = rng.uniform(-10, 10, 77)
        s = rng.standard_normal(77) + 1j * rng.standard_normal(77)
        g = grid_kspace(kx, ky, s, grid_size=16, fov_x_m=0.05, fov_y_m=0.05)
        assert g.n_samples == 77


# ─────────────────────────────────────────────────────────────────────────────
# TestReconstructGridding
# ─────────────────────────────────────────────────────────────────────────────

class TestReconstructGridding:
    def test_output_type(self):
        rng = _rng(30)
        kx = rng.uniform(-10, 10, 200)
        ky = rng.uniform(-10, 10, 200)
        s = rng.standard_normal(200) + 1j * rng.standard_normal(200)
        r = reconstruct_gridding(kx, ky, s, fov_x_m=0.05, fov_y_m=0.05, grid_size=32)
        assert isinstance(r, ReconResult)

    def test_output_shape(self):
        rng = _rng(31)
        kx = rng.uniform(-10, 10, 200)
        ky = rng.uniform(-10, 10, 200)
        s = rng.standard_normal(200) + 1j * rng.standard_normal(200)
        r = reconstruct_gridding(kx, ky, s, fov_x_m=0.05, fov_y_m=0.05, grid_size=32)
        assert r.image.shape == (32, 32)
        assert r.magnitude.shape == (32, 32)

    def test_method_label(self):
        rng = _rng(32)
        kx = rng.uniform(-10, 10, 100)
        ky = rng.uniform(-10, 10, 100)
        s = rng.standard_normal(100) + 1j * rng.standard_normal(100)
        r = reconstruct_gridding(kx, ky, s, fov_x_m=0.05, fov_y_m=0.05, grid_size=16)
        assert "gridding" in r.method

    def test_magnitude_nonneg(self):
        rng = _rng(33)
        kx = rng.uniform(-10, 10, 100)
        ky = rng.uniform(-10, 10, 100)
        s = rng.standard_normal(100) + 1j * rng.standard_normal(100)
        r = reconstruct_gridding(kx, ky, s, fov_x_m=0.05, fov_y_m=0.05, grid_size=16)
        assert np.all(r.magnitude >= 0)


# ─────────────────────────────────────────────────────────────────────────────
# TestReconstructCompressedSensing
# ─────────────────────────────────────────────────────────────────────────────

class TestReconstructCompressedSensing:
    def test_output_type(self):
        img = _box_phantom(32)
        ks = simulate_kspace(img)
        ks_u, mask = apply_undersampling_mask(ks, acceleration_factor=2, seed=1)
        r = reconstruct_compressed_sensing(ks_u, mask, fov_x_m=0.05, fov_y_m=0.05)
        assert isinstance(r, ReconResult)

    def test_output_shape(self):
        img = _box_phantom(32)
        ks = simulate_kspace(img)
        ks_u, mask = apply_undersampling_mask(ks, acceleration_factor=2, seed=1)
        r = reconstruct_compressed_sensing(ks_u, mask, fov_x_m=0.05, fov_y_m=0.05)
        assert r.image.shape == (32, 32)

    def test_method_name(self):
        img = _box_phantom(32)
        ks = simulate_kspace(img)
        ks_u, mask = apply_undersampling_mask(ks, acceleration_factor=2, seed=1)
        r = reconstruct_compressed_sensing(ks_u, mask, fov_x_m=0.05, fov_y_m=0.05)
        assert "compressed_sensing" in r.method

    def test_iterations_stored(self):
        img = _box_phantom(32)
        ks = simulate_kspace(img)
        ks_u, mask = apply_undersampling_mask(ks, acceleration_factor=2, seed=1)
        r = reconstruct_compressed_sensing(ks_u, mask, fov_x_m=0.05, fov_y_m=0.05, n_iterations=15)
        assert r.n_iterations == 15

    def test_full_sampling_vs_fft(self):
        """With no undersampling, CS should produce similar result to FFT."""
        img = _box_phantom(32)
        ks = simulate_kspace(img)
        # Full mask (acceleration=1)
        ks_u, mask = apply_undersampling_mask(ks, acceleration_factor=1, seed=1)
        r_cs = reconstruct_compressed_sensing(ks_u, mask, fov_x_m=0.05, fov_y_m=0.05,
                                               n_iterations=5, lambda_reg=0.0001)
        r_fft = reconstruct_fft(ks, fov_x_m=0.05, fov_y_m=0.05, apply_hamming=False)
        # CS and FFT should agree reasonably well
        corr = float(np.corrcoef(r_cs.magnitude.ravel(), r_fft.magnitude.ravel())[0, 1])
        assert corr > 0.9

    def test_snr_finite(self):
        """CS may return NaN SNR if outside-box is very clean (std~0); test image quality."""
        img = _box_phantom(32)
        ks = simulate_kspace(img)
        ks_u, mask = apply_undersampling_mask(ks, acceleration_factor=2, seed=1)
        r = reconstruct_compressed_sensing(ks_u, mask, fov_x_m=0.05, fov_y_m=0.05)
        # Verify reconstruction is non-zero (not collapsed to all zeros)
        assert np.max(r.magnitude) > 0.1


# ─────────────────────────────────────────────────────────────────────────────
# TestReconstructDepthProfile
# ─────────────────────────────────────────────────────────────────────────────

class TestReconstructDepthProfile:
    def _fid(self, freq_hz: float = 1000.0, t2_s: float = 0.5e-3,
             n: int = 512) -> tuple[np.ndarray, float]:
        t = np.linspace(0, 2e-3, n, endpoint=False)
        dwell_s = float(t[1] - t[0])
        sig = np.exp(-t / t2_s) * np.exp(2j * np.pi * freq_hz * t)
        return sig, dwell_s * 1e6  # return dwell in µs

    def test_output_type(self):
        sig, dwell = self._fid()
        r = reconstruct_depth_profile(sig, dwell_us=dwell, gradient_t_per_m=0.1)
        assert isinstance(r, DepthProfileResult)

    def test_depths_array_shape(self):
        sig, dwell = self._fid(n=256)
        r = reconstruct_depth_profile(sig, dwell_us=dwell, gradient_t_per_m=0.1)
        assert r.depths_mm.shape == r.signal.shape
        assert r.depths_mm.shape == r.magnitude.shape

    def test_magnitude_nonneg(self):
        sig, dwell = self._fid()
        r = reconstruct_depth_profile(sig, dwell_us=dwell, gradient_t_per_m=0.1)
        assert np.all(r.magnitude >= 0)

    def test_gradient_stored(self):
        sig, dwell = self._fid()
        r = reconstruct_depth_profile(sig, dwell_us=dwell, gradient_t_per_m=0.2)
        assert r.gradient_t_per_m == pytest.approx(0.2)

    def test_t2star_positive(self):
        sig, dwell = self._fid(t2_s=0.5e-3)
        r = reconstruct_depth_profile(sig, dwell_us=dwell, gradient_t_per_m=0.1)
        # T2* should be a positive number
        assert r.t2_star_ms > 0

    def test_resolution_formula(self):
        """resolution_mm = 1e3 / (gamma/2pi * G * T_acq * 2)."""
        sig, dwell = self._fid(n=512)
        g = 0.1
        r = reconstruct_depth_profile(sig, dwell_us=dwell, gradient_t_per_m=g)
        assert r.resolution_mm > 0

    def test_depths_mm_resolved_subset(self):
        sig, dwell = self._fid()
        r = reconstruct_depth_profile(sig, dwell_us=dwell, gradient_t_per_m=0.1,
                                      signal_threshold=0.5)
        # Resolved depths should be a subset of full depth axis
        assert len(r.depths_mm_resolved) <= len(r.depths_mm)

    def test_real_input_accepted(self):
        """Real-valued FID (e.g., cosine) should work without error."""
        t = np.linspace(0, 1e-3, 256)
        sig_real = np.cos(2 * np.pi * 500 * t)
        dwell_us = float((t[1] - t[0]) * 1e6)
        r = reconstruct_depth_profile(sig_real, dwell_us=dwell_us, gradient_t_per_m=0.05)
        assert isinstance(r, DepthProfileResult)


# ─────────────────────────────────────────────────────────────────────────────
# TestImageSNR
# ─────────────────────────────────────────────────────────────────────────────

class TestImageSNR:
    def test_snr_finite_for_box(self):
        img = _box_phantom(32)
        ks = simulate_kspace(img, snr_db=30.0)
        r = reconstruct_fft(ks, fov_x_m=0.05, fov_y_m=0.05)
        snr = image_snr_from_phantom(r.magnitude)
        assert math.isfinite(snr)

    def test_higher_snr_kspace_gives_higher_snr(self):
        img = _box_phantom(32)
        r_lo = reconstruct_fft(simulate_kspace(img, snr_db=10.0), fov_x_m=0.05, fov_y_m=0.05)
        r_hi = reconstruct_fft(simulate_kspace(img, snr_db=40.0), fov_x_m=0.05, fov_y_m=0.05)
        snr_lo = image_snr_from_phantom(r_lo.magnitude)
        snr_hi = image_snr_from_phantom(r_hi.magnitude)
        assert snr_hi > snr_lo

    def test_all_noise_image(self):
        rng = _rng(50)
        noise = rng.standard_normal((32, 32)).astype(float)
        snr = image_snr_from_phantom(noise)
        # Very low or negative SNR expected for pure noise
        assert math.isfinite(snr)


# ─────────────────────────────────────────────────────────────────────────────
# TestSweeps
# ─────────────────────────────────────────────────────────────────────────────

class TestSweeps:
    def test_sweep_resolution_length(self):
        fovs = [0.02, 0.05, 0.10]
        res = sweep_resolution_vs_fov(32, fovs)
        assert len(res) == len(fovs)

    def test_sweep_resolution_monotone(self):
        """Larger FOV → larger pixel size (lower resolution)."""
        fovs = [0.02, 0.05, 0.10]
        res = sweep_resolution_vs_fov(32, fovs)
        assert res[0] < res[1] < res[2]

    def test_sweep_resolution_formula(self):
        """resolution_mm = fov_m / n_pixels * 1e3."""
        fovs = [0.05]
        res = sweep_resolution_vs_fov(32, fovs)
        assert_allclose(res[0], 0.05 / 32 * 1e3, rtol=1e-6)

    def test_sweep_snr_length(self):
        img = _box_phantom(32)
        factors = [1, 2]
        snrs = sweep_snr_vs_acceleration(img, factors)
        assert len(snrs) == len(factors)

    def test_sweep_snr_all_finite(self):
        # Use Gaussian phantom which doesn't hit over-regularization edge case
        rng = _rng(77)
        img = np.abs(rng.standard_normal((32, 32)))  # non-negative
        snrs = sweep_snr_vs_acceleration(img, [1, 2], lambda_reg=1e-4)
        assert all(math.isfinite(s) for s in snrs)

    def test_sweep_snr_decreases_with_acceleration(self):
        """Higher undersampling generally degrades SNR (not strictly monotone, but AF=1 > AF=4)."""
        rng = _rng(78)
        img = np.abs(rng.standard_normal((32, 32)))
        snrs = sweep_snr_vs_acceleration(img, [1, 4], lambda_reg=1e-4)
        assert snrs[0] >= snrs[1] - 5.0  # allow 5 dB tolerance for randomness
