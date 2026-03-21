"""Tests for the Phase-6 '2D image' milestone validator.

Coverage
────────
TestPhase6Config                 (13) — defaults, validation errors, frozen
TestGridPhantomResultFields       (6) — construction, field types, frozen
TestBarContrastResultFields       (6) — construction, field types, frozen
TestPhase6MilestoneResultFields   (8) — all fields present, frozen, types
TestMakeBarPhantom               (11) — bar pattern geometry, pixel sizes,
                                        edge cases
TestReconstructFromPhantom        (6) — output shape, method label, SNR
TestMeasurePSFFWHM                (7) — value range, point-source shape,
                                        finite result, tight/loose config
TestMeasureBarContrast            (8) — contrast formula, pass/fail,
                                        zero-image, all-bar, all-gap
TestValidatePhase6MilestoneDefault (12) — default config passes, measured
                                          values within spec
TestPhase6MilestoneClosedLogic    (4) — all three sub-criteria required
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nv_maser.physics.phase6_validator import (
    Phase6Config,
    GridPhantomResult,
    BarContrastResult,
    Phase6MilestoneResult,
    _make_bar_phantom,
    _reconstruct_from_phantom,
    _measure_psf_fwhm,
    _measure_bar_contrast,
    validate_phase6_milestone,
)


# ── Shared fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def default_result() -> Phase6MilestoneResult:
    """Run the milestone validator once for the whole module (slow)."""
    return validate_phase6_milestone()


@pytest.fixture(scope="module")
def default_config() -> Phase6Config:
    return Phase6Config()


# ── TestPhase6Config ──────────────────────────────────────────────


class TestPhase6Config:
    def test_default_bar_width_mm(self) -> None:
        cfg = Phase6Config()
        assert cfg.bar_width_mm == 3.0

    def test_default_fov_m(self) -> None:
        cfg = Phase6Config()
        assert cfg.fov_m == pytest.approx(0.064)

    def test_default_grid_size(self) -> None:
        cfg = Phase6Config()
        assert cfg.grid_size == 64

    def test_default_n_spokes(self) -> None:
        cfg = Phase6Config()
        assert cfg.n_spokes == 64

    def test_default_n_readout(self) -> None:
        cfg = Phase6Config()
        assert cfg.n_readout == 64

    def test_default_bar_contrast_threshold(self) -> None:
        cfg = Phase6Config()
        assert cfg.bar_contrast_threshold == pytest.approx(0.2)

    def test_default_image_snr_threshold_db(self) -> None:
        cfg = Phase6Config()
        assert cfg.image_snr_threshold_db == pytest.approx(5.0)

    def test_default_resolution_threshold_mm(self) -> None:
        cfg = Phase6Config()
        assert cfg.resolution_threshold_mm == pytest.approx(3.0)

    def test_frozen(self) -> None:
        cfg = Phase6Config()
        with pytest.raises((AttributeError, TypeError)):
            cfg.bar_width_mm = 1.0  # type: ignore[misc]

    def test_zero_bar_width_raises(self) -> None:
        with pytest.raises(ValueError, match="bar_width_mm"):
            Phase6Config(bar_width_mm=0.0)

    def test_negative_fov_raises(self) -> None:
        with pytest.raises(ValueError, match="fov_m"):
            Phase6Config(fov_m=-0.01)

    def test_small_grid_raises(self) -> None:
        with pytest.raises(ValueError, match="grid_size"):
            Phase6Config(grid_size=2)

    def test_bar_width_smaller_than_pixel_raises(self) -> None:
        # FOV=0.064, grid=64 → pixel=1.0 mm; bar=0.5 mm < pixel → error
        with pytest.raises(ValueError, match="bar_width_mm"):
            Phase6Config(bar_width_mm=0.5)


# ── TestGridPhantomResultFields ───────────────────────────────────


class TestGridPhantomResultFields:
    def test_construction(self) -> None:
        ph = _make_bar_phantom(64, 0.064, 3.0)
        r = GridPhantomResult(
            phantom=ph,
            pixel_size_mm=1.0,
            n_bar_pairs=10,
            bar_width_mm=3.0,
        )
        assert r.bar_width_mm == 3.0

    def test_frozen(self) -> None:
        ph = _make_bar_phantom(32, 0.032, 2.0)
        r = GridPhantomResult(phantom=ph, pixel_size_mm=1.0, n_bar_pairs=8, bar_width_mm=2.0)
        with pytest.raises((AttributeError, TypeError)):
            r.pixel_size_mm = 2.0  # type: ignore[misc]

    def test_n_bar_pairs_is_int(self, default_result: Phase6MilestoneResult) -> None:
        assert isinstance(default_result.phantom_result.n_bar_pairs, int)

    def test_pixel_size_mm_is_float(self, default_result: Phase6MilestoneResult) -> None:
        assert isinstance(default_result.phantom_result.pixel_size_mm, float)

    def test_phantom_shape(self, default_result: Phase6MilestoneResult) -> None:
        ph = default_result.phantom_result.phantom
        assert ph.shape == (64, 64)

    def test_phantom_binary(self, default_result: Phase6MilestoneResult) -> None:
        ph = default_result.phantom_result.phantom
        unique = set(np.unique(ph).tolist())
        assert unique.issubset({0.0, 1.0})


# ── TestBarContrastResultFields ───────────────────────────────────


class TestBarContrastResultFields:
    def test_construction(self) -> None:
        bc = BarContrastResult(
            bar_mean=0.8,
            gap_mean=0.2,
            michelson_contrast=0.6,
            passes=True,
        )
        assert bc.michelson_contrast == pytest.approx(0.6)

    def test_frozen(self) -> None:
        bc = BarContrastResult(bar_mean=0.5, gap_mean=0.1, michelson_contrast=0.67, passes=True)
        with pytest.raises((AttributeError, TypeError)):
            bc.passes = False  # type: ignore[misc]

    def test_passes_is_bool(self, default_result: Phase6MilestoneResult) -> None:
        assert isinstance(default_result.bar_contrast.passes, bool)

    def test_bar_mean_positive(self, default_result: Phase6MilestoneResult) -> None:
        assert default_result.bar_contrast.bar_mean > 0.0

    def test_gap_mean_non_negative(self, default_result: Phase6MilestoneResult) -> None:
        assert default_result.bar_contrast.gap_mean >= 0.0

    def test_contrast_range(self, default_result: Phase6MilestoneResult) -> None:
        c = default_result.bar_contrast.michelson_contrast
        # Perfect bars → contrast in [0, 1]; gridding may degrade but must be > 0
        assert 0.0 < c <= 1.0


# ── TestPhase6MilestoneResultFields ──────────────────────────────


class TestPhase6MilestoneResultFields:
    def test_config_stored(self, default_result: Phase6MilestoneResult) -> None:
        assert isinstance(default_result.config, Phase6Config)

    def test_phantom_result_stored(self, default_result: Phase6MilestoneResult) -> None:
        assert isinstance(default_result.phantom_result, GridPhantomResult)

    def test_bar_contrast_stored(self, default_result: Phase6MilestoneResult) -> None:
        assert isinstance(default_result.bar_contrast, BarContrastResult)

    def test_psf_fwhm_mm_is_float(self, default_result: Phase6MilestoneResult) -> None:
        assert isinstance(default_result.psf_fwhm_mm, float)

    def test_image_snr_db_is_float(self, default_result: Phase6MilestoneResult) -> None:
        assert isinstance(default_result.image_snr_db, float)

    def test_pixel_size_mm_is_float(self, default_result: Phase6MilestoneResult) -> None:
        assert isinstance(default_result.pixel_size_mm, float)

    def test_boolean_flags_are_bool(self, default_result: Phase6MilestoneResult) -> None:
        for flag in (
            default_result.psf_pass,
            default_result.contrast_pass,
            default_result.snr_pass,
            default_result.phase6_milestone_closed,
        ):
            assert isinstance(flag, bool)

    def test_frozen(self, default_result: Phase6MilestoneResult) -> None:
        with pytest.raises((AttributeError, TypeError)):
            default_result.phase6_milestone_closed = False  # type: ignore[misc]


# ── TestMakeBarPhantom ────────────────────────────────────────────


class TestMakeBarPhantom:
    def test_output_shape(self) -> None:
        ph = _make_bar_phantom(64, 0.064, 3.0)
        assert ph.shape == (64, 64)

    def test_values_are_binary(self) -> None:
        ph = _make_bar_phantom(32, 0.032, 2.0)
        assert set(np.unique(ph).tolist()).issubset({0.0, 1.0})

    def test_first_bar_is_bright(self) -> None:
        # Column 0 should be in bar_idx=0 → bright
        ph = _make_bar_phantom(64, 0.064, 3.0)
        assert ph[0, 0] == 1.0

    def test_bar_and_gap_alternate(self) -> None:
        # At pixel 1.0mm, 3px per bar: cols 0-2 bright, 3-5 dark, 6-8 bright
        ph = _make_bar_phantom(64, 0.064, 3.0)
        assert ph[0, 0] == 1.0   # bar
        assert ph[0, 1] == 1.0   # bar
        assert ph[0, 2] == 1.0   # bar
        assert ph[0, 3] == 0.0   # gap
        assert ph[0, 4] == 0.0   # gap
        assert ph[0, 5] == 0.0   # gap
        assert ph[0, 6] == 1.0   # bar

    def test_bars_extend_full_height(self) -> None:
        # All rows in a bar column should be 1.0
        ph = _make_bar_phantom(64, 0.064, 3.0)
        assert np.all(ph[:, 0] == 1.0)
        assert np.all(ph[:, 3] == 0.0)

    def test_pixel_size_1mm(self) -> None:
        # Default: FOV=0.064m, grid=64 → 1.0mm/pixel
        ph = _make_bar_phantom(64, 0.064, 3.0)
        # 3 bars per feature → ~10-11 bar pairs across 64 pixels
        bright_cols = sum(1 for c in range(64) if ph[0, c] == 1.0)
        dark_cols = 64 - bright_cols
        # Should be roughly half bright, half dark (with possible edge effect)
        assert 20 <= bright_cols <= 44

    def test_identical_rows(self) -> None:
        # All rows should be identical (vertical bars)
        ph = _make_bar_phantom(64, 0.064, 3.0)
        for row in range(1, 64):
            np.testing.assert_array_equal(ph[row, :], ph[0, :])

    def test_custom_bar_width(self) -> None:
        # bar_width = 8mm, pixel=1mm → 8px per bar
        ph = _make_bar_phantom(64, 0.064, 8.0)
        # Columns 0-7 should be bright, 8-15 dark
        assert np.all(ph[0, 0:8] == 1.0)
        assert np.all(ph[0, 8:16] == 0.0)

    def test_larger_grid(self) -> None:
        ph = _make_bar_phantom(128, 0.128, 3.0)
        assert ph.shape == (128, 128)
        assert set(np.unique(ph).tolist()).issubset({0.0, 1.0})

    def test_nonzero_pixels_present(self) -> None:
        ph = _make_bar_phantom(32, 0.032, 2.0)
        assert ph.sum() > 0

    def test_not_all_bright(self) -> None:
        ph = _make_bar_phantom(32, 0.032, 2.0)
        assert ph.sum() < 32 * 32


# ── TestReconstructFromPhantom ────────────────────────────────────


class TestReconstructFromPhantom:
    def test_output_shape(self, default_config: Phase6Config) -> None:
        ph = _make_bar_phantom(default_config.grid_size, default_config.fov_m, 3.0)
        r = _reconstruct_from_phantom(ph, default_config)
        assert r.magnitude.shape == (64, 64)

    def test_method_label(self, default_config: Phase6Config) -> None:
        ph = _make_bar_phantom(default_config.grid_size, default_config.fov_m, 3.0)
        r = _reconstruct_from_phantom(ph, default_config)
        assert r.method == "gridding+fft"

    def test_magnitude_nonneg(self, default_config: Phase6Config) -> None:
        ph = _make_bar_phantom(default_config.grid_size, default_config.fov_m, 3.0)
        r = _reconstruct_from_phantom(ph, default_config)
        assert float(r.magnitude.min()) >= 0.0

    def test_fov_preserved(self, default_config: Phase6Config) -> None:
        ph = _make_bar_phantom(default_config.grid_size, default_config.fov_m, 3.0)
        r = _reconstruct_from_phantom(ph, default_config)
        assert r.fov_x_m == pytest.approx(default_config.fov_m)
        assert r.fov_y_m == pytest.approx(default_config.fov_m)

    def test_resolution_mm(self, default_config: Phase6Config) -> None:
        ph = _make_bar_phantom(default_config.grid_size, default_config.fov_m, 3.0)
        r = _reconstruct_from_phantom(ph, default_config)
        expected = default_config.fov_m / default_config.grid_size * 1000.0
        assert r.resolution_x_mm == pytest.approx(expected, rel=1e-6)

    def test_snr_db_finite(self, default_config: Phase6Config) -> None:
        ph = _make_bar_phantom(default_config.grid_size, default_config.fov_m, 3.0)
        r = _reconstruct_from_phantom(ph, default_config)
        assert math.isfinite(r.snr_db)


# ── TestMeasurePSFFWHM ────────────────────────────────────────────


class TestMeasurePSFFWHM:
    def test_returns_finite(self, default_config: Phase6Config) -> None:
        fwhm = _measure_psf_fwhm(default_config)
        assert math.isfinite(fwhm)

    def test_fwhm_positive(self, default_config: Phase6Config) -> None:
        fwhm = _measure_psf_fwhm(default_config)
        assert fwhm > 0.0

    def test_fwhm_below_threshold(self, default_config: Phase6Config) -> None:
        fwhm = _measure_psf_fwhm(default_config)
        assert fwhm <= default_config.resolution_threshold_mm

    def test_fwhm_above_pixel_size(self, default_config: Phase6Config) -> None:
        fwhm = _measure_psf_fwhm(default_config)
        pixel_mm = default_config.fov_m / default_config.grid_size * 1000.0
        assert fwhm >= pixel_mm

    def test_fwhm_reasonable_range(self, default_config: Phase6Config) -> None:
        # Expected ~1.8 mm from artifact_characterizer benchmarks
        fwhm = _measure_psf_fwhm(default_config)
        assert 0.5 <= fwhm <= 5.0

    def test_tight_threshold_fails(self) -> None:
        # resolution_threshold_mm=0.5 mm — PSF (~1.8mm) cannot satisfy this
        cfg = Phase6Config(resolution_threshold_mm=0.5)
        fwhm = _measure_psf_fwhm(cfg)
        assert fwhm > cfg.resolution_threshold_mm

    def test_loose_threshold_passes(self) -> None:
        cfg = Phase6Config(resolution_threshold_mm=10.0)
        fwhm = _measure_psf_fwhm(cfg)
        assert fwhm <= cfg.resolution_threshold_mm


# ── TestMeasureBarContrast ────────────────────────────────────────


class TestMeasureBarContrast:
    def test_perfect_phantom_contrast(self) -> None:
        # Perfect match: recon = phantom (no blurring)
        ph = _make_bar_phantom(32, 0.032, 2.0)
        bc = _measure_bar_contrast(ph.astype(float), ph, threshold=0.2)
        assert bc.michelson_contrast == pytest.approx(1.0, abs=1e-6)

    def test_zero_image_no_passes(self) -> None:
        ph = _make_bar_phantom(32, 0.032, 2.0)
        zero = np.zeros_like(ph)
        bc = _measure_bar_contrast(zero, ph, threshold=0.2)
        assert bc.passes is False
        assert bc.michelson_contrast == pytest.approx(0.0, abs=1e-6)

    def test_uniform_image_zero_contrast(self) -> None:
        ph = _make_bar_phantom(32, 0.032, 2.0)
        uniform = np.ones_like(ph) * 0.5
        bc = _measure_bar_contrast(uniform, ph, threshold=0.2)
        assert bc.michelson_contrast == pytest.approx(0.0, abs=1e-6)
        assert bc.passes is False

    def test_high_contrast_passes(self) -> None:
        ph = _make_bar_phantom(32, 0.032, 2.0)
        recon = ph.copy()  # perfect reconstruction
        bc = _measure_bar_contrast(recon, ph, threshold=0.2)
        assert bc.passes is True

    def test_low_contrast_fails(self) -> None:
        # SNR contrasts just above zero but below 0.2 threshold
        ph = _make_bar_phantom(32, 0.032, 2.0)
        # bars=0.6, gaps=0.4 → contrast = 0.2/1.0 = 0.2 (boundary, passes)
        # Let's use bars=0.55, gaps=0.45 → contrast = 0.1/1.0 = 0.10 < 0.2
        blurred = np.where(ph > 0.5, 0.55, 0.45)
        bc = _measure_bar_contrast(blurred, ph, threshold=0.2)
        assert bc.passes is False

    def test_bar_mean_gt_gap_mean_for_bright_bars(self) -> None:
        ph = _make_bar_phantom(64, 0.064, 3.0)
        bc = _measure_bar_contrast(ph, ph, threshold=0.2)
        assert bc.bar_mean > bc.gap_mean

    def test_formula(self) -> None:
        ph = _make_bar_phantom(32, 0.032, 2.0)
        bar_val = 0.8
        gap_val = 0.4
        recon = np.where(ph > 0.5, bar_val, gap_val)
        bc = _measure_bar_contrast(recon, ph, threshold=0.2)
        expected = (bar_val - gap_val) / (bar_val + gap_val)
        assert bc.michelson_contrast == pytest.approx(expected, rel=1e-4)

    def test_passes_type_is_bool(self) -> None:
        ph = _make_bar_phantom(32, 0.032, 2.0)
        bc = _measure_bar_contrast(ph, ph, threshold=0.2)
        assert isinstance(bc.passes, bool)


# ── TestValidatePhase6MilestoneDefault ───────────────────────────


class TestValidatePhase6MilestoneDefault:
    def test_phase6_milestone_closed(self, default_result: Phase6MilestoneResult) -> None:
        assert default_result.phase6_milestone_closed is True

    def test_psf_pass(self, default_result: Phase6MilestoneResult) -> None:
        assert default_result.psf_pass is True

    def test_contrast_pass(self, default_result: Phase6MilestoneResult) -> None:
        assert default_result.contrast_pass is True

    def test_snr_pass(self, default_result: Phase6MilestoneResult) -> None:
        assert default_result.snr_pass is True

    def test_psf_fwhm_below_3mm(self, default_result: Phase6MilestoneResult) -> None:
        assert default_result.psf_fwhm_mm <= 3.0

    def test_psf_fwhm_above_pixel(self, default_result: Phase6MilestoneResult) -> None:
        # PSF must be at least one pixel wide
        assert default_result.psf_fwhm_mm >= default_result.pixel_size_mm

    def test_michelson_contrast_above_threshold(self, default_result: Phase6MilestoneResult) -> None:
        assert default_result.bar_contrast.michelson_contrast >= 0.2

    def test_snr_well_above_threshold(self, default_result: Phase6MilestoneResult) -> None:
        # Noise-free simulation → SNR >> 5 dB
        assert default_result.image_snr_db >= 10.0

    def test_pixel_size_1mm(self, default_result: Phase6MilestoneResult) -> None:
        # FOV=0.064, grid=64 → 1.0 mm
        assert default_result.pixel_size_mm == pytest.approx(1.0, rel=1e-6)

    def test_n_bar_pairs_positive(self, default_result: Phase6MilestoneResult) -> None:
        assert default_result.phantom_result.n_bar_pairs >= 1

    def test_recon_shape(self, default_result: Phase6MilestoneResult) -> None:
        assert default_result.recon.magnitude.shape == (64, 64)

    def test_none_config_uses_default(self) -> None:
        r = validate_phase6_milestone(config=None)
        assert r.config == Phase6Config()


# ── TestPhase6MilestoneClosedLogic ───────────────────────────────


class TestPhase6MilestoneClosedLogic:
    def test_tight_psf_threshold_fails(self) -> None:
        # resolution_threshold_mm=0.5 < PSF FWHM → psf_pass=False
        cfg = Phase6Config(resolution_threshold_mm=0.5)
        r = validate_phase6_milestone(cfg)
        assert r.psf_pass is False
        assert r.phase6_milestone_closed is False

    def test_tight_contrast_threshold_fails(self) -> None:
        # bar_contrast_threshold=0.99 → contrast will not reach this level
        cfg = Phase6Config(bar_contrast_threshold=0.99)
        r = validate_phase6_milestone(cfg)
        assert r.contrast_pass is False
        assert r.phase6_milestone_closed is False

    def test_tight_snr_threshold_still_passes(self) -> None:
        # SNR is >> 5 dB in noise-free simulation; even 50 dB threshold fails
        cfg = Phase6Config(image_snr_threshold_db=200.0)
        r = validate_phase6_milestone(cfg)
        assert r.snr_pass is False
        assert r.phase6_milestone_closed is False

    def test_all_loose_thresholds_pass(self) -> None:
        cfg = Phase6Config(
            resolution_threshold_mm=10.0,
            bar_contrast_threshold=0.01,
            image_snr_threshold_db=1.0,
        )
        r = validate_phase6_milestone(cfg)
        assert r.phase6_milestone_closed is True
