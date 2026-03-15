"""Tests for Halbach permanent magnet array multipole field model."""
import math

import numpy as np
import pytest

from nv_maser.config import FieldConfig, GridConfig, HalbachConfig
from nv_maser.physics.grid import SpatialGrid
from nv_maser.physics.halbach import (
    MultipoleCoefficients,
    compute_halbach_field,
    compute_manufacturing_errors,
    compute_multipole_coefficients,
    compute_segmentation_harmonics,
    evaluate_multipole_field,
)
from nv_maser.physics.base_field import compute_base_field


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def grid() -> SpatialGrid:
    return SpatialGrid(GridConfig())


@pytest.fixture
def halbach_cfg() -> HalbachConfig:
    return HalbachConfig(enabled=True, seed=42)


@pytest.fixture
def field_cfg() -> FieldConfig:
    return FieldConfig(b0_tesla=0.05)


# ── HalbachConfig ────────────────────────────────────────────────


class TestHalbachConfig:
    def test_default_disabled(self) -> None:
        cfg = HalbachConfig()
        assert cfg.enabled is False

    def test_ideal_b0_formula(self) -> None:
        cfg = HalbachConfig(num_segments=8, inner_radius_mm=7.0,
                            outer_radius_mm=15.0, remanence_tesla=1.4)
        expected = 1.4 * math.log(15.0 / 7.0) * math.sin(math.pi / 8) / (math.pi / 8)
        assert cfg.ideal_b0_tesla == pytest.approx(expected, rel=1e-10)

    def test_ideal_b0_positive(self) -> None:
        cfg = HalbachConfig()
        assert cfg.ideal_b0_tesla > 0

    def test_more_segments_larger_b0(self) -> None:
        """sin(π/N)/(π/N) → 1 as N → ∞, so more segments ≈ higher field."""
        b0_8 = HalbachConfig(num_segments=8).ideal_b0_tesla
        b0_24 = HalbachConfig(num_segments=24).ideal_b0_tesla
        assert b0_24 > b0_8


# ── Segmentation harmonics ───────────────────────────────────────


class TestSegmentationHarmonics:
    def test_only_cosine_terms(self, halbach_cfg: HalbachConfig) -> None:
        """Symmetric arrangement → b_n all zero."""
        _, b_n = compute_segmentation_harmonics(halbach_cfg, b0=0.05)
        np.testing.assert_array_equal(b_n, 0.0)

    def test_harmonics_at_kN_pm_1(self, halbach_cfg: HalbachConfig) -> None:
        """For N=8, expect non-zero at orders 7 (8-1) and 9 (8+1)."""
        a_n, _ = compute_segmentation_harmonics(halbach_cfg, b0=0.05)
        # a_n indices: order 2 → idx 0, order 7 → idx 5, order 9 → idx 7
        assert a_n[5] != 0.0  # order 7 = 8-1
        assert a_n[7] != 0.0  # order 9 = 8+1

    def test_non_harmonic_orders_zero(self, halbach_cfg: HalbachConfig) -> None:
        """Orders that are NOT kN±1 should be zero (systematic only)."""
        a_n, _ = compute_segmentation_harmonics(halbach_cfg, b0=0.05)
        # With N=8, max_order=12: kN±1 = {7,9}.  Orders {2,3,4,5,6,8,10,11,12} should be 0.
        non_harmonic_orders = [2, 3, 4, 5, 6, 8, 10, 11, 12]
        for n in non_harmonic_orders:
            assert a_n[n - 2] == pytest.approx(0.0), f"order {n} should be zero"

    def test_amplitude_proportional_to_b0(self, halbach_cfg: HalbachConfig) -> None:
        a1, _ = compute_segmentation_harmonics(halbach_cfg, b0=0.05)
        a2, _ = compute_segmentation_harmonics(halbach_cfg, b0=0.10)
        np.testing.assert_allclose(a2, 2.0 * a1, rtol=1e-12)


# ── Manufacturing errors ─────────────────────────────────────────


class TestManufacturingErrors:
    def test_seed_reproducibility(self, halbach_cfg: HalbachConfig) -> None:
        rng1 = np.random.default_rng(42)
        a1, b1 = compute_manufacturing_errors(halbach_cfg, 0.05, rng1)
        rng2 = np.random.default_rng(42)
        a2, b2 = compute_manufacturing_errors(halbach_cfg, 0.05, rng2)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)

    def test_zero_tolerances_zero_errors(self) -> None:
        cfg = HalbachConfig(
            enabled=True,
            br_tolerance_pct=0.0,
            angle_tolerance_deg=0.0,
            position_tolerance_mm=0.0,
            seed=99,
        )
        rng = np.random.default_rng(99)
        a_n, b_n = compute_manufacturing_errors(cfg, 0.05, rng)
        np.testing.assert_array_equal(a_n, 0.0)
        np.testing.assert_array_equal(b_n, 0.0)

    def test_errors_have_both_cosine_and_sine(self, halbach_cfg: HalbachConfig) -> None:
        rng = np.random.default_rng(42)
        a_n, b_n = compute_manufacturing_errors(halbach_cfg, 0.05, rng)
        # Random errors → both a_n and b_n should have non-zero entries
        assert np.any(a_n != 0.0)
        assert np.any(b_n != 0.0)

    def test_errors_scale_with_tolerances(self) -> None:
        """Larger tolerances → larger errors (on average)."""
        cfg_small = HalbachConfig(
            enabled=True, br_tolerance_pct=0.1,
            angle_tolerance_deg=0.1, position_tolerance_mm=0.01, seed=42,
        )
        cfg_large = HalbachConfig(
            enabled=True, br_tolerance_pct=5.0,
            angle_tolerance_deg=2.0, position_tolerance_mm=0.3, seed=42,
        )
        rng1 = np.random.default_rng(42)
        a_sm, b_sm = compute_manufacturing_errors(cfg_small, 0.05, rng1)
        rng2 = np.random.default_rng(42)
        a_lg, b_lg = compute_manufacturing_errors(cfg_large, 0.05, rng2)

        rms_small = np.sqrt(np.mean(a_sm**2 + b_sm**2))
        rms_large = np.sqrt(np.mean(a_lg**2 + b_lg**2))
        assert rms_large > rms_small


# ── MultipoleCoefficients ────────────────────────────────────────


class TestMultipoleCoefficients:
    def test_total_rms_error_finite(self, halbach_cfg: HalbachConfig) -> None:
        coeffs = compute_multipole_coefficients(halbach_cfg, 0.05)
        rms = coeffs.total_rms_error(r_fraction=0.6)
        assert np.isfinite(rms)
        assert rms >= 0.0

    def test_total_rms_error_increases_with_radius(self, halbach_cfg: HalbachConfig) -> None:
        coeffs = compute_multipole_coefficients(halbach_cfg, 0.05)
        rms_inner = coeffs.total_rms_error(0.2)
        rms_outer = coeffs.total_rms_error(0.9)
        assert rms_outer > rms_inner

    def test_combines_systematic_and_random(self, halbach_cfg: HalbachConfig) -> None:
        coeffs = compute_multipole_coefficients(halbach_cfg, 0.05)
        # With tolerances > 0 and seed, b_n should be non-zero (random contributes sine terms)
        assert np.any(coeffs.b_n != 0.0)
        # Systematic contributes to a_n at kN±1 orders
        assert coeffs.a_n[5] != 0.0  # order 7


# ── evaluate_multipole_field ─────────────────────────────────────


class TestEvaluateMultipoleField:
    def test_shape(self, grid: SpatialGrid, halbach_cfg: HalbachConfig) -> None:
        coeffs = compute_multipole_coefficients(halbach_cfg, 0.05)
        field = evaluate_multipole_field(grid, coeffs, halbach_cfg.inner_radius_mm)
        assert field.shape == grid.shape

    def test_dtype_float32(self, grid: SpatialGrid, halbach_cfg: HalbachConfig) -> None:
        coeffs = compute_multipole_coefficients(halbach_cfg, 0.05)
        field = evaluate_multipole_field(grid, coeffs, halbach_cfg.inner_radius_mm)
        assert field.dtype == np.float32

    def test_center_equals_b0(self, grid: SpatialGrid) -> None:
        """At r=0, all multipole terms vanish → field = B₀ exactly."""
        cfg = HalbachConfig(enabled=True, seed=42)
        b0 = 0.05
        coeffs = compute_multipole_coefficients(cfg, b0)
        field = evaluate_multipole_field(grid, coeffs, cfg.inner_radius_mm)
        # Center of 64×64 grid: indices (32, 32) is at (0,0) for even size
        # Actually for linspace(-5, 5, 64) the values near center aren't exactly 0
        # but at r≈0 the (r/R_in)^(n-1) terms → 0 for n≥2
        center = grid.size // 2
        center_val = field[center, center]
        # At r ≈ 0.08 mm (first grid point near center), multipole terms tiny
        assert center_val == pytest.approx(b0, abs=1e-5)

    def test_field_varies_spatially(self, grid: SpatialGrid, halbach_cfg: HalbachConfig) -> None:
        """Multipole errors → field is not perfectly uniform."""
        coeffs = compute_multipole_coefficients(halbach_cfg, 0.05)
        field = evaluate_multipole_field(grid, coeffs, halbach_cfg.inner_radius_mm)
        assert np.std(field) > 0


# ── compute_halbach_field (top-level) ────────────────────────────


class TestComputeHalbachField:
    def test_shape(self, grid: SpatialGrid, field_cfg: FieldConfig,
                   halbach_cfg: HalbachConfig) -> None:
        field = compute_halbach_field(grid, field_cfg, halbach_cfg)
        assert field.shape == grid.shape

    def test_dtype(self, grid: SpatialGrid, field_cfg: FieldConfig,
                   halbach_cfg: HalbachConfig) -> None:
        field = compute_halbach_field(grid, field_cfg, halbach_cfg)
        assert field.dtype == np.float32

    def test_mean_near_b0(self, grid: SpatialGrid, field_cfg: FieldConfig,
                          halbach_cfg: HalbachConfig) -> None:
        """Mean field should be close to B₀ (errors are small perturbations)."""
        field = compute_halbach_field(grid, field_cfg, halbach_cfg)
        assert np.mean(field) == pytest.approx(field_cfg.b0_tesla, abs=1e-3)

    def test_seed_reproducibility(self, grid: SpatialGrid, field_cfg: FieldConfig) -> None:
        cfg1 = HalbachConfig(enabled=True, seed=42)
        cfg2 = HalbachConfig(enabled=True, seed=42)
        f1 = compute_halbach_field(grid, field_cfg, cfg1)
        f2 = compute_halbach_field(grid, field_cfg, cfg2)
        np.testing.assert_array_equal(f1, f2)


# ── Integration with base_field ──────────────────────────────────


class TestBaseFieldHalbachIntegration:
    def test_halbach_disabled_flat_field(self, grid: SpatialGrid) -> None:
        """When halbach is disabled, field should be uniform (legacy behaviour)."""
        cfg = FieldConfig(b0_tesla=0.05)
        halbach = HalbachConfig(enabled=False)
        field = compute_base_field(grid, cfg, halbach)
        assert np.std(field) == pytest.approx(0.0, abs=1e-6)

    def test_halbach_none_flat_field(self, grid: SpatialGrid) -> None:
        """When halbach is None, field should be uniform (backward compat)."""
        cfg = FieldConfig(b0_tesla=0.05)
        field = compute_base_field(grid, cfg, halbach=None)
        assert np.std(field) == pytest.approx(0.0, abs=1e-6)

    def test_halbach_enabled_adds_structure(self, grid: SpatialGrid) -> None:
        """When halbach enabled, field has spatial structure (std > 0)."""
        cfg = FieldConfig(b0_tesla=0.05)
        halbach = HalbachConfig(enabled=True, seed=42)
        field = compute_base_field(grid, cfg, halbach)
        assert np.std(field) > 0

    def test_halbach_enabled_preserves_mean(self, grid: SpatialGrid) -> None:
        """Mean field ≈ B₀ regardless of multipole errors."""
        b0 = 0.05
        cfg = FieldConfig(b0_tesla=b0)
        halbach = HalbachConfig(enabled=True, seed=42)
        field = compute_base_field(grid, cfg, halbach)
        assert np.mean(field) == pytest.approx(b0, abs=1e-3)

    def test_zero_tolerances_nearly_uniform(self, grid: SpatialGrid) -> None:
        """Zero manufacturing tolerances → essentially uniform (only segmentation)."""
        cfg = FieldConfig(b0_tesla=0.05)
        halbach = HalbachConfig(
            enabled=True, seed=42,
            br_tolerance_pct=0.0,
            angle_tolerance_deg=0.0,
            position_tolerance_mm=0.0,
        )
        field = compute_base_field(grid, cfg, halbach)
        # Segmentation harmonics for N=8 at orders 7,9 produce ~0.04% std
        assert np.std(field) < 1e-3


# ── Backward compatibility ───────────────────────────────────────


class TestBackwardCompatibility:
    def test_existing_tests_unaffected(self, grid: SpatialGrid) -> None:
        """compute_base_field without halbach arg produces same result as before."""
        cfg = FieldConfig(b0_tesla=0.05, b0_gradient_ppm_per_mm=0.0)
        field = compute_base_field(grid, cfg)
        np.testing.assert_allclose(field, 0.05, atol=1e-6)

    def test_gradient_still_works(self, grid: SpatialGrid) -> None:
        """Linear gradient path still functional."""
        cfg = FieldConfig(b0_tesla=0.05, b0_gradient_ppm_per_mm=10.0)
        field = compute_base_field(grid, cfg)
        assert np.std(field) > 0
