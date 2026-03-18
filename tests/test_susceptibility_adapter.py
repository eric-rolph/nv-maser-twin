"""
Tests for the susceptibility adapter module.

Covers:
- Physical correctness of field-shift calculations
- Numerical properties (uniform tissue → zero shift)
- Known boundary cases (fat-muscle, air-tissue)
- Integration with TissueLayer and depth profile data
- Cross-validation utilities
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nv_maser.physics.depth_profile import (
    FOREARM_LAYERS,
    TissueLayer,
    _assign_layers,
)
from nv_maser.physics.susceptibility_adapter import (
    SUSCEPTIBILITY_TABLE,
    SusceptibilityProfile,
    SusceptibilityCorrectedProfile,
    apply_susceptibility_correction,
    compute_dephasing_signal_loss,
    compute_frequency_shift,
    compute_susceptibility_field_shift,
    cross_validate_susceptibility,
    estimate_susceptibility_impact,
)

# ── Constants ─────────────────────────────────────────────────────
_GAMMA_P_HZ_T = 42.577e6   # Hz/T
_B0_DEFAULT = 0.050        # 50 mT operating field


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def uniform_layers() -> list[TissueLayer]:
    """Single tissue type — should produce zero ΔB₀ everywhere."""
    return [
        TissueLayer("muscle", thickness_mm=30.0, proton_density=1.0,
                    t1_ms=600, t2_ms=35, susceptibility_ppm=-9.05),
    ]


@pytest.fixture
def fat_muscle_layers() -> list[TissueLayer]:
    """Fat layer over muscle — creates a single susceptibility boundary."""
    return [
        TissueLayer("fat", thickness_mm=10.0, proton_density=0.9,
                    t1_ms=250, t2_ms=80, susceptibility_ppm=-7.79),
        TissueLayer("muscle", thickness_mm=20.0, proton_density=1.0,
                    t1_ms=600, t2_ms=35, susceptibility_ppm=-9.05),
    ]


@pytest.fixture
def air_tissue_layers() -> list[TissueLayer]:
    """Air gap over muscle — maximum Δχ in expected tissues."""
    return [
        TissueLayer("air", thickness_mm=2.0, proton_density=0.0,
                    t1_ms=1e6, t2_ms=1e6, susceptibility_ppm=0.36),
        TissueLayer("muscle", thickness_mm=28.0, proton_density=1.0,
                    t1_ms=600, t2_ms=35, susceptibility_ppm=-9.05),
    ]


def _make_depths(max_mm: float = 30.0, step_mm: float = 0.5) -> np.ndarray:
    return np.arange(step_mm / 2, max_mm, step_mm)


def _make_b0(depths: np.ndarray, b0: float = _B0_DEFAULT) -> np.ndarray:
    """Uniform B₀ profile for all test depths."""
    return np.full_like(depths, b0)


# ── Test: TissueLayer susceptibility field ─────────────────────────

class TestTissueLayerSusceptibility:
    def test_default_susceptibility_ppm(self):
        layer = TissueLayer("water", thickness_mm=10.0)
        assert layer.susceptibility_ppm == pytest.approx(-9.05)

    def test_custom_susceptibility_ppm(self):
        layer = TissueLayer("fat", thickness_mm=5.0, susceptibility_ppm=-7.79)
        assert layer.susceptibility_ppm == pytest.approx(-7.79)

    def test_forearm_layers_have_susceptibility(self):
        for layer in FOREARM_LAYERS:
            assert hasattr(layer, "susceptibility_ppm")
            assert -25 < layer.susceptibility_ppm < 5  # all diamagnetic

    def test_susceptibility_table_completeness(self):
        """Check canonical values exist for common tissues."""
        for tissue in ("water", "fat", "muscle", "bone"):
            assert tissue in SUSCEPTIBILITY_TABLE or any(
                t in SUSCEPTIBILITY_TABLE for t in [tissue, tissue + "_cortex", "subcutaneous_" + tissue]
            )

    def test_assign_layers_passes_susceptibility(self):
        layers = [TissueLayer("skin", 5.0, susceptibility_ppm=-9.4)]
        depths = np.array([1.0, 3.0, 5.0])
        props = _assign_layers(depths, layers)
        for p in props:
            assert "susceptibility_ppm" in p
            assert p["susceptibility_ppm"] == pytest.approx(-9.4)


# ── Test: Field shift computation ─────────────────────────────────

class TestFieldShiftComputation:
    def test_uniform_tissue_zero_shift(self, uniform_layers):
        depths = _make_depths()
        b0 = _make_b0(depths)
        profile = compute_susceptibility_field_shift(
            uniform_layers, b0, depths, reference_chi_ppm=-9.05
        )
        # Δχ = 0 everywhere → ΔB₀ = 0
        np.testing.assert_allclose(profile.delta_b0_tesla, 0.0, atol=1e-15)

    def test_uniform_tissue_zero_freq_shift(self, uniform_layers):
        depths = _make_depths()
        b0 = _make_b0(depths)
        profile = compute_susceptibility_field_shift(
            uniform_layers, b0, depths, reference_chi_ppm=-9.05
        )
        np.testing.assert_allclose(profile.delta_freq_hz, 0.0, atol=1e-6)

    def test_fat_over_reference_positive_shift(self, fat_muscle_layers):
        """Fat (χ=-7.79) is less diamagnetic than reference water (χ=-9.05).
        → Δχ > 0 → ΔB₀ > 0 in fat region."""
        depths = _make_depths()
        b0 = _make_b0(depths)
        profile = compute_susceptibility_field_shift(
            fat_muscle_layers, b0, depths, reference_chi_ppm=-9.05
        )
        # Fat region: depths <= 10 mm
        fat_mask = depths <= 10.0
        assert np.any(profile.delta_b0_tesla[fat_mask] > 0), "Fat should show positive ΔB₀"

    def test_muscle_at_reference_near_zero(self, fat_muscle_layers):
        """Muscle χ = reference χ = -9.05 → ΔB₀ ≈ 0 in muscle region."""
        depths = _make_depths()
        b0 = _make_b0(depths)
        profile = compute_susceptibility_field_shift(
            fat_muscle_layers, b0, depths, reference_chi_ppm=-9.05
        )
        muscle_mask = depths > 10.0
        np.testing.assert_allclose(
            profile.delta_b0_tesla[muscle_mask], 0.0, atol=1e-15
        )

    def test_magnitude_at_50mt_fat_muscle(self, fat_muscle_layers):
        """Physics check: fat-muscle Δχ = 1.26 ppm, sphere geometry, B₀=50 mT.
        ΔB₀ = (1.26e-6 / 3) × 0.050 ≈ 2.1e-8 T = 21 nT."""
        depths = _make_depths()
        b0 = _make_b0(depths, b0=0.050)
        profile = compute_susceptibility_field_shift(
            fat_muscle_layers, b0, depths, reference_chi_ppm=-9.05, geometry="sphere"
        )
        fat_delta = profile.delta_b0_tesla[depths <= 10.0]
        expected = (1.26e-6 / 3) * 0.050  # ≈ 2.1e-8 T
        np.testing.assert_allclose(fat_delta, expected, rtol=0.05)

    def test_air_tissue_boundary_larger_shift(self, air_tissue_layers):
        """Air (χ=+0.36) vs reference (-9.05): Δχ = 9.41 ppm — much larger than fat."""
        depths = _make_depths()
        b0 = _make_b0(depths)
        profile = compute_susceptibility_field_shift(
            air_tissue_layers, b0, depths, reference_chi_ppm=-9.05
        )
        air_delta = np.max(np.abs(profile.delta_b0_tesla))
        fat_delta = (1.26e-6 / 3) * 0.050  # fat-muscle reference
        assert air_delta > fat_delta * 5, "Air-tissue shift should be >> fat-muscle shift"

    def test_slab_geometry_larger_than_sphere(self, fat_muscle_layers):
        """Slab model (N=1) gives 3× the shift of sphere model (N=1/3)."""
        depths = _make_depths()
        b0 = _make_b0(depths)
        sphere = compute_susceptibility_field_shift(
            fat_muscle_layers, b0, depths, geometry="sphere"
        )
        slab = compute_susceptibility_field_shift(
            fat_muscle_layers, b0, depths, geometry="slab"
        )
        fat_mask = depths <= 10.0
        ratio = np.mean(np.abs(slab.delta_b0_tesla[fat_mask])) / np.mean(np.abs(sphere.delta_b0_tesla[fat_mask]))
        assert ratio == pytest.approx(3.0, rel=0.01)

    def test_invalid_geometry_raises(self, fat_muscle_layers):
        depths = _make_depths()
        b0 = _make_b0(depths)
        with pytest.raises(ValueError, match="geometry"):
            compute_susceptibility_field_shift(
                fat_muscle_layers, b0, depths, geometry="cylinder"
            )

    def test_profile_fields_populated(self, fat_muscle_layers):
        depths = _make_depths()
        b0 = _make_b0(depths)
        profile = compute_susceptibility_field_shift(fat_muscle_layers, b0, depths)
        assert isinstance(profile, SusceptibilityProfile)
        assert len(profile.chi_ppm) == len(depths)
        assert len(profile.delta_freq_hz) == len(depths)
        assert len(profile.dephasing_factor) == len(depths)
        assert len(profile.tissue_labels) == len(depths)

    def test_frequency_shift_accessor(self, fat_muscle_layers):
        depths = _make_depths()
        b0 = _make_b0(depths)
        profile = compute_susceptibility_field_shift(fat_muscle_layers, b0, depths)
        freq_shift = compute_frequency_shift(profile)
        np.testing.assert_array_equal(freq_shift, profile.delta_freq_hz)


# ── Test: Dephasing signal loss ────────────────────────────────────

class TestDephasingSignalLoss:
    def test_zero_gradient_no_loss(self):
        """Uniform ΔB₀ → zero gradient → no dephasing attenuation."""
        delta_b0 = np.full(60, 2e-8)  # constant, no gradient
        loss = compute_dephasing_signal_loss(
            delta_b0, te_s=0.010, voxel_size_mm=3.0, depth_step_mm=0.5
        )
        # Interior points should be ~1 (minimal gradient)
        np.testing.assert_allclose(loss[2:-2], 1.0, atol=1e-3)

    def test_large_gradient_reduces_signal(self):
        """Sharp ΔB₀ step should cause significant dephasing."""
        # Create a step function: 0 for first half, large value for second half
        delta_b0 = np.concatenate([
            np.zeros(30),
            np.full(30, 1e-5)  # very large ΔB₀ step
        ])
        loss = compute_dephasing_signal_loss(
            delta_b0, te_s=0.010, voxel_size_mm=3.0, depth_step_mm=0.5
        )
        # At the boundary, loss should be < 1
        assert np.any(loss < 0.99), "Sharp boundary should show dephasing loss"

    def test_loss_values_in_range(self, fat_muscle_layers):
        depths = _make_depths()
        b0 = _make_b0(depths)
        profile = compute_susceptibility_field_shift(fat_muscle_layers, b0, depths)
        loss = compute_dephasing_signal_loss(
            profile.delta_b0_tesla, te_s=0.010, voxel_size_mm=3.0, depth_step_mm=0.5
        )
        assert np.all(loss >= 0.0)
        assert np.all(loss <= 1.0 + 1e-10)


# ── Test: Apply correction ─────────────────────────────────────────

class TestApplySusceptibilityCorrection:
    def test_uniform_tissue_no_correction(self, uniform_layers):
        depths = _make_depths()
        b0 = _make_b0(depths)
        profile = compute_susceptibility_field_shift(
            uniform_layers, b0, depths, reference_chi_ppm=-9.05
        )
        signal_in = np.ones_like(depths)
        result = apply_susceptibility_correction(
            signal_in, profile, te_s=0.010, voxel_size_mm=3.0, depth_step_mm=0.5
        )
        assert isinstance(result, SusceptibilityCorrectedProfile)
        # Uniform tissue → correction_factor ≈ 1 everywhere (interior)
        np.testing.assert_allclose(result.correction_factor[2:-2], 1.0, atol=1e-3)

    def test_corrected_signal_leq_original(self, FOREARM_LAYERS_fixture=None):
        """Susceptibility correction can only reduce (or preserve) signal."""
        layers = FOREARM_LAYERS
        depths = _make_depths()
        b0 = _make_b0(depths)
        profile = compute_susceptibility_field_shift(layers, b0, depths)
        signal_in = np.abs(np.random.default_rng(42).normal(1.0, 0.1, len(depths)))
        result = apply_susceptibility_correction(
            signal_in, profile, te_s=0.010, voxel_size_mm=3.0, depth_step_mm=0.5
        )
        # After dephasing correction, signal should be <= original
        np.testing.assert_array_less(
            result.signal_corrected - 1e-10, result.signal_original + 1e-10
        )


# ── Test: Estimate impact ──────────────────────────────────────────

class TestEstimateSusceptibilityImpact:
    def test_uniform_tissue_zero_impact(self, uniform_layers):
        impact = estimate_susceptibility_impact(uniform_layers, b0_mean_t=0.050)
        assert impact["max_delta_chi_ppm"] == pytest.approx(0.0, abs=1e-10)
        assert impact["max_delta_b0_ut"] == pytest.approx(0.0, abs=1e-10)
        assert impact["boundary_count"] == 0.0

    def test_fat_muscle_boundary_impact(self, fat_muscle_layers):
        impact = estimate_susceptibility_impact(fat_muscle_layers, b0_mean_t=0.050)
        assert impact["max_delta_chi_ppm"] == pytest.approx(1.26, rel=0.05)
        # ΔB₀ = (1.26e-6 / 3) × 0.050 ≈ 21 nT = 0.021 μT
        assert impact["max_delta_b0_ut"] == pytest.approx(0.021, rel=0.05)
        assert impact["boundary_count"] == 1.0

    def test_forearm_boundary_count(self):
        impact = estimate_susceptibility_impact(FOREARM_LAYERS, b0_mean_t=0.050)
        # skin→fat, fat→muscle, muscle→bone = 3 boundaries with distinct chi
        assert impact["boundary_count"] >= 2  # at minimum fat-muscle and muscle-bone

    def test_impact_returns_dict_with_required_keys(self, fat_muscle_layers):
        impact = estimate_susceptibility_impact(fat_muscle_layers, b0_mean_t=0.050)
        for key in ("max_delta_chi_ppm", "max_delta_b0_ut", "max_delta_freq_hz", "boundary_count"):
            assert key in impact


# ── Test: Cross-validation ─────────────────────────────────────────

class TestCrossValidateSusceptibility:
    def test_identical_profiles_zero_diff(self):
        signal = np.array([1.0, 0.8, 0.6, 0.4])
        result = cross_validate_susceptibility(signal, signal.copy())
        assert result["max_relative_change"] == pytest.approx(0.0, abs=1e-12)
        assert result["mean_relative_change"] == pytest.approx(0.0, abs=1e-12)
        assert result["correlation"] == pytest.approx(1.0, abs=1e-10)

    def test_small_correction_small_diff(self):
        signal = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        correction = np.array([0.99, 0.995, 0.998, 0.999, 0.999])
        corrected = signal * correction
        result = cross_validate_susceptibility(signal, corrected)
        assert result["max_relative_change"] < 0.02  # < 2%

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="shape"):
            cross_validate_susceptibility(np.ones(10), np.ones(12))

    def test_high_correlation_preserved(self, fat_muscle_layers):
        depths = _make_depths()
        b0 = _make_b0(depths)
        profile = compute_susceptibility_field_shift(fat_muscle_layers, b0, depths)
        signal_in = np.random.default_rng(7).uniform(0.1, 1.0, len(depths))
        result = apply_susceptibility_correction(
            signal_in, profile, te_s=0.010, voxel_size_mm=3.0, depth_step_mm=0.5
        )
        cv = cross_validate_susceptibility(
            result.signal_original, result.signal_corrected
        )
        # Tissue susceptibility at 50 mT causes small but non-zero correction
        assert cv["correlation"] > 0.999
