"""Tests for the MRzero-Core Bloch-equation adapter.

Validates that MRzero's isochromat Bloch simulation agrees with our
analytical depth-profile contrast model across a range of tissue
parameters and sequence timings.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nv_maser.physics.mrzero_adapter import (
    MRZERO_AVAILABLE,
    BlochSignal,
    DepthValidation,
    build_single_voxel_phantom,
    build_spin_echo_sequence,
    simulate_single_voxel_bloch,
    compute_analytical_contrast,
    simulate_depth_bloch,
    cross_validate_depth,
    cross_validate_contrast,
)
from nv_maser.physics.depth_profile import (
    TissueLayer,
    FOREARM_LAYERS,
    HEMORRHAGE_LAYERS,
)

pytestmark = pytest.mark.skipif(
    not MRZERO_AVAILABLE, reason="MRzeroCore not installed"
)


# ── Constants ─────────────────────────────────────────────────────

# Tolerance for Bloch vs analytical comparison.  The Bloch sim uses
# randomised intra-voxel spins, so we allow ~2 % tolerance.
RTOL = 0.03
ATOL = 1e-4


# ── Fixture ───────────────────────────────────────────────────────


@pytest.fixture
def simple_layers() -> list[TissueLayer]:
    """Two-layer phantom for fast tests."""
    return [
        TissueLayer("water", thickness_mm=10.0, proton_density=1.0,
                     t1_ms=1000, t2_ms=100),
        TissueLayer("fat", thickness_mm=10.0, proton_density=0.9,
                     t1_ms=250, t2_ms=80),
    ]


# ══════════════════════════════════════════════════════════════════
#  1.  Phantom construction
# ══════════════════════════════════════════════════════════════════


class TestBuildPhantom:
    """Tests for CustomVoxelPhantom construction."""

    def test_single_voxel_properties(self):
        phantom = build_single_voxel_phantom(1.0, 0.05, 1.0)
        assert phantom.T1.item() == pytest.approx(1.0)
        assert phantom.T2.item() == pytest.approx(0.05)
        assert phantom.PD.item() == pytest.approx(1.0)

    def test_build_sim_data(self):
        phantom = build_single_voxel_phantom(0.5, 0.03, 0.8)
        data = phantom.build()
        assert data.PD.numel() == 1
        assert data.T1.item() == pytest.approx(0.5)
        assert data.T2.item() == pytest.approx(0.03)

    def test_custom_b0_offset(self):
        phantom = build_single_voxel_phantom(1.0, 0.05, 1.0, b0_hz=100.0)
        assert phantom.B0.item() == pytest.approx(100.0)

    def test_custom_t2dash(self):
        phantom = build_single_voxel_phantom(1.0, 0.05, 1.0, t2dash_s=0.02)
        assert phantom.T2dash.item() == pytest.approx(0.02)


# ══════════════════════════════════════════════════════════════════
#  2.  Sequence construction
# ══════════════════════════════════════════════════════════════════


class TestBuildSequence:
    """Tests for spin-echo sequence building."""

    def test_single_rep_structure(self):
        seq = build_spin_echo_sequence(te_s=0.01, tr_s=0.1, n_repetitions=1)
        # 2 reps per TR: excite + refocus
        assert len(seq) == 2

    def test_multi_rep_structure(self):
        seq = build_spin_echo_sequence(te_s=0.01, tr_s=0.1, n_repetitions=5)
        assert len(seq) == 10

    def test_adc_count(self):
        seq = build_spin_echo_sequence(te_s=0.01, tr_s=0.1, n_repetitions=3)
        import torch
        adc_total = sum(
            int((rep.adc_usage > 0).sum().item()) for rep in seq
        )
        # One ADC per TR
        assert adc_total == 3

    def test_excitation_angle(self):
        seq = build_spin_echo_sequence(te_s=0.01, tr_s=0.1, n_repetitions=1)
        # First rep is excitation
        angle = seq[0].pulse.angle.item()
        assert angle == pytest.approx(math.pi / 2, abs=1e-6)

    def test_refocus_angle(self):
        seq = build_spin_echo_sequence(te_s=0.01, tr_s=0.1, n_repetitions=1)
        angle = seq[1].pulse.angle.item()
        assert angle == pytest.approx(math.pi, abs=1e-6)


# ══════════════════════════════════════════════════════════════════
#  3.  Single-voxel Bloch physics
# ══════════════════════════════════════════════════════════════════


class TestSingleVoxelBloch:
    """Validate Bloch sim against known analytical results."""

    def test_t2_decay_pure(self):
        """With long T₁ and short TE, signal ≈ PD × exp(−TE/T₂)."""
        t2 = 0.05
        te = 0.01
        expected = math.exp(-te / t2)
        sig = simulate_single_voxel_bloch(
            t1_s=10.0, t2_s=t2, proton_density=1.0,
            te_s=te, tr_s=10.0,
            spin_count=200, n_repetitions=1,
        )
        assert sig == pytest.approx(expected, rel=RTOL)

    def test_proton_density_scaling(self):
        """Signal should scale linearly with PD."""
        kwargs = dict(t1_s=5.0, t2_s=0.1, te_s=0.01, tr_s=5.0,
                      spin_count=200, n_repetitions=1)
        sig_1 = simulate_single_voxel_bloch(proton_density=1.0, **kwargs)
        sig_half = simulate_single_voxel_bloch(proton_density=0.5, **kwargs)
        assert sig_half == pytest.approx(sig_1 * 0.5, rel=RTOL)

    def test_t1_saturation_steady_state(self):
        """After many TRs the signal approaches the spin-echo steady state.

        The exact steady-state accounts for the 180° refocusing pulse
        inverting Mz at TE/2:

            Mz_ss = 1 − [2 − exp(−TE/(2T₁))] × exp(−(TR−TE/2)/T₁)
            Signal = Mz_ss × exp(−TE/T₂)
        """
        t1, t2, te, tr = 1.0, 0.1, 0.01, 0.1
        # Correct spin-echo steady state (180° flips residual Mz)
        mz_after_180 = -(1 - math.exp(-te / (2 * t1)))
        mz_ss = 1 - (1 - mz_after_180) * math.exp(-(tr - te / 2) / t1)
        expected = mz_ss * math.exp(-te / t2)
        sig = simulate_single_voxel_bloch(
            t1_s=t1, t2_s=t2, proton_density=1.0,
            te_s=te, tr_s=tr,
            spin_count=200, n_repetitions=20,
        )
        assert sig == pytest.approx(expected, rel=0.05)

    def test_long_t2_near_unity(self):
        """With very long T₂ and single TR from equilibrium, signal ≈ PD."""
        sig = simulate_single_voxel_bloch(
            t1_s=10.0, t2_s=10.0, proton_density=1.0,
            te_s=0.001, tr_s=10.0,
            spin_count=200, n_repetitions=1,
        )
        assert sig == pytest.approx(1.0, rel=RTOL)

    def test_very_short_t2_near_zero(self):
        """With T₂ ≪ TE, signal → 0."""
        sig = simulate_single_voxel_bloch(
            t1_s=10.0, t2_s=0.001, proton_density=1.0,
            te_s=0.05, tr_s=10.0,
            spin_count=200, n_repetitions=1,
        )
        assert sig < 1e-10

    def test_two_different_t2_values(self):
        """Shorter T₂ should produce smaller signal at same TE."""
        kwargs = dict(t1_s=10.0, proton_density=1.0,
                      te_s=0.02, tr_s=10.0,
                      spin_count=200, n_repetitions=1)
        sig_long = simulate_single_voxel_bloch(t2_s=0.1, **kwargs)
        sig_short = simulate_single_voxel_bloch(t2_s=0.03, **kwargs)
        assert sig_short < sig_long


# ══════════════════════════════════════════════════════════════════
#  4.  Analytical contrast
# ══════════════════════════════════════════════════════════════════


class TestAnalyticalContrast:
    """Tests for the pure contrast computation (no Bloch sim needed)."""

    def test_output_shape(self, simple_layers):
        depths, contrast, labels = compute_analytical_contrast(
            simple_layers, max_depth_mm=20.0, depth_resolution_mm=1.0,
            te_s=0.01, tr_s=0.1,
        )
        assert len(depths) == len(contrast) == len(labels)
        assert len(depths) == 20

    def test_labels_match_layers(self, simple_layers):
        depths, _, labels = compute_analytical_contrast(
            simple_layers, max_depth_mm=20.0, depth_resolution_mm=1.0,
            te_s=0.01, tr_s=0.1,
        )
        # First 10mm = water, next 10mm = fat
        assert all(l == "water" for l in labels[:10])
        assert all(l == "fat" for l in labels[10:])

    def test_contrast_formula(self):
        """Verify the contrast formula for a single tissue type."""
        layers = [TissueLayer("test", 10.0, proton_density=0.8,
                              t1_ms=500, t2_ms=40)]
        depths, contrast, _ = compute_analytical_contrast(
            layers, max_depth_mm=5.0, depth_resolution_mm=1.0,
            te_s=0.01, tr_s=0.1,
        )
        expected = 0.8 * (1 - math.exp(-0.1 / 0.5)) * math.exp(-0.01 / 0.04)
        assert all(c == pytest.approx(expected) for c in contrast)

    def test_bone_near_zero(self):
        """Bone cortex (T₂ = 0.5 ms) should give ~0 signal at TE = 10 ms."""
        layers = [TissueLayer("bone", 10.0, proton_density=0.05,
                              t1_ms=1000, t2_ms=0.5)]
        _, contrast, _ = compute_analytical_contrast(
            layers, max_depth_mm=5.0, depth_resolution_mm=1.0,
            te_s=0.01, tr_s=0.1,
        )
        assert all(c < 1e-6 for c in contrast)


# ══════════════════════════════════════════════════════════════════
#  5.  Depth Bloch simulation
# ══════════════════════════════════════════════════════════════════


class TestSimulateDepthBloch:
    """Tests for the full depth-profile Bloch simulation."""

    def test_result_type(self, simple_layers):
        result = simulate_depth_bloch(
            simple_layers,
            max_depth_mm=10.0, depth_resolution_mm=2.0,
            te_ms=10.0, tr_ms=100.0,
            spin_count=50, n_repetitions=5,
        )
        assert isinstance(result, BlochSignal)

    def test_depth_grid(self, simple_layers):
        result = simulate_depth_bloch(
            simple_layers,
            max_depth_mm=10.0, depth_resolution_mm=2.0,
            te_ms=10.0, tr_ms=100.0,
            spin_count=50, n_repetitions=5,
        )
        expected_depths = np.arange(1.0, 10.0, 2.0)
        np.testing.assert_allclose(result.depths_mm, expected_depths)

    def test_signal_positive(self, simple_layers):
        result = simulate_depth_bloch(
            simple_layers,
            max_depth_mm=10.0, depth_resolution_mm=2.0,
            te_ms=10.0, tr_ms=100.0,
            spin_count=50, n_repetitions=5,
        )
        assert all(s >= 0 for s in result.signal_magnitude)

    def test_bone_layer_low_signal(self):
        """At typical TE, bone cortex (T₂ = 0.5 ms) → negligible signal."""
        layers = [
            TissueLayer("muscle", 5.0, 1.0, t1_ms=600, t2_ms=35),
            TissueLayer("bone", 5.0, 0.05, t1_ms=1000, t2_ms=0.5),
        ]
        result = simulate_depth_bloch(
            layers,
            max_depth_mm=10.0, depth_resolution_mm=2.0,
            te_ms=10.0, tr_ms=100.0,
            spin_count=100, n_repetitions=5,
        )
        # Bone entries (depth > 5mm) should be near zero
        bone_mask = np.array(result.depths_mm) > 5.0
        assert all(s < 1e-6 for s in result.signal_magnitude[bone_mask])


# ══════════════════════════════════════════════════════════════════
#  6.  Cross-validation
# ══════════════════════════════════════════════════════════════════


class TestCrossValidation:
    """Tests for analytical ↔ Bloch cross-validation."""

    def test_contrast_correlation_simple(self, simple_layers):
        """Correlation between analytical contrast and Bloch should be > 0.99."""
        result = cross_validate_contrast(
            simple_layers,
            max_depth_mm=20.0, depth_resolution_mm=2.0,
            te_ms=10.0, tr_ms=100.0,
            spin_count=200, n_repetitions=15,
        )
        assert isinstance(result, DepthValidation)
        assert result.correlation > 0.99

    def test_contrast_relative_error(self, simple_layers):
        """Mean relative error should be < 5 %."""
        result = cross_validate_contrast(
            simple_layers,
            max_depth_mm=20.0, depth_resolution_mm=2.0,
            te_ms=10.0, tr_ms=100.0,
            spin_count=200, n_repetitions=15,
        )
        assert result.mean_relative_error < 0.05

    def test_forearm_layered_contrast(self):
        """Cross-validate the forearm tissue stack (coarse grid, fast)."""
        result = cross_validate_contrast(
            FOREARM_LAYERS,
            max_depth_mm=20.0, depth_resolution_mm=5.0,
            te_ms=10.0, tr_ms=100.0,
            spin_count=100, n_repetitions=10,
        )
        assert result.correlation > 0.98

    def test_cross_validate_depth_api(self):
        """Verify the DepthProfile ↔ BlochSignal comparison API."""
        from nv_maser.physics.depth_profile import DepthProfile

        depths = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        analytical = DepthProfile(
            depths_mm=depths,
            b0_tesla=np.full(5, 0.05),
            larmor_mhz=np.full(5, 2.13),
            signal=np.array([1.0, 0.9, 0.8, 0.1, 0.05]),
            noise_v=0.01,
            snr=np.array([100, 90, 80, 10, 5]),
            tissue_labels=["a", "a", "a", "b", "b"],
            scan_time_s=10.0,
        )
        bloch = BlochSignal(
            depths_mm=depths,
            signal_magnitude=np.array([0.5, 0.45, 0.4, 0.05, 0.025]),
            tissue_labels=["a", "a", "a", "b", "b"],
            te_s=0.01,
            tr_s=0.1,
            spin_count=100,
        )
        result = cross_validate_depth(analytical, bloch)
        assert isinstance(result, DepthValidation)
        # Same shape (scaled by 2x) → perfect correlation
        assert result.correlation > 0.999

    def test_depth_grid_mismatch_raises(self):
        """Different depth grids should raise ValueError."""
        from nv_maser.physics.depth_profile import DepthProfile

        analytical = DepthProfile(
            depths_mm=np.array([1.0, 2.0]),
            b0_tesla=np.zeros(2),
            larmor_mhz=np.zeros(2),
            signal=np.ones(2),
            noise_v=0.01,
            snr=np.ones(2),
            tissue_labels=["a", "a"],
            scan_time_s=1.0,
        )
        bloch = BlochSignal(
            depths_mm=np.array([1.0, 2.0, 3.0]),
            signal_magnitude=np.ones(3),
            tissue_labels=["a", "a", "a"],
            te_s=0.01,
            tr_s=0.1,
            spin_count=100,
        )
        with pytest.raises(ValueError, match="mismatch"):
            cross_validate_depth(analytical, bloch)
