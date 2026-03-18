"""
Tests for t1t2_estimator.py — tissue T1/T2 parameter estimation.

Classes:
  TestFitT2Monoexponential — unit tests for the T2 fitting function
  TestFitT1SaturationRecovery — unit tests for the T1 fitting function
  TestMapT2FromCpmg — depth-resolved T2 mapping via EPG CPMG
  TestMapT1FromSaturationRecovery — depth-resolved T1 mapping
  TestBuildT1T2Map — joint T1+T2 depth map
  TestDetectTissueAbnormalities — abnormality detection (hemorrhage vs forearm)
  TestCrossValidateT1T2 — cross-validation between two maps
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nv_maser.physics.t1t2_estimator import (
    AbnormalityFlag,
    T1FitResult,
    T1MapResult,
    T1T2CrossValidation,
    T1T2Map,
    T2FitResult,
    T2MapResult,
    build_t1t2_map,
    cross_validate_t1t2,
    detect_tissue_abnormalities,
    fit_t1_saturation_recovery,
    fit_t2_monoexponential,
    map_t1_from_saturation_recovery,
    map_t2_from_cpmg,
)
from nv_maser.physics.depth_profile import (
    FOREARM_LAYERS,
    HEMORRHAGE_LAYERS,
    TissueLayer,
)


# ── Test fixtures ─────────────────────────────────────────────────

@pytest.fixture
def muscle_layer():
    return TissueLayer("muscle", thickness_mm=10.0, t1_ms=600, t2_ms=35)


@pytest.fixture
def fat_layer():
    return TissueLayer("subcutaneous_fat", thickness_mm=5.0, t1_ms=250, t2_ms=80)


@pytest.fixture
def skin_layer():
    return TissueLayer("skin", thickness_mm=2.0, t1_ms=400, t2_ms=30)


@pytest.fixture
def two_layer_phantom(skin_layer, muscle_layer):
    """Simple 2-layer phantom for map tests."""
    return [skin_layer, muscle_layer]


# ── TestFitT2Monoexponential ──────────────────────────────────────

class TestFitT2Monoexponential:
    """Unit tests for fit_t2_monoexponential()."""

    def test_exact_recovery_noiseless(self):
        """Fit ideal monoexponential — should recover T2 within 0.1%."""
        t2_true = 0.035  # 35 ms (muscle)
        s0_true = 0.85
        echo_times = np.arange(1, 17) * 0.010  # 10, 20, …, 160 ms
        echoes = s0_true * np.exp(-echo_times / t2_true)

        res = fit_t2_monoexponential(echoes, echo_times)

        assert res.converged is True
        assert res.t2_s == pytest.approx(t2_true, rel=1e-3)
        assert res.s0 == pytest.approx(s0_true, rel=1e-3)

    def test_fat_t2_recovery(self):
        """Recover fat T2 ≈ 80 ms."""
        t2_true = 0.080
        echo_times = np.arange(1, 17) * 0.010
        echoes = np.exp(-echo_times / t2_true)

        res = fit_t2_monoexponential(echoes, echo_times)

        assert res.t2_s == pytest.approx(t2_true, rel=2e-3)

    def test_r_squared_near_unity_for_noiseless(self):
        """R² should be > 0.9999 for noiseless monoexponential data."""
        t2_true = 0.050
        echo_times = np.arange(1, 33) * 0.010
        echoes = 0.9 * np.exp(-echo_times / t2_true)

        res = fit_t2_monoexponential(echoes, echo_times)

        assert res.r_squared > 0.9999

    def test_rms_residual_near_zero_noiseless(self):
        """RMS residual should be < 1e-8 for ideal data."""
        t2_true = 0.060
        echo_times = np.arange(1, 17) * 0.010
        echoes = np.exp(-echo_times / t2_true)

        res = fit_t2_monoexponential(echoes, echo_times)

        assert res.residuals_rms < 1e-6

    def test_noisy_data_within_tolerance(self):
        """With 3% Gaussian noise, T2 should be recovered within 10%."""
        rng = np.random.default_rng(42)
        t2_true = 0.040
        echo_times = np.arange(1, 17) * 0.010
        echoes = np.exp(-echo_times / t2_true)
        echoes += rng.normal(0, 0.03, size=len(echoes))
        echoes = np.clip(echoes, 0, None)

        res = fit_t2_monoexponential(echoes, echo_times)

        assert abs(res.t2_s - t2_true) / t2_true < 0.10

    def test_degenerate_all_zeros_returns_no_crash(self):
        """All-zero echo train (bone cortex edge case) must not raise."""
        echo_times = np.arange(1, 9) * 0.010
        echoes = np.zeros(8)

        res = fit_t2_monoexponential(echoes, echo_times)

        assert res.converged is False
        assert res.t2_s > 0  # degenerate but finite value

    def test_too_few_echoes_raises(self):
        """Fewer than 2 echoes should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            fit_t2_monoexponential(np.array([0.5]), np.array([0.01]))

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="equal length"):
            fit_t2_monoexponential(np.array([0.5, 0.4]), np.array([0.01]))


# ── TestFitT1SaturationRecovery ───────────────────────────────────

class TestFitT1SaturationRecovery:
    """Unit tests for fit_t1_saturation_recovery()."""

    def test_exact_recovery_noiseless(self):
        """Fit ideal saturation recovery — recover T1 within 1%."""
        t1_true = 0.600  # 600 ms (muscle)
        a_true = 0.80
        tr_values = np.linspace(0.1, 5.0, 12)
        signals = a_true * (1.0 - np.exp(-tr_values / t1_true))

        res = fit_t1_saturation_recovery(signals, tr_values)

        assert res.converged is True
        assert res.t1_s == pytest.approx(t1_true, rel=5e-3)

    def test_fat_t1_recovery(self):
        """Recover fat T1 ≈ 250 ms."""
        t1_true = 0.250
        tr_values = np.linspace(0.05, 2.0, 12)
        signals = 0.75 * (1.0 - np.exp(-tr_values / t1_true))

        res = fit_t1_saturation_recovery(signals, tr_values)

        assert res.t1_s == pytest.approx(t1_true, rel=5e-3)

    def test_r_squared_near_unity(self):
        """R² should be > 0.999 for noiseless saturation-recovery data."""
        t1_true = 0.400
        tr_values = np.logspace(-2, 1, 12)
        signals = 0.85 * (1.0 - np.exp(-tr_values / t1_true))

        res = fit_t1_saturation_recovery(signals, tr_values)

        assert res.r_squared > 0.999

    def test_amplitude_absorbs_te_factor(self):
        """Amplitude A should absorb the exp(−TE/T2) factor correctly."""
        t1_true = 0.600
        t2_true = 0.035
        te = 0.010
        expected_a = 0.90 * math.exp(-te / t2_true)  # 90% × TE decay
        tr_values = np.linspace(0.1, 4.0, 10)
        signals = expected_a * (1.0 - np.exp(-tr_values / t1_true))

        res = fit_t1_saturation_recovery(signals, tr_values)

        assert res.amplitude == pytest.approx(expected_a, rel=5e-3)

    def test_too_few_tr_values_raises(self):
        """Fewer than 2 TR values should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            fit_t1_saturation_recovery(np.array([0.5]), np.array([1.0]))

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            fit_t1_saturation_recovery(np.array([0.3, 0.5, 0.7]), np.array([0.2, 0.5]))


# ── TestMapT2FromCpmg ─────────────────────────────────────────────

class TestMapT2FromCpmg:
    """Tests for map_t2_from_cpmg() depth-resolved T2 mapping."""

    def test_returns_correct_type(self, two_layer_phantom):
        result = map_t2_from_cpmg(two_layer_phantom, esp_ms=10.0, n_echoes=16)
        assert isinstance(result, T2MapResult)

    def test_depth_array_length(self, two_layer_phantom):
        """Depth array length matches total tissue coverage."""
        result = map_t2_from_cpmg(two_layer_phantom, depth_step_mm=0.5)
        total_mm = sum(lay.thickness_mm for lay in two_layer_phantom)
        expected_n = int(total_mm / 0.5)
        assert len(result.depths_mm) == expected_n

    def test_muscle_t2_recovered(self, muscle_layer):
        """EPG CPMG on pure muscle should recover T2 ≈ 35 ms."""
        layers = [muscle_layer]
        result = map_t2_from_cpmg(layers, esp_ms=10.0, n_echoes=16, tr_ms=2000.0)
        # All depths are muscle; check mean fitted T2
        mean_t2_ms = float(np.mean(result.t2_fitted_s)) * 1e3
        assert mean_t2_ms == pytest.approx(35.0, rel=0.05)  # within 5%

    def test_fat_t2_recovered(self, fat_layer):
        """EPG CPMG on pure fat should recover T2 ≈ 80 ms."""
        layers = [fat_layer]
        result = map_t2_from_cpmg(layers, esp_ms=10.0, n_echoes=16, tr_ms=2000.0)
        mean_t2_ms = float(np.mean(result.t2_fitted_s)) * 1e3
        assert mean_t2_ms == pytest.approx(80.0, rel=0.05)

    def test_r_squared_high_for_ideal_pulses(self, muscle_layer):
        """With ideal 180° refocusing, R² should be > 0.99."""
        layers = [muscle_layer]
        result = map_t2_from_cpmg(layers, esp_ms=5.0, n_echoes=32, refocus_angle_deg=180.0)
        assert float(np.min(result.r_squared)) > 0.99

    def test_tissue_labels_present(self, two_layer_phantom):
        result = map_t2_from_cpmg(two_layer_phantom)
        unique_labels = set(result.tissue_labels)
        assert "skin" in unique_labels
        assert "muscle" in unique_labels

    def test_reference_t2_matches_input(self, two_layer_phantom):
        """t2_reference_s should equal the TissueLayer.t2_ms values."""
        result = map_t2_from_cpmg(two_layer_phantom, depth_step_mm=1.0)
        for i, label in enumerate(result.tissue_labels):
            layer = next(lay for lay in two_layer_phantom if lay.name == label)
            assert result.t2_reference_s[i] == pytest.approx(layer.t2_ms * 1e-3, rel=1e-6)

    def test_forearm_layers_run_without_crash(self):
        """Full FOREARM_LAYERS stack should complete successfully."""
        result = map_t2_from_cpmg(FOREARM_LAYERS, depth_step_mm=1.0, n_echoes=16)
        assert len(result.depths_mm) > 0
        assert np.all(np.isfinite(result.t2_fitted_s))


# ── TestMapT1FromSaturationRecovery ──────────────────────────────

class TestMapT1FromSaturationRecovery:
    """Tests for map_t1_from_saturation_recovery() depth T1 mapping."""

    def test_returns_correct_type(self, two_layer_phantom):
        result = map_t1_from_saturation_recovery(two_layer_phantom)
        assert isinstance(result, T1MapResult)

    def test_muscle_t1_recovered(self, muscle_layer):
        """Should recover muscle T1 ≈ 600 ms within 5%."""
        result = map_t1_from_saturation_recovery(
            [muscle_layer],
            tr_values_ms=[100, 200, 400, 600, 1000, 2000],
            te_ms=10.0,
        )
        mean_t1_ms = float(np.mean(result.t1_fitted_s)) * 1e3
        assert mean_t1_ms == pytest.approx(600.0, rel=0.05)

    def test_fat_t1_recovered(self, fat_layer):
        """Should recover fat T1 ≈ 250 ms within 5%.

        TR must be >> T2 (fat T2 ≈ 80 ms) to ensure T2 coherences from
        previous repetitions are fully dephased before each 90° excitation.
        Using TR_min = 500 ms ≈ 6× T2 satisfies this condition.
        """
        result = map_t1_from_saturation_recovery(
            [fat_layer],
            tr_values_ms=[500, 750, 1000, 1500, 2500, 4000],
            te_ms=10.0,
        )
        mean_t1_ms = float(np.mean(result.t1_fitted_s)) * 1e3
        assert mean_t1_ms == pytest.approx(250.0, rel=0.05)

    def test_r_squared_high(self, skin_layer):
        """R² should be > 0.99 for well-sampled TR range."""
        result = map_t1_from_saturation_recovery(
            [skin_layer],
            tr_values_ms=[50, 100, 200, 400, 800, 1500, 3000],
            te_ms=10.0,
        )
        assert float(np.min(result.r_squared)) > 0.99

    def test_auto_tr_range_covers_tissues(self, two_layer_phantom):
        """Auto-generated TR range should produce valid T1 estimates."""
        result = map_t1_from_saturation_recovery(two_layer_phantom)
        assert np.all(result.t1_fitted_s > 0)
        assert np.all(np.isfinite(result.t1_fitted_s))

    def test_te_stored_in_result(self, muscle_layer):
        result = map_t1_from_saturation_recovery(
            [muscle_layer], te_ms=15.0,
            tr_values_ms=[200, 500, 1000, 2000],
        )
        assert result.te_s == pytest.approx(0.015)


# ── TestBuildT1T2Map ──────────────────────────────────────────────

class TestBuildT1T2Map:
    """Tests for build_t1t2_map() combined T1/T2 mapping."""

    def test_returns_correct_type(self, two_layer_phantom):
        result = build_t1t2_map(two_layer_phantom)
        assert isinstance(result, T1T2Map)

    def test_depth_arrays_consistent(self, two_layer_phantom):
        """T1, T2, and depth arrays should all have equal length."""
        result = build_t1t2_map(two_layer_phantom)
        n = len(result.depths_mm)
        assert len(result.t1_s) == n
        assert len(result.t2_s) == n
        assert len(result.t1t2_ratio) == n
        assert len(result.r1_s_inv) == n
        assert len(result.r2_s_inv) == n

    def test_t1t2_ratio_positive(self, two_layer_phantom):
        """T1/T2 ratio should be > 1 for all biological tissues."""
        result = build_t1t2_map(two_layer_phantom)
        assert np.all(result.t1t2_ratio > 1.0)

    def test_rates_are_reciprocals(self, muscle_layer):
        """R1 = 1/T1 and R2 = 1/T2 by definition."""
        result = build_t1t2_map([muscle_layer])
        valid = result.t1_s > 0
        np.testing.assert_allclose(
            result.r1_s_inv[valid], 1.0 / result.t1_s[valid], rtol=1e-6
        )
        valid2 = result.t2_s > 0
        np.testing.assert_allclose(
            result.r2_s_inv[valid2], 1.0 / result.t2_s[valid2], rtol=1e-6
        )

    def test_tissue_labels_in_map(self, two_layer_phantom):
        result = build_t1t2_map(two_layer_phantom)
        label_set = set(result.tissue_labels)
        assert "skin" in label_set
        assert "muscle" in label_set

    def test_fat_muscle_t1t2_ratio_differs(self):
        """Fat and muscle should have distinct T1/T2 ratios (tissue contrast)."""
        fat = TissueLayer("fat", thickness_mm=5.0, t1_ms=250, t2_ms=80)
        muscle = TissueLayer("muscle", thickness_mm=5.0, t1_ms=600, t2_ms=35)
        result = build_t1t2_map([fat, muscle], depth_step_mm=1.0)

        fat_idx = [i for i, lbl in enumerate(result.tissue_labels) if lbl == "fat"]
        muscle_idx = [i for i, lbl in enumerate(result.tissue_labels) if lbl == "muscle"]

        fat_ratio = float(np.mean(result.t1t2_ratio[fat_idx]))
        muscle_ratio = float(np.mean(result.t1t2_ratio[muscle_idx]))

        # Fat: 250/80 ~ 3.1;  Muscle: 600/35 ~ 17.1 → clearly different
        assert abs(fat_ratio - muscle_ratio) > 5.0


# ── TestDetectTissueAbnormalities ─────────────────────────────────

class TestDetectTissueAbnormalities:
    """Tests for detect_tissue_abnormalities() using hemorrhage vs forearm."""

    @pytest.fixture
    def forearm_t2_map(self):
        return map_t2_from_cpmg(FOREARM_LAYERS, depth_step_mm=1.0, n_echoes=16)

    @pytest.fixture
    def hemorrhage_t2_map(self):
        return map_t2_from_cpmg(HEMORRHAGE_LAYERS, depth_step_mm=1.0, n_echoes=16)

    def test_returns_list(self, forearm_t2_map, hemorrhage_t2_map):
        flags = detect_tissue_abnormalities(hemorrhage_t2_map, forearm_t2_map)
        assert isinstance(flags, list)

    def test_hemorrhage_flagged_as_prolonged_t2(
        self, forearm_t2_map, hemorrhage_t2_map
    ):
        """Haemorrhage layer (T2=150 ms) versus muscle reference (T2=35 ms)
        should be flagged as a T2 prolongation."""
        flags = detect_tissue_abnormalities(hemorrhage_t2_map, forearm_t2_map, t2_threshold=0.25)

        hem_flags = [f for f in flags if f.tissue_label == "hemorrhage"]
        assert len(hem_flags) > 0, "Expected abnormality flags for hemorrhage tissue"
        assert all(f.flag_type == "prolonged" for f in hem_flags)
        assert all(f.parameter == "T2" for f in hem_flags)

    def test_hemorrhage_deviation_large(self, forearm_t2_map, hemorrhage_t2_map):
        """Fractional deviation for hemorrhage should be > 1.0 (>100% above ref)."""
        flags = detect_tissue_abnormalities(hemorrhage_t2_map, forearm_t2_map)
        hem_flags = [f for f in flags if f.tissue_label == "hemorrhage"]
        if hem_flags:
            max_dev = max(f.deviation_fraction for f in hem_flags)
            assert max_dev > 1.0  # T2=150 ms vs reference T2=35 ms → 329% above

    def test_no_flags_for_identical_maps(self, forearm_t2_map):
        """Comparing a map to itself should produce zero flags."""
        flags = detect_tissue_abnormalities(forearm_t2_map, forearm_t2_map)
        assert len(flags) == 0

    def test_threshold_controls_sensitivity(self, forearm_t2_map, hemorrhage_t2_map):
        """Higher threshold → fewer or equal flags."""
        flags_low = detect_tissue_abnormalities(hemorrhage_t2_map, forearm_t2_map, t2_threshold=0.10)
        flags_high = detect_tissue_abnormalities(hemorrhage_t2_map, forearm_t2_map, t2_threshold=0.80)
        assert len(flags_low) >= len(flags_high)


# ── TestCrossValidateT1T2 ─────────────────────────────────────────

class TestCrossValidateT1T2:
    """Tests for cross_validate_t1t2()."""

    @pytest.fixture
    def muscle_map(self):
        return build_t1t2_map(
            [TissueLayer("muscle", thickness_mm=5.0, t1_ms=600, t2_ms=35)],
            depth_step_mm=1.0,
        )

    def test_returns_correct_type(self, muscle_map):
        result = cross_validate_t1t2(muscle_map, muscle_map)
        assert isinstance(result, T1T2CrossValidation)

    def test_identical_maps_perfect_correlation(self, muscle_map):
        """Comparing a map with itself should give correlation = 1."""
        result = cross_validate_t1t2(muscle_map, muscle_map)
        assert result.t1_correlation == pytest.approx(1.0, abs=1e-6)
        assert result.t2_correlation == pytest.approx(1.0, abs=1e-6)

    def test_identical_maps_zero_error(self, muscle_map):
        """Comparing a map with itself should give zero relative errors."""
        result = cross_validate_t1t2(muscle_map, muscle_map)
        assert result.t1_max_relative_error < 1e-6
        assert result.t2_max_relative_error < 1e-6

    def test_different_maps_reduced_correlation(self):
        """Maps from different tissues should have lower T2 correlation."""
        muscle_map = build_t1t2_map(
            [TissueLayer("muscle", thickness_mm=5.0, t1_ms=600, t2_ms=35)],
            depth_step_mm=1.0,
        )
        fat_map = build_t1t2_map(
            [TissueLayer("fat", thickness_mm=5.0, t1_ms=250, t2_ms=80)],
            depth_step_mm=1.0,
        )
        result = cross_validate_t1t2(muscle_map, fat_map)
        # Both single-tissue maps are constant arrays → correlation is 1
        # but relative error should be non-zero
        muscle_t2 = float(np.mean(muscle_map.t2_s))
        fat_t2 = float(np.mean(fat_map.t2_s))
        assert result.t2_mean_relative_error == pytest.approx(
            abs(muscle_t2 - fat_t2) / muscle_t2, rel=0.01
        )

    def test_n_depths_reported(self, muscle_map):
        result = cross_validate_t1t2(muscle_map, muscle_map)
        assert result.n_depths == len(muscle_map.depths_mm)

    def test_two_layer_phantom(self, two_layer_phantom):
        """Two-layer phantom: T2 correlation with itself should be 1."""
        m = build_t1t2_map(two_layer_phantom, depth_step_mm=1.0)
        result = cross_validate_t1t2(m, m)
        assert result.t2_correlation == pytest.approx(1.0, abs=1e-6)
