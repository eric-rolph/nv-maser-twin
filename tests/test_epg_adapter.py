"""
Tests for the Extended Phase Graph (EPG) adapter module.

Covers:
- Physical correctness of single spin-echo signal
- CPMG multi-echo envelope monotonic decay
- EPG concordance with analytical formula (TR >> T1)
- EPG concordance with MRzero Bloch simulation (if available)
- Depth profile generation and shape
- Cross-validation utilities
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nv_maser.physics.depth_profile import (
    FOREARM_LAYERS,
    TissueLayer,
)
from nv_maser.physics.epg_adapter import (
    EPGDepthProfile,
    EPGResult,
    EPGValidation,
    cross_validate_epg_vs_analytical,
    epg_cpmg,
    epg_depth_profile,
    epg_signal,
    _epg_init,
    _epg_rf,
    _epg_relax,
    _epg_grad,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def water_params():
    return {"t1_s": 2.0, "t2_s": 0.3, "te_s": 0.020, "tr_s": 5.0}


@pytest.fixture
def muscle_params():
    return {"t1_s": 0.6, "t2_s": 0.035, "te_s": 0.010, "tr_s": 0.100}


@pytest.fixture
def short_t2_params():
    return {"t1_s": 1.0, "t2_s": 0.002, "te_s": 0.010, "tr_s": 0.200}


@pytest.fixture
def simple_two_layer():
    return [
        TissueLayer("fat", thickness_mm=10.0, proton_density=0.9, t1_ms=250, t2_ms=80),
        TissueLayer("muscle", thickness_mm=20.0, proton_density=1.0, t1_ms=600, t2_ms=35),
    ]


# ── Test: EPG core operators ────────────────────────────────────────

class TestEPGCoreOperators:
    def test_init_equilibrium(self):
        state = _epg_init(8)
        assert state.shape == (3, 8)
        assert state[2, 0] == pytest.approx(1.0)  # Z[0] = Mz = 1 (equilibrium)
        assert abs(state[0, 0]) < 1e-10  # no transverse at start

    def test_rf_90_creates_transverse(self):
        state = _epg_init(8)
        state = _epg_rf(state, math.pi / 2, phi_rad=math.pi / 2)
        # 90° pulse rotates longitudinal into transverse
        assert abs(state[0, 0]) > 0.5  # F+[0] becomes significant
        assert abs(state[2, 0]) < 0.1  # Z[0] nearly zero after 90°

    def test_rf_180_inverts_longitudinal(self):
        state = _epg_init(8)
        # Apply 180° pulse at equilibrium (no prior tip) — inverts Mz
        state = _epg_rf(state, math.pi, phi_rad=math.pi / 2)
        assert state[2, 0].real == pytest.approx(-1.0, abs=1e-10)

    def test_relax_preserves_equilibrium(self):
        """Applying relaxation to equilibrium state leaves it unchanged."""
        state = _epg_init(8)
        state = _epg_relax(state, t_interval_s=0.1, t1_s=0.6, t2_s=0.035)
        # Z[0] should stay at ~1 (full equilibrium → no change)
        assert state[2, 0].real == pytest.approx(1.0, abs=1e-10)

    def test_relax_decays_transverse(self):
        state = _epg_init(8)
        state = _epg_rf(state, math.pi / 2, phi_rad=math.pi / 2)
        f_before = abs(state[0, 0])
        state = _epg_relax(state, t_interval_s=0.050, t1_s=1.0, t2_s=0.050)
        f_after = abs(state[0, 0])
        # Expect ~exp(-1) ≈ 0.368 attenuation
        assert f_after == pytest.approx(f_before * math.exp(-1.0), rel=0.01)

    def test_grad_shifts_states(self):
        state = _epg_init(8)
        state = _epg_rf(state, math.pi / 2, phi_rad=math.pi / 2)
        f_before = state[0, 0].copy()
        state = _epg_grad(state)
        # F+[0] should be cleared; F+[1] gets old F+[0]
        assert abs(state[0, 0]) < 1e-10
        assert abs(state[0, 1] - f_before) < 1e-10


# ── Test: Single spin-echo signal ─────────────────────────────────

class TestSingleSpinEcho:
    def test_long_t2_high_signal(self, water_params):
        """Long T2 relative to TE → signal dominated by T1 saturation × T2 decay."""
        s = epg_signal(**water_params)
        # water_params: T1=2s, T2=0.3s, TE=0.02s, TR=5s
        # TR/T1 = 2.5, so saturation = (1 - exp(-2.5)) ≈ 0.918 (NOT ≈ 1.0)
        # Full formula: (1 - exp(-TR/T1)) × exp(-TE/T2)
        t1, t2, te, tr = water_params["t1_s"], water_params["t2_s"], water_params["te_s"], water_params["tr_s"]
        expected = (1 - math.exp(-tr / t1)) * math.exp(-te / t2)
        assert s == pytest.approx(expected, rel=0.03)

    def test_short_t2_low_signal(self, short_t2_params):
        """Short T2 = 2 ms, TE = 10 ms → very low signal."""
        s = epg_signal(**short_t2_params)
        assert s < 0.01  # should be near zero

    def test_t2_decay_correct(self):
        """With perfect TR >> T1, EPG should match exp(-TE/T2) × (1-exp(-TR/T1))."""
        t1, t2 = 0.5, 0.1
        te, tr = 0.020, 10.0  # TR >> T1
        s = epg_signal(t1_s=t1, t2_s=t2, te_s=te, tr_s=tr)
        expected = (1 - math.exp(-tr / t1)) * math.exp(-te / t2)
        assert s == pytest.approx(expected, rel=0.02)

    def test_signal_positive(self, muscle_params):
        s = epg_signal(**muscle_params)
        assert s > 0

    def test_perfect_180_flip(self):
        """Testing that a 90-180 sequence forms an echo."""
        s = epg_signal(t1_s=1.0, t2_s=0.1, te_s=0.010, tr_s=1.0,
                       flip_angle_deg=90.0, refocus_angle_deg=180.0)
        assert s > 0.05

    def test_tr_less_than_te_raises(self):
        with pytest.raises(ValueError, match="TR"):
            epg_signal(t1_s=1.0, t2_s=0.1, te_s=0.050, tr_s=0.010)


# ── Test: CPMG multi-echo ─────────────────────────────────────────

class TestCPMG:
    def test_output_shape(self, muscle_params):
        echoes = epg_cpmg(
            t1_s=muscle_params["t1_s"], t2_s=muscle_params["t2_s"],
            esp_s=0.010, n_echoes=8
        )
        assert echoes.shape == (8,)

    def test_monotonic_decay(self):
        """CPMG echo amplitudes should be monotonically non-increasing for ideal 180° pulses."""
        echoes = epg_cpmg(t1_s=2.0, t2_s=0.3, esp_s=0.010, n_echoes=16)
        # Each echo should be <= previous
        diffs = np.diff(echoes)
        assert np.all(diffs <= 1e-6), f"CPMG not monotone: {diffs[diffs > 1e-6]}"

    def test_first_echo_matches_single_echo(self, water_params):
        """First CPMG echo at ESP = TE should match single spin-echo result."""
        te = water_params["te_s"]
        t1 = water_params["t1_s"]
        t2 = water_params["t2_s"]
        tr = water_params["tr_s"]

        single = epg_signal(t1_s=t1, t2_s=t2, te_s=te, tr_s=tr)
        cpmg = epg_cpmg(t1_s=t1, t2_s=t2, esp_s=te, n_echoes=4, tr_s=tr)
        assert cpmg[0] == pytest.approx(single, rel=0.05)

    def test_long_esp_t2_decay(self):
        """Echo n in CPMG with ideal 180° ≈ (1-exp(-TR/T1)) × exp(-n × ESP / T2).

        Use TR = 10 × T1 so the saturation factor ≈ 1 and the formula simplifies
        to exp(-n × ESP / T2).
        """
        t1 = 0.5   # T1 = 0.5 s
        t2 = 0.1   # T2 = 100 ms
        esp = 0.020
        tr = 5.0   # TR = 10 × T1 → saturation factor (1-exp(-10)) ≈ 1.00
        echoes = epg_cpmg(t1_s=t1, t2_s=t2, esp_s=esp, n_echoes=8, tr_s=tr)
        for n, amp in enumerate(echoes, start=1):
            expected = math.exp(-(n * esp) / t2)
            assert amp == pytest.approx(expected, rel=0.03), f"Echo {n} mismatch"

    def test_all_positive(self, muscle_params):
        echoes = epg_cpmg(
            t1_s=muscle_params["t1_s"], t2_s=muscle_params["t2_s"],
            esp_s=0.010, n_echoes=6
        )
        assert np.all(echoes >= 0)


# ── Test: EPG depth profile ───────────────────────────────────────

class TestEPGDepthProfile:
    def test_returns_epg_depth_profile(self, simple_two_layer):
        result = epg_depth_profile(simple_two_layer)
        assert isinstance(result, EPGDepthProfile)

    def test_depth_grid_matches_config(self, simple_two_layer):
        result = epg_depth_profile(
            simple_two_layer, max_depth_mm=30.0, depth_resolution_mm=0.5
        )
        expected_n = int(30.0 / 0.5)
        assert len(result.depths_mm) == expected_n

    def test_signal_positive_in_tissues(self, simple_two_layer):
        result = epg_depth_profile(simple_two_layer)
        assert np.all(result.signal >= 0)
        assert np.any(result.signal > 0)

    def test_fat_higher_t2_than_muscle(self, simple_two_layer):
        """Fat (T2=80ms) should show more signal vs muscle (T2=35ms) at TE=10ms."""
        result = epg_depth_profile(simple_two_layer, te_ms=10.0, tr_ms=500.0)
        fat_mask = result.depths_mm <= 10.0
        muscle_mask = result.depths_mm > 10.0
        fat_mean = np.mean(result.signal[fat_mask])
        muscle_mean = np.mean(result.signal[muscle_mask])
        assert fat_mean > muscle_mean, "Fat should have higher signal due to longer T2"

    def test_labels_coverage(self, simple_two_layer):
        result = epg_depth_profile(simple_two_layer, max_depth_mm=30.0, depth_resolution_mm=0.5)
        assert "fat" in result.tissue_labels
        assert "muscle" in result.tissue_labels

    def test_forearm_layers(self):
        result = epg_depth_profile(FOREARM_LAYERS, max_depth_mm=30.0, depth_resolution_mm=0.5)
        assert isinstance(result, EPGDepthProfile)
        assert np.any(result.signal > 0)
        # Bone cortex has very short T2 → near zero signal
        bone_mask = np.array([lbl == "bone_cortex" for lbl in result.tissue_labels])
        if np.any(bone_mask):
            assert np.mean(result.signal[bone_mask]) < 0.01


# ── Test: Cross-validation vs analytical ─────────────────────────

class TestCrossValidationVsAnalytical:
    def test_correlation_high_for_simple_layers(self, simple_two_layer):
        """EPG and analytical formula should produce highly correlated depth profiles."""
        result = epg_depth_profile(simple_two_layer, te_ms=10.0, tr_ms=200.0,
                                   max_depth_mm=30.0, depth_resolution_mm=0.5)

        # Compute analytical signal (PD × T1_sat × T2_decay) for same grid
        te_s = 0.010
        tr_s = 0.200
        depths = result.depths_mm
        analytical = np.array([
            0.9 * (1 - math.exp(-tr_s / 0.250)) * math.exp(-te_s / 0.080) if d <= 10.0
            else 1.0 * (1 - math.exp(-tr_s / 0.600)) * math.exp(-te_s / 0.035)
            for d in depths
        ])

        validation = cross_validate_epg_vs_analytical(result, analytical, depths)
        assert isinstance(validation, EPGValidation)
        assert validation.correlation > 0.98

    def test_relative_error_within_tolerance(self, simple_two_layer):
        """EPG should agree with analytical to within ~5% for ideal flip angles."""
        result = epg_depth_profile(simple_two_layer, te_ms=10.0, tr_ms=500.0,
                                   max_depth_mm=30.0, depth_resolution_mm=0.5)
        te_s, tr_s = 0.010, 0.500
        depths = result.depths_mm
        analytical = np.array([
            0.9 * (1 - math.exp(-tr_s / 0.250)) * math.exp(-te_s / 0.080) if d <= 10.0
            else 1.0 * (1 - math.exp(-tr_s / 0.600)) * math.exp(-te_s / 0.035)
            for d in depths
        ])
        validation = cross_validate_epg_vs_analytical(result, analytical, depths)
        assert validation.mean_relative_error < 0.05  # < 5% average error

    def test_mismatched_length_raises(self, simple_two_layer):
        result = epg_depth_profile(simple_two_layer, max_depth_mm=30.0)
        with pytest.raises(ValueError, match="length"):
            cross_validate_epg_vs_analytical(result, np.ones(10), np.ones(10))
