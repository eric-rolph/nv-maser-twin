"""Tests for pulse_sequence.py — analytical NMR pulse sequence simulator."""
from __future__ import annotations

import math
import pytest
import numpy as np

from nv_maser.physics.pulse_sequence import (
    SpinEchoResult,
    CPMGResult,
    GREResult,
    InversionRecoveryResult,
    SNREfficiency,
    simulate_spin_echo,
    simulate_cpmg,
    simulate_gre,
    simulate_inversion_recovery,
    ernst_angle,
    optimal_te_for_contrast,
    snr_efficiency,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared tissue parameters (typical muscle at low field)
# ─────────────────────────────────────────────────────────────────────────────
T1_MUSCLE = 600.0   # ms (low-field muscle T1)
T2_MUSCLE = 35.0    # ms
T2S_MUSCLE = 25.0   # ms (T2* ~ shorter than T2 due to susceptibility)
T1_FAT = 250.0      # ms
T2_FAT = 80.0       # ms
T1_BLOOD = 900.0    # ms
T2_BLOOD = 150.0    # ms


# ═════════════════════════════════════════════════════════════════════════════
# TestSpinEcho
# ═════════════════════════════════════════════════════════════════════════════


class TestSpinEcho:
    """Unit tests for simulate_spin_echo."""

    def test_returns_spin_echo_result(self):
        r = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, tr_ms=500.0, te_ms=10.0)
        assert isinstance(r, SpinEchoResult)

    def test_signal_is_float(self):
        r = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 500.0, 10.0)
        assert isinstance(r.signal_normalized, float)

    def test_signal_positive(self):
        r = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 500.0, 10.0)
        assert r.signal_normalized > 0

    def test_signal_leq_one(self):
        """Normalised signal must not exceed 1 (bounded by M₀)."""
        r = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 5000.0, 1.0)
        assert r.signal_normalized <= 1.0 + 1e-9

    def test_e1_formula(self):
        r = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 500.0, 10.0)
        assert abs(r.e1 - math.exp(-500.0 / T1_MUSCLE)) < 1e-12

    def test_e2_formula(self):
        r = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 500.0, 10.0)
        assert abs(r.e2 - math.exp(-10.0 / T2_MUSCLE)) < 1e-12

    def test_long_tr_approaches_e2(self):
        """For TR >> T1, saturation factor → 1, signal → E₂."""
        r = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, tr_ms=1e6, te_ms=5.0)
        expected_e2 = math.exp(-5.0 / T2_MUSCLE)
        assert abs(r.signal_normalized - expected_e2) < 1e-4

    def test_long_te_approaches_zero(self):
        r = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, tr_ms=5000.0, te_ms=5 * T2_MUSCLE)
        assert r.signal_normalized < 0.01

    def test_shorter_te_gives_larger_signal(self):
        r1 = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 2000.0, 10.0)
        r2 = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 2000.0, 50.0)
        assert r1.signal_normalized > r2.signal_normalized

    def test_longer_tr_gives_larger_signal(self):
        r1 = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 200.0, 10.0)
        r2 = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 2000.0, 10.0)
        assert r2.signal_normalized > r1.signal_normalized

    def test_flip_angle_90_is_default_and_maximum(self):
        r_90 = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 2000.0, 10.0, flip_angle_deg=90.0)
        r_45 = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 2000.0, 10.0, flip_angle_deg=45.0)
        assert r_90.signal_normalized > r_45.signal_normalized

    def test_flip_angle_zero_gives_zero_signal(self):
        r = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 2000.0, 10.0, flip_angle_deg=0.0)
        assert abs(r.signal_normalized) < 1e-12

    def test_flip_angle_180_gives_zero_signal(self):
        """sin(180°) = 0, so no transverse magnetisation."""
        r = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 2000.0, 10.0, flip_angle_deg=180.0)
        assert abs(r.signal_normalized) < 1e-10

    def test_t2_decay_factor_matches_e2(self):
        r = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 500.0, 10.0)
        assert abs(r.t2_decay_factor - r.e2) < 1e-12

    def test_validation_te_ge_tr_raises(self):
        with pytest.raises(ValueError, match="TE"):
            simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, tr_ms=10.0, te_ms=10.0)

    def test_validation_te_greater_tr_raises(self):
        with pytest.raises(ValueError):
            simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, tr_ms=10.0, te_ms=15.0)

    def test_validation_zero_t1_raises(self):
        with pytest.raises(ValueError):
            simulate_spin_echo(0.0, T2_MUSCLE, 500.0, 10.0)

    def test_validation_negative_t2_raises(self):
        with pytest.raises(ValueError):
            simulate_spin_echo(T1_MUSCLE, -5.0, 500.0, 10.0)

    def test_stored_parameters_preserved(self):
        r = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 600.0, 12.0, 75.0)
        assert r.te_ms == 12.0
        assert r.tr_ms == 600.0
        assert r.flip_angle_deg == 75.0
        assert r.t1_ms == T1_MUSCLE
        assert r.t2_ms == T2_MUSCLE

    def test_high_snr_tissue_fat_vs_muscle(self):
        """Fat & muscle should give different signals (T1/T2 contrast)."""
        r_fat = simulate_spin_echo(T1_FAT, T2_FAT, 500.0, 20.0)
        r_mus = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 500.0, 20.0)
        assert abs(r_fat.signal_normalized - r_mus.signal_normalized) > 0.01

    def test_signal_immutable(self):
        r = simulate_spin_echo(T1_MUSCLE, T2_MUSCLE, 500.0, 10.0)
        with pytest.raises(Exception):
            r.signal_normalized = 99.0  # type: ignore[misc]


# ═════════════════════════════════════════════════════════════════════════════
# TestCPMG
# ═════════════════════════════════════════════════════════════════════════════


class TestCPMG:
    """Unit tests for simulate_cpmg."""

    def test_returns_cpmg_result(self):
        r = simulate_cpmg(T1_MUSCLE, T2_MUSCLE, tr_ms=2000.0, esp_ms=10.0, n_echoes=8)
        assert isinstance(r, CPMGResult)

    def test_n_echoes_match(self):
        r = simulate_cpmg(T1_MUSCLE, T2_MUSCLE, 2000.0, 10.0, 8)
        assert r.n_echoes == 8
        assert len(r.echo_amplitudes) == 8
        assert len(r.echo_times_ms) == 8

    def test_echo_amplitudes_positive(self):
        r = simulate_cpmg(T1_MUSCLE, T2_MUSCLE, 2000.0, 10.0, 8)
        assert all(a > 0 for a in r.echo_amplitudes)

    def test_echo_train_monotonically_decreasing(self):
        r = simulate_cpmg(T1_MUSCLE, T2_MUSCLE, 2000.0, 10.0, 16)
        amps = r.echo_amplitudes
        for i in range(len(amps) - 1):
            assert amps[i] > amps[i + 1], f"Echo {i}: {amps[i]} not > {amps[i+1]}"

    def test_echo_times_match_esp(self):
        esp = 8.0
        r = simulate_cpmg(T1_MUSCLE, T2_MUSCLE, 2000.0, esp, 4)
        expected = [esp, 2 * esp, 3 * esp, 4 * esp]
        for got, exp in zip(r.echo_times_ms, expected):
            assert abs(got - exp) < 1e-9

    def test_custom_te_first(self):
        r = simulate_cpmg(T1_MUSCLE, T2_MUSCLE, 2000.0, esp_ms=10.0, n_echoes=3, te_first_ms=5.0)
        assert abs(r.echo_times_ms[0] - 5.0) < 1e-9
        assert abs(r.echo_times_ms[1] - 15.0) < 1e-9
        assert abs(r.echo_times_ms[2] - 25.0) < 1e-9

    def test_t2_eff_close_to_t2(self):
        """For ideal CPMG, fitted T₂_eff should match input T₂."""
        r = simulate_cpmg(T1_MUSCLE, T2_MUSCLE, tr_ms=1e6, esp_ms=5.0, n_echoes=32)
        assert abs(r.t2_eff_ms - T2_MUSCLE) / T2_MUSCLE < 0.05  # 5% tolerance

    def test_single_echo_case(self):
        r = simulate_cpmg(T1_MUSCLE, T2_MUSCLE, 2000.0, 10.0, 1)
        assert r.n_echoes == 1
        assert len(r.echo_amplitudes) == 1

    def test_validation_n_echoes_zero_raises(self):
        with pytest.raises(ValueError, match="n_echoes"):
            simulate_cpmg(T1_MUSCLE, T2_MUSCLE, 2000.0, 10.0, 0)

    def test_validation_negative_esp_raises(self):
        with pytest.raises(ValueError):
            simulate_cpmg(T1_MUSCLE, T2_MUSCLE, 2000.0, -5.0, 8)

    def test_validation_zero_t2_raises(self):
        with pytest.raises(ValueError):
            simulate_cpmg(T1_MUSCLE, 0.0, 2000.0, 10.0, 8)

    def test_flip_angle_90_maximises_first_echo(self):
        r90 = simulate_cpmg(T1_MUSCLE, T2_MUSCLE, 2000.0, 10.0, 4, flip_angle_deg=90.0)
        r45 = simulate_cpmg(T1_MUSCLE, T2_MUSCLE, 2000.0, 10.0, 4, flip_angle_deg=45.0)
        assert r90.echo_amplitudes[0] > r45.echo_amplitudes[0]

    def test_stored_parameterspreserved(self):
        r = simulate_cpmg(T1_MUSCLE, T2_MUSCLE, 2000.0, 12.0, 6)
        assert r.esp_ms == 12.0
        assert r.t1_ms == T1_MUSCLE
        assert r.t2_ms == T2_MUSCLE


# ═════════════════════════════════════════════════════════════════════════════
# TestGRE
# ═════════════════════════════════════════════════════════════════════════════


class TestGRE:
    """Unit tests for simulate_gre."""

    def test_returns_gre_result(self):
        r = simulate_gre(T1_MUSCLE, T2S_MUSCLE, tr_ms=100.0, te_ms=5.0, flip_angle_deg=30.0)
        assert isinstance(r, GREResult)

    def test_signal_positive(self):
        r = simulate_gre(T1_MUSCLE, T2S_MUSCLE, 100.0, 5.0, 30.0)
        assert r.signal_normalized > 0

    def test_signal_leq_one(self):
        r = simulate_gre(T1_MUSCLE, T2S_MUSCLE, 1e6, 1.0, 90.0)
        assert r.signal_normalized <= 1.0 + 1e-9

    def test_ernst_angle_formula(self):
        r = simulate_gre(T1_MUSCLE, T2S_MUSCLE, 100.0, 5.0, 30.0)
        expected_ea = math.degrees(math.acos(math.exp(-100.0 / T1_MUSCLE)))
        assert abs(r.ernst_angle_deg - expected_ea) < 1e-9

    def test_ernst_angle_gives_maximum_signal(self):
        """Signal at Ernst angle should exceed signal at arbitrary angle."""
        ea = ernst_angle(T1_MUSCLE, 100.0)
        r_ernst = simulate_gre(T1_MUSCLE, T2S_MUSCLE, 100.0, 5.0, ea)
        r_other = simulate_gre(T1_MUSCLE, T2S_MUSCLE, 100.0, 5.0, ea / 2)
        assert r_ernst.signal_normalized > r_other.signal_normalized

    def test_flip_zero_gives_zero(self):
        r = simulate_gre(T1_MUSCLE, T2S_MUSCLE, 100.0, 5.0, 0.0)
        assert abs(r.signal_normalized) < 1e-12

    def test_e1_formula(self):
        r = simulate_gre(T1_MUSCLE, T2S_MUSCLE, 100.0, 5.0, 30.0)
        assert abs(r.e1 - math.exp(-100.0 / T1_MUSCLE)) < 1e-12

    def test_e2_star_formula(self):
        r = simulate_gre(T1_MUSCLE, T2S_MUSCLE, 100.0, 5.0, 30.0)
        assert abs(r.e2_star - math.exp(-5.0 / T2S_MUSCLE)) < 1e-12

    def test_shorter_te_gives_larger_signal(self):
        r1 = simulate_gre(T1_MUSCLE, T2S_MUSCLE, 100.0, 2.0, 30.0)
        r2 = simulate_gre(T1_MUSCLE, T2S_MUSCLE, 100.0, 15.0, 30.0)
        assert r1.signal_normalized > r2.signal_normalized

    def test_validation_te_ge_tr_raises(self):
        with pytest.raises(ValueError, match="TE"):
            simulate_gre(T1_MUSCLE, T2S_MUSCLE, tr_ms=5.0, te_ms=5.0)

    def test_validation_zero_t1_raises(self):
        with pytest.raises(ValueError):
            simulate_gre(0.0, T2S_MUSCLE, 100.0, 5.0)

    def test_stored_parameters(self):
        r = simulate_gre(T1_MUSCLE, T2S_MUSCLE, 150.0, 7.0, 25.0)
        assert r.te_ms == 7.0
        assert r.tr_ms == 150.0
        assert r.flip_angle_deg == 25.0
        assert r.t1_ms == T1_MUSCLE
        assert r.t2_star_ms == T2S_MUSCLE

    def test_long_tr_approaches_e2_star(self):
        """TR >> T1 → saturation → 1; GRE signal → sin(α) · E₂*."""
        r = simulate_gre(T1_MUSCLE, T2S_MUSCLE, tr_ms=1e6, te_ms=5.0, flip_angle_deg=90.0)
        expected = math.exp(-5.0 / T2S_MUSCLE)
        assert abs(r.signal_normalized - expected) < 1e-4


# ═════════════════════════════════════════════════════════════════════════════
# TestInversionRecovery
# ═════════════════════════════════════════════════════════════════════════════


class TestInversionRecovery:
    """Unit tests for simulate_inversion_recovery."""

    def test_returns_ir_result(self):
        r = simulate_inversion_recovery(T1_MUSCLE, tr_ms=3000.0, ti_ms=400.0)
        assert isinstance(r, InversionRecoveryResult)

    def test_ti_zero_gives_negative_one(self):
        """At TI → 0+ signal = 1 − 2×1 + E₁ = −1 + E₁ ≈ −1 for long TR."""
        r = simulate_inversion_recovery(T1_MUSCLE, tr_ms=1e6, ti_ms=0.001)
        assert r.signal_normalized < -0.9

    def test_ti_eq_tr_long_approaches_positive_one(self):
        """At TI = TR → ∞, signal → 1 − 2 × E₁ + E₁ = 1 − E₁ ≈ 1."""
        r = simulate_inversion_recovery(T1_MUSCLE, tr_ms=1e6, ti_ms=1e6)
        assert r.signal_normalized > 0.9

    def test_null_point_formula_long_tr(self):
        """For TR >> T₁, null point ≈ T₁ ln(2)."""
        r = simulate_inversion_recovery(T1_MUSCLE, tr_ms=1e6, ti_ms=T1_MUSCLE)
        expected_null = T1_MUSCLE * math.log(2.0)
        # 5% tolerance (for very long TR the exact formula converges)
        assert abs(r.null_point_ms - expected_null) / expected_null < 0.005

    def test_signal_at_null_point_is_zero(self):
        """Signal evaluated at the null point should be ≈ 0."""
        t1 = T1_MUSCLE
        tr = 1e6
        r_ref = simulate_inversion_recovery(t1, tr, t1)
        null = r_ref.null_point_ms
        r_null = simulate_inversion_recovery(t1, tr, null)
        assert abs(r_null.signal_normalized) < 1e-8

    def test_validation_ti_gt_tr_raises(self):
        with pytest.raises(ValueError, match="TI"):
            simulate_inversion_recovery(T1_MUSCLE, tr_ms=500.0, ti_ms=600.0)

    def test_validation_zero_t1_raises(self):
        with pytest.raises(ValueError):
            simulate_inversion_recovery(0.0, tr_ms=500.0, ti_ms=200.0)

    def test_stored_parameters(self):
        r = simulate_inversion_recovery(700.0, 3000.0, 500.0)
        assert r.t1_ms == 700.0
        assert r.tr_ms == 3000.0
        assert r.ti_ms == 500.0


# ═════════════════════════════════════════════════════════════════════════════
# TestErnstAngle
# ═════════════════════════════════════════════════════════════════════════════


class TestErnstAngle:

    def test_long_tr_approaches_90(self):
        """TR >> T₁ → E₁ → 0 → Ernst angle → arccos(0) = 90°."""
        ea = ernst_angle(T1_MUSCLE, 1e9)
        assert abs(ea - 90.0) < 0.001

    def test_short_tr_gives_small_angle(self):
        """TR << T₁ → E₁ → 1 → Ernst angle → arccos(1) = 0°."""
        ea = ernst_angle(T1_MUSCLE, 0.01)
        assert ea < 5.0

    def test_returns_float(self):
        assert isinstance(ernst_angle(T1_MUSCLE, 100.0), float)

    def test_positive_angle(self):
        ea = ernst_angle(T1_MUSCLE, 100.0)
        assert 0.0 < ea < 90.0

    def test_validation_zero_t1_raises(self):
        with pytest.raises(ValueError):
            ernst_angle(0.0, 100.0)

    def test_validation_zero_tr_raises(self):
        with pytest.raises(ValueError):
            ernst_angle(T1_MUSCLE, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# TestOptimalTE
# ═════════════════════════════════════════════════════════════════════════════


class TestOptimalTE:

    def test_between_t2_values(self):
        """Optimal TE should lie between the two T₂ values."""
        te_opt = optimal_te_for_contrast(80.0, 35.0)
        assert 35.0 < te_opt < 80.0

    def test_symmetric(self):
        """Order of arguments should not matter (contrast is symmetric)."""
        te_1 = optimal_te_for_contrast(80.0, 35.0)
        te_2 = optimal_te_for_contrast(35.0, 80.0)
        assert abs(te_1 - te_2) < 1e-9

    def test_equal_t2_raises(self):
        with pytest.raises(ValueError, match="differ"):
            optimal_te_for_contrast(80.0, 80.0)

    def test_zero_t2_raises(self):
        with pytest.raises(ValueError):
            optimal_te_for_contrast(0.0, 35.0)

    def test_returns_float(self):
        assert isinstance(optimal_te_for_contrast(80.0, 35.0), float)


# ═════════════════════════════════════════════════════════════════════════════
# TestSNREfficiency
# ═════════════════════════════════════════════════════════════════════════════


class TestSNREfficiency:

    def test_returns_snr_efficiency(self):
        r = snr_efficiency(T1_MUSCLE, T2_MUSCLE, 500.0, 10.0)
        assert isinstance(r, SNREfficiency)

    def test_snr_per_sqrt_time_positive(self):
        r = snr_efficiency(T1_MUSCLE, T2_MUSCLE, 500.0, 10.0)
        assert r.snr_per_sqrt_scan_time > 0

    def test_snr_efficiency_formula(self):
        r = snr_efficiency(T1_MUSCLE, T2_MUSCLE, 200.0, 10.0)
        tr_s = 200.0e-3
        expected = r.signal_normalized / math.sqrt(tr_s)
        assert abs(r.snr_per_sqrt_scan_time - expected) < 1e-10

    def test_scan_time_matches_tr(self):
        r = snr_efficiency(T1_MUSCLE, T2_MUSCLE, 300.0, 10.0)
        assert abs(r.scan_time_per_slice_s - 0.3) < 1e-12

    def test_ernst_angle_present(self):
        r = snr_efficiency(T1_MUSCLE, T2_MUSCLE, 100.0, 5.0)
        ea = ernst_angle(T1_MUSCLE, 100.0)
        assert abs(r.ernst_angle_deg - ea) < 1e-9
