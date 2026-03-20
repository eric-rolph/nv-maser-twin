"""Tests for physics/gain_bandwidth_match.py (R2 – maser gain-bandwidth matching)."""

from __future__ import annotations

import pytest

from nv_maser.physics import (
    GainBandwidthConfig,
    BandwidthMatchResult,
    compute_maser_gain_bandwidth,
    compute_b0_drift_tolerance,
    compute_bandwidth_match,
    sweep_q_vs_gain_bandwidth,
    sweep_b0_drift_vs_overlap,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

NOMINAL = GainBandwidthConfig()  # Q=30 000, f0=1.4699 GHz, readout=20 kHz, B0=50 mT


# ---------------------------------------------------------------------------
# TestGainBandwidthConfig
# ---------------------------------------------------------------------------

class TestGainBandwidthConfig:
    def test_defaults(self):
        cfg = GainBandwidthConfig()
        assert cfg.cavity_q == pytest.approx(30_000.0)
        assert cfg.f0_hz == pytest.approx(1.4699e9)
        assert cfg.readout_bw_hz == pytest.approx(20_000.0)
        assert cfg.b0_tesla == pytest.approx(0.050)

    def test_custom_values(self):
        cfg = GainBandwidthConfig(cavity_q=10_000, f0_hz=2.0e9, readout_bw_hz=10_000.0)
        assert cfg.cavity_q == pytest.approx(10_000.0)
        assert cfg.f0_hz == pytest.approx(2.0e9)
        assert cfg.readout_bw_hz == pytest.approx(10_000.0)

    def test_frozen(self):
        cfg = GainBandwidthConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.cavity_q = 999  # type: ignore[misc]

    def test_gyro_ratio_default(self):
        from nv_maser.physics.gain_bandwidth_match import PROTON_GYRO_HZ_T
        assert GainBandwidthConfig().gyro_ratio_hz_t == pytest.approx(PROTON_GYRO_HZ_T)


# ---------------------------------------------------------------------------
# TestComputeMaserGainBandwidth
# ---------------------------------------------------------------------------

class TestComputeMaserGainBandwidth:
    def test_nominal_approx_49khz(self):
        bw = compute_maser_gain_bandwidth(30_000, 1.4699e9)
        assert bw == pytest.approx(1.4699e9 / 30_000, rel=1e-9)
        assert bw == pytest.approx(48_996.7, rel=1e-4)  # ≈49 kHz

    def test_q10k_gives_147khz(self):
        bw = compute_maser_gain_bandwidth(10_000, 1.4699e9)
        assert bw == pytest.approx(1.4699e5, rel=1e-4)

    def test_zero_q_raises(self):
        with pytest.raises(ValueError, match="cavity_q"):
            compute_maser_gain_bandwidth(0.0, 1.4699e9)

    def test_negative_q_raises(self):
        with pytest.raises(ValueError, match="cavity_q"):
            compute_maser_gain_bandwidth(-1.0, 1.4699e9)

    def test_negative_f0_raises(self):
        with pytest.raises(ValueError, match="f0_hz"):
            compute_maser_gain_bandwidth(10_000, -1.0)

    def test_inversely_proportional_to_q(self):
        bw1 = compute_maser_gain_bandwidth(10_000, 1.0e9)
        bw2 = compute_maser_gain_bandwidth(20_000, 1.0e9)
        assert bw1 == pytest.approx(2.0 * bw2, rel=1e-9)

    def test_proportional_to_f0(self):
        bw1 = compute_maser_gain_bandwidth(10_000, 1.0e9)
        bw2 = compute_maser_gain_bandwidth(10_000, 2.0e9)
        assert bw2 == pytest.approx(2.0 * bw1, rel=1e-9)

    def test_returns_float(self):
        assert isinstance(compute_maser_gain_bandwidth(30_000, 1.4699e9), float)


# ---------------------------------------------------------------------------
# TestComputeB0DriftTolerance
# ---------------------------------------------------------------------------

class TestComputeB0DriftTolerance:
    def test_nominal_nonzero(self):
        tol_t, tol_ppm = compute_b0_drift_tolerance(NOMINAL)
        assert tol_t > 0.0
        assert tol_ppm > 0.0

    def test_nominal_formula(self):
        bw_maser = 1.4699e9 / 30_000
        margin_hz = (bw_maser - 20_000.0) / 2.0
        expected_t = margin_hz / GainBandwidthConfig().gyro_ratio_hz_t
        expected_ppm = expected_t / 0.050 * 1.0e6
        tol_t, tol_ppm = compute_b0_drift_tolerance(NOMINAL)
        assert tol_t == pytest.approx(expected_t, rel=1e-6)
        assert tol_ppm == pytest.approx(expected_ppm, rel=1e-6)

    def test_readout_exceeds_maser_bw_returns_zero(self):
        cfg = GainBandwidthConfig(readout_bw_hz=200_000.0)  # >> 49 kHz
        tol_t, tol_ppm = compute_b0_drift_tolerance(cfg)
        assert tol_t == pytest.approx(0.0)
        assert tol_ppm == pytest.approx(0.0)

    def test_high_q_no_margin(self):
        # Q=100 000 → BW = 14.7 kHz < readout 20 kHz → no margin
        cfg = GainBandwidthConfig(cavity_q=100_000)
        tol_t, tol_ppm = compute_b0_drift_tolerance(cfg)
        assert tol_ppm == pytest.approx(0.0)

    def test_lower_q_more_tolerance(self):
        low_q = GainBandwidthConfig(cavity_q=5_000)   # BW ≈ 294 kHz
        high_q = GainBandwidthConfig(cavity_q=30_000)  # BW ≈ 49 kHz
        tol_low, _ = compute_b0_drift_tolerance(low_q)
        tol_high, _ = compute_b0_drift_tolerance(high_q)
        assert tol_low > tol_high

    def test_returns_tuple_of_floats(self):
        result = compute_b0_drift_tolerance(NOMINAL)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# TestComputeBandwidthMatch
# ---------------------------------------------------------------------------

class TestComputeBandwidthMatch:
    def test_returns_result_type(self):
        assert isinstance(compute_bandwidth_match(NOMINAL), BandwidthMatchResult)

    def test_passes_criterion_nominal(self):
        assert compute_bandwidth_match(NOMINAL).passes_criterion is True

    def test_nominal_gain_bw_approx_49khz(self):
        result = compute_bandwidth_match(NOMINAL)
        assert result.maser_gain_bw_hz == pytest.approx(1.4699e9 / 30_000, rel=1e-9)
        # Approximately 49 kHz (within 2 %)
        assert result.maser_gain_bw_hz == pytest.approx(49_000.0, rel=0.02)

    def test_readout_bw_echoed(self):
        result = compute_bandwidth_match(NOMINAL)
        assert result.readout_bw_hz == pytest.approx(NOMINAL.readout_bw_hz)

    def test_frequency_margin_positive_nominal(self):
        result = compute_bandwidth_match(NOMINAL)
        assert result.frequency_margin_hz > 0.0

    def test_overlap_fraction_between_0_and_1(self):
        result = compute_bandwidth_match(NOMINAL)
        assert 0.0 < result.overlap_fraction < 1.0

    def test_overlap_fraction_formula(self):
        result = compute_bandwidth_match(NOMINAL)
        expected = max(0.0, 1.0 - 20_000.0 / result.maser_gain_bw_hz)
        assert result.overlap_fraction == pytest.approx(expected, rel=1e-9)

    def test_fails_when_readout_too_wide(self):
        cfg = GainBandwidthConfig(readout_bw_hz=200_000.0)
        result = compute_bandwidth_match(cfg)
        assert result.passes_criterion is False
        assert result.overlap_fraction == pytest.approx(0.0)

    def test_drift_tolerance_consistent_with_standalone(self):
        result = compute_bandwidth_match(NOMINAL)
        tol_t, tol_ppm = compute_b0_drift_tolerance(NOMINAL)
        assert result.b0_drift_tolerance_t == pytest.approx(tol_t)
        assert result.b0_drift_tolerance_ppm == pytest.approx(tol_ppm)

    def test_architecture_doc_bandwidth_margin_is_generous(self):
        # margin_hz = (49 kHz − 20 kHz) / 2 ≈ 14.5 kHz
        # ΔB_max = margin_hz / γ_p ≈ 350 µT
        # In ppm at B₀ = 50 mT: 350 µT / 50 mT × 1e6 ≈ 7 000 ppm
        # This is far more permissive than the 10 ppm B₀ stability required
        # for NMR image quality — confirming R2 is mitigated by design.
        result = compute_bandwidth_match(NOMINAL)
        assert result.b0_drift_tolerance_ppm > 100.0   # generous: >> 10 ppm NMR limit
        assert result.b0_drift_tolerance_t > 1.0e-6    # > 1 µT tolerance


# ---------------------------------------------------------------------------
# TestSweepQVsGainBandwidth
# ---------------------------------------------------------------------------

class TestSweepQVsGainBandwidth:
    def test_returns_correct_length(self):
        bws = sweep_q_vs_gain_bandwidth([10_000, 20_000, 30_000])
        assert len(bws) == 3

    def test_monotone_decreasing_in_q(self):
        qs = [5_000, 10_000, 20_000, 50_000]
        bws = sweep_q_vs_gain_bandwidth(qs)
        for i in range(len(bws) - 1):
            assert bws[i] > bws[i + 1]

    def test_matches_standalone_fn(self):
        q = 25_000
        expected = compute_maser_gain_bandwidth(q, 1.4699e9)
        assert sweep_q_vs_gain_bandwidth([q])[0] == pytest.approx(expected)

    def test_single_element(self):
        bws = sweep_q_vs_gain_bandwidth([30_000])
        assert len(bws) == 1
        assert bws[0] == pytest.approx(1.4699e9 / 30_000, rel=1e-9)

    def test_custom_f0(self):
        bws = sweep_q_vs_gain_bandwidth([10_000], f0_hz=2.0e9)
        assert bws[0] == pytest.approx(2.0e9 / 10_000, rel=1e-9)


# ---------------------------------------------------------------------------
# TestSweepB0DriftVsOverlap
# ---------------------------------------------------------------------------

class TestSweepB0DriftVsOverlap:
    def test_returns_correct_length(self):
        drifts = [0.0, 100.0, 200.0, 300.0]
        out = sweep_b0_drift_vs_overlap(drifts, NOMINAL)
        assert len(out) == len(drifts)

    def test_zero_drift_equals_static_overlap(self):
        out = sweep_b0_drift_vs_overlap([0.0], NOMINAL)
        expected = compute_bandwidth_match(NOMINAL).overlap_fraction
        assert out[0] == pytest.approx(expected, rel=1e-9)

    def test_large_drift_clips_to_zero(self):
        out = sweep_b0_drift_vs_overlap([1.0e9], NOMINAL)  # 1e9 µT = 1 kT drift
        assert out[0] == pytest.approx(0.0)

    def test_monotone_decreasing_in_drift(self):
        drifts = [0.0, 50.0, 100.0, 200.0, 500.0]
        out = sweep_b0_drift_vs_overlap(drifts, NOMINAL)
        for i in range(len(out) - 1):
            assert out[i] >= out[i + 1]

    def test_negative_drift_same_as_positive(self):
        pos = sweep_b0_drift_vs_overlap([100.0], NOMINAL)
        neg = sweep_b0_drift_vs_overlap([-100.0], NOMINAL)
        assert pos[0] == pytest.approx(neg[0])

    def test_all_values_in_unit_interval(self):
        drifts = list(range(0, 2000, 100))
        out = sweep_b0_drift_vs_overlap(drifts, NOMINAL)
        assert all(0.0 <= v <= 1.0 for v in out)


# ---------------------------------------------------------------------------
# TestPublicAPI
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_all_symbols_importable_from_physics(self):
        from nv_maser.physics import (  # noqa: F401
            GainBandwidthConfig,
            BandwidthMatchResult,
            compute_maser_gain_bandwidth,
            compute_b0_drift_tolerance,
            compute_bandwidth_match,
            sweep_q_vs_gain_bandwidth,
            sweep_b0_drift_vs_overlap,
        )
        assert GainBandwidthConfig is not None
        assert BandwidthMatchResult is not None
