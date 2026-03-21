"""
Tests for physics/field_tolerance_calculator.py — B₀ field tolerance model (SS20, R1).

Coverage:
    T01–T08  FieldToleranceConfig defaults and validation
    T09–T18  compute_b0_sensitivity_point — nominal, deviation, edge cases
    T19–T29  compute_homogeneity_point — ppm→T2*, FID loss, bandwidth
    T30–T37  sweep_b0_sensitivity and sweep_homogeneity shape/ordering
    T38–T43  compute_field_tolerance with default and custom config
    T44–T47  R1 closure confirmation
    T48–T52  Internal helper functions (analytical thresholds)
"""
from __future__ import annotations

import math
import pytest

from nv_maser.physics.field_tolerance_calculator import (
    FieldToleranceConfig,
    B0SensitivityPoint,
    HomogeneityPoint,
    FieldToleranceResult,
    compute_b0_sensitivity_point,
    compute_homogeneity_point,
    sweep_b0_sensitivity,
    sweep_homogeneity,
    compute_field_tolerance,
    _b0_for_snr_loss_db,
    _uniformity_for_fid_loss_db,
    _maser_limit_ppm,
    _GAMMA_BAR_P,
    _PI,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def default_cfg() -> FieldToleranceConfig:
    return FieldToleranceConfig()


@pytest.fixture
def fast_cfg() -> FieldToleranceConfig:
    """Smaller sweep counts for fast tests."""
    return FieldToleranceConfig(n_b0_sweep=5, n_uniformity_sweep=5)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T01–T08  FieldToleranceConfig defaults and validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_T01_default_b0_nominal(default_cfg):
    assert default_cfg.b0_nominal_t == pytest.approx(0.050)


def test_T02_default_sweep_range(default_cfg):
    assert default_cfg.b0_sweep_min_t == pytest.approx(0.030)
    assert default_cfg.b0_sweep_max_t == pytest.approx(0.070)


def test_T03_default_v1_tolerance(default_cfg):
    assert default_cfg.v1_b0_tolerance_t == pytest.approx(0.005)
    assert default_cfg.v1_uniformity_ppm == pytest.approx(500.0)


def test_T04_default_maser_bandwidth(default_cfg):
    """Maser bandwidth should match ADR-020 value (≈ 49 kHz)."""
    assert default_cfg.maser_bandwidth_hz == pytest.approx(49_000.0)


def test_T05_default_te_t2(default_cfg):
    assert default_cfg.te_fid_us == pytest.approx(100.0)
    assert default_cfg.t2_tissue_ms == pytest.approx(50.0)


def test_T06_config_is_frozen(default_cfg):
    with pytest.raises((AttributeError, TypeError)):
        default_cfg.b0_nominal_t = 0.060  # type: ignore[misc]


def test_T07_custom_config_roundtrip():
    cfg = FieldToleranceConfig(b0_nominal_t=0.060, te_fid_us=200.0)
    assert cfg.b0_nominal_t == pytest.approx(0.060)
    assert cfg.te_fid_us == pytest.approx(200.0)


def test_T08_config_sweep_counts(default_cfg):
    assert default_cfg.n_b0_sweep == 21
    assert default_cfg.n_uniformity_sweep == 20


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T09–T18  compute_b0_sensitivity_point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_T09_nominal_point_zero_loss():
    pt = compute_b0_sensitivity_point(0.050, 0.050)
    assert pt.snr_factor == pytest.approx(1.0)
    assert pt.snr_loss_db == pytest.approx(0.0, abs=1e-9)


def test_T10_nominal_point_b0_values():
    pt = compute_b0_sensitivity_point(0.050, 0.050)
    assert pt.b0_tesla == pytest.approx(0.050)
    assert pt.b0_mT == pytest.approx(50.0)
    assert pt.b0_deviation_pct == pytest.approx(0.0, abs=1e-9)


def test_T11_larmor_frequency_at_nominal():
    pt = compute_b0_sensitivity_point(0.050, 0.050)
    expected_hz = _GAMMA_BAR_P * 0.050
    assert pt.larmor_frequency_hz == pytest.approx(expected_hz)


def test_T12_larmor_at_50mT_is_2129kHz():
    """γ̄ × 50 mT = 42.577 MHz/T × 0.05 T = 2.129 MHz (proton Larmor)."""
    pt = compute_b0_sensitivity_point(0.050, 0.050)
    assert pt.larmor_frequency_hz == pytest.approx(2.129e6, rel=1e-3)


def test_T13_snr_factor_below_nominal():
    """B₀ = 45 mT → SNR factor = (45/50)² = 0.81."""
    pt = compute_b0_sensitivity_point(0.050, 0.045)
    assert pt.snr_factor == pytest.approx((45 / 50) ** 2, rel=1e-6)


def test_T14_snr_loss_at_45mT():
    """SNR loss at 45 mT should be ≈ 1.84 dB (well below 3 dB)."""
    pt = compute_b0_sensitivity_point(0.050, 0.045)
    # snr_loss_db = -20 * log10((45/50)^2) = 40 * log10(50/45)
    expected = -20.0 * math.log10((45 / 50) ** 2)
    assert pt.snr_loss_db == pytest.approx(expected, rel=1e-6)
    assert pt.snr_loss_db < 3.0  # well below 3 dB


def test_T15_snr_factor_above_nominal():
    """B₀ = 55 mT → SNR factor > 1 (gain, not loss)."""
    pt = compute_b0_sensitivity_point(0.050, 0.055)
    assert pt.snr_factor == pytest.approx((55 / 50) ** 2, rel=1e-6)
    assert pt.snr_loss_db < 0.0   # negative loss = gain


def test_T16_polarization_equals_signal_frequency():
    """Both M₀ and ω₀ contribute one power of B, so both factors should be equal."""
    pt = compute_b0_sensitivity_point(0.050, 0.045)
    assert pt.polarization_factor == pytest.approx(pt.signal_frequency_factor)
    assert pt.polarization_factor == pytest.approx(0.045 / 0.050)


def test_T17_invalid_b0_nominal_raises():
    with pytest.raises(ValueError, match="b0_nominal_t"):
        compute_b0_sensitivity_point(0.0, 0.050)


def test_T18_invalid_b0_actual_raises():
    with pytest.raises(ValueError, match="b0_actual_t"):
        compute_b0_sensitivity_point(0.050, -0.010)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T19–T29  compute_homogeneity_point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_T19_zero_ppm_perfect_homogeneity(default_cfg):
    pt = compute_homogeneity_point(default_cfg, 0.0)
    assert pt.delta_b0_mt == pytest.approx(0.0)
    assert pt.delta_frequency_hz == pytest.approx(0.0)
    assert math.isinf(pt.t2star_inhom_ms)
    assert pt.snr_loss_fid_factor == pytest.approx(1.0)
    assert pt.snr_loss_fid_db == pytest.approx(0.0, abs=1e-9)


def test_T20_zero_ppm_within_maser_bw(default_cfg):
    pt = compute_homogeneity_point(default_cfg, 0.0)
    assert pt.within_maser_bandwidth is True


def test_T21_delta_b0_from_ppm(default_cfg):
    """500 ppm at 50 mT → ΔB₀ = 0.025 mT."""
    pt = compute_homogeneity_point(default_cfg, 500.0)
    expected_mt = 0.050 * 500e-6 * 1e3
    assert pt.delta_b0_mt == pytest.approx(expected_mt)


def test_T22_delta_frequency_500ppm(default_cfg):
    """500 ppm → Δν ≈ 1064 Hz."""
    pt = compute_homogeneity_point(default_cfg, 500.0)
    delta_b0_t = 0.050 * 500e-6
    expected_hz = _GAMMA_BAR_P * delta_b0_t
    assert pt.delta_frequency_hz == pytest.approx(expected_hz, rel=1e-6)
    assert pt.delta_frequency_hz == pytest.approx(1064.0, rel=5e-3)


def test_T23_t2star_inhom_500ppm(default_cfg):
    """T2*_inhom = 1/(π × Δν) at 500 ppm."""
    pt = compute_homogeneity_point(default_cfg, 500.0)
    delta_b0_t = 0.050 * 500e-6
    delta_freq = _GAMMA_BAR_P * delta_b0_t
    expected_ms = 1.0 / (_PI * delta_freq) * 1e3
    assert pt.t2star_inhom_ms == pytest.approx(expected_ms, rel=1e-6)


def test_T24_t2star_eff_shorter_than_t2(default_cfg):
    """T2*_eff must always be shorter than (or equal to) tissue T2."""
    for ppm in [100.0, 500.0, 2000.0]:
        pt = compute_homogeneity_point(default_cfg, ppm)
        assert pt.t2star_eff_ms <= default_cfg.t2_tissue_ms


def test_T25_t2star_eff_formula(default_cfg):
    """1/T2*_eff = 1/T2_tissue + 1/T2*_inhom."""
    pt = compute_homogeneity_point(default_cfg, 500.0)
    inv_t2 = 1.0 / default_cfg.t2_tissue_ms
    inv_inhom = 1.0 / pt.t2star_inhom_ms
    expected_eff = 1.0 / (inv_t2 + inv_inhom)
    assert pt.t2star_eff_ms == pytest.approx(expected_eff, rel=1e-6)


def test_T26_fid_loss_factor_monotone_decreasing(default_cfg):
    """FID loss factor should decrease (more loss) as uniformity worsens."""
    ppms = [100.0, 300.0, 500.0, 1000.0, 3000.0]
    factors = [compute_homogeneity_point(default_cfg, p).snr_loss_fid_factor
               for p in ppms]
    for a, b in zip(factors[:-1], factors[1:]):
        assert a >= b, "Loss factor must be non-increasing with ppm"


def test_T27_fid_loss_db_positive(default_cfg):
    """FID loss should be ≥ 0 dB for any ppm > 0."""
    for ppm in [10.0, 500.0, 2000.0]:
        pt = compute_homogeneity_point(default_cfg, ppm)
        assert pt.snr_loss_fid_db >= 0.0


def test_T28_maser_bandwidth_flag_low_ppm(default_cfg):
    """500 ppm → ≈ 1 kHz spectral BW, well within 49 kHz maser BW."""
    pt = compute_homogeneity_point(default_cfg, 500.0)
    assert pt.within_maser_bandwidth is True


def test_T29_negative_ppm_raises(default_cfg):
    with pytest.raises(ValueError, match="uniformity_ppm"):
        compute_homogeneity_point(default_cfg, -1.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T30–T37  Sweep functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_T30_b0_sweep_length(default_cfg):
    pts = sweep_b0_sensitivity(default_cfg)
    assert len(pts) == default_cfg.n_b0_sweep


def test_T31_b0_sweep_endpoints(default_cfg):
    pts = sweep_b0_sensitivity(default_cfg)
    assert pts[0].b0_tesla == pytest.approx(default_cfg.b0_sweep_min_t)
    assert pts[-1].b0_tesla == pytest.approx(default_cfg.b0_sweep_max_t)


def test_T32_b0_sweep_ascending(default_cfg):
    pts = sweep_b0_sensitivity(default_cfg)
    for a, b in zip(pts[:-1], pts[1:]):
        assert a.b0_tesla < b.b0_tesla


def test_T33_b0_sweep_snr_factor_at_nominal_1(default_cfg):
    """The sweep should include the nominal point with SNR factor = 1."""
    pts = sweep_b0_sensitivity(default_cfg)
    nominal_pts = [p for p in pts
                   if abs(p.b0_tesla - default_cfg.b0_nominal_t) < 1e-6]
    assert len(nominal_pts) >= 1
    assert nominal_pts[0].snr_factor == pytest.approx(1.0, abs=1e-6)


def test_T34_uniformity_sweep_length(default_cfg):
    pts = sweep_homogeneity(default_cfg)
    assert len(pts) == default_cfg.n_uniformity_sweep


def test_T35_uniformity_sweep_endpoints(default_cfg):
    pts = sweep_homogeneity(default_cfg)
    assert pts[0].uniformity_ppm == pytest.approx(
        default_cfg.uniformity_sweep_min_ppm
    )
    assert pts[-1].uniformity_ppm == pytest.approx(
        default_cfg.uniformity_sweep_max_ppm
    )


def test_T36_uniformity_sweep_ascending(default_cfg):
    pts = sweep_homogeneity(default_cfg)
    for a, b in zip(pts[:-1], pts[1:]):
        assert a.uniformity_ppm < b.uniformity_ppm


def test_T37_b0_sweep_invalid_config_raises():
    bad_cfg = FieldToleranceConfig(n_b0_sweep=1)
    with pytest.raises(ValueError, match="n_b0_sweep"):
        sweep_b0_sensitivity(bad_cfg)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T38–T43  compute_field_tolerance — defaults and structure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_T38_field_tolerance_returns_result():
    result = compute_field_tolerance()
    assert isinstance(result, FieldToleranceResult)


def test_T39_field_tolerance_none_config_uses_defaults():
    result = compute_field_tolerance(None)
    assert result.config.b0_nominal_t == pytest.approx(0.050)


def test_T40_sweep_lengths_in_result(fast_cfg):
    result = compute_field_tolerance(fast_cfg)
    assert len(result.b0_sensitivity) == fast_cfg.n_b0_sweep
    assert len(result.homogeneity) == fast_cfg.n_uniformity_sweep


def test_T41_b0_3db_threshold_below_nominal(default_cfg):
    result = compute_field_tolerance(default_cfg)
    assert result.b0_3db_loss_t < default_cfg.b0_nominal_t


def test_T42_b0_1db_threshold_between_3db_and_nominal(default_cfg):
    result = compute_field_tolerance(default_cfg)
    assert result.b0_3db_loss_t < result.b0_1db_loss_t < default_cfg.b0_nominal_t


def test_T43_uniformity_maser_limit_above_v1_spec(default_cfg):
    """Maser BW limit must be >> 500 ppm (much more relaxed than thermal spec)."""
    result = compute_field_tolerance(default_cfg)
    assert result.uniformity_maser_limit_ppm > default_cfg.v1_uniformity_ppm * 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T44–T47  R1 closure confirmation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_T44_r1_closed_with_default_config():
    """R1 risk must be confirmed closed for the default V1 arch spec."""
    result = compute_field_tolerance()
    assert result.r1_risk_closed is True


def test_T45_v1_snr_loss_below_3db():
    """Worst-case SNR loss at 45 mT should be < 3 dB."""
    result = compute_field_tolerance()
    assert result.v1_snr_loss_at_b0_min_db < 3.0
    # Specific value ≈ 1.84 dB
    expected_loss = -20.0 * math.log10((0.045 / 0.050) ** 2)
    assert result.v1_snr_loss_at_b0_min_db == pytest.approx(expected_loss, rel=1e-5)


def test_T46_v1_spectral_bandwidth_fits_maser():
    """500 ppm signal bandwidth (≈ 1 064 Hz) must fit inside maser BW (49 kHz)."""
    result = compute_field_tolerance()
    # Signal BW at 500 ppm ≈ 1064 Hz; maser half-BW ≈ 24500 Hz
    assert result.v1_spectral_bandwidth_at_spec_hz < result.config.maser_bandwidth_hz / 2
    assert result.v1_maser_bandwidth_margin_hz > 0.0
    # Margin should be >> signal BW (at least 20×)
    assert result.v1_maser_bandwidth_margin_hz > result.v1_spectral_bandwidth_at_spec_hz * 20


def test_T47_r1_open_when_tolerance_too_tight():
    """Artificially tighten the B₀ tolerance so 45 mT is no longer acceptable.

    R1 should open if v1_b0_tolerance is reduced to make b0_min fall below
    the 3 dB SNR loss point.  The 3 dB point is at ≈ 42.07 mT, so a
    tolerance of 0.001 T (49 mT minimum) should still close R1; however,
    making nominal closer to b0_3db threshold while also requiring ppm
    beyond maser BW would open it.  We test direct manipulation of the
    maser BW to engineer an open condition.
    """
    # Force maser BW so small that even 500 ppm exceeds it
    tiny_bw_cfg = FieldToleranceConfig(maser_bandwidth_hz=100.0)
    result = compute_field_tolerance(tiny_bw_cfg)
    # With 100 Hz maser BW, 500 ppm → ≈1064 Hz > 50 Hz half-BW → maser margin < 0
    assert result.v1_maser_bandwidth_margin_hz < 0.0
    assert result.r1_risk_closed is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T48–T52  Analytical helper functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_T48_b0_for_3db_loss():
    """B₀ at 3 dB loss should satisfy (B₀/50mT)² = 10^(-3/20)."""
    b0 = _b0_for_snr_loss_db(0.050, 3.0)
    snr_factor = (b0 / 0.050) ** 2
    assert snr_factor == pytest.approx(10 ** (-3 / 20), rel=1e-6)


def test_T49_b0_for_3db_loss_value():
    """At 50 mT nominal, 3 dB occurs at ≈ 42.07 mT."""
    b0 = _b0_for_snr_loss_db(0.050, 3.0)
    assert b0 == pytest.approx(0.050 * 10 ** (-3 / 40), rel=1e-6)
    # Must be below 45 mT (V1 min) — confirms V1 range never crosses 3 dB
    assert b0 < 0.045


def test_T50_uniformity_for_fid_loss_zero_te():
    """Zero TE means no FID dephasing regardless of ppm — returns infinity."""
    cfg_zero_te = FieldToleranceConfig(te_fid_us=0.0)
    ppm = _uniformity_for_fid_loss_db(cfg_zero_te, 3.0)
    assert math.isinf(ppm)


def test_T51_maser_limit_ppm_formula():
    """maser_limit_ppm = (BW_maser/2) / (γ̄ × B₀_nom) × 1e6."""
    cfg = FieldToleranceConfig(b0_nominal_t=0.050, maser_bandwidth_hz=49_000.0)
    expected = (cfg.maser_bandwidth_hz / 2.0) / (_GAMMA_BAR_P * cfg.b0_nominal_t) * 1e6
    assert _maser_limit_ppm(cfg) == pytest.approx(expected, rel=1e-8)


def test_T52_maser_limit_value():
    """At 50 mT and 49 kHz BW, maser limit ≈ 11 508 ppm — much > 500 ppm spec."""
    limit = _maser_limit_ppm(FieldToleranceConfig())
    assert limit == pytest.approx(
        (49_000 / 2.0) / (_GAMMA_BAR_P * 0.050) * 1e6, rel=1e-6
    )
    assert limit > 10_000.0  # >> 500 ppm spec
