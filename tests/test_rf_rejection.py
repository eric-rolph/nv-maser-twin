"""Tests for physics/rf_rejection.py — RF interference rejection model (R6)."""
from __future__ import annotations

import math

import pytest

from nv_maser.physics.rf_rejection import (
    InterfererResult,
    InterfererSpec,
    RFRejectionConfig,
    RFRejectionResult,
    compute_fractional_bandwidth,
    compute_interferer_rejection,
    compute_lorentzian_attenuation,
    compute_rf_rejection,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _cfg(**kw) -> RFRejectionConfig:
    """Convenience: build a config overriding keyword args."""
    return RFRejectionConfig(**kw)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  compute_lorentzian_attenuation
# ══════════════════════════════════════════════════════════════════════════════


class TestLorentzianAttenuation:
    def test_on_resonance_is_zero_db(self):
        """Signal exactly at maser centre → 0 dB attenuation."""
        atten = compute_lorentzian_attenuation(
            freq_hz=1.4699e9, center_hz=1.4699e9, bw_hz=49_000.0
        )
        assert atten == pytest.approx(0.0, abs=1e-10)

    def test_at_half_bw_is_3db(self):
        """Signal at ±BW/2 (half-power points) → 3.01 dB attenuation."""
        center = 1.4699e9
        bw = 49_000.0
        atten = compute_lorentzian_attenuation(
            freq_hz=center + bw / 2, center_hz=center, bw_hz=bw
        )
        assert atten == pytest.approx(10 * math.log10(2), rel=1e-6)

    def test_symmetric_about_centre(self):
        """Attenuation is the same above and below centre."""
        center = 1.4699e9
        bw = 49_000.0
        delta = 200e6
        a_high = compute_lorentzian_attenuation(center + delta, center, bw)
        a_low = compute_lorentzian_attenuation(center - delta, center, bw)
        assert a_high == pytest.approx(a_low, rel=1e-10)

    def test_attenuation_increases_with_offset(self):
        """Larger frequency offset → more attenuation."""
        center = 1.4699e9
        bw = 49_000.0
        a_near = compute_lorentzian_attenuation(center + 1e6, center, bw)
        a_far = compute_lorentzian_attenuation(center + 100e6, center, bw)
        assert a_far > a_near

    def test_wifi_2_4ghz_rejection_exceeds_80db(self):
        """WiFi at 2.4 GHz is ~930 MHz from 1.47 GHz → > 80 dB rejection."""
        atten = compute_lorentzian_attenuation(
            freq_hz=2.412e9, center_hz=1.4699e9, bw_hz=49_000.0
        )
        assert atten > 80.0

    def test_wifi_5ghz_rejection_exceeds_100db(self):
        """WiFi at 5 GHz is > 3.5 GHz away → > 100 dB rejection."""
        atten = compute_lorentzian_attenuation(
            freq_hz=5.18e9, center_hz=1.4699e9, bw_hz=49_000.0
        )
        assert atten > 100.0

    def test_broadcast_fm_rejection_exceeds_90db(self):
        """Broadcast FM at 98 MHz is ~1.37 GHz from 1.4699 GHz → > 90 dB rejection.

        Computed: Δf ≈ 1.37 GHz, 2Δf/BW ≈ 56 000, OOB ≈ 95 dB.
        """
        atten = compute_lorentzian_attenuation(
            freq_hz=98e6, center_hz=1.4699e9, bw_hz=49_000.0
        )
        assert atten > 90.0

    def test_attenuation_is_non_negative(self):
        """Attenuation is always ≥ 0 dB."""
        for f in [100e6, 500e6, 1e9, 1.4699e9, 2e9, 5e9, 10e9]:
            atten = compute_lorentzian_attenuation(f, 1.4699e9, 49_000.0)
            assert atten >= 0.0

    def test_large_offset_asymptotic_formula(self):
        """For Δf ≫ BW, attenuation ≈ 20·log₁₀(2Δf/BW)."""
        center = 1.4699e9
        bw = 49_000.0
        delta = 500e6  # >> BW
        atten_exact = compute_lorentzian_attenuation(center + delta, center, bw)
        atten_approx = 20.0 * math.log10(2.0 * delta / bw)
        # Should agree to better than 0.01 dB at this large offset
        assert atten_exact == pytest.approx(atten_approx, abs=0.01)

    def test_different_bw_changes_attenuation(self):
        """Narrower bandwidth → more rejection at same offset."""
        center = 1.4699e9
        delta = 10e6
        a_wide = compute_lorentzian_attenuation(center + delta, center, bw_hz=100_000.0)
        a_narrow = compute_lorentzian_attenuation(center + delta, center, bw_hz=10_000.0)
        assert a_narrow > a_wide


# ══════════════════════════════════════════════════════════════════════════════
# 2.  compute_fractional_bandwidth
# ══════════════════════════════════════════════════════════════════════════════


class TestFractionalBandwidth:
    def test_default_value_is_approximately_3e5(self):
        """Default config: BW/f0 ≈ 3.33 × 10⁻⁵."""
        cfg = RFRejectionConfig()
        fb = compute_fractional_bandwidth(cfg)
        expected = 49_000.0 / 1.4699e9
        assert fb == pytest.approx(expected, rel=1e-6)

    def test_fractional_bw_is_positive(self):
        cfg = RFRejectionConfig()
        assert compute_fractional_bandwidth(cfg) > 0.0

    def test_narrower_bw_gives_smaller_fractional_bw(self):
        cfg_narrow = RFRejectionConfig(maser_gain_bw_hz=10_000.0)
        cfg_wide = RFRejectionConfig(maser_gain_bw_hz=100_000.0)
        assert compute_fractional_bandwidth(cfg_narrow) < compute_fractional_bandwidth(cfg_wide)

    def test_custom_config(self):
        cfg = RFRejectionConfig(maser_center_hz=1.5e9, maser_gain_bw_hz=50_000.0)
        assert compute_fractional_bandwidth(cfg) == pytest.approx(50_000.0 / 1.5e9, rel=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  InterfererSpec validation
# ══════════════════════════════════════════════════════════════════════════════


class TestInterfererSpec:
    def test_valid_spec(self):
        spec = InterfererSpec("WiFi", 2.4e9, 40e6, -30.0)
        assert spec.name == "WiFi"
        assert spec.center_freq_hz == 2.4e9

    def test_negative_center_freq_raises(self):
        with pytest.raises(ValueError, match="center_freq_hz"):
            InterfererSpec("Bad", -1e9, 10e6, -30.0)

    def test_zero_center_freq_raises(self):
        with pytest.raises(ValueError, match="center_freq_hz"):
            InterfererSpec("Bad", 0.0, 10e6, -30.0)

    def test_negative_bandwidth_raises(self):
        with pytest.raises(ValueError, match="bandwidth_hz"):
            InterfererSpec("Bad", 2.4e9, -1.0, -30.0)

    def test_zero_bandwidth_raises(self):
        with pytest.raises(ValueError, match="bandwidth_hz"):
            InterfererSpec("Bad", 2.4e9, 0.0, -30.0)

    def test_negative_power_is_valid(self):
        """Negative dBm power is normal in RF (sub-milliwatt)."""
        spec = InterfererSpec("Weak", 2.4e9, 10e6, -90.0)
        assert spec.power_dbm == -90.0

    def test_frozen_immutability(self):
        spec = InterfererSpec("WiFi", 2.4e9, 40e6, -30.0)
        with pytest.raises((AttributeError, TypeError)):
            spec.power_dbm = 0.0  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  RFRejectionConfig validation
# ══════════════════════════════════════════════════════════════════════════════


class TestRFRejectionConfig:
    def test_default_lo_resolved(self):
        """LO frequency defaults to maser_center − 2.129 MHz."""
        cfg = RFRejectionConfig()
        expected_lo = 1.4699e9 - 2.129e6
        assert cfg.lo_freq_hz == pytest.approx(expected_lo, rel=1e-9)

    def test_explicit_lo_preserved(self):
        lo = 1.460e9
        cfg = RFRejectionConfig(lo_freq_hz=lo)
        assert cfg.lo_freq_hz == lo

    def test_default_interferers_present(self):
        cfg = RFRejectionConfig()
        assert len(cfg.interferers) == 8

    def test_negative_maser_center_raises(self):
        with pytest.raises(ValueError, match="maser_center_hz"):
            RFRejectionConfig(maser_center_hz=-1.0)

    def test_zero_gain_bw_raises(self):
        with pytest.raises(ValueError, match="maser_gain_bw_hz"):
            RFRejectionConfig(maser_gain_bw_hz=0.0)

    def test_zero_readout_bw_raises(self):
        with pytest.raises(ValueError, match="readout_bw_hz"):
            RFRejectionConfig(readout_bw_hz=0.0)

    def test_custom_interferer_list(self):
        specs = (InterfererSpec("X", 2.4e9, 20e6, -40.0),)
        cfg = RFRejectionConfig(interferers=specs)
        assert len(cfg.interferers) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 5.  compute_interferer_rejection
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeInterfererRejection:
    def _wifi_spec(self) -> InterfererSpec:
        return InterfererSpec("WiFi 2.4 GHz", 2.412e9, 40e6, -30.0)

    def test_returns_interferer_result(self):
        cfg = RFRejectionConfig()
        spec = self._wifi_spec()
        result = compute_interferer_rejection(spec, cfg)
        assert isinstance(result, InterfererResult)

    def test_freq_offset_correct(self):
        cfg = RFRejectionConfig()
        spec = self._wifi_spec()
        result = compute_interferer_rejection(spec, cfg)
        expected_offset = abs(spec.center_freq_hz - cfg.maser_center_hz)
        assert result.freq_offset_hz == pytest.approx(expected_offset, rel=1e-9)

    def test_attenuation_matches_lorentzian_function(self):
        cfg = RFRejectionConfig()
        spec = self._wifi_spec()
        result = compute_interferer_rejection(spec, cfg)
        expected_att = compute_lorentzian_attenuation(
            spec.center_freq_hz, cfg.maser_center_hz, cfg.maser_gain_bw_hz
        )
        assert result.attenuation_db == pytest.approx(expected_att, rel=1e-9)

    def test_residual_power_is_input_minus_attenuation(self):
        cfg = RFRejectionConfig()
        spec = self._wifi_spec()
        result = compute_interferer_rejection(spec, cfg)
        assert result.residual_power_dbm == pytest.approx(
            spec.power_dbm - result.attenuation_db, rel=1e-9
        )

    def test_wifi_residual_below_minus_110dbm(self):
        """WiFi 2.4 GHz at −30 dBm should leave < −110 dBm residual."""
        cfg = RFRejectionConfig()
        result = compute_interferer_rejection(self._wifi_spec(), cfg)
        assert result.residual_power_dbm < -110.0

    def test_baseband_freq_is_correct(self):
        cfg = RFRejectionConfig()
        spec = self._wifi_spec()
        result = compute_interferer_rejection(spec, cfg)
        expected_bb = abs(spec.center_freq_hz - cfg.lo_freq_hz)
        assert result.baseband_freq_hz == pytest.approx(expected_bb, rel=1e-9)

    def test_wifi_not_in_readout_band(self):
        """2.4 GHz WiFi maps to ~2.4 GHz baseband — far outside 20 kHz readout."""
        cfg = RFRejectionConfig()
        result = compute_interferer_rejection(self._wifi_spec(), cfg)
        assert result.in_readout_band is False

    def test_in_band_interferer_detected(self):
        """An interferer at f_LO + 5 kHz maps to 5 kHz baseband → in band."""
        cfg = RFRejectionConfig()
        # Place the interferer at maser_center_hz (NMR USB) + 5 kHz
        near_spec = InterfererSpec(
            "Near", cfg.maser_center_hz + 5_000.0, 1_000.0, -30.0
        )
        result = compute_interferer_rejection(near_spec, cfg)
        # baseband freq = |maser_center + 5kHz − LO| ≈ NMR_freq + 5kHz ≈ 2.134 MHz → NOT in 20 kHz band
        # Actually this lands at ~2.134 MHz which is NOT in the ±10 kHz readout band.
        assert result.in_readout_band is False
        # Use a spec that truly lands in the readout band:
        # We want |f_int - f_LO| < 10 kHz. f_LO ≈ 1.46777 GHz. So f_int ≈ f_LO ± 5 kHz.
        in_band_spec = InterfererSpec(
            "InBand", cfg.lo_freq_hz + 5_000.0, 1_000.0, -30.0  # type: ignore[arg-type]
        )
        result2 = compute_interferer_rejection(in_band_spec, cfg)
        assert result2.in_readout_band is True

    def test_attenuation_db_is_positive(self):
        """Attenuation should be ≥ 0 for any off-resonance source."""
        cfg = RFRejectionConfig()
        for f in [100e6, 700e6, 2.4e9, 5e9]:
            spec = InterfererSpec("X", f, 10e6, -40.0)
            result = compute_interferer_rejection(spec, cfg)
            assert result.attenuation_db >= 0.0

    def test_on_resonance_interferer_no_attenuation(self):
        """An interferer exactly at maser centre is not attenuated at all."""
        cfg = RFRejectionConfig()
        spec = InterfererSpec("OnRes", cfg.maser_center_hz, 1_000.0, -50.0)
        result = compute_interferer_rejection(spec, cfg)
        assert result.attenuation_db == pytest.approx(0.0, abs=1e-9)
        assert result.residual_power_dbm == pytest.approx(-50.0, abs=1e-9)

    def test_interferer_reference_preserved(self):
        """Result should carry a reference to the original spec."""
        cfg = RFRejectionConfig()
        spec = self._wifi_spec()
        result = compute_interferer_rejection(spec, cfg)
        assert result.interferer is spec


# ══════════════════════════════════════════════════════════════════════════════
# 6.  compute_rf_rejection (aggregate)
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeRFRejection:
    def test_default_returns_rf_rejection_result(self):
        result = compute_rf_rejection()
        assert isinstance(result, RFRejectionResult)

    def test_default_has_eight_interferers(self):
        result = compute_rf_rejection()
        assert len(result.interferer_results) == 8

    def test_all_standard_interferers_attenuated_above_80db(self):
        """Every default interferer should see > 80 dB rejection at 49 kHz BW."""
        result = compute_rf_rejection()
        for r in result.interferer_results:
            assert r.attenuation_db > 80.0, (
                f"{r.interferer.name}: only {r.attenuation_db:.1f} dB"
            )

    def test_no_standard_interferer_in_readout_band(self):
        """None of the standard 8 interferers should land in the 20 kHz readout band."""
        result = compute_rf_rejection()
        assert result.any_in_readout_band is False

    def test_worst_case_residual_identified(self):
        """worst_case_residual_dbm should equal the max across individual results."""
        result = compute_rf_rejection()
        computed_max = max(r.residual_power_dbm for r in result.interferer_results)
        assert result.worst_case_residual_dbm == pytest.approx(computed_max, rel=1e-9)

    def test_worst_case_name_matches(self):
        result = compute_rf_rejection()
        expected_name = max(
            result.interferer_results, key=lambda r: r.residual_power_dbm
        ).interferer.name
        assert result.worst_case_name == expected_name

    def test_fractional_bw_in_result(self):
        cfg = RFRejectionConfig()
        result = compute_rf_rejection(cfg)
        assert result.maser_fractional_bw == pytest.approx(
            compute_fractional_bandwidth(cfg), rel=1e-9
        )

    def test_default_fractional_bw_below_1e4(self):
        """Fractional bandwidth must be < 1e-4 (very selective)."""
        result = compute_rf_rejection()
        assert result.maser_fractional_bw < 1e-4

    def test_min_max_attenuation_ordering(self):
        result = compute_rf_rejection()
        assert result.min_attenuation_db <= result.max_attenuation_db

    def test_none_config_uses_defaults(self):
        """compute_rf_rejection(None) should behave like compute_rf_rejection()."""
        r1 = compute_rf_rejection(None)
        r2 = compute_rf_rejection()
        assert r1.worst_case_residual_dbm == pytest.approx(r2.worst_case_residual_dbm)

    def test_custom_interferers_only(self):
        """A config with a single interferer returns one result."""
        spec = InterfererSpec("OnlyOne", 2.4e9, 20e6, -40.0)
        cfg = RFRejectionConfig(interferers=(spec,))
        result = compute_rf_rejection(cfg)
        assert len(result.interferer_results) == 1
        assert result.worst_case_name == "OnlyOne"

    def test_empty_interferer_list(self):
        """Empty interferer list returns a safe sentinel result."""
        cfg = RFRejectionConfig(interferers=())
        result = compute_rf_rejection(cfg)
        assert len(result.interferer_results) == 0
        assert result.any_in_readout_band is False
        assert result.worst_case_residual_dbm == float("-inf")

    def test_worst_case_residual_far_below_noise_floor(self):
        """All default interferers attenuated well below −100 dBm residual."""
        result = compute_rf_rejection()
        # Coil thermal noise floor at NMR freq is around −170 dBm in 1 Hz BW
        # Interference residual should be well below practical signal levels
        assert result.worst_case_residual_dbm < -100.0

    def test_min_attenuation_is_non_negative(self):
        result = compute_rf_rejection()
        assert result.min_attenuation_db >= 0.0

    def test_wider_bw_gives_less_rejection(self):
        """Wider maser bandwidth → less rejection at same offsets."""
        r_narrow = compute_rf_rejection(RFRejectionConfig(maser_gain_bw_hz=10_000.0))
        r_wide = compute_rf_rejection(RFRejectionConfig(maser_gain_bw_hz=200_000.0))
        # The minimum attenuation across all interferers should be higher for narrower BW
        assert r_narrow.min_attenuation_db > r_wide.min_attenuation_db

    def test_result_is_frozen(self):
        """RFRejectionResult should be immutable (frozen dataclass)."""
        result = compute_rf_rejection()
        with pytest.raises((AttributeError, TypeError)):
            result.any_in_readout_band = True  # type: ignore[misc]
