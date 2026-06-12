"""Tests for up_conversion.py — RF up-conversion mixer model."""
from __future__ import annotations

import pytest

from nv_maser.physics.up_conversion import (
    DEFAULT_MIXER,
    MixerSpec,
    UpConversionNoiseContribution,
    UpConversionResult,
    compute_bandwidth_utilization,
    compute_lo_frequency_ghz,
    compute_mixer_noise_contribution,
    compute_up_conversion,
    friis_system_temperature_with_mixer,
)

# ─────────────────────────────────────────────────────────────────────────────
# Reference values for handheld probe
# ─────────────────────────────────────────────────────────────────────────────
NMR_FREQ_HZ = 2.129e6       # 50 mT Larmor frequency
MASER_FREQ_GHZ = 1.47       # NV maser frequency
MASER_BW_HZ = 340_000.0     # Maser 3-dB BW (~340 kHz from Wang 2024)
NMR_BW_HZ = 10_000.0        # 10 kHz readout bandwidth

_T0 = 290.0  # IEEE reference temperature
_KB = 1.380649e-23


# ═════════════════════════════════════════════════════════════════════════════
# TestMixerSpec
# ═════════════════════════════════════════════════════════════════════════════


class TestMixerSpec:

    def test_default_spec_creates(self):
        m = MixerSpec()
        assert m.conversion_loss_db == 6.5

    def test_effective_nf_defaults_to_conv_loss(self):
        m = MixerSpec(conversion_loss_db=7.0)
        assert m.effective_nf_db == 7.0

    def test_explicit_nf_overrides_default(self):
        m = MixerSpec(conversion_loss_db=6.5, noise_figure_ssb_db=8.0)
        assert m.effective_nf_db == 8.0

    def test_negative_conv_loss_raises(self):
        with pytest.raises(ValueError, match="loss"):
            MixerSpec(conversion_loss_db=-1.0)

    def test_negative_nf_raises(self):
        with pytest.raises(ValueError, match="noise"):
            MixerSpec(conversion_loss_db=6.0, noise_figure_ssb_db=-0.5)

    def test_zero_conv_loss_allowed(self):
        m = MixerSpec(conversion_loss_db=0.0)
        assert m.conversion_loss_db == 0.0

    def test_frozen_immutable(self):
        m = MixerSpec()
        with pytest.raises(Exception):
            m.conversion_loss_db = 99.0  # type: ignore[misc]


# ═════════════════════════════════════════════════════════════════════════════
# TestComputeUpConversion
# ═════════════════════════════════════════════════════════════════════════════


class TestComputeUpConversion:

    def test_returns_result(self):
        r = compute_up_conversion(NMR_FREQ_HZ, MASER_FREQ_GHZ, NMR_BW_HZ, MASER_BW_HZ)
        assert isinstance(r, UpConversionResult)

    def test_usb_equals_maser_frequency(self):
        """USB must land exactly at the maser centre frequency."""
        r = compute_up_conversion(NMR_FREQ_HZ, MASER_FREQ_GHZ, NMR_BW_HZ, MASER_BW_HZ)
        assert abs(r.usb_frequency_ghz - MASER_FREQ_GHZ) < 1e-9

    def test_lo_formula(self):
        r = compute_up_conversion(NMR_FREQ_HZ, MASER_FREQ_GHZ, NMR_BW_HZ, MASER_BW_HZ)
        expected_lo = MASER_FREQ_GHZ - NMR_FREQ_HZ / 1e9
        assert abs(r.lo_frequency_ghz - expected_lo) < 1e-9

    def test_lsb_formula(self):
        r = compute_up_conversion(NMR_FREQ_HZ, MASER_FREQ_GHZ, NMR_BW_HZ, MASER_BW_HZ)
        expected_lsb = r.lo_frequency_ghz - NMR_FREQ_HZ / 1e9
        assert abs(r.lsb_frequency_ghz - expected_lsb) < 1e-9

    def test_image_equals_lsb(self):
        r = compute_up_conversion(NMR_FREQ_HZ, MASER_FREQ_GHZ, NMR_BW_HZ, MASER_BW_HZ)
        assert abs(r.image_frequency_ghz - r.lsb_frequency_ghz) < 1e-12

    def test_noise_temperature_positive(self):
        r = compute_up_conversion(NMR_FREQ_HZ, MASER_FREQ_GHZ, NMR_BW_HZ, MASER_BW_HZ)
        assert r.noise_temperature_k > 0

    def test_noise_temperature_formula(self):
        """T_mixer = T₀ · (10^(NF/10) − 1)."""
        mixer = MixerSpec(conversion_loss_db=6.5, noise_figure_ssb_db=8.0)
        r = compute_up_conversion(NMR_FREQ_HZ, MASER_FREQ_GHZ, NMR_BW_HZ, MASER_BW_HZ, mixer)
        expected_t = _T0 * (10.0 ** (8.0 / 10.0) - 1.0)
        assert abs(r.noise_temperature_k - expected_t) < 0.01

    def test_bandwidth_utilization_small(self):
        """10 kHz NMR BW inside 340 kHz maser BW → << 1."""
        r = compute_up_conversion(NMR_FREQ_HZ, MASER_FREQ_GHZ, NMR_BW_HZ, MASER_BW_HZ)
        assert r.bandwidth_utilization < 1.0
        assert abs(r.bandwidth_utilization - NMR_BW_HZ / MASER_BW_HZ) < 1e-9

    def test_bw_utilization_gt_one_when_nmr_wider(self):
        r = compute_up_conversion(NMR_FREQ_HZ, MASER_FREQ_GHZ, 400_000.0, MASER_BW_HZ)
        assert r.bandwidth_utilization > 1.0

    def test_default_mixer_used_when_none(self):
        r = compute_up_conversion(NMR_FREQ_HZ, MASER_FREQ_GHZ, NMR_BW_HZ, MASER_BW_HZ)
        assert r.conversion_loss_db == DEFAULT_MIXER.conversion_loss_db

    def test_stored_fields(self):
        r = compute_up_conversion(NMR_FREQ_HZ, MASER_FREQ_GHZ, NMR_BW_HZ, MASER_BW_HZ)
        assert r.nmr_frequency_hz == NMR_FREQ_HZ
        assert r.maser_frequency_ghz == MASER_FREQ_GHZ
        assert r.nmr_bandwidth_hz == NMR_BW_HZ
        assert r.maser_bandwidth_hz == MASER_BW_HZ

    def test_validation_zero_nmr_freq_raises(self):
        with pytest.raises(ValueError, match="nmr_frequency"):
            compute_up_conversion(0.0, MASER_FREQ_GHZ, NMR_BW_HZ, MASER_BW_HZ)

    def test_validation_zero_maser_freq_raises(self):
        with pytest.raises(ValueError, match="maser_frequency"):
            compute_up_conversion(NMR_FREQ_HZ, 0.0, NMR_BW_HZ, MASER_BW_HZ)

    def test_validation_negative_nmr_bw_raises(self):
        with pytest.raises(ValueError, match="nmr_bandwidth"):
            compute_up_conversion(NMR_FREQ_HZ, MASER_FREQ_GHZ, -1.0, MASER_BW_HZ)

    def test_validation_zero_maser_bw_raises(self):
        with pytest.raises(ValueError, match="maser_bandwidth"):
            compute_up_conversion(NMR_FREQ_HZ, MASER_FREQ_GHZ, NMR_BW_HZ, 0.0)

    def test_lo_is_near_maser_frequency(self):
        """For NMR at ~2 MHz and maser at 1.47 GHz, LO ≈ f_maser."""
        r = compute_up_conversion(NMR_FREQ_HZ, MASER_FREQ_GHZ, NMR_BW_HZ, MASER_BW_HZ)
        # LO should be very close to maser freq (NMR freq << maser freq)
        assert abs(r.lo_frequency_ghz - MASER_FREQ_GHZ) < 0.01  # within 10 MHz

    def test_result_is_immutable(self):
        r = compute_up_conversion(NMR_FREQ_HZ, MASER_FREQ_GHZ, NMR_BW_HZ, MASER_BW_HZ)
        with pytest.raises(Exception):
            r.usb_frequency_ghz = 99.0  # type: ignore[misc]


# ═════════════════════════════════════════════════════════════════════════════
# TestMixerNoiseContribution
# ═════════════════════════════════════════════════════════════════════════════


class TestMixerNoiseContribution:

    def test_returns_result(self):
        r = compute_mixer_noise_contribution(DEFAULT_MIXER, 10_000.0)
        assert isinstance(r, UpConversionNoiseContribution)

    def test_thermal_noise_formula(self):
        bw = 10_000.0
        r = compute_mixer_noise_contribution(DEFAULT_MIXER, bw)
        expected = _KB * _T0 * bw
        assert abs(r.thermal_noise_w - expected) / expected < 1e-9

    def test_mixer_noise_formula(self):
        bw = 10_000.0
        mixer = MixerSpec(noise_figure_ssb_db=8.0)
        r = compute_mixer_noise_contribution(mixer, bw)
        t_mixer = _T0 * (10.0 ** (8.0 / 10.0) - 1.0)
        expected = _KB * t_mixer * bw
        assert abs(r.mixer_added_noise_w - expected) / expected < 1e-9

    def test_total_is_sum(self):
        r = compute_mixer_noise_contribution(DEFAULT_MIXER, 10_000.0)
        assert abs(r.total_input_referred_noise_w - (r.thermal_noise_w + r.mixer_added_noise_w)) < 1e-30

    def test_bandwidth_scales_noise(self):
        r1 = compute_mixer_noise_contribution(DEFAULT_MIXER, 1_000.0)
        r2 = compute_mixer_noise_contribution(DEFAULT_MIXER, 10_000.0)
        assert abs(r2.total_input_referred_noise_w / r1.total_input_referred_noise_w - 10.0) < 1e-9

    def test_validation_zero_bw_raises(self):
        with pytest.raises(ValueError):
            compute_mixer_noise_contribution(DEFAULT_MIXER, 0.0)

    def test_noise_temperature_matches(self):
        mixer = MixerSpec(noise_figure_ssb_db=8.0)
        r = compute_mixer_noise_contribution(mixer, 10_000.0)
        expected_t = _T0 * (10.0 ** (8.0 / 10.0) - 1.0)
        assert abs(r.noise_temperature_k - expected_t) < 0.01


# ═════════════════════════════════════════════════════════════════════════════
# TestHelperFunctions
# ═════════════════════════════════════════════════════════════════════════════


class TestLOFrequency:

    def test_lo_formula(self):
        lo = compute_lo_frequency_ghz(NMR_FREQ_HZ, MASER_FREQ_GHZ)
        assert abs(lo - (MASER_FREQ_GHZ - NMR_FREQ_HZ / 1e9)) < 1e-12

    def test_lo_positive(self):
        lo = compute_lo_frequency_ghz(NMR_FREQ_HZ, MASER_FREQ_GHZ)
        assert lo > 0

    def test_lo_just_below_maser(self):
        lo = compute_lo_frequency_ghz(NMR_FREQ_HZ, MASER_FREQ_GHZ)
        assert lo < MASER_FREQ_GHZ


class TestBandwidthUtilization:

    def test_equal_bw_gives_one(self):
        bw = compute_bandwidth_utilization(100_000.0, 100_000.0)
        assert abs(bw - 1.0) < 1e-9

    def test_narrow_nmr_gives_small_fraction(self):
        bw = compute_bandwidth_utilization(10_000.0, 340_000.0)
        assert bw < 0.1

    def test_wide_nmr_gives_gt_one(self):
        bw = compute_bandwidth_utilization(500_000.0, 340_000.0)
        assert bw > 1.0

    def test_validation_raises_on_zero(self):
        with pytest.raises(ValueError):
            compute_bandwidth_utilization(0.0, 340_000.0)
        with pytest.raises(ValueError):
            compute_bandwidth_utilization(10_000.0, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# TestFriisSystemTemperature
# ═════════════════════════════════════════════════════════════════════════════


class TestFriisSystemTemperature:

    def test_returns_positive(self):
        t_sys = friis_system_temperature_with_mixer(
            coil_temperature_k=300.0,
            mixer=DEFAULT_MIXER,
            maser_noise_temperature_k=5.0,
            maser_gain_db=30.0,
            lna_noise_temperature_k=75.0,
        )
        assert t_sys > 0

    def test_coil_dominates_at_300k(self):
        """At room temperature the coil noise should dominate the total."""
        t_sys = friis_system_temperature_with_mixer(
            coil_temperature_k=300.0,
            mixer=DEFAULT_MIXER,
            maser_noise_temperature_k=5.0,
            maser_gain_db=40.0,
            lna_noise_temperature_k=75.0,
        )
        # Coil at 300 K is the dominant term; system temp > 300 K
        assert t_sys > 300.0

    def test_high_maser_gain_suppresses_lna(self):
        """Increasing maser gain should reduce the LNA contribution to T_sys."""
        common = dict(
            coil_temperature_k=300.0,
            mixer=DEFAULT_MIXER,
            maser_noise_temperature_k=5.0,
            lna_noise_temperature_k=500.0,
        )
        t_low_gain = friis_system_temperature_with_mixer(**common, maser_gain_db=10.0)
        t_high_gain = friis_system_temperature_with_mixer(**common, maser_gain_db=50.0)
        assert t_high_gain < t_low_gain

    def test_ideal_zero_noise_mixer_zero_noise_maser(self):
        """With noiseless mixer and maser, T_sys = coil_temperature."""
        noiseless = MixerSpec(conversion_loss_db=0.0, noise_figure_ssb_db=0.0)
        t_sys = friis_system_temperature_with_mixer(
            coil_temperature_k=300.0,
            mixer=noiseless,
            maser_noise_temperature_k=0.0,
            maser_gain_db=40.0,
            lna_noise_temperature_k=0.0,
        )
        assert abs(t_sys - 300.0) < 1e-9

    def test_cooled_coil_reduces_t_sys(self):
        """Cooling the coil should reduce total system noise temperature."""
        common = dict(
            mixer=DEFAULT_MIXER,
            maser_noise_temperature_k=5.0,
            maser_gain_db=30.0,
            lna_noise_temperature_k=75.0,
        )
        t_warm = friis_system_temperature_with_mixer(coil_temperature_k=300.0, **common)
        t_cool = friis_system_temperature_with_mixer(coil_temperature_k=200.0, **common)
        assert t_cool < t_warm
