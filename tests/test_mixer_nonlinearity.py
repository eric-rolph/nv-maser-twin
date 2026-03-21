"""Tests for physics/mixer_nonlinearity.py — Mixer IMD3 model (R9).

Coverage matrix
---------------
compute_imd3_power_dbm
  ✓ equal-tone formula reduces to 3P − 2·IIP3
  ✓ 2f1-f2 product with unequal tones (P1 dominant)
  ✓ 2f2-f1 product with unequal tones (P2 dominant)
  ✓ stronger input → higher IMD3 power
  ✓ higher IIP3 → lower IMD3 power
  ✓ power scales 2:1 with dominant tone (1 dB dominant → 2 dB IM3)
  ✓ power scales 1:1 with secondary tone
  ✓ raises ValueError on bad product_type

compute_imd3_frequency_hz
  ✓ 2f1-f2 formula is exact
  ✓ 2f2-f1 formula is exact
  ✓ equal-frequency tones → both products equal f
  ✓ negative result possible (f2 > 2*f1)
  ✓ raises ValueError on bad product_type

compute_imd3_pair
  ✓ returns exactly 2 products
  ✓ product types are "2f1-f2" and "2f2-f1"
  ✓ interferer_1 ordering preserved
  ✓ is_physical False when product_freq ≤ 0
  ✓ in_maser_band True when product frequency inside BW
  ✓ in_maser_band False when product frequency outside BW
  ✓ non-physical product → in_maser_band False
  ✓ freq_offset_from_maser_hz correct for physical product

MixerNonlinearityConfig
  ✓ default iip3_dbm = 5.0
  ✓ default maser_center_hz = 1.4699e9
  ✓ default maser_gain_bw_hz = 49_000.0
  ✓ default interferers = 8 hospital sources
  ✓ raises ValueError on non-positive maser_center_hz
  ✓ raises ValueError on non-positive maser_gain_bw_hz
  ✓ custom interferers accepted

compute_mixer_nonlinearity (default config)
  ✓ returns MixerNonlinearityResult
  ✓ n_pairs_evaluated == C(8,2) == 28
  ✓ n_products_evaluated == 56
  ✓ any_in_band is False for default 8 interferers
  ✓ worst_in_band_power_dbm == -inf when no in-band products
  ✓ in_band_products is empty tuple
  ✓ max_imd3_power_dbm is finite (worst physical product computed)
  ✓ None config uses default
  ✓ stronger dominant tone → higher max_imd3

compute_mixer_nonlinearity (custom in-band config)
  ✓ two interferers near maser centre → any_in_band True
  ✓ in_band_products non-empty
  ✓ worst_in_band_power_dbm equals max power of those in-band
  ✓ worst_in_band_power_dbm <= max_imd3_power_dbm

topology / pair counting
  ✓ N=1 interferer → 0 pairs, 0 products
  ✓ N=2 interferers → 1 pair, 2 products
  ✓ N=3 interferers → 3 pairs, 6 products
  ✓ each unordered pair appears exactly once

IMD3Product fields
  ✓ product_type tag matches frequency formula
  ✓ power values roundtrip with compute_imd3_power_dbm
"""
from __future__ import annotations

import math
import pytest

from nv_maser.physics.mixer_nonlinearity import (
    IMD3Product,
    MixerNonlinearityConfig,
    MixerNonlinearityResult,
    PRODUCT_2F1_MINUS_F2,
    PRODUCT_2F2_MINUS_F1,
    compute_imd3_frequency_hz,
    compute_imd3_pair,
    compute_imd3_power_dbm,
    compute_mixer_nonlinearity,
)
from nv_maser.physics.rf_rejection import InterfererSpec

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_spec(
    freq_hz: float,
    power_dbm: float,
    name: str = "test",
) -> InterfererSpec:
    return InterfererSpec(
        name=name,
        center_freq_hz=freq_hz,
        bandwidth_hz=1e6,
        power_dbm=power_dbm,
    )


MASER_CENTER = 1.4699e9
MASER_BW = 49_000.0
DEFAULT_IIP3 = 5.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  compute_imd3_power_dbm                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestComputeImd3PowerDbm:
    def test_equal_tones_formula(self) -> None:
        """Equal tones: P_IM3 = 3P − 2·IIP3."""
        p = -30.0
        iip3 = 5.0
        expected = 3 * p - 2 * iip3  # = -100 dBm
        assert compute_imd3_power_dbm(p, p, iip3, PRODUCT_2F1_MINUS_F2) == pytest.approx(expected)
        assert compute_imd3_power_dbm(p, p, iip3, PRODUCT_2F2_MINUS_F1) == pytest.approx(expected)

    def test_unequal_tones_2f1_minus_f2(self) -> None:
        """2f1-f2 product: P_IM3 = 2·P1 + P2 − 2·IIP3."""
        p1, p2, iip3 = -30.0, -40.0, 5.0
        expected = 2 * p1 + p2 - 2 * iip3  # = −110 dBm
        assert compute_imd3_power_dbm(p1, p2, iip3, PRODUCT_2F1_MINUS_F2) == pytest.approx(expected)

    def test_unequal_tones_2f2_minus_f1(self) -> None:
        """2f2-f1 product: P_IM3 = P1 + 2·P2 − 2·IIP3."""
        p1, p2, iip3 = -30.0, -40.0, 5.0
        expected = p1 + 2 * p2 - 2 * iip3  # = −120 dBm
        assert compute_imd3_power_dbm(p1, p2, iip3, PRODUCT_2F2_MINUS_F1) == pytest.approx(expected)

    def test_stronger_input_higher_imd3_2f1_f2(self) -> None:
        """Stronger P1 → higher IMD3 power for 2f1-f2 product."""
        iip3 = 5.0
        p_weak = compute_imd3_power_dbm(-40.0, -40.0, iip3, PRODUCT_2F1_MINUS_F2)
        p_strong = compute_imd3_power_dbm(-30.0, -40.0, iip3, PRODUCT_2F1_MINUS_F2)
        assert p_strong > p_weak

    def test_stronger_input_higher_imd3_2f2_f1(self) -> None:
        """Stronger P2 → higher IMD3 power for 2f2-f1 product."""
        iip3 = 5.0
        p_weak = compute_imd3_power_dbm(-40.0, -40.0, iip3, PRODUCT_2F2_MINUS_F1)
        p_strong = compute_imd3_power_dbm(-40.0, -30.0, iip3, PRODUCT_2F2_MINUS_F1)
        assert p_strong > p_weak

    def test_higher_iip3_lower_imd3(self) -> None:
        """Higher IIP3 → weaker IMD3 (better linearity)."""
        p1, p2 = -30.0, -40.0
        p_bad = compute_imd3_power_dbm(p1, p2, iip3_dbm=0.0)
        p_good = compute_imd3_power_dbm(p1, p2, iip3_dbm=20.0)
        assert p_good < p_bad

    def test_dominant_tone_2_to_1_slope(self) -> None:
        """Increasing dominant tone 1 dB raises 2f1-f2 product by 2 dB."""
        iip3 = 5.0
        p2 = -40.0
        for p1_base in (-50.0, -40.0, -30.0):
            base = compute_imd3_power_dbm(p1_base, p2, iip3, PRODUCT_2F1_MINUS_F2)
            bumped = compute_imd3_power_dbm(p1_base + 1.0, p2, iip3, PRODUCT_2F1_MINUS_F2)
            assert bumped - base == pytest.approx(2.0)

    def test_secondary_tone_1_to_1_slope(self) -> None:
        """Increasing secondary tone 1 dB raises 2f1-f2 product by 1 dB."""
        iip3 = 5.0
        p1 = -30.0
        for p2_base in (-60.0, -50.0):
            base = compute_imd3_power_dbm(p1, p2_base, iip3, PRODUCT_2F1_MINUS_F2)
            bumped = compute_imd3_power_dbm(p1, p2_base + 1.0, iip3, PRODUCT_2F1_MINUS_F2)
            assert bumped - base == pytest.approx(1.0)

    def test_bad_product_type_raises(self) -> None:
        with pytest.raises(ValueError, match="product_type"):
            compute_imd3_power_dbm(-30.0, -40.0, 5.0, "bad_type")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  compute_imd3_frequency_hz                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestComputeImd3FrequencyHz:
    def test_2f1_minus_f2_exact(self) -> None:
        f1, f2 = 2412e6, 2437e6
        assert compute_imd3_frequency_hz(f1, f2, PRODUCT_2F1_MINUS_F2) == pytest.approx(2 * f1 - f2)

    def test_2f2_minus_f1_exact(self) -> None:
        f1, f2 = 2412e6, 2437e6
        assert compute_imd3_frequency_hz(f1, f2, PRODUCT_2F2_MINUS_F1) == pytest.approx(2 * f2 - f1)

    def test_equal_frequencies(self) -> None:
        """Both IM3 products land at f when f1 == f2."""
        f = 2.4e9
        assert compute_imd3_frequency_hz(f, f, PRODUCT_2F1_MINUS_F2) == pytest.approx(f)
        assert compute_imd3_frequency_hz(f, f, PRODUCT_2F2_MINUS_F1) == pytest.approx(f)

    def test_negative_product_possible(self) -> None:
        """f2 > 2*f1 → 2f1-f2 < 0 (non-physical, allowed algebraically)."""
        f1, f2 = 100e6, 300e6  # 2*f1 = 200 MHz < 300 MHz
        result = compute_imd3_frequency_hz(f1, f2, PRODUCT_2F1_MINUS_F2)
        assert result < 0.0

    def test_bad_product_type_raises(self) -> None:
        with pytest.raises(ValueError, match="product_type"):
            compute_imd3_frequency_hz(1e9, 2e9, "3f1")

    def test_default_product_type_is_2f1_f2(self) -> None:
        f1, f2 = 1.2e9, 1.5e9
        assert compute_imd3_frequency_hz(f1, f2) == pytest.approx(2 * f1 - f2)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  compute_imd3_pair                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestComputeImd3Pair:
    def _default_config(self) -> MixerNonlinearityConfig:
        return MixerNonlinearityConfig(
            iip3_dbm=DEFAULT_IIP3,
            maser_center_hz=MASER_CENTER,
            maser_gain_bw_hz=MASER_BW,
            interferers=(),  # not used in pair function
        )

    def test_returns_two_products(self) -> None:
        s1 = _make_spec(2412e6, -30.0)
        s2 = _make_spec(2437e6, -40.0)
        result = compute_imd3_pair(s1, s2, self._default_config())
        assert len(result) == 2

    def test_product_type_tags(self) -> None:
        s1 = _make_spec(2412e6, -30.0)
        s2 = _make_spec(2437e6, -40.0)
        pa, pb = compute_imd3_pair(s1, s2, self._default_config())
        assert pa.product_type == PRODUCT_2F1_MINUS_F2
        assert pb.product_type == PRODUCT_2F2_MINUS_F1

    def test_interferer_ordering_preserved(self) -> None:
        s1 = _make_spec(2412e6, -30.0, "A")
        s2 = _make_spec(2437e6, -40.0, "B")
        pa, pb = compute_imd3_pair(s1, s2, self._default_config())
        assert pa.interferer_1.name == "A"
        assert pa.interferer_2.name == "B"

    def test_is_physical_false_when_negative_freq(self) -> None:
        """LTE 700 MHz (f1) + LTE 2600 MHz (f2): 2f1-f2 < 0."""
        s1 = _make_spec(746e6, -60.0)
        s2 = _make_spec(2600e6, -60.0)
        pa, _ = compute_imd3_pair(s1, s2, self._default_config())
        assert pa.product_freq_hz < 0.0
        assert pa.is_physical is False

    def test_non_physical_not_in_maser_band(self) -> None:
        s1 = _make_spec(746e6, -60.0)
        s2 = _make_spec(2600e6, -60.0)
        pa, _ = compute_imd3_pair(s1, s2, self._default_config())
        assert pa.in_maser_band is False

    def test_in_band_when_product_within_bw(self) -> None:
        """Two tones slightly above maser centre → 2f1-f2 hits maser band."""
        # 2f1 - f2 = maser_centre if f2 = 2f1 - maser_centre
        # Use f1 = maser_centre + 1 kHz, f2 = maser_centre + 2 kHz
        # → product = 2*(maser+1k) - (maser+2k) = maser → exactly in band
        f1 = MASER_CENTER + 1_000.0
        f2 = MASER_CENTER + 2_000.0
        s1 = _make_spec(f1, -30.0)
        s2 = _make_spec(f2, -30.0)
        pa, _ = compute_imd3_pair(s1, s2, self._default_config())
        assert pa.in_maser_band is True
        assert pa.freq_offset_from_maser_hz == pytest.approx(0.0, abs=1.0)

    def test_out_of_band_when_product_far(self) -> None:
        s1 = _make_spec(2412e6, -30.0)
        s2 = _make_spec(2437e6, -40.0)
        cfg = self._default_config()
        pa, pb = compute_imd3_pair(s1, s2, cfg)
        # 2*2412 - 2437 = 2387 MHz → far from 1.4699 GHz
        assert pa.in_maser_band is False
        assert pb.in_maser_band is False

    def test_freq_offset_magnitude_correct(self) -> None:
        s1 = _make_spec(2412e6, -30.0)
        s2 = _make_spec(2437e6, -40.0)
        pa, _ = compute_imd3_pair(s1, s2, self._default_config())
        expected_offset = abs(2 * 2412e6 - 2437e6 - MASER_CENTER)
        assert pa.freq_offset_from_maser_hz == pytest.approx(expected_offset)

    def test_power_matches_formula(self) -> None:
        s1 = _make_spec(2412e6, -30.0)
        s2 = _make_spec(2437e6, -40.0)
        pa, pb = compute_imd3_pair(s1, s2, self._default_config())
        expected_a = compute_imd3_power_dbm(-30.0, -40.0, DEFAULT_IIP3, PRODUCT_2F1_MINUS_F2)
        expected_b = compute_imd3_power_dbm(-30.0, -40.0, DEFAULT_IIP3, PRODUCT_2F2_MINUS_F1)
        assert pa.product_power_dbm == pytest.approx(expected_a)
        assert pb.product_power_dbm == pytest.approx(expected_b)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MixerNonlinearityConfig                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestMixerNonlinearityConfig:
    def test_default_iip3(self) -> None:
        cfg = MixerNonlinearityConfig()
        assert cfg.iip3_dbm == pytest.approx(5.0)

    def test_default_maser_center(self) -> None:
        cfg = MixerNonlinearityConfig()
        assert cfg.maser_center_hz == pytest.approx(1.4699e9)

    def test_default_maser_bw(self) -> None:
        cfg = MixerNonlinearityConfig()
        assert cfg.maser_gain_bw_hz == pytest.approx(49_000.0)

    def test_default_interferers_count(self) -> None:
        cfg = MixerNonlinearityConfig()
        assert len(cfg.interferers) == 8

    def test_nonpositive_center_raises(self) -> None:
        with pytest.raises(ValueError, match="maser_center_hz"):
            MixerNonlinearityConfig(maser_center_hz=0.0)

    def test_nonpositive_bw_raises(self) -> None:
        with pytest.raises(ValueError, match="maser_gain_bw_hz"):
            MixerNonlinearityConfig(maser_gain_bw_hz=-1.0)

    def test_custom_interferers_accepted(self) -> None:
        s1 = _make_spec(1.0e9, -30.0)
        s2 = _make_spec(2.0e9, -40.0)
        cfg = MixerNonlinearityConfig(interferers=(s1, s2))
        assert len(cfg.interferers) == 2


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  compute_mixer_nonlinearity — default config                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestComputeMixerNonlinearityDefault:
    @pytest.fixture(scope="class")
    def result(self) -> MixerNonlinearityResult:
        return compute_mixer_nonlinearity()

    def test_returns_result_type(self, result: MixerNonlinearityResult) -> None:
        assert isinstance(result, MixerNonlinearityResult)

    def test_n_pairs_is_28(self, result: MixerNonlinearityResult) -> None:
        """C(8,2) = 28 unordered pairs from 8 default interferers."""
        assert result.n_pairs_evaluated == 28

    def test_n_products_is_56(self, result: MixerNonlinearityResult) -> None:
        """28 pairs × 2 products = 56."""
        assert result.n_products_evaluated == 56

    def test_imd3_products_length_matches(self, result: MixerNonlinearityResult) -> None:
        assert len(result.imd3_products) == result.n_products_evaluated

    def test_no_in_band_products_default(self, result: MixerNonlinearityResult) -> None:
        """All 8 default interferers are far from 1.4699 GHz — no IMD3 in band."""
        assert result.any_in_band is False

    def test_worst_in_band_power_is_neg_inf(self, result: MixerNonlinearityResult) -> None:
        assert result.worst_in_band_power_dbm == float("-inf")

    def test_in_band_products_empty(self, result: MixerNonlinearityResult) -> None:
        assert result.in_band_products == ()

    def test_max_imd3_power_finite(self, result: MixerNonlinearityResult) -> None:
        """There are physical products; max power must be finite."""
        assert math.isfinite(result.max_imd3_power_dbm)

    def test_none_config_uses_default(self) -> None:
        r1 = compute_mixer_nonlinearity(None)
        r2 = compute_mixer_nonlinearity(MixerNonlinearityConfig())
        assert r1.n_pairs_evaluated == r2.n_pairs_evaluated
        assert r1.any_in_band == r2.any_in_band
        assert r1.max_imd3_power_dbm == pytest.approx(r2.max_imd3_power_dbm)

    def test_stronger_dominant_raises_worst_power(self) -> None:
        """Boosting the strongest interferer raises max_imd3_power_dbm."""
        from nv_maser.physics.rf_rejection import _DEFAULT_INTERFERERS

        boosted = [
            InterfererSpec(name=n, center_freq_hz=f, bandwidth_hz=bw, power_dbm=p - 20.0
                           if p == -20.0 else p)  # leave Broadcast FM alone
            for n, f, bw, p in _DEFAULT_INTERFERERS
        ]
        # Actually just boost the first (WiFi 2.4 GHz from -30 to -10)
        boosted = [
            InterfererSpec(
                name=n,
                center_freq_hz=f,
                bandwidth_hz=bw,
                power_dbm=p + 20.0 if "WiFi 2.4 GHz" == n else p,
            )
            for n, f, bw, p in _DEFAULT_INTERFERERS
        ]
        cfg_boosted = MixerNonlinearityConfig(interferers=tuple(boosted))
        r_boosted = compute_mixer_nonlinearity(cfg_boosted)
        r_default = compute_mixer_nonlinearity()
        assert r_boosted.max_imd3_power_dbm > r_default.max_imd3_power_dbm


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  compute_mixer_nonlinearity — custom in-band config                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestComputeMixerNonlinearityInBand:
    """Force in-band IMD3 by constructing interferers near maser centre."""

    @pytest.fixture(scope="class")
    def in_band_result(self) -> MixerNonlinearityResult:
        # f1 = maser_centre + 1 kHz, f2 = maser_centre + 2 kHz
        # → 2f1-f2 = maser_centre → exactly in band
        f1 = MASER_CENTER + 1_000.0
        f2 = MASER_CENTER + 2_000.0
        s1 = _make_spec(f1, -30.0, "near_1")
        s2 = _make_spec(f2, -30.0, "near_2")
        cfg = MixerNonlinearityConfig(
            iip3_dbm=DEFAULT_IIP3,
            maser_center_hz=MASER_CENTER,
            maser_gain_bw_hz=MASER_BW,
            interferers=(s1, s2),
        )
        return compute_mixer_nonlinearity(cfg)

    def test_any_in_band_true(self, in_band_result: MixerNonlinearityResult) -> None:
        assert in_band_result.any_in_band is True

    def test_in_band_products_non_empty(self, in_band_result: MixerNonlinearityResult) -> None:
        assert len(in_band_result.in_band_products) >= 1

    def test_worst_in_band_power_finite(self, in_band_result: MixerNonlinearityResult) -> None:
        assert math.isfinite(in_band_result.worst_in_band_power_dbm)

    def test_worst_in_band_le_max_imd3(self, in_band_result: MixerNonlinearityResult) -> None:
        """In-band worst power cannot exceed global worst power."""
        assert in_band_result.worst_in_band_power_dbm <= in_band_result.max_imd3_power_dbm + 1e-9

    def test_worst_in_band_equals_max_of_in_band_set(
        self, in_band_result: MixerNonlinearityResult
    ) -> None:
        products = in_band_result.in_band_products
        assert in_band_result.worst_in_band_power_dbm == pytest.approx(
            max(p.product_power_dbm for p in products)
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Topology / pair counting                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestTopologyPairCounting:
    def _cfg(self, n: int) -> MixerNonlinearityConfig:
        specs = tuple(
            _make_spec(freq_hz=(1.0 + i * 0.1) * 1e9, power_dbm=-30.0, name=f"s{i}")
            for i in range(n)
        )
        return MixerNonlinearityConfig(interferers=specs)

    def test_n1_zero_pairs(self) -> None:
        r = compute_mixer_nonlinearity(self._cfg(1))
        assert r.n_pairs_evaluated == 0
        assert r.n_products_evaluated == 0

    def test_n2_one_pair_two_products(self) -> None:
        r = compute_mixer_nonlinearity(self._cfg(2))
        assert r.n_pairs_evaluated == 1
        assert r.n_products_evaluated == 2

    def test_n3_three_pairs_six_products(self) -> None:
        r = compute_mixer_nonlinearity(self._cfg(3))
        assert r.n_pairs_evaluated == 3
        assert r.n_products_evaluated == 6

    def test_each_unordered_pair_appears_once(self) -> None:
        """No pair is double-counted: both orderings should not appear."""
        r = compute_mixer_nonlinearity(self._cfg(4))
        seen: set[frozenset[str]] = set()
        for prod in r.imd3_products:
            pair_key = frozenset((prod.interferer_1.name, prod.interferer_2.name))
            # Each pair appears in exactly 2 products (the two IM3 sidebands)
            seen.add(pair_key)
        # 4 interferers → C(4,2)=6 unique pairs
        assert len(seen) == 6


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  IMD3Product field consistency                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestImd3ProductFieldConsistency:
    def _pair(self) -> tuple[IMD3Product, IMD3Product]:
        s1 = _make_spec(2.412e9, -30.0, "A")
        s2 = _make_spec(2.437e9, -40.0, "B")
        cfg = MixerNonlinearityConfig()
        return compute_imd3_pair(s1, s2, cfg)

    def test_product_type_matches_frequency_formula(self) -> None:
        s1 = _make_spec(2.412e9, -30.0)
        s2 = _make_spec(2.437e9, -40.0)
        pa, pb = self._pair()
        assert pa.product_freq_hz == pytest.approx(
            compute_imd3_frequency_hz(s1.center_freq_hz, s2.center_freq_hz, PRODUCT_2F1_MINUS_F2)
        )
        assert pb.product_freq_hz == pytest.approx(
            compute_imd3_frequency_hz(s1.center_freq_hz, s2.center_freq_hz, PRODUCT_2F2_MINUS_F1)
        )

    def test_power_roundtrips_with_formula(self) -> None:
        pa, pb = self._pair()
        assert pa.product_power_dbm == pytest.approx(
            compute_imd3_power_dbm(
                pa.interferer_1.power_dbm,
                pa.interferer_2.power_dbm,
                MixerNonlinearityConfig().iip3_dbm,
                PRODUCT_2F1_MINUS_F2,
            )
        )
        assert pb.product_power_dbm == pytest.approx(
            compute_imd3_power_dbm(
                pb.interferer_1.power_dbm,
                pb.interferer_2.power_dbm,
                MixerNonlinearityConfig().iip3_dbm,
                PRODUCT_2F2_MINUS_F1,
            )
        )
