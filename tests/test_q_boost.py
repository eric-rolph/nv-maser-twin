"""
Tests for q_boost.py — electronic Q-boosting (active cavity-loss compensation).

Covers:
  - QBoostResult dataclass (frozen, types, fields)
  - compute_q_boost: derived values, formulas, edge cases, validation
  - compute_minimum_boost: threshold boost computation
  - compute_noise_temperature_boosted: Wang Eq. 4 with Q-boost, SQL approach
  - compute_sql_limit_ratio: ratio T_a/T_SQL
  - Wang 2024 quantitative validation (Q_L=1.1×10⁴ → 6.5×10⁵, B≈59)
"""
from __future__ import annotations

import math

import pytest

from src.nv_maser.physics.q_boost import (
    QBoostResult,
    compute_minimum_boost,
    compute_noise_temperature_boosted,
    compute_q_boost,
    compute_sql_limit_ratio,
)


# ── Shared constants ───────────────────────────────────────────────────────

# Wang et al. (2024) experimental parameters
_WANG_QL = 11_000.0        # passive loaded Q
_WANG_BETA = 1.0           # critical coupling
_WANG_QL_BOOSTED = 650_000.0  # boosted Q_L reported in paper
_WANG_BOOST = _WANG_QL_BOOSTED / _WANG_QL  # ≈ 59.1
_WANG_FC_GHZ = 9.4056       # cavity frequency


# ═══════════════════════════════════════════════════════════════════════════
# 1. QBoostResult dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestQBoostResultDataclass:
    def test_is_frozen(self):
        r = compute_q_boost(_WANG_QL, _WANG_BETA, boost_factor=2.0)
        with pytest.raises((AttributeError, TypeError)):
            r.boost_factor = 1.0  # type: ignore[misc]

    def test_all_fields_are_float(self):
        r = compute_q_boost(_WANG_QL, _WANG_BETA, boost_factor=2.0)
        for field in (
            r.q_l_native, r.q0_native, r.boost_factor, r.gain_feedback,
            r.q_l_effective, r.q0_effective, r.threshold_qm,
            r.threshold_reduction_factor,
        ):
            assert isinstance(field, float)

    def test_repr_contains_boost_factor(self):
        r = compute_q_boost(_WANG_QL, _WANG_BETA, boost_factor=2.0)
        assert "boost_factor" in repr(r)

    def test_type_is_q_boost_result(self):
        r = compute_q_boost(_WANG_QL, _WANG_BETA, boost_factor=5.0)
        assert isinstance(r, QBoostResult)


# ═══════════════════════════════════════════════════════════════════════════
# 2. compute_q_boost — basic formulas
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeQBoost:
    def test_q0_native_formula(self):
        """Q₀ = Q_L × (1 + β)."""
        r = compute_q_boost(q_l_native=1000.0, coupling_beta=1.0, boost_factor=1.0)
        assert r.q0_native == pytest.approx(2000.0)

    def test_q0_native_beta_two(self):
        r = compute_q_boost(q_l_native=1000.0, coupling_beta=2.0, boost_factor=1.0)
        assert r.q0_native == pytest.approx(3000.0)

    def test_no_boost_gives_native_q_l(self):
        """B=1 → Q_L_eff = Q_L_native."""
        r = compute_q_boost(1000.0, 1.0, boost_factor=1.0)
        assert r.q_l_effective == pytest.approx(1000.0)

    def test_no_boost_gives_native_q0(self):
        r = compute_q_boost(1000.0, 1.0, boost_factor=1.0)
        assert r.q0_effective == pytest.approx(r.q0_native)

    def test_q_l_eff_equals_boost_times_native(self):
        q_l, B = 1000.0, 5.0
        r = compute_q_boost(q_l, 1.0, B)
        assert r.q_l_effective == pytest.approx(B * q_l)

    def test_q0_eff_equals_boost_times_q0_native(self):
        q_l, B = 1000.0, 5.0
        r = compute_q_boost(q_l, 1.0, B)
        assert r.q0_effective == pytest.approx(B * r.q0_native)

    def test_gain_feedback_formula(self):
        """g_fb = 1 - 1/B."""
        B = 4.0
        r = compute_q_boost(1000.0, 1.0, B)
        assert r.gain_feedback == pytest.approx(1.0 - 1.0 / B)

    def test_gain_feedback_zero_when_no_boost(self):
        r = compute_q_boost(1000.0, 1.0, boost_factor=1.0)
        assert r.gain_feedback == pytest.approx(0.0)

    def test_gain_feedback_approaches_one_for_large_boost(self):
        r = compute_q_boost(1000.0, 1.0, boost_factor=1_000_000.0)
        assert r.gain_feedback == pytest.approx(1.0, abs=1e-5)

    def test_threshold_qm_equals_q_l_eff(self):
        """threshold_qm = Q_L_eff (oscillation condition Q_m ≤ Q_L_eff)."""
        r = compute_q_boost(1000.0, 1.0, 5.0)
        assert r.threshold_qm == pytest.approx(r.q_l_effective)

    def test_threshold_reduction_factor_equals_boost(self):
        r = compute_q_boost(1000.0, 1.0, 10.0)
        assert r.threshold_reduction_factor == pytest.approx(10.0)

    def test_q_l_native_stored_correctly(self):
        r = compute_q_boost(12345.0, 1.5, 3.0)
        assert r.q_l_native == pytest.approx(12345.0)

    def test_boost_factor_stored_correctly(self):
        r = compute_q_boost(1000.0, 1.0, 7.0)
        assert r.boost_factor == pytest.approx(7.0)

    def test_ratio_q0_to_q_l_preserved_after_boost(self):
        """Q₀_eff / Q_L_eff = Q₀_native / Q_L_native (= 1 + β)."""
        r = compute_q_boost(1000.0, 1.5, 10.0)
        assert r.q0_effective / r.q_l_effective == pytest.approx(
            r.q0_native / r.q_l_native
        )


class TestComputeQBoostValidation:
    def test_raises_on_negative_q_l(self):
        with pytest.raises(ValueError):
            compute_q_boost(-100.0, 1.0, 2.0)

    def test_raises_on_zero_q_l(self):
        with pytest.raises(ValueError):
            compute_q_boost(0.0, 1.0, 2.0)

    def test_raises_on_negative_beta(self):
        with pytest.raises(ValueError):
            compute_q_boost(1000.0, -1.0, 2.0)

    def test_raises_on_zero_beta(self):
        with pytest.raises(ValueError):
            compute_q_boost(1000.0, 0.0, 2.0)

    def test_raises_on_boost_less_than_one(self):
        with pytest.raises(ValueError):
            compute_q_boost(1000.0, 1.0, 0.5)

    def test_boost_exactly_one_is_allowed(self):
        r = compute_q_boost(1000.0, 1.0, 1.0)
        assert r.boost_factor == pytest.approx(1.0)


class TestComputeQBoostWang2024:
    def test_wang_q_l_effective(self):
        """Wang 2024: Q_L = 1.1×10⁴ boosted to 6.5×10⁵."""
        r = compute_q_boost(_WANG_QL, _WANG_BETA, _WANG_BOOST)
        assert r.q_l_effective == pytest.approx(_WANG_QL_BOOSTED, rel=0.01)

    def test_wang_q0_native(self):
        """β=1 → Q₀ = 2 × Q_L = 2.2×10⁴."""
        r = compute_q_boost(_WANG_QL, _WANG_BETA, _WANG_BOOST)
        assert r.q0_native == pytest.approx(2 * _WANG_QL)

    def test_wang_boost_factor_approx_59(self):
        r = compute_q_boost(_WANG_QL, _WANG_BETA, _WANG_BOOST)
        assert r.boost_factor == pytest.approx(59.09, rel=0.01)

    def test_wang_threshold_qm_equals_boosted_q_l(self):
        r = compute_q_boost(_WANG_QL, _WANG_BETA, _WANG_BOOST)
        assert r.threshold_qm == pytest.approx(r.q_l_effective)

    def test_wang_qm_1589_below_threshold_after_boost(self):
        """Q_m ≈ 1589 < Q_L_eff ≈ 6.5×10⁵ → oscillation regime with boost."""
        r = compute_q_boost(_WANG_QL, _WANG_BETA, _WANG_BOOST)
        qm_wang = 1589.0
        assert qm_wang < r.threshold_qm


# ═══════════════════════════════════════════════════════════════════════════
# 3. compute_minimum_boost
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeMinimumBoost:
    def test_already_above_threshold_returns_one(self):
        """Q_m ≤ Q_L → already oscillating, no boost needed → B_min = 1."""
        assert compute_minimum_boost(q_m=500.0, q_l=1000.0) == pytest.approx(1.0)

    def test_equal_returns_one(self):
        assert compute_minimum_boost(q_m=1000.0, q_l=1000.0) == pytest.approx(1.0)

    def test_formula_when_above_passive_threshold(self):
        """Q_m = 2 × Q_L → B_min = 2."""
        assert compute_minimum_boost(q_m=2000.0, q_l=1000.0) == pytest.approx(2.0)

    def test_large_q_m_requires_large_boost(self):
        b = compute_minimum_boost(q_m=100_000.0, q_l=1000.0)
        assert b == pytest.approx(100.0)

    def test_zero_q_l_returns_inf(self):
        assert compute_minimum_boost(q_m=1000.0, q_l=0.0) == float("inf")

    def test_result_at_least_one(self):
        """B_min is always ≥ 1."""
        for q_m, q_l in [(100, 200), (200, 200), (300, 200), (1000, 50)]:
            assert compute_minimum_boost(q_m, q_l) >= 1.0

    def test_wang_2024_minimum_boost(self):
        """Wang 2024: Q_m ≈ 1589, Q_L = 1.1×10⁴ → B_min = 1589/11000 < 1 → returns 1."""
        b = compute_minimum_boost(q_m=1589.0, q_l=_WANG_QL)
        # Q_m < Q_L → already at threshold → B_min = 1
        assert b == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. compute_noise_temperature_boosted
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeNoiseTempBoosted:
    # Shortcut: Q_boost with large boost factor for "infinite Q" limit
    @pytest.fixture
    def large_boost(self):
        return compute_q_boost(1000.0, 1.0, boost_factor=1_000.0)

    @pytest.fixture
    def small_boost(self):
        return compute_q_boost(1000.0, 1.0, boost_factor=2.0)

    def test_returns_finite_for_valid_input(self, small_boost):
        # small_boost: Q₀_eff = 2×2000 = 4000; use Q_m=1000 < Q₀_eff
        t = compute_noise_temperature_boosted(
            magnetic_q=1000.0,
            q_boost_result=small_boost,
            spin_temperature_k=1.0,
            bath_temperature_k=300.0,
        )
        assert math.isfinite(t)

    def test_nan_when_qm_exceeds_q0_eff(self, small_boost):
        """Q_m ≥ Q₀_eff → at or above oscillation threshold → nan."""
        t = compute_noise_temperature_boosted(
            magnetic_q=10_000_000.0,  # >> Q₀_eff
            q_boost_result=small_boost,
            spin_temperature_k=1.0,
        )
        assert math.isnan(t)

    def test_nan_propagates_from_spin_temp(self, small_boost):
        t = compute_noise_temperature_boosted(
            magnetic_q=500.0,
            q_boost_result=small_boost,
            spin_temperature_k=float("nan"),
        )
        assert math.isnan(t)

    def test_approaches_spin_temperature_for_large_boost(self, large_boost):
        """T_a → T_s as B → ∞ (Q₀_eff → ∞, Q_m ≪ Q₀_eff)."""
        t_spin = 5.0   # K
        # Q₀_eff = 1000 × 2000 = 2e6; Q_m=500 → residual bath term ≈ 0.075 K
        q_m = 500.0
        t = compute_noise_temperature_boosted(q_m, large_boost, t_spin, 300.0)
        # 1% relative isn't tight enough for residual bath term; use abs tolerance
        assert t == pytest.approx(t_spin, abs=0.1)

    def test_lower_with_boost_than_without(self):
        """Boosting Q reduces T_a for the same Q_m (when Q_m is below both Q₀_eff)."""
        # No-boost: Q₀_eff = 2000; 100x boost: Q₀_eff = 200_000
        # Use Q_m=1500 < 2000 so both cases are sub-threshold and finite
        q_m = 1500.0
        rb_low = compute_q_boost(1000.0, 1.0, boost_factor=1.0)   # Q₀_eff = 2000
        rb_high = compute_q_boost(1000.0, 1.0, boost_factor=100.0)  # Q₀_eff = 200000
        t_na = compute_noise_temperature_boosted(q_m, rb_low, 1.0, 300.0)
        t_nb = compute_noise_temperature_boosted(q_m, rb_high, 1.0, 300.0)
        # Boosted: much larger denominator → smaller T_a
        assert math.isfinite(t_na)
        assert math.isfinite(t_nb)
        assert t_nb < t_na

    def test_positive_for_valid_below_threshold_input(self):
        rb = compute_q_boost(1000.0, 1.0, 10.0)
        t = compute_noise_temperature_boosted(1000.0, rb, 1.0, 300.0)
        assert t > 0.0

    def test_wang_2024_noise_temperature_order_of_magnitude(self):
        """Wang 2024 achieved T_a ≈ 172 K before boost; should be lower after."""
        rb = compute_q_boost(_WANG_QL, _WANG_BETA, _WANG_BOOST)
        # Q_m in amplifier regime: just below passive threshold
        q_m_passive = 1589.0  # just above Q_L_passive=1337, amplifier mode
        t = compute_noise_temperature_boosted(
            magnetic_q=q_m_passive,
            q_boost_result=rb,
            spin_temperature_k=-0.001,  # cold spins (Wang paper)
            bath_temperature_k=300.0,
        )
        # With huge Q-boost, T_a should be much less than passive 172 K
        assert math.isfinite(t)
        assert t < 172.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. compute_sql_limit_ratio
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeSqlLimitRatio:
    def test_returns_positive_for_valid_input(self):
        ratio = compute_sql_limit_ratio(noise_temperature_k=1.0, cavity_frequency_ghz=1.47)
        assert ratio > 0.0

    def test_ratio_below_one_means_sub_sql(self):
        """If T_a < T_SQL, ratio < 1 (quantum-limited amplifier achieved)."""
        # T_SQL at 1.47 GHz ≈ 35 mK
        t_a_sub_sql = 0.010  # 10 mK
        ratio = compute_sql_limit_ratio(t_a_sub_sql, 1.47)
        assert ratio < 1.0

    def test_ratio_one_means_at_sql(self):
        hbar = 1.054571817e-34
        kb = 1.380649e-23
        import math
        omega = 2 * math.pi * 1.47e9
        t_sql = hbar * omega / (2 * kb)
        ratio = compute_sql_limit_ratio(t_sql, 1.47)
        assert ratio == pytest.approx(1.0, rel=1e-9)

    def test_scales_inversely_with_frequency(self):
        """Higher frequency → larger T_SQL → smaller ratio for same T_a."""
        t_a = 1.0  # K
        ratio_low = compute_sql_limit_ratio(t_a, 1.0)
        ratio_high = compute_sql_limit_ratio(t_a, 10.0)
        assert ratio_low > ratio_high

    def test_zero_frequency_returns_inf(self):
        ratio = compute_sql_limit_ratio(1.0, 0.0)
        assert ratio == float("inf")
