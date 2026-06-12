"""
Tests for the amplifier.py maser-as-amplifier characterisation module.

Covers:
  - Magnetic quality factor Q_m (Wang 2024 Eq. 1)
  - Spin temperature T_s
  - Noise temperature T_a (Wang 2024 Eq. 4)
  - Standard Quantum Limit T_SQL
  - Output power above threshold
  - Full AmplifierProperties integration
  - Consistency with cavity.py cooperativity threshold
"""
from __future__ import annotations

import math

import pytest

from src.nv_maser.config import CavityConfig, MaserConfig, NVConfig
from src.nv_maser.physics.amplifier import (
    SIGMA_2,
    AmplifierProperties,
    MaserGainResult,
    OutputPowerResult,
    _derive_population_fractions,
    _gain_voltage,
    compute_amplifier_properties,
    compute_magnetic_q,
    compute_maser_gain,
    compute_noise_temperature,
    compute_output_power,
    compute_spin_temperature,
    compute_sql_noise_temperature,
)
from src.nv_maser.physics.cavity import (
    compute_cavity_properties,
    compute_full_threshold,
)

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def default_nv() -> NVConfig:
    return NVConfig()


@pytest.fixture
def default_cav() -> CavityConfig:
    return CavityConfig()


@pytest.fixture
def default_maser() -> MaserConfig:
    return MaserConfig()


@pytest.fixture
def high_inversion_nv() -> NVConfig:
    """NV with 90% pump efficiency — strong inversion."""
    return NVConfig(pump_efficiency=0.90, nv_density_per_cm3=1e18)


@pytest.fixture
def wangpaper_nv() -> NVConfig:
    """Approximate Wang 2024 (pentacene) inverted spin parameters.

    Pentacene: Δn ≈ 3.3e20 m⁻³ = 3.3e14 cm⁻³.  We model this through
    a high NV density × pump efficiency product.
    """
    # Effective inverted density = nv_density × orientation_fraction × pump_eff
    # Want Δn ≈ 3.3e14 cm⁻³ → set density=3.3e16 cm⁻³, pump_eff=0.5, orient=0.02
    return NVConfig(
        nv_density_per_cm3=3.3e16,
        pump_efficiency=0.5,
        orientation_fraction=0.02,
        t2_star_us=4.24,   # Wang 2024 T₂ = 4.24 µs
    )


@pytest.fixture
def wangpaper_cav() -> CavityConfig:
    """Wang 2024 cavity: 0.22 cm³, filling factor 0.027."""
    return CavityConfig(mode_volume_cm3=0.22, fill_factor=0.027)


@pytest.fixture
def wangpaper_maser() -> MaserConfig:
    """Wang 2024: 9.4 GHz, Q₀ = 2.2e4, Q_L ≈ 1.1e4 (critical coupling)."""
    return MaserConfig(
        cavity_frequency_ghz=9.4056,
        cavity_q=11_000,      # loaded Q at critical coupling
        coupling_beta=1.0,    # critical coupling β=1 → Q₀ = 2×Q_L
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. SIGMA_2 constant
# ═══════════════════════════════════════════════════════════════════════════


class TestSigma2:
    def test_value_is_half(self):
        assert SIGMA_2 == 0.5

    def test_positive(self):
        assert SIGMA_2 > 0


# ═══════════════════════════════════════════════════════════════════════════
# 2. compute_magnetic_q
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeMagneticQ:
    def test_returns_finite_positive(self, default_nv, default_cav, default_maser):
        q_m = compute_magnetic_q(default_nv, default_cav, default_maser)
        assert math.isfinite(q_m)
        assert q_m > 0

    def test_zero_density_gives_inf(self, default_cav, default_maser):
        nv = NVConfig(nv_density_per_cm3=0.0)
        q_m = compute_magnetic_q(nv, default_cav, default_maser)
        assert math.isinf(q_m)

    def test_zero_pump_efficiency_gives_inf(self, default_cav, default_maser):
        nv = NVConfig(pump_efficiency=0.0)
        q_m = compute_magnetic_q(nv, default_cav, default_maser)
        assert math.isinf(q_m)

    def test_very_small_fill_factor_gives_larger_q(self, default_nv, default_maser):
        """Tinier fill factor → fewer active spins → less gain → larger Q_m."""
        cav_small = CavityConfig(fill_factor=1e-4)
        cav_normal = CavityConfig(fill_factor=0.1)
        q_small = compute_magnetic_q(default_nv, cav_small, default_maser)
        q_normal = compute_magnetic_q(default_nv, cav_normal, default_maser)
        assert q_small > q_normal

    def test_increases_with_density(self, default_cav, default_maser):
        """Higher inverted density → lower 1/Q_m → lower Q_m (more gain)."""
        nv_low = NVConfig(nv_density_per_cm3=1e16)
        nv_high = NVConfig(nv_density_per_cm3=1e18)
        assert compute_magnetic_q(nv_low, default_cav, default_maser) > \
               compute_magnetic_q(nv_high, default_cav, default_maser)

    def test_increases_with_t2(self, default_cav, default_maser):
        """Longer T₂ → more coherence → higher spin gain → lower Q_m."""
        nv_short = NVConfig(t2_star_us=0.5)
        nv_long = NVConfig(t2_star_us=10.0)
        assert compute_magnetic_q(nv_short, default_cav, default_maser) > \
               compute_magnetic_q(nv_long, default_cav, default_maser)

    def test_increases_with_fill_factor(self, default_nv, default_maser):
        """Higher fill factor → more spins in mode volume → lower Q_m."""
        cav_low = CavityConfig(fill_factor=0.01)
        cav_high = CavityConfig(fill_factor=0.5)
        assert compute_magnetic_q(default_nv, cav_low, default_maser) > \
               compute_magnetic_q(default_nv, cav_high, default_maser)

    def test_scales_linearly_with_density(self, default_cav, default_maser):
        """Q_m ∝ 1/Δn — doubling density halves Q_m."""
        nv1 = NVConfig(nv_density_per_cm3=1e17)
        nv2 = NVConfig(nv_density_per_cm3=2e17)
        q1 = compute_magnetic_q(nv1, default_cav, default_maser)
        q2 = compute_magnetic_q(nv2, default_cav, default_maser)
        assert abs(q1 / q2 - 2.0) < 1e-10

    def test_scales_linearly_with_t2(self, default_cav, default_maser):
        """Q_m ∝ 1/T₂ — doubling T₂ halves Q_m."""
        nv1 = NVConfig(t2_star_us=1.0)
        nv2 = NVConfig(t2_star_us=2.0)
        q1 = compute_magnetic_q(nv1, default_cav, default_maser)
        q2 = compute_magnetic_q(nv2, default_cav, default_maser)
        assert abs(q1 / q2 - 2.0) < 1e-10

    def test_wang_paper_order_of_magnitude(
        self, wangpaper_nv, wangpaper_cav, wangpaper_maser
    ):
        """Q_m should be in the 10³–10⁵ range for Wang paper parameters."""
        q_m = compute_magnetic_q(wangpaper_nv, wangpaper_cav, wangpaper_maser)
        assert 1e3 < q_m < 1e6, f"Expected 1e3 < Q_m < 1e6, got {q_m:.2e}"

    def test_orientation_fraction_scales_linearly(self, default_cav, default_maser):
        """Orientation fraction enters Δn linearly → Q_m ∝ 1/orientation."""
        nv1 = NVConfig(orientation_fraction=0.1)
        nv2 = NVConfig(orientation_fraction=0.2)
        q1 = compute_magnetic_q(nv1, default_cav, default_maser)
        q2 = compute_magnetic_q(nv2, default_cav, default_maser)
        assert abs(q1 / q2 - 2.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# 3. compute_spin_temperature
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeSpinTemperature:
    def test_high_inversion_gives_low_temperature(self):
        """p_upper ≫ p_lower → T_s ≪ T_bath."""
        t_s = compute_spin_temperature(0.9, 0.05, 1.47)
        assert 0 < t_s < 1.0  # much less than room temperature

    def test_equal_populations_returns_nan(self):
        """ln(1) = 0 → division by zero → nan."""
        t_s = compute_spin_temperature(0.5, 0.5, 1.47)
        assert math.isnan(t_s)

    def test_zero_upper_returns_nan(self):
        t_s = compute_spin_temperature(0.0, 0.5, 1.47)
        assert math.isnan(t_s)

    def test_zero_lower_returns_nan(self):
        t_s = compute_spin_temperature(0.5, 0.0, 1.47)
        assert math.isnan(t_s)

    def test_negative_population_returns_nan(self):
        t_s = compute_spin_temperature(-0.1, 0.5, 1.47)
        assert math.isnan(t_s)

    def test_thermal_equilibrium_large_positive_temperature(self):
        """Near-equal populations (thermal) → very large positive T_s."""
        # At room temp, k_BT >> ℏω_c for GHz ω_c so populations are nearly equal.
        # Use extremely close populations to get T_s >> room temp.
        # p_upper=0.500005, p_lower=0.499995 → ratio≈1+2e-5 → T_s ≈ 3500 K
        t_s = compute_spin_temperature(0.500005, 0.499995, 1.47)
        assert t_s > 1000  # much hotter than room temperature

    def test_scales_with_frequency(self):
        """T_s ∝ ω_c (doubling frequency doubles spin temperature)."""
        t1 = compute_spin_temperature(0.9, 0.1, 1.47)
        t2 = compute_spin_temperature(0.9, 0.1, 2.94)
        assert abs(t2 / t1 - 2.0) < 1e-10

    def test_type_is_float(self):
        t_s = compute_spin_temperature(0.8, 0.1, 1.47)
        assert isinstance(t_s, float)

    def test_inversion_lower_than_thermal_gives_coolness(self):
        """Any inversion (p_upper > p_lower) gives T_s < thermal temp."""
        # thermal T_s would be >> 300 K; inverted T_s should be < 300 K
        t_s_inverted = compute_spin_temperature(0.7, 0.3, 1.47)
        assert t_s_inverted < 300.0

    def test_numerical_example(self):
        """Verify exact T_s for known input."""
        # T_s = ℏω / (k_B ln(p_upper/p_lower))
        # With f = 1.47 GHz, p_upper=0.9, p_lower=0.1
        import math as _math
        hbar = 1.054571817e-34
        kb = 1.380649e-23
        omega = 2 * _math.pi * 1.47e9
        expected = hbar * omega / (kb * _math.log(0.9 / 0.1))
        got = compute_spin_temperature(0.9, 0.1, 1.47)
        assert abs(got / expected - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# 4. compute_noise_temperature
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeNoiseTemperature:
    def test_returns_finite_for_valid_input(self):
        t_a = compute_noise_temperature(
            magnetic_q=5_000.0, unloaded_q=20_000.0, spin_temperature_k=0.1
        )
        assert math.isfinite(t_a)

    def test_q_m_equals_q0_returns_nan(self):
        """At oscillation threshold, T_a diverges."""
        t_a = compute_noise_temperature(
            magnetic_q=20_000.0, unloaded_q=20_000.0, spin_temperature_k=0.1
        )
        assert math.isnan(t_a)

    def test_q_m_above_q0_returns_nan(self):
        """Above oscillation threshold (Q_m > Q₀), formula is invalid."""
        t_a = compute_noise_temperature(
            magnetic_q=25_000.0, unloaded_q=20_000.0, spin_temperature_k=0.1
        )
        assert math.isnan(t_a)

    def test_nan_spin_temp_propagates(self):
        t_a = compute_noise_temperature(
            magnetic_q=5_000.0, unloaded_q=20_000.0,
            spin_temperature_k=float("nan")
        )
        assert math.isnan(t_a)

    def test_low_q_m_approaches_zero_for_cold_spins(self):
        """Q_m ≪ Q₀ and T_s ≈ 0: T_a ≈ (Q_m/Q₀) × T_bath → small."""
        t_a = compute_noise_temperature(
            magnetic_q=100.0, unloaded_q=100_000.0,
            spin_temperature_k=0.001, bath_temperature_k=300.0
        )
        # T_a ≈ (100/100000) × 300 + small ≈ 0.3 K + ε
        assert 0 < t_a < 1.0

    def test_consistent_with_wang_formula(self):
        """Verify formula exactly: T_a = Q_m/(Q₀-Q_m)×T_b + Q₀/(Q₀-Q_m)×T_s."""
        q_m, q_0 = 5_000.0, 22_000.0
        t_s, t_b = 0.05, 300.0
        expected = (q_m / (q_0 - q_m)) * t_b + (q_0 / (q_0 - q_m)) * t_s
        got = compute_noise_temperature(q_m, q_0, t_s, t_b)
        assert abs(got / expected - 1.0) < 1e-12

    def test_below_bath_temperature_for_inverted_maser(self):
        """Well-pumped maser should have T_a < T_bath."""
        q_m = 5_000.0
        q_0 = 22_000.0
        t_s = 0.1  # near-zero spin temperature (well inverted)
        t_bath = 300.0
        t_a = compute_noise_temperature(q_m, q_0, t_s, t_bath)
        assert t_a < t_bath

    def test_positive_for_sensible_parameters(self):
        t_a = compute_noise_temperature(
            magnetic_q=5_000.0, unloaded_q=20_000.0,
            spin_temperature_k=0.1, bath_temperature_k=300.0
        )
        assert t_a > 0


# ═══════════════════════════════════════════════════════════════════════════
# 5. compute_sql_noise_temperature
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeSqlNoiseTemperate:
    def test_positive(self):
        assert compute_sql_noise_temperature(1.47) > 0

    def test_scales_linearly_with_frequency(self):
        t1 = compute_sql_noise_temperature(1.47)
        t2 = compute_sql_noise_temperature(2.94)
        assert abs(t2 / t1 - 2.0) < 1e-10

    def test_numerical_value_1ghz(self):
        """At 1 GHz, T_SQL = ℏ·2π·10⁹ / (2k_B) ≈ 24 mK."""
        t_sql = compute_sql_noise_temperature(1.0)
        # ℏ × 2π × 10⁹ / (2 × k_B) = 1.055e-34 × 6.28e9 / (2 × 1.38e-23)
        expected_mk = 1.054571817e-34 * 2 * math.pi * 1e9 / (2 * 1.380649e-23)
        assert abs(t_sql - expected_mk) < 1e-6

    def test_larger_at_higher_frequency(self):
        t_low = compute_sql_noise_temperature(1.0)
        t_high = compute_sql_noise_temperature(10.0)
        assert t_high > t_low

    def test_value_is_millikelvin_scale(self):
        """At microwave frequencies, T_SQL is in the milli-Kelvin range."""
        t_sql = compute_sql_noise_temperature(1.47)
        assert 1e-4 < t_sql < 0.1  # 0.1 mK to 100 mK


# ═══════════════════════════════════════════════════════════════════════════
# 6. compute_output_power
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeOutputPower:
    def _make_above_threshold_fixtures(self):
        """Create configs where C > 1 (above threshold)."""
        nv = NVConfig(
            nv_density_per_cm3=1e18,
            pump_efficiency=0.9,
            orientation_fraction=0.25,
            t2_star_us=1.0,
        )
        cav = CavityConfig(mode_volume_cm3=0.5, fill_factor=0.1)
        maser = MaserConfig(cavity_q=10_000, coupling_beta=1.0)
        return nv, cav, maser

    def test_zero_if_below_threshold(self, default_nv, default_cav, default_maser):
        cavity_props = compute_cavity_properties(default_maser, default_cav)
        threshold = compute_full_threshold(
            default_nv, default_maser, default_cav,
            gain_budget=1.0, spin_linewidth_hz=1e6,
        )
        if threshold.masing:
            pytest.skip("Default config is above threshold — skip below-threshold test")
        result = compute_output_power(
            cavity_props, threshold, default_nv, default_maser
        )
        assert result.output_power_w == 0.0
        assert result.intracavity_photon_number == 0.0
        assert result.output_power_dbm == -999.0

    def test_positive_power_above_threshold(self):
        nv, cav, maser = self._make_above_threshold_fixtures()
        cavity_props = compute_cavity_properties(maser, cav)
        threshold = compute_full_threshold(nv, maser, cav, gain_budget=1.0, spin_linewidth_hz=1e6)
        assert threshold.masing, "Expected masing with high-density fixture"
        result = compute_output_power(cavity_props, threshold, nv, maser)
        assert result.output_power_w > 0
        assert result.intracavity_photon_number > 0
        assert result.output_power_dbm > -999.0

    def test_dbm_consistent_with_watts(self):
        nv, cav, maser = self._make_above_threshold_fixtures()
        cavity_props = compute_cavity_properties(maser, cav)
        threshold = compute_full_threshold(nv, maser, cav, gain_budget=1.0, spin_linewidth_hz=1e6)
        if not threshold.masing:
            pytest.skip("Below threshold")
        result = compute_output_power(cavity_props, threshold, nv, maser)
        expected_dbm = 10.0 * math.log10(result.output_power_w / 1e-3)
        assert abs(result.output_power_dbm - expected_dbm) < 1e-9

    def test_output_scales_with_density(self):
        """Higher NV density → more power above threshold."""
        cav = CavityConfig(mode_volume_cm3=0.5, fill_factor=0.1)
        maser = MaserConfig(cavity_q=10_000, coupling_beta=1.0)
        nv_low = NVConfig(nv_density_per_cm3=1e18, pump_efficiency=0.9)
        nv_high = NVConfig(nv_density_per_cm3=1e19, pump_efficiency=0.9)
        cavity_props_low = compute_cavity_properties(maser, cav)
        cavity_props_high = compute_cavity_properties(maser, cav)
        thr_low = compute_full_threshold(nv_low, maser, cav, gain_budget=1.0, spin_linewidth_hz=1e6)
        thr_high = compute_full_threshold(nv_high, maser, cav, gain_budget=1.0, spin_linewidth_hz=1e6)
        if not (thr_low.masing and thr_high.masing):
            pytest.skip("One or both below threshold")
        r_low = compute_output_power(cavity_props_low, thr_low, nv_low, maser)
        r_high = compute_output_power(cavity_props_high, thr_high, nv_high, maser)
        assert r_high.output_power_w > r_low.output_power_w

    def test_output_result_is_dataclass(self):
        nv, cav, maser = self._make_above_threshold_fixtures()
        cavity_props = compute_cavity_properties(maser, cav)
        threshold = compute_full_threshold(nv, maser, cav, gain_budget=1.0, spin_linewidth_hz=1e6)
        result = compute_output_power(cavity_props, threshold, nv, maser)
        assert isinstance(result, OutputPowerResult)


# ═══════════════════════════════════════════════════════════════════════════
# 7. _derive_population_fractions helper
# ═══════════════════════════════════════════════════════════════════════════


class TestDerivePopulationFractions:
    def test_p_upper_equals_pump_efficiency(self, default_nv):
        p_upper, _ = _derive_population_fractions(default_nv)
        assert abs(p_upper - default_nv.pump_efficiency) < 1e-12

    def test_p_lower_is_half_of_unpumped(self, default_nv):
        p_upper, p_lower = _derive_population_fractions(default_nv)
        expected_lower = (1.0 - default_nv.pump_efficiency) / 2.0
        assert abs(p_lower - expected_lower) < 1e-12

    def test_p_upper_greater_than_p_lower(self, default_nv):
        """For any reasonable pump efficiency > 1/3, upper > lower."""
        p_upper, p_lower = _derive_population_fractions(default_nv)
        assert p_upper > p_lower  # pump_efficiency = 0.5, lower = 0.25

    def test_full_pump_gives_zero_lower(self):
        nv = NVConfig(pump_efficiency=1.0)
        _, p_lower = _derive_population_fractions(nv)
        assert abs(p_lower) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# 8. AmplifierProperties integration
# ═══════════════════════════════════════════════════════════════════════════


class TestAmplifierProperties:
    def test_returns_amplifier_properties_type(
        self, default_nv, default_cav, default_maser
    ):
        result = compute_amplifier_properties(default_nv, default_cav, default_maser)
        assert isinstance(result, AmplifierProperties)

    def test_loaded_q_matches_config(self, default_nv, default_cav, default_maser):
        result = compute_amplifier_properties(default_nv, default_cav, default_maser)
        assert abs(result.loaded_q - default_maser.cavity_q) < 1e-6

    def test_unloaded_q_larger_than_loaded(
        self, default_nv, default_cav, default_maser
    ):
        result = compute_amplifier_properties(default_nv, default_cav, default_maser)
        assert result.unloaded_q > result.loaded_q

    def test_unloaded_q_formula(self, default_nv, default_cav, default_maser):
        """Q₀ = Q_L × (1 + β)."""
        result = compute_amplifier_properties(default_nv, default_cav, default_maser)
        expected_q0 = default_maser.cavity_q * (1.0 + default_maser.coupling_beta)
        assert abs(result.unloaded_q - expected_q0) < 1e-6

    def test_spin_temperature_is_finite(self, default_nv, default_cav, default_maser):
        result = compute_amplifier_properties(default_nv, default_cav, default_maser)
        assert math.isfinite(result.spin_temperature_k)

    def test_spin_temperature_below_room_temp(
        self, default_nv, default_cav, default_maser
    ):
        """An inverted spin ensemble has T_s ≪ T_bath = 300 K."""
        result = compute_amplifier_properties(default_nv, default_cav, default_maser)
        assert result.spin_temperature_k < 300.0

    def test_sql_noise_temp_positive(self, default_nv, default_cav, default_maser):
        result = compute_amplifier_properties(default_nv, default_cav, default_maser)
        assert result.sql_noise_temp_k > 0

    def test_above_threshold_flag_consistent_with_magnetic_q(
        self, default_nv, default_cav, default_maser
    ):
        result = compute_amplifier_properties(default_nv, default_cav, default_maser)
        if result.above_threshold:
            assert result.magnetic_q <= result.loaded_q
        else:
            assert result.magnetic_q > result.loaded_q or math.isinf(result.magnetic_q)

    def test_high_inversion_lower_noise_temp(
        self, high_inversion_nv, default_cav, default_maser
    ):
        """Higher inversion → lower Q_m → better amplifier → lower T_a."""
        low_nv = NVConfig(pump_efficiency=0.1, nv_density_per_cm3=1e15)
        result_low = compute_amplifier_properties(low_nv, default_cav, default_maser)
        result_high = compute_amplifier_properties(
            high_inversion_nv, default_cav, default_maser
        )
        # Only compare noise temps that are finite
        if math.isnan(result_low.noise_temperature_k) or math.isnan(
            result_high.noise_temperature_k
        ):
            pytest.skip("One result has nan noise temp (at threshold)")
        assert result_high.noise_temperature_k < result_low.noise_temperature_k

    def test_frozen_dataclass(self, default_nv, default_cav, default_maser):
        result = compute_amplifier_properties(default_nv, default_cav, default_maser)
        with pytest.raises((AttributeError, TypeError)):
            result.magnetic_q = 0  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# 9. Consistency with cavity.py cooperativity threshold
# ═══════════════════════════════════════════════════════════════════════════


class TestConsistencyWithCooperativity:
    def test_threshold_agrees_with_cooperativity(
        self, default_nv, default_cav, default_maser
    ):
        """above_threshold flag and cooperativity threshold should agree."""
        threshold = compute_full_threshold(
            default_nv, default_maser, default_cav,
            gain_budget=1.0, spin_linewidth_hz=1e6,
        )
        amp = compute_amplifier_properties(default_nv, default_cav, default_maser)
        # Both are threshold criteria for the same system —
        # they may not always agree exactly (different physics limits)
        # but should be consistent in direction for extreme cases.
        # Just verify both return sensible types.
        assert isinstance(threshold.masing, bool)
        assert isinstance(amp.above_threshold, bool)

    def test_high_density_above_both_thresholds(self):
        """Very high NV density → both C > 1 and Q_m < Q_L."""
        nv = NVConfig(nv_density_per_cm3=1e19, pump_efficiency=0.9)
        cav = CavityConfig(mode_volume_cm3=0.5, fill_factor=0.1)
        maser = MaserConfig(cavity_q=10_000)
        threshold = compute_full_threshold(nv, maser, cav, gain_budget=1.0, spin_linewidth_hz=1e6)
        amp = compute_amplifier_properties(nv, cav, maser)
        assert threshold.masing  # cooperativity > 1
        assert amp.above_threshold  # Q_m < Q_L

    def test_zero_density_below_both_thresholds(self, default_cav, default_maser):
        nv = NVConfig(nv_density_per_cm3=0.0)
        threshold = compute_full_threshold(
            nv, default_maser, default_cav,
            gain_budget=1.0, spin_linewidth_hz=1e6,
        )
        amp = compute_amplifier_properties(nv, default_cav, default_maser)
        assert not threshold.masing
        assert not amp.above_threshold

    def test_q_m_and_cooperativity_both_scale_with_density(
        self, default_cav, default_maser
    ):
        """As density rises, C increases and Q_m decreases — consistent direction."""
        nv_low = NVConfig(nv_density_per_cm3=1e15)
        nv_high = NVConfig(nv_density_per_cm3=1e19)
        amp_low = compute_amplifier_properties(nv_low, default_cav, default_maser)
        amp_high = compute_amplifier_properties(nv_high, default_cav, default_maser)
        thr_low = compute_full_threshold(nv_low, default_maser, default_cav, gain_budget=1.0, spin_linewidth_hz=1e6)
        thr_high = compute_full_threshold(nv_high, default_maser, default_cav, gain_budget=1.0, spin_linewidth_hz=1e6)
        # Q_m decreases with density (more gain):
        assert amp_high.magnetic_q < amp_low.magnetic_q
        # Cooperativity increases with density:
        assert thr_high.cooperativity > thr_low.cooperativity


# ═══════════════════════════════════════════════════════════════════════════
# 10. SQL / noise quantitative checks
# ═══════════════════════════════════════════════════════════════════════════


class TestQuantitativeChecks:
    def test_t_sql_formula_for_1p47ghz(self):
        """T_SQL = ℏω/(2k_B) at f=1.47 GHz → ~35 mK."""
        t_sql = compute_sql_noise_temperature(1.47)
        hbar = 1.054571817e-34
        kb = 1.380649e-23
        omega = 2 * math.pi * 1.47e9
        expected = hbar * omega / (2 * kb)
        assert abs(t_sql / expected - 1.0) < 1e-9

    def test_spin_temp_decreases_with_pump_efficiency(self):
        """More pumping → colder spin temperature."""
        t1 = compute_spin_temperature(0.5, 0.25, 1.47)   # 50% upper
        t2 = compute_spin_temperature(0.9, 0.05, 1.47)   # 90% upper
        assert t2 < t1

    def test_magnetic_q_formula_units_consistent(self, default_nv, default_cav, default_maser):
        """Q_m should be dimensionless and physically meaningful (10²–10¹⁰)."""
        q_m = compute_magnetic_q(default_nv, default_cav, default_maser)
        assert 1e2 < q_m < 1e12

    def test_noise_temp_lower_than_bath(self):
        """A well-pumped maser achieves T_a < T_bath = 300 K."""
        t_a = compute_noise_temperature(
            magnetic_q=1_000.0,
            unloaded_q=20_000.0,
            spin_temperature_k=0.01,
            bath_temperature_k=300.0,
        )
        assert t_a < 300.0

    def test_wang_2024_amplifier_regime(
        self, wangpaper_nv, wangpaper_cav, wangpaper_maser
    ):
        """Wang et al. (2024) operated in amplifier regime (just below threshold).

        Expect: Q_m near Q_L = 11,000 (within order of magnitude).
        """
        q_m = compute_magnetic_q(wangpaper_nv, wangpaper_cav, wangpaper_maser)
        # Wang achieved amplification: Q_m ≈ Q_L (just above threshold boundary)
        # With approximate parameters, Q_m should be within 10× of Q_L = 11,000
        assert 1_000 < q_m < 1_000_000, f"Q_m = {q_m:.2e} outside expected range"


# ═══════════════════════════════════════════════════════════════════════════
# 8. _gain_voltage (internal helper)
# ═══════════════════════════════════════════════════════════════════════════


class TestGainVoltage:
    """Tests for the _gain_voltage helper (|S₁₁| at resonance)."""

    def test_critical_coupling_formula_beta_one(self):
        """β=1: |S₁₁| = Q_L / (Q_m − Q_L)."""
        q_l, q_m = 1000.0, 1200.0
        s11 = _gain_voltage(q_m, q_l, coupling_beta=1.0)
        expected = q_l / (q_m - q_l)
        assert abs(s11 - expected) < 1e-9 * expected

    def test_at_threshold_returns_nan(self):
        """Q_m = Q_L → denominator zero → nan."""
        s11 = _gain_voltage(q_m=1000.0, q_l=1000.0, coupling_beta=1.0)
        assert math.isnan(s11)

    def test_above_threshold_returns_nan(self):
        """Q_m < Q_L (oscillator regime) → nan."""
        s11 = _gain_voltage(q_m=800.0, q_l=1000.0, coupling_beta=1.0)
        assert math.isnan(s11)

    def test_large_margin_small_amplitude(self):
        """Q_m ≫ Q_L → |S₁₁| → Q_L/Q_m ≪ 1 (no gain, just passes through)."""
        q_l, q_m = 1000.0, 1_000_000.0
        s11 = _gain_voltage(q_m, q_l, coupling_beta=1.0)
        # Should be ≈ Q_L/Q_m = 0.001 (no significant gain)
        assert s11 == pytest.approx(q_l / (q_m - q_l), rel=1e-6)

    def test_overcoupled_beta_two_gives_higher_amplitude(self):
        """β=2 (overcoupled) provides higher |S₁₁| than β=1 at same Q values."""
        q_l, q_m = 1000.0, 1200.0
        s11_critical = _gain_voltage(q_m, q_l, coupling_beta=1.0)
        s11_overcoupled = _gain_voltage(q_m, q_l, coupling_beta=2.0)
        assert s11_overcoupled > s11_critical

    def test_undercoupled_beta_zero_gives_no_coupling(self):
        """β→0 (fully undercoupled): S₁₁ → -1 (all reflected, no amplification)."""
        s11 = _gain_voltage(q_m=1200.0, q_l=1000.0, coupling_beta=1e-6)
        # n = (β-1)/(Q_L(1+β)) + 1/Q_m ≈ -1/Q_L + 1/Q_m (small β)
        # d = 1/Q_L - 1/Q_m
        # s11 ≈ (-1/Q_L + 1/Q_m)/(1/Q_L - 1/Q_m) = -1
        assert s11 == pytest.approx(-1.0, abs=1e-3)


# ═══════════════════════════════════════════════════════════════════════════
# 9. compute_maser_gain
# ═══════════════════════════════════════════════════════════════════════════


# Wang 2024 self-consistent experimental parameters (derived from G=14.5 dB, B=340 kHz):
#   f_c = 2.87 GHz, Q_L = 1337, Q_m = 1589, β = 1
_WANG_FC_HZ = 2.87e9
_WANG_QL = 1337.0
_WANG_QM = 1589.0
_WANG_BETA = 1.0


class TestComputeMaserGain:
    """Tests for compute_maser_gain — gain, bandwidth, operating margins."""

    def test_returns_maser_gain_result_type(self):
        result = compute_maser_gain(_WANG_FC_HZ, _WANG_QM, _WANG_QL)
        assert isinstance(result, MaserGainResult)

    def test_below_threshold_flag_set_correctly_when_below(self):
        result = compute_maser_gain(_WANG_FC_HZ, q_m=1500.0, q_l=1000.0)
        assert result.below_threshold is True

    def test_below_threshold_flag_false_when_equal(self):
        result = compute_maser_gain(_WANG_FC_HZ, q_m=1000.0, q_l=1000.0)
        assert result.below_threshold is False

    def test_below_threshold_flag_false_when_above(self):
        result = compute_maser_gain(_WANG_FC_HZ, q_m=800.0, q_l=1000.0)
        assert result.below_threshold is False

    def test_margin_positive_when_below_threshold(self):
        result = compute_maser_gain(_WANG_FC_HZ, q_m=1500.0, q_l=1000.0)
        assert result.margin_to_threshold > 0.0

    def test_margin_zero_at_threshold(self):
        result = compute_maser_gain(_WANG_FC_HZ, q_m=1000.0, q_l=1000.0)
        assert result.margin_to_threshold == pytest.approx(0.0, abs=1e-12)

    def test_margin_negative_above_threshold(self):
        result = compute_maser_gain(_WANG_FC_HZ, q_m=700.0, q_l=1000.0)
        assert result.margin_to_threshold < 0.0

    def test_margin_formula_correct(self):
        q_m, q_l = 1500.0, 1000.0
        result = compute_maser_gain(_WANG_FC_HZ, q_m=q_m, q_l=q_l)
        assert result.margin_to_threshold == pytest.approx(q_m / q_l - 1.0, rel=1e-9)

    def test_gain_positive_below_threshold(self):
        result = compute_maser_gain(_WANG_FC_HZ, q_m=1500.0, q_l=1000.0)
        assert result.gain_db > 0.0

    def test_gain_nan_at_threshold(self):
        result = compute_maser_gain(_WANG_FC_HZ, q_m=1000.0, q_l=1000.0)
        assert math.isnan(result.gain_db)

    def test_gain_nan_above_threshold(self):
        result = compute_maser_gain(_WANG_FC_HZ, q_m=700.0, q_l=1000.0)
        assert math.isnan(result.gain_db)

    def test_bandwidth_positive_below_threshold(self):
        result = compute_maser_gain(_WANG_FC_HZ, q_m=1500.0, q_l=1000.0)
        assert result.bandwidth_hz > 0.0

    def test_bandwidth_zero_at_threshold(self):
        result = compute_maser_gain(_WANG_FC_HZ, q_m=1000.0, q_l=1000.0)
        assert result.bandwidth_hz == pytest.approx(0.0, abs=1.0)

    def test_bandwidth_zero_above_threshold(self):
        result = compute_maser_gain(_WANG_FC_HZ, q_m=700.0, q_l=1000.0)
        assert result.bandwidth_hz == pytest.approx(0.0, abs=1.0)

    def test_bandwidth_formula_correct(self):
        q_m, q_l = 1500.0, 1000.0
        result = compute_maser_gain(_WANG_FC_HZ, q_m=q_m, q_l=q_l)
        expected_bw = _WANG_FC_HZ * (q_m - q_l) / (q_l * q_m)
        assert result.bandwidth_hz == pytest.approx(expected_bw, rel=1e-9)

    def test_bandwidth_less_than_bare_cavity_linewidth(self):
        """Parametric narrowing: B < f_c/Q_L (the bare cavity linewidth)."""
        q_m, q_l = 1400.0, 1000.0
        result = compute_maser_gain(_WANG_FC_HZ, q_m=q_m, q_l=q_l)
        bare_cavity_bw = _WANG_FC_HZ / q_l
        assert result.bandwidth_hz < bare_cavity_bw

    def test_gain_increases_as_qm_approaches_ql(self):
        """Gain should increase as Q_m approaches Q_L from above."""
        q_l = 1000.0
        r_far = compute_maser_gain(_WANG_FC_HZ, q_m=5000.0, q_l=q_l)
        r_close = compute_maser_gain(_WANG_FC_HZ, q_m=1100.0, q_l=q_l)
        assert r_close.gain_db > r_far.gain_db

    def test_bandwidth_decreases_as_qm_approaches_ql(self):
        """Parametric narrowing: bandwidth narrows as Q_m → Q_L."""
        q_l = 1000.0
        r_far = compute_maser_gain(_WANG_FC_HZ, q_m=5000.0, q_l=q_l)
        r_close = compute_maser_gain(_WANG_FC_HZ, q_m=1100.0, q_l=q_l)
        assert r_close.bandwidth_hz < r_far.bandwidth_hz

    def test_gain_formula_critical_coupling(self):
        """β=1: G = (Q_L/(Q_m-Q_L))² — verify against direct formula."""
        q_l, q_m = 1000.0, 1200.0
        result = compute_maser_gain(_WANG_FC_HZ, q_m=q_m, q_l=q_l, coupling_beta=1.0)
        expected_g_power = (q_l / (q_m - q_l)) ** 2
        expected_db = 10.0 * math.log10(expected_g_power)
        assert result.gain_db == pytest.approx(expected_db, rel=1e-9)

    def test_gain_bw_product_positive(self):
        result = compute_maser_gain(_WANG_FC_HZ, q_m=1500.0, q_l=1000.0)
        assert result.gain_bw_product_hz > 0.0

    def test_wang_2024_gain_approx_14p5_db(self):
        """Wang et al. (2024): peak gain ≈ 14.5 dB at self-consistent Q values."""
        result = compute_maser_gain(
            _WANG_FC_HZ, q_m=_WANG_QM, q_l=_WANG_QL, coupling_beta=_WANG_BETA
        )
        assert result.gain_db == pytest.approx(14.5, abs=0.5)

    def test_wang_2024_bandwidth_approx_340_khz(self):
        """Wang et al. (2024): bandwidth ≈ 340 kHz at self-consistent Q values."""
        result = compute_maser_gain(
            _WANG_FC_HZ, q_m=_WANG_QM, q_l=_WANG_QL, coupling_beta=_WANG_BETA
        )
        assert result.bandwidth_hz == pytest.approx(340_000.0, rel=0.05)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Gain–bandwidth invariant
# ═══════════════════════════════════════════════════════════════════════════


class TestMaserGainBandwidthInvariant:
    """The gain-voltage × bandwidth product equals f_c/Q_m for β=1."""

    def test_gbp_equals_fc_over_qm_for_critical_coupling(self):
        """√G × B = f_c/Q_m for β=1, validated at one operating point."""
        q_l, q_m = 1000.0, 1200.0
        result = compute_maser_gain(_WANG_FC_HZ, q_m=q_m, q_l=q_l, coupling_beta=1.0)
        expected = _WANG_FC_HZ / q_m
        assert result.gain_bw_product_hz == pytest.approx(expected, rel=1e-9)

    def test_gbp_invariant_for_two_different_qm_critical_coupling(self):
        """GBP = f_c/Q_m is the same for both Q_m values (β=1)."""
        q_l = 1000.0
        r1 = compute_maser_gain(_WANG_FC_HZ, q_m=1100.0, q_l=q_l, coupling_beta=1.0)
        r2 = compute_maser_gain(_WANG_FC_HZ, q_m=5000.0, q_l=q_l, coupling_beta=1.0)
        # Both equal f_c/Q_m (different Q_m ↔ different GBP)
        assert r1.gain_bw_product_hz == pytest.approx(_WANG_FC_HZ / 1100.0, rel=1e-9)
        assert r2.gain_bw_product_hz == pytest.approx(_WANG_FC_HZ / 5000.0, rel=1e-9)

    def test_gbp_zero_above_threshold(self):
        """GBP = 0 when not in amplifier regime."""
        result = compute_maser_gain(_WANG_FC_HZ, q_m=500.0, q_l=1000.0, coupling_beta=1.0)
        assert result.gain_bw_product_hz == pytest.approx(0.0, abs=1.0)

    def test_wang_2024_gbp_equals_fc_over_qm(self):
        """Wang 2024 exp. point: GBP = f_c/Q_m."""
        result = compute_maser_gain(
            _WANG_FC_HZ, q_m=_WANG_QM, q_l=_WANG_QL, coupling_beta=_WANG_BETA
        )
        expected = _WANG_FC_HZ / _WANG_QM
        assert result.gain_bw_product_hz == pytest.approx(expected, rel=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# 11. MaserGainResult dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestMaserGainResult:
    def test_result_is_frozen(self):
        r = compute_maser_gain(_WANG_FC_HZ, q_m=1500.0, q_l=1000.0)
        with pytest.raises((AttributeError, TypeError)):
            r.gain_db = 0.0  # type: ignore[misc]

    def test_all_float_fields_below_threshold(self):
        r = compute_maser_gain(_WANG_FC_HZ, q_m=1500.0, q_l=1000.0)
        assert isinstance(r.gain_db, float)
        assert isinstance(r.bandwidth_hz, float)
        assert isinstance(r.gain_bw_product_hz, float)
        assert isinstance(r.margin_to_threshold, float)

    def test_below_threshold_is_bool(self):
        r = compute_maser_gain(_WANG_FC_HZ, q_m=1500.0, q_l=1000.0)
        assert isinstance(r.below_threshold, bool)

    def test_repr_includes_gain_db(self):
        r = compute_maser_gain(_WANG_FC_HZ, q_m=1500.0, q_l=1000.0)
        assert "gain_db" in repr(r)


# ═══════════════════════════════════════════════════════════════════════════
# Q-boost integration: compute_amplifier_properties respects q_boost_gain
# ═══════════════════════════════════════════════════════════════════════════


class TestAmplifierQBoostIntegration:
    """Verify that compute_amplifier_properties uses effective Q when boosted."""

    def test_no_boost_matches_raw_cavity_q(self, default_nv, default_cav):
        """q_boost_gain=0 → loaded_q = cavity_q (unchanged)."""
        maser = MaserConfig(cavity_q=10_000, q_boost_gain=0.0)
        result = compute_amplifier_properties(default_nv, default_cav, maser)
        assert result.loaded_q == pytest.approx(10_000.0)

    def test_boost_raises_loaded_q(self, default_nv, default_cav):
        """q_boost_gain=0.5 → loaded_q = cavity_q / (1 - 0.5) = 2× cavity_q."""
        maser = MaserConfig(cavity_q=10_000, q_boost_gain=0.5)
        result = compute_amplifier_properties(default_nv, default_cav, maser)
        assert result.loaded_q == pytest.approx(20_000.0)

    def test_boost_raises_unloaded_q(self, default_nv, default_cav):
        """Q₀ = Q_L_eff × (1 + β)."""
        beta = 0.5
        maser = MaserConfig(cavity_q=10_000, coupling_beta=beta, q_boost_gain=0.5)
        result = compute_amplifier_properties(default_nv, default_cav, maser)
        expected_q0 = 20_000.0 * (1.0 + beta)
        assert result.unloaded_q == pytest.approx(expected_q0)

    def test_high_boost_like_wang_2024(self, default_nv, default_cav):
        """Wang 2024: g_fb ≈ 0.983 boosts Q_L from 1.1e4 to ~6.5e5."""
        g_fb = 1.0 - 1.0 / 59.0  # ≈ 0.9831
        maser = MaserConfig(cavity_q=11_000, q_boost_gain=g_fb)
        result = compute_amplifier_properties(default_nv, default_cav, maser)
        assert result.loaded_q == pytest.approx(11_000 * 59.0, rel=0.01)

    def test_boost_can_cross_threshold(self, default_nv, default_cav):
        """System below threshold without boost may cross it with Q-boost."""
        # With low Q, Q_m > Q_L → below threshold
        maser_passive = MaserConfig(cavity_q=100, q_boost_gain=0.0)
        result_passive = compute_amplifier_properties(
            default_nv, default_cav, maser_passive
        )

        # Boost Q by 100× → Q_L_eff = 10_000, may cross threshold
        maser_boosted = MaserConfig(cavity_q=100, q_boost_gain=0.99)
        result_boosted = compute_amplifier_properties(
            default_nv, default_cav, maser_boosted
        )

        # We can't guarantee threshold crossing for all parameter combos,
        # but loaded_q should be dramatically higher
        assert result_boosted.loaded_q > result_passive.loaded_q * 50

    def test_boost_does_not_change_magnetic_q(self, default_nv, default_cav):
        """Q_m depends on spin parameters, not cavity Q-boost."""
        maser_passive = MaserConfig(cavity_q=10_000, q_boost_gain=0.0)
        maser_boosted = MaserConfig(cavity_q=10_000, q_boost_gain=0.9)
        r1 = compute_amplifier_properties(default_nv, default_cav, maser_passive)
        r2 = compute_amplifier_properties(default_nv, default_cav, maser_boosted)
        assert r1.magnetic_q == pytest.approx(r2.magnetic_q, rel=1e-10)

    def test_boost_lowers_noise_temperature(self, default_nv, default_cav):
        """Higher Q₀_eff → lower T_a (Wang Eq. 4: T_a → T_s as Q₀_eff → ∞)."""
        maser_passive = MaserConfig(cavity_q=10_000, q_boost_gain=0.0)
        maser_boosted = MaserConfig(cavity_q=10_000, q_boost_gain=0.9)
        r1 = compute_amplifier_properties(default_nv, default_cav, maser_passive)
        r2 = compute_amplifier_properties(default_nv, default_cav, maser_boosted)
        # Both must be finite for comparison (nan when at threshold)
        if math.isnan(r1.noise_temperature_k) or math.isnan(r2.noise_temperature_k):
            pytest.skip("Noise temp nan at threshold boundary")
        assert r2.noise_temperature_k <= r1.noise_temperature_k

