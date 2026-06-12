"""Tests for spin_squeezing.py — projection noise and quantum-enhanced sensitivity."""
import math

import pytest

from nv_maser.config import NVConfig
from nv_maser.physics.spin_squeezing import (
    REGIME_COHERENT,
    REGIME_NEAR_HEISENBERG,
    REGIME_SQUEEZED,
    QuantumEnhancementResult,
    SpinSqueezingResult,
    classify_squeezing_regime,
    compute_hl_field_sensitivity,
    compute_hl_phase_sensitivity,
    compute_metrological_gain_db,
    compute_oat_optimal_squeezing,
    compute_projection_noise,
    compute_quantum_enhancement,
    compute_spin_squeezing,
    compute_sql_field_sensitivity,
    compute_sql_phase_sensitivity,
    compute_wineland_squeezing,
)

# ── Shared constants ───────────────────────────────────────────────────────
_GAMMA_E = 28.025e9  # Hz/T  (NV |0⟩→|−1⟩)
_TWO_PI = 2.0 * math.pi


# ═══════════════════════════════════════════════════════════════════════════
# SQL phase sensitivity
# ═══════════════════════════════════════════════════════════════════════════


def test_sql_phase_sensitivity_one_spin():
    """Single spin: phase uncertainty = 1 (maximum)."""
    assert compute_sql_phase_sensitivity(1.0) == pytest.approx(1.0, rel=1e-9)


def test_sql_phase_sensitivity_four_spins():
    """N = 4 → δφ = 0.5."""
    assert compute_sql_phase_sensitivity(4.0) == pytest.approx(0.5, rel=1e-9)


def test_sql_phase_sensitivity_hundred_spins():
    """N = 100 → δφ = 0.1."""
    assert compute_sql_phase_sensitivity(100.0) == pytest.approx(0.1, rel=1e-9)


def test_sql_phase_sensitivity_sqrt_n_scaling():
    """Quadrupling N halves the phase uncertainty."""
    ph_n = compute_sql_phase_sensitivity(100.0)
    ph_4n = compute_sql_phase_sensitivity(400.0)
    assert ph_4n == pytest.approx(ph_n / 2.0, rel=1e-9)


def test_sql_phase_sensitivity_negative_n_raises():
    with pytest.raises(ValueError, match="n_spins must be > 0"):
        compute_sql_phase_sensitivity(-1.0)


def test_sql_phase_sensitivity_zero_n_raises():
    with pytest.raises(ValueError, match="n_spins must be > 0"):
        compute_sql_phase_sensitivity(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# HL phase sensitivity
# ═══════════════════════════════════════════════════════════════════════════


def test_hl_phase_sensitivity_one_spin():
    """Single spin: HL = SQL = 1."""
    assert compute_hl_phase_sensitivity(1.0) == pytest.approx(1.0, rel=1e-9)


def test_hl_phase_sensitivity_hundred_spins():
    """N = 100 → δφ_HL = 0.01."""
    assert compute_hl_phase_sensitivity(100.0) == pytest.approx(0.01, rel=1e-9)


def test_hl_better_than_sql_for_n_gt_1():
    """Heisenberg uncertainty is always smaller than SQL for N > 1."""
    for n in [2, 10, 1000, 1e12]:
        assert compute_hl_phase_sensitivity(n) < compute_sql_phase_sensitivity(n)


def test_hl_sql_ratio_is_sqrt_n():
    """δφ_SQL / δφ_HL = √N exactly."""
    n = 1e6
    ratio = compute_sql_phase_sensitivity(n) / compute_hl_phase_sensitivity(n)
    assert ratio == pytest.approx(math.sqrt(n), rel=1e-9)


def test_hl_phase_sensitivity_zero_n_raises():
    with pytest.raises(ValueError, match="n_spins must be > 0"):
        compute_hl_phase_sensitivity(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# SQL field sensitivity
# ═══════════════════════════════════════════════════════════════════════════


def test_sql_field_sensitivity_formula():
    """Verify η_B_SQL = 1/(2π γ_e √(N τ)) against direct computation."""
    n, tau = 1e12, 1e-6
    expected = 1.0 / (_TWO_PI * _GAMMA_E * math.sqrt(n * tau))
    result = compute_sql_field_sensitivity(n, tau, _GAMMA_E)
    assert result == pytest.approx(expected, rel=1e-9)


def test_sql_field_sensitivity_physical_scale():
    """For NV defaults (N=1e12, τ=1 µs): η ≈ 5–7 fT/√Hz."""
    eta = compute_sql_field_sensitivity(1e12, 1e-6, _GAMMA_E)
    assert 1e-15 < eta < 1e-14, f"Expected fT/√Hz scale, got {eta}"


def test_sql_field_sensitivity_n_scaling():
    """4× more spins → 2× better sensitivity."""
    eta_n = compute_sql_field_sensitivity(1e10, 1e-6, _GAMMA_E)
    eta_4n = compute_sql_field_sensitivity(4e10, 1e-6, _GAMMA_E)
    assert eta_4n == pytest.approx(eta_n / 2.0, rel=1e-9)


def test_sql_field_sensitivity_tau_scaling():
    """4× longer interrogation → 2× better sensitivity."""
    eta_t = compute_sql_field_sensitivity(1e10, 1e-6, _GAMMA_E)
    eta_4t = compute_sql_field_sensitivity(1e10, 4e-6, _GAMMA_E)
    assert eta_4t == pytest.approx(eta_t / 2.0, rel=1e-9)


def test_sql_field_sensitivity_zero_n_raises():
    with pytest.raises(ValueError):
        compute_sql_field_sensitivity(0.0, 1e-6, _GAMMA_E)


def test_sql_field_sensitivity_zero_tau_raises():
    with pytest.raises(ValueError):
        compute_sql_field_sensitivity(1e10, 0.0, _GAMMA_E)


# ═══════════════════════════════════════════════════════════════════════════
# HL field sensitivity
# ═══════════════════════════════════════════════════════════════════════════


def test_hl_field_sensitivity_formula():
    """Verify η_B_HL = 1/(2π γ_e N √τ) against direct computation."""
    n, tau = 1e12, 1e-6
    expected = 1.0 / (_TWO_PI * _GAMMA_E * n * math.sqrt(tau))
    result = compute_hl_field_sensitivity(n, tau, _GAMMA_E)
    assert result == pytest.approx(expected, rel=1e-9)


def test_hl_better_than_sql_field():
    """η_B_HL < η_B_SQL for N > 1."""
    n, tau = 1e10, 1e-6
    assert compute_hl_field_sensitivity(n, tau, _GAMMA_E) < compute_sql_field_sensitivity(n, tau, _GAMMA_E)


def test_hl_sql_field_ratio_is_sqrt_n():
    """η_B_SQL / η_B_HL = √N exactly."""
    n, tau = 1e8, 1e-5
    sql = compute_sql_field_sensitivity(n, tau, _GAMMA_E)
    hl = compute_hl_field_sensitivity(n, tau, _GAMMA_E)
    assert sql / hl == pytest.approx(math.sqrt(n), rel=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# ProjectionNoiseResult
# ═══════════════════════════════════════════════════════════════════════════


def test_projection_noise_result_types():
    """All fields are finite floats."""
    pn = compute_projection_noise(1e10, 1e-6)
    assert math.isfinite(pn.sql_phase_rad)
    assert math.isfinite(pn.hl_phase_rad)
    assert math.isfinite(pn.sql_field_t_per_sqrthz)
    assert math.isfinite(pn.hl_field_t_per_sqrthz)
    assert math.isfinite(pn.heisenberg_advantage)


def test_projection_noise_heisenberg_advantage_is_sqrt_n():
    """heisenberg_advantage = √N exactly."""
    n = 1e8
    pn = compute_projection_noise(n, 1e-6)
    assert pn.heisenberg_advantage == pytest.approx(math.sqrt(n), rel=1e-9)


def test_projection_noise_sql_better_than_hl():
    """SQL field sensitivity is worse (larger) than HL."""
    pn = compute_projection_noise(1e10, 1e-6)
    assert pn.sql_field_t_per_sqrthz > pn.hl_field_t_per_sqrthz


def test_projection_noise_default_gamma():
    """Default γ_e = 28.025 GHz/T (NV |0⟩→|−1⟩ transition)."""
    n, tau = 1e10, 1e-6
    pn_default = compute_projection_noise(n, tau)
    pn_explicit = compute_projection_noise(n, tau, gamma_e_hz_per_t=28.025e9)
    assert pn_default.sql_field_t_per_sqrthz == pytest.approx(
        pn_explicit.sql_field_t_per_sqrthz, rel=1e-9
    )


# ═══════════════════════════════════════════════════════════════════════════
# Wineland squeezing parameter
# ═══════════════════════════════════════════════════════════════════════════


def test_wineland_coherent_spin_state():
    """For CSS: ⟨ΔJy²⟩ = N/4, ⟨Jz⟩ = N/2 → ξ²_R = 1."""
    n = 1000.0
    var_css = n / 4.0
    mean_jz_css = n / 2.0
    xi2 = compute_wineland_squeezing(n, var_css, mean_jz_css)
    assert xi2 == pytest.approx(1.0, rel=1e-9)


def test_wineland_below_sql():
    """Reduced variance with same mean spin → ξ²_R < 1."""
    n = 1000.0
    mean_jz = n / 2.0
    var_sql = n / 4.0
    xi2_squeezed = compute_wineland_squeezing(n, var_sql / 2.0, mean_jz)
    assert xi2_squeezed < 1.0


def test_wineland_ghz_state():
    """GHZ state: ⟨ΔJy²⟩ = N²/4, but each spin contributes N times —
    actually for GHZ: ξ²_R ≈ 1/N via the collective variance.
    Here test via the formula directly: ξ²_R = N × (1/4) / (N/2)² = 1/N."""
    n = 100.0
    # For ideal GHZ: variance in optimal direction = (N/2)²/N = N/4... actually
    # GHZ has δJy²= 0 in squeezed direction AND ⟨Jz⟩² = (N/2)²
    # leading to ξ²_R = 0 (impossible to distinguish from maximally squeezed)
    # More practically: use the formula with values that give 1/N
    var = 1.0 / 4.0  # variance for one spin pair
    mean = n / 2.0
    xi2 = compute_wineland_squeezing(n, var, mean)
    # xi2 = N × (1/4) / (N/2)² = N/(4 × N²/4) = 1/N
    assert xi2 == pytest.approx(1.0 / n, rel=1e-9)


def test_wineland_zero_mean_jz_raises():
    with pytest.raises(ValueError, match="mean_jz must be non-zero"):
        compute_wineland_squeezing(100.0, 25.0, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# OAT optimal squeezing
# ═══════════════════════════════════════════════════════════════════════════


def test_oat_optimal_squeezing_n_scaling():
    """ξ²_R^{OAT} scales as N^{-2/3}."""
    xi_100 = compute_oat_optimal_squeezing(100.0)
    xi_1000 = compute_oat_optimal_squeezing(1000.0)
    # ratio should be (1000/100)^{2/3} = 10^{2/3} ≈ 4.642
    expected_ratio = (1000.0 / 100.0) ** (2.0 / 3.0)
    assert xi_100 / xi_1000 == pytest.approx(expected_ratio, rel=1e-9)


def test_oat_below_sql_for_n_gt_1():
    """OAT optimal squeezing is always below the SQL for N > 1."""
    for n in [2, 10, 100, 1e6, 1e12]:
        assert compute_oat_optimal_squeezing(n) < 1.0


def test_oat_above_hl_for_n_gt_1():
    """OAT optimal squeezing stays above the Heisenberg limit N^{-1} for N > 1."""
    for n in [2, 10, 100, 1e6]:
        oat = compute_oat_optimal_squeezing(n)
        hl = 1.0 / n
        assert oat > hl, f"OAT should be above HL for N={n}"


def test_oat_optimal_squeezing_zero_n_raises():
    with pytest.raises(ValueError):
        compute_oat_optimal_squeezing(0.0)


def test_oat_optimal_squeezing_one_spin():
    """For N = 1: N^{-2/3} = 1.0 (no squeezing possible with single spin)."""
    assert compute_oat_optimal_squeezing(1.0) == pytest.approx(1.0, rel=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# Metrological gain
# ═══════════════════════════════════════════════════════════════════════════


def test_metrological_gain_coherent_state():
    """ξ²_R = 1 → G = 0 dB."""
    assert compute_metrological_gain_db(1.0) == pytest.approx(0.0, abs=1e-9)


def test_metrological_gain_ten_db():
    """ξ²_R = 0.1 → G = 10 dB."""
    assert compute_metrological_gain_db(0.1) == pytest.approx(10.0, rel=1e-9)


def test_metrological_gain_twenty_db():
    """ξ²_R = 0.01 → G = 20 dB."""
    assert compute_metrological_gain_db(0.01) == pytest.approx(20.0, rel=1e-9)


def test_metrological_gain_positive_for_squeezed():
    """G > 0 for any ξ²_R < 1."""
    for xi2 in [0.5, 0.1, 0.01, 1e-6]:
        assert compute_metrological_gain_db(xi2) > 0.0


def test_metrological_gain_zero_xi_raises():
    with pytest.raises(ValueError):
        compute_metrological_gain_db(0.0)


def test_metrological_gain_negative_xi_raises():
    with pytest.raises(ValueError):
        compute_metrological_gain_db(-0.5)


# ═══════════════════════════════════════════════════════════════════════════
# Regime classification
# ═══════════════════════════════════════════════════════════════════════════


def test_classify_coherent_state():
    """ξ²_R = 1.0 → COHERENT."""
    assert classify_squeezing_regime(1.0, 1000.0) == REGIME_COHERENT


def test_classify_above_sql_is_coherent():
    """ξ²_R > 1.0 → COHERENT (anti-squeezed state)."""
    assert classify_squeezing_regime(2.0, 1000.0) == REGIME_COHERENT


def test_classify_squeezed():
    """ξ²_R = 0.1 with large N → SQUEEZED."""
    # N = 1000, near-HL threshold = 1/√1000 ≈ 0.0316
    # ξ²_R = 0.1 > 0.0316 → SQUEEZED
    assert classify_squeezing_regime(0.1, 1000.0) == REGIME_SQUEEZED


def test_classify_near_heisenberg():
    """ξ²_R ≤ 1/√N → NEAR_HEISENBERG."""
    n = 1000.0
    threshold = 1.0 / math.sqrt(n)  # ≈ 0.0316
    # slightly below threshold
    assert classify_squeezing_regime(threshold * 0.5, n) == REGIME_NEAR_HEISENBERG


def test_classify_exactly_at_threshold_is_near_heisenberg():
    """ξ²_R exactly at 1/√N boundary → NEAR_HEISENBERG."""
    n = 1000.0
    threshold = 1.0 / math.sqrt(n)
    assert classify_squeezing_regime(threshold, n) == REGIME_NEAR_HEISENBERG


# ═══════════════════════════════════════════════════════════════════════════
# SpinSqueezingResult
# ═══════════════════════════════════════════════════════════════════════════


def test_spin_squeezing_result_coherent_state():
    """Coherent state ξ²_R = 1 → below_sql = False, regime = COHERENT."""
    result = compute_spin_squeezing(1000.0, 1.0)
    assert isinstance(result, SpinSqueezingResult)
    assert result.below_sql is False
    assert result.regime == REGIME_COHERENT
    assert result.metrological_gain_db == pytest.approx(0.0, abs=1e-9)


def test_spin_squeezing_result_squeezed():
    """Squeezed state ξ²_R = 0.1 → below_sql = True, G = 10 dB."""
    result = compute_spin_squeezing(1000.0, 0.1)
    assert result.below_sql is True
    assert result.metrological_gain_db == pytest.approx(10.0, rel=1e-9)
    assert result.regime == REGIME_SQUEEZED


def test_spin_squeezing_result_xi2_stored():
    """ξ²_R value is stored verbatim."""
    xi2 = 0.05
    result = compute_spin_squeezing(100.0, xi2)
    assert result.xi2_r == xi2


def test_spin_squeezing_zero_xi_raises():
    with pytest.raises(ValueError):
        compute_spin_squeezing(1000.0, 0.0)


def test_spin_squeezing_zero_n_raises():
    with pytest.raises(ValueError):
        compute_spin_squeezing(0.0, 0.5)


# ═══════════════════════════════════════════════════════════════════════════
# QuantumEnhancementResult
# ═══════════════════════════════════════════════════════════════════════════


def test_quantum_enhancement_default_config():
    """Runs without error with NVConfig defaults."""
    nv = NVConfig()
    result = compute_quantum_enhancement(nv, n_effective=1e12)
    assert isinstance(result, QuantumEnhancementResult)


def test_quantum_enhancement_coherent_state_sql():
    """No squeezing (ξ²_R = 1): squeezed sensitivity = SQL sensitivity."""
    nv = NVConfig()
    result = compute_quantum_enhancement(nv, n_effective=1e10, xi2_r=1.0)
    assert result.squeezed_field_sensitivity_t_per_sqrthz == pytest.approx(
        result.projection_noise.sql_field_t_per_sqrthz, rel=1e-9
    )


def test_quantum_enhancement_squeezing_improves_sensitivity():
    """Squeezing (ξ²_R < 1) reduces the field sensitivity below SQL."""
    nv = NVConfig()
    result_sq = compute_quantum_enhancement(nv, n_effective=1e10, xi2_r=0.1)
    result_css = compute_quantum_enhancement(nv, n_effective=1e10, xi2_r=1.0)
    assert result_sq.squeezed_field_sensitivity_t_per_sqrthz < \
           result_css.squeezed_field_sensitivity_t_per_sqrthz


def test_quantum_enhancement_default_interrogation_time():
    """When interrogation_time_s is None, T₂* from config is used."""
    nv = NVConfig()
    t2_s = nv.t2_star_us * 1e-6
    result_none = compute_quantum_enhancement(nv, n_effective=1e10)
    result_explicit = compute_quantum_enhancement(nv, n_effective=1e10,
                                                  interrogation_time_s=t2_s)
    assert result_none.projection_noise.sql_field_t_per_sqrthz == pytest.approx(
        result_explicit.projection_noise.sql_field_t_per_sqrthz, rel=1e-9
    )


def test_quantum_enhancement_oat_better_than_sql():
    """OAT optimum sensitivity is always better than SQL."""
    nv = NVConfig()
    result = compute_quantum_enhancement(nv, n_effective=1e10)
    assert result.oat_best_sensitivity_t_per_sqrthz < \
           result.projection_noise.sql_field_t_per_sqrthz


def test_quantum_enhancement_heisenberg_is_hl():
    """heisenberg_sensitivity == projection_noise.hl_field_t_per_sqrthz."""
    nv = NVConfig()
    result = compute_quantum_enhancement(nv, n_effective=1e10)
    assert result.heisenberg_sensitivity_t_per_sqrthz == pytest.approx(
        result.projection_noise.hl_field_t_per_sqrthz, rel=1e-9
    )


def test_quantum_enhancement_hierarchy():
    """Sensitivity floor ordering: HL ≤ OAT ≤ squeezed (CSS) = SQL."""
    nv = NVConfig()
    result = compute_quantum_enhancement(nv, n_effective=1e10, xi2_r=1.0)
    hl = result.heisenberg_sensitivity_t_per_sqrthz
    oat = result.oat_best_sensitivity_t_per_sqrthz
    sql = result.projection_noise.sql_field_t_per_sqrthz
    assert hl <= oat <= sql


def test_quantum_enhancement_zero_n_raises():
    with pytest.raises(ValueError):
        compute_quantum_enhancement(NVConfig(), n_effective=0.0)


def test_quantum_enhancement_zero_xi_raises():
    with pytest.raises(ValueError):
        compute_quantum_enhancement(NVConfig(), n_effective=1e10, xi2_r=0.0)
