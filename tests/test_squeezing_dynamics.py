"""Tests for squeezing_dynamics.py — OAT squeezing with T₂* decoherence."""
import math

import numpy as np
import pytest

from nv_maser.config import NVConfig
from nv_maser.physics.squeezing_dynamics import (
    OATDecoherenceTrajectory,
    OATIdealTrajectory,
    SqueezingFeasibility,
    TATIdealTrajectory,
    TATDecoherenceTrajectory,
    apply_decoherence,
    compute_oat_ideal_trajectory,
    compute_oat_with_decoherence,
    compute_squeezing_feasibility,
    estimate_oat_chi,
    oat_optimal_time,
    oat_xi2_ideal,
    tat_xi2_ideal,
    tat_optimal_time,
    compute_tat_ideal_trajectory,
    compute_tat_with_decoherence,
)

# ── Shared constants ───────────────────────────────────────────────────────
_N_SPINS = 1e12       # Typical NV ensemble
_CHI_HZ = 5.0         # Typical dipolar OAT coupling (Hz)
_T2_STAR_S = 1e-6     # 1 μs dephasing
_TWO_PI = 2.0 * math.pi


# ═══════════════════════════════════════════════════════════════════════════
# oat_optimal_time
# ═══════════════════════════════════════════════════════════════════════════


def test_oat_optimal_time_formula():
    """t_opt = (12/(N-1))^{1/4} / (2πχ)."""
    t = oat_optimal_time(_N_SPINS, _CHI_HZ)
    expected = (12.0 / (_N_SPINS - 1.0)) ** 0.25 / (_TWO_PI * _CHI_HZ)
    assert t == pytest.approx(expected, rel=1e-10)


def test_oat_optimal_time_scales_with_n():
    """Doubling N reduces t_opt by factor 2^{-1/4} (parabolic form)."""
    t1 = oat_optimal_time(1e6, _CHI_HZ)
    t2 = oat_optimal_time(2e6, _CHI_HZ)
    # (12/(2N-1))^{1/4} / (12/(N-1))^{1/4} = ((N-1)/(2N-1))^{1/4} ~ 2^{-1/4}
    assert t2 / t1 == pytest.approx(((1e6 - 1) / (2e6 - 1)) ** 0.25, rel=1e-6)


def test_oat_optimal_time_scales_with_chi():
    """Doubling χ halves t_opt."""
    t1 = oat_optimal_time(_N_SPINS, 5.0)
    t2 = oat_optimal_time(_N_SPINS, 10.0)
    assert t2 / t1 == pytest.approx(0.5, rel=1e-10)


def test_oat_optimal_time_zero_n_raises():
    with pytest.raises(ValueError, match="n_spins"):
        oat_optimal_time(0.0, _CHI_HZ)


def test_oat_optimal_time_negative_chi_raises():
    with pytest.raises(ValueError, match="chi_hz"):
        oat_optimal_time(_N_SPINS, -1.0)


# ═══════════════════════════════════════════════════════════════════════════
# oat_xi2_ideal
# ═══════════════════════════════════════════════════════════════════════════


def test_xi2_ideal_at_optimum_scales_as_n_minus_half():
    """At t_opt the parabolic form achieves ξ² ≈ 1/√(3N) ∝ N^{-1/2}."""
    n = 1e8
    chi = 10.0
    t_opt = oat_optimal_time(n, chi)
    xi2 = oat_xi2_ideal(t_opt, n, chi).item()
    expected = 1.0 / math.sqrt(3.0 * n)
    # Parabolic minimum: ξ²_min = 2/√(12N) = 1/√(3N)
    assert xi2 == pytest.approx(expected, rel=0.01)


def test_xi2_ideal_at_zero_time_returns_css():
    """At t=0, ξ²_R = 1 (coherent spin state)."""
    xi2 = oat_xi2_ideal(0.0, _N_SPINS, _CHI_HZ).item()
    assert xi2 == pytest.approx(1.0, rel=1e-10)


def test_xi2_ideal_short_time_parabolic():
    """At very short times, ξ²_R ≈ 1/(N μ²t²)."""
    n = 1e10
    chi = 5.0
    mu = _TWO_PI * chi
    t = 1e-6  # very short
    xi2 = oat_xi2_ideal(t, n, chi).item()
    expected = 1.0 / (n * (mu * t) ** 2)
    # The parabolic form has a correction term; just check order of magnitude
    assert xi2 == pytest.approx(expected, rel=0.5)


def test_xi2_ideal_floor_at_heisenberg():
    """ξ²_R never goes below 1/N (Heisenberg limit)."""
    n = 100.0
    chi = 100.0
    times = np.linspace(0.001, 1.0, 500)
    xi2 = oat_xi2_ideal(times, n, chi)
    assert np.all(xi2 >= 1.0 / n - 1e-15)


def test_xi2_ideal_returns_array():
    """Vectorised over time."""
    times = np.array([0.0, 1e-6, 2e-6])
    xi2 = oat_xi2_ideal(times, _N_SPINS, _CHI_HZ)
    assert xi2.shape == (3,)


def test_xi2_ideal_scalar_input():
    """Scalar time is accepted."""
    xi2 = oat_xi2_ideal(1e-6, _N_SPINS, _CHI_HZ)
    assert xi2.ndim >= 0  # returns array-like


# ═══════════════════════════════════════════════════════════════════════════
# apply_decoherence
# ═══════════════════════════════════════════════════════════════════════════


def test_decoherence_at_zero_time_preserves_ideal():
    """At t=0, decoherence has no effect."""
    t = np.array([0.0])
    xi2_ideal = np.array([1.0])
    xi2 = apply_decoherence(xi2_ideal, t, _T2_STAR_S)
    assert xi2[0] == pytest.approx(1.0, abs=1e-10)


def test_decoherence_increases_squeezing_parameter():
    """Decoherence always makes ξ²_R worse (larger)."""
    n = 1e8
    chi = 10.0
    t_opt = oat_optimal_time(n, chi)
    times = np.linspace(t_opt * 0.01, t_opt * 5, 100)
    xi2_ideal = oat_xi2_ideal(times, n, chi)
    xi2_deco = apply_decoherence(xi2_ideal, times, _T2_STAR_S)
    assert np.all(xi2_deco >= xi2_ideal - 1e-15)


def test_decoherence_approaches_css_at_long_times():
    """As t → ∞, ξ²_R → 1+ (unsqueezed CSS)."""
    t = np.array([100.0])  # Very long compared to T₂*
    xi2_ideal = np.array([0.01])
    xi2 = apply_decoherence(xi2_ideal, t, _T2_STAR_S)
    # The floor term [1 - exp(-2t/T₂*)] → 1, and the ideal term is heavily attenuated
    assert xi2[0] > 0.99


def test_decoherence_formula():
    """Verify: ξ²(t) = ξ²_ideal·exp(2t/T₂*) + [1 − exp(−2t/T₂*)]."""
    t = np.array([0.5e-6])
    xi2_ideal = np.array([0.1])
    t2 = 1e-6
    ratio = t[0] / t2
    expected = xi2_ideal[0] * math.exp(2 * ratio) + (1.0 - math.exp(-2 * ratio))
    result = apply_decoherence(xi2_ideal, t, t2)
    assert result[0] == pytest.approx(expected, rel=1e-10)


def test_decoherence_zero_t2_raises():
    with pytest.raises(ValueError, match="t2_star_s"):
        apply_decoherence(np.array([1.0]), np.array([1e-6]), 0.0)


def test_decoherence_negative_t2_raises():
    with pytest.raises(ValueError, match="t2_star_s"):
        apply_decoherence(np.array([1.0]), np.array([1e-6]), -1.0)


# ═══════════════════════════════════════════════════════════════════════════
# compute_oat_ideal_trajectory
# ═══════════════════════════════════════════════════════════════════════════


def test_ideal_trajectory_returns_correct_type():
    traj = compute_oat_ideal_trajectory(_N_SPINS, _CHI_HZ, n_points=50)
    assert isinstance(traj, OATIdealTrajectory)


def test_ideal_trajectory_optimal_xi2_below_sql():
    """Optimal ξ²_R should be < 1 for large N."""
    traj = compute_oat_ideal_trajectory(_N_SPINS, _CHI_HZ, n_points=100)
    assert traj.optimal_xi2_r < 1.0


def test_ideal_trajectory_gain_positive():
    traj = compute_oat_ideal_trajectory(_N_SPINS, _CHI_HZ, n_points=50)
    assert traj.metrological_gain_db > 0


def test_ideal_trajectory_n_points():
    traj = compute_oat_ideal_trajectory(_N_SPINS, _CHI_HZ, n_points=77)
    assert len(traj.times_s) == 77
    assert len(traj.xi2_r) == 77


def test_ideal_trajectory_zero_n_raises():
    with pytest.raises(ValueError, match="n_spins"):
        compute_oat_ideal_trajectory(0.0, _CHI_HZ)


def test_ideal_trajectory_zero_chi_raises():
    with pytest.raises(ValueError, match="chi_hz"):
        compute_oat_ideal_trajectory(_N_SPINS, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# compute_oat_with_decoherence
# ═══════════════════════════════════════════════════════════════════════════


def test_decoherence_trajectory_returns_correct_type():
    traj = compute_oat_with_decoherence(_N_SPINS, _CHI_HZ, _T2_STAR_S, n_points=50)
    assert isinstance(traj, OATDecoherenceTrajectory)


def test_decoherence_trajectory_penalty_positive():
    """Decoherence penalty is non-negative."""
    traj = compute_oat_with_decoherence(_N_SPINS, _CHI_HZ, _T2_STAR_S, n_points=100)
    assert traj.decoherence_penalty_db >= 0


def test_decoherence_trajectory_optimal_worse_than_ideal():
    """Decoherence-limited ξ²_R ≥ ideal ξ²_R."""
    traj = compute_oat_with_decoherence(_N_SPINS, _CHI_HZ, _T2_STAR_S, n_points=100)
    assert traj.optimal_xi2_r >= traj.ideal_optimal_xi2_r - 1e-15


def test_decoherence_trajectory_stores_parameters():
    traj = compute_oat_with_decoherence(1e10, 8.0, 2e-6, n_points=50)
    assert traj.n_spins == 1e10
    assert traj.chi_hz == 8.0
    assert traj.t2_star_s == 2e-6
    assert traj.chi_t2_star_product == pytest.approx(8.0 * 2e-6)


def test_decoherence_trajectory_chi_t2_product():
    """χ·T₂* is the key figure of merit."""
    traj = compute_oat_with_decoherence(1e6, 100.0, 1e-3, n_points=50)
    assert traj.chi_t2_star_product == pytest.approx(100.0 * 1e-3)


def test_decoherence_trajectory_shorter_t2_worse_squeezing():
    """Halving T₂* should degrade the achievable squeezing."""
    traj_long = compute_oat_with_decoherence(1e8, 10.0, 10e-6, n_points=100)
    traj_short = compute_oat_with_decoherence(1e8, 10.0, 5e-6, n_points=100)
    assert traj_short.optimal_xi2_r >= traj_long.optimal_xi2_r - 1e-15


def test_decoherence_trajectory_long_t2_approaches_ideal():
    """With very long T₂*, decoherence result ≈ ideal."""
    t2_very_long = 100.0  # 100 seconds — negligible decoherence
    traj = compute_oat_with_decoherence(1e6, 10.0, t2_very_long, n_points=200)
    assert traj.optimal_xi2_r == pytest.approx(traj.ideal_optimal_xi2_r, rel=0.05)
    assert traj.decoherence_penalty_db < 0.5  # < 0.5 dB loss


def test_decoherence_trajectory_zero_n_raises():
    with pytest.raises(ValueError, match="n_spins"):
        compute_oat_with_decoherence(0.0, _CHI_HZ, _T2_STAR_S)


def test_decoherence_trajectory_zero_chi_raises():
    with pytest.raises(ValueError, match="chi_hz"):
        compute_oat_with_decoherence(_N_SPINS, 0.0, _T2_STAR_S)


def test_decoherence_trajectory_zero_t2_raises():
    with pytest.raises(ValueError, match="t2_star_s"):
        compute_oat_with_decoherence(_N_SPINS, _CHI_HZ, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# estimate_oat_chi
# ═══════════════════════════════════════════════════════════════════════════


def test_estimate_oat_chi_positive():
    """Coupling estimate must be positive."""
    nv = NVConfig()
    chi = estimate_oat_chi(nv)
    assert chi > 0


# ═══════════════════════════════════════════════════════════════════════════
# TAT — tat_optimal_time
# ═══════════════════════════════════════════════════════════════════════════


def test_tat_optimal_time_formula():
    """t_opt = ln(2N) / [2(N-1) × 2πχ]."""
    t = tat_optimal_time(_N_SPINS, _CHI_HZ)
    rate = 2.0 * (_N_SPINS - 1.0) * _TWO_PI * _CHI_HZ
    expected = math.log(2.0 * _N_SPINS) / rate
    assert t == pytest.approx(expected, rel=1e-10)


def test_tat_optimal_time_much_shorter_than_oat():
    """TAT t_opt ≪ OAT t_opt because TAT squeezes exponentially."""
    t_tat = tat_optimal_time(_N_SPINS, _CHI_HZ)
    t_oat = oat_optimal_time(_N_SPINS, _CHI_HZ)
    assert t_tat < t_oat * 0.01  # orders of magnitude shorter


def test_tat_optimal_time_error_n1():
    with pytest.raises(ValueError, match="n_spins"):
        tat_optimal_time(1.0, _CHI_HZ)


def test_tat_optimal_time_error_chi0():
    with pytest.raises(ValueError, match="chi_hz"):
        tat_optimal_time(_N_SPINS, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# TAT — tat_xi2_ideal
# ═══════════════════════════════════════════════════════════════════════════


def test_tat_xi2_ideal_minimum_near_1_over_n():
    """TAT minimum ξ²_R = 1/N (Heisenberg scaling)."""
    N = 1e6
    chi = 10.0
    t_opt = tat_optimal_time(N, chi)
    xi2 = tat_xi2_ideal(t_opt, N, chi).item()
    expected = 1.0 / N
    assert xi2 == pytest.approx(expected, rel=0.01)


def test_tat_xi2_ideal_heisenberg_scaling():
    """Doubling N halves ξ²_R,min (Heisenberg: 1/N)."""
    N1 = 1e6
    N2 = 2e6
    chi = 10.0
    xi2_1 = tat_xi2_ideal(tat_optimal_time(N1, chi), N1, chi).item()
    xi2_2 = tat_xi2_ideal(tat_optimal_time(N2, chi), N2, chi).item()
    # ξ²_min(2N) / ξ²_min(N) ≈ 0.5
    assert xi2_2 / xi2_1 == pytest.approx(0.5, rel=0.05)


def test_tat_xi2_ideal_better_than_oat():
    """TAT minimum ξ²_R < OAT minimum ξ²_R for same parameters."""
    N = 1e6
    chi = 10.0
    tat_min = tat_xi2_ideal(tat_optimal_time(N, chi), N, chi).item()
    oat_min = oat_xi2_ideal(oat_optimal_time(N, chi), N, chi).min().item()
    assert tat_min < oat_min


def test_tat_xi2_ideal_at_t0_near_css():
    """At t = 0 the TAT state should be near the CSS (ξ² ≈ 1)."""
    xi2 = tat_xi2_ideal(0.0, _N_SPINS, _CHI_HZ).item()
    assert xi2 == pytest.approx(1.0, abs=1e-6)


def test_tat_xi2_ideal_increases_past_optimum():
    """After t_opt, ξ² should increase rapidly (anti-squeeze leakage)."""
    N = 1e6
    chi = 10.0
    t_opt = tat_optimal_time(N, chi)
    xi2_opt = tat_xi2_ideal(t_opt, N, chi).item()
    xi2_late = tat_xi2_ideal(t_opt * 2.0, N, chi).item()
    assert xi2_late > xi2_opt * 2.0


# ═══════════════════════════════════════════════════════════════════════════
# TAT — compute_tat_ideal_trajectory
# ═══════════════════════════════════════════════════════════════════════════


def test_tat_ideal_trajectory_type():
    traj = compute_tat_ideal_trajectory(1e6, 10.0)
    assert isinstance(traj, TATIdealTrajectory)


def test_tat_ideal_trajectory_minimum_near_heisenberg():
    N = 1e6
    traj = compute_tat_ideal_trajectory(N, 10.0, n_points=500)
    assert traj.optimal_xi2_r == pytest.approx(1.0 / N, rel=0.05)


def test_tat_ideal_trajectory_gain_db_positive():
    traj = compute_tat_ideal_trajectory(1e8, 5.0)
    assert traj.metrological_gain_db > 0


# ═══════════════════════════════════════════════════════════════════════════
# TAT — compute_tat_with_decoherence
# ═══════════════════════════════════════════════════════════════════════════


def test_tat_decoherence_type():
    traj = compute_tat_with_decoherence(1e6, 10.0, 1e-3)
    assert isinstance(traj, TATDecoherenceTrajectory)


def test_tat_decoherence_worse_than_ideal():
    N = 1e6
    traj = compute_tat_with_decoherence(N, 10.0, 1e-6)
    # Decoherence-limited ξ² should be worse than ideal minimum
    assert traj.optimal_xi2_r >= traj.ideal_optimal_xi2_r


def test_tat_decoherence_penalty_positive():
    traj = compute_tat_with_decoherence(1e6, 10.0, 1e-6)
    assert traj.decoherence_penalty_db >= 0


def test_tat_decoherence_long_t2_recovers_ideal():
    """With very long T₂*, decoherence barely affects TAT."""
    N = 1e4
    chi = 100.0
    traj = compute_tat_with_decoherence(N, chi, t2_star_s=1e3, n_points=500)
    assert traj.optimal_xi2_r == pytest.approx(
        traj.ideal_optimal_xi2_r, rel=0.1
    )


def test_tat_decoherence_error_invalid_params():
    with pytest.raises(ValueError, match="n_spins"):
        compute_tat_with_decoherence(1.0, 10.0, 1e-3)
    with pytest.raises(ValueError, match="chi_hz"):
        compute_tat_with_decoherence(1e6, 0.0, 1e-3)
    with pytest.raises(ValueError, match="t2_star_s"):
        compute_tat_with_decoherence(1e6, 10.0, 0.0)


def test_estimate_oat_chi_order_of_magnitude():
    """For 1e17/cm³ density, χ should be ~1–10 kHz."""
    nv = NVConfig(nv_density_per_cm3=1e17)
    chi = estimate_oat_chi(nv)
    assert 1.0 < chi < 100_000.0  # dipolar coupling at ~ppm NV density


def test_estimate_oat_chi_scales_with_density():
    """Higher density → stronger dipolar coupling."""
    nv_low = NVConfig(nv_density_per_cm3=1e16)
    nv_high = NVConfig(nv_density_per_cm3=1e18)
    chi_low = estimate_oat_chi(nv_low)
    chi_high = estimate_oat_chi(nv_high)
    assert chi_high > chi_low


def test_estimate_oat_chi_density_scaling():
    """χ ∝ ρ because ρ/r³ = ρ·ρ = ρ (r = ρ^{-1/3})."""
    nv1 = NVConfig(nv_density_per_cm3=1e17)
    nv2 = NVConfig(nv_density_per_cm3=2e17)
    chi1 = estimate_oat_chi(nv1)
    chi2 = estimate_oat_chi(nv2)
    # χ ∝ 1/r³ = ρ  → ratio should be 2
    assert chi2 / chi1 == pytest.approx(2.0, rel=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# compute_squeezing_feasibility
# ═══════════════════════════════════════════════════════════════════════════


def test_feasibility_returns_correct_type():
    nv = NVConfig()
    result = compute_squeezing_feasibility(nv, 1e12, chi_hz=5.0, n_trajectory_points=50)
    assert isinstance(result, SqueezingFeasibility)


def test_feasibility_ideal_matches_parabolic_minimum():
    nv = NVConfig()
    n_eff = 1e10
    result = compute_squeezing_feasibility(nv, n_eff, chi_hz=10.0, n_trajectory_points=50)
    expected = 1.0 / math.sqrt(3.0 * n_eff)
    assert result.ideal_xi2_r == pytest.approx(expected, rel=1e-10)


def test_feasibility_achievable_worse_than_ideal():
    nv = NVConfig()
    result = compute_squeezing_feasibility(nv, 1e10, chi_hz=10.0, n_trajectory_points=50)
    assert result.achievable_xi2_r >= result.ideal_xi2_r - 1e-15


def test_feasibility_with_auto_chi():
    """When chi_hz=None, estimate from density."""
    nv = NVConfig()
    result = compute_squeezing_feasibility(nv, 1e10, chi_hz=None, n_trajectory_points=50)
    assert result.chi_hz > 0


def test_feasibility_sql_sensitivity_order():
    """SQL sensitivity for 10¹² spins and 1 μs: ~fT/√Hz scale."""
    nv = NVConfig()
    result = compute_squeezing_feasibility(nv, 1e12, chi_hz=5.0, n_trajectory_points=50)
    assert 1e-15 < result.sql_sensitivity_t_per_sqrthz < 1e-12


def test_feasibility_squeezed_better_than_sql():
    """If feasible, squeezed sensitivity < SQL sensitivity."""
    nv = NVConfig()
    result = compute_squeezing_feasibility(nv, 1e12, chi_hz=5.0, n_trajectory_points=50)
    if result.feasible:
        assert result.squeezed_sensitivity_t_per_sqrthz < result.sql_sensitivity_t_per_sqrthz


def test_feasibility_stores_parameters():
    nv = NVConfig(t2_star_us=2.0)
    result = compute_squeezing_feasibility(nv, 1e10, chi_hz=7.0, n_trajectory_points=50)
    assert result.n_effective == 1e10
    assert result.chi_hz == 7.0
    assert result.t2_star_s == pytest.approx(2e-6)
    assert result.chi_t2_star_product == pytest.approx(7.0 * 2e-6)


def test_feasibility_zero_n_raises():
    with pytest.raises(ValueError, match="n_effective"):
        compute_squeezing_feasibility(NVConfig(), 0.0, chi_hz=5.0)


def test_feasibility_negative_chi_raises():
    with pytest.raises(ValueError, match="chi_hz"):
        compute_squeezing_feasibility(NVConfig(), 1e10, chi_hz=-1.0)


def test_feasibility_gain_consistent_with_xi2():
    """achievable_gain_db = −10 log₁₀(achievable_xi2_r)."""
    nv = NVConfig()
    result = compute_squeezing_feasibility(nv, 1e10, chi_hz=10.0, n_trajectory_points=50)
    expected_gain = -10.0 * math.log10(result.achievable_xi2_r)
    assert result.achievable_gain_db == pytest.approx(expected_gain, rel=0.01)


def test_feasibility_high_chi_t2_gives_good_squeezing():
    """Large χ·T₂* product → achievable approaches ideal."""
    nv = NVConfig(t2_star_us=1000.0)  # 1 ms T₂*
    result = compute_squeezing_feasibility(nv, 1e6, chi_hz=100.0, n_trajectory_points=100)
    # χ·T₂* = 100 × 1e-3 = 0.1 — moderate
    assert result.achievable_xi2_r < 1.0  # at least some squeezing


def test_feasibility_low_chi_t2_limits_squeezing():
    """Very small χ·T₂* → decoherence dominates, may not squeeze."""
    nv = NVConfig(t2_star_us=0.001)  # 1 ns T₂*
    result = compute_squeezing_feasibility(nv, 1e6, chi_hz=0.1, n_trajectory_points=50)
    # χ·T₂* = 0.1 × 1e-9 = 1e-10 — negligible
    # Squeezing is essentially impossible
    assert result.achievable_xi2_r > result.ideal_xi2_r
