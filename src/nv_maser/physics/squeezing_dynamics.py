"""OAT squeezing dynamics with T₂* decoherence (ADR-013 deferred item).

One-axis twisting (OAT) under H = χ J_z² generates spin-squeezed states
that surpass the Standard Quantum Limit.  However, decoherence during the
squeezing evolution limits the practically achievable squeezing.

Physical model
──────────────
The OAT Hamiltonian generates squeezing at rate χ (the spin–spin nonlinear
coupling constant, Hz).  For an ensemble of N spins, the *ideal* Wineland
squeezing parameter after evolution time t is approximated by the large-N
parabolic form (Kitagawa & Ueda 1993):

    ξ²_R(t) ≈ 1 / [N (μt)²]  +  (N−1)/(12N) (μt)²

where μ = 2πχ is the angular twisting rate (rad/s).  This two-term
expansion has a minimum at:

    μ t_opt = (12/N)^{1/4}       ξ²_R,min ≈ 1/√(3N) ∝ N^{−1/2}

Note: the *exact* Kitagawa–Ueda formula (including mean-spin depletion)
achieves the stronger scaling ξ²_R,min ∝ N^{-2/3}.  The parabolic form
used here provides a conservative (pessimistic) estimate suitable for
the decoherence overlay.

When T₂* decoherence is present, the squeezed state undergoes dephasing.
The combined effect is modelled as (André & Lukin 2002; Schleier-Smith
et al. 2010):

    ξ²_R(t; T₂*) = ξ²_R,ideal(t) · exp(2t/T₂*) + [1 − exp(−2t/T₂*)]

The first term represents the degraded squeezed component, the second
represents the noise floor from decoherence-induced depolarisation.

The *effective optimal time* shifts earlier than the ideal t_opt because
decoherence penalises long evolution.  We find it numerically.

Two-axis twisting (TAT)
───────────────────────
TAT with H = χ(J_x² − J_y²) achieves Heisenberg scaling ξ²_R ∝ 1/N but
requires opposing interactions.  We include the TAT ideal limit for
comparison, with the same decoherence overlay.

References
──────────
Kitagawa, M. & Ueda, M., PRA 47, 5138 (1993).
Ma, J. et al., Phys. Rep. 509, 89 (2011), Sec. 3.2–3.4.
André, A. & Lukin, M. D., PRA 65, 053819 (2002).
Schleier-Smith, M. H. et al., PRL 104, 073604 (2010).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import NVConfig

# ── Physical constants ────────────────────────────────────────────────────
_TWO_PI = 2.0 * math.pi


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class OATIdealTrajectory:
    """Time-resolved OAT squeezing trajectory without decoherence.

    Attributes
    ----------
    times_s : NDArray
        Evolution times sampled (s).
    xi2_r : NDArray
        Wineland squeezing parameter ξ²_R(t) at each time.
    optimal_time_s : float
        Time of minimum ξ²_R (best squeezing).
    optimal_xi2_r : float
        Minimum ξ²_R achieved.
    metrological_gain_db : float
        −10 log₁₀(optimal_xi2_r) at the optimum.
    n_spins : float
        Number of spins N.
    chi_hz : float
        OAT coupling strength χ (Hz).
    """

    times_s: NDArray[np.float64]
    xi2_r: NDArray[np.float64]
    optimal_time_s: float
    optimal_xi2_r: float
    metrological_gain_db: float
    n_spins: float
    chi_hz: float


@dataclass(frozen=True)
class OATDecoherenceTrajectory:
    """Time-resolved OAT squeezing with T₂* decoherence.

    Attributes
    ----------
    times_s : NDArray
        Evolution times sampled (s).
    xi2_r_ideal : NDArray
        Ideal ξ²_R(t) (no decoherence).
    xi2_r_with_decoherence : NDArray
        ξ²_R(t) including T₂* decoherence penalty.
    optimal_time_s : float
        Time that minimises ξ²_R with decoherence.
    optimal_xi2_r : float
        Best achievable ξ²_R with decoherence.
    ideal_optimal_xi2_r : float
        Best ξ²_R *without* decoherence (for comparison).
    metrological_gain_db : float
        −10 log₁₀(optimal_xi2_r) at the decoherence-limited optimum.
    decoherence_penalty_db : float
        Loss of metrological gain due to T₂*:
        G_ideal − G_achievable (positive means degradation).
    n_spins : float
        Number of spins N.
    chi_hz : float
        OAT coupling strength χ (Hz).
    t2_star_s : float
        Dephasing time T₂* (s).
    chi_t2_star_product : float
        Dimensionless figure of merit χ·T₂* — how many twisting
        rotations fit within the coherence time.
    """

    times_s: NDArray[np.float64]
    xi2_r_ideal: NDArray[np.float64]
    xi2_r_with_decoherence: NDArray[np.float64]
    optimal_time_s: float
    optimal_xi2_r: float
    ideal_optimal_xi2_r: float
    metrological_gain_db: float
    decoherence_penalty_db: float
    n_spins: float
    chi_hz: float
    t2_star_s: float
    chi_t2_star_product: float


@dataclass(frozen=True)
class SqueezingFeasibility:
    """Practical feasibility assessment for OAT squeezing of the NV ensemble.

    Attributes
    ----------
    n_effective : float
        Number of oriented + pumped NV spins.
    chi_hz : float
        Estimated OAT coupling χ (Hz).
    t2_star_s : float
        T₂* decoherence time (s).
    chi_t2_star_product : float
        χ·T₂* dimensionless ratio.  Need ≫ 1 for useful squeezing.
    ideal_xi2_r : float
        OAT-optimal ξ²_R ignoring decoherence: N^{−2/3}.
    achievable_xi2_r : float
        Best ξ²_R accounting for T₂*.
    ideal_gain_db : float
        Metrological gain at OAT ideal.
    achievable_gain_db : float
        Metrological gain after T₂* penalty.
    feasible : bool
        True if achievable ξ²_R < 1 (any squeezing below SQL).
    sql_sensitivity_t_per_sqrthz : float
        SQL field sensitivity η_B^SQL for reference.
    squeezed_sensitivity_t_per_sqrthz : float
        Achievable field sensitivity after squeezing.
    """

    n_effective: float
    chi_hz: float
    t2_star_s: float
    chi_t2_star_product: float
    ideal_xi2_r: float
    achievable_xi2_r: float
    ideal_gain_db: float
    achievable_gain_db: float
    feasible: bool
    sql_sensitivity_t_per_sqrthz: float
    squeezed_sensitivity_t_per_sqrthz: float


@dataclass(frozen=True)
class TATIdealTrajectory:
    """Time-resolved TAT squeezing trajectory without decoherence.

    Attributes
    ----------
    times_s : NDArray
        Evolution times sampled (s).
    xi2_r : NDArray
        Wineland squeezing parameter ξ²_R(t) at each time.
    optimal_time_s : float
        Time of minimum ξ²_R (best squeezing).
    optimal_xi2_r : float
        Minimum ξ²_R achieved.
    metrological_gain_db : float
        −10 log₁₀(optimal_xi2_r) at the optimum.
    n_spins : float
        Number of spins N.
    chi_hz : float
        TAT coupling strength χ (Hz).
    """

    times_s: NDArray[np.float64]
    xi2_r: NDArray[np.float64]
    optimal_time_s: float
    optimal_xi2_r: float
    metrological_gain_db: float
    n_spins: float
    chi_hz: float


@dataclass(frozen=True)
class TATDecoherenceTrajectory:
    """Time-resolved TAT squeezing with T₂* decoherence.

    Attributes
    ----------
    times_s : NDArray
        Evolution times sampled (s).
    xi2_r_ideal : NDArray
        Ideal ξ²_R(t) (no decoherence).
    xi2_r_with_decoherence : NDArray
        ξ²_R(t) including T₂* decoherence penalty.
    optimal_time_s : float
        Time that minimises ξ²_R with decoherence.
    optimal_xi2_r : float
        Best achievable ξ²_R with decoherence.
    ideal_optimal_xi2_r : float
        Best ξ²_R *without* decoherence (for comparison).
    metrological_gain_db : float
        −10 log₁₀(optimal_xi2_r) at the decoherence-limited optimum.
    decoherence_penalty_db : float
        Loss of metrological gain due to T₂*.
    n_spins : float
        Number of spins N.
    chi_hz : float
        TAT coupling strength χ (Hz).
    t2_star_s : float
        Dephasing time T₂* (s).
    chi_t2_star_product : float
        Dimensionless figure of merit χ·T₂*.
    """

    times_s: NDArray[np.float64]
    xi2_r_ideal: NDArray[np.float64]
    xi2_r_with_decoherence: NDArray[np.float64]
    optimal_time_s: float
    optimal_xi2_r: float
    ideal_optimal_xi2_r: float
    metrological_gain_db: float
    decoherence_penalty_db: float
    n_spins: float
    chi_hz: float
    t2_star_s: float
    chi_t2_star_product: float


# ═══════════════════════════════════════════════════════════════════════════
# Core computation functions
# ═══════════════════════════════════════════════════════════════════════════


def oat_xi2_ideal(t: float | NDArray, n_spins: float, chi_hz: float) -> NDArray:
    """Ideal OAT squeezing parameter ξ²_R(t) for large N.

    Uses the analytic large-N approximation from Ma et al. (2011) Eq. (3.24):

        ξ²_R(t) = [ 1 + ½N(μt)² − ½√(N²(μt)⁴ + N(μt)²) ] / (N−1)(μt)²)
                 ≈ 1/[N(μt)²]  for short times (μt ≪ 1)

    with μ = 2πχ the angular twisting rate.  For very large N the
    simplified short-time form is accurate near the optimum.

    We use the exact Kitagawa–Ueda variance expression (valid for all t):

        (ΔJ_⊥)²_min = N/4 [ A − √(A² − B²) ]

    where:
        A = 1 + ½(N−1)[1 − cos^{N−2}(2χt)]
        B = ½(N−1) sin(2χt) cos^{N−2}(χt)     (B is NOT used — see note)

    For computational tractability at N ~ 10¹² we use the large-N limit:

        ξ²_R(t) ≈ (1/N) + (N−1)/(2N) × Φ(μt, N)

    where:
        Φ(θ, N) = 1 − cos^{N−2}(θ) − √[(1 − cos^{N−2}(θ))² − sin²(θ)·cos^{2(N−2)}(θ)]

    For extremely large N, cos^{N-2}(θ) → 0 for any θ > 0, and the
    squeezing reduces to the short-time parabolic formula:

        ξ²_R(t) ≈ 1 / [ N (2πχ t)² ]    for 0 < 2πχt ≪ N^{-1/4}

    with parabolic minimum at μ t_opt = (12/N)^{1/4}, giving
    ξ²_R,min ≈ 1/√(3N) ∝ N^{-1/2}.

    We use the short-time parabolic approximation which is accurate near
    the optimum for N ≫ 1.

    Parameters
    ----------
    t : float or NDArray
        Evolution time(s) in seconds.
    n_spins : float
        Number of spins N.  Must be > 0.
    chi_hz : float
        OAT coupling strength χ (Hz).  Must be > 0.

    Returns
    -------
    NDArray
        ξ²_R(t) at each time point (always ≥ 1/N, clipped).
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
    mu = _TWO_PI * chi_hz  # angular twisting rate (rad/s)
    mu_t = mu * t_arr

    # Short-time parabolic form (valid near optimum for large N):
    #   ξ²_R ≈ 1 / [N μ²t²]  +  (N−1)/(12N) μ²t²
    # The first term dominates at short times, the second at longer times.
    # Their sum has a minimum at μt_opt = (6/N)^{1/3}.
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = 1.0 / (n_spins * mu_t**2)
        term2 = (n_spins - 1.0) / (12.0 * n_spins) * mu_t**2
    xi2 = term1 + term2

    # At t=0 the parametric form diverges; clamp to 1.0 (CSS)
    xi2 = np.where(mu_t > 0, xi2, 1.0)

    # Physical floor: Heisenberg limit ξ²_R = 1/N
    xi2 = np.clip(xi2, 1.0 / n_spins, None)

    return xi2


def oat_optimal_time(n_spins: float, chi_hz: float) -> float:
    """Optimal OAT squeezing time t_opt (s) for the parabolic approximation.

    Minimises ξ²_R = 1/(N μ²t²) + (N-1)/(12N) μ²t², giving:

        μ⁴ t⁴ = 12 / (N-1)
        t_opt  = [12 / (N-1)]^{1/4} / μ   where μ = 2πχ

    Parameters
    ----------
    n_spins : float
        Number of spins N.  Must be > 1.
    chi_hz : float
        OAT coupling strength χ (Hz).  Must be > 0.

    Returns
    -------
    float
        Optimal squeezing time in seconds.

    Raises
    ------
    ValueError
        If n_spins ≤ 1 or chi_hz ≤ 0.
    """
    if n_spins <= 1:
        raise ValueError(f"n_spins must be > 1, got {n_spins}")
    if chi_hz <= 0:
        raise ValueError(f"chi_hz must be > 0, got {chi_hz}")
    mu = _TWO_PI * chi_hz
    return (12.0 / (n_spins - 1.0)) ** 0.25 / mu


def apply_decoherence(
    xi2_ideal: NDArray,
    times_s: NDArray,
    t2_star_s: float,
) -> NDArray:
    """Apply T₂* decoherence penalty to ideal squeezing trajectory.

    The decoherence model follows André & Lukin (2002):

        ξ²_R(t) = ξ²_ideal(t) · exp(2t/T₂*) + [1 − exp(−2t/T₂*)]

    The first term represents the squeezed-state variance growing
    exponentially as coherence is lost.  The second term represents the
    depolarisation floor approaching 1 (unsqueezed CSS) as t → ∞.

    Parameters
    ----------
    xi2_ideal : NDArray
        Ideal squeezing trajectory ξ²_R(t).
    times_s : NDArray
        Time array matching xi2_ideal (s).
    t2_star_s : float
        T₂* dephasing time (s).  Must be > 0.

    Returns
    -------
    NDArray
        ξ²_R(t) with decoherence applied.

    Raises
    ------
    ValueError
        If t2_star_s ≤ 0.
    """
    if t2_star_s <= 0:
        raise ValueError(f"t2_star_s must be > 0, got {t2_star_s}")
    ratio = np.asarray(times_s / t2_star_s)
    decay = np.exp(-2.0 * ratio)
    # Clip decay to avoid division by zero for very large t/T₂*
    decay_safe = np.clip(decay, 1e-30, None)
    return xi2_ideal * (1.0 / decay_safe) + (1.0 - decay)


def compute_oat_ideal_trajectory(
    n_spins: float,
    chi_hz: float,
    *,
    n_points: int = 200,
    t_max_factor: float = 5.0,
) -> OATIdealTrajectory:
    """Compute the ideal OAT squeezing trajectory (no decoherence).

    Parameters
    ----------
    n_spins : float
        Number of spins N.  Must be > 0.
    chi_hz : float
        OAT coupling χ (Hz).  Must be > 0.
    n_points : int
        Number of time samples.
    t_max_factor : float
        Sample up to t_max_factor × t_opt.

    Returns
    -------
    OATIdealTrajectory
        Full trajectory with optimal time and ξ²_R.

    Raises
    ------
    ValueError
        If n_spins or chi_hz ≤ 0.
    """
    if n_spins <= 0:
        raise ValueError(f"n_spins must be > 0, got {n_spins}")
    if chi_hz <= 0:
        raise ValueError(f"chi_hz must be > 0, got {chi_hz}")

    t_opt = oat_optimal_time(n_spins, chi_hz)
    times = np.linspace(t_opt * 0.01, t_opt * t_max_factor, n_points)
    xi2 = oat_xi2_ideal(times, n_spins, chi_hz)

    idx_min = int(np.argmin(xi2))
    opt_xi2 = float(xi2[idx_min])
    gain_db = -10.0 * math.log10(max(opt_xi2, 1e-300))

    return OATIdealTrajectory(
        times_s=times,
        xi2_r=xi2,
        optimal_time_s=float(times[idx_min]),
        optimal_xi2_r=opt_xi2,
        metrological_gain_db=gain_db,
        n_spins=n_spins,
        chi_hz=chi_hz,
    )


def compute_oat_with_decoherence(
    n_spins: float,
    chi_hz: float,
    t2_star_s: float,
    *,
    n_points: int = 200,
    t_max_factor: float = 5.0,
) -> OATDecoherenceTrajectory:
    """Compute OAT squeezing trajectory with T₂* decoherence.

    Parameters
    ----------
    n_spins : float
        Number of spins N.  Must be > 0.
    chi_hz : float
        OAT coupling χ (Hz).  Must be > 0.
    t2_star_s : float
        Dephasing time T₂* (s).  Must be > 0.
    n_points : int
        Number of time samples.
    t_max_factor : float
        Sample up to t_max_factor × t_opt.

    Returns
    -------
    OATDecoherenceTrajectory
        Trajectory with both ideal and decoherence-limited squeezing.

    Raises
    ------
    ValueError
        If any parameter ≤ 0.
    """
    if n_spins <= 0:
        raise ValueError(f"n_spins must be > 0, got {n_spins}")
    if chi_hz <= 0:
        raise ValueError(f"chi_hz must be > 0, got {chi_hz}")
    if t2_star_s <= 0:
        raise ValueError(f"t2_star_s must be > 0, got {t2_star_s}")

    t_opt_ideal = oat_optimal_time(n_spins, chi_hz)
    times = np.linspace(t_opt_ideal * 0.01, t_opt_ideal * t_max_factor, n_points)

    xi2_ideal = oat_xi2_ideal(times, n_spins, chi_hz)
    xi2_deco = apply_decoherence(xi2_ideal, times, t2_star_s)

    # Find optima
    idx_ideal = int(np.argmin(xi2_ideal))
    ideal_opt = float(xi2_ideal[idx_ideal])

    idx_deco = int(np.argmin(xi2_deco))
    deco_opt = float(xi2_deco[idx_deco])

    gain_deco = -10.0 * math.log10(max(deco_opt, 1e-300))
    gain_ideal = -10.0 * math.log10(max(ideal_opt, 1e-300))
    penalty = gain_ideal - gain_deco

    return OATDecoherenceTrajectory(
        times_s=times,
        xi2_r_ideal=xi2_ideal,
        xi2_r_with_decoherence=xi2_deco,
        optimal_time_s=float(times[idx_deco]),
        optimal_xi2_r=deco_opt,
        ideal_optimal_xi2_r=ideal_opt,
        metrological_gain_db=gain_deco,
        decoherence_penalty_db=penalty,
        n_spins=n_spins,
        chi_hz=chi_hz,
        t2_star_s=t2_star_s,
        chi_t2_star_product=chi_hz * t2_star_s,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAT (Two-Axis Twisting) functions
# ═══════════════════════════════════════════════════════════════════════════


def tat_xi2_ideal(t: float | NDArray, n_spins: float, chi_hz: float) -> NDArray:
    """Ideal TAT squeezing parameter ξ²_R(t) for large N.

    Two-axis twisting H = χ(J_x² − J_y²) generates faster squeezing
    than OAT and reaches the Heisenberg limit ξ²_R ∝ 1/N.

    The large-N model uses a two-exponential form
    (Kitagawa & Ueda 1993, Ma et al. 2011 Sec. 3.4):

        ξ²_R(t) = exp(−Γt) + exp(+Γt) / (4N²)

    where Γ = 2(N−1)μ and μ = 2πχ is the angular twisting rate.

    Key properties:

    * At t → 0⁺: ξ² → 1 + 1/(4N²) ≈ 1 (coherent spin state).
    * Minimum: ξ²_min = 1/N at t_opt = ln(2N)/Γ (Heisenberg scaling).
    * After optimum: anti-squeezed quadrature leakage causes rapid
      degradation.

    Parameters
    ----------
    t : float or NDArray
        Evolution time(s) in seconds.
    n_spins : float
        Number of spins N.  Must be > 0.
    chi_hz : float
        TAT coupling strength χ (Hz).  Must be > 0.

    Returns
    -------
    NDArray
        ξ²_R(t) at each time point (clipped to Heisenberg floor 1/N).
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
    mu = _TWO_PI * chi_hz
    rate = 2.0 * (n_spins - 1.0) * mu  # Γ
    gt = rate * t_arr

    with np.errstate(over="ignore"):
        squeeze = np.exp(-gt)
        anti_squeeze = np.exp(gt) / (4.0 * n_spins**2)

    xi2 = squeeze + anti_squeeze
    # At t=0 the model gives 1 + 1/(4N²) ≈ 1; clamp CSS at 1.0
    xi2 = np.where(gt > 0, xi2, 1.0)
    # Heisenberg floor
    xi2 = np.clip(xi2, 1.0 / n_spins, None)
    return xi2


def tat_optimal_time(n_spins: float, chi_hz: float) -> float:
    """Optimal TAT squeezing time t_opt (s).

    Minimises ξ²_R = exp(−Γt) + exp(+Γt)/(4N²), giving:

        t_opt = ln(2N) / Γ  =  ln(2N) / [2(N−1) × 2πχ]

    This is much shorter than the OAT optimal time and scales as
    ln(N)/(Nχ) for large N.

    Parameters
    ----------
    n_spins : float
        Number of spins N.  Must be > 1.
    chi_hz : float
        TAT coupling strength χ (Hz).  Must be > 0.

    Returns
    -------
    float
        Optimal squeezing time in seconds.

    Raises
    ------
    ValueError
        If n_spins ≤ 1 or chi_hz ≤ 0.
    """
    if n_spins <= 1:
        raise ValueError(f"n_spins must be > 1, got {n_spins}")
    if chi_hz <= 0:
        raise ValueError(f"chi_hz must be > 0, got {chi_hz}")
    mu = _TWO_PI * chi_hz
    rate = 2.0 * (n_spins - 1.0) * mu
    return math.log(2.0 * n_spins) / rate


def compute_tat_ideal_trajectory(
    n_spins: float,
    chi_hz: float,
    *,
    n_points: int = 200,
    t_max_factor: float = 3.0,
) -> TATIdealTrajectory:
    """Compute the ideal TAT squeezing trajectory (no decoherence).

    Parameters
    ----------
    n_spins : float
        Number of spins N.  Must be > 1.
    chi_hz : float
        TAT coupling χ (Hz).  Must be > 0.
    n_points : int
        Number of time samples.
    t_max_factor : float
        Sample up to t_max_factor × t_opt.

    Returns
    -------
    TATIdealTrajectory
    """
    if n_spins <= 1:
        raise ValueError(f"n_spins must be > 1, got {n_spins}")
    if chi_hz <= 0:
        raise ValueError(f"chi_hz must be > 0, got {chi_hz}")

    t_opt = tat_optimal_time(n_spins, chi_hz)
    times = np.linspace(t_opt * 0.01, t_opt * t_max_factor, n_points)
    xi2 = tat_xi2_ideal(times, n_spins, chi_hz)

    idx_min = int(np.argmin(xi2))
    opt_xi2 = float(xi2[idx_min])
    gain_db = -10.0 * math.log10(max(opt_xi2, 1e-300))

    return TATIdealTrajectory(
        times_s=times,
        xi2_r=xi2,
        optimal_time_s=float(times[idx_min]),
        optimal_xi2_r=opt_xi2,
        metrological_gain_db=gain_db,
        n_spins=n_spins,
        chi_hz=chi_hz,
    )


def compute_tat_with_decoherence(
    n_spins: float,
    chi_hz: float,
    t2_star_s: float,
    *,
    n_points: int = 200,
    t_max_factor: float = 3.0,
) -> TATDecoherenceTrajectory:
    """Compute TAT squeezing trajectory with T₂* decoherence.

    Uses the same André & Lukin (2002) decoherence overlay as OAT:

        ξ²_R(t; T₂*) = ξ²_ideal(t) · exp(2t/T₂*) + [1 − exp(−2t/T₂*)]

    Parameters
    ----------
    n_spins : float
        Number of spins N.  Must be > 1.
    chi_hz : float
        TAT coupling χ (Hz).  Must be > 0.
    t2_star_s : float
        Dephasing time T₂* (s).  Must be > 0.
    n_points : int
        Number of time samples.
    t_max_factor : float
        Sample up to t_max_factor × t_opt.

    Returns
    -------
    TATDecoherenceTrajectory
    """
    if n_spins <= 1:
        raise ValueError(f"n_spins must be > 1, got {n_spins}")
    if chi_hz <= 0:
        raise ValueError(f"chi_hz must be > 0, got {chi_hz}")
    if t2_star_s <= 0:
        raise ValueError(f"t2_star_s must be > 0, got {t2_star_s}")

    t_opt = tat_optimal_time(n_spins, chi_hz)
    times = np.linspace(t_opt * 0.01, t_opt * t_max_factor, n_points)

    xi2_ideal = tat_xi2_ideal(times, n_spins, chi_hz)
    xi2_deco = apply_decoherence(xi2_ideal, times, t2_star_s)

    idx_ideal = int(np.argmin(xi2_ideal))
    ideal_opt = float(xi2_ideal[idx_ideal])

    idx_deco = int(np.argmin(xi2_deco))
    deco_opt = float(xi2_deco[idx_deco])

    gain_deco = -10.0 * math.log10(max(deco_opt, 1e-300))
    gain_ideal = -10.0 * math.log10(max(ideal_opt, 1e-300))
    penalty = gain_ideal - gain_deco

    return TATDecoherenceTrajectory(
        times_s=times,
        xi2_r_ideal=xi2_ideal,
        xi2_r_with_decoherence=xi2_deco,
        optimal_time_s=float(times[idx_deco]),
        optimal_xi2_r=deco_opt,
        ideal_optimal_xi2_r=ideal_opt,
        metrological_gain_db=gain_deco,
        decoherence_penalty_db=penalty,
        n_spins=n_spins,
        chi_hz=chi_hz,
        t2_star_s=t2_star_s,
        chi_t2_star_product=chi_hz * t2_star_s,
    )


def estimate_oat_chi(nv_config: NVConfig) -> float:
    """Estimate the OAT coupling strength χ for the NV ensemble.

    The OAT coupling arises from dipolar spin-spin interactions within
    the diamond.  For NV centres at concentration ρ (per m³), the mean
    dipolar coupling is:

        χ_dipolar ≈ (μ₀ γ² ℏ) / (4π r³)

    where r ≈ ρ^{-1/3} is the mean NV–NV separation.  We use
    γ = 2π × 28.025 GHz/T (NV electron spin).

    For ρ = 10¹⁷/cm³ = 10²³/m³:
        r ≈ (10²³)^{-1/3} ≈ 2.15 nm
        χ ≈ 3–10 Hz (varies by lattice averaging factor)

    This is a rough order-of-magnitude estimate; experimental values
    should be used when available.

    Parameters
    ----------
    nv_config : NVConfig
        NV centre configuration (uses density).

    Returns
    -------
    float
        Estimated χ in Hz.
    """
    rho_m3 = nv_config.nv_density_per_cm3 * 1e6  # per cm³ → per m³
    r_mean = rho_m3 ** (-1.0 / 3.0)  # mean separation (m)

    # Physical constants
    mu0 = 4e-7 * math.pi  # T·m/A
    hbar = 1.0546e-34  # J·s
    gamma = _TWO_PI * nv_config.gamma_e_ghz_per_t * 1e9  # rad/s/T

    # Dipolar coupling energy / ℏ → angular frequency → Hz
    chi_rad_s = (mu0 * gamma**2 * hbar) / (4.0 * math.pi * r_mean**3)
    return chi_rad_s / _TWO_PI


def compute_squeezing_feasibility(
    nv_config: NVConfig,
    n_effective: float,
    chi_hz: float | None = None,
    *,
    n_trajectory_points: int = 200,
) -> SqueezingFeasibility:
    """Assess practical feasibility of OAT squeezing for the NV ensemble.

    Brings together the OAT coupling estimate, T₂* decoherence time,
    and effective spin count to determine whether sub-SQL sensitivity
    is achievable.

    Parameters
    ----------
    nv_config : NVConfig
        NV centre configuration.
    n_effective : float
        Effectively inverted spin count N_eff.  Must be > 0.
    chi_hz : float or None
        OAT coupling χ in Hz.  If None, estimated from nv_config density.
    n_trajectory_points : int
        Resolution for the internal trajectory computation.

    Returns
    -------
    SqueezingFeasibility
        Feasibility assessment with ideal and achievable figures.

    Raises
    ------
    ValueError
        If n_effective ≤ 0.
    """
    if n_effective <= 0:
        raise ValueError(f"n_effective must be > 0, got {n_effective}")

    if chi_hz is None:
        chi_hz = estimate_oat_chi(nv_config)
    if chi_hz <= 0:
        raise ValueError(f"chi_hz must be > 0, got {chi_hz}")

    t2_star_s = nv_config.t2_star_us * 1e-6

    # Ideal OAT optimum (parabolic approximation: 1/sqrt(3N))
    ideal_xi2 = 1.0 / math.sqrt(3.0 * n_effective)
    ideal_gain = -10.0 * math.log10(max(ideal_xi2, 1e-300))

    # Decoherence-limited trajectory
    traj = compute_oat_with_decoherence(
        n_spins=n_effective,
        chi_hz=chi_hz,
        t2_star_s=t2_star_s,
        n_points=n_trajectory_points,
    )

    # SQL sensitivity for reference
    gamma_hz = nv_config.gamma_e_ghz_per_t * 1e9
    sql_b = 1.0 / (_TWO_PI * gamma_hz * math.sqrt(n_effective * t2_star_s))
    squeezed_b = math.sqrt(traj.optimal_xi2_r) * sql_b

    return SqueezingFeasibility(
        n_effective=n_effective,
        chi_hz=chi_hz,
        t2_star_s=t2_star_s,
        chi_t2_star_product=chi_hz * t2_star_s,
        ideal_xi2_r=ideal_xi2,
        achievable_xi2_r=traj.optimal_xi2_r,
        ideal_gain_db=ideal_gain,
        achievable_gain_db=traj.metrological_gain_db,
        feasible=traj.optimal_xi2_r < 1.0,
        sql_sensitivity_t_per_sqrthz=sql_b,
        squeezed_sensitivity_t_per_sqrthz=squeezed_b,
    )
