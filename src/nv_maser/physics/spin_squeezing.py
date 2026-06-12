"""Spin projection noise and quantum-enhanced magnetometry limits.

For N independent spin-½ particles, the measurement uncertainty is bounded
below by the Standard Quantum Limit (SQL) set by quantum projection noise.
Entanglement (spin squeezing) can reduce fluctuations below the SQL, towards
the fundamental Heisenberg Limit (HL).

This module computes the three complementary quantum sensitivity floors for
the NV diamond maser magnetometer and characterises how spin squeezing
potentially improves field sensitivity beyond the projection-noise floor.

Key quantities
──────────────
Projection noise:  δφ_SQL = 1/√N                           [rad/shot]
SQL sensitivity:   η_B_SQL = 1 / (γ √(N τ))               [T/√Hz]
Heisenberg limit:  η_B_HL  = 1 / (γ  N  √τ)               [T/√Hz]
Wineland criterion:ξ²_R = N ⟨ΔJ_⊥²⟩_min / ⟨J_z⟩²         [dimensionless]
OAT optimum:       ξ²_R^{min,OAT} ≈ N^{-2/3}              (large-N)
TAT scaling:       ξ²_R^{min,TAT} ∝ N^{-1}                (Heisenberg)
Metrological gain: G = −10 log₁₀(ξ²_R)                   [dB]

Physical picture
────────────────
The NV maser can operate in two distinct sensing modes:

1. **Oscillator mode** (Schawlow-Townes): The cavity output is a microwave
   signal whose frequency stability is limited by spontaneous-emission phase
   diffusion (quantum Langevin noise).  This is captured by ``stability.py``
   and ``sensitivity.py``.

2. **Spin-ensemble mode** (Ramsey / CPT): The NV spins accumulate phase due
   to the field.  The readout is limited by spin projection noise → SQL.

For N_eff ≈ 10¹² spins and T₂* = 1 µs the SQL evaluates to:

    η_B_SQL = 1 / (2π × 28.025e9 × √(10¹² × 10⁻⁶)) ≈ 6 fT/√Hz

This is the fundamental floor that sets the ultimate goal for the shimming
controller: no algorithm can improve sensitivity below the SQL without
entanglement.  Spin squeezing (OAT/TAT interactions) would in principle
allow approaching the Heisenberg limit η_B_HL ≈ 6 aT/√Hz.

Squeezing regimes (Wineland criterion)
───────────────────────────────────────
ξ²_R = 1           coherent spin state (CSS), at the SQL
ξ²_R < 1           metrologically useful entanglement (below SQL)
ξ²_R = N^{-2/3}    OAT optimal squeezing (large-N asymptote)
ξ²_R = 1/N         Heisenberg-limited (maximally entangled GHZ)

References
──────────
Wineland et al., PRA 46, R6797 (1992).
Kitagawa & Ueda, PRA 47, 5138 (1993).
Degen, Reinhard & Cappellaro, Rev. Mod. Phys. 89, 035002 (2017).
Ma et al., Phys. Rep. 509, 89-165 (2011).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import NVConfig

# ── Physical constant ──────────────────────────────────────────────────────
_TWO_PI = 2.0 * math.pi

# ── Regime labels ──────────────────────────────────────────────────────────
REGIME_COHERENT = "COHERENT"
"""Coherent spin state: ξ²_R ≥ 1.  At the SQL."""

REGIME_SQUEEZED = "SQUEEZED"
"""Spin-squeezed below the SQL: 1/N < ξ²_R < 1."""

REGIME_NEAR_HEISENBERG = "NEAR_HEISENBERG"
"""Close to the Heisenberg limit: ξ²_R ≤ N^{-1/2}."""


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ProjectionNoiseResult:
    """Standard Quantum Limit and Heisenberg Limit for N independent spins.

    All sensitivity figures are spectral densities in T/√Hz
    (one-sided, equivalent to the sensitivity at 1 Hz bandwidth).
    """

    n_spins: float
    """Number of effective spins contributing to the measurement."""

    interrogation_time_s: float
    """Free-precession (Ramsey) interrogation time τ [s].
    The maximum useful value is T₂* (ensemble dephasing time)."""

    # ── Phase sensitivities [rad per shot] ─────────────────────────────────
    sql_phase_rad: float
    """Projection-noise phase uncertainty per shot: δφ_SQL = 1/√N  [rad]."""

    hl_phase_rad: float
    """Heisenberg-limited phase uncertainty per shot: δφ_HL = 1/N  [rad]."""

    # ── Spectral density field sensitivities [T/√Hz] ───────────────────────
    sql_field_t_per_sqrthz: float
    """SQL field sensitivity: η_B_SQL = 1/(2π γ_e √(N τ))  [T/√Hz]."""

    hl_field_t_per_sqrthz: float
    """Heisenberg limit: η_B_HL = 1/(2π γ_e N √τ)  [T/√Hz]."""

    # ── Quantum advantage ──────────────────────────────────────────────────
    heisenberg_advantage: float
    """√N: ratio η_B_SQL / η_B_HL.
    The maximum improvement entanglement can offer over the SQL."""


@dataclass(frozen=True)
class SpinSqueezingResult:
    """Wineland squeezing parameter and metrological characterisation."""

    n_spins: float
    """Number of spins the squeezing was evaluated for."""

    xi2_r: float
    """Wineland squeezing parameter ξ²_R = N ⟨ΔJ_⊥²⟩ / ⟨J_z⟩²  [dimensionless].
    ξ²_R = 1 → coherent state (at SQL).
    ξ²_R < 1 → metrologically useful entanglement (below SQL)."""

    metrological_gain_db: float
    """Metrological gain G = −10 log₁₀(ξ²_R)  [dB].
    G = 0 dB → coherent state.  G > 0 dB → improvement over SQL."""

    below_sql: bool
    """True when ξ²_R < 1 (state is metrologically entangled)."""

    regime: str
    """One of REGIME_COHERENT / REGIME_SQUEEZED / REGIME_NEAR_HEISENBERG."""


@dataclass(frozen=True)
class QuantumEnhancementResult:
    """Full quantum measurement enhancement analysis for the NV ensemble.

    Combines projection-noise limits, squeezing characterisation, and
    the theoretical OAT-optimal sensitivity for the given spin count.
    """

    projection_noise: ProjectionNoiseResult
    """SQL and HL limits for the spin ensemble."""

    squeezing: SpinSqueezingResult
    """Wineland characterisation of the spin state."""

    # ── Sensitivity floors [T/√Hz] ─────────────────────────────────────────
    squeezed_field_sensitivity_t_per_sqrthz: float
    """Sensitivity after squeezing: η_B = ξ_R × η_B_SQL  [T/√Hz]."""

    oat_optimal_xi2_r: float
    """Best ξ²_R achievable via one-axis twisting: N^{-2/3}."""

    oat_best_sensitivity_t_per_sqrthz: float
    """Sensitivity at OAT optimum: √(ξ²_R^{OAT}) × η_B_SQL  [T/√Hz]."""

    heisenberg_sensitivity_t_per_sqrthz: float
    """Fundamental Heisenberg limit η_B_HL  [T/√Hz].  See projection_noise."""


# ═══════════════════════════════════════════════════════════════════════════
# Primitive functions
# ═══════════════════════════════════════════════════════════════════════════


def compute_sql_phase_sensitivity(n_spins: float) -> float:
    """Standard Quantum Limit phase uncertainty per shot: δφ_SQL = 1/√N  [rad].

    For N independent spin-½ particles in a maximally stretched coherent
    spin state the angular-momentum measurement variance satisfies:
        ⟨ΔJ_⊥²⟩ = N/4
    giving a per-shot standard deviation in the inferred phase of 1/√N.

    Args:
        n_spins: Number of spins N.  Must be > 0.

    Returns:
        SQL phase uncertainty in radians.

    Raises:
        ValueError: If n_spins ≤ 0.
    """
    if n_spins <= 0.0:
        raise ValueError(f"n_spins must be > 0, got {n_spins!r}")
    return 1.0 / math.sqrt(n_spins)


def compute_hl_phase_sensitivity(n_spins: float) -> float:
    """Heisenberg-limited phase uncertainty per shot: δφ_HL = 1/N  [rad].

    For maximally entangled N-particle states (e.g. GHZ/NOON states) the
    phase uncertainty scales as 1/N rather than 1/√N.

    Args:
        n_spins: Number of spins N.  Must be > 0.

    Returns:
        Heisenberg-limited phase uncertainty in radians.

    Raises:
        ValueError: If n_spins ≤ 0.
    """
    if n_spins <= 0.0:
        raise ValueError(f"n_spins must be > 0, got {n_spins!r}")
    return 1.0 / n_spins


def compute_sql_field_sensitivity(
    n_spins: float,
    interrogation_time_s: float,
    gamma_e_hz_per_t: float,
) -> float:
    """SQL field sensitivity spectral density [T/√Hz].

    From Degen, Reinhard & Cappellaro (2017) Eq. (9):

        η_B_SQL = 1 / (2π γ_e √(N τ))

    where γ_e is the gyromagnetic ratio in Hz/T and τ is the free-
    precession interrogation time.

    Args:
        n_spins: Number of spins N.  Must be > 0.
        interrogation_time_s: Free-precession time τ [s].  Must be > 0.
        gamma_e_hz_per_t: Gyromagnetic ratio γ_e [Hz/T] (= 28.025e9 for NV).

    Returns:
        SQL field sensitivity [T/√Hz].

    Raises:
        ValueError: If n_spins or interrogation_time_s ≤ 0.
    """
    if n_spins <= 0.0:
        raise ValueError(f"n_spins must be > 0, got {n_spins!r}")
    if interrogation_time_s <= 0.0:
        raise ValueError(
            f"interrogation_time_s must be > 0, got {interrogation_time_s!r}"
        )
    gamma_rad = _TWO_PI * gamma_e_hz_per_t
    return 1.0 / (gamma_rad * math.sqrt(n_spins * interrogation_time_s))


def compute_hl_field_sensitivity(
    n_spins: float,
    interrogation_time_s: float,
    gamma_e_hz_per_t: float,
) -> float:
    """Heisenberg-limited field sensitivity spectral density [T/√Hz].

        η_B_HL = 1 / (2π γ_e N √τ)

    The Heisenberg limit is √N times better than the SQL.

    Args:
        n_spins: Number of spins N.  Must be > 0.
        interrogation_time_s: Free-precession time τ [s].  Must be > 0.
        gamma_e_hz_per_t: Gyromagnetic ratio γ_e [Hz/T].

    Returns:
        Heisenberg-limited field sensitivity [T/√Hz].

    Raises:
        ValueError: If n_spins or interrogation_time_s ≤ 0.
    """
    if n_spins <= 0.0:
        raise ValueError(f"n_spins must be > 0, got {n_spins!r}")
    if interrogation_time_s <= 0.0:
        raise ValueError(
            f"interrogation_time_s must be > 0, got {interrogation_time_s!r}"
        )
    gamma_rad = _TWO_PI * gamma_e_hz_per_t
    return 1.0 / (gamma_rad * n_spins * math.sqrt(interrogation_time_s))


def compute_projection_noise(
    n_spins: float,
    interrogation_time_s: float,
    gamma_e_hz_per_t: float = 28.025e9,
) -> ProjectionNoiseResult:
    """Assemble SQL and Heisenberg-limit results for a given spin ensemble.

    Args:
        n_spins: Number of effectively inverted spins N.  Must be > 0.
        interrogation_time_s: Free-precession time τ [s].  Must be > 0.
        gamma_e_hz_per_t: NV gyromagnetic ratio [Hz/T].
            Default: 28.025e9 (|0⟩→|−1⟩ transition).

    Returns:
        ProjectionNoiseResult with SQL, HL, and Heisenberg advantage.
    """
    sql_phi = compute_sql_phase_sensitivity(n_spins)
    hl_phi = compute_hl_phase_sensitivity(n_spins)
    sql_b = compute_sql_field_sensitivity(n_spins, interrogation_time_s, gamma_e_hz_per_t)
    hl_b = compute_hl_field_sensitivity(n_spins, interrogation_time_s, gamma_e_hz_per_t)

    return ProjectionNoiseResult(
        n_spins=n_spins,
        interrogation_time_s=interrogation_time_s,
        sql_phase_rad=sql_phi,
        hl_phase_rad=hl_phi,
        sql_field_t_per_sqrthz=sql_b,
        hl_field_t_per_sqrthz=hl_b,
        heisenberg_advantage=math.sqrt(n_spins),
    )


def compute_wineland_squeezing(
    n_spins: float,
    variance_perp: float,
    mean_jz: float,
) -> float:
    """Wineland squeezing parameter ξ²_R from measured spin statistics.

    ξ²_R = N ⟨ΔJ_⊥²⟩_min / ⟨J_z⟩²

    where ⟨ΔJ_⊥²⟩_min is the minimum variance of a spin component
    perpendicular to the mean-spin direction, and ⟨J_z⟩ is the
    mean-spin projection along the reference axis.

    Reference values:
    - Coherent spin state: ξ²_R = 1.  (⟨ΔJy²⟩ = N/4, ⟨Jz⟩ = N/2)
    - GHZ / NOON state:   ξ²_R = 1/N.  (Heisenberg limited)

    Args:
        n_spins: Number of spins N.
        variance_perp: Minimum perpendicular spin variance ⟨ΔJ_⊥²⟩ [dimensionless].
        mean_jz: Mean spin projection ⟨J_z⟩ along the reference axis [dimensionless].
            For a fully polarised CSS: ⟨Jz⟩ = N/2.

    Returns:
        Wineland squeezing parameter ξ²_R (dimensionless).

    Raises:
        ValueError: If mean_jz is zero (undefined polarisation axis).
    """
    if mean_jz == 0.0:
        raise ValueError("mean_jz must be non-zero (undefined polarisation axis)")
    return n_spins * variance_perp / (mean_jz**2)


def compute_oat_optimal_squeezing(n_spins: float) -> float:
    """Minimum ξ²_R achievable by one-axis twisting (large-N approximation).

    Under OAT (H = χ J_z²), the minimum Wineland squeezing parameter after
    optimising the evolution time scales asymptotically as:

        ξ²_R^{min,OAT} ≈ N^{-2/3}    (large N)

    This is better than the SQL (ξ²_R = 1) but worse than the Heisenberg
    limit (ξ²_R = 1/N).  Achieving the full Heisenberg scaling requires
    two-axis twisting (TAT) or equivalent protocols.

    Reference: Kitagawa & Ueda, PRA 47, 5138 (1993);
               Ma et al., Phys. Rep. 509, 89 (2011), Sec. 3.2.

    Args:
        n_spins: Number of spins N.  Must be > 0.

    Returns:
        ξ²_R^{min,OAT} ≈ N^{-2/3}  [dimensionless].

    Raises:
        ValueError: If n_spins ≤ 0.
    """
    if n_spins <= 0.0:
        raise ValueError(f"n_spins must be > 0, got {n_spins!r}")
    return n_spins ** (-2.0 / 3.0)


def compute_metrological_gain_db(xi2_r: float) -> float:
    """Metrological gain from squeezing G = −10 log₁₀(ξ²_R)  [dB].

    G = 0 dB  → coherent spin state, at the SQL.
    G > 0 dB  → below the SQL; 10 dB gain ≡ ξ²_R = 0.1.

    Args:
        xi2_r: Wineland squeezing parameter ξ²_R.  Must be > 0.

    Returns:
        Metrological gain in decibels.

    Raises:
        ValueError: If xi2_r ≤ 0.
    """
    if xi2_r <= 0.0:
        raise ValueError(f"xi2_r must be > 0, got {xi2_r!r}")
    return -10.0 * math.log10(xi2_r)


def classify_squeezing_regime(xi2_r: float, n_spins: float) -> str:
    """Classify the spin state into a squeezing regime.

    Thresholds:
    - COHERENT:        ξ²_R ≥ 1               (at or above the SQL)
    - NEAR_HEISENBERG: ξ²_R ≤ 1/√N            (within factor √N of the HL)
    - SQUEEZED:        1/√N < ξ²_R < 1         (below SQL, not yet near HL)

    Args:
        xi2_r: Wineland squeezing parameter ξ²_R.
        n_spins: Number of spins N (used for the near-Heisenberg threshold).

    Returns:
        One of REGIME_COHERENT, REGIME_SQUEEZED, REGIME_NEAR_HEISENBERG.

    Raises:
        ValueError: If n_spins ≤ 0.
    """
    if n_spins <= 0.0:
        raise ValueError(f"n_spins must be > 0, got {n_spins!r}")
    if xi2_r >= 1.0:
        return REGIME_COHERENT
    near_hl_threshold = 1.0 / math.sqrt(n_spins)
    if xi2_r <= near_hl_threshold:
        return REGIME_NEAR_HEISENBERG
    return REGIME_SQUEEZED


def compute_spin_squeezing(n_spins: float, xi2_r: float) -> SpinSqueezingResult:
    """Assemble a full squeezing characterisation from ξ²_R and N.

    Args:
        n_spins: Number of spins N.  Must be > 0.
        xi2_r: Wineland squeezing parameter ξ²_R.  Must be > 0.

    Returns:
        SpinSqueezingResult with gain, regime, and below-SQL flag.

    Raises:
        ValueError: If n_spins or xi2_r ≤ 0.
    """
    if n_spins <= 0.0:
        raise ValueError(f"n_spins must be > 0, got {n_spins!r}")
    if xi2_r <= 0.0:
        raise ValueError(f"xi2_r must be > 0, got {xi2_r!r}")

    gain_db = compute_metrological_gain_db(xi2_r)
    regime = classify_squeezing_regime(xi2_r, n_spins)

    return SpinSqueezingResult(
        n_spins=n_spins,
        xi2_r=xi2_r,
        metrological_gain_db=gain_db,
        below_sql=xi2_r < 1.0,
        regime=regime,
    )


# ═══════════════════════════════════════════════════════════════════════════
# High-level analysis
# ═══════════════════════════════════════════════════════════════════════════


def compute_quantum_enhancement(
    nv_config: NVConfig,
    n_effective: float,
    interrogation_time_s: float | None = None,
    xi2_r: float = 1.0,
) -> QuantumEnhancementResult:
    """Full quantum measurement enhancement analysis for the NV ensemble.

    Computes the SQL, Heisenberg limit, OAT optimum, and the sensitivity
    achievable with a given squeezing parameter ξ²_R, all derived from the
    NV configuration and the number of effectively contributing spins.

    Args:
        nv_config: NV centre parameters (for γ_e and T₂*).
        n_effective: Number of effectively inverted spins N_eff.  Must be > 0.
        interrogation_time_s: Free-precession time τ [s].  Defaults to T₂*
            from nv_config when None.
        xi2_r: Wineland squeezing parameter ξ²_R of the current spin state.
            Default 1.0 → coherent state (no squeezing, at the SQL).

    Returns:
        QuantumEnhancementResult with all sensitivity floors.

    Raises:
        ValueError: If n_effective ≤ 0 or xi2_r ≤ 0.
    """
    if n_effective <= 0.0:
        raise ValueError(f"n_effective must be > 0, got {n_effective!r}")
    if xi2_r <= 0.0:
        raise ValueError(f"xi2_r must be > 0, got {xi2_r!r}")

    t_meas = (
        interrogation_time_s
        if interrogation_time_s is not None
        else nv_config.t2_star_us * 1e-6
    )

    gamma_e = nv_config.gamma_e_ghz_per_t * 1e9  # Hz/T

    pn = compute_projection_noise(n_effective, t_meas, gamma_e)
    sq = compute_spin_squeezing(n_effective, xi2_r)

    # Sensitivity floors
    xi_r = math.sqrt(xi2_r)  # amplitude squeezing factor
    squeezed_b = xi_r * pn.sql_field_t_per_sqrthz

    oat_xi2 = compute_oat_optimal_squeezing(n_effective)
    oat_b = math.sqrt(oat_xi2) * pn.sql_field_t_per_sqrthz

    return QuantumEnhancementResult(
        projection_noise=pn,
        squeezing=sq,
        squeezed_field_sensitivity_t_per_sqrthz=squeezed_b,
        oat_optimal_xi2_r=oat_xi2,
        oat_best_sensitivity_t_per_sqrthz=oat_b,
        heisenberg_sensitivity_t_per_sqrthz=pn.hl_field_t_per_sqrthz,
    )
