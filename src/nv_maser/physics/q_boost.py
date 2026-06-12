"""
Electronic Q-boosting: active cavity-loss compensation.

Models the Wang et al. (2024) feedback loop that electronically cancels part of
the total cavity decay rate, boosting the effective loaded Q factor (Q_L_eff)
and reducing the oscillation threshold.

Physical picture
────────────────
In the passive cavity the total field decay rate is

    κ_total = κ_i + κ_e   (rad/s)    [κ_i = internal loss, κ_e = coupling loss]

The electronic controller measures the output field and feeds back a signal
that **subtracts** a fraction of the decay:

    κ_eff = κ_total × (1 − g_fb)   where 0 ≤ g_fb < 1

The *effective* loaded Q is therefore:

    Q_L_eff = ω_c / κ_eff = Q_L / (1 − g_fb) = B × Q_L   (B ≥ 1)

Because Q₀ / Q_L = const at fixed coupling β, the internal Q boosts by the
same factor:

    Q₀_eff = B × Q₀

Threshold implications
──────────────────────
Masing condition (spin gain ≥ cavity loss):

    1/Q_m ≥ 1/Q_L_eff   ↔   Q_m ≤ Q_L_eff = B × Q_L

A boost factor B = 59 (Wang 2024: Q_L = 1.1×10⁴ → 6.5×10⁵) reduces the
minimum spin gain needed for oscillation by the same factor.

Noise temperature with boost
────────────────────────────
Substituting Q₀_eff into Wang Eq. 4:

    T_a = Q_m / (Q₀_eff − Q_m) T_bath + Q₀_eff / (Q₀_eff − Q_m) T_s

As B → ∞ the noise temperature approaches the spin temperature T_s — the
ideal quantum-limited maser limit.

Reference
─────────
Wang, J. et al. Tailoring coherent microwave emission from spin-gain-enhanced
resonator for room-temperature maser. *Advanced Science* 11 (2024).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ── Physical constants ────────────────────────────────────────────────────
from .constants import HBAR as _HBAR
from .constants import KB as _KB

# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class QBoostResult:
    """Effective cavity parameters after electronic Q-boosting.

    Validated against Wang et al. (2024):
        boost_factor ≈ 59, Q_L_native = 1.1×10⁴ → Q_L_effective = 6.5×10⁵.
    """

    # ---------- Inputs ------------------------------------------------
    q_l_native: float
    """Passive loaded quality factor Q_L (without feedback)."""

    q0_native: float
    """Passive unloaded quality factor Q₀ = Q_L × (1 + β) (without feedback)."""

    boost_factor: float
    """B = Q_L_eff / Q_L_native ≥ 1.  B = 1 means no boost."""

    # ---------- Derived -----------------------------------------------
    gain_feedback: float
    """Feedback gain g_fb = 1 − 1/B.

    Represents the fraction of total cavity decay cancelled by the controller.
    g_fb = 0: passive (B = 1).
    g_fb → 1: near-complete cancellation (B → ∞).
    """

    q_l_effective: float
    """Effective loaded Q_L_eff = B × Q_L_native."""

    q0_effective: float
    """Effective unloaded Q₀_eff = B × Q₀_native."""

    # ---------- Operational context -----------------------------------
    threshold_qm: float
    """Minimum spin-gain quality factor that triggers oscillation with boost.

    Oscillation condition: Q_m ≤ Q_L_eff.
    Below this value the system oscillates; above it is the amplifier regime.
    """

    threshold_reduction_factor: float
    """Factor by which Q-boost reduces the threshold spin gain requirement.

    threshold_reduction_factor = B.  A higher spin gain is needed in the
    PASSIVE cavity; boosting by B reduces that requirement by the same factor.
    """


# ── Core functions ────────────────────────────────────────────────────────

def compute_q_boost(
    q_l_native: float,
    coupling_beta: float,
    boost_factor: float,
) -> QBoostResult:
    r"""Compute effective cavity parameters after electronic Q-boosting.

    The feedback controller reduces the net cavity decay rate by a factor B,
    yielding:

    .. math::

        Q_{L,\text{eff}} = B \cdot Q_L,\quad
        Q_{0,\text{eff}} = B \cdot Q_0 = B \cdot Q_L (1+\beta)

    Args:
        q_l_native:    Passive loaded Q factor Q_L.
        coupling_beta: Cavity coupling coefficient β = Q₀/Q_e.
        boost_factor:  Multiplicative Q-boost factor B ≥ 1.
                       B = 59 is the Wang 2024 experimental value.

    Returns:
        :class:`QBoostResult` with boosted Q values and threshold context.

    Raises:
        ValueError: If boost_factor < 1 or q_l_native ≤ 0 or coupling_beta ≤ 0.
    """
    if q_l_native <= 0:
        raise ValueError(f"q_l_native must be positive, got {q_l_native}")
    if coupling_beta <= 0:
        raise ValueError(f"coupling_beta must be positive, got {coupling_beta}")
    if boost_factor < 1.0:
        raise ValueError(f"boost_factor must be ≥ 1, got {boost_factor}")

    q0_native = q_l_native * (1.0 + coupling_beta)
    g_fb = 1.0 - 1.0 / boost_factor
    q_l_eff = boost_factor * q_l_native
    q0_eff = boost_factor * q0_native

    return QBoostResult(
        q_l_native=q_l_native,
        q0_native=q0_native,
        boost_factor=boost_factor,
        gain_feedback=g_fb,
        q_l_effective=q_l_eff,
        q0_effective=q0_eff,
        threshold_qm=q_l_eff,
        threshold_reduction_factor=boost_factor,
    )


def compute_minimum_boost(q_m: float, q_l: float) -> float:
    r"""Minimum boost factor required to reach oscillation threshold.

    The oscillation condition Q_m ≤ B × Q_L gives:

    .. math::

        B_\min = Q_m / Q_L

    If Q_m ≤ Q_L the system already oscillates without boost (returns 1.0).

    Args:
        q_m: Magnetic quality factor (spin gain).
        q_l: Passive loaded cavity Q_L.

    Returns:
        Minimum boost factor B_min ≥ 1.0.
    """
    if q_l <= 0:
        return float("inf")
    raw = q_m / q_l
    return max(1.0, raw)


def compute_noise_temperature_boosted(
    magnetic_q: float,
    q_boost_result: QBoostResult,
    spin_temperature_k: float,
    bath_temperature_k: float = 300.0,
) -> float:
    r"""Noise temperature of the maser amplifier with Q-boost applied.

    Substitutes Q₀_eff into Wang et al. (2024), Eq. 4:

    .. math::

        T_a = \frac{Q_m}{Q_{0,\text{eff}} - Q_m} T_\text{bath}
            + \frac{Q_{0,\text{eff}}}{Q_{0,\text{eff}} - Q_m} T_s

    As B → ∞ (Q₀_eff → ∞): T_a → T_s (spin-temperature-limited amplifier).

    Args:
        magnetic_q:       Q_m (spin gain quality factor).
        q_boost_result:   Boost parameters from :func:`compute_q_boost`.
        spin_temperature_k: T_s from the NV spin inversion (K).
        bath_temperature_k: Physical cavity temperature (K).

    Returns:
        Noise temperature T_a in Kelvin.
        Returns float('nan') if in/above oscillation threshold (Q_m ≥ Q₀_eff).
    """
    if math.isnan(spin_temperature_k):
        return float("nan")

    q0_eff = q_boost_result.q0_effective
    denominator = q0_eff - magnetic_q
    if denominator <= 0:
        return float("nan")

    return (
        (magnetic_q / denominator) * bath_temperature_k
        + (q0_eff / denominator) * spin_temperature_k
    )


def compute_sql_limit_ratio(
    noise_temperature_k: float,
    cavity_frequency_ghz: float,
) -> float:
    r"""Ratio T_a / T_SQL — how close the amplifier is to the quantum limit.

    .. math::

        T_\text{SQL} = \frac{\hbar \omega_c}{2 k_B}

    A ratio ≤ 1 indicates quantum-limited (sub-SQL) performance.

    Args:
        noise_temperature_k:  Effective noise temperature (K).
        cavity_frequency_ghz: Cavity frequency (GHz).

    Returns:
        T_a / T_SQL (dimensionless, ≥ 0).  Returns float('inf') if T_SQL = 0.
    """
    omega_c = 2.0 * math.pi * cavity_frequency_ghz * 1.0e9
    t_sql = _HBAR * omega_c / (2.0 * _KB)
    if t_sql <= 0:
        return float("inf")
    return noise_temperature_k / t_sql
