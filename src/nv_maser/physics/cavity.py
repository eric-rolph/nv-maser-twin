"""
Microwave cavity properties and maser threshold from cavity QED.

Computes the single-spin vacuum Rabi coupling, Purcell enhancement factor,
and ensemble cooperativity that determines whether the NV diamond maser
reaches oscillation threshold.

Key quantities
──────────────
Zero-point field:  B_zpf = √(μ₀ ℏω / (2 V_mode))
Coupling:          g₀ = γe · B_zpf          (single spin, rad/s)
Cavity decay:      κ  = ω / Q               (rad/s)
Spin dephasing:    γ⊥ = 2π · Γ_eff          (rad/s)
Ensemble coupling: g_N = g₀ · √(N_eff)
Cooperativity:     C   = 4 g_N² / (κ · γ⊥)
Purcell factor:    F_P = (3 Q λ³) / (4π² V_mode)
Threshold:         C > 1

References
──────────
Breeze et al., Nature 555, 493 (2018).
Jin et al., Nat. Commun. 6, 8251 (2015).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import CavityConfig, MaserConfig, NVConfig


# ── Physical constants ────────────────────────────────────────────
_HBAR = 1.054571817e-34  # J·s
_MU0 = 1.2566370614e-6  # H/m  (vacuum permeability)
_C = 2.99792458e8  # m/s


@dataclass(frozen=True)
class CavityProperties:
    """Derived cavity-mode quantities."""

    mode_volume_m3: float
    zpf_field_tesla: float  # B_zpf
    single_spin_coupling_hz: float  # g₀ / (2π)
    cavity_linewidth_hz: float  # κ / (2π)
    purcell_factor: float


@dataclass(frozen=True)
class ThresholdResult:
    """Ensemble cooperativity and maser threshold status."""

    n_effective: float
    ensemble_coupling_hz: float  # g_N / (2π)
    cooperativity: float
    threshold_margin: float  # C − 1 (positive ⇒ above threshold)
    masing: bool


def compute_cavity_properties(
    maser_config: MaserConfig,
    cavity_config: CavityConfig,
) -> CavityProperties:
    """
    Derive cavity-mode properties from geometry and Q factor.

    Args:
        maser_config: Cavity Q and resonance frequency.
        cavity_config: Mode volume.

    Returns:
        CavityProperties with B_zpf, g₀, κ, and Purcell factor.
    """
    omega = 2 * math.pi * maser_config.cavity_frequency_ghz * 1e9  # rad/s
    v_mode = cavity_config.mode_volume_cm3 * 1e-6  # m³

    # Zero-point magnetic fluctuation of the cavity mode
    b_zpf = math.sqrt(_MU0 * _HBAR * omega / (2 * v_mode))

    # Single-spin vacuum Rabi coupling (in Hz, i.e. g₀/(2π))
    gamma_e_hz = 28.025e9  # Hz/T (= γe / 2π)
    g0_hz = gamma_e_hz * b_zpf

    # Cavity linewidth κ/(2π)
    kappa_hz = maser_config.cavity_frequency_ghz * 1e9 / maser_config.cavity_q

    # Purcell factor  F_P = 3 Q λ³ / (4π² V_mode)
    lambda_m = _C / (maser_config.cavity_frequency_ghz * 1e9)
    f_purcell = (
        3.0 * maser_config.cavity_q * lambda_m**3 / (4 * math.pi**2 * v_mode)
    )

    return CavityProperties(
        mode_volume_m3=v_mode,
        zpf_field_tesla=b_zpf,
        single_spin_coupling_hz=g0_hz,
        cavity_linewidth_hz=kappa_hz,
        purcell_factor=f_purcell,
    )


def compute_maser_threshold(
    cavity_props: CavityProperties,
    n_effective: float,
    spin_linewidth_hz: float,
) -> ThresholdResult:
    """
    Evaluate maser threshold via ensemble cooperativity.

    C = 4 g_N² / (κ · γ⊥)

    where g_N = g₀ √N_eff and all quantities are angular frequencies,
    but since the 2π factors cancel (both numerator and denominator
    scale the same way) we can use ordinary-frequency (Hz) values.

    Args:
        cavity_props: Pre-computed cavity quantities.
        n_effective: Number of effectively inverted NV spins.
        spin_linewidth_hz: Total spin linewidth Γ_eff (Hz).

    Returns:
        ThresholdResult with cooperativity and margin.
    """
    if n_effective <= 0:
        return ThresholdResult(
            n_effective=0.0,
            ensemble_coupling_hz=0.0,
            cooperativity=0.0,
            threshold_margin=-1.0,
            masing=False,
        )

    g0 = cavity_props.single_spin_coupling_hz
    g_ens = g0 * math.sqrt(n_effective)

    kappa = cavity_props.cavity_linewidth_hz
    gamma_perp = spin_linewidth_hz

    denom = kappa * gamma_perp
    if denom <= 0:
        cooperativity = float("inf") if g_ens > 0 else 0.0
    else:
        cooperativity = 4.0 * g_ens**2 / denom

    return ThresholdResult(
        n_effective=n_effective,
        ensemble_coupling_hz=g_ens,
        cooperativity=cooperativity,
        threshold_margin=cooperativity - 1.0,
        masing=cooperativity > 1.0,
    )


def compute_n_effective(
    nv_config: NVConfig,
    cavity_config: CavityConfig,
    gain_budget: float,
) -> float:
    """
    Number of NV spins that effectively contribute to masing.

    N_eff = n_NV × V_mode × η_fill × η_pump × gain_budget

    The fill factor (V_diamond / V_mode) selects the NV centers
    inside the mode volume; pump efficiency gives the inverted
    fraction; gain_budget de-rates for inhomogeneous broadening.

    Args:
        nv_config: NV density, pump efficiency.
        cavity_config: Fill factor.
        gain_budget: Spectral overlap fraction (0–1).

    Returns:
        Effective number of inverted spins in the cavity mode.
    """
    v_mode_m3 = cavity_config.mode_volume_cm3 * 1e-6
    v_diamond_m3 = v_mode_m3 * cavity_config.fill_factor
    n_nv = nv_config.nv_density_per_cm3 * 1e6  # convert /cm³ → /m³
    return n_nv * v_diamond_m3 * nv_config.pump_efficiency * gain_budget


def compute_full_threshold(
    nv_config: NVConfig,
    maser_config: MaserConfig,
    cavity_config: CavityConfig,
    gain_budget: float,
    spin_linewidth_hz: float,
) -> ThresholdResult:
    """
    One-shot maser threshold evaluation.

    Convenience wrapper combining cavity properties, N_eff, and
    threshold calculation.

    Args:
        nv_config: NV center parameters.
        maser_config: Cavity Q and frequency.
        cavity_config: Mode volume and fill factor.
        gain_budget: Current spectral overlap fraction (0–1).
        spin_linewidth_hz: Total spin linewidth Γ_eff (Hz).

    Returns:
        ThresholdResult with cooperativity and masing status.
    """
    props = compute_cavity_properties(maser_config, cavity_config)
    n_eff = compute_n_effective(nv_config, cavity_config, gain_budget)
    return compute_maser_threshold(props, n_eff, spin_linewidth_hz)
