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


# ── Q-boosting (Wang 2024) ────────────────────────────────────────


def compute_effective_q(maser_config: MaserConfig) -> float:
    """
    Effective cavity Q with electronic feedback Q-boosting.

    Q_eff = Q_0 / (1 - G_loop)

    where G_loop is the electronic feedback loop gain.
    G_loop = 0 → no boost (Q_eff = Q_0).
    G_loop → 1 → Q_eff → ∞ (oscillation onset).

    Wang et al. (2024) boosted Q_L from 1.1×10⁴ to 6.5×10⁵
    using active dissipation compensation.

    Args:
        maser_config: Maser parameters including cavity_q and q_boost_gain.

    Returns:
        Effective quality factor after electronic Q-boosting.
    """
    g = maser_config.q_boost_gain
    return maser_config.cavity_q / (1.0 - g)


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
    return (
        n_nv
        * v_diamond_m3
        * nv_config.pump_efficiency
        * nv_config.orientation_fraction
        * gain_budget
    )


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


# ── Magnetic quality factor (Wang 2024, Eq. 1) ──────────────────


@dataclass(frozen=True)
class MagneticQResult:
    """Magnetic quality factor of the inverted spin ensemble."""

    q_magnetic: float  # Q_m  (dimensionless, negative → gain)
    inverted_density_m3: float  # Δn  (inverted spins / m³)
    fill_factor: float
    t2_star_s: float


def compute_magnetic_q(
    nv_config: NVConfig,
    cavity_config: CavityConfig,
) -> MagneticQResult:
    """
    Magnetic quality factor Q_m of the active spin medium.

    Q_m⁻¹ = μ₀ γₑ² Δn η T₂* / 2

    where Δn is the inverted spin density (per m³), η is the
    fill factor, and T₂* is the ensemble dephasing time.
    A negative Q_m (inverted medium) means net gain.

    Source: Wang et al. 2024, Nature Electronics 7, 780 — Eq. 1.

    Args:
        nv_config: NV centre parameters (density, pump, orientation).
        cavity_config: Fill factor.

    Returns:
        MagneticQResult with Q_m and intermediate quantities.
    """
    gamma_e_rad = 2 * math.pi * nv_config.gamma_e_ghz_per_t * 1e9  # rad s⁻¹ T⁻¹
    n_nv_m3 = nv_config.nv_density_per_cm3 * 1e6
    delta_n = (
        n_nv_m3
        * nv_config.pump_efficiency
        * nv_config.orientation_fraction
    )
    eta = cavity_config.fill_factor
    t2_s = nv_config.t2_star_us * 1e-6

    q_inv = _MU0 * gamma_e_rad**2 * delta_n * eta * t2_s / 2.0
    q_m = 1.0 / q_inv if q_inv > 0 else float("inf")

    return MagneticQResult(
        q_magnetic=q_m,
        inverted_density_m3=delta_n,
        fill_factor=eta,
        t2_star_s=t2_s,
    )


# ── Spectral overlap ratio (Wang 2024) ──────────────────────────


def compute_spectral_overlap(
    cavity_props: CavityProperties,
    spin_linewidth_hz: float,
) -> float:
    """
    Spectral overlap ratio R = κ / γ⊥.

    R < 1 ("good-cavity" regime): cavity linewidth narrower than spin
    line — only a fraction of inverted spins couple efficiently.
    R > 1 ("bad-cavity" regime): cavity wider than spin line — all
    spins contribute but cavity losses are high.

    Args:
        cavity_props: Pre-computed cavity quantities (includes κ/(2π)).
        spin_linewidth_hz: Total spin linewidth Γ_eff (Hz).

    Returns:
        Dimensionless overlap ratio R.
    """
    if spin_linewidth_hz <= 0:
        return float("inf")
    return cavity_props.cavity_linewidth_hz / spin_linewidth_hz
