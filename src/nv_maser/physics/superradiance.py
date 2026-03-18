"""
Superradiant dynamics model for the NV diamond maser.

The NV maser can operate in one of three distinct regimes determined by
the collective spin-cavity coupling g_eff = g₀ √N_eff relative to the
cavity half-linewidth κ/2 and the spin dephasing time T₂*:

  Below threshold  (C ≤ 1):          no masing, fluorescence only
  CW masing        (g_eff ≤ κ/2):    bad-cavity limit, steady CW output
  Superradiant     (g_eff > κ/2,
                    C > 1,
                    τ_SR < T₂*):     cooperative burst, P_peak ∝ N²_eff

In the superradiant regime all inverted spins emit cooperatively in a
single pulse (Dicke 1954).  Key observables differ qualitatively from
CW masing:

  Peak power         P_SR = ℏω × κ_rad × N_eff / 4   (scales as N)
  Pulse duration     τ_SR = 1 / (2π × g_eff)          (shortens with N)
  Superradiant delay t_D  ≈ τ_SR × ln(N_eff) / 2      (statistical seed)
  Pulse energy       E_SR = N_eff × ℏω                 (all angular momentum)

The module provides an **analytic** regime classifier and pulse estimator.
No ODE integration is performed here; for time-domain simulation of
spectral hole-burning and burst trains see ``spectral_maxwell_bloch.py``.

Convention
──────────
All couplings and linewidths are expressed in ordinary frequency (Hz),
i.e. as ω/(2π).  Angular-frequency factors (2π) appear only at the
final output-power expression.

References
──────────
Dicke, R. H. (1954) Phys. Rev. 93, 99.
Kersten et al., Nature Physics (2026), PMC12811124.
Breeze et al., Nature 555, 493 (2018).
Tavis & Cummings (1968) Phys. Rev. 170, 379.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import MaserConfig, NVConfig
from .cavity import CavityProperties, ThresholdResult, compute_cavity_properties


# ── Physical constants ────────────────────────────────────────────
_HBAR = 1.054571817e-34  # J·s

# ── Regime string constants ───────────────────────────────────────
BELOW_THRESHOLD: str = "below_threshold"
MASING: str = "masing"
SUPERRADIANT: str = "superradiant"


# ── Result dataclass ──────────────────────────────────────────────


@dataclass(frozen=True)
class SuperradianceResult:
    """Analytic superradiant dynamics estimate for the NV maser.

    All frequency quantities are in ordinary Hz (i.e. ω/(2π)).
    Time quantities are in seconds; power in watts; energy in joules.

    Attributes:
        regime: Operating regime string — one of ``BELOW_THRESHOLD``,
            ``MASING``, or ``SUPERRADIANT``.
        is_superradiant: ``True`` iff ``regime == SUPERRADIANT``.
        n_effective: Effective number of cooperatively coupled spins N_eff.
        single_spin_coupling_hz: Single-spin vacuum Rabi coupling g₀/(2π) [Hz].
        collective_coupling_hz: Ensemble coupling g_eff = g₀√N_eff [Hz].
        cavity_linewidth_hz: Cavity half-power linewidth κ/(2π) [Hz].
        superradiant_threshold_coupling_hz: Strong-coupling boundary κ/2 [Hz].
            g_eff must exceed this for the superradiant regime.
        cooperativity: Cavity QED cooperativity C = 4g_eff²/(κ·γ⊥).
        pulse_duration_s: Analytic SR pulse half-duration τ_SR = 1/(2π g_eff) [s].
            Computed even in the masing regime as a lower bound.
        delay_s: Statistical superradiant delay t_D ≈ τ_SR ln(N)/2 [s].
        peak_photons: Peak intracavity photon number N̄_SR = N_eff/4.
        peak_power_w: Peak output power P_SR = ℏω × κ_rad × N̄_SR [W].
        pulse_energy_j: Total SR pulse energy E_SR = N_eff × ℏω [J].
        cw_power_w: CW masing output power from Maxwell-Bloch result, or
            ``math.nan`` if no MB result was supplied.
        power_enhancement: Ratio P_SR_peak / P_CW, or ``math.nan`` if
            CW power is unavailable.
        cavity_frequency_hz: Cavity resonance frequency [Hz].
    """

    regime: str
    is_superradiant: bool

    # Coupling parameters
    n_effective: float
    single_spin_coupling_hz: float
    collective_coupling_hz: float
    cavity_linewidth_hz: float
    superradiant_threshold_coupling_hz: float
    cooperativity: float

    # SR pulse estimates (computed for all regimes; NaN only if g_eff == 0)
    pulse_duration_s: float
    delay_s: float
    peak_photons: float
    peak_power_w: float
    pulse_energy_j: float

    # Comparison with CW masing
    cw_power_w: float
    power_enhancement: float

    # Context
    cavity_frequency_hz: float


# ── Low-level building blocks ─────────────────────────────────────


def compute_collective_coupling(
    single_spin_coupling_hz: float,
    n_collective: float,
) -> float:
    """Ensemble collective coupling g_eff = g₀ × √N_eff [Hz].

    The collective coupling sets the rate at which N_eff coherently
    inverted spins exchange energy with a single cavity mode.

    Args:
        single_spin_coupling_hz: Single-spin vacuum Rabi coupling g₀/(2π) [Hz].
        n_collective: Effective number of cooperatively coupled spins N_eff.
            Values ≤ 0 return 0.

    Returns:
        Collective coupling g_eff [Hz].  Zero if n_collective ≤ 0.
    """
    if n_collective <= 0.0:
        return 0.0
    return single_spin_coupling_hz * math.sqrt(n_collective)


def determine_regime(
    collective_coupling_hz: float,
    cavity_linewidth_hz: float,
    cooperativity: float,
    t2_star_s: float,
) -> str:
    """Classify the maser operating regime.

    Three conditions must all hold for the superradiant regime:

    1. Above oscillation threshold:  C > 1
    2. Strong-coupling (good-cavity): g_eff > κ/2
    3. Coherent emission wins dephasing: τ_SR < T₂*
       where τ_SR = 1/(2π × g_eff)

    If condition 1 fails → ``BELOW_THRESHOLD``.
    If condition 1 holds but 2 or 3 fail → ``MASING``.
    If all three hold → ``SUPERRADIANT``.

    Args:
        collective_coupling_hz: g_eff = g₀√N_eff [Hz].
        cavity_linewidth_hz: Cavity linewidth κ/(2π) [Hz].
        cooperativity: C = 4g_eff²/(κ·γ⊥), dimensionless.
        t2_star_s: Ensemble dephasing time T₂* [s].

    Returns:
        One of ``BELOW_THRESHOLD``, ``MASING``, ``SUPERRADIANT``.
    """
    if cooperativity <= 1.0:
        return BELOW_THRESHOLD

    # Strong-coupling criterion: collective emission rate > cavity decay rate
    strong_coupling = collective_coupling_hz > cavity_linewidth_hz / 2.0

    if not strong_coupling:
        return MASING

    # Coherence criterion: superradiant burst completes before T₂* dephasing
    if collective_coupling_hz > 0.0:
        tau_sr = 1.0 / (2.0 * math.pi * collective_coupling_hz)
        coherent = tau_sr < t2_star_s
    else:
        coherent = False

    return SUPERRADIANT if coherent else MASING


def compute_superradiant_pulse_duration(collective_coupling_hz: float) -> float:
    """Superradiant Rabi half-period τ_SR = 1 / (2π × g_eff) [seconds].

    This is the characteristic time for collective inversion; the actual
    burst FWHM is typically ∼ τ_SR/π, but the Tavis-Cummings peak occurs
    at t = τ_SR.

    Args:
        collective_coupling_hz: g_eff [Hz].  Must be > 0.

    Returns:
        Pulse half-duration τ_SR [s].

    Raises:
        ValueError: If collective_coupling_hz ≤ 0.
    """
    if collective_coupling_hz <= 0.0:
        raise ValueError(
            f"collective_coupling_hz must be > 0, got {collective_coupling_hz!r}"
        )
    return 1.0 / (2.0 * math.pi * collective_coupling_hz)


def compute_superradiant_delay(
    collective_coupling_hz: float,
    n_collective: float,
) -> float:
    """Statistical superradiant delay t_D ≈ τ_SR × ln(N_eff) / 2 [seconds].

    The delay from t = 0 (all spins inverted) to the burst maximum
    arises from quantum fluctuations seeding the coherence.  For a
    Fock-state initial condition the mean delay is ∼ τ_SR × ln(N)/2.

    Args:
        collective_coupling_hz: g_eff [Hz].  Zero or negative → 0.
        n_collective: N_eff.  Values < 1 are clamped to 1 before log.

    Returns:
        Delay t_D [s].  Zero if collective_coupling_hz or n_collective ≤ 0.
    """
    if collective_coupling_hz <= 0.0 or n_collective <= 0.0:
        return 0.0
    tau_sr = compute_superradiant_pulse_duration(collective_coupling_hz)
    n_safe = max(n_collective, 1.0)
    return tau_sr * math.log(n_safe) / 2.0


def compute_superradiant_peak(
    collective_coupling_hz: float,
    n_collective: float,
    cavity_linewidth_hz: float,
    cavity_frequency_hz: float,
) -> dict[str, float]:
    """Peak SR burst observables from the Tavis-Cummings semiclassical model.

    At the moment of maximum photon number (t = τ_SR), the intracavity
    occupation is N̄_SR = N_eff / 4 and the total energy radiated over
    the full burst equals N_eff × ℏω (all angular momentum converted).

    Args:
        collective_coupling_hz: g_eff [Hz] (used only for documentation;
            physics here depends on n_collective directly).
        n_collective: N_eff (effective coupled spins).
        cavity_linewidth_hz: κ/(2π) [Hz] — sets photon decay rate.
        cavity_frequency_hz: Cavity resonance f [Hz] — sets ℏω.

    Returns:
        dict with keys:

        - ``peak_photons``  : N̄_SR = N_eff / 4
        - ``peak_power_w``  : P_SR = ℏω × (2π κ) × N̄_SR  [W]
        - ``pulse_energy_j``: E_SR = N_eff × ℏω  [J]
    """
    peak_photons = n_collective / 4.0

    omega_rad = 2.0 * math.pi * cavity_frequency_hz   # rad/s
    kappa_rad = 2.0 * math.pi * cavity_linewidth_hz   # rad/s

    peak_power_w = _HBAR * omega_rad * kappa_rad * peak_photons
    pulse_energy_j = n_collective * _HBAR * omega_rad

    return {
        "peak_photons": peak_photons,
        "peak_power_w": peak_power_w,
        "pulse_energy_j": pulse_energy_j,
    }


# ── Top-level entry point ─────────────────────────────────────────


def compute_superradiance(
    cavity_props: CavityProperties,
    threshold_result: ThresholdResult,
    nv_config: NVConfig,
    maser_config: MaserConfig,
    mb_result=None,
) -> SuperradianceResult:
    """Classify the maser operating regime and estimate SR pulse parameters.

    Combines ``determine_regime`` with full analytic pulse calculations to
    produce a single comprehensive result.  No ODE integration is performed;
    for time-domain burst-train simulation use ``solve_spectral_maxwell_bloch``.

    Args:
        cavity_props: Pre-computed cavity quantities from
            ``compute_cavity_properties``.
        threshold_result: Output of ``compute_full_threshold`` or
            ``compute_maser_threshold``.  Supplies g_eff and N_eff.
        nv_config: NV parameters including ``t2_star_us`` for the
            dephasing coherence check.
        maser_config: Cavity parameters; ``cavity_frequency_ghz`` sets ℏω.
        mb_result: Optional Maxwell-Bloch result (``MaxwellBlochResult``
            or ``SpectralMBResult``).  If supplied, ``output_power_w``
            is used as the CW comparison and ``power_enhancement`` is
            computed.  If ``None``, both fields are ``math.nan``.

    Returns:
        :class:`SuperradianceResult` with regime classification and all
        analytic SR observables.
    """
    g0 = cavity_props.single_spin_coupling_hz
    kappa = cavity_props.cavity_linewidth_hz
    n_eff = threshold_result.n_effective
    g_eff = threshold_result.ensemble_coupling_hz
    cooperativity = threshold_result.cooperativity

    f_hz = maser_config.cavity_frequency_ghz * 1e9
    t2_star_s = nv_config.t2_star_us * 1e-6

    regime = determine_regime(g_eff, kappa, cooperativity, t2_star_s)
    is_sr = regime == SUPERRADIANT

    # Analytic SR pulse estimates (computed for all regimes for comparison)
    if g_eff > 0.0:
        tau_sr = compute_superradiant_pulse_duration(g_eff)
        t_delay = compute_superradiant_delay(g_eff, n_eff)
        peak = compute_superradiant_peak(g_eff, n_eff, kappa, f_hz)
    else:
        tau_sr = math.nan
        t_delay = math.nan
        peak = {"peak_photons": 0.0, "peak_power_w": 0.0, "pulse_energy_j": 0.0}

    # CW power comparison
    if mb_result is not None:
        cw_power = float(mb_result.output_power_w)
    else:
        cw_power = math.nan

    if not math.isnan(cw_power) and cw_power > 0.0:
        power_enh = peak["peak_power_w"] / cw_power
    else:
        power_enh = math.nan

    return SuperradianceResult(
        regime=regime,
        is_superradiant=is_sr,
        n_effective=n_eff,
        single_spin_coupling_hz=g0,
        collective_coupling_hz=g_eff,
        cavity_linewidth_hz=kappa,
        superradiant_threshold_coupling_hz=kappa / 2.0,
        cooperativity=cooperativity,
        pulse_duration_s=tau_sr,
        delay_s=t_delay,
        peak_photons=peak["peak_photons"],
        peak_power_w=peak["peak_power_w"],
        pulse_energy_j=peak["pulse_energy_j"],
        cw_power_w=cw_power,
        power_enhancement=power_enh,
        cavity_frequency_hz=f_hz,
    )
