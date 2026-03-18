"""Magnetic field sensitivity model for the NV diamond maser magnetometer.

Three complementary sensitivity floors are computed and compared:

1. **Schawlow-Townes (quantum phase diffusion)**
   The maser output phase undergoes a random walk seeded by spontaneous
   emission.  Over integration time τ, the accumulated phase uncertainty
   Δφ = √(2π Δν_ST τ) gives a frequency uncertainty

       δf_ST = √(Δν_ST / (2π τ))   [Hz]

   which translates to a field sensitivity

       η_ST = √(Δν_ST / (2π)) / γ_e   [T/√Hz]

   where γ_e is the NV gyromagnetic ratio (Hz/T).  This is the
   fundamental quantum-noise-limited sensitivity, arising from
   vacuum fluctuations and stimulated emission.

2. **Thermal-SNR (classical LNA chain)**
   The maser output signal, after cavity coupling and insertion loss,
   is detected via a mixer/frequency discriminator.  With system noise
   temperature T_sys, the residual noise on the carrier sets a frequency
   uncertainty via the Lorentzian discriminator slope at resonance:

       η_T = (κ_c / γ_e) × √(k_B T_sys / P_out)   [T/√Hz]

   where κ_c = cavity_linewidth_hz is the half-bandwidth (FWHM/2π) and
   P_out is the coupled maser output power.

3. **Friis-enhanced (quantum maser pre-amplifier)**
   Same formula as thermal-SNR but with T_sys replaced by the Friis
   cascade temperature T_friis ≈ T_maser ≈ 0.11 K.  This is only
   computed when a MaserNoiseResult is available; otherwise ``nan``.

       η_F = (κ_c / γ_e) × √(k_B T_friis / P_out)   [T/√Hz]

Comparison context
──────────────────
- Room-temperature SQUID magnetometers: ~1 fT/√Hz (liquid helium)
- Optically pumped atomic magnetometers: ~10 fT/√Hz
- NV ensemble sensors (DC): ~1 pT/√Hz (best reported ~100 fT/√Hz)
- This maser (ST limit): ~1–2 nT/√Hz for Δν_ST ≈ 1 kHz, γ_e = 28 GHz/T
  With Friis pre-amp advantage (~35 dB), thermal floor → ~35 pT/√Hz

References
──────────
Breeze et al., "Enhanced magnetic Purcell effect in room-temperature diamond",
npj Quantum Information 7, 45 (2021).
Angerer et al., "Superradiant emission from colour centres in diamond",
Nature Physics 14, 1168 (2018).
Schawlow & Townes, "Infrared and Optical Masers", Phys. Rev. 112, 1940 (1958).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import NVConfig
from .quantum_noise import MaserNoiseResult
from .signal_chain import SignalChainBudget

# ── Physical constants ─────────────────────────────────────────────────────
_KB = 1.380649e-23  # J/K
_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class SensitivityResult:
    """Complete field sensitivity analysis for the NV maser magnetometer.

    All sensitivity figures are one-sided spectral densities in T/√Hz.
    Convenience copies in nT/√Hz and pT/√Hz are included for comparison
    with atomic magnetometers and SQUID specifications.
    """

    # ── Schawlow-Townes quantum limit ──────────────────────────────────────
    schawlow_townes_t_per_sqrthz: float
    """η_ST = √(Δν_ST / (2π)) / γ_e  [T/√Hz].
    Fundamental quantum limit set by spontaneous emission phase diffusion."""

    schawlow_townes_nt_per_sqrthz: float
    """η_ST in nT/√Hz (×10⁹)."""

    schawlow_townes_pt_per_sqrthz: float
    """η_ST in pT/√Hz (×10¹²)."""

    # ── Thermal-SNR classical limit ────────────────────────────────────────
    thermal_snr_t_per_sqrthz: float
    """η_T = (κ_c / γ_e) × √(k_B T_sys / P_out)  [T/√Hz].
    Set by system noise temperature of the classical LNA chain."""

    thermal_snr_nt_per_sqrthz: float
    """η_T in nT/√Hz."""

    thermal_snr_pt_per_sqrthz: float
    """η_T in pT/√Hz."""

    # ── Friis-enhanced quantum limit ───────────────────────────────────────
    friis_t_per_sqrthz: float
    """η_F = (κ_c / γ_e) × √(k_B T_friis / P_out)  [T/√Hz].
    Thermal sensitivity with quantum Friis cascade replacing LNA chain.
    math.nan when SignalChainBudget.friis_system_temperature_k is not supplied."""

    friis_nt_per_sqrthz: float
    """η_F in nT/√Hz.  math.nan when Friis T_sys not available."""

    friis_pt_per_sqrthz: float
    """η_F in pT/√Hz.  math.nan when Friis T_sys not available."""

    # ── Cross-floor comparisons ────────────────────────────────────────────
    thermal_vs_st_ratio: float
    """η_T / η_ST — how many times worse the thermal floor is vs the ST limit.
    Values >> 1 mean the system is thermal-noise-limited, not quantum-limited."""

    friis_vs_st_ratio: float
    """η_F / η_ST — ratio of Friis floor to quantum ST limit.
    math.nan when Friis data not available."""

    friis_advantage_over_thermal_db: float
    """20 × log₁₀(η_T / η_F) [dB].
    Field-amplitude improvement from switching LNA chain → Friis cascade.
    math.nan when Friis data not available."""

    # ── Stored parameters for provenance ──────────────────────────────────
    schawlow_townes_linewidth_hz: float
    """Δν_ST used in the ST calculation (Hz)."""

    cavity_linewidth_hz: float
    """κ_c = cavity half-bandwidth (FWHM/(2π)) used in thermal formulas (Hz)."""

    system_noise_temperature_k: float
    """Classical T_sys from the signal chain budget (K)."""

    friis_system_temperature_k: float
    """Friis cascade T_sys (K).  math.nan when not supplied."""

    output_power_w: float
    """Maser coupled output power (W) used in thermal sensitivity estimates."""

    gamma_e_hz_per_t: float
    """NV gyromagnetic ratio γ_e (Hz/T) = gamma_e_ghz_per_t × 10⁹."""

    # ── Allan deviation (1 s integration) ─────────────────────────────────
    allan_deviation_1s_t: float
    """σ_B(τ=1s) = η_ST / √(2π)  [T].
    Stability of the maser frequency at 1 second averaging time."""


def compute_schawlow_townes_sensitivity(
    maser_noise_result: MaserNoiseResult,
    nv_config: NVConfig,
) -> float:
    """Schawlow-Townes quantum-limited field sensitivity [T/√Hz].

    The fundamental noise floor arises from quantum phase diffusion:
    spontaneous emission events cause the maser field phase to undergo
    a random walk described by the Schawlow-Townes linewidth Δν_ST.

    Over integration time τ, the accumulated phase error is:

        Δφ(τ) = √(2π Δν_ST τ)   [rad]

    giving a fractional frequency error σ_y(τ) = √(Δν_ST / (2π ν₀² τ))
    and a minimum detectable field:

        η_ST = √(Δν_ST / (2π)) / γ_e   [T/√Hz]

    Args:
        maser_noise_result: Quantum noise characterisation (Δν_ST source).
        nv_config: NV center parameters (γ_e source).

    Returns:
        Schawlow-Townes field sensitivity in T/√Hz.  Always finite and positive.
    """
    delta_nu_st = maser_noise_result.schawlow_townes_linewidth_hz
    gamma_e = nv_config.gamma_e_ghz_per_t * 1e9  # Hz/T
    return math.sqrt(delta_nu_st / _TWO_PI) / gamma_e


def compute_thermal_sensitivity(
    budget: SignalChainBudget,
    maser_noise_result: MaserNoiseResult,
    nv_config: NVConfig,
) -> float:
    """Thermal-SNR-limited field sensitivity using the classical LNA chain [T/√Hz].

    The frequency discriminator operating on the maser carrier with
    cavity half-bandwidth κ_c converts a frequency shift δf into a
    detectable power change.  With system noise temperature T_sys, the
    minimum detectable frequency shift per unit bandwidth is:

        δf_min = κ_c × √(k_B T_sys / P_out)   [Hz/√Hz]

    giving

        η_T = (κ_c / γ_e) × √(k_B T_sys / P_out)   [T/√Hz]

    where the cavity linewidth κ_c sets the discriminator slope and
    P_out is the coupled (received) maser power.

    Args:
        budget: Signal chain budget (T_sys and coupled power).
        maser_noise_result: Quantum noise result (κ_c = cavity_linewidth_hz).
        nv_config: NV center parameters (γ_e source).

    Returns:
        Thermal-SNR-limited field sensitivity in T/√Hz.  Returns inf if
        the received power is zero (no output from maser).
    """
    t_sys = budget.system_noise_temperature_k
    p_out = budget.received_power_w
    kappa_c = maser_noise_result.cavity_linewidth_hz
    gamma_e = nv_config.gamma_e_ghz_per_t * 1e9  # Hz/T

    if p_out <= 0.0:
        return math.inf

    return (kappa_c / gamma_e) * math.sqrt(_KB * t_sys / p_out)


def compute_friis_sensitivity(
    budget: SignalChainBudget,
    maser_noise_result: MaserNoiseResult,
    nv_config: NVConfig,
) -> float:
    """Friis-enhanced field sensitivity using quantum maser pre-amplifier [T/√Hz].

    Same formula as ``compute_thermal_sensitivity`` but with T_sys replaced
    by the Friis cascade noise temperature T_friis ≈ T_maser ≈ 0.11 K:

        η_F = (κ_c / γ_e) × √(k_B T_friis / P_out)   [T/√Hz]

    Returns ``math.nan`` when ``budget.friis_system_temperature_k`` is ``nan``
    (i.e. no MaserNoiseResult was supplied to ``compute_signal_chain_budget``).

    Args:
        budget: Signal chain budget with Friis temperature populated.
        maser_noise_result: Quantum noise result (κ_c source).
        nv_config: NV center parameters (γ_e source).

    Returns:
        Friis-enhanced field sensitivity in T/√Hz, or ``math.nan`` if the
        Friis temperature is not available in the budget.
    """
    t_friis = budget.friis_system_temperature_k
    if math.isnan(t_friis):
        return math.nan

    p_out = budget.received_power_w
    kappa_c = maser_noise_result.cavity_linewidth_hz
    gamma_e = nv_config.gamma_e_ghz_per_t * 1e9  # Hz/T

    if p_out <= 0.0:
        return math.nan

    return (kappa_c / gamma_e) * math.sqrt(_KB * t_friis / p_out)


def compute_sensitivity(
    budget: SignalChainBudget,
    maser_noise_result: MaserNoiseResult,
    nv_config: NVConfig,
) -> SensitivityResult:
    """Complete field sensitivity analysis for the NV maser magnetometer.

    Computes three complementary sensitivity floors and their ratios:

    - Schawlow-Townes (quantum phase diffusion,  Δν_ST driven)
    - Thermal-SNR    (classical LNA chain, T_sys driven)
    - Friis-enhanced (quantum pre-amplifier, T_friis driven)

    The result gives the full picture of where the system stands relative
    to fundamental quantum limits and how much improvement the Friis cascade
    provides over the classical receiver chain.

    Args:
        budget: Completed signal chain budget (from compute_signal_chain_budget).
            Must have been called with maser_noise_result to populate Friis fields;
            otherwise Friis sensitivity will be math.nan.
        maser_noise_result: Quantum noise characterisation.  Provides Δν_ST and
            cavity linewidth κ_c used in the sensitivity formulas.
        nv_config: NV center parameters.  Provides γ_e for T/Hz ↔ T/√Hz conversion.

    Returns:
        SensitivityResult with all floors in T/√Hz, nT/√Hz, and pT/√Hz.
    """
    gamma_e = nv_config.gamma_e_ghz_per_t * 1e9  # Hz/T

    # ── Three sensitivity floors ───────────────────────────────────────────
    eta_st = compute_schawlow_townes_sensitivity(maser_noise_result, nv_config)
    eta_t = compute_thermal_sensitivity(budget, maser_noise_result, nv_config)
    eta_f = compute_friis_sensitivity(budget, maser_noise_result, nv_config)

    # ── Cross-floor ratios ────────────────────────────────────────────────
    thermal_vs_st = eta_t / eta_st if (eta_st > 0.0 and math.isfinite(eta_t)) else math.nan

    if not math.isnan(eta_f) and eta_f > 0.0 and math.isfinite(eta_f):
        friis_vs_st = eta_f / eta_st if eta_st > 0.0 else math.nan
        friis_adv_db = (
            20.0 * math.log10(eta_t / eta_f)
            if (math.isfinite(eta_t) and eta_t > 0.0)
            else math.nan
        )
    else:
        friis_vs_st = math.nan
        friis_adv_db = math.nan

    # ── Allan deviation at 1 s ────────────────────────────────────────────
    # σ_B(τ=1s) = η_ST [T/√Hz] evaluated at τ=1s: σ_B = η_ST × 1/√1 = η_ST
    # More precisely: from σ_y(τ) = √(Δν_ST/(2π ν₀² τ)),
    # σ_B(τ=1s) = σ_y(1) × ν₀ / γ_e = √(Δν_ST/(2π)) / γ_e = η_ST
    allan_1s = eta_st  # T, at τ = 1 second

    def _to_nt(val: float) -> float:
        return val * 1e9 if math.isfinite(val) else math.nan

    def _to_pt(val: float) -> float:
        return val * 1e12 if math.isfinite(val) else math.nan

    return SensitivityResult(
        # ST floor
        schawlow_townes_t_per_sqrthz=eta_st,
        schawlow_townes_nt_per_sqrthz=_to_nt(eta_st),
        schawlow_townes_pt_per_sqrthz=_to_pt(eta_st),
        # Thermal floor
        thermal_snr_t_per_sqrthz=eta_t,
        thermal_snr_nt_per_sqrthz=_to_nt(eta_t),
        thermal_snr_pt_per_sqrthz=_to_pt(eta_t),
        # Friis floor
        friis_t_per_sqrthz=eta_f,
        friis_nt_per_sqrthz=_to_nt(eta_f),
        friis_pt_per_sqrthz=_to_pt(eta_f),
        # Cross-floor ratios
        thermal_vs_st_ratio=thermal_vs_st,
        friis_vs_st_ratio=friis_vs_st,
        friis_advantage_over_thermal_db=friis_adv_db,
        # Provenance
        schawlow_townes_linewidth_hz=maser_noise_result.schawlow_townes_linewidth_hz,
        cavity_linewidth_hz=maser_noise_result.cavity_linewidth_hz,
        system_noise_temperature_k=budget.system_noise_temperature_k,
        friis_system_temperature_k=budget.friis_system_temperature_k,
        output_power_w=budget.received_power_w,
        gamma_e_hz_per_t=gamma_e,
        # Allan deviation
        allan_deviation_1s_t=allan_1s,
    )



