"""
RF signal chain SNR budget for the NV diamond maser.

Models the complete receive path from stimulated emission in the NV
ensemble through the microwave cavity, coupling port, LNA, and ADC
to a digitised signal.

Signal power model
──────────────────
The maser emits into the cavity mode.  Power extracted at the coupling
port is:

    P_signal = ℏω · R_stim · N_active · η_pump · β

where R_stim is the stimulated emission rate (≈ Γ_h for an inverted
two-level system at threshold), N_active is the number of NV centers
in the mode volume, η_pump is the pump efficiency, and β is the
cavity coupling factor (fraction of intracavity power coupled out).

Noise contributors
──────────────────
1. **Thermal (Johnson-Nyquist)** — kB · T_phys · Δf
2. **Amplifier**                — kB · T_amp · Δf  where T_amp = T₀(F−1)
3. **ADC quantisation**         — P_FS / (1.5 · 4^bits)  (uniformly distributed)

SNR = P_signal / (P_thermal + P_amp + P_quant)   [linear]
    = 10·log₁₀( … )                              [dB]

References
──────────
Breeze et al., "Continuous-wave room-temperature diamond maser",
Nature 555, 493 (2018).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import NVConfig, MaserConfig, SignalChainConfig

# ── Physical constants ────────────────────────────────────────────
_HBAR = 1.054571817e-34  # J·s
_KB = 1.380649e-23  # J/K
_T0 = 290.0  # IEEE reference temperature (K)


@dataclass(frozen=True)
class SignalChainBudget:
    """Complete SNR budget snapshot.

    All powers in Watts, SNR in dB.
    """

    # Signal
    maser_emission_power_w: float
    coupled_power_w: float  # after cavity coupling (β)
    received_power_w: float  # after insertion loss

    # Noise
    thermal_noise_w: float
    amplifier_noise_w: float
    quantisation_noise_w: float
    total_noise_w: float

    # System
    system_noise_temperature_k: float
    snr_linear: float
    snr_db: float

    # Context
    detection_bandwidth_hz: float
    cavity_frequency_ghz: float


def compute_maser_emission_power(
    nv_config: NVConfig,
    maser_config: MaserConfig,
    gain_budget: float,
) -> float:
    """
    Estimate maser stimulated emission power from the NV ensemble.

    P = ℏω · Γ_h · N_active · η_pump · gain_budget

    The gain_budget acts as a de-rating factor: at perfect field
    uniformity (gain_budget=1) the full population contributes;
    with inhomogeneous broadening, only the fraction within the
    homogeneous linewidth participates effectively.

    Args:
        nv_config: NV center parameters.
        maser_config: Cavity/maser parameters.
        gain_budget: Fraction of peak gain retained (0–1).

    Returns:
        Stimulated emission power (Watts).  Zero if gain_budget ≤ 0.
    """
    if gain_budget <= 0:
        return 0.0

    omega = 2 * math.pi * maser_config.cavity_frequency_ghz * 1e9  # rad/s

    # Number of active NV centers in the diamond
    # Volume = area × thickness.  We approximate area from the diamond
    # as a cylinder cross-section matching the grid active zone.
    # The active_zone is ~60% of 10 mm → 6 mm square → effective area ≈ (6mm)²
    # For a cylindrical diamond of comparable extent:
    thickness_m = nv_config.diamond_thickness_mm * 1e-3
    # Use a fixed 3 mm active radius (conservative for a 6 mm active zone)
    active_radius_m = 3.0e-3
    volume_m3 = math.pi * active_radius_m**2 * thickness_m
    n_nv = nv_config.nv_density_per_cm3 * 1e6 * volume_m3  # convert /cm³ to /m³

    # Stimulated emission rate ≈ homogeneous linewidth (at threshold)
    t2_s = nv_config.t2_star_us * 1e-6
    gamma_h = 1.0 / (math.pi * t2_s)  # Hz

    power = _HBAR * omega * gamma_h * n_nv * nv_config.pump_efficiency * gain_budget
    return power


def compute_thermal_noise(
    temperature_k: float,
    bandwidth_hz: float,
) -> float:
    """
    Johnson-Nyquist thermal noise power.

    P_thermal = kB · T · Δf

    Args:
        temperature_k: Physical temperature (K).
        bandwidth_hz: Detection bandwidth (Hz).

    Returns:
        Noise power (Watts).
    """
    return _KB * temperature_k * bandwidth_hz


def compute_amplifier_noise(
    noise_figure_db: float,
    bandwidth_hz: float,
) -> float:
    """
    LNA added noise power.

    T_amp = T₀ · (F − 1)   where F = 10^(NF/10)
    P_amp = kB · T_amp · Δf

    Args:
        noise_figure_db: LNA noise figure (dB).
        bandwidth_hz: Detection bandwidth (Hz).

    Returns:
        Amplifier noise power (Watts).
    """
    f_linear = 10.0 ** (noise_figure_db / 10.0)
    t_amp = _T0 * (f_linear - 1.0)
    return _KB * t_amp * bandwidth_hz


def compute_quantisation_noise(
    adc_bits: int,
    full_scale_dbm: float,
) -> float:
    """
    ADC quantisation noise power.

    For a uniform quantiser:
        P_quant = P_FS / (1.5 · 4^bits)

    where P_FS is the full-scale power.

    Args:
        adc_bits: ADC resolution (bits).
        full_scale_dbm: Full-scale input power (dBm).

    Returns:
        Quantisation noise power (Watts).
    """
    p_fs_w = 1e-3 * 10.0 ** (full_scale_dbm / 10.0)  # dBm → W
    return p_fs_w / (1.5 * 4.0**adc_bits)


def compute_signal_chain_budget(
    nv_config: NVConfig,
    maser_config: MaserConfig,
    signal_config: SignalChainConfig,
    gain_budget: float,
) -> SignalChainBudget:
    """
    Compute the complete SNR budget for the receiver chain.

    Signal path: NV emission → cavity coupling (β) → insertion loss → LNA
    Noise path:  thermal + amplifier + ADC quantisation

    Args:
        nv_config: NV center parameters.
        maser_config: Cavity and coupling parameters.
        signal_config: Receiver chain parameters.
        gain_budget: Current maser gain budget (0–1) from field uniformity.

    Returns:
        SignalChainBudget with all power components and SNR.
    """
    # ── Signal power ──────────────────────────────────────────────
    p_emission = compute_maser_emission_power(nv_config, maser_config, gain_budget)
    p_coupled = p_emission * maser_config.coupling_beta
    insertion_linear = 10.0 ** (-signal_config.insertion_loss_db / 10.0)
    p_received = p_coupled * insertion_linear

    # ── Noise powers ──────────────────────────────────────────────
    bw = signal_config.detection_bandwidth_hz
    p_thermal = compute_thermal_noise(signal_config.physical_temperature_k, bw)
    p_amp = compute_amplifier_noise(signal_config.lna_noise_figure_db, bw)
    p_quant = compute_quantisation_noise(
        signal_config.adc_bits, signal_config.adc_full_scale_dbm
    )
    p_noise = p_thermal + p_amp + p_quant

    # ── System noise temperature ──────────────────────────────────
    t_sys = p_noise / (_KB * bw) if bw > 0 else float("inf")

    # ── SNR ───────────────────────────────────────────────────────
    snr_lin = p_received / p_noise if p_noise > 0 else float("inf")
    snr_db = 10.0 * math.log10(snr_lin) if snr_lin > 0 else -math.inf

    return SignalChainBudget(
        maser_emission_power_w=p_emission,
        coupled_power_w=p_coupled,
        received_power_w=p_received,
        thermal_noise_w=p_thermal,
        amplifier_noise_w=p_amp,
        quantisation_noise_w=p_quant,
        total_noise_w=p_noise,
        system_noise_temperature_k=t_sys,
        snr_linear=snr_lin,
        snr_db=snr_db,
        detection_bandwidth_hz=bw,
        cavity_frequency_ghz=maser_config.cavity_frequency_ghz,
    )


def compute_snr_vs_field_uniformity(
    nv_config: NVConfig,
    maser_config: MaserConfig,
    signal_config: SignalChainConfig,
    b_std_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute SNR (dB) as a function of field non-uniformity σ(B).

    Useful for visualising the shimming → SNR link.

    Args:
        nv_config: NV center parameters.
        maser_config: Cavity parameters.
        signal_config: Receiver parameters.
        b_std_values: Array of σ(B) values in Tesla.

    Returns:
        Array of SNR values in dB, same shape as b_std_values.
    """
    from .nv_spin import homogeneous_linewidth_ghz

    gamma_h = homogeneous_linewidth_ghz(nv_config.t2_star_us)
    gamma_e = nv_config.gamma_e_ghz_per_t

    snr_db = np.empty_like(b_std_values)
    for i, b_std in enumerate(b_std_values):
        gamma_inh = gamma_e * b_std
        gamma_eff = gamma_h + gamma_inh
        gain_budget = gamma_h / gamma_eff if gamma_eff > 0 else 0.0

        budget = compute_signal_chain_budget(
            nv_config, maser_config, signal_config, gain_budget
        )
        snr_db[i] = budget.snr_db

    return snr_db
