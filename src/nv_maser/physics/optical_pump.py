"""
Optical pump model for the NV diamond maser.

A 532 nm CW laser drives the spin-selective intersystem crossing (ISC)
that polarises the NV⁻ ground state into |0⟩, creating population
inversion for the |0⟩ → |−1⟩ (or |+1⟩) microwave transition.

Pump physics
────────────
Intensity:       I₀ = 2P / (π w₀²)              [W/m²]
Pump rate:       Γ_pump = σ_abs · I₀ / (ℏ ω_L)  [Hz per NV]
Saturation:      s = Γ_pump / (Γ_pump + 1/T₁)
Inversion:       η_eff = s × (2/3)
                 (from thermal 1/3 per sublevel → all in |0⟩ at saturation)

Absorbed power:  P_abs = P_laser × (1 − e^{−α d})
                 where α = n_NV × σ_abs
Thermal load:    P_heat = P_abs × quantum_defect_fraction

References
──────────
Tetienne et al., NJP 14, 103033 (2012).
Breeze et al., Nature 555, 493 (2018).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import NVConfig, OpticalPumpConfig


# ── Physical constants ────────────────────────────────────────────
_HBAR = 1.054571817e-34  # J·s
_C = 2.99792458e8  # m/s


@dataclass(frozen=True)
class PumpState:
    """State of the optical pump and its effects on the diamond."""

    pump_rate_hz: float  # Γ_pump per NV center
    pump_saturation: float  # s = Γ_pump / (Γ_pump + 1/T₁)  ∈ [0, 1]
    effective_pump_efficiency: float  # η_eff replacing flat NVConfig.pump_efficiency
    absorbed_power_w: float  # optical power deposited in the diamond
    thermal_load_w: float  # heat into diamond (absorbed × quantum defect)
    pump_intensity_w_per_m2: float  # peak intensity at beam centre


def compute_pump_rate(
    pump_config: OpticalPumpConfig,
) -> float:
    """
    Optical excitation rate per NV centre at the beam centre.

    Γ_pump = σ_abs × I₀ / (ℏ ω_L)

    where I₀ = 2P / (π w₀²) is the peak Gaussian intensity.

    Args:
        pump_config: Laser parameters and NV cross-section.

    Returns:
        Pump rate in Hz.
    """
    omega_laser = 2 * math.pi * _C / (pump_config.laser_wavelength_nm * 1e-9)
    w0 = pump_config.beam_waist_mm * 1e-3  # m
    i0 = 2.0 * pump_config.laser_power_w / (math.pi * w0**2)
    return pump_config.absorption_cross_section_m2 * i0 / (_HBAR * omega_laser)


def compute_absorbed_power(
    pump_config: OpticalPumpConfig,
    nv_config: NVConfig,
) -> float:
    """
    Optical power absorbed by the diamond (single-pass Beer-Lambert).

    P_abs = P_laser × (1 − exp(−α d))  where α = n_NV × σ_abs

    Args:
        pump_config: Laser power, cross-section.
        nv_config: NV density, diamond thickness.

    Returns:
        Absorbed power (Watts).
    """
    n_nv_m3 = nv_config.nv_density_per_cm3 * 1e6
    alpha = n_nv_m3 * pump_config.absorption_cross_section_m2  # /m
    d = nv_config.diamond_thickness_mm * 1e-3  # m
    return pump_config.laser_power_w * (1.0 - math.exp(-alpha * d))


def compute_pump_state(
    pump_config: OpticalPumpConfig,
    nv_config: NVConfig,
) -> PumpState:
    """
    Full optical pump characterisation.

    Combines pump rate, saturation, effective inversion efficiency,
    absorbed power, and thermal load into a single snapshot.

    Args:
        pump_config: Laser and cross-section parameters.
        nv_config: NV density, thickness, T₁.

    Returns:
        PumpState with all derived quantities.
    """
    # ── Pump rate ─────────────────────────────────────────────────
    gamma_pump = compute_pump_rate(pump_config)

    # ── Saturation ────────────────────────────────────────────────
    # NOTE: spin_t1_ms is treated as temperature-independent here.
    # In reality T₁ shortens at elevated temperatures (Orbach process),
    # which would reduce saturation.  A future refinement could accept
    # T₁(T) from the thermal model.
    gamma_relax = 1.0 / (pump_config.spin_t1_ms * 1e-3)  # Hz
    saturation = gamma_pump / (gamma_pump + gamma_relax)

    # ── Effective inversion ───────────────────────────────────────
    # At saturation → ~100% in |0⟩.  Two-thirds excess over thermal
    # equilibrium (1/3 per sublevel) → η_eff = s × 2/3 ∈ [0, 2/3].
    eta_eff = saturation * (2.0 / 3.0)

    # ── Absorbed power & thermal load ─────────────────────────────
    p_abs = compute_absorbed_power(pump_config, nv_config)
    p_heat = p_abs * pump_config.quantum_defect_fraction

    # ── Peak intensity ────────────────────────────────────────────
    w0 = pump_config.beam_waist_mm * 1e-3
    i0 = 2.0 * pump_config.laser_power_w / (math.pi * w0**2)

    return PumpState(
        pump_rate_hz=gamma_pump,
        pump_saturation=saturation,
        effective_pump_efficiency=eta_eff,
        absorbed_power_w=p_abs,
        thermal_load_w=p_heat,
        pump_intensity_w_per_m2=i0,
    )


# ── Depth-resolved pump model ────────────────────────────────────


@dataclass(frozen=True)
class DepthResolvedPumpResult:
    """Depth-resolved optical pump characterisation.

    The pump beam is attenuated through the diamond via Beer-Lambert:
    I(z) = I₀ exp(-α z).  Each depth slice has its own pump rate,
    saturation, and inversion efficiency.  The weighted average gives
    a more accurate effective inversion than the uniform model.
    """

    z_m: tuple[float, ...]  # depth positions (m)
    pump_rate_hz: tuple[float, ...]  # Γ_pump(z) per slice
    saturation: tuple[float, ...]  # s(z) per slice
    inversion: tuple[float, ...]  # η(z) per slice
    effective_pump_efficiency: float  # depth-averaged η_eff
    absorbed_power_w: float  # total absorbed (same as uniform model)
    thermal_load_w: float  # total heat
    front_back_ratio: float  # I(0) / I(d) — pump non-uniformity


def compute_depth_resolved_pump(
    pump_config: OpticalPumpConfig,
    nv_config: NVConfig,
    n_slices: int = 20,
) -> DepthResolvedPumpResult:
    """
    Depth-resolved pump model accounting for Beer-Lambert attenuation.

    At high NV density (≳10 ppm), the pump beam attenuates significantly
    through the crystal.  Front-surface NVs are fully saturated while
    back-surface NVs may be barely pumped.

    Kollarics et al. 2024 (12 ppm, 0.5 mm): α×d ≈ 3.3, 25× front-to-back
    intensity drop.

    Args:
        pump_config: Laser parameters and NV cross-section.
        nv_config: NV density, diamond thickness, T₁.
        n_slices: Number of depth slices (≥1).

    Returns:
        DepthResolvedPumpResult with per-slice profiles and weighted average.
    """
    if n_slices < 1:
        raise ValueError("n_slices must be >= 1")

    d = nv_config.diamond_thickness_mm * 1e-3  # m
    n_nv_m3 = nv_config.nv_density_per_cm3 * 1e6
    alpha = n_nv_m3 * pump_config.absorption_cross_section_m2  # /m

    omega_laser = 2 * math.pi * _C / (pump_config.laser_wavelength_nm * 1e-9)
    w0 = pump_config.beam_waist_mm * 1e-3  # m
    i0 = 2.0 * pump_config.laser_power_w / (math.pi * w0**2)
    gamma_relax = 1.0 / (pump_config.spin_t1_ms * 1e-3)

    # Slice midpoints
    dz = d / n_slices
    z_list: list[float] = []
    rate_list: list[float] = []
    sat_list: list[float] = []
    inv_list: list[float] = []

    for i in range(n_slices):
        z = (i + 0.5) * dz  # midpoint
        i_z = i0 * math.exp(-alpha * z)
        gamma_pump = pump_config.absorption_cross_section_m2 * i_z / (_HBAR * omega_laser)
        s = gamma_pump / (gamma_pump + gamma_relax)
        eta = s * (2.0 / 3.0)

        z_list.append(z)
        rate_list.append(gamma_pump)
        sat_list.append(s)
        inv_list.append(eta)

    # Depth-averaged effective efficiency (uniform weight per slice)
    eta_avg = sum(inv_list) / n_slices

    # Absorbed power and thermal load (same as uniform model)
    p_abs = pump_config.laser_power_w * (1.0 - math.exp(-alpha * d))
    p_heat = p_abs * pump_config.quantum_defect_fraction

    # Front-to-back intensity ratio
    fb_ratio = math.exp(alpha * d) if alpha * d > 0 else 1.0

    return DepthResolvedPumpResult(
        z_m=tuple(z_list),
        pump_rate_hz=tuple(rate_list),
        saturation=tuple(sat_list),
        inversion=tuple(inv_list),
        effective_pump_efficiency=eta_avg,
        absorbed_power_w=p_abs,
        thermal_load_w=p_heat,
        front_back_ratio=fb_ratio,
    )
