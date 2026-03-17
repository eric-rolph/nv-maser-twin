"""
Flat spiral surface coil for single-sided NMR transmit/receive.

Models the RF coil placed on the tissue surface for the handheld MRI
probe.  Computes B₁ sensitivity, impedance, thermal noise, and body
noise as functions of frequency and depth into tissue.

B₁ sensitivity model
─────────────────────
A flat circular loop of N turns, radius a, carrying current I produces
an axial magnetic field at depth d on-axis:

    B₁(d) = µ₀ N I a² / [2 (a² + d²)^(3/2)]

The sensitivity (reciprocity) is:

    S(d) = B₁/I = µ₀ N a² / [2 (a² + d²)^(3/2)]

This is the key quantity: it determines both the excitation efficiency
and the receive sensitivity (by reciprocity).

Noise model
───────────
Three noise sources:

1. **Coil thermal (Johnson) noise:**
   V_n,coil = √(4 kB T R Δf)
   where R is the AC resistance at the Larmor frequency.

2. **Body noise** (dielectric loss in tissue):
   V_n,body = √(4 kB T_body R_body Δf)
   R_body ∝ ω² σ V_eff / (loading factor)
   At 2 MHz this is negligible (R_body << R_coil).

3. **Preamplifier noise** (handled by maser model, not here).

References
──────────
Demas et al., "Electronic characterization of lithographically patterned
microcoils for NMR", J. Magn. Reson. 200, 56 (2009).

Hoult & Richards, "The signal-to-noise ratio of the NMR experiment",
J. Magn. Reson. 24, 71 (1976).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import SurfaceCoilConfig

# ── Physical constants ────────────────────────────────────────────
_MU0 = 4.0 * math.pi * 1e-7  # T·m/A
_KB = 1.380649e-23  # J/K
_RHO_CU = 1.68e-8  # copper resistivity at 20 °C (Ω·m)


@dataclass(frozen=True)
class CoilProperties:
    """Electromagnetic properties of the surface coil at operating frequency."""

    dc_resistance_ohm: float
    ac_resistance_ohm: float
    inductance_h: float
    skin_depth_mm: float
    resonant_frequency_hz: float  # self-resonant (untuned)


@dataclass(frozen=True)
class NoiseComponents:
    """Noise voltage breakdown for the surface coil."""

    coil_thermal_v: float  # V_rms from coil resistance
    body_noise_v: float  # V_rms from tissue dielectric loss
    total_noise_v: float  # RSS of all noise sources
    bandwidth_hz: float


def sensitivity_on_axis(
    config: SurfaceCoilConfig,
    depths_mm: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute B₁/I sensitivity along the central axis.

    Args:
        config: surface coil configuration.
        depths_mm: depth below coil plane (mm).

    Returns:
        Sensitivity B₁/I in T/A at each depth.
    """
    a = config.coil_radius_mm * 1e-3  # metres
    n = config.n_turns
    d = depths_mm * 1e-3  # metres

    return _MU0 * n * a**2 / (2.0 * (a**2 + d**2) ** 1.5)


def sensitivity_off_axis(
    config: SurfaceCoilConfig,
    rho_mm: NDArray[np.float64],
    depth_mm: float,
) -> NDArray[np.float64]:
    """B₁/I at lateral offset rho and fixed depth (approximate).

    For a single-turn loop, the off-axis field involves elliptic
    integrals.  We use a simplified model valid for |ρ| < a:

        S(ρ, d) ≈ S(0, d) × [1 − (3/2)(ρ/a)² × a²/(a²+d²)]

    This is the first correction from the multipole expansion.

    Args:
        config: surface coil configuration.
        rho_mm: lateral displacement from axis (mm).
        depth_mm: depth below coil plane (mm).

    Returns:
        Approximate B₁/I in T/A at each lateral position.
    """
    a = config.coil_radius_mm * 1e-3
    d = depth_mm * 1e-3
    rho = rho_mm * 1e-3

    s0 = float(sensitivity_on_axis(config, np.array([depth_mm]))[0])
    correction = 1.0 - 1.5 * (rho / a) ** 2 * a**2 / (a**2 + d**2)
    return s0 * np.clip(correction, 0.0, None)


def compute_coil_properties(
    config: SurfaceCoilConfig,
    frequency_hz: float,
) -> CoilProperties:
    """Compute electromagnetic properties of the coil at operating frequency.

    Args:
        config: surface coil configuration.
        frequency_hz: operating (Larmor) frequency in Hz.

    Returns:
        CoilProperties with resistance, inductance, skin depth.
    """
    a = config.coil_radius_mm * 1e-3  # mean radius (m)
    n = config.n_turns
    d_wire = config.wire_diameter_mm * 1e-3  # wire diameter (m)

    # DC resistance: R_dc = ρ × L_wire / A_wire
    wire_length = 2 * math.pi * a * n
    wire_area = math.pi * (d_wire / 2) ** 2
    r_dc = _RHO_CU * wire_length / wire_area

    # Skin depth at operating frequency
    omega = 2 * math.pi * frequency_hz
    skin_depth = math.sqrt(2 * _RHO_CU / (_MU0 * omega)) if omega > 0 else float("inf")

    # AC resistance: if skin_depth < wire_radius, current is confined
    wire_radius = d_wire / 2
    if skin_depth < wire_radius:
        # Effective area is annular: A_eff ≈ π d_wire × δ
        area_eff = math.pi * d_wire * skin_depth
        r_ac = _RHO_CU * wire_length / area_eff
    else:
        r_ac = r_dc

    # Inductance of flat spiral (Wheeler approximation)
    # L ≈ µ₀ N² a / 2 × [ln(8a/d_wire) − 2]  for single-layer
    if d_wire > 0 and a > 0:
        inductance = _MU0 * n**2 * a / 2 * (math.log(8 * a / d_wire) - 2)
    else:
        inductance = 0.0

    # Self-resonant frequency (parasitic capacitance estimate)
    # Very rough: f_sr ≈ 1/(2π√(LC)), C_parasitic ≈ ε₀ × pitch_length
    # For now, just report a placeholder
    f_sr = 1.0 / (2 * math.pi * math.sqrt(max(inductance, 1e-15) * 1e-12))

    return CoilProperties(
        dc_resistance_ohm=r_dc,
        ac_resistance_ohm=r_ac,
        inductance_h=inductance,
        skin_depth_mm=skin_depth * 1e3,
        resonant_frequency_hz=f_sr,
    )


def compute_noise(
    config: SurfaceCoilConfig,
    frequency_hz: float,
    bandwidth_hz: float,
) -> NoiseComponents:
    """Compute noise voltage components.

    Args:
        config: surface coil configuration.
        frequency_hz: Larmor frequency (Hz).
        bandwidth_hz: detection bandwidth (Hz).

    Returns:
        NoiseComponents with coil thermal and body noise voltages.
    """
    props = compute_coil_properties(config, frequency_hz)

    # Coil thermal noise
    v_coil = math.sqrt(4 * _KB * config.temperature_k * props.ac_resistance_ohm * bandwidth_hz)

    # Body noise: R_body ∝ ω² σ a³ / 5 (Hoult & Lauterbur approximation)
    # At 2 MHz, this is typically << R_coil for surface coils
    omega = 2 * math.pi * frequency_hz
    a = config.coil_radius_mm * 1e-3
    sigma = config.tissue_conductivity_s_per_m
    r_body = omega**2 * _MU0**2 * sigma * a**3 / 5
    v_body = math.sqrt(4 * _KB * 310.0 * r_body * bandwidth_hz)  # body at 310 K

    v_total = math.sqrt(v_coil**2 + v_body**2)

    return NoiseComponents(
        coil_thermal_v=v_coil,
        body_noise_v=v_body,
        total_noise_v=v_total,
        bandwidth_hz=bandwidth_hz,
    )


def snr_per_voxel(
    config: SurfaceCoilConfig,
    depth_mm: float,
    voxel_size_mm: float,
    b0_tesla: float,
    frequency_hz: float,
    bandwidth_hz: float,
    n_averages: int = 1,
    body_temperature_k: float = 310.0,
) -> float:
    """Compute single-acquisition SNR for one voxel at a given depth.

    Uses the reciprocity principle:
        SNR = ω₀ M₀ V_voxel × (B₁/I) / V_noise

    Args:
        config: surface coil configuration.
        depth_mm: voxel depth below coil plane (mm).
        voxel_size_mm: isotropic voxel side length (mm).
        b0_tesla: B₀ field strength at the voxel (Tesla).
        frequency_hz: Larmor frequency (Hz).
        bandwidth_hz: readout bandwidth (Hz).
        n_averages: number of signal averages (NEX).
        body_temperature_k: tissue temperature (K).

    Returns:
        SNR (linear, dimensionless).
    """
    # Equilibrium magnetisation: M₀ = n_H γ²ℏ²I(I+1) B₀ / (3 kB T)
    # For water protons at body temperature:
    n_h = 6.69e28  # proton density in water (per m³)
    gamma = 2.675e8  # proton gyromagnetic ratio (rad/s/T)
    hbar = 1.054571817e-34
    spin_i = 0.5
    m0 = n_h * gamma**2 * hbar**2 * spin_i * (spin_i + 1) * b0_tesla / (3 * _KB * body_temperature_k)

    # Voxel volume
    v_voxel = (voxel_size_mm * 1e-3) ** 3

    # Sensitivity at depth
    sens = float(sensitivity_on_axis(config, np.array([depth_mm]))[0])

    # EMF per voxel: ε = ω₀ M₀ V × (B₁/I)
    omega0 = 2 * math.pi * frequency_hz
    emf = omega0 * m0 * v_voxel * sens

    # Noise
    noise = compute_noise(config, frequency_hz, bandwidth_hz)

    # SNR with averaging
    return abs(emf) / noise.total_noise_v * math.sqrt(n_averages)


class SurfaceCoil:
    """Stateful surface coil object for convenient multi-query use."""

    def __init__(self, config: SurfaceCoilConfig) -> None:
        self.config = config

    def b1_per_amp(self, depths_mm: NDArray[np.float64]) -> NDArray[np.float64]:
        """B₁/I sensitivity at given depths (T/A)."""
        return sensitivity_on_axis(self.config, depths_mm)

    def properties(self, frequency_hz: float) -> CoilProperties:
        """Coil electromagnetic properties at operating frequency."""
        return compute_coil_properties(self.config, frequency_hz)

    def noise(self, frequency_hz: float, bandwidth_hz: float) -> NoiseComponents:
        """Noise voltage components at operating frequency."""
        return compute_noise(self.config, frequency_hz, bandwidth_hz)

    def snr(
        self,
        depth_mm: float,
        voxel_size_mm: float,
        b0_tesla: float,
        frequency_hz: float,
        bandwidth_hz: float,
        n_averages: int = 1,
    ) -> float:
        """Single-voxel SNR at given depth."""
        return snr_per_voxel(
            self.config, depth_mm, voxel_size_mm,
            b0_tesla, frequency_hz, bandwidth_hz, n_averages,
        )
