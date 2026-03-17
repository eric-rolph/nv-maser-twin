"""
1D NMR depth profiling through a single-sided magnet gradient.

In a single-sided geometry the B₀ field varies with depth (the residual
gradient outside the sweet spot).  Each depth layer has a unique Larmor
frequency, so a frequency-resolved NMR acquisition maps directly to a
depth profile — analogous to A-mode ultrasound.

Tissue model
────────────
Tissue is modelled as stacked planar layers, each with:
- thickness (mm)
- proton density (relative to water = 1.0)
- T1 relaxation time (ms)
- T2 relaxation time (ms)
- name (e.g. "skin", "fat", "muscle", "bone cortex")

Signal model
────────────
For each depth slice at depth z:

1. Larmor frequency: f(z) = γ B₀(z)
2. Equilibrium magnetisation: M₀(z) ∝ ρ(z) × B₀(z) / T_body
3. Signal after excitation:
   - Spin echo: S(z) ∝ M₀(z) × [1 − exp(−TR/T1)] × exp(−TE/T2)
   - CPMG: S(z, echo_n) ∝ M₀(z) × exp(−2n·τ/T2) × [1 − exp(−TR/T1)]
4. Coil sensitivity: S(z) × (B₁/I)(z) — falls off as ~1/(a²+z²)^(3/2)
5. Detected EMF per depth slice: ε(z) = ω(z) × M₀(z) × V_slice × S(z) × sens(z)

The depth profile is the 1D map of detected signal vs. depth.

References
──────────
Blümich et al., "Mobile single-sided NMR", Prog. NMR Spectroscopy 52, 197 (2008).
Casanova et al., "Single-Sided NMR" (Springer, 2011).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..config import DepthProfileConfig
from .single_sided_magnet import SingleSidedMagnet
from .surface_coil import SurfaceCoil, sensitivity_on_axis

# ── Physical constants ────────────────────────────────────────────
_KB = 1.380649e-23  # J/K
_HBAR = 1.054571817e-34  # J·s
_GAMMA_P = 2.675e8  # proton gyromagnetic ratio (rad/s/T)
_N_PROTONS_WATER = 6.69e28  # protons per m³ in water


@dataclass
class TissueLayer:
    """A single tissue layer in the depth model."""

    name: str
    thickness_mm: float
    proton_density: float = 1.0  # relative to water
    t1_ms: float = 600.0
    t2_ms: float = 50.0

    def __post_init__(self) -> None:
        if self.thickness_mm <= 0:
            raise ValueError(f"Thickness must be > 0, got {self.thickness_mm}")
        if self.proton_density < 0:
            raise ValueError(f"Proton density must be >= 0, got {self.proton_density}")


@dataclass(frozen=True)
class DepthProfile:
    """Result of a 1D NMR depth profile simulation."""

    depths_mm: NDArray[np.float64]
    b0_tesla: NDArray[np.float64]
    larmor_mhz: NDArray[np.float64]
    signal: NDArray[np.float64]  # detected EMF per depth bin (V)
    noise_v: float  # RMS noise voltage
    snr: NDArray[np.float64]  # signal / noise at each depth
    tissue_labels: list[str]  # tissue name at each depth
    scan_time_s: float


# ── Default tissue models ─────────────────────────────────────────

FOREARM_LAYERS: list[TissueLayer] = [
    TissueLayer("skin", thickness_mm=2.0, proton_density=0.7, t1_ms=400, t2_ms=30),
    TissueLayer("subcutaneous_fat", thickness_mm=5.0, proton_density=0.9, t1_ms=250, t2_ms=80),
    TissueLayer("muscle", thickness_mm=20.0, proton_density=1.0, t1_ms=600, t2_ms=35),
    TissueLayer("bone_cortex", thickness_mm=3.0, proton_density=0.05, t1_ms=1000, t2_ms=0.5),
]

HEMORRHAGE_LAYERS: list[TissueLayer] = [
    TissueLayer("skin", thickness_mm=2.0, proton_density=0.7, t1_ms=400, t2_ms=30),
    TissueLayer("subcutaneous_fat", thickness_mm=3.0, proton_density=0.9, t1_ms=250, t2_ms=80),
    TissueLayer("hemorrhage", thickness_mm=8.0, proton_density=1.0, t1_ms=900, t2_ms=150),
    TissueLayer("muscle", thickness_mm=15.0, proton_density=1.0, t1_ms=600, t2_ms=35),
]


def _equilibrium_magnetisation(
    b0: float, proton_density: float, temperature_k: float = 310.0
) -> float:
    """Equilibrium nuclear magnetisation M₀ (A/m).

    M₀ = n_H ρ γ²ℏ² I(I+1) B₀ / (3 kB T)
    """
    n = _N_PROTONS_WATER * proton_density
    spin_i = 0.5
    return n * _GAMMA_P**2 * _HBAR**2 * spin_i * (spin_i + 1) * abs(b0) / (3 * _KB * temperature_k)


def simulate_depth_profile(
    magnet: SingleSidedMagnet,
    coil: SurfaceCoil,
    config: DepthProfileConfig,
    tissue_layers: list[TissueLayer] | None = None,
    sequence: str = "spin_echo",
    te_ms: float = 10.0,
) -> DepthProfile:
    """Simulate a 1D NMR depth profile.

    Args:
        magnet: single-sided magnet providing B₀(z).
        coil: surface coil providing sensitivity(z) and noise.
        config: depth profile simulation parameters.
        tissue_layers: tissue model (stacked layers). Defaults to forearm.
        sequence: pulse sequence type ("spin_echo" or "cpmg").
        te_ms: echo time (ms).

    Returns:
        DepthProfile with signal, noise, and SNR vs. depth.
    """
    if tissue_layers is None:
        tissue_layers = FOREARM_LAYERS

    # Build depth grid
    depths = np.arange(
        config.depth_resolution_mm / 2,
        config.max_depth_mm,
        config.depth_resolution_mm,
    )

    # Assign tissue properties at each depth
    layer_props = _assign_layers(depths, tissue_layers)

    # B₀ field at each depth (from magnet model)
    b0 = magnet.field_on_axis(depths)

    # Larmor frequency
    larmor_hz = np.abs(b0) * config.proton_larmor_mhz_per_t * 1e6
    larmor_mhz = larmor_hz / 1e6

    # Coil sensitivity at each depth
    sens = sensitivity_on_axis(coil.config, depths)

    # Voxel volume (depth slice × lateral area)
    # For depth profiling, the "voxel" is a thin slice of depth = depth_resolution
    # with lateral extent determined by coil sensitivity (≈ coil area)
    slice_thickness_m = config.depth_resolution_mm * 1e-3
    lateral_area_m2 = math.pi * (config.voxel_size_mm * 1e-3 / 2) ** 2
    v_slice = slice_thickness_m * lateral_area_m2

    # Signal at each depth
    signal = np.zeros_like(depths)
    labels: list[str] = []

    tr_s = config.repetition_time_ms * 1e-3
    te_s = te_ms * 1e-3

    for i, d in enumerate(depths):
        props = layer_props[i]
        labels.append(props["name"])

        # Equilibrium magnetisation
        m0 = _equilibrium_magnetisation(b0[i], props["proton_density"])

        # T1 saturation factor
        t1_s = props["t1_ms"] * 1e-3
        sat = 1.0 - math.exp(-tr_s / t1_s) if t1_s > 0 else 1.0

        # T2 decay
        t2_s = props["t2_ms"] * 1e-3
        if sequence == "spin_echo":
            decay = math.exp(-te_s / t2_s) if t2_s > 0 else 0.0
        elif sequence == "cpmg":
            # First echo of CPMG train
            decay = math.exp(-te_s / t2_s) if t2_s > 0 else 0.0
        else:
            raise ValueError(f"Unknown sequence: {sequence}")

        # EMF = ω₀ × M₀ × V × (B₁/I) × sat × decay
        omega0 = 2 * math.pi * larmor_hz[i]
        emf = omega0 * m0 * v_slice * sens[i] * sat * decay
        signal[i] = abs(emf)

    # Noise (from coil at a representative frequency)
    mean_freq = float(np.mean(larmor_hz[larmor_hz > 0])) if np.any(larmor_hz > 0) else 2.13e6
    noise_comp = coil.noise(mean_freq, config.readout_bandwidth_hz)
    noise_v = noise_comp.total_noise_v

    # Apply averaging
    signal_averaged = signal * math.sqrt(config.n_averages)
    snr = signal_averaged / noise_v if noise_v > 0 else np.full_like(signal, float("inf"))

    # Scan time
    scan_time = config.n_averages * config.repetition_time_ms * 1e-3

    return DepthProfile(
        depths_mm=depths,
        b0_tesla=b0,
        larmor_mhz=larmor_mhz,
        signal=signal_averaged,
        noise_v=noise_v,
        snr=snr,
        tissue_labels=labels,
        scan_time_s=scan_time,
    )


def _assign_layers(
    depths_mm: NDArray[np.float64],
    layers: list[TissueLayer],
) -> list[dict]:
    """Map depth positions to tissue layer properties.

    Layers are stacked from surface (depth=0) downward.

    Returns:
        List of dicts with tissue properties at each depth.
    """
    result = []
    boundaries: list[float] = []
    cumulative = 0.0
    for layer in layers:
        cumulative += layer.thickness_mm
        boundaries.append(cumulative)

    for d in depths_mm:
        assigned = False
        cumulative = 0.0
        for j, layer in enumerate(layers):
            if d <= boundaries[j]:
                result.append({
                    "name": layer.name,
                    "proton_density": layer.proton_density,
                    "t1_ms": layer.t1_ms,
                    "t2_ms": layer.t2_ms,
                })
                assigned = True
                break
        if not assigned:
            # Beyond defined layers — use last layer's properties
            last = layers[-1]
            result.append({
                "name": last.name,
                "proton_density": last.proton_density,
                "t1_ms": last.t1_ms,
                "t2_ms": last.t2_ms,
            })
    return result


def add_noise(
    profile: DepthProfile,
    rng: np.random.Generator | None = None,
) -> DepthProfile:
    """Add realistic Gaussian noise to a simulated depth profile.

    Args:
        profile: clean simulated profile.
        rng: random number generator (for reproducibility).

    Returns:
        New DepthProfile with noise added to the signal.
    """
    if rng is None:
        rng = np.random.default_rng()

    noise_realisation = rng.normal(0, profile.noise_v, size=profile.signal.shape)
    noisy_signal = profile.signal + noise_realisation
    noisy_snr = np.abs(noisy_signal) / profile.noise_v if profile.noise_v > 0 else profile.snr

    return DepthProfile(
        depths_mm=profile.depths_mm,
        b0_tesla=profile.b0_tesla,
        larmor_mhz=profile.larmor_mhz,
        signal=noisy_signal,
        noise_v=profile.noise_v,
        snr=noisy_snr,
        tissue_labels=profile.tissue_labels,
        scan_time_s=profile.scan_time_s,
    )
