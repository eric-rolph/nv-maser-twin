"""
Magnetic susceptibility field-shift modeling for 1D NMR depth profiling.

Background
──────────
Tissues are weakly diamagnetic/paramagnetic.  In a uniform external field B₀
each voxel acquires an additional magnetisation M = χ B₀ / μ₀ that slightly
perturbs the local field.  At tissue boundaries the discontinuity in χ creates
a spatially-varying ΔB₀ that:

1. Shifts the Larmor frequency by Δf(z) = γ_p ΔB₀(z) / 2π
2. Creates intravoxel dephasing — spins across a voxel precess at slightly
   different frequencies, causing amplitude loss proportional to sinc(Δφ/2)
   where Δφ = 2π Δf TE is the accumulated phase spread.

Model choices
─────────────
We use a **1-D planar slab** geometry (the dominant geometry in single-sided
NMR with layered soft tissue):

- Demagnetisation factor N = 0 for an infinite slab perpendicular to B₀
  → ΔB₀ = μ₀ χ_eff B₀  (full susceptibility effect, no shape correction)
- Alternatively, N = 1/3 for a sphere (Lorentz sphere / "soft tissue" model)
  → ΔB₀ = (χ_eff / 3) B₀

In practice soft-tissue imaging literature uses the sphere model because
tissue voxels are roughly isotropic.  We expose both via `geometry` parameter.

The **field shift at each depth** is relative to the reference susceptibility
(default: water, χ_ref = -9.05 ppm).  Each tissue layer contributes:

    Δχ_eff(z) = χ_tissue(z) - χ_ref  [ppm]
    ΔB₀(z)   = Δχ_eff(z) × 1e-6 × N_eff × B₀(z)  [T]

where N_eff = 1 (slab) or 1/3 (sphere).

Intravoxel dephasing
────────────────────
Within a voxel of thickness Δz, if ΔB₀ varies linearly by δB across the
voxel, the signal is attenuated by |sinc(γ_p δB TE / 2)|.  We approximate
δB as the gradient |dΔB₀/dz| × Δz_voxel.

References
──────────
Schenck (1996) "The role of magnetic susceptibility in MRI: MRI compatibility
and artifacts", Magn Reson Med 36(2):199–210.
De Graaf (2007) "In Vivo NMR Spectroscopy", Table 2.1, Wiley.
Haacke et al. (2015) "Susceptibility Weighted Imaging in MRI", Wiley-Blackwell.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .depth_profile import TissueLayer, _assign_layers

# ── Proton gyromagnetic ratio ─────────────────────────────────────
_GAMMA_P = 2.675e8  # rad / s / T

# ── Canonical susceptibility values (ppm volume SI) ──────────────
# Source: De Graaf (2007) Table 2.1; Schenck (1996).
SUSCEPTIBILITY_TABLE: dict[str, float] = {
    "water": -9.05,
    "skin": -9.40,
    "subcutaneous_fat": -7.79,
    "fat": -7.79,
    "muscle": -9.05,
    "bone_cortex": -8.86,
    "bone": -8.86,
    "blood": -9.05,        # oxygenated
    "hemorrhage": -9.05,   # acute: deoxy-Hb shifts toward +3.1 ppm, use water default
    "air": +0.36,
    "diamond": -21.5,      # NV diamond crystal (probe body susceptibility)
    "copper": -9.6,        # surface coil (probe body susceptibility)
}

# Demagnetisation factors for supported geometry models
_N_FACTORS: dict[str, float] = {
    "slab": 1.0,    # infinite plane perpendicular to B₀ — N = 0 along B₀ → full effect
    "sphere": 1 / 3,  # isotropic sphere (Lorentz sphere correction)
}


@dataclass(frozen=True)
class SusceptibilityProfile:
    """Result of a susceptibility field-shift computation."""

    depths_mm: NDArray[np.float64]
    chi_ppm: NDArray[np.float64]          # susceptibility at each depth (ppm)
    delta_chi_ppm: NDArray[np.float64]    # deviation from reference χ (ppm)
    delta_b0_tesla: NDArray[np.float64]   # ΔB₀ perturbation at each depth (T)
    delta_freq_hz: NDArray[np.float64]    # Larmor frequency shift (Hz)
    dephasing_factor: NDArray[np.float64] # signal amplitude factor ∈ [0, 1]
    tissue_labels: list[str]
    b0_tesla: NDArray[np.float64]         # unperturbed B₀ used for calculation
    reference_chi_ppm: float
    geometry: str


@dataclass(frozen=True)
class SusceptibilityCorrectedProfile:
    """Depth profile signal after applying susceptibility correction."""

    depths_mm: NDArray[np.float64]
    signal_original: NDArray[np.float64]  # signal without susceptibility correction
    signal_corrected: NDArray[np.float64] # signal including dephasing loss + any freq shift
    correction_factor: NDArray[np.float64] # per-depth multiplicative correction (≤ 1)
    susceptibility: SusceptibilityProfile
    tissue_labels: list[str]


# ── Core computation functions ────────────────────────────────────

def compute_susceptibility_field_shift(
    tissue_layers: list[TissueLayer],
    b0_profile: NDArray[np.float64],
    depths_mm: NDArray[np.float64],
    reference_chi_ppm: float = -9.05,
    geometry: str = "sphere",
) -> SusceptibilityProfile:
    """Compute susceptibility-induced ΔB₀ perturbation at each depth.

    Args:
        tissue_layers: stacked tissue layers with ``susceptibility_ppm`` field.
        b0_profile: unperturbed B₀ field (T) at each depth.
        depths_mm: depth positions (mm) corresponding to ``b0_profile``.
        reference_chi_ppm: susceptibility of the reference medium (ppm).
            Shifts are computed relative to this value.  Default: water.
        geometry: demagnetisation model — 'sphere' (default, N=1/3) or 'slab' (N=1).

    Returns:
        SusceptibilityProfile with ΔB₀, Δf, and dephasing factors.
    """
    if geometry not in _N_FACTORS:
        raise ValueError(f"geometry must be one of {list(_N_FACTORS)}, got '{geometry}'")

    n_factor = _N_FACTORS[geometry]

    # Assign tissue properties at each depth (includes susceptibility_ppm)
    layer_props = _assign_layers(depths_mm, tissue_layers)

    chi = np.array([p["susceptibility_ppm"] for p in layer_props])
    labels: list[str] = [p["name"] for p in layer_props]

    # Susceptibility difference from reference (ppm → dimensionless SI × 1e-6)
    delta_chi_ppm = chi - reference_chi_ppm
    delta_chi_si = delta_chi_ppm * 1e-6

    # ΔB₀ perturbation (T): ΔB₀ = N_eff × Δχ × B₀
    delta_b0 = n_factor * delta_chi_si * b0_profile

    # Larmor frequency shift (Hz): Δf = γ_p ΔB₀ / 2π
    delta_freq_hz = (_GAMMA_P / (2 * math.pi)) * delta_b0

    # Intravoxel dephasing: spins across a voxel accumulate phase spread
    # Estimate dephasing factor from inter-depth gradient of ΔB₀
    dephasing = _compute_dephasing_factor(delta_b0, depths_mm)

    return SusceptibilityProfile(
        depths_mm=depths_mm,
        chi_ppm=chi,
        delta_chi_ppm=delta_chi_ppm,
        delta_b0_tesla=delta_b0,
        delta_freq_hz=delta_freq_hz,
        dephasing_factor=dephasing,
        tissue_labels=labels,
        b0_tesla=b0_profile,
        reference_chi_ppm=reference_chi_ppm,
        geometry=geometry,
    )


def compute_frequency_shift(
    susceptibility_profile: SusceptibilityProfile,
) -> NDArray[np.float64]:
    """Return Larmor frequency shift (Hz) at each depth.

    Convenience accessor — identical to ``susceptibility_profile.delta_freq_hz``.
    """
    return susceptibility_profile.delta_freq_hz


def compute_dephasing_signal_loss(
    delta_b0: NDArray[np.float64],
    te_s: float,
    voxel_size_mm: float,
    depth_step_mm: float,
) -> NDArray[np.float64]:
    """Compute per-voxel signal attenuation from intravoxel B₀ dephasing.

    Uses the sinc-function model for a linear B₀ gradient across a voxel:

        attenuation = |sinc(Δφ / 2π)|

    where Δφ = γ_p × |dΔB₀/dz| × voxel_size × TE is the total phase spread.

    Args:
        delta_b0: ΔB₀ array (T) at each depth.
        te_s: echo time (s).
        voxel_size_mm: voxel thickness in the depth direction (mm).
        depth_step_mm: depth grid spacing (mm) used to estimate gradients.

    Returns:
        Array of signal attenuation factors ∈ [0, 1].
    """
    voxel_size_m = voxel_size_mm * 1e-3
    depth_step_m = depth_step_mm * 1e-3

    # Finite-difference gradient of ΔB₀ along depth
    gradient = np.gradient(delta_b0, depth_step_m)  # T/m

    # Phase spread across a voxel
    delta_phi = _GAMMA_P * np.abs(gradient) * voxel_size_m * te_s  # radians

    # sinc attenuation: |sinc(x)| = |sin(πx)| / (πx), argument = Δφ/(2π)
    x = delta_phi / (2 * math.pi)
    attenuation = np.where(x < 1e-10, np.ones_like(x), np.abs(np.sinc(x)))
    return attenuation


def apply_susceptibility_correction(
    signal: NDArray[np.float64],
    susceptibility_profile: SusceptibilityProfile,
    te_s: float,
    voxel_size_mm: float,
    depth_step_mm: float,
) -> SusceptibilityCorrectedProfile:
    """Apply susceptibility-induced dephasing correction to a depth profile signal.

    Args:
        signal: raw depth profile signal array (V).
        susceptibility_profile: computed susceptibility profile.
        te_s: echo time used in the acquisition (s).
        voxel_size_mm: voxel size in depth direction (mm).
        depth_step_mm: depth grid spacing (mm).

    Returns:
        SusceptibilityCorrectedProfile with original and corrected signal arrays.
    """
    correction = compute_dephasing_signal_loss(
        susceptibility_profile.delta_b0_tesla,
        te_s=te_s,
        voxel_size_mm=voxel_size_mm,
        depth_step_mm=depth_step_mm,
    )
    corrected = signal * correction

    return SusceptibilityCorrectedProfile(
        depths_mm=susceptibility_profile.depths_mm,
        signal_original=signal,
        signal_corrected=corrected,
        correction_factor=correction,
        susceptibility=susceptibility_profile,
        tissue_labels=susceptibility_profile.tissue_labels,
    )


def estimate_susceptibility_impact(
    tissue_layers: list[TissueLayer],
    b0_mean_t: float,
    reference_chi_ppm: float = -9.05,
    geometry: str = "sphere",
) -> dict[str, float]:
    """Quick summary of the susceptibility impact across a tissue stack.

    Returns a dict with:
    - ``max_delta_chi_ppm``: largest |Δχ| in the stack
    - ``max_delta_b0_ut``: peak ΔB₀ perturbation (μT)
    - ``max_delta_freq_hz``: peak Larmor frequency shift (Hz)
    - ``boundary_count``: number of distinct tissue interfaces

    Args:
        tissue_layers: tissue stack.
        b0_mean_t: representative B₀ field (T) for the estimate.
        reference_chi_ppm: reference susceptibility (ppm, default water).
        geometry: demagnetisation model ('sphere' or 'slab').
    """
    n_factor = _N_FACTORS[geometry]
    chi_values = [layer.susceptibility_ppm for layer in tissue_layers]
    delta_chi = [abs(c - reference_chi_ppm) for c in chi_values]
    max_delta_chi = max(delta_chi) if delta_chi else 0.0
    max_delta_b0_t = n_factor * max_delta_chi * 1e-6 * b0_mean_t
    max_delta_freq = (_GAMMA_P / (2 * math.pi)) * max_delta_b0_t

    # Count distinct boundaries (consecutive layers with different chi)
    boundaries = sum(
        1 for i in range(1, len(tissue_layers))
        if abs(tissue_layers[i].susceptibility_ppm - tissue_layers[i - 1].susceptibility_ppm) > 0.01
    )

    return {
        "max_delta_chi_ppm": max_delta_chi,
        "max_delta_b0_ut": max_delta_b0_t * 1e6,
        "max_delta_freq_hz": max_delta_freq,
        "boundary_count": float(boundaries),
    }


def cross_validate_susceptibility(
    profile_without: NDArray[np.float64],
    profile_with: NDArray[np.float64],
) -> dict[str, float]:
    """Compare depth profiles with and without susceptibility correction.

    Returns:
        Dict with:
        - ``max_relative_change``: peak |ΔS/S| across depth
        - ``mean_relative_change``: mean |ΔS/S|
        - ``correlation``: Pearson correlation between the two profiles
        - ``rms_fractional_diff``: RMS of (S_with - S_without) / S_without
    """
    if profile_without.shape != profile_with.shape:
        raise ValueError("Profile arrays must have the same shape.")

    denom = np.where(np.abs(profile_without) > 1e-30, profile_without, np.nan)
    rel_diff = np.abs(profile_with - profile_without) / np.abs(denom)

    max_rel = float(np.nanmax(rel_diff))
    mean_rel = float(np.nanmean(rel_diff))
    rms_frac = float(np.sqrt(np.nanmean(rel_diff**2)))

    # Pearson correlation on valid (non-nan, non-zero) points
    mask = np.isfinite(rel_diff)
    if mask.sum() >= 2:
        corr = float(np.corrcoef(profile_without[mask], profile_with[mask])[0, 1])
    else:
        corr = float("nan")

    return {
        "max_relative_change": max_rel,
        "mean_relative_change": mean_rel,
        "correlation": corr,
        "rms_fractional_diff": rms_frac,
    }


# ── Internal helpers ──────────────────────────────────────────────

def _compute_dephasing_factor(
    delta_b0: NDArray[np.float64],
    depths_mm: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Estimate signal dephasing factor from gradient of ΔB₀.

    Uses np.gradient for central differences.  At uniform regions (gradient≈0)
    the factor is 1.0 (no dephasing).  At sharp boundaries the factor is < 1.
    """
    depth_step_m = np.mean(np.diff(depths_mm)) * 1e-3 if len(depths_mm) > 1 else 1e-3
    gradient = np.gradient(delta_b0, depth_step_m)  # T/m

    # Use a nominal TE = 10 ms and voxel size = 3 mm for the default dephasing estimate
    _TE_DEFAULT = 0.010   # s
    _VOXEL_DEFAULT = 3e-3  # m

    delta_phi = _GAMMA_P * np.abs(gradient) * _VOXEL_DEFAULT * _TE_DEFAULT
    x = delta_phi / (2 * math.pi)
    return np.where(x < 1e-10, np.ones_like(x), np.abs(np.sinc(x)))
