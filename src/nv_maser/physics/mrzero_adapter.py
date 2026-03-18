"""MRzero-Core adapter for Bloch-equation validation of depth profiles.

Bridges our analytical 1D NMR depth profiling model with MRzero-Core's
isochromat Bloch simulation to provide ground-truth validation.

Validation approach
───────────────────
Our analytical model computes EMF at each depth as:

    ε(z) ∝ ω₀(z) × M₀(z) × V × sens(z) × [1 − exp(−TR/T₁)] × exp(−TE/T₂)

The Bloch simulation evolves spin magnetisation vectors through the full
Rodrigues rotation (flip → relax → dephase → precess) pipeline, producing
a signal that implicitly captures T₁ saturation and T₂ decay.

By comparing normalised signal profiles from both models, we validate that
the analytical contrast formulas (T₁ weighting, T₂ decay) are correct at
every depth.  Factors like ω₀, coil sensitivity, and voxel volume are
geometric/electromagnetic and cancel in the normalised comparison.

Dependencies
────────────
Requires ``MRzeroCore >= 0.4.6`` and ``torch >= 1.12`` (optional;
graceful degradation if absent).

References
──────────
Endres et al., "MRzero — Automated discovery of MRI sequences using
supervised learning", Magn. Reson. Med. 2024.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .depth_profile import TissueLayer, DepthProfile, _assign_layers

if TYPE_CHECKING:
    from ..config import DepthProfileConfig

try:
    import torch
    import MRzeroCore as mr0

    MRZERO_AVAILABLE = True
except ImportError:  # pragma: no cover
    MRZERO_AVAILABLE = False


# ── Result data classes ───────────────────────────────────────────


@dataclass(frozen=True)
class BlochSignal:
    """Bloch-simulated NMR signal at each depth point."""

    depths_mm: NDArray[np.float64]
    signal_magnitude: NDArray[np.float64]
    tissue_labels: list[str]
    te_s: float
    tr_s: float
    spin_count: int


@dataclass(frozen=True)
class DepthValidation:
    """Cross-validation between analytical and Bloch depth profiles."""

    depths_mm: NDArray[np.float64]
    analytical_normalized: NDArray[np.float64]
    bloch_normalized: NDArray[np.float64]
    correlation: float
    max_relative_error: float
    mean_relative_error: float


# ── Internal helpers ──────────────────────────────────────────────


def _require_mrzero() -> None:
    if not MRZERO_AVAILABLE:
        raise ImportError(
            "MRzeroCore is not installed.  pip install MRzeroCore"
        )


# ── Phantom builders ─────────────────────────────────────────────


def build_single_voxel_phantom(
    t1_s: float,
    t2_s: float,
    proton_density: float,
    b0_hz: float = 0.0,
    t2dash_s: float = 0.1,
) -> "mr0.CustomVoxelPhantom":
    """Build a :class:`CustomVoxelPhantom` containing one voxel at the origin.

    Parameters
    ----------
    t1_s : float
        T₁ relaxation time (seconds).
    t2_s : float
        T₂ relaxation time (seconds).
    proton_density : float
        Relative proton density (water = 1.0).
    b0_hz : float
        Off-resonance frequency offset (Hz).
    t2dash_s : float
        T₂' inhomogeneous dephasing time (seconds).

    Returns
    -------
    mr0.CustomVoxelPhantom
    """
    _require_mrzero()
    return mr0.CustomVoxelPhantom(
        pos=[[0.0, 0.0, 0.0]],
        PD=proton_density,
        T1=t1_s,
        T2=t2_s,
        T2dash=t2dash_s,
        B0=b0_hz,
        B1=1.0,
        voxel_size=0.001,  # 1 mm
        voxel_shape="sinc",
    )


# ── Sequence builders ────────────────────────────────────────────


def build_spin_echo_sequence(
    te_s: float,
    tr_s: float,
    n_repetitions: int = 1,
) -> "mr0.Sequence":
    """Build a spin-echo sequence with ``n_repetitions`` TR periods.

    Each TR consists of two MRzero :class:`Repetition` objects:

    1. Excitation (90°) → wait TE/2
    2. Refocusing (180°) → wait TE/2 (ADC) → dead-time (TR − TE)

    Parameters
    ----------
    te_s : float
        Echo time (seconds).
    tr_s : float
        Repetition time (seconds).
    n_repetitions : int
        Number of TR periods (use ≥5 for approximate steady state).

    Returns
    -------
    mr0.Sequence
    """
    _require_mrzero()
    seq = mr0.Sequence()

    half_te = te_s / 2
    dead_time = max(tr_s - te_s, 1e-6)

    for _ in range(n_repetitions):
        # ── Excitation: 90° ─────────────────────────────────────
        excite = mr0.Repetition(
            pulse=mr0.Pulse(
                mr0.PulseUsage.EXCIT,
                torch.tensor(math.pi / 2),
                torch.tensor(0.0),
                torch.tensor([[1.0, 0.0]]),
                False,
            ),
            event_time=torch.tensor([half_te]),
            gradm=torch.zeros(1, 3),
            adc_phase=torch.zeros(1),
            adc_usage=torch.zeros(1, dtype=torch.int32),
        )
        seq.append(excite)

        # ── Refocusing: 180° → ADC → dead time ─────────────────
        refocus = mr0.Repetition(
            pulse=mr0.Pulse(
                mr0.PulseUsage.REFOC,
                torch.tensor(math.pi),
                torch.tensor(0.0),
                torch.tensor([[1.0, 0.0]]),
                False,
            ),
            event_time=torch.tensor([half_te, dead_time]),
            gradm=torch.zeros(2, 3),
            adc_phase=torch.zeros(2),
            adc_usage=torch.tensor([1, 0], dtype=torch.int32),
        )
        seq.append(refocus)

    return seq


# ── Single-voxel Bloch simulation ────────────────────────────────


def simulate_single_voxel_bloch(
    t1_s: float,
    t2_s: float,
    proton_density: float,
    te_s: float,
    tr_s: float,
    spin_count: int = 100,
    n_repetitions: int = 10,
) -> float:
    """Bloch-simulate a single voxel and return spin-echo signal magnitude.

    Runs ``n_repetitions`` TR periods so the magnetisation approaches
    steady state, then returns the last echo's absolute signal.

    Parameters
    ----------
    t1_s, t2_s, proton_density : float
        Tissue properties (SI units: seconds, relative PD).
    te_s, tr_s : float
        Echo time and repetition time (seconds).
    spin_count : int
        Isochromats per voxel (more → smoother but slower).
    n_repetitions : int
        TR periods before reading (≥5 recommended).

    Returns
    -------
    float
        Absolute signal magnitude of the last echo.
    """
    _require_mrzero()

    phantom = build_single_voxel_phantom(t1_s, t2_s, proton_density)
    data = phantom.build()
    seq = build_spin_echo_sequence(te_s, tr_s, n_repetitions)

    signal = mr0.isochromat_sim(
        seq, data, spin_count, print_progress=False
    )
    # signal shape: (n_repetitions, coil_count=1)
    # Last echo = steady state
    return float(signal[-1].abs().item())


# ── Analytical contrast (for direct comparison with Bloch) ────────


def compute_analytical_contrast(
    tissue_layers: list[TissueLayer],
    max_depth_mm: float,
    depth_resolution_mm: float,
    te_s: float,
    tr_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Compute the pure T₁/T₂ contrast curve for the tissue stack.

    This matches the physics that MRzero's Bloch sim captures (PD × T₁
    saturation × T₂ decay) without the geometric/electromagnetic factors
    (ω₀, coil sensitivity, voxel volume) present in the full
    :func:`simulate_depth_profile`.

    Returns
    -------
    tuple of (depths_mm, contrast, labels)
        ``contrast[i] = PD × [1 − exp(−TR/T₁)] × exp(−TE/T₂)`` at each depth.
    """
    depths = np.arange(
        depth_resolution_mm / 2,
        max_depth_mm,
        depth_resolution_mm,
    )
    layer_props = _assign_layers(depths, tissue_layers)

    contrast = np.zeros(len(depths))
    labels: list[str] = []

    for i, props in enumerate(layer_props):
        labels.append(props["name"])
        pd = props["proton_density"]
        t1 = props["t1_ms"] * 1e-3
        t2 = props["t2_ms"] * 1e-3

        sat = 1.0 - math.exp(-tr_s / t1) if t1 > 0 else 1.0
        decay = math.exp(-te_s / t2) if t2 > 0 else 0.0
        contrast[i] = pd * sat * decay

    return depths, contrast, labels


# ── Full depth-profile Bloch simulation ──────────────────────────


def simulate_depth_bloch(
    tissue_layers: list[TissueLayer],
    max_depth_mm: float = 30.0,
    depth_resolution_mm: float = 0.5,
    te_ms: float = 10.0,
    tr_ms: float = 100.0,
    spin_count: int = 100,
    n_repetitions: int = 10,
) -> BlochSignal:
    """Bloch-simulate the NMR signal at each depth point.

    For each depth slice, a single-voxel Bloch simulation is run with the
    tissue parameters (T₁, T₂, PD) of the layer at that depth.

    Parameters
    ----------
    tissue_layers : list of TissueLayer
        Stacked tissue model from surface downward.
    max_depth_mm : float
        Maximum simulation depth (mm).
    depth_resolution_mm : float
        Depth step size (mm).
    te_ms : float
        Echo time (ms).
    tr_ms : float
        Repetition time (ms).
    spin_count : int
        Isochromats per voxel.
    n_repetitions : int
        TR periods for steady-state approach.

    Returns
    -------
    BlochSignal
    """
    _require_mrzero()

    depths = np.arange(
        depth_resolution_mm / 2,
        max_depth_mm,
        depth_resolution_mm,
    )
    layer_props = _assign_layers(depths, tissue_layers)

    te_s = te_ms * 1e-3
    tr_s = tr_ms * 1e-3

    signals = np.zeros(len(depths))
    labels: list[str] = []

    for i, props in enumerate(layer_props):
        labels.append(props["name"])
        pd = props["proton_density"]
        t1 = props["t1_ms"] * 1e-3
        t2 = props["t2_ms"] * 1e-3

        if pd < 1e-6 or t2 < 1e-6:
            signals[i] = 0.0
            continue

        signals[i] = simulate_single_voxel_bloch(
            t1, t2, pd, te_s, tr_s, spin_count, n_repetitions,
        )

    return BlochSignal(
        depths_mm=depths,
        signal_magnitude=signals,
        tissue_labels=labels,
        te_s=te_s,
        tr_s=tr_s,
        spin_count=spin_count,
    )


# ── Cross-validation ─────────────────────────────────────────────


def cross_validate_depth(
    analytical: DepthProfile,
    bloch: BlochSignal,
) -> DepthValidation:
    """Compare analytical depth profile with Bloch simulation.

    Both profiles are normalised to unit maximum before computing the
    Pearson correlation and relative error statistics.

    Parameters
    ----------
    analytical : DepthProfile
        From :func:`simulate_depth_profile`.
    bloch : BlochSignal
        From :func:`simulate_depth_bloch`.

    Returns
    -------
    DepthValidation
    """
    if len(analytical.depths_mm) != len(bloch.depths_mm):
        raise ValueError(
            f"Depth grid mismatch: analytical has {len(analytical.depths_mm)} "
            f"points, Bloch has {len(bloch.depths_mm)}"
        )
    if not np.allclose(analytical.depths_mm, bloch.depths_mm, atol=0.01):
        raise ValueError("Depth positions differ between analytical and Bloch")

    a_sig = np.abs(analytical.signal)
    b_sig = np.abs(bloch.signal_magnitude)

    a_max = np.max(a_sig)
    b_max = np.max(b_sig)

    a_norm = a_sig / a_max if a_max > 0 else a_sig
    b_norm = b_sig / b_max if b_max > 0 else b_sig

    # Pearson correlation
    corr = float(np.corrcoef(a_norm, b_norm)[0, 1])

    # Relative errors where both signals are significant
    mask = (a_norm > 0.01) & (b_norm > 0.01)
    if mask.any():
        denom = np.maximum(a_norm[mask], b_norm[mask])
        rel_err = np.abs(a_norm[mask] - b_norm[mask]) / denom
        max_rel = float(np.max(rel_err))
        mean_rel = float(np.mean(rel_err))
    else:
        max_rel = 0.0
        mean_rel = 0.0

    return DepthValidation(
        depths_mm=analytical.depths_mm,
        analytical_normalized=a_norm,
        bloch_normalized=b_norm,
        correlation=corr,
        max_relative_error=max_rel,
        mean_relative_error=mean_rel,
    )


def cross_validate_contrast(
    tissue_layers: list[TissueLayer],
    max_depth_mm: float = 30.0,
    depth_resolution_mm: float = 0.5,
    te_ms: float = 10.0,
    tr_ms: float = 100.0,
    spin_count: int = 100,
    n_repetitions: int = 10,
) -> DepthValidation:
    """Compare pure analytical contrast with Bloch sim (no ω₀/coil factors).

    This is the purest validation: both sides compute only
    PD × T₁-weighting × T₂-decay, so differences reveal genuine
    Bloch vs. formula discrepancies.

    Parameters
    ----------
    tissue_layers : list of TissueLayer
        Stacked tissue model.
    max_depth_mm, depth_resolution_mm : float
        Depth grid parameters (mm).
    te_ms, tr_ms : float
        Sequence timing (ms).
    spin_count : int
        Isochromats per voxel.
    n_repetitions : int
        TR periods for steady-state approach.

    Returns
    -------
    DepthValidation
    """
    _require_mrzero()

    te_s = te_ms * 1e-3
    tr_s = tr_ms * 1e-3

    depths, contrast, labels = compute_analytical_contrast(
        tissue_layers, max_depth_mm, depth_resolution_mm, te_s, tr_s,
    )

    bloch = simulate_depth_bloch(
        tissue_layers,
        max_depth_mm=max_depth_mm,
        depth_resolution_mm=depth_resolution_mm,
        te_ms=te_ms,
        tr_ms=tr_ms,
        spin_count=spin_count,
        n_repetitions=n_repetitions,
    )

    a_max = np.max(np.abs(contrast))
    b_max = np.max(bloch.signal_magnitude)

    a_norm = np.abs(contrast) / a_max if a_max > 0 else contrast
    b_norm = bloch.signal_magnitude / b_max if b_max > 0 else bloch.signal_magnitude

    corr = float(np.corrcoef(a_norm, b_norm)[0, 1])

    mask = (a_norm > 0.01) & (b_norm > 0.01)
    if mask.any():
        denom = np.maximum(a_norm[mask], b_norm[mask])
        rel_err = np.abs(a_norm[mask] - b_norm[mask]) / denom
        max_rel = float(np.max(rel_err))
        mean_rel = float(np.mean(rel_err))
    else:
        max_rel = 0.0
        mean_rel = 0.0

    return DepthValidation(
        depths_mm=depths,
        analytical_normalized=a_norm,
        bloch_normalized=b_norm,
        correlation=corr,
        max_relative_error=max_rel,
        mean_relative_error=mean_rel,
    )
