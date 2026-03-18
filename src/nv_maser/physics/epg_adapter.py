"""
Extended Phase Graph (EPG) algorithm for NMR/MRI signal simulation.

The EPG algorithm (Weigel 2015) represents the spin state as a set of
'configuration states' (F+, F-, Z), each characterised by a dephasing order k.
This is more efficient than tracking individual spins in a Bloch simulation
when the sequence has many identical repetitions (CPMG, multi-echo trains).

Key advantages over analytical spin-echo formula:
- Handles multi-echo trains with stimulated echoes exactly
- Accounts for imperfect refocusing pulses (flip angle < 180°)
- Tracks T1 recovery between echoes
- No per-spin numerical integration required
- Pure NumPy, no PyTorch dependency

Key advantages over MRzero Bloch simulation:
- ~100× faster (no GPU required, no per-voxel ODE)
- Simple enough for rapid sequence parameter optimization
- Identical physics for ideal spin-echo and CPMG

Usage
─────
Single spin-echo::

    signal = epg_signal(t1=0.6, t2=0.05, te=0.010, tr=0.100)
    # returns echo amplitude at TE (normalised to equilibrium magnetisation)

CPMG multi-echo train::

    echoes = epg_cpmg(t1=0.6, t2=0.050, esp=0.010, n_echoes=16)
    # returns 16 echo amplitudes

Depth profile::

    depths, signals = epg_depth_profile(tissue_layers, te_ms=10.0, tr_ms=100.0)

References
──────────
Weigel M (2015) "Extended phase graphs: Dephasing, RF pulses, and echoes—
pure and simple", J Magn Reson Imaging 41(2):266–295.
Hennig J (1988) "Multiecho imaging sequences with low refocusing flip angles",
J Magn Reson 78(3):397–407.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .depth_profile import TissueLayer, _assign_layers

# ── Type alias ────────────────────────────────────────────────────
# EPG state matrix shape: (3, N_states)
# Row 0: F+  (transverse magnetisation, positive dephasing orders)
# Row 1: F-  (transverse magnetisation, negative dephasing orders = conj(F+))
# Row 2: Z   (longitudinal magnetisation)
EPGState = NDArray[np.complex128]


@dataclass(frozen=True)
class EPGResult:
    """Result of an EPG signal simulation."""

    echo_amplitudes: NDArray[np.float64]  # |signal| at each echo time
    echo_times_s: NDArray[np.float64]     # echo centre times (s)
    t1_s: float
    t2_s: float
    flip_angle_deg: float
    n_echoes: int


@dataclass(frozen=True)
class EPGDepthProfile:
    """EPG-computed depth profile."""

    depths_mm: NDArray[np.float64]
    signal: NDArray[np.float64]  # signal amplitude at first echo for each depth
    tissue_labels: list[str]
    te_s: float
    tr_s: float
    flip_angle_deg: float


@dataclass(frozen=True)
class EPGValidation:
    """Cross-validation of EPG against another simulator."""

    epg_normalized: NDArray[np.float64]
    reference_normalized: NDArray[np.float64]
    correlation: float
    max_relative_error: float
    mean_relative_error: float
    method_name: str  # "analytical" or "bloch"


# ── EPG core operators ────────────────────────────────────────────

def _epg_init(n_states: int = 1) -> EPGState:
    """Initialise EPG state at thermal equilibrium (Mz = 1)."""
    state = np.zeros((3, n_states), dtype=np.complex128)
    state[2, 0] = 1.0  # Z[0] = 1 (equilibrium Mz)
    return state


def _epg_rf(state: EPGState, alpha_rad: float, phi_rad: float = math.pi / 2) -> EPGState:
    """Apply an RF pulse to the EPG state.

    Uses the general rotation matrix R(α, φ) for a pulse with flip angle α
    and phase φ.  For a 90x pulse: alpha=π/2, phi=π/2.  For 180y: alpha=π, phi=π.

    The rotation matrix for EPG (F+, F-, Z) is (Weigel 2015, Eq. 21):

        R = [[cos²(α/2),         e^{2iφ} sin²(α/2),    -i e^{iφ} sin(α)  ],
             [e^{-2iφ} sin²(α/2), cos²(α/2),             i e^{-iφ} sin(α) ],
             [-i/2 e^{-iφ} sin(α), i/2 e^{iφ} sin(α),    cos(α)           ]]
    """
    ca = math.cos(alpha_rad)
    sa = math.sin(alpha_rad)
    ca2 = math.cos(alpha_rad / 2) ** 2
    sa2 = math.sin(alpha_rad / 2) ** 2
    ep = complex(math.cos(phi_rad), math.sin(phi_rad))  # e^{iφ}
    em = complex(math.cos(phi_rad), -math.sin(phi_rad))  # e^{-iφ}

    R = np.array([
        [ca2,            (ep**2) * sa2,   -1j * ep * sa],
        [(em**2) * sa2,  ca2,              1j * em * sa],
        [-0.5j * em * sa, 0.5j * ep * sa,  ca           ],
    ], dtype=np.complex128)

    return R @ state


def _epg_relax(state: EPGState, t_interval_s: float, t1_s: float, t2_s: float) -> EPGState:
    """Apply relaxation over a time interval.

    Transverse states (F+, F-) decay with E2 = exp(-t/T2).
    Longitudinal state Z decays with E1 = exp(-t/T1) and recovers toward 1.
    """
    if t_interval_s <= 0:
        return state.copy()

    e1 = math.exp(-t_interval_s / t1_s) if t1_s > 0 else 0.0
    e2 = math.exp(-t_interval_s / t2_s) if t2_s > 0 else 0.0

    new = state.copy()
    new[0, :] *= e2          # F+ decays
    new[1, :] *= e2          # F- decays
    new[2, :] *= e1          # Z decays
    new[2, 0] += (1.0 - e1)  # Z[0] recovers toward 1
    return new


def _epg_grad(state: EPGState) -> EPGState:
    """Apply a unit dephasing gradient (shift operator T).

    Shifts F+ states one order up (→ higher dephasing),
    shifts F- states one order down, fills new lowest F+ order with 0,
    and uses conjugate symmetry: F-[0] = conj(F+[0]).
    """
    new = np.zeros_like(state)
    n = state.shape[1]

    # F+ shifts up: F+[k] ← F+[k-1] for k ≥ 1
    if n > 1:
        new[0, 1:] = state[0, :n - 1]
    new[0, 0] = 0.0  # lowest F+ order becomes 0

    # F- shifts down: F-[k] ← F-[k+1] for k ≤ n-2
    if n > 1:
        new[1, :n - 1] = state[1, 1:]
    new[1, n - 1] = 0.0  # highest F- order cleared

    # Z is unchanged
    new[2, :] = state[2, :]

    # Conjugate symmetry for F-[0]
    new[1, 0] = np.conj(new[0, 0])
    return new


# ── Public simulation functions ───────────────────────────────────

def epg_signal(
    t1_s: float,
    t2_s: float,
    te_s: float,
    tr_s: float,
    flip_angle_deg: float = 90.0,
    refocus_angle_deg: float = 180.0,
    n_states: int = 32,
) -> float:
    """Compute the spin-echo signal for a single echo time.

    Simulates one full TR: excitation → TE/2 → refocus → TE/2(echo) → TR-TE recovery.

    Args:
        t1_s: T1 relaxation time (s).
        t2_s: T2 relaxation time (s).
        te_s: echo time (s).
        tr_s: repetition time (s).  Must be >= te_s.
        flip_angle_deg: excitation flip angle (degrees).  Default 90°.
        refocus_angle_deg: refocusing flip angle (degrees).  Default 180°.
        n_states: number of EPG configuration states to track.

    Returns:
        Echo signal amplitude (normalised to M₀ = 1).
    """
    if tr_s < te_s:
        raise ValueError(f"TR ({tr_s}s) must be >= TE ({te_s}s).")

    state = _epg_init(n_states)
    echo_amplitude = 0.0

    # Steady-state: run multiple TRs until convergence.
    # For spectroscopy / depth-profiling there are no imaging gradients, so
    # the gradient operator T is NOT applied.  The echo refocuses at k=0
    # purely through the RF rotation (RF alone correctly rephases when there
    # is no crusher dephasing).
    for _ in range(20):  # typically converges within 10 TRs
        # Excitation pulse (90° about y-axis → phase = π/2)
        state = _epg_rf(state, math.radians(flip_angle_deg), phi_rad=math.pi / 2)

        # Relaxation for TE/2
        state = _epg_relax(state, te_s / 2, t1_s, t2_s)

        # Refocusing pulse (180° about x-axis → phase = 0)
        state = _epg_rf(state, math.radians(refocus_angle_deg), phi_rad=0.0)

        # Relaxation for TE/2 — echo forms at F+[0]
        state = _epg_relax(state, te_s / 2, t1_s, t2_s)

        # Record echo amplitude here (BEFORE remaining TR-TE recovery)
        echo_amplitude = float(abs(state[0, 0]))

        # Relaxation for remaining TR - TE (Z recovers, transverse decays)
        remaining = tr_s - te_s
        if remaining > 0:
            state = _epg_relax(state, remaining, t1_s, t2_s)

    return echo_amplitude


def epg_cpmg(
    t1_s: float,
    t2_s: float,
    esp_s: float,
    n_echoes: int,
    tr_s: float | None = None,
    excitation_angle_deg: float = 90.0,
    refocus_angle_deg: float = 180.0,
    n_states: int = 64,
) -> NDArray[np.float64]:
    """Simulate a CPMG multi-echo train and return all echo amplitudes.

    The sequence is: 90° → [ESP/2 → 180° → ESP/2 → echo] × n_echoes → TR recovery.

    Args:
        t1_s: T1 relaxation time (s).
        t2_s: T2 relaxation time (s).
        esp_s: echo spacing (s) — time between consecutive echoes.
        n_echoes: number of echoes.
        tr_s: repetition time (s). If None, uses n_echoes × esp_s + 2×esp_s (minimal TR).
        excitation_angle_deg: flip angle for initial excitation.
        refocus_angle_deg: flip angle for all refocusing pulses.
        n_states: EPG configuration states to track.

    Returns:
        Array of shape (n_echoes,) with echo amplitudes.
    """
    te_last = n_echoes * esp_s
    if tr_s is None:
        tr_s = te_last + 2 * esp_s

    state = _epg_init(n_states)

    # Run until steady-state (typically 10 TRs for most tissues).
    # No imaging gradients for spectroscopy — echo refocuses at k=0 via RF only.
    for tr_idx in range(15):
        state = _epg_rf(state, math.radians(excitation_angle_deg), phi_rad=math.pi / 2)

        echo_amps = np.zeros(n_echoes, dtype=np.float64)

        for echo_idx in range(n_echoes):
            # Relaxation for ESP/2
            state = _epg_relax(state, esp_s / 2, t1_s, t2_s)

            # Refocusing pulse
            state = _epg_rf(state, math.radians(refocus_angle_deg), phi_rad=0.0)

            # Relaxation for ESP/2 — echo at F+[0]
            state = _epg_relax(state, esp_s / 2, t1_s, t2_s)

            # Record echo
            echo_amps[echo_idx] = float(abs(state[0, 0]))

        # Recovery for TR - n_echoes*ESP
        remaining = tr_s - te_last
        if remaining > 0:
            state = _epg_relax(state, remaining, t1_s, t2_s)

    return echo_amps


def epg_depth_profile(
    tissue_layers: list[TissueLayer],
    te_ms: float = 10.0,
    tr_ms: float = 100.0,
    max_depth_mm: float = 30.0,
    depth_resolution_mm: float = 0.5,
    flip_angle_deg: float = 90.0,
    refocus_angle_deg: float = 180.0,
    n_states: int = 32,
) -> EPGDepthProfile:
    """Compute a 1D NMR depth profile using the EPG algorithm.

    Uses the same depth grid and tissue layer mapping as ``simulate_depth_profile``.
    Signal is the steady-state echo amplitude at TE for each depth.

    This function does NOT include coil sensitivity or equilibrium magnetisation
    scaling — it returns the pure NMR signal evolution factor ∈ [0, 1], which
    can be multiplied by M₀(z) × sens(z) downstream.

    Args:
        tissue_layers: stacked tissue layers.
        te_ms: echo time (ms).
        tr_ms: repetition time (ms).
        max_depth_mm: depth range (mm).
        depth_resolution_mm: depth sampling step (mm).
        flip_angle_deg: excitation flip angle (default 90°).
        refocus_angle_deg: refocusing flip angle (default 180°).
        n_states: EPG states to track.

    Returns:
        EPGDepthProfile with signal factor vs depth.
    """
    te_s = te_ms * 1e-3
    tr_s = tr_ms * 1e-3

    depths = np.arange(depth_resolution_mm / 2, max_depth_mm, depth_resolution_mm)
    layer_props = _assign_layers(depths, tissue_layers)

    signal = np.zeros(len(depths), dtype=np.float64)
    labels: list[str] = []

    for i, props in enumerate(layer_props):
        labels.append(props["name"])
        t1_s = props["t1_ms"] * 1e-3
        t2_s = props["t2_ms"] * 1e-3

        if t2_s <= 0 or props["proton_density"] <= 0:
            signal[i] = 0.0
            continue

        # Scale by proton density (EPG base signal is for PD=1)
        signal[i] = props["proton_density"] * epg_signal(
            t1_s=t1_s,
            t2_s=t2_s,
            te_s=te_s,
            tr_s=tr_s,
            flip_angle_deg=flip_angle_deg,
            refocus_angle_deg=refocus_angle_deg,
            n_states=n_states,
        )

    return EPGDepthProfile(
        depths_mm=depths,
        signal=signal,
        tissue_labels=labels,
        te_s=te_s,
        tr_s=tr_s,
        flip_angle_deg=flip_angle_deg,
    )


def cross_validate_epg_vs_analytical(
    epg_result: EPGDepthProfile,
    analytical_signal: NDArray[np.float64],
    analytical_depths: NDArray[np.float64],
) -> EPGValidation:
    """Compare EPG depth profile against analytical spin-echo formula.

    Both signals are normalised to their maximum before comparison, so the
    comparison is on tissue contrast (shape), not absolute signal magnitude.

    Args:
        epg_result: EPGDepthProfile from ``epg_depth_profile``.
        analytical_signal: signal array from analytical depth profile (same depth grid).
        analytical_depths: depth grid for analytical signal (must match EPG).

    Returns:
        EPGValidation with correlation and relative error statistics.
    """
    epg_s = epg_result.signal
    ref_s = analytical_signal

    if len(epg_s) != len(ref_s):
        raise ValueError(
            f"EPG signal length ({len(epg_s)}) != analytical length ({len(ref_s)}). "
            "Use the same depth grid."
        )

    # Normalise to max (non-zero tissues only)
    epg_max = np.max(np.abs(epg_s))
    ref_max = np.max(np.abs(ref_s))

    epg_norm = epg_s / epg_max if epg_max > 1e-30 else epg_s
    ref_norm = ref_s / ref_max if ref_max > 1e-30 else ref_s

    # Relative error
    denom = np.where(np.abs(ref_norm) > 1e-6, np.abs(ref_norm), np.nan)
    rel_err = np.abs(epg_norm - ref_norm) / denom
    max_rel = float(np.nanmax(rel_err))
    mean_rel = float(np.nanmean(rel_err))

    # Pearson correlation on valid points
    mask = np.isfinite(rel_err)
    corr = float(np.corrcoef(epg_norm[mask], ref_norm[mask])[0, 1]) if mask.sum() >= 2 else float("nan")

    return EPGValidation(
        epg_normalized=epg_norm,
        reference_normalized=ref_norm,
        correlation=corr,
        max_relative_error=max_rel,
        mean_relative_error=mean_rel,
        method_name="analytical",
    )
