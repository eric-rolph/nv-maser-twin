"""SigPy adapter for 1D depth-profile reconstruction and RF pulse design.

Bridges our single-sided NMR depth profiling with SigPy's reconstruction
algorithms and RF pulse design tools.

Problem formulation
───────────────────
In a single-sided magnet the B₀ gradient is *non-uniform*: each depth z
maps to a unique Larmor frequency f(z) = γ·|B₀(z)|.  An FID or echo
acquisition S(t) in this gradient encodes spatial information, but the
non-uniform mapping means standard FFT is insufficient — we need either:

  1. NUFFT (non-uniform FFT) for the non-linear f↔z mapping, or
  2. An explicit encoding matrix E and iterative reconstruction.

This module provides both approaches plus compressed-sensing (L1-wavelet)
and total-variation regularised reconstruction for under-sampled or noisy
acquisitions, and basic RF pulse design for selective excitation in the
inhomogeneous sweet-spot field.

Dependencies
────────────
Requires ``sigpy >= 0.1.25`` (optional; graceful degradation if absent).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..config import DepthProfileConfig
    from .single_sided_magnet import SingleSidedMagnet

try:
    import sigpy as sp
    import sigpy.mri
    import sigpy.mri.rf

    SIGPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SIGPY_AVAILABLE = False

# ── Physical constants ────────────────────────────────────────────
_GAMMA_P_HZ_PER_T = 42.577e6  # proton gyromagnetic ratio (Hz/T)


# ── Custom SigPy linear operators ─────────────────────────────────

class _EncodingLinop(sp.linop.Linop if SIGPY_AVAILABLE else object):
    """Forward encoding: y = E @ x."""

    def __init__(self, E: NDArray[np.complex128]):
        self._E = E
        n_time, n_depth = E.shape
        super().__init__([n_time], [n_depth])

    def _apply(self, x: NDArray) -> NDArray:
        return self._E @ x

    def _adjoint_linop(self):
        return _EncodingAdjointLinop(self._E)


class _EncodingAdjointLinop(sp.linop.Linop if SIGPY_AVAILABLE else object):
    """Adjoint encoding: x = E^H @ y."""

    def __init__(self, E: NDArray[np.complex128]):
        self._E = E
        n_time, n_depth = E.shape
        super().__init__([n_depth], [n_time])

    def _apply(self, x: NDArray) -> NDArray:
        return self._E.conj().T @ x

    def _adjoint_linop(self):
        return _EncodingLinop(self._E)


# ── Data classes ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ReconResult:
    """Result of a 1D depth-profile reconstruction."""

    depths_mm: NDArray[np.float64]
    profile: NDArray[np.float64]  # reconstructed proton density (a.u.)
    residual_norm: float  # ‖E·x − y‖₂
    iterations: int
    method: str  # "l1_wavelet", "total_variation", "least_squares"


@dataclass(frozen=True)
class EncodingInfo:
    """Frequency-to-depth encoding geometry for a single-sided magnet."""

    depths_mm: NDArray[np.float64]
    b0_tesla: NDArray[np.float64]
    larmor_hz: NDArray[np.float64]
    gradient_t_per_m: NDArray[np.float64]  # dB₀/dz at each depth


@dataclass(frozen=True)
class RFPulseResult:
    """Result of an RF excitation pulse design."""

    pulse: NDArray[np.complex128]  # RF waveform (complex, a.u.)
    time_us: NDArray[np.float64]  # time axis (µs)
    duration_us: float
    bandwidth_hz: float
    time_bandwidth: float


# ── Encoding geometry ─────────────────────────────────────────────

def build_encoding_info(
    magnet: SingleSidedMagnet,
    config: DepthProfileConfig,
) -> EncodingInfo:
    """Compute the frequency-to-depth encoding from the magnet model.

    Args:
        magnet: single-sided magnet providing B₀(z).
        config: depth profile config (depth range, resolution).

    Returns:
        EncodingInfo with B₀, Larmor frequency, and gradient at each depth.
    """
    depths = np.arange(
        config.depth_resolution_mm / 2,
        config.max_depth_mm,
        config.depth_resolution_mm,
    )
    b0 = magnet.field_on_axis(depths)
    larmor_hz = np.abs(b0) * _GAMMA_P_HZ_PER_T

    # Gradient via central differences (T/m)
    dz_m = config.depth_resolution_mm * 1e-3
    grad = np.gradient(b0, dz_m)

    return EncodingInfo(
        depths_mm=depths,
        b0_tesla=b0,
        larmor_hz=larmor_hz,
        gradient_t_per_m=grad,
    )


def build_encoding_matrix(
    encoding: EncodingInfo,
    n_time_points: int,
    dwell_time_s: float,
) -> NDArray[np.complex128]:
    """Build the explicit encoding matrix E for non-uniform gradient.

    E[k, j] = exp(−i 2π f_j t_k)  where f_j = γ·B₀(z_j)

    The acquired signal is  y[k] = Σ_j E[k,j] · ρ[j]  + noise.

    Args:
        encoding: frequency-to-depth mapping.
        n_time_points: number of acquisition time points.
        dwell_time_s: time between samples (s).

    Returns:
        Complex encoding matrix of shape (n_time_points, n_depths).
    """
    t = np.arange(n_time_points) * dwell_time_s  # (K,)
    f = encoding.larmor_hz  # (N,)
    # Shift frequencies to baseband (centred on mean Larmor)
    f_shifted = f - np.mean(f)
    # E[k, j] = exp(-i 2π f_j t_k)
    return np.exp(-1j * 2 * math.pi * np.outer(t, f_shifted))


# ── Forward simulation ────────────────────────────────────────────

def simulate_signal(
    encoding: EncodingInfo,
    profile: NDArray[np.float64],
    n_time_points: int,
    dwell_time_s: float,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """Simulate the time-domain NMR signal for a given depth profile.

    Args:
        encoding: frequency-to-depth encoding.
        profile: proton density at each depth (length = n_depths).
        n_time_points: number of time samples.
        dwell_time_s: dwell time per sample (s).
        noise_std: standard deviation of complex Gaussian noise.
        rng: random generator for reproducibility.

    Returns:
        Complex signal vector of length n_time_points.
    """
    E = build_encoding_matrix(encoding, n_time_points, dwell_time_s)
    signal = E @ profile.astype(np.complex128)

    if noise_std > 0:
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.normal(0, noise_std, signal.shape) + 1j * rng.normal(
            0, noise_std, signal.shape
        )
        signal = signal + noise

    return signal


# ── Reconstruction ────────────────────────────────────────────────

def reconstruct_least_squares(
    signal: NDArray[np.complex128],
    encoding: EncodingInfo,
    dwell_time_s: float,
    max_iter: int = 100,
) -> ReconResult:
    """Reconstruct depth profile via conjugate-gradient least squares.

    Solves  min_x ‖E·x − y‖₂²  using SigPy's ConjugateGradient.

    Args:
        signal: acquired complex signal (n_time_points,).
        encoding: frequency-to-depth encoding.
        dwell_time_s: dwell time per sample (s).
        max_iter: maximum CG iterations.

    Returns:
        ReconResult with reconstructed profile.
    """
    if not SIGPY_AVAILABLE:
        raise RuntimeError("sigpy is required for reconstruction")

    n_time = len(signal)
    E = build_encoding_matrix(encoding, n_time, dwell_time_s)

    E_op = _EncodingLinop(E)
    app = sp.app.LinearLeastSquares(
        E_op, signal, max_iter=max_iter, show_pbar=False
    )
    x = app.run()

    residual = float(np.linalg.norm(E @ x - signal))
    return ReconResult(
        depths_mm=encoding.depths_mm.copy(),
        profile=np.abs(x),
        residual_norm=residual,
        iterations=max_iter,
        method="least_squares",
    )


def reconstruct_l1_wavelet(
    signal: NDArray[np.complex128],
    encoding: EncodingInfo,
    dwell_time_s: float,
    lamda: float = 0.01,
    max_iter: int = 200,
) -> ReconResult:
    """Compressed-sensing reconstruction with L1-wavelet regularisation.

    Solves  min_x ‖E·x − y‖₂² + λ‖Ψx‖₁
    where Ψ is a wavelet transform.

    Best for profiles with smooth regions and sharp transitions
    (e.g. tissue boundaries in depth profiling).

    Args:
        signal: acquired complex signal (n_time_points,).
        encoding: frequency-to-depth encoding.
        dwell_time_s: dwell time per sample (s).
        lamda: L1 regularisation weight.
        max_iter: maximum ADMM iterations.

    Returns:
        ReconResult with reconstructed profile.
    """
    if not SIGPY_AVAILABLE:
        raise RuntimeError("sigpy is required for reconstruction")

    n_time = len(signal)
    n_depth = len(encoding.depths_mm)
    E = build_encoding_matrix(encoding, n_time, dwell_time_s)

    E_op = _EncodingLinop(E)
    W_op = sp.linop.Wavelet((n_depth,), wave_name="db4")

    # min_x ‖Ex − y‖₂² + λ‖Wx‖₁  via ADMM
    proxg = sp.prox.L1Reg(W_op.oshape, lamda)
    app = sp.app.LinearLeastSquares(
        E_op, signal, proxg=proxg, G=W_op, max_iter=max_iter, show_pbar=False
    )
    x = app.run()

    residual = float(np.linalg.norm(E @ x - signal))
    return ReconResult(
        depths_mm=encoding.depths_mm.copy(),
        profile=np.abs(x),
        residual_norm=residual,
        iterations=max_iter,
        method="l1_wavelet",
    )


def reconstruct_total_variation(
    signal: NDArray[np.complex128],
    encoding: EncodingInfo,
    dwell_time_s: float,
    lamda: float = 0.005,
    max_iter: int = 200,
) -> ReconResult:
    """Reconstruction with total-variation regularisation.

    Solves  min_x ‖E·x − y‖₂² + λ‖∇x‖₁

    Best for piecewise-constant profiles (sharp tissue layer boundaries).

    Args:
        signal: acquired complex signal (n_time_points,).
        encoding: frequency-to-depth encoding.
        dwell_time_s: dwell time per sample (s).
        lamda: TV regularisation weight.
        max_iter: maximum iterations.

    Returns:
        ReconResult with reconstructed profile.
    """
    if not SIGPY_AVAILABLE:
        raise RuntimeError("sigpy is required for reconstruction")

    n_time = len(signal)
    n_depth = len(encoding.depths_mm)
    E = build_encoding_matrix(encoding, n_time, dwell_time_s)

    E_op = _EncodingLinop(E)
    G_op = sp.linop.FiniteDifference((n_depth,))

    proxg = sp.prox.L1Reg(G_op.oshape, lamda)
    app = sp.app.LinearLeastSquares(
        E_op, signal, proxg=proxg, G=G_op, max_iter=max_iter, show_pbar=False
    )
    x = app.run()

    residual = float(np.linalg.norm(E @ x - signal))
    return ReconResult(
        depths_mm=encoding.depths_mm.copy(),
        profile=np.abs(x),
        residual_norm=residual,
        iterations=max_iter,
        method="total_variation",
    )


# ── RF pulse design ──────────────────────────────────────────────

def design_excitation_pulse(
    bandwidth_hz: float,
    duration_us: float = 1000.0,
    n_points: int = 256,
    pulse_type: str = "sinc",
) -> RFPulseResult:
    """Design a frequency-selective RF excitation pulse.

    For the single-sided magnet, the sweet-spot bandwidth determines
    the slice (depth range) excited.  This designs a pulse that
    excites a band of ``bandwidth_hz`` centred on the Larmor frequency.

    Args:
        bandwidth_hz: excitation bandwidth (Hz).
        duration_us: pulse duration (µs).
        n_points: number of waveform samples.
        pulse_type: "sinc" (windowed sinc via SLR) or "adiabatic" (HS pulse).

    Returns:
        RFPulseResult with complex RF waveform and timing.
    """
    if not SIGPY_AVAILABLE:
        raise RuntimeError("sigpy is required for RF pulse design")

    duration_s = duration_us * 1e-6
    dt_s = duration_s / n_points
    time_us = np.arange(n_points) * (duration_us / n_points)
    tbw = bandwidth_hz * duration_s  # time-bandwidth product

    if pulse_type == "sinc":
        # SLR-based excitation pulse (small-tip approximation)
        # tbw must be >= 2 for SLR; clamp if needed
        tbw_clamped = max(tbw, 2.0)
        pulse = sigpy.mri.rf.dzrf(
            n_points,
            tbw_clamped,
            ptype="ex",
            ftype="ls",
            d1=0.01,
            d2=0.01,
        )
    elif pulse_type == "adiabatic":
        # Hyperbolic secant adiabatic pulse
        # hypsec returns (am, fm) tuple — amplitude and frequency modulation
        n_points_actual = max(n_points, 512)
        beta = 800.0  # modulation parameter
        mu = 4.9  # adiabatic parameter
        am, fm = sigpy.mri.rf.hypsec(n_points_actual, beta=beta, mu=mu)
        # Construct complex RF waveform from AM/FM
        dt_norm = 1.0 / n_points_actual
        phase = np.cumsum(fm) * 2 * math.pi * dt_norm
        pulse = am * np.exp(1j * phase)
        time_us = np.arange(n_points_actual) * (duration_us / n_points_actual)
    else:
        raise ValueError(f"Unknown pulse type: {pulse_type!r} (use 'sinc' or 'adiabatic')")

    return RFPulseResult(
        pulse=np.asarray(pulse, dtype=np.complex128),
        time_us=time_us,
        duration_us=duration_us,
        bandwidth_hz=bandwidth_hz,
        time_bandwidth=tbw,
    )


def bloch_simulate_pulse(
    pulse: NDArray[np.complex128],
    dt_s: float,
    freq_offsets_hz: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Simulate the excitation profile of an RF pulse via Bloch equations.

    Uses SigPy's ``blochsim`` to compute the transverse magnetisation
    Mxy at each frequency offset after the given RF pulse.

    Args:
        pulse: complex RF waveform (n_points,).
        dt_s: dwell time per sample (s).
        freq_offsets_hz: array of frequency offsets (Hz) to evaluate.

    Returns:
        Complex Mxy at each frequency offset (a.u.).
    """
    if not SIGPY_AVAILABLE:
        raise RuntimeError("sigpy is required for Bloch simulation")

    n_offsets = len(freq_offsets_hz)
    mxy = np.zeros(n_offsets, dtype=np.complex128)

    for i, df in enumerate(freq_offsets_hz):
        # sigpy blochsim expects: rf (n,), grad (n,), dt, df
        # b1 pulse in radians (assume small tip, scale doesn't matter for profile shape)
        a, b = sigpy.mri.rf.abrm(
            pulse,
            np.ones(1) * 2 * math.pi * df * dt_s,
        )
        # Transverse magnetisation from Cayley-Klein parameters
        mxy[i] = 2 * np.conj(a.ravel()[-1]) * b.ravel()[-1]

    return mxy
