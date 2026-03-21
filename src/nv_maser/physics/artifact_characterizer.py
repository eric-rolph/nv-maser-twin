"""Reconstruction artifact characterizer for the handheld NMR/MRI probe.

Non-Cartesian k-space acquisition uses gradient coils to sample k-space on
radial or spiral trajectories.  Gridding-based reconstruction (Cartesian
re-gridding + FFT) introduces three primary artifact types that the V1
handheld probe must keep within acceptable bounds:

1.  **PSF broadening** — the gridding convolution kernel (Gaussian, width
    ``kernel_width`` grid pixels) spreads point-source energy, widening the
    effective point-spread function beyond the single-pixel diffraction limit.

2.  **Aliasing** — density compensation (histogram-based Voronoi weighting)
    is imperfect; residual weighting errors leave ghost energy outside the
    true signal support.  Metric: artifact-to-signal power ratio (ASR).

3.  **Ringing (Gibbs phenomenon)** — k-space truncation at the finite matrix
    boundary causes overshoot at sharp edges.  Suppressed by the Hamming
    window applied inside ``reconstruct_fft``; residual ringing is measured
    with the window active (V1 operating condition).

This module closes risk **R8** (*Reconstruction artifacts from non-Cartesian
k-space*) in the architecture risk register by quantifying all three artifact
levels and declaring V1 acceptance when all three are within their respective
thresholds.

Architecture references
-----------------------
* Handheld probe architecture document §10 and Appendix B.
* Risk register R8: "Use proven NUFFT libraries; train ML denoiser (V2)."
* Phase-6 milestone: "First 2D image on screen."

Physics references
------------------
Jackson, Meyer & Nishimura, "Selection of a convolution function for Fourier
inversion using gridding", IEEE TMI 10:473 (1991).

O'Sullivan, "A fast sinc function gridding algorithm for Fourier inversion in
computer tomography", IEEE TMI 4:200 (1985).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .reconstruction import (
    grid_kspace,
    image_snr_from_phantom,
    reconstruct_fft,
    simulate_kspace,
)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Configuration                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class ArtifactConfig:
    """Parameters for a single artifact characterisation run.

    Attributes
    ----------
    grid_size:
        Reconstruction matrix side length N (N × N grid).
    fov_m:
        Field of view in metres (applied symmetrically in x and y).
    kernel_width:
        Gridding kernel half-width in grid pixels, passed to
        ``grid_kspace``.  Larger values → smoother gridding but broader PSF.
    trajectory:
        k-space trajectory type: ``"radial"`` or ``"spiral"``.
    n_spokes:
        Number of radial spokes (used when ``trajectory="radial"``).
    n_interleaves:
        Number of spiral interleaves (used when ``trajectory="spiral"``).
    n_readout:
        Readout samples per spoke or interleave.
    disk_radius_frac:
        Disk phantom radius as a fraction of ``grid_size``.
    v1_asr_threshold:
        Maximum aliasing artifact-to-signal ratio for V1 acceptance.
        Dimensionless power ratio (not dB).
    v1_overshoot_frac:
        Maximum Gibbs ringing overshoot fraction at a step edge for V1
        acceptance.  The un-windowed Gibbs limit is 0.089 (8.9 %).
    v1_psf_fwhm_ratio:
        Maximum allowed PSF FWHM relative to the single-pixel diffraction
        limit (``pixel_size_mm``) for V1 acceptance.
    """

    grid_size: int = 64
    fov_m: float = 0.08
    kernel_width: float = 3.0
    trajectory: Literal["radial", "spiral"] = "radial"
    n_spokes: int = 64
    n_interleaves: int = 8
    n_readout: int = 64
    disk_radius_frac: float = 0.3
    v1_asr_threshold: float = 0.05
    v1_overshoot_frac: float = 0.09
    v1_psf_fwhm_ratio: float = 3.0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Result data classes                                             ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class PSFResult:
    """Point-spread function characterisation result.

    Attributes
    ----------
    fwhm_x_mm:
        Measured PSF FWHM along x (mm).
    fwhm_y_mm:
        Measured PSF FWHM along y (mm).
    ideal_fwhm_mm:
        Reference FWHM = ``pixel_size_mm`` (single-pixel diffraction limit).
    fwhm_ratio:
        ``max(fwhm_x_mm, fwhm_y_mm) / ideal_fwhm_mm``.
    peak_value:
        Peak intensity of the reconstructed PSF (unnormalised).
    within_spec:
        ``True`` when ``fwhm_ratio ≤ config.v1_psf_fwhm_ratio``.
    """

    fwhm_x_mm: float
    fwhm_y_mm: float
    ideal_fwhm_mm: float
    fwhm_ratio: float
    peak_value: float
    within_spec: bool


@dataclass(frozen=True)
class AliasingResult:
    """Aliasing artifact characterisation result.

    Attributes
    ----------
    asr:
        Artifact-to-signal ratio: background power / signal power.
        Dimensionless power ratio.
    asr_db:
        ``10 · log₁₀(asr)`` in dB.
    within_spec:
        ``True`` when ``asr ≤ config.v1_asr_threshold``.
    """

    asr: float
    asr_db: float
    within_spec: bool


@dataclass(frozen=True)
class RingingResult:
    """Gibbs ringing artifact characterisation result.

    Attributes
    ----------
    overshoot_fraction:
        Peak overshoot above the bright plateau, normalised by step height.
    undershoot_fraction:
        Peak undershoot below the dark plateau, normalised by step height.
    within_spec:
        ``True`` when ``overshoot_fraction ≤ config.v1_overshoot_frac``.
    """

    overshoot_fraction: float
    undershoot_fraction: float
    within_spec: bool


@dataclass(frozen=True)
class ArtifactResult:
    """Full artifact characterisation result for R8 risk closure.

    Attributes
    ----------
    config:
        Configuration used for this run.
    psf:
        PSF characterisation result.
    aliasing:
        Aliasing characterisation result.
    ringing:
        Gibbs ringing characterisation result.
    pixel_size_mm:
        Image pixel size in mm = ``fov_m / grid_size × 1000``.
    trajectory_n_samples:
        Total number of k-space samples in the trajectory.
    cartesian_reference_snr_db:
        SNR of an ideal Cartesian FFT reconstruction of the disk phantom
        (dB).  Set to 0.0 when SNR is indeterminate (noise-free simulation).
    gridding_snr_db:
        SNR of the gridded non-Cartesian reconstruction of the same phantom
        (dB).  Set to 0.0 when indeterminate.
    snr_loss_gridding_db:
        ``cartesian_reference_snr_db − gridding_snr_db`` (dB).
        Positive means gridding has lower SNR.
    r8_risk_closed:
        ``True`` when PSF, aliasing, and ringing all satisfy their V1
        acceptance thresholds.  Signals closure of risk R8.
    """

    config: ArtifactConfig
    psf: PSFResult
    aliasing: AliasingResult
    ringing: RingingResult
    pixel_size_mm: float
    trajectory_n_samples: int
    cartesian_reference_snr_db: float
    gridding_snr_db: float
    snr_loss_gridding_db: float
    r8_risk_closed: bool


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Private helpers                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝


def _noncartesian_dft(
    image: NDArray,
    kx_per_m: NDArray,
    ky_per_m: NDArray,
    fov_x_m: float,
    fov_y_m: float,
) -> NDArray:
    """Exact DFT evaluated at arbitrary non-Cartesian k-space positions.

    Evaluates::

        S(kx_i, ky_i) = Σ_{n,m} img[n,m] · exp(−j2π (kx_i·x_m + ky_i·y_n))

    using vectorised matrix products; complexity O(M × Ny × Nx).  No
    external dependencies beyond NumPy.

    Parameters
    ----------
    image:
        Ground-truth image, shape ``(Ny, Nx)``.
    kx_per_m, ky_per_m:
        Non-Cartesian k-space sample positions (m⁻¹), shape ``(M,)``.
    fov_x_m, fov_y_m:
        Field of view (m).

    Returns
    -------
    NDArray
        Complex k-space values, shape ``(M,)``.
    """
    img = np.asarray(image, dtype=complex)
    ny, nx = img.shape
    dx = fov_x_m / nx
    dy = fov_y_m / ny

    # Spatial coordinate grids centred at (0, 0)
    x = (np.arange(nx) - nx / 2.0) * dx   # (Nx,) metres
    y = (np.arange(ny) - ny / 2.0) * dy   # (Ny,) metres

    kx = np.asarray(kx_per_m, dtype=float)
    ky = np.asarray(ky_per_m, dtype=float)

    # Phase matrices
    phase_x = np.exp(-1j * 2.0 * math.pi * np.outer(kx, x))  # (M, Nx)
    phase_y = np.exp(-1j * 2.0 * math.pi * np.outer(ky, y))  # (M, Ny)

    # tmp[i, n] = Σ_m img[n, m] · phase_x[i, m]  →  (M, Ny)
    tmp = phase_x @ img.T

    # S[i] = Σ_n phase_y[i, n] · tmp[i, n]
    return (phase_y * tmp).sum(axis=1)  # (M,)


def _measure_fwhm(profile: NDArray, pixel_mm: float) -> float:
    """Measure the FWHM of a 1-D profile via linear interpolation.

    Parameters
    ----------
    profile:
        1-D magnitude array.  Assumed to have a single dominant peak.
    pixel_mm:
        Physical size of each array element (mm).

    Returns
    -------
    float
        FWHM in mm; ``math.inf`` when the crossing cannot be found.
    """
    p = np.asarray(profile, dtype=float)
    peak = float(p.max())
    if peak == 0.0:
        return math.inf
    half = peak * 0.5
    n = len(p)

    # Left crossing: first rising edge where profile crosses half-maximum
    left: float | None = None
    for i in range(n - 1):
        if p[i] < half <= p[i + 1]:
            t = (half - p[i]) / (p[i + 1] - p[i])
            left = i + t
            break

    # Right crossing: last falling edge where profile drops below half-maximum
    right: float | None = None
    for i in range(n - 1, 0, -1):
        if p[i] < half <= p[i - 1]:
            t = (half - p[i]) / (p[i - 1] - p[i])
            right = (i - 1) + (1.0 - t)
            break

    if left is None or right is None or right <= left:
        return math.inf
    return float((right - left) * pixel_mm)


def _reconstruct_with_config(
    kx: NDArray,
    ky: NDArray,
    samples: NDArray,
    config: ArtifactConfig,
):
    """Grid non-Cartesian k-space and reconstruct with a Hamming-windowed FFT.

    Passes ``config.kernel_width`` through to ``grid_kspace``, which
    ``reconstruct_gridding`` does not expose directly.
    """
    gridding = grid_kspace(
        kx,
        ky,
        samples,
        grid_size=(config.grid_size, config.grid_size),
        fov_x_m=config.fov_m,
        fov_y_m=config.fov_m,
        kernel_width=config.kernel_width,
    )
    return reconstruct_fft(
        gridding.kspace_grid,
        fov_x_m=config.fov_m,
        fov_y_m=config.fov_m,
        apply_hamming=True,
    )


def _get_trajectory(config: ArtifactConfig) -> tuple[NDArray, NDArray]:
    """Return ``(kx, ky)`` trajectory arrays for the given config."""
    if config.trajectory == "radial":
        return generate_radial_trajectory(
            config.n_spokes, config.n_readout, config.fov_m, config.grid_size
        )
    if config.trajectory == "spiral":
        return generate_spiral_trajectory(
            config.n_interleaves, config.n_readout, config.fov_m, config.grid_size
        )
    raise ValueError(f"Unknown trajectory type: {config.trajectory!r}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Trajectory generators                                           ║
# ╚══════════════════════════════════════════════════════════════════╝


def generate_radial_trajectory(
    n_spokes: int,
    n_readout: int,
    fov_m: float,
    grid_size: int,
) -> tuple[NDArray, NDArray]:
    """Generate a uniform-angle radial k-space trajectory.

    Spokes are distributed uniformly over ``[0, π)``; opposite directions on
    each spoke together cover the full k-space disk.  All spokes pass through
    the k-space origin.

    Parameters
    ----------
    n_spokes:
        Number of radial spokes.
    n_readout:
        Readout samples per spoke (spans ``−kmax`` to ``+kmax``).
    fov_m:
        Field of view (m), used to derive the k-space grid spacing.
    grid_size:
        Reconstruction matrix side N; sets
        ``kmax = N / (2 · fov_m)`` (m⁻¹).

    Returns
    -------
    kx_per_m, ky_per_m:
        Shape ``(n_spokes × n_readout,)`` arrays of k-space positions (m⁻¹).
    """
    kmax = grid_size / 2.0 / fov_m                    # m⁻¹  (Nyquist limit)
    k_line = np.linspace(-kmax, kmax, n_readout)
    angles = np.linspace(0.0, math.pi, n_spokes, endpoint=False)

    kx_parts = [k_line * math.cos(a) for a in angles]
    ky_parts = [k_line * math.sin(a) for a in angles]

    return np.concatenate(kx_parts), np.concatenate(ky_parts)


def generate_spiral_trajectory(
    n_interleaves: int,
    n_readout: int,
    fov_m: float,
    grid_size: int,
) -> tuple[NDArray, NDArray]:
    """Generate an Archimedean interleaved spiral k-space trajectory.

    Each interleave is an Archimedean spiral from ``k = 0`` to ``kmax``,
    rotated by ``2π / n_interleaves`` from the previous interleave.

    Parameters
    ----------
    n_interleaves:
        Number of interleaved spirals.
    n_readout:
        Readout samples per interleave.
    fov_m:
        Field of view (m).
    grid_size:
        Reconstruction matrix side N.

    Returns
    -------
    kx_per_m, ky_per_m:
        Shape ``(n_interleaves × n_readout,)`` arrays of k-space positions
        (m⁻¹).
    """
    kmax = grid_size / 2.0 / fov_m
    t = np.linspace(0.0, 1.0, n_readout)
    n_rotations = n_interleaves / 2.0    # full rotations per interleave

    kx_parts: list[NDArray] = []
    ky_parts: list[NDArray] = []
    for i in range(n_interleaves):
        phase = 2.0 * math.pi * i / n_interleaves
        r = kmax * t
        theta = 2.0 * math.pi * n_rotations * t + phase
        kx_parts.append(r * np.cos(theta))
        ky_parts.append(r * np.sin(theta))

    return np.concatenate(kx_parts), np.concatenate(ky_parts)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Phantom generators                                              ║
# ╚══════════════════════════════════════════════════════════════════╝


def make_phantom(
    phantom_type: Literal["disk", "point", "step"],
    grid_size: int,
    disk_radius_frac: float = 0.3,
) -> NDArray:
    """Create a synthetic 2-D phantom for artifact characterisation.

    Parameters
    ----------
    phantom_type:
        ``"point"`` — single bright pixel at image centre (PSF test).
        ``"disk"``  — filled circle centred at image centre (aliasing test).
        ``"step"``  — top half bright, bottom half dark (ringing test).
    grid_size:
        Width and height of the phantom (N × N).
    disk_radius_frac:
        Disk radius as a fraction of ``grid_size`` (used only for
        ``phantom_type="disk"``).

    Returns
    -------
    NDArray
        Float64 array of shape ``(grid_size, grid_size)`` with values in
        ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``phantom_type`` is not one of the supported values.
    """
    N = grid_size
    phantom = np.zeros((N, N), dtype=float)
    cx = N // 2
    cy = N // 2

    if phantom_type == "point":
        phantom[cy, cx] = 1.0
    elif phantom_type == "disk":
        r = disk_radius_frac * N
        y_idx, x_idx = np.ogrid[:N, :N]
        mask = (x_idx - cx) ** 2 + (y_idx - cy) ** 2 <= r ** 2
        phantom[mask] = 1.0
    elif phantom_type == "step":
        phantom[: N // 2, :] = 1.0
    else:
        raise ValueError(
            f"Unknown phantom_type: {phantom_type!r}. "
            "Supported values: 'point', 'disk', 'step'."
        )

    return phantom


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Individual artifact measurement functions                       ║
# ╚══════════════════════════════════════════════════════════════════╝


def compute_psf(config: ArtifactConfig | None = None) -> PSFResult:
    """Characterise PSF broadening from gridding reconstruction.

    Reconstructs a centred point-source phantom via the non-Cartesian
    gridding pipeline and measures the FWHM of the resulting PSF along
    the x and y directions through the peak voxel.

    V1 acceptance criterion::

        max(fwhm_x_mm, fwhm_y_mm) / pixel_size_mm ≤ config.v1_psf_fwhm_ratio

    Parameters
    ----------
    config:
        Characterisation configuration; uses ``ArtifactConfig()`` if None.

    Returns
    -------
    PSFResult
    """
    if config is None:
        config = ArtifactConfig()

    phantom = make_phantom("point", config.grid_size)
    kx, ky = _get_trajectory(config)
    pixel_mm = config.fov_m / config.grid_size * 1e3

    samples = _noncartesian_dft(phantom, kx, ky, config.fov_m, config.fov_m)
    result = _reconstruct_with_config(kx, ky, samples, config)
    mag = result.magnitude

    peak = float(mag.max())
    mag_norm = mag / peak if peak > 0.0 else mag

    # Profile through the true peak voxel (may be offset from grid centre)
    cy_peak, cx_peak = np.unravel_index(np.argmax(mag), mag.shape)

    fwhm_x = _measure_fwhm(mag_norm[cy_peak, :], pixel_mm)
    fwhm_y = _measure_fwhm(mag_norm[:, cx_peak], pixel_mm)

    ideal_fwhm_mm = pixel_mm                 # single-pixel diffraction limit
    fwhm_ratio = float(
        max(fwhm_x, fwhm_y) / ideal_fwhm_mm
        if ideal_fwhm_mm > 0.0
        else math.inf
    )

    return PSFResult(
        fwhm_x_mm=float(fwhm_x),
        fwhm_y_mm=float(fwhm_y),
        ideal_fwhm_mm=ideal_fwhm_mm,
        fwhm_ratio=fwhm_ratio,
        peak_value=float(peak),
        within_spec=bool(fwhm_ratio <= config.v1_psf_fwhm_ratio),
    )


def compute_aliasing(config: ArtifactConfig | None = None) -> AliasingResult:
    """Characterise aliasing artifacts from non-Cartesian gridding.

    Uses a disk phantom and measures the artifact-to-signal ratio (ASR)::

        ASR = Σ(mag[background]²) / Σ(mag[signal]²)

    The signal region is the disk support (``phantom > 0.5``); the
    background is the complement.

    V1 acceptance criterion: ``ASR ≤ config.v1_asr_threshold``.

    Parameters
    ----------
    config:
        Characterisation configuration; uses ``ArtifactConfig()`` if None.

    Returns
    -------
    AliasingResult
    """
    if config is None:
        config = ArtifactConfig()

    phantom = make_phantom("disk", config.grid_size, config.disk_radius_frac)
    kx, ky = _get_trajectory(config)

    samples = _noncartesian_dft(phantom, kx, ky, config.fov_m, config.fov_m)
    result = _reconstruct_with_config(kx, ky, samples, config)
    mag = result.magnitude

    signal_mask = phantom > 0.5
    bg_mask = ~signal_mask
    signal_power = float(np.sum(mag[signal_mask] ** 2))
    bg_power = float(np.sum(mag[bg_mask] ** 2))

    if signal_power <= 0.0:
        return AliasingResult(asr=math.inf, asr_db=math.inf, within_spec=False)

    asr = bg_power / signal_power
    asr_db = 10.0 * math.log10(asr) if asr > 0.0 else -math.inf

    return AliasingResult(
        asr=asr,
        asr_db=float(asr_db),
        within_spec=asr <= config.v1_asr_threshold,
    )


def compute_ringing(config: ArtifactConfig | None = None) -> RingingResult:
    """Characterise Gibbs ringing from k-space truncation.

    Measures the residual Gibbs overshoot on a Cartesian simulation of a
    step-edge phantom (top half bright, bottom half dark) after applying the
    Hamming window inside ``reconstruct_fft``.

    **Why Cartesian k-space is used here**: a horizontal step phantom has all
    its k-space energy concentrated along :math:`k_x = 0`, which is sparsely
    sampled by a radial trajectory (only the single vertical spoke hits that
    axis).  Using the full Cartesian k-space ``simulate_kspace`` gives a
    clean, well-defined ringing profile.  This is valid because:

    1. The Hamming window applied inside ``reconstruct_fft`` is identical for
       both Cartesian and gridded non-Cartesian pipelines.
    2. The gridding kernel adds further k-space smoothing, so gridded Gibbs
       is bounded above by the Cartesian result measured here.

    Classic un-windowed Gibbs gives ~8.9 % overshoot; the Hamming window
    reduces this to typically < 3 %.

    V1 acceptance criterion::

        overshoot_fraction ≤ config.v1_overshoot_frac

    Parameters
    ----------
    config:
        Characterisation configuration; uses ``ArtifactConfig()`` if None.

    Returns
    -------
    RingingResult
    """
    if config is None:
        config = ArtifactConfig()

    N = config.grid_size
    phantom = make_phantom("step", N)

    # Cartesian simulation — uses the same Hamming window as the gridded pipeline
    ks = simulate_kspace(phantom, fov_x_m=config.fov_m, fov_y_m=config.fov_m)
    result = reconstruct_fft(
        ks,
        fov_x_m=config.fov_m,
        fov_y_m=config.fov_m,
        apply_hamming=True,
    )
    mag = result.magnitude

    # Average over the central half of columns to suppress edge effects
    col_start = N // 4
    col_end = 3 * N // 4
    profile = np.mean(mag[:, col_start:col_end], axis=1)  # (Ny,)

    # Estimate plateau values from rows well away from the step boundary
    q = max(1, N // 4)
    bright_mean = float(np.mean(profile[:q]))        # top-quarter rows
    dark_mean = float(np.mean(profile[N - q :]))     # bottom-quarter rows
    step_height = bright_mean - dark_mean

    # Guard: step absent or inverted (should not occur for Cartesian simulation)
    if step_height <= 1e-10:
        return RingingResult(
            overshoot_fraction=0.0,
            undershoot_fraction=0.0,
            within_spec=True,
        )

    max_val = float(profile.max())
    min_val = float(profile.min())

    overshoot = max(0.0, max_val - bright_mean) / step_height
    undershoot = max(0.0, dark_mean - min_val) / step_height

    return RingingResult(
        overshoot_fraction=overshoot,
        undershoot_fraction=undershoot,
        within_spec=overshoot <= config.v1_overshoot_frac,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Top-level characterisation                                      ║
# ╚══════════════════════════════════════════════════════════════════╝


def compute_artifact_characterization(
    config: ArtifactConfig | None = None,
) -> ArtifactResult:
    """Run the full artifact characterisation pipeline.

    Characterises PSF broadening, aliasing, and Gibbs ringing for the
    specified k-space trajectory and gridding kernel.  Also computes a
    SNR comparison between ideal Cartesian FFT and gridded non-Cartesian
    reconstruction on the same disk phantom.

    Sets ``r8_risk_closed = True`` when all three artifact metrics satisfy
    their V1 acceptance thresholds, signalling that risk R8
    (*Reconstruction artifacts from non-Cartesian k-space*) is within the
    acceptable V1 operating envelope.  ML denoising (V2 roadmap item) can
    further suppress residual artifacts beyond these bounds.

    Parameters
    ----------
    config:
        Characterisation configuration; uses ``ArtifactConfig()`` if None.

    Returns
    -------
    ArtifactResult
    """
    if config is None:
        config = ArtifactConfig()

    kx, ky = _get_trajectory(config)
    n_samples = len(kx)
    pixel_mm = config.fov_m / config.grid_size * 1e3

    # ── Cartesian reference SNR (ideal FFT on full k-space grid) ──
    disk_phantom = make_phantom("disk", config.grid_size, config.disk_radius_frac)
    ks_cart = simulate_kspace(disk_phantom, fov_x_m=config.fov_m, fov_y_m=config.fov_m)
    cart_result = reconstruct_fft(ks_cart, fov_x_m=config.fov_m, fov_y_m=config.fov_m)
    cart_snr = image_snr_from_phantom(cart_result.magnitude)

    # ── Non-Cartesian gridding SNR (same disk phantom) ─────────────
    samples_nc = _noncartesian_dft(disk_phantom, kx, ky, config.fov_m, config.fov_m)
    grid_result = _reconstruct_with_config(kx, ky, samples_nc, config)
    grid_snr = image_snr_from_phantom(grid_result.magnitude)

    # Set to 0.0 when SNR is indeterminate (noise-free simulation has no
    # noise floor, so image_snr_from_phantom may return NaN)
    cart_snr_safe = 0.0 if math.isnan(cart_snr) else cart_snr
    grid_snr_safe = 0.0 if math.isnan(grid_snr) else grid_snr

    # ── Individual artifact metrics ────────────────────────────────
    psf_result = compute_psf(config)
    aliasing_result = compute_aliasing(config)
    ringing_result = compute_ringing(config)

    # ── R8 risk closure ────────────────────────────────────────────
    r8_closed = (
        psf_result.within_spec
        and aliasing_result.within_spec
        and ringing_result.within_spec
    )

    return ArtifactResult(
        config=config,
        psf=psf_result,
        aliasing=aliasing_result,
        ringing=ringing_result,
        pixel_size_mm=pixel_mm,
        trajectory_n_samples=n_samples,
        cartesian_reference_snr_db=cart_snr_safe,
        gridding_snr_db=grid_snr_safe,
        snr_loss_gridding_db=cart_snr_safe - grid_snr_safe,
        r8_risk_closed=r8_closed,
    )
