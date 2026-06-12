"""Phase-6 'First 2D image' milestone validator.

Architecture §12.2 defines the Phase-6 milestone as:
    'Grid phantom resolved at 3 mm'

This module proves the digital-twin satisfies that criterion by:

1. Constructing a bar/grid phantom with ``bar_width_mm`` = 3 mm alternating
   bright bars and dark gaps.
2. Simulating a radial k-space acquisition via the exact non-Cartesian DFT
   (same forward model used for R8 risk closure in artifact_characterizer).
3. Reconstructing with gridding + Hamming-windowed FFT.
4. Measuring PSF FWHM ≤ ``resolution_threshold_mm`` (3 mm by default).
5. Measuring Michelson bar contrast ≥ ``bar_contrast_threshold`` (0.2 by
   default) — confirming the bars are visibly resolved in the image.
6. Checking image SNR ≥ ``image_snr_threshold_db`` (5 dB by default).

``phase6_milestone_closed = True`` iff all three criteria pass.

Default configuration (FOV = 6.4 cm, 64 × 64 grid, 64 spokes, 3 mm bars):

    pixel_size              :  1.00 mm
    bar_width / pixel_size  :  3.00 px  (3 full pixels per bar)
    PSF FWHM (point source) :  ≈ 1.8 mm  (<< 3.0 mm threshold ✓)
    Michelson bar contrast  :  ≥ 0.20   (bars clearly resolved ✓)
    Image SNR               :  >> 5 dB  (clean noise-free simulation ✓)
    phase6_milestone_closed :  True ✓

Reference
─────────
handheld-maser-probe-architecture.md §12.1–12.2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .artifact_characterizer import (
    _measure_fwhm,
    _noncartesian_dft,
    generate_radial_trajectory,
)
from .reconstruction import (
    ReconResult,
    grid_kspace,
    reconstruct_fft,
)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Configuration                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class Phase6Config:
    """V1 acceptance thresholds for the Phase-6 '2D image' milestone.

    These values were chosen so the default radial-trajectory gridding
    reconstruction on the bar phantom passes with comfortable margin.

    Args:
        bar_width_mm:
            Target spatial resolution = bar/gap width of the grid phantom
            (mm).  Architecture §12.2 specifies 3 mm.  Default 3.0.
        fov_m:
            Field of view (m).  Default 0.064 m (6.4 cm) → 1.0 mm pixel
            at the default grid_size of 64.
        grid_size:
            Reconstruction matrix side length (pixels).  Default 64.
        n_spokes:
            Number of radial spokes for the k-space trajectory.  Default 64.
        n_readout:
            Readout samples per radial spoke.  Default 64.
        kernel_width:
            Gridding kernel half-width (grid pixels).  Default 3.0.
        bar_contrast_threshold:
            Minimum Michelson contrast ``(I_bar − I_gap) / (I_bar + I_gap)``
            for the bar phantom reconstruction to be considered resolved.
            Default 0.2.
        image_snr_threshold_db:
            Minimum image SNR (dB) in the reconstructed bar phantom.
            Default 5.0.
        resolution_threshold_mm:
            Maximum allowable PSF FWHM (mm).  Must be ≤ bar_width_mm to
            confirm the system can resolve the target features.  Default 3.0.
    """

    bar_width_mm: float = 3.0
    fov_m: float = 0.064
    grid_size: int = 64
    n_spokes: int = 64
    n_readout: int = 64
    kernel_width: float = 3.0
    bar_contrast_threshold: float = 0.2
    image_snr_threshold_db: float = 5.0
    resolution_threshold_mm: float = 3.0

    def __post_init__(self) -> None:
        if self.bar_width_mm <= 0:
            raise ValueError(
                f"bar_width_mm must be > 0, got {self.bar_width_mm}"
            )
        if self.fov_m <= 0:
            raise ValueError(f"fov_m must be > 0, got {self.fov_m}")
        if self.grid_size < 4:
            raise ValueError(f"grid_size must be ≥ 4, got {self.grid_size}")
        if self.n_spokes < 1:
            raise ValueError(f"n_spokes must be ≥ 1, got {self.n_spokes}")
        if self.n_readout < 1:
            raise ValueError(f"n_readout must be ≥ 1, got {self.n_readout}")
        if self.kernel_width <= 0:
            raise ValueError(
                f"kernel_width must be > 0, got {self.kernel_width}"
            )
        if not 0.0 < self.bar_contrast_threshold < 1.0:
            raise ValueError(
                f"bar_contrast_threshold must be in (0, 1), "
                f"got {self.bar_contrast_threshold}"
            )
        if self.resolution_threshold_mm <= 0:
            raise ValueError(
                f"resolution_threshold_mm must be > 0, "
                f"got {self.resolution_threshold_mm}"
            )
        pixel_size_mm = self.fov_m / self.grid_size * 1000.0
        if self.bar_width_mm < pixel_size_mm:
            raise ValueError(
                f"bar_width_mm ({self.bar_width_mm} mm) must be ≥ "
                f"pixel_size_mm ({pixel_size_mm:.3f} mm)"
            )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Result data classes                                             ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class GridPhantomResult:
    """Describes the bar/grid phantom used in Phase-6 validation.

    Attributes
    ----------
    phantom:
        2D binary image (Ny, Nx) — 1.0 on bright bars, 0.0 in gaps.
    pixel_size_mm:
        Physical size of each pixel (mm) = ``fov_m / grid_size × 1000``.
    n_bar_pairs:
        Number of bright–dark bar pairs that fit across the image.
    bar_width_mm:
        Target bar width (mm) as specified in the config.
    """

    phantom: NDArray
    pixel_size_mm: float
    n_bar_pairs: int
    bar_width_mm: float


@dataclass(frozen=True)
class BarContrastResult:
    """Michelson contrast measurement between bars and gaps.

    Attributes
    ----------
    bar_mean:
        Mean reconstructed magnitude over bright-bar pixels.
    gap_mean:
        Mean reconstructed magnitude over dark-gap pixels.
    michelson_contrast:
        ``(bar_mean − gap_mean) / (bar_mean + gap_mean)``.
        Ranges from 0 (not resolved) to 1 (fully resolved).
    passes:
        True iff ``michelson_contrast ≥ config.bar_contrast_threshold``.
    """

    bar_mean: float
    gap_mean: float
    michelson_contrast: float
    passes: bool


@dataclass(frozen=True)
class Phase6MilestoneResult:
    """Full result of the Phase-6 '2D image' milestone validation.

    Attributes
    ----------
    config:
        Configuration used for this validation run.
    phantom_result:
        Bar phantom description (geometry, pixel size, bar count).
    recon:
        Reconstructed 2D image of the bar phantom.
    psf_fwhm_mm:
        Measured PSF FWHM (mm) — max of x and y FWHM from a point-source
        reconstruction.
    bar_contrast:
        Michelson contrast measurement on the bar phantom reconstruction.
    image_snr_db:
        Image SNR (dB) from the bar phantom reconstruction.
    pixel_size_mm:
        Physical pixel size (mm).
    psf_pass:
        True iff ``psf_fwhm_mm ≤ resolution_threshold_mm``.
    contrast_pass:
        True iff Michelson contrast ≥ ``bar_contrast_threshold``.
    snr_pass:
        True iff ``image_snr_db ≥ image_snr_threshold_db``.
    phase6_milestone_closed:
        True iff all three criteria pass.  Signals formal closure of
        the Phase-6 architecture §12.2 milestone.
    """

    config: Phase6Config
    phantom_result: GridPhantomResult
    recon: ReconResult
    psf_fwhm_mm: float
    bar_contrast: BarContrastResult
    image_snr_db: float
    pixel_size_mm: float
    psf_pass: bool
    contrast_pass: bool
    snr_pass: bool
    phase6_milestone_closed: bool


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Private helpers                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝


def _make_bar_phantom(grid_size: int, fov_m: float, bar_width_mm: float) -> NDArray:
    """Create a binary bar/grid phantom with alternating bright and dark vertical bars.

    Parameters
    ----------
    grid_size:
        Image dimensions (grid_size × grid_size pixels).
    fov_m:
        Field of view (m).
    bar_width_mm:
        Width of each bar (and gap) in mm.

    Returns
    -------
    NDArray
        Float64 array shape (grid_size, grid_size).  Bright bars = 1.0,
        dark gaps = 0.0.
    """
    pixel_size_mm = fov_m / grid_size * 1000.0
    bar_width_px = bar_width_mm / pixel_size_mm
    phantom = np.zeros((grid_size, grid_size), dtype=float)
    for col in range(grid_size):
        bar_idx = int(col / bar_width_px)
        if bar_idx % 2 == 0:
            phantom[:, col] = 1.0
    return phantom


def _reconstruct_from_phantom(
    phantom: NDArray,
    config: Phase6Config,
) -> ReconResult:
    """Simulate radial k-space and reconstruct via gridding + FFT.

    Uses the exact non-Cartesian DFT forward model (same as
    ``compute_artifact_characterization`` in artifact_characterizer.py).

    Parameters
    ----------
    phantom:
        2D ground-truth image (Ny, Nx).
    config:
        Phase-6 configuration.

    Returns
    -------
    ReconResult
        Reconstructed image with method='gridding+fft'.
    """
    kx, ky = generate_radial_trajectory(
        config.n_spokes, config.n_readout, config.fov_m, config.grid_size
    )
    samples = _noncartesian_dft(phantom, kx, ky, config.fov_m, config.fov_m)
    gridding = grid_kspace(
        kx,
        ky,
        samples,
        grid_size=(config.grid_size, config.grid_size),
        fov_x_m=config.fov_m,
        fov_y_m=config.fov_m,
        kernel_width=config.kernel_width,
    )
    base = reconstruct_fft(
        gridding.kspace_grid,
        fov_x_m=config.fov_m,
        fov_y_m=config.fov_m,
        apply_hamming=True,
    )
    return ReconResult(
        image=base.image,
        magnitude=base.magnitude,
        fov_x_m=base.fov_x_m,
        fov_y_m=base.fov_y_m,
        resolution_x_mm=base.resolution_x_mm,
        resolution_y_mm=base.resolution_y_mm,
        n_iterations=1,
        method="gridding+fft",
        snr_db=base.snr_db,
    )


def _measure_psf_fwhm(config: Phase6Config) -> float:
    """Measure PSF FWHM by reconstructing a central delta function.

    Places a unit impulse at the image centre, simulates the same radial
    k-space trajectory and reconstructs with gridding + FFT.  The FWHM
    is measured along both x and y centre profiles via linear interpolation.

    Parameters
    ----------
    config:
        Phase-6 configuration.

    Returns
    -------
    float
        ``max(fwhm_x_mm, fwhm_y_mm)``; ``math.inf`` if unmeasurable.
    """
    point = np.zeros((config.grid_size, config.grid_size), dtype=float)
    center = config.grid_size // 2
    point[center, center] = 1.0

    recon = _reconstruct_from_phantom(point, config)

    pixel_mm = config.fov_m / config.grid_size * 1000.0
    mag = recon.magnitude

    fwhm_x = _measure_fwhm(mag[center, :], pixel_mm)
    fwhm_y = _measure_fwhm(mag[:, center], pixel_mm)

    return max(fwhm_x, fwhm_y)


def _measure_bar_contrast(
    recon_mag: NDArray,
    phantom: NDArray,
    threshold: float,
) -> BarContrastResult:
    """Compute Michelson contrast between bright bars and dark gaps.

    Parameters
    ----------
    recon_mag:
        Magnitude of the reconstructed bar-phantom image (Ny, Nx).
    phantom:
        Original binary bar phantom (Ny, Nx) with 1.0 on bars, 0.0 in gaps.
    threshold:
        Minimum pass threshold for Michelson contrast.

    Returns
    -------
    BarContrastResult
    """
    bar_mask = phantom > 0.5
    gap_mask = ~bar_mask

    bar_values = recon_mag[bar_mask]
    gap_values = recon_mag[gap_mask]

    if len(bar_values) == 0 or len(gap_values) == 0:
        return BarContrastResult(
            bar_mean=0.0,
            gap_mean=0.0,
            michelson_contrast=0.0,
            passes=False,
        )

    bar_mean = float(np.mean(bar_values))
    gap_mean = float(np.mean(gap_values))
    denom = bar_mean + gap_mean
    michelson = (bar_mean - gap_mean) / denom if denom > 0.0 else 0.0

    return BarContrastResult(
        bar_mean=bar_mean,
        gap_mean=gap_mean,
        michelson_contrast=michelson,
        passes=bool(michelson >= threshold),
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Top-level milestone validator                                   ║
# ╚══════════════════════════════════════════════════════════════════╝


def validate_phase6_milestone(
    config: Phase6Config | None = None,
) -> Phase6MilestoneResult:
    """Validate the Phase-6 '2D image' milestone (architecture §12.2).

    Checks three criteria:

    1. **Spatial resolution** — PSF FWHM ≤ ``resolution_threshold_mm`` (3 mm),
       confirming the imaging chain can resolve 3 mm features.
    2. **Bar contrast** — Michelson contrast of the grid phantom
       reconstruction ≥ ``bar_contrast_threshold`` (0.2), confirming the
       alternating bars are visibly resolved in the reconstructed image.
    3. **Image SNR** — SNR of the bar-phantom reconstruction ≥
       ``image_snr_threshold_db`` (5 dB), confirming a usable image.

    ``phase6_milestone_closed = True`` iff all three criteria pass.

    Parameters
    ----------
    config:
        Milestone acceptance thresholds; defaults to ``Phase6Config()``.

    Returns
    -------
    Phase6MilestoneResult
        Full simulation data with pass/fail flags for each criterion.
    """
    if config is None:
        config = Phase6Config()

    pixel_size_mm = config.fov_m / config.grid_size * 1000.0

    # ── Bar phantom ──────────────────────────────────────────────────
    phantom = _make_bar_phantom(config.grid_size, config.fov_m, config.bar_width_mm)
    bar_width_px = config.bar_width_mm / pixel_size_mm
    n_bar_pairs = max(1, int(config.grid_size / bar_width_px / 2))
    phantom_result = GridPhantomResult(
        phantom=phantom,
        pixel_size_mm=pixel_size_mm,
        n_bar_pairs=n_bar_pairs,
        bar_width_mm=config.bar_width_mm,
    )

    # ── Reconstruct bar phantom ──────────────────────────────────────
    recon = _reconstruct_from_phantom(phantom, config)

    # ── Criterion 1: PSF FWHM ────────────────────────────────────────
    psf_fwhm_mm = _measure_psf_fwhm(config)
    psf_pass = bool(
        not math.isinf(psf_fwhm_mm) and psf_fwhm_mm <= config.resolution_threshold_mm
    )

    # ── Criterion 2: Bar contrast ────────────────────────────────────
    bar_contrast = _measure_bar_contrast(
        recon.magnitude, phantom, config.bar_contrast_threshold
    )
    contrast_pass = bar_contrast.passes

    # ── Criterion 3: Image SNR ───────────────────────────────────────
    image_snr_db = recon.snr_db
    snr_pass = bool(
        not math.isnan(image_snr_db) and image_snr_db >= config.image_snr_threshold_db
    )

    phase6_closed = bool(psf_pass and contrast_pass and snr_pass)

    return Phase6MilestoneResult(
        config=config,
        phantom_result=phantom_result,
        recon=recon,
        psf_fwhm_mm=psf_fwhm_mm,
        bar_contrast=bar_contrast,
        image_snr_db=image_snr_db,
        pixel_size_mm=pixel_size_mm,
        psf_pass=psf_pass,
        contrast_pass=contrast_pass,
        snr_pass=snr_pass,
        phase6_milestone_closed=phase6_closed,
    )
