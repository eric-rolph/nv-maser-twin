"""
k-space → image reconstruction for the handheld NMR/MRI probe.

The probe acquires raw data as a free-induction decay (FID) or echo
signal that has been spatially encoded by gradient coils.  This module
converts that raw k-space data into a reconstructed image using:

    1. Direct FFT (Cartesian k-space, uniform sampling)
    2. Gridding + FFT (non-Cartesian / radial / spiral k-space)
    3. Iterative compressed sensing (exploiting sparsity in image domain)
    4. 1-D depth profile reconstruction (dedicated for CPMG / A-mode)

Signal model
────────────
The MRI signal from a 2D spin distribution ρ(x, y) sampled at k-space
positions (kx, ky) is:

    S(kx, ky) = ∫∫ ρ(x,y) × exp(−j 2π (kx·x + ky·y)) dx dy

Reconstruction = inverse Fourier transform (exact for Cartesian sampling;
approximate via gridding for non-Cartesian).

Compressed sensing
──────────────────
For an under-sampled acquisition the missing k-space data is inferred
by solving the convex programme (basis pursuit denoising):

    min ‖Ψ ρ‖₁   subject to  ‖E ρ − s‖₂ ≤ ε

where:
    E    = encoding operator (FFT of sampled k-space positions)
    Ψ    = sparsifying transform (Haar wavelet in this implementation)
    ρ    = image to reconstruct
    s    = acquired data

We use a simple iterative soft-thresholding algorithm (ISTA) that does
not require external dependencies beyond NumPy.

References
──────────
Lustig, Donoho & Pauly, "Sparse MRI: the application of compressed
sensing for rapid MRI", MRM 58, 1182 (2007).

Pruessmann et al., "SENSE: sensitivity encoding for fast MRI",
MRM 42, 952 (1999).

Handheld probe architecture document, §10 and Appendix B.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# ── Physical constants ────────────────────────────────────────────
_GAMMA_P: float = 2.0 * math.pi * 42.577e6  # rad/s/T (proton)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Result data classes                                             ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class ReconResult:
    """Result of a 2D k-space reconstruction.

    Attributes
    ----------
    image:
        Reconstructed image (complex or magnitude), shape (Ny, Nx).
    magnitude:
        Magnitude image (always real), shape (Ny, Nx).
    fov_x_m:
        Field of view in x (m).
    fov_y_m:
        Field of view in y (m).
    resolution_x_mm:
        Pixel size in x (mm).
    resolution_y_mm:
        Pixel size in y (mm).
    n_iterations:
        Number of solver iterations used (1 for FFT; >1 for CS).
    method:
        Reconstruction method name.
    snr_db:
        Estimated signal-to-noise ratio in the reconstructed image (dB).
        Computed as 20·log₁₀(|signal_peak| / noise_std); NaN if noise
        region not computable.
    """

    image: NDArray
    magnitude: NDArray
    fov_x_m: float
    fov_y_m: float
    resolution_x_mm: float
    resolution_y_mm: float
    n_iterations: int
    method: str
    snr_db: float


@dataclass(frozen=True)
class DepthProfileResult:
    """Result of 1-D depth profile reconstruction.

    Attributes
    ----------
    depths_mm:
        Depth axis (mm).  Shape (Nd,).
    signal:
        Complex echo signal vs. depth.  Shape (Nd,).
    magnitude:
        |signal|.  Shape (Nd,).
    depths_mm_resolved:
        Depths where |signal| > threshold × peak.
    peak_depth_mm:
        Depth of the signal peak (mm).
    t2_star_ms:
        Apparent T₂* estimated from FID decay (ms); NaN if not
        determinable.
    gradient_t_per_m:
        Gradient used for depth encoding (T/m).
    resolution_mm:
        Spatial resolution = 1 / (2 × k_max) / (γ/2π) / gradient (mm).
    """

    depths_mm: NDArray
    signal: NDArray
    magnitude: NDArray
    depths_mm_resolved: NDArray
    peak_depth_mm: float
    t2_star_ms: float
    gradient_t_per_m: float
    resolution_mm: float


@dataclass(frozen=True)
class GriddingResult:
    """Intermediate result of the gridding step.

    Attributes
    ----------
    kspace_grid:
        Gridded k-space on a Cartesian grid.  Shape (Ny, Nx).
    density_weights:
        Density compensation weights applied.  Shape (n_samples,).
    n_samples:
        Number of input k-space samples.
    grid_size:
        (Ny, Nx) of the output grid.
    """

    kspace_grid: NDArray
    density_weights: NDArray
    n_samples: int
    grid_size: tuple[int, int]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  FFT reconstruction (Cartesian k-space)                          ║
# ╚══════════════════════════════════════════════════════════════════╝


def reconstruct_fft(
    kspace: NDArray,
    *,
    fov_x_m: float = 0.08,
    fov_y_m: float = 0.08,
    apply_hamming: bool = True,
) -> ReconResult:
    """Reconstruct a 2D image from Cartesian k-space data via FFT.

    Parameters
    ----------
    kspace:
        Complex k-space data, shape (Ny, Nx).  Row index = ky,
        column index = kx.  Assumed to be sampled on a uniform
        Cartesian grid centred at (kx=0, ky=0).
    fov_x_m:
        Field of view in x (m).
    fov_y_m:
        Field of view in y (m).
    apply_hamming:
        Apply a 2D Hamming window before the FFT to reduce Gibbs ringing.

    Returns
    -------
    ReconResult
    """
    kspace = np.asarray(kspace, dtype=complex)
    ny, nx = kspace.shape

    if apply_hamming:
        wx = np.hamming(nx)
        wy = np.hamming(ny)
        kspace = kspace * np.outer(wy, wx)

    # Shift so DC is at centre, IFFT, shift image back to (0,0)
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))

    res_x_mm = fov_x_m / nx * 1e3
    res_y_mm = fov_y_m / ny * 1e3

    magnitude = np.abs(image)
    snr = _estimate_snr(magnitude)

    return ReconResult(
        image=image,
        magnitude=magnitude,
        fov_x_m=fov_x_m,
        fov_y_m=fov_y_m,
        resolution_x_mm=res_x_mm,
        resolution_y_mm=res_y_mm,
        n_iterations=1,
        method="fft",
        snr_db=snr,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Gridding (non-Cartesian → Cartesian)                             ║
# ╚══════════════════════════════════════════════════════════════════╝


def grid_kspace(
    kx_per_m: NDArray,
    ky_per_m: NDArray,
    samples: NDArray,
    *,
    grid_size: tuple[int, int] = (64, 64),
    fov_x_m: float = 0.08,
    fov_y_m: float = 0.08,
    kernel_width: float = 3.0,
) -> GriddingResult:
    """Grid non-Cartesian k-space onto a uniform Cartesian grid.

    Uses a nearest-neighbour + density-compensation gridding approach
    implemented with pure NumPy (no external dependencies).

    Parameters
    ----------
    kx_per_m, ky_per_m:
        Non-Cartesian k-space sample locations (m⁻¹).  Shape (N,).
    samples:
        Complex k-space values.  Shape (N,).
    grid_size:
        (Ny, Nx) of the output Cartesian grid.
    fov_x_m, fov_y_m:
        Field of view (m).  Determines k-space grid spacing.
    kernel_width:
        Gridding kernel half-width in grid pixels.  Larger → smoother
        but lower spatial resolution.

    Returns
    -------
    GriddingResult
    """
    if isinstance(grid_size, int):
        ny = nx = grid_size
    else:
        ny, nx = grid_size
    kx = np.asarray(kx_per_m, dtype=float)
    ky = np.asarray(ky_per_m, dtype=float)
    s = np.asarray(samples, dtype=complex)

    # k-space grid spacing
    dkx = 1.0 / fov_x_m   # m⁻¹
    dky = 1.0 / fov_y_m   # m⁻¹

    # Normalise k to grid pixel coordinates (centre = Nx/2, Ny/2)
    ix = kx / dkx + nx / 2.0
    iy = ky / dky + ny / 2.0

    # Simple density compensation: voronoi-like cell count approximation
    # (histogram of sample density → weight ∝ 1/density)
    hist, xedges, yedges = np.histogram2d(ix, iy, bins=[nx, ny])
    # Interpolate density weight for each sample
    xi_clip = np.clip(np.floor(ix).astype(int), 0, nx - 1)
    yi_clip = np.clip(np.floor(iy).astype(int), 0, ny - 1)
    density = hist[xi_clip, yi_clip]
    density = np.where(density < 1, 1.0, density)
    density_weights = 1.0 / density

    # Accumulate into grid using nearest-neighbour kernel
    grid = np.zeros((ny, nx), dtype=complex)
    weight_sum = np.zeros((ny, nx), dtype=float)
    half = int(math.ceil(kernel_width))

    for k_idx in range(len(kx)):
        x0 = int(round(ix[k_idx]))
        y0 = int(round(iy[k_idx]))
        w = density_weights[k_idx]
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                xi = x0 + dx
                yi = y0 + dy
                if 0 <= xi < nx and 0 <= yi < ny:
                    dist2 = dx**2 + dy**2
                    kern = math.exp(-0.5 * dist2 / (kernel_width**2))
                    grid[yi, xi] += s[k_idx] * w * kern
                    weight_sum[yi, xi] += w * kern

    # Normalise by accumulated weight
    mask = weight_sum > 0
    grid[mask] /= weight_sum[mask]

    return GriddingResult(
        kspace_grid=grid,
        density_weights=density_weights,
        n_samples=len(kx),
        grid_size=(ny, nx),
    )


def reconstruct_gridding(
    kx_per_m: NDArray,
    ky_per_m: NDArray,
    samples: NDArray,
    *,
    grid_size: tuple[int, int] = (64, 64),
    fov_x_m: float = 0.08,
    fov_y_m: float = 0.08,
    apply_hamming: bool = True,
) -> ReconResult:
    """Reconstruct from non-Cartesian k-space using gridding + FFT.

    Parameters
    ----------
    kx_per_m, ky_per_m:
        k-space sample coordinates (m⁻¹).
    samples:
        Complex k-space values.
    grid_size:
        Reconstruction matrix size (Ny, Nx).
    fov_x_m, fov_y_m:
        Field of view (m).
    apply_hamming:
        Window the gridded k-space before FFT.

    Returns
    -------
    ReconResult
    """
    gridding = grid_kspace(
        kx_per_m,
        ky_per_m,
        samples,
        grid_size=grid_size,
        fov_x_m=fov_x_m,
        fov_y_m=fov_y_m,
    )
    result = reconstruct_fft(
        gridding.kspace_grid,
        fov_x_m=fov_x_m,
        fov_y_m=fov_y_m,
        apply_hamming=apply_hamming,
    )
    # Wrap in new result with gridding method label
    return ReconResult(
        image=result.image,
        magnitude=result.magnitude,
        fov_x_m=result.fov_x_m,
        fov_y_m=result.fov_y_m,
        resolution_x_mm=result.resolution_x_mm,
        resolution_y_mm=result.resolution_y_mm,
        n_iterations=1,
        method="gridding+fft",
        snr_db=result.snr_db,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Wavelet transform utilities (sparsifying basis for CS)          ║
# ╚══════════════════════════════════════════════════════════════════╝


def haar_wavelet_transform(image: NDArray, levels: int = 2) -> NDArray:
    """Forward multi-level 2D Haar wavelet transform.

    Parameters
    ----------
    image:
        2D array (Ny, Nx).  Size must be divisible by 2^levels.
    levels:
        Number of decomposition levels.

    Returns
    -------
    NDArray
        Wavelet coefficient array, same shape as image.
    """
    coeffs = np.array(image, dtype=complex)
    ny, nx = coeffs.shape
    cur_ny, cur_nx = ny, nx
    for _ in range(levels):
        half_nx = cur_nx // 2
        half_ny = cur_ny // 2
        # Horizontal transform: compute both bands before any write (even/odd are views)
        sub = coeffs[:cur_ny, :cur_nx]
        even = sub[:, 0::2]
        odd  = sub[:, 1::2]
        low_h  = (even + odd) / math.sqrt(2)    # materialized copy
        high_h = (even - odd) / math.sqrt(2)    # materialized copy (even still original)
        coeffs[:cur_ny, :half_nx]      = low_h
        coeffs[:cur_ny, half_nx:cur_nx] = high_h
        # Vertical transform: compute both bands before any write
        sub  = coeffs[:cur_ny, :cur_nx]
        even = sub[0::2, :]
        odd  = sub[1::2, :]
        low_v  = (even + odd) / math.sqrt(2)
        high_v = (even - odd) / math.sqrt(2)
        coeffs[:half_ny, :cur_nx]       = low_v
        coeffs[half_ny:cur_ny, :cur_nx] = high_v
        cur_ny = half_ny
        cur_nx = half_nx
    return coeffs


def haar_wavelet_inverse(coeffs: NDArray, levels: int = 2) -> NDArray:
    """Inverse multi-level 2D Haar wavelet transform.

    Parameters
    ----------
    coeffs:
        Wavelet coefficient array (Ny, Nx).
    levels:
        Number of reconstruction levels (must match forward transform).

    Returns
    -------
    NDArray
        Reconstructed image array, same shape as coeffs.
    """
    image = np.array(coeffs, dtype=complex)
    ny_full, nx_full = image.shape
    # Start from the coarsest level and work back up
    cur_ny = ny_full // (2 ** levels)
    cur_nx = nx_full // (2 ** levels)
    for _ in range(levels):
        cur_ny *= 2
        cur_nx *= 2
        half_ny = cur_ny // 2
        half_nx = cur_nx // 2
        # Vertical inverse on current subband
        approx_v = image[:half_ny, :cur_nx].copy()
        detail_v = image[half_ny:cur_ny, :cur_nx].copy()
        image[:cur_ny:2, :cur_nx] = (approx_v + detail_v) / math.sqrt(2)
        image[1:cur_ny:2, :cur_nx] = (approx_v - detail_v) / math.sqrt(2)
        # Horizontal inverse on current subband
        approx_h = image[:cur_ny, :half_nx].copy()
        detail_h = image[:cur_ny, half_nx:cur_nx].copy()
        image[:cur_ny, :cur_nx:2] = (approx_h + detail_h) / math.sqrt(2)
        image[:cur_ny, 1:cur_nx:2] = (approx_h - detail_h) / math.sqrt(2)
    return image


def soft_threshold(x: NDArray, threshold: float) -> NDArray:
    """Complex soft-thresholding operator.

    Parameters
    ----------
    x:
        Array of complex values.
    threshold:
        Threshold magnitude.

    Returns
    -------
    NDArray
        Thresholded array.
    """
    mag = np.abs(x)
    scale = np.where(mag > threshold, (mag - threshold) / mag, 0.0)
    return x * scale


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Compressed sensing reconstruction (ISTA)                        ║
# ╚══════════════════════════════════════════════════════════════════╝


def reconstruct_compressed_sensing(
    kspace_sampled: NDArray,
    sampling_mask: NDArray,
    *,
    fov_x_m: float = 0.08,
    fov_y_m: float = 0.08,
    n_iterations: int = 50,
    lambda_reg: float = 0.01,
    wavelet_levels: int = 2,
    step_size: float = 1.0,
) -> ReconResult:
    """Reconstruct under-sampled Cartesian k-space via ISTA.

    Iterative Shrinkage/Thresholding Algorithm minimises:

        J(ρ) = ‖M(Fρ − s)‖₂² + λ ‖Ψρ‖₁

    where M = sampling mask, F = 2D FFT, Ψ = Haar wavelet.

    Parameters
    ----------
    kspace_sampled:
        Under-sampled k-space.  Zero at unacquired locations.
        Shape (Ny, Nx).
    sampling_mask:
        Boolean mask; True = acquired sample.  Shape (Ny, Nx).
    fov_x_m, fov_y_m:
        Field of view (m).
    n_iterations:
        Number of ISTA iterations.
    lambda_reg:
        Regularisation weight.  Larger → sparser image.
    wavelet_levels:
        Haar wavelet decomposition levels.  Image size must be
        divisible by 2^wavelet_levels.
    step_size:
        Gradient descent step size (≤ 1/L where L = Lipschitz constant
        of ‖M·F·ρ‖₂² term; 1.0 is valid for normalised F).

    Returns
    -------
    ReconResult
    """
    ks = np.asarray(kspace_sampled, dtype=complex)
    mask = np.asarray(sampling_mask, dtype=bool)
    ny, nx = ks.shape

    # Initialise with zero-filled FFT estimate
    rho = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ks)))

    for _ in range(n_iterations):
        # Forward step: gradient of data fidelity
        kspace_current = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(rho)))
        residual = np.where(mask, kspace_current - ks, 0.0)
        grad = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(residual)))

        # Gradient descent step
        rho_update = rho - step_size * grad

        # Proximal (soft-threshold in wavelet domain)
        wavelet_coeffs = haar_wavelet_transform(rho_update, levels=wavelet_levels)
        wavelet_thresh = soft_threshold(wavelet_coeffs, lambda_reg * step_size)
        rho = haar_wavelet_inverse(wavelet_thresh, levels=wavelet_levels)

    magnitude = np.abs(rho)
    snr = _estimate_snr(magnitude)
    res_x_mm = fov_x_m / nx * 1e3
    res_y_mm = fov_y_m / ny * 1e3

    return ReconResult(
        image=rho,
        magnitude=magnitude,
        fov_x_m=fov_x_m,
        fov_y_m=fov_y_m,
        resolution_x_mm=res_x_mm,
        resolution_y_mm=res_y_mm,
        n_iterations=n_iterations,
        method="compressed_sensing_ista",
        snr_db=snr,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  1-D depth profile reconstruction                                ║
# ╚══════════════════════════════════════════════════════════════════╝


def reconstruct_depth_profile(
    time_signal_v: NDArray,
    *,
    dwell_us: float = 5.0,
    gradient_t_per_m: float = 0.005,
    b0_tesla: float = 0.050,
    apply_apodization: bool = True,
    signal_threshold: float = 0.1,
) -> DepthProfileResult:
    """Reconstruct a 1-D depth profile from a free-induction decay (FID).

    The B₀ gradient maps depth to frequency:
        f(z) = γ_p / (2π) × G × z   (relative to Larmor frequency)

    An FT of the time-domain FID gives signal vs. frequency, which
    maps to signal vs. depth via:
        z [m] = f [Hz] / (γ_p/(2π) × G [T/m])

    Parameters
    ----------
    time_signal_v:
        Complex baseband FID signal (V).  Shape (Nt,).
    dwell_us:
        ADC dwell time (µs).
    gradient_t_per_m:
        Frequency-encoding gradient (T/m).
    b0_tesla:
        Static field at probe face (T; used for Larmor frequency report).
    apply_apodization:
        Apply exponential apodization (equivalent to Lorentzian broadening)
        before FFT to reduce noise spikes.
    signal_threshold:
        Threshold relative to peak magnitude for `depths_mm_resolved`.

    Returns
    -------
    DepthProfileResult
    """
    sig = np.asarray(time_signal_v, dtype=complex)
    nt = len(sig)
    dwell_s = dwell_us * 1e-6
    acq_time_s = nt * dwell_s

    if apply_apodization:
        # Exponential decay: τ = acquisition time / 3
        tau = acq_time_s / 3.0
        t = np.arange(nt) * dwell_s
        sig = sig * np.exp(-t / tau)

    # FFT → spectrum (shift DC to centre)
    spectrum = np.fft.fftshift(np.fft.fft(sig))

    # Frequency axis (Hz)
    freqs_hz = np.fft.fftshift(np.fft.fftfreq(nt, d=dwell_s))

    # Map frequency → depth
    #   Δf = (γ/2π) × G × Δz  →  Δz = Δf / ((γ/2π) × G)
    gamma_over_2pi = _GAMMA_P / (2.0 * math.pi)
    hz_per_m = gamma_over_2pi * gradient_t_per_m
    depths_m = freqs_hz / hz_per_m
    depths_mm = depths_m * 1e3

    magnitude = np.abs(spectrum)
    peak_idx = int(np.argmax(magnitude))
    peak_depth_mm = float(depths_mm[peak_idx])

    # T2* from exponential fit to |FID| envelope
    t2_star_ms = _estimate_t2_star(np.abs(sig), dwell_s)

    # Resolved depths (above threshold)
    threshold_val = signal_threshold * float(magnitude[peak_idx])
    resolved_mask = magnitude > threshold_val
    depths_resolved = depths_mm[resolved_mask]

    # Spatial resolution = 1 / (2 × k_max) converted to depth
    # k_max = (γ/2π) × G × acq_time_s / 2
    k_max_per_m = gamma_over_2pi * gradient_t_per_m * acq_time_s / 2.0
    resolution_m = 1.0 / (2.0 * k_max_per_m) if k_max_per_m > 0 else float("inf")
    resolution_mm = resolution_m * 1e3

    return DepthProfileResult(
        depths_mm=depths_mm,
        signal=spectrum,
        magnitude=magnitude,
        depths_mm_resolved=depths_resolved,
        peak_depth_mm=peak_depth_mm,
        t2_star_ms=t2_star_ms,
        gradient_t_per_m=gradient_t_per_m,
        resolution_mm=resolution_mm,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Utility functions                                               ║
# ╚══════════════════════════════════════════════════════════════════╝


def _estimate_snr(magnitude: NDArray) -> float:
    """Estimate SNR from a magnitude image (dB).

    Signal = peak magnitude.
    Noise = std of the lowest-intensity 25% of pixels (background region).

    Returns NaN if the image is flat (all zeros or all equal).
    """
    if magnitude.size == 0:
        return float("nan")
    flat = magnitude.ravel()
    threshold = np.percentile(flat, 25)
    noise_region = flat[flat <= threshold]
    if len(noise_region) < 4:
        return float("nan")
    noise_std = float(np.std(noise_region))
    if noise_std == 0.0:
        return float("nan")
    peak = float(np.max(flat))
    return 20.0 * math.log10(peak / noise_std) if peak > 0 else float("nan")


def _estimate_t2_star(envelope: NDArray, dwell_s: float) -> float:
    """Estimate T₂* from magnitude FID envelope via log-linear fit (ms).

    Returns NaN if the fit fails or the signal is below noise floor.
    """
    if len(envelope) < 4:
        return float("nan")
    env = np.asarray(envelope, dtype=float)
    env = np.where(env > 1e-30, env, 1e-30)
    log_env = np.log(env)
    nt = len(log_env)
    t = np.arange(nt) * dwell_s
    # Linear regression: log(E) = log(E0) − t/T2*
    try:
        slope, _ = np.polyfit(t, log_env, 1)
    except Exception:
        return float("nan")
    if slope >= 0:
        return float("nan")
    t2_star_s = -1.0 / slope
    return t2_star_s * 1e3  # ms


def simulate_kspace(
    image: NDArray,
    *,
    fov_x_m: float = 0.08,
    fov_y_m: float = 0.08,
    snr_db: float | None = None,
) -> NDArray:
    """Forward model: image → k-space (for testing).

    Parameters
    ----------
    image:
        Ground-truth image.  Shape (Ny, Nx).
    fov_x_m, fov_y_m:
        Field of view (m).
    snr_db:
        If given, add white complex Gaussian noise to k-space to give
        this SNR (dB) in the reconstructed image.

    Returns
    -------
    NDArray
        Complex k-space data, shape (Ny, Nx).
    """
    img = np.asarray(image, dtype=complex)
    ks = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img)))

    if snr_db is not None:
        signal_power = float(np.mean(np.abs(ks) ** 2))
        noise_power = signal_power / (10 ** (snr_db / 10))
        sigma = math.sqrt(noise_power / 2)
        rng = np.random.default_rng(seed=42)
        noise = rng.standard_normal(ks.shape) + 1j * rng.standard_normal(ks.shape)
        ks = ks + sigma * noise

    return ks


def apply_undersampling_mask(
    kspace: NDArray,
    acceleration_factor: float = 2.0,
    *,
    keep_centre_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[NDArray, NDArray]:
    """Create and apply a pseudo-random undersampling mask.

    Parameters
    ----------
    kspace:
        Full k-space data, shape (Ny, Nx).
    acceleration_factor:
        Desired acceleration (e.g. 2.0 → keep 50% of k-space lines).
    keep_centre_fraction:
        Fraction of central k-space lines always retained.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    kspace_under:
        Undersampled k-space (zeros at unacquired positions).
    mask:
        Boolean mask, True = acquired.
    """
    ks = np.asarray(kspace, dtype=complex)
    ny, nx = ks.shape

    rng = np.random.default_rng(seed)
    mask = np.zeros((ny, nx), dtype=bool)

    # Always keep central lines
    n_centre = max(1, int(round(ny * keep_centre_fraction)))
    centre_start = ny // 2 - n_centre // 2
    centre_end = centre_start + n_centre
    mask[centre_start:centre_end, :] = True

    # Randomly select additional lines
    n_keep_total = max(1, int(round(ny / acceleration_factor)))
    remaining = n_keep_total - n_centre
    if remaining > 0:
        outer_lines = [i for i in range(ny) if i < centre_start or i >= centre_end]
        chosen = rng.choice(outer_lines, size=min(remaining, len(outer_lines)), replace=False)
        mask[chosen, :] = True

    return np.where(mask, ks, 0.0), mask


def estimate_acceleration_factor(mask: NDArray) -> float:
    """Estimate the k-space acceleration factor from a sampling mask.

    Parameters
    ----------
    mask:
        Boolean array, True = acquired.

    Returns
    -------
    float
        Acceleration = total_samples / acquired_samples.
    """
    total = mask.size
    acquired = int(np.sum(mask))
    if acquired == 0:
        return float("inf")
    return total / acquired


def image_snr_from_phantom(
    magnitude: NDArray,
    *,
    signal_fraction: float = 0.1,
    noise_fraction: float = 0.25,
) -> float:
    """Measure SNR in a phantom image.

    Parameters
    ----------
    magnitude:
        Magnitude image (Ny, Nx).
    signal_fraction:
        Fraction of pixels (by magnitude) considered signal.
    noise_fraction:
        Fraction of lowest-magnitude pixels used as noise floor.

    Returns
    -------
    float
        SNR in dB.
    """
    flat = magnitude.ravel()
    n_sig = max(1, int(round(len(flat) * signal_fraction)))
    n_noise = max(1, int(round(len(flat) * noise_fraction)))
    signal_pixels = np.sort(flat)[-n_sig:]
    noise_pixels = np.sort(flat)[:n_noise]
    sig_mean = float(np.mean(signal_pixels))
    noise_std = float(np.std(noise_pixels))
    if noise_std == 0.0:
        return float("nan")
    return 20.0 * math.log10(sig_mean / noise_std)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Parametric sweeps                                               ║
# ╚══════════════════════════════════════════════════════════════════╝


def sweep_snr_vs_acceleration(
    image: NDArray,
    acceleration_factors: list[float],
    *,
    fov_x_m: float = 0.08,
    fov_y_m: float = 0.08,
    n_iterations: int = 30,
    lambda_reg: float = 0.01,
) -> NDArray:
    """Reconstructed SNR vs. k-space acceleration factor.

    Parameters
    ----------
    image:
        Ground-truth image (Ny, Nx).  Size must be divisible by 4
        (for wavelet levels=2).
    acceleration_factors:
        List of acceleration factors to evaluate.
    fov_x_m, fov_y_m:
        Field of view (m).
    n_iterations:
        CS iterations per reconstruction.
    lambda_reg:
        CS regularisation weight.

    Returns
    -------
    NDArray
        SNR (dB) for each acceleration factor.  Shape (len(factors),).
    """
    kspace = simulate_kspace(image, fov_x_m=fov_x_m, fov_y_m=fov_y_m)
    snrs = []
    for af in acceleration_factors:
        ks_under, mask = apply_undersampling_mask(kspace, acceleration_factor=af)
        result = reconstruct_compressed_sensing(
            ks_under,
            mask,
            fov_x_m=fov_x_m,
            fov_y_m=fov_y_m,
            n_iterations=n_iterations,
            lambda_reg=lambda_reg,
        )
        snrs.append(result.snr_db)
    return np.array(snrs)


def sweep_resolution_vs_fov(
    n_pixels: int = 64,
    fov_values_m: list[float] | None = None,
) -> NDArray:
    """Pixel size (mm) as a function of FOV for fixed pixel count.

    Parameters
    ----------
    n_pixels:
        Number of pixels per side.
    fov_values_m:
        FOV values to evaluate (m).

    Returns
    -------
    NDArray
        Pixel size (mm) for each FOV.
    """
    if fov_values_m is None:
        fov_values_m = [0.04, 0.06, 0.08, 0.10, 0.12]
    return np.array([fov / n_pixels * 1e3 for fov in fov_values_m])
