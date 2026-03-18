"""
ODMR spectrum simulator — inspired by NVision (shoham-b/NVision).

Bridges our field-level physics (nv_spin, maser_gain) with the experimental
ODMR observable.  The key models:

  S(f) = background − [L_left + L_center + L_right]

Each dip is a Lorentzian or Voigt (Lorentzian⊗Gaussian) profile accounting
for hyperfine splitting of the NV⁻ ground-state triplet.

Patterns borrowed from NVision's architecture:
  • Lorentzian + Voigt lineshape (signal/nv_center.py)
  • Composable noise (Poisson shot noise, temporal drift)
  • Cross-validation between analytical models and spectral fits
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import wofz
from scipy.optimize import curve_fit

from ..config import NVConfig
from .nv_spin import (
    transition_frequencies,
    effective_linewidth_ghz,
    homogeneous_linewidth_ghz,
)


# ── Data containers ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ODMRResult:
    """Result of an ODMR sweep simulation.

    Attributes
    ----------
    frequencies_ghz : array, shape (N,)
        Microwave frequencies swept.
    signal : array, shape (N,)
        Normalised fluorescence signal (1 = background, dips < 1).
    noisy_signal : array, shape (N,) or None
        Signal with noise applied (None if no noise requested).
    center_freq_ghz : float
        Central transition frequency (ν−).
    linewidth_ghz : float
        Effective linewidth used for the simulation.
    contrast : float
        Depth of the deepest dip relative to background.
    """

    frequencies_ghz: NDArray[np.float64]
    signal: NDArray[np.float64]
    noisy_signal: NDArray[np.float64] | None
    center_freq_ghz: float
    linewidth_ghz: float
    contrast: float


@dataclass(frozen=True)
class FitResult:
    """Result of fitting a Lorentzian model to an ODMR spectrum.

    Attributes
    ----------
    center_freq_ghz : float
        Fitted central transition frequency.
    linewidth_ghz : float
        Fitted HWHM linewidth.
    splitting_ghz : float
        Fitted hyperfine splitting (0 if single-dip fit).
    contrast : float
        Fitted dip depth.
    residual_rms : float
        RMS residual of the fit.
    params : dict
        All fitted parameters.
    """

    center_freq_ghz: float
    linewidth_ghz: float
    splitting_ghz: float
    contrast: float
    residual_rms: float
    params: dict


@dataclass(frozen=True)
class CrossValidation:
    """Cross-validation between analytical linewidth and spectral fit.

    Attributes
    ----------
    analytical_linewidth_ghz : float
        From effective_linewidth_ghz().
    fitted_linewidth_ghz : float
        From Lorentzian fit to simulated ODMR.
    relative_error : float
        |fitted − analytical| / analytical.
    consistent : bool
        True if relative_error < tolerance.
    """

    analytical_linewidth_ghz: float
    fitted_linewidth_ghz: float
    relative_error: float
    consistent: bool


# ── Lineshape functions ──────────────────────────────────────────────

def _lorentzian(f: NDArray, f0: float, gamma: float) -> NDArray:
    """Normalised Lorentzian dip profile (peak = 1/γ²).

    L(f) = 1 / ((f − f0)² + γ²)
    """
    return 1.0 / ((f - f0) ** 2 + gamma ** 2)


def _voigt(f: NDArray, f0: float, gamma_l: float, sigma_g: float) -> NDArray:
    """Voigt profile via Faddeeva function.

    Real part of w(z) where z = (f − f0 + iγ) / (σ√2).
    Falls back to Lorentzian if sigma_g ≈ 0.
    """
    if sigma_g < 1e-15:
        return _lorentzian(f, f0, gamma_l)
    z = ((f - f0) + 1j * gamma_l) / (sigma_g * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma_g * np.sqrt(2.0 * np.pi))


# ── Core ODMR spectrum ───────────────────────────────────────────────

def compute_odmr_spectrum(
    frequencies_ghz: NDArray[np.float64],
    center_freq_ghz: float,
    linewidth_ghz: float,
    *,
    splitting_ghz: float = 0.0,
    k_np: float = 1.0,
    contrast: float = 0.03,
    background: float = 1.0,
    profile: Literal["lorentzian", "voigt"] = "lorentzian",
    gaussian_fwhm_ghz: float = 0.0,
) -> NDArray[np.float64]:
    """Compute clean ODMR fluorescence spectrum.

    Parameters
    ----------
    frequencies_ghz : array
        Microwave frequency sweep points in GHz.
    center_freq_ghz : float
        Central ODMR transition frequency (ν−) in GHz.
    linewidth_ghz : float
        Lorentzian HWHM linewidth in GHz.
    splitting_ghz : float
        ¹⁴N hyperfine splitting (~2.2 MHz = 0.0022 GHz).
    k_np : float
        Non-polarisation factor (amplitude ratio between peaks).
        Left = A/k_np, centre = A, right = A*k_np.
    contrast : float
        Maximum dip depth as a fraction of background (e.g. 0.03 = 3%).
        The raw Lorentzian amplitude is computed internally so that the
        deepest point of the spectrum is ``background − contrast``.
    background : float
        Normalised background fluorescence level.
    profile : str
        "lorentzian" or "voigt".
    gaussian_fwhm_ghz : float
        Gaussian FWHM for Voigt profile (inhomogeneous broadening).
    """
    f = np.asarray(frequencies_ghz, dtype=np.float64)
    gamma = linewidth_ghz

    if profile == "voigt" and gaussian_fwhm_ghz > 0:
        sigma = gaussian_fwhm_ghz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    else:
        sigma = 0.0

    # Compute peak height of a single profile at its centre so we can
    # scale the amplitude parameter to achieve the requested contrast.
    peak_height = float(_voigt(np.array([0.0]), 0.0, gamma, sigma)[0])

    # Total peak multiplier from all 3 dips when they overlap maximally
    total_k = k_np + 1.0 + 1.0 / k_np
    raw_amp = contrast / (total_k * peak_height) if peak_height > 0 else 0.0

    if splitting_ghz < 1e-10:
        dip = total_k * raw_amp * _voigt(f, center_freq_ghz, gamma, sigma)
    else:
        left = (raw_amp / k_np) * _voigt(f, center_freq_ghz - splitting_ghz, gamma, sigma)
        center = raw_amp * _voigt(f, center_freq_ghz, gamma, sigma)
        right = (raw_amp * k_np) * _voigt(f, center_freq_ghz + splitting_ghz, gamma, sigma)
        dip = left + center + right

    return background - dip


# ── Full sweep simulation with noise ─────────────────────────────────

def simulate_odmr_sweep(
    b_field_tesla: float,
    nv_config: NVConfig,
    *,
    n_points: int = 501,
    span_ghz: float = 0.1,
    contrast: float = 0.03,
    splitting_ghz: float = 0.0022,
    k_np: float = 1.0,
    profile: Literal["lorentzian", "voigt"] = "lorentzian",
    gaussian_fwhm_ghz: float = 0.0,
    photon_count: float = 0.0,
    noise_std: float = 0.0,
    drift_per_point: float = 0.0,
    seed: int | None = None,
) -> ODMRResult:
    """Simulate a complete ODMR frequency sweep.

    Parameters
    ----------
    b_field_tesla : float
        Magnetic field magnitude at the NV centre.
    nv_config : NVConfig
        NV centre configuration.
    n_points : int
        Number of frequency points in the sweep.
    span_ghz : float
        Total frequency span centred on the transition.
    contrast : float
        ODMR contrast parameter.
    splitting_ghz : float
        ¹⁴N hyperfine splitting in GHz.
    k_np : float
        Non-polarisation factor.
    profile : str
        Lineshape: "lorentzian" or "voigt".
    gaussian_fwhm_ghz : float
        Gaussian FWHM for Voigt broadening.
    photon_count : float
        Mean photon count for Poisson shot noise (0 = no shot noise).
    noise_std : float
        Additive Gaussian noise standard deviation.
    drift_per_point : float
        Linear drift added per sweep point (temporal baseline drift).
    seed : int or None
        Random seed for reproducibility.
    """
    # Transition frequency from our physics model
    b = np.array([b_field_tesla], dtype=np.float32)
    _, nu_minus = transition_frequencies(b, nv_config)
    center_freq = float(nu_minus[0])

    # Linewidth from T2*
    linewidth = homogeneous_linewidth_ghz(nv_config.t2_star_us)

    # Frequency sweep
    f_start = center_freq - span_ghz / 2.0
    f_end = center_freq + span_ghz / 2.0
    frequencies = np.linspace(f_start, f_end, n_points)

    # Clean spectrum
    signal = compute_odmr_spectrum(
        frequencies,
        center_freq,
        linewidth,
        splitting_ghz=splitting_ghz,
        k_np=k_np,
        contrast=contrast,
        profile=profile,
        gaussian_fwhm_ghz=gaussian_fwhm_ghz,
    )

    # Apply noise
    noisy = None
    if photon_count > 0 or noise_std > 0 or drift_per_point != 0.0:
        rng = np.random.default_rng(seed)
        noisy = signal.copy()

        # Poisson shot noise (NVision pattern)
        if photon_count > 0:
            counts = rng.poisson(noisy * photon_count)
            noisy = counts / photon_count

        # Additive Gaussian noise
        if noise_std > 0:
            noisy = noisy + rng.normal(0, noise_std, size=len(noisy))

        # Linear drift (NVision OverProbeDriftNoise pattern)
        if drift_per_point != 0.0:
            drift = np.arange(len(noisy)) * drift_per_point
            drift -= drift.mean()  # zero-mean so it doesn't bias the background
            noisy = noisy + drift

    # Compute realised contrast
    actual_contrast = float(1.0 - np.min(signal))

    return ODMRResult(
        frequencies_ghz=frequencies,
        signal=signal,
        noisy_signal=noisy,
        center_freq_ghz=center_freq,
        linewidth_ghz=linewidth,
        contrast=actual_contrast,
    )


# ── Spectral fitting ─────────────────────────────────────────────────

def _lorentzian_3dip_model(f, freq, linewidth, split, contrast, k_np, background):
    """3-dip Lorentzian model for curve_fit.

    ``contrast`` is the fractional dip depth (0-1).  The raw Lorentzian
    amplitude is derived internally so the deepest point sits at
    ``background − contrast``.
    """
    gamma = linewidth
    total_k = k_np + 1.0 + 1.0 / k_np
    # Peak height of Lorentzian at centre = 1/γ²
    raw_amp = contrast * gamma ** 2 / total_k

    if split < 1e-10:
        dip = total_k * raw_amp / ((f - freq) ** 2 + gamma ** 2)
    else:
        left = (raw_amp / k_np) / ((f - (freq - split)) ** 2 + gamma ** 2)
        center = raw_amp / ((f - freq) ** 2 + gamma ** 2)
        right = (raw_amp * k_np) / ((f - (freq + split)) ** 2 + gamma ** 2)
        dip = left + center + right
    return background - dip


def fit_odmr_spectrum(
    frequencies_ghz: NDArray[np.float64],
    signal: NDArray[np.float64],
    *,
    initial_freq_ghz: float | None = None,
    fit_splitting: bool = True,
) -> FitResult:
    """Fit a 3-dip Lorentzian model to an ODMR spectrum.

    Parameters
    ----------
    frequencies_ghz : array
        Measured frequency points.
    signal : array
        Measured fluorescence signal.
    initial_freq_ghz : float or None
        Initial guess for centre frequency. If None, uses frequency of
        the minimum signal value.
    fit_splitting : bool
        If True, fit hyperfine splitting. If False, fix to 0 (single dip).
    """
    f = np.asarray(frequencies_ghz, dtype=np.float64)
    s = np.asarray(signal, dtype=np.float64)

    # Initial guesses
    if initial_freq_ghz is None:
        initial_freq_ghz = float(f[np.argmin(s)])
    bg_guess = float(np.median(s))
    dip_depth = bg_guess - float(np.min(s))
    span = f[-1] - f[0]
    gamma_guess = span * 0.01  # 1% of span as initial linewidth

    if fit_splitting:
        contrast_guess = dip_depth / bg_guess if bg_guess > 0 else dip_depth
        p0 = [initial_freq_ghz, gamma_guess, 0.002, contrast_guess, 1.0, bg_guess]
        bounds_low = [f[0], 1e-6, 0.0, 0.0, 0.5, 0.5]
        bounds_high = [f[-1], span * 0.5, span * 0.5, 0.5, 5.0, 1.5]
    else:
        contrast_guess = dip_depth / bg_guess if bg_guess > 0 else dip_depth
        p0 = [initial_freq_ghz, gamma_guess, 0.0, contrast_guess, 1.0, bg_guess]
        bounds_low = [f[0], 1e-6, -1e-12, 0.0, 0.99, 0.5]
        bounds_high = [f[-1], span * 0.5, 1e-12, 0.5, 1.01, 1.5]

    try:
        popt, _ = curve_fit(
            _lorentzian_3dip_model,
            f,
            s,
            p0=p0,
            bounds=(bounds_low, bounds_high),
            maxfev=10000,
        )
    except RuntimeError:
        # Return initial guesses if fit fails
        popt = np.array(p0)

    fitted = _lorentzian_3dip_model(f, *popt)
    rms = float(np.sqrt(np.mean((s - fitted) ** 2)))

    freq_fit, lw_fit, split_fit, contrast_fit, knp_fit, bg_fit = popt
    # Compute actual contrast from fitted params
    peak_signal = _lorentzian_3dip_model(freq_fit, *popt)
    actual_contrast = float(bg_fit - peak_signal)

    return FitResult(
        center_freq_ghz=float(freq_fit),
        linewidth_ghz=float(lw_fit),
        splitting_ghz=float(split_fit),
        contrast=actual_contrast,
        residual_rms=rms,
        params={
            "frequency": float(freq_fit),
            "linewidth": float(lw_fit),
            "splitting": float(split_fit),
            "contrast": float(contrast_fit),
            "k_np": float(knp_fit),
            "background": float(bg_fit),
        },
    )


# ── Cross-validation ─────────────────────────────────────────────────

def cross_validate_linewidth(
    b_field: NDArray[np.float32],
    active_mask: NDArray[np.bool_],
    nv_config: NVConfig,
    *,
    n_points: int = 501,
    contrast: float = 0.03,
    tolerance: float = 0.3,
) -> CrossValidation:
    """Compare analytical linewidth model vs. ODMR spectral fit.

    Generates a clean ODMR spectrum from the mean B-field over the active
    zone, then fits a Lorentzian to extract the linewidth. Compares with
    effective_linewidth_ghz() from nv_spin.py.

    Parameters
    ----------
    b_field : array, (H, W)
        Magnetic field map in Tesla.
    active_mask : array, (H, W)
        Boolean mask for diamond active zone.
    nv_config : NVConfig
        NV centre configuration.
    n_points : int
        Sweep points for the simulation.
    contrast : float
        ODMR contrast parameter.
    tolerance : float
        Relative error threshold for "consistent" flag.
    """
    # Analytical linewidth
    gamma_eff, _, _ = effective_linewidth_ghz(b_field, active_mask, nv_config)

    # Mean field for ODMR simulation
    mean_b = float(np.mean(b_field[active_mask]))

    # Use effective linewidth (including inhomogeneous) for the spectrum
    result = simulate_odmr_sweep(
        mean_b,
        nv_config,
        n_points=n_points,
        contrast=contrast,
        splitting_ghz=0.0,  # single dip for clean linewidth comparison
    )

    # Fit
    fit = fit_odmr_spectrum(
        result.frequencies_ghz,
        result.signal,
        fit_splitting=False,
    )

    # Compare
    if gamma_eff > 0:
        rel_err = abs(fit.linewidth_ghz - gamma_eff) / gamma_eff
    else:
        rel_err = float("inf") if fit.linewidth_ghz > 0 else 0.0

    return CrossValidation(
        analytical_linewidth_ghz=gamma_eff,
        fitted_linewidth_ghz=fit.linewidth_ghz,
        relative_error=rel_err,
        consistent=rel_err < tolerance,
    )
