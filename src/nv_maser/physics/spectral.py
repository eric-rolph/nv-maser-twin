"""
Frequency-resolved spin inversion profile for inhomogeneous NV ensembles.

Models the spectral distribution p(Δ) of spin inversion across the
inhomogeneously broadened line, where Δ = ω_spin − ω_cavity is the
detuning from cavity resonance.

The inhomogeneous lineshape is a q-Gaussian (Tsallis distribution),
which interpolates smoothly between Gaussian (q = 1) and Lorentzian (q → 2).

Key physics:
    - **Spectral hole burning**: on-resonance spins are depleted faster by
      stimulated emission, creating a spectral hole at Δ ≈ 0.
    - **Spectral refilling**: dipole-dipole interactions (modelled in
      ``dipolar.py``) transport inversion from off-resonant wings into
      the hole, enabling repeated superradiant bursts.

References
──────────
Kersten et al., "Self-induced superradiant masing …",
    Nature Physics (2026), PMC12811124.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..config import NVConfig, SpectralConfig


def build_detuning_grid(config: SpectralConfig) -> NDArray[np.float64]:
    """Create the frequency detuning grid Δ/(2π) in Hz.

    Returns an array of shape ``(n_freq_bins,)`` spanning
    [−freq_range, +freq_range] in Hz, centred on Δ = 0.
    """
    half_range_hz = config.freq_range_mhz * 1e6
    return np.linspace(-half_range_hz, half_range_hz, config.n_freq_bins)


def q_gaussian(
    delta: NDArray[np.float64],
    sigma_hz: float,
    q: float = 1.0,
) -> NDArray[np.float64]:
    r"""Normalised q-Gaussian (Tsallis) distribution.

    .. math::

        f_q(\Delta) = \frac{C_q}{\sigma}
            \left[1 - (1 - q)\,\frac{\Delta^2}{2\sigma^2}\right]_+^{1/(1-q)}

    where :math:`[x]_+ = \max(x, 0)`.

    * q = 1  → standard Gaussian
    * q → 2  → Lorentzian (Cauchy)

    The distribution is normalised so that ∫ f_q dΔ = 1.

    Args:
        delta: Detuning array (Hz).
        sigma_hz: Width parameter σ (Hz).  Related to FWHM by a
            q-dependent factor; for q = 1, FWHM = 2σ√(2 ln 2).
        q: Shape parameter, 1 ≤ q < 2.

    Returns:
        Probability density at each Δ, same shape as *delta*.
    """
    if sigma_hz <= 0:
        raise ValueError("sigma_hz must be positive")

    x2 = (delta / sigma_hz) ** 2

    if abs(q - 1.0) < 1e-12:
        # Pure Gaussian
        pdf = np.exp(-0.5 * x2)
    else:
        base = 1.0 - (1.0 - q) * 0.5 * x2
        np.maximum(base, 0.0, out=base)
        pdf = base ** (1.0 / (1.0 - q))

    # Normalise numerically
    dx = delta[1] - delta[0] if len(delta) > 1 else 1.0
    total = np.sum(pdf) * dx
    if total > 0:
        pdf /= total
    return pdf


def fwhm_to_sigma(fwhm_hz: float, q: float = 1.0) -> float:
    """Convert FWHM (Hz) to q-Gaussian σ parameter.

    For q = 1 (Gaussian): FWHM = 2σ√(2 ln 2).
    For q → 2 (Lorentzian): FWHM = 2σ.
    General formula derived from the half-maximum condition.
    """
    if abs(q - 1.0) < 1e-12:
        return fwhm_hz / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    # Half-maximum condition: [1 − (1−q) Δ²/(2σ²)]^{1/(1−q)} = 0.5
    # => Δ_hm² = 2σ²/(1−q) · (1 − 0.5^{1−q})
    exponent = 1.0 - q
    half_sq = 2.0 * (1.0 - 0.5**exponent) / (1.0 - q)
    if half_sq <= 0:
        return fwhm_hz / 2.0
    return fwhm_hz / (2.0 * math.sqrt(half_sq))


def build_initial_inversion(
    nv_config: NVConfig,
    spectral_config: SpectralConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the initial frequency-resolved inversion profile p(Δ).

    The profile is p(Δ) = sz₀ · g(Δ) / g(0), normalised so that
    the on-resonance inversion equals sz₀ = pump_efficiency / 2
    and the shape follows the inhomogeneous lineshape.

    Returns:
        (delta_hz, p_delta): detuning grid (Hz) and inversion profile.
        p_delta has the same shape as delta_hz; its on-resonance value
        equals sz₀.
    """
    delta = build_detuning_grid(spectral_config)
    sigma = fwhm_to_sigma(
        spectral_config.inhomogeneous_linewidth_mhz * 1e6,
        spectral_config.q_parameter,
    )
    lineshape = q_gaussian(delta, sigma, spectral_config.q_parameter)

    # Normalise so peak = sz0
    sz0 = nv_config.pump_efficiency / 2.0
    peak = lineshape.max()
    if peak > 0:
        p_delta = sz0 * (lineshape / peak)
    else:
        p_delta = np.full_like(delta, sz0)

    return delta, p_delta


def compute_on_resonance_inversion(
    p_delta: NDArray[np.float64],
    delta_hz: NDArray[np.float64],
    cavity_linewidth_hz: float,
) -> float:
    """Average inversion within the cavity linewidth.

    Computes the mean of p(Δ) over the frequency window
    |Δ| ≤ κ_c / 2, which determines the effective inversion
    seen by the cavity mode.

    Args:
        p_delta: Inversion profile.
        delta_hz: Detuning grid (Hz).
        cavity_linewidth_hz: Full cavity linewidth κ_c/(2π) in Hz.

    Returns:
        Mean inversion within the cavity bandwidth.
    """
    half_kappa = cavity_linewidth_hz / 2.0
    mask = np.abs(delta_hz) <= half_kappa
    if not np.any(mask):
        # Grid too coarse — return value at closest-to-zero bin
        idx = np.argmin(np.abs(delta_hz))
        return float(p_delta[idx])
    return float(np.mean(p_delta[mask]))


def burn_spectral_hole(
    p_delta: NDArray[np.float64],
    delta_hz: NDArray[np.float64],
    cavity_linewidth_hz: float,
    burn_depth: float,
) -> NDArray[np.float64]:
    """Deplete on-resonance inversion (spectral hole burning).

    Applies a Lorentzian-shaped depletion centred at Δ = 0 with width
    equal to the cavity linewidth.  Mimics the selective depletion of
    spins resonant with the cavity mode during stimulated emission.

    Args:
        p_delta: Current inversion profile.
        delta_hz: Detuning grid (Hz).
        cavity_linewidth_hz: Cavity linewidth (Hz).
        burn_depth: Maximum fractional depletion at Δ = 0
            (0 = no burn, 1 = complete depletion).

    Returns:
        Updated inversion profile (new array, input is not modified).
    """
    burn_depth = max(0.0, min(1.0, burn_depth))
    half_kappa = cavity_linewidth_hz / 2.0
    lorentzian = half_kappa**2 / (delta_hz**2 + half_kappa**2)
    return p_delta * (1.0 - burn_depth * lorentzian)


def spectral_overlap_weights(
    delta_hz: NDArray[np.float64],
    cavity_linewidth_hz: float,
) -> NDArray[np.float64]:
    """Lorentzian spectral weight for each detuning bin.

    Returns w(Δ) = (κ_c/2)² / (Δ² + (κ_c/2)²), the relative coupling
    strength of each detuning class to the cavity mode.

    Args:
        delta_hz: Detuning grid (Hz).
        cavity_linewidth_hz: Cavity linewidth (Hz).

    Returns:
        Weight array, same shape as delta_hz.  Peak value = 1.0 at Δ = 0.
    """
    half_kappa = cavity_linewidth_hz / 2.0
    return half_kappa**2 / (delta_hz**2 + half_kappa**2)
