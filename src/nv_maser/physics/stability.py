"""Oscillator stability and Allan deviation model for the NV diamond maser.

The NV maser oscillates at a microwave frequency ν₀ ≈ 1.47 GHz whose
instantaneous phase undergoes quantum phase diffusion (random walk) seeded
by vacuum fluctuations and spontaneous emission.  The characteristic width
of the resulting Lorentzian output spectrum is the Schawlow-Townes linewidth
Δν_ST.  This module converts Δν_ST (and optionally a full phase noise PSD)
into the **Allan deviation** σ_y(τ) and the corresponding field uncertainty
σ_B(τ) as a function of integration time τ.

Physical picture
────────────────
For white frequency noise — the dominant regime when Δν_ST drives all
instabilities — the Allan deviation has the familiar τ^{−1/2} slope:

    σ_y(τ)  = √(Δν_ST / (2π ν₀² τ))          [dimensionless]
    σ_B(τ)  = σ_y(τ) × ν₀ / γ_e              [T]
             = √(Δν_ST / (2π)) / (γ_e × √τ)
             = η_ST / √τ

where η_ST is the Schawlow-Townes sensitivity (T/√Hz) from ``sensitivity.py``
and γ_e = 28.025 GHz/T is the NV gyromagnetic ratio for the |0⟩→|−1⟩
transition.

At τ = 1 s, σ_B(1) = η_ST, linking this module to
``SensitivityResult.allan_deviation_1s_t``.

Derivation of σ_y(τ) from random-walk phase diffusion
───────────────────────────────────────────────────────
The maser phase φ(t) undergoes a Wiener process with diffusion coefficient
D_φ = 2π Δν_ST (rad²/s).  For two consecutive averaging intervals of
duration τ the two-sample Allan variance is:

    σ_y²(τ) = (1/2)⟨(ȳ₂ − ȳ₁)²⟩

where ȳ_k = Δφ_k / (2π ν₀ τ) is the mean fractional frequency deviation
over the k-th interval.  Since the increments Δφ_k are independent with
variance D_φ τ = 2π Δν_ST τ:

    σ_y²(τ) = D_φ / (4π² ν₀² τ) = Δν_ST / (2π ν₀² τ)

Equivalently, S_y(f) = f²/ν₀² × S_φ(f) = Δν_ST/(π ν₀²) = h₀ (white FM),
and the standard ADEV–PSD relation gives σ_y²(τ) = h₀/τ = Δν_ST/(πν₀²τ)…
wait — with the one-sided ADEV integral:

    σ_y²(τ) = 2 ∫₀^∞ S_y(f) sin⁴(πfτ)/(πfτ)² df
             = 2 h₀ × 1/(4τ) = h₀/(2τ)

and h₀ = Δν_ST/(π ν₀²), giving σ_y²(τ) = Δν_ST/(2π ν₀² τ).  ✓

Numerical ADEV from phase noise PSD
────────────────────────────────────
When a full ``PhaseNoiseSpectrum`` is available, the Allan deviation is
computed by direct numerical integration:

    σ_y²(τ) = 2 ∫₀^∞ S_y(f) × sin⁴(πfτ)/(πfτ)²  df

where S_y(f) = (f/ν₀)² × S_φ(f) is the one-sided fractional frequency
noise PSD and the trapezoidal rule is applied over the supplied frequency
grid.  Accuracy requires the grid to span at least a decade on either side
of f_char ≈ 0.37/τ (the peak of the ADEV kernel).

Log-log slope interpretation
────────────────────────────
The ``allan_slope`` field captures d(log σ_B)/d(log τ) at each τ point:

    slope ≈ −0.5  →  white frequency noise  (Schawlow-Townes regime)
    slope ≈  0.0  →  flicker (1/f) frequency noise
    slope ≈ +0.5  →  random walk frequency noise  (thermal drift)

References
──────────
Schawlow, A. L., Townes, C. H. (1958) Phys. Rev. 112, 1940.
Allan, D. W. (1966) Proc. IEEE 54, 221–230.
Rutman, J. (1978) Proc. IEEE 66, 1048–1075.
Breeze et al. (2021) npj Quantum Information 7, 45.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import NVConfig
from .quantum_noise import MaserNoiseResult, PhaseNoiseSpectrum

# ── Physical constant ──────────────────────────────────────────────────────
_TWO_PI = 2.0 * math.pi

# NumPy ≥ 2.0 uses np.trapezoid; earlier versions used np.trapz.
try:
    _trapz = np.trapezoid  # type: ignore[attr-defined]  # NumPy ≥ 2.0
except AttributeError:
    _trapz = np.trapz  # type: ignore[attr-defined]  # NumPy < 2.0


# ── Result container ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class OscillatorStabilityResult:
    """Allan deviation and field stability characterisation of the NV maser.

    All field stability quantities are in Tesla (T), nT, and pT.
    ``sigma_y`` is the dimensionless fractional frequency Allan deviation.

    The ``sigma_b_st_floor_t`` array always holds the analytical white-FM
    floor η_ST / √τ derived directly from Δν_ST.  ``sigma_b_t`` equals
    this floor unless a full ``PhaseNoiseSpectrum`` was passed to
    ``compute_oscillator_stability``, in which case it is the result of
    the numerical PSD integral.
    """

    # ── Integration time axis ──────────────────────────────────────────────
    tau_s: NDArray[np.float64]
    """Integration time array τ (s).  Log-spacing recommended for ADEV plots."""

    # ── Primary field Allan deviation ──────────────────────────────────────
    sigma_b_t: NDArray[np.float64]
    """σ_B(τ) [T].  PSD-derived when a spectrum is supplied; ST floor otherwise."""

    sigma_b_nt: NDArray[np.float64]
    """σ_B(τ) [nT] = sigma_b_t × 10⁹."""

    sigma_b_pt: NDArray[np.float64]
    """σ_B(τ) [pT] = sigma_b_t × 10¹²."""

    # ── Fractional frequency Allan deviation ───────────────────────────────
    sigma_y: NDArray[np.float64]
    """σ_y(τ) = σ_B(τ) × γ_e / ν₀  [dimensionless]."""

    # ── Analytical Schawlow-Townes floor ───────────────────────────────────
    sigma_b_st_floor_t: NDArray[np.float64]
    """η_ST / √τ  [T].  Exact analytical white-FM limit computed from Δν_ST alone."""

    # ── Local log-log slope ────────────────────────────────────────────────
    allan_slope: NDArray[np.float64]
    """d(log σ_B) / d(log τ) at each τ point.
    ≈ −0.5 for white FM; ≈ 0 for flicker FM; ≈ +0.5 for random-walk FM."""

    # ── Scalar summary ─────────────────────────────────────────────────────
    sigma_b_at_1s_t: float
    """σ_B(τ = 1 s) = η_ST  [T].  Equals ``SensitivityResult.allan_deviation_1s_t``."""

    # ── Provenance ─────────────────────────────────────────────────────────
    schawlow_townes_linewidth_hz: float
    """Δν_ST (Hz) used for the ST floor and the σ_B_at_1s scalar."""

    gamma_e_hz_per_t: float
    """γ_e (Hz/T) = NVConfig.gamma_e_ghz_per_t × 10⁹."""

    carrier_frequency_hz: float
    """ν₀ (Hz) — maser carrier frequency used for σ_y ↔ σ_B conversion."""


# ── Core physics functions ─────────────────────────────────────────────────


def compute_white_fm_allan_deviation(
    tau_s: NDArray[np.float64],
    maser_noise_result: MaserNoiseResult,
    nv_config: NVConfig,
) -> NDArray[np.float64]:
    """Field Allan deviation σ_B(τ) in the pure white frequency noise limit  [T].

    Uses the Schawlow-Townes linewidth as the sole noise source:

        σ_B(τ) = η_ST / √τ  =  √(Δν_ST / (2π)) / (γ_e × √τ)

    This is exact for pure quantum phase diffusion (no flicker FM, no
    thermal drift, no technical noise).  At τ = 1 s it equals η_ST, the
    Schawlow-Townes field sensitivity stored in ``SensitivityResult.
    schawlow_townes_t_per_sqrthz`` and also in ``allan_deviation_1s_t``.

    Args:
        tau_s: Integration time array (s).  All values must be > 0.
        maser_noise_result: Quantum noise characterisation supplying Δν_ST.
        nv_config: NV center parameters supplying γ_e.

    Returns:
        NDArray of σ_B(τ) values (T), same shape as ``tau_s``.

    Raises:
        ValueError: if any element of ``tau_s`` is ≤ 0.
    """
    tau = np.asarray(tau_s, dtype=np.float64)
    if np.any(tau <= 0.0):
        raise ValueError("All tau_s values must be > 0.")

    delta_nu_st = maser_noise_result.schawlow_townes_linewidth_hz
    gamma_e = nv_config.gamma_e_ghz_per_t * 1.0e9  # Convert GHz/T → Hz/T
    # η_ST = √(Δν_ST / (2π)) / γ_e  — the Schawlow-Townes sensitivity [T/√Hz]
    eta_st = math.sqrt(delta_nu_st / _TWO_PI) / gamma_e
    return eta_st / np.sqrt(tau)


def compute_allan_deviation_from_psd(
    tau_s: NDArray[np.float64],
    phase_noise_spectrum: PhaseNoiseSpectrum,
    carrier_frequency_hz: float,
    gamma_e_hz_per_t: float,
) -> NDArray[np.float64]:
    """Field Allan deviation σ_B(τ) from the phase noise PSD via numerical integration  [T].

    Evaluates the two-sample Allan variance integral:

        σ_y²(τ) = 2 ∫₀^∞  S_y(f) × sin⁴(πfτ) / (πfτ)²  df

    where S_y(f) = (f / ν₀)² × S_φ(f) is the one-sided fractional
    frequency noise PSD.  The field Allan deviation follows from

        σ_B(τ) = σ_y(τ) × ν₀ / γ_e

    Accuracy requirements for the frequency grid of ``phase_noise_spectrum``:

    - Lower cutoff: f_min ≪ 0.37 / τ_max so that the ADEV kernel peak is
      captured at the longest integration time.
    - Upper cutoff: f_max ≫ 0.37 / τ_min so that all kernel area is included
      at the shortest integration time.
    - A log-spaced grid with ≥ 10 000 points from 10⁻³ Hz to 10⁷ Hz gives
      sub-percent accuracy for τ ∈ [0.1 s, 100 s].

    Args:
        tau_s: Integration time array (s).  All values must be > 0.
        phase_noise_spectrum: Phase noise PSD from ``compute_phase_noise_spectrum``.
        carrier_frequency_hz: ν₀ — maser carrier/cavity resonance (Hz). Must be > 0.
        gamma_e_hz_per_t: γ_e — NV gyromagnetic ratio (Hz/T).

    Returns:
        NDArray of σ_B(τ) values (T), same shape as ``tau_s``.

    Raises:
        ValueError: if any tau_s ≤ 0 or carrier_frequency_hz ≤ 0.
    """
    tau = np.asarray(tau_s, dtype=np.float64)
    if np.any(tau <= 0.0):
        raise ValueError("All tau_s values must be > 0.")
    if carrier_frequency_hz <= 0.0:
        raise ValueError(
            f"carrier_frequency_hz must be > 0, got {carrier_frequency_hz}."
        )

    f = phase_noise_spectrum.freq_offsets_hz      # (N,) Hz, all > 0
    S_phi = phase_noise_spectrum.psd_rad2_per_hz  # (N,) rad²/Hz

    # S_y(f) = (f/ν₀)² × S_φ(f)  — one-sided fractional frequency PSD [1/Hz]
    S_y = (f / carrier_frequency_hz) ** 2 * S_phi

    sigma_b = np.empty(len(tau), dtype=np.float64)
    for i, t in enumerate(tau):
        # ADEV kernel: 2 × sin⁴(πfτ) / (πfτ)²
        # Prefactor 2 comes from the one-sided PSD convention.
        x = math.pi * f * t                    # (N,) element-wise, all > 0
        kernel = 2.0 * np.sin(x) ** 4 / (x * x)  # (N,)
        integrand = S_y * kernel               # (N,)
        sigma_y_sq = float(_trapz(integrand, f))  # type: ignore[operator]
        sigma_y = math.sqrt(max(sigma_y_sq, 0.0))
        sigma_b[i] = sigma_y * carrier_frequency_hz / gamma_e_hz_per_t

    return sigma_b


def _compute_log_slope(
    tau_s: NDArray[np.float64],
    sigma_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Local log-log slope d(log σ_B) / d(log τ) via NumPy's gradient.

    Uses central differences at interior points and first-order one-sided
    differences at the endpoints.  Returns an array of the same length as
    ``tau_s``.

    For white FM noise, the slope should be ≈ −0.5 everywhere.
    For flicker FM noise, ≈ 0.  For random-walk FM, ≈ +0.5.

    Args:
        tau_s: Integration time array (s), all positive.
        sigma_b: σ_B(τ) array (T), all positive.

    Returns:
        Slope array of the same length as ``tau_s``.
    """
    log_tau = np.log(tau_s)
    log_sigma = np.log(sigma_b)
    if len(log_tau) < 2:
        # np.gradient requires ≥ 2 points; return zero slope for a single sample.
        return np.zeros_like(log_sigma)
    return np.gradient(log_sigma, log_tau)


def compute_oscillator_stability(
    tau_s: NDArray[np.float64],
    maser_noise_result: MaserNoiseResult,
    nv_config: NVConfig,
    phase_noise_spectrum: PhaseNoiseSpectrum | None = None,
) -> OscillatorStabilityResult:
    """Complete oscillator stability characterisation of the NV diamond maser.

    Computes the field Allan deviation σ_B(τ) and fractional frequency
    stability σ_y(τ) across the supplied integration time array.  Two
    computation paths are available:

    **Analytical (default)**: σ_B(τ) = η_ST / √τ derived from the
    Schawlow-Townes linewidth.  Exact for pure white-FM (quantum phase
    diffusion) noise.  Requires only ``maser_noise_result``.

    **Numerical PSD**: when ``phase_noise_spectrum`` is supplied, σ_B(τ)
    is computed from the full PSD integral (see ``compute_allan_deviation_
    from_psd``).  For a pure 1/f² phase noise spectrum the two paths agree
    to < 1 % when the frequency grid spans a decade on either side of
    0.37/τ.

    The ``allan_slope`` field (d log σ_B / d log τ) identifies the dominant
    noise type at each τ:

    - −0.5 → white frequency noise (Schawlow-Townes quantum limit)
    -  0.0 → flicker (1/f) frequency noise
    - +0.5 → random walk frequency noise (thermal drift)

    The scalar ``sigma_b_at_1s_t`` equals η_ST and is therefore identical to
    ``SensitivityResult.allan_deviation_1s_t`` when the same
    ``maser_noise_result`` is used for both.

    Args:
        tau_s: Integration time array (s).  Must contain only positive values.
            Log-spacing (e.g. ``np.logspace(-3, 3, 100)``) gives the most
            informative ADEV plot.
        maser_noise_result: Quantum noise characterisation.  Provides Δν_ST,
            ν₀ (cavity_frequency_hz), and κ_c.
        nv_config: NV center parameters.  Provides γ_e (gamma_e_ghz_per_t).
        phase_noise_spectrum: Optional full phase noise spectrum from
            ``compute_phase_noise_spectrum``.  When supplied the numerical PSD
            integral is used for ``sigma_b_t``; otherwise the analytical ST
            floor is used and ``sigma_b_t is sigma_b_st_floor_t`` element-wise.

    Returns:
        ``OscillatorStabilityResult`` with all stability metrics.

    Raises:
        ValueError: if any element of ``tau_s`` is ≤ 0.
    """
    tau = np.asarray(tau_s, dtype=np.float64)
    delta_nu_st = maser_noise_result.schawlow_townes_linewidth_hz
    nu0 = maser_noise_result.cavity_frequency_hz
    gamma_e = nv_config.gamma_e_ghz_per_t * 1.0e9  # Hz/T

    # ── Analytical Schawlow-Townes floor (always computed) ─────────────────
    sigma_b_st = compute_white_fm_allan_deviation(tau, maser_noise_result, nv_config)

    # ── Primary σ_B: PSD-integral or ST floor ─────────────────────────────
    if phase_noise_spectrum is not None:
        sigma_b = compute_allan_deviation_from_psd(tau, phase_noise_spectrum, nu0, gamma_e)
    else:
        sigma_b = sigma_b_st.copy()

    # ── Fractional frequency Allan deviation ───────────────────────────────
    # σ_y(τ) = σ_B(τ) × γ_e / ν₀
    sigma_y = sigma_b * (gamma_e / nu0)

    # ── Unit conversions ───────────────────────────────────────────────────
    sigma_b_nt = sigma_b * 1.0e9
    sigma_b_pt = sigma_b * 1.0e12

    # ── Local log-log slope ────────────────────────────────────────────────
    allan_slope = _compute_log_slope(tau, sigma_b)

    # ── Scalar: σ_B(τ=1 s) = η_ST (white FM limit) ────────────────────────
    # This equals SensitivityResult.allan_deviation_1s_t for the same noise result.
    eta_st = math.sqrt(delta_nu_st / _TWO_PI) / gamma_e
    sigma_b_at_1s = eta_st  # = η_ST / √1 = η_ST

    return OscillatorStabilityResult(
        tau_s=tau,
        sigma_b_t=sigma_b,
        sigma_b_nt=sigma_b_nt,
        sigma_b_pt=sigma_b_pt,
        sigma_y=sigma_y,
        sigma_b_st_floor_t=sigma_b_st,
        allan_slope=allan_slope,
        sigma_b_at_1s_t=sigma_b_at_1s,
        schawlow_townes_linewidth_hz=delta_nu_st,
        gamma_e_hz_per_t=gamma_e,
        carrier_frequency_hz=nu0,
    )
