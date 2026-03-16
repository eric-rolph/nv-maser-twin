"""
Mean-field dipolar spin-spin interaction model.

NV centres at 10 ppm concentrations are separated by ~8 nm, giving
nearest-neighbour magnetic dipole-dipole coupling strengths ~100 kHz.
These flip-flop interactions transport spin inversion across the
inhomogeneous lineshape, refilling spectral holes burned by
superradiant emission.

Two complementary models are provided:

1. **Stretched-exponential refilling** (phenomenological, fast):
       p(Δ, t) → p̄ − (p̄ − p(Δ, t₀)) exp(−(δt / T_r)^α)
   with α = 0.5 as the hallmark of 3D 1/r³ dipolar interactions.

2. **Spectral diffusion** (mean-field PDE, more physical):
       dp/dt |_dipolar = D_ss · ∂²p/∂Δ²
   modelling the frequency-space diffusion of inversion.

References
──────────
Kersten et al., "Self-induced superradiant masing …",
    Nature Physics (2026), PMC12811124, Eqs. 2, 8c.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..config import DipolarConfig, SpectralConfig


def stretched_exponential_refill(
    p_delta: NDArray[np.float64],
    p_equilibrium: NDArray[np.float64],
    dt_s: float,
    refilling_time_s: float,
    alpha: float = 0.5,
) -> NDArray[np.float64]:
    r"""Apply stretched-exponential spectral refilling over time dt.

    .. math::

        p(\Delta, t+dt) = \bar{p}(\Delta)
        - \left[\bar{p}(\Delta) - p(\Delta, t)\right]
          \exp\!\left(-\left(\frac{dt}{T_r}\right)^\alpha\right)

    This drives the inversion profile back toward its pumped equilibrium
    on a timescale T_r, with sub-exponential (α < 1) dynamics caused by
    the broad distribution of 1/r³ couplings in 3D.

    Args:
        p_delta: Current inversion profile, shape ``(n_bins,)``.
        p_equilibrium: Equilibrium (pumped) inversion profile, same shape.
        dt_s: Time step (seconds).
        refilling_time_s: Characteristic refilling time T_r (seconds).
        alpha: Stretched exponent (0 < α ≤ 1).  α = 0.5 for 3D dipolar.

    Returns:
        Updated inversion profile (new array).
    """
    if dt_s <= 0:
        return p_delta.copy()

    ratio = dt_s / refilling_time_s
    # Clamp the exponent (not the ratio) to avoid exp underflow
    exponent = min(ratio ** alpha, 700.0)
    decay = math.exp(-exponent)
    return p_equilibrium - (p_equilibrium - p_delta) * decay


def spectral_diffusion_step(
    p_delta: NDArray[np.float64],
    d_freq_hz: float,
    diffusion_coeff_hz2_per_s: float,
    dt_s: float,
) -> NDArray[np.float64]:
    r"""One explicit Euler step of spectral diffusion.

    .. math::

        \frac{\partial p}{\partial t}\bigg|_{\text{dipolar}}
        = D_{ss} \frac{\partial^2 p}{\partial \Delta^2}

    Discretised with second-order central differences and zero-flux
    (Neumann) boundary conditions.

    Args:
        p_delta: Current inversion profile, shape ``(n_bins,)``.
        d_freq_hz: Frequency spacing between adjacent bins (Hz).
        diffusion_coeff_hz2_per_s: Diffusion coefficient D_ss (Hz²/s).
        dt_s: Time step (seconds).

    Returns:
        Updated inversion profile (new array).

    Raises:
        ValueError: If the CFL stability condition is violated
            (dt < d_freq² / (2 D_ss)).
    """
    if diffusion_coeff_hz2_per_s <= 0 or dt_s <= 0:
        return p_delta.copy()

    # CFL stability check
    cfl_limit = d_freq_hz**2 / (2.0 * diffusion_coeff_hz2_per_s)
    if dt_s > cfl_limit:
        raise ValueError(
            f"CFL violation: dt={dt_s:.2e} s > limit={cfl_limit:.2e} s. "
            f"Reduce dt or diffusion_coeff."
        )

    # Central difference Laplacian with Neumann BCs (zero flux)
    laplacian = np.empty_like(p_delta)
    laplacian[1:-1] = (p_delta[2:] - 2.0 * p_delta[1:-1] + p_delta[:-2])
    laplacian[0] = p_delta[1] - p_delta[0]       # dp/dΔ = 0 at left
    laplacian[-1] = p_delta[-2] - p_delta[-1]     # dp/dΔ = 0 at right
    laplacian /= d_freq_hz**2

    return p_delta + diffusion_coeff_hz2_per_s * dt_s * laplacian


def estimate_dipolar_coupling_hz(nv_density_per_m3: float) -> float:
    r"""Estimate nearest-neighbour dipolar coupling from NV density.

    The mean nearest-neighbour distance in a random 3D distribution is:
        r_nn ≈ (3 / (4π n))^{1/3}

    The secular dipolar coupling between two electron spins is:
        J = (μ₀ γ²ₑ ℏ) / (4π r³) × (1 − 3cos²θ)

    Taking the RMS angular factor ⟨(1−3cos²θ)²⟩^{1/2} = 2/√5 ≈ 0.894:
        J_typ ≈ (μ₀ γ²ₑ ℏ) / (4π r_nn³) × 0.894

    Returns coupling in Hz.
    """
    if nv_density_per_m3 <= 0:
        return 0.0

    mu0 = 4e-7 * math.pi          # T·m/A
    gamma_e = 2 * math.pi * 28.025e9  # rad/s/T
    hbar = 1.054571817e-34         # J·s

    r_nn = (3.0 / (4.0 * math.pi * nv_density_per_m3)) ** (1.0 / 3.0)
    j_rad = (mu0 * gamma_e**2 * hbar) / (4.0 * math.pi * r_nn**3) * 0.894
    return j_rad / (2.0 * math.pi)


def estimate_refilling_time_us(nv_density_per_m3: float) -> float:
    """Estimate spectral hole refilling time from NV density.

    Empirical scaling: T_r ∝ 1/n_NV (faster refilling at higher density).
    Calibrated to Kersten et al. 2026: T_r = 11.6 μs at ~10 ppm
    (1.76 × 10²⁴ m⁻³).
    """
    n_ref = 1.76e24  # 10 ppm in m⁻³
    t_r_ref = 11.6   # μs

    if nv_density_per_m3 <= 0:
        return float("inf")

    return t_r_ref * (n_ref / nv_density_per_m3)


def apply_dipolar_refilling(
    p_delta: NDArray[np.float64],
    p_equilibrium: NDArray[np.float64],
    delta_hz: NDArray[np.float64],
    dipolar_config: DipolarConfig,
    dt_s: float,
) -> NDArray[np.float64]:
    """Apply one step of dipolar spectral refilling.

    Dispatches to the stretched-exponential or diffusion model
    based on configuration.

    Args:
        p_delta: Current inversion profile.
        p_equilibrium: Equilibrium (pumped) profile.
        delta_hz: Detuning grid (Hz).
        dipolar_config: Dipolar interaction settings.
        dt_s: Time step (seconds).

    Returns:
        Updated inversion profile.
    """
    if not dipolar_config.enable:
        return p_delta.copy()

    t_r_s = dipolar_config.refilling_time_us * 1e-6

    if dipolar_config.diffusion_coefficient_mhz2_per_us > 0:
        d_ss = dipolar_config.diffusion_coefficient_mhz2_per_us * 1e12 * 1e6
        d_freq = delta_hz[1] - delta_hz[0] if len(delta_hz) > 1 else 1.0
        return spectral_diffusion_step(p_delta, d_freq, d_ss, dt_s)

    return stretched_exponential_refill(
        p_delta, p_equilibrium, dt_s, t_r_s, dipolar_config.stretch_exponent,
    )
