"""
Spectral Maxwell-Bloch solver: frequency-resolved maser dynamics.

Extends the scalar Maxwell-Bloch solver (``maxwell_bloch.py``) to
evolve a frequency-resolved inversion profile p(Δ, t) across M
detuning bins, capturing:

- **Spectral hole burning** at the cavity resonance
- **Dipolar spectral refilling** (stretched-exponential or diffusive)
- **Superradiant pulse trains** from repeated hole/refill cycles

The ODE system follows Kersten et al., Nature Physics (2026), Eqs. 8a–8c,
using a mean-field per-bin approach.

State vector
────────────
    y = [a_r, a_i, s₀_r, s₀_i, p₀, s₁_r, s₁_i, p₁, …, s_{M-1}_r, s_{M-1}_i, p_{M-1}]

    - a_r, a_i : cavity field (Re, Im parts of ⟨a⟩)
    - sⱼ_r, sⱼ_i : per-bin spin coherence ⟨σⱼ⁻⟩
    - pⱼ : per-bin inversion ⟨σⱼᶻ⟩

Total dimension: 2 + 3·M  (M = number of frequency bins, typically 51–201).

Per-bin coupling
────────────────
Each bin j contains nⱼ spins (set by the inhomogeneous lineshape weight).
The per-bin collective coupling is gⱼ = g₀ √nⱼ.  The summed spin drive
on the cavity is ∑ⱼ gⱼ σⱼ⁻.

ODEs (rotating frame, ω_cavity = 0)
────────────────────────────────────
    ȧ   = −(κ/2)a − i ∑ⱼ gⱼ σⱼ⁻
    σ̇ⱼ⁻ = −(γ⊥/2 + iΔⱼ) σⱼ⁻ + i gⱼ a pⱼ
    ṗⱼ  = 2i gⱼ (a† σⱼ⁻ − a σⱼ⁺) − γ(pⱼ − p̄ⱼ)

Dipolar refilling is applied as an operator-split step after each ODE
integration interval.

References
──────────
Kersten et al., Nature Physics (2026), PMC12811124, Eqs. 8a–8c.
Wang et al., Advanced Science (2024), PMC11425272, Eqs. 9-11.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from ..config import (
    CavityConfig,
    DipolarConfig,
    MaserConfig,
    MaxwellBlochConfig,
    NVConfig,
    SpectralConfig,
)
from .cavity import compute_cavity_properties, compute_n_effective
from .dipolar import apply_dipolar_refilling
from .spectral import (
    build_detuning_grid,
    build_initial_inversion,
    fwhm_to_sigma,
    q_gaussian,
)

# ── Physical constants ────────────────────────────────────────────
_HBAR = 1.054571817e-34  # J·s


@dataclass(frozen=True)
class SpectralMBResult:
    """Result of a spectral Maxwell-Bloch simulation.

    Arrays with suffix ``_t`` have shape ``(n_time_points,)``.
    ``inversion_profile`` has shape ``(n_time_points, n_bins)``.
    """

    time_s: NDArray[np.float64]
    cavity_re: NDArray[np.float64]  # Re[a(t)]
    cavity_im: NDArray[np.float64]  # Im[a(t)]
    photon_number: NDArray[np.float64]  # |a(t)|²

    delta_hz: NDArray[np.float64]  # detuning grid, shape (n_bins,)
    inversion_profile: NDArray[np.float64]  # p(Δ, t), shape (n_time, n_bins)
    on_resonance_inversion: NDArray[np.float64]  # p(0, t), shape (n_time,)

    # Steady-state summary
    steady_state_photons: float
    steady_state_on_res_inversion: float
    output_power_w: float
    cooperativity: float
    n_bursts: int  # number of superradiant bursts detected


# ── Per-bin spin weights ──────────────────────────────────────────


def _compute_bin_weights(
    delta_hz: NDArray[np.float64],
    spectral_config: SpectralConfig,
    n_eff_total: float,
) -> NDArray[np.float64]:
    """Compute number of spins per detuning bin.

    The inhomogeneous lineshape g(Δ) is normalised so that
    ∑ⱼ nⱼ = N_eff_total.

    Returns:
        n_per_bin: array of shape ``(n_bins,)`` giving the effective
            number of spins in each frequency bin.
    """
    sigma = fwhm_to_sigma(
        spectral_config.inhomogeneous_linewidth_mhz * 1e6,
        spectral_config.q_parameter,
    )
    lineshape = q_gaussian(delta_hz, sigma, spectral_config.q_parameter)
    d_freq = delta_hz[1] - delta_hz[0] if len(delta_hz) > 1 else 1.0

    # lineshape is a density (1/Hz); multiply by bin width to get weight
    weights = lineshape * d_freq
    total_weight = weights.sum()
    if total_weight > 0:
        weights *= n_eff_total / total_weight
    return weights


# ── ODE right-hand side ──────────────────────────────────────────


def _spectral_mb_rhs(
    t: float,
    y: NDArray[np.float64],
    n_bins: int,
    delta_hz: NDArray[np.float64],
    g_bin: NDArray[np.float64],
    kappa_c: float,
    kappa_s: float,
    gamma: float,
    p_eq: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""
    RHS of the spectral Maxwell-Bloch ODE.

    State layout: y = [a_r, a_i, (s_r, s_i, p) × n_bins]
    Dimension: 2 + 3 * n_bins

    Uses per-bin collective coupling g_bin[j] = g₀ √(n_j) (rad/s).
    """
    ar = y[0]
    ai = y[1]

    # Extract per-bin arrays (views into y)
    spin_block = y[2:].reshape(n_bins, 3)
    sr = spin_block[:, 0]  # Re[σ⁻]
    si = spin_block[:, 1]  # Im[σ⁻]
    pj = spin_block[:, 2]  # inversion

    # Cavity: ȧ = −(κ/2)a − i ∑ⱼ gⱼ σⱼ⁻
    # Real:  ȧ_r = −(κ/2)a_r + ∑ⱼ gⱼ sⱼ_i
    # Imag:  ȧ_i = −(κ/2)a_i − ∑ⱼ gⱼ sⱼ_r
    dar = -(kappa_c / 2) * ar + np.dot(g_bin, si)
    dai = -(kappa_c / 2) * ai - np.dot(g_bin, sr)

    # Spin coherence: σ̇ⱼ⁻ = −(γ⊥/2 + iΔⱼ) σⱼ⁻ + i gⱼ a pⱼ
    # Real:  ṡ_r = −(γ⊥/2)s_r + Δⱼ s_i − gⱼ a_i pⱼ
    # Imag:  ṡ_i = −(γ⊥/2)s_i − Δⱼ s_r + gⱼ a_r pⱼ
    delta_rad = 2 * math.pi * delta_hz
    dsr = -(kappa_s / 2) * sr + delta_rad * si - g_bin * ai * pj
    dsi = -(kappa_s / 2) * si - delta_rad * sr + g_bin * ar * pj

    # Inversion: ṗⱼ = 2i gⱼ (a† σⱼ⁻ − a σⱼ⁺) − γ(pⱼ − p̄ⱼ)
    # = −2 gⱼ Im(a† σⱼ⁻) − γ(pⱼ − p̄ⱼ)
    # Im(a† σ⁻) = a_r s_i − a_i s_r
    dpj = -2 * g_bin * (ar * si - ai * sr) - gamma * (pj - p_eq)

    # Assemble output
    dydt = np.empty_like(y)
    dydt[0] = dar
    dydt[1] = dai
    dydt[2:].reshape(n_bins, 3)[:, 0] = dsr
    dydt[2:].reshape(n_bins, 3)[:, 1] = dsi
    dydt[2:].reshape(n_bins, 3)[:, 2] = dpj
    return dydt


# ── Main solver ──────────────────────────────────────────────────


def solve_spectral_maxwell_bloch(
    nv_config: NVConfig,
    maser_config: MaserConfig,
    cavity_config: CavityConfig,
    mb_config: MaxwellBlochConfig,
    spectral_config: SpectralConfig,
    dipolar_config: DipolarConfig | None = None,
    gain_budget: float = 1.0,
) -> SpectralMBResult:
    r"""
    Solve the spectral Maxwell-Bloch equations.

    The solver uses operator splitting: ODE integration handles the
    coherent cavity–spin dynamics and T₁ relaxation, while dipolar
    refilling is applied as a discrete step between ODE intervals.

    By default the full time span is solved in one ODE call.
    When dipolar_config is enabled, the time span is split into
    sub-intervals of length ~ T_r/10, and dipolar refilling is
    applied to the inversion profile between intervals.

    Args:
        nv_config: NV-centre parameters.
        maser_config: Cavity parameters.
        cavity_config: Mode geometry.
        mb_config: Time-domain solver settings.
        spectral_config: Frequency grid and lineshape.
        dipolar_config: Dipolar interaction settings (None = disabled).
        gain_budget: Spectral overlap fraction.

    Returns:
        SpectralMBResult with full time/frequency evolution.
    """
    # ── Derived physical quantities ──────────────────────────────
    omega = 2 * math.pi * maser_config.cavity_frequency_ghz * 1e9
    kappa_c = omega / maser_config.cavity_q
    kappa_s = 2.0 / (nv_config.t2_star_us * 1e-6)
    gamma = 1.0 / (nv_config.t1_ms * 1e-3)

    cavity_props = compute_cavity_properties(maser_config, cavity_config)
    g0 = 2 * math.pi * cavity_props.single_spin_coupling_hz  # rad/s

    n_eff = compute_n_effective(nv_config, cavity_config, gain_budget)

    # Build detuning grid and per-bin spin counts
    delta_hz = build_detuning_grid(spectral_config)
    n_bins = len(delta_hz)
    n_per_bin = _compute_bin_weights(delta_hz, spectral_config, n_eff)

    # Per-bin collective coupling: gⱼ = g₀ √nⱼ
    g_bin = g0 * np.sqrt(np.maximum(n_per_bin, 0.0))

    # Total cooperativity: C = 4 (∑ gⱼ²) / (κ_c κ_s)
    g_ens_sq = np.sum(g_bin**2)
    cooperativity = 4 * g_ens_sq / (kappa_c * kappa_s) if kappa_c * kappa_s > 0 else 0.0

    # Equilibrium inversion profile (pumped)
    sz0 = nv_config.pump_efficiency / 2.0
    _, p_eq = build_initial_inversion(nv_config, spectral_config)

    # ── Initial conditions ───────────────────────────────────────
    y0 = np.zeros(2 + 3 * n_bins)
    y0[0] = 1e-6  # seed cavity photon (breaks symmetry)
    # Spin coherences start at zero; inversion starts at equilibrium
    y0[2:].reshape(n_bins, 3)[:, 2] = p_eq

    # ── Time setup ───────────────────────────────────────────────
    t_max = mb_config.t_max_us * 1e-6
    n_pts = mb_config.n_time_points

    # Dipolar operator-splitting intervals
    dipolar_enabled = (
        dipolar_config is not None
        and dipolar_config.enable
        and dipolar_config.refilling_time_us > 0
    )

    if dipolar_enabled:
        assert dipolar_config is not None
        t_r = dipolar_config.refilling_time_us * 1e-6
        # Sub-interval ~ T_r / 10, but at least 10 intervals
        n_intervals = max(10, int(math.ceil(t_max / (t_r / 10))))
    else:
        n_intervals = 1

    dt_interval = t_max / n_intervals

    # Collect results: allocate output arrays
    t_out = np.linspace(0, t_max, n_pts)
    ar_out = np.empty(n_pts)
    ai_out = np.empty(n_pts)
    inv_out = np.empty((n_pts, n_bins))

    # We integrate interval-by-interval and interpolate onto t_out
    collected_t: list[NDArray] = []
    collected_ar: list[NDArray] = []
    collected_ai: list[NDArray] = []
    collected_inv: list[NDArray] = []

    y_current = y0.copy()
    t_start = 0.0

    for i_int in range(n_intervals):
        t_end_int = min(t_start + dt_interval, t_max)
        if t_end_int <= t_start:
            break

        # Points in this interval from t_out
        mask = (t_out >= t_start - 1e-15) & (t_out <= t_end_int + 1e-15)
        t_eval_int = t_out[mask].copy()
        # Clamp to strictly within [t_start, t_end_int] for solve_ivp
        if len(t_eval_int) > 0:
            t_eval_int = np.clip(t_eval_int, t_start, t_end_int)
        else:
            # Still need to integrate to evolve state
            t_eval_int = np.array([t_end_int])

        sol = solve_ivp(
            _spectral_mb_rhs,
            [t_start, t_end_int],
            y_current,
            method=mb_config.solver_method,
            t_eval=t_eval_int,
            args=(n_bins, delta_hz, g_bin, kappa_c, kappa_s, gamma, p_eq),
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success:
            raise RuntimeError(
                f"Spectral MB solver failed at interval {i_int}: {sol.message}"
            )

        collected_t.append(sol.t)
        collected_ar.append(sol.y[0])
        collected_ai.append(sol.y[1])

        # Extract inversion profiles at each time point
        for k in range(sol.y.shape[1]):
            inv_k = sol.y[2:, k].reshape(n_bins, 3)[:, 2]
            collected_inv.append(inv_k)

        # Update state from end of interval
        y_current = sol.y[:, -1].copy()

        # Apply dipolar refilling to inversion profile
        if dipolar_enabled:
            assert dipolar_config is not None
            spin_block = y_current[2:].reshape(n_bins, 3)
            p_now = spin_block[:, 2].copy()
            p_refilled = apply_dipolar_refilling(
                p_now, p_eq, delta_hz, dipolar_config, dt_interval,
            )
            spin_block[:, 2] = p_refilled

        t_start = t_end_int

    # ── Assemble output arrays ───────────────────────────────────
    all_t = np.concatenate(collected_t)
    all_ar = np.concatenate(collected_ar)
    all_ai = np.concatenate(collected_ai)
    all_inv = np.array(collected_inv)  # shape: (n_total_pts, n_bins)

    # Deduplicate and sort (interval boundaries may overlap)
    _, unique_idx = np.unique(all_t, return_index=True)
    all_t = all_t[unique_idx]
    all_ar = all_ar[unique_idx]
    all_ai = all_ai[unique_idx]
    all_inv = all_inv[unique_idx]

    # Interpolate onto the requested t_out grid
    ar_out = np.interp(t_out, all_t, all_ar)
    ai_out = np.interp(t_out, all_t, all_ai)

    # For the inversion profile, do per-bin interpolation
    inv_out = np.empty((n_pts, n_bins))
    for j in range(n_bins):
        inv_out[:, j] = np.interp(t_out, all_t, all_inv[:, j])

    photon_number = ar_out**2 + ai_out**2

    # On-resonance inversion: value at Δ = 0 (centre bin)
    centre_bin = n_bins // 2
    on_res_inv = inv_out[:, centre_bin]

    # ── Steady-state extraction (last 10 %) ──────────────────────
    n_avg = max(1, n_pts // 10)
    n_ss = float(np.mean(photon_number[-n_avg:]))

    sz_on_res_ss = float(np.mean(on_res_inv[-n_avg:]))

    kappa_e = kappa_c * maser_config.coupling_beta
    p_out = _HBAR * omega * kappa_e * n_ss

    # Count superradiant bursts (peaks in photon number)
    n_bursts = _count_bursts(photon_number)

    return SpectralMBResult(
        time_s=t_out,
        cavity_re=ar_out,
        cavity_im=ai_out,
        photon_number=photon_number,
        delta_hz=delta_hz,
        inversion_profile=inv_out,
        on_resonance_inversion=on_res_inv,
        steady_state_photons=n_ss,
        steady_state_on_res_inversion=sz_on_res_ss,
        output_power_w=p_out,
        cooperativity=cooperativity,
        n_bursts=n_bursts,
    )


def _count_bursts(photon_number: NDArray[np.float64]) -> int:
    """Count superradiant bursts as peaks exceeding 10× the median."""
    median_n = float(np.median(photon_number))
    if median_n <= 0:
        threshold = float(np.max(photon_number)) * 0.1
    else:
        threshold = median_n * 10

    if threshold <= 0:
        return 0

    above = photon_number > threshold
    # Count rising edges (False → True transitions)
    edges = np.diff(above.astype(int))
    return int(np.sum(edges == 1))
