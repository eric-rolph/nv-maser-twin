"""
Time-domain Maxwell-Bloch solver for the NV diamond maser.

Implements semiclassical Maxwell-Bloch ODEs for the coupled
cavity-spin system, supporting both free-running maser (V = 0) and
driven amplifier (V > 0) modes.

Equations (mean-field, after Wang et al., Adv. Sci. 2024, Eqs. 9-11)
─────────────────────────────────────────────────────────────────────
    da/dt   = -(κ_c/2) a  − i g_ens S⁻  − i V
    dS⁻/dt  = -(κ_s/2) S⁻ + i g_ens a Sz
    dSz/dt  = -γ (Sz − Sz₀) + i g_ens (a† S⁻ − a S⁺)

where g_ens = g₀√N is the ensemble Rabi coupling (rad/s),
and S⁻, Sz are dimensionless per-spin order parameters.
This convention gives cooperativity C = 4 g_ens² / (κ_c κ_s),
consistent with cavity.py.

    a      – cavity photon amplitude (complex)
    S⁻, Sz – per-spin collective coherence / inversion
    g_ens  – ensemble coupling g₀√N (rad/s)
    κ_c    – cavity decay rate ω/Q (rad/s)
    κ_s    – spin dephasing rate 2/T₂ (rad/s)
    γ      – spin-lattice relaxation rate 1/T₁ (s⁻¹)
    Sz₀    – pump-maintained equilibrium inversion
    V      – external drive amplitude (rad/s; 0 = free-running)

References
──────────
Wang et al., "Tailoring coherent microwave emission …",
    Advanced Science (2024), PMC11425272.
Breeze et al., Nature 555, 493 (2018).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from ..config import CavityConfig, MaserConfig, MaxwellBlochConfig, NVConfig
from .cavity import compute_cavity_properties, compute_n_effective

# ── Physical constants ────────────────────────────────────────────
_HBAR = 1.054571817e-34  # J·s


@dataclass(frozen=True)
class MaxwellBlochResult:
    """Result of a Maxwell-Bloch time-domain simulation.

    Arrays have shape ``(n_time_points,)``.  Steady-state values are
    averages over the final 10 % of the time trace.
    """

    time_s: NDArray[np.float64]
    cavity_re: NDArray[np.float64]  # Re[a(t)]
    cavity_im: NDArray[np.float64]  # Im[a(t)]
    coherence_re: NDArray[np.float64]  # Re[S⁻(t)]
    coherence_im: NDArray[np.float64]  # Im[S⁻(t)]
    inversion: NDArray[np.float64]  # Sz(t)
    photon_number: NDArray[np.float64]  # |a(t)|²

    # Steady-state summary
    steady_state_photons: float
    steady_state_inversion: float
    output_power_w: float
    gain_db: float | None  # only meaningful for driven mode
    cooperativity: float


# ── ODE right-hand side ──────────────────────────────────────────


def _maxwell_bloch_rhs(
    t: float,
    y: list[float],
    kappa_c: float,
    kappa_s: float,
    g_ens: float,
    gamma: float,
    sz0: float,
    drive: float,
) -> list[float]:
    r"""
    RHS of the real-valued Maxwell-Bloch ODE system.

    State vector y = [a_r, a_i, s_r, s_i, sz] where a = a_r + i·a_i
    and S⁻ = s_r + i·s_i.

    Uses the ensemble coupling g_ens = g₀√N in all equations
    (mean-field convention consistent with cooperativity C = 4g_ens²/(κ_c·κ_s)).

    The drive *V* is taken as a real amplitude applied to the imaginary
    quadrature: −iV adds ``−V`` to ``da_i/dt``.
    """
    ar, ai, sr, si, sz = y

    # Cavity amplitude
    dar = -(kappa_c / 2) * ar + g_ens * si
    dai = -(kappa_c / 2) * ai - g_ens * sr - drive

    # Spin coherence
    dsr = -(kappa_s / 2) * sr - g_ens * ai * sz
    dsi = -(kappa_s / 2) * si + g_ens * ar * sz

    # Inversion  (Im(a† S⁻) = a_r·s_i − a_i·s_r)
    dsz = -gamma * (sz - sz0) - 2 * g_ens * (ar * si - ai * sr)

    return [dar, dai, dsr, dsi, dsz]


# ── Main solver ──────────────────────────────────────────────────


def solve_maxwell_bloch(
    nv_config: NVConfig,
    maser_config: MaserConfig,
    cavity_config: CavityConfig,
    mb_config: MaxwellBlochConfig,
    gain_budget: float = 1.0,
) -> MaxwellBlochResult:
    r"""
    Solve the Maxwell-Bloch equations for the NV maser system.

    Physical parameters are derived from existing configuration classes:

    - κ_c from cavity Q and resonance frequency
    - κ_s from T₂\*
    - g₀ from cavity zero-point field
    - N_eff from NV density, mode volume, fill factor, pump efficiency
    - γ from T₁
    - Sz₀ = pump_efficiency / 2

    Args:
        nv_config: NV-centre parameters (incl. T₁, T₂\*, pump efficiency).
        maser_config: Cavity Q, frequency, coupling β.
        cavity_config: Mode volume, fill factor.
        mb_config: Solver settings (time span, drive, method).
        gain_budget: Spectral overlap fraction (0–1).

    Returns:
        MaxwellBlochResult with full time evolution and steady-state summary.

    Raises:
        RuntimeError: If the ODE integrator fails to converge.
    """
    # ── Derived physical quantities ──────────────────────────────
    omega = 2 * math.pi * maser_config.cavity_frequency_ghz * 1e9  # rad/s
    kappa_c = omega / maser_config.cavity_q  # cavity decay (rad/s)

    t2_s = nv_config.t2_star_us * 1e-6
    kappa_s = 2.0 / t2_s  # spin dephasing rate (rad/s)

    t1_s = nv_config.t1_ms * 1e-3
    gamma = 1.0 / t1_s  # spin-lattice relaxation (s⁻¹)

    cavity_props = compute_cavity_properties(maser_config, cavity_config)
    g0 = 2 * math.pi * cavity_props.single_spin_coupling_hz  # rad/s

    n_eff = compute_n_effective(nv_config, cavity_config, gain_budget)
    g_ens = g0 * math.sqrt(n_eff) if n_eff > 0 else 0.0

    sz0 = nv_config.pump_efficiency / 2.0  # equilibrium inversion (per spin)

    # Cooperativity C = 4 g_ens² / (κ_c · κ_s)
    denom = kappa_c * kappa_s
    cooperativity = 4 * g_ens**2 / denom if denom > 0 else 0.0

    # Drive amplitude (convert Hz → rad/s)
    drive = mb_config.drive_amplitude_hz * 2 * math.pi

    # ── Integration setup ────────────────────────────────────────
    t_max = mb_config.t_max_us * 1e-6
    t_eval = np.linspace(0, t_max, mb_config.n_time_points)

    # Initial conditions: pumped equilibrium, tiny seed photon to break symmetry
    y0 = [1e-6, 0.0, 0.0, 0.0, sz0]

    sol = solve_ivp(
        _maxwell_bloch_rhs,
        [0, t_max],
        y0,
        method=mb_config.solver_method,
        t_eval=t_eval,
        args=(kappa_c, kappa_s, g_ens, gamma, sz0, drive),
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"Maxwell-Bloch solver failed: {sol.message}")

    ar, ai, sr, si, sz = sol.y
    photon_number = ar**2 + ai**2

    # ── Steady-state extraction (average last 10 %) ──────────────
    n_avg = max(1, len(t_eval) // 10)
    n_ss = float(np.mean(photon_number[-n_avg:]))
    sz_ss = float(np.mean(sz[-n_avg:]))

    # Output power: P_out = ℏω · κ_e · ⟨n⟩_ss
    kappa_e = kappa_c * maser_config.coupling_beta
    p_out = _HBAR * omega * kappa_e * n_ss

    # Gain (amplifier mode only)
    gain_db: float | None = None
    if drive > 0:
        # Input power: P_in = ℏω · V² / κ_e  (input-output theory)
        p_in = _HBAR * omega * drive**2 / kappa_e if kappa_e > 0 else float("inf")
        if p_in > 0 and p_out > 0:
            gain_db = 10 * math.log10(p_out / p_in)
        elif p_out == 0:
            gain_db = -math.inf

    return MaxwellBlochResult(
        time_s=sol.t,
        cavity_re=ar,
        cavity_im=ai,
        coherence_re=sr,
        coherence_im=si,
        inversion=sz,
        photon_number=photon_number,
        steady_state_photons=n_ss,
        steady_state_inversion=sz_ss,
        output_power_w=p_out,
        gain_db=gain_db,
        cooperativity=cooperativity,
    )


# ── Analytical steady-state ──────────────────────────────────────


def compute_steady_state_power(
    nv_config: NVConfig,
    maser_config: MaserConfig,
    cavity_config: CavityConfig,
    gain_budget: float = 1.0,
) -> float:
    r"""
    Analytical steady-state output power for a free-running maser.

    Above threshold (C · Sz₀ > 1):

        n_ss = (γ / κ_c) · (Sz₀ − 1/C)

        P_out = ℏω · κ_e · n_ss

    Derivation: at steady state the inversion clamps at Sz_ss = 1/C,
    and the energy balance gives n_ss = γ (Sz₀ − Sz_ss) / κ_c.

    Below threshold the output is zero.

    Args:
        nv_config: NV-centre parameters.
        maser_config: Cavity parameters.
        cavity_config: Mode volume and fill factor.
        gain_budget: Spectral overlap fraction (0–1).

    Returns:
        Steady-state output power (Watts).  0 if below threshold.
    """
    omega = 2 * math.pi * maser_config.cavity_frequency_ghz * 1e9
    kappa_c = omega / maser_config.cavity_q

    t1_s = nv_config.t1_ms * 1e-3
    gamma = 1.0 / t1_s

    t2_s = nv_config.t2_star_us * 1e-6
    kappa_s = 2.0 / t2_s

    cavity_props = compute_cavity_properties(maser_config, cavity_config)
    g0 = 2 * math.pi * cavity_props.single_spin_coupling_hz

    n_eff = compute_n_effective(nv_config, cavity_config, gain_budget)
    if n_eff <= 0 or g0 == 0:
        return 0.0

    g_ens = g0 * math.sqrt(n_eff)
    denom = kappa_c * kappa_s
    if denom <= 0:
        return 0.0

    cooperativity = 4 * g_ens**2 / denom
    sz0 = nv_config.pump_efficiency / 2.0

    # Masing condition: C · sz0 > 1
    if cooperativity * sz0 <= 1.0:
        return 0.0

    # Intracavity photon number: n_ss = γ(sz0 − 1/C)/κ_c
    sz_clamped = 1.0 / cooperativity
    n_ss = (gamma / kappa_c) * (sz0 - sz_clamped)
    if n_ss < 0:
        return 0.0

    kappa_e = kappa_c * maser_config.coupling_beta
    return _HBAR * omega * kappa_e * n_ss
