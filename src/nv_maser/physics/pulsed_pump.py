"""
Pulsed optical pump model for the NV diamond maser.

Models LED or laser pumping with finite pulse duration and repetition,
as in Long et al., Communications Engineering (2025) — LED-pumped
room-temperature maser with 7–200 µs pulses at ~130 W peak.

During pump ON, the population inversion builds up toward the CW
steady-state value. During pump OFF, inversion decays with T₁.

The key ODE is:

    dη/dt = Γ_pump(t) × (2/3 − η) − η / T₁

where:
    η(t)      – effective inversion efficiency (0 to 2/3)
    Γ_pump(t) – optical pump rate (Hz), zero when pump is off
    T₁        – spin-lattice relaxation time
    2/3       – maximum inversion for S=1 triplet

The equivalent CW power for threshold comparison:

    P_eq = P_peak × (τ_pulse / τ_period)

References
──────────
Long et al., Commun. Eng. (2025), PMC12241473.
Breeze et al., Nature 555, 493 (2018).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from ..config import NVConfig, OpticalPumpConfig
from .optical_pump import compute_pump_rate


@dataclass(frozen=True)
class PulsedPumpResult:
    """Result of a pulsed pump inversion calculation.

    Contains the full time evolution of inversion over multiple
    pump cycles, plus summary metrics for threshold estimation.
    """

    time_s: NDArray[np.float64]  # time array
    inversion: NDArray[np.float64]  # η(t) at each time point
    peak_inversion: float  # max η achieved during any pulse
    mean_inversion: float  # time-averaged η over the computed window
    equivalent_cw_power_w: float  # P_peak × duty_cycle
    duty_cycle: float  # τ_on / τ_period
    pump_on_fraction: float  # fraction of time pump is on (same as duty_cycle)
    n_cycles: int  # number of pump cycles simulated


def pulsed_pump_rate(
    t: float,
    gamma_pump_cw: float,
    pulse_duration_s: float,
    pulse_period_s: float,
) -> float:
    """
    Instantaneous pump rate at time t under pulsed operation.

    Returns full CW pump rate during ON phase and zero during OFF.

    Args:
        t: Time (seconds).
        gamma_pump_cw: CW pump rate at full power (Hz).
        pulse_duration_s: Duration of each pump pulse (s).
        pulse_period_s: Period between pulse starts (s).

    Returns:
        Pump rate at time t (Hz).
    """
    if pulse_period_s <= 0:
        return gamma_pump_cw  # CW fallback
    phase = t % pulse_period_s
    if phase < pulse_duration_s:
        return gamma_pump_cw
    return 0.0


def _pulsed_pump_rhs(
    t: float,
    y: list[float],
    gamma_pump_cw: float,
    gamma_relax: float,
    pulse_duration_s: float,
    pulse_period_s: float,
) -> list[float]:
    """ODE RHS for pulsed inversion dynamics."""
    eta = y[0]
    gp = pulsed_pump_rate(t, gamma_pump_cw, pulse_duration_s, pulse_period_s)
    # dη/dt = Γ_pump(t) × (2/3 − η) − η / T₁
    d_eta = gp * (2.0 / 3.0 - eta) - gamma_relax * eta
    return [d_eta]


def compute_pulsed_inversion(
    pump_config: OpticalPumpConfig,
    nv_config: NVConfig,
    n_cycles: int = 5,
    points_per_cycle: int = 200,
) -> PulsedPumpResult:
    """
    Solve the pulsed inversion ODE over multiple pump cycles.

    Simulates the build-up and decay of population inversion for a
    pulsed pump (LED or laser) with given pulse duration and period.

    Args:
        pump_config: Laser/pump parameters including pulsed mode settings.
        nv_config: NV-centre parameters (T₁).
        n_cycles: Number of pump cycles to simulate.
        points_per_cycle: Time resolution per cycle.

    Returns:
        PulsedPumpResult with full time evolution and summary metrics.

    Raises:
        ValueError: If pulse_duration_us > pulse_period_us.
        RuntimeError: If the ODE integrator fails.
    """
    tau_pulse = pump_config.pulse_duration_us * 1e-6  # s
    tau_period = pump_config.pulse_period_us * 1e-6  # s

    if tau_pulse > tau_period:
        raise ValueError(
            f"pulse_duration_us ({pump_config.pulse_duration_us}) "
            f"cannot exceed pulse_period_us ({pump_config.pulse_period_us})"
        )

    gamma_pump_cw = compute_pump_rate(pump_config)
    gamma_relax = 1.0 / (nv_config.t1_ms * 1e-3)  # 1/T₁

    t_max = n_cycles * tau_period
    n_points = n_cycles * points_per_cycle
    t_eval = np.linspace(0, t_max, n_points)

    # Initial state: thermal equilibrium, no inversion
    y0 = [0.0]

    sol = solve_ivp(
        _pulsed_pump_rhs,
        [0, t_max],
        y0,
        method="RK45",
        t_eval=t_eval,
        args=(gamma_pump_cw, gamma_relax, tau_pulse, tau_period),
        rtol=1e-8,
        atol=1e-12,
        max_step=tau_pulse / 10,  # resolve pulse edges
    )

    if not sol.success:
        raise RuntimeError(f"Pulsed pump solver failed: {sol.message}")

    eta = sol.y[0]
    duty_cycle = tau_pulse / tau_period if tau_period > 0 else 1.0

    return PulsedPumpResult(
        time_s=sol.t,
        inversion=eta,
        peak_inversion=float(np.max(eta)),
        mean_inversion=float(np.mean(eta)),
        equivalent_cw_power_w=pump_config.laser_power_w * duty_cycle,
        duty_cycle=duty_cycle,
        pump_on_fraction=duty_cycle,
        n_cycles=n_cycles,
    )


def compute_equivalent_cw_power(pump_config: OpticalPumpConfig) -> float:
    """
    Equivalent CW power for a pulsed pump.

    P_eq = P_peak × (τ_pulse / τ_period)

    This is the power a CW laser would need to deliver the same
    time-averaged energy as the pulsed source.

    Reference: Long et al. 2025, Eq. for P_pth,eq.
    """
    if pump_config.pulse_period_us <= 0:
        return pump_config.laser_power_w
    duty = pump_config.pulse_duration_us / pump_config.pulse_period_us
    return pump_config.laser_power_w * duty
