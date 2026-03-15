"""
Thermal coupling model for the NV maser digital twin.

Temperature fluctuations affect every subsystem simultaneously:

1. **Halbach magnets (NdFeB)** — Reversible tempco drifts B₀.
   ΔB₀/B₀ = α_mag · ΔT, where α_mag ≈ -0.12%/°C for N52.

2. **Diamond NV T2*** — Phonon-limited dephasing degrades with temperature.
   T2*(T) = T2*(T_ref) · (T_ref_K / T_K)^n, n=1..2.

3. **Microwave cavity Q** — Wall resistivity increases with temperature.
   Q(T) = Q_ref / √(1 + α_wall · ΔT).

4. **Coil resistance** — Copper tempco changes R, shifting L/R time constant.
   R(T) = R_ref · (1 + α_Cu · ΔT).

This module provides:
- ThermalState: snapshot of temperature-dependent parameters at a given T
- ThermalModel: time-evolving temperature with drift + noise
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import ThermalConfig, NVConfig, MaserConfig, FeedbackConfig, FieldConfig


@dataclass(frozen=True)
class ThermalState:
    """Snapshot of all temperature-affected parameters at a given temperature.

    This is what the rest of the physics stack reads to get the
    effective values of parameters that depend on temperature.
    """

    temperature_c: float
    """Current temperature (°C)."""

    b0_shift_tesla: float
    """Additive shift to B₀ from magnet tempco (Tesla)."""

    effective_t2_star_us: float
    """Temperature-adjusted T2* (μs)."""

    effective_cavity_q: float
    """Temperature-adjusted cavity Q factor."""

    effective_coil_resistance_ohm: float
    """Temperature-adjusted coil DC resistance (Ω)."""

    @property
    def effective_coil_time_constant_us(self) -> float:
        """L/R time constant using temperature-adjusted resistance.

        Note: inductance is essentially temperature-independent
        (it depends on geometry, not resistivity).
        """
        # This needs inductance from FeedbackConfig; we store R here
        # and let the caller combine with L. This property is a convenience
        # when L is known externally.
        return float("inf") if self.effective_coil_resistance_ohm == 0 else 0.0


def compute_thermal_state(
    temperature_c: float,
    thermal: ThermalConfig,
    field: FieldConfig,
    nv: NVConfig,
    maser: MaserConfig,
    feedback: FeedbackConfig,
) -> ThermalState:
    """
    Compute all temperature-dependent parameter values at a given temperature.

    Args:
        temperature_c: Current temperature in °C.
        thermal: Thermal coupling configuration.
        field: Base field configuration (for B₀).
        nv: NV center configuration (for T2*).
        maser: Maser configuration (for cavity Q).
        feedback: Feedback configuration (for coil resistance).

    Returns:
        ThermalState with shifted parameter values.
    """
    delta_t = temperature_c - thermal.reference_temperature_c

    # 1. Halbach magnet B₀ shift
    #    ΔB₀ = B₀ × (α/100) × ΔT  (α is in %/°C)
    b0_shift = field.b0_tesla * (thermal.magnet_tempco_pct_per_c / 100.0) * delta_t

    # 2. Diamond T2* degradation
    #    T2*(T) = T2*(T_ref) × (T_ref_K / T_K)^n
    #    Convert °C to K for the ratio
    t_ref_k = thermal.reference_temperature_c + 273.15
    t_k = temperature_c + 273.15
    if t_k > 0 and thermal.t2_star_tempco_exponent > 0:
        ratio = t_ref_k / t_k
        t2_factor = ratio ** thermal.t2_star_tempco_exponent
    else:
        t2_factor = 1.0
    effective_t2 = nv.t2_star_us * t2_factor

    # 3. Cavity Q degradation
    #    Q ∝ 1/√ρ, and ρ(T) = ρ_ref × (1 + α_wall × ΔT)
    #    → Q(T) = Q_ref / √(1 + α_wall × ΔT)
    rho_factor = 1.0 + thermal.cavity_wall_tempco_per_c * delta_t
    if rho_factor > 0:
        q_factor = 1.0 / np.sqrt(rho_factor)
    else:
        q_factor = 1.0  # unphysical negative: clamp
    effective_q = maser.cavity_q * q_factor

    # 4. Coil resistance shift
    #    R(T) = R_ref × (1 + α_Cu × ΔT)
    r_factor = 1.0 + thermal.coil_tempco_per_c * delta_t
    effective_r = feedback.coil_resistance_ohm * max(r_factor, 0.01)  # floor to prevent ≤0

    return ThermalState(
        temperature_c=temperature_c,
        b0_shift_tesla=float(b0_shift),
        effective_t2_star_us=float(effective_t2),
        effective_cavity_q=float(effective_q),
        effective_coil_resistance_ohm=float(effective_r),
    )


class ThermalModel:
    """
    Time-evolving temperature model with linear drift and noise.

    Temperature evolves as:
        T(t) = T_ambient + drift_rate × t + noise(t)

    where noise is sampled from N(0, σ²) at each query.

    Provides ThermalState snapshots for the physics stack.
    """

    def __init__(self, config: ThermalConfig, seed: int | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self._base_temperature = config.ambient_temperature_c

    def temperature_at(self, t: float) -> float:
        """
        Compute temperature at time t (seconds).

        Args:
            t: Time in seconds from simulation start.

        Returns:
            Temperature in °C.
        """
        drift = self.config.thermal_drift_rate_c_per_s * t
        noise = (
            float(self.rng.normal(0, self.config.thermal_noise_std_c))
            if self.config.thermal_noise_std_c > 0
            else 0.0
        )
        return self._base_temperature + drift + noise

    def state_at(
        self,
        t: float,
        field: FieldConfig,
        nv: NVConfig,
        maser: MaserConfig,
        feedback: FeedbackConfig,
    ) -> ThermalState:
        """
        Get the full ThermalState at time t.

        Args:
            t: Time in seconds.
            field: Base field config.
            nv: NV center config.
            maser: Maser config.
            feedback: Feedback config.

        Returns:
            ThermalState with all temperature-adjusted parameters.
        """
        temp = self.temperature_at(t)
        return compute_thermal_state(
            temp, self.config, field, nv, maser, feedback
        )

    def trajectory(
        self,
        duration_s: float,
        dt_s: float,
        field: FieldConfig,
        nv: NVConfig,
        maser: MaserConfig,
        feedback: FeedbackConfig,
    ) -> list[ThermalState]:
        """
        Compute a time series of ThermalStates.

        Useful for analyzing thermal sensitivity over a simulation run.

        Args:
            duration_s: Total duration (seconds).
            dt_s: Time step (seconds).
            field, nv, maser, feedback: Config objects.

        Returns:
            List of ThermalState snapshots at each time step.
        """
        states = []
        t = 0.0
        while t <= duration_s:
            states.append(self.state_at(t, field, nv, maser, feedback))
            t += dt_s
        return states
