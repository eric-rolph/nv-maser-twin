"""
Closed-loop shimming simulation with realistic hardware in the loop.

This module runs a time-stepping simulation where:

1. Disturbance evolves in time
2. Hall sensors measure the field (with noise)
3. Controller predicts coil currents (with computation latency)
4. DAC quantizes the current commands
5. Coils settle toward target (L/R dynamics)
6. Net field is evaluated for maser viability

This is the "hardware-in-the-loop" test that tells you whether
your trained controller will actually work on real hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
from numpy.typing import NDArray

from ..config import SimConfig
from .environment import FieldEnvironment
from .feedback import CoilDynamics, HallSensorArray, quantize_currents
from .maser_gain import compute_maser_metrics


@dataclass
class LoopStepResult:
    """Results from one control loop iteration."""

    time_us: float
    commanded_currents: NDArray[np.float32]
    quantized_currents: NDArray[np.float32]
    actual_currents: NDArray[np.float32]
    sensor_readings: NDArray[np.float32]
    field_variance: float
    field_std: float
    gain_budget: float
    masing: bool


@dataclass
class ClosedLoopResult:
    """Aggregate results from a full closed-loop simulation run."""

    steps: list[LoopStepResult] = dc_field(default_factory=list)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def mean_variance(self) -> float:
        if not self.steps:
            return float("inf")
        return float(np.mean([s.field_variance for s in self.steps]))

    @property
    def mean_gain_budget(self) -> float:
        if not self.steps:
            return 0.0
        return float(np.mean([s.gain_budget for s in self.steps]))

    @property
    def masing_fraction(self) -> float:
        """Fraction of timesteps where maser is above threshold."""
        if not self.steps:
            return 0.0
        return float(np.mean([s.masing for s in self.steps]))

    @property
    def current_quantization_error_rms(self) -> float:
        """RMS difference between commanded and quantized currents."""
        if not self.steps:
            return 0.0
        errors = [
            np.sqrt(np.mean((s.commanded_currents - s.quantized_currents) ** 2))
            for s in self.steps
        ]
        return float(np.mean(errors))

    @property
    def current_settling_error_rms(self) -> float:
        """RMS difference between quantized and actual (settled) currents."""
        if not self.steps:
            return 0.0
        errors = [
            np.sqrt(np.mean((s.quantized_currents - s.actual_currents) ** 2))
            for s in self.steps
        ]
        return float(np.mean(errors))

    def summary(self) -> dict[str, float]:
        """Compact summary for logging/comparison."""
        return {
            "num_steps": self.num_steps,
            "mean_variance": self.mean_variance,
            "mean_gain_budget": self.mean_gain_budget,
            "masing_fraction": self.masing_fraction,
            "quantization_error_rms_amps": self.current_quantization_error_rms,
            "settling_error_rms_amps": self.current_settling_error_rms,
        }


class ClosedLoopSimulator:
    """
    Time-stepping closed-loop shimming simulation.

    Chains: disturbance → sensor → controller → DAC → coil dynamics → field.

    The controller is provided as a callable: field_map → currents.
    This decouples the simulation from any specific NN architecture.
    """

    def __init__(
        self,
        config: SimConfig,
        controller_fn: "Callable[[NDArray[np.float32]], NDArray[np.float32]]",  # noqa: F821
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.controller_fn = controller_fn

        # Build physics components
        self.env = FieldEnvironment(config, thermal_seed=seed)
        self.sensors = HallSensorArray(
            self.env.grid, config.feedback, seed=seed
        )
        self.coil_dynamics = CoilDynamics(config.feedback)
        self.coil_dynamics.reset(config.coils.num_coils)

    def run(
        self,
        duration_us: float,
        dt_disturbance_us: float | None = None,
    ) -> ClosedLoopResult:
        """
        Run a closed-loop simulation for the given duration.

        Args:
            duration_us: Total simulation time in microseconds.
            dt_disturbance_us: How often the disturbance evolves.
                Defaults to control_loop_period_us.

        Returns:
            ClosedLoopResult with per-step and aggregate metrics.
        """
        fb = self.config.feedback
        loop_period = fb.control_loop_period_us
        if dt_disturbance_us is None:
            dt_disturbance_us = loop_period

        result = ClosedLoopResult()
        t_us = 0.0

        while t_us < duration_us:
            step_result = self._step(t_us, loop_period, dt_disturbance_us)
            result.steps.append(step_result)
            t_us += loop_period

        return result

    def _step(
        self,
        t_us: float,
        loop_period_us: float,
        dt_disturbance_us: float,
    ) -> LoopStepResult:
        """Execute one control loop iteration."""
        fb = self.config.feedback

        # 1. Evolve disturbance + thermal state (convert μs → seconds)
        t_seconds = t_us * 1e-6
        self.env.step(t_seconds)

        # 1b. Update coil dynamics L/R if temperature has shifted resistance
        thermal = self.env.thermal_state
        if thermal is not None:
            # Temporarily adjust tau for this step based on effective resistance
            effective_tau = fb.coil_inductance_uh / thermal.effective_coil_resistance_ohm
            self.coil_dynamics.tau_us = effective_tau

        # 2. Get current field (B₀ + thermal shift + disturbance + current coil correction)
        if self.coil_dynamics.current_state is not None:
            current_field = self.env.apply_correction(
                self.coil_dynamics.current_state
            )
        else:
            current_field = self.env.distorted_field

        # 3. Controller sees the full field map
        commanded = self.controller_fn(current_field)

        # 4. DAC quantization
        quantized = quantize_currents(
            commanded,
            self.config.coils.max_current_amps,
            fb.dac_bits,
        )

        # 5. Coil L/R settling over the available time
        settling_time = max(0, loop_period_us - fb.total_loop_latency_us)
        actual = self.coil_dynamics.step(quantized, settling_time)

        # 6. Compute the resulting net field with actual currents
        net_field = self.env.apply_correction(actual)

        # 7. Evaluate performance with thermally-adjusted NV/maser params
        mask = self.env.grid.active_zone_mask
        active = net_field[mask]
        var_b = float(np.var(active))
        std_b = float(np.std(active))

        nv_config = self.config.nv
        maser_config = self.config.maser
        if thermal is not None:
            nv_config = nv_config.model_copy(
                update={"t2_star_us": thermal.effective_t2_star_us}
            )
            maser_config = maser_config.model_copy(
                update={"cavity_q": thermal.effective_cavity_q}
            )

        maser = compute_maser_metrics(net_field, mask, nv_config, maser_config)

        # 8. Sensor readings (for diagnostic/logging)
        sensor_readings = self.sensors.measure(net_field)

        return LoopStepResult(
            time_us=t_us,
            commanded_currents=commanded,
            quantized_currents=quantized,
            actual_currents=actual,
            sensor_readings=sensor_readings,
            field_variance=var_b,
            field_std=std_b,
            gain_budget=maser["gain_budget"],
            masing=maser["masing"],
        )
