"""
Realistic hardware models for the closed-loop shimming feedback chain.

Models three physical effects that degrade shimming performance
compared to the ideal (instantaneous, infinite-resolution) case:

1. **Hall sensor noise** — Gaussian noise on field measurements
2. **DAC quantization** — Finite-resolution current commands
3. **Coil L/R dynamics** — First-order exponential current settling

Each model can be applied independently or chained together in the
closed-loop simulation (see closed_loop.py).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import FeedbackConfig, CoilConfig
from .grid import SpatialGrid


# ─── Hall sensor model ─────────────────────────────────────────────


class HallSensorArray:
    """
    Sparse Hall-effect sensor array for field measurement.

    Sensors are placed in a regular grid pattern within the active zone.
    Each measurement is corrupted by additive white Gaussian noise.

    In a real device, the controller sees only these sparse, noisy
    measurements — not the full 2D field map that the training loop uses.
    """

    def __init__(
        self,
        grid: SpatialGrid,
        config: FeedbackConfig,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)

        # Place sensors in a regular sub-grid within the active zone
        n = config.num_sensors
        side = int(np.ceil(np.sqrt(n)))
        active_half = (grid.extent * grid.active_fraction) / 2.0

        sx = np.linspace(-active_half * 0.8, active_half * 0.8, side)
        sy = np.linspace(-active_half * 0.8, active_half * 0.8, side)
        sx_grid, sy_grid = np.meshgrid(sx, sy, indexing="xy")

        # Flatten and take first n positions
        self.sensor_x = sx_grid.ravel()[:n].astype(np.float32)
        self.sensor_y = sy_grid.ravel()[:n].astype(np.float32)

        # Precompute nearest-grid-point indices for each sensor
        self._indices = self._compute_indices(grid)

    def _compute_indices(self, grid: SpatialGrid) -> list[tuple[int, int]]:
        """Find nearest grid point for each sensor location."""
        indices = []
        for sx, sy in zip(self.sensor_x, self.sensor_y):
            # grid.x and grid.y are (size, size) meshgrids
            dist = (grid.x - sx) ** 2 + (grid.y - sy) ** 2
            idx = np.unravel_index(np.argmin(dist), dist.shape)
            indices.append((int(idx[0]), int(idx[1])))
        return indices

    def measure(self, field: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Read field values at sensor locations with noise.

        Args:
            field: (H, W) magnetic field in Tesla.

        Returns:
            (num_sensors,) noisy field measurements in Tesla.
        """
        readings = np.array(
            [field[i, j] for i, j in self._indices], dtype=np.float32
        )
        noise = self.rng.normal(
            0, self.config.sensor_noise_tesla, size=len(readings)
        ).astype(np.float32)
        return readings + noise

    def measure_clean(self, field: NDArray[np.float32]) -> NDArray[np.float32]:
        """Read exact field values at sensor locations (no noise)."""
        return np.array(
            [field[i, j] for i, j in self._indices], dtype=np.float32
        )


# ─── DAC quantization model ───────────────────────────────────────


def quantize_currents(
    currents: NDArray[np.float32],
    max_current: float,
    dac_bits: int,
) -> NDArray[np.float32]:
    """
    Quantize continuous current commands to DAC resolution.

    Maps the continuous range [-max_current, +max_current] onto
    2^dac_bits discrete levels (uniform mid-tread quantization).

    Args:
        currents:    (...) desired currents in Amps.
        max_current: Maximum current magnitude.
        dac_bits:    DAC resolution in bits.

    Returns:
        (...) quantized currents, same shape and dtype.
    """
    # Clamp to valid range
    clamped = np.clip(currents, -max_current, max_current)

    # Number of levels (signed: -2^(N-1) to +2^(N-1)-1)
    n_levels = 2**dac_bits
    step = (2 * max_current) / n_levels

    # Quantize: round to nearest step
    quantized = np.round(clamped / step) * step
    return np.clip(quantized, -max_current, max_current).astype(np.float32)


# ─── Coil L/R dynamics model ──────────────────────────────────────


class CoilDynamics:
    """
    First-order L/R model for shim coil current settling.

    When the DAC commands a new current I_target, the actual coil
    current follows:

        I(t) = I_target + (I_prev - I_target) · exp(-t/τ)

    where τ = L/R is the coil time constant.

    This means fast-changing corrections are attenuated — the coils
    can't instantly jump to new current values.
    """

    def __init__(self, config: FeedbackConfig) -> None:
        self.config = config
        self.tau_us = config.coil_time_constant_us
        self._current_state: NDArray[np.float32] | None = None

    @property
    def current_state(self) -> NDArray[np.float32] | None:
        """Current actual coil currents (may lag commanded values)."""
        return self._current_state

    def reset(self, num_coils: int) -> None:
        """Reset all coil currents to zero."""
        self._current_state = np.zeros(num_coils, dtype=np.float32)

    def step(
        self,
        target_currents: NDArray[np.float32],
        dt_us: float,
    ) -> NDArray[np.float32]:
        """
        Advance coil currents toward target over time interval dt.

        Args:
            target_currents: (num_coils,) commanded current in Amps.
            dt_us:           Time step in microseconds.

        Returns:
            (num_coils,) actual coil currents after settling.
        """
        if self._current_state is None:
            self._current_state = np.zeros_like(target_currents)

        if self.tau_us <= 0:
            # No inductance — instant response
            self._current_state = target_currents.copy()
        else:
            # First-order exponential settling
            alpha = float(np.exp(-dt_us / self.tau_us))
            self._current_state = (
                target_currents + (self._current_state - target_currents) * alpha
            ).astype(np.float32)

        return self._current_state.copy()

    def settling_fraction(self, dt_us: float) -> float:
        """Fraction of target reached after dt microseconds."""
        if self.tau_us <= 0:
            return 1.0
        return 1.0 - float(np.exp(-dt_us / self.tau_us))
