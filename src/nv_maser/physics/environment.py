"""
Composes the full field environment: B₀ + disturbance + coil corrections.
This is the main interface for the training loop and visualization.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import SimConfig, NVConfig, MaserConfig
from .grid import SpatialGrid
from .base_field import compute_base_field
from .disturbance import DisturbanceGenerator
from .coils import ShimCoilArray
from .maser_gain import compute_maser_metrics, max_tolerable_b_std
from .thermal import ThermalModel, ThermalState, compute_thermal_state
from .signal_chain import compute_signal_chain_budget


class FieldEnvironment:
    """
    Complete simulation environment.

    Manages the spatial grid, base field, disturbance generator, and coil array.
    Provides the interface for:

    - Generating distorted field observations (for the AI controller input)
    - Applying coil corrections and computing the net field
    - Evaluating field uniformity (for the loss function)
    """

    def __init__(self, config: SimConfig, thermal_seed: int | None = None) -> None:
        self.config = config
        self.grid = SpatialGrid(config.grid)
        self.base_field = compute_base_field(self.grid, config.field, config.halbach)
        self.disturbance_gen = DisturbanceGenerator(self.grid, config.disturbance)
        self.coils = ShimCoilArray(self.grid, config.coils)

        # Thermal model (always created; at default 25°C → zero offset)
        self.thermal_model = ThermalModel(config.thermal, seed=thermal_seed)
        self._thermal_state: ThermalState | None = None

        # Current state
        self._current_disturbance: NDArray[np.float32] | None = None

    @property
    def thermal_state(self) -> ThermalState | None:
        """Current thermal state, if step() has been called."""
        return self._thermal_state

    @property
    def effective_base_field(self) -> NDArray[np.float32]:
        """B₀ adjusted for thermal drift."""
        if self._thermal_state is not None:
            return self.base_field + np.float32(self._thermal_state.b0_shift_tesla)
        return self.base_field

    @property
    def distorted_field(self) -> NDArray[np.float32]:
        """B₀ (thermally shifted) + current disturbance (before correction)."""
        if self._current_disturbance is None:
            self._current_disturbance = self.disturbance_gen.generate()
        return self.effective_base_field + self._current_disturbance

    def step(self, t: float = 0.0) -> NDArray[np.float32]:
        """
        Advance the environment: generate a new disturbance at time t.
        Also updates the thermal state.

        Returns the distorted field (without correction).
        """
        self._current_disturbance = self.disturbance_gen.generate(t)
        self._thermal_state = self.thermal_model.state_at(
            t, self.config.field, self.config.nv,
            self.config.maser, self.config.feedback,
        )
        return self.distorted_field

    def apply_correction(
        self, currents: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Apply coil currents and return the net (corrected) field.

        Args:
            currents: (num_coils,) coil current array.

        Returns:
            (size, size) net field = B₀ + disturbance + coil_field.
        """
        coil_field = self.coils.compute_field(currents)
        return self.distorted_field + coil_field

    def compute_uniformity_metric(
        self, net_field: NDArray[np.float32]
    ) -> dict[str, float]:
        """
        Compute field uniformity metrics over the active zone.

        When thermal state is available, uses temperature-adjusted T2* and Q.

        Returns dict with:
            - variance: Var(B) over active zone (the primary loss target)
            - std: std(B) over active zone in Tesla
            - ppm: peak-to-peak homogeneity in ppm
            - max_deviation: max |B - B₀| over active zone
            - temperature_c: current temperature (if thermal active)
        """
        mask = self.grid.active_zone_mask
        active = net_field[mask]

        mean_b = float(np.mean(active))
        var_b = float(np.var(active))
        std_b = float(np.std(active))
        min_b = float(np.min(active))
        max_b = float(np.max(active))

        ppm = ((max_b - min_b) / mean_b * 1e6) if mean_b > 0 else float("inf")
        max_dev = float(np.max(np.abs(active - self.config.field.b0_tesla)))

        # Use thermally-adjusted NV/maser params if available
        nv_config = self.config.nv
        maser_config = self.config.maser
        if self._thermal_state is not None:
            nv_config = nv_config.model_copy(
                update={"t2_star_us": self._thermal_state.effective_t2_star_us}
            )
            maser_config = maser_config.model_copy(
                update={"cavity_q": self._thermal_state.effective_cavity_q}
            )

        maser = compute_maser_metrics(net_field, mask, nv_config, maser_config)

        result = {
            "variance": var_b,
            "std": std_b,
            "ppm": ppm,
            "max_deviation": max_dev,
            **maser,
        }

        if self._thermal_state is not None:
            result["temperature_c"] = self._thermal_state.temperature_c
            result["b0_shift_tesla"] = self._thermal_state.b0_shift_tesla

        # Signal chain SNR budget
        gain_budget = maser["gain_budget"]
        snr_budget = compute_signal_chain_budget(
            nv_config, maser_config, self.config.signal_chain, gain_budget
        )
        result["snr_db"] = snr_budget.snr_db
        result["received_power_w"] = snr_budget.received_power_w
        result["total_noise_w"] = snr_budget.total_noise_w
        result["system_noise_temperature_k"] = snr_budget.system_noise_temperature_k

        return result

    def generate_training_data(
        self, num_samples: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Generate a dataset of distorted fields for training.

        Returns:
            distorted_fields: (num_samples, size, size) — model inputs
            disturbances:     (num_samples, size, size) — ground-truth disturbances
        """
        disturbances = self.disturbance_gen.generate_batch(num_samples)
        distorted = self.base_field[np.newaxis, :, :] + disturbances
        return distorted, disturbances
