"""
Composes the full field environment: B₀ + disturbance + coil corrections.
This is the main interface for the training loop and visualization.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import SimConfig
from .grid import SpatialGrid
from .base_field import compute_base_field
from .disturbance import DisturbanceGenerator
from .coils import ShimCoilArray


class FieldEnvironment:
    """
    Complete simulation environment.

    Manages the spatial grid, base field, disturbance generator, and coil array.
    Provides the interface for:

    - Generating distorted field observations (for the AI controller input)
    - Applying coil corrections and computing the net field
    - Evaluating field uniformity (for the loss function)
    """

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.grid = SpatialGrid(config.grid)
        self.base_field = compute_base_field(self.grid, config.field)
        self.disturbance_gen = DisturbanceGenerator(self.grid, config.disturbance)
        self.coils = ShimCoilArray(self.grid, config.coils)

        # Current state
        self._current_disturbance: NDArray[np.float32] | None = None

    @property
    def distorted_field(self) -> NDArray[np.float32]:
        """B₀ + current disturbance (before correction)."""
        if self._current_disturbance is None:
            self._current_disturbance = self.disturbance_gen.generate()
        return self.base_field + self._current_disturbance

    def step(self, t: float = 0.0) -> NDArray[np.float32]:
        """
        Advance the environment: generate a new disturbance at time t.

        Returns the distorted field (without correction).
        """
        self._current_disturbance = self.disturbance_gen.generate(t)
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

        Returns dict with:
            - variance: Var(B) over active zone (the primary loss target)
            - std: std(B) over active zone in Tesla
            - ppm: peak-to-peak homogeneity in ppm
            - max_deviation: max |B - B₀| over active zone
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

        return {
            "variance": var_b,
            "std": std_b,
            "ppm": ppm,
            "max_deviation": max_dev,
        }

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
