"""
Simulated micro-coil array using simplified Biot-Savart field model.

Each coil produces a field that falls off as 1/r² from its position.
The superposition of all coils with given currents produces the corrective field.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import CoilConfig
from .grid import SpatialGrid


class ShimCoilArray:
    """
    Array of N shim coils arranged in a circle around the diamond slab.

    Physics model (simplified Biot-Savart)::

        B_coil_i(x, y) = (scale_factor * I_i) / (r_i² + ε²)

    where r_i is the distance from coil i to point (x,y), I_i is the current,
    and ε is a small softening term to prevent singularity at the coil center.
    """

    def __init__(self, grid: SpatialGrid, config: CoilConfig) -> None:
        self.grid = grid
        self.config = config
        self.num_coils = config.num_coils

        # Place coils in a circle
        angles = np.linspace(0, 2 * np.pi, self.num_coils, endpoint=False)
        self.coil_x = config.coil_radius_mm * np.cos(angles)  # (num_coils,)
        self.coil_y = config.coil_radius_mm * np.sin(angles)  # (num_coils,)

        # Precompute the influence matrix: field contribution of each coil at unit current
        # Shape: (num_coils, grid_size, grid_size)
        self.influence_matrix = self._compute_influence_matrix()

    def _compute_influence_matrix(self) -> NDArray[np.float32]:
        """
        Precompute field contribution of each coil at 1 Amp current.

        Uses vectorized broadcasting — no Python loops over grid points.

        Returns:
            (num_coils, size, size) influence matrix in Tesla/Amp.
        """
        # Coil positions: (num_coils, 1, 1)
        cx = self.coil_x[:, np.newaxis, np.newaxis]
        cy = self.coil_y[:, np.newaxis, np.newaxis]

        # Grid positions: (1, size, size)
        gx = self.grid.x[np.newaxis, :, :]
        gy = self.grid.y[np.newaxis, :, :]

        # Distance squared: (num_coils, size, size)
        r_sq = (gx - cx) ** 2 + (gy - cy) ** 2

        # Softening epsilon to avoid division by zero (0.1 mm)
        eps_sq = 0.01

        # Simplified Biot-Savart: field ∝ 1/r²
        influence = self.config.field_scale_factor / (r_sq + eps_sq)

        return influence.astype(np.float32)

    def compute_field(self, currents: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Compute the total corrective field for given coil currents.

        Args:
            currents: (num_coils,) or (batch, num_coils) array of currents in Amps.

        Returns:
            (size, size) or (batch, size, size) total coil field in Tesla.
        """
        # Clamp currents to physical limits
        currents = np.clip(
            currents, -self.config.max_current_amps, self.config.max_current_amps
        )

        if currents.ndim == 1:
            # Single sample: (num_coils,) · (num_coils, N, N) → (N, N)
            return np.einsum("c,cij->ij", currents, self.influence_matrix)
        else:
            # Batch: (B, num_coils) · (num_coils, N, N) → (B, N, N)
            return np.einsum("bc,cij->bij", currents, self.influence_matrix)

    def compute_field_torch(
        self,
        currents: "torch.Tensor",  # noqa: F821
        influence_tensor: "torch.Tensor",  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        """
        PyTorch version for differentiable training.

        Args:
            currents: (batch, num_coils) torch tensor on device.
            influence_tensor: (num_coils, size, size) tensor on device.

        Returns:
            (batch, size, size) corrective field tensor.
        """
        import torch  # local import keeps physics module torch-free when not needed

        return torch.einsum("bc,cij->bij", currents, influence_tensor)
