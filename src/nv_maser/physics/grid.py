"""
Defines the 2D spatial grid representing the NV diamond slab cross-section.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import GridConfig


class SpatialGrid:
    """
    2D spatial grid with physical coordinates.

    The grid represents a square cross-section of the NV diamond slab.
    Coordinates are in millimeters, centered at (0, 0).
    """

    def __init__(self, config: GridConfig) -> None:
        self.size = config.size
        self.extent = config.physical_extent_mm
        self.active_fraction = config.active_zone_fraction

        half = self.extent / 2.0
        axis = np.linspace(-half, half, self.size, dtype=np.float32)
        self.x, self.y = np.meshgrid(axis, axis, indexing="xy")
        # Precompute radial distance from center (used by coils)
        self.r = np.sqrt(self.x**2 + self.y**2)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    @property
    def active_zone_mask(self) -> NDArray[np.bool_]:
        """
        Boolean mask: True for grid points inside the central 'active zone'.

        The active zone is where we care about field uniformity.
        """
        half_active = (self.extent * self.active_fraction) / 2.0
        return (np.abs(self.x) <= half_active) & (np.abs(self.y) <= half_active)

    @property
    def num_active_points(self) -> int:
        return int(self.active_zone_mask.sum())
