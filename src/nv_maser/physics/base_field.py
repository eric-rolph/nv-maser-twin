"""
Halbach array permanent magnet base field B₀.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import FieldConfig
from .grid import SpatialGrid


def compute_base_field(grid: SpatialGrid, config: FieldConfig) -> NDArray[np.float32]:
    """
    Generate the static base magnetic field B₀ across the grid.

    For an ideal Halbach array, this is perfectly uniform.
    Optionally adds a small linear gradient to simulate manufacturing imperfections.

    Returns:
        (size, size) array of field values in Tesla.
    """
    field = np.full(grid.shape, config.b0_tesla, dtype=np.float32)

    if config.b0_gradient_ppm_per_mm > 0:
        # Linear gradient along x-axis (ppm of B₀ per mm)
        gradient = config.b0_tesla * (config.b0_gradient_ppm_per_mm * 1e-6) * grid.x
        field = field + gradient

    return field
