"""
Halbach array permanent magnet base field B₀.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import FieldConfig, HalbachConfig
from .grid import SpatialGrid
from .halbach import compute_halbach_field


def compute_base_field(
    grid: SpatialGrid,
    config: FieldConfig,
    halbach: HalbachConfig | None = None,
) -> NDArray[np.float32]:
    """
    Generate the static base magnetic field B₀ across the grid.

    When *halbach* is provided and enabled, delegates to the analytical
    multipole Halbach model.  Otherwise falls back to the legacy uniform
    field with optional linear gradient.

    Returns:
        (size, size) array of field values in Tesla.
    """
    if halbach is not None and halbach.enabled:
        return compute_halbach_field(grid, config, halbach)

    field = np.full(grid.shape, config.b0_tesla, dtype=np.float32)

    if config.b0_gradient_ppm_per_mm > 0:
        # Linear gradient along x-axis (ppm of B₀ per mm)
        gradient = config.b0_tesla * (config.b0_gradient_ppm_per_mm * 1e-6) * grid.x
        field = field + gradient

    return field
