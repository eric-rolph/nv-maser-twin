"""Tests for the base field generator."""
import numpy as np
import pytest

from nv_maser.config import FieldConfig, GridConfig
from nv_maser.physics.grid import SpatialGrid
from nv_maser.physics.base_field import compute_base_field


@pytest.fixture
def grid() -> SpatialGrid:
    return SpatialGrid(GridConfig())


def test_uniform_field_zero_std(grid: SpatialGrid) -> None:
    """Default config → perfectly uniform field (within float32 precision)."""
    cfg = FieldConfig(b0_tesla=0.05, b0_gradient_ppm_per_mm=0.0)
    field = compute_base_field(grid, cfg)
    assert np.std(field) == pytest.approx(0.0, abs=1e-6)


def test_gradient_increases_std(grid: SpatialGrid) -> None:
    """Non-zero gradient → std > 0."""
    cfg = FieldConfig(b0_tesla=0.05, b0_gradient_ppm_per_mm=10.0)
    field = compute_base_field(grid, cfg)
    assert np.std(field) > 0.0


def test_field_values_near_b0(grid: SpatialGrid) -> None:
    """All field values within reasonable range of B₀."""
    b0 = 0.05
    cfg = FieldConfig(b0_tesla=b0, b0_gradient_ppm_per_mm=0.0)
    field = compute_base_field(grid, cfg)
    assert np.allclose(field, b0, atol=1e-6)


def test_field_shape(grid: SpatialGrid) -> None:
    """Output shape matches grid shape."""
    field = compute_base_field(grid, FieldConfig())
    assert field.shape == grid.shape


def test_field_dtype(grid: SpatialGrid) -> None:
    """Output is float32."""
    field = compute_base_field(grid, FieldConfig())
    assert field.dtype == np.float32


def test_gradient_varies_along_x(grid: SpatialGrid) -> None:
    """Gradient causes variation along x-axis but not y-axis."""
    cfg = FieldConfig(b0_tesla=0.05, b0_gradient_ppm_per_mm=100.0)
    field = compute_base_field(grid, cfg)
    # Column std (along y) should be ~0; row std (along x) should be > 0
    col_std = np.std(field[:, 0])  # all rows, first column
    row_std = np.std(field[0, :])  # first row, all columns
    assert col_std == pytest.approx(0.0, abs=1e-8)
    assert row_std > 0.0
