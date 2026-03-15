"""Tests for the SpatialGrid class."""
import numpy as np
import pytest

from nv_maser.config import GridConfig
from nv_maser.physics.grid import SpatialGrid


@pytest.fixture
def default_grid() -> SpatialGrid:
    return SpatialGrid(GridConfig())


def test_grid_shape(default_grid: SpatialGrid) -> None:
    """Grid x, y arrays are (size, size)."""
    assert default_grid.x.shape == (64, 64)
    assert default_grid.y.shape == (64, 64)
    assert default_grid.shape == (64, 64)


def test_grid_coordinates_centered(default_grid: SpatialGrid) -> None:
    """Grid ranges from -extent/2 to +extent/2."""
    assert pytest.approx(default_grid.x[0, 0], abs=0.01) == -5.0
    assert pytest.approx(default_grid.x[0, -1], abs=0.01) == 5.0
    assert pytest.approx(default_grid.y[0, 0], abs=0.01) == -5.0
    assert pytest.approx(default_grid.y[-1, 0], abs=0.01) == 5.0


def test_grid_dtype(default_grid: SpatialGrid) -> None:
    """Grid arrays are float32."""
    assert default_grid.x.dtype == np.float32
    assert default_grid.y.dtype == np.float32


def test_active_zone_mask_subset(default_grid: SpatialGrid) -> None:
    """Active zone has fewer points than full grid."""
    total = default_grid.size * default_grid.size
    assert default_grid.num_active_points < total
    assert default_grid.active_zone_mask.shape == (64, 64)


def test_active_zone_centered(default_grid: SpatialGrid) -> None:
    """Active zone mask is symmetric around center."""
    mask = default_grid.active_zone_mask
    # Horizontally symmetric
    assert np.array_equal(mask, mask[:, ::-1])
    # Vertically symmetric
    assert np.array_equal(mask, mask[::-1, :])


def test_radial_distance(default_grid: SpatialGrid) -> None:
    """Radial distance r is computed correctly."""
    r_expected = np.sqrt(default_grid.x**2 + default_grid.y**2)
    assert np.allclose(default_grid.r, r_expected, atol=1e-5)


def test_custom_grid_size() -> None:
    """Custom grid size works correctly."""
    cfg = GridConfig(size=32, physical_extent_mm=5.0)
    grid = SpatialGrid(cfg)
    assert grid.shape == (32, 32)
    assert pytest.approx(grid.x[0, 0], abs=0.01) == -2.5
