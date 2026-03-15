"""Tests for the ShimCoilArray."""
import numpy as np
import pytest

from nv_maser.config import CoilConfig, GridConfig
from nv_maser.physics.grid import SpatialGrid
from nv_maser.physics.coils import ShimCoilArray


@pytest.fixture
def grid() -> SpatialGrid:
    return SpatialGrid(GridConfig())


@pytest.fixture
def coils(grid: SpatialGrid) -> ShimCoilArray:
    return ShimCoilArray(grid, CoilConfig())


def test_influence_matrix_shape(coils: ShimCoilArray) -> None:
    """Shape is (num_coils, size, size)."""
    assert coils.influence_matrix.shape == (8, 64, 64)


def test_zero_current_zero_field(coils: ShimCoilArray) -> None:
    """Zero currents → zero coil field."""
    currents = np.zeros(8, dtype=np.float32)
    field = coils.compute_field(currents)
    assert np.allclose(field, 0.0)


def test_influence_matrix_full_rank(coils: ShimCoilArray) -> None:
    """All 8 coil patterns are linearly independent (gradient basis has full rank)."""
    A = coils.influence_matrix.reshape(coils.num_coils, -1)
    rank = int(np.linalg.matrix_rank(A))
    assert rank == coils.num_coils, f"Expected rank {coils.num_coils}, got {rank}"


def test_single_coil_falloff(coils: ShimCoilArray) -> None:
    """Coil 0 (X-dipole) has higher influence at the +x edge than at x=0."""
    influence = coils.influence_matrix[0]
    center_row = coils.grid.size // 2

    # Horizontal slice at y≈0
    row = influence[center_row, :]
    # X-dipole peaks at rightmost column (x = +extent/2)
    assert row[-1] > row[coils.grid.size // 2], (
        "X-dipole coil influence should be greater at +x edge than at center"
    )


def test_current_clamping(coils: ShimCoilArray) -> None:
    """Currents exceeding max are clamped."""
    max_c = coils.config.max_current_amps
    over = np.full(8, max_c * 10, dtype=np.float32)
    clamped = np.full(8, max_c, dtype=np.float32)
    assert np.allclose(
        coils.compute_field(over), coils.compute_field(clamped)
    )


def test_batch_computation(coils: ShimCoilArray) -> None:
    """Batch input (B, num_coils) → (B, size, size) output."""
    B = 4
    currents = np.random.randn(B, 8).astype(np.float32)
    field = coils.compute_field(currents)
    assert field.shape == (B, 64, 64)


def test_influence_matrix_dtype(coils: ShimCoilArray) -> None:
    """Influence matrix is float32."""
    assert coils.influence_matrix.dtype == np.float32


def test_coil_positions_on_circle(coils: ShimCoilArray) -> None:
    """Coils are placed at the correct radius."""
    r = np.sqrt(coils.coil_x**2 + coils.coil_y**2)
    assert np.allclose(r, coils.config.coil_radius_mm, atol=1e-5)
