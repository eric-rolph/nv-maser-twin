"""
Simulated shim coil array using analytically-defined spatial harmonic patterns.

Each coil produces one of the standard 2D magnetic shim-coil basis functions
(spatial harmonics: X, Y, XY, X²-Y², …), which is how real NMR/NV shim coils
are wound — each explicitly to cancel one specific term in the field-error
expansion.

This replaces the earlier 1/r² approximation, which produced nearly-flat patterns
inside the active zone and was therefore unable to cancel sinusoidal disturbances.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import CoilConfig
from .grid import SpatialGrid


# ---------------------------------------------------------------------------
# Gradient-coil basis functions (2D spatial harmonics, normalised to ‖·‖∞ = 1)
# ---------------------------------------------------------------------------

def _gradient_basis(xn: NDArray, yn: NDArray) -> list[NDArray]:
    """
    Return the eight standard 2D shim-coil basis patterns.

    Coordinates *xn* and *yn* should already be normalised so that the
    active-zone edge sits at ±1.  Each pattern is divided by its own L∞ norm
    so that at full current (1 A) the peak field contribution is exactly
    ``field_scale_factor`` Tesla regardless of polynomial order.

    Basis:
        0  X-dipole              ∝ x
        1  Y-dipole              ∝ y
        2  X²−Y² quadrupole      ∝ x²−y²
        3  XY quadrupole         ∝ xy
        4  Circular (Z₂ analog)  ∝ x²+y²
        5  X₃ hexapole           ∝ x³−3xy²
        6  Y₃ hexapole           ∝ 3x²y−y³
        7  XY(X²−Y²) octupole    ∝ xy(x²−y²)
    """
    raw = [
        xn,                          # [0] X-dipole
        yn,                          # [1] Y-dipole
        xn**2 - yn**2,               # [2] X²−Y² quadrupole
        xn * yn,                     # [3] XY quadrupole
        xn**2 + yn**2,               # [4] circular (Z₂ analog)
        xn * (xn**2 - 3 * yn**2),    # [5] X₃ hexapole
        yn * (3 * xn**2 - yn**2),    # [6] Y₃ hexapole
        xn * yn * (xn**2 - yn**2),   # [7] XY(X²−Y²) octupole
    ]
    # Normalise each pattern to L∞ = 1 so field_scale_factor is meaningful
    normalised = []
    for p in raw:
        max_abs = float(np.abs(p).max())
        normalised.append(p / (max_abs + 1e-12))
    return normalised


class ShimCoilArray:
    """
    Array of N shim coils whose field patterns are 2D spatial harmonics.

    Each coil corresponds to one basis function in the gradient expansion,
    so the controller has direct access to each spatial mode independently.

    Physics model::

        B_coil_c(x, y) = field_scale_factor × I_c × H_c(x̂, ŷ)

    where H_c is the c-th normalised harmonic, I_c is the coil current (A),
    x̂ = x / (extent/2), ŷ = y / (extent/2).
    """

    def __init__(self, grid: SpatialGrid, config: CoilConfig) -> None:
        self.grid = grid
        self.config = config
        self.num_coils = config.num_coils

        # Coil positions remain defined (used for visualisation / legacy tests)
        angles = np.linspace(0, 2 * np.pi, self.num_coils, endpoint=False)
        self.coil_x = config.coil_radius_mm * np.cos(angles)
        self.coil_y = config.coil_radius_mm * np.sin(angles)

        # Precompute the influence matrix
        self.influence_matrix = self._compute_influence_matrix()

    def _compute_influence_matrix(self) -> NDArray[np.float32]:
        """
        Precompute field contribution of each coil at 1 Amp current.

        Uses analytically-defined 2D spatial harmonic patterns (the standard
        basis for magnetic field shimming).  Each pattern is L∞-normalised
        then scaled by ``field_scale_factor`` so the caller can reason about
        peak field amplitudes directly.

        Returns:
            (num_coils, size, size) influence matrix in Tesla/Amp.
        """
        half_extent = self.grid.extent / 2.0        # mm
        xn = self.grid.x / half_extent              # ∈ [−1, 1]
        yn = self.grid.y / half_extent              # ∈ [−1, 1]

        basis = _gradient_basis(xn, yn)

        # Cycle through basis patterns for num_coils coils
        # (if num_coils > 8 the basis repeats — fine for testing)
        patterns = [basis[c % len(basis)] for c in range(self.num_coils)]
        influence = np.stack(patterns, axis=0) * self.config.field_scale_factor

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
        PyTorch differentiable version.

        Args:
            currents: (B, num_coils) tensor on the training device.
            influence_tensor: pre-baked (num_coils, H, W) on the same device.

        Returns:
            (B, H, W) coil field tensor.
        """
        return torch.einsum("bc,cij->bij", currents, influence_tensor)
