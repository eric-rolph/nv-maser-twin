"""
Halbach permanent magnet array — analytical multipole field model.

A K=2 (dipole) Halbach cylinder made from N discrete magnets produces:

1. **Dominant dipolar field** B₀ = Br × ln(R_out/R_in) × sin(π/N)/(π/N)
2. **Segmentation harmonics** at multipole orders n = kN±1 (k=1,2,…)
   Amplitudes: A_n = B₀ × (2/N) × sin(nπ/N)/(nπ/N - 1)  (approximate)
   Spatial dependence: ∝ (r/R_in)^(n-1)  → fall off sharply inside bore
3. **Manufacturing errors** — random multipole coefficients from:
   - Br variation (remanence scatter between segments)
   - Magnetisation angle errors (misaligned segment dipoles)
   - Position errors (radial displacement of segments)

All computed analytically via 2D multipole expansion — no FEM required.

Reference: Raich & Blümler, "Design and construction of a dipolar
Halbach array…", Concepts in MR Part B, 2004.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import HalbachConfig, FieldConfig
from .grid import SpatialGrid


@dataclass(frozen=True)
class MultipoleCoefficients:
    """Multipole expansion coefficients (a_n, b_n) for orders 2..max_order.

    The field at polar coordinates (r, θ) relative to bore centre is:
        B(r,θ) = B₀ + Σ_n [a_n·cos(nθ) + b_n·sin(nθ)] × (r/r_in)^(n-1)

    a_n: cosine (normal) coefficients, shape (max_order-1,) for n=2..max_order
    b_n: sine (skew) coefficients, shape (max_order-1,) for n=2..max_order
    """

    a_n: NDArray[np.float64]
    b_n: NDArray[np.float64]
    max_order: int
    b0_tesla: float

    def total_rms_error(self, r_fraction: float = 0.6) -> float:
        """RMS multipole error at a given fractional radius.

        Args:
            r_fraction: radius as fraction of inner_radius (default 0.6 = active zone edge).

        Returns:
            RMS field error in Tesla.
        """
        orders = np.arange(2, self.max_order + 1)
        radial = r_fraction ** (orders - 1)
        a_weighted = self.a_n * radial
        b_weighted = self.b_n * radial
        return float(np.sqrt(np.mean(a_weighted**2 + b_weighted**2)))


def compute_segmentation_harmonics(
    config: HalbachConfig,
    b0: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute systematic multipole coefficients from discrete segmentation.

    For N segments, field errors appear at orders n = kN±1 (k=1,2,…).
    Amplitude of each term falls as (2/N) × |sin(nπ/N)/(nπ/N)|.
    Only cosine (a_n) terms for a symmetric arrangement; b_n = 0.

    Args:
        config: Halbach geometry configuration.
        b0: Nominal dipolar field strength (Tesla).

    Returns:
        (a_n, b_n) arrays for orders 2..max_multipole_order.
    """
    N = config.num_segments
    max_n = config.max_multipole_order
    num_coeffs = max_n - 1  # orders 2..max_n

    a_n = np.zeros(num_coeffs, dtype=np.float64)
    b_n = np.zeros(num_coeffs, dtype=np.float64)

    for k in range(1, max_n + 1):
        for sign in (+1, -1):
            n = k * N + sign
            if n < 2 or n > max_n:
                continue
            # Segmentation harmonic amplitude (fraction of B₀)
            # A_n ≈ B₀ · (2/N) · |1/(n-1)| for the dominant term
            # More precisely: geometric series from each segment's contribution
            amp = b0 * (2.0 / N) * (1.0 / (n - 1))
            idx = n - 2  # array index for order n
            a_n[idx] += amp  # symmetric → cosine only

    return a_n, b_n


def compute_manufacturing_errors(
    config: HalbachConfig,
    b0: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute random multipole coefficients from manufacturing tolerances.

    Each tolerance source contributes to all multipole orders with
    random phase and amplitude scaling:
    - Br variation: ΔB₀ × σ_Br, distributed across orders
    - Angle error: B₀ × σ_angle, higher order contribution
    - Position error: B₀ × (σ_pos / R_in), gradient-like

    Args:
        config: Halbach geometry configuration.
        b0: Nominal dipolar field strength (Tesla).
        rng: NumPy random generator for reproducibility.

    Returns:
        (a_n, b_n) random multipole coefficients for orders 2..max_order.
    """
    max_n = config.max_multipole_order
    num_coeffs = max_n - 1
    N = config.num_segments

    a_n = np.zeros(num_coeffs, dtype=np.float64)
    b_n = np.zeros(num_coeffs, dtype=np.float64)

    # Scale factors for each error source (per segment, RSS over N segments → /√N)
    sqrt_n = np.sqrt(N)

    # 1. Remanence variation
    if config.br_tolerance_pct > 0:
        sigma_br = b0 * (config.br_tolerance_pct / 100.0) / sqrt_n
        a_n += rng.normal(0, sigma_br, num_coeffs)
        b_n += rng.normal(0, sigma_br, num_coeffs)

    # 2. Magnetisation angle error
    if config.angle_tolerance_deg > 0:
        sigma_angle = np.radians(config.angle_tolerance_deg)
        # Angle error couples as B₀ × sin(σ) ≈ B₀ × σ
        sigma_field = b0 * sigma_angle / sqrt_n
        a_n += rng.normal(0, sigma_field, num_coeffs)
        b_n += rng.normal(0, sigma_field, num_coeffs)

    # 3. Position error (gradient-like, strongest at low orders)
    if config.position_tolerance_mm > 0:
        sigma_pos = config.position_tolerance_mm / config.inner_radius_mm
        sigma_field = b0 * sigma_pos / sqrt_n
        # Weight toward lower orders (position errors are gradient-like)
        orders = np.arange(2, max_n + 1, dtype=np.float64)
        order_weight = 1.0 / orders  # lower orders get more contribution
        a_n += rng.normal(0, sigma_field, num_coeffs) * order_weight
        b_n += rng.normal(0, sigma_field, num_coeffs) * order_weight

    return a_n, b_n


def compute_multipole_coefficients(
    config: HalbachConfig,
    b0: float,
) -> MultipoleCoefficients:
    """
    Compute total multipole coefficients (systematic + manufacturing).

    Args:
        config: Halbach configuration.
        b0: Nominal B₀ (Tesla), either from config.ideal_b0_tesla or FieldConfig.

    Returns:
        MultipoleCoefficients with combined a_n, b_n arrays.
    """
    a_sys, b_sys = compute_segmentation_harmonics(config, b0)

    rng = np.random.default_rng(config.seed)
    a_rand, b_rand = compute_manufacturing_errors(config, b0, rng)

    return MultipoleCoefficients(
        a_n=a_sys + a_rand,
        b_n=b_sys + b_rand,
        max_order=config.max_multipole_order,
        b0_tesla=b0,
    )


def evaluate_multipole_field(
    grid: SpatialGrid,
    coefficients: MultipoleCoefficients,
    inner_radius_mm: float,
) -> NDArray[np.float32]:
    """
    Evaluate the multipole expansion on a 2D spatial grid.

    B(x,y) = B₀ + Σ_n [a_n·cos(nθ) + b_n·sin(nθ)] × (r/R_in)^(n-1)

    Points outside the bore (r > R_in) are clamped to r = R_in.

    Args:
        grid: 2D spatial grid with x, y coordinates in mm.
        coefficients: Multipole expansion coefficients.
        inner_radius_mm: Bore inner radius for normalisation.

    Returns:
        (size, size) field map in Tesla (float32).
    """
    # Polar coordinates
    r = grid.r.astype(np.float64)  # mm
    theta = np.arctan2(grid.y.astype(np.float64), grid.x.astype(np.float64))

    # Clamp r to bore radius (field expansion only valid inside)
    r_clamped = np.minimum(r, inner_radius_mm)
    rho = r_clamped / inner_radius_mm  # normalised radius ∈ [0, 1]

    # Start with B₀
    field = np.full_like(r, coefficients.b0_tesla)

    # Add multipole terms
    for idx, n in enumerate(range(2, coefficients.max_order + 1)):
        radial = rho ** (n - 1)
        field += (
            coefficients.a_n[idx] * np.cos(n * theta)
            + coefficients.b_n[idx] * np.sin(n * theta)
        ) * radial

    return field.astype(np.float32)


def compute_halbach_field(
    grid: SpatialGrid,
    field_config: FieldConfig,
    halbach_config: HalbachConfig,
) -> NDArray[np.float32]:
    """
    Compute the full Halbach field map on a 2D grid.

    Uses b0_tesla from FieldConfig as the nominal field (allowing override),
    then adds multipole error structure from the Halbach geometry.

    Args:
        grid: 2D spatial grid.
        field_config: Base field configuration (provides b0_tesla).
        halbach_config: Halbach array geometry and tolerances.

    Returns:
        (size, size) field map in Tesla.
    """
    b0 = field_config.b0_tesla

    coefficients = compute_multipole_coefficients(halbach_config, b0)
    return evaluate_multipole_field(grid, coefficients, halbach_config.inner_radius_mm)
