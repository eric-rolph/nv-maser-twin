"""
Magpylib adapter for cross-validating single-sided magnet models.

Builds a magpylib 3D magnet collection from our ``SingleSidedMagnetConfig``
so we can compare our analytical on-axis/off-axis field calculations against
magpylib's validated analytical expressions (Cuboid/CylinderSegment sources).

Magpylib uses SI units throughout (metres and Tesla).  Our config stores
dimensions in millimetres; this module handles the conversion.

Usage
─────
>>> from nv_maser.config import SingleSidedMagnetConfig
>>> from nv_maser.physics.magpylib_adapter import build_magpylib_collection
>>> cfg = SingleSidedMagnetConfig()
>>> col = build_magpylib_collection(cfg)
>>> col.getB([0, 0, 0.020])  # B-vector at 20 mm depth (in Tesla)

Cross-validation
────────────────
>>> from nv_maser.physics.magpylib_adapter import compare_on_axis
>>> diffs = compare_on_axis(cfg, depths_mm=np.linspace(5, 40, 50))
>>> print(f"Max relative diff: {diffs['max_relative_diff']:.2%}")
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..config import SingleSidedMagnetConfig

try:
    import magpylib as magpy

    MAGPYLIB_AVAILABLE = True
except ImportError:
    MAGPYLIB_AVAILABLE = False


def _require_magpylib() -> None:
    if not MAGPYLIB_AVAILABLE:
        raise ImportError(
            "magpylib is required for 3D magnet validation. "
            "Install it with: pip install magpylib"
        )


def build_magpylib_collection(
    config: SingleSidedMagnetConfig,
) -> "magpy.Collection":
    """Convert a barrel magnet config into a magpylib Collection.

    Each annular ring is modelled as two coaxial ``CylinderSegment``
    sources (a full 360° segment with inner/outer radii) with axial
    polarization matching the ring polarity.

    Coordinate convention
    ─────────────────────
    - Magnet surface is at z = 0.
    - Rings are stacked *below* the surface (negative z).
    - Tissue/probe space is at positive z.

    This matches our ``SingleSidedMagnet`` convention where
    ``depths_mm > 0`` means above the magnet surface.

    Returns:
        magpylib.Collection containing all ring sources.
    """
    _require_magpylib()

    if config.magnet_type != "barrel":
        raise NotImplementedError(
            f"magpylib adapter only supports 'barrel' type, got '{config.magnet_type}'"
        )

    sources = []
    z_offset_mm = 0.0  # cumulative depth below surface (positive downward)

    for i in range(config.num_rings):
        r_inner = config.ring_inner_radii_mm[i]  # mm
        r_outer = config.ring_outer_radii_mm[i]  # mm
        h = config.ring_heights_mm[i]  # mm

        # Convert mm to m for magpylib (SI units)
        r_inner_m = r_inner * 1e-3
        r_outer_m = r_outer * 1e-3
        h_m = h * 1e-3

        # Polarization: Br along z, sign from polarity
        pol = config.ring_polarities[i]
        polarization = (0, 0, pol * config.remanence_tesla)

        # CylinderSegment: full 360° annular ring
        # dimension = (r_inner, r_outer, height, phi_start, phi_end)
        ring = magpy.magnet.CylinderSegment(
            polarization=polarization,
            dimension=(r_inner_m, r_outer_m, h_m, 0, 360),
        )

        # Position: centre of ring below surface
        # Ring stacks downward: centre at -(z_offset + h/2)
        z_centre_m = -(z_offset_mm + h / 2) * 1e-3
        ring.position = (0, 0, z_centre_m)

        sources.append(ring)
        z_offset_mm += h

    return magpy.Collection(*sources)


def field_on_axis_magpylib(
    config: SingleSidedMagnetConfig,
    depths_mm: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Bz along the central axis using magpylib.

    Args:
        config: Barrel magnet configuration.
        depths_mm: Depth positions above surface (mm), positive = into tissue.

    Returns:
        Bz in Tesla at each depth (1D array).
    """
    _require_magpylib()
    collection = build_magpylib_collection(config)

    # Observation points on z-axis: (0, 0, depth_m)
    observers = np.zeros((len(depths_mm), 3))
    observers[:, 2] = depths_mm * 1e-3  # convert mm to m

    # magpylib returns shape (N, 3) — Bx, By, Bz
    b_vectors = np.atleast_2d(collection.getB(observers))
    return b_vectors[:, 2]  # extract Bz component


def field_map_2d_magpylib(
    config: SingleSidedMagnetConfig,
    depth_range_mm: tuple[float, float] = (1.0, 40.0),
    lateral_extent_mm: float = 30.0,
    resolution: int = 32,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute a 2D Bz field map in the (x, z) plane using magpylib.

    Returns:
        Tuple of (bz, x_mm, z_mm) where bz has shape (resolution, resolution).
    """
    _require_magpylib()
    collection = build_magpylib_collection(config)

    x_mm = np.linspace(-lateral_extent_mm, lateral_extent_mm, resolution)
    z_mm = np.linspace(depth_range_mm[0], depth_range_mm[1], resolution)
    xx, zz = np.meshgrid(x_mm, z_mm, indexing="ij")

    observers = np.zeros((xx.size, 3))
    observers[:, 0] = xx.ravel() * 1e-3
    observers[:, 2] = zz.ravel() * 1e-3

    b_vectors = collection.getB(observers)
    bz = b_vectors[:, 2].reshape(xx.shape)

    return bz, x_mm, z_mm


def compare_on_axis(
    config: SingleSidedMagnetConfig,
    depths_mm: NDArray[np.float64] | None = None,
) -> dict:
    """Compare our analytical model against magpylib on the central axis.

    Returns a dict with:
        - depths_mm: depth array
        - bz_analytical: field from our model (T)
        - bz_magpylib: field from magpylib (T)
        - abs_diff: absolute difference (T)
        - rel_diff: relative difference (fraction)
        - max_abs_diff: max absolute difference (T)
        - max_relative_diff: max relative difference (fraction)
        - rms_relative_diff: RMS relative difference
    """
    from .single_sided_magnet import SingleSidedMagnet

    _require_magpylib()

    if depths_mm is None:
        depths_mm = np.linspace(5.0, 50.0, 100)
    depths_mm = np.asarray(depths_mm, dtype=np.float64)

    # Our analytical model
    mag = SingleSidedMagnet(config)
    bz_analytical = mag.field_on_axis(depths_mm)

    # Magpylib reference
    bz_magpylib = field_on_axis_magpylib(config, depths_mm)

    abs_diff = np.abs(bz_analytical - bz_magpylib)

    # Relative difference (avoid division by zero)
    bz_ref = np.maximum(np.abs(bz_magpylib), 1e-15)
    rel_diff = abs_diff / bz_ref

    return {
        "depths_mm": depths_mm,
        "bz_analytical": bz_analytical,
        "bz_magpylib": bz_magpylib,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "max_abs_diff": float(np.max(abs_diff)),
        "max_relative_diff": float(np.max(rel_diff)),
        "rms_relative_diff": float(np.sqrt(np.mean(rel_diff**2))),
    }


def find_sweet_spot_magpylib(
    config: SingleSidedMagnetConfig,
    depth_range_mm: tuple[float, float] = (1.0, 60.0),
    n_points: int = 2000,
) -> dict:
    """Find the sweet spot (dBz/dz = 0) using magpylib field computation.

    Returns dict with depth_mm, b0_tesla, gradient_at_sweet_spot.
    """
    _require_magpylib()

    depths = np.linspace(depth_range_mm[0], depth_range_mm[1], n_points)
    bz = field_on_axis_magpylib(config, depths)

    dz_m = (depths[1] - depths[0]) * 1e-3
    dbz_dz = np.gradient(bz, dz_m)

    # Find zero-crossings
    sign_changes = np.where(np.diff(np.sign(dbz_dz)))[0]

    if len(sign_changes) == 0:
        idx = int(np.argmin(np.abs(dbz_dz)))
    else:
        idx = sign_changes[0]
        # Linear interpolation
        z0, z1 = depths[idx], depths[idx + 1]
        g0, g1 = dbz_dz[idx], dbz_dz[idx + 1]
        frac = -g0 / (g1 - g0 + 1e-30)
        depth_sweet = z0 + frac * (z1 - z0)
        b0_sweet = float(np.interp(depth_sweet, depths, bz))

        return {
            "depth_mm": float(depth_sweet),
            "b0_tesla": b0_sweet,
            "gradient_at_sweet_spot": float(np.interp(depth_sweet, depths, dbz_dz)),
        }

    return {
        "depth_mm": float(depths[idx]),
        "b0_tesla": float(bz[idx]),
        "gradient_at_sweet_spot": float(dbz_dz[idx]),
    }
