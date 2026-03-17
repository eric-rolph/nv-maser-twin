"""
FieldMap: standard container for 2-D magnetic field measurements.

Hardware units
--------------
- Field (b_z):        Tesla (float32)
- Positions (x, y):   millimetres (float32)
- Uniformity:         ppm (parts-per-million) = (ΔB / B₀) × 10⁶

File format
-----------
A single .npz archive with:
  b_z               float32  (height, width)     field values
  x_mm              float32  (width,)             x positions
  y_mm              float32  (height,)            y positions
  b0_nominal_tesla  float64  scalar               expected nominal field
  active_radius_mm  float64  scalar               active zone radius
  source            str                           "simulation" | "measurement"
  timestamp         str                           ISO 8601 (empty if unknown)
  notes             str                           free-form notes

See nv-maser-hardware/calibration/FORMAT.md for full details.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator


@dataclass
class FieldMap:
    """2-D magnetic field map.

    Attributes:
        b_z:               Field values in Tesla, shape (height, width).
        x_mm:              X-axis positions in mm, shape (width,).
        y_mm:              Y-axis positions in mm, shape (height,).
        b0_nominal_tesla:  Nominal (target) field strength in Tesla.
        active_radius_mm:  Radius of the active zone for uniformity metrics.
        source:            Data provenance: "simulation" or "measurement".
        timestamp:         ISO 8601 timestamp string (empty if unknown).
        notes:             Free-form annotation string.
    """

    b_z: NDArray[np.float32]
    x_mm: NDArray[np.float32]
    y_mm: NDArray[np.float32]
    b0_nominal_tesla: float
    active_radius_mm: float
    source: Literal["simulation", "measurement"] = "simulation"
    timestamp: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        self.b_z = np.asarray(self.b_z, dtype=np.float32)
        self.x_mm = np.asarray(self.x_mm, dtype=np.float32)
        self.y_mm = np.asarray(self.y_mm, dtype=np.float32)
        if self.b_z.ndim != 2:
            raise ValueError(f"b_z must be 2-D, got shape {self.b_z.shape}")
        if self.b_z.shape != (len(self.y_mm), len(self.x_mm)):
            raise ValueError(
                f"b_z shape {self.b_z.shape} inconsistent with "
                f"(len(y_mm)={len(self.y_mm)}, len(x_mm)={len(self.x_mm)})"
            )
        if self.b0_nominal_tesla <= 0:
            raise ValueError("b0_nominal_tesla must be positive")
        if self.active_radius_mm <= 0:
            raise ValueError("active_radius_mm must be positive")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def height(self) -> int:
        return int(self.b_z.shape[0])

    @property
    def width(self) -> int:
        return int(self.b_z.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return self.b_z.shape  # type: ignore[return-value]

    @property
    def active_zone_mask(self) -> NDArray[np.bool_]:
        """Boolean mask of pixels within the active zone (circular)."""
        xx, yy = np.meshgrid(self.x_mm, self.y_mm)
        return (xx**2 + yy**2) <= self.active_radius_mm**2


@dataclass
class CompareResult:
    """Statistics from comparing two FieldMaps on the same grid."""

    rms_residual_tesla: float
    max_abs_residual_tesla: float
    mean_residual_tesla: float
    correlation: float
    rms_residual_ppm: float


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------


def uniformity_ppm(fm: FieldMap, active_radius_mm: float | None = None) -> float:
    """Peak-to-peak field uniformity within the active zone, in ppm.

    Uniformity = (B_max − B_min) / B₀_nominal × 10⁶

    Args:
        fm:               FieldMap to evaluate.
        active_radius_mm: Override active zone radius (default: fm.active_radius_mm).

    Returns:
        Peak-to-peak uniformity in ppm.  0 ppm = perfectly uniform field.
    """
    radius = active_radius_mm if active_radius_mm is not None else fm.active_radius_mm
    xx, yy = np.meshgrid(fm.x_mm, fm.y_mm)
    mask = (xx**2 + yy**2) <= radius**2
    active_vals = fm.b_z[mask]
    if active_vals.size == 0:
        raise ValueError(
            f"No pixels within active_radius_mm={radius} mm. "
            f"Map spans x=[{fm.x_mm.min():.1f}, {fm.x_mm.max():.1f}], "
            f"y=[{fm.y_mm.min():.1f}, {fm.y_mm.max():.1f}]"
        )
    peak_to_peak = float(active_vals.max()) - float(active_vals.min())
    return float(peak_to_peak / fm.b0_nominal_tesla * 1e6)


def save_field_map(path: str | Path, fm: FieldMap) -> None:
    """Save a FieldMap to a .npz archive.

    Args:
        path:  Output path (.npz extension recommended).
        fm:    FieldMap to serialise.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        b_z=fm.b_z,
        x_mm=fm.x_mm,
        y_mm=fm.y_mm,
        b0_nominal_tesla=np.float64(fm.b0_nominal_tesla),
        active_radius_mm=np.float64(fm.active_radius_mm),
        source=np.array(fm.source, dtype="U"),
        timestamp=np.array(fm.timestamp, dtype="U"),
        notes=np.array(fm.notes, dtype="U"),
    )


def load_field_map(path: str | Path) -> FieldMap:
    """Load a FieldMap from a .npz archive.

    Args:
        path:  .npz file written by :func:`save_field_map`.

    Returns:
        Reconstructed FieldMap.

    Raises:
        ValueError: If required arrays are missing from the archive.
    """
    path = Path(path)
    data = np.load(str(path), allow_pickle=False)

    required = {"b_z", "x_mm", "y_mm", "b0_nominal_tesla", "active_radius_mm"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"FieldMap archive missing keys: {missing}")

    source_raw = data["source"].item() if "source" in data.files else "measurement"
    timestamp_raw = data["timestamp"].item() if "timestamp" in data.files else ""
    notes_raw = data["notes"].item() if "notes" in data.files else ""

    if source_raw not in ("simulation", "measurement"):
        source_raw = "measurement"

    return FieldMap(
        b_z=data["b_z"],
        x_mm=data["x_mm"],
        y_mm=data["y_mm"],
        b0_nominal_tesla=float(data["b0_nominal_tesla"]),
        active_radius_mm=float(data["active_radius_mm"]),
        source=source_raw,  # type: ignore[arg-type]
        timestamp=str(timestamp_raw),
        notes=str(notes_raw),
    )


def compare_maps(measured: FieldMap, reference: FieldMap) -> CompareResult:
    """Quantify the residual between two FieldMaps.

    Both maps must share the same grid (x_mm, y_mm).  If they differ in shape,
    the reference is regridded to the measured map's grid before comparison.

    Args:
        measured:   FieldMap from hardware measurement.
        reference:  FieldMap from simulation or another measurement.

    Returns:
        CompareResult with residual statistics.
    """
    if measured.shape != reference.shape or not (
        np.allclose(measured.x_mm, reference.x_mm)
        and np.allclose(measured.y_mm, reference.y_mm)
    ):
        reference = regrid(reference, measured.x_mm, measured.y_mm)

    residual = measured.b_z.astype(np.float64) - reference.b_z.astype(np.float64)
    rms = float(np.sqrt(np.mean(residual**2)))
    max_abs = float(np.max(np.abs(residual)))
    mean_res = float(np.mean(residual))

    # Pearson correlation over active zone of measured map
    mask = measured.active_zone_mask
    m_vals = measured.b_z[mask].astype(np.float64)
    r_vals = reference.b_z[mask].astype(np.float64)
    if m_vals.std() < 1e-15 or r_vals.std() < 1e-15:
        corr = 1.0 if np.allclose(m_vals, r_vals) else 0.0
    else:
        corr = float(np.corrcoef(m_vals, r_vals)[0, 1])

    rms_ppm = rms / measured.b0_nominal_tesla * 1e6

    return CompareResult(
        rms_residual_tesla=rms,
        max_abs_residual_tesla=max_abs,
        mean_residual_tesla=mean_res,
        correlation=corr,
        rms_residual_ppm=rms_ppm,
    )


def regrid(
    fm: FieldMap,
    target_x_mm: NDArray[np.float32],
    target_y_mm: NDArray[np.float32],
) -> FieldMap:
    """Interpolate a FieldMap onto a new (x, y) grid using bilinear interpolation.

    Args:
        fm:           Source FieldMap.
        target_x_mm: Target x positions in mm, 1-D monotonic.
        target_y_mm: Target y positions in mm, 1-D monotonic.

    Returns:
        New FieldMap on the target grid with the same metadata.
    """
    # RegularGridInterpolator expects ascending axes
    interp = RegularGridInterpolator(
        (fm.y_mm.astype(np.float64), fm.x_mm.astype(np.float64)),
        fm.b_z.astype(np.float64),
        method="linear",
        bounds_error=False,
        fill_value=None,  # extrapolate at edges
    )
    xx, yy = np.meshgrid(
        target_x_mm.astype(np.float64), target_y_mm.astype(np.float64)
    )
    pts = np.stack([yy.ravel(), xx.ravel()], axis=-1)
    new_b_z = interp(pts).reshape(len(target_y_mm), len(target_x_mm)).astype(np.float32)

    return FieldMap(
        b_z=new_b_z,
        x_mm=target_x_mm.copy(),
        y_mm=target_y_mm.copy(),
        b0_nominal_tesla=fm.b0_nominal_tesla,
        active_radius_mm=fm.active_radius_mm,
        source=fm.source,
        timestamp=fm.timestamp,
        notes=fm.notes,
    )


def simulated_field_map(
    config: "SimConfig",  # noqa: F821 – forward reference
    *,
    add_disturbance: bool = False,
    disturbance_seed: int | None = None,
) -> FieldMap:
    """Generate a FieldMap from the digital-twin simulator.

    This is the reference map for hardware commissioning: the hardware team
    compares their measured FieldMap against this to quantify the
    simulation ↔ reality gap.

    Args:
        config:           SimConfig (uses field, halbach, grid sections).
        add_disturbance:  If True, add one synthetic disturbance sample.
        disturbance_seed: Seed for disturbance RNG (if add_disturbance=True).

    Returns:
        FieldMap with source="simulation".
    """
    from ..physics.environment import FieldEnvironment
    from ..physics.disturbance import DisturbanceGenerator

    env = FieldEnvironment(config)
    field_2d = env.base_field.astype(np.float32)  # (size, size)

    if add_disturbance:
        dist_config = config.disturbance
        if disturbance_seed is not None:
            import copy
            dist_config = copy.copy(dist_config)
            dist_config = dist_config.model_copy(update={"seed": disturbance_seed})
        gen = DisturbanceGenerator(env.grid, dist_config)
        gen.randomize()
        field_2d = field_2d + gen.generate(t=0.0).astype(np.float32)

    grid = env.grid
    x_mm = grid.x[0, :].astype(np.float32)   # (size,)  first row
    y_mm = grid.y[:, 0].astype(np.float32)   # (size,)  first column

    return FieldMap(
        b_z=field_2d,
        x_mm=x_mm,
        y_mm=y_mm,
        b0_nominal_tesla=config.field.b0_tesla,
        active_radius_mm=config.grid.physical_extent_mm / 2.0 * config.grid.active_zone_fraction,
        source="simulation",
    )
