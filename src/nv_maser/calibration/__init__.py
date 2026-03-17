"""
Hardware calibration data interface for NV Maser digital twin.

Provides a standard FieldMap format so measured field maps from the hardware
repo (nv-maser-hardware/calibration/) plug directly into the simulation.

Workflow
--------
Simulation pipeline defines specs → hardware builds to spec → hardware
measures B₀ field → stores as FieldMap .npz → twin loads it as training
disturbance, closing the simulation-reality gap.

See nv-maser-hardware/calibration/FORMAT.md for the on-disk schema.
"""

from .field_map import (
    CompareResult,
    FieldMap,
    compare_maps,
    load_field_map,
    regrid,
    save_field_map,
    simulated_field_map,
    uniformity_ppm,
)

__all__ = [
    "CompareResult",
    "FieldMap",
    "compare_maps",
    "load_field_map",
    "regrid",
    "save_field_map",
    "simulated_field_map",
    "uniformity_ppm",
]
