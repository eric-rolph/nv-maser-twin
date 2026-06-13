"""
Halbach buildability study: can a real magnet be shimmed into masing range?

The twin's own physics says the maser tolerates a field std of ~11 uT
(227 ppm of 50 mT) before inhomogeneous broadening kills the gain. A
discrete Halbach array has manufacturing errors (segment remanence,
magnetisation angle, position) that produce multipole field errors. This
study answers, per manufacturing-quality scenario:

  1. How big is the raw field error of a realistic array?
  2. Does static least-squares shimming with the 8-harmonic coil basis
     bring it inside the masing tolerance?

Unlike the synthetic training disturbances (band-limited harmonics that the
coil basis captures poorly), Halbach segmentation errors are dominated by
low-order multipoles inside the bore — exactly what shim coils are wound
to cancel. This is the physically real shimming problem.

Usage:
    python scripts/halbach_tolerance_study.py
    python scripts/halbach_tolerance_study.py --realizations 500
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nv_maser.config import SimConfig
from nv_maser.model.lstsq_baseline import LeastSquaresShimmer
from nv_maser.physics.base_field import compute_base_field
from nv_maser.physics.coils import ShimCoilArray
from nv_maser.physics.grid import SpatialGrid
from nv_maser.physics.maser_gain import max_tolerable_b_std


@dataclass(frozen=True)
class Scenario:
    name: str
    br_tolerance_pct: float
    angle_tolerance_deg: float
    position_tolerance_mm: float


SCENARIOS = [
    Scenario("precision (sorted N52)", 0.5, 0.5, 0.02),
    Scenario("standard (catalog N52)", 1.0, 1.0, 0.05),
    Scenario("budget (unsorted)", 2.0, 1.0, 0.10),
    Scenario("hand assembly", 5.0, 2.0, 0.30),
]


def run_study(n_realizations: int, num_segments: int) -> None:
    config = SimConfig()
    grid = SpatialGrid(config.grid)
    coils = ShimCoilArray(grid, config.coils)
    mask = grid.active_zone_mask
    shimmer = LeastSquaresShimmer(
        coils.influence_matrix,
        mask,
        config.coils.max_current_amps,
        ridge=0.0,
    )
    tolerable = max_tolerable_b_std(config.nv, config.maser)
    b0 = config.field.b0_tesla

    print(
        f"Halbach buildability study — {num_segments} segments, "
        f"{n_realizations} realizations per scenario"
    )
    print(
        f"masing tolerance: field std <= {tolerable*1e6:.2f} uT "
        f"({tolerable/b0*1e6:.0f} ppm of B0 = {b0*1e3:.0f} mT)\n"
    )
    header = (
        f"{'scenario':<24} {'raw std uT':>12} {'raw mase %':>11} "
        f"{'shim std uT':>12} {'shim p95 uT':>12} {'shim mase %':>12} "
        f"{'max |I| A':>10}"
    )
    print(header)
    print("-" * len(header))

    for sc in SCENARIOS:
        halbach = config.halbach.model_copy(
            update={
                "enabled": True,
                "num_segments": num_segments,
                "br_tolerance_pct": sc.br_tolerance_pct,
                "angle_tolerance_deg": sc.angle_tolerance_deg,
                "position_tolerance_mm": sc.position_tolerance_mm,
            }
        )

        raw_stds, shim_stds, peak_currents = [], [], []
        for seed in range(n_realizations):
            hb = halbach.model_copy(update={"seed": seed})
            field = compute_base_field(grid, config.field, hb)

            raw_stds.append(float(np.std(field[mask])))

            currents = shimmer.solve(field)
            net = field + coils.compute_field(currents)
            shim_stds.append(float(np.std(net[mask])))
            peak_currents.append(float(np.max(np.abs(currents))))

        raw = np.array(raw_stds)
        shim = np.array(shim_stds)
        print(
            f"{sc.name:<24} {np.median(raw)*1e6:>12.2f} "
            f"{np.mean(raw <= tolerable)*100:>10.1f}% "
            f"{np.median(shim)*1e6:>12.3f} "
            f"{np.percentile(shim, 95)*1e6:>12.3f} "
            f"{np.mean(shim <= tolerable)*100:>11.1f}% "
            f"{np.max(peak_currents):>10.3f}"
        )

    print(
        "\nraw  = as-built array, no shimming"
        "\nshim = after one static least-squares shim "
        "(8-harmonic coil basis, currents clipped to "
        f"+/-{config.coils.max_current_amps:g} A)"
        "\nmase % = fraction of realizations with field std inside the "
        "masing tolerance"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Halbach manufacturing-tolerance buildability study."
    )
    parser.add_argument("--realizations", type=int, default=200)
    parser.add_argument(
        "--segments",
        type=int,
        default=8,
        help="Number of Halbach segments (default 8)",
    )
    args = parser.parse_args()
    run_study(args.realizations, args.segments)


if __name__ == "__main__":
    main()
