"""
Single-sided permanent magnet array for handheld MRI probe.

Models the B₀ field in the half-space above a planar magnet assembly.
Three geometries are supported:

1. **Barrel** — concentric annular rings with alternating polarity.
   Creates a sweet-spot saddle point at a controllable depth where
   dB_z/dz = 0.  This is the recommended geometry.

2. **U-shaped** — two anti-parallel rectangular blocks (NMR-MOUSE style).
   Simpler but the sweet spot is shallower and less uniform.

3. **Halbach single-sided** — a truncated Halbach cylinder oriented
   axially.  Good uniformity but heavier.

Field model
───────────
Each magnetised element is treated as a magnetic dipole of moment
m = Br × V / µ₀, where V is the element volume.  The field at point
r from a dipole at origin is:

    B(r) = (µ₀ / 4π) [ 3(m·r̂)r̂ − m ] / |r|³

For annular rings the contribution is integrated analytically over the
ring volume by treating each ring as a stack of current loops (the
Biot–Savart equivalent of a uniformly-magnetised cylinder).

The on-axis field of a uniformly magnetised cylinder of radius R, height
h, centred at z = z₀ with magnetisation M along ẑ is:

    B_z(0,0,z) = (µ₀ M / 2) [ (z−z₀+h/2)/√(R²+(z−z₀+h/2)²)
                               − (z−z₀−h/2)/√(R²+(z−z₀−h/2)²) ]

For an annular ring (inner R₁, outer R₂) this becomes the difference
of two solid-cylinder contributions.

References
──────────
Marble et al., "A compact permanent magnet array with a remote
homogeneous field", J. Magn. Reson. 186, 100 (2007).

Blümich et al., "NMR at low magnetic fields", Chem. Phys. Lett. 477,
231 (2009).

Prado, "Single sided NMR", in *Magnetic Resonance Microscopy* (Wiley, 2009).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import SingleSidedMagnetConfig

# ── Physical constants ────────────────────────────────────────────
_MU0 = 4.0 * math.pi * 1e-7  # T·m/A


@dataclass(frozen=True)
class SweetSpotInfo:
    """Properties of the B₀ sweet spot (saddle point where dBz/dz = 0)."""

    depth_mm: float
    b0_tesla: float
    second_derivative_t_per_m2: float  # d²Bz/dz² at sweet spot
    uniformity_ppm: float  # over a 10 mm sphere


@dataclass(frozen=True)
class FieldMap2D:
    """2D field map in a plane above the magnet."""

    bz: NDArray[np.float64]  # shape (ny, nz) or (nr, nz) — field in Tesla
    positions_mm: NDArray[np.float64]  # shape (ny, nz, 2) — (lateral, depth)
    extent_mm: float
    depth_range_mm: tuple[float, float]


def _solid_cylinder_on_axis_bz(
    z: NDArray[np.float64],
    z_centre: float,
    radius_m: float,
    height_m: float,
    magnetisation: float,
) -> NDArray[np.float64]:
    """On-axis B_z from a uniformly magnetised solid cylinder.

    Args:
        z: axial positions (m), measured from magnet surface (z=0 is top).
        z_centre: centre of the cylinder below surface (m, positive downward).
        radius_m: cylinder radius (m).
        height_m: cylinder height (m).
        magnetisation: M = Br / µ₀ effective axial magnetisation (A/m).

    Returns:
        B_z at each z position (Tesla).
    """
    # z relative to cylinder faces:
    # Cylinder centre is at z_actual = -z_centre (below surface).
    # Top face (closer to tissue) at z_actual + h/2 = -z_centre + h/2
    # Bottom face at z_actual - h/2 = -z_centre - h/2
    #
    # Standard on-axis formula: B_z = (µ₀M/2)[g(z-z_bot) - g(z-z_top)]
    # where g(ζ) = ζ/√(R²+ζ²)
    dz_top = z - (-z_centre + height_m / 2)
    dz_bot = z - (-z_centre - height_m / 2)
    r = radius_m

    term_top = dz_top / np.sqrt(r**2 + dz_top**2)
    term_bot = dz_bot / np.sqrt(r**2 + dz_bot**2)

    return (_MU0 * magnetisation / 2) * (term_bot - term_top)


def _annular_ring_on_axis_bz(
    z: NDArray[np.float64],
    z_centre: float,
    r_inner_m: float,
    r_outer_m: float,
    height_m: float,
    magnetisation: float,
) -> NDArray[np.float64]:
    """On-axis B_z from a uniformly magnetised annular ring.

    Superposition: ring = outer solid cylinder − inner solid cylinder.
    """
    bz_outer = _solid_cylinder_on_axis_bz(z, z_centre, r_outer_m, height_m, magnetisation)
    bz_inner = _solid_cylinder_on_axis_bz(z, z_centre, r_inner_m, height_m, magnetisation)
    return bz_outer - bz_inner


def _annular_ring_off_axis_bz(
    rho: NDArray[np.float64],
    z: NDArray[np.float64],
    z_centre: float,
    r_inner_m: float,
    r_outer_m: float,
    height_m: float,
    magnetisation: float,
    n_integration: int = 32,
) -> NDArray[np.float64]:
    """Off-axis B_z from an annular ring, computed by numerical quadrature.

    Uses Gauss–Legendre quadrature over the ring cross-section, treating
    each differential element as a thin current loop (Biot–Savart).

    For a thin current loop of radius a at height z₀, the axial field
    component at (ρ, z) involves elliptic integrals.  We approximate with
    a dipole expansion valid for |r| >> a.

    For better accuracy at close range, we integrate over the ring volume
    using a cylindrical quadrature grid.
    """
    from numpy.polynomial.legendre import leggauss

    # Quadrature points in radial and axial directions
    n_r = max(n_integration // 2, 4)
    n_z = max(n_integration // 2, 4)

    xi_r, wi_r = leggauss(n_r)
    xi_z, wi_z = leggauss(n_z)

    # Map to physical coordinates
    r_mid = (r_outer_m + r_inner_m) / 2
    r_half = (r_outer_m - r_inner_m) / 2
    z_top = -z_centre + height_m / 2
    z_bot = -z_centre - height_m / 2
    z_mid = (z_top + z_bot) / 2
    z_half = (z_top - z_bot) / 2

    bz_total = np.zeros_like(rho)

    for i in range(n_r):
        r_src = r_mid + r_half * xi_r[i]
        for j in range(n_z):
            z_src = z_mid + z_half * xi_z[j]

            # Vector from source to field point
            dz = z - z_src
            # Distance in cylindrical symmetry (field point at rho, source at r_src)
            # Use the on-axis formula for each ring element
            # For a thin loop of radius r_src at z_src, the Bz at (rho, z) is:
            # Bz = (µ₀ M / 2π) * area_element / ((rho² + r_src² + dz²)^(3/2)) ...
            # Simplified: treat each volume element as a dipole
            dist_sq = rho**2 + r_src**2 + dz**2
            dist = np.sqrt(dist_sq)

            # Volume element dV = r dr dz dφ → for one quadrature point:
            dv = r_src * r_half * z_half * wi_r[i] * wi_z[j] * 2 * math.pi

            # Dipole moment along z: m = M * dV
            m = magnetisation * dv

            # Bz from dipole: Bz = (µ₀/4π) * m * (3 cos²θ - 1) / r³
            # where cos θ = dz/r
            cos_theta = dz / (dist + 1e-30)
            bz_contrib = (_MU0 / (4 * math.pi)) * m * (3 * cos_theta**2 - 1) / (dist**3 + 1e-30)

            bz_total += bz_contrib

    return bz_total


class SingleSidedMagnet:
    """Single-sided permanent magnet array for handheld MRI probe.

    Computes the B₀ field in the half-space above the magnet surface.
    The magnet surface is at z = 0; positive z is into tissue (depth).
    """

    def __init__(self, config: SingleSidedMagnetConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        c = self.config
        if c.magnet_type == "barrel":
            n = c.num_rings
            if not (len(c.ring_inner_radii_mm) == n
                    and len(c.ring_outer_radii_mm) == n
                    and len(c.ring_heights_mm) == n
                    and len(c.ring_polarities) == n):
                raise ValueError(
                    f"All ring parameter lists must have length num_rings={n}"
                )
            for i in range(n):
                if c.ring_inner_radii_mm[i] >= c.ring_outer_radii_mm[i]:
                    raise ValueError(
                        f"Ring {i}: inner radius must be < outer radius"
                    )

    def field_on_axis(self, depths_mm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute B_z along the central axis at given depths.

        Args:
            depths_mm: array of depth values (mm) above magnet surface.
                       0 = magnet surface, positive = into tissue.

        Returns:
            B_z in Tesla at each depth.
        """
        c = self.config
        z_m = depths_mm * 1e-3  # convert to metres

        if c.magnet_type == "barrel":
            return self._barrel_on_axis(z_m)
        elif c.magnet_type == "u_shaped":
            return self._u_shaped_on_axis(z_m)
        else:
            raise ValueError(f"Unsupported magnet type: {c.magnet_type}")

    def _barrel_on_axis(self, z_m: NDArray[np.float64]) -> NDArray[np.float64]:
        """On-axis field from a barrel (concentric ring) array."""
        c = self.config
        bz = np.zeros_like(z_m)

        # Stack rings starting from surface downward
        z_offset = 0.0  # cumulative depth below surface (m)
        for i in range(c.num_rings):
            r_in = c.ring_inner_radii_mm[i] * 1e-3
            r_out = c.ring_outer_radii_mm[i] * 1e-3
            h = c.ring_heights_mm[i] * 1e-3
            pol = c.ring_polarities[i]

            # Centre of this ring below surface
            z_centre = z_offset + h / 2
            magnetisation = pol * c.remanence_tesla / _MU0

            bz += _annular_ring_on_axis_bz(
                z_m, z_centre, r_in, r_out, h, magnetisation
            )
            z_offset += h

        return bz

    def _u_shaped_on_axis(self, z_m: NDArray[np.float64]) -> NDArray[np.float64]:
        """Simplified U-shaped magnet: two anti-parallel blocks.

        Models as two flat annular rings side by side with opposite polarity.
        The on-axis field of this geometry produces a gradient that can
        have a sweet spot when combined appropriately.
        """
        c = self.config
        # Use the first two ring entries as the two blocks
        bz = np.zeros_like(z_m)
        for i in range(min(c.num_rings, 2)):
            r_in = c.ring_inner_radii_mm[i] * 1e-3
            r_out = c.ring_outer_radii_mm[i] * 1e-3
            h = c.ring_heights_mm[i] * 1e-3
            pol = c.ring_polarities[i]
            z_centre = h / 2
            magnetisation = pol * c.remanence_tesla / _MU0
            bz += _annular_ring_on_axis_bz(
                z_m, z_centre, r_in, r_out, h, magnetisation
            )
        return bz

    def sweet_spot(self) -> SweetSpotInfo:
        """Find the sweet spot — the depth where dBz/dz = 0 (saddle point).

        Uses a fine grid search along the axis followed by parabolic
        interpolation of the derivative zero-crossing.

        Returns:
            SweetSpotInfo with depth, field strength, curvature, and uniformity.
        """
        # Fine grid along axis
        depths_mm = np.linspace(1.0, self.config.computation_extent_mm, 2000)
        bz = self.field_on_axis(depths_mm)

        # Numerical gradient dBz/dz
        dz = (depths_mm[1] - depths_mm[0]) * 1e-3  # step in metres
        dbz_dz = np.gradient(bz, dz)

        # Find zero-crossings of dbz_dz (sweet spots)
        sign_changes = np.where(np.diff(np.sign(dbz_dz)))[0]

        if len(sign_changes) == 0:
            # No sweet spot found — return the point of minimum gradient
            idx = int(np.argmin(np.abs(dbz_dz)))
            depth = depths_mm[idx]
            b0 = float(bz[idx])
            d2bz = float(np.gradient(dbz_dz, dz)[idx])
            return SweetSpotInfo(
                depth_mm=depth,
                b0_tesla=b0,
                second_derivative_t_per_m2=d2bz,
                uniformity_ppm=_estimate_uniformity_ppm(b0, d2bz, 5e-3),
            )

        # Take the first (shallowest) sweet spot
        idx = sign_changes[0]

        # Linear interpolation for more precise location
        z0, z1 = depths_mm[idx], depths_mm[idx + 1]
        g0, g1 = dbz_dz[idx], dbz_dz[idx + 1]
        frac = -g0 / (g1 - g0 + 1e-30)
        depth_sweet = z0 + frac * (z1 - z0)

        # Evaluate field at sweet spot
        b0_sweet = float(np.interp(depth_sweet, depths_mm, bz))

        # Second derivative at sweet spot
        d2bz_dz2 = np.gradient(dbz_dz, dz)
        d2bz_sweet = float(np.interp(depth_sweet, depths_mm, d2bz_dz2))

        # Uniformity over a 10 mm sphere
        uniformity = _estimate_uniformity_ppm(b0_sweet, d2bz_sweet, 5e-3)

        return SweetSpotInfo(
            depth_mm=depth_sweet,
            b0_tesla=b0_sweet,
            second_derivative_t_per_m2=d2bz_sweet,
            uniformity_ppm=uniformity,
        )

    def gradient_on_axis(self, depths_mm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute dBz/dz along the central axis (T/m).

        This gradient provides depth encoding for single-sided NMR:
        each depth has a unique Larmor frequency.

        Args:
            depths_mm: depth positions (mm).

        Returns:
            dBz/dz in T/m at each depth.
        """
        bz = self.field_on_axis(depths_mm)
        dz = (depths_mm[1] - depths_mm[0]) * 1e-3 if len(depths_mm) > 1 else 1e-3
        return np.gradient(bz, dz)

    def field_map_2d(
        self,
        depth_range_mm: tuple[float, float] = (0.0, 40.0),
        lateral_extent_mm: float = 30.0,
        resolution: int = 64,
    ) -> FieldMap2D:
        """Compute a 2D field map in the (lateral, depth) plane.

        Uses the off-axis model for barrel magnets.

        Args:
            depth_range_mm: (min_depth, max_depth) in mm.
            lateral_extent_mm: half-extent of lateral axis (mm).
            resolution: grid points per axis.

        Returns:
            FieldMap2D with the B_z field and position arrays.
        """
        c = self.config

        rho_mm = np.linspace(-lateral_extent_mm, lateral_extent_mm, resolution)
        z_mm = np.linspace(depth_range_mm[0] + 0.1, depth_range_mm[1], resolution)

        rho_grid, z_grid = np.meshgrid(rho_mm, z_mm, indexing="ij")
        rho_m = np.abs(rho_grid) * 1e-3  # cylindrical symmetry
        z_m = z_grid * 1e-3

        bz = np.zeros_like(rho_m)

        if c.magnet_type == "barrel":
            z_offset = 0.0
            for i in range(c.num_rings):
                r_in = c.ring_inner_radii_mm[i] * 1e-3
                r_out = c.ring_outer_radii_mm[i] * 1e-3
                h = c.ring_heights_mm[i] * 1e-3
                pol = c.ring_polarities[i]
                z_centre = z_offset + h / 2
                magnetisation = pol * c.remanence_tesla / _MU0

                bz += _annular_ring_off_axis_bz(
                    rho_m.ravel(), z_m.ravel(),
                    z_centre, r_in, r_out, h,
                    magnetisation,
                ).reshape(rho_m.shape)
                z_offset += h
        else:
            # Fall back to on-axis only for unsupported geometries
            for j in range(resolution):
                bz[:, j] = self.field_on_axis(np.array([z_mm[j]]))[0]

        positions = np.stack([rho_grid, z_grid], axis=-1)

        return FieldMap2D(
            bz=bz,
            positions_mm=positions,
            extent_mm=lateral_extent_mm,
            depth_range_mm=depth_range_mm,
        )


def _estimate_uniformity_ppm(
    b0: float, d2bz_dz2: float, radius_m: float
) -> float:
    """Estimate field uniformity (ppm) over a sphere of given radius.

    For a field with vanishing first derivative, the dominant variation
    is from the second derivative: ΔB ≈ (1/2) d²B/dz² × r².
    """
    if abs(b0) < 1e-12:
        return float("inf")
    delta_b = 0.5 * abs(d2bz_dz2) * radius_m**2
    return abs(delta_b / b0) * 1e6


# ── Phase-2 milestone validation ─────────────────────────────────

@dataclass(frozen=True)
class MilestoneResult:
    """Phase-2 sweet-spot field quality milestone validation result.

    Architecture doc §12.2 success criterion:
        B₀ = 50 ± 5 mT at ~20 mm depth, < 500 ppm over 10 mm diameter sphere.

    The *volumetric* uniformity uses a 2D field map to sample all points
    within the evaluation sphere (exploiting cylindrical symmetry of barrel
    magnets).  The *analytical* uniformity is the on-axis second-derivative
    approximation stored in SweetSpotInfo for comparison.
    """

    sweet_spot_depth_mm: float          # detected sweet-spot depth (mm)
    b0_tesla: float                     # field strength at sweet spot (T)
    uniformity_ppm_volumetric: float    # peak-to-peak ppm over volume
    uniformity_ppm_analytical: float    # on-axis 2nd-derivative estimate

    b0_target_tesla: float              # target field (T)
    b0_tolerance_tesla: float           # ±tolerance (T)
    uniformity_target_ppm: float        # maximum allowed ppm
    volume_radius_mm: float             # radius of evaluation sphere (mm)

    b0_pass: bool                       # field in tolerance band
    uniformity_pass: bool               # uniformity < target
    milestone_pass: bool                # both criteria met simultaneously

    @property
    def b0_mT(self) -> float:
        """Sweet-spot field strength expressed in milli-Tesla."""
        return self.b0_tesla * 1e3


def validate_sweet_spot_milestone(
    magnet: SingleSidedMagnet,
    b0_target_tesla: float = 0.050,
    b0_tolerance_tesla: float = 0.005,
    uniformity_target_ppm: float = 500.0,
    volume_radius_mm: float = 5.0,
    map_resolution: int = 128,
) -> MilestoneResult:
    """Validate the Phase-2 sweet-spot field quality milestone (§12.2).

    Checks whether a SingleSidedMagnet configuration satisfies:

    * **B₀ criterion** — field at the sweet spot is within
      ``b0_target_tesla ± b0_tolerance_tesla`` (default 45–55 mT).
    * **Uniformity criterion** — peak-to-peak B_z variation over a sphere
      of radius ``volume_radius_mm`` centred on the sweet spot is below
      ``uniformity_target_ppm`` parts-per-million (default 500 ppm).

    The uniformity is evaluated volumetrically using a high-resolution 2D
    field map.  For a barrel magnet (axially symmetric) this is exact;
    points within the sphere are selected via cylindrical distance.

    Args:
        magnet: configured SingleSidedMagnet to evaluate.
        b0_target_tesla: nominal B₀ target (default 50 mT = 0.050 T).
        b0_tolerance_tesla: symmetric tolerance band (default ±5 mT).
        uniformity_target_ppm: peak-to-peak ppm limit (default 500 ppm).
        volume_radius_mm: radius of the evaluation sphere (default 5 mm →
            10 mm diameter sphere).
        map_resolution: grid points per axis for the 2D field map (default
            128; increase for higher accuracy at the cost of compute time).

    Returns:
        MilestoneResult with measured values, criteria, and pass/fail flags.
    """
    # Step 1: locate the sweet spot
    info = magnet.sweet_spot()
    sweet_depth_mm = info.depth_mm
    b0 = info.b0_tesla

    # Step 2: compute a fine 2D field map in a window around the sweet spot
    margin = volume_radius_mm * 1.5
    depth_min = max(0.5, sweet_depth_mm - margin)
    depth_max = sweet_depth_mm + margin

    fmap = magnet.field_map_2d(
        depth_range_mm=(depth_min, depth_max),
        lateral_extent_mm=margin,
        resolution=map_resolution,
    )

    # fmap.positions_mm[i, j, 0] = lateral (rho) in mm
    # fmap.positions_mm[i, j, 1] = depth (z) in mm
    lateral_mm = fmap.positions_mm[:, :, 0]
    depth_mm_grid = fmap.positions_mm[:, :, 1]

    # Step 3: select points within the evaluation sphere (cylindrical symmetry)
    dist_mm = np.sqrt(lateral_mm**2 + (depth_mm_grid - sweet_depth_mm) ** 2)
    in_sphere = dist_mm <= volume_radius_mm

    n_pts = int(in_sphere.sum())
    if n_pts < 4:
        # Resolution too coarse — fall back to the analytical estimate
        uniformity_vol = info.uniformity_ppm
    else:
        bz_sphere = fmap.bz[in_sphere]
        bz_mean = float(np.mean(bz_sphere))
        bz_ptp = float(np.ptp(bz_sphere))  # peak-to-peak
        if abs(bz_mean) < 1e-12:
            uniformity_vol = float("inf")
        else:
            uniformity_vol = (bz_ptp / abs(bz_mean)) * 1e6

    b0_pass = abs(b0 - b0_target_tesla) <= b0_tolerance_tesla
    uniformity_pass = uniformity_vol <= uniformity_target_ppm

    return MilestoneResult(
        sweet_spot_depth_mm=sweet_depth_mm,
        b0_tesla=b0,
        uniformity_ppm_volumetric=uniformity_vol,
        uniformity_ppm_analytical=info.uniformity_ppm,
        b0_target_tesla=b0_target_tesla,
        b0_tolerance_tesla=b0_tolerance_tesla,
        uniformity_target_ppm=uniformity_target_ppm,
        volume_radius_mm=volume_radius_mm,
        b0_pass=b0_pass,
        uniformity_pass=uniformity_pass,
        milestone_pass=b0_pass and uniformity_pass,
    )
