"""Mu-metal magnetic shielding model for the maser module (Risk R3 mitigation).

The handheld probe architecture places a single-sided imaging magnet adjacent to
the maser module.  Its fringe field is the primary environmental disturbance that
can destabilise oscillation.  This module models the passive isolation provided
by a closed mu-metal shell surrounding the maser cavity, which is the first
defence layer before active shimming coil compensation.

Risk R3 (Architecture doc §13):
    "Stray field from imaging magnet destabilizes maser — Maser fails — Mu-metal
    shielding, distance, active compensation (twin models this)."

Architecture target (§6.5):
    ~50 dB attenuation to reduce ~30 mT stray at 40 mm separation to < 100 µT.

Physics
-------
Spherical shell (exact, uniform external field, Jackson §5.12):

    S_sphere = [(2µ_r+1)(µ_r+2) − 2(µ_r−1)²(r₁/r₂)³] / (9µ_r)

Cylindrical shell (exact, transverse uniform field, infinite cylinder):

    S_cyl = [(µ_r+1)² − (µ_r−1)²(r₁/r₂)²] / (4µ_r)

Thin-shell approximations (t ≪ r₁, µ_r ≫ 1):

    S_sphere ≈ 2µ_r t / (3 r₁)
    S_cyl    ≈   µ_r t / (2 r₁)

Multi-layer (n concentric shells, product approximation):

    S_total ≈ ∏ᵢ Sᵢ   (valid when interlayer gap ≫ shell thickness)

References
----------
- Jackson, J. D., *Classical Electrodynamics* (3rd ed.) §5.12
- Rikitake, T., *Magnetic Shielding*, 1966 (cylindrical formula)
- Handheld probe architecture doc §6.5 and §13 Risk Register R3
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ── Default mu-metal material constants (ASTM A753 type 4 / Permalloy 80) ─────
_DEFAULT_MU_R: float = 50_000.0          # typical DC relative permeability
_DEFAULT_DENSITY_KG_M3: float = 8_700.0  # kg/m³ (≈ Permalloy density)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Configuration dataclasses                                       ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class MuMetalShellConfig:
    """Geometry and material parameters for a mu-metal shielding enclosure.

    The enclosure is a single closed shell (or n nested shells) of soft
    ferromagnetic material surrounding the maser cavity.  Both sphere and
    closed-end-cap cylinder geometries are supported.

    Args:
        inner_radius_mm:
            Inner radius of the innermost shell (mm).  Must exceed the maser
            Halbach bore radius (~7 mm) plus any diamond/coil clearance.
            Default 15 mm (fits the 14 mm-bore Halbach with 1 mm clearance).
        thickness_mm:
            Wall thickness of each mu-metal shell (mm).  Commercial mu-metal
            sheet is available from 0.1 mm to 3 mm.  Default 1.0 mm.
        mu_r:
            Relative permeability of the mu-metal at DC / near-DC.  Typical
            range for Ni-Fe 80/20 Permalloy: 20 000–100 000.  Default 50 000
            (conservative; matches architecture doc assumption).
        shell_shape:
            "sphere" — spherical enclosure, uses Jackson §5.12 formula.
            "cylinder" — cylindrical enclosure with closed end-caps, uses
            the infinite-cylinder transverse-field formula for the side wall.
            Default "sphere".
        height_mm:
            Cylinder height in mm.  Ignored for sphere.  None (default) →
            auto-set to 2 × inner_radius_mm (square cross-section aspect ratio).
        n_layers:
            Number of concentric shells.  The inner radius of the i-th layer
            (i = 0 … n-1) is:
                r_i = inner_radius_mm + i × (thickness_mm + interlayer_gap_mm)
            Product attenuation requires interlayer gap ≫ thickness.
            Default 1.
        interlayer_gap_mm:
            Radial air gap between consecutive shell surfaces (mm).  Larger
            gaps reduce magnetic coupling and make the product approximation
            more accurate.  Default 5 mm (large relative to each 1 mm shell).
        density_kg_m3:
            Material bulk density (kg/m³).  Default 8 700 (Permalloy 80).
    """

    inner_radius_mm: float = 15.0
    thickness_mm: float = 1.0
    mu_r: float = _DEFAULT_MU_R
    shell_shape: str = "sphere"       # "sphere" | "cylinder"
    height_mm: float | None = None    # cylinder only; None → 2 × inner_radius_mm
    n_layers: int = 1
    interlayer_gap_mm: float = 5.0
    density_kg_m3: float = _DEFAULT_DENSITY_KG_M3


@dataclass(frozen=True)
class ShieldingResult:
    """Result of a mu-metal shielding calculation.

    Attributes:
        attenuation_linear:
            Shielding factor S = B_external / B_internal (dimensionless, ≥ 1).
            ``S = 1`` means no shielding; ``S = 316`` ≈ 50 dB.
        attenuation_db:
            20 × log₁₀(S) in dB.  Architecture target: ≥ 50 dB.
        residual_field_tesla:
            Magnetic field magnitude inside the enclosure: B_external / S (T).
        incident_field_tesla:
            Externally applied stray-field magnitude before shielding (T).
        shell_mass_kg:
            Total combined mass of all shell layers (kg).
        config:
            The ``MuMetalShellConfig`` that produced this result.
    """

    attenuation_linear: float
    attenuation_db: float
    residual_field_tesla: float
    incident_field_tesla: float
    shell_mass_kg: float
    config: MuMetalShellConfig


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Private single-layer helpers                                    ║
# ╚══════════════════════════════════════════════════════════════════╝


def _spherical_single_layer_attenuation(
    inner_radius_mm: float,
    thickness_mm: float,
    mu_r: float,
) -> float:
    """Exact shielding factor for a uniform field through a spherical shell.

    Formula (Jackson, Classical Electrodynamics, §5.12):

        S = [(2µ_r+1)(µ_r+2) − 2(µ_r−1)²(r₁/r₂)³] / (9µ_r)

    Returns 1.0 for zero-thickness or µ_r = 1 (no shielding).
    """
    if thickness_mm <= 0.0:
        return 1.0
    r1 = inner_radius_mm
    r2 = inner_radius_mm + thickness_mm
    ratio3 = (r1 / r2) ** 3
    s = (
        (2.0 * mu_r + 1.0) * (mu_r + 2.0)
        - 2.0 * (mu_r - 1.0) ** 2 * ratio3
    ) / (9.0 * mu_r)
    return max(s, 1.0)


def _cylindrical_single_layer_attenuation(
    inner_radius_mm: float,
    thickness_mm: float,
    mu_r: float,
) -> float:
    """Exact shielding factor for a transverse uniform field through a cylindrical shell.

    Formula (Rikitake 1966; standard magnetostatics result):

        S = [(µ_r+1)² − (µ_r−1)²(r₁/r₂)²] / (4µ_r)

    Returns 1.0 for zero-thickness or µ_r = 1 (no shielding).
    """
    if thickness_mm <= 0.0:
        return 1.0
    r1 = inner_radius_mm
    r2 = inner_radius_mm + thickness_mm
    ratio2 = (r1 / r2) ** 2
    s = ((mu_r + 1.0) ** 2 - (mu_r - 1.0) ** 2 * ratio2) / (4.0 * mu_r)
    return max(s, 1.0)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Public attenuation functions                                    ║
# ╚══════════════════════════════════════════════════════════════════╝


def compute_single_layer_attenuation(config: MuMetalShellConfig) -> float:
    """Return the shielding factor S for a single mu-metal shell.

    Dispatches to the exact Jackson (sphere) or Rikitake (cylinder) formula.
    Multi-layer stacks should use :func:`compute_multilayer_attenuation`.

    Args:
        config: Shell geometry and material parameters.  Only the first shell
            is used; ``n_layers`` and ``interlayer_gap_mm`` are ignored.

    Returns:
        Dimensionless shielding factor S ≥ 1.0.
        S = B_external / B_internal for a uniform applied field.
    """
    if config.shell_shape == "cylinder":
        return _cylindrical_single_layer_attenuation(
            config.inner_radius_mm, config.thickness_mm, config.mu_r
        )
    return _spherical_single_layer_attenuation(
        config.inner_radius_mm, config.thickness_mm, config.mu_r
    )


def compute_multilayer_attenuation(config: MuMetalShellConfig) -> float:
    """Return the combined shielding factor for n_layers concentric shells.

    The inner radius of the i-th shell (0-indexed) is:

        r_inner_i = inner_radius_mm + i × (thickness_mm + interlayer_gap_mm)

    Total attenuation is approximated as the product of individual factors:

        S_total ≈ ∏ᵢ Sᵢ

    This is exact in the limit of large interlayer gaps (decoupled shells) and
    conservative when the gap equals the interlayer_gap_mm default of 5 mm,
    which is large compared to the 1 mm shell thickness.

    Args:
        config: Shell configuration, including ``n_layers`` and
            ``interlayer_gap_mm``.

    Returns:
        Combined shielding factor S_total ≥ 1.0 (dimensionless).
    """
    if config.n_layers < 1:
        return 1.0

    total_s = 1.0
    r_inner = config.inner_radius_mm
    for _ in range(config.n_layers):
        if config.shell_shape == "cylinder":
            s_i = _cylindrical_single_layer_attenuation(
                r_inner, config.thickness_mm, config.mu_r
            )
        else:
            s_i = _spherical_single_layer_attenuation(
                r_inner, config.thickness_mm, config.mu_r
            )
        total_s *= s_i
        r_inner += config.thickness_mm + config.interlayer_gap_mm
    return total_s


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Mass estimation                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝


def compute_shell_mass_kg(config: MuMetalShellConfig) -> float:
    """Estimate the total mass of all mu-metal shell layers (kg).

    Volume formulas:

    * **Sphere**: V = (4π/3)(r₂³ − r₁³)
    * **Cylinder**: V = π(r₂²-r₁²)·h  +  2π·r₂²·t   (side wall + 2 end-caps)
      where h = ``height_mm`` (None → 2 × ``inner_radius_mm``), t = ``thickness_mm``,
      and end-cap outer radius is r₂ (conservative upper bound on cap area).

    Args:
        config: Shell configuration.

    Returns:
        Total shell mass in kilograms.
    """
    total_mass = 0.0
    r_inner = config.inner_radius_mm  # mm

    for _ in range(max(config.n_layers, 0)):
        r1_m = r_inner * 1e-3
        r2_m = (r_inner + config.thickness_mm) * 1e-3

        if config.shell_shape == "sphere":
            vol = (4.0 / 3.0) * math.pi * (r2_m**3 - r1_m**3)
        else:
            h_m = (
                config.height_mm * 1e-3
                if config.height_mm is not None
                else 2.0 * r_inner * 1e-3
            )
            vol_side = math.pi * (r2_m**2 - r1_m**2) * h_m
            t_m = config.thickness_mm * 1e-3
            vol_caps = 2.0 * math.pi * r2_m**2 * t_m
            vol = vol_side + vol_caps

        total_mass += vol * config.density_kg_m3
        r_inner += config.thickness_mm + config.interlayer_gap_mm

    return total_mass


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Primary entry point                                             ║
# ╚══════════════════════════════════════════════════════════════════╝


def compute_shielding(
    incident_field_tesla: float,
    config: MuMetalShellConfig,
) -> ShieldingResult:
    """Compute the effect of a mu-metal shield on an external stray field.

    This is the main entry point for using the shielding model.  Given the
    stray-field magnitude at the unshielded maser module location and the
    shell geometry, it returns the residual internal field and all engineering
    metrics needed to verify R3 mitigation.

    Typical call chain to model the full disturbance path::

        from nv_maser.physics import (
            compute_imaging_magnet_stray_field,
            compute_shielding,
            MuMetalShellConfig,
            SpatialGrid,
            ImagingMagnetDisturbanceConfig,
        )

        grid = SpatialGrid(size=1, extent_mm=1.0)          # single-point grid
        disturbance_cfg = ImagingMagnetDisturbanceConfig(offset_x_mm=40.0)
        b_stray = float(compute_imaging_magnet_stray_field(grid, disturbance_cfg))

        result = compute_shielding(abs(b_stray), MuMetalShellConfig())
        assert result.attenuation_db > 50.0  # architecture R3 target

    Args:
        incident_field_tesla:
            External stray-field magnitude (T) at the maser module centre,
            before any shielding.  Typically obtained from
            :func:`~nv_maser.physics.disturbance.compute_imaging_magnet_stray_field`
            evaluated at a single point (the maser centre).
        config:
            Shell geometry and material parameters.

    Returns:
        :class:`ShieldingResult` with attenuation factor, residual field, and
        shell mass.
    """
    s = compute_multilayer_attenuation(config)
    attenuation_db = 20.0 * math.log10(max(s, 1e-30))
    residual = incident_field_tesla / s if s > 0.0 else float("inf")
    mass = compute_shell_mass_kg(config)
    return ShieldingResult(
        attenuation_linear=s,
        attenuation_db=attenuation_db,
        residual_field_tesla=residual,
        incident_field_tesla=incident_field_tesla,
        shell_mass_kg=mass,
        config=config,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Shell sizing utility                                            ║
# ╚══════════════════════════════════════════════════════════════════╝


def find_thickness_for_target_db(
    target_db: float,
    config_template: MuMetalShellConfig,
    *,
    t_lo_mm: float = 0.001,
    t_hi_mm: float = 100.0,
    tol_db: float = 0.01,
    max_iter: int = 80,
) -> MuMetalShellConfig:
    """Find the minimum shell thickness that achieves a target shielding level.

    Performs bisection on ``thickness_mm`` while holding all other parameters
    fixed (inner radius, mu_r, shape, n_layers, interlayer gap).

    When ``n_layers > 1``, each shell's inner radius shifts outward by
    ``(thickness_mm + interlayer_gap_mm)`` per layer, so the total attenuation
    varies monotonically with thickness — bisection converges reliably.

    Args:
        target_db:
            Required attenuation in decibels (e.g. 50.0 for the R3 architecture
            target).
        config_template:
            ``MuMetalShellConfig`` providing all parameters except
            ``thickness_mm``, which is solved for.
        t_lo_mm:
            Lower bound for the thickness search (mm).  Must give attenuation
            *below* ``target_db``; default 0.001 mm.
        t_hi_mm:
            Upper bound for the thickness search (mm).  Must give attenuation
            *above* ``target_db``; default 100 mm.
        tol_db:
            Convergence tolerance in dB (default 0.01 dB).
        max_iter:
            Maximum bisection iterations (default 80).

    Returns:
        A new ``MuMetalShellConfig`` identical to ``config_template`` except
        with ``thickness_mm`` set to achieve (within ``tol_db``) the target.

    Raises:
        ValueError: If ``t_lo_mm`` already gives attenuation ≥ ``target_db``
            (lower bound is too high).
        ValueError: If ``t_hi_mm`` is insufficient to reach ``target_db``
            (increase ``t_hi_mm``, ``mu_r``, or ``n_layers``).
    """

    def _db_at(t: float) -> float:
        cfg = MuMetalShellConfig(
            inner_radius_mm=config_template.inner_radius_mm,
            thickness_mm=t,
            mu_r=config_template.mu_r,
            shell_shape=config_template.shell_shape,
            height_mm=config_template.height_mm,
            n_layers=config_template.n_layers,
            interlayer_gap_mm=config_template.interlayer_gap_mm,
            density_kg_m3=config_template.density_kg_m3,
        )
        return 20.0 * math.log10(max(compute_multilayer_attenuation(cfg), 1.0))

    db_lo = _db_at(t_lo_mm)
    db_hi = _db_at(t_hi_mm)

    if db_lo > target_db:
        raise ValueError(
            f"t_lo_mm={t_lo_mm:.4f} already gives {db_lo:.1f} dB > target "
            f"{target_db:.1f} dB; lower t_lo_mm to find minimum thickness."
        )
    if db_hi < target_db:
        raise ValueError(
            f"t_hi_mm={t_hi_mm:.1f} only gives {db_hi:.1f} dB < target "
            f"{target_db:.1f} dB; increase t_hi_mm, mu_r, or n_layers."
        )

    lo, hi = t_lo_mm, t_hi_mm
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        db_mid = _db_at(mid)
        if abs(db_mid - target_db) <= tol_db:
            lo = hi = mid
            break
        if db_mid < target_db:
            lo = mid
        else:
            hi = mid

    final_t = (lo + hi) / 2.0
    return MuMetalShellConfig(
        inner_radius_mm=config_template.inner_radius_mm,
        thickness_mm=final_t,
        mu_r=config_template.mu_r,
        shell_shape=config_template.shell_shape,
        height_mm=config_template.height_mm,
        n_layers=config_template.n_layers,
        interlayer_gap_mm=config_template.interlayer_gap_mm,
        density_kg_m3=config_template.density_kg_m3,
    )
