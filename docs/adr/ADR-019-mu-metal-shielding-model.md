# ADR-019: Mu-Metal Shielding Model for Stray-Field Isolation (SS15)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-20 |
| **Module** | `src/nv_maser/physics/shielding.py` |
| **Tests** | `tests/test_shielding.py` (39 tests) |
| **Risk mitigated** | R3 — Stray field from imaging magnet destabilises maser |

---

## Context

The handheld probe places a single-sided imaging magnet (≈ 0.5–1.0 kg NdFeB,
B₀ ≈ 50 mT at the sweet spot) alongside a small maser module (14 mm Halbach
bore).  At 40 mm lateral separation, the imaging magnet's dipole fringe field
is approximately:

```
B_stray ≈ (µ₀/4π) m / r³  with  m = Br·V/µ₀ ≈ 1.3 × 2×10⁻⁵ / (4π×10⁻⁷) ≈ 20.7 A·m²
B_stray ≈ (10⁻⁷ × 20.7) / (0.04)³ ≈ 32 mT
```

The NV maser's Halbach magnet must stay at 50 ± 5 mT (< 10% change).  A 32 mT
perturbation would shift the field by ≈ 64%, completely disrupting masing.
The architecture doc §6.5 specifies three layers of defence:

1. **Distance** — > 5 cm separation (fringe field ∝ 1/r³)
2. **Shielding** — mu-metal shell at ~50 dB attenuation (`this ADR`)
3. **Active compensation** — 8 shimming coils (`disturbance.py`, `closed_loop.py`)

The digital twin must model layer 2 quantitatively so designers can choose shell
geometry (sphere vs. cylinder, wall thickness, number of layers) against mass
and cost constraints.

---

## Decision

Implement `physics/shielding.py` with the following public API:

| Symbol | Type | Purpose |
|--------|------|---------|
| `MuMetalShellConfig` | frozen dataclass | Shell geometry + material |
| `ShieldingResult` | frozen dataclass | Attenuation, residual field, mass |
| `compute_single_layer_attenuation(config)` | `float` | Exact formula, single shell |
| `compute_multilayer_attenuation(config)` | `float` | Product law, n layers |
| `compute_shell_mass_kg(config)` | `float` | Engineering mass estimate |
| `compute_shielding(incident_T, config)` | `ShieldingResult` | Primary entry point |
| `find_thickness_for_target_db(target, template)` | `MuMetalShellConfig` | Sizing utility |

### Formulas used

**Spherical shell** (Jackson, *Classical Electrodynamics* §5.12, exact):

$$S_\text{sphere} = \frac{(2\mu_r+1)(\mu_r+2) - 2(\mu_r-1)^2 (r_1/r_2)^3}{9\mu_r}$$

**Thin-shell approximation** (t ≪ r₁, µ_r ≫ 1):

$$S_\text{sphere} \approx \frac{2\mu_r t}{3 r_1}$$

**Cylindrical shell** (Rikitake 1966, transverse uniform field, exact):

$$S_\text{cyl} = \frac{(\mu_r+1)^2 - (\mu_r-1)^2 (r_1/r_2)^2}{4\mu_r}$$

$$S_\text{cyl} \approx \frac{\mu_r t}{2 r_1} \quad (t \ll r_1,\; \mu_r \gg 1)$$

**Multi-layer product** (approximate, valid for gap ≫ thickness):

$$S_\text{total} \approx \prod_{i=0}^{n-1} S_i(r_1 + i(t+g),\; t,\; \mu_r)$$

where *g* = `interlayer_gap_mm`.

### Benchmark values

For the default parameters (`inner_radius_mm = 15`, `thickness_mm = 1.0`,
`mu_r = 50 000`, sphere, single layer):

| Quantity | Value |
|----------|-------|
| (r₁/r₂)³ | (15/16)³ ≈ 0.824 |
| S (exact) | ≈ 1 956 |
| Attenuation | **≈ 65.8 dB** ≫ 50 dB target |
| Residual at 30 mT incident | ≈ **15 nT** |
| Shell mass | ≈ 26 g |

Minimum thickness to achieve exactly 50 dB (single layer, sphere, default r and
µ) is ≈ **0.14 mm**, well within commercially available 0.1–3 mm mu-metal sheet.

### Why sphere > cylinder

The sphere formula contains a 1/3r₁ factor in the thin-shell limit; the cylinder
uses 1/2r₁.  Ratio of S_sphere/S_cyl ≈ (2/3)/(1/2) = 4/3.  A spherical
enclosure provides ~33% more attenuation per unit thickness.  The module
supports both to let the designer match probe mechanical constraints.

---

## Alternatives Considered

### Alternative 1 — Add a `shielding_db_formula()` scalar function

*Rejected.*  A single formula returning dB from (µ, t, r) is accurate only for
the thin-shell limit and does not support multi-layer stacking, cylindrical
geometry, or mass estimation.  The overhead of a proper module is justified by
the R3 criticality rating.

### Alternative 2 — Finite-element field simulation (FEA)

*Rejected for the digital twin.*  FEA (e.g. via FEMM or ngsolve) gives exact
results for arbitrary geometry but requires an external solver, discretisation
setup, and 10 000× more compute time per evaluation.  The analytical formulas
are exact for the idealised geometries (uniform external field, uniform shell),
and the deviations from the true handheld geometry are covered by using the
conservative single-point incident field from `compute_imaging_magnet_stray_field()`.
FEA remains appropriate for final hardware validation; the digital twin uses the
analytical formulas for design-space exploration.

### Alternative 3 — Encode shielding directly in `ImagingMagnetDisturbanceConfig.shield_attenuation_db`

`disturbance.py` already has `shield_attenuation_db` as a scalar pass-through
attenuation in `ImagingMagnetDisturbanceConfig`.  

*Rejected as the primary model.*  The existing scalar field does not compute
the attenuation from physical parameters; it only applies a user-supplied
number.  `physics/shielding.py` gives the twin the ability to *derive* that
number from geometry, then feed it back:

```python
result = compute_shielding(b_stray, shell_cfg)
disturbance_cfg = ImagingMagnetDisturbanceConfig(
    ...,
    shield_attenuation_db=result.attenuation_db,
)
```

This separation of concerns lets `disturbance.py` remain geometry-agnostic
while `shielding.py` owns the physics.

---

## Physical Model Details

### Validity of the product approximation (multi-layer)

For two shells separated by gap *g*, the magnetic field re-couples through the
gap.  The product approximation S_total ≈ S₁ × S₂ under-estimates coupling and
is *conservative* (predicts lower attenuation than reality for closely-spaced
shells).  For the default gap of 5 mm vs. shell thickness 1 mm, the coupling
correction factor is ≲ 10% (Gubser & Wolf 1965), so the product formula is
adequate for digital-twin design exploration.  Hardware qualification should use
FEA.

### End-cap treatment for cylindrical shells

The cylindrical formula uses the infinite-cylinder result for the side wall —
this matches the dominant path for a laterally incident field.  The two closed
end-caps are included in the mass calculation (treated as flat disks of outer
radius r₂ and thickness t) but not in the attenuation formula, which slightly
under-estimates the effective shielding for axially incident fields.  This is
again conservative.

### Frequency dependence

mu-metal's permeability is approximately constant from DC to ~1 kHz and then
rolls off due to eddy-current and domain-wall resonance effects.  The relevant
disturbance is the quasi-static fringe field of the NdFeB imaging magnet, which
is DC-to-slow-drift.  No frequency correction is needed for the current model.

---

## Consequences

**Positive:**
- R3 is now quantitatively modelled in the digital twin.  A designer can call
  `find_thickness_for_target_db(50.0, MuMetalShellConfig())` and immediately
  get the minimum wall thickness (≈ 0.14 mm for defaults).
- `compute_shielding()` chains cleanly with `compute_imaging_magnet_stray_field()`
  to model the full passive-isolation path before the active shimming loop.
- The sizing function `find_thickness_for_target_db()` enables parametric
  sweeps (e.g. trading mass vs. attenuation) without external tools.

**Negative / trade-offs:**
- The analytical formulas assume a uniform external field and perfect geometry.
  Real mu-metal enclosures have seams, feed-throughs, and variable permeability
  after fabrication stress.  A de-rating factor of 2–3 dB is prudent in hardware
  design.
- The module adds 7 new public symbols to `nv_maser.physics` — manageable given
  the justified scope.

---

## Usage Example

```python
from nv_maser.physics import (
    ImagingMagnetDisturbanceConfig,
    MuMetalShellConfig,
    SpatialGrid,
    compute_imaging_magnet_stray_field,
    compute_shielding,
    find_thickness_for_target_db,
)

# 1. Stray field at the maser centre (single-point grid)
grid = SpatialGrid(size=1, extent_mm=1.0)          # 1×1 grid at origin
disturbance_cfg = ImagingMagnetDisturbanceConfig(offset_x_mm=40.0)
b_stray = abs(float(compute_imaging_magnet_stray_field(grid, disturbance_cfg)))
print(f"Stray field at maser: {b_stray*1e3:.1f} mT")  # ~29–32 mT

# 2. Default single-layer sphere
result = compute_shielding(b_stray, MuMetalShellConfig())
print(f"Attenuation: {result.attenuation_db:.1f} dB")   # ~65 dB
print(f"Residual:    {result.residual_field_tesla*1e9:.0f} nT")  # ~15 nT
print(f"Shell mass:  {result.shell_mass_kg*1e3:.1f} g")  # ~26 g

# 3. Find minimum thickness for exactly 50 dB (R3 requirement)
min_shell = find_thickness_for_target_db(50.0, MuMetalShellConfig())
print(f"Min thickness for 50 dB: {min_shell.thickness_mm:.3f} mm")  # ~0.14 mm

# 4. Feed derived attenuation back into DisturbanceGenerator
shielded_disturbance_cfg = ImagingMagnetDisturbanceConfig(
    offset_x_mm=40.0,
    shield_attenuation_db=result.attenuation_db,
)
```

---

## References

- Jackson, J. D., *Classical Electrodynamics* (3rd ed.), §5.12, pp. 197–200
- Rikitake, T., "Magnetic Shielding", *Bulletin of the Earthquake Research
  Institute*, 44 (1966)
- Gubser, D. U. & Wolf, S. A., "Magnetic shielding with multiple shells",
  *Journal of Applied Physics*, 36 (1965)
- Handheld probe architecture document §6.5 (Magnetic Isolation) and §13 (Risk R3)
