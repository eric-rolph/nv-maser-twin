# ADR-026: Phase-4 Depth-Profile Milestone Validator

**Status:** Accepted  
**Date:** 2025-07-15  
**Milestone:** Phase 4 — "First NMR signal" and "Resolvable layers in layered phantom"  
**Session:** SS22  

---

## Context

The architecture document (`docs/research/handheld-maser-probe-architecture.md`,
§12.2) defines a hardware validation roadmap with six milestones.  Sessions
SS14 to SS21 closed all ten technical risks (R1–R10) in the risk register.  The
natural next deliverable after risk closure is to add *formal milestone
validators* to the digital twin — functions that run the relevant physics
simulation and return a structured pass/fail report against the architecture
criteria.

Phase-2 already has `validate_sweet_spot_milestone` in `single_sided_magnet.py`
(ADR prior).  Phase-4 was the next unvalidated milestone.

### Phase-4 criteria (architecture §12.2)

| Criterion | Definition | V1 Threshold |
|-----------|------------|--------------|
| **First NMR signal** | Peak SNR across the depth profile within the V1 operating range | ≥ 3 (phantom demonstration) |
| **Resolvable layers** | All tissue-layer boundaries that fall inside the operating range are detectable | T2 contrast ratio ≥ 1.5 AND SNR at boundary ≥ 1.0 |
| **Scan time** | Total scan time for the depth profile (n_averages × TR) | ≤ 120 s |

The **V1 operating range** is defined as 3–15 mm depth, capturing the
subcutaneous fat and muscle layers of the forearm phantom (skin → fat at 2 mm,
fat → muscle boundary at 7 mm, muscle → bone at 27 mm).

### Why 3 mm lower bound (not 1 mm)?

A single-sided barrel magnet produces its sweet-spot field at a distance from
the magnet surface.  At depths < 3 mm, the field is still in the rising slope
below the sweet spot and the SNR is near zero.  Using 1 mm as the lower bound
would count depths with physically meaningless near-zero SNR and cause
spurious failures for the minimum-SNR check.

### Why SNR threshold = 3 (not 5)?

`depth_limit_calculator.py` uses `target_snr = 5.0` for *clinical* depth
mapping.  Phase-4 is a *phantom demonstration* milestone, not a clinical
operating point.  SNR = 3 is the conventional "first signal" threshold.

---

## Decision

Implement `src/nv_maser/physics/phase4_validator.py` containing:

### Data types

```python
@dataclass(frozen=True)
class Phase4Config:
    signal_snr_threshold: float = 3.0
    contrast_ratio_threshold: float = 1.5
    snr_at_boundary_threshold: float = 1.0
    scan_time_limit_s: float = 120.0
    depth_range_mm: tuple[float, float] = (3.0, 15.0)

@dataclass(frozen=True)
class LayerContrastResult:
    layer_a_name: str
    layer_b_name: str
    t2_a_ms: float
    t2_b_ms: float
    t2_contrast_ratio: float       # max(T2_a, T2_b) / min(T2_a, T2_b); inf if either ≤ 0
    boundary_depth_mm: float
    snr_at_boundary: float
    in_depth_range: bool
    detectable: bool               # contrast_ratio ≥ threshold AND snr ≥ threshold AND in_range

@dataclass(frozen=True)
class Phase4MilestoneResult:
    depth_profile: DepthProfile
    layer_contrasts: tuple[LayerContrastResult, ...]
    n_depths_evaluated: int
    max_snr_in_range: float
    min_snr_in_range: float
    scan_time_s: float
    all_in_range_layers_detectable: bool   # vacuously True if no in-range boundaries
    snr_pass: bool                          # max_snr_in_range >= signal_snr_threshold
    scan_time_pass: bool                    # scan_time_s <= scan_time_limit_s
    phase4_milestone_closed: bool           # all three pass
```

### Functions

| Function | Purpose |
|----------|---------|
| `compute_layer_contrast(layer_a, layer_b, boundary_depth_mm, profile, config)` | Compute T2 contrast ratio and boundary SNR for one adjacent pair |
| `validate_phase4_milestone(magnet, coil, config, tissue_layers, te_ms)` | Top-level entry point; all arguments optional |

### Default simulation settings

The validator uses `DepthProfileConfig()` defaults: `n_averages=1024`,
`TR=100 ms` → `scan_time_s = 102.4 s` (within the 120 s budget).

---

## Verification (smoke test results)

```
phase4_milestone_closed : True
snr_pass                : True   max_snr_in_range = 20.75 dB  (>  3.0 ✓)
scan_time_pass          : True   scan_time_s      = 102.4 s   (< 120.0 ✓)
all_in_range_detectable : True

Layer boundaries on FOREARM_LAYERS phantom:
  skin → subcutaneous_fat  @ 2.0 mm  T2ratio = 2.67  SNR =  1.96  in_range=False  skipped
  subcutaneous_fat → muscle @ 7.0 mm  T2ratio = 2.29  SNR = 16.44  in_range=True   detectable=True ✓
  muscle → bone_cortex     @ 27.0 mm T2ratio = 70.0  SNR =  0.02  in_range=False  skipped
```

---

## Consequences

### Positive
- Phase-4 milestone is now formally validated in the digital twin with
  quantitative pass/fail criteria drawn directly from the architecture spec.
- `LayerContrastResult` is reusable for future multi-layer phantoms (e.g.,
  haemorrhage phantom for clinical studies).
- All three sub-criteria are independently testable, making regression clear.

### Negative / Trade-offs
- `validate_phase4_milestone()` calls `simulate_depth_profile()` internally,
  which takes ~0.1 s per call.  Tests that call it many times use a
  module-scope fixture to minimise overhead.

### Out of scope
- Phase-4 hardware milestone requires physical phantom construction and
  lab measurements; this ADR covers the digital twin component only.
- Phase-6 ("First 2D image") remains unvalidated in the digital twin;
  that will be the target of a future session (SS23+).
