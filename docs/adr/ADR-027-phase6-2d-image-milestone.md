# ADR-027: Phase-6 '2D Image' Milestone Validator

**Status:** Accepted  
**Date:** 2025-07-15  
**Milestone:** Phase 6 — "First 2D image — Grid phantom resolved at 3 mm"  
**Session:** SS23  

---

## Context

The architecture document (`docs/research/handheld-maser-probe-architecture.md`,
§12.2) defines a hardware validation roadmap.  Sessions SS14–SS21 closed all
ten technical risks (R1–R10).  Sessions SS22 delivered a formal milestone
validator for Phase-4.  This ADR documents the Phase-6 milestone validator
delivered in SS23.

### Phase-6 criterion (architecture §12.2)

| Milestone | Criterion | Phase |
|-----------|-----------|-------|
| **First 2D image** | Grid phantom resolved at 3 mm | 6 |

"Grid phantom resolved at 3 mm" means the imaging reconstruction pipeline must
produce a 2D image in which alternating bars and gaps of width 3 mm are
visibly distinguishable.  This requires:

1. **Spatial resolution** — the point-spread function (PSF) FWHM must be
   ≤ 3 mm, so 3 mm features cannot be washed out by reconstruction blur.
2. **Bar contrast** — the Michelson contrast between bright bars and dark
   gaps in the reconstructed image must be above a threshold, confirming
   the features are actually visible (not just theoretically resolvable).
3. **Image SNR** — the reconstructed image must have adequate signal-to-noise
   ratio to be a usable image.

### Design choices

#### Forward model: exact non-Cartesian DFT (same as R8 validator)

The Phase-6 validator reuses `_noncartesian_dft` from
`artifact_characterizer.py` — the same function that proved risk R8 closed
in SS21.  This ensures the forward model is consistent, validated, and
dependency-free (pure NumPy).

#### Trajectory: radial (same defaults as ArtifactConfig)

A 64-spoke radial trajectory with 64 readout points per spoke is used,
identical to the R8 risk-closure parameters.  This is the simplest
non-Cartesian trajectory and the most common for field-portable MRI.

#### FOV = 6.4 cm (not 8 cm)

`ArtifactConfig` defaults to FOV = 8 cm.  The Phase-6 validator uses
FOV = 6.4 cm (0.064 m) because:

- At 64 × 64, this gives a pixel size of exactly **1.0 mm/pixel**.
- The 3 mm bar width is exactly **3.0 pixels** — a clean, easy-to-verify ratio.
- 6.4 cm is within the V1 target lateral FOV range (3–8 cm) from §4.

At the ArtifactConfig default (8 cm), pixel size would be 1.25 mm and bars
would span 2.4 pixels — fractional bar boundaries create staircase artefacts
that complicate the contrast measurement without adding realism.

#### Three-criterion closure (PSF + contrast + SNR)

Using PSF FWHM alone is necessary but not sufficient — it is a theoretical
measure of the imaging system bandwidth.  The bar-contrast test is an
*empirical* confirmation that the actual gridding + FFT reconstruction chain
produces a recognisable image of the target structure.  The SNR check ensures
the image quality is practically usable and not just "technically resolved but
too noisy to see".

#### Bar contrast metric: Michelson contrast

Michelson contrast `C = (I_bar − I_gap) / (I_bar + I_gap)` is the natural
choice for periodic bar patterns (equivalent to the spatial modulation
transfer function at the bar spatial frequency).  The threshold of 0.2 is
conservative — human observers can typically detect periodic patterns at
contrast C ≥ 0.03–0.05.

---

## Decision

Implement `src/nv_maser/physics/phase6_validator.py` containing:

### Data types

```python
@dataclass(frozen=True)
class Phase6Config:
    bar_width_mm: float = 3.0          # architecture §12.2 target
    fov_m: float = 0.064               # 6.4 cm → 1.0 mm pixel at grid_size=64
    grid_size: int = 64
    n_spokes: int = 64
    n_readout: int = 64
    kernel_width: float = 3.0
    bar_contrast_threshold: float = 0.2
    image_snr_threshold_db: float = 5.0
    resolution_threshold_mm: float = 3.0

@dataclass(frozen=True)
class GridPhantomResult:
    phantom: NDArray          # (64, 64) binary bar image
    pixel_size_mm: float      # fov_m / grid_size * 1000
    n_bar_pairs: int          # number of bright-dark pairs
    bar_width_mm: float

@dataclass(frozen=True)
class BarContrastResult:
    bar_mean: float
    gap_mean: float
    michelson_contrast: float  # (bar_mean - gap_mean) / (bar_mean + gap_mean)
    passes: bool               # michelson_contrast ≥ bar_contrast_threshold

@dataclass(frozen=True)
class Phase6MilestoneResult:
    config: Phase6Config
    phantom_result: GridPhantomResult
    recon: ReconResult          # gridding + FFT reconstruction of bar phantom
    psf_fwhm_mm: float          # max(fwhm_x, fwhm_y) from point-source recon
    bar_contrast: BarContrastResult
    image_snr_db: float
    pixel_size_mm: float
    psf_pass: bool
    contrast_pass: bool
    snr_pass: bool
    phase6_milestone_closed: bool   # True iff all three pass
```

### Functions

- `_make_bar_phantom(grid_size, fov_m, bar_width_mm)` → NDArray  
  Generates the binary bar/grid phantom (alternating vertical bright/dark bars).

- `_reconstruct_from_phantom(phantom, config)` → ReconResult  
  Simulates radial k-space via `_noncartesian_dft` and reconstructs with
  `grid_kspace` + `reconstruct_fft` (Hamming-windowed).

- `_measure_psf_fwhm(config)` → float  
  Places a unit impulse at image centre, reconstructs it, and measures FWHM
  via `_measure_fwhm` along x and y centre profiles.  Returns
  `max(fwhm_x, fwhm_y)`.

- `_measure_bar_contrast(recon_mag, phantom, threshold)` → BarContrastResult  
  Computes Michelson contrast using the original phantom as the bar/gap mask.

- `validate_phase6_milestone(config=None)` → Phase6MilestoneResult  
  Top-level entrypoint.  Calls all helpers and assembles the three pass/fail
  flags into `phase6_milestone_closed`.

---

## Consequences

### Positive

- **Phase-6 formally closed** in the digital twin.  `phase6_milestone_closed = True`
  with default configuration and comfortable margins:
  - PSF FWHM ≈ 1.79 mm (threshold 3.0 mm, margin 1.21 mm)
  - Michelson contrast ≈ 0.32 (threshold 0.2, margin 0.12)
  - Image SNR ≈ 65 dB (threshold 5 dB; noise-free simulation)

- **Milestone validators now cover** Phase-2 (sweet-spot), Phase-4 (depth
  profile), and Phase-6 (2D image).  Only Phase-9 (tissue contrast) remains.

- **Reuses validated forward model** (`_noncartesian_dft`) — no new physics
  approximations introduced.

### Negative / tradeoffs

- **Noise-free simulation**: The SNR check is trivially satisfied because the
  simulation has no noise floor.  The SNR criterion is retained for structural
  completeness and as a guard against degenerate reconstruction failures.

- **Private imports**: `_noncartesian_dft` and `_measure_fwhm` are private
  functions in `artifact_characterizer.py` (underscore prefix).  This is
  acceptable because `phase6_validator.py` is an internal physics module and
  the functions are tested through their callers.

---

## References

- Architecture §12.2 milestone table
- ADR-025: Reconstruction artifact characterisation (R8 risk closure)
- ADR-026: Phase-4 depth-profile milestone validator (SS22)
- `src/nv_maser/physics/phase6_validator.py`
- `tests/test_phase6_validator.py` (81 tests)
