# ADR-006: SOTA EM/MRI Simulation Scope

**Status:** Accepted  
**Date:** 2026-07-09  
**Deciders:** AI Engineer, Physics Lead

---

## Context

A review of state-of-the-art electromagnetic and MRI physics simulation techniques was conducted to determine which are relevant to the NV-diamond maser digital twin (50 mT Halbach magnet, 20 mm sweet spot, 2.13 MHz proton Larmor frequency).

The techniques evaluated were:

| Technique | Tools / References |
|-----------|-------------------|
| FDTD full-wave EM | Sim4Life, XFdtd, CST Studio, HFSS |
| SAR/RF safety modelling | Sim4Life, ISO 10974 |
| Extended Phase Graph (EPG) | Weigel 2015, Hennig 1988 |
| Tissue magnetic susceptibility | Schenck 1996, De Graaf 2007 |
| Patient-specific head models | PHASE, Duke/Ella phantoms |
| Hybrid EM/Huygens' Box | Multi-scale coupling |
| B₀ inhomogeneity / shimming | FDTD + perturbation theory |

---

## Decisions

### In scope — implemented in this PR

#### 1. Tissue magnetic susceptibility (χ) modelling

**Rationale:**  
Biological tissue susceptibility differences (Δχ ≈ 1–2 ppm at interfaces) create ΔB₀ field perturbations and intravoxel dephasing signal loss. At 50 mT (vs 1.5–3 T in clinical MRI) the absolute ΔB₀ magnitudes are 40–60× smaller, but they are still measurable when the sweet-spot gradient is already minimised, and they directly affect T2* contrast and depth-profile SNR.

**Implementation:**
- `SusceptibilityProfile` / `SusceptibilityCorrectedProfile` dataclasses
- `compute_susceptibility_field_shift()` — 1D slab/sphere demagnetisation model  
  ΔB₀ = N × Δχ × B₀, where N = 1 (slab) or 1/3 (sphere)
- `compute_dephasing_signal_loss()` — sinc attenuation from intravoxel ∇B₀  
  loss(z) = |sinc(γ_p |∇B₀| × δz × TE / 2π)|
- `estimate_susceptibility_impact()` — quick summary for any tissue stack
- `cross_validate_susceptibility()` — compare profiles with/without correction
- χ values from De Graaf (2007) Table 2.1 and Schenck (1996)
- `include_susceptibility=True` flag on `simulate_depth_profile()` applies correction

#### 2. Extended Phase Graph (EPG) signal simulation

**Rationale:**  
The device uses CPMG-style spin-echo acquisitions. The standard Bloch equation per-spin integration accurately computes single-echo signal but is slow for multi-echo trains (O(N_echoes × N_spins)). EPG represents the spin system as a discrete set of configuration states (F+, F−, Z) and propagates the full echo train with O(N_states) matrix operations per echo, enabling fast, exact multi-echo contrast prediction including T1 saturation and stimulated echo contributions.

**Implementation:**
- Pure NumPy (no PyTorch dependency)
- State matrix shape: (3, N_states) — rows = F+, F−, Z
- RF rotation matrix: Weigel (2015) Eq. 21
- `epg_signal()` — steady-state single spin-echo
- `epg_cpmg()` — full echo-train (N_echoes)
- `epg_depth_profile()` — 1D depth profile with per-layer T1/T2/PD
- `cross_validate_epg_vs_analytical()` — correlation against analytical SE formula

---

### Out of scope

| Technique | Reason |
|-----------|--------|
| **FDTD full-wave EM** (Sim4Life, XFdtd, CST, HFSS) | Designed for RF safety and high-field (≥1.5 T) wavelength effects. At 2.13 MHz the electromagnetic wavelength in tissue is ~15 km; near-field quasistatic approximation is exact. No RF standing-wave or SAR problem exists. |
| **SAR / RF radiation safety** | SAR is defined by time-averaged RF power deposition. At 2.13 MHz and the milli-watt RF levels used for spin-echo excitation, SAR is immeasurably small. Regulatory concern arises only above ~1 MHz at Watt-level CW power — not applicable here. |
| **Patient-specific head/body models** (PHASE, Duke) | High-resolution 3D voxel models are built for whole-body MRI bore geometries. This device is a handheld surface probe; the 1D stacked-layer tissue model is the correct abstraction at 20 mm depth with mm-scale depth resolution. |
| **Hybrid EM / Huygens' Box coupling** | Multi-scale domain coupling is used to inject antenna near-fields into FDTD body models. Not applicable at quasi-static frequencies. |
| **High-resolution 3D susceptibility phantoms** | Sufficient at 1D layer level for depth profiling. 3D phantom would add >10× complexity with negligible difference in predicted signal at the current hardware geometry. |

---

### Deferred to Phase C

| Item | Condition for revisiting |
|------|--------------------------|
| **Probe-body metal artifact modelling** (diamond + copper χ vs tissue) | Requires final hardware geometry from `nv-maser-hardware`. Diamond χ = −21.5 ppm and copper χ = −9.6 ppm differ significantly from tissue; near-probe susceptibility effects depend on exact coil/crystal geometry. |
| **B₀ shimming via susceptibility inserts** | Only relevant after experimental B₀ maps are acquired and compared to the model. |

---

## Consequences

### Positive
- Both adapters are pure Python / NumPy — no commercial tool license required.
- Susceptibility correction can be toggled on/off; existing depth-profile tests are unaffected when `include_susceptibility=False` (default).
- EPG gives sub-millisecond CPMG train simulation, enabling future parameter mapping (T1/T2 maps from depth profiles).
- The scope boundary is explicit, preventing premature investment in FDTD toolchains.

### Negative / Risks
- The 1D slab/sphere susceptibility model neglects 3D geometry effects at tissue-probe interfaces; error is bounded by the demagnetisation factor approximation (≤3× for realistic geometries).
- EPG steady-state assumption (20 TR iterations) may not converge for very long T1 (>5 s) at short TR; `epg_signal()` will silently underestimate recovery in that regime.

---

## References

- Schenck, J.F. (1996). "The role of magnetic susceptibility in magnetic resonance imaging." *Magn. Reson. Med.* **36**(2):199–210.
- De Graaf, R.A. (2007). *In Vivo NMR Spectroscopy*, 2nd ed. Table 2.1 (volume susceptibility values).
- Weigel, M. (2015). "Extended phase graphs: Dephasing, RF pulses, and echoes." *J. Magn. Reson. Imaging* **41**(2):266–295.
- Hennig, J. (1988). "Multiecho imaging sequences with low refocusing flip angles." *J. Magn. Reson.* **78**(3):397–407.
- Blümich, B. et al. (2008). "Mobile single-sided NMR." *Prog. NMR Spectroscopy* **52**:197–269.
