# ADR-024: B₀ Field Strength and Homogeneity Tolerance Model (R1 Closure)

**Status:** Accepted
**Date:** 2025-07-15
**Risk register:** R1 — Sweet-spot magnet field too weak at target depth
**Session:** SS20

---

## Context

The handheld NV-maser probe uses a single-sided permanent magnet array to
produce a localised "sweet spot" — a quasi-homogeneous B₀ region at a
target depth of 20 mm from the probe coil face.  The Phase-2 milestone
(arch doc §12.2) specifies:

> B₀ = **50 ± 5 mT** at 20 mm depth, **< 500 ppm** over a 10 mm diameter sphere

Risk R1 asks: *if the magnet cannot achieve exactly 50 mT, how much SNR does
the system lose, and does that loss exceed the V1 operating envelope?*

### Existing coverage gap

`single_sided_magnet.py` (SS-prior) already models the sweet-spot field
location and volumetric uniformity via `validate_sweet_spot_milestone()`.
That module answers **"did we hit 50 mT at 20 mm?"** but does not answer:

1. **If B₀ = 47 mT (slightly short), how much SNR is lost?**
2. **At 500 ppm uniformity, how wide is the NMR signal spectrum, and does it
   fit inside the maser gain passband?**
3. **Is the full ±5 mT tolerance band safe for all SNR budgets?**

`gain_bandwidth_match.py` (ADR-020, R2) models the overlap between the NMR
readout bandwidth and the maser gain envelope, and computes the maximum B₀
drift tolerable before the signal centre frequency drifts outside the maser
passband.  That module addresses **dynamic drift** during operation; R1 is
about **static manufacturing tolerance**: is 45–55 mT an achievable and safe
range?

---

## Decision

Implement `physics/field_tolerance_calculator.py` with the following model:

### Part 1 — B₀ strength sensitivity (SNR ∝ B₀²)

The NMR signal voltage at the coil depends on two B₀-proportional quantities:

* Equilibrium magnetisation M₀ ∝ B₀ (Boltzmann polarisation)
* Detection EMF ∝ ω₀ = γ B₀ (Faraday induction, Hoult & Richards 1976)

With thermal-noise-dominated detection (body + coil noise independent of B₀):

```
SNR ∝ ω₀ × M₀ ∝ B₀²
SNR_factor = (B₀_actual / B₀_nominal)²
SNR_loss_dB = −20 log₁₀(SNR_factor) = 40 log₁₀(B₀_nom / B₀_actual)
```

The 3 dB SNR loss point (SNR halved in power, −29 % in amplitude) occurs at:

```
B₀_3dB = B₀_nom × 10^(−3/40) ≈ 0.050 × 0.8414 ≈ 42.07 mT
```

Since the V1 tolerance floor is 45 mT (≥ 42.07 mT), **the V1 range never
crosses the 3 dB boundary.**  Worst-case SNR loss at 45 mT is 1.84 dB —
equivalent to requiring ≤ 53 % more signal averages.

### Part 2 — Field inhomogeneity: FID T2* dephasing

Peak-to-peak field variation ΔB₀ across the sweet-spot volume spreads the
Larmor frequencies:

```
Δν = γ̄ × ΔB₀ = 42.577 MHz/T × ΔB₀
T2*_inhom = 1/(π × Δν)
1/T2*_eff = 1/T2_tissue + 1/T2*_inhom
FID_amplitude_ratio = exp(−TE / T2*_inhom)
```

At 500 ppm and 50 mT: ΔB₀ = 25 µT → Δν ≈ 1 064 Hz → T2*_inhom ≈ 299 µs.
For TE = 100 µs, FID amplitude loss ≈ 3 dB.

**However**, the probe's primary acquisition sequence is CPMG (multi-echo T2
measurement for haemorrhage detection).  CPMG 180° refocusing pulses
re-phase statically dephased spins between echoes; T2* does not govern CPMG
echo amplitudes.  The FID T2* calculation is provided for reference and
future single-pulse modes.

### Part 3 — Field inhomogeneity: spectral bandwidth vs maser passband

The NMR signal from the sweet-spot volume occupies a bandwidth Δν ≈ 1 064 Hz
at 500 ppm.  The maser gain half-bandwidth (from ADR-020) is ≈ 24 500 Hz.
The maser bandwidth limit in ppm:

```
limit_ppm = (BW_maser / 2) / (γ̄ × B₀_nom) × 10⁶
           = 24 500 / (42.577 MHz/T × 0.050 T) × 10⁶
           ≈ 11 508 ppm
```

The 500 ppm spec is **23× below** the maser BW limit.  Signal fits inside
the maser passband with > 46 kHz of margin.

---

## Consequences

### Quantitative summary

| Criterion | V1 spec | Worst-case value | Margin |
|-----------|---------|-----------------|--------|
| B₀ deviation (strength) | ≤ ±5 mT | 1.84 dB SNR loss at 45 mT | Below 3 dB threshold |
| Signal spectral BW at 500 ppm | < maser BW/2 = 24.5 kHz | ≈ 1 064 Hz | 23× below limit |
| FID T2* at 500 ppm (TE=100 µs) | — | ≈ 3 dB amplitude loss | CPMG refocuses this |

### R1 closure

R1 is **confirmed closed at V1 specification**.

* Within the ±5 mT tolerance band, SNR loss is bounded and manageable
  (worst case 1.84 dB at 45 mT → ≤ 53 % more averages).
* At 500 ppm field uniformity the NMR spectral bandwidth (≈ 1 064 Hz)
  is far below the maser gain bandwidth (49 kHz), providing > 46 kHz margin.
* The primary acquisition sequence (CPMG) is immune to static T2* dephasing.
* `r1_risk_closed = True` is produced by default `FieldToleranceConfig`.

### Module produced

| Item | Reference |
|------|-----------|
| Module | `src/nv_maser/physics/field_tolerance_calculator.py` |
| Tests  | `tests/test_field_tolerance_calculator.py` (52 tests) |
| Exported | `physics.__init__` — five new public symbols |

### Key classes and functions

| Symbol | Purpose |
|--------|---------|
| `FieldToleranceConfig` | Nominal B₀, sweep parameters, maser BW, V1 criteria |
| `B0SensitivityPoint` | SNR factor and loss at a given B₀ |
| `HomogeneityPoint` | T2*_eff, FID loss, spectral BW, maser-BW flag |
| `FieldToleranceResult` | Full sweeps + analytical thresholds + R1 closure |
| `compute_b0_sensitivity_point` | SNR penalty at one B₀ value |
| `compute_homogeneity_point` | FID and spectral penalties at one ppm value |
| `sweep_b0_sensitivity` | B₀ sensitivity profile sweep |
| `sweep_homogeneity` | Uniformity penalty profile sweep |
| `compute_field_tolerance` | Master function — sweeps + verdict |

### Residual scope

* Full 2D off-axis field mapping and magnet geometry optimisation remain
  in `single_sided_magnet.py`; this ADR adds the SNR-impact layer.
* Reconstruction artefacts from non-Cartesian k-space (R8) remain open
  for a future session.
