# ADR-022: Mixer IMD3 Nonlinearity Model (R9 — Mixer Distortion)

**Status:** Accepted  
**Date:** 2025-07-13  
**Risk register:** R9 — Up-conversion mixer noise (IMD3 component)  
**Session:** SS18  

---

## Context

The NV-maser signal chain up-converts the detected NMR signal (~2.13 MHz)
to the maser resonance frequency (~1.4699 GHz) using a passive RF mixer.
Any RF interference present at the probe input reaches the mixer's RF port.

### Existing coverage gap

`up_conversion.py` (SS13) implements a Friis thermal-noise cascade for the
mixer, correctly accounting for conversion loss and noise figure.  The
`MixerSpec` dataclass stores `ip3_dbm = 5.0 dBm` with the docstring note
"Used for linearity / dynamic-range checks" — documenting the intent but
leaving the calculation unimplemented.

`rf_rejection.py` (ADR-021, SS17) models the Lorentzian bandpass filter
provided by the maser cavity, which attenuates each *individual* OOB
interferer by 80–180 dB depending on its frequency offset.  However, this
model only applies to single-frequency signals processed *after* the mixer.

### The nonlinear distortion mechanism

A fundamental limitation of passive mixers is their weak (but non-zero) cubic
non-linearity characterised by the third-order intercept point IIP3.  When
*two* OOB interferers are present simultaneously at the mixer input with
frequencies f₁ and f₂, the non-linearity produces spurious output products at:

```
f_IM3⁺ = 2f₁ − f₂        (third-order intermodulation product)
f_IM3⁻ = 2f₂ − f₁
```

Their powers are (Pozar §10.3, two-tone unequal-power formula):

```
P_IM3(2f₁−f₂)  =  2·P₁ + P₂ − 2·IIP3     (dBm)   ... (1)
P_IM3(2f₂−f₁)  =  P₁ + 2·P₂ − 2·IIP3     (dBm)   ... (2)
```

where P₁ and P₂ are the *input* powers of tones at f₁ and f₂, and IIP3 is
the mixer's input-referred third-order intercept.  For equal tones (P₁ = P₂ = P)
both formulae reduce to the classical result P_IM3 = 3P − 2·IIP3.

**Key difference from ADR-021:**  The maser cavity rejects individual OOB
interferers *after* the mixer.  IMD3 products are generated *inside* the mixer
and emerge at the mixer *output* at the new frequencies f_IM3.  If f_IM3 falls
within the maser gain passband (±BW/2 of centre), the Lorentzian bandpass
provides no rejection — the product is amplified just like the wanted NMR
signal.

---

## Decision

Implement `physics/mixer_nonlinearity.py` as a standalone module that:

1. Accepts a list of `InterfererSpec` objects (reusing the dataclass from
   `rf_rejection.py`) and a `MixerNonlinearityConfig` (IIP3, maser centre,
   maser BW).
2. Iterates every unique (unordered) pair of interferers using
   `itertools.combinations`.
3. Computes both IMD3 products per pair via the exact two-tone formulas (1) and
   (2).
4. Marks each product `is_physical = product_freq > 0` and
   `in_maser_band = |f_IM3 − f_maser| < BW/2`.
5. Returns a `MixerNonlinearityResult` aggregate reporting whether any in-band
   product exists, its worst-case power, and the full product list.

The default configuration reuses the same 8-source hospital-environment
interferer set from `rf_rejection.py` with `IIP3 = 5.0 dBm` (matching
`DEFAULT_MIXER.ip3_dbm` in `up_conversion.py`).

---

## Alternatives Considered

### A — Hardware pre-filter before the mixer

A bandpass filter before the mixer would attenuate OOB interferers before they
enter the nonlinear device, suppressing IMD3 products quadratically with
attenuation.  **Deferred to hardware V2**: adds loss that degrades noise
figure, requires custom design at 1.4699 GHz, and is unnecessary given the
finding below.

### B — Replace mixer with higher-IIP3 device

A mixer with IIP3 > 20 dBm is commercially available (e.g. Mini-Circuits
ZX05-43MH, IIP3 ≈ +18 dBm).  **Available as a mitigation** if screening shows
in-band IMD3 from a non-default interferer set.  Modelling the current baseline
first is the right decision gate.

### C — Compute only worst-case OIP3 bound (no per-product model)

A single OIP3 bound was considered sufficient.  Rejected: it cannot identify
*which*, if any, IMD3 products fall in-band, and therefore cannot distinguish
between "high IMD3 power but all out-of-band" (safe) and "low IMD3 power but
in-band" (dangerous).

---

## Consequences

### Positive

* R9 is now fully quantified.  The Friis thermal contribution (existing
  `up_conversion.py`) and the nonlinear IMD3 contribution (this ADR) together
  characterise the mixer's signal-degradation budget.
* Reuses `InterfererSpec` from ADR-021 — consistent failure mode enumeration.
* `MixerNonlinearityConfig` exposes `iip3_dbm` as a first-class parameter for
  sensitivity analysis (raises IIP3 → reduces all IMD3 powers by 2 dB per dB).

### Validation (default 8-interferer hospital environment)

| Computation | Value |
|-------------|-------|
| Interferer pairs evaluated | C(8,2) = **28** |
| IMD3 products computed | 28 × 2 = **56** |
| Products within ±24.5 kHz of 1.4699 GHz | **0** |
| `any_in_band` | **False** |
| `worst_in_band_power_dbm` | **−∞** |
| Highest physical IMD3 power | ≈ −100 dBm (WiFi + WiFi, equal tones, IIP3=5) |

The default interferer set is safe: all OOB interferers are separated from the
maser centre by > 75 MHz, and no IMD3 mixing product of any pair lands within
the 49 kHz maser gain window.

### Risk trigger conditions

The IMD3 threat materialises only if an interferer operates very close to half
or double the maser centre frequency such that a two-tone product falls within
the maser band.  Likely realisation: spurious LO harmonics from nearby
electronics, or a co-channel interferer from medical telemetry equipment
operating near 1.47 GHz.  The model will flag this automatically when such an
interferer is added to the configuration.

### Negative

* C(N,2) complexity: 100 interferers → 4 950 pairs.  This is negligible for
  any realistic RF environment (<1 ms on modern hardware) but is documented
  for completeness.

---

## Implementation Notes

* `compute_imd3_power_dbm(p1_dbm, p2_dbm, iip3_dbm, product_type)` — pure
  formula, unit-testable in isolation.
* `compute_imd3_frequency_hz(f1_hz, f2_hz, product_type)` — pure formula.
* `compute_imd3_pair(spec1, spec2, config)` — evaluates one pair, returns
  `(IMD3Product, IMD3Product)`.
* `compute_mixer_nonlinearity(config)` — top-level aggregator.
* All dataclasses are `frozen=True` (immutable, hashable).
* No external dependencies beyond the standard library (`itertools`,
  `dataclasses`, `math`).

---

## References

* Pozar, D.M. — *Microwave Engineering*, 4th ed. (2012), §10.3 "Intermodulation
  Distortion".
* Razavi, B. — *Design of Analog CMOS Integrated Circuits*, 2nd ed. (2017),
  Appendix B §B.3.
* `ADR-021` — RF interference rejection / Lorentzian bandpass model (SS17).
* `ADR-013` — Up-conversion mixer thermal noise cascade (SS13).
* `up_conversion.py` — Friis noise model, `DEFAULT_MIXER.ip3_dbm = 5.0 dBm`.
* `rf_rejection.py` — `InterfererSpec`, `_DEFAULT_INTERFERERS` (8 hospital
  sources).
