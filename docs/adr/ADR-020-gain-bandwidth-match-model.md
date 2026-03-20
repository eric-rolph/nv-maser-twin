# ADR-020: Maser Gain-Bandwidth vs NMR Readout Matching Model (SS16)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-20 |
| **Module** | `src/nv_maser/physics/gain_bandwidth_match.py` |
| **Tests** | `tests/test_gain_bandwidth_match.py` (40 tests) |
| **Risk mitigated** | R2 — Maser gain bandwidth too narrow for readout |

---

## Context

The NV-maser cavity has a Lorentzian gain profile centred at the N-V electron
spin-resonance transition frequency (f₀ ≈ 1.4699 GHz at the sweet-spot field
of 50 mT).  The full-width half-maximum (FWHM) of this gain envelope is:

    BW_maser = f₀ / Q_loaded

The NMR readout bandwidth (gradient-encoded spatial information) spans a
window of width *BW_readout* around the proton Larmor frequency (~2.129 MHz at
50 mT).  After up-conversion to ~1.47 GHz, this band must sit entirely within
the maser gain envelope for distortion-free amplification.

Risk Register entry R2 states:
> "Maser gain bandwidth too narrow for readout → truncated signal → artefacts.
>  Mitigation: match maser Q to readout BW; use lower Q cavity or
>  frequency-tracking."

Without a quantitative model the risk could not be evaluated against design
parameters (Q, readout BW, B₀ stability).

---

## Decision

We implement `physics/gain_bandwidth_match.py` with a Lorentzian gain-bandwidth
model and a B₀ drift tolerance analysis.

### Physical model

The key relationships:

| Formula | Description |
|---------|-------------|
| `BW_maser = f₀ / Q_loaded` | FWHM of the maser gain profile |
| `margin = (BW_maser − BW_readout) / 2` | One-sided frequency slack |
| `ΔB_max = margin / γ_p` | Max B₀ drift before Larmor exits margin [T] |
| `ΔB_ppm = ΔB_max / B₀ × 10⁶` | Same limit in ppm of B₀ |

### Nominal parameter values (architecture doc §12)

| Parameter | Value | Notes |
|-----------|-------|-------|
| f₀ | 1.4699 GHz | NV ESR at sweet-spot 50 mT |
| Q_loaded | 30 000 | Yields BW_maser ≈ 49 kHz ≈ stated "~50 kHz" |
| BW_readout | 20 kHz | Mid-range of ±10–25 kHz stated range |
| B₀ | 50 mT | Sweet-spot field |
| γ_p | 42.577 MHz/T | Proton gyromagnetic ratio |

> **Note on Q notation**: The architecture doc quotes "Q = 10 000" alongside
> "BW ≈ 50 kHz".  These two numbers are inconsistent via `BW = f₀/Q`
> (10 000 gives BW ≈ 147 kHz, not 50 kHz).  The loaded Q that reproduces
> the stated 50 kHz at 1.4699 GHz is ≈ 29 400, rounded to 30 000 here.
> The "10 000" figure may refer to the unloaded (cold-cavity) Q.

### Benchmark result (nominal configuration)

| Quantity | Value |
|---------|-------|
| BW_maser | 49.0 kHz |
| BW_readout | 20.0 kHz |
| Frequency margin (one-sided) | 14.5 kHz |
| Overlap fraction | 59 % |
| B₀ drift tolerance | 340 µT / **6 810 ppm** |

### Key finding — R2 is mitigated by design

The one-sided frequency margin of 14.5 kHz corresponds to a B₀ drift tolerance
of **~340 µT (≈ 6 810 ppm at 50 mT)**.  This is orders of magnitude larger
than the 10 ppm B₀ stability required for NMR image resolution (architecture
doc §12).  Therefore:

* The readout bandwidth **easily** fits within the maser gain envelope under
  nominal operating conditions.
* B₀ drift at the level achievable with the digital twin's shimming loop
  (< 10 ppm) moves the Larmor frequency by only ~0.5 kHz — negligible against
  the 14.5 kHz margin.
* R2 becomes a live risk only if Q_loaded is increased to ≈ 73 000 (BW drops
  to ≤ 20 kHz, eliminating all margin) or if BW_readout is widened to ≥ 49 kHz.

The `sweep_q_vs_gain_bandwidth` and `sweep_b0_drift_vs_overlap` utilities
allow continuous monitoring of these margins as design parameters evolve.

---

## Consequences

### Positive
* R2 is quantitatively closed for the nominal design point.
* `find_q_for_target_margin` can be derived if a tighter Q spec emerges.
* Sweep utilities provide visualisation-ready data for design trade-off plots.

### Negative / watch-points
* The model assumes a rigid Lorentzian gain profile.  In practice, gain
  saturation and power broadening at high pump rates may alter the effective
  BW; this is not modelled here.
* The analysis addresses *static* B₀ drift only.  Dynamic field fluctuations
  (vibration-induced) require a separate spectral analysis.
* If a higher-Q cavity is used for lower noise, the margin shrinks linearly.
  At Q = 70 000, BW ≈ 21 kHz, which barely accommodates 20 kHz readout with
  only ~500 Hz one-sided margin (~24 ppm at 50 mT).

---

## Alternatives considered

| Option | Why rejected |
|--------|-------------|
| **Embed in `disturbance.py`** | Bandwidth matching is a separate physical phenomenon; conflating it with stray-field disturbance would obscure both. |
| **Numerical gain-profile simulation** | The Lorentzian model is exact for a linear cavity.  Full-wave simulation adds complexity without additional insight at this design stage. |
| **Single scalar pass-through** (like the original `shield_attenuation_db`) | Provides no insight into how Q or readout BW choices affect the margin.  Sweep utilities are essential for design exploration. |

---

## References

* Architecture doc §7 — "The Critical Bandwidth Problem"
* Architecture doc §12 — Parameters table and "Critical point" note
* Architecture doc §13 — Risk register, R2
* Saleh & Teich, *Fundamentals of Photonics* §15.1 (resonator bandwidth)
