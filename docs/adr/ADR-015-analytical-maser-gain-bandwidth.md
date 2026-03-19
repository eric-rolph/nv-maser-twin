# ADR-015: Analytical Maser Gain and Bandwidth Model

**Status:** Accepted  
**Date:** 2026-03-18  
**Module:** `src/nv_maser/physics/amplifier.py`  
**Tests:** `tests/test_amplifier.py` — `TestGainVoltage`, `TestComputeMaserGain`, `TestMaserGainBandwidthInvariant`, `TestMaserGainResult`

---

## Context

The NV-diamond maser operated below oscillation threshold functions as a
low-noise microwave amplifier.  ADR-014 added the magnetic quality factor
Q_m, noise temperature T_a, and CW output power from Wang et al. (2024).
Those are static "read-point" metrics; what was missing was the dynamic
amplifier characterisation:

1. **Signal power gain G** — how much the device amplifies a weak probe tone.
2. **Effective bandwidth B** — the parametric-narrowed 3 dB linewidth of the
   gain curve.
3. **Gain–bandwidth product** — the invariant that bounds the gain × bandwidth
   trade-off.

These three quantities are needed to position the maser in the context of
broader amplifier technology (comparison with HEMTs, SQUIDs, etc.) and to
reproduce the headline experimental result of Wang et al.: "14.5 dB gain,
340 kHz bandwidth" at room temperature.

---

## Decision

Add three new public interfaces to `amplifier.py`:

### `_gain_voltage(q_m, q_l, coupling_beta) -> float`

Internal helper computing the reflection amplitude |S₁₁(ω₀)| at resonance
from cavity input-output theory:

```
         (β − 1)/[Q_L (1+β)] + 1/Q_m
S₁₁ = ─────────────────────────────────
              1/Q_L − 1/Q_m
```

At critical coupling (β = 1) this reduces to:

```
S₁₁ = Q_L / (Q_m − Q_L)
```

Returns `nan` when Q_m ≤ Q_L (at or above oscillation threshold, where the
denominator ≤ 0 and the device enters the oscillator regime).

### `MaserGainResult` (frozen dataclass)

Fields:

| Field | Type | Description |
|-------|------|-------------|
| `gain_db` | `float` | Signal power gain in dB; `nan` at/above threshold |
| `bandwidth_hz` | `float` | 3 dB bandwidth after parametric narrowing (Hz) |
| `gain_bw_product_hz` | `float` | \|S₁₁\| × B = f_c/Q_m for β=1 (Hz) |
| `below_threshold` | `bool` | True when Q_m > Q_L |
| `margin_to_threshold` | `float` | Q_m/Q_L − 1 |

### `compute_maser_gain(cavity_frequency_hz, q_m, q_l, coupling_beta=1.0) -> MaserGainResult`

Main entry point.  Combines gain, bandwidth, and margin into a single
result object.

**Gain formula (power):**

```
G = S₁₁²  (power ratio)
G_dB = 10 log₁₀(G)
```

**Bandwidth (parametric narrowing):**

```
B = f_c · (Q_m − Q_L) / (Q_L · Q_m)
```

As Q_m → Q_L the bandwidth narrows to zero while gain diverges — the
parametric-narrowing effect of Wang et al. (2024) Fig. 3.

**Gain–bandwidth invariant (β = 1):**

```
√G · B = f_c / Q_m
```

The sqrt-gain × bandwidth equals the spin-gain linewidth f_c/Q_m,
which depends only on the NV material (not on how close Q_m is to Q_L).

---

## Physics Derivation

### Cavity Input-Output Theory

For a single-port microwave cavity with:

- κ_i = ω_c/Q₀  — internal loss rate
- κ_e = ω_c/Q_e — external coupling rate (to transmission line)
- κ_m = ω_c/Q_m — stimulated emission rate (spin gain, replaces loss)

the reflection coefficient at resonance (ω = ω_c) is:

```
S₁₁(ω_c) = (κ_e − κ_i + κ_m) / (κ_e + κ_i − κ_m)
```

The coupling coefficient β = Q₀/Q_e, so:

```
κ_i = ω_c/Q₀ = ω_c / [Q_L(1+β)]
κ_e = ω_c/Q_e = ω_c β / [Q_L(1+β)]
```

Substituting and cancelling ω_c gives the formula above.

### Q-factor relationships

```
1/Q_L = 1/Q₀ + 1/Q_e  (loaded = internal + coupling)
Q₀ = Q_L (1+β)
Q_e = Q₀ / β = Q_L (1+β) / β
```

For β = 1 (critical coupling): Q₀ = 2Q_L, Q_e = Q₀.

### Bandwidth

The effective decay rate of the cavity-with-gain is:

```
κ_eff = κ_e + κ_i − κ_m = ω_c/Q_L − ω_c/Q_m
```

The 3 dB bandwidth of the Lorentzian gain spectrum (valid for high G):

```
B = κ_eff / (2π) = f_c · (Q_m − Q_L) / (Q_L · Q_m)
```

This is the "parametric narrowing" of Wang et al. (2024).

---

## Validation Against Wang et al. (2024)

The following Q values are *derived* from the paper's headline results
(G = 14.5 dB, B ≈ 340 kHz, f_c = 2.87 GHz, β = 1):

From the gain formula (β = 1):  `S₁₁ = Q_L/(Q_m−Q_L) = √G = 5.31`
→ `Q_m − Q_L = Q_L / 5.31`

From the bandwidth formula: `B = f_c · (Q_m−Q_L) / (Q_L · Q_m)`
→ Solving yields **Q_L ≈ 1337, Q_m ≈ 1589**.

Cross-check:
```
G = (1337/(1589−1337))² = (1337/252)² = 5.306² = 28.15 → 14.5 dB  ✓
B = 2.87e9 × 252 / (1337 × 1589) = 341 kHz ≈ 340 kHz             ✓
GBP = f_c/Q_m = 2.87e9/1589 = 1.807 MHz                           ✓
```

These values are used in the quantitative tests (`test_wang_2024_*`).

---

## Alternatives Considered

### Maxwell-Bloch time-domain solver (already implemented)

`maxwell_bloch.py` computes gain numerically via the driven-mode ODE
(`gain_db` in `MaxwellBlochResult`).  This is accurate but requires
full ODE integration, making it unsuitable for rapid parameter sweeps or
design-space exploration.  The analytical formula is the complement.

### Coupling-independent gain formula

An approximation valid only at critical coupling (β = 1) would omit the
β-dependent numerator.  Rejected in favour of the general formula, which
degrades gracefully for β ≠ 1 and recovers the critical-coupling case
exactly.

---

## Consequences

**Positive:**
- Enables rapid (microsecond) computation of gain vs Q_m operating curves.
- Reproduces the Wang 2024 headline numbers from first principles.
- Provides the gain–bandwidth invariant f_c/Q_m, useful for material comparison.
- `compute_spectral_overlap` (already in `cavity.py`) is now also exported via
  `__init__.py`, completing the ADR-006 spectral-overlap model.

**Negative / Limitations:**
- Formula applies only in the *linear* amplifier regime (small-signal, below threshold).
- Does not account for: noise squeezing, frequency pulling, mode competition,
  or spatial mode effects.
- Above threshold (`gain_db = nan`) the device enters the CW oscillator regime;
  use `compute_output_power` for that case.

---

## References

- Wang, J. et al. *Room-Temperature Solid-State Maser as Microwave Amplifier.*
  Advanced Science 11 (2024).
- Caves, C. M. *Quantum limits on noise in linear amplifiers.*
  Phys. Rev. D 26, 1817 (1982).
- Collett, M. J. & Gardiner, C. W. *Squeezing of intracavity and travelling-wave
  light fields produced in parametric amplification.* Phys. Rev. A 30, 1386 (1984).
