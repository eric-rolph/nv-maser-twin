# ADR-011: Field Sensitivity Model — Schawlow-Townes, Thermal-SNR, Friis-Enhanced

**Status**: Accepted  
**Date**: 2025-07-27  
**Builds on**: ADR-010 (quantum noise → signal chain Friis integration)

---

## Context

The NV diamond maser digital twin had, by ADR-010, a complete chain of physics models:

- Quantum noise (Schawlow-Townes linewidth Δν_ST, noise temperature T_noise, output power P_out)
- Signal chain SNR budget (T_sys ≈ 375 K classical, T_friis ≈ 0.11 K Friis-enhanced)

But nowhere in the codebase was this information combined into the magnetometer's primary figure
of merit: **minimum detectable magnetic field B_min in T/√Hz**. The application objective is
sensing — every other physics module ultimately serves this goal. ADR-011 closes this gap.

---

## Decision

Add `src/nv_maser/physics/sensitivity.py` with a `SensitivityResult` dataclass and four
public functions:

| Function | Purpose |
|---|---|
| `compute_schawlow_townes_sensitivity` | η_ST — quantum phase diffusion floor |
| `compute_thermal_sensitivity` | η_T — classical LNA chain floor |
| `compute_friis_sensitivity` | η_F — Friis-enhanced quantum pre-amp floor |
| `compute_sensitivity` | Assembles all three + unit conversions + ratios |

Three complementary floors give users a complete picture of where the current system
operates relative to all relevant limits.

---

## Physics Derivation

### 1. Schawlow-Townes quantum phase diffusion (η_ST)

A maser oscillator with Schawlow-Townes linewidth Δν_ST [Hz, FWHM] undergoes phase diffusion
driven by spontaneous emission. The single-sided frequency noise power spectral density is:

$$S_\nu(f) = \frac{\Delta\nu_{ST}}{2\pi} \quad \text{[Hz}^2\text{/Hz]}$$

This gives a minimum detectable field:

$$\eta_{ST} = \frac{\sqrt{S_\nu}}{\gamma_e} = \frac{\sqrt{\Delta\nu_{ST} / (2\pi)}}{\gamma_e} \quad \text{[T/\sqrt{Hz}]}$$

where γ_e = 28.025 GHz/T is the NV gyromagnetic ratio. This is the fundamental quantum limit
— no detection improvement can beat it.

**Convention clarification**: In Maxwell-Bloch/quantum noise modules, `cavity_linewidth_hz`
and `schawlow_townes_linewidth_hz` are stored as **κ/(2π)** in Hz (half-bandwidth in
cycles-per-second, i.e. FWHM). The 2π factor in the denominator of the frequency noise PSD
is required to convert from the Lorentzian FWHM to the white frequency noise floor.

### 2. Thermal-SNR limited (η_T)

A frequency discriminator based on a cavity resonator with half-bandwidth κ_c [Hz] detects
a field-induced frequency shift δν = γ_e B by monitoring the transmitted power. The
cavity-discriminator slope at resonance converts δν to a power change; with received signal
power P_rcv and system noise temperature T_sys, the SNR = 1 condition gives:

$$\eta_T = \frac{\kappa_c}{\gamma_e} \sqrt{\frac{k_B T_{sys}}{P_{rcv}}} \quad \text{[T/\sqrt{Hz}]}$$

where P_rcv = `budget.received_power_w` (power at the LNA input after coupling and insertion
loss, which is the reference plane for T_sys).

### 3. Friis-enhanced (η_F)

Identical formula with T_sys replaced by the Friis cascade temperature T_friis from ADR-010:

$$\eta_F = \frac{\kappa_c}{\gamma_e} \sqrt{\frac{k_B T_{friis}}{P_{rcv}}} \quad \text{[T/\sqrt{Hz}]}$$

Since T_friis ≈ T_maser ≈ 0.11 K ≪ T_sys ≈ 375 K, the Friis advantage in field sensitivity is:

$$\text{Advantage} = 20\log_{10}\!\left(\frac{\eta_T}{\eta_F}\right) = 10\log_{10}\!\left(\frac{T_{sys}}{T_{friis}}\right) \approx 35 \text{ dB}$$

Note this is exactly **half the SNR advantage** from ADR-010 (which was a power ratio, ×10 dB
= 35 dB here vs. 35 dB SNR advantage → consistent since B_min ∝ √P_noise).

---

## Numerical Values (Default Config)

| Quantity | Value | Notes |
|---|---|---|
| Δν_ST | ~551 Hz | n_sp=1.5, κ_c=147 kHz, N̄=200 |
| η_ST | ~0.33 nT/√Hz | Quantum limit |
| T_sys (classical) | ~375 K | 300K + 75K LNA (NF=1 dB) |
| T_friis (Friis) | ~0.12 K | T_maser + T_LNA/G_maser |
| Friis advantage | ~35 dB | η_T/η_F in field amplitude |

Comparison context:
- Liquid-He SQUID: ~1 fT/√Hz (cryogenic)
- Room-temperature OPM: ~10 fT/√Hz
- NV ensemble (DC): ~100 fT/√Hz – 1 pT/√Hz (best reported)
- This maser (ST limit): ~0.3–1 nT/√Hz (compact, room-temperature)

The maser is not competing with cryogenic SQUIDs on absolute sensitivity; it competes on
form factor, operating temperature, and radio-frequency operation.

---

## `SensitivityResult` Field Design

Fields are grouped into four logical sections:

1. **Three sensitivity floors** in T/√Hz, nT/√Hz, pT/√Hz (3 × 3 = 9 fields)
2. **Cross-floor ratios**: `thermal_vs_st_ratio`, `friis_vs_st_ratio`,
   `friis_advantage_over_thermal_db`  
   All are `nan` when Friis data is unavailable; `friis_vs_st_ratio` and
   `friis_advantage_over_thermal_db` are always `nan` without a Friis-enabled budget.
3. **Provenance**: stores the raw inputs (Δν_ST, κ_c, T_sys, T_friis, P_rcv, γ_e) so the
   result is self-documenting without needing to re-fetch source objects.
4. **Allan deviation**: `allan_deviation_1s_t` = η_ST [T] (numerically equal at τ=1s) for
   direct comparison with oscillator stability literature.

**`nan` propagation rule**: Any Friis-related field is `nan` when
`SignalChainBudget.friis_system_temperature_k` is `nan` (i.e. the budget was not built
with a `MaserNoiseResult`). The ST and thermal fields are never `nan` for a valid maser.

---

## Interface Contracts

### New (this ADR)

```python
# In src/nv_maser/physics/sensitivity.py

def compute_schawlow_townes_sensitivity(
    maser_noise_result: MaserNoiseResult,
    nv_config: NVConfig,
) -> float: ...
# Returns η_ST = √(Δν_ST / 2π) / γ_e  [T/√Hz]

def compute_thermal_sensitivity(
    budget: SignalChainBudget,
    maser_noise_result: MaserNoiseResult,
    nv_config: NVConfig,
) -> float: ...
# Returns η_T = (κ_c / γ_e) × √(k_B T_sys / P_rcv)  [T/√Hz]

def compute_friis_sensitivity(
    budget: SignalChainBudget,
    maser_noise_result: MaserNoiseResult,
    nv_config: NVConfig,
) -> float: ...
# Returns η_F = (κ_c / γ_e) × √(k_B T_friis / P_rcv)  [T/√Hz]; nan if no Friis

def compute_sensitivity(
    budget: SignalChainBudget,
    maser_noise_result: MaserNoiseResult,
    nv_config: NVConfig,
) -> SensitivityResult: ...
# Assembles all three + nT/pT conversions + ratios + Allan deviation
```

All four functions exported from `src/nv_maser/physics/__init__.py`.

### Unchanged (backward-compatible)

All existing APIs from ADR-010 and earlier are unchanged.

---

## Alternatives Considered

**A. Single `compute_field_sensitivity(budget, noise, nv_cfg)` returning a scalar**

Rejected: a scalar doesn't communicate which limit applies or the advantage from the Friis
cascade. The dataclass makes all three floors visible simultaneously.

**B. Embedding sensitivity directly in `SignalChainBudget`**

Rejected: the signal chain module doesn't have the NVConfig (γ_e). Adding it would increase
coupling. A separate `sensitivity.py` layer keeps concerns separated and imports clean.

**C. Using `maser_noise_result.output_power_w` for P in thermal formula**

Rejected in favour of `budget.received_power_w`. The system noise temperature T_sys is
defined at the LNA input, so the SNR reference plane must be the same — after coupling and
insertion loss. Using raw maser P_out would give an optimistic (incorrect) thermal floor.

---

## Consequences

**Positive**
- The primary magnetometer figure of merit is now computable from any simulation run.
- All three sensitivity limits (quantum, thermal, Friis) are available side by side.
- Unit conversions (T/nT/pT) are pre-computed, removing boilerplate in downstream analysis.
- The 35 dB Friis advantage from ADR-010 is now expressed in the physically meaningful units
  (T/√Hz, field sensitivity) rather than in noise power (dB SNR).

**Neutral / trade-offs**
- `compute_sensitivity` requires both a `SignalChainBudget` *and* a `MaserNoiseResult`,
  meaning the caller must have run both modules. This is by design: the three-floor answer
  requires all three physics models.
- η_T and η_F use `budget.received_power_w` (classical emission model), while `MaserNoiseResult.output_power_w` is from the quantum noise model. These differ slightly; the budget power is the correct reference for thermal-chain sensitivity.

**Future work**
- Sweep `compute_sensitivity` over field uniformity (gain_budget scan) to produce
  sensitivity-vs-field-uniformity curves.
- Add `integrate_sensitivity_over_volume()` for spatially averaged performance.
- Phase D: field maps → volumetric sensitivity budget.
