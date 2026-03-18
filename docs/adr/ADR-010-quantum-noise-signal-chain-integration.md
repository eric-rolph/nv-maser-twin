# ADR-010: Quantum Noise → Signal Chain Friis Cascade Integration

**Status:** Accepted  
**Date:** 2025-07-17  
**Author:** Eric Rolph

---

## Context

`compute_signal_chain_budget()` (Phase A, ADR-001) computed the receiver SNR budget
using only classical noise sources — Johnson-Nyquist thermal noise and LNA added noise
characterised by the LNA noise figure in dB.  The system noise temperature returned was
typically 350–400 K for a room-temperature 300 K environment with a 1 dB LNA.

Phase C introduced `compute_maser_noise()` (ADR-009) which computes the quantum noise
temperature of the NV maser using the quantum Langevin theory result:

    T_noise = ℏω n_sp / k_B

For the NV maser at 1.47 GHz with n_sp ≈ 1.5 (η_pump = 0.5):

    T_noise ≈ 0.11 K

This is the fundamental lower bound on the added noise of the maser acting as a
microwave amplifier (Caves 1982).  Despite being the dominant first-stage amplification
element in the receive chain, this value was never plumbed into `SignalChainBudget`.
The maser's ~35 dB noise advantage over a conventional LNA chain was invisible to
system-level SNR analysis.

---

## Decision

Integrate `MaserNoiseResult.noise_temperature_k` into `compute_signal_chain_budget()`
via the Friis cascade formula.

### Friis Cascade

When a maser pre-amplifier precedes the LNA, the receiver system noise temperature is:

    T_cascade = T_maser + T_LNA / G_maser

where:
- `T_maser  = ℏω n_sp / k_B` — quantum noise temperature from `MaserNoiseResult`
- `T_LNA    = T₀ (F_LNA − 1)` — LNA noise temperature (T₀ = 290 K)
- `G_maser  = P_out / (k_B T_maser Δf)` — maser gain estimated from output power

As `G_maser → ∞`, `T_cascade → T_maser ≈ 0.11 K` — limited only by vacuum fluctuations.

### Design Choices

1. **Backward-compatible optional parameter** — `compute_signal_chain_budget()` gains a
   fifth keyword-only parameter `maser_noise_result: MaserNoiseResult | None = None`.
   All existing call sites continue to work unchanged.

2. **Sentinel value `math.nan`** — The three new fields
   (`maser_noise_temperature_k`, `friis_system_temperature_k`, `quantum_advantage_db`)
   are set to `math.nan` when `maser_noise_result` is not supplied.  This is an explicit
   signal rather than a silent default (e.g. 0 or −1) that could be misinterpreted.

3. **Classical budget unchanged** — `system_noise_temperature_k`, `snr_db`, and all
   other pre-existing fields are computed identically regardless of whether a
   `MaserNoiseResult` is supplied.  The Friis cascade lives in parallel new fields only.

4. **Gain estimated from output power** — `G_maser = P_out / (k_B T_maser Δf)`.
   This is self-consistent with the quantum noise model and avoids adding a new gain
   parameter to `SignalChainConfig` or `MaserConfig`.  The gain is clamped to 1 at minimum.

5. **Dedicated helper `compute_friis_system_temperature()`** — Exported publicly so that
   higher-level tools (e.g. sensitivity optimisers) can call it directly with arbitrary
   gain values without going through the full budget computation.

---

## New API Surface

### `SignalChainBudget` — three additional frozen fields

| Field | Type | Description |
|---|---|---|
| `maser_noise_temperature_k` | `float` | T_noise = ℏω n_sp / k_B.  `nan` if not supplied. |
| `friis_system_temperature_k` | `float` | T_maser + T_LNA / G_maser.  `nan` if not supplied. |
| `quantum_advantage_db` | `float` | 10 log₁₀(T_sys_classical / T_friis).  `nan` if not supplied. |

### New public function

```python
def compute_friis_system_temperature(
    maser_noise_result: MaserNoiseResult,
    lna_noise_figure_db: float,
    maser_gain_linear: float,
) -> float:
    """Friis cascade: T_maser + T_LNA / G_maser  [K]."""
```

### Updated signature

```python
def compute_signal_chain_budget(
    nv_config: NVConfig,
    maser_config: MaserConfig,
    signal_config: SignalChainConfig,
    gain_budget: float,
    maser_noise_result: MaserNoiseResult | None = None,
) -> SignalChainBudget:
```

---

## Consequences

### Positive
- The system-level SNR model now reflects the maser's principal advantage: sub-Kelvin
  quantum noise temperature reduces the effective noise floor by ~35 dB vs a room-temperature LNA.
- Closes the gap between Phase C quantum noise model and the Phase A signal chain — all
  major physics modules are now cross-wired.
- Zero regression risk: all existing callers and tests are unmodified.

### Neutral
- The gain estimate `G_maser = P_out / (k_B T_maser Δf)` is bandwidth-dependent.
  For a fixed output power, wider detection bandwidths give lower estimated gain, raising
  `T_friis` slightly.  This is physically correct — at wider bandwidths, the maser gain
  per unit bandwidth is lower.

### Negative / Open Questions
- `G_maser` derived from `P_out` and `T_maser` is an approximation.  A future ADR
  could add an explicit `maser_gain_db` field to `MaserConfig` for more precise control.
- The budget does not yet model insertion loss *between* the maser and LNA; this would
  degrade the Friis advantage by adding a thermal noise contribution before the LNA.
  Tracked as a future enhancement.

---

## Alternatives Considered

### A: Replace classical budget with quantum budget
Replace `system_noise_temperature_k` with the Friis result when `maser_noise_result` is
provided.  **Rejected** — would silently change the SNR value returned to existing
callers, violating backward compatibility and making it harder to compare classical vs
quantum cases.

### B: New separate function `compute_quantum_signal_chain_budget()`
A parallel function with quantum fields baked in.  **Rejected** — duplicates the entire
budget logic and splits maintenance.  The optional-parameter approach is cleaner.

### C: Add `maser_gain_db` to `SignalChainConfig`
Make gain explicit in the config.  **Deferred** — requires user to know the maser gain
independently of `MaserNoiseResult`; the self-consistent estimate from output power is
more convenient and physically consistent for the current use case.

---

## References

- Caves, C. M. (1982). *Quantum limits on noise in linear amplifiers.* Physical Review D, 26(8), 1817.
- Siegman, A. E. (1964). *Microwave Solid-State Masers.* McGraw-Hill.
- Friis, H. T. (1944). *Noise figures of radio receivers.* Proceedings of the IRE, 32(7), 419–422.
- ADR-001: Signal chain SNR budget (Phase A baseline)
- ADR-009: Quantum noise model — Langevin formalism (Phase C)
