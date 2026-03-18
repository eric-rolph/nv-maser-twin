# ADR-007: T1/T2 Tissue Parameter Mapping Strategy

**Status:** Accepted  
**Date:** 2025-07-14  
**Authors:** NV Maser Digital Twin team  
**Tags:** Phase B, EPG, tissue characterisation, diagnostics

---

## Context

The NV-diamond handheld probe operates at 50 mT.  Previous Phase B work
(ADR-006, `epg_adapter.py`) established that Extended Phase Graphs (EPG)
accurately model steady-state spin-echo and CPMG sequences for the stacked
tissue geometry captured in `depth_profile.py`.  

To achieve the probe's clinical value proposition — detecting subacute
haemorrhage and peri-haemorrhagic oedema in superficial tissue — the
digital twin must be able to:

1. **Infer T2** per depth layer from a CPMG echo train.
2. **Infer T1** per depth layer from multiple-TR spin-echo acquisitions.
3. **Flag outliers** — depth positions whose T1/T2 differs significantly
   from the healthy forearm reference.

The two approaches considered were:

| Approach | Advantages | Disadvantages |
|---|---|---|
| **A — Monoexponential CPMG + saturation recovery** | Simple 2-parameter fits; scipy `curve_fit` with bounds; round-trip accuracy < 0.1% for ideal pulses | Requires TR >> T2 for T1 branch; single-component model; no multi-pool effects |
| **B — Full MRzero Bloch simulation fit** | Exact model, multi-pool, field inhomogeneity | Very slow (minutes per voxel); requires MRzero installation; OOT for probe latencies |

Approach A was chosen as **sufficient for Phase B diagnostics** because the
target tissue anomaly (haemorrhage T2 ≈ 150 ms vs. muscle T2 ≈ 35 ms) is a
large, easily resolved deviation that does not require multi-pool modelling.

---

## Decision

Implement `t1t2_estimator.py` with the following algorithm:

### T2 mapping — CPMG echo-train fit

For each depth slice with tissue properties (T1, T2):

1. Simulate `n_echoes` CPMG echoes via `epg_cpmg(t1_s, t2_s, esp_s, ...)`.
2. Fit `S(n⋅ESP) = S₀ × exp(−n⋅ESP / T₂)` using `scipy.optimize.curve_fit`.
3. Cache results by `(T1_key, T2_key)` to avoid re-running the same tissue.

**Validity condition**: For ideal 180° refocusing in spectroscopy mode (no
gradient operators), the EPG CPMG echo amplitudes follow monoexponential T2
decay exactly.  Round-trip recovery < 0.1% for noiseless data (verified by
`test_fit_t2_monoexponential_exact` and `test_muscle_t2_recovered`).

**Bone cortex edge case**: T2 = 0.5 ms with ESP = 10 ms gives
exp(−10/0.5) ≈ 2×10⁻⁹ — all echoes ≈ 0.  Guarded by the
`max(amp) < 1e-10` check that returns `converged=False` without crashing.

### T1 mapping — saturation-recovery fit

For each depth slice:

1. Compute `epg_signal(t1, t2, te, TR_i)` for each TR value in the TR grid.
2. Fit `S(TR) = A × (1 − exp(−TR / T₁))` where A = M₀ × exp(−TE/T₂)
   absorbs the TE-dependent factor.
3. Auto-generate TR grid spanning `[max(0.5×T1_min, TE+0.1ms), 5×T1_max]`
   with 10 log-spaced points when `tr_values_ms` is not provided.

**Validity condition**: `S(TR) = A(1−exp(−TR/T₁))` holds only when TR >> T2
so that T2 coherences from previous repetitions have fully decayed.  For fat
(T2 = 80 ms), TR_min must exceed ~400 ms (5×T2).  The auto-generated grid
satisfies this when T1_min is large enough; otherwise the caller must supply
explicit TR values.  Documented in test `test_fat_t1_recovered`.

### Abnormality detection

`detect_tissue_abnormalities(observed_map, reference_map, t2_threshold=0.25)`
compares the fitted T2 at each depth against the nearest depth in the
reference map.  Flags are raised when:

```
|T2_obs − T2_ref| / T2_ref > t2_threshold
```

For HEMORRHAGE_LAYERS versus FOREARM_LAYERS:
- Haemorrhage (T2 ≈ 150 ms) vs. muscle reference (T2 ≈ 35 ms):
  deviation = (150 − 35) / 35 ≈ 329% >> 25% → **prolonged** flag ✓
- Fat layers in haemorrhage scenario vs. fat reference: deviation ≈ 0 → no flag ✓

---

## Consequences

### Positive

- Extends the clinical utility of the digital twin from a "field uniformity
  metric" to a **practical tissue differential diagnosis tool**.
- The T1/T2 ratio distinguishes tissue types (fat ≈ 3.1, muscle ≈ 17.1)
  providing contrast independent of absolute field strength.
- Round-trip EPG→fit accuracy < 0.1% for T2, < 1% for T1 (validated by
  45 new tests in `tests/test_t1t2_estimator.py`).
- Caching by (T1, T2) key makes depth-resolved maps O(unique tissues), not
  O(depth samples), keeping latency acceptable.

### Negative / Trade-offs

- Monoexponential model is incorrect for multi-pool tissues (e.g., myelin
  water component in muscle at high field).  At 50 mT this is a second-order
  effect.
- Saturation-recovery T1 fit is invalid for TR < 5×T2 (fat, TR < 400 ms).
  Callers must ensure the TR grid satisfies this constraint.

### Future work

- Add `build_reference_t2_table(tissue_layers)` helper to convert a healthy
  baseline model into a per-label lookup dictionary.
- Consider biexponential T2 fit for tissues with multiple water pools if
  in-vivo measurements show systematic residuals from monoexponential model.
- The `cross_validate_t1t2` function currently uses Pearson correlation; a
  Bland-Altman analysis would be more appropriate for clinical-grade
  validation.

---

## Related

- ADR-006 (`ADR-006-sota-em-simulation-scope.md`) — EPG adapter and depth
  profile
- `src/nv_maser/physics/t1t2_estimator.py` — implementation
- `src/nv_maser/physics/epg_adapter.py` — `epg_cpmg`, `epg_signal`
- `src/nv_maser/physics/depth_profile.py` — `FOREARM_LAYERS`,
  `HEMORRHAGE_LAYERS`, `TissueLayer`
- `tests/test_t1t2_estimator.py` — 45 tests (45 pass, 0 fail)
