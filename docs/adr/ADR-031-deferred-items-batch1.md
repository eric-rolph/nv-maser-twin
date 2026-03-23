# ADR-031 — Deferred Items Batch 1: OAT Decoherence, Bland-Altman, SRF Fix

**Status**: Accepted  
**Date**: SS27  
**Deciders**: Engineering team

---

## Context

The project audit (SS27 discovery) identified 10 deferred items across ADRs.
Three were selected as unblocked and implementable without hardware:

1. **OAT squeezing decoherence** (ADR-013 §Future work) — the existing
   `spin_squeezing.py` computes static SQL/HL/OAT parameters but does not
   model the time-domain squeezing trajectory or the effect of T₂*
   decoherence during the OAT evolution.

2. **Bland-Altman analysis** (ADR-007 §Future work) — `cross_validate_t1t2`
   uses Pearson correlation and relative error, but clinical validation
   requires Bland–Altman bias and limits of agreement (Bland & Altman 1986).

3. **Surface coil SRF placeholder** (`surface_coil.py:177`) — the self-
   resonant frequency used a hardcoded 1 pF parasitic capacitance instead of
   an interwinding capacitance model.

---

## Decision

### 1. New module: `squeezing_dynamics.py`

Created `src/nv_maser/physics/squeezing_dynamics.py` implementing OAT
squeezing with T₂* decoherence overlay:

- **`oat_xi2_ideal(t, N, χ)`** — Wineland squeezing parameter ξ²_R(t) using
  the large-N parabolic approximation.  The parabolic form gives
  ξ²_min ≈ 1/√(3N) ∝ N^{−1/2}.  Note: the *exact* Kitagawa–Ueda formula
  achieves the stronger N^{-2/3} scaling but requires numerically challenging
  evaluation of cos^{N-2}(θ) terms for N ∼ 10¹².

- **`oat_optimal_time(N, χ)`** — time that minimises the parabolic ξ²_R:
  μt_opt = (12/(N−1))^{1/4}.

- **`apply_decoherence(ξ²_ideal, t, T₂*)`** — André & Lukin (2002) model:
  ξ²_R(t; T₂*) = ξ²_ideal · exp(2t/T₂*) + [1 − exp(−2t/T₂*)].

- **`compute_oat_ideal_trajectory()`** — full time-domain trajectory.

- **`compute_oat_with_decoherence()`** — decoherence-limited trajectory with
  penalty in dB and shifted optimal time.

- **`estimate_oat_chi(NVConfig)`** — rough dipolar coupling estimate from
  NV density.

- **`compute_squeezing_feasibility(NVConfig, ...)`** — end-to-end assessment
  including SQL vs squeezed field sensitivity.

### 2. Bland-Altman in `t1t2_estimator.py`

Added `BlandAltmanResult`, `BlandAltmanT1T2` dataclasses and
`bland_altman_t1t2()` function.  Computes bias, std of differences, and 95%
limits of agreement (bias ± 1.96σ) for both T1 and T2.  Returns per-point
means and diffs arrays for plotting.

### 3. Surface coil SRF model

Replaced the hardcoded `1e-12 F` parasitic capacitance with an interwinding
capacitance model:

    C_pair ≈ π ε₀ × 2πa / arccosh(pitch / d_wire)
    C_total = C_pair / (N − 1)        (series combination)
    f_SRF = 1 / (2π √(L × C_total))

For the default 15 mm, 5-turn coil: SRF ≈ 387 MHz (well above 2 MHz Larmor).

---

## Tests

- `tests/test_squeezing_dynamics.py` — 49 tests (new module).
- `tests/test_t1t2_estimator.py` — 10 new tests in `TestBlandAltmanT1T2` class.
- `tests/test_surface_coil.py` — 23 existing tests pass (no SRF assertions broken).

---

## Consequences

- Closes ADR-013 deferred item "decoherence during squeezing".
- Closes ADR-007 deferred item "Bland-Altman analysis".
- Closes surface_coil.py SRF placeholder.
- The parabolic OAT approximation is documented as conservative; upgrading
  to the exact KU formula is a future enhancement if N^{-2/3} precision is
  needed.

---

## References

- Kitagawa, M. & Ueda, M., PRA 47, 5138 (1993).
- André, A. & Lukin, M. D., PRA 65, 053819 (2002).
- Schleier-Smith, M. H. et al., PRL 104, 073604 (2010).
- Bland, J. M. & Altman, D. G., Lancet 327, 307 (1986).
