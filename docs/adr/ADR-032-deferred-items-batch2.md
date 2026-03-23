# ADR-032 — Deferred Items Batch 2: TAT Squeezing, Biexponential T2, Allan τ Scaling

**Status**: Accepted  
**Date**: SS28  
**Deciders**: Engineering team

---

## Context

Continuing from ADR-031 (batch 1), SS28 addresses three more deferred items
from the project audit.  These were selected as unblocked, moderate-scope,
and complementary to SS27's work:

1. **TAT squeezing** (ADR-013 §Future work) — the module docstring
   promised two-axis twisting (TAT) but only OAT was implemented.  TAT
   achieves Heisenberg-limited squeezing (ξ² ∝ 1/N) versus OAT's weaker
   N^{−1/2} scaling.

2. **Biexponential T2 fit** (ADR-007 §Future work) — only monoexponential
   T2 fitting existed.  Multi-pool tissues (bound/free water) require a
   two-component model with AIC-based model selection.

3. **Allan deviation τ scaling** (ADR-012 §Future work) — only white FM
   noise (slope −0.5) was modelled.  Real oscillators exhibit white PM
   (slope −1), flicker FM (flat floor), and random-walk FM (slope +0.5).

---

## Decision

### 1. TAT squeezing in `squeezing_dynamics.py`

Added TAT dynamics using the two-exponential large-N model from
Kitagawa & Ueda (1993) and Ma et al. (2011) §3.4:

- **`TATIdealTrajectory`** / **`TATDecoherenceTrajectory`** — frozen
  dataclasses mirroring the OAT trajectory types.

- **`tat_xi2_ideal(t, N, χ)`** — ξ²_R(t) = exp(−Γt) + exp(+Γt)/(4N²)
  where Γ = 2(N−1)×2πχ.  The minimum ξ²_min = 1/N (Heisenberg scaling)
  at t_opt = ln(2N)/Γ ∝ ln(N)/(Nχ).

- **`tat_optimal_time(N, χ)`** — closed-form optimal squeezing time.

- **`compute_tat_ideal_trajectory()`** / **`compute_tat_with_decoherence()`**
  — full trajectory computation, reusing `apply_decoherence()`.

**Key physics**: TAT squeezes exponentially faster than OAT (t_opt ∝ ln(N)
vs N^{1/3}) and reaches the Heisenberg limit (1/N vs 1/√N).

### 2. Biexponential T2 in `t1t2_estimator.py`

Added multi-pool T2 fitting with AIC model selection:

- **`BiexpT2FitResult`** — dataclass with T2_short, T2_long, fraction_short,
  s0, R², RMS residual, AIC, convergence flag.

- **`fit_t2_biexponential()`** — 4-parameter constrained curve_fit:
  S(t) = A_s·exp(−t/T₂_s) + A_l·exp(−t/T₂_l).  Seeded from monoexp fit
  for robustness. Bounds [0.5 ms, 100 s]. Auto-swaps to enforce
  T₂_short < T₂_long.

- **`_compute_aic(n, rss, k)`** — AIC = n·ln(RSS/n) + 2k.

- **`select_t2_model()`** — returns ("mono"|"biexp", result) based on
  AIC comparison.  Falls back to mono if n < 4.

### 3. Allan deviation τ scaling in `stability.py`

Added three noise process models and RSS combination:

- **`NoiseProcessADEV`** / **`CombinedADEVResult`** — dataclasses for
  individual and combined σ_B(τ) contributions.

- **`compute_white_pm_adev(τ, h₂, f_H, ν₀, γ_e)`** — white phase
  modulation: σ_y²(τ) = 3h₂f_H/(4π²τ²), slope −1.

- **`compute_flicker_fm_adev(τ, h₋₁, ν₀, γ_e)`** — flicker FM floor:
  σ_y²(τ) = 2ln(2)·h₋₁, slope 0.

- **`compute_random_walk_fm_adev(τ, h₋₂, ν₀, γ_e)`** — random-walk FM:
  σ_y²(τ) = (2π²/3)·h₋₂·τ, slope +0.5.

- **`compute_combined_allan_deviation(τ, components)`** — RSS combination
  producing the "ADEV bathtub curve" with optimal τ finder.

---

## Test Coverage

| Feature | New Tests | Key Assertions |
|---------|-----------|----------------|
| TAT squeezing | 17 tests | Heisenberg 1/N scaling, t_opt formula, TAT < OAT, decoherence penalty, CSS at t=0 |
| Biexponential T2 | 12 tests | Two-pool recovery, T2_s < T2_l, AIC selection, <4 echoes guard, monoexp vs biexp |
| Allan τ scaling | 15 tests | WPM slope −1, flicker flat, RW FM slope +0.5, bathtub minimum, RSS, unit conversions |

**Total**: 44 new tests.  Full suite: 2626 passed, 75 skipped.

---

## Remaining Deferred Items

After SS27 (batch 1, 3 items) and SS28 (batch 2, 3 items), 4 items remain:

| Item | ADR | Status |
|------|-----|--------|
| Non-barrel magnet geometries | ADR-011 | Unblocked, complex |
| Far-from-threshold stability | ADR-018 | Unblocked, moderate |
| Probe-body RF artifacts | ADR-010 | Blocked (needs FEM data) |
| B₀ shimming optimisation | ADR-004 | Blocked (needs field maps) |

---

## References

- Kitagawa, M. & Ueda, M. (1993). Squeezed spin states. Phys. Rev. A 47, 5138.
- Ma, J. et al. (2011). Quantum spin squeezing. Physics Reports 509, 89–165.
- Allan, D. W. (1966). Statistics of atomic frequency standards. Proc. IEEE 54, 221.
- Rutman, J. (1978). Characterization of phase and frequency instabilities. Proc. IEEE 66, 1048.
- Bland, J.M. & Altman, D.G. (1986). Statistical methods for assessing agreement. Lancet i, 307–310.
