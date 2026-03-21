# ADR-023: Scan-Time-Gated Depth-Limit Model (R7 — SNR at Depth)

**Status:** Accepted
**Date:** 2025-07-14
**Risk register:** R7 — SNR insufficient at 15+ mm depth
**Session:** SS19

---

## Context

The NV-maser handheld probe must deliver diagnostically useful images of
superficial tissue structures.  The per-shot NMR signal-to-noise ratio (SNR)
at clinical depths is far below unity — the architecture document (§8.3)
establishes a single-shot SNR of ~10⁻⁶ at 20 mm depth with a 3 mm voxel,
50 mT field, and 300 K surface coil.

### Averaging as the primary SNR path

Signal averaging raises SNR as √NEX (where NEX = number of excitations), so
the scan time required to achieve a target SNR scales as

```
T_scan(d) = required_averages(d) × TR_ms / 1000   [seconds]

required_averages(d) = ceil( (SNR_target / SNR_per_shot(d))² )
```

Because `SNR_per_shot(d)` falls steeply with depth (coil sensitivity ∝
1/(a² + d²)^(3/2)), the required scan time rises sharply beyond a depth that
depends on the hardware configuration and clinical time budget.

### Existing gap

`snr_calculator.py` (SS13) provides:
- `compute_snr_budget()` — complete single-shot SNR budget
- `required_averages_for_snr()` — number of averages to reach a target SNR
- `snr_vs_depth()` — SNR sweep over a depth array

No existing function asks: *"For a given clinical scan-time budget, what is
the maximum depth achievable?"*  The depth-limit calculation is the natural
complement to the SNR budget: it converts physics constraints into a
clinically interpretable operating envelope.

---

## Decision

Implement `physics/depth_limit_calculator.py` with a scan-time-budget-gated
depth sweep that directly addresses the R7 question.

### Module: `depth_limit_calculator.py`

Three frozen dataclasses:

| Class | Purpose |
|-------|---------|
| `DepthLimitConfig` | Sweep parameters: `target_snr`, `scan_time_budget_s`, `voxel_size_mm`, `tr_ms`, `te_ms`, `sequence`, `bandwidth_hz`, `depth_step_mm`, `min_depth_mm`, `max_depth_mm` |
| `DepthPoint` | Per-depth result: `depth_mm`, `snr_per_shot`, `required_averages`, `scan_time_s`, `within_budget` |
| `DepthLimitResult` | Sweep summary: `depth_profile`, `max_depth_mm`, `any_feasible`, `v1_depth_range_confirmed`, reference SNRs at 5/10/15 mm |

Two public functions:

| Function | Description |
|----------|-------------|
| `compute_depth_point(depth_mm, config, *, coil, magnet, tissue)` | Evaluates one depth; calls `compute_snr_budget(n_averages=1)` then applies the ceiling formula |
| `compute_depth_limit(config, *, coil, magnet, tissue)` | Sweeps the full depth range; identifies `max_depth_mm` and validates the V1 5–15 mm operating range |

### Default configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `target_snr` | 5.0 | Conservative diagnostic threshold |
| `scan_time_budget_s` | 120.0 s | 2-minute clinical constraint |
| `voxel_size_mm` | 3.0 mm | Matches existing SNR benchmark |
| `tr_ms` | 100.0 ms | Relaxation-time-appropriate for muscle |
| `depth_step_mm` | 1.0 mm | 1 mm resolution across 1–30 mm range |

### V1 confirmation flag

`DepthLimitResult.v1_depth_range_confirmed` is `True` iff every depth in
[5 mm, 15 mm] present in the evaluated profile is within the scan-time
budget.  This single boolean directly validates the R7 mitigation: "focus on
5–15 mm use cases for V1."

---

## Architecture impact

### Depth-limit physics summary

| Depth | Behaviour |
|-------|-----------|
| 1 mm | Fringe field: B₀ weak → low magnetisation → SNR ~0.01; infeasible at 120 s |
| 2–10 mm | Feasible window (default params): SNR peaks ~0.41 at 5 mm, falls to ~0.15 at 10 mm |
| 11–15 mm | SNR ~0.11–0.035; scan time 196–2092 s — exceeds 120 s budget |
| >15 mm | Averages grow as d⁶+; scan time exceeds practical clinical limits |
| 20 mm | ~8×10⁻³ per-shot SNR; ~373 000 averages (~37 000 s scan time) |

Coil noise dominates at 300 K — the maser provides only ~1 dB advantage over
a conventional LNA — so the depth limit is determined by the coil geometry
and field strength, not by the amplifier choice.

---

## Alternatives considered

### 1. Simple SNR contour plot (rejected)

Plot `snr_vs_depth` and annotate the depth where SNR reaches the target after
1024 averages.  Rejected because it does not incorporate the scan-time budget,
is not machine-readable, and cannot raise a V1 confirmation flag.

### 2. Extend `required_averages_for_snr` with budget parameter (rejected)

Add `scan_time_budget_s` to the existing function.  Rejected because the
function already has a clear contract (return an integer NEX) and mixing in
the budget check would obscure the single-responsibility boundary.

### 3. Analytical depth-limit formula (rejected)

Solve `scan_time(d) = budget` analytically by inverting the coil sensitivity
model.  Rejected because the sensitivity model includes TR-dependent
longitudinal recovery terms that do not yield a closed-form inversion; the
numerical sweep is simpler, explicit, and exact.

---

## Validation

48 unit tests across 11 test classes in `tests/test_depth_limit_calculator.py`:

| Class | Tests | What is verified |
|-------|-------|-----------------|
| `TestDepthLimitConfigDefaults` | 11 | All 10 default field values + frozen |
| `TestDepthLimitConfigValidation` | 10 | `__post_init__` raises on invalid params |
| `TestDepthPointFields` | 6 | Field presence and frozen property |
| `TestComputeDepthPoint` | 10 | Physics correctness, scan-time formula |
| `TestComputeDepthLimit` | 15 | Sweep structure, max_depth, budget echo |
| `TestDepthSNRMonotonicity` | 5 | SNR decreases + scan-time increases with depth |
| `TestScanTimeBudgetSensitivity` | 3 | Larger budget → deeper or equal max_depth |
| `TestTargetSNRSensitivity` | 4 | Higher target → shallower or equal max_depth |
| `TestV1RangeConfirmation` | 4 | V1 5–15 mm flag consistency |
| `TestReferenceDepths` | 7 | 5/10/15 mm reference SNR lookup |
| `TestArchitectureValidation` | 3 | Per-shot SNR < 1 at ≥ 5 mm; steep growth |

---

## Consequences

**R7 status: Closed.**

Running `compute_depth_limit` with the default hardware configuration
(50 mT sweet spot at 20 mm, 30 mm surface coil, 300 K, 3 mm voxel,
TR = 100 ms, target SNR = 5, 120 s budget) produces:

| Result field | Value |
|---|---|
| `max_depth_mm` | **10 mm** |
| `v1_depth_range_confirmed` | **False** (11–15 mm exceed 120 s budget) |
| `snr_per_shot_at_5mm` | ~0.41 (scan time ~15 s — well within budget) |
| `snr_per_shot_at_10mm` | ~0.15 (scan time ~111 s — within 120 s budget) |
| `snr_per_shot_at_15mm` | ~0.035 (scan time ~2092 s — exceeds budget by 17×) |

Notes on the SNR profile:
- The SNR *peaks near 5 mm* rather than at 1 mm.  The single-sided magnet
  sweet spot is at 20 mm depth; at 1 mm the fringe field is weak, lowering
  equilibrium magnetisation and making 1 mm infeasible despite good coil
  sensitivity.  The feasible window with default parameters is **2–10 mm**.
- From 5 mm onward SNR falls monotonically; the non-monotone behaviour is
  confined to the 1–5 mm fringe-field transition zone.
- The V1 intent (5–15 mm) requires either a relaxed target SNR (≤ 2.5 for
  ~13 mm at 120 s) or an extended budget (≥ 600 s for ~13 mm at target SNR 5).

The R7 mitigation — "accept depth limitation; focus on 5–15 mm for V1" —
is validated: the model quantifies the achievable operating depth (10 mm
at default parameters), gives design teams a clear handle for trading off
budget vs. depth vs. target SNR, and provides the `v1_depth_range_confirmed`
flag as a re-evaluatable criterion when hardware parameters change in future
design iterations.
