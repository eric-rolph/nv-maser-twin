# ADR-005: Phase A — Literature-Driven Physics Upgrades

**Status:** Accepted  
**Date:** 2026-07-08  
**Deciders:** AI Engineer, Physics Lead

---

## Context

A literature review of four recent papers (documented in `docs/research/four_papers_analysis.md`) identified gaps in the maser threshold, noise, and signal models.

| Paper | Key insight for this codebase |
|-------|-------------------------------|
| Kollarics et al. 2024 | Only 1/4 of NV centres are resonant (⟨111⟩ orientation) |
| Wang et al. 2024 | Magnetic Q_m and amplifier noise temperature formulas |
| Long et al. 2025 | LED-pumped maser threshold validation |
| Kersten et al. 2026 | Superradiant masing dynamics (Phase B/C scope) |

Four analytical corrections were implemented in Phase A:

1. **A1 — Magnetic quality factor Q_m** (Wang 2024, Eq. 1)
2. **A2 — Maser amplifier noise temperature** (Wang 2024, Eq. 4)
3. **A3 — NV orientation fraction** (Kollarics 2024)
4. **A4 — Spectral overlap ratio R** (Wang 2024)

---

## Decisions

### A1: Magnetic Q_m

**Formula:** Q_m⁻¹ = μ₀ γₑ² Δn η T₂* / 2

Added `MagneticQResult` dataclass and `compute_magnetic_q()` to `cavity.py`.
Exposed as `q_magnetic` in `environment.py` metrics.

### A2: Noise Temperature

**Formula:** T_a = Q_m/(Q₀−Q_m) · T_bath + Q₀/(Q₀−Q_m) · T_s

Added `compute_maser_noise_temperature()` to `signal_chain.py`.
Exposed as `maser_noise_temperature_k` in `environment.py` metrics.

### A3: NV Orientation Correction

Added `NVConfig.orientation_fraction = 0.25` (1/4 for B₀ ∥ one ⟨111⟩ axis).
Wired into `compute_n_effective()` (cavity.py) and `compute_maser_emission_power()` (signal_chain.py).

**Impact:** N_eff reduced 4×. Default config cooperativity drops from ~20 to ~5 — still above threshold (C > 1). No default retuning needed.

### A4: Spectral Overlap Ratio R

**Formula:** R = κ / γ⊥ (cavity linewidth / spin linewidth)

Added `compute_spectral_overlap()` to `cavity.py`.
Exposed as `spectral_overlap_R` in `environment.py` metrics.

---

## Consequences

- All 396 tests pass (20 new tests added, 0 regressions)
- `environment.compute_uniformity_metric()` now returns 3 additional keys: `q_magnetic`, `spectral_overlap_R`, `maser_noise_temperature_k`
- Downstream RL environment and API automatically see new metrics
- Phase B (Maxwell-Bloch solver) and Phase C (spectral dynamics) can build on these foundations

## Files Changed

| File | Change |
|------|--------|
| `src/nv_maser/config.py` | Added `orientation_fraction` to `NVConfig` |
| `src/nv_maser/physics/cavity.py` | Added `MagneticQResult`, `compute_magnetic_q()`, `compute_spectral_overlap()`; wired orientation correction into `compute_n_effective()` |
| `src/nv_maser/physics/signal_chain.py` | Added `compute_maser_noise_temperature()`; wired orientation correction into `compute_maser_emission_power()` |
| `src/nv_maser/physics/environment.py` | Exposed new metrics in `compute_uniformity_metric()` |
| `tests/test_cavity.py` | 14 new tests (orientation, Q_m, spectral overlap) |
| `tests/test_signal_chain.py` | 8 new tests (noise temperature, new metrics) |
