# ADR-012: Oscillator Stability — Allan Deviation Model

**Status**: Accepted  
**Date**: 2025-07-18  
**Deciders**: AI pair-programming session  
**Supersedes**: —  
**Related**: ADR-008 (quantum noise), ADR-011 (field sensitivity)

---

## Context

ADR-011 introduced `SensitivityResult`, which stores three white-noise
sensitivity floors (η_ST, η_T, η_F) and a single scalar
`allan_deviation_1s_t = η_ST` — the field stability at exactly τ = 1 s.

That scalar is insufficient for experimental cross-validation against
Breeze et al. (2021), which reports **Allan deviation curves** σ_B(τ)
spanning integration times from milliseconds to thousands of seconds.
It is also insufficient for characterizing the real system's noise
floor, which may contain contributions beyond pure white frequency noise
(e.g., 1/f flicker at mid-τ, thermal random walk at long τ).

A full σ_B(τ) curve reveals:
- The dominant noise type at each integration time (via the log-log slope)
- The optimal averaging time if a noise floor minimum exists
- The break points where different noise processes dominate

Both an **analytical path** (exact for Schawlow-Townes white FM noise)
and a **numerical path** (PSD integral for arbitrary spectra) are needed.

---

## Decision

Implement `src/nv_maser/physics/stability.py` providing:

1. **`compute_white_fm_allan_deviation(tau_s, maser_noise_result, nv_config)`**  
   Analytical σ_B(τ) = η_ST / √τ from the Schawlow-Townes linewidth alone.

2. **`compute_allan_deviation_from_psd(tau_s, phase_noise_spectrum, nu0, gamma_e)`**  
   Numerical ADEV from the phase noise PSD via the standard two-sample
   Allan variance integral.

3. **`compute_oscillator_stability(tau_s, maser_noise_result, nv_config, phase_noise_spectrum=None)`**  
   High-level assembly function returning an `OscillatorStabilityResult`.

4. **`OscillatorStabilityResult`** (frozen dataclass)  
   Stores σ_B(τ) in T/nT/pT, σ_y(τ), the analytical ST floor, log-log
   slopes, the σ_B(τ=1s) scalar, and provenance fields.

---

## Physics

### Random-walk derivation

The maser phase φ(t) follows a Wiener process with diffusion coefficient
D_φ = 2π Δν_ST rad²/s.  For the two-sample ADEV over consecutive
intervals of duration τ, with independent increments Δφ_k:

    σ_y²(τ) = (1/2)⟨(ȳ₂ − ȳ₁)²⟩
             = D_φ / (4π² ν₀² τ)
             = Δν_ST / (2π ν₀² τ)

Converting to field units via σ_B = σ_y × ν₀ / γ_e:

    σ_B(τ) = √(Δν_ST / (2π)) / (γ_e × √τ) = η_ST / √τ   [T]

At τ = 1 s this equals η_ST — identical to `SensitivityResult.allan_deviation_1s_t`. ✓

### ADEV–PSD integral (numerical path)

The standard two-sample Allan variance in terms of the one-sided
fractional frequency PSD S_y(f) = (f/ν₀)² × S_φ(f):

    σ_y²(τ) = 2 ∫₀^∞  S_y(f) × sin⁴(πfτ) / (πfτ)²  df

For the Schawlow-Townes spectrum S_φ(f) = Δν_ST/(πf²), this reduces to
the analytical result above (using ∫₀^∞ sin⁴(u)/u² du = π/4).

The numerical path enables future extensions where S_φ(f) includes
1/f flicker, 1/f² random-walk, or technical noise contributions.

### Log-log slope interpretation

    d(log σ_B)/d(log τ) ≈ −0.5  → white frequency noise (ST quantum limit)
    d(log σ_B)/d(log τ) ≈  0.0  → flicker (1/f) frequency noise
    d(log σ_B)/d(log τ) ≈ +0.5  → random walk frequency noise (thermal drift)

Computed via `np.gradient` (central differences) on the log-log scale.

---

## Consequences

### Positive

- **Experimental cross-validation**: σ_B(τ) curves can be plotted directly
  against Breeze et al. (2021) Fig. 5 (frequency stability vs averaging time).
- **Design guidance**: the slope field identifies which noise mechanism
  dominates at each operating point.
- **Consistency**: `OscillatorStabilityResult.sigma_b_at_1s_t` is
  mathematically identical to `SensitivityResult.allan_deviation_1s_t`
  when the same `MaserNoiseResult` is used — no duplication of truth.
- **Extensibility**: `compute_oscillator_stability` accepts an optional
  `PhaseNoiseSpectrum` for the numerical path; passing `None` gives the
  exact analytical result with zero overhead.

### Negative / Trade-offs

- **Numerical path accuracy**: the PSD integral accuracy depends on the
  frequency grid coverage.  Well-resolved grids (logspace −3 to 7 Hz,
  ≥ 10 000 points) give < 1 % error for τ ∈ [0.1, 100] s.  Very short
  or very long τ require extending the grid.
- **Thermal/Friis ADEV not modelled**: η_T and η_F (from ADR-011) arise
  from receiver chain noise (additive white noise) rather than phase
  diffusion.  Their τ scaling is different and left to a future ADR.

---

## Alternatives Considered

### A: Extend `SensitivityResult` with a τ array

Rejected.  `SensitivityResult` is a scalar summary of the noise floor.
Mixing arrays into it would violate its single-responsibility design
(confirmed by the size of ADR-011's dataclass).

### B: Compute ADEV only numerically from PSD

Rejected.  The analytical white FM formula is exact and O(N) faster than
the numerical integral over a large frequency grid.  The analytical path
is the default; numerical is opt-in.

### C: Only provide σ_B(τ) without σ_y(τ) or slope

Rejected.  `sigma_y` is needed for direct comparison with frequency-
stability literature values (dimensionless, independent of magnetometer
calibration).  `allan_slope` is cheap to compute and immediately reveals
the noise regime without requiring the user to plot and examine the curve.

---

## Implementation

| File | Change |
|------|--------|
| `src/nv_maser/physics/stability.py` | New module (297 lines) |
| `src/nv_maser/physics/__init__.py` | Added 4 exports from `stability` |
| `tests/test_stability.py` | 35 new tests |

### Test coverage highlights

| Test | Physics asserted |
|------|-----------------|
| `test_exact_formula_single_tau` | σ_B(τ) = η_ST/√τ to numerical precision |
| `test_quadruple_tau_halves_sigma` | White FM averaging law: σ_B(4τ) = σ_B(τ)/2 |
| `test_sigma_b_at_1s_matches_sensitivity_result` | Links σ_B(1s) to ADR-011 scalar |
| `test_st_psd_matches_analytical_within_1pct` | Validates the numerical integral |
| `test_slope_minus_half_for_white_fm` | Allan slope ≈ −0.5 throughout τ array |
| `test_slope_zero_for_constant_sigma` | Flicker regime detection |
| `test_slope_plus_half_for_random_walk` | Random-walk regime detection |

---

## References

- Allan, D. W. (1966) "Statistics of Atomic Frequency Standards". *Proc. IEEE* 54, 221–230.
- Rutman, J. (1978) "Characterization of Phase and Frequency Instabilities". *Proc. IEEE* 66, 1048–1075.
- Schawlow, A. L., Townes, C. H. (1958) *Phys. Rev.* 112, 1940.
- Breeze, J. et al. (2021) "Room-Temperature Diamond Maser". *npj Quantum Inf.* 7, 45.
- IEEE Std 1139-2008 "Definitions of Physical Quantities for Fundamental Frequency and Time Metrology".
