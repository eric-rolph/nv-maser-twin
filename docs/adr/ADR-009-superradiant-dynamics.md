# ADR-009: Superradiant Dynamics Model

**Status**: Accepted  
**Date**: 2025-01  
**Author**: NV Maser Digital Twin Project  

---

## Context

The spectral Maxwell-Bloch solver (`spectral_maxwell_bloch.py`) already integrates
the Kersten et al. (2026) ODE system and counts superradiant burst events in the
time-domain simulation.  However, the project lacked an **analytic** regime
classifier that works without running the ODE — a computationally cheap function
that accepts cavity parameters and outputs the operating regime and burst estimates
directly.

### Why this matters

1. **Rapid parameter sweeps**: Regime maps over (N_eff, Q, T₂*) space require
   evaluating millions of parameter combinations.  Running a full ODE for each
   is prohibitive; an analytic estimate is ~10⁶× faster.

2. **Design guidance**: Hardware designers need to know *a priori* whether a given
   diamond / cavity combination will produce CW masing (stable output) or
   superradiant bursts (pulsed energy, stronger peak signal but intermittent).

3. **Cross-validation**: The analytic model produces `burst_detected = g_eff > κ/2`
   predictions that can be checked against `SpectralMBResult.n_bursts > 0` from the
   ODE simulation, providing a self-consistency test.

---

## Decision

Implement `src/nv_maser/physics/superradiance.py` — an analytic superradiant
dynamics module with:

1. **Regime classification** via three conditions (Dicke 1954 / Kersten 2026):
   - `C > 1` (oscillation threshold)
   - `g_eff > κ/2` (strong-coupling / good-cavity criterion)
   - `τ_SR < T₂*` (cooperative emission completes before dephasing)

2. **Analytic pulse observables** from the Tavis-Cummings semiclassical model:
   - `τ_SR = 1/(2π g_eff)` — Rabi half-period
   - `t_D ≈ τ_SR ln(N)/2` — statistical delay from quantum seed
   - `N̄_SR = N_eff/4` — peak intracavity photons
   - `P_SR = ℏω × κ_rad × N̄_SR` — peak output power
   - `E_SR = N_eff × ℏω` — total burst energy

3. **CW comparison**: If a `MaxwellBlochResult` or `SpectralMBResult` is provided,
   compute `power_enhancement = P_SR_peak / P_CW`.

---

## Physics Derivation

### Collective coupling and cooperativity

The ensemble of N_eff NV spins couples to the cavity mode with collective coupling:

```
g_eff = g₀ × √N_eff       [Hz, i.e. g_N / (2π)]
```

The cooperativity is:

```
C = 4 g_eff² / (κ · γ⊥)
```

Both quantities are already computed by `compute_full_threshold` in `cavity.py`.

### Regime boundary: bad-cavity vs good-cavity

The NV maser can operate in two distinct oscillating regimes above threshold (C > 1):

| Regime | Condition | Physics |
|--------|-----------|---------|
| CW masing (bad-cavity) | `g_eff ≤ κ/2` | Cavity decay dominates; spins emit incoherently into mode; steady state |
| Superradiant (good-cavity) | `g_eff > κ/2` | Spin-cavity coupling dominates; spins synchronize; cooperative burst |

The boundary `g_eff = κ/2` is the *strong-coupling threshold* in cavity QED, equivalent
to the single-pass gain equalling cavity loss for the collective emission channel.

An additional coherence requirement ensures the cooperative Rabi cycle completes before
inhomogeneous dephasing destroys the phase synchrony:

```
τ_SR < T₂*    where τ_SR = 1/(2π g_eff)
```

If this fails (e.g., poor-quality diamond with short T₂*), the ensemble dephases before
the burst peaks and the output reverts to incoherent CW masing despite `g_eff > κ/2`.

### Superradiant pulse quantities (Tavis-Cummings result)

From the semiclassical mean-field solution of n_eff two-level systems coupled to a
single cavity mode (Tavis & Cummings 1968), the pulse solution from a fully-inverted
initial state gives:

```
Peak photon number:  N̄_SR = N_eff / 4       (at t = τ_SR from maximum)
Peak output power:   P_SR = ℏω × κ_rad × N̄_SR = ℏω × (2π κ) × N_eff / 4
Total pulse energy:  E_SR = N_eff × ℏω       (full angular momentum conversion)
Statistical delay:   t_D  = τ_SR × ln(N_eff) / 2
```

The peak power scales linearly with N_eff (same as CW masing peak photon number).
The **burst** nature means energy is concentrated in a time τ_SR, making the *peak
intensity* scale as N_eff / τ_SR ∝ N_eff × g_eff ∝ N_eff^{3/2} — superlinear in N.

### Typical NV maser parameters

With default `CavityConfig` (V_mode = 0.5 cm³) and `NVConfig` (n_NV = 10¹⁷/cm³):

| Quantity | Value |
|----------|-------|
| g₀ | ≈ 100 Hz |
| N_eff | ≈ 6 × 10¹³ |
| g_eff | ≈ 800 MHz |
| κ = f/Q | ≈ 147 kHz (Q = 10⁴) |
| g_eff / (κ/2) | ≈ 10⁴ → deeply superradiant |
| τ_SR | ≈ 0.2 ns ≪ T₂* = 1 µs ✓ |
| P_SR peak | ≈ 10 µW |

This demonstrates the NV maser is *deeply* in the superradiant regime at typical
operating densities — the spectral MB simulation's burst-counting (`n_bursts > 0`)
should agree.

---

## Alternatives Considered

### A. Integrate superradiance analytics into `spectral_maxwell_bloch.py`
**Rejected**: That module already handles time-domain simulation.  Mixing analytic
estimates with ODE output would blur the separation of concerns.  The analytic model
is most useful *before* running the ODE for parameter-space exploration.

### B. Extend `ThresholdResult` with SR fields
**Rejected**: `cavity.py` computes cavity-QED quantities; the *interpretation* of
those quantities as SR vs CW regimes belongs in a separate module.  This follows the
existing architecture where `cavity.py` computes g₀, κ, C and `maxwell_bloch.py`
uses them for dynamics.

### C. Use Γ_SR = g_eff² / (κ/2) as the regime parameter
Considered, but the `g_eff vs κ/2` comparison is cleaner and more physically
transparent.  Both criteria agree at the boundary and agree with Kersten 2026 Table 1.

---

## Consequences

### Positive
- Analytic regime maps run in microseconds vs seconds for ODE.
- `power_enhancement` quantifies the burst advantage over CW for hardware tradeoffs.
- Three-condition classify logic matches Kersten 2026 exactly — provides a direct
  reference implementation for the experimental paper.
- `compute_superradiance` integrates cleanly with existing `CavityProperties` and
  `ThresholdResult` without any new config fields.

### Negative / Accepted risks
- Analytic pulse amplitudes assume a *uniform inversion* initial state.  The spectral
  MB model uses a qGaussian-distributed inhomogeneous inversion; analytic estimates
  will slightly overestimate peak power (< 2× for typical Q-parameters).
- `t_D` formula is for a Fock-state seed; real NV maser starts from a thermal state.
  The delay estimate is accurate to within a factor ∼ ln(n_sp) where n_sp ≈ 1.

---

## References

- Dicke R. H. (1954) "Coherence in spontaneous radiation processes." *Phys. Rev.* **93**, 99.
- Tavis M. & Cummings F. W. (1968) *Phys. Rev.* **170**, 379.
- Kersten et al. (2026) "Superradiant masing from NV centres." *Nature Physics*, PMC12811124.
- Breeze J. et al. (2018) "Continuous-wave room-temperature diamond maser." *Nature* **555**, 493.
