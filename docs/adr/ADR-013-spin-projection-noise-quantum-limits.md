# ADR-013: Spin Projection Noise and Quantum-Enhanced Magnetometry

**Status**: Accepted  
**Date**: 2025-07-20  
**Module**: `src/nv_maser/physics/spin_squeezing.py`  
**Tests**: `tests/test_spin_squeezing.py` (50 tests)

---

## Context

The NV diamond maser digital twin already models two quantum-noise-limited
sensitivity floors:

| Module | Mechanism | Limit |
|--------|-----------|-------|
| `quantum_noise.py` | Phase diffusion (Schawlow-Townes) | Oscillator linewidth Δν_ST |
| `sensitivity.py` | S-T converted to field | η_B_ST [T/√Hz] |
| `stability.py` | Allan deviation σ_B(τ) | Time-domain from η_ST |

These all characterise the **oscillator mode**: cavity photon shot noise sets
the frequency stability of the maser's microwave output.

Missing from the model was the **spin-ensemble mode**: when the NV spins are
used directly as a Ramsey sensor (free precession + projection readout), the
measurement is limited by *spin projection noise* (quantum projection noise,
QPN) — the irreducible variance of a spin-½ observable on a coherent spin
state.

The two limits are complementary and coexist in the same device:

```
    ┌──────────────────────────────────────────────────────────────────┐
    │  NV maser magnetometer                                           │
    │                                                                  │
    │  Oscillator channel (cavity photons)                             │
    │    → Schawlow-Townes → η_ST (sensitivity.py / stability.py)     │
    │                                                                  │
    │  Spin channel (NV ensemble)                          [NEW ADR]  │
    │    → Projection noise → η_SQL (spin_squeezing.py)               │
    │    → Entanglement    → η_OAT, η_HL                              │
    └──────────────────────────────────────────────────────────────────┘
```

For the default NV configuration (N_eff ≈ 10¹²,  T₂* = 1 µs):

```
η_B_SQL = 1 / (2π × 28.025e9 × √(10¹² × 10⁻⁶)) ≈ 5.7 fT/√Hz
η_B_HL  = 1 / (2π × 28.025e9 × 10¹² × √10⁻⁶)   ≈ 5.7 aT/√Hz

Heisenberg advantage (√N_eff) ≈ 3.2 × 10⁵ (×320 000 better with full entanglement)
```

This complete the Phase C *quantum correlations* roadmap item.

---

## Decision

Implement `src/nv_maser/physics/spin_squeezing.py` providing:

### Data Classes

| Class | Contents |
|-------|----------|
| `ProjectionNoiseResult` | SQL + HL phase and field sensitivities, Heisenberg advantage |
| `SpinSqueezingResult` | ξ²_R, G_met [dB], below-SQL flag, regime label |
| `QuantumEnhancementResult` | Combines above; OAT optimum; hierarchy of sensitivities |

### Functions

| Function | Purpose |
|----------|---------|
| `compute_sql_phase_sensitivity(N)` | δφ_SQL = 1/√N |
| `compute_hl_phase_sensitivity(N)` | δφ_HL = 1/N |
| `compute_sql_field_sensitivity(N, τ, γ)` | η_B_SQL = 1/(2πγ √(Nτ)) [T/√Hz] |
| `compute_hl_field_sensitivity(N, τ, γ)` | η_B_HL = 1/(2πγ N √τ) [T/√Hz] |
| `compute_projection_noise(N, τ, γ)` | Assembles `ProjectionNoiseResult` |
| `compute_wineland_squeezing(N, var, ⟨Jz⟩)` | ξ²_R = N ⟨ΔJ_⊥²⟩ / ⟨Jz⟩² |
| `compute_oat_optimal_squeezing(N)` | ξ²_R^{min,OAT} ≈ N^{-2/3} |
| `compute_metrological_gain_db(ξ²_R)` | G = −10 log₁₀(ξ²_R) [dB] |
| `classify_squeezing_regime(ξ²_R, N)` | COHERENT / SQUEEZED / NEAR_HEISENBERG |
| `compute_spin_squeezing(N, ξ²_R)` | Assembles `SpinSqueezingResult` |
| `compute_quantum_enhancement(nv_config, N_eff, τ, ξ²_R)` | Full analysis |

---

## Physics

### Standard Quantum Limit (SQL)

For N independent spin-½ particles in a Ramsey sequence with interrogation
time τ, the minimum detectable field is:

$$
\eta_{B}^{\text{SQL}} = \frac{1}{2\pi\,\gamma_e \sqrt{N\tau}}
\quad [\text{T}/\sqrt{\text{Hz}}]
$$

(Degen, Reinhard & Cappellaro, Rev. Mod. Phys. 89, 035002 (2017), Eq. 9)

This arises from quantum projection noise: measuring along ±z, each spin
collapses with standard deviation σ = ½, giving collective phase uncertainty
δφ_SQL = 1/√N.

### Heisenberg Limit (HL)

For maximally entangled N-particle states (GHZ / NOON), the phase uncertainty
scales as 1/N rather than 1/√N:

$$
\eta_{B}^{\text{HL}} = \frac{1}{2\pi\,\gamma_e\, N\sqrt{\tau}}
\quad [\text{T}/\sqrt{\text{Hz}}]
$$

The ratio η_SQL / η_HL = √N sets the maximum gain entanglement can provide.

### Wineland Squeezing Parameter

The Wineland criterion (Wineland et al., PRA 46, R6797, 1992) quantifies
metrologically useful entanglement:

$$
\xi_R^2 = \frac{N\,\langle\Delta J_\perp^2\rangle_{\min}}{\langle J_z\rangle^2}
$$

where ⟨ΔJ_⊥²⟩_min is the variance minimised over transverse directions.

| State | ξ²_R | Regime |
|-------|------|--------|
| Coherent spin state (CSS) | 1 | At SQL |
| OAT-squeezed (large N) | ≈ N^{-2/3} | Below SQL |
| GHZ / NOON state | 1/N | Heisenberg limit |

### One-Axis Twisting (OAT) Optimal Squeezing

Under the OAT Hamiltonian H = χJ_z², the minimum achievable squeezing
scales asymptotically as (Kitagawa & Ueda, PRA 47, 5138, 1993):

$$
\xi_{R,\min}^{2,\text{OAT}} \approx N^{-2/3} \quad (N \gg 1)
$$

OAT provides sub-SQL sensitivity but does not reach the Heisenberg limit.
Two-axis twisting (TAT) achieves ξ²_R ∝ N^{-1}, at the cost of requiring
competing interactions.

---

## Consequences

### Positive
- **Completes Phase C "quantum correlations"** roadmap item.
- **Explicit sensitivity hierarchy**: users can now compare η_HL ≤ η_OAT ≤ η_SQL
  to the Schawlow-Townes floor from `sensitivity.py` in one simulation run.
- **Metrological gain [dB]** provides an actionable figure of merit for
  evaluating future squeezing protocols.
- **Wineland criterion** accepts measured spin statistics, enabling
  experimental validation against quantum-optics literature values.
- **Zero new dependencies**: pure Python/math, no NumPy required.

### Negative / Trade-offs
- OAT optimal squeezing is a large-N asymptote; for N < 10 the formula
  ξ²_R ≈ N^{-2/3} overestimates the achievable squeezing.  For the NV
  ensemble (N_eff ≈ 10¹²) the asymptotic result is highly accurate.
- Decoherence during squeezing (T₂* limited) is not modelled.  In practice,
  the OAT squeezing time τ* ~ N^{-2/3}/χ must satisfy τ* ≤ T₂*, which
  limits the practically achievable ξ²_R to values larger than N^{-2/3}.
  This is deferred to a future `squeezing_dynamics.py` module.
- TAT squeezing (Heisenberg scaling) is not implemented; only the OAT
  asymptote is provided as a benchmark.

---

## Alternatives Considered

| Option | Reason Rejected |
|--------|-----------------|
| Add projection noise to `sensitivity.py` | Would conflate two distinct physical mechanisms (oscillator vs sensor mode) |
| Add to `quantum_noise.py` | `quantum_noise.py` is already 500+ lines and focuses on maser photon noise |
| Postpone to Phase D | Directly enables comparisons in unit tests and the inference API `/metrics` endpoint; low implementation cost |

---

## Cross-Links

| Module | Relationship |
|--------|-------------|
| `sensitivity.py` | Parallel sensitivity floor (oscillator); compare S-T vs SQL |
| `quantum_noise.py` | Phase noise spectra; SQL is the *spin* counterpart of photon shot noise |
| `stability.py` | Allan deviation from S-T; SQL gives the spin-ensemble stability analogue |
| `cavity.py` | N_eff from `compute_n_effective()` feeds `compute_quantum_enhancement()` |
| `superradiance.py` | Cooperative emission; collective coupling g_eff uses √N scaling like SQL |

---

## References

1. Wineland, D. J. et al., "Spin squeezing and reduced quantum noise in
   spectroscopy," *PRA* **46**, R6797 (1992).
2. Kitagawa, M. & Ueda, M., "Squeezed spin states," *PRA* **47**, 5138 (1993).
3. Degen, C. L., Reinhard, F. & Cappellaro, P., "Quantum sensing,"
   *Rev. Mod. Phys.* **89**, 035002 (2017).  Eq. (9) for SQL sensitivity.
4. Ma, J. et al., "Quantum spin squeezing," *Phys. Rep.* **509**, 89 (2011).
   Sec. 3.2 for OAT optimal squeezing scaling.
