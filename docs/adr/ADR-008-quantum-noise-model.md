# ADR-008: Quantum Langevin Noise Model

**Status**: Accepted  
**Date**: 2025-01  
**Author**: NV Maser Digital Twin Project  

---

## Context

The semiclassical Maxwell-Bloch solver (`maxwell_bloch.py`) predicts the deterministic
mean-field dynamics of the NV maser: the evolution of the cavity field amplitude, spin
coherence, and population inversion, converging to a steady-state intracavity photon
number N̄ and output power P_out.

This description is *noise-free*.  It gives no information about the fundamental quantum
noise performance of the maser — its output linewidth, phase noise, noise temperature, or
relative intensity noise (RIN).  Without these, the `signal_chain.py` SNR budget misses
the maser's principal value: its sub-Kelvin noise temperature (orders of magnitude below
resistive or solid-state amplifiers).

### What Phase C requires

To complete the noise budget and make the digital twin useful for hardware design
decisions, we need a quantum noise layer that:

1. Quantifies **why** the NV maser is compelling (noise temperature < 0.5 K vs. ~290 K
   for a room-temperature low-noise amplifier).
2. Provides the **Schawlow-Townes linewidth** — the ultimate spectral purity limit,
   which constrains the minimum detectable frequency shift in NV magnetometry.
3. Computes **phase noise PSD** and **RIN spectrum** in standard RF engineering formats
   (dBc/Hz, dBc/Hz) for comparison against datasheet specifications.
4. Enables downstream sensitivity analysis: how does noise temperature scale with cavity
   Q, pump efficiency, and intracavity photon number?

---

## Decision

Implement a **quantum Langevin noise model** in
`src/nv_maser/physics/quantum_noise.py`, computing:

| Quantity | Formula | Physical meaning |
|---|---|---|
| Population inversion factor | n_sp = (1+η)/(2η) | Excess spontaneous emission noise |
| Added noise (Caves theorem) | n_add = n_sp | Quantum amplifier noise floor |
| Schawlow-Townes linewidth | Δν = κ_c n_sp / (2N̄) | Maser coherence time |
| Noise temperature | T_n = ℏω n_sp / k_B | Equivalent thermal noise |
| Phase noise PSD | S_φ(f) = Δν/(πf²) | 1/f² phase diffusion |
| RIN spectrum | RIN(f) = (2n_sp/N̄)/(1+(f/κ_c)²) | Cavity-filtered shot noise |

### Why this approach

**Quantum Langevin equations** (from input-output theory) rigorously describe quantum
fluctuations in driven-dissipative cavity QED systems.  In the mean-field (above-
threshold, large-N̄) limit, they reduce to:

- **Phase sector**: random walk of the field phase with diffusion coefficient D_φ =
  κ_c n_sp / (4N̄) rad²/s → Schawlow-Townes linewidth Δν = D_φ/π.
- **Amplitude sector**: photon-number fluctuations with correlation time 1/κ_c → the
  Lorentzian RIN spectrum.

This is the standard treatment (Lax 1966; Yamamoto & Haus 1986) and gives closed-form
results with no additional free parameters beyond those already in the model.

### Design boundaries

This module intentionally omits:

- **Technical noise** (vibration, temperature drift, power supply noise) — these are not
  fundamental quantum limits.
- **Relaxation oscillations** — the NV maser has overdamped amplitude dynamics
  (1/T₁ ≈ 200 Hz ≪ κ_c/2π ≈ 147 kHz) so no oscillation peak appears in the RIN.
- **Multi-mode effects** — single-mode cavity approximation is valid for the 50 mT
  Halbach geometry with mode spacing >> gain linewidth.
- **Quantum correlations / squeezing** — only coherent-state mean-field statistics; below
  the single-photon threshold the Schawlow-Townes formula does not apply.

---

## Consequences

### Positive

- **Noise temperature closes the SNR budget**: `signal_chain.py` can use
  `MaserNoiseResult.noise_temperature_k` as the LNA input noise figure.
- **Linewidth enables sensitivity calculation**: Schawlow-Townes linewidth bounds the
  minimum measurable frequency shift → minimum detectable B-field.
- **Zero new dependencies**: pure NumPy/SciPy; no new packages.
- **Composable**: the module takes `CavityProperties + MaxwellBlochResult` as inputs,
  maintaining the established pipeline architecture.

### Neutral

- The model uses `pump_efficiency` (η) from `NVConfig` as a proxy for the population
  inversion factor n_sp.  This is appropriate for the two-level approximation but does
  not account for the ms=0 → ms=-1 → ms=+1 shelving dynamics in NV centres.  A more
  detailed model would require solving the NV density matrix (Phase D scope).

### Negative / Risks

- **Below-threshold behaviour**: when N̄ → 0, the Schawlow-Townes formula diverges.
  The implementation handles this by returning κ_c/(2π) as a conservative upper bound,
  but this should be interpreted cautiously (the maser is not coherent below threshold).
- **Single-mode approximation**: for very wide cavity linewidths (low Q < 1000), the
  multi-mode cavity could support several maser lines; this module would undercount the
  noise.

---

## Numerical example (default NVConfig / MaserConfig)

```
cavity_frequency     = 1.47 GHz
cavity_Q             = 10 000
κ_c / (2π)          = 1.47 GHz / 10 000 = 147 kHz
pump_efficiency (η)  = 0.5
n_sp                 = (1 + 0.5) / (2 × 0.5) = 1.5
noise temperature    = ℏ × 2π × 1.47e9 × 1.5 / k_B ≈ 0.11 K

For N̄ = 1000 photons intracavity:
  Schawlow-Townes Δν  = 147 000 × 1.5 / (2 × 1000) = 110.25 Hz
  Phase noise L(1 Hz) = 10 log10(110.25 / (2π))     ≈ 12.4 dBc/Hz
  RIN floor            = 2 × 1.5 / 1000              = 3 × 10⁻³ Hz⁻¹ = −25.2 dBc/Hz
```

Compare to a room-temperature LNA: T_noise ≈ 50–300 K → SNR improvement factor ≈
500–3000.  This is the defining advantage of the NV maser architecture.

---

## References

1. Schawlow, A. L., Townes, C. H. (1958). "Infrared and Optical Masers." *Phys. Rev.*
   112, 1940–1949.
2. Caves, C. M. (1982). "Quantum limits on noise in linear amplifiers." *Phys. Rev. D*
   26, 1817–1839.
3. Lax, M. (1966). "Quantum Noise. X: Density-Matrix Treatment of Field and
   Population-Difference Fluctuations." *Phys. Rev.* 157, 213–231.
4. Yamamoto, Y., Haus, H. A. (1986). "Preparation, measurement and information capacity
   of optical quantum states." *Rev. Mod. Phys.* 58, 1001.
5. Henry, C. H. (1982). "Theory of the linewidth of semiconductor lasers." *IEEE J.
   Quantum Electron.* 18, 259–264.
6. Wang, J. et al. (2024). "Room-temperature diamond maser operating at a magic angle."
   *Advanced Science*, PMC11425272.
