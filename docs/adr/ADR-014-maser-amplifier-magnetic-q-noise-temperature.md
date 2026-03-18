# ADR-014: Maser-as-Amplifier Model — Magnetic Q Factor, Noise Temperature, and Output Power

**Status**: Accepted
**Date**: 2026-03-18
**Deciders**: NV Maser Digital Twin team
**Supersedes**: —
**Cross-links**: ADR-008 (quantum noise), ADR-010 (signal chain), ADR-011 (sensitivity), ADR-013 (spin squeezing)

---

## Context

Prior to this ADR, the digital twin evaluated maser performance solely via the cooperativity criterion (ADR-005, ADR-010):

```
C = 4 g_N² / (κ · γ⊥) > 1  →  masing
```

While correct, this steady-state threshold criterion provides only a binary yes/no verdict.  It does not answer the experimentally critical questions:

1. **How good an amplifier is the sub-threshold maser?** — characterised by noise temperature T_a.
2. **What output power does the oscillator produce above threshold?** — needed for SNR budget validation.
3. **How does the "magnetic quality factor" Q_m complement cooperativity?** — Wang et al. (2024) use Q_m as their primary threshold metric.

Wang et al. (Advanced Science, 2024) demonstrated a room-temperature X-band pentacene maser as a low-noise microwave amplifier and provided closed-form expressions for these missing quantities.  The same physics applies to NV diamond masers.

---

## Decision

Implement `src/nv_maser/physics/amplifier.py` with three new physical models from Wang et al. (2024):

1. **Magnetic quality factor Q_m** — characterises spin gain strength.
2. **Spin temperature T_s and noise temperature T_a** — characterises amplifier noise.
3. **CW output power** — estimates above-threshold oscillator output.

---

## Physics Background

### 1. Magnetic Quality Factor (Wang 2024, Eq. 1)

The spin-gain strength is quantified as:

$$Q_m^{-1} = \frac{\mu_0 \hbar \gamma_e^2 \sigma_2 \eta \, \Delta n \, T_2}{2}$$

| Symbol | Meaning | Typical NV value |
|--------|---------|-----------------|
| $\mu_0$ | Vacuum permeability | $4\pi \times 10^{-7}$ H/m |
| $\hbar$ | Reduced Planck constant | $1.055 \times 10^{-34}$ J·s |
| $\gamma_e$ | Electron gyromagnetic ratio (angular) | $2\pi \times 28.025 \times 10^9$ rad/s/T |
| $\sigma_2$ | Transition matrix element for S=1, $\pi$-polarised B₁ | 0.5 |
| $\eta$ | Cavity filling factor $V_\text{diamond}/V_\text{mode}$ | 0.01 |
| $\Delta n$ | Inverted spin density $n_\text{NV} \times \eta_\text{pump} \times \eta_\text{orient}$ | $\sim 10^{17}$–$10^{23}$ m⁻³ |
| $T_2$ | Spin coherence time | 1 µs |

**Threshold condition**: $Q_m \leq Q_L$ (loaded cavity Q), which is equivalent to $C \geq 1$ (see [Relationship to Cooperativity](#relationship-to-cooperativity) below).

### 2. Spin and Noise Temperature (Wang 2024, Eq. 4)

**Spin temperature** of the NV ensemble:

$$T_s = \frac{\hbar \omega_c}{k_B \ln(p_\text{upper} / p_\text{lower})}$$

For the NV maser transition (ms=0 upper, ms=+1 lower):
- Optical pumping fills ms=0 → $p_\text{upper} \approx \eta_\text{pump}$
- Lower state: $p_\text{lower} \approx (1 - \eta_\text{pump})/2$
- At $\eta_\text{pump} = 0.5$, $f_c = 1.47$ GHz: $T_s \approx 0.12$ K — far colder than the 300 K bath.

**Amplifier noise temperature** (Wang 2024, Eq. 4):

$$T_a = \frac{Q_m}{Q_0 - Q_m} T_\text{bath} + \frac{Q_0}{Q_0 - Q_m} T_s$$

where $Q_0 = Q_L (1 + \beta)$ is the unloaded cavity Q and $\beta$ is the coupling coefficient.

For a well-inverted maser ($T_s \ll T_\text{bath}$, $Q_m \ll Q_0$):

$$T_a \approx \frac{Q_m}{Q_0} T_\text{bath}$$

This is substantially below $T_\text{bath}$ = 300 K, confirming the maser's utility as a low-noise amplifier.

**Standard Quantum Limit** (Caves 1982):

$$T_\text{SQL} = \frac{\hbar \omega_c}{2 k_B}$$

Any linear phase-preserving amplifier satisfies $T_a \geq T_\text{SQL}$.  A maser can approach (but not surpass) this limit.

### 3. CW Output Power

Above threshold, energy conservation gives:

$$P_\text{out} = \frac{\hbar \omega_c}{2 T_1} \cdot N_\text{eff} \cdot (p_0 - p_\text{th}) \cdot \frac{\beta}{1+\beta}$$

where $p_\text{th} = 1/C$ is the threshold inversion fraction.  The $\beta/(1+\beta)$ factor routes the fraction of generated power to the output port.

Intracavity photon number:

$$n_\text{ss} = \frac{P_\text{out}}{\hbar \omega_c \kappa_\text{ext}}, \quad \kappa_\text{ext} = \frac{\omega_c}{Q_L} \cdot \frac{\beta}{1+\beta}$$

---

## Relationship to Cooperativity

The two threshold criteria (C > 1 and Q_m < Q_L) are equivalent but derived from different physical pictures.  To show this, note (at microwave regime):

$$C = \frac{4 g_N^2}{\kappa \gamma_\perp} \propto \frac{\Delta n \cdot \eta \cdot T_2}{Q_L^{-1}}$$

and

$$Q_m^{-1} \propto \mu_0 \hbar \gamma_e^2 \cdot \eta \cdot \Delta n \cdot T_2$$

Both scale identically with the key parameters ($\Delta n$, $\eta$, $T_2$) and both threshold at:

$$C > 1 \iff Q_m < Q_L$$

(The numerical coefficients differ because C uses the cavity-mode vacuum coupling strength while Q_m uses the bulk susceptibility formulation.)  The digital twin now provides **both** diagnostic metrics, enabling comparison with papers using either convention.

---

## NV Energy Level Convention

For the NV diamond maser at $B_0 \approx 0.05$ T along the NV axis, the relevant spin levels are:

| State | Energy (Hz) | Role in maser |
|-------|-------------|--------------|
| ms=0 | 0 | **Upper lasing state** (populated by optical pumping) |
| ms=+1 | $D - \gamma_e B_0 = 1.47$ GHz | **Lower lasing state** |
| ms=−1 | $D + \gamma_e B_0 = 4.27$ GHz | Spectator (off-resonant) |

Hamiltonian used: $H/\hbar \omega = D S_z^2 - \gamma_e B_0 S_z$ (positive $\gamma_e$ convention).

This energy ordering means optical pumping into ms=0 creates population inversion for the 1.47 GHz transition.

---

## Module Design

```python
# src/nv_maser/physics/amplifier.py

SIGMA_2 = 0.5  # S=1 transition matrix element

@dataclass(frozen=True)
class AmplifierProperties:
    magnetic_q: float           # Q_m
    loaded_q: float             # Q_L
    unloaded_q: float           # Q₀ = Q_L(1+β)
    above_threshold: bool       # Q_m ≤ Q_L
    spin_temperature_k: float   # T_s
    noise_temperature_k: float  # T_a
    sql_noise_temp_k: float     # T_SQL = ℏω/(2k_B)
    below_sql: bool             # T_a ≤ T_SQL?

@dataclass(frozen=True)
class OutputPowerResult:
    intracavity_photon_number: float  # n_ss
    output_power_w: float             # P_out (W)
    output_power_dbm: float           # P_out (dBm)

def compute_magnetic_q(nv_config, cavity_config, maser_config) -> float
def compute_spin_temperature(p_upper, p_lower, cavity_freq_ghz) -> float
def compute_noise_temperature(q_m, q0, T_s, T_bath=300.0) -> float
def compute_sql_noise_temperature(cavity_freq_ghz) -> float
def compute_output_power(cavity_props, threshold, nv_config, maser_config) -> OutputPowerResult
def compute_amplifier_properties(nv_config, cavity_config, maser_config, T_bath=300.0) -> AmplifierProperties
```

---

## Test Coverage (55 tests)

| Test class | Focus | Count |
|------------|-------|-------|
| `TestSigma2` | σ₂ constant value | 2 |
| `TestComputeMagneticQ` | Q_m formula, scaling, edge cases | 11 |
| `TestComputeSpinTemperature` | T_s formula, inversion, NaN cases | 10 |
| `TestComputeNoiseTemperature` | T_a formula, threshold NaN, Wang check | 8 |
| `TestComputeSqlNoiseTemperate` | T_SQL formula, scaling, milliKelvin range | 5 |
| `TestComputeOutputPower` | Below thrsh = 0, above threshold > 0 | 5 |
| `TestDerivePopulationFractions` | Population model | 4 |
| `TestAmplifierProperties` | All fields, frozen dataclass, noise comparison | 10 |
| `TestConsistencyWithCooperativity` | C > 1 ↔ Q_m < Q_L | 4 |
| `TestQuantitativeChecks` | Physical-scale + Wang 2024 cross-check | 5 |

---

## Consequences

### Positive
- **Noise temperature is now computable**: Circuit designers can now budget T_a vs. front-end LNA (e.g., SiGe: 2–15 K).
- **Output power model**: Enables SNR predictions without full Maxwell-Bloch simulation.
- **Wang 2024 cross-validation**: Direct comparison with the paper's amplifier characterisation data.
- **Q_m supplements cooperativity C**: Both threshold metrics available; Q_m is more natural when comparing to susceptibility-based measurements.

### Negative / Constraints
- **Simplified population model**: `p_upper = pump_efficiency`, `p_lower = (1-η_p)/2` — a rough approximation. A more rigorous calculation would require integrating `optical_pump.py` steady-state.
- **Homogeneous broadening assumed**: T_s formula ignores the inhomogeneous distribution of NV transition frequencies (spectral broadening by crystal strain, dipolar fields).
- **Output power model is analytic**: Not derived from Maxwell-Bloch ODE; valid in the strongly-pumped steady-state limit well above threshold.

---

## Alternatives Considered

1. **Extend cavity.py** — Q_m is closely related to cavity QED threshold, but combining amplifier characterisation with the cavity module would mix two distinct physical concepts (mode coupling vs. bulk susceptibility). Separate module is cleaner.

2. **Import compute_noise_temperature from quantum_noise.py** — `quantum_noise.py` already exports a `compute_noise_temperature` but uses the Friis/Caves formalism (noise figure from photon statistics). Wang's formula is a different (susceptibility-based) derivation. Both are preserved under distinct names: `compute_noise_temperature` (Friis, in `quantum_noise.py`) vs. `compute_noise_temperature` imported as `compute_amplifier_noise_temperature` in `__init__.py`.

3. **Skip spin temperature** — T_s is needed for Wang Eq. 4; without it, T_a cannot be calculated from first principles.

---

## References

1. **Wang et al.**, "Tailoring Coherent Microwave Emission with Supramolecular Spin-Photon Interactions", *Advanced Science*, 2024. PMC11425272. [Key equations: Eq. 1 (Q_m), Eq. 4 (T_a)]

2. **Caves, C. M.** (1982). "Quantum limits on noise in linear amplifiers". *Physical Review D*, 26(8), 1817. [SQL for phase-preserving amplifiers]

3. **Breeze et al.**, "Continuous-wave room-temperature diamond maser", *Nature* 555, 493 (2018). [NV maser oscillator, sets target for the twin]

4. **Jin et al.**, "Proposal for a room-temperature diamond maser", *Nature Communications* 6, 8251 (2015). [Theoretical foundation]
