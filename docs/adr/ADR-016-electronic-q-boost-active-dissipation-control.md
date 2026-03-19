# ADR-016 — Electronic Q-Boosting: Active Cavity-Loss Compensation

**Status:** Accepted  
**Date:** 2025-07-20  
**Module:** `src/nv_maser/physics/q_boost.py`  
**Tests:** `tests/test_q_boost.py` (48 tests)

---

## Context

Wang et al. (2024) demonstrated room-temperature masing in two distinct regimes from a
single pentacene-loaded sapphire resonator:

1. **Pulsed amplification** (no Q-boost): Q_m ≈ 1589 > Q_L ≈ 1337 → gain ≈ 14.5 dB.
2. **CW oscillation** (with Q-boost B ≈ 59): Q_m ≈ 1589 ≪ Q_L_eff ≈ 6.5 × 10⁵.

The transition is accomplished by an electronic feedback controller that cancels the
majority of the cavity's total decay rate, effectively boosting the loaded quality
factor from 1.1 × 10⁴ to 6.5 × 10⁵ — a factor of ≈ 59.

Our prior models (`amplifier.py`, `maser_gain.py`) compute gain and noise temperature
for a passive cavity.  They do not model active dissipation control and therefore
cannot reproduce regime (2) or quantify the noise-temperature improvement it provides.

## Decision

Implement a standalone `q_boost.py` module that:

- Computes the effective *loaded* and *unloaded* Q factors after boost.
- Gives the revised oscillation-threshold condition and threshold-reduction factor.
- Evaluates the Wang Eq. 4 noise temperature with the boosted Q₀.
- Provides `compute_sql_limit_ratio` to benchmark against the standard quantum limit.

The module deliberately does **not** model the loop dynamics (bandwidth, stability
margins) of the electronic controller — only its steady-state effect on cavity decay.

## Physical Model

### Passive cavity

| Quantity | Symbol | Definition |
|----------|--------|------------|
| Internal loss rate | κ_i | ω_c / Q₀ |
| Coupling rate | κ_e | ω_c / Q_e ; Q_e = Q₀/β |
| Total decay rate | κ_total | κ_i + κ_e = ω_c / Q_L ; Q_L = Q₀/(1+β) |

### Active reduction of decay rate

The feedback controller measures the output field and re-injects a signal that cancels
fraction g_fb of κ_total:

```
κ_eff  = κ_total × (1 − g_fb) = κ_total / B        B ≥ 1
g_fb   = 1 − 1/B
```

Because κ_eff rescales κ_total uniformly, both Q_L and Q₀ scale by the same factor B:

```
Q_L_eff = B × Q_L        (≡ threshold_qm in QBoostResult)
Q₀_eff  = B × Q₀  = B × Q_L × (1 + β)
```

### Oscillation threshold

Masing condition: spin gain ≥ net cavity loss:

```
1/Q_m ≥ 1/Q_L_eff    ↔    Q_m ≤ Q_L_eff = B × Q_L
```

The passive threshold Q_m = Q_L is replaced by B × Q_L; a boost B reduces
the required spin gain Q_m by the same factor.

### Noise temperature (amplifier regime, Q_m > Q_L_eff)

Substituting Q₀_eff into Wang (2024), Eq. 4:

```
T_a = [ Q_m / (Q₀_eff − Q_m) ] × T_bath
    + [ Q₀_eff / (Q₀_eff − Q_m) ] × T_s
```

Limiting behaviour:

| Condition | T_a approaches |
|-----------|----------------|
| B = 1 (no boost) | Passive formula from ADR-014 |
| B → ∞ | T_s (spin-temperature limited — ideal quantum maser) |

If Q_m ≥ Q₀_eff the system is in oscillation; the formula is undefined and
`compute_noise_temperature_boosted` returns `float('nan')`.

### Standard Quantum Limit (SQL)

```
T_SQL = ℏ ω_c / (2 k_B)
```

At 9.4 GHz (Wang cavity): T_SQL ≈ 226 mK.  A value T_a / T_SQL < 1 is
quantum-limited (sub-SQL) performance.

## Wang 2024 Validation

| Quantity | Paper value | Module output | Check |
|----------|-------------|---------------|-------|
| Q_L_native | 1.1 × 10⁴ | 11 000 (input) | — |
| Q_L_effective | 6.5 × 10⁵ | 6.50 × 10⁵ | ≤ 1 % |
| Boost factor B | ≈ 59 | 59.09 | ✓ |
| Q₀_native (β=1) | 2.2 × 10⁴ | 22 000 | ✓ |
| Q₀_effective | 1.30 × 10⁶ | 1.30 × 10⁶ | ✓ |
| g_fb | 1 − 1/59 ≈ 0.983 | 0.9831 | ✓ |
| Q_m (passive amplifier) | 1589 > Q_L = 1337 | amplifier | ✓ |
| Q_m with boost | 1589 ≪ 6.5 × 10⁵ | oscillation | ✓ |

## API

```python
from nv_maser.physics.q_boost import (
    QBoostResult,            # frozen dataclass
    compute_q_boost,         # main factory function
    compute_minimum_boost,   # B_min for a given Q_m, Q_L
    compute_noise_temperature_boosted,
    compute_sql_limit_ratio,
)

# Wang 2024 — CW oscillation setup
rb = compute_q_boost(q_l_native=11_000.0, coupling_beta=1.0, boost_factor=59.0)
# → rb.q_l_effective ≈ 6.5e5

# How much boost is needed to oscillate at Q_m = 1589 with Q_L = 11 000?
# (Q_m < Q_L already → B_min = 1.0, no boost needed for threshold)
b_min = compute_minimum_boost(q_m=1589.0, q_l=11_000.0)   # → 1.0

# Noise temperature in amplifier regime (Q_m > Q_L_eff)
t_a = compute_noise_temperature_boosted(
    magnetic_q=8e5,         # just above Q_L_eff = 6.5e5
    q_boost_result=rb,
    spin_temperature_k=-0.001,
    bath_temperature_k=300.0,
)

# SQL ratio at 9.4 GHz
ratio = compute_sql_limit_ratio(t_a, cavity_frequency_ghz=9.4056)
```

## Alternatives Considered

**Extend `amplifier.py`** with a `boost_factor` parameter — rejected.  
The boost changes the *cavity* parameters, not the spin/amplifier physics.
Keeping it separate honours single-responsibility and makes both modules
independently testable.

**Model feedback loop dynamics** — deferred.  
Controller bandwidth and phase margin affect stability but not the steady-state
Q values.  A dynamic model would require the loop-gain transfer function which
is not available from Wang 2024.

## Consequences

- `QBoostResult` is composable with existing functions:
  pass `q0_effective` to `compute_noise_temperature` from ADR-014.
- `compute_maser_gain` (ADR-015) still applies when operating in the amplifier
  regime by substituting `q_l_effective` for `q_l`.
- Future work: integrate `ClosedLoopSimulator` with Q-boost by injecting
  `q_l_effective` into the loop gain at each step.

## References

Wang, J. et al. "Tailoring coherent microwave emission from spin-gain-enhanced
resonator for room-temperature maser." *Advanced Science* 11, 2309826 (2024).
