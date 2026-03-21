# ADR-029: Phase-1 Maser Oscillation Milestone Validator

**Status**: Accepted  
**Date**: 2025-01-01  
**Session**: SS25

---

## Context

Architecture doc §12.2 lists six critical milestones for the handheld NV maser
probe.  Sessions SS22–SS24 created formal digital-twin validators for four of
them (Phase-4 depth profile, Phase-6 first 2D image, Phase-9 tissue contrast).
The initial **Phase-1 maser oscillation** milestone — the earliest hardware
checkpoint — lacked a dedicated validator module.

The Phase-1 success criterion per §12.2 is:

> **Maser oscillation — Detectable stimulated emission at 1.47 GHz**  (Phase 1)

This is the foundational prerequisite for all subsequent milestones: without
CW oscillation at the NV zero-field microwave transition (~2.87 GHz shifted to
~1.47 GHz at 50 mT), there is no maser amplification and no image.

---

## Decision

Implement `src/nv_maser/physics/phase1_validator.py` following the same
validator pattern established by `phase4_validator.py` / `phase6_validator.py`
/ `phase9_validator.py`.

The validator uses two independent physical frameworks to assess threshold,
plus a frequency and power check:

### Criterion 1 — Oscillation threshold (two sub-criteria)

**(a) Wang 2024 magnetic-Q criterion** (`amplifier.py::compute_amplifier_properties`):

$$Q_m^{-1} = \frac{\mu_0 \hbar \gamma_e^2 \sigma_2 \eta \Delta n T_2}{2}$$

Oscillation requires $Q_m \leq Q_L$.  With defaults ($\rho_\text{NV}=10^{17}$
/cm³, $T_2^*=1$µs, $\eta=0.01$, $Q_L=10\,000$):

$$Q_m \approx 7\,788 < Q_L = 10\,000 \quad \Rightarrow \quad \text{above\_threshold} = \text{True}$$

**(b) Cooperativity criterion** (Breeze 2018, `cavity.py::compute_full_threshold`):

$$C = \frac{4 g_N^2}{\kappa \gamma_\perp}, \quad g_N = g_0 \sqrt{N_\text{eff}}$$

Oscillation requires $C > 1$.  Default result: $C \approx 2.57$.

Both sub-criteria must be satisfied for `threshold_met = True`.

### Criterion 2 — Frequency

$$|\nu_\text{cavity} - 1.47\text{ GHz}| \leq 50\text{ MHz}$$

50 MHz tolerance accommodates small variations in the 50 mT static field.

### Criterion 3 — Output power

$$P_\text{out} \geq -100\text{ dBm}$$

The default CW power is approximately $−69.5$ dBm — some 4.5 dB above the
−100 dBm floor, which is itself 74 dB above room-temperature thermal noise
in a 1 Hz bandwidth.

### Public API

```python
from nv_maser.physics.phase1_validator import (
    Phase1Config,
    OscillationThresholdResult,
    Phase1MilestoneResult,
    validate_phase1_milestone,
)

result = validate_phase1_milestone()
assert result.phase1_milestone_closed
```

---

## Baseline values (default configuration)

| Quantity | Value | Criterion | Pass? |
|----------|-------|-----------|-------|
| Q_m | 7 788 | < Q_L = 10 000 | ✅ |
| Q_m / Q_L | 0.779 | < 1 | ✅ |
| Cooperativity C | 2.57 | > 1 | ✅ |
| Oscillation frequency | 1.47 GHz | ±50 MHz from target | ✅ |
| CW output power | −69.5 dBm | ≥ −100 dBm | ✅ |
| Spin temperature | 0.10 K | (diagnostic only) | — |
| **phase1_milestone_closed** | **True** | all three pass | **✅** |

---

## Consequences

### Positive

* Closes the last open §12.2 critical milestone — the full set of six
  milestones (Maser oscillation, Sweet-spot, Depth profile, 2D image,
  Tissue contrast) now has formal digital-twin validators.
* Two independent threshold criteria (Wang Q-factor + Breeze cooperativity)
  provide complementary physical evidence, reducing the risk of a false pass.
* Test suite extended by 69 tests (+3% from 2405 baseline).

### Neutral

* The validator intentionally uses existing `amplifier.py` and `cavity.py`
  APIs without modification — no physics code was changed.
* `Phase1Config.gain_budget` defaults to 0.5, matching
  `MaserConfig.min_gain_budget`; this can be adjusted to reflect measured
  spectral overlap in a real device.

### Negative / Trade-offs

* The CW output power formula assumes steady-state oscillation above threshold;
  it does not model start-up transients or mode competition.
* The frequency criterion checks `maser_config.cavity_frequency_ghz` directly
  — it does not currently simulate the Zeeman shift from the imaging magnet
  (addressed separately by the field-tolerance module, R1 closure ADR-024).

---

## References

* Wang et al., Advanced Science (2024) PMC11425272 — Eq. 1 (Q_m), Eq. 4 (T_a).
* Breeze et al., *Nature* 555, 493–496 (2018) — Cooperativity threshold.
* handheld-maser-probe-architecture.md §12.1–12.2 (milestone table).
* ADR-018 — Gain-lock PI control (SS14, R10 closure).
* ADR-026 — Phase-4 depth-profile milestone (SS22).
* ADR-027 — Phase-6 grid-phantom milestone (SS23).
* ADR-028 — Phase-9 tissue-contrast milestone (SS24).
