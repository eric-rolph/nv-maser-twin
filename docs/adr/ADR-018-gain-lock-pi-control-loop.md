# ADR-018: Gain-Lock PI Control Loop for Maser Threshold Stabilisation

**Status**: Accepted  
**Date**: 2025-07-22  
**Module**: `src/nv_maser/physics/gain_lock.py`  
**Tests**: `tests/test_gain_lock.py` (39 tests)  
**Risk Mitigated**: R10 — *Maser regenerative mode unstable*

---

## Context

The NV diamond maser operates in regenerative mode (CW self-oscillation) when
the ensemble cooperativity C ≥ 1:

    C = 4 g_N² / (κ · γ_⊥)

where

| Symbol | Quantity | Typical value |
|--------|----------|--------------|
| g_N = g₀√N_eff | ensemble vacuum Rabi coupling | ~500 kHz |
| κ = ω_c/Q_L | cavity field decay rate (Hz) | ~147 kHz (Q_L = 10 000) |
| γ_⊥ = 1/(π T₂*) | spin decoherence rate (HWHM, Hz) | ~318 kHz (T₂* = 1 μs) |
| N_eff = N × η_fill × η_pump | effective inverted spins | ~3×10¹⁴ |

Near threshold (C ≈ 1), the gain is extremely sensitive to laser pump power P
because η_pump(P) sets N_eff through the optical saturation curve.  Small
perturbations (laser pointing jitter, vibration, temperature drift) push C
above or below 1, causing the maser to oscillate chaotically or quench
entirely.

The handheld probe architecture (§13, Risk Register R10) states:

> *"Digital twin models threshold precisely; implement gain-lock control loop."*

Without active stabilisation, the device cannot be used in an uncontrolled
emergency-triage environment.

### Physics of the pump–cooperativity map

The mapping P → C is:

1. **Pump rate**: Γ_pump(P) = σ_abs × I₀(P) / (ℏ ω_L),  
   where I₀(P) = 2P/(π w₀²) is the peak Gaussian intensity.
2. **Saturation**: s(P) = Γ_pump / (Γ_pump + 1/T₁) ∈ [0, 1)
3. **Pump efficiency**: η_pump = s × (2/3)
4. **Effective spin count**: N_eff = N_total × η_fill × η_pump
5. **Cooperativity**: C ∝ N_eff (via `compute_full_threshold()`)

The function C(P) is monotone and sub-linear: it rises steeply near threshold
(dC/dP ≈ 256 C/W at P ≈ 3.5 mW for default NV/cavity parameters) and
flattens as the pump saturates at higher powers.

### Why standard closed-loop designs fail

A naïve proportional controller with gain K_p tries to correct pump power by
ΔP = K_p × e (where e = C_target - C).  Stability requires:

    K_p × dC/dP < 1   →   K_p < 1/256 ≈ 0.004 W/C

Many textbook choices (e.g. K_p = 0.05–0.1) are 10–25× too large and
cause persistent limit-cycle oscillations near threshold.

---

## Decision

Implement **`gain_lock.py`** as a self-contained gain-lock PI control loop
module in `nv_maser.physics`.

### Design choices

1. **Control variable**: laser power P (W) — directly accessible via the pump
   laser driver in hardware.

2. **Observable**: ensemble cooperativity C(P), computed on each step from
   `compute_cooperativity()` (calls `compute_pump_state()` + `compute_full_threshold()`).
   In hardware this would be derived from output power via a low-noise diode
   detector; here the exact physics model is used.

3. **PI law** (Euler forward, step Δt ):
   ```
   integral ← integral + e × Δt
   ΔP = Kp × e + Ki × integral
   P_new = clip(P + ΔP, P_min, P_max)
   ```
   where e = C_target − C.

4. **Default gains**: K_p = 0.001 W/C, K_i = 1.0 W·s⁻¹/C.  
   Chosen so that the closed-loop gain K_p × (dC/dP)_near_threshold = 0.001 × 256 = 0.26 < 1 (stable).  
   The integral term provides steady-state error rejection at timescales > 1 ms.

5. **Configuration**: all gains/limits encapsulated in frozen `GainLockConfig`
   dataclass.  No mutable global state.

6. **Step record**: each simulation step logs both the cooperative measured at
   the start (from the previous pump power) and the pump power commanded for
   the next step.  This is a standard "sample-and-update" digital control
   convention.

7. **Noise injection**: optional additive Gaussian noise on the C observable
   (controlled by `coop_noise_sigma`) models laser shot noise and pointing
   jitter for robustness testing.

8. **Threshold finder**: `find_threshold_pump_power()` uses bisection
   (60-iteration safety cap, 0.1 mW default tolerance) to locate P_th where
   C = 1 exactly.  This gives the hardware team the minimum pump power required
   to sustain oscillation under a given field quality (gain_budget).

---

## Alternatives Considered

### A. Output-power PI (measure P_out, control P pump)

The maser output power P_out ∝ max(0, C − 1) × H(gain_budget) is a natural
observable.  However it is zero below threshold, so the loop cannot acquire
lock from a cold start (quenched maser).  The cooperativity observable used
here is always positive and drives the pump toward threshold even when the
maser is not yet oscillating.

**Rejected**: fails to acquire lock from cold-start.

### B. Frequency-tracking PLL (adjust cavity frequency to match spin transition)

Frequency tracking (adjusting B₀ or cavity temperature) could be used to
maintain the spin transition inside the cavity bandwidth.  This addresses R10
indirectly but:
- Requires sub-MHz cavity frequency actuator (complex hardware)
- Does not address absolute gain margin; the maser can quench even on resonance
  if the pump is insufficient
- Already partially handled by the shimming controller (closed_loop.py)

**Rejected** as the primary R10 mitigation; may be implemented as a secondary
loop in future.

### C. Q-boost feedback as the control variable

`q_boost.py` implements electronic Q-boosting via a delay-line feedback loop.
Increasing Q_boost gain lowers the effective Q_L, increasing C.  This is faster
than pump-power adjustment (electrical vs. optical response times).  However:
- Q_boost gain < 1 is required for stability (ADR-016)
- Changing the Q_boost gain affects the noise temperature (Eq. 4, Wang 2024)
- The pump-power lever is the primary and simpler actuator

**Deferred**: could be a supplementary fast inner loop (ms bandwidth) around
the slower pump-power outer loop (100 ms bandwidth).

---

## Physical Model

### Gain-lock linearised stability analysis

Near the operating point P* = P_eq (where C(P*) = C_target):

    ΔC/ΔP ≈ dC/dP|_{P*}  (plant gain, C/W)

For default NV/cavity parameters, evaluated at P* ≈ 3.57 mW (C* = 1.10):

    Γ_pump(P*) ≈ 8.4 Hz    T₁ = 5 ms    1/T₁ = 200 Hz
    s* = 8.4/208.4 = 0.040    η* = 0.027
    N_eff* = 1.35 × 10¹²
    dC/dP|_{P*} ≈ 256 C/W

Stability margin (proportional only):
    Loop gain = K_p × 256 = 0.001 × 256 = 0.26   →   stable, phase margin > 45°

### Integral anti-windup note

The current implementation accumulates the integral unconditionally.  For large
initial deviations (cold start), windup can cause overshoot.  Operators can:
- Use anti-windup clamping (clip integral contribution)
- Start with a pre-computed initial pump power near P_th (from
  `find_threshold_pump_power()`) to minimise initial error magnitude

### Threshold sensitivity to field quality

        P_th ∝ 1/gain_budget

A field-quality degradation from GB=1.0 to GB=0.5 (doubles the inhomogeneous
broadening) doubles the required pump power to reach threshold.  This directly
couples the shimming controller (closed_loop.py) to the gain-lock: poor
shimming increases the minimum pump power, thermal load, and system power budget.

---

## Implementation Notes

```python
# Find the minimum laser power to sustain masing:
p_th = find_threshold_pump_power(nv_config, maser_config, cavity_config, pump_cfg)

# Simulate 100 steps of gain-lock recovery from cold start:
lock_cfg = GainLockConfig(kp=0.001, ki=1.0, target_cooperativity=1.10)
result = run_gain_lock_simulation(
    n_steps=200,
    nv_config=nv_config,
    maser_config=maser_config,
    cavity_config=cavity_config,
    pump_config_template=pump_cfg,
    lock_config=lock_cfg,
    initial_pump_power_w=p_th * 0.5,   # start at half threshold
)
print(f"Locked at step {result.locked_at_step} "
      f"({result.locked_at_us:.0f} µs)")
```

### Imports (no circular dependencies)

```
gain_lock.py
  → optical_pump.compute_pump_state()   (P → η_pump)
  → cavity.compute_full_threshold()     (η_pump → C)
  ← amplifier, q_boost                 (not imported; avoids circularity)
```

---

## Consequences

### Positive

- **R10 directly mitigated**: the digital twin can now simulate whether a
  proposed gain-lock design will stabilise the maser under pump noise and
  field disturbances before hardware is built.
- **Threshold calculator**: `find_threshold_pump_power()` gives the hardware
  team the pump budget needed for a given diamond and cavity quality.
- **Coupling to shimming**: the `gain_budget` parameter links field quality
  (shimming result) to gain margin, making the two subsystems composable.
- **Composable with RL**: the gain-lock can be wrapped as an inner loop
  inside `ProbeShimmingEnv` to improve physical realism in RL training.

### Negative

- **Integral windup**: no anti-windup clamp is implemented in V1; large
  initial deviations may overshoot the setpoint before settling.
- **Linearised gain analysis only**: the nonlinear C(P) curve means that gains
  that are stable near threshold may be unstable far from it.  A future V2
  could use gain scheduling.
- **No actuator deadtime model**: real laser drivers have response times
  (μs–ms); the control step dt_us should be set ≥ actuator response time.

---

## References

- Breeze et al. (2018) *Enhanced magnetic Purcell effect in room-temperature
  diamond*. Nature 555, 493.  
- Siegman, A. E. (1986) *Lasers* — Chapter 13, CW threshold and locking.  
- Åström, K. J. & Hägglund, T. (1995) *PID Controllers* — integral windup and  
  gain scheduling.  
- Handheld probe architecture doc §5 (System Architecture), §6 (Probe Head),
  §13 (Risk Register R10).  
- ADR-016: `q_boost.py` — complementary fast inner control loop candidate.
