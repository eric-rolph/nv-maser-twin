# ADR-017 — Probe-Aware RL Shimming Environment (ProbeShimmingEnv)

**Status:** Accepted  
**Date:** 2026-03-19  
**Module:** `src/nv_maser/rl/probe_env.py`  
**Tests:** `tests/test_probe_env.py` (37 tests)

---

## Context

The base `ShimmingEnv` environment (ADR-003) optimises for B₀ field uniformity over
the maser sensing volume but is unaware of:

1. **Imaging-magnet stray field** — the handheld probe's imaging magnet (§6.5,
   §13 R3) produces a spatially non-uniform field that contaminates the maser's
   holding field, destabilising oscillation.
2. **Maser-chain SNR** — the clinical utility of the probe depends on the
   end-to-end probe SNR (`snr_db`, `maser_noise_temperature_k`) computed by
   `environment.py::compute_uniformity_metric()`, not just the RMS uniformity
   scalar that drives the RL reward.

`ShimmingEnv` episodes expose a `probe_snr_db` reward component in neither the
observation vector nor the reward signal, making it impossible to train policies
that balance field uniformity against probe SNR.

A coupling layer is needed that:
- registers the imaging magnet disturbance *once* (its stray field is a physical
  constant of the hardware design, not an episode variable);
- surfaces `maser_noise_temperature_k` and `probe_snr_db` in the info dict at
  every step (for offline analysis and curriculum learning);
- provides an *optional* SNR-weighted reward shaping term that can be switched on
  for future training experiments without breaking existing baselines.

---

## Decision

Extend `ShimmingEnv` with a thin subclass `ProbeShimmingEnv` governed by a
`ProbeShimmingConfig` dataclass.  The subclass:

1. Accepts a `ProbeShimmingConfig` alongside the existing `SimConfig`.
2. Calls `self._disturbance.add_imaging_magnet(cfg.imaging_magnet)` **once** in
   `__init__()`, caching the stray-field RMS for later reporting.
3. Overrides `step()` to call `super().step()`, then query
   `compute_uniformity_metric()` for physics metrics, and append three keys to the
   info dict:
   - `"maser_noise_temp_k"` — maser noise temperature (K)
   - `"probe_snr_db"` — probe SNR (dB)
   - `"stray_field_rms_mt"` — imaging magnet RMS stray field on the maser grid (mT)
4. When `use_probe_reward=True`, adds `probe_snr_weight × snr_db` to the reward.

---

## Alternatives Considered

### A. Add SNR to `ShimmingEnv` directly

**Rejected.** `ShimmingEnv` is a well-tested baseline used by multiple downstream
tests and training scripts.  Injecting imaging-magnet coupling into it would break
backward compatibility.  A subclass is strictly additive.

### B. Modify the disturbance per episode in `reset()`

**Rejected.** The imaging magnet is a fixed hardware component.  Registering it in
`reset()` would clear the stray field on every episode re-seed and incorrectly make
`stray_field_rms_mt` variable.  The stray field changes only when the hardware
design changes, not episode-by-episode.

### C. Separate wrapper / `gym.Wrapper`

**Considered.** A `gym.Wrapper` would give the cleanest separation of concerns.
However `ShimmingEnv` is not a strict `gym.Env` (it wraps an internal `_env`
object), so a `Wrapper` would require re-exposing `num_coils`, `max_steps`, and the
full `step` / `reset` protocol.  A subclass is simpler and requires only one
`__init__` + one `step` override — 50 lines vs. 150+ for a wrapper.

---

## Physical Model

### Imaging-magnet disturbance

`add_imaging_magnet(cfg)` accumulates a 2-D stray-field array on the maser grid
(shape `(size, size)`) using a dipole model parameterised by:

| Parameter | Symbol | Default |
|-----------|--------|---------|
| Distance from maser (mm) | d | 80 mm |
| Dipole moment (A·m²) | m | 10.0 |
| Orientation (polar, azimuthal) | θ, φ | 0°, 0° |

### Physics coupling chain

```
field uniformity (σ/B₀)
      ↓
  T₂* broadening           T₂* ∝ (σ/B₀)⁻¹
      ↓
maser Q_m reduction         Q_m ∝ T₂*
      ↓
maser noise temperature     T_a = ħω / k_B × (1−G)/G
      ↓
probe SNR                   snr_db = 10 log₁₀(S/k_B T_sys Δν)
```

All four quantities are computed by `environment.py::compute_uniformity_metric()`
and surfaced in the step info dict.

### Reward shaping (optional)

With `use_probe_reward=True`:

```
r_total = r_base + w_snr × snr_db
```

where `r_base` is the base `ShimmingEnv` reward (uniformity improvement), `w_snr`
is `probe_snr_weight` (default 0.1), and `snr_db` is in dB.  

The default `use_probe_reward=False` leaves `r_total = r_base`, preserving full
backward compatibility with policies trained on `ShimmingEnv`.

---

## Consequences

### Positive

- **Zero backward-compatibility risk.** `ShimmingEnv` is unchanged.  Existing
  training scripts continue to work without modification.
- **Richer telemetry.** Every PPO rollout now records maser noise temperature and
  probe SNR for offline analysis.
- **Foundation for curriculum learning.** Future work can condition the reward
  weight `w_snr` on training progress (e.g., warm-start with uniformity-only,
  then graduate to SNR-shaping).
- **Stray field cached at construction.** Hardware engineers can inspect
  `stray_field_rms_mt` to verify that the shielding budget is met without running
  an episode.

### Negative / Risks

- **`compute_uniformity_metric()` is called at every step.** This adds ≈ 0.1 ms
  per step on CPU.  For short episodes (≤ 200 steps) this is < 20 ms per episode —
  acceptable.  If profiling shows this is a hot path, it can be replaced with a
  cached linear approximation.
- **`ProbeShimmingEnv` assumes axial symmetry** in the stray-field model.  Off-axis
  probe tilts are not modelled until `ImagingMagnetDisturbanceConfig` gains an
  angular parameter.

---

## Implementation Notes

```python
# Construction: register magnet once
self._disturbance.add_imaging_magnet(self._probe_config.imaging_magnet)
imf = self._disturbance.imaging_magnet_field          # shape (size, size)
self._stray_field_rms_mt = float(np.sqrt(np.mean(imf**2))) * 1e3

# Step: augment info dict
phys = self._env.compute_uniformity_metric(self._current_field)
info["maser_noise_temp_k"]  = float(phys.get("maser_noise_temperature_k", math.nan))
info["probe_snr_db"]        = float(phys.get("snr_db", math.nan))
info["stray_field_rms_mt"]  = self._stray_field_rms_mt
```

---

## References

- Architecture doc §6.5 — Magnetic isolation between imaging magnet and maser
- Architecture doc §12.2 — Critical milestones (stray-field mitigation)
- Architecture doc §13 — Risk R3: stray field destabilises maser
- ADR-003 — RL environment design (base ShimmingEnv)
- `src/nv_maser/physics/disturbance.py` — `add_imaging_magnet()`, `imaging_magnet_field`
- `src/nv_maser/physics/environment.py` — `compute_uniformity_metric()`
