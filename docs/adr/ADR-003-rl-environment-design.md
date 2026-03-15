# ADR-003: RL Environment Design

**Status:** Accepted  
**Date:** 2026-03-14  
**Deciders:** AI Engineer, Software Architect

---

## Context

The shimming policy must be trained via reinforcement learning. This requires an environment that presents the agent with observed field maps, accepts correction-current actions, and returns a scalar reward signal. Three approaches were considered:

1. **`gymnasium` library** — subclass `gymnasium.Env`; inherits standard reset/step/render contracts and ecosystem tooling (stable-baselines3, RLlib).
2. **Standalone gym-compatible interface** — implement the same 5-tuple step API without taking `gymnasium` as a runtime dependency.
3. **Purely functional interface** — `reset(state)` / `step(state, action)` returning plain tuples; no class or OOP.

The environment wraps the existing `DisturbanceGenerator` and `FieldUniformityLoss` and must be usable in both the RL training loop (`scripts/train_rl.py`) and interactive notebooks without installing heavyweight RL frameworks.

---

## Decision

**Standalone gym-compatible implementation (option 2).**

`MagneticShimmingEnv` is a plain Python class that exposes exactly the interface required by modern gym 0.26+ / gymnasium 0.26+:

```python
obs, info           = env.reset(seed=N)
obs, reward, terminated, truncated, info = env.step(action)
```

No `gymnasium` import is required at runtime. Consuming code that needs `gymnasium.Env` compatibility can wrap the class with a one-line adapter.

### Key design choices

| Design parameter | Choice | Rationale |
|-----------------|--------|-----------|
| Observation shape | `(1, grid_size, grid_size)` float32 | Channel-first; zero-copy compatible with CNN/LSTM/MLP inputs |
| Action space | `(num_coils,)` float32 ∈ `[−max_current, max_current]` | Matches controller output range; continuous for gradient methods |
| Reward | `distorted_var − corrected_var` | Positive when correction reduces field non-uniformity; zero for no-op; negative when correction worsens field |
| Episode length | 200 steps (configurable `max_steps`) | Covers typical disturbance evolution timescale; prevents infinite loops |
| Reset reproducibility | `reset(seed=N)` seeds internal RNG and `DisturbanceGenerator` | Enables deterministic evaluation runs and CI smoke tests |

---

## Consequences

### Positive
- **No extra runtime dependency**: `gymnasium` and `stable-baselines3` are development/research packages. Removing them from the production dependency graph reduces Docker image size and avoids version-pinning conflicts with PyTorch.
- **Transparent reward signal**: The improvement-based reward (`Δvariance`) has a clear physical interpretation — it directly measures whether the agent's action reduced field non-uniformity — making reward hacking detectable by inspection.
- **Drop-in compatibility**: Any RL library that accepts a duck-typed env (SB3, CleanRL, custom loops) works without wrapping.
- **Reproducible evaluation**: Seeded reset enables fixed evaluation suites for benchmark comparisons across policy checkpoints.

### Negative / Trade-offs
- **No vectorised env support out of the box**: `gymnasium.vector.VectorEnv` parallelism is unavailable. Parallel data collection requires either a custom multiprocessing wrapper or adopting Ray RLlib.
- **Manual spec maintenance**: Without inheriting `gymnasium.Env`, observation/action space descriptors (`Box`, `spaces`) are not enforced by a base class. Shape mismatches surface at training time rather than import time.

---

## Alternatives Considered

### `gymnasium.Env` subclass (option 1)
- Enables direct use of stable-baselines3, RLlib, and the `gymnasium.make` registry.
- **Con**: Adds `gymnasium` as a non-optional runtime dependency; version pinning conflicts with recent PyTorch / NumPy releases have been observed in CI.
- **Decision**: Deferred; reintroduce if a specific SB3 algorithm is adopted for the production policy.

### Purely functional interface (option 3)
- Eliminates class state entirely; each call is a pure function of `(state, action)`.
- **Con**: Tracking episode step count, RNG state, and accumulated metrics requires the caller to manage mutable state, shifting complexity to `train_rl.py` and notebooks.
- **Decision**: Rejected; the marginal FP purity gain does not justify the ergonomic cost.

---

## References
- `src/nv_maser/env.py` — `MagneticShimmingEnv`
- `src/nv_maser/simulator.py` — `DisturbanceGenerator`
- `src/nv_maser/model/loss.py` — `FieldUniformityLoss`
- `scripts/train_rl.py` — RL training loop
