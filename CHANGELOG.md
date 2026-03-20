# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added (SS15)

- `physics/shielding.py` — Mu-metal magnetic shielding model (Risk R3 mitigation).
  Implements `MuMetalShellConfig` (frozen dataclass: inner radius, wall thickness,
  µ_r, shape, n_layers, interlayer gap, density), `ShieldingResult` (frozen dataclass:
  attenuation_linear, attenuation_db, residual_field_tesla, incident_field_tesla,
  shell_mass_kg, config), `compute_single_layer_attenuation()` (exact Jackson sphere
  or Rikitake cylinder formula), `compute_multilayer_attenuation()` (product law for n
  concentric shells), `compute_shell_mass_kg()` (sphere/cylinder volume formulas,
  including end-caps), `compute_shielding()` (primary entry point returning
  ShieldingResult), and `find_thickness_for_target_db()` (bisection sizing utility).
  Default parameters (15 mm inner radius, 1 mm wall, µ_r = 50 000, spherical, 1 layer)
  give ≈ 65.8 dB attenuation, reducing a 30 mT stray field to ~15 nT — exceeding the
  architecture §6.5 target of ≥ 50 dB.  Minimum thickness for 50 dB is ≈ 0.14 mm.
  All 7 symbols exported from `nv_maser.physics`.
- `docs/adr/ADR-019-mu-metal-shielding-model.md` — Full architecture decision record:
  physics background (Jackson §5.12 spherical formula, Rikitake cylindrical formula),
  benchmark values table, 3 alternatives considered (scalar function, FEA, encode in
  disturbance config), validity of product approximation for multi-layer shells,
  end-cap treatment, frequency-dependence note, chain usage example.

### Tests (SS15)

- `tests/test_shielding.py` — 39 tests across 7 classes:
  `TestMuMetalShellConfig` (4), `TestComputeSingleLayerAttenuation` (10),
  `TestComputeMultilayerAttenuation` (5), `TestComputeShellMassKg` (6),
  `TestComputeShielding` (7), `TestFindThicknessForTargetDb` (6),
  `TestPublicAPI` (1).  Key assertions: µ_r = 1 → S = 1 exactly; sphere > cylinder
  for equal dimensions; architecture 50 dB target exceeded; 30 mT stray → < 100 µT
  residual; bisection achieves target within ±0.1 dB; `ValueError` on invalid bracket.
  Suite total: **1849 passed, 75 skipped, 10 warnings**.

### Added (SS14)

- `physics/gain_lock.py` — Gain-lock PI control loop for maser threshold
  stabilisation (Risk R10 mitigation).  Implements `GainLockConfig` (frozen
  dataclass with K_p, K_i, setpoint, power bounds, noise sigma), `GainLockStep`
  (per-step snapshot), `GainLockResult` (simulation summary with lock time),
  `compute_cooperativity()` (maps laser power → ensemble C via optical pump
  saturation and `compute_full_threshold()`), `find_threshold_pump_power()`
  (bisection for C = 1), and `run_gain_lock_simulation()` (N-step PI loop with
  optional noise injection and reproducible RNG).  All 6 symbols exported from
  `nv_maser.physics`.  Default gains (K_p = 0.001 W/C, K_i = 1.0 W·s⁻¹/C)
  chosen for stability: loop gain K_p × dC/dP ≈ 0.26 < 1 near threshold.
- `docs/adr/ADR-018-gain-lock-pi-control-loop.md` — Full architecture decision
  record: physics background, PI stability analysis, 3 alternatives considered
  (output-power PI, frequency-tracking PLL, Q-boost inner loop), linearised
  gain derivation, integral windup discussion, code example.

### Added

- `rl/__init__.py` — public API exports for the RL sub-package:
  `ShimmingEnv`, `ProbeShimmingConfig`, `ProbeShimmingEnv`, `PPOConfig`,
  `PPOTrainer`, `ActorCritic`, `RolloutBuffer`, `compute_gae`,
  `load_ppo_controller`, `load_supervised_controller`,
  `validate_policy_closed_loop`
- `physics/single_sided_magnet.py` — Phase-2 milestone validation via
  `validate_sweet_spot_milestone()` and `MilestoneResult` dataclass.
  Checks B₀ = 50 ± 5 mT and < 500 ppm volumetric uniformity over a
  10 mm sphere; uses the 2D off-axis field map for accurate non-axial
  sampling (§12.2 criterion). Exported from `nv_maser.physics`.
- `rl/probe_env.py` (SS12) — `ProbeShimmingConfig` dataclass and
  `ProbeShimmingEnv(ShimmingEnv)` coupling imaging-magnet stray-field
  disturbance with maser-chain SNR into the RL shimming episode. Imaging
  magnet registered once at construction; `step()` enriches the info dict
  with `maser_noise_temp_k`, `probe_snr_db`, and `stray_field_rms_mt`.
  Optional `use_probe_reward` flag (default off) enables SNR-weighted
  reward shaping for future training runs.
- `physics/disturbance.py` (SS11.1) — `imaging_magnet_field` read-only
  property on `DisturbanceGenerator` for external inspection of the
  accumulated imaging-magnet stray field (returns `None` if unset).
- `physics/probe.py` (SS11.1) — `ProbeIntegration` module coupling the
  maser signal chain with the probe environment; exports
  `compute_probe_snr`, `ProbeMetrics`.

### Fixed

- `physics/reconstruction.py` — `soft_threshold()` divide-by-zero
  `RuntimeWarning` eliminated via `safe_mag` guard (`np.where(mag > 0,
  mag, 1.0)` before the division). Promotes NumPy from evaluating
  `0/0` in the inactive branch of `np.where`. 7 warnings removed
  (total suite warnings: 17 → 10).

### Tests

- 39 new tests in `tests/test_gain_lock.py`
  (`TestGainLockConfig` × 3, `TestGainLockStep` × 3,
  `TestComputeCooperativity` × 8, `TestFindThresholdPumpPower` × 5,
  `TestRunGainLockSimulation` × 14, `TestGainLockResult` × 3,
  `TestPublicAPI` × 1). Total suite: 1810 passed, 75 skipped, 10 warnings.
- 17 new tests in `tests/test_single_sided_magnet.py`
  (`TestMilestoneResult` × 5, `TestValidateSweetSpotMilestone` × 12).
- 37 new tests in `tests/test_probe_env.py` covering
  `ProbeShimmingConfig`, `ProbeShimmingEnv` construction / reset / step,
  reward shaping, space compatibility, episode mechanics, and physics
  correctness (shielding, magnet-size scaling). 1752 passed, 75 skipped.

## [0.7.0] - 2025-07-XX

### Added

- Multi-stage `Dockerfile` (`python:3.11-slim` builder + runtime layers) and `docker-compose.yml` with healthcheck and `checkpoints/` volume mount
- `/metrics` endpoint in Prometheus exposition format; thread-safe `_Metrics` class tracking `shim_requests_total`, error count, and rolling average latency
- Content-addressed dataset caching in `src/nv_maser/data/dataset.py` — SHA-256 hash of `grid + disturbance + field + coils` config, stored as `.npz`; `force_rebuild` flag; progress logging every 10 %
- `scripts/build_dataset.py` CLI entry point for offline dataset generation
- CI matrix expanded to Python 3.10 / 3.11 / 3.12 with `actions/cache` for pip; separate `lint` job (ruff) gating the `test` matrix
- `config/default.yaml` documenting all `SimConfig` defaults with inline comments
- 3 new metrics tests in `tests/test_api.py` (16 tests total)
- `tests/test_dataset.py` — 4 tests: shape validation, cache hit, hash invalidation, `force_rebuild`
- `.dockerignore` to exclude venv, checkpoints cache, and notebook output from build context

## [0.6.0] - 2025-07-XX

### Added

- Full README rewrite with architecture diagram (module tree), performance table, and FastAPI endpoint reference

### Fixed

- FastAPI security hardening: 1 MB request-body guard, dimension bounds check (`[2, 512]` each), NaN/Inf sanitisation before inference, CORS restricted to `localhost` origins only, `X-Content-Type-Options` / `X-Frame-Options` security-headers middleware
- Dead code `max_current * 0` expression removed from `scripts/train_rl.py`

### Security

- CORS `allow_origins` restricted to `["http://localhost", "http://127.0.0.1"]`; wildcard origin removed

## [0.5.0] - 2025-07-XX

### Added

- `benchmarks/benchmark_inference.py` — multi-architecture latency comparison for CNN, MLP, and LSTM controllers (all < 1 ms median on CPU)
- `scripts/train_rl.py` — REINFORCE policy-gradient baseline with `StochasticShimmingPolicy`, gradient clipping (`max_norm=1.0`), and episode logging
- `tests/test_rl_train.py` — 2 tests: single episode runs without error, policy gradient step updates parameters
- ADR-003: RL environment design (episode termination, reward shaping, action scale)
- ADR-004: Temporal controller design — rationale for LSTM over GRU/Transformer for real-time latency budget

## [0.4.0] - 2025-07-XX

### Added

- `LSTMController` architecture: CNN spatial extractor feeding a two-layer LSTM, hidden-state carry across inference calls
- `ShimmingEnv` — gymnasium-compatible RL environment without a hard `gymnasium` dependency; `reset()` / `step()` / `render()` interface
- `DisturbanceGenerator.randomize()` helper for stochastic episode initialisation
- `tests/test_api.py` — 10 endpoint tests using `httpx.AsyncClient` (health, shim valid input, shim error paths)
- LSTM checkpoint key extraction (`model_state`) aligned with supervised training output format

### Fixed

- `server.py` duplicate `/shim` route definition removed (caused silent shadowing)
- Checkpoint loading now extracts `model_state` sub-key from training dict; previously failed silently on new-format checkpoints

## [0.3.0] - 2025-07-XX

### Added

- `src/nv_maser/api/server.py` — FastAPI inference server with `/health` and `/shim` endpoints
- `uvicorn` entry point: `python -m nv_maser serve`
- Inference latency benchmark: 0.27 ms median end-to-end (CNN, batch-1, CPU)
- ADR-001: Controller architecture overview (CNN chosen as default; MLP as ablation; LSTM for temporal)
- ADR-002: Single-server deployment rationale (no reverse proxy for v0; revisit at production scale)

## [0.2.0] - 2025-07-XX

### Added

- GitHub Actions CI workflow (`.github/workflows/ci.yml`) running `pytest` + `ruff` on push and pull request to `main`
- `notebooks/exploration.ipynb` — interactive walkthrough of physics simulation, controller training, and field visualisation
- YAML deep-merge config overrides: `python -m nv_maser train --config my_overrides.yaml` merges on top of `config/default.yaml`
- `src/nv_maser/__main__.py` enabling `python -m nv_maser` entry point

## [0.1.0] - 2025-07-XX

### Added

- Core physics simulation: `NVCentreHamiltonian`, `FieldEnvironment`, `DisturbanceGenerator`, `ShimCoilArray` (Biot-Savart)
- `CNNController` and `MLPController` neural shimming controllers
- `Trainer` with AdamW optimiser, cosine/step LR scheduler, early stopping on validation loss plateau
- `FieldUniformityLoss` + current-penalty regulariser
- `Pydantic`-based `SimConfig` for all tunable simulation parameters
- PyQtGraph real-time field dashboard (`python -m nv_maser demo`)
- 51 passing pytest tests across physics, model, and training modules
