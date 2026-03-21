# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added (SS18)

- `physics/mixer_nonlinearity.py` — Mixer IMD3 intermodulation distortion model
  (Risk R9 — nonlinear component).  Implements `IMD3Product` (frozen dataclass:
  interferer_1, interferer_2, product_type, product_freq_hz, is_physical,
  product_power_dbm, freq_offset_from_maser_hz, in_maser_band),
  `MixerNonlinearityConfig` (frozen dataclass: iip3_dbm = 5.0 dBm, maser_center_hz
  = 1.4699 GHz, maser_gain_bw_hz = 49 000 Hz, interferers = default 8-source
  hospital set), `MixerNonlinearityResult` (frozen dataclass: imd3_products,
  n_pairs_evaluated, n_products_evaluated, any_in_band, worst_in_band_power_dbm,
  in_band_products, max_imd3_power_dbm), `compute_imd3_power_dbm()` (two-tone
  formula: 2f₁−f₂ → 2P₁+P₂−2·IIP3; 2f₂−f₁ → P₁+2P₂−2·IIP3, all dBm),
  `compute_imd3_frequency_hz()` (exact frequency of each product), `compute_imd3_pair()`
  (evaluates both products for one interferer pair), and `compute_mixer_nonlinearity()`
  (top-level aggregator over all C(N,2) pairs).  Default config: 8 hospital-environment
  interferers, IIP3 = 5.0 dBm (matches `DEFAULT_MIXER.ip3_dbm` in `up_conversion.py`).
  Benchmark: 28 unordered pairs × 2 products = 56 IMD3 products evaluated; all 56
  fall > 75 MHz outside the 49 kHz maser gain window (`any_in_band = False`), closing
  R9 nonlinear distortion for the default hospital environment.  All 9 public symbols
  exported from `nv_maser.physics`.
- `docs/adr/ADR-022-mixer-imd3-nonlinearity-model.md` — Architecture decision
  record: two-tone IMD3 model for mixer R9 nonlinear distortion; validation table (56
  products, 0 in-band); distinguishes IMD3 mechanism from single-tone Lorentzian
  rejection (ADR-021); 3 alternatives considered (hardware pre-filter deferred,
  high-IIP3 mixer available as mitigation, OIP3-only bound rejected for inability
  to identify in-band products).

### Tests (SS18)

- `tests/test_mixer_nonlinearity.py` — 52 tests across 8 classes:
  `TestComputeImd3PowerDbm` (8): equal-tone reduces to 3P−2·IIP3, unequal tones both
  product types, stronger input raises power, higher IIP3 lowers power, 2:1 dominant-tone
  slope, 1:1 secondary-tone slope, bad product_type raises;
  `TestComputeImd3FrequencyHz` (6): exact formulas for both product types, equal
  frequencies, negative-frequency algebraically allowed, bad product_type raises,
  default product type;
  `TestComputeImd3Pair` (9): two products returned, type tags, interferer ordering,
  non-positive frequency is_physical=False, non-physical not in-band, exact in-band
  hit, out-of-band pair, freq_offset magnitude, power matches formula;
  `TestMixerNonlinearityConfig` (7): defaults, non-positive center/bw raises, custom
  interferers;
  `TestComputeMixerNonlinearityDefault` (9): type, n_pairs=28, n_products=56, no
  in-band products, worst=-inf, empty in-band tuple, finite max power, None config
  uses default, stronger dominant raises max;
  `TestComputeMixerNonlinearityInBand` (5): constructed near-maser pair flags in-band,
  non-empty in-band products, finite worst power, worst≤max, worst equals max-of-in-band;
  `TestTopologyPairCounting` (4): N=1/2/3 topology, no duplicate pairs;
  `TestImd3ProductFieldConsistency` (2): type tag matches frequency formula, power
  roundtrips with formula.
  Suite total: **1996 passed, 75 skipped, 10 warnings**.

### Added (SS17)

- `physics/rf_rejection.py` — RF interference rejection model (Risk R6 mitigation).
  Implements `InterfererSpec` (frozen dataclass: name, center_freq_hz, bandwidth_hz,
  power_dbm), `RFRejectionConfig` (frozen dataclass: maser_center_hz = 1.4699 GHz,
  maser_gain_bw_hz = 49 000 Hz, readout_bw_hz = 20 000 Hz, lo_freq_hz resolved from
  maser_center − 2.129 MHz, interferers = 8-source hospital-environment default),
  `InterfererResult` (frozen dataclass: interferer, freq_offset_hz, attenuation_db,
  residual_power_dbm, baseband_freq_hz, in_readout_band), `RFRejectionResult` (frozen
  dataclass: interferer_results, worst_case_residual_dbm, worst_case_name,
  any_in_readout_band, maser_fractional_bw, min_attenuation_db, max_attenuation_db),
  `compute_lorentzian_attenuation(freq_hz, center_hz, bw_hz)` (core physics:
  OOB_dB = 10 log₁₀[1 + (2Δf/BW)²]), `compute_fractional_bandwidth()` (BW/f₀),
  `compute_interferer_rejection()` (per-interferer analysis), and
  `compute_rf_rejection()` (primary entry point). Default 8 interferers span WiFi
  2.4 GHz, WiFi 5 GHz, Bluetooth 2.4 GHz, LTE 700/1800/2600 MHz, hospital
  Wi-Fi, and broadcast FM. Benchmark: maser fractional BW ≈ 3.33 × 10⁻⁵;
  worst-case residual ≈ −115 dBm (broadcast FM at −20 dBm input); no standard
  interferer maps into the ±10 kHz NMR readout band, closing R6 by passive physics.
  All 8 public symbols exported from `nv_maser.physics`.
- `docs/adr/ADR-021-rf-interference-rejection-model.md` — Architecture decision
  record: Lorentzian cavity rejection model for R6; computed attenuation table for
  8 interferers; 3 alternatives considered (Faraday cage rejected, passive filter
  and active cancellation deferred); quantitative justification that no additional
  shielding hardware is required.

### Tests (SS17)

- `tests/test_rf_rejection.py` — 55 tests across 6 classes:
  `TestLorentzianAttenuation` (10): on-resonance zero dB, −3 dB at half-power point,
  symmetry, monotonicity, per-source floor checks (WiFi >80 dB, 5 GHz >100 dB,
  FM >90 dB), asymptotic formula agreement, BW sweep; `TestFractionalBandwidth` (4):
  default value, positivity, monotonicity, custom config; `TestInterfererSpec` (7):
  valid construction, negative/zero freq/bw raises, negative power valid, frozen;
  `TestRFRejectionConfig` (7): LO resolution, explicit LO, 8 default interferers,
  validation raises, custom list; `TestComputeInterfererRejection` (12): field-by-field
  correctness, WiFi residual < −110 dBm, in-band detection, on-resonance no attenuation,
  interferer reference preserved; `TestComputeRFRejection` (15): aggregate correctness,
  all 8 defaults > 80 dB, none in readout band, worst-case identification/naming,
  fractional BW, fractional BW < 1e-4, min/max ordering, None config, custom/empty
  lists, residual < −100 dBm, wider BW less rejection, frozen result.
  Suite total: **1944 passed, 75 skipped, 10 warnings**.

### Added (SS16)

- `physics/gain_bandwidth_match.py` — Maser gain-bandwidth vs NMR readout
  matching model (Risk R2 mitigation).  Implements `GainBandwidthConfig`
  (frozen dataclass: cavity_q = 30 000, f0_hz = 1.4699 GHz, readout_bw_hz =
  20 kHz, b0_tesla = 50 mT, gyro_ratio_hz_t = 42.577 MHz/T),
  `BandwidthMatchResult` (frozen dataclass: maser_gain_bw_hz,
  readout_bw_hz, frequency_margin_hz, overlap_fraction,
  b0_drift_tolerance_t, b0_drift_tolerance_ppm, passes_criterion),
  `compute_maser_gain_bandwidth()` (BW = f₀/Q), `compute_b0_drift_tolerance()`
  (returns max B₀ drift before Larmor exits gain margin, as (T, ppm) tuple),
  `compute_bandwidth_match()` (primary entry point returning
  BandwidthMatchResult), `sweep_q_vs_gain_bandwidth()` (Q-sweep utility),
  and `sweep_b0_drift_vs_overlap()` (B₀-drift vs. overlap-fraction sweep).
  Nominal benchmark: BW_maser ≈ 49 kHz, one-sided margin ≈ 14.5 kHz,
  B₀ drift tolerance ≈ 340 µT / 6 810 ppm — far exceeding the 10 ppm
  B₀ stability required for NMR image resolution, quantitatively closing R2
  by design.  All 7 symbols exported from `nv_maser.physics`.
- `docs/adr/ADR-020-gain-bandwidth-match-model.md` — Architecture decision
  record: Lorentzian gain model, loaded-Q vs unloaded-Q note, benchmark table,
  key finding (R2 mitigated by design margin), positive / watch-point
  consequences, 3 alternatives considered.

### Tests (SS16)

- `tests/test_gain_bandwidth_match.py` — 40 tests across 7 classes:
  `TestGainBandwidthConfig` (4), `TestComputeMaserGainBandwidth` (8),
  `TestComputeB0DriftTolerance` (6), `TestComputeBandwidthMatch` (10),
  `TestSweepQVsGainBandwidth` (5), `TestSweepB0DriftVsOverlap` (6),
  `TestPublicAPI` (1).  Key assertions: BW = f₀/Q exactly; ValueError on
  non-positive Q or f₀; BW inversely proportional to Q; drift tolerance zero
  when readout exceeds gain BW; overlap fraction in [0, 1]; drift tolerance
  ≫ 100 ppm at nominal; zero-drift overlap matches static result; large drift
  clips to 0; monotone sweep behaviours; symmetry under sign of drift.
  Suite total: **1889 passed, 75 skipped, 10 warnings**.

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
