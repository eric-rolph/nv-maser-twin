# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added (SS28)

- **TAT squeezing** in `squeezing_dynamics.py` — two-axis twisting dynamics
  with Heisenberg-limited scaling (ξ² ∝ 1/N).  2 dataclasses + 4 functions:
  `tat_xi2_ideal`, `tat_optimal_time`, `compute_tat_ideal_trajectory`,
  `compute_tat_with_decoherence`.  Closes ADR-013 deferred item "TAT".
- **Biexponential T2 fit** in `t1t2_estimator.py` — multi-pool T2 model
  with AIC-based selection: `BiexpT2FitResult`, `fit_t2_biexponential`,
  `select_t2_model`, `_compute_aic`.  Closes ADR-007 deferred item.
- **Allan deviation τ scaling** in `stability.py` — white PM (slope −1),
  flicker FM (flat), random-walk FM (slope +0.5), and RSS combination:
  `NoiseProcessADEV`, `CombinedADEVResult`, `compute_white_pm_adev`,
  `compute_flicker_fm_adev`, `compute_random_walk_fm_adev`,
  `compute_combined_allan_deviation`.  Closes ADR-012 deferred item.
- 44 new tests across 3 test files (17 TAT, 12 biexp, 15 ADEV).
- ADR: `docs/adr/ADR-032-deferred-items-batch2.md`.

### Added (SS27)

- `src/nv_maser/physics/squeezing_dynamics.py` — OAT squeezing dynamics with
  T₂* decoherence overlay (André & Lukin 2002 model).  7 public functions,
  3 dataclasses.  Closes ADR-013 deferred item "decoherence during squeezing".
- `tests/test_squeezing_dynamics.py` — 49 tests for the new module.
- `BlandAltmanResult`, `BlandAltmanT1T2`, `bland_altman_t1t2()` in
  `t1t2_estimator.py` — Bland–Altman agreement analysis for clinical
  validation of T1/T2 maps.  Closes ADR-007 deferred item.
- 10 new tests in `TestBlandAltmanT1T2` class in `test_t1t2_estimator.py`.
- ADR: `docs/adr/ADR-031-deferred-items-batch1.md`.

### Changed (SS27)

- `surface_coil.py`: replaced hardcoded 1 pF parasitic capacitance with
  interwinding capacitance model for self-resonant frequency estimation.
  Default coil SRF now ~387 MHz (was ~5 GHz with the old placeholder).

### Added (SS26)

- `tests/test_main.py` — CLI entry-point unit tests; lifts `main.py` from
  0 % to **94 % branch coverage** (7 non-testable lines remain: `cmd_demo`
  GUI body + `__main__` guard).  49 tests across 10 classes:
  `TestDeepMergeConfig` (8 — pure-function tests for `_deep_merge_config`
  including flat/nested overrides and immutability), `TestMainDispatch` (6 —
  argparse routing for `train`, `evaluate`, `visualize-coils`, `dataset`,
  `export`, `serve`), `TestMainConfigOverrides` (5 — `--device`, `--arch`,
  `--epochs`, `--samples` flag application), `TestMainYAMLConfig` (4 —
  valid flat/nested YAML overrides and Pydantic `ValidationError → SystemExit(1)`
  path), `TestCmdTrain` (3 — Trainer construction, `train()` call,
  `plot_training_history` call), `TestCmdEvaluate` (4 — `load_best` /
  `eval` calls, `generate_training_data(500)`, stdout output verification
  with real tiny torch tensors 500 × 4 × 4), `TestCmdDataset` (6 —
  `build_dataset` call, `num_samples` precedence logic, `force_rebuild`,
  stdout), `TestCmdExport` (5 — `export_model` dispatch, custom
  `output_path/checkpoint/opset`), `TestCmdServe` (3 — `uvicorn.run`
  call with default and custom host/port), `TestCmdVisualizeCoils` (5 —
  `FieldEnvironment`, `plot_coil_influence`, `plot_disturbance_spectrum`,
  `plt.show`, `disturbance_gen.generate`).
  ADR: `docs/adr/ADR-030-main-cli-coverage.md`.
  Tests: `tests/test_main.py` (49 tests).

### Added (SS25)

- `physics/phase1_validator.py` — Phase-1 'maser oscillation' milestone
  validator; formally validates architecture §12.2 criterion "Detectable
  stimulated emission at 1.47 GHz".  Completes the full set of six §12.2
  milestone validators (Maser oscillation, Sweet-spot, Depth profile, 2D
  image, Tissue contrast).  Implements `Phase1Config` (frozen dataclass:
  `target_frequency_ghz=1.47`, `frequency_tolerance_mhz=50.0`,
  `min_output_power_dbm=-100.0`, `gain_budget=0.5`),
  `OscillationThresholdResult` (frozen: `magnetic_q`, `loaded_q`, `q_ratio`,
  `above_threshold_wang`, `cooperativity`, `threshold_margin`, `masing`,
  `spin_temperature_k`), `Phase1MilestoneResult` (frozen: `oscillation`,
  `frequency_ghz`, `target_frequency_ghz`, `frequency_deviation_mhz`,
  `frequency_tolerance_mhz`, `output_power_w`, `output_power_dbm`,
  `min_output_power_dbm`, `threshold_met`, `frequency_met`, `power_met`,
  `phase1_milestone_closed`, `closing_message`).  Top-level:
  `validate_phase1_milestone`.  Three criteria: (1) oscillation threshold
  assessed via *two* independent frameworks — Wang 2024 magnetic-Q
  (Q_m/Q_L ≈ 0.779 < 1) and Breeze cooperativity (C ≈ 2.57 > 1); (2)
  cavity frequency within ±50 MHz of 1.47 GHz; (3) CW output power ≥
  −100 dBm (default ≈ −69.5 dBm).  All three pass by default →
  `phase1_milestone_closed=True`.
  ADR: `docs/adr/ADR-029-phase1-maser-oscillation-milestone.md`.
  Tests: `tests/test_phase1_validator.py` (69 tests).

### Added (SS24)

- `physics/phase9_validator.py` — Phase-9 'tissue contrast' milestone
  validator; formally validates architecture §12.2 criterion "T2 difference
  visible between fat and muscle".  Implements `Phase9Config` (frozen
  dataclass: `te_short_ms=10.0`, `te_long_ms=80.0`,
  `t2_contrast_ratio_threshold=1.5`, `snr_threshold=3.0`,
  `scan_time_limit_s=120.0`, `fat_center_depth_mm=4.5`,
  `muscle_center_depth_mm=10.0`), `T2ContrastResult` (frozen: `fat_depth_mm`,
  `muscle_depth_mm`, `fat_signal_short`, `muscle_signal_short`,
  `fat_signal_long`, `muscle_signal_long`, `t2_contrast_ratio`, `passes`),
  `Phase9MilestoneResult` (frozen: `config`, `profile_short`, `profile_long`,
  `t2_contrast`, `fat_snr`, `muscle_snr`, `contrast_pass`, `snr_pass`,
  `scan_time_s`, `scan_time_pass`, `phase9_milestone_closed`).  Top-level:
  `validate_phase9_milestone`.  Design: SNR criterion evaluated at short TE
  (10 ms, both tissues bright); T2 contrast ratio evaluated at long TE
  (80 ms, fat/muscle ≈ 19×) so the depth-dependent coil-sensitivity is
  cancelled.  Default outcome on FOREARM_LAYERS phantom: T2 contrast ratio
  ≈ 19× (> 1.5× ✓), fat SNR ≈ 20.3 (> 3.0 ✓), muscle SNR ≈ 3.3 (> 3.0 ✓),
  scan time ≈ 102 s (< 120 s ✓) — `phase9_milestone_closed=True`.
  ADR: `docs/adr/ADR-028-phase9-tissue-contrast-milestone.md`.
  Tests: `tests/test_phase9_validator.py` (67 tests).

### Added (SS23)

- `physics/phase6_validator.py` — Phase-6 '2D image' milestone validator;
  formally validates architecture §12.2 criterion "Grid phantom resolved at
  3 mm".  Implements `Phase6Config` (frozen dataclass: `bar_width_mm=3.0`,
  `fov_m=0.064`, `grid_size=64`, `n_spokes=64`, `n_readout=64`,
  `kernel_width=3.0`, `bar_contrast_threshold=0.2`,
  `image_snr_threshold_db=5.0`, `resolution_threshold_mm=3.0`),
  `GridPhantomResult` (frozen: `phantom`, `pixel_size_mm`, `n_bar_pairs`,
  `bar_width_mm`), `BarContrastResult` (frozen: `bar_mean`, `gap_mean`,
  `michelson_contrast`, `passes`), `Phase6MilestoneResult` (frozen: `config`,
  `phantom_result`, `recon`, `psf_fwhm_mm`, `bar_contrast`, `image_snr_db`,
  `pixel_size_mm`, `psf_pass`, `contrast_pass`, `snr_pass`,
  `phase6_milestone_closed`).  Helpers: `_make_bar_phantom`,
  `_reconstruct_from_phantom`, `_measure_psf_fwhm`, `_measure_bar_contrast`.
  Top-level: `validate_phase6_milestone`.  Forward model reuses
  `_noncartesian_dft` from `artifact_characterizer.py` (same validated path
  as R8 risk closure).  Default outcome (FOV=6.4 cm, 64×64, 3 mm bars,
  1.0 mm pixel): PSF FWHM ≈ 1.79 mm (< 3.0 mm ✓), Michelson contrast ≈ 0.32
  (> 0.2 ✓), image SNR ≈ 65 dB (> 5 dB ✓) — `phase6_milestone_closed=True`.
  ADR: `docs/adr/ADR-027-phase6-2d-image-milestone.md`.
  Tests: `tests/test_phase6_validator.py` (81 tests).

### Added (SS22)

- `physics/phase4_validator.py` — Phase-4 depth-profile milestone validator;
  formally validates architecture §12.2 criteria "First NMR signal" and
  "Resolvable layers in layered phantom".  Implements `Phase4Config` (frozen
  dataclass: `signal_snr_threshold=3.0`, `contrast_ratio_threshold=1.5`,
  `snr_at_boundary_threshold=1.0`, `scan_time_limit_s=120.0`,
  `depth_range_mm=(3.0, 15.0)`), `LayerContrastResult` (frozen: `layer_a_name`,
  `layer_b_name`, `t2_a_ms`, `t2_b_ms`, `t2_contrast_ratio`,
  `boundary_depth_mm`, `snr_at_boundary`, `in_depth_range`, `detectable`),
  `Phase4MilestoneResult` (frozen: `depth_profile`, `layer_contrasts`,
  `n_depths_evaluated`, `max_snr_in_range`, `min_snr_in_range`, `scan_time_s`,
  `all_in_range_layers_detectable`, `snr_pass`, `scan_time_pass`,
  `phase4_milestone_closed`).  Functions: `compute_layer_contrast`,
  `validate_phase4_milestone`.  Default outcome on FOREARM_LAYERS phantom:
  `phase4_milestone_closed=True` — fat→muscle boundary at 7 mm T2 ratio 2.29
  (> 1.5 ✓), SNR at boundary 16.44 (> 1.0 ✓), max SNR in range 20.75
  (> 3.0 ✓), scan_time 102.4 s (< 120 ✓).  ADR:
  `docs/adr/ADR-026-phase4-depth-profile-milestone.md`.
  Tests: `tests/test_phase4_validator.py` (71 tests).

### Added (SS21)

- `physics/artifact_characterizer.py` — Reconstruction artifact characterisation
  model (Risk R8 — reconstruction artifacts from non-Cartesian k-space,
  **confirmed closed**).  Implements `ArtifactConfig` (frozen dataclass:
  `grid_size=64`, `fov_m=0.08`, `kernel_width=3.0`, `trajectory="radial"`,
  `n_spokes=64`, `n_readout=64`, `v1_asr_threshold=0.05`,
  `v1_overshoot_frac=0.09`, `v1_psf_fwhm_ratio=3.0`), `PSFResult` (frozen:
  `fwhm_x_mm`, `fwhm_y_mm`, `ideal_fwhm_mm`, `fwhm_ratio`, `peak_value`,
  `within_spec`), `AliasingResult` (frozen: `asr`, `asr_db`, `within_spec`),
  `RingingResult` (frozen: `overshoot_fraction`, `undershoot_fraction`,
  `within_spec`), `ArtifactResult` (frozen: all three sub-results plus
  `pixel_size_mm`, `trajectory_n_samples`, `cartesian_reference_snr_db`,
  `gridding_snr_db`, `snr_loss_gridding_db`, `r8_risk_closed`).  Functions:
  `generate_radial_trajectory`, `generate_spiral_trajectory`, `make_phantom`,
  `compute_psf`, `compute_aliasing`, `compute_ringing`,
  `compute_artifact_characterization`.  Uses an exact non-Cartesian DFT
  (`_noncartesian_dft`, O(M × Ny × Nx), pure NumPy) as the forward model.
  R8 closure: default 64 × 64 radial config gives PSF FWHM ratio 1.79×
  (< 3.0 threshold), aliasing ASR 0.48 % (< 5 %), Gibbs overshoot 1.58 %
  (< 9 %) — all V1 thresholds met; `r8_risk_closed = True`.
  ADR: `docs/adr/ADR-025-reconstruction-artifact-characterisation.md`.
  Tests: `tests/test_artifact_characterizer.py` (59 tests).

### Added (SS20)

- `physics/field_tolerance_calculator.py` — B₀ field strength and homogeneity
  tolerance model (Risk R1 — sweet-spot magnet field too weak, **confirmed
  closed**).  Implements `FieldToleranceConfig` (frozen dataclass: `b0_nominal_t=0.050`,
  `b0_sweep_min_t=0.030`, `b0_sweep_max_t=0.070`, `n_b0_sweep=21`,
  `uniformity_sweep_min_ppm=10`, `uniformity_sweep_max_ppm=5000`,
  `n_uniformity_sweep=20`, `te_fid_us=100`, `t2_tissue_ms=50`,
  `maser_bandwidth_hz=49000`, `v1_b0_tolerance_t=0.005`,
  `v1_uniformity_ppm=500`), `B0SensitivityPoint` (frozen: `b0_tesla`, `b0_mT`,
  `b0_deviation_pct`, `polarization_factor`, `signal_frequency_factor`,
  `snr_factor`, `snr_loss_db`, `larmor_frequency_hz`), `HomogeneityPoint`
  (frozen: `uniformity_ppm`, `delta_b0_mt`, `delta_frequency_hz`,
  `t2star_inhom_ms`, `t2star_eff_ms`, `snr_loss_fid_factor`,
  `snr_loss_fid_db`, `within_maser_bandwidth`), `FieldToleranceResult` (frozen:
  both sweep tuples, `b0_3db_loss_t`, `b0_1db_loss_t`,
  `uniformity_3db_fid_loss_ppm`, `uniformity_1db_fid_loss_ppm`,
  `uniformity_maser_limit_ppm`, `v1_snr_loss_at_b0_min_db`,
  `v1_spectral_bandwidth_at_spec_hz`, `v1_maser_bandwidth_margin_hz`,
  `r1_risk_closed`).  Functions: `compute_b0_sensitivity_point`,
  `compute_homogeneity_point`, `sweep_b0_sensitivity`, `sweep_homogeneity`,
  `compute_field_tolerance`.  R1 closure: worst-case SNR loss at 45 mT is
  1.84 dB (< 3 dB); 500 ppm spectral bandwidth ≈ 1 064 Hz vs 24 500 Hz maser
  half-BW (23× margin); `r1_risk_closed = True` with default config.
- ADR-024: documents the B₀ sensitivity (SNR ∝ B₀²), FID T2* dephasing,
  spectral-bandwidth model, and R1 closure rationale.
- `tests/test_field_tolerance_calculator.py` — 52 tests (T01–T52).
- **Test count**: 2127 passed (baseline SS19: 2075).

### Added (SS19)

- `physics/depth_limit_calculator.py` — Scan-time-gated depth-limit model
  (Risk R7 — SNR insufficient at depth).  Implements `DepthLimitConfig`
  (frozen dataclass: `target_snr=5.0`, `scan_time_budget_s=120.0`,
  `voxel_size_mm=3.0`, `tr_ms=100.0`, `te_ms=10.0`, `sequence="spin_echo"`,
  `bandwidth_hz=10_000.0`, `depth_step_mm=1.0`, range [1, 30] mm; validated
  in `__post_init__`), `DepthPoint` (frozen dataclass: `depth_mm`,
  `snr_per_shot`, `required_averages`, `scan_time_s`, `within_budget`),
  `DepthLimitResult` (frozen dataclass: `depth_profile`, `max_depth_mm`,
  `n_depths_evaluated`, `scan_time_budget_s`, `target_snr`,
  `snr_per_shot_at_limit`, `required_averages_at_limit`,
  `scan_time_at_limit_s`, `any_feasible`, `v1_depth_range_confirmed`,
  `snr_per_shot_at_5mm/10mm/15mm`), `compute_depth_point()` (calls
  `compute_snr_budget(n_averages=1)` then applies
  `required_averages = ceil((target_snr / snr_per_shot)²)` and
  `scan_time_s = required_averages × TR_ms / 1000`), and
  `compute_depth_limit()` (sweeps depth range with integer-step indexing,
  identifies deepest feasible depth, sets `v1_depth_range_confirmed = True`
  iff all depths in [5, 15] mm are within the scan-time budget).
  Architecture context: single-shot SNR at 20 mm ≈ 10⁻⁶ (coil-noise
  dominated at 300 K); maser advantage ~1 dB; V1 design intent 5–15 mm.
  All 5 public symbols exported from `nv_maser.physics`.
- `docs/adr/ADR-023-depth-limit-snr-budget-analysis.md` — Architecture
  decision record: scan-time-budget approach to characterise R7 depth
  limitation; validates V1 5–15 mm operating range; 3 alternatives
  considered (SNR contour plot, budget parameter on existing function,
  analytical inversion — all rejected); R7 status closed.

### Tests (SS19)

- `tests/test_depth_limit_calculator.py` — 48 tests across 11 classes:
  `TestDepthLimitConfigDefaults` (11): all 10 default field values verified,
  frozen check; `TestDepthLimitConfigValidation` (10): `__post_init__` raises
  `ValueError` for negative/zero `target_snr`, `scan_time_budget_s`,
  `voxel_size_mm`, `tr_ms`, `depth_step_mm`, and `min_depth_mm ≥ max_depth_mm`;
  `TestDepthPointFields` (6): field presence and frozen property;
  `TestComputeDepthPoint` (10): returns `DepthPoint`, depth echoed, SNR > 0,
  averages ≥ 1, `scan_time_s = averages × TR`, `within_budget` flag correct
  for large/tiny budgets, consistency check, deeper requires more averages;
  `TestComputeDepthLimit` (15): returns `DepthLimitResult`, sweep length,
  profile length matches, depths increasing, max_depth is feasible, max_depth
  is largest feasible, `any_feasible` with large/tiny budget, `max_depth=0`
  when infeasible, budget/target echoed, `None` config uses defaults, frozen
  result, `scan_time_at_limit ≤ budget`, first/last profile depths equal
  min/max; `TestDepthSNRMonotonicity` (5): per-shot SNR strictly decreasing,
  required_averages non-decreasing, scan_time non-decreasing, all SNR > 0,
  infeasibility propagates monotonically; `TestScanTimeBudgetSensitivity` (3):
  larger budget → deeper or equal max_depth, smaller budget → shallower,
  scan_time_at_limit ≤ budget for 3 representative budgets;
  `TestTargetSNRSensitivity` (4): higher target → shallower or equal,
  lower target → deeper or equal, target echoed, required averages increase
  with target; `TestV1RangeConfirmation` (4): confirmed with enormous budget,
  not confirmed with microscopic budget, consistent with per-point profile,
  correct when 15 mm is outside evaluated range; `TestReferenceDepths` (7):
  5/10/15 mm SNR present in default profile, 5 mm > 10 mm > 15 mm ordering,
  values match depth profile, None when depth not in profile;
  `TestArchitectureValidation` (3): per-shot SNR < 1 at all depths ≥ 5 mm,
  required averages at 20 mm >> at 10 mm, default profile has 30 points.

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
