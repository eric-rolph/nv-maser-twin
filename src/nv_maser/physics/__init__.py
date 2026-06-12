"""Physics simulation modules for the NV Maser Digital Twin."""

from .amplifier import (  # noqa: F401
    SIGMA_2,
    AmplifierProperties,
    MaserGainResult,
    OutputPowerResult,
    compute_amplifier_properties,
    compute_magnetic_q,
    compute_maser_gain,
    compute_output_power,
    compute_spin_temperature,
    compute_sql_noise_temperature,
)
from .amplifier import (  # noqa: F401
    compute_noise_temperature as compute_amplifier_noise_temperature,
)
from .artifact_characterizer import (  # noqa: F401
    AliasingResult,
    ArtifactConfig,
    ArtifactResult,
    PSFResult,
    RingingResult,
    compute_aliasing,
    compute_artifact_characterization,
    compute_psf,
    compute_ringing,
    generate_radial_trajectory,
    generate_spiral_trajectory,
    make_phantom,
)
from .cavity import (  # noqa: F401
    CavityProperties,
    ThresholdResult,
    compute_cavity_properties,
    compute_effective_q,
    compute_full_threshold,
    compute_maser_threshold,
    compute_n_effective,
    compute_spectral_overlap,
)
from .closed_loop import ClosedLoopResult, ClosedLoopSimulator  # noqa: F401
from .depth_limit_calculator import (  # noqa: F401
    DepthLimitConfig,
    DepthLimitResult,
    DepthPoint,
    compute_depth_limit,
    compute_depth_point,
)
from .depth_profile import (  # noqa: F401
    FOREARM_LAYERS,
    HEMORRHAGE_LAYERS,
    DepthProfile,
    TissueLayer,
    add_noise,
    simulate_depth_profile,
)
from .dipolar import (  # noqa: F401
    apply_dipolar_refilling,
    estimate_dipolar_coupling_hz,
    estimate_refilling_time_us,
    spectral_diffusion_step,
    stretched_exponential_refill,
)
from .disturbance import (  # noqa: F401
    DisturbanceGenerator,
    ImagingMagnetDisturbanceConfig,
    compute_imaging_magnet_stray_field,
)
from .environment import UniformityReport  # noqa: F401
from .epg_adapter import (  # noqa: F401
    EPGDepthProfile,
    EPGResult,
    EPGValidation,
    cross_validate_epg_vs_analytical,
    epg_cpmg,
    epg_depth_profile,
    epg_signal,
)
from .feedback import CoilDynamics, HallSensorArray, quantize_currents  # noqa: F401
from .field_tolerance_calculator import (  # noqa: F401
    B0SensitivityPoint,
    FieldToleranceConfig,
    FieldToleranceResult,
    HomogeneityPoint,
    compute_b0_sensitivity_point,
    compute_field_tolerance,
    compute_homogeneity_point,
    sweep_b0_sensitivity,
    sweep_homogeneity,
)
from .gain_bandwidth_match import (  # noqa: F401
    BandwidthMatchResult,
    GainBandwidthConfig,
    compute_b0_drift_tolerance,
    compute_bandwidth_match,
    compute_maser_gain_bandwidth,
    sweep_b0_drift_vs_overlap,
    sweep_q_vs_gain_bandwidth,
)
from .gain_lock import (  # noqa: F401
    GainLockConfig,
    GainLockResult,
    GainLockStep,
    compute_cooperativity,
    find_threshold_pump_power,
    run_gain_lock_simulation,
)
from .halbach import (  # noqa: F401
    MultipoleCoefficients,
    compute_halbach_field,
    compute_multipole_coefficients,
)
from .magpylib_adapter import (  # noqa: F401
    MAGPYLIB_AVAILABLE,
    build_magpylib_collection,
    compare_on_axis,
    field_map_2d_magpylib,
    field_on_axis_magpylib,
    find_sweet_spot_magpylib,
)
from .maser_gain import MaserMetrics, compute_maser_metrics, max_tolerable_b_std  # noqa: F401
from .maser_gain_frequency import (  # noqa: F401
    GainCurveResult,
    bandwidth_analytical,
    compute_bandwidth_3db,
    compute_gain_bandwidth_product,
    compute_gain_curve,
    compute_saturation_power,
    gain_curve_from_mb_result,
)
from .maxwell_bloch import (  # noqa: F401
    MaxwellBlochResult,
    compute_steady_state_power,
    solve_maxwell_bloch,
)
from .mixer_nonlinearity import (  # noqa: F401
    PRODUCT_2F1_MINUS_F2,
    PRODUCT_2F2_MINUS_F1,
    IMD3Product,
    MixerNonlinearityConfig,
    MixerNonlinearityResult,
    compute_imd3_frequency_hz,
    compute_imd3_pair,
    compute_imd3_power_dbm,
    compute_mixer_nonlinearity,
)
from .mrzero_adapter import (  # noqa: F401
    MRZERO_AVAILABLE,
    BlochSignal,
    DepthValidation,
    build_single_voxel_phantom,
    build_spin_echo_sequence,
    compute_analytical_contrast,
    cross_validate_contrast,
    cross_validate_depth,
    simulate_depth_bloch,
    simulate_single_voxel_bloch,
)
from .nv_spin import effective_linewidth_ghz, transition_frequencies  # noqa: F401
from .odmr_simulator import (  # noqa: F401
    CrossValidation,
    FitResult,
    ODMRResult,
    compute_odmr_spectrum,
    cross_validate_linewidth,
    fit_odmr_spectrum,
    simulate_odmr_sweep,
)
from .optical_pump import (  # noqa: F401
    DepthResolvedPumpResult,
    PumpState,
    compute_absorbed_power,
    compute_depth_resolved_pump,
    compute_pump_rate,
    compute_pump_state,
)
from .phase1_validator import (  # noqa: F401
    OscillationThresholdResult,
    Phase1Config,
    Phase1MilestoneResult,
    validate_phase1_milestone,
)
from .phase4_validator import (  # noqa: F401
    LayerContrastResult,
    Phase4Config,
    Phase4MilestoneResult,
    compute_layer_contrast,
    validate_phase4_milestone,
)
from .phase6_validator import (  # noqa: F401
    BarContrastResult,
    GridPhantomResult,
    Phase6Config,
    Phase6MilestoneResult,
    validate_phase6_milestone,
)
from .phase9_validator import (  # noqa: F401
    Phase9Config,
    Phase9MilestoneResult,
    T2ContrastResult,
    validate_phase9_milestone,
)
from .planar_gradient import (  # noqa: F401
    DEFAULT_GX,
    DEFAULT_GY,
    GradientCoilSpec,
    GradientPulseResult,
    GradientWaveform,
    PhaseEncodeScheme,
    build_phase_encode_scheme,
    compute_coil_resistance,
    compute_gradient_efficiency,
    compute_inductance,
    compute_k_position,
    compute_k_trajectory,
    compute_max_gradient,
    compute_power_dissipation,
    current_for_gradient,
    evaluate_waveform,
    gradient_field_1d,
    linearity_error,
    sweep_efficiency_vs_radius,
    sweep_k_max_vs_fov,
    sweep_max_gradient_vs_current,
    sweep_resolution_vs_n_lines,
)
from .probe import (  # noqa: F401
    HandheldProbe,
    ProbeConfig,
    ProbePerformanceReport,
    compute_probe_performance,
    compute_stray_field_rms,
    sweep_depth_resolution_vs_bandwidth,
    sweep_lateral_resolution_vs_n_lines,
    sweep_snr_vs_averages,
    sweep_snr_vs_depth,
    sweep_stray_field_vs_separation,
)
from .pulse_sequence import (  # noqa: F401
    CPMGResult,
    GREResult,
    InversionRecoveryResult,
    SNREfficiency,
    SpinEchoResult,
    ernst_angle,
    optimal_te_for_contrast,
    simulate_cpmg,
    simulate_gre,
    simulate_inversion_recovery,
    simulate_spin_echo,
    snr_efficiency,
)
from .pulsed_pump import (  # noqa: F401
    PulsedPumpResult,
    compute_equivalent_cw_power,
    compute_pulsed_inversion,
    pulsed_pump_rate,
)
from .pulsed_pump_optimizer import (  # noqa: F401
    CWvsPulsedReport,
    OptimizedSequence,
    PulseCandidate,
    PulsedThresholdResult,
    compare_cw_vs_pulsed,
    compute_pulsed_threshold,
    optimize_pulse_sequence,
)
from .q_boost import (  # noqa: F401
    QBoostResult,
    compute_minimum_boost,
    compute_noise_temperature_boosted,
    compute_q_boost,
    compute_sql_limit_ratio,
)
from .quantum_noise import (  # noqa: F401
    MaserNoiseResult,
    PhaseNoiseSpectrum,
    RINSpectrum,
    compute_added_noise,
    compute_maser_noise,
    compute_noise_temperature,
    compute_phase_noise_spectrum,
    compute_population_inversion_factor,
    compute_rin_spectrum,
    compute_schawlow_townes_linewidth,
)
from .reconstruction import (  # noqa: F401
    DepthProfileResult,
    GriddingResult,
    apply_undersampling_mask,
    estimate_acceleration_factor,
    grid_kspace,
    haar_wavelet_inverse,
    haar_wavelet_transform,
    image_snr_from_phantom,
    reconstruct_compressed_sensing,
    reconstruct_depth_profile,
    reconstruct_fft,
    reconstruct_gridding,
    simulate_kspace,
    soft_threshold,
    sweep_resolution_vs_fov,
    sweep_snr_vs_acceleration,
)
from .reconstruction import (  # noqa: F401
    ReconResult as KSpaceReconResult,
)
from .rf_rejection import (  # noqa: F401
    InterfererResult,
    InterfererSpec,
    RFRejectionConfig,
    RFRejectionResult,
    compute_fractional_bandwidth,
    compute_interferer_rejection,
    compute_lorentzian_attenuation,
    compute_rf_rejection,
)
from .sensitivity import (  # noqa: F401
    SensitivityResult,
    compute_friis_sensitivity,
    compute_schawlow_townes_sensitivity,
    compute_sensitivity,
    compute_thermal_sensitivity,
)
from .shielding import (  # noqa: F401
    MuMetalShellConfig,
    ShieldingResult,
    compute_multilayer_attenuation,
    compute_shell_mass_kg,
    compute_shielding,
    compute_single_layer_attenuation,
    find_thickness_for_target_db,
)
from .signal_chain import (  # noqa: F401
    SignalChainBudget,
    compute_friis_system_temperature,
    compute_signal_chain_budget,
    compute_snr_vs_field_uniformity,
)
from .sigpy_adapter import (  # noqa: F401
    SIGPY_AVAILABLE,
    EncodingInfo,
    ReconResult,
    RFPulseResult,
    bloch_simulate_pulse,
    build_encoding_info,
    build_encoding_matrix,
    design_excitation_pulse,
    reconstruct_l1_wavelet,
    reconstruct_least_squares,
    reconstruct_total_variation,
    simulate_signal,
)
from .single_sided_magnet import (  # noqa: F401
    FieldMap2D,
    MilestoneResult,
    SingleSidedMagnet,
    SweetSpotInfo,
    validate_sweet_spot_milestone,
)
from .snr_calculator import (  # noqa: F401
    SNRBudget,
    compute_snr_budget,
    required_averages_for_snr,
    snr_vs_averages,
    snr_vs_depth,
    snr_vs_voxel_size,
)
from .spectral import (  # noqa: F401
    build_detuning_grid,
    build_initial_inversion,
    burn_spectral_hole,
    compute_on_resonance_inversion,
    q_gaussian,
    spectral_overlap_weights,
)
from .spectral_maxwell_bloch import (  # noqa: F401
    SpectralMBResult,
    solve_spectral_maxwell_bloch,
)
from .spin_squeezing import (  # noqa: F401
    REGIME_COHERENT,
    REGIME_NEAR_HEISENBERG,
    REGIME_SQUEEZED,
    ProjectionNoiseResult,
    QuantumEnhancementResult,
    SpinSqueezingResult,
    classify_squeezing_regime,
    compute_hl_field_sensitivity,
    compute_hl_phase_sensitivity,
    compute_metrological_gain_db,
    compute_oat_optimal_squeezing,
    compute_projection_noise,
    compute_quantum_enhancement,
    compute_spin_squeezing,
    compute_sql_field_sensitivity,
    compute_sql_phase_sensitivity,
    compute_wineland_squeezing,
)
from .squeezing_dynamics import (  # noqa: F401
    OATDecoherenceTrajectory,
    OATIdealTrajectory,
    SqueezingFeasibility,
    TATDecoherenceTrajectory,
    TATIdealTrajectory,
    apply_decoherence,
    compute_oat_ideal_trajectory,
    compute_oat_with_decoherence,
    compute_squeezing_feasibility,
    compute_tat_ideal_trajectory,
    compute_tat_with_decoherence,
    estimate_oat_chi,
    oat_optimal_time,
    oat_xi2_ideal,
    tat_optimal_time,
    tat_xi2_ideal,
)
from .stability import (  # noqa: F401
    CombinedADEVResult,
    NoiseProcessADEV,
    OscillatorStabilityResult,
    compute_allan_deviation_from_psd,
    compute_combined_allan_deviation,
    compute_flicker_fm_adev,
    compute_oscillator_stability,
    compute_random_walk_fm_adev,
    compute_white_fm_allan_deviation,
    compute_white_pm_adev,
)
from .superradiance import (  # noqa: F401
    BELOW_THRESHOLD,
    MASING,
    SUPERRADIANT,
    SuperradianceResult,
    compute_collective_coupling,
    compute_superradiance,
    compute_superradiant_delay,
    compute_superradiant_peak,
    compute_superradiant_pulse_duration,
    determine_regime,
)
from .surface_coil import (  # noqa: F401
    CoilProperties,
    NoiseComponents,
    SurfaceCoil,
    compute_coil_properties,
    compute_noise,
    sensitivity_off_axis,
    sensitivity_on_axis,
    snr_per_voxel,
)
from .susceptibility_adapter import (  # noqa: F401
    SUSCEPTIBILITY_TABLE,
    SusceptibilityCorrectedProfile,
    SusceptibilityProfile,
    apply_susceptibility_correction,
    compute_dephasing_signal_loss,
    compute_frequency_shift,
    compute_susceptibility_field_shift,
    cross_validate_susceptibility,
    estimate_susceptibility_impact,
)
from .t1t2_estimator import (  # noqa: F401
    AbnormalityFlag,
    BiexpT2FitResult,
    BlandAltmanResult,
    BlandAltmanT1T2,
    T1FitResult,
    T1MapResult,
    T1T2CrossValidation,
    T1T2Map,
    T2FitResult,
    T2MapResult,
    bland_altman_t1t2,
    build_t1t2_map,
    cross_validate_t1t2,
    detect_tissue_abnormalities,
    fit_t1_saturation_recovery,
    fit_t2_biexponential,
    fit_t2_monoexponential,
    map_t1_from_saturation_recovery,
    map_t2_from_cpmg,
    select_t2_model,
)
from .thermal import ThermalModel, ThermalState, compute_thermal_state  # noqa: F401
from .up_conversion import (  # noqa: F401
    DEFAULT_MIXER,
    MixerSpec,
    UpConversionNoiseContribution,
    UpConversionResult,
    compute_bandwidth_utilization,
    compute_lo_frequency_ghz,
    compute_mixer_noise_contribution,
    compute_up_conversion,
    friis_system_temperature_with_mixer,
)
