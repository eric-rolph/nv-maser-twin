"""Physics simulation modules for the NV Maser Digital Twin."""

from .nv_spin import transition_frequencies, effective_linewidth_ghz  # noqa: F401
from .maser_gain import compute_maser_metrics, max_tolerable_b_std  # noqa: F401
from .feedback import HallSensorArray, quantize_currents, CoilDynamics  # noqa: F401
from .closed_loop import ClosedLoopSimulator, ClosedLoopResult  # noqa: F401
from .thermal import ThermalModel, ThermalState, compute_thermal_state  # noqa: F401
from .halbach import (  # noqa: F401
    MultipoleCoefficients,
    compute_halbach_field,
    compute_multipole_coefficients,
)
from .signal_chain import (  # noqa: F401
    SignalChainBudget,
    compute_signal_chain_budget,
    compute_friis_system_temperature,
    compute_snr_vs_field_uniformity,
)
from .cavity import (  # noqa: F401
    CavityProperties,
    ThresholdResult,
    compute_cavity_properties,
    compute_maser_threshold,
    compute_n_effective,
    compute_full_threshold,
    compute_effective_q,
    compute_spectral_overlap,
)
from .optical_pump import (  # noqa: F401
    PumpState,
    DepthResolvedPumpResult,
    compute_pump_rate,
    compute_absorbed_power,
    compute_pump_state,
    compute_depth_resolved_pump,
)
from .pulsed_pump import (  # noqa: F401
    PulsedPumpResult,
    pulsed_pump_rate,
    compute_pulsed_inversion,
    compute_equivalent_cw_power,
)
from .pulsed_pump_optimizer import (  # noqa: F401
    PulseCandidate,
    OptimizedSequence,
    PulsedThresholdResult,
    CWvsPulsedReport,
    optimize_pulse_sequence,
    compute_pulsed_threshold,
    compare_cw_vs_pulsed,
)
from .maxwell_bloch import (  # noqa: F401
    MaxwellBlochResult,
    solve_maxwell_bloch,
    compute_steady_state_power,
)
from .spectral import (  # noqa: F401
    build_detuning_grid,
    build_initial_inversion,
    burn_spectral_hole,
    compute_on_resonance_inversion,
    q_gaussian,
    spectral_overlap_weights,
)
from .dipolar import (  # noqa: F401
    stretched_exponential_refill,
    spectral_diffusion_step,
    apply_dipolar_refilling,
    estimate_dipolar_coupling_hz,
    estimate_refilling_time_us,
)
from .spectral_maxwell_bloch import (  # noqa: F401
    SpectralMBResult,
    solve_spectral_maxwell_bloch,
)
from .single_sided_magnet import (  # noqa: F401
    SingleSidedMagnet,
    SweetSpotInfo,
    FieldMap2D,
)
from .magpylib_adapter import (  # noqa: F401
    MAGPYLIB_AVAILABLE,
    build_magpylib_collection,
    field_on_axis_magpylib,
    field_map_2d_magpylib,
    compare_on_axis,
    find_sweet_spot_magpylib,
)
from .surface_coil import (  # noqa: F401
    SurfaceCoil,
    CoilProperties,
    NoiseComponents,
    sensitivity_on_axis,
    sensitivity_off_axis,
    compute_coil_properties,
    compute_noise,
    snr_per_voxel,
)
from .depth_profile import (  # noqa: F401
    TissueLayer,
    DepthProfile,
    FOREARM_LAYERS,
    HEMORRHAGE_LAYERS,
    simulate_depth_profile,
    add_noise,
)
from .sigpy_adapter import (  # noqa: F401
    SIGPY_AVAILABLE,
    EncodingInfo,
    ReconResult,
    RFPulseResult,
    build_encoding_info,
    build_encoding_matrix,
    simulate_signal,
    reconstruct_least_squares,
    reconstruct_l1_wavelet,
    reconstruct_total_variation,
    design_excitation_pulse,
    bloch_simulate_pulse,
)
from .odmr_simulator import (  # noqa: F401
    ODMRResult,
    FitResult,
    CrossValidation,
    compute_odmr_spectrum,
    simulate_odmr_sweep,
    fit_odmr_spectrum,
    cross_validate_linewidth,
)
from .mrzero_adapter import (  # noqa: F401
    MRZERO_AVAILABLE,
    BlochSignal,
    DepthValidation,
    build_single_voxel_phantom,
    build_spin_echo_sequence,
    simulate_single_voxel_bloch,
    compute_analytical_contrast,
    simulate_depth_bloch,
    cross_validate_depth,
    cross_validate_contrast,
)
from .susceptibility_adapter import (  # noqa: F401
    SUSCEPTIBILITY_TABLE,
    SusceptibilityProfile,
    SusceptibilityCorrectedProfile,
    compute_susceptibility_field_shift,
    compute_frequency_shift,
    compute_dephasing_signal_loss,
    apply_susceptibility_correction,
    estimate_susceptibility_impact,
    cross_validate_susceptibility,
)
from .epg_adapter import (  # noqa: F401
    EPGResult,
    EPGDepthProfile,
    EPGValidation,
    epg_signal,
    epg_cpmg,
    epg_depth_profile,
    cross_validate_epg_vs_analytical,
)
from .t1t2_estimator import (  # noqa: F401
    T2FitResult,
    T1FitResult,
    T2MapResult,
    T1MapResult,
    T1T2Map,
    AbnormalityFlag,
    T1T2CrossValidation,
    fit_t2_monoexponential,
    fit_t1_saturation_recovery,
    map_t2_from_cpmg,
    map_t1_from_saturation_recovery,
    build_t1t2_map,
    detect_tissue_abnormalities,
    cross_validate_t1t2,
)
from .quantum_noise import (  # noqa: F401
    MaserNoiseResult,
    PhaseNoiseSpectrum,
    RINSpectrum,
    compute_population_inversion_factor,
    compute_schawlow_townes_linewidth,
    compute_added_noise,
    compute_noise_temperature,
    compute_phase_noise_spectrum,
    compute_rin_spectrum,
    compute_maser_noise,
)
from .superradiance import (  # noqa: F401
    SuperradianceResult,
    BELOW_THRESHOLD,
    MASING,
    SUPERRADIANT,
    compute_collective_coupling,
    determine_regime,
    compute_superradiant_pulse_duration,
    compute_superradiant_delay,
    compute_superradiant_peak,
    compute_superradiance,
)
from .sensitivity import (  # noqa: F401
    SensitivityResult,
    compute_schawlow_townes_sensitivity,
    compute_thermal_sensitivity,
    compute_friis_sensitivity,
    compute_sensitivity,
)
from .stability import (  # noqa: F401
    OscillatorStabilityResult,
    compute_white_fm_allan_deviation,
    compute_allan_deviation_from_psd,
    compute_oscillator_stability,
)
from .spin_squeezing import (  # noqa: F401
    REGIME_COHERENT,
    REGIME_SQUEEZED,
    REGIME_NEAR_HEISENBERG,
    ProjectionNoiseResult,
    SpinSqueezingResult,
    QuantumEnhancementResult,
    compute_sql_phase_sensitivity,
    compute_hl_phase_sensitivity,
    compute_sql_field_sensitivity,
    compute_hl_field_sensitivity,
    compute_projection_noise,
    compute_wineland_squeezing,
    compute_oat_optimal_squeezing,
    compute_metrological_gain_db,
    classify_squeezing_regime,
    compute_spin_squeezing,
    compute_quantum_enhancement,
)
from .amplifier import (  # noqa: F401
    SIGMA_2,
    AmplifierProperties,
    MaserGainResult,
    OutputPowerResult,
    compute_magnetic_q,
    compute_spin_temperature,
    compute_noise_temperature as compute_amplifier_noise_temperature,
    compute_sql_noise_temperature,
    compute_output_power,
    compute_amplifier_properties,
    compute_maser_gain,
)
from .q_boost import (  # noqa: F401
    QBoostResult,
    compute_q_boost,
    compute_minimum_boost,
    compute_noise_temperature_boosted,
    compute_sql_limit_ratio,
)
from .maser_gain_frequency import (  # noqa: F401
    GainCurveResult,
    compute_gain_curve,
    compute_bandwidth_3db,
    bandwidth_analytical,
    compute_gain_bandwidth_product,
    compute_saturation_power,
    gain_curve_from_mb_result,
)
from .pulse_sequence import (  # noqa: F401
    SpinEchoResult,
    CPMGResult,
    GREResult,
    InversionRecoveryResult,
    SNREfficiency,
    simulate_spin_echo,
    simulate_cpmg,
    simulate_gre,
    simulate_inversion_recovery,
    ernst_angle,
    optimal_te_for_contrast,
    snr_efficiency,
)
from .up_conversion import (  # noqa: F401
    MixerSpec,
    DEFAULT_MIXER,
    UpConversionResult,
    UpConversionNoiseContribution,
    compute_up_conversion,
    compute_mixer_noise_contribution,
    compute_lo_frequency_ghz,
    compute_bandwidth_utilization,
    friis_system_temperature_with_mixer,
)
from .snr_calculator import (  # noqa: F401
    SNRBudget,
    compute_snr_budget,
    snr_vs_depth,
    snr_vs_averages,
    snr_vs_voxel_size,
    required_averages_for_snr,
)
from .planar_gradient import (  # noqa: F401
    GradientCoilSpec,
    GradientWaveform,
    GradientPulseResult,
    PhaseEncodeScheme,
    DEFAULT_GX,
    DEFAULT_GY,
    compute_gradient_efficiency,
    compute_coil_resistance,
    compute_inductance,
    compute_power_dissipation,
    compute_max_gradient,
    current_for_gradient,
    compute_k_position,
    compute_k_trajectory,
    evaluate_waveform,
    build_phase_encode_scheme,
    gradient_field_1d,
    linearity_error,
    sweep_efficiency_vs_radius,
    sweep_max_gradient_vs_current,
    sweep_k_max_vs_fov,
    sweep_resolution_vs_n_lines,
)
from .reconstruction import (  # noqa: F401
    ReconResult as KSpaceReconResult,
    DepthProfileResult,
    GriddingResult,
    reconstruct_fft,
    grid_kspace,
    reconstruct_gridding,
    haar_wavelet_transform,
    haar_wavelet_inverse,
    soft_threshold,
    reconstruct_compressed_sensing,
    reconstruct_depth_profile,
    simulate_kspace,
    apply_undersampling_mask,
    estimate_acceleration_factor,
    image_snr_from_phantom,
    sweep_snr_vs_acceleration,
    sweep_resolution_vs_fov,
)
