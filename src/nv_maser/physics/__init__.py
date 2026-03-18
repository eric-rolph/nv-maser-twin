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
