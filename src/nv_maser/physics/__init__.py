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
