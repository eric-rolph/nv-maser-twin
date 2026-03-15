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
