"""
Central configuration for the NV Maser Digital Twin.
All tunable parameters are defined here with sensible defaults.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ModelArchitecture(str, Enum):
    CNN = "cnn"
    MLP = "mlp"
    LSTM = "lstm"


class GridConfig(BaseModel):
    """Spatial grid parameters."""

    size: int = Field(64, description="Grid points per axis (size × size)")
    physical_extent_mm: float = Field(10.0, description="Physical size of grid in mm")
    active_zone_fraction: float = Field(
        0.6,
        ge=0.1,
        le=1.0,
        description=(
            "Fraction of grid considered the 'active zone' for loss calc. "
            "0.6 means the central 60% of each axis."
        ),
    )


class FieldConfig(BaseModel):
    """Base magnetic field parameters."""

    b0_tesla: float = Field(
        0.05, description="Nominal B₀ field strength in Tesla (Halbach array)"
    )
    b0_gradient_ppm_per_mm: float = Field(
        0.0,
        description=(
            "Optional linear gradient across B₀ in ppm/mm. "
            "Set >0 to simulate imperfect Halbach geometry."
        ),
    )


class HalbachConfig(BaseModel):
    """Halbach permanent magnet array geometry and tolerances.

    Models a K=2 (dipole) Halbach cylinder built from N discrete magnets.
    The field inside has:
    - Dominant dipolar term B₀ from geometry
    - Systematic segmentation harmonics at orders kN±1
    - Random multipole errors from manufacturing tolerances

    The multipole expansion in 2D:
        B(x,y) = B₀ + Σ_n [a_n·cos(nθ) + b_n·sin(nθ)]·(r/r_in)^(n-1)

    where (r,θ) are polar coordinates relative to the bore center.
    """

    enabled: bool = Field(
        False,
        description=(
            "Enable Halbach multipole model. When False, the flat B₀ + "
            "optional gradient from FieldConfig is used (backward compatible)."
        ),
    )

    # ── Array geometry ─────────────────────────────────────────────
    num_segments: int = Field(
        8,
        ge=4,
        le=48,
        description=(
            "Number of discrete permanent magnet segments. "
            "8 is common for compact designs; 16-24 for high homogeneity."
        ),
    )
    inner_radius_mm: float = Field(
        7.0,
        gt=0,
        description="Bore inner radius (mm). Must be > physical_extent_mm / 2.",
    )
    outer_radius_mm: float = Field(
        15.0,
        gt=0,
        description="Outer shell radius (mm). Larger ratio → stronger field.",
    )
    remanence_tesla: float = Field(
        1.4,
        gt=0,
        description="NdFeB remanence Br (Tesla). N52: ~1.4 T, N42: ~1.3 T.",
    )

    # ── Multipole expansion ───────────────────────────────────────
    max_multipole_order: int = Field(
        12,
        ge=2,
        le=32,
        description=(
            "Highest multipole order in the expansion. "
            "Orders kN±1 carry the segmentation harmonics; "
            "higher orders fall off as (r/r_in)^(n-1)."
        ),
    )

    # ── Manufacturing tolerances (stochastic errors) ──────────────
    br_tolerance_pct: float = Field(
        1.0,
        ge=0.0,
        description=(
            "RMS segment-to-segment remanence variation (% of Br). "
            "Grade-sorted N52: ~0.5-1%. Unsorted: 2-5%."
        ),
    )
    angle_tolerance_deg: float = Field(
        0.5,
        ge=0.0,
        description=(
            "RMS magnetisation angle error per segment (degrees). "
            "Precision magnetised: 0.5°. Standard: 1-2°."
        ),
    )
    position_tolerance_mm: float = Field(
        0.05,
        ge=0.0,
        description=(
            "RMS segment radial position error (mm). "
            "Precision assembly: 0.02-0.05 mm. Hand assembly: 0.1-0.3 mm."
        ),
    )

    seed: int | None = Field(
        None,
        description="Random seed for manufacturing tolerance realisation.",
    )

    @property
    def ideal_b0_tesla(self) -> float:
        """Nominal B₀ from ideal Halbach dipole formula with segmentation correction.

        B₀ = Br × ln(R_out/R_in) × sin(π/N)/(π/N)
        """
        import math

        n = self.num_segments
        geo = math.log(self.outer_radius_mm / self.inner_radius_mm)
        seg_factor = math.sin(math.pi / n) / (math.pi / n) if n > 0 else 1.0
        return self.remanence_tesla * geo * seg_factor


class DisturbanceConfig(BaseModel):
    """Interference generator parameters."""

    num_modes: int = Field(
        5, ge=1, le=20, description="Number of superimposed spatial harmonic modes"
    )
    max_amplitude_tesla: float = Field(
        0.005, description="Maximum peak disturbance amplitude in Tesla"
    )
    min_spatial_freq: float = Field(
        0.1,
        description=(
            "Minimum spatial frequency (cycles across grid). "
            "Lower values produce gradient-like disturbances that the "
            "shim coil basis can cancel effectively."
        ),
    )
    max_spatial_freq: float = Field(
        1.5,
        description=(
            "Maximum spatial frequency (cycles across grid). "
            "Kept below ~2 so the 8-coil harmonic basis captures the "
            "dominant disturbance energy."
        ),
    )
    temporal_drift_rate: float = Field(
        0.1,
        description="Rate of temporal evolution for time-varying disturbances (Hz-equivalent)",
    )
    seed: int | None = Field(
        None, description="Random seed for reproducibility (None = random)"
    )


class CoilConfig(BaseModel):
    """Shim coil array parameters."""

    num_coils: int = Field(8, ge=4, le=32, description="Number of shim coils")
    coil_radius_mm: float = Field(
        6.0,
        description=(
            "Physical radius defining coil placement for visualisation. "
            "Not used by the gradient-coil influence model "
            "(patterns are analytically defined spatial harmonics)."
        ),
    )
    max_current_amps: float = Field(
        1.0,
        description="Maximum allowed current magnitude per coil (physical constraint)",
    )
    field_scale_factor: float = Field(
        0.005,
        description=(
            "Peak field produced by one coil at 1 A (Tesla). "
            "Set equal to max_amplitude_tesla so each coil at 1 A can "
            "cancel a full-amplitude disturbance in its spatial mode. "
            "Gradient patterns are L∞-normalised before this scaling."
        ),
    )


class NVConfig(BaseModel):
    """NV center spin physics parameters."""

    zero_field_splitting_ghz: float = Field(
        2.87, description="Zero-field splitting D (GHz) for NV⁻ ground state"
    )
    gamma_e_ghz_per_t: float = Field(
        28.025, description="Electron gyromagnetic ratio γe (GHz/T)"
    )
    t2_star_us: float = Field(
        1.0,
        gt=0,
        description=(
            "T2* ensemble dephasing time (μs). "
            "Determines homogeneous linewidth Γ_h = 1/(π·T2*). "
            "Good CVD diamond: 1-10 μs. Mediocre: 0.1-1 μs."
        ),
    )
    nv_density_per_cm3: float = Field(
        1e17,
        description=(
            "NV center concentration (per cm³). "
            "1 ppm ≈ 1.76×10¹⁷/cm³ in diamond."
        ),
    )
    pump_efficiency: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of NV centers effectively inverted by optical pumping. "
            "Ideal saturation: ~0.5. Realistic with losses: 0.2-0.4."
        ),
    )
    diamond_thickness_mm: float = Field(
        0.5,
        gt=0,
        description="Diamond thickness along cavity axis (mm).",
    )


class MaserConfig(BaseModel):
    """Microwave maser cavity parameters."""

    cavity_q: float = Field(
        10_000,
        gt=0,
        description="Loaded quality factor of the microwave cavity.",
    )
    cavity_frequency_ghz: float = Field(
        1.47,
        description=(
            "Cavity resonant frequency (GHz). "
            "Should match one NV transition: D ± γe·B₀. "
            "Default: D - γe·0.05T ≈ 1.47 GHz (lower branch at 50 mT)."
        ),
    )
    min_gain_budget: float = Field(
        0.5,
        gt=0,
        le=1.0,
        description=(
            "Minimum gain budget (Γ_h/Γ_eff) for maser oscillation. "
            "Below this, inhomogeneous broadening kills the gain. "
            "Depends on cavity Q, NV concentration, pump power. "
            "0.5 = maser needs ≥50%% of peak gain to overcome cavity losses."
        ),
    )
    coupling_beta: float = Field(
        0.5,
        gt=0.0,
        le=1.0,
        description=(
            "Cavity coupling factor β = Q_ext / Q_loaded. "
            "Fraction of intracavity power extracted at the coupling port. "
            "Critical coupling β = 0.5. Over-coupled > 0.5."
        ),
    )


class CavityConfig(BaseModel):
    """Microwave cavity geometry for threshold and Purcell calculations.

    The zero-point magnetic field in the cavity mode sets the single-spin
    coupling strength g₀ = γe · B_zpf.  Together with the cavity linewidth
    κ = ω/Q and ensemble size, this determines the cooperativity
    C = 4 g² N_eff / (κ · γ⊥) that must exceed 1 for masing.
    """

    mode_volume_cm3: float = Field(
        0.5,
        gt=0,
        description=(
            "Effective microwave mode volume (cm³). "
            "TE₀₁₁ at ~1.5 GHz in 12 mm bore: ~0.3-1.0 cm³. "
            "Smaller → stronger coupling."
        ),
    )
    fill_factor: float = Field(
        0.01,
        gt=0,
        le=1.0,
        description=(
            "Overlap fill factor V_diamond / V_mode. "
            "Typical 0.005-0.05 for thin diamond in microwave cavity."
        ),
    )


class OpticalPumpConfig(BaseModel):
    """532 nm optical pump parameters for NV population inversion.

    The pump laser drives the spin-selective intersystem crossing (ISC)
    that polarises the NV⁻ ground state into |0⟩, creating the population
    inversion needed for stimulated microwave emission.

    Thermal load from the pump couples to the ThermalConfig model:
    absorbed power × quantum defect heats the diamond.
    """

    laser_power_w: float = Field(
        2.0,
        ge=0,
        description=(
            "CW pump laser power at 532 nm (Watts). "
            "Breeze et al. used ~4 W for CW masing. "
            "2 W is a reasonable starting point for threshold estimates."
        ),
    )
    laser_wavelength_nm: float = Field(
        532.0,
        gt=0,
        description="Pump laser wavelength (nm). 532 nm standard for NV.",
    )
    beam_waist_mm: float = Field(
        1.5,
        gt=0,
        description=(
            "1/e² Gaussian beam waist radius (mm). "
            "Determines peak intensity at diamond centre."
        ),
    )
    absorption_cross_section_m2: float = Field(
        3.1e-21,
        gt=0,
        description=(
            "NV⁻ absorption cross-section at 532 nm (m²). "
            "Literature: 3.1×10⁻²¹ m² (Wee et al. 2007)."
        ),
    )
    quantum_defect_fraction: float = Field(
        0.6,
        ge=0,
        le=1.0,
        description=(
            "Fraction of absorbed optical power converted to heat. "
            "Accounts for Stokes shift + non-radiative decay. "
            "0.5-0.7 typical for NV at 532 nm."
        ),
    )
    spin_t1_ms: float = Field(
        5.0,
        gt=0,
        description=(
            "NV ground-state spin-lattice relaxation T₁ (ms). "
            "Room temperature: 5-6 ms. Sets the decay rate competing "
            "against optical pumping."
        ),
    )


class SignalChainConfig(BaseModel):
    """RF signal chain parameters for SNR budget.

    Models the complete receiver chain from maser emission to digitised signal:
        NV maser → cavity coupling → LNA → bandpass filter → ADC

    The SNR budget determines whether the maser output is detectable above
    the combined thermal + amplifier + quantisation noise floor.
    """

    # ── LNA (Low Noise Amplifier) ─────────────────────────────────
    lna_noise_figure_db: float = Field(
        1.0,
        ge=0.0,
        description=(
            "LNA noise figure (dB). Sets the added receiver noise. "
            "Cryogenic HEMT: 0.1-0.3 dB. Room-temp GaAs: 0.5-1.5 dB. "
            "Default 1.0 dB is a good room-temperature LNA."
        ),
    )
    lna_gain_db: float = Field(
        30.0,
        ge=0.0,
        description=(
            "LNA power gain (dB). Typical 20-40 dB. "
            "High gain reduces noise contribution of subsequent stages."
        ),
    )

    # ── System temperature ────────────────────────────────────────
    physical_temperature_k: float = Field(
        300.0,
        gt=0,
        description=(
            "Physical temperature of the receiver (K). "
            "Room temperature 300 K. Cryogenic: 4-77 K."
        ),
    )

    # ── Measurement bandwidth ─────────────────────────────────────
    detection_bandwidth_hz: float = Field(
        1000.0,
        gt=0,
        description=(
            "Post-detection bandwidth (Hz). Determines noise floor. "
            "Narrower → higher SNR but slower response. "
            "1 kHz matches the shimming control loop rate."
        ),
    )

    # ── ADC ───────────────────────────────────────────────────────
    adc_bits: int = Field(
        14,
        ge=8,
        le=24,
        description=(
            "ADC resolution (bits). Determines quantisation noise floor. "
            "14-bit: typical high-speed RF ADC. 16-bit: precision."
        ),
    )
    adc_full_scale_dbm: float = Field(
        -10.0,
        description=(
            "ADC full-scale input power (dBm). "
            "Sets the reference for quantisation noise."
        ),
    )

    # ── Cable / insertion losses ──────────────────────────────────
    insertion_loss_db: float = Field(
        2.0,
        ge=0.0,
        description=(
            "Total insertion loss between cavity and LNA (dB). "
            "Cables, connectors, isolator. 1-3 dB typical."
        ),
    )


class ModelConfig(BaseModel):
    """Neural network controller parameters."""

    architecture: ModelArchitecture = Field(ModelArchitecture.CNN)
    # CNN-specific
    cnn_channels: list[int] = Field(
        [16, 32, 64], description="Channel progression for conv layers"
    )
    cnn_kernel_size: int = Field(3)
    # MLP-specific
    mlp_hidden_dims: list[int] = Field(
        [512, 256, 128], description="Hidden layer sizes for MLP"
    )
    dropout: float = Field(0.1, ge=0.0, le=0.5)
    activation: str = Field("relu", description="Activation function: relu, gelu, silu")
    # LSTM-specific
    lstm_hidden_size: int = Field(256, description="Hidden state size for LSTM layers")
    lstm_num_layers: int = Field(2, description="Number of LSTM layers")


class TrainingConfig(BaseModel):
    """Training loop parameters."""

    num_samples: int = Field(
        10_000, description="Number of distorted field samples to generate"
    )
    batch_size: int = Field(64)
    epochs: int = Field(50)
    learning_rate: float = Field(1e-3)
    lr_scheduler: str = Field("cosine", description="cosine | step | none")
    lr_step_size: int = Field(15, description="Step size for StepLR scheduler")
    lr_gamma: float = Field(0.5, description="Gamma for StepLR scheduler")
    weight_decay: float = Field(1e-5)
    val_split: float = Field(0.1, description="Fraction of data for validation")
    checkpoint_dir: str = Field("checkpoints/")
    early_stopping_patience: int = Field(
        20, description="Epochs without improvement before stopping"
    )
    current_penalty_weight: float = Field(
        1e-6,
        description=(
            "L2 penalty weight on coil currents (A²). "
            "Must be  field_scale_factor² so the penalty does not "
            "dominate the field-variance term and force zero-current predictions. "
            "Rule of thumb: ≈ (field_scale_factor / max_current)² × 0.01."
        ),
    )
    loss_type: str = Field(
        "variance",
        description=(
            "Loss function type: 'variance' (original Var(B)+λ||I||²) or "
            "'physics' (adds gain_budget + cooperativity penalty terms)."
        ),
    )
    gain_budget_penalty_weight: float = Field(
        1e-5,
        ge=0.0,
        description="Weight α for −log(gain_budget) penalty in physics loss.",
    )
    cooperativity_penalty_weight: float = Field(
        1e-5,
        ge=0.0,
        description="Weight β for max(0, 1−C) penalty in physics loss.",
    )
    reward_shaping: bool = Field(
        False,
        description=(
            "Enable physics reward shaping in RL environment. "
            "Adds gain_budget improvement to variance-based reward."
        ),
    )
    reward_shaping_weight: float = Field(
        0.1,
        ge=0.0,
        description="Weight γ for gain_budget term in shaped RL reward.",
    )
    auto_export_onnx: bool = Field(
        False,
        description="Automatically export ONNX model after training completes",
    )
    onnx_export_path: str = Field(
        "checkpoints/model.onnx",
        description="Output path for automatic ONNX export",
    )


class FeedbackConfig(BaseModel):
    """Closed-loop feedback hardware parameters."""

    # ── Hall sensor ────────────────────────────────────────────────
    sensor_noise_tesla: float = Field(
        1e-7,
        ge=0,
        description=(
            "RMS noise floor of Hall-effect sensors (Tesla). "
            "Typical AH49E: ~100 nT. High-end MLX90395: ~10 nT. "
            "Default 100 nT is conservative."
        ),
    )
    num_sensors: int = Field(
        4,
        ge=1,
        le=64,
        description=(
            "Number of point Hall sensors in the active zone. "
            "Sensors provide sparse field measurements for the controller."
        ),
    )
    sensor_bandwidth_hz: float = Field(
        1000.0,
        gt=0,
        description="Sensor measurement bandwidth (Hz). Limits update rate.",
    )

    # ── DAC (Digital-to-Analog Converter) ──────────────────────────
    dac_bits: int = Field(
        16,
        ge=8,
        le=24,
        description=(
            "DAC resolution in bits. Determines current quantization. "
            "16-bit over ±1A → step = 30.5 μA."
        ),
    )
    dac_settling_time_us: float = Field(
        10.0,
        ge=0,
        description="DAC settling time (μs). Adds latency to control loop.",
    )

    # ── Coil electrical dynamics ───────────────────────────────────
    coil_inductance_uh: float = Field(
        100.0,
        ge=0,
        description=(
            "Coil inductance (μH). Flex polyimide micro-coils: 10-500 μH. "
            "Combined with resistance, sets the L/R time constant."
        ),
    )
    coil_resistance_ohm: float = Field(
        5.0,
        gt=0,
        description="Coil DC resistance (Ω). Flex polyimide: 1-20 Ω.",
    )

    # ── Control loop timing ───────────────────────────────────────
    control_loop_period_us: float = Field(
        1000.0,
        gt=0,
        description=(
            "Control loop period (μs). How often the NN runs. "
            "1000 μs = 1 kHz update rate."
        ),
    )
    computation_latency_us: float = Field(
        100.0,
        ge=0,
        description=(
            "NN inference + data transfer latency (μs). "
            "CNN on MCU: ~100-500 μs. On FPGA: ~10 μs."
        ),
    )

    @property
    def dac_lsb_amps(self) -> float:
        """Current step size for DAC quantization."""
        # Full range is ±max_current, so 2×max / 2^bits
        # We use 2.0 as the range (±1A default), but this is
        # referenced externally with the actual max_current.
        return 2.0 / (2**self.dac_bits)

    @property
    def coil_time_constant_us(self) -> float:
        """L/R time constant (μs)."""
        return self.coil_inductance_uh / self.coil_resistance_ohm

    @property
    def total_loop_latency_us(self) -> float:
        """Total latency: computation + DAC settling."""
        return self.computation_latency_us + self.dac_settling_time_us


class ThermalConfig(BaseModel):
    """Thermal coupling parameters.

    Models how temperature fluctuations affect every subsystem simultaneously:
    - Halbach magnets: NdFeB tempco drifts B₀
    - Diamond: phonon-limited T2* degradation
    - Cavity: wall resistivity changes Q
    - Coils: copper resistivity changes R → L/R time constant
    """

    # ── Operating point ───────────────────────────────────────────
    reference_temperature_c: float = Field(
        25.0,
        description="Reference temperature (°C) at which all nominal params are specified.",
    )
    ambient_temperature_c: float = Field(
        25.0,
        description="Current ambient temperature (°C). Set differently to model drift.",
    )

    # ── Halbach magnet (NdFeB) ────────────────────────────────────
    magnet_tempco_pct_per_c: float = Field(
        -0.12,
        description=(
            "NdFeB reversible temperature coefficient (%/°C). "
            "Typical N52: -0.12%/°C. Br drops as temperature rises."
        ),
    )

    # ── Diamond NV T2* ────────────────────────────────────────────
    t2_star_tempco_exponent: float = Field(
        1.0,
        ge=0.0,
        le=3.0,
        description=(
            "Temperature exponent for T2* degradation: "
            "T2*(T) = T2*(T_ref) × (T_ref/T)^n. "
            "n=1 for single-phonon Raman, n≈2 for Orbach. "
            "Conservative default n=1."
        ),
    )

    # ── Microwave cavity ──────────────────────────────────────────
    cavity_wall_tempco_per_c: float = Field(
        0.004,
        ge=0.0,
        description=(
            "Temperature coefficient of cavity wall resistivity (/°C). "
            "Copper: ~0.004/°C, aluminum: ~0.004/°C. "
            "Q ∝ 1/√ρ, so Q(T) = Q_ref × √(1/(1 + α·ΔT))."
        ),
    )

    # ── Coil copper ───────────────────────────────────────────────
    coil_tempco_per_c: float = Field(
        0.004,
        ge=0.0,
        description=(
            "Temperature coefficient of coil DC resistance (/°C). "
            "Copper: ~0.00393/°C. Affects L/R time constant."
        ),
    )

    # ── Temporal dynamics ─────────────────────────────────────────
    thermal_drift_rate_c_per_s: float = Field(
        0.01,
        ge=0.0,
        description=(
            "Rate of ambient temperature drift (°C/s). "
            "Models slow environmental changes. 0.01 °C/s = 36 °C/hr (worst case). "
            "Well-insulated lab: ~0.001 °C/s."
        ),
    )
    thermal_noise_std_c: float = Field(
        0.1,
        ge=0.0,
        description=(
            "Standard deviation of fast temperature fluctuations (°C). "
            "Models short-term noise on top of drift. "
            "Well-controlled: 0.01 °C. Moderate: 0.1 °C."
        ),
    )

    # ── External heat source (pump → thermal coupling) ─────────────
    external_heat_w: float = Field(
        0.0,
        ge=0.0,
        description=(
            "External heat deposited into the diamond (Watts). "
            "Set from optical pump thermal_load_w to close the "
            "pump → thermal feedback loop. 0 = backward compatible."
        ),
    )
    thermal_resistance_c_per_w: float = Field(
        5.0,
        gt=0.0,
        description=(
            "Thermal resistance from diamond to heat sink (°C/W). "
            "Determines steady-state temperature rise from pump heating: "
            "ΔT = P_heat × R_th. Typical 3-10 °C/W for diamond on Cu heat sink."
        ),
    )

    @property
    def delta_t(self) -> float:
        """Temperature offset from reference (°C)."""
        return self.ambient_temperature_c - self.reference_temperature_c


class VizConfig(BaseModel):
    """Visualization parameters."""

    update_interval_ms: int = Field(
        50, description="Dashboard refresh interval in ms"
    )
    colormap: str = Field("RdBu_r", description="Matplotlib/PyQtGraph colormap name")
    show_coil_positions: bool = Field(True)
    field_range_tesla: tuple[float, float] = Field(
        (0.045, 0.055),
        description="Color scale range for field heatmaps",
    )


class SimConfig(BaseModel):
    """Top-level simulation configuration."""

    grid: GridConfig = GridConfig()
    field: FieldConfig = FieldConfig()
    halbach: HalbachConfig = HalbachConfig()
    disturbance: DisturbanceConfig = DisturbanceConfig()
    coils: CoilConfig = CoilConfig()
    nv: NVConfig = NVConfig()
    maser: MaserConfig = MaserConfig()
    signal_chain: SignalChainConfig = SignalChainConfig()
    cavity: CavityConfig = CavityConfig()
    optical_pump: OpticalPumpConfig = OpticalPumpConfig()
    feedback: FeedbackConfig = FeedbackConfig()
    thermal: ThermalConfig = ThermalConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    viz: VizConfig = VizConfig()
    device: str = Field("auto", description="'auto' | 'cuda' | 'cpu'")

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
