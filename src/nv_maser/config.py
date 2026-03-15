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
    auto_export_onnx: bool = Field(
        False,
        description="Automatically export ONNX model after training completes",
    )
    onnx_export_path: str = Field(
        "checkpoints/model.onnx",
        description="Output path for automatic ONNX export",
    )


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
    disturbance: DisturbanceConfig = DisturbanceConfig()
    coils: CoilConfig = CoilConfig()
    nv: NVConfig = NVConfig()
    maser: MaserConfig = MaserConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    viz: VizConfig = VizConfig()
    device: str = Field("auto", description="'auto' | 'cuda' | 'cpu'")

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
