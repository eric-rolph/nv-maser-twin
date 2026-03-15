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
        0.5, description="Minimum spatial frequency (cycles across grid)"
    )
    max_spatial_freq: float = Field(
        4.0, description="Maximum spatial frequency (cycles across grid)"
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
            "Radial distance of coils from grid center in mm. "
            "Should be slightly outside the grid extent."
        ),
    )
    max_current_amps: float = Field(
        1.0,
        description="Maximum allowed current magnitude per coil (physical constraint)",
    )
    field_scale_factor: float = Field(
        1e-4,
        description=(
            "Proportionality constant: Tesla per Amp at 1mm distance. "
            "Tunes the coil influence strength."
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
        10, description="Epochs without improvement before stopping"
    )
    current_penalty_weight: float = Field(
        0.01,
        description="L2 penalty weight on coil currents to prefer minimal corrections",
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
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    viz: VizConfig = VizConfig()
    device: str = Field("auto", description="'auto' | 'cuda' | 'cpu'")

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
