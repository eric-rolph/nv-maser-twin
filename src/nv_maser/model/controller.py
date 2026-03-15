"""
Neural network shimming controllers.

Input:  Distorted field observation — (batch, 1, size, size) for CNN, (batch, 1, size, size) for MLP
Output: Coil currents — (batch, num_coils) in [-max_current, +max_current]
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import CoilConfig, ModelConfig


def _activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[name]()


class CNNController(nn.Module):
    """
    Convolutional controller.

    Architecture::

        Input: (B, 1, 64, 64) field heatmap
        → Conv2d stack with BatchNorm and activation
        → AdaptiveAvgPool2d → flatten
        → Linear head → tanh * max_current → (B, num_coils)
    """

    def __init__(
        self, grid_size: int, model_cfg: ModelConfig, coil_cfg: CoilConfig
    ) -> None:
        super().__init__()
        self.max_current = coil_cfg.max_current_amps

        # Build conv stack
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in model_cfg.cnn_channels:
            layers.extend(
                [
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=model_cfg.cnn_kernel_size,
                        padding=model_cfg.cnn_kernel_size // 2,
                    ),
                    nn.BatchNorm2d(out_ch),
                    _activation(model_cfg.activation),
                    nn.MaxPool2d(2),
                ]
            )
            in_ch = out_ch

        layers.append(nn.AdaptiveAvgPool2d(1))
        self.conv_stack = nn.Sequential(*layers)

        # Compute flattened size after conv stack
        with torch.no_grad():
            dummy = torch.zeros(1, 1, grid_size, grid_size)
            flat_size = self.conv_stack(dummy).view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.Dropout(model_cfg.dropout),
            _activation(model_cfg.activation),
            nn.Linear(128, coil_cfg.num_coils),
            nn.Tanh(),  # Output in [-1, 1], scaled to [-max_current, max_current]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, size, size) distorted field tensor.

        Returns:
            (batch, num_coils) coil currents in physical units (Amps).
        """
        features = self.conv_stack(x)
        features = features.view(features.size(0), -1)
        return self.head(features) * self.max_current


class MLPController(nn.Module):
    """
    Multi-layer perceptron controller.

    Architecture::

        Input: (B, 1, size, size) → flatten → (B, size*size)
        → Hidden layers with activation and dropout
        → tanh * max_current → (B, num_coils)
    """

    def __init__(
        self, grid_size: int, model_cfg: ModelConfig, coil_cfg: CoilConfig
    ) -> None:
        super().__init__()
        self.max_current = coil_cfg.max_current_amps
        self.grid_size = grid_size

        layers: list[nn.Module] = []
        in_dim = grid_size * grid_size
        for hidden_dim in model_cfg.mlp_hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    _activation(model_cfg.activation),
                    nn.Dropout(model_cfg.dropout),
                ]
            )
            in_dim = hidden_dim

        layers.extend(
            [
                nn.Linear(in_dim, coil_cfg.num_coils),
                nn.Tanh(),
            ]
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, size, size) distorted field tensor.

        Returns:
            (batch, num_coils) coil currents in physical units (Amps).
        """
        flat = x.view(x.size(0), -1)
        return self.network(flat) * self.max_current


def build_controller(
    grid_size: int, model_cfg: ModelConfig, coil_cfg: CoilConfig
) -> nn.Module:
    """Factory function to build the selected controller architecture."""
    arch = model_cfg.architecture.value
    if arch == "cnn":
        return CNNController(grid_size, model_cfg, coil_cfg)
    elif arch == "mlp":
        return MLPController(grid_size, model_cfg, coil_cfg)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
