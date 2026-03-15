"""
Custom loss function for magnetic field uniformity.

The loss is the variance of the corrected field over the active zone,
plus a regularization penalty on coil currents.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FieldUniformityLoss(nn.Module):
    """
    Loss = Var(B_net over active zone) + λ · ||I||²

    Where:
        B_net = distorted_field + coil_field(currents)
        active zone = precomputed boolean mask on the grid
        λ = current_penalty_weight (encourages minimal current usage)
    """

    def __init__(
        self,
        active_zone_mask: torch.Tensor,
        current_penalty_weight: float = 0.01,
    ) -> None:
        super().__init__()
        # Register as buffer so it moves to device with the model
        self.register_buffer("mask", active_zone_mask)  # (size, size) bool
        self.current_penalty = current_penalty_weight
        self._num_active = int(active_zone_mask.sum().item())

    def forward(
        self,
        net_field: torch.Tensor,
        currents: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            net_field: (batch, size, size) corrected field.
            currents:  (batch, num_coils) applied currents.

        Returns:
            total_loss: Scalar tensor (differentiable).
            metrics: Dict of detached float metrics for logging.
        """
        # Extract active zone: (batch, num_active_points)
        active = net_field[:, self.mask]  # type: ignore[index]

        # Primary loss: field variance over active zone (per sample, then mean)
        field_var = torch.var(active, dim=1).mean()

        # Regularization: L2 norm of currents (encourages small corrections)
        current_l2 = torch.mean(currents**2)

        total = field_var + self.current_penalty * current_l2

        metrics = {
            "field_variance": float(field_var.detach().cpu()),
            "current_l2": float(current_l2.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
        }
        return total, metrics
