"""
Custom loss functions for magnetic field uniformity.

The primary loss is the variance of the corrected field over the active zone,
plus a regularization penalty on coil currents.

Optionally, a physics-informed loss adds penalties for poor gain budget
and low cooperativity.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..physics.environment import FieldEnvironment


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


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss: Var(B) + λ||I||² + α/gain_budget + β·max(0, 1−C).

    Wraps :class:`FieldUniformityLoss` and adds scalar penalties computed
    from the physics environment.  The physics terms are non-differentiable
    scalars (numpy), so they act as an adaptive weighting on each batch
    rather than gradient sources.
    """

    def __init__(
        self,
        active_zone_mask: torch.Tensor,
        env: FieldEnvironment,
        current_penalty_weight: float = 0.01,
        gain_budget_weight: float = 0.1,
        cooperativity_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_loss = FieldUniformityLoss(active_zone_mask, current_penalty_weight)
        self.env = env
        self.gain_budget_weight = gain_budget_weight
        self.cooperativity_weight = cooperativity_weight

    def forward(
        self,
        net_field: torch.Tensor,
        currents: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute physics-informed loss.

        The gain-budget and cooperativity penalties are computed from
        a representative sample (first item in batch) via the physics
        environment and added as non-differentiable scalar offsets.
        """
        base_loss, metrics = self.base_loss(net_field, currents)

        # Physics penalties from a representative sample in the batch
        sample_field = net_field[0].detach().cpu().numpy()
        phys = self.env.compute_uniformity_metric(sample_field)

        gain_budget = phys.get("gain_budget", 1.0)
        cooperativity = phys.get("cooperativity", 1.0)

        # 1/gain_budget penalty: smaller budget → larger penalty
        gb_penalty = 1.0 / max(gain_budget, 1e-6)
        # Cooperativity shortfall: penalty when C < 1
        coop_penalty = max(0.0, 1.0 - cooperativity)

        physics_term = (
            self.gain_budget_weight * gb_penalty
            + self.cooperativity_weight * coop_penalty
        )

        total = base_loss + physics_term

        metrics["gain_budget"] = gain_budget
        metrics["cooperativity"] = cooperativity
        metrics["physics_penalty"] = physics_term
        metrics["total_loss"] = float(total.detach().cpu())
        return total, metrics
