"""Tests for FieldUniformityLoss."""
import pytest
import torch

from nv_maser.model.loss import FieldUniformityLoss


@pytest.fixture
def mask() -> torch.Tensor:
    m = torch.zeros(64, 64, dtype=torch.bool)
    m[16:48, 16:48] = True
    return m


def test_zero_variance_uniform_field(mask: torch.Tensor) -> None:
    """Perfectly uniform field → field_variance ≈ 0."""
    loss_fn = FieldUniformityLoss(mask, current_penalty_weight=0.0)
    # Uniform field
    net = torch.ones(4, 64, 64) * 0.05
    currents = torch.zeros(4, 8)
    total, metrics = loss_fn(net, currents)
    assert metrics["field_variance"] == pytest.approx(0.0, abs=1e-8)
    assert total.item() == pytest.approx(0.0, abs=1e-8)


def test_nonzero_variance(mask: torch.Tensor) -> None:
    """Non-uniform field → loss > 0."""
    loss_fn = FieldUniformityLoss(mask, current_penalty_weight=0.0)
    net = torch.randn(4, 64, 64)
    currents = torch.zeros(4, 8)
    total, _ = loss_fn(net, currents)
    assert total.item() > 0


def test_current_penalty(mask: torch.Tensor) -> None:
    """Current penalty increases loss for non-zero currents."""
    loss_fn = FieldUniformityLoss(mask, current_penalty_weight=1.0)
    net = torch.ones(4, 64, 64) * 0.05
    zero_currents = torch.zeros(4, 8)
    nonzero_currents = torch.ones(4, 8)
    _, m0 = loss_fn(net, zero_currents)
    _, m1 = loss_fn(net, nonzero_currents)
    assert m1["total_loss"] > m0["total_loss"]


def test_backward_differentiable(mask: torch.Tensor) -> None:
    """Loss backward pass runs without error."""
    loss_fn = FieldUniformityLoss(mask)
    net = torch.randn(4, 64, 64, requires_grad=True)
    currents = torch.randn(4, 8, requires_grad=True)
    total, _ = loss_fn(net, currents)
    total.backward()
    assert net.grad is not None
    assert currents.grad is not None
