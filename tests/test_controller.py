"""Tests for neural network controllers."""
import pytest
import torch

from nv_maser.config import CoilConfig, ModelArchitecture, ModelConfig, SimConfig
from nv_maser.model.controller import CNNController, LSTMController, MLPController, build_controller


@pytest.fixture
def cnn_model() -> CNNController:
    return CNNController(64, ModelConfig(architecture=ModelArchitecture.CNN), CoilConfig())


@pytest.fixture
def mlp_model() -> MLPController:
    return MLPController(64, ModelConfig(architecture=ModelArchitecture.MLP), CoilConfig())


def _dummy_input(batch: int = 4) -> torch.Tensor:
    return torch.randn(batch, 1, 64, 64)


def test_cnn_output_shape(cnn_model: CNNController) -> None:
    """CNN: (B, 1, 64, 64) → (B, 8)."""
    x = _dummy_input()
    out = cnn_model(x)
    assert out.shape == (4, 8)


def test_mlp_output_shape(mlp_model: MLPController) -> None:
    """MLP: (B, 1, 64, 64) → (B, 8)."""
    x = _dummy_input()
    out = mlp_model(x)
    assert out.shape == (4, 8)


def test_cnn_output_range(cnn_model: CNNController) -> None:
    """CNN outputs within [-max_current, +max_current]."""
    max_c = CoilConfig().max_current_amps
    out = cnn_model(_dummy_input(16))
    assert float(out.abs().max().item()) <= max_c + 1e-5


def test_mlp_output_range(mlp_model: MLPController) -> None:
    """MLP outputs within [-max_current, +max_current]."""
    max_c = CoilConfig().max_current_amps
    out = mlp_model(_dummy_input(16))
    assert float(out.abs().max().item()) <= max_c + 1e-5


def test_factory_builds_cnn() -> None:
    """build_controller returns CNN."""
    cfg = ModelConfig(architecture=ModelArchitecture.CNN)
    model = build_controller(64, cfg, CoilConfig())
    assert isinstance(model, CNNController)


def test_factory_builds_mlp() -> None:
    """build_controller returns MLP."""
    cfg = ModelConfig(architecture=ModelArchitecture.MLP)
    model = build_controller(64, cfg, CoilConfig())
    assert isinstance(model, MLPController)


def test_factory_unknown_arch() -> None:
    """build_controller raises for unknown architecture."""
    cfg = ModelConfig()
    cfg.architecture = "unknown"  # type: ignore[assignment]
    with pytest.raises((ValueError, KeyError, AttributeError)):
        build_controller(64, cfg, CoilConfig())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_cuda_transfer() -> None:
    """Model transfers to CUDA cleanly."""
    model = build_controller(64, ModelConfig(), CoilConfig()).to("cuda")
    x = _dummy_input().to("cuda")
    out = model(x)
    assert out.device.type == "cuda"
    assert out.shape == (4, 8)


# ---------------------------------------------------------------------------
# LSTM tests
# ---------------------------------------------------------------------------


@pytest.fixture
def lstm_model() -> LSTMController:
    return LSTMController(64, ModelConfig(architecture=ModelArchitecture.LSTM), CoilConfig())


def test_lstm_output_shape(lstm_model: LSTMController) -> None:
    """LSTM: (B, 1, 64, 64) → (B, 8)."""
    out = lstm_model(_dummy_input())
    assert out.shape == (4, 8)


def test_lstm_output_range(lstm_model: LSTMController) -> None:
    """LSTM outputs within [-max_current, +max_current]."""
    max_c = CoilConfig().max_current_amps
    out = lstm_model(_dummy_input(16))
    assert float(out.abs().max().item()) <= max_c + 1e-5


def test_factory_builds_lstm() -> None:
    """build_controller returns LSTMController."""
    cfg = ModelConfig(architecture=ModelArchitecture.LSTM)
    model = build_controller(64, cfg, CoilConfig())
    assert isinstance(model, LSTMController)


def test_backward_pass(cnn_model: CNNController) -> None:
    """Backward pass runs without error."""
    x = _dummy_input()
    out = cnn_model(x)
    loss = out.sum()
    loss.backward()
