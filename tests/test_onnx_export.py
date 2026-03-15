"""Tests for ONNX export of shimming controllers."""
from __future__ import annotations

import pytest

from nv_maser.config import ModelArchitecture, ModelConfig, SimConfig
from nv_maser.export import OnnxExportResult, export_model


@pytest.fixture
def default_config() -> SimConfig:
    """Default SimConfig (CNN, 64×64 grid, 8 coils)."""
    return SimConfig()


# ---------------------------------------------------------------------------
# Test 1: export creates the .onnx file on disk
# ---------------------------------------------------------------------------
def test_export_creates_file(tmp_path, default_config):
    out = tmp_path / "model.onnx"
    result = export_model(default_config, output_path=out, checkpoint_path=None)
    assert out.exists(), "export_model must create the .onnx file"
    assert out.stat().st_size > 0, ".onnx file must be non-empty"
    assert result.path == out


# ---------------------------------------------------------------------------
# Test 2: result dataclass fields are correct for default config
# ---------------------------------------------------------------------------
def test_export_returns_result(tmp_path, default_config):
    result = export_model(
        default_config, output_path=tmp_path / "model.onnx", checkpoint_path=None
    )
    assert isinstance(result, OnnxExportResult)
    assert result.arch == "cnn"
    assert result.grid_size == default_config.grid.size
    assert result.num_coils == default_config.coils.num_coils
    assert result.opset == 17


# ---------------------------------------------------------------------------
# Test 3: input / output shapes are correct
# ---------------------------------------------------------------------------
def test_export_input_output_shapes(tmp_path, default_config):
    grid = default_config.grid.size
    num_coils = default_config.coils.num_coils
    result = export_model(
        default_config, output_path=tmp_path / "model.onnx", checkpoint_path=None
    )
    assert result.input_shape == (1, 1, grid, grid)
    assert result.output_shape == (1, num_coils)


# ---------------------------------------------------------------------------
# Test 4: MLP architecture — export succeeds, arch field is correct
# ---------------------------------------------------------------------------
def test_export_mlp_arch(tmp_path):
    config = SimConfig()
    config.model = ModelConfig(architecture=ModelArchitecture.MLP)
    out = tmp_path / "mlp_model.onnx"
    result = export_model(config, output_path=out, checkpoint_path=None)
    assert out.exists()
    assert result.arch == "mlp"
    assert result.output_shape == (1, config.coils.num_coils)


# ---------------------------------------------------------------------------
# Test 5: onnxruntime inference (skipped if onnxruntime is not installed)
# ---------------------------------------------------------------------------
def test_onnx_runtime_inference(tmp_path, default_config):
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    out = tmp_path / "rt_model.onnx"
    result = export_model(default_config, output_path=out, checkpoint_path=None)

    sess = ort.InferenceSession(str(out))
    dummy = np.random.randn(1, 1, 64, 64).astype(np.float32)
    outputs = sess.run(None, {"distorted_field": dummy})

    assert len(outputs) == 1, "Expected exactly one output tensor"
    assert tuple(outputs[0].shape) == (1, default_config.coils.num_coils), (
        f"Expected output shape (1, {default_config.coils.num_coils}), "
        f"got {tuple(outputs[0].shape)}"
    )
