"""Tests for checkpoint metadata enrichment and load_checkpoint_meta utility."""
import pytest
import torch
from pathlib import Path

from nv_maser.config import SimConfig
from nv_maser.model.training import Trainer, load_checkpoint_meta


def _minimal_config(tmp_path: Path) -> SimConfig:
    config = SimConfig()
    config.training.epochs = 1
    config.training.num_samples = 20
    config.training.batch_size = 10
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")
    config.disturbance.seed = 42
    config.device = "cpu"
    return config


def test_checkpoint_has_meta_after_training(tmp_path: Path) -> None:
    """Checkpoint saved after training must contain a 'meta' key."""
    config = _minimal_config(tmp_path)
    trainer = Trainer(config)
    trainer.train()

    ckpt_path = Path(config.training.checkpoint_dir) / "best.pt"
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    assert "meta" in saved, "Checkpoint must include a 'meta' key"
    assert saved["meta"]["arch"] == config.model.architecture.value


def test_meta_fields_correct(tmp_path: Path) -> None:
    """Checkpoint meta contains correct grid_size, num_coils, and epochs_trained."""
    config = _minimal_config(tmp_path)
    trainer = Trainer(config)
    trainer.train()

    ckpt_path = Path(config.training.checkpoint_dir) / "best.pt"
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = saved["meta"]

    assert meta["grid_size"] == config.grid.size
    assert meta["num_coils"] == config.coils.num_coils
    assert meta["epochs_trained"] >= 1


def test_load_checkpoint_meta_returns_dict(tmp_path: Path) -> None:
    """load_checkpoint_meta returns a dict with key 'arch'."""
    config = _minimal_config(tmp_path)
    trainer = Trainer(config)
    trainer.train()

    ckpt_path = Path(config.training.checkpoint_dir) / "best.pt"
    meta = load_checkpoint_meta(ckpt_path)

    assert isinstance(meta, dict)
    assert "arch" in meta


def test_load_checkpoint_meta_missing_file() -> None:
    """load_checkpoint_meta raises FileNotFoundError for a nonexistent path."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint_meta("nonexistent/path.pt")


def test_auto_export_onnx_flag_false(tmp_path: Path) -> None:
    """With auto_export_onnx=False (default), no .onnx file is created."""
    config = _minimal_config(tmp_path)
    config.training.auto_export_onnx = False
    trainer = Trainer(config)
    trainer.train()

    ckpt_dir = Path(config.training.checkpoint_dir)
    onnx_files = list(ckpt_dir.glob("*.onnx"))
    assert len(onnx_files) == 0, f"No ONNX file should be created, found: {onnx_files}"
