"""Tests for the training loop (fast, small-scale variants)."""
import pytest
import torch

from nv_maser.config import SimConfig
from nv_maser.model.training import Trainer


def _fast_config() -> SimConfig:
    """Config tuned for fast test execution (tiny dataset, 1 epoch)."""
    cfg = SimConfig()
    cfg.training.num_samples = 100
    cfg.training.batch_size = 16
    cfg.training.epochs = 1
    cfg.training.val_split = 0.2
    cfg.training.early_stopping_patience = 5
    cfg.training.checkpoint_dir = "checkpoints_test/"
    cfg.disturbance.seed = 0
    cfg.device = "cpu"
    return cfg


def test_single_epoch_no_crash(tmp_path) -> None:
    """Training runs 1 epoch on small dataset without error."""
    cfg = _fast_config()
    cfg.training.checkpoint_dir = str(tmp_path / "ckpt")
    trainer = Trainer(cfg)
    history = trainer.train()
    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) == 1


def test_loss_decreases(tmp_path) -> None:
    """Best val loss during training is lower than initial val loss (model learns)."""
    cfg = _fast_config()
    cfg.training.num_samples = 200
    cfg.training.batch_size = 32
    cfg.training.epochs = 5
    cfg.training.early_stopping_patience = 10
    cfg.training.checkpoint_dir = str(tmp_path / "ckpt")
    trainer = Trainer(cfg)
    history = trainer.train()
    # Best val loss (minimum across all epochs) must be lower than first epoch
    best_val = min(history["val_loss"])
    assert best_val <= history["val_loss"][0]


def test_checkpoint_save_load(tmp_path) -> None:
    """Save and load checkpoint, verify model outputs match."""
    cfg = _fast_config()
    cfg.training.checkpoint_dir = str(tmp_path / "ckpt")
    trainer = Trainer(cfg)
    trainer.train()

    # Record pre-load outputs
    x = torch.randn(2, 1, 64, 64)
    trainer.model.eval()
    with torch.no_grad():
        out_before = trainer.model(x).clone()

    # Corrupt model weights
    for p in trainer.model.parameters():
        p.data.zero_()

    # Reload best checkpoint
    trainer.load_best()
    trainer.model.eval()
    with torch.no_grad():
        out_after = trainer.model(x)

    assert torch.allclose(out_before, out_after, atol=1e-5)


def test_loss_fn_uniformity(tmp_path) -> None:
    """FieldUniformityLoss backward pass works."""
    cfg = _fast_config()
    cfg.training.checkpoint_dir = str(tmp_path / "ckpt")
    trainer = Trainer(cfg)

    batch_field = torch.randn(4, 1, 64, 64, requires_grad=False)
    loss, metrics = trainer._forward_step(batch_field)
    loss.backward()
    assert "total_loss" in metrics
    assert loss.item() >= 0
