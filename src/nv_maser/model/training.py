"""
Training pipeline: dataset generation, training loop, validation, checkpointing.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from ..config import SimConfig
from ..physics.environment import FieldEnvironment
from .controller import build_controller
from .loss import FieldUniformityLoss


class Trainer:
    """
    Manages the full training pipeline.

    Workflow:
        1. Generate distorted field dataset from FieldEnvironment.
        2. Train the ShimController to minimize FieldUniformityLoss.
        3. Validate each epoch; save the best checkpoint.
        4. Early stop if validation loss plateaus.
    """

    def __init__(self, config: SimConfig, tracker=None) -> None:
        self.config = config
        self.device = config.resolve_device()
        self._tracker = tracker
        self._run_id: int | None = None

        # Build physics environment
        self.env = FieldEnvironment(config)

        # Build model
        self.model = build_controller(
            config.grid.size, config.model, config.coils
        ).to(self.device)

        # Pre-bake influence tensor on device for differentiable coil field
        self.influence_tensor = torch.tensor(
            self.env.coils.influence_matrix,
            dtype=torch.float32,
            device=self.device,
        )

        # Active zone mask on device
        self.active_mask = torch.tensor(
            self.env.grid.active_zone_mask,
            dtype=torch.bool,
            device=self.device,
        )

        # Loss function
        self.loss_fn = FieldUniformityLoss(
            self.active_mask,
            config.training.current_penalty_weight,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        self.scheduler = self._build_scheduler()
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _build_scheduler(self):
        tc = self.config.training
        if tc.lr_scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=tc.epochs)
        elif tc.lr_scheduler == "step":
            return StepLR(
                self.optimizer, step_size=tc.lr_step_size, gamma=tc.lr_gamma
            )
        return None

    def generate_dataset(self) -> tuple[DataLoader, DataLoader]:
        """Generate training and validation dataloaders."""
        tc = self.config.training
        print(
            f"[Trainer] Generating {tc.num_samples} training samples on "
            f"{self.device}…"
        )
        t0 = time.time()

        distorted, _ = self.env.generate_training_data(tc.num_samples)
        X = torch.tensor(distorted, dtype=torch.float32).unsqueeze(1)  # (N,1,H,W)

        val_n = int(tc.num_samples * tc.val_split)
        train_n = tc.num_samples - val_n

        train_ds = TensorDataset(X[:train_n])
        val_ds = TensorDataset(X[train_n:])

        print(
            f"[Trainer] Dataset ready in {time.time()-t0:.1f}s "
            f"(train={train_n}, val={val_n})"
        )

        train_loader = DataLoader(
            train_ds, batch_size=tc.batch_size, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(val_ds, batch_size=tc.batch_size, pin_memory=True)
        return train_loader, val_loader

    def _forward_step(
        self, batch_field: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Single forward pass: field → currents → coil field → net field → loss.

        Args:
            batch_field: (B, 1, H, W) on device.

        Returns:
            loss: scalar tensor (differentiable).
            metrics: dict of float values.
        """
        batch_field = batch_field.to(self.device)
        currents = self.model(batch_field)  # (B, num_coils)

        # Differentiable coil field computation
        coil_field = torch.einsum(
            "bc,cij->bij", currents, self.influence_tensor
        )  # (B, H, W)

        net_field = batch_field.squeeze(1) + coil_field  # (B, H, W)
        loss, metrics = self.loss_fn(net_field, currents)
        return loss, metrics

    def train(self) -> dict:
        """
        Full training loop.

        Returns:
            history dict with lists: train_loss, val_loss per epoch.
        """
        train_loader, val_loader = self.generate_dataset()
        tc = self.config.training
        ckpt_dir = Path(tc.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        if self._tracker is not None:
            self._run_id = self._tracker.start_run(
                arch=self.config.model.architecture.value,
                config=self.config,
                notes=f"lr={tc.learning_rate} wd={tc.weight_decay}",
            )

        print(
            f"[Trainer] Training {type(self.model).__name__} "
            f"for {tc.epochs} epochs on {self.device}"
        )

        for epoch in range(tc.epochs):
            # ── Training ──────────────────────────────────────────────────
            self.model.train()
            epoch_train_loss = 0.0
            for (batch,) in train_loader:
                self.optimizer.zero_grad()
                loss, _ = self._forward_step(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_train_loss += loss.item()

            epoch_train_loss /= len(train_loader)

            # ── Validation ────────────────────────────────────────────────
            self.model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    loss, _ = self._forward_step(batch)
                    epoch_val_loss += loss.item()
            epoch_val_loss /= len(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)

            if self._tracker is not None and self._run_id is not None:
                self._tracker.log_epoch(
                    self._run_id,
                    epoch=epoch + 1,
                    train_loss=epoch_train_loss,
                    val_loss=epoch_val_loss,
                )

            print(
                f"  Epoch {epoch+1:3d}/{tc.epochs}  "
                f"train={epoch_train_loss:.4e}  val={epoch_val_loss:.4e}"
            )

            # ── Checkpoint ────────────────────────────────────────────────
            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                self.patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "val_loss": epoch_val_loss,
                    },
                    ckpt_dir / "best.pt",
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= tc.early_stopping_patience:
                    print(
                        f"[Trainer] Early stopping at epoch {epoch+1} "
                        f"(no improvement for {tc.early_stopping_patience} epochs)"
                    )
                    break

        if self._tracker is not None and self._run_id is not None:
            self._tracker.finish_run(self._run_id, best_val_loss=self.best_val_loss)

        print(f"[Trainer] Best val loss: {self.best_val_loss:.4e}")
        return history

    def load_best(self) -> None:
        """Load the best checkpoint."""
        ckpt_path = Path(self.config.training.checkpoint_dir) / "best.pt"
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        print(f"[Trainer] Loaded best model from epoch {checkpoint['epoch']+1}")
