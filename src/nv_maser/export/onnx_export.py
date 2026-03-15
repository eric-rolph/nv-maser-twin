"""ONNX export utilities for NV Maser shimming controllers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..config import SimConfig


@dataclass
class OnnxExportResult:
    path: Path
    grid_size: int
    num_coils: int
    arch: str
    opset: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


def export_model(
    config: "SimConfig",
    output_path: str | Path = "checkpoints/model.onnx",
    checkpoint_path: str | Path | None = "checkpoints/best.pt",
    opset_version: int = 17,
) -> OnnxExportResult:
    """
    Export a trained (or random-init) shimming controller to ONNX.

    Args:
        config: SimConfig used to build the model architecture.
        output_path: Where to write the .onnx file.
        checkpoint_path: Path to a .pt checkpoint.  If None or file missing,
            exports the random-initialised model.
        opset_version: ONNX opset (default 17, compatible with onnxruntime >=1.16).

    Returns:
        OnnxExportResult dataclass with export metadata.
    """
    from ..model.controller import build_controller

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_controller(config.grid.size, config.model, config.coils)

    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            saved = torch.load(ckpt, map_location="cpu", weights_only=True)
            state_dict = (
                saved.get("model_state", saved) if isinstance(saved, dict) else saved
            )
            model.load_state_dict(state_dict)

    model.eval()
    grid = config.grid.size
    dummy_input = torch.zeros(1, 1, grid, grid)

    with torch.no_grad():
        dummy_output = model(dummy_input)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["distorted_field"],
        output_names=["coil_currents"],
        dynamic_axes={
            "distorted_field": {0: "batch_size"},
            "coil_currents": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    return OnnxExportResult(
        path=output_path,
        grid_size=grid,
        num_coils=config.coils.num_coils,
        arch=config.model.architecture.value,
        opset=opset_version,
        input_shape=tuple(dummy_input.shape),
        output_shape=tuple(dummy_output.shape),
    )
