"""
Export the shimming controller to ONNX format.

Usage:
    python scripts/export_onnx.py [--output checkpoints/model.onnx]
                                   [--checkpoint checkpoints/best.pt]
                                   [--arch cnn] [--opset 17]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/export_onnx.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the NV Maser shimming controller to ONNX"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/model.onnx",
        help="Output .onnx path (default: checkpoints/model.onnx)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="Checkpoint to load; skipped if file is missing (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["cnn", "mlp", "lstm"],
        default="cnn",
        help="Model architecture to export (default: cnn)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    args = parser.parse_args()

    from nv_maser.config import ModelArchitecture, SimConfig
    from nv_maser.export import export_model

    config = SimConfig()
    config.model.architecture = ModelArchitecture(args.arch)

    result = export_model(
        config,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        opset_version=args.opset,
    )

    print(f"[export] ONNX model written to: {result.path}")
    print(f"         arch        = {result.arch}")
    print(f"         grid_size   = {result.grid_size}")
    print(f"         num_coils   = {result.num_coils}")
    print(f"         opset       = {result.opset}")
    print(f"         input_shape = {result.input_shape}")
    print(f"         output_shape= {result.output_shape}")

    # Optional onnxruntime verification
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(result.path))
        dummy = np.zeros(result.input_shape, dtype=np.float32)
        outputs = sess.run(None, {"distorted_field": dummy})
        out_shape = tuple(outputs[0].shape)
        if out_shape == result.output_shape:
            print(f"\n[onnxruntime] Verification PASSED — output shape: {out_shape}")
        else:
            print(
                f"\n[onnxruntime] WARNING: expected shape {result.output_shape}, got {out_shape}"
            )
    except ImportError:
        print("\n[onnxruntime] Not installed — skipping runtime verification.")


if __name__ == "__main__":
    main()
