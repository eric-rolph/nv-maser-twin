"""
CLI entry point for the NV Maser Digital Twin.

Usage::

    python -m nv_maser train              # Train the shimming controller
    python -m nv_maser demo               # Run the real-time dashboard with trained model
    python -m nv_maser evaluate           # Run evaluation metrics on test disturbances
    python -m nv_maser visualize-coils    # Show coil influence matrix plots
"""
from __future__ import annotations

import argparse

from .config import SimConfig


def _deep_merge_config(base: SimConfig, overrides: dict) -> SimConfig:
    """Deep merge YAML overrides on top of the default SimConfig.

    Only keys present in *overrides* are changed; nested sub-configs are merged
    field-by-field rather than completely replaced.  This means a YAML file that
    only contains ``training.epochs: 10`` will not wipe out the rest of the
    training defaults.
    """
    merged = base.model_dump()
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return SimConfig(**merged)


def cmd_train(config: SimConfig) -> None:
    from .model.training import Trainer
    from .viz.plots import plot_training_history

    trainer = Trainer(config)
    history = trainer.train()
    plot_training_history(history, save_path="training_curves.png")


def cmd_demo(config: SimConfig) -> None:
    from .model.training import Trainer
    from .viz.dashboard import run_dashboard

    trainer = Trainer(config)
    trainer.load_best()
    trainer.model.eval()
    run_dashboard(trainer.env, trainer.model, trainer.influence_tensor, config)


def cmd_evaluate(config: SimConfig) -> None:
    import numpy as np
    import torch
    from .model.training import Trainer

    trainer = Trainer(config)
    trainer.load_best()
    trainer.model.eval()

    n_test = 500
    print(f"\n[Evaluate] Generating {n_test} test disturbances…")
    test_fields, _ = trainer.env.generate_training_data(n_test)
    X = torch.tensor(test_fields, dtype=torch.float32).unsqueeze(1).to(trainer.device)

    with torch.no_grad():
        currents = trainer.model(X)
        coil_fields = torch.einsum("bc,cij->bij", currents, trainer.influence_tensor)
        net = X.squeeze(1) + coil_fields

    mask = trainer.active_mask.cpu()
    for label, field in [("Before correction", X.squeeze(1)), ("After correction", net)]:
        active = field.cpu()[:, mask].numpy()
        stds = np.std(active, axis=1)
        print(f"\n  {label}:")
        print(f"    Mean std:       {np.mean(stds):.2e} T")
        print(f"    Median std:     {np.median(stds):.2e} T")
        print(f"    Worst-case std: {np.max(stds):.2e} T")

    before_stds = np.std(X.squeeze(1).cpu()[:, mask].numpy(), axis=1)
    after_stds = np.std(net.cpu()[:, mask].numpy(), axis=1)
    improvement = np.mean(before_stds) / np.mean(after_stds)
    print(f"\n  Average improvement factor: {improvement:.1f}x")


def cmd_visualize_coils(config: SimConfig) -> None:
    import matplotlib.pyplot as plt
    from .physics.environment import FieldEnvironment
    from .viz.plots import plot_coil_influence, plot_disturbance_spectrum

    env = FieldEnvironment(config)
    plot_coil_influence(env.coils, save_path="coil_influence.png")
    disturbance = env.disturbance_gen.generate()
    plot_disturbance_spectrum(disturbance, save_path="disturbance_spectrum.png")
    plt.show()


def cmd_dataset(config: SimConfig, args) -> None:
    from .data.dataset import build_dataset

    num_samples = args.num_samples or config.training.num_samples
    ds = build_dataset(
        config,
        num_samples=num_samples,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild,
    )
    print(f"[dataset] Built dataset with {len(ds)} samples -> {args.cache_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="NV Maser Digital Twin")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config override"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument("--arch", type=str, default=None, choices=["cnn", "mlp"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Train the shimming controller")
    subparsers.add_parser("demo", help="Run the real-time dashboard")
    subparsers.add_parser("evaluate", help="Evaluate on test disturbances")
    subparsers.add_parser("visualize-coils", help="Plot coil influence matrix")

    dataset_parser = subparsers.add_parser("dataset", help="Build and cache training dataset")
    dataset_parser.add_argument("--num-samples", type=int, default=None,
                                 help="Number of samples (overrides config)")
    dataset_parser.add_argument("--cache-dir", type=str, default="dataset_cache",
                                 help="Directory for cached .npz files")
    dataset_parser.add_argument("--force-rebuild", action="store_true",
                                 help="Ignore cache and rebuild from scratch")

    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI inference server")
    serve_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Bind port (default: 8000)"
    )

    args = parser.parse_args()

    config = SimConfig()

    if args.config:
        import yaml
        from pydantic import ValidationError

        with open(args.config) as f:
            overrides = yaml.safe_load(f)
        try:
            config = _deep_merge_config(config, overrides)
        except ValidationError as e:
            print(f"Config validation error:\n{e}")
            raise SystemExit(1)

    config.device = args.device
    if args.arch:
        config.model.architecture = args.arch  # type: ignore[assignment]
    if args.epochs:
        config.training.epochs = args.epochs
    if args.samples:
        config.training.num_samples = args.samples

    if args.command == "dataset":
        cmd_dataset(config, args)
        return

    if args.command == "serve":
        import uvicorn
        uvicorn.run(
            "nv_maser.api.server:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
        return

    commands = {
        "train": cmd_train,
        "demo": cmd_demo,
        "evaluate": cmd_evaluate,
        "visualize-coils": cmd_visualize_coils,
    }
    commands[args.command](config)


if __name__ == "__main__":
    main()
