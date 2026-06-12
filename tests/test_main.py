"""Unit tests for nv_maser.main CLI entry point.

These tests exercise:
  • _deep_merge_config       — pure function, no side-effects
  • main()                   — argparse routing, config flag overrides, YAML loading
  • cmd_train / cmd_evaluate / cmd_dataset / cmd_visualize_coils (unit-level, all
    heavy dependencies mocked via unittest.mock.patch)
  • export and serve dispatch paths through main()

Coverage goal: lift main.py from 0 % → ~85 %.

Run with:
    pytest tests/test_main.py -v
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
import torch

from nv_maser.config import SimConfig
from nv_maser.main import (
    _deep_merge_config,
    cmd_dataset,
    cmd_evaluate,
    cmd_train,
    cmd_visualize_coils,
    main,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfg() -> SimConfig:
    """Return a fresh default SimConfig."""
    return SimConfig()


# ─────────────────────────────────────────────────────────────────────────────
# 1. _deep_merge_config
# ─────────────────────────────────────────────────────────────────────────────

class TestDeepMergeConfig:
    """Pure-function tests — no I/O or side-effects."""

    def test_empty_overrides_is_noop(self):
        base = _cfg()
        merged = _deep_merge_config(base, {})
        assert merged == base

    def test_flat_key_override(self):
        base = _cfg()
        merged = _deep_merge_config(base, {"device": "cpu"})
        assert merged.device == "cpu"

    def test_nested_partial_override_preserves_other_fields(self):
        base = _cfg()
        original_batch = base.training.batch_size
        merged = _deep_merge_config(base, {"training": {"epochs": 3}})
        assert merged.training.epochs == 3
        assert merged.training.batch_size == original_batch  # untouched

    def test_nested_override_grid_size(self):
        base = _cfg()
        merged = _deep_merge_config(base, {"grid": {"size": 32}})
        assert merged.grid.size == 32
        # physical_extent_mm should remain default
        assert merged.grid.physical_extent_mm == base.grid.physical_extent_mm

    def test_multiple_flat_overrides(self):
        base = _cfg()
        merged = _deep_merge_config(base, {"device": "cuda"})
        assert merged.device == "cuda"
        # Training settings untouched
        assert merged.training.epochs == base.training.epochs

    def test_returns_simconfig_instance(self):
        base = _cfg()
        merged = _deep_merge_config(base, {})
        assert isinstance(merged, SimConfig)

    def test_original_base_is_unchanged(self):
        base = _cfg()
        orig_device = base.device
        _deep_merge_config(base, {"device": "cpu"})
        assert base.device == orig_device  # base is immutable / not mutated

    def test_override_training_learning_rate(self):
        base = _cfg()
        merged = _deep_merge_config(base, {"training": {"learning_rate": 0.01}})
        assert merged.training.learning_rate == pytest.approx(0.01)
        assert merged.training.epochs == base.training.epochs  # unchanged


# ─────────────────────────────────────────────────────────────────────────────
# 2. main() — sub-command dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestMainDispatch:
    """Verify that main() dispatches each sub-command to the right handler."""

    def test_train_dispatch(self):
        with patch.object(sys, "argv", ["nv_maser", "train"]), \
             patch("nv_maser.main.cmd_train") as mock_cmd:
            main()
        mock_cmd.assert_called_once()

    def test_evaluate_dispatch(self):
        with patch.object(sys, "argv", ["nv_maser", "evaluate"]), \
             patch("nv_maser.main.cmd_evaluate") as mock_cmd:
            main()
        mock_cmd.assert_called_once()

    def test_visualize_coils_dispatch(self):
        with patch.object(sys, "argv", ["nv_maser", "visualize-coils"]), \
             patch("nv_maser.main.cmd_visualize_coils") as mock_cmd:
            main()
        mock_cmd.assert_called_once()

    def test_dataset_dispatch(self):
        with patch.object(sys, "argv", ["nv_maser", "dataset", "--cache-dir", "tmp_cache"]), \
             patch("nv_maser.main.cmd_dataset") as mock_cmd:
            main()
        mock_cmd.assert_called_once()

    def test_export_dispatch(self):
        fake_result = MagicMock()
        with patch.object(sys, "argv", ["nv_maser", "export"]), \
             patch("nv_maser.export.export_model", return_value=fake_result):
            main()  # should not raise

    def test_serve_dispatch(self):
        with patch.object(sys, "argv", ["nv_maser", "serve"]), \
             patch("uvicorn.run") as mock_run:
            main()
        mock_run.assert_called_once_with(
            "nv_maser.api.server:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. main() — CLI flag overrides applied to SimConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestMainConfigOverrides:
    """Verify --device / --arch / --epochs / --samples flags mutate config."""

    def _dispatch_and_capture(self, *extra_args) -> SimConfig:
        """Run main() with 'train' sub-command; capture config received by cmd_train."""
        captured: dict[str, SimConfig] = {}

        def _spy(config: SimConfig) -> None:
            captured["config"] = config

        with patch.object(sys, "argv", ["nv_maser"] + list(extra_args) + ["train"]), \
             patch("nv_maser.main.cmd_train", side_effect=_spy):
            main()
        return captured["config"]

    def test_device_override(self):
        config = self._dispatch_and_capture("--device", "cpu")
        assert config.device == "cpu"

    def test_arch_override(self):
        config = self._dispatch_and_capture("--arch", "mlp")
        # After CLI assignment, architecture is a plain string (not enum)
        arch_val = getattr(config.model.architecture, "value", config.model.architecture)
        assert arch_val == "mlp"

    def test_epochs_override(self):
        config = self._dispatch_and_capture("--epochs", "5")
        assert config.training.epochs == 5

    def test_samples_override(self):
        config = self._dispatch_and_capture("--samples", "100")
        assert config.training.num_samples == 100

    def test_no_overrides_uses_defaults(self):
        defaults = SimConfig()
        config = self._dispatch_and_capture()
        assert config.training.epochs == defaults.training.epochs
        assert config.device == defaults.device


# ─────────────────────────────────────────────────────────────────────────────
# 4. main() — YAML config loading
# ─────────────────────────────────────────────────────────────────────────────

class TestMainYAMLConfig:
    def _dispatch_and_capture(self, yaml_path: str) -> SimConfig:
        captured: dict[str, SimConfig] = {}

        def _spy(config: SimConfig) -> None:
            captured["config"] = config

        with patch.object(sys, "argv", ["nv_maser", "--config", yaml_path, "train"]), \
             patch("nv_maser.main.cmd_train", side_effect=_spy):
            main()
        return captured["config"]

    def test_valid_yaml_flat_override(self, tmp_path):
        # Use a field not re-applied from CLI flags (device is overwritten by args.device)
        yaml_file = tmp_path / "override.yaml"
        yaml_file.write_text("training:\n  batch_size: 128\n")
        config = self._dispatch_and_capture(str(yaml_file))
        assert config.training.batch_size == 128

    def test_valid_yaml_nested_override(self, tmp_path):
        yaml_file = tmp_path / "nested.yaml"
        yaml_file.write_text("training:\n  epochs: 7\n")
        config = self._dispatch_and_capture(str(yaml_file))
        assert config.training.epochs == 7

    def test_yaml_preserves_non_overridden_fields(self, tmp_path):
        yaml_file = tmp_path / "partial.yaml"
        yaml_file.write_text("training:\n  epochs: 2\n")
        config = self._dispatch_and_capture(str(yaml_file))
        # batch_size not in YAML → should remain at default
        assert config.training.batch_size == SimConfig().training.batch_size

    def test_yaml_validation_error_exits_with_code_1(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        # grid.size must be an int; "not_an_int" triggers Pydantic ValidationError
        yaml_file.write_text("grid:\n  size: not_an_int\n")
        with patch.object(sys, "argv", ["nv_maser", "--config", str(yaml_file), "train"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. cmd_train
# ─────────────────────────────────────────────────────────────────────────────

class TestCmdTrain:

    def _make_mock_trainer(self) -> MagicMock:
        mt = MagicMock()
        mt.train.return_value = {"train_loss": [0.5, 0.4], "val_loss": [0.6, 0.45]}
        return mt

    def test_creates_trainer_with_config(self):
        config = _cfg()
        mock_trainer = self._make_mock_trainer()
        with patch("nv_maser.model.training.Trainer", return_value=mock_trainer) as MockT, \
             patch("nv_maser.tracking.ExperimentTracker", return_value=MagicMock()), \
             patch("nv_maser.viz.plots.plot_training_history"):
            cmd_train(config)
        MockT.assert_called_once_with(config, tracker=ANY)

    def test_calls_trainer_train(self):
        config = _cfg()
        mock_trainer = self._make_mock_trainer()
        with patch("nv_maser.model.training.Trainer", return_value=mock_trainer), \
             patch("nv_maser.tracking.ExperimentTracker", return_value=MagicMock()), \
             patch("nv_maser.viz.plots.plot_training_history"):
            cmd_train(config)
        mock_trainer.train.assert_called_once()

    def test_calls_plot_training_history(self):
        config = _cfg()
        mock_trainer = self._make_mock_trainer()
        with patch("nv_maser.model.training.Trainer", return_value=mock_trainer), \
             patch("nv_maser.tracking.ExperimentTracker", return_value=MagicMock()), \
             patch("nv_maser.viz.plots.plot_training_history") as mock_plot:
            cmd_train(config)
        mock_plot.assert_called_once()
        call_kwargs = mock_plot.call_args.kwargs
        assert call_kwargs.get("save_path") == "training_curves.png"


# ─────────────────────────────────────────────────────────────────────────────
# 6. cmd_evaluate
# ─────────────────────────────────────────────────────────────────────────────

class TestCmdEvaluate:
    """Uses real torch ops on tiny tensors to cover all inline computations."""

    # Small but valid grid dimensions
    H, W, C = 4, 4, 2
    N_TEST = 500  # hard-coded inside cmd_evaluate

    def _build_mock_trainer(self) -> MagicMock:
        mt = MagicMock()
        mt.device = "cpu"
        mt.env.generate_training_data.return_value = (
            np.zeros((self.N_TEST, self.H, self.W), dtype=np.float32),
            None,
        )
        # model returns (N, C) currents tensor
        mt.model.return_value = torch.zeros(self.N_TEST, self.C)
        # influence_tensor shape: (C, H, W)
        mt.influence_tensor = torch.zeros(self.C, self.H, self.W)
        # active_mask shape: (H, W) bool — same layout as env.grid.active_zone_mask
        mt.active_mask = torch.ones(self.H, self.W, dtype=torch.bool)
        return mt

    def test_calls_load_best_and_eval(self):
        config = _cfg()
        mt = self._build_mock_trainer()
        with patch("nv_maser.model.training.Trainer", return_value=mt):
            cmd_evaluate(config)
        mt.load_best.assert_called_once()
        mt.model.eval.assert_called_once()

    def test_calls_generate_training_data_with_500(self):
        config = _cfg()
        mt = self._build_mock_trainer()
        with patch("nv_maser.model.training.Trainer", return_value=mt):
            cmd_evaluate(config)
        mt.env.generate_training_data.assert_called_once_with(500)

    def test_prints_before_and_after_correction(self, capsys):
        config = _cfg()
        mt = self._build_mock_trainer()
        with patch("nv_maser.model.training.Trainer", return_value=mt):
            cmd_evaluate(config)
        out = capsys.readouterr().out
        assert "Before correction" in out
        assert "After correction" in out

    def test_prints_improvement_factor(self, capsys):
        config = _cfg()
        mt = self._build_mock_trainer()
        with patch("nv_maser.model.training.Trainer", return_value=mt):
            cmd_evaluate(config)
        out = capsys.readouterr().out
        assert "improvement factor" in out


# ─────────────────────────────────────────────────────────────────────────────
# 7. cmd_dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestCmdDataset:

    def _make_args(
        self,
        num_samples: int | None = 10,
        cache_dir: str = "test_cache",
        force_rebuild: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            num_samples=num_samples,
            cache_dir=cache_dir,
            force_rebuild=force_rebuild,
        )

    def _mock_ds(self, size: int = 10) -> MagicMock:
        ds = MagicMock()
        ds.__len__ = lambda _: size
        return ds

    def test_calls_build_dataset(self):
        config = _cfg()
        with patch("nv_maser.data.dataset.build_dataset", return_value=self._mock_ds()) as mock_bd:
            cmd_dataset(config, self._make_args())
        mock_bd.assert_called_once()

    def test_passes_args_num_samples(self):
        config = _cfg()
        with patch("nv_maser.data.dataset.build_dataset", return_value=self._mock_ds()) as mock_bd:
            cmd_dataset(config, self._make_args(num_samples=42))
        assert mock_bd.call_args.kwargs["num_samples"] == 42

    def test_falls_back_to_config_num_samples_when_arg_is_none(self):
        config = _cfg()
        with patch("nv_maser.data.dataset.build_dataset", return_value=self._mock_ds()) as mock_bd:
            cmd_dataset(config, self._make_args(num_samples=None))
        assert mock_bd.call_args.kwargs["num_samples"] == config.training.num_samples

    def test_passes_cache_dir(self):
        config = _cfg()
        with patch("nv_maser.data.dataset.build_dataset", return_value=self._mock_ds()) as mock_bd:
            cmd_dataset(config, self._make_args(cache_dir="my_dir"))
        assert mock_bd.call_args.kwargs["cache_dir"] == "my_dir"

    def test_passes_force_rebuild(self):
        config = _cfg()
        with patch("nv_maser.data.dataset.build_dataset", return_value=self._mock_ds()) as mock_bd:
            cmd_dataset(config, self._make_args(force_rebuild=True))
        assert mock_bd.call_args.kwargs["force_rebuild"] is True

    def test_prints_cache_dir_in_output(self, capsys):
        config = _cfg()
        with patch("nv_maser.data.dataset.build_dataset", return_value=self._mock_ds()):
            cmd_dataset(config, self._make_args(cache_dir="my_special_cache"))
        out = capsys.readouterr().out
        assert "my_special_cache" in out


# ─────────────────────────────────────────────────────────────────────────────
# 8. export dispatch through main()
# ─────────────────────────────────────────────────────────────────────────────

class TestCmdExport:

    def _fake_result(self, path: str = "out.onnx") -> MagicMock:
        fr = MagicMock()
        fr.path = path
        fr.arch = "cnn"
        fr.grid_size = 64
        fr.num_coils = 8
        fr.opset = 17
        return fr

    def test_export_calls_export_model(self):
        fr = self._fake_result()
        with patch.object(sys, "argv", ["nv_maser", "export"]), \
             patch("nv_maser.export.export_model", return_value=fr) as mock_em:
            main()
        mock_em.assert_called_once()

    def test_export_prints_onnx_path(self, capsys):
        fr = self._fake_result(path="checkpoints/model.onnx")
        with patch.object(sys, "argv", ["nv_maser", "export"]), \
             patch("nv_maser.export.export_model", return_value=fr):
            main()
        out = capsys.readouterr().out
        assert "ONNX" in out

    def test_export_custom_output_path(self, tmp_path):
        out_path = str(tmp_path / "custom.onnx")
        fr = self._fake_result(path=out_path)
        with patch.object(sys, "argv", ["nv_maser", "export", "--output", out_path]), \
             patch("nv_maser.export.export_model", return_value=fr) as mock_em:
            main()
        assert mock_em.call_args.kwargs["output_path"] == out_path

    def test_export_custom_opset(self):
        fr = self._fake_result()
        with patch.object(sys, "argv", ["nv_maser", "export", "--opset", "16"]), \
             patch("nv_maser.export.export_model", return_value=fr) as mock_em:
            main()
        assert mock_em.call_args.kwargs["opset_version"] == 16

    def test_export_custom_checkpoint(self, tmp_path):
        ckpt = str(tmp_path / "best.pt")
        fr = self._fake_result()
        with patch.object(sys, "argv", ["nv_maser", "export", "--checkpoint", ckpt]), \
             patch("nv_maser.export.export_model", return_value=fr) as mock_em:
            main()
        assert mock_em.call_args.kwargs["checkpoint_path"] == ckpt


# ─────────────────────────────────────────────────────────────────────────────
# 9. serve dispatch through main()
# ─────────────────────────────────────────────────────────────────────────────

class TestCmdServe:

    def test_default_host_and_port(self):
        with patch.object(sys, "argv", ["nv_maser", "serve"]), \
             patch("uvicorn.run") as mock_run:
            main()
        mock_run.assert_called_once_with(
            "nv_maser.api.server:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
        )

    def test_custom_host_and_port(self):
        with patch.object(sys, "argv", ["nv_maser", "serve", "--host", "0.0.0.0", "--port", "9000"]), \
             patch("uvicorn.run") as mock_run:
            main()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["host"] == "0.0.0.0"
        assert call_kwargs["port"] == 9000

    def test_serve_passes_reload_false(self):
        with patch.object(sys, "argv", ["nv_maser", "serve"]), \
             patch("uvicorn.run") as mock_run:
            main()
        assert mock_run.call_args.kwargs["reload"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 10. cmd_visualize_coils
# ─────────────────────────────────────────────────────────────────────────────

class TestCmdVisualizeCoils:

    def _run_with_mocks(self, config: SimConfig) -> dict[str, MagicMock]:
        """Run cmd_visualize_coils with all heavy deps mocked; return mock objects."""
        mocks: dict[str, MagicMock] = {}

        mock_env = MagicMock()
        mocks["env"] = mock_env

        with patch("nv_maser.physics.environment.FieldEnvironment", return_value=mock_env) as MockFE, \
             patch("nv_maser.viz.plots.plot_coil_influence") as mock_pci, \
             patch("nv_maser.viz.plots.plot_disturbance_spectrum") as mock_pds, \
             patch("matplotlib.pyplot.show") as mock_show:
            mocks["MockFE"] = MockFE
            mocks["plot_coil_influence"] = mock_pci
            mocks["plot_disturbance_spectrum"] = mock_pds
            mocks["plt_show"] = mock_show
            cmd_visualize_coils(config)

        return mocks

    def test_creates_field_environment(self):
        config = _cfg()
        mocks = self._run_with_mocks(config)
        mocks["MockFE"].assert_called_once_with(config)

    def test_calls_plot_coil_influence(self):
        config = _cfg()
        mocks = self._run_with_mocks(config)
        mock_env = mocks["env"]
        mocks["plot_coil_influence"].assert_called_once_with(
            mock_env.coils, save_path="coil_influence.png"
        )

    def test_calls_plot_disturbance_spectrum(self):
        config = _cfg()
        mocks = self._run_with_mocks(config)
        mocks["plot_disturbance_spectrum"].assert_called_once()
        # second positional arg should be the generated disturbance
        called_kwargs = mocks["plot_disturbance_spectrum"].call_args.kwargs
        assert called_kwargs.get("save_path") == "disturbance_spectrum.png"

    def test_calls_plt_show(self):
        config = _cfg()
        mocks = self._run_with_mocks(config)
        mocks["plt_show"].assert_called_once()

    def test_generates_disturbance(self):
        config = _cfg()
        mocks = self._run_with_mocks(config)
        mock_env = mocks["env"]
        mock_env.disturbance_gen.generate.assert_called_once()
