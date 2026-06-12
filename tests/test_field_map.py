"""Tests for the calibration FieldMap module and reference-map wiring.

Covers the .npz container (save/load/regrid/compare/uniformity) and the
calibration.reference_map_path → FieldEnvironment base-field path that lets
measured hardware maps replace the simulated B₀.
"""
from __future__ import annotations

import argparse

import numpy as np
import pytest

from nv_maser.calibration import (
    FieldMap,
    compare_maps,
    load_field_map,
    regrid,
    save_field_map,
    simulated_field_map,
    uniformity_ppm,
)
from nv_maser.config import SimConfig
from nv_maser.main import cmd_fieldmap
from nv_maser.physics.environment import FieldEnvironment

_B0 = 0.05  # Tesla — matches SimConfig field.b0_tesla default


def _axis(n: int = 64, extent_mm: float = 10.0) -> np.ndarray:
    half = extent_mm / 2.0
    return np.linspace(-half, half, n, dtype=np.float32)


def _make_map(
    n: int = 64,
    extent_mm: float = 10.0,
    b0: float = _B0,
    gradient_t_per_mm: float = 0.0,
    active_radius_mm: float = 3.0,
) -> FieldMap:
    ax = _axis(n, extent_mm)
    xx, _ = np.meshgrid(ax, ax)
    b_z = (b0 + gradient_t_per_mm * xx).astype(np.float32)
    return FieldMap(
        b_z=b_z,
        x_mm=ax,
        y_mm=ax.copy(),
        b0_nominal_tesla=b0,
        active_radius_mm=active_radius_mm,
    )


# ═══════════════════════════════════════════════════════════════════
# FieldMap validation
# ═══════════════════════════════════════════════════════════════════

class TestFieldMapValidation:
    def test_shape_mismatch_raises(self) -> None:
        ax = _axis(8)
        with pytest.raises(ValueError, match="inconsistent"):
            FieldMap(
                b_z=np.zeros((8, 9), dtype=np.float32),
                x_mm=ax,
                y_mm=ax.copy(),
                b0_nominal_tesla=_B0,
                active_radius_mm=3.0,
            )

    def test_non_2d_raises(self) -> None:
        ax = _axis(8)
        with pytest.raises(ValueError, match="2-D"):
            FieldMap(
                b_z=np.zeros(8, dtype=np.float32),
                x_mm=ax,
                y_mm=ax.copy(),
                b0_nominal_tesla=_B0,
                active_radius_mm=3.0,
            )

    def test_nonpositive_b0_raises(self) -> None:
        ax = _axis(8)
        with pytest.raises(ValueError, match="b0_nominal_tesla"):
            FieldMap(
                b_z=np.zeros((8, 8), dtype=np.float32),
                x_mm=ax,
                y_mm=ax.copy(),
                b0_nominal_tesla=0.0,
                active_radius_mm=3.0,
            )

    def test_nonpositive_radius_raises(self) -> None:
        ax = _axis(8)
        with pytest.raises(ValueError, match="active_radius_mm"):
            FieldMap(
                b_z=np.zeros((8, 8), dtype=np.float32),
                x_mm=ax,
                y_mm=ax.copy(),
                b0_nominal_tesla=_B0,
                active_radius_mm=0.0,
            )


# ═══════════════════════════════════════════════════════════════════
# Save / load round-trip
# ═══════════════════════════════════════════════════════════════════

class TestSaveLoadRoundTrip:
    def test_values_preserved(self, tmp_path) -> None:
        fm = _make_map(gradient_t_per_mm=1e-5)
        path = tmp_path / "map.npz"
        save_field_map(path, fm)
        loaded = load_field_map(path)
        np.testing.assert_allclose(loaded.b_z, fm.b_z)
        np.testing.assert_allclose(loaded.x_mm, fm.x_mm)
        np.testing.assert_allclose(loaded.y_mm, fm.y_mm)
        assert loaded.b0_nominal_tesla == pytest.approx(fm.b0_nominal_tesla)
        assert loaded.active_radius_mm == pytest.approx(fm.active_radius_mm)

    def test_metadata_preserved(self, tmp_path) -> None:
        fm = _make_map()
        fm.source = "measurement"
        fm.timestamp = "2026-06-12T00:00:00"
        fm.notes = "bench test"
        path = tmp_path / "map.npz"
        save_field_map(path, fm)
        loaded = load_field_map(path)
        assert loaded.source == "measurement"
        assert loaded.timestamp == "2026-06-12T00:00:00"
        assert loaded.notes == "bench test"

    def test_missing_required_key_raises(self, tmp_path) -> None:
        path = tmp_path / "broken.npz"
        np.savez(path, x_mm=_axis(8), y_mm=_axis(8))
        with pytest.raises(ValueError, match="missing keys"):
            load_field_map(path)


# ═══════════════════════════════════════════════════════════════════
# uniformity_ppm
# ═══════════════════════════════════════════════════════════════════

class TestUniformityPpm:
    def test_uniform_field_is_zero_ppm(self) -> None:
        assert uniformity_ppm(_make_map()) == pytest.approx(0.0)

    def test_linear_gradient_matches_analytic(self) -> None:
        grad = 1e-6  # T/mm
        fm = _make_map(gradient_t_per_mm=grad, active_radius_mm=3.0)
        # Peak-to-peak inside a radius-r circle of a linear x-gradient ≈ 2·g·r
        expected_ppm = 2 * grad * 3.0 / _B0 * 1e6
        # Grid sampling means the extreme pixels sit slightly inside r
        assert uniformity_ppm(fm) == pytest.approx(expected_ppm, rel=0.06)

    def test_radius_override(self) -> None:
        grad = 1e-6
        fm = _make_map(gradient_t_per_mm=grad, active_radius_mm=3.0)
        # Halving the radius roughly halves the peak-to-peak
        full = uniformity_ppm(fm)
        half = uniformity_ppm(fm, active_radius_mm=1.5)
        assert half == pytest.approx(full / 2, rel=0.1)

    def test_empty_active_zone_raises(self) -> None:
        fm = _make_map(n=64, extent_mm=10.0)
        # 64-point grid has no pixel at exactly (0,0); a tiny radius is empty
        with pytest.raises(ValueError, match="No pixels"):
            uniformity_ppm(fm, active_radius_mm=0.01)


# ═══════════════════════════════════════════════════════════════════
# regrid / compare_maps
# ═══════════════════════════════════════════════════════════════════

class TestRegrid:
    def test_identity_on_same_axes(self) -> None:
        fm = _make_map(gradient_t_per_mm=1e-5)
        out = regrid(fm, fm.x_mm, fm.y_mm)
        np.testing.assert_allclose(out.b_z, fm.b_z, atol=1e-9)

    def test_linear_field_interpolates_exactly(self) -> None:
        # Bilinear interpolation reproduces a linear field exactly
        coarse = _make_map(n=33, extent_mm=12.0, gradient_t_per_mm=1e-5)
        fine_ax = _axis(64, 10.0)
        out = regrid(coarse, fine_ax, fine_ax)
        xx, _ = np.meshgrid(fine_ax, fine_ax)
        expected = (_B0 + 1e-5 * xx).astype(np.float32)
        np.testing.assert_allclose(out.b_z, expected, atol=1e-7)


class TestCompareMaps:
    def test_identical_maps(self) -> None:
        fm = _make_map(gradient_t_per_mm=1e-5)
        result = compare_maps(fm, _make_map(gradient_t_per_mm=1e-5))
        assert result.rms_residual_tesla == pytest.approx(0.0, abs=1e-12)
        assert result.correlation == pytest.approx(1.0)

    def test_known_offset(self) -> None:
        fm_a = _make_map(gradient_t_per_mm=1e-5)
        fm_b = _make_map(gradient_t_per_mm=1e-5)
        fm_b.b_z = fm_b.b_z + np.float32(1e-4)
        result = compare_maps(fm_b, fm_a)
        assert result.mean_residual_tesla == pytest.approx(1e-4, rel=1e-3)
        assert result.max_abs_residual_tesla == pytest.approx(1e-4, rel=1e-3)


# ═══════════════════════════════════════════════════════════════════
# simulated_field_map
# ═══════════════════════════════════════════════════════════════════

class TestSimulatedFieldMap:
    def test_matches_environment_base_field(self) -> None:
        config = SimConfig()
        fm = simulated_field_map(config)
        env = FieldEnvironment(config)
        np.testing.assert_allclose(fm.b_z, env.base_field)
        assert fm.source == "simulation"

    def test_disturbance_changes_field(self) -> None:
        config = SimConfig()
        clean = simulated_field_map(config)
        noisy = simulated_field_map(
            config, add_disturbance=True, disturbance_seed=7
        )
        assert not np.allclose(clean.b_z, noisy.b_z)


# ═══════════════════════════════════════════════════════════════════
# Reference-map wiring: calibration.reference_map_path → base field
# ═══════════════════════════════════════════════════════════════════

class TestReferenceBaseFieldWiring:
    def _config_with_map(self, path) -> SimConfig:
        return SimConfig(calibration={"reference_map_path": str(path)})

    def test_environment_uses_reference_map(self, tmp_path) -> None:
        fm = _make_map(gradient_t_per_mm=2e-6)
        path = tmp_path / "ref.npz"
        save_field_map(path, fm)

        env = FieldEnvironment(self._config_with_map(path))
        np.testing.assert_allclose(env.base_field, fm.b_z, atol=1e-9)

    def test_training_data_built_on_reference_map(self, tmp_path) -> None:
        fm = _make_map(gradient_t_per_mm=2e-6)
        path = tmp_path / "ref.npz"
        save_field_map(path, fm)

        env = FieldEnvironment(self._config_with_map(path))
        distorted, disturbances = env.generate_training_data(3)
        # distorted = base + disturbance, so the difference recovers the map
        np.testing.assert_allclose(
            distorted - disturbances,
            np.broadcast_to(fm.b_z, distorted.shape),
            atol=1e-9,
        )

    def test_coarser_map_is_regridded(self, tmp_path) -> None:
        # 33-point map over a larger extent; linear field regrids exactly
        fm = _make_map(n=33, extent_mm=12.0, gradient_t_per_mm=2e-6)
        path = tmp_path / "coarse.npz"
        save_field_map(path, fm)

        env = FieldEnvironment(self._config_with_map(path))
        ax = _axis(64, 10.0)
        xx, _ = np.meshgrid(ax, ax)
        expected = (_B0 + 2e-6 * xx).astype(np.float32)
        assert env.base_field.shape == (64, 64)
        np.testing.assert_allclose(env.base_field, expected, atol=1e-7)

    def test_b0_mismatch_raises(self, tmp_path) -> None:
        fm = _make_map(b0=0.06)  # 20% off the 0.05 T config default
        path = tmp_path / "wrong_b0.npz"
        save_field_map(path, fm)

        with pytest.raises(ValueError, match="field.b0_tesla"):
            FieldEnvironment(self._config_with_map(path))

    def test_b0_within_tolerance_accepted(self, tmp_path) -> None:
        fm = _make_map(b0=0.0512)  # 2.4% off — a realistic magnet
        path = tmp_path / "close_b0.npz"
        save_field_map(path, fm)

        env = FieldEnvironment(self._config_with_map(path))
        np.testing.assert_allclose(env.base_field, fm.b_z, atol=1e-9)

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="reference_map_path"):
            FieldEnvironment(self._config_with_map(tmp_path / "nope.npz"))

    def test_empty_path_uses_simulated_field(self) -> None:
        config = SimConfig()
        assert config.calibration.reference_map_path == ""
        env = FieldEnvironment(config)
        sim = simulated_field_map(config)
        np.testing.assert_allclose(env.base_field, sim.b_z)


# ═══════════════════════════════════════════════════════════════════
# fieldmap CLI command
# ═══════════════════════════════════════════════════════════════════

class TestCmdFieldmap:
    def test_writes_loadable_map(self, tmp_path, capsys) -> None:
        config = SimConfig()
        out = tmp_path / "cli_map.npz"
        args = argparse.Namespace(
            output=str(out), add_disturbance=False, seed=None
        )
        cmd_fieldmap(config, args)

        assert out.exists()
        fm = load_field_map(out)
        env = FieldEnvironment(config)
        np.testing.assert_allclose(fm.b_z, env.base_field)
        assert "uniformity" in capsys.readouterr().out

    def test_round_trip_through_environment(self, tmp_path) -> None:
        """fieldmap output is directly consumable as a reference map."""
        config = SimConfig()
        out = tmp_path / "round_trip.npz"
        cmd_fieldmap(
            config,
            argparse.Namespace(output=str(out), add_disturbance=False, seed=None),
        )

        env = FieldEnvironment(
            SimConfig(calibration={"reference_map_path": str(out)})
        )
        np.testing.assert_allclose(
            env.base_field, FieldEnvironment(config).base_field, atol=1e-9
        )
