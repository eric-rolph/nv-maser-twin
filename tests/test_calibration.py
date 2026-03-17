"""
Tests for src/nv_maser/calibration/field_map.py

Covers:
  - FieldMap construction and validation
  - save / load round-trip
  - uniformity_ppm calculations
  - compare_maps statistics
  - regrid interpolation
  - simulated_field_map output shape and provenance
"""
from __future__ import annotations

import numpy as np
import pytest

from nv_maser.calibration import (
    CompareResult,
    FieldMap,
    compare_maps,
    load_field_map,
    regrid,
    save_field_map,
    simulated_field_map,
    uniformity_ppm,
)
from nv_maser.config import SimConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fm(
    size: int = 16,
    b0: float = 0.05,
    active_radius: float = 4.0,
    source: str = "simulation",
) -> FieldMap:
    """Create a uniform FieldMap for testing."""
    x_mm = np.linspace(-5.0, 5.0, size, dtype=np.float32)
    y_mm = np.linspace(-5.0, 5.0, size, dtype=np.float32)
    b_z = np.full((size, size), b0, dtype=np.float32)
    return FieldMap(
        b_z=b_z,
        x_mm=x_mm,
        y_mm=y_mm,
        b0_nominal_tesla=b0,
        active_radius_mm=active_radius,
        source=source,  # type: ignore[arg-type]
    )


def _make_gradient_fm(size: int = 32, b0: float = 0.05) -> FieldMap:
    """FieldMap with a known linear gradient across the active zone."""
    x_mm = np.linspace(-5.0, 5.0, size, dtype=np.float32)
    y_mm = np.linspace(-5.0, 5.0, size, dtype=np.float32)
    xx, _ = np.meshgrid(x_mm, y_mm)
    # Gradient: 0.001 T over [-5, +5] mm → ΔB = 0.001 T
    b_z = (b0 + 0.0001 * xx).astype(np.float32)
    return FieldMap(
        b_z=b_z,
        x_mm=x_mm,
        y_mm=y_mm,
        b0_nominal_tesla=b0,
        active_radius_mm=4.0,
    )


# ===========================================================================
# 1. FieldMap construction
# ===========================================================================


class TestFieldMapConstruction:
    def test_basic_construction(self):
        fm = _make_fm()
        assert fm.shape == (16, 16)
        assert fm.b0_nominal_tesla == pytest.approx(0.05)
        assert fm.source == "simulation"

    def test_shape_properties(self):
        fm = _make_fm(size=8)
        assert fm.height == 8
        assert fm.width == 8

    def test_non_square_shape(self):
        x_mm = np.linspace(-5, 5, 10, dtype=np.float32)
        y_mm = np.linspace(-5, 5, 20, dtype=np.float32)
        b_z = np.ones((20, 10), dtype=np.float32)
        fm = FieldMap(b_z=b_z, x_mm=x_mm, y_mm=y_mm, b0_nominal_tesla=0.05, active_radius_mm=3.0)
        assert fm.shape == (20, 10)

    def test_wrong_shape_raises(self):
        x_mm = np.linspace(-5, 5, 8, dtype=np.float32)
        y_mm = np.linspace(-5, 5, 16, dtype=np.float32)
        b_z = np.ones((8, 8), dtype=np.float32)  # wrong: should be (16, 8)
        with pytest.raises(ValueError, match="inconsistent"):
            FieldMap(b_z=b_z, x_mm=x_mm, y_mm=y_mm, b0_nominal_tesla=0.05, active_radius_mm=3.0)

    def test_1d_bz_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            FieldMap(
                b_z=np.ones(16),
                x_mm=np.linspace(-5, 5, 16, dtype=np.float32),
                y_mm=np.linspace(-5, 5, 16, dtype=np.float32),
                b0_nominal_tesla=0.05,
                active_radius_mm=3.0,
            )

    def test_non_positive_b0_raises(self):
        with pytest.raises(ValueError, match="b0_nominal_tesla must be positive"):
            _make_fm(b0=-0.05)

    def test_non_positive_active_radius_raises(self):
        x_mm = np.linspace(-5, 5, 8, dtype=np.float32)
        y_mm = np.linspace(-5, 5, 8, dtype=np.float32)
        with pytest.raises(ValueError, match="active_radius_mm must be positive"):
            FieldMap(b_z=np.ones((8, 8), dtype=np.float32), x_mm=x_mm, y_mm=y_mm,
                     b0_nominal_tesla=0.05, active_radius_mm=-1.0)

    def test_active_zone_mask_shape(self):
        fm = _make_fm(size=16, active_radius=4.0)
        mask = fm.active_zone_mask
        assert mask.shape == (16, 16)
        assert mask.dtype == np.bool_
        # Centre pixel should be inside
        assert mask[8, 8]

    def test_active_zone_mask_excludes_corners(self):
        fm = _make_fm(size=16, active_radius=4.0)
        mask = fm.active_zone_mask
        # Corners at ±5, ±5 mm are outside radius 4 mm
        assert not mask[0, 0]
        assert not mask[15, 15]


# ===========================================================================
# 2. Save / Load round-trip
# ===========================================================================


class TestSaveLoad:
    def test_roundtrip_uniform(self, tmp_path):
        fm = _make_fm(size=8, b0=0.05, active_radius=3.0)
        path = tmp_path / "fm.npz"
        save_field_map(path, fm)
        loaded = load_field_map(path)
        np.testing.assert_array_equal(loaded.b_z, fm.b_z)
        np.testing.assert_array_almost_equal(loaded.x_mm, fm.x_mm)
        np.testing.assert_array_almost_equal(loaded.y_mm, fm.y_mm)
        assert loaded.b0_nominal_tesla == pytest.approx(fm.b0_nominal_tesla)
        assert loaded.active_radius_mm == pytest.approx(fm.active_radius_mm)
        assert loaded.source == fm.source

    def test_roundtrip_metadata(self, tmp_path):
        fm = FieldMap(
            b_z=np.ones((4, 4), dtype=np.float32) * 0.05,
            x_mm=np.linspace(-2, 2, 4, dtype=np.float32),
            y_mm=np.linspace(-2, 2, 4, dtype=np.float32),
            b0_nominal_tesla=0.05,
            active_radius_mm=1.5,
            source="measurement",
            timestamp="2026-03-17T12:00:00Z",
            notes="Phase 1 Halbach measurement",
        )
        path = tmp_path / "meta.npz"
        save_field_map(path, fm)
        loaded = load_field_map(path)
        assert loaded.source == "measurement"
        assert loaded.timestamp == "2026-03-17T12:00:00Z"
        assert loaded.notes == "Phase 1 Halbach measurement"

    def test_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "fm.npz"
        save_field_map(deep_path, _make_fm())
        assert deep_path.exists()

    def test_load_missing_key_raises(self, tmp_path):
        path = tmp_path / "broken.npz"
        # Save without 'b_z'
        np.savez(str(path), x_mm=np.array([0.0]), y_mm=np.array([0.0]))
        with pytest.raises(ValueError, match="missing keys"):
            load_field_map(path)

    def test_roundtrip_gradient(self, tmp_path):
        fm = _make_gradient_fm(size=16)
        path = tmp_path / "grad.npz"
        save_field_map(path, fm)
        loaded = load_field_map(path)
        np.testing.assert_array_almost_equal(loaded.b_z, fm.b_z, decimal=6)


# ===========================================================================
# 3. uniformity_ppm
# ===========================================================================


class TestUniformityPpm:
    def test_perfect_uniform_field(self):
        fm = _make_fm(size=32, b0=0.05, active_radius=4.0)
        ppm = uniformity_ppm(fm)
        assert ppm == pytest.approx(0.0, abs=1e-9)

    def test_gradient_field_known_value(self):
        """Linear gradient 0.0001 T/mm over 8 mm diameter → ΔB = 0.0008 T.

        Active zone: radius 4 mm, so x spans [-4, +4] mm.
        ΔB = 0.0001 × 8 = 0.0008 T
        ppm = 0.0008 / 0.05 × 1e6 = 16000 ppm
        (approximate — exact value depends on discrete grid points)
        """
        fm = _make_gradient_fm(size=64, b0=0.05)
        ppm = uniformity_ppm(fm)
        # With 64 sample points, we expect ~16000 ppm ±5%
        assert 14000 < ppm < 18000, f"Expected ~16000 ppm, got {ppm:.1f}"

    def test_active_radius_override(self):
        fm = _make_gradient_fm(size=64, b0=0.05)
        # Smaller active zone → smaller ΔB → smaller ppm
        ppm_small = uniformity_ppm(fm, active_radius_mm=1.0)
        ppm_large = uniformity_ppm(fm, active_radius_mm=4.0)
        assert ppm_small < ppm_large

    def test_active_zone_too_small_raises(self):
        fm = _make_fm(size=8, active_radius=0.001)
        # Grid spans ±5 mm; radius 0.001 mm → no pixels inside
        with pytest.raises(ValueError, match="No pixels"):
            uniformity_ppm(fm)

    def test_measurement_source_works(self):
        fm = _make_fm(source="measurement")
        ppm = uniformity_ppm(fm)
        assert ppm == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# 4. compare_maps
# ===========================================================================


class TestCompareMaps:
    def test_identical_maps_zero_residual(self):
        fm = _make_fm(size=16)
        result = compare_maps(fm, fm)
        assert isinstance(result, CompareResult)
        assert result.rms_residual_tesla == pytest.approx(0.0, abs=1e-9)
        assert result.max_abs_residual_tesla == pytest.approx(0.0, abs=1e-9)
        assert result.correlation == pytest.approx(1.0, abs=1e-6)

    def test_known_offset_residual(self):
        """Measured is uniformly 0.001 T higher than reference."""
        ref = _make_fm(size=16, b0=0.05)
        x_mm = ref.x_mm.copy()
        y_mm = ref.y_mm.copy()
        meas = FieldMap(
            b_z=ref.b_z + 0.001,
            x_mm=x_mm,
            y_mm=y_mm,
            b0_nominal_tesla=0.05,
            active_radius_mm=ref.active_radius_mm,
            source="measurement",
        )
        result = compare_maps(meas, ref)
        assert result.rms_residual_tesla == pytest.approx(0.001, rel=1e-5)
        assert result.mean_residual_tesla == pytest.approx(0.001, rel=1e-5)

    def test_auto_regrid_on_different_shapes(self):
        fm_coarse = _make_fm(size=8)
        fm_fine = _make_fm(size=16)
        # Should not raise; reference is regridded to measured grid
        result = compare_maps(fm_coarse, fm_fine)
        assert result.rms_residual_tesla < 1e-5  # same uniform field

    def test_rms_ppm_consistent(self):
        ref = _make_fm(size=16, b0=0.05)
        meas = FieldMap(
            b_z=ref.b_z + 0.001,
            x_mm=ref.x_mm,
            y_mm=ref.y_mm,
            b0_nominal_tesla=0.05,
            active_radius_mm=ref.active_radius_mm,
            source="measurement",
        )
        result = compare_maps(meas, ref)
        expected_ppm = result.rms_residual_tesla / meas.b0_nominal_tesla * 1e6
        assert result.rms_residual_ppm == pytest.approx(expected_ppm, rel=1e-5)

    def test_anticorrelated_maps(self):
        """One map is the negative of the other → correlation = -1."""
        x_mm = np.linspace(-5, 5, 16, dtype=np.float32)
        y_mm = np.linspace(-5, 5, 16, dtype=np.float32)
        xx, _ = np.meshgrid(x_mm, y_mm)
        pos = FieldMap(b_z=(0.05 + 0.001 * xx).astype(np.float32), x_mm=x_mm, y_mm=y_mm,
                       b0_nominal_tesla=0.05, active_radius_mm=4.0, source="simulation")
        neg = FieldMap(b_z=(0.05 - 0.001 * xx).astype(np.float32), x_mm=x_mm, y_mm=y_mm,
                       b0_nominal_tesla=0.05, active_radius_mm=4.0, source="simulation")
        result = compare_maps(pos, neg)
        assert result.correlation == pytest.approx(-1.0, abs=1e-4)


# ===========================================================================
# 5. regrid
# ===========================================================================


class TestRegrid:
    def test_regrid_same_grid_identity(self):
        fm = _make_gradient_fm(size=16)
        regridded = regrid(fm, fm.x_mm.copy(), fm.y_mm.copy())
        np.testing.assert_array_almost_equal(regridded.b_z, fm.b_z, decimal=5)

    def test_regrid_coarser_shape(self):
        fm = _make_gradient_fm(size=32)
        x_coarse = np.linspace(-5, 5, 8, dtype=np.float32)
        y_coarse = np.linspace(-5, 5, 8, dtype=np.float32)
        coarse = regrid(fm, x_coarse, y_coarse)
        assert coarse.shape == (8, 8)
        assert coarse.b0_nominal_tesla == fm.b0_nominal_tesla

    def test_regrid_finer_shape(self):
        fm = _make_gradient_fm(size=8)
        x_fine = np.linspace(-5, 5, 64, dtype=np.float32)
        y_fine = np.linspace(-5, 5, 64, dtype=np.float32)
        fine = regrid(fm, x_fine, y_fine)
        assert fine.shape == (64, 64)

    def test_regrid_preserves_uniform_field(self):
        fm = _make_fm(size=16, b0=0.05)
        x_new = np.linspace(-4, 4, 12, dtype=np.float32)
        y_new = np.linspace(-4, 4, 12, dtype=np.float32)
        r = regrid(fm, x_new, y_new)
        np.testing.assert_array_almost_equal(
            r.b_z, np.full((12, 12), 0.05, dtype=np.float32), decimal=5
        )

    def test_regrid_metadata_preserved(self):
        fm = FieldMap(
            b_z=np.ones((8, 8), dtype=np.float32) * 0.05,
            x_mm=np.linspace(-5, 5, 8, dtype=np.float32),
            y_mm=np.linspace(-5, 5, 8, dtype=np.float32),
            b0_nominal_tesla=0.05,
            active_radius_mm=3.0,
            source="measurement",
            timestamp="T0",
            notes="test",
        )
        x_new = np.linspace(-4, 4, 16, dtype=np.float32)
        y_new = np.linspace(-4, 4, 16, dtype=np.float32)
        r = regrid(fm, x_new, y_new)
        assert r.source == "measurement"
        assert r.timestamp == "T0"
        assert r.notes == "test"
        assert r.active_radius_mm == pytest.approx(3.0)


# ===========================================================================
# 6. simulated_field_map
# ===========================================================================


class TestSimulatedFieldMap:
    def test_shape_matches_grid_size(self):
        config = SimConfig()
        fm = simulated_field_map(config)
        assert fm.shape == (config.grid.size, config.grid.size)

    def test_source_is_simulation(self):
        fm = simulated_field_map(SimConfig())
        assert fm.source == "simulation"

    def test_b0_nominal_matches_config(self):
        config = SimConfig()
        fm = simulated_field_map(config)
        assert fm.b0_nominal_tesla == pytest.approx(config.field.b0_tesla)

    def test_mean_field_near_b0(self):
        config = SimConfig()
        fm = simulated_field_map(config)
        assert np.mean(fm.b_z) == pytest.approx(config.field.b0_tesla, rel=0.01)

    def test_with_disturbance_differs(self):
        config = SimConfig()
        clean = simulated_field_map(config, add_disturbance=False)
        disturbed = simulated_field_map(config, add_disturbance=True, disturbance_seed=7)
        assert not np.allclose(clean.b_z, disturbed.b_z)

    def test_disturbance_seed_reproducible(self):
        config = SimConfig()
        a = simulated_field_map(config, add_disturbance=True, disturbance_seed=42)
        b = simulated_field_map(config, add_disturbance=True, disturbance_seed=42)
        np.testing.assert_array_equal(a.b_z, b.b_z)

    def test_active_radius_consistent(self):
        config = SimConfig()
        fm = simulated_field_map(config)
        expected_radius = (
            config.grid.physical_extent_mm / 2.0 * config.grid.active_zone_fraction
        )
        assert fm.active_radius_mm == pytest.approx(expected_radius)

    def test_x_y_span_matches_grid(self):
        config = SimConfig()
        fm = simulated_field_map(config)
        half = config.grid.physical_extent_mm / 2.0
        assert fm.x_mm.min() == pytest.approx(-half, rel=0.01)
        assert fm.x_mm.max() == pytest.approx(half, rel=0.01)

    def test_halbach_enabled_changes_field(self):
        config = SimConfig()
        flat = simulated_field_map(config)  # halbach disabled by default

        config_h = SimConfig()
        config_h.halbach.enabled = True
        halbach_fm = simulated_field_map(config_h)

        # Halbach adds multipole errors → field is no longer perfectly uniform
        assert not np.allclose(flat.b_z, halbach_fm.b_z, atol=1e-6)
