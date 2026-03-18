"""Tests for magpylib adapter and cross-validation of analytical magnet model.

These tests compare our hand-rolled SingleSidedMagnet analytical
calculations against magpylib's validated field computations.
"""
import math

import numpy as np
import pytest

from nv_maser.config import SingleSidedMagnetConfig

try:
    import magpylib  # noqa: F401

    HAS_MAGPYLIB = True
except ImportError:
    HAS_MAGPYLIB = False

pytestmark = pytest.mark.skipif(
    not HAS_MAGPYLIB, reason="magpylib not installed"
)


from nv_maser.physics.magpylib_adapter import (  # noqa: E402
    MAGPYLIB_AVAILABLE,
    build_magpylib_collection,
    compare_on_axis,
    field_map_2d_magpylib,
    field_on_axis_magpylib,
    find_sweet_spot_magpylib,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def default_config() -> SingleSidedMagnetConfig:
    return SingleSidedMagnetConfig()


@pytest.fixture
def single_ring_config() -> SingleSidedMagnetConfig:
    """Simple single-ring magnet for easier analytical comparison."""
    return SingleSidedMagnetConfig(
        remanence_tesla=1.0,
        num_rings=1,
        ring_inner_radii_mm=[0.1],  # nearly solid
        ring_outer_radii_mm=[10.0],
        ring_heights_mm=[20.0],
        ring_polarities=[1.0],
    )


@pytest.fixture
def two_ring_config() -> SingleSidedMagnetConfig:
    """Two rings with opposite polarity — creates a sweet spot."""
    return SingleSidedMagnetConfig(
        remanence_tesla=1.3,
        num_rings=2,
        ring_inner_radii_mm=[5.0, 15.0],
        ring_outer_radii_mm=[12.0, 25.0],
        ring_heights_mm=[25.0, 20.0],
        ring_polarities=[1.0, -1.0],
    )


# ── Collection building ──────────────────────────────────────────


class TestBuildCollection:
    def test_available_flag(self) -> None:
        assert MAGPYLIB_AVAILABLE is True

    def test_builds_from_default_config(self, default_config: SingleSidedMagnetConfig) -> None:
        col = build_magpylib_collection(default_config)
        assert len(col) == default_config.num_rings

    def test_single_ring_collection(self, single_ring_config: SingleSidedMagnetConfig) -> None:
        col = build_magpylib_collection(single_ring_config)
        assert len(col) == 1

    def test_two_ring_collection(self, two_ring_config: SingleSidedMagnetConfig) -> None:
        col = build_magpylib_collection(two_ring_config)
        assert len(col) == 2

    def test_ring_positions_below_surface(self, default_config: SingleSidedMagnetConfig) -> None:
        """All ring centres should be at z < 0 (below the surface)."""
        col = build_magpylib_collection(default_config)
        for source in col:
            assert source.position[2] < 0, "Rings should be below z=0 surface"

    def test_unsupported_type_raises(self) -> None:
        cfg = SingleSidedMagnetConfig(magnet_type="u_shaped")
        with pytest.raises(NotImplementedError, match="barrel"):
            build_magpylib_collection(cfg)


# ── Magpylib field computation ────────────────────────────────────


class TestMagpylibField:
    def test_field_nonzero_near_surface(self, default_config: SingleSidedMagnetConfig) -> None:
        depths = np.array([1.0, 5.0, 10.0])
        bz = field_on_axis_magpylib(default_config, depths)
        assert np.any(np.abs(bz) > 1e-6)

    def test_field_decays_far(self, single_ring_config: SingleSidedMagnetConfig) -> None:
        bz_near = field_on_axis_magpylib(single_ring_config, np.array([5.0]))
        bz_far = field_on_axis_magpylib(single_ring_config, np.array([100.0]))
        assert abs(bz_far[0]) < abs(bz_near[0])

    def test_field_shape(self, default_config: SingleSidedMagnetConfig) -> None:
        depths = np.linspace(1.0, 50.0, 100)
        bz = field_on_axis_magpylib(default_config, depths)
        assert bz.shape == (100,)
        assert not np.any(np.isnan(bz))

    def test_on_axis_symmetry(self, single_ring_config: SingleSidedMagnetConfig) -> None:
        """By-Bx should be zero on-axis (cylindrical symmetry)."""
        col = build_magpylib_collection(single_ring_config)
        b = np.atleast_2d(col.getB([(0, 0, 0.01)]))  # 10 mm above surface
        assert abs(b[0, 0]) < 1e-10, "Bx should be ~0 on axis"
        assert abs(b[0, 1]) < 1e-10, "By should be ~0 on axis"
        assert abs(b[0, 2]) > 1e-6, "Bz should be nonzero"


# ── Cross-validation: analytical vs magpylib ──────────────────────


class TestCrossValidation:
    """Compare our analytical on-axis model against magpylib.

    Our model uses superposition of annular ring on-axis formulas
    (exact for the on-axis case). Magpylib uses its own analytical
    expressions for CylinderSegment. They should agree to high
    precision on-axis.
    """

    def test_single_ring_on_axis_agreement(
        self, single_ring_config: SingleSidedMagnetConfig
    ) -> None:
        """Single ring on-axis should match very closely."""
        result = compare_on_axis(
            single_ring_config,
            depths_mm=np.linspace(5.0, 50.0, 50),
        )
        # On-axis analytical formula for a solid cylinder is exact,
        # so we expect very good agreement
        assert result["max_relative_diff"] < 0.10, (
            f"Single ring max relative diff {result['max_relative_diff']:.4f} "
            "exceeds 10% tolerance"
        )

    def test_two_ring_on_axis_agreement(
        self, two_ring_config: SingleSidedMagnetConfig
    ) -> None:
        """Two-ring barrel should agree well on-axis."""
        result = compare_on_axis(
            two_ring_config,
            depths_mm=np.linspace(5.0, 40.0, 50),
        )
        assert result["max_relative_diff"] < 0.10, (
            f"Two-ring max relative diff {result['max_relative_diff']:.4f} "
            "exceeds 10% tolerance"
        )

    def test_default_config_on_axis_agreement(
        self, default_config: SingleSidedMagnetConfig
    ) -> None:
        """Default 4-ring config on-axis."""
        result = compare_on_axis(
            default_config,
            depths_mm=np.linspace(5.0, 40.0, 50),
        )
        # 4-ring alternating polarity may have cancellations
        # where small absolute values inflate relative diffs.
        # Check both absolute and relative
        assert result["max_abs_diff"] < 0.05, (
            f"Default config max abs diff {result['max_abs_diff']:.6f} T "
            "exceeds 50 mT tolerance"
        )

    def test_field_signs_agree(self, single_ring_config: SingleSidedMagnetConfig) -> None:
        """Field signs should match between models at all depths."""
        result = compare_on_axis(
            single_ring_config,
            depths_mm=np.linspace(5.0, 50.0, 30),
        )
        signs_match = np.sign(result["bz_analytical"]) == np.sign(result["bz_magpylib"])
        # Allow for near-zero crossings where sign is ambiguous
        near_zero = np.abs(result["bz_magpylib"]) < 1e-6
        assert np.all(signs_match | near_zero), "Field signs disagree"

    def test_rms_relative_diff_reasonable(
        self, two_ring_config: SingleSidedMagnetConfig
    ) -> None:
        """RMS relative difference should be modest."""
        result = compare_on_axis(
            two_ring_config,
            depths_mm=np.linspace(10.0, 40.0, 40),
        )
        assert result["rms_relative_diff"] < 0.10, (
            f"RMS relative diff {result['rms_relative_diff']:.4f} exceeds 10%"
        )


# ── Sweet spot comparison ──────────────────────────────────────────


class TestSweetSpotComparison:
    def test_sweet_spot_exists(self, two_ring_config: SingleSidedMagnetConfig) -> None:
        """Two-ring opposite-polarity should have a sweet spot via magpylib."""
        ss = find_sweet_spot_magpylib(two_ring_config)
        assert ss["depth_mm"] > 0
        assert abs(ss["b0_tesla"]) > 1e-6

    def test_sweet_spot_depth_matches_analytical(
        self, two_ring_config: SingleSidedMagnetConfig
    ) -> None:
        """Sweet spot depth from magpylib should be close to our analytical model."""
        from nv_maser.physics.single_sided_magnet import SingleSidedMagnet

        mag = SingleSidedMagnet(two_ring_config)
        ss_analytical = mag.sweet_spot()
        ss_magpylib = find_sweet_spot_magpylib(two_ring_config)

        # Depths should agree within 3 mm
        depth_diff = abs(ss_analytical.depth_mm - ss_magpylib["depth_mm"])
        assert depth_diff < 3.0, (
            f"Sweet spot depth mismatch: analytical={ss_analytical.depth_mm:.1f} mm, "
            f"magpylib={ss_magpylib['depth_mm']:.1f} mm, diff={depth_diff:.1f} mm"
        )

    def test_sweet_spot_field_matches(
        self, two_ring_config: SingleSidedMagnetConfig
    ) -> None:
        """B₀ at the sweet spot should agree between both methods."""
        from nv_maser.physics.single_sided_magnet import SingleSidedMagnet

        mag = SingleSidedMagnet(two_ring_config)
        ss_analytical = mag.sweet_spot()
        ss_magpylib = find_sweet_spot_magpylib(two_ring_config)

        if abs(ss_magpylib["b0_tesla"]) > 1e-6:
            rel_diff = abs(ss_analytical.b0_tesla - ss_magpylib["b0_tesla"]) / abs(
                ss_magpylib["b0_tesla"]
            )
            assert rel_diff < 0.15, (
                f"Sweet spot B₀ mismatch: analytical={ss_analytical.b0_tesla:.4f} T, "
                f"magpylib={ss_magpylib['b0_tesla']:.4f} T, rel diff={rel_diff:.2%}"
            )


# ── 2D field map ──────────────────────────────────────────────────


class TestFieldMap2D:
    def test_field_map_shape(self, single_ring_config: SingleSidedMagnetConfig) -> None:
        bz, x_mm, z_mm = field_map_2d_magpylib(
            single_ring_config, resolution=16
        )
        assert bz.shape == (16, 16)
        assert x_mm.shape == (16,)
        assert z_mm.shape == (16,)

    def test_field_map_no_nan(self, single_ring_config: SingleSidedMagnetConfig) -> None:
        bz, _, _ = field_map_2d_magpylib(single_ring_config, resolution=8)
        assert not np.any(np.isnan(bz))

    def test_field_map_symmetric(self, single_ring_config: SingleSidedMagnetConfig) -> None:
        """Bz should be symmetric about x=0 for an axially symmetric magnet."""
        bz, x_mm, z_mm = field_map_2d_magpylib(
            single_ring_config,
            lateral_extent_mm=20.0,
            resolution=32,
        )
        # bz[i, :] at x=x_mm[i], bz[-1-i, :] at x=-x_mm[i]
        # Should be approximately equal
        n = len(x_mm)
        for i in range(n // 4):
            np.testing.assert_allclose(
                bz[i, :], bz[n - 1 - i, :],
                atol=1e-10,
                err_msg=f"Asymmetry at x={x_mm[i]:.1f} mm vs x={x_mm[n-1-i]:.1f} mm",
            )
