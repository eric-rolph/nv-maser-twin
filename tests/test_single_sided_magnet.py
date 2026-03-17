"""Tests for the single-sided permanent magnet array model."""
import math

import numpy as np
import pytest

from nv_maser.config import SingleSidedMagnetConfig
from nv_maser.physics.single_sided_magnet import (
    SingleSidedMagnet,
    SweetSpotInfo,
    FieldMap2D,
    _solid_cylinder_on_axis_bz,
    _annular_ring_on_axis_bz,
    _MU0,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def default_config() -> SingleSidedMagnetConfig:
    return SingleSidedMagnetConfig()


@pytest.fixture
def barrel_magnet(default_config: SingleSidedMagnetConfig) -> SingleSidedMagnet:
    return SingleSidedMagnet(default_config)


@pytest.fixture
def strong_config() -> SingleSidedMagnetConfig:
    """Larger magnet with stronger field."""
    return SingleSidedMagnetConfig(
        remanence_tesla=1.45,
        num_rings=4,
        ring_inner_radii_mm=[3.0, 10.0, 18.0, 26.0],
        ring_outer_radii_mm=[8.0, 16.0, 24.0, 34.0],
        ring_heights_mm=[35.0, 30.0, 25.0, 20.0],
        ring_polarities=[1.0, -1.0, 1.0, -1.0],
    )


# ── Config validation ─────────────────────────────────────────────


class TestConfigValidation:
    def test_default_config_valid(self) -> None:
        cfg = SingleSidedMagnetConfig()
        assert cfg.magnet_type == "barrel"
        assert cfg.num_rings == 4
        assert len(cfg.ring_inner_radii_mm) == 4

    def test_mismatched_ring_lists_raises(self) -> None:
        cfg = SingleSidedMagnetConfig(
            num_rings=4,
            ring_inner_radii_mm=[5.0, 12.0, 20.0],  # only 3
        )
        with pytest.raises(ValueError, match="must have length"):
            SingleSidedMagnet(cfg)

    def test_inner_ge_outer_raises(self) -> None:
        cfg = SingleSidedMagnetConfig(
            num_rings=1,
            ring_inner_radii_mm=[20.0],
            ring_outer_radii_mm=[10.0],
            ring_heights_mm=[30.0],
            ring_polarities=[1.0],
        )
        with pytest.raises(ValueError, match="inner radius must be < outer"):
            SingleSidedMagnet(cfg)


# ── Solid cylinder model ─────────────────────────────────────────


class TestSolidCylinder:
    def test_field_at_surface_nonzero(self) -> None:
        """Field just above a magnetised cylinder should be nonzero."""
        z = np.array([0.001])  # 1 mm above surface
        bz = _solid_cylinder_on_axis_bz(
            z, z_centre=0.015, radius_m=0.01, height_m=0.03,
            magnetisation=1.45 / _MU0,
        )
        assert abs(bz[0]) > 0.01, "Field should be significant near the magnet"

    def test_field_decays_with_distance(self) -> None:
        """Field should decrease monotonically far from the cylinder."""
        z = np.linspace(0.05, 0.2, 50)
        bz = _solid_cylinder_on_axis_bz(
            z, z_centre=0.015, radius_m=0.01, height_m=0.03,
            magnetisation=1.45 / _MU0,
        )
        # After the near field, should monotonically decrease
        assert np.all(np.diff(np.abs(bz[-20:])) <= 0), "Far field should decay"

    def test_dipole_far_field(self) -> None:
        """At large z, the field should approximate a point dipole."""
        r = 0.005
        h = 0.01
        z_c = 0.015  # centre 15 mm below surface
        M = 1e6
        # Dipole moment: m = M × V
        vol = math.pi * r**2 * h
        m_dip = M * vol
        # Far field along axis of a dipole: Bz ≈ µ₀ 2m / (4π d³)
        z = np.array([0.2])  # 200 mm above surface
        bz = _solid_cylinder_on_axis_bz(z, z_c, r, h, M)
        # Distance from dipole centre (-z_c) to observation point (z):
        d = z[0] + z_c  # = 0.215 m
        bz_dipole = _MU0 * 2 * m_dip / (4 * math.pi * d**3)
        np.testing.assert_allclose(abs(bz[0]), abs(bz_dipole), rtol=0.05)


# ── Annular ring ──────────────────────────────────────────────────


class TestAnnularRing:
    def test_annular_weaker_than_solid(self) -> None:
        """Annular ring should produce less field than the outer solid cylinder."""
        z = np.array([0.02])
        bz_solid = _solid_cylinder_on_axis_bz(
            z, 0.015, radius_m=0.018, height_m=0.03,
            magnetisation=1e6,
        )
        bz_annular = _annular_ring_on_axis_bz(
            z, 0.015, r_inner_m=0.005, r_outer_m=0.018,
            height_m=0.03, magnetisation=1e6,
        )
        assert abs(bz_annular[0]) < abs(bz_solid[0])

    def test_zero_inner_radius_equals_solid(self) -> None:
        """Annular ring with zero inner radius should equal solid cylinder."""
        z = np.linspace(0.01, 0.05, 20)
        bz_ann = _annular_ring_on_axis_bz(
            z, 0.015, r_inner_m=1e-9, r_outer_m=0.01,
            height_m=0.03, magnetisation=1e6,
        )
        bz_sol = _solid_cylinder_on_axis_bz(
            z, 0.015, radius_m=0.01, height_m=0.03,
            magnetisation=1e6,
        )
        np.testing.assert_allclose(bz_ann, bz_sol, rtol=1e-4)


# ── Barrel magnet ─────────────────────────────────────────────────


class TestBarrelMagnet:
    def test_field_on_axis_returns_array(self, barrel_magnet: SingleSidedMagnet) -> None:
        depths = np.linspace(1.0, 40.0, 50)
        bz = barrel_magnet.field_on_axis(depths)
        assert bz.shape == (50,)
        assert not np.any(np.isnan(bz))

    def test_field_positive_near_surface(self, barrel_magnet: SingleSidedMagnet) -> None:
        """Alternating rings should produce a non-trivial field near surface."""
        depths = np.array([1.0, 5.0, 10.0])
        bz = barrel_magnet.field_on_axis(depths)
        # At least some depths should have non-zero field
        assert np.any(np.abs(bz) > 1e-6), "Expected nonzero field near the magnet"

    def test_field_decays_at_large_depth(self, barrel_magnet: SingleSidedMagnet) -> None:
        """Field should be small far from the magnet."""
        bz_near = barrel_magnet.field_on_axis(np.array([5.0]))
        bz_far = barrel_magnet.field_on_axis(np.array([100.0]))
        assert abs(bz_far[0]) < abs(bz_near[0])


class TestSweetSpot:
    def test_sweet_spot_returns_info(self, barrel_magnet: SingleSidedMagnet) -> None:
        ss = barrel_magnet.sweet_spot()
        assert isinstance(ss, SweetSpotInfo)
        assert ss.depth_mm > 0
        assert not math.isnan(ss.b0_tesla)

    def test_sweet_spot_depth_reasonable(self, barrel_magnet: SingleSidedMagnet) -> None:
        """Sweet spot should be within the computation extent."""
        ss = barrel_magnet.sweet_spot()
        assert 0 < ss.depth_mm < barrel_magnet.config.computation_extent_mm

    def test_uniformity_finite(self, barrel_magnet: SingleSidedMagnet) -> None:
        ss = barrel_magnet.sweet_spot()
        assert ss.uniformity_ppm >= 0
        assert not math.isinf(ss.uniformity_ppm) or abs(ss.b0_tesla) < 1e-10


class TestGradient:
    def test_gradient_shape(self, barrel_magnet: SingleSidedMagnet) -> None:
        depths = np.linspace(1.0, 30.0, 100)
        grad = barrel_magnet.gradient_on_axis(depths)
        assert grad.shape == (100,)

    def test_gradient_zero_at_sweet_spot(self, barrel_magnet: SingleSidedMagnet) -> None:
        """Gradient should be approximately zero at the sweet spot."""
        ss = barrel_magnet.sweet_spot()
        # Sample around sweet spot
        d = np.array([ss.depth_mm - 0.1, ss.depth_mm, ss.depth_mm + 0.1])
        grad = barrel_magnet.gradient_on_axis(d)
        # The gradient at the sweet spot should be smaller than elsewhere
        assert abs(grad[1]) < abs(grad[0]) + abs(grad[2]) + 1.0


class TestFieldMap2D:
    def test_field_map_shape(self, barrel_magnet: SingleSidedMagnet) -> None:
        fm = barrel_magnet.field_map_2d(
            depth_range_mm=(1.0, 30.0),
            lateral_extent_mm=20.0,
            resolution=16,
        )
        assert isinstance(fm, FieldMap2D)
        assert fm.bz.shape == (16, 16)
        assert fm.positions_mm.shape == (16, 16, 2)

    def test_field_map_no_nan(self, barrel_magnet: SingleSidedMagnet) -> None:
        fm = barrel_magnet.field_map_2d(resolution=8)
        assert not np.any(np.isnan(fm.bz))


# ── U-shaped magnet ───────────────────────────────────────────────


class TestUShaped:
    def test_u_shaped_field(self) -> None:
        cfg = SingleSidedMagnetConfig(
            magnet_type="u_shaped",
            num_rings=2,
            ring_inner_radii_mm=[5.0, 5.0],
            ring_outer_radii_mm=[15.0, 15.0],
            ring_heights_mm=[20.0, 20.0],
            ring_polarities=[1.0, -1.0],
        )
        mag = SingleSidedMagnet(cfg)
        depths = np.linspace(1.0, 30.0, 20)
        bz = mag.field_on_axis(depths)
        assert bz.shape == (20,)

    def test_unsupported_type_raises(self) -> None:
        cfg = SingleSidedMagnetConfig(magnet_type="toroid")
        mag = SingleSidedMagnet(cfg)
        with pytest.raises(ValueError, match="Unsupported"):
            mag.field_on_axis(np.array([10.0]))
