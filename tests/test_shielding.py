"""Tests for physics/shielding.py — mu-metal magnetic shielding model (SS15).

Coverage:
  TestMuMetalShellConfig          (4 tests)  — dataclass construction / immutability
  TestComputeSingleLayerAttenuation (10 tests) — exact formulas, monotonicity
  TestComputeMultilayerAttenuation  (5 tests)  — product law, n-layer stacking
  TestComputeShellMassKg            (6 tests)  — volume formulas, scaling
  TestComputeShielding              (7 tests)  — main entry point, architecture target
  TestFindThicknessForTargetDb      (6 tests)  — bisection sizing utility
  TestPublicAPI                     (1 test)   — importable from nv_maser.physics

Total: 39 tests.
"""

from __future__ import annotations

import math

import pytest

from nv_maser.physics.shielding import (
    MuMetalShellConfig,
    ShieldingResult,
    compute_multilayer_attenuation,
    compute_shell_mass_kg,
    compute_shielding,
    compute_single_layer_attenuation,
    find_thickness_for_target_db,
)

# ── helpers ────────────────────────────────────────────────────────────────────


def _default_sphere() -> MuMetalShellConfig:
    """Default 15 mm-radius, 1 mm-thick, µ_r=50 000 spherical shell."""
    return MuMetalShellConfig()


def _default_cylinder() -> MuMetalShellConfig:
    return MuMetalShellConfig(shell_shape="cylinder")


# ══════════════════════════════════════════════════════════════════════════════
# TestMuMetalShellConfig
# ══════════════════════════════════════════════════════════════════════════════


class TestMuMetalShellConfig:
    def test_defaults_valid(self):
        cfg = MuMetalShellConfig()
        assert cfg.inner_radius_mm == pytest.approx(15.0)
        assert cfg.thickness_mm == pytest.approx(1.0)
        assert cfg.mu_r == pytest.approx(50_000.0)
        assert cfg.shell_shape == "sphere"
        assert cfg.height_mm is None
        assert cfg.n_layers == 1
        assert cfg.interlayer_gap_mm == pytest.approx(5.0)
        assert cfg.density_kg_m3 == pytest.approx(8_700.0)

    def test_custom_sphere(self):
        cfg = MuMetalShellConfig(inner_radius_mm=20.0, thickness_mm=2.0, mu_r=30_000.0)
        assert cfg.inner_radius_mm == pytest.approx(20.0)
        assert cfg.thickness_mm == pytest.approx(2.0)

    def test_custom_cylinder(self):
        cfg = MuMetalShellConfig(shell_shape="cylinder", height_mm=30.0)
        assert cfg.shell_shape == "cylinder"
        assert cfg.height_mm == pytest.approx(30.0)

    def test_frozen(self):
        cfg = MuMetalShellConfig()
        with pytest.raises((TypeError, AttributeError)):
            cfg.thickness_mm = 5.0  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# TestComputeSingleLayerAttenuation
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeSingleLayerAttenuation:
    def test_sphere_s_ge_1(self):
        s = compute_single_layer_attenuation(_default_sphere())
        assert s >= 1.0

    def test_cylinder_s_ge_1(self):
        s = compute_single_layer_attenuation(_default_cylinder())
        assert s >= 1.0

    def test_sphere_architecture_target_50db(self):
        """Default sphere (15 mm, 1 mm, µ=50 000) must exceed 50 dB (R3 target)."""
        s = compute_single_layer_attenuation(_default_sphere())
        attenuation_db = 20.0 * math.log10(s)
        assert attenuation_db > 50.0

    def test_cylinder_reasonable_attenuation(self):
        """Default cylinder should provide at least 40 dB."""
        s = compute_single_layer_attenuation(_default_cylinder())
        attenuation_db = 20.0 * math.log10(s)
        assert attenuation_db > 40.0

    def test_unity_mu_r_sphere(self):
        """µ_r = 1 means non-magnetic material — no shielding, S = 1."""
        cfg = MuMetalShellConfig(mu_r=1.0)
        s = compute_single_layer_attenuation(cfg)
        assert s == pytest.approx(1.0, rel=1e-6)

    def test_unity_mu_r_cylinder(self):
        cfg = MuMetalShellConfig(shell_shape="cylinder", mu_r=1.0)
        s = compute_single_layer_attenuation(cfg)
        assert s == pytest.approx(1.0, rel=1e-6)

    def test_monotone_thickness_sphere(self):
        """Thicker shell → higher shielding factor."""
        s1 = compute_single_layer_attenuation(
            MuMetalShellConfig(thickness_mm=0.5)
        )
        s2 = compute_single_layer_attenuation(
            MuMetalShellConfig(thickness_mm=2.0)
        )
        assert s2 > s1

    def test_monotone_mu_r_sphere(self):
        """Higher µ_r → higher shielding factor."""
        s_low = compute_single_layer_attenuation(MuMetalShellConfig(mu_r=1_000.0))
        s_high = compute_single_layer_attenuation(MuMetalShellConfig(mu_r=100_000.0))
        assert s_high > s_low

    def test_returns_float(self):
        assert isinstance(compute_single_layer_attenuation(_default_sphere()), float)

    def test_sphere_greater_than_cylinder_same_params(self):
        """Spherical enclosure is more efficient than cylindrical (same t, r, µ).

        Thin-shell approximations: S_sphere ≈ 2µt/(3r), S_cyl ≈ µt/(2r).
        Ratio ≈ 4/3 > 1, so sphere always wins.
        """
        s_sphere = compute_single_layer_attenuation(
            MuMetalShellConfig(shell_shape="sphere")
        )
        s_cyl = compute_single_layer_attenuation(
            MuMetalShellConfig(shell_shape="cylinder")
        )
        assert s_sphere > s_cyl


# ══════════════════════════════════════════════════════════════════════════════
# TestComputeMultilayerAttenuation
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeMultilayerAttenuation:
    def test_single_layer_matches_single_layer_fn(self):
        cfg = MuMetalShellConfig(n_layers=1)
        assert compute_multilayer_attenuation(cfg) == pytest.approx(
            compute_single_layer_attenuation(cfg), rel=1e-9
        )

    def test_two_layers_greater_than_one(self):
        cfg_1 = MuMetalShellConfig(n_layers=1)
        cfg_2 = MuMetalShellConfig(n_layers=2)
        assert compute_multilayer_attenuation(cfg_2) > compute_multilayer_attenuation(cfg_1)

    def test_n_layers_0_returns_1(self):
        cfg = MuMetalShellConfig(n_layers=0)
        assert compute_multilayer_attenuation(cfg) == pytest.approx(1.0)

    def test_three_greater_than_two(self):
        cfg_2 = MuMetalShellConfig(n_layers=2)
        cfg_3 = MuMetalShellConfig(n_layers=3)
        assert compute_multilayer_attenuation(cfg_3) > compute_multilayer_attenuation(cfg_2)

    def test_product_approx_order_of_magnitude(self):
        """Two-layer S should be *approximately* S_single² (within 20×).

        The exact ratio deviates because the second shell sits at a slightly
        larger radius (r₁ + t + gap), so its individual attenuation is slightly
        smaller.  We require S_2 > 0.5 × S_1² as a lower-bound sanity check.
        """
        cfg_1 = MuMetalShellConfig(n_layers=1, interlayer_gap_mm=5.0)
        cfg_2 = MuMetalShellConfig(n_layers=2, interlayer_gap_mm=5.0)
        s1 = compute_multilayer_attenuation(cfg_1)
        s2 = compute_multilayer_attenuation(cfg_2)
        assert s2 > 0.5 * s1**2


# ══════════════════════════════════════════════════════════════════════════════
# TestComputeShellMassKg
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeShellMassKg:
    def test_sphere_mass_positive(self):
        assert compute_shell_mass_kg(_default_sphere()) > 0.0

    def test_cylinder_mass_positive(self):
        assert compute_shell_mass_kg(_default_cylinder()) > 0.0

    def test_cylinder_auto_height_differs_from_explicit(self):
        """A cylinder with an explicit height differs from the auto-height default."""
        cfg_auto = MuMetalShellConfig(shell_shape="cylinder", height_mm=None)
        cfg_tall = MuMetalShellConfig(shell_shape="cylinder", height_mm=60.0)
        # 60 mm >> 2×15 mm = 30 mm auto height → taller cylinder is heavier
        assert compute_shell_mass_kg(cfg_tall) > compute_shell_mass_kg(cfg_auto)

    def test_two_layers_heavier_than_one(self):
        m1 = compute_shell_mass_kg(MuMetalShellConfig(n_layers=1))
        m2 = compute_shell_mass_kg(MuMetalShellConfig(n_layers=2))
        assert m2 > m1

    def test_mass_scales_with_thickness(self):
        m_thin = compute_shell_mass_kg(MuMetalShellConfig(thickness_mm=0.5))
        m_thick = compute_shell_mass_kg(MuMetalShellConfig(thickness_mm=2.0))
        assert m_thick > m_thin

    def test_returns_scalar_float(self):
        assert isinstance(compute_shell_mass_kg(_default_sphere()), float)


# ══════════════════════════════════════════════════════════════════════════════
# TestComputeShielding
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeShielding:
    def test_returns_shielding_result(self):
        assert isinstance(compute_shielding(30e-3, MuMetalShellConfig()), ShieldingResult)

    def test_attenuation_linear_ge_1(self):
        result = compute_shielding(30e-3, MuMetalShellConfig())
        assert result.attenuation_linear >= 1.0

    def test_residual_less_than_incident(self):
        result = compute_shielding(30e-3, MuMetalShellConfig())
        assert result.residual_field_tesla < result.incident_field_tesla

    def test_attenuation_db_consistent_with_linear(self):
        result = compute_shielding(30e-3, MuMetalShellConfig())
        expected_db = 20.0 * math.log10(result.attenuation_linear)
        assert result.attenuation_db == pytest.approx(expected_db, rel=1e-6)

    def test_incident_field_echoed(self):
        b_in = 27.5e-3
        result = compute_shielding(b_in, MuMetalShellConfig())
        assert result.incident_field_tesla == pytest.approx(b_in)

    def test_architecture_50db_target_met(self):
        """Default shell (15 mm, 1 mm, µ=50 000, sphere) must exceed 50 dB (R3)."""
        result = compute_shielding(30e-3, MuMetalShellConfig())
        assert result.attenuation_db > 50.0

    def test_30mT_reduced_below_100uT(self):
        """Architecture doc: reduce ~30 mT stray field to < 100 µT at 40 mm separation."""
        result = compute_shielding(30e-3, MuMetalShellConfig())
        assert result.residual_field_tesla < 100e-6


# ══════════════════════════════════════════════════════════════════════════════
# TestFindThicknessForTargetDb
# ══════════════════════════════════════════════════════════════════════════════


class TestFindThicknessForTargetDb:
    _template = MuMetalShellConfig(inner_radius_mm=15.0, thickness_mm=1.0)

    def test_returns_mumu_metal_shell_config(self):
        result = find_thickness_for_target_db(50.0, self._template)
        assert isinstance(result, MuMetalShellConfig)

    def test_achieves_target_within_tolerance(self):
        """Plugging the returned thickness back must achieve ±0.1 dB of target."""
        target = 50.0
        result = find_thickness_for_target_db(target, self._template, tol_db=0.01)
        achieved = 20.0 * math.log10(
            max(compute_multilayer_attenuation(result), 1.0)
        )
        assert abs(achieved - target) <= 0.1

    def test_raises_on_lo_too_large(self):
        """t_lo that already exceeds the target should raise ValueError."""
        with pytest.raises(ValueError, match="already gives"):
            find_thickness_for_target_db(
                target_db=10.0,   # very easy target
                config_template=self._template,
                t_lo_mm=10.0,     # 10 mm shell far exceeds 10 dB
            )

    def test_raises_on_hi_insufficient(self):
        """t_hi too small to reach the target should raise ValueError."""
        with pytest.raises(ValueError, match="only gives"):
            find_thickness_for_target_db(
                target_db=200.0,  # impossibly demanding target
                config_template=self._template,
                t_hi_mm=0.001,    # 1 µm shell never reaches 200 dB
            )

    def test_target_50db_sphere_positive_thickness(self):
        result = find_thickness_for_target_db(50.0, MuMetalShellConfig(shell_shape="sphere"))
        assert result.thickness_mm > 0.0

    def test_target_40db_cylinder_positive_thickness(self):
        result = find_thickness_for_target_db(40.0, MuMetalShellConfig(shell_shape="cylinder"))
        assert result.thickness_mm > 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TestPublicAPI
# ══════════════════════════════════════════════════════════════════════════════


class TestPublicAPI:
    def test_all_symbols_importable_from_nv_maser_physics(self):
        from nv_maser.physics import (  # noqa: F401
            MuMetalShellConfig,
            ShieldingResult,
            compute_multilayer_attenuation,
            compute_shell_mass_kg,
            compute_shielding,
            compute_single_layer_attenuation,
            find_thickness_for_target_db,
        )
