"""Tests for microwave cavity properties and maser threshold model."""
import math

import pytest

from nv_maser.config import CavityConfig, MaserConfig, NVConfig
from nv_maser.physics.cavity import (
    CavityProperties,
    MagneticQResult,
    ThresholdResult,
    compute_cavity_properties,
    compute_effective_q,
    compute_full_threshold,
    compute_magnetic_q,
    compute_maser_threshold,
    compute_n_effective,
    compute_spectral_overlap,
    _HBAR,
    _MU0,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def nv_cfg() -> NVConfig:
    return NVConfig()


@pytest.fixture
def maser_cfg() -> MaserConfig:
    return MaserConfig()


@pytest.fixture
def cavity_cfg() -> CavityConfig:
    return CavityConfig()


@pytest.fixture
def props(maser_cfg: MaserConfig, cavity_cfg: CavityConfig) -> CavityProperties:
    return compute_cavity_properties(maser_cfg, cavity_cfg)


# ── CavityConfig ──────────────────────────────────────────────────


class TestCavityConfig:
    def test_defaults(self) -> None:
        cfg = CavityConfig()
        assert cfg.mode_volume_cm3 == 0.5
        assert cfg.fill_factor == 0.01

    def test_custom(self) -> None:
        cfg = CavityConfig(mode_volume_cm3=1.0, fill_factor=0.05)
        assert cfg.mode_volume_cm3 == 1.0
        assert cfg.fill_factor == 0.05


# ── Cavity properties ────────────────────────────────────────────


class TestCavityProperties:
    def test_returns_dataclass(self, props: CavityProperties) -> None:
        assert isinstance(props, CavityProperties)

    def test_mode_volume_conversion(self, cavity_cfg: CavityConfig,
                                     props: CavityProperties) -> None:
        assert props.mode_volume_m3 == pytest.approx(
            cavity_cfg.mode_volume_cm3 * 1e-6, rel=1e-10
        )

    def test_zpf_field_positive(self, props: CavityProperties) -> None:
        assert props.zpf_field_tesla > 0

    def test_zpf_formula(self, maser_cfg: MaserConfig,
                          cavity_cfg: CavityConfig) -> None:
        omega = 2 * math.pi * maser_cfg.cavity_frequency_ghz * 1e9
        v = cavity_cfg.mode_volume_cm3 * 1e-6
        expected = math.sqrt(_MU0 * _HBAR * omega / (2 * v))
        props = compute_cavity_properties(maser_cfg, cavity_cfg)
        assert props.zpf_field_tesla == pytest.approx(expected, rel=1e-10)

    def test_coupling_positive(self, props: CavityProperties) -> None:
        assert props.single_spin_coupling_hz > 0

    def test_cavity_linewidth(self, maser_cfg: MaserConfig,
                               props: CavityProperties) -> None:
        expected = maser_cfg.cavity_frequency_ghz * 1e9 / maser_cfg.cavity_q
        assert props.cavity_linewidth_hz == pytest.approx(expected, rel=1e-10)

    def test_purcell_positive(self, props: CavityProperties) -> None:
        assert props.purcell_factor > 0

    def test_higher_q_stronger_purcell(self, cavity_cfg: CavityConfig) -> None:
        low_q = MaserConfig(cavity_q=5000)
        high_q = MaserConfig(cavity_q=20000)
        p_low = compute_cavity_properties(low_q, cavity_cfg)
        p_high = compute_cavity_properties(high_q, cavity_cfg)
        assert p_high.purcell_factor > p_low.purcell_factor

    def test_smaller_mode_volume_stronger_coupling(self, maser_cfg: MaserConfig) -> None:
        small = CavityConfig(mode_volume_cm3=0.1)
        large = CavityConfig(mode_volume_cm3=1.0)
        p_small = compute_cavity_properties(maser_cfg, small)
        p_large = compute_cavity_properties(maser_cfg, large)
        assert p_small.single_spin_coupling_hz > p_large.single_spin_coupling_hz
        assert p_small.zpf_field_tesla > p_large.zpf_field_tesla


# ── N_effective ───────────────────────────────────────────────────


class TestNEffective:
    def test_positive(self, nv_cfg: NVConfig, cavity_cfg: CavityConfig) -> None:
        n = compute_n_effective(nv_cfg, cavity_cfg, gain_budget=1.0)
        assert n > 0

    def test_scales_with_gain_budget(self, nv_cfg: NVConfig,
                                      cavity_cfg: CavityConfig) -> None:
        n1 = compute_n_effective(nv_cfg, cavity_cfg, 0.5)
        n2 = compute_n_effective(nv_cfg, cavity_cfg, 1.0)
        assert n2 == pytest.approx(2.0 * n1, rel=1e-10)

    def test_zero_budget(self, nv_cfg: NVConfig, cavity_cfg: CavityConfig) -> None:
        n = compute_n_effective(nv_cfg, cavity_cfg, 0.0)
        assert n == 0.0

    def test_scales_with_fill_factor(self, nv_cfg: NVConfig) -> None:
        c1 = CavityConfig(fill_factor=0.01)
        c2 = CavityConfig(fill_factor=0.02)
        n1 = compute_n_effective(nv_cfg, c1, 1.0)
        n2 = compute_n_effective(nv_cfg, c2, 1.0)
        assert n2 == pytest.approx(2.0 * n1, rel=1e-10)

    def test_scales_with_density(self, cavity_cfg: CavityConfig) -> None:
        nv_lo = NVConfig(nv_density_per_cm3=1e16)
        nv_hi = NVConfig(nv_density_per_cm3=1e17)
        n_lo = compute_n_effective(nv_lo, cavity_cfg, 1.0)
        n_hi = compute_n_effective(nv_hi, cavity_cfg, 1.0)
        assert n_hi == pytest.approx(10.0 * n_lo, rel=1e-10)


# ── Maser threshold ──────────────────────────────────────────────


class TestMaserThreshold:
    def test_returns_dataclass(self, props: CavityProperties) -> None:
        result = compute_maser_threshold(props, n_effective=1e12, spin_linewidth_hz=1e6)
        assert isinstance(result, ThresholdResult)

    def test_zero_spins(self, props: CavityProperties) -> None:
        result = compute_maser_threshold(props, n_effective=0, spin_linewidth_hz=1e6)
        assert result.cooperativity == 0.0
        assert result.masing is False

    def test_cooperativity_formula(self, props: CavityProperties) -> None:
        n_eff = 1e12
        gamma = 1e6  # Hz
        result = compute_maser_threshold(props, n_eff, gamma)
        g_ens = props.single_spin_coupling_hz * math.sqrt(n_eff)
        expected_c = 4 * g_ens**2 / (props.cavity_linewidth_hz * gamma)
        assert result.cooperativity == pytest.approx(expected_c, rel=1e-10)

    def test_more_spins_higher_cooperativity(self, props: CavityProperties) -> None:
        r1 = compute_maser_threshold(props, n_effective=1e10, spin_linewidth_hz=1e6)
        r2 = compute_maser_threshold(props, n_effective=1e12, spin_linewidth_hz=1e6)
        assert r2.cooperativity > r1.cooperativity

    def test_narrower_linewidth_higher_cooperativity(self,
                                                      props: CavityProperties) -> None:
        r_wide = compute_maser_threshold(props, n_effective=1e12, spin_linewidth_hz=1e7)
        r_narrow = compute_maser_threshold(props, n_effective=1e12, spin_linewidth_hz=1e6)
        assert r_narrow.cooperativity > r_wide.cooperativity

    def test_threshold_margin_sign(self, props: CavityProperties) -> None:
        """Negative margin means below threshold."""
        # Very few spins → below threshold
        r = compute_maser_threshold(props, n_effective=1.0, spin_linewidth_hz=1e6)
        assert r.cooperativity < 1.0
        assert r.threshold_margin < 0
        assert r.masing is False

    def test_above_threshold_with_many_spins(self, props: CavityProperties) -> None:
        """Enough spins should reach C > 1."""
        r = compute_maser_threshold(props, n_effective=1e15, spin_linewidth_hz=1e6)
        assert r.cooperativity > 1.0
        assert r.masing is True
        assert r.threshold_margin > 0


# ── Full threshold convenience ────────────────────────────────────


class TestFullThreshold:
    def test_returns_threshold_result(self, nv_cfg: NVConfig, maser_cfg: MaserConfig,
                                       cavity_cfg: CavityConfig) -> None:
        result = compute_full_threshold(nv_cfg, maser_cfg, cavity_cfg, 1.0, 1e6)
        assert isinstance(result, ThresholdResult)

    def test_gain_budget_affects_threshold(self, nv_cfg: NVConfig,
                                            maser_cfg: MaserConfig,
                                            cavity_cfg: CavityConfig) -> None:
        r_low = compute_full_threshold(nv_cfg, maser_cfg, cavity_cfg, 0.1, 1e6)
        r_high = compute_full_threshold(nv_cfg, maser_cfg, cavity_cfg, 1.0, 1e6)
        assert r_high.cooperativity > r_low.cooperativity

    def test_zero_gain_budget_no_masing(self, nv_cfg: NVConfig,
                                         maser_cfg: MaserConfig,
                                         cavity_cfg: CavityConfig) -> None:
        result = compute_full_threshold(nv_cfg, maser_cfg, cavity_cfg, 0.0, 1e6)
        assert result.cooperativity == 0.0
        assert result.masing is False


# ── Orientation fraction affects N_eff ────────────────────────────


class TestOrientationFraction:
    def test_default_is_quarter(self) -> None:
        cfg = NVConfig()
        assert cfg.orientation_fraction == 0.25

    def test_n_eff_scales_with_orientation(self, cavity_cfg: CavityConfig) -> None:
        nv_full = NVConfig(orientation_fraction=1.0)
        nv_quarter = NVConfig(orientation_fraction=0.25)
        n_full = compute_n_effective(nv_full, cavity_cfg, 1.0)
        n_quarter = compute_n_effective(nv_quarter, cavity_cfg, 1.0)
        assert n_full == pytest.approx(4.0 * n_quarter, rel=1e-10)

    def test_default_config_still_mases(self, maser_cfg: MaserConfig,
                                         cavity_cfg: CavityConfig) -> None:
        """Default NVConfig (orientation_fraction=0.25) should still reach threshold."""
        nv = NVConfig()
        result = compute_full_threshold(nv, maser_cfg, cavity_cfg, 1.0, 1e6)
        assert result.masing is True


# ── Magnetic quality factor ───────────────────────────────────────


class TestMagneticQ:
    def test_returns_dataclass(self, nv_cfg: NVConfig,
                                cavity_cfg: CavityConfig) -> None:
        result = compute_magnetic_q(nv_cfg, cavity_cfg)
        assert isinstance(result, MagneticQResult)

    def test_positive_q(self, nv_cfg: NVConfig,
                         cavity_cfg: CavityConfig) -> None:
        result = compute_magnetic_q(nv_cfg, cavity_cfg)
        assert result.q_magnetic > 0

    def test_higher_density_lower_q(self, cavity_cfg: CavityConfig) -> None:
        """More inverted spins → stronger magnetic loading → lower Q_m."""
        nv_lo = NVConfig(nv_density_per_cm3=1e16)
        nv_hi = NVConfig(nv_density_per_cm3=1e17)
        q_lo = compute_magnetic_q(nv_lo, cavity_cfg).q_magnetic
        q_hi = compute_magnetic_q(nv_hi, cavity_cfg).q_magnetic
        assert q_hi < q_lo  # more spins → lower Q

    def test_longer_t2_lower_q(self, cavity_cfg: CavityConfig) -> None:
        nv_short = NVConfig(t2_star_us=0.5)
        nv_long = NVConfig(t2_star_us=2.0)
        q_short = compute_magnetic_q(nv_short, cavity_cfg).q_magnetic
        q_long = compute_magnetic_q(nv_long, cavity_cfg).q_magnetic
        assert q_long < q_short

    def test_fill_factor_affects_q(self) -> None:
        nv = NVConfig()
        c_small = CavityConfig(fill_factor=0.005)
        c_large = CavityConfig(fill_factor=0.05)
        q_small = compute_magnetic_q(nv, c_small).q_magnetic
        q_large = compute_magnetic_q(nv, c_large).q_magnetic
        assert q_large < q_small

    def test_orientation_affects_q(self, cavity_cfg: CavityConfig) -> None:
        nv_full = NVConfig(orientation_fraction=1.0)
        nv_quarter = NVConfig(orientation_fraction=0.25)
        q_full = compute_magnetic_q(nv_full, cavity_cfg).q_magnetic
        q_quarter = compute_magnetic_q(nv_quarter, cavity_cfg).q_magnetic
        assert q_full < q_quarter  # more oriented spins → lower Q


# ── Spectral overlap ratio ────────────────────────────────────────


class TestSpectralOverlap:
    def test_ratio_formula(self, props: CavityProperties) -> None:
        r = compute_spectral_overlap(props, spin_linewidth_hz=1e6)
        assert r == pytest.approx(props.cavity_linewidth_hz / 1e6, rel=1e-10)

    def test_narrower_spin_line_larger_r(self, props: CavityProperties) -> None:
        r_wide = compute_spectral_overlap(props, 1e7)
        r_narrow = compute_spectral_overlap(props, 1e5)
        assert r_narrow > r_wide

    def test_zero_linewidth_returns_inf(self, props: CavityProperties) -> None:
        r = compute_spectral_overlap(props, 0.0)
        assert r == float("inf")


# ── Q-boosting (Wang 2024) ────────────────────────────────────────


class TestEffectiveQ:
    def test_no_boost(self) -> None:
        """G_loop = 0 → Q_eff = Q_0."""
        cfg = MaserConfig(cavity_q=10_000, q_boost_gain=0.0)
        assert compute_effective_q(cfg) == pytest.approx(10_000, rel=1e-10)

    def test_moderate_boost(self) -> None:
        """G_loop = 0.5 → Q_eff = 2 × Q_0."""
        cfg = MaserConfig(cavity_q=10_000, q_boost_gain=0.5)
        assert compute_effective_q(cfg) == pytest.approx(20_000, rel=1e-10)

    def test_high_boost(self) -> None:
        """G_loop = 0.9 → Q_eff = 10 × Q_0."""
        cfg = MaserConfig(cavity_q=10_000, q_boost_gain=0.9)
        assert compute_effective_q(cfg) == pytest.approx(100_000, rel=1e-10)

    def test_wang_2024_reproduction(self) -> None:
        """Wang et al. 2024: Q_L from 1.1e4 to 6.5e5 → G ≈ 0.9831."""
        q0 = 1.1e4
        q_target = 6.5e5
        g = 1.0 - q0 / q_target  # ≈ 0.9831
        cfg = MaserConfig(cavity_q=q0, q_boost_gain=g)
        assert compute_effective_q(cfg) == pytest.approx(q_target, rel=1e-3)

    def test_boost_monotonic(self) -> None:
        """Higher gain → higher Q_eff."""
        q0 = 10_000
        gains = [0.0, 0.3, 0.6, 0.9]
        q_effs = [compute_effective_q(MaserConfig(cavity_q=q0, q_boost_gain=g)) for g in gains]
        for i in range(len(q_effs) - 1):
            assert q_effs[i + 1] > q_effs[i]

    def test_small_gain_small_effect(self) -> None:
        """G_loop = 0.01 → ~1% Q increase."""
        cfg = MaserConfig(cavity_q=10_000, q_boost_gain=0.01)
        q_eff = compute_effective_q(cfg)
        ratio = q_eff / 10_000
        assert ratio == pytest.approx(1.0 / 0.99, rel=1e-6)


class TestQBoostEnvironmentIntegration:
    def test_q_boost_metrics_present(self) -> None:
        from nv_maser.config import SimConfig
        from nv_maser.physics.environment import FieldEnvironment

        cfg = SimConfig(maser=MaserConfig(cavity_q=10_000, q_boost_gain=0.5))
        env = FieldEnvironment(cfg)
        result = env.compute_uniformity_metric(env.distorted_field)
        assert "q_boost_effective_q" in result
        assert result["q_boost_effective_q"] == pytest.approx(20_000, rel=1e-6)
        assert result["q_boost_gain"] == pytest.approx(0.5, rel=1e-10)

    def test_no_q_boost_no_metrics(self) -> None:
        from nv_maser.config import SimConfig
        from nv_maser.physics.environment import FieldEnvironment

        cfg = SimConfig(maser=MaserConfig(cavity_q=10_000, q_boost_gain=0.0))
        env = FieldEnvironment(cfg)
        result = env.compute_uniformity_metric(env.distorted_field)
        assert "q_boost_effective_q" not in result

    def test_q_boost_lowers_threshold(self) -> None:
        """Q-boosting should increase cooperativity (lower threshold)."""
        from nv_maser.config import SimConfig
        from nv_maser.physics.environment import FieldEnvironment

        cfg_no_boost = SimConfig(maser=MaserConfig(cavity_q=10_000, q_boost_gain=0.0))
        cfg_boosted = SimConfig(maser=MaserConfig(cavity_q=10_000, q_boost_gain=0.9))
        env_no = FieldEnvironment(cfg_no_boost)
        env_yes = FieldEnvironment(cfg_boosted)
        r_no = env_no.compute_uniformity_metric(env_no.distorted_field)
        r_yes = env_yes.compute_uniformity_metric(env_yes.distorted_field)
        # Higher Q → higher cooperativity
        assert r_yes["cooperativity"] > r_no["cooperativity"]
