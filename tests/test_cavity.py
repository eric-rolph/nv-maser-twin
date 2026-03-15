"""Tests for microwave cavity properties and maser threshold model."""
import math

import pytest

from nv_maser.config import CavityConfig, MaserConfig, NVConfig
from nv_maser.physics.cavity import (
    CavityProperties,
    ThresholdResult,
    compute_cavity_properties,
    compute_full_threshold,
    compute_maser_threshold,
    compute_n_effective,
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
