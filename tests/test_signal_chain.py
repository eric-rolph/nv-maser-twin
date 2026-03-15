"""Tests for the RF signal chain SNR budget model."""
import math

import numpy as np
import pytest

from nv_maser.config import (
    FieldConfig,
    GridConfig,
    MaserConfig,
    NVConfig,
    SignalChainConfig,
    SimConfig,
)
from nv_maser.physics.grid import SpatialGrid
from nv_maser.physics.signal_chain import (
    SignalChainBudget,
    compute_amplifier_noise,
    compute_maser_emission_power,
    compute_quantisation_noise,
    compute_signal_chain_budget,
    compute_snr_vs_field_uniformity,
    compute_thermal_noise,
    _KB,
    _T0,
)
from nv_maser.physics.environment import FieldEnvironment


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def nv_cfg() -> NVConfig:
    return NVConfig()


@pytest.fixture
def maser_cfg() -> MaserConfig:
    return MaserConfig()


@pytest.fixture
def signal_cfg() -> SignalChainConfig:
    return SignalChainConfig()


# ── MaserConfig coupling_beta ─────────────────────────────────────


class TestMaserConfigCouplingBeta:
    def test_default_coupling(self) -> None:
        cfg = MaserConfig()
        assert cfg.coupling_beta == 0.5

    def test_coupling_range(self) -> None:
        cfg = MaserConfig(coupling_beta=0.99)
        assert cfg.coupling_beta == 0.99


# ── SignalChainConfig ─────────────────────────────────────────────


class TestSignalChainConfig:
    def test_defaults(self) -> None:
        cfg = SignalChainConfig()
        assert cfg.lna_noise_figure_db == 1.0
        assert cfg.lna_gain_db == 30.0
        assert cfg.physical_temperature_k == 300.0
        assert cfg.detection_bandwidth_hz == 1000.0
        assert cfg.adc_bits == 14
        assert cfg.adc_full_scale_dbm == -10.0
        assert cfg.insertion_loss_db == 2.0


# ── Thermal noise ─────────────────────────────────────────────────


class TestThermalNoise:
    def test_room_temperature(self) -> None:
        """kB × 300K × 1kHz ≈ 4.14e-18 W."""
        p = compute_thermal_noise(300.0, 1000.0)
        expected = _KB * 300.0 * 1000.0
        assert p == pytest.approx(expected, rel=1e-10)

    def test_proportional_to_temperature(self) -> None:
        p1 = compute_thermal_noise(300.0, 1000.0)
        p2 = compute_thermal_noise(600.0, 1000.0)
        assert p2 == pytest.approx(2.0 * p1, rel=1e-10)

    def test_proportional_to_bandwidth(self) -> None:
        p1 = compute_thermal_noise(300.0, 1000.0)
        p2 = compute_thermal_noise(300.0, 2000.0)
        assert p2 == pytest.approx(2.0 * p1, rel=1e-10)

    def test_positive(self) -> None:
        assert compute_thermal_noise(300.0, 1000.0) > 0


# ── Amplifier noise ──────────────────────────────────────────────


class TestAmplifierNoise:
    def test_zero_noise_figure(self) -> None:
        """NF = 0 dB → F = 1 → T_amp = 0 → no added noise."""
        p = compute_amplifier_noise(0.0, 1000.0)
        assert p == pytest.approx(0.0, abs=1e-30)

    def test_positive_noise_figure(self) -> None:
        p = compute_amplifier_noise(1.0, 1000.0)
        assert p > 0

    def test_formula(self) -> None:
        """NF=3dB → F=2 → T_amp = 290K → P = kB·290·Δf."""
        p = compute_amplifier_noise(3.0, 1000.0)
        f_lin = 10.0 ** (3.0 / 10.0)
        expected = _KB * _T0 * (f_lin - 1.0) * 1000.0
        assert p == pytest.approx(expected, rel=1e-6)

    def test_increases_with_nf(self) -> None:
        p1 = compute_amplifier_noise(1.0, 1000.0)
        p3 = compute_amplifier_noise(3.0, 1000.0)
        assert p3 > p1


# ── Quantisation noise ───────────────────────────────────────────


class TestQuantisationNoise:
    def test_positive(self) -> None:
        assert compute_quantisation_noise(14, -10.0) > 0

    def test_more_bits_less_noise(self) -> None:
        p14 = compute_quantisation_noise(14, -10.0)
        p16 = compute_quantisation_noise(16, -10.0)
        assert p16 < p14

    def test_formula(self) -> None:
        """P_quant = P_FS / (1.5 × 4^bits)."""
        p_fs = 1e-3 * 10.0 ** (-10.0 / 10.0)
        expected = p_fs / (1.5 * 4.0**14)
        actual = compute_quantisation_noise(14, -10.0)
        assert actual == pytest.approx(expected, rel=1e-10)

    def test_scales_with_full_scale(self) -> None:
        p_low = compute_quantisation_noise(14, -20.0)
        p_high = compute_quantisation_noise(14, -10.0)
        assert p_high > p_low


# ── Maser emission power ─────────────────────────────────────────


class TestMaserEmissionPower:
    def test_positive_with_gain(self, nv_cfg: NVConfig, maser_cfg: MaserConfig) -> None:
        p = compute_maser_emission_power(nv_cfg, maser_cfg, gain_budget=1.0)
        assert p > 0

    def test_zero_with_zero_budget(self, nv_cfg: NVConfig, maser_cfg: MaserConfig) -> None:
        p = compute_maser_emission_power(nv_cfg, maser_cfg, gain_budget=0.0)
        assert p == 0.0

    def test_proportional_to_gain_budget(self, nv_cfg: NVConfig, maser_cfg: MaserConfig) -> None:
        p1 = compute_maser_emission_power(nv_cfg, maser_cfg, gain_budget=0.5)
        p2 = compute_maser_emission_power(nv_cfg, maser_cfg, gain_budget=1.0)
        assert p2 == pytest.approx(2.0 * p1, rel=1e-10)

    def test_increases_with_pump_efficiency(self, maser_cfg: MaserConfig) -> None:
        nv_low = NVConfig(pump_efficiency=0.2)
        nv_high = NVConfig(pump_efficiency=0.5)
        p_low = compute_maser_emission_power(nv_low, maser_cfg, 1.0)
        p_high = compute_maser_emission_power(nv_high, maser_cfg, 1.0)
        assert p_high > p_low

    def test_increases_with_nv_density(self, maser_cfg: MaserConfig) -> None:
        nv_sparse = NVConfig(nv_density_per_cm3=1e16)
        nv_dense = NVConfig(nv_density_per_cm3=1e17)
        p_sparse = compute_maser_emission_power(nv_sparse, maser_cfg, 1.0)
        p_dense = compute_maser_emission_power(nv_dense, maser_cfg, 1.0)
        assert p_dense > p_sparse


# ── Full budget ───────────────────────────────────────────────────


class TestSignalChainBudget:
    def test_returns_dataclass(self, nv_cfg: NVConfig, maser_cfg: MaserConfig,
                                signal_cfg: SignalChainConfig) -> None:
        budget = compute_signal_chain_budget(nv_cfg, maser_cfg, signal_cfg, 1.0)
        assert isinstance(budget, SignalChainBudget)

    def test_snr_positive_with_gain(self, nv_cfg: NVConfig, maser_cfg: MaserConfig,
                                     signal_cfg: SignalChainConfig) -> None:
        budget = compute_signal_chain_budget(nv_cfg, maser_cfg, signal_cfg, 1.0)
        assert budget.snr_db > 0
        assert budget.snr_linear > 1.0

    def test_snr_decreases_with_lower_gain(self, nv_cfg: NVConfig,
                                            maser_cfg: MaserConfig,
                                            signal_cfg: SignalChainConfig) -> None:
        b_high = compute_signal_chain_budget(nv_cfg, maser_cfg, signal_cfg, 1.0)
        b_low = compute_signal_chain_budget(nv_cfg, maser_cfg, signal_cfg, 0.1)
        assert b_high.snr_db > b_low.snr_db

    def test_insertion_loss_reduces_signal(self, nv_cfg: NVConfig,
                                           maser_cfg: MaserConfig) -> None:
        no_loss = SignalChainConfig(insertion_loss_db=0.0)
        with_loss = SignalChainConfig(insertion_loss_db=6.0)
        b1 = compute_signal_chain_budget(nv_cfg, maser_cfg, no_loss, 1.0)
        b2 = compute_signal_chain_budget(nv_cfg, maser_cfg, with_loss, 1.0)
        assert b1.received_power_w > b2.received_power_w

    def test_coupling_beta_scales_signal(self, nv_cfg: NVConfig,
                                          signal_cfg: SignalChainConfig) -> None:
        maser_half = MaserConfig(coupling_beta=0.5)
        maser_full = MaserConfig(coupling_beta=1.0)
        b1 = compute_signal_chain_budget(nv_cfg, maser_half, signal_cfg, 1.0)
        b2 = compute_signal_chain_budget(nv_cfg, maser_full, signal_cfg, 1.0)
        assert b2.coupled_power_w == pytest.approx(2.0 * b1.coupled_power_w, rel=1e-10)

    def test_noise_components_positive(self, nv_cfg: NVConfig, maser_cfg: MaserConfig,
                                        signal_cfg: SignalChainConfig) -> None:
        budget = compute_signal_chain_budget(nv_cfg, maser_cfg, signal_cfg, 1.0)
        assert budget.thermal_noise_w > 0
        assert budget.amplifier_noise_w > 0
        assert budget.quantisation_noise_w > 0
        assert budget.total_noise_w > 0

    def test_noise_sum(self, nv_cfg: NVConfig, maser_cfg: MaserConfig,
                       signal_cfg: SignalChainConfig) -> None:
        budget = compute_signal_chain_budget(nv_cfg, maser_cfg, signal_cfg, 1.0)
        expected = budget.thermal_noise_w + budget.amplifier_noise_w + budget.quantisation_noise_w
        assert budget.total_noise_w == pytest.approx(expected, rel=1e-10)

    def test_system_noise_temp_reasonable(self, nv_cfg: NVConfig, maser_cfg: MaserConfig,
                                          signal_cfg: SignalChainConfig) -> None:
        """System noise temp should be > physical temp (LNA adds noise)."""
        budget = compute_signal_chain_budget(nv_cfg, maser_cfg, signal_cfg, 1.0)
        assert budget.system_noise_temperature_k > signal_cfg.physical_temperature_k

    def test_zero_gain_budget(self, nv_cfg: NVConfig, maser_cfg: MaserConfig,
                               signal_cfg: SignalChainConfig) -> None:
        budget = compute_signal_chain_budget(nv_cfg, maser_cfg, signal_cfg, 0.0)
        assert budget.maser_emission_power_w == 0.0
        assert budget.snr_db == -math.inf


# ── SNR vs field uniformity ───────────────────────────────────────


class TestSNRvsFieldUniformity:
    def test_shape(self, nv_cfg: NVConfig, maser_cfg: MaserConfig,
                   signal_cfg: SignalChainConfig) -> None:
        b_std = np.linspace(0, 1e-4, 10)
        snr = compute_snr_vs_field_uniformity(nv_cfg, maser_cfg, signal_cfg, b_std)
        assert snr.shape == b_std.shape

    def test_decreases_with_b_std(self, nv_cfg: NVConfig, maser_cfg: MaserConfig,
                                   signal_cfg: SignalChainConfig) -> None:
        """More field non-uniformity → lower SNR."""
        b_std = np.array([0.0, 1e-5, 1e-4, 1e-3])
        snr = compute_snr_vs_field_uniformity(nv_cfg, maser_cfg, signal_cfg, b_std)
        # Each value should be ≥ the next (monotonically decreasing)
        for i in range(len(snr) - 1):
            assert snr[i] >= snr[i + 1], f"SNR should decrease: snr[{i}]={snr[i]} < snr[{i+1}]={snr[i+1]}"

    def test_perfect_field_highest_snr(self, nv_cfg: NVConfig, maser_cfg: MaserConfig,
                                        signal_cfg: SignalChainConfig) -> None:
        b_std = np.array([0.0, 1e-3])
        snr = compute_snr_vs_field_uniformity(nv_cfg, maser_cfg, signal_cfg, b_std)
        assert snr[0] > snr[1]


# ── Environment integration ──────────────────────────────────────


class TestEnvironmentSNRIntegration:
    def test_metrics_include_snr(self) -> None:
        """compute_uniformity_metric() now returns SNR fields."""
        config = SimConfig()
        env = FieldEnvironment(config)
        net_field = env.distorted_field
        metrics = env.compute_uniformity_metric(net_field)
        assert "snr_db" in metrics
        assert "received_power_w" in metrics
        assert "total_noise_w" in metrics
        assert "system_noise_temperature_k" in metrics

    def test_snr_is_finite(self) -> None:
        config = SimConfig()
        env = FieldEnvironment(config)
        net_field = env.distorted_field
        metrics = env.compute_uniformity_metric(net_field)
        assert np.isfinite(metrics["snr_db"])
        assert metrics["snr_db"] > 0

    def test_better_field_higher_snr(self) -> None:
        """Corrected (uniform) field → higher SNR than distorted field."""
        config = SimConfig()
        env = FieldEnvironment(config)
        _ = env.step(0.0)

        # Distorted (uncorrected)
        distorted = env.distorted_field
        m_bad = env.compute_uniformity_metric(distorted)

        # "Perfect" correction → uniform base field
        uniform = env.effective_base_field
        m_good = env.compute_uniformity_metric(uniform)

        assert m_good["snr_db"] >= m_bad["snr_db"]
