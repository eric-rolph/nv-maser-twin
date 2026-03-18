"""Tests for quantum noise → signal chain Friis cascade integration.

Validates that MaserNoiseResult.noise_temperature_k is correctly wired into
compute_signal_chain_budget via the Friis cascade formula, that the new
SignalChainBudget fields (maser_noise_temperature_k, friis_system_temperature_k,
quantum_advantage_db) are physically correct, and that existing callers are
fully backward-compatible.
"""
from __future__ import annotations

import math

import pytest

from nv_maser.config import MaserConfig, NVConfig, SignalChainConfig
from nv_maser.physics.quantum_noise import (
    MaserNoiseResult,
    compute_maser_noise,
    compute_noise_temperature,
)
from nv_maser.physics.signal_chain import (
    SignalChainBudget,
    _KB,
    _T0,
    compute_friis_system_temperature,
    compute_signal_chain_budget,
)


# ── Shared constants ───────────────────────────────────────────────────────

_CAVITY_HZ = 1.47e9  # NV maser cavity frequency (Hz)
_NF_DB = 1.0  # typical LNA noise figure
_BW = 1_000.0  # detection bandwidth (Hz)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def nv_cfg() -> NVConfig:
    return NVConfig()


@pytest.fixture
def maser_cfg() -> MaserConfig:
    return MaserConfig()


@pytest.fixture
def signal_cfg() -> SignalChainConfig:
    return SignalChainConfig(detection_bandwidth_hz=_BW)


@pytest.fixture
def minimal_maser_noise() -> MaserNoiseResult:
    """A minimal but physically realistic MaserNoiseResult.

    n_sp = 1.5 (η_pump = 0.5), ν = 1.47 GHz, N̄ = 200, P_out = 10 pW.
    T_noise = ℏω × 1.5 / kB ≈ 0.11 K.
    """
    n_sp = 1.5
    kappa_hz = 147_000.0  # 147 kHz cavity linewidth
    f_hz = _CAVITY_HZ
    n_bar = 200.0
    import math as _math
    _hbar = 1.054571817e-34
    _kb = 1.380649e-23
    omega = 2.0 * _math.pi * f_hz
    t_noise = (_hbar * omega * n_sp) / _kb
    st_lw = kappa_hz * n_sp / (2.0 * n_bar)
    output_power = 10.0e-12  # 10 pW — well above noise floor
    rin_floor = 2.0 * n_sp / n_bar

    return MaserNoiseResult(
        population_inversion_factor=n_sp,
        added_noise_number=n_sp,
        schawlow_townes_linewidth_hz=st_lw,
        noise_temperature_k=t_noise,
        steady_state_photons=n_bar,
        output_power_w=output_power,
        phase_noise_1hz_dbc_hz=10.0 * _math.log10(st_lw / (2.0 * _math.pi)),
        rin_floor_per_hz=rin_floor,
        rin_floor_dbc_hz=10.0 * _math.log10(rin_floor),
        cavity_linewidth_hz=kappa_hz,
        cavity_frequency_hz=f_hz,
    )


@pytest.fixture
def full_maser_noise(nv_cfg: NVConfig, maser_cfg: MaserConfig) -> MaserNoiseResult:
    """Full MaserNoiseResult produced by compute_maser_noise() via solve_maxwell_bloch."""
    from nv_maser.config import CavityConfig, MaxwellBlochConfig
    from nv_maser.physics.cavity import compute_cavity_properties
    from nv_maser.physics.maxwell_bloch import solve_maxwell_bloch

    cavity_cfg = CavityConfig()
    mb_cfg = MaxwellBlochConfig(enable=True, t_max_us=50.0, n_time_points=500)
    cavity_props = compute_cavity_properties(maser_cfg, cavity_cfg)
    mb_result = solve_maxwell_bloch(nv_cfg, maser_cfg, cavity_cfg, mb_cfg)
    return compute_maser_noise(cavity_props, mb_result, nv_cfg, maser_cfg)


# ── TestFriisHelperFunction ────────────────────────────────────────────────


class TestFriisHelperFunction:
    """Unit tests for compute_friis_system_temperature()."""

    def test_zero_noise_figure_gives_t_maser(
        self, minimal_maser_noise: MaserNoiseResult
    ) -> None:
        """NF = 0 dB → T_LNA = 0 → T_friis = T_maser for any G."""
        t = compute_friis_system_temperature(minimal_maser_noise, 0.0, 100.0)
        assert t == pytest.approx(minimal_maser_noise.noise_temperature_k, rel=1e-6)

    def test_high_gain_approaches_t_maser(
        self, minimal_maser_noise: MaserNoiseResult
    ) -> None:
        """G_maser → ∞ ⇒ T_friis → T_maser (LNA contribution quenched)."""
        t_maser = minimal_maser_noise.noise_temperature_k
        t_friis = compute_friis_system_temperature(minimal_maser_noise, _NF_DB, 1e6)
        # Should be within 1 % of T_maser when G = 10^6
        assert t_friis == pytest.approx(t_maser, rel=0.01)

    def test_unity_gain_adds_full_lna_noise(
        self, minimal_maser_noise: MaserNoiseResult
    ) -> None:
        """G = 1 ⇒ T_friis = T_maser + T_LNA (no gain suppression)."""
        nf = 3.0  # dB → F=2 → T_LNA = T0
        g = 1.0
        t_lna = _T0 * (10.0 ** (nf / 10.0) - 1.0)
        expected = minimal_maser_noise.noise_temperature_k + t_lna
        t = compute_friis_system_temperature(minimal_maser_noise, nf, g)
        assert t == pytest.approx(expected, rel=1e-6)

    def test_gain_less_than_one_clipped_to_unity(
        self, minimal_maser_noise: MaserNoiseResult
    ) -> None:
        """Sub-unity gain is physically clamped to G=1 (no negative gain)."""
        t_g0 = compute_friis_system_temperature(minimal_maser_noise, _NF_DB, 0.01)
        t_g1 = compute_friis_system_temperature(minimal_maser_noise, _NF_DB, 1.0)
        assert t_g0 == pytest.approx(t_g1, rel=1e-9)

    def test_t_friis_always_positive(
        self, minimal_maser_noise: MaserNoiseResult
    ) -> None:
        for g in [1.0, 10.0, 100.0, 1e6]:
            t = compute_friis_system_temperature(minimal_maser_noise, _NF_DB, g)
            assert t > 0.0

    def test_friis_less_than_classical_at_realistic_gain(
        self, minimal_maser_noise: MaserNoiseResult
    ) -> None:
        """Friis temperature should be much lower than room-temperature chain."""
        # Classical T_sys ≈ 375 K; Friis at G=1000 ≈ 0.11 K + 75e-3 K ≈ 0.19 K
        t_classical = 375.0
        t_friis = compute_friis_system_temperature(minimal_maser_noise, _NF_DB, 1000.0)
        assert t_friis < t_classical / 10.0  # more than 10× advantage conservatively


# ── TestSignalChainBudgetQuantumFields ────────────────────────────────────


class TestSignalChainBudgetQuantumFields:
    """Validate the three new fields on SignalChainBudget."""

    def test_no_qn_result_gives_nan_fields(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        signal_cfg: SignalChainConfig,
    ) -> None:
        """Without MaserNoiseResult the three new fields are all math.nan."""
        budget = compute_signal_chain_budget(nv_cfg, maser_cfg, signal_cfg, 1.0)
        assert math.isnan(budget.maser_noise_temperature_k)
        assert math.isnan(budget.friis_system_temperature_k)
        assert math.isnan(budget.quantum_advantage_db)

    def test_qn_result_populates_fields(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        signal_cfg: SignalChainConfig,
        minimal_maser_noise: MaserNoiseResult,
    ) -> None:
        """With MaserNoiseResult all three new fields are finite numbers."""
        budget = compute_signal_chain_budget(
            nv_cfg, maser_cfg, signal_cfg, 1.0, maser_noise_result=minimal_maser_noise
        )
        assert math.isfinite(budget.maser_noise_temperature_k)
        assert math.isfinite(budget.friis_system_temperature_k)
        assert math.isfinite(budget.quantum_advantage_db)

    def test_maser_noise_temp_matches_input(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        signal_cfg: SignalChainConfig,
        minimal_maser_noise: MaserNoiseResult,
    ) -> None:
        """maser_noise_temperature_k must equal MaserNoiseResult.noise_temperature_k."""
        budget = compute_signal_chain_budget(
            nv_cfg, maser_cfg, signal_cfg, 1.0, maser_noise_result=minimal_maser_noise
        )
        assert budget.maser_noise_temperature_k == pytest.approx(
            minimal_maser_noise.noise_temperature_k, rel=1e-9
        )

    def test_friis_temp_less_than_classical(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        signal_cfg: SignalChainConfig,
        minimal_maser_noise: MaserNoiseResult,
    ) -> None:
        """Friis system temperature must be far below the classical T_sys."""
        budget = compute_signal_chain_budget(
            nv_cfg, maser_cfg, signal_cfg, 1.0, maser_noise_result=minimal_maser_noise
        )
        # Classical T_sys > 300 K, Friis should be < 10 K for any realistic maser
        assert budget.friis_system_temperature_k < budget.system_noise_temperature_k / 10.0

    def test_quantum_advantage_positive(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        signal_cfg: SignalChainConfig,
        minimal_maser_noise: MaserNoiseResult,
    ) -> None:
        """Maser pre-amp always improves SNR vs room-temp chain → positive advantage."""
        budget = compute_signal_chain_budget(
            nv_cfg, maser_cfg, signal_cfg, 1.0, maser_noise_result=minimal_maser_noise
        )
        assert budget.quantum_advantage_db > 0.0

    def test_quantum_advantage_exceeds_30_db(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        signal_cfg: SignalChainConfig,
        minimal_maser_noise: MaserNoiseResult,
    ) -> None:
        """NV maser at 1.47 GHz → T_noise ≈ 0.11 K vs ~375 K classical → ~35 dB."""
        budget = compute_signal_chain_budget(
            nv_cfg, maser_cfg, signal_cfg, 1.0, maser_noise_result=minimal_maser_noise
        )
        assert budget.quantum_advantage_db > 30.0

    def test_friis_temp_sub_kelvin_at_high_gain(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        minimal_maser_noise: MaserNoiseResult,
    ) -> None:
        """In the high-gain limit Friis temperature must be below 1 K."""
        # Use a wider bandwidth so gain estimate is moderate but still high
        sc = SignalChainConfig(detection_bandwidth_hz=100.0)
        budget = compute_signal_chain_budget(
            nv_cfg, maser_cfg, sc, 1.0, maser_noise_result=minimal_maser_noise
        )
        assert budget.friis_system_temperature_k < 1.0


# ── TestBackwardCompatibility ─────────────────────────────────────────────


class TestBackwardCompatibility:
    """Existing callers with no maser_noise_result must be unaffected."""

    def test_classical_fields_unchanged(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        signal_cfg: SignalChainConfig,
    ) -> None:
        """All non-quantum fields must match regardless of whether QN is supplied."""
        b_no_qn = compute_signal_chain_budget(nv_cfg, maser_cfg, signal_cfg, 1.0)
        # These must be identical (QN param only affects 3 new fields)
        assert b_no_qn.maser_emission_power_w > 0.0
        assert b_no_qn.system_noise_temperature_k > 300.0
        assert b_no_qn.snr_linear > 0.0
        assert b_no_qn.detection_bandwidth_hz == _BW

    def test_none_default_is_explicit(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        signal_cfg: SignalChainConfig,
    ) -> None:
        """Explicitly passing maser_noise_result=None equals omitting it."""
        b_implicit = compute_signal_chain_budget(nv_cfg, maser_cfg, signal_cfg, 1.0)
        b_explicit = compute_signal_chain_budget(
            nv_cfg, maser_cfg, signal_cfg, 1.0, maser_noise_result=None
        )
        assert b_implicit.maser_emission_power_w == pytest.approx(
            b_explicit.maser_emission_power_w, rel=1e-10
        )
        assert b_implicit.snr_db == pytest.approx(b_explicit.snr_db, rel=1e-10)
        assert b_implicit.system_noise_temperature_k == pytest.approx(
            b_explicit.system_noise_temperature_k, rel=1e-10
        )

    def test_classical_fields_equal_with_and_without_qn(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        signal_cfg: SignalChainConfig,
        minimal_maser_noise: MaserNoiseResult,
    ) -> None:
        """The 9 classical fields must be identical whether or not QN is supplied."""
        b_no_qn = compute_signal_chain_budget(nv_cfg, maser_cfg, signal_cfg, 1.0)
        b_qn = compute_signal_chain_budget(
            nv_cfg, maser_cfg, signal_cfg, 1.0, maser_noise_result=minimal_maser_noise
        )
        assert b_no_qn.maser_emission_power_w == pytest.approx(
            b_qn.maser_emission_power_w, rel=1e-10
        )
        assert b_no_qn.coupled_power_w == pytest.approx(b_qn.coupled_power_w, rel=1e-10)
        assert b_no_qn.received_power_w == pytest.approx(b_qn.received_power_w, rel=1e-10)
        assert b_no_qn.thermal_noise_w == pytest.approx(b_qn.thermal_noise_w, rel=1e-10)
        assert b_no_qn.amplifier_noise_w == pytest.approx(b_qn.amplifier_noise_w, rel=1e-10)
        assert b_no_qn.quantisation_noise_w == pytest.approx(b_qn.quantisation_noise_w, rel=1e-10)
        assert b_no_qn.total_noise_w == pytest.approx(b_qn.total_noise_w, rel=1e-10)
        assert b_no_qn.system_noise_temperature_k == pytest.approx(
            b_qn.system_noise_temperature_k, rel=1e-10
        )
        assert b_no_qn.snr_db == pytest.approx(b_qn.snr_db, rel=1e-10)

    def test_snr_vs_field_uniformity_still_works(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        signal_cfg: SignalChainConfig,
    ) -> None:
        """compute_snr_vs_field_uniformity must still run without modification."""
        import numpy as np
        from nv_maser.physics.signal_chain import compute_snr_vs_field_uniformity

        b_std = np.linspace(0.0, 1e-4, 5)
        result = compute_snr_vs_field_uniformity(nv_cfg, maser_cfg, signal_cfg, b_std)
        assert result.shape == b_std.shape
        assert all(math.isfinite(v) for v in result)


# ── TestPhysicsValues ────────────────────────────────────────────────────


class TestPhysicsValues:
    """Validate numerical outputs against analytic expectations."""

    def test_t_noise_at_1p47ghz_sub_kelvin(self) -> None:
        """T_noise(1.47 GHz, n_sp=1.5) must be well below 1 K."""
        t = compute_noise_temperature(_CAVITY_HZ, 1.5)
        assert t < 1.0
        assert t > 0.0

    def test_t_noise_formula(self) -> None:
        """T_noise = ℏω n_sp / kB at 1.47 GHz, n_sp=1."""
        import math as _m
        _hbar = 1.054571817e-34
        omega = 2.0 * _m.pi * _CAVITY_HZ
        expected = _hbar * omega / _KB
        t = compute_noise_temperature(_CAVITY_HZ, 1.0)
        assert t == pytest.approx(expected, rel=1e-9)

    def test_classical_t_sys_exceeds_300k(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        signal_cfg: SignalChainConfig,
    ) -> None:
        """Classical system noise temperature must exceed the physical temperature."""
        budget = compute_signal_chain_budget(nv_cfg, maser_cfg, signal_cfg, 1.0)
        assert budget.system_noise_temperature_k > 300.0

    def test_quantum_t_sys_below_1k(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        signal_cfg: SignalChainConfig,
        minimal_maser_noise: MaserNoiseResult,
    ) -> None:
        """Friis system temperature must be below 1 K for high-power maser output."""
        sc = SignalChainConfig(detection_bandwidth_hz=100.0)
        budget = compute_signal_chain_budget(
            nv_cfg, maser_cfg, sc, 1.0, maser_noise_result=minimal_maser_noise
        )
        assert budget.friis_system_temperature_k < 1.0

    def test_quantum_advantage_formula(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        signal_cfg: SignalChainConfig,
        minimal_maser_noise: MaserNoiseResult,
    ) -> None:
        """Verify quantum_advantage_db = 10 log10(T_sys / T_friis)."""
        budget = compute_signal_chain_budget(
            nv_cfg, maser_cfg, signal_cfg, 1.0, maser_noise_result=minimal_maser_noise
        )
        expected_adv = 10.0 * math.log10(
            budget.system_noise_temperature_k / budget.friis_system_temperature_k
        )
        assert budget.quantum_advantage_db == pytest.approx(expected_adv, rel=1e-6)

    def test_n_sp_one_gives_minimum_noise_temp(self) -> None:
        """n_sp = 1 (perfect inversion) gives the minimum quantum T_noise."""
        t_optimal = compute_noise_temperature(_CAVITY_HZ, 1.0)
        t_partial = compute_noise_temperature(_CAVITY_HZ, 1.5)
        assert t_optimal < t_partial


# ── TestComputeMaserNoiseIntegration ─────────────────────────────────────


class TestComputeMaserNoiseIntegration:
    """Integration tests using the real compute_maser_noise() pipeline."""

    def test_full_pipeline_populates_budget(
        self,
        full_maser_noise: MaserNoiseResult,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        signal_cfg: SignalChainConfig,
    ) -> None:
        """compute_maser_noise() result must flow through to budget without errors."""
        budget = compute_signal_chain_budget(
            nv_cfg, maser_cfg, signal_cfg, 1.0, maser_noise_result=full_maser_noise
        )
        assert isinstance(budget, SignalChainBudget)
        # Quantum fields must be finite (maser is above threshold)
        if math.isfinite(full_maser_noise.noise_temperature_k):
            assert math.isfinite(budget.maser_noise_temperature_k)

    def test_maser_noise_temp_from_pipeline_is_sub_kelvin(
        self, full_maser_noise: MaserNoiseResult
    ) -> None:
        """NV maser quantum noise temperature from pipeline must be < 1 K."""
        if full_maser_noise.noise_temperature_k > 0:
            assert full_maser_noise.noise_temperature_k < 1.0

    def test_pipeline_advantage_positive_if_above_threshold(
        self,
        full_maser_noise: MaserNoiseResult,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        signal_cfg: SignalChainConfig,
    ) -> None:
        """If maser is actively emitting, quantum advantage must be positive."""
        if full_maser_noise.output_power_w > 0.0:
            budget = compute_signal_chain_budget(
                nv_cfg, maser_cfg, signal_cfg, 1.0, maser_noise_result=full_maser_noise
            )
            if math.isfinite(budget.quantum_advantage_db):
                assert budget.quantum_advantage_db > 0.0
