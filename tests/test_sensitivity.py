"""Tests for field sensitivity analysis of the NV diamond maser magnetometer.

Covers:
- compute_schawlow_townes_sensitivity: η_ST = √(Δν_ST / 2π) / γ_e
- compute_thermal_sensitivity: η_T = (κ_c / γ_e) × √(k_B T_sys / P_received)
- compute_friis_sensitivity: η_F = (κ_c / γ_e) × √(k_B T_friis / P_received)
- compute_sensitivity: assembles all three + unit conversions + ratios
- SensitivityResult: unit conversions T → nT → pT, frozen dataclass
- Physics ordering: η_F ≤ η_T (Friis pre-amp can only improve over classical)
"""
from __future__ import annotations

import math

import pytest

from nv_maser.config import MaserConfig, NVConfig, SignalChainConfig
from nv_maser.physics.quantum_noise import MaserNoiseResult
from nv_maser.physics.sensitivity import (
    SensitivityResult,
    compute_friis_sensitivity,
    compute_schawlow_townes_sensitivity,
    compute_sensitivity,
    compute_thermal_sensitivity,
)
from nv_maser.physics.signal_chain import (
    SignalChainBudget,
    compute_signal_chain_budget,
    _KB,
)

_TWO_PI = 2.0 * math.pi
_HBAR = 1.054571817e-34


# ── Shared helpers ─────────────────────────────────────────────────────────


def _make_noise(
    n_sp: float = 1.5,
    kappa_hz: float = 147_000.0,
    f_hz: float = 1.47e9,
    n_bar: float = 200.0,
    output_power_w: float = 10.0e-12,
) -> MaserNoiseResult:
    """Construct a MaserNoiseResult with fully controlled parameters."""
    t_noise = (_HBAR * _TWO_PI * f_hz * n_sp) / _KB
    st_lw = kappa_hz * n_sp / (2.0 * n_bar)
    rin = 2.0 * n_sp / n_bar
    return MaserNoiseResult(
        population_inversion_factor=n_sp,
        added_noise_number=n_sp,
        schawlow_townes_linewidth_hz=st_lw,
        noise_temperature_k=t_noise,
        steady_state_photons=n_bar,
        output_power_w=output_power_w,
        phase_noise_1hz_dbc_hz=10.0 * math.log10(st_lw / _TWO_PI),
        rin_floor_per_hz=rin,
        rin_floor_dbc_hz=10.0 * math.log10(rin),
        cavity_linewidth_hz=kappa_hz,
        cavity_frequency_hz=f_hz,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def nv_cfg() -> NVConfig:
    return NVConfig()


@pytest.fixture
def maser_cfg() -> MaserConfig:
    return MaserConfig()


@pytest.fixture
def signal_cfg() -> SignalChainConfig:
    return SignalChainConfig()


@pytest.fixture
def minimal_noise() -> MaserNoiseResult:
    """n_sp=1.5, κ=147 kHz, N̄=200, P_out=10 pW → Δν_ST=551.25 Hz, T_noise≈0.106 K."""
    return _make_noise()


@pytest.fixture
def budget_no_qn(
    nv_cfg: NVConfig,
    maser_cfg: MaserConfig,
    signal_cfg: SignalChainConfig,
) -> SignalChainBudget:
    return compute_signal_chain_budget(nv_cfg, maser_cfg, signal_cfg, 1.0)


@pytest.fixture
def budget_with_qn(
    nv_cfg: NVConfig,
    maser_cfg: MaserConfig,
    signal_cfg: SignalChainConfig,
    minimal_noise: MaserNoiseResult,
) -> SignalChainBudget:
    return compute_signal_chain_budget(
        nv_cfg, maser_cfg, signal_cfg, 1.0, maser_noise_result=minimal_noise
    )


# ══ TestSchawlowTownesSensitivity ══════════════════════════════════════════


class TestSchawlowTownesSensitivity:
    """Unit tests for compute_schawlow_townes_sensitivity()."""

    def test_returns_finite_positive(
        self, minimal_noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        eta = compute_schawlow_townes_sensitivity(minimal_noise, nv_cfg)
        assert math.isfinite(eta)
        assert eta > 0.0

    def test_exact_formula(
        self, minimal_noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """η_ST = √(Δν_ST / (2π)) / γ_e."""
        delta_nu_st = minimal_noise.schawlow_townes_linewidth_hz
        gamma_e = nv_cfg.gamma_e_ghz_per_t * 1e9
        expected = math.sqrt(delta_nu_st / _TWO_PI) / gamma_e
        actual = compute_schawlow_townes_sensitivity(minimal_noise, nv_cfg)
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_sub_nanotesla_range(
        self, minimal_noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """ST sensitivity for realistic NV maser should be sub-nT/√Hz."""
        eta_nt = compute_schawlow_townes_sensitivity(minimal_noise, nv_cfg) * 1e9
        assert 0.01 < eta_nt < 10.0

    def test_scales_as_sqrt_delta_nu_st(self, nv_cfg: NVConfig) -> None:
        """Doubling Δν_ST (halving N̄) raises η_ST by exactly √2."""
        noise_lo = _make_noise(n_bar=200.0)  # Δν_ST = 551.25 Hz
        noise_hi = _make_noise(n_bar=100.0)  # Δν_ST = 1102.5 Hz — double
        eta_lo = compute_schawlow_townes_sensitivity(noise_lo, nv_cfg)
        eta_hi = compute_schawlow_townes_sensitivity(noise_hi, nv_cfg)
        assert eta_hi == pytest.approx(eta_lo * math.sqrt(2.0), rel=1e-9)

    def test_inversely_proportional_to_gamma_e(
        self, minimal_noise: MaserNoiseResult
    ) -> None:
        """Doubling γ_e halves η_ST."""
        cfg1 = NVConfig(gamma_e_ghz_per_t=28.025)
        cfg2 = NVConfig(gamma_e_ghz_per_t=56.050)
        eta1 = compute_schawlow_townes_sensitivity(minimal_noise, cfg1)
        eta2 = compute_schawlow_townes_sensitivity(minimal_noise, cfg2)
        assert eta2 == pytest.approx(eta1 / 2.0, rel=1e-9)


# ══ TestThermalSensitivity ═════════════════════════════════════════════════


class TestThermalSensitivity:
    """Unit tests for compute_thermal_sensitivity()."""

    def test_returns_finite_positive(
        self, budget_no_qn: SignalChainBudget, minimal_noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        eta = compute_thermal_sensitivity(budget_no_qn, minimal_noise, nv_cfg)
        assert math.isfinite(eta)
        assert eta > 0.0

    def test_exact_formula(
        self, budget_no_qn: SignalChainBudget, minimal_noise: MaserNoiseResult, nv_cfg: NVConfig
    ) -> None:
        """η_T = (κ_c / γ_e) × √(k_B T_sys / P_received)."""
        kappa_c = minimal_noise.cavity_linewidth_hz
        gamma_e = nv_cfg.gamma_e_ghz_per_t * 1e9
        t_sys = budget_no_qn.system_noise_temperature_k
        p_rcv = budget_no_qn.received_power_w
        expected = (kappa_c / gamma_e) * math.sqrt(_KB * t_sys / p_rcv)
        actual = compute_thermal_sensitivity(budget_no_qn, minimal_noise, nv_cfg)
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_improves_with_higher_power(self, nv_cfg: NVConfig, maser_cfg: MaserConfig) -> None:
        """Higher NV density → more power → lower (better) thermal sensitivity."""
        noise = _make_noise()
        sig = SignalChainConfig()
        # More NV centers ↑ → higher emission power ↑ → lower η_T ↓
        nv_dense = NVConfig(nv_density_per_cm3=NVConfig().nv_density_per_cm3 * 10.0)
        b_sparse = compute_signal_chain_budget(NVConfig(), maser_cfg, sig, 1.0)
        b_dense = compute_signal_chain_budget(nv_dense, maser_cfg, sig, 1.0)
        eta_sparse = compute_thermal_sensitivity(b_sparse, noise, nv_cfg)
        eta_dense = compute_thermal_sensitivity(b_dense, noise, nv_cfg)
        assert eta_dense < eta_sparse

    def test_degrades_with_higher_temperature(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        minimal_noise: MaserNoiseResult,
    ) -> None:
        """Higher physical temperature → higher T_sys → larger (worse) η_T."""
        sig_cool = SignalChainConfig(physical_temperature_k=200.0)
        sig_warm = SignalChainConfig(physical_temperature_k=400.0)
        b_cool = compute_signal_chain_budget(nv_cfg, maser_cfg, sig_cool, 1.0)
        b_warm = compute_signal_chain_budget(nv_cfg, maser_cfg, sig_warm, 1.0)
        eta_cool = compute_thermal_sensitivity(b_cool, minimal_noise, nv_cfg)
        eta_warm = compute_thermal_sensitivity(b_warm, minimal_noise, nv_cfg)
        assert eta_warm > eta_cool

    def test_zero_received_power_gives_inf(
        self, minimal_noise: MaserNoiseResult, nv_cfg: NVConfig,
        maser_cfg: MaserConfig, signal_cfg: SignalChainConfig
    ) -> None:
        """gain_budget=0 → no maser emission → P_received=0 → η_T=inf."""
        budget = compute_signal_chain_budget(nv_cfg, maser_cfg, signal_cfg, 0.0)
        eta = compute_thermal_sensitivity(budget, minimal_noise, nv_cfg)
        assert eta == math.inf


# ══ TestFriisEnhancedSensitivity ═══════════════════════════════════════════


class TestFriisEnhancedSensitivity:
    """Unit tests for compute_friis_sensitivity()."""

    def test_nan_when_friis_temperature_not_available(
        self,
        budget_no_qn: SignalChainBudget,
        minimal_noise: MaserNoiseResult,
        nv_cfg: NVConfig,
    ) -> None:
        """Budget built without QN result → friis_system_temperature_k = nan → η_F = nan."""
        assert math.isnan(budget_no_qn.friis_system_temperature_k)
        eta = compute_friis_sensitivity(budget_no_qn, minimal_noise, nv_cfg)
        assert math.isnan(eta)

    def test_finite_when_friis_available(
        self,
        budget_with_qn: SignalChainBudget,
        minimal_noise: MaserNoiseResult,
        nv_cfg: NVConfig,
    ) -> None:
        """Budget built with QN result → η_F is finite and positive."""
        eta = compute_friis_sensitivity(budget_with_qn, minimal_noise, nv_cfg)
        assert math.isfinite(eta)
        assert eta > 0.0

    def test_friis_better_than_thermal(
        self,
        budget_with_qn: SignalChainBudget,
        minimal_noise: MaserNoiseResult,
        nv_cfg: NVConfig,
    ) -> None:
        """η_F < η_T always (T_friis < T_sys)."""
        assert budget_with_qn.friis_system_temperature_k < budget_with_qn.system_noise_temperature_k
        eta_t = compute_thermal_sensitivity(budget_with_qn, minimal_noise, nv_cfg)
        eta_f = compute_friis_sensitivity(budget_with_qn, minimal_noise, nv_cfg)
        assert eta_f < eta_t

    def test_exact_formula(
        self,
        budget_with_qn: SignalChainBudget,
        minimal_noise: MaserNoiseResult,
        nv_cfg: NVConfig,
    ) -> None:
        """η_F = (κ_c / γ_e) × √(k_B T_friis / P_received)."""
        kappa_c = minimal_noise.cavity_linewidth_hz
        gamma_e = nv_cfg.gamma_e_ghz_per_t * 1e9
        t_friis = budget_with_qn.friis_system_temperature_k
        p_rcv = budget_with_qn.received_power_w
        expected = (kappa_c / gamma_e) * math.sqrt(_KB * t_friis / p_rcv)
        actual = compute_friis_sensitivity(budget_with_qn, minimal_noise, nv_cfg)
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_advantage_scales_as_sqrt_temperature_ratio(
        self,
        budget_with_qn: SignalChainBudget,
        minimal_noise: MaserNoiseResult,
        nv_cfg: NVConfig,
    ) -> None:
        """η_T / η_F = √(T_sys / T_friis)."""
        eta_t = compute_thermal_sensitivity(budget_with_qn, minimal_noise, nv_cfg)
        eta_f = compute_friis_sensitivity(budget_with_qn, minimal_noise, nv_cfg)
        t_ratio = budget_with_qn.system_noise_temperature_k / budget_with_qn.friis_system_temperature_k
        assert eta_t / eta_f == pytest.approx(math.sqrt(t_ratio), rel=1e-9)


# ══ TestSensitivityResultDataclass ════════════════════════════════════════


class TestSensitivityResultDataclass:
    """Verify SensitivityResult fields and unit conversions."""

    @pytest.fixture
    def result(
        self,
        budget_with_qn: SignalChainBudget,
        minimal_noise: MaserNoiseResult,
        nv_cfg: NVConfig,
    ) -> SensitivityResult:
        return compute_sensitivity(budget_with_qn, minimal_noise, nv_cfg)

    def test_is_frozen_dataclass(self, result: SensitivityResult) -> None:
        with pytest.raises((AttributeError, TypeError)):
            result.schawlow_townes_t_per_sqrthz = 999.0  # type: ignore[misc]

    def test_nt_per_sqrthz_is_t_times_1e9(self, result: SensitivityResult) -> None:
        assert result.schawlow_townes_nt_per_sqrthz == pytest.approx(
            result.schawlow_townes_t_per_sqrthz * 1e9, rel=1e-12
        )
        assert result.thermal_snr_nt_per_sqrthz == pytest.approx(
            result.thermal_snr_t_per_sqrthz * 1e9, rel=1e-12
        )
        assert result.friis_nt_per_sqrthz == pytest.approx(
            result.friis_t_per_sqrthz * 1e9, rel=1e-12
        )

    def test_pt_per_sqrthz_is_t_times_1e12(self, result: SensitivityResult) -> None:
        assert result.schawlow_townes_pt_per_sqrthz == pytest.approx(
            result.schawlow_townes_t_per_sqrthz * 1e12, rel=1e-12
        )
        assert result.thermal_snr_pt_per_sqrthz == pytest.approx(
            result.thermal_snr_t_per_sqrthz * 1e12, rel=1e-12
        )
        assert result.friis_pt_per_sqrthz == pytest.approx(
            result.friis_t_per_sqrthz * 1e12, rel=1e-12
        )

    def test_provenance_fields_match_inputs(
        self,
        budget_with_qn: SignalChainBudget,
        minimal_noise: MaserNoiseResult,
        nv_cfg: NVConfig,
        result: SensitivityResult,
    ) -> None:
        assert result.schawlow_townes_linewidth_hz == minimal_noise.schawlow_townes_linewidth_hz
        assert result.cavity_linewidth_hz == minimal_noise.cavity_linewidth_hz
        assert result.system_noise_temperature_k == budget_with_qn.system_noise_temperature_k
        assert result.gamma_e_hz_per_t == pytest.approx(
            nv_cfg.gamma_e_ghz_per_t * 1e9, rel=1e-12
        )

    def test_allan_deviation_equals_st_sensitivity(
        self, result: SensitivityResult
    ) -> None:
        """σ_B(τ=1s) = η_ST (numeric identity for τ=1s)."""
        assert result.allan_deviation_1s_t == pytest.approx(
            result.schawlow_townes_t_per_sqrthz, rel=1e-12
        )


# ══ TestComputeSensitivityPhysics ══════════════════════════════════════════


class TestComputeSensitivityPhysics:
    """Physics ordering and ratio tests for compute_sensitivity()."""

    @pytest.fixture
    def result_no_qn(
        self,
        budget_no_qn: SignalChainBudget,
        minimal_noise: MaserNoiseResult,
        nv_cfg: NVConfig,
    ) -> SensitivityResult:
        return compute_sensitivity(budget_no_qn, minimal_noise, nv_cfg)

    @pytest.fixture
    def result_with_qn(
        self,
        budget_with_qn: SignalChainBudget,
        minimal_noise: MaserNoiseResult,
        nv_cfg: NVConfig,
    ) -> SensitivityResult:
        return compute_sensitivity(budget_with_qn, minimal_noise, nv_cfg)

    def test_st_sensitivity_always_finite_positive(
        self, result_no_qn: SensitivityResult
    ) -> None:
        assert math.isfinite(result_no_qn.schawlow_townes_t_per_sqrthz)
        assert result_no_qn.schawlow_townes_t_per_sqrthz > 0.0

    def test_friis_fields_nan_without_qn_budget(
        self, result_no_qn: SensitivityResult
    ) -> None:
        """No QN in budget → all Friis fields are nan."""
        assert math.isnan(result_no_qn.friis_t_per_sqrthz)
        assert math.isnan(result_no_qn.friis_nt_per_sqrthz)
        assert math.isnan(result_no_qn.friis_pt_per_sqrthz)
        assert math.isnan(result_no_qn.friis_vs_st_ratio)
        assert math.isnan(result_no_qn.friis_advantage_over_thermal_db)

    def test_friis_better_than_thermal_in_sensitivity(
        self, result_with_qn: SensitivityResult
    ) -> None:
        """Friis-enhanced floor must be lower (better) than classical thermal floor."""
        assert result_with_qn.friis_t_per_sqrthz < result_with_qn.thermal_snr_t_per_sqrthz

    def test_friis_advantage_positive_db(
        self, result_with_qn: SensitivityResult
    ) -> None:
        """Friis advantage should be a positive number (improvement in dB)."""
        assert math.isfinite(result_with_qn.friis_advantage_over_thermal_db)
        assert result_with_qn.friis_advantage_over_thermal_db > 0.0

    def test_friis_advantage_consistent_with_temperature_ratio(
        self, result_with_qn: SensitivityResult
    ) -> None:
        """20 log₁₀(η_T / η_F) = 10 log₁₀(T_sys / T_friis)."""
        t_sys = result_with_qn.system_noise_temperature_k
        t_friis = result_with_qn.friis_system_temperature_k
        expected_db = 10.0 * math.log10(t_sys / t_friis)
        assert result_with_qn.friis_advantage_over_thermal_db == pytest.approx(
            expected_db, rel=1e-6
        )

    def test_thermal_vs_st_ratio_positive(
        self, result_no_qn: SensitivityResult
    ) -> None:
        assert math.isfinite(result_no_qn.thermal_vs_st_ratio)
        assert result_no_qn.thermal_vs_st_ratio > 0.0

    def test_st_sensitivity_independent_of_budget_qn(
        self,
        result_no_qn: SensitivityResult,
        result_with_qn: SensitivityResult,
    ) -> None:
        """ST limit depends only on Δν_ST, not on signal chain budget."""
        assert result_no_qn.schawlow_townes_t_per_sqrthz == pytest.approx(
            result_with_qn.schawlow_townes_t_per_sqrthz, rel=1e-12
        )


# ══ TestComputeSensitivityIntegration ═════════════════════════════════════


class TestComputeSensitivityIntegration:
    """Full physics pipeline: solve_maxwell_bloch → compute_maser_noise
    → compute_signal_chain_budget → compute_sensitivity."""

    @pytest.fixture(scope="class")
    def full_result(self) -> SensitivityResult:
        """End-to-end sensitivity result using the full physics stack."""
        from nv_maser.config import CavityConfig, MaxwellBlochConfig
        from nv_maser.physics.cavity import compute_cavity_properties
        from nv_maser.physics.maxwell_bloch import solve_maxwell_bloch
        from nv_maser.physics.quantum_noise import compute_maser_noise
        from nv_maser.physics.signal_chain import compute_signal_chain_budget

        nv_cfg = NVConfig()
        maser_cfg = MaserConfig()
        signal_cfg = SignalChainConfig()
        cavity_cfg = CavityConfig()
        mb_cfg = MaxwellBlochConfig(enable=True, t_max_us=50.0, n_time_points=500)

        cavity_props = compute_cavity_properties(maser_cfg, cavity_cfg)
        mb_result = solve_maxwell_bloch(nv_cfg, maser_cfg, cavity_cfg, mb_cfg)
        noise = compute_maser_noise(cavity_props, mb_result, nv_cfg, maser_cfg)
        budget = compute_signal_chain_budget(
            nv_cfg, maser_cfg, signal_cfg, 1.0, maser_noise_result=noise
        )
        return compute_sensitivity(budget, noise, nv_cfg)

    def test_st_sensitivity_finite_positive(self, full_result: SensitivityResult) -> None:
        """ST limit should be a finite positive number regardless of operating regime."""
        assert math.isfinite(full_result.schawlow_townes_t_per_sqrthz)
        assert full_result.schawlow_townes_t_per_sqrthz > 0.0

    def test_friis_floor_finite_and_positive(self, full_result: SensitivityResult) -> None:
        assert math.isfinite(full_result.friis_t_per_sqrthz)
        assert full_result.friis_t_per_sqrthz > 0.0

    def test_friis_floor_smaller_than_thermal(self, full_result: SensitivityResult) -> None:
        assert full_result.friis_t_per_sqrthz < full_result.thermal_snr_t_per_sqrthz

    def test_advantage_greater_than_10db(self, full_result: SensitivityResult) -> None:
        """Quantum Friis advantage should be substantial (> 10 dB field sensitivity)."""
        assert full_result.friis_advantage_over_thermal_db > 10.0

    def test_result_exported_from_physics_init(self) -> None:
        """SensitivityResult and compute_sensitivity importable from physics package."""
        from nv_maser.physics import (  # noqa: F401
            SensitivityResult,
            compute_friis_sensitivity,
            compute_schawlow_townes_sensitivity,
            compute_sensitivity,
            compute_thermal_sensitivity,
        )
