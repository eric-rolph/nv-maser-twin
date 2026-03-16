"""Tests for the Maxwell-Bloch time-domain solver."""
import math

import numpy as np
import pytest

from nv_maser.config import (
    CavityConfig,
    MaserConfig,
    MaxwellBlochConfig,
    NVConfig,
)
from nv_maser.physics.maxwell_bloch import (
    MaxwellBlochResult,
    _maxwell_bloch_rhs,
    compute_steady_state_power,
    solve_maxwell_bloch,
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
def mb_cfg() -> MaxwellBlochConfig:
    return MaxwellBlochConfig(enable=True, t_max_us=50.0, n_time_points=500)


@pytest.fixture
def mb_driven_cfg() -> MaxwellBlochConfig:
    return MaxwellBlochConfig(
        enable=True,
        drive_amplitude_hz=1e3,
        t_max_us=50.0,
        n_time_points=500,
    )


# ── Config ────────────────────────────────────────────────────────


class TestMaxwellBlochConfig:
    def test_defaults(self) -> None:
        cfg = MaxwellBlochConfig()
        assert cfg.enable is False
        assert cfg.drive_amplitude_hz == 0.0
        assert cfg.t_max_us == 100.0
        assert cfg.n_time_points == 1000
        assert cfg.solver_method == "RK45"

    def test_nv_t1_default(self) -> None:
        nv = NVConfig()
        assert nv.t1_ms == 5.0

    def test_nv_t1_custom(self) -> None:
        nv = NVConfig(t1_ms=10.0)
        assert nv.t1_ms == 10.0


# ── ODE RHS ───────────────────────────────────────────────────────


class TestODERHS:
    """Verify the right-hand side function directly."""

    def test_zero_state_no_drive(self) -> None:
        """All zeros except sz0-maintained inversion → only inversion recovery."""
        rhs = _maxwell_bloch_rhs(
            0, [0, 0, 0, 0, 0.2],
            kappa_c=1e6, kappa_s=1e6, g_ens=1e3,
            gamma=200, sz0=0.25, drive=0,
        )
        # da/dt = 0, ds/dt = 0 (no coupling since a=0 and s=0)
        assert rhs[0] == 0.0
        assert rhs[1] == 0.0
        assert rhs[2] == 0.0
        assert rhs[3] == 0.0
        # dsz/dt = -γ(sz - sz0) = -200*(0.2 - 0.25) = +10
        assert pytest.approx(rhs[4], rel=1e-10) == 10.0

    def test_cavity_decay(self) -> None:
        """Cavity amplitude decays at rate κ_c/2."""
        rhs = _maxwell_bloch_rhs(
            0, [1.0, 0.0, 0.0, 0.0, 0.0],
            kappa_c=2e6, kappa_s=1e6, g_ens=0,
            gamma=200, sz0=0.25, drive=0,
        )
        # dar/dt = -(κ_c/2)*ar = -1e6
        assert pytest.approx(rhs[0], rel=1e-10) == -1e6
        assert rhs[1] == 0.0

    def test_drive_only(self) -> None:
        """External drive appears in imaginary quadrature of cavity."""
        rhs = _maxwell_bloch_rhs(
            0, [0, 0, 0, 0, 0.25],
            kappa_c=1e6, kappa_s=1e6, g_ens=1e3,
            gamma=200, sz0=0.25, drive=100.0,
        )
        # dai/dt includes -drive term
        assert rhs[1] == pytest.approx(-100.0, rel=1e-10)

    def test_coupling_uses_g_ens(self) -> None:
        """g_ens appears in cavity, spin, and inversion equations."""
        g_ens = 1000.0
        rhs = _maxwell_bloch_rhs(
            0, [0, 0, 0, 1.0, 0.5],
            kappa_c=0, kappa_s=0, g_ens=g_ens,
            gamma=0, sz0=0, drive=0,
        )
        # dar/dt = g_ens * si = 1000 * 1.0
        assert pytest.approx(rhs[0], rel=1e-10) == 1000.0
        # dsi/dt = g_ens * ar * sz = 1000 * 0 * 0.5 = 0 (ar=0)
        assert rhs[3] == 0.0


# ── Solver ────────────────────────────────────────────────────────


class TestSolveMaxwellBloch:
    """Integration tests using the full solver."""

    def test_returns_result_type(
        self, nv_cfg, maser_cfg, cavity_cfg, mb_cfg
    ) -> None:
        result = solve_maxwell_bloch(nv_cfg, maser_cfg, cavity_cfg, mb_cfg)
        assert isinstance(result, MaxwellBlochResult)

    def test_array_shapes(
        self, nv_cfg, maser_cfg, cavity_cfg, mb_cfg
    ) -> None:
        result = solve_maxwell_bloch(nv_cfg, maser_cfg, cavity_cfg, mb_cfg)
        n = mb_cfg.n_time_points
        assert result.time_s.shape == (n,)
        assert result.cavity_re.shape == (n,)
        assert result.cavity_im.shape == (n,)
        assert result.coherence_re.shape == (n,)
        assert result.coherence_im.shape == (n,)
        assert result.inversion.shape == (n,)
        assert result.photon_number.shape == (n,)

    def test_time_span(
        self, nv_cfg, maser_cfg, cavity_cfg, mb_cfg
    ) -> None:
        result = solve_maxwell_bloch(nv_cfg, maser_cfg, cavity_cfg, mb_cfg)
        assert result.time_s[0] == pytest.approx(0.0)
        assert result.time_s[-1] == pytest.approx(mb_cfg.t_max_us * 1e-6)

    def test_photon_number_non_negative(
        self, nv_cfg, maser_cfg, cavity_cfg, mb_cfg
    ) -> None:
        result = solve_maxwell_bloch(nv_cfg, maser_cfg, cavity_cfg, mb_cfg)
        assert np.all(result.photon_number >= 0)

    def test_output_power_non_negative(
        self, nv_cfg, maser_cfg, cavity_cfg, mb_cfg
    ) -> None:
        result = solve_maxwell_bloch(nv_cfg, maser_cfg, cavity_cfg, mb_cfg)
        assert result.output_power_w >= 0

    def test_cooperativity_positive(
        self, nv_cfg, maser_cfg, cavity_cfg, mb_cfg
    ) -> None:
        result = solve_maxwell_bloch(nv_cfg, maser_cfg, cavity_cfg, mb_cfg)
        assert result.cooperativity > 0

    def test_no_gain_for_free_running(
        self, nv_cfg, maser_cfg, cavity_cfg, mb_cfg
    ) -> None:
        """gain_db is None when drive_amplitude is 0."""
        result = solve_maxwell_bloch(nv_cfg, maser_cfg, cavity_cfg, mb_cfg)
        assert result.gain_db is None

    def test_inversion_clamped_above_threshold(
        self, nv_cfg, maser_cfg, cavity_cfg
    ) -> None:
        """Above threshold, steady-state inversion should be below initial."""
        # Use long simulation to allow masing buildup (~100 µs)
        mb_long = MaxwellBlochConfig(
            enable=True, t_max_us=500.0, n_time_points=2000,
        )
        result = solve_maxwell_bloch(nv_cfg, maser_cfg, cavity_cfg, mb_long)
        if result.cooperativity > 1:
            # Inversion is clamped below the pump value
            sz0 = nv_cfg.pump_efficiency / 2.0
            assert result.steady_state_inversion < sz0


class TestBelowThreshold:
    """Verify behaviour when cooperativity < 1."""

    def test_no_masing_low_density(self, maser_cfg, cavity_cfg, mb_cfg) -> None:
        nv_low = NVConfig(nv_density_per_cm3=1e10)  # very low density
        result = solve_maxwell_bloch(nv_low, maser_cfg, cavity_cfg, mb_cfg)
        assert result.cooperativity < 1
        # Photon number stays negligible
        assert result.steady_state_photons < 1.0

    def test_no_power_below_threshold(self, maser_cfg, cavity_cfg, mb_cfg) -> None:
        nv_low = NVConfig(nv_density_per_cm3=1e10)
        result = solve_maxwell_bloch(nv_low, maser_cfg, cavity_cfg, mb_cfg)
        assert result.output_power_w < 1e-30


class TestDrivenMode:
    """Verify amplifier (driven) mode."""

    def test_driven_has_gain_db(
        self, nv_cfg, maser_cfg, cavity_cfg, mb_driven_cfg
    ) -> None:
        result = solve_maxwell_bloch(nv_cfg, maser_cfg, cavity_cfg, mb_driven_cfg)
        assert result.gain_db is not None

    def test_driven_produces_output(
        self, nv_cfg, maser_cfg, cavity_cfg, mb_driven_cfg
    ) -> None:
        result = solve_maxwell_bloch(nv_cfg, maser_cfg, cavity_cfg, mb_driven_cfg)
        assert result.output_power_w > 0

    def test_driven_below_threshold_still_transmits(
        self, maser_cfg, cavity_cfg, mb_driven_cfg
    ) -> None:
        """Even below threshold, a drive still puts photons in the cavity."""
        nv_low = NVConfig(nv_density_per_cm3=1e10)
        result = solve_maxwell_bloch(nv_low, maser_cfg, cavity_cfg, mb_driven_cfg)
        # There should be non-zero photon count from the drive
        assert result.steady_state_photons > 0


# ── Analytical steady-state ───────────────────────────────────────


class TestSteadyStatePower:
    def test_zero_below_threshold(self, maser_cfg, cavity_cfg) -> None:
        nv_low = NVConfig(nv_density_per_cm3=1e10)
        power = compute_steady_state_power(nv_low, maser_cfg, cavity_cfg)
        assert power == 0.0

    def test_positive_above_threshold(self, nv_cfg, maser_cfg, cavity_cfg) -> None:
        power = compute_steady_state_power(nv_cfg, maser_cfg, cavity_cfg)
        # Default config should be above threshold
        assert power > 0

    def test_increases_with_pump(self, maser_cfg, cavity_cfg) -> None:
        p1 = compute_steady_state_power(
            NVConfig(pump_efficiency=0.3), maser_cfg, cavity_cfg,
        )
        p2 = compute_steady_state_power(
            NVConfig(pump_efficiency=0.5), maser_cfg, cavity_cfg,
        )
        assert p2 > p1

    def test_increases_with_q(self, nv_cfg, cavity_cfg) -> None:
        p1 = compute_steady_state_power(
            nv_cfg, MaserConfig(cavity_q=5000), cavity_cfg,
        )
        p2 = compute_steady_state_power(
            nv_cfg, MaserConfig(cavity_q=20000), cavity_cfg,
        )
        assert p2 > p1

    def test_zero_gain_budget_zero_power(self, nv_cfg, maser_cfg, cavity_cfg) -> None:
        power = compute_steady_state_power(nv_cfg, maser_cfg, cavity_cfg, gain_budget=0.0)
        assert power == 0.0


# ── Numerical vs Analytical cross-validation ──────────────────────


class TestNumericalVsAnalytical:
    """Cross-validate ODE steady-state with analytical formula."""

    def test_powers_agree_above_threshold(
        self, nv_cfg, maser_cfg, cavity_cfg
    ) -> None:
        """Numerical and analytical power should agree within 50 %.

        The maser exhibits relaxation oscillations (spiking) with a
        recovery timescale set by T₁ ≈ 5 ms.  The ODE needs ~20 × T₁
        to converge to the true steady state.
        """
        mb_long = MaxwellBlochConfig(
            enable=True, t_max_us=100_000.0, n_time_points=5000,
        )
        numerical = solve_maxwell_bloch(nv_cfg, maser_cfg, cavity_cfg, mb_long)
        analytical = compute_steady_state_power(nv_cfg, maser_cfg, cavity_cfg)

        if analytical > 0 and numerical.output_power_w > 0:
            ratio = numerical.output_power_w / analytical
            assert 0.5 < ratio < 2.0, (
                f"Numerical {numerical.output_power_w:.2e} vs "
                f"analytical {analytical:.2e}, ratio={ratio:.2f}"
            )


# ── Environment integration ───────────────────────────────────────


class TestEnvironmentIntegration:
    """Verify MB metrics appear in environment output when enabled."""

    def test_mb_disabled_no_keys(self) -> None:
        from nv_maser.config import SimConfig
        from nv_maser.physics.environment import FieldEnvironment

        config = SimConfig()  # maxwell_bloch.enable defaults to False
        env = FieldEnvironment(config)
        field = env.step(0.0)
        net = env.apply_correction(np.zeros(config.coils.num_coils, dtype=np.float32))
        metrics = env.compute_uniformity_metric(net)
        assert "mb_output_power_w" not in metrics

    def test_mb_enabled_has_keys(self) -> None:
        from nv_maser.config import SimConfig
        from nv_maser.physics.environment import FieldEnvironment

        config = SimConfig(
            maxwell_bloch=MaxwellBlochConfig(
                enable=True, t_max_us=10.0, n_time_points=100,
            ),
        )
        env = FieldEnvironment(config)
        field = env.step(0.0)
        net = env.apply_correction(np.zeros(config.coils.num_coils, dtype=np.float32))
        metrics = env.compute_uniformity_metric(net)
        assert "mb_output_power_w" in metrics
        assert "mb_steady_state_photons" in metrics
        assert "mb_cooperativity" in metrics
        assert "mb_analytical_power_w" in metrics
