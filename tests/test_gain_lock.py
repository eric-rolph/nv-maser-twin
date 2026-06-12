"""Tests for gain-lock PI control loop (R10 maser threshold stabilisation)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nv_maser.config import CavityConfig, MaserConfig, NVConfig, OpticalPumpConfig
from nv_maser.physics.gain_lock import (
    GainLockConfig,
    GainLockResult,
    GainLockStep,
    compute_cooperativity,
    find_threshold_pump_power,
    run_gain_lock_simulation,
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
def pump_tmpl() -> OpticalPumpConfig:
    """Default 532 nm pump template (2 W starting point)."""
    return OpticalPumpConfig()


@pytest.fixture
def lock_cfg() -> GainLockConfig:
    """Default gain-lock config."""
    return GainLockConfig()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


# ── GainLockConfig ────────────────────────────────────────────────


class TestGainLockConfig:
    def test_defaults(self, lock_cfg: GainLockConfig) -> None:
        assert lock_cfg.target_cooperativity == pytest.approx(1.10)
        assert lock_cfg.kp > 0
        assert lock_cfg.ki > 0
        assert lock_cfg.dt_us > 0
        assert lock_cfg.min_pump_power_w > 0
        assert lock_cfg.max_pump_power_w > lock_cfg.min_pump_power_w
        assert lock_cfg.coop_noise_sigma == 0.0
        assert lock_cfg.lock_tolerance > 0

    def test_custom(self) -> None:
        cfg = GainLockConfig(target_cooperativity=1.2, kp=0.1, ki=10.0)
        assert cfg.target_cooperativity == pytest.approx(1.2)
        assert cfg.kp == pytest.approx(0.1)
        assert cfg.ki == pytest.approx(10.0)

    def test_frozen(self, lock_cfg: GainLockConfig) -> None:
        with pytest.raises((AttributeError, TypeError)):
            lock_cfg.kp = 999.0  # type: ignore[misc]


# ── GainLockStep ──────────────────────────────────────────────────


class TestGainLockStep:
    def test_construction(self) -> None:
        step = GainLockStep(
            step=0, time_us=10.0, pump_power_w=2.0,
            cooperativity=1.05, error=0.05, integral=0.0, locked=True,
        )
        assert step.step == 0
        assert step.locked is True

    def test_locked_field_false(self) -> None:
        step = GainLockStep(
            step=5, time_us=50.0, pump_power_w=0.5,
            cooperativity=0.3, error=0.8, integral=1e-4, locked=False,
        )
        assert step.locked is False

    def test_frozen(self) -> None:
        step = GainLockStep(
            step=0, time_us=10.0, pump_power_w=2.0,
            cooperativity=1.0, error=0.1, integral=0.0, locked=False,
        )
        with pytest.raises((AttributeError, TypeError)):
            step.cooperativity = 99.0  # type: ignore[misc]


# ── compute_cooperativity ─────────────────────────────────────────


class TestComputeCooperativity:
    def test_zero_pump_returns_zero(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        c = compute_cooperativity(0.0, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl)
        assert c == 0.0

    def test_returns_float(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        c = compute_cooperativity(2.0, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl)
        assert isinstance(c, float)
        assert math.isfinite(c)

    def test_positive_at_nominal_power(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        c = compute_cooperativity(2.0, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl)
        assert c > 0.0

    def test_monotone_with_power(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        """Higher pump power → higher cooperativity."""
        c_low = compute_cooperativity(0.1, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl)
        c_high = compute_cooperativity(5.0, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl)
        assert c_high > c_low

    def test_above_threshold_at_nominal(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        """Default 2 W pump should drive the maser above threshold (C > 1)."""
        c = compute_cooperativity(2.0, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl)
        assert c > 1.0

    def test_below_threshold_at_very_low_power(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        """Near-zero pump should be below threshold."""
        c = compute_cooperativity(1e-3, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl)
        assert c < 1.0

    def test_gain_budget_scales_c(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        """Lower gain_budget (worse field quality) reduces cooperativity."""
        c_ideal = compute_cooperativity(
            2.0, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, gain_budget=1.0
        )
        c_degrade = compute_cooperativity(
            2.0, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, gain_budget=0.5
        )
        assert c_ideal > c_degrade

    def test_no_mutation_of_configs(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        """Config objects must not be mutated by the function."""
        original_eta = nv_cfg.pump_efficiency
        original_power = pump_tmpl.laser_power_w
        compute_cooperativity(0.5, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl)
        assert nv_cfg.pump_efficiency == pytest.approx(original_eta)
        assert pump_tmpl.laser_power_w == pytest.approx(original_power)


# ── find_threshold_pump_power ────────────────────────────────────


class TestFindThresholdPumpPower:
    def test_returns_positive_float(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        p_th = find_threshold_pump_power(
            nv_cfg, maser_cfg, cavity_cfg, pump_tmpl
        )
        assert isinstance(p_th, float)
        assert p_th > 0.0

    def test_cooperativity_near_one_at_threshold(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        """C(P_th) should be ≈ 1.0 within bisection tolerance."""
        p_th = find_threshold_pump_power(
            nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, tol=1e-4
        )
        c_th = compute_cooperativity(
            p_th, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl
        )
        assert c_th == pytest.approx(1.0, abs=0.01)

    def test_threshold_within_bracket(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        p_th = find_threshold_pump_power(
            nv_cfg, maser_cfg, cavity_cfg, pump_tmpl,
            p_lo=1e-3, p_hi=10.0,
        )
        assert 1e-3 <= p_th <= 10.0

    def test_raises_when_lo_is_above_threshold(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        """ValueError when C(p_lo) ≥ 1."""
        with pytest.raises(ValueError, match="already above threshold"):
            find_threshold_pump_power(
                nv_cfg, maser_cfg, cavity_cfg, pump_tmpl,
                p_lo=5.0, p_hi=10.0,  # both well above threshold
            )

    def test_raises_when_hi_is_below_threshold(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        """ValueError when C(p_hi) < 1."""
        with pytest.raises(ValueError, match="unreachable"):
            find_threshold_pump_power(
                nv_cfg, maser_cfg, cavity_cfg, pump_tmpl,
                p_lo=1e-6, p_hi=1e-5,  # both far below threshold
            )

    def test_threshold_below_nominal_power(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        """Threshold power should be less than the nominal 2 W."""
        p_th = find_threshold_pump_power(
            nv_cfg, maser_cfg, cavity_cfg, pump_tmpl
        )
        assert p_th < pump_tmpl.laser_power_w


# ── run_gain_lock_simulation ─────────────────────────────────────


class TestRunGainLockSimulation:
    def test_returns_gain_lock_result(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
        lock_cfg: GainLockConfig,
    ) -> None:
        result = run_gain_lock_simulation(
            10, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, lock_cfg
        )
        assert isinstance(result, GainLockResult)

    def test_step_count_matches(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
        lock_cfg: GainLockConfig,
    ) -> None:
        n = 25
        result = run_gain_lock_simulation(
            n, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, lock_cfg
        )
        assert result.n_steps == n
        assert len(result.steps) == n

    def test_raises_zero_steps(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
        lock_cfg: GainLockConfig,
    ) -> None:
        with pytest.raises(ValueError):
            run_gain_lock_simulation(
                0, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, lock_cfg
            )

    def test_pump_stays_within_bounds(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
        lock_cfg: GainLockConfig,
    ) -> None:
        result = run_gain_lock_simulation(
            50, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, lock_cfg
        )
        for step in result.steps:
            assert lock_cfg.min_pump_power_w <= step.pump_power_w <= lock_cfg.max_pump_power_w

    def test_error_matches_cooperativity(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
        lock_cfg: GainLockConfig,
    ) -> None:
        """error = target_cooperativity − cooperativity at each step."""
        result = run_gain_lock_simulation(
            10, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, lock_cfg
        )
        for step in result.steps:
            expected_error = lock_cfg.target_cooperativity - step.cooperativity
            assert step.error == pytest.approx(expected_error, abs=1e-9)

    def test_locked_consistent_with_tolerance(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
        lock_cfg: GainLockConfig,
    ) -> None:
        """locked ↔ |error| < lock_tolerance."""
        result = run_gain_lock_simulation(
            20, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, lock_cfg
        )
        for step in result.steps:
            expected_locked = abs(step.error) < lock_cfg.lock_tolerance
            assert step.locked == expected_locked

    def test_time_stamp_increments(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
        lock_cfg: GainLockConfig,
    ) -> None:
        """time_us should increase by dt_us each step."""
        result = run_gain_lock_simulation(
            5, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, lock_cfg
        )
        for i, step in enumerate(result.steps):
            assert step.time_us == pytest.approx((i + 1) * lock_cfg.dt_us)

    def test_converges_from_underpump(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        """Starting below threshold, loop should lock within 200 steps."""
        cfg = GainLockConfig(kp=0.001, ki=1.0, dt_us=10.0, target_cooperativity=1.10)
        result = run_gain_lock_simulation(
            200,
            nv_cfg,
            maser_cfg,
            cavity_cfg,
            pump_tmpl,
            cfg,
            initial_pump_power_w=0.001,  # below threshold (~3 mW)
        )
        assert result.locked_at_step >= 0, (
            "Loop did not lock within 200 steps starting from underpump. "
            f"Final cooperativity = {result.final_cooperativity:.4f}"
        )

    def test_converges_from_overpump(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        """Starting moderately above target, loop should lock within 200 steps."""
        cfg = GainLockConfig(kp=0.001, ki=1.0, dt_us=10.0, target_cooperativity=1.10)
        result = run_gain_lock_simulation(
            200,
            nv_cfg,
            maser_cfg,
            cavity_cfg,
            pump_tmpl,
            cfg,
            initial_pump_power_w=0.01,  # above threshold (~3 mW), C≈~2
        )
        assert result.locked_at_step >= 0, (
            "Loop did not lock within 200 steps starting from overpump. "
            f"Final cooperativity = {result.final_cooperativity:.4f}"
        )

    def test_noise_does_not_crash(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
        rng: np.random.Generator,
    ) -> None:
        """Noise injection must not raise or produce NaN."""
        cfg = GainLockConfig(coop_noise_sigma=0.05)
        result = run_gain_lock_simulation(
            20, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, cfg, rng=rng
        )
        for step in result.steps:
            assert math.isfinite(step.pump_power_w)
            assert math.isfinite(step.cooperativity)

    def test_final_fields_consistent(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
        lock_cfg: GainLockConfig,
    ) -> None:
        result = run_gain_lock_simulation(
            10, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, lock_cfg
        )
        last = result.steps[-1]
        assert result.final_cooperativity == pytest.approx(last.cooperativity)
        assert result.final_pump_power_w == pytest.approx(last.pump_power_w)
        assert result.converged == last.locked

    def test_locked_at_step_is_first_locked(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        """locked_at_step must be the index of the first locked step."""
        cfg = GainLockConfig(kp=0.2, ki=100.0, dt_us=10.0)
        result = run_gain_lock_simulation(
            200, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, cfg,
            initial_pump_power_w=0.01,
        )
        if result.locked_at_step >= 0:
            assert result.steps[result.locked_at_step].locked is True
            for step in result.steps[: result.locked_at_step]:
                assert step.locked is False

    def test_never_locked_returns_neg_one(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        """When the loop never converges, locked_at_step = -1 and locked_at_us is NaN."""
        # Tiny gains → loop barely moves
        cfg = GainLockConfig(kp=1e-10, ki=1e-10, dt_us=10.0)
        result = run_gain_lock_simulation(
            3, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, cfg,
            initial_pump_power_w=0.002,  # far from target
        )
        if not result.converged:
            assert result.locked_at_step == -1
            assert math.isnan(result.locked_at_us)

    def test_reproducible_with_rng(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        """Same seed → identical simulation trajectories."""
        cfg = GainLockConfig(coop_noise_sigma=0.02)
        res_a = run_gain_lock_simulation(
            15, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, cfg,
            rng=np.random.default_rng(99),
        )
        res_b = run_gain_lock_simulation(
            15, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, cfg,
            rng=np.random.default_rng(99),
        )
        for a, b in zip(res_a.steps, res_b.steps):
            assert a.cooperativity == pytest.approx(b.cooperativity)
            assert a.pump_power_w == pytest.approx(b.pump_power_w)

    def test_step_index_matches_position(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
        lock_cfg: GainLockConfig,
    ) -> None:
        result = run_gain_lock_simulation(
            8, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, lock_cfg
        )
        for i, step in enumerate(result.steps):
            assert step.step == i


# ── GainLockResult properties ─────────────────────────────────────


class TestGainLockResult:
    def test_n_steps_property(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
        lock_cfg: GainLockConfig,
    ) -> None:
        n = 12
        result = run_gain_lock_simulation(
            n, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, lock_cfg
        )
        assert result.n_steps == n

    def test_lock_time_t1_units_nan_when_unlocked(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        cfg = GainLockConfig(kp=1e-12, ki=1e-12)
        result = run_gain_lock_simulation(
            3, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, cfg,
            initial_pump_power_w=0.002,
        )
        if result.locked_at_step < 0:
            assert math.isnan(result.lock_time_t1_units)

    def test_lock_time_t1_units_finite_when_locked(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        pump_tmpl: OpticalPumpConfig,
    ) -> None:
        cfg = GainLockConfig(kp=0.2, ki=100.0)
        result = run_gain_lock_simulation(
            500, nv_cfg, maser_cfg, cavity_cfg, pump_tmpl, cfg,
            initial_pump_power_w=0.01,
        )
        if result.locked_at_step >= 0:
            assert math.isfinite(result.lock_time_t1_units)
            assert result.lock_time_t1_units > 0


# ── Public API available from physics package ────────────────────


class TestPublicAPI:
    def test_importable_from_physics_package(self) -> None:
        from nv_maser.physics import (  # noqa: F401
            GainLockConfig,
            GainLockResult,
            GainLockStep,
            compute_cooperativity,
            find_threshold_pump_power,
            run_gain_lock_simulation,
        )
