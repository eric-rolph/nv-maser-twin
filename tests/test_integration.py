"""
Integration tests verifying closed feedback loops across physics modules.

These tests confirm that changes in one subsystem (e.g., optical pump power)
propagate correctly through the coupled chain to downstream outputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from nv_maser.config import SimConfig
from nv_maser.physics.environment import FieldEnvironment
from nv_maser.model.loss import PhysicsInformedLoss
from nv_maser.rl.env import ShimmingEnv
import torch


# ── Helpers ────────────────────────────────────────────────────────────


def _make_env(
    laser_power_w: float = 2.0,
    grid_size: int = 16,
    seed: int = 42,
    **thermal_overrides: float,
) -> FieldEnvironment:
    """Create a FieldEnvironment with specified pump and thermal settings."""
    thermal_kwargs = {
        "thermal_noise_std_c": 0.0,
        "thermal_drift_rate_c_per_s": 0.0,
        **thermal_overrides,
    }
    cfg = SimConfig(
        grid={"size": grid_size},
        disturbance={"seed": seed},
        optical_pump={"laser_power_w": laser_power_w},
        thermal=thermal_kwargs,
    )
    return FieldEnvironment(cfg, thermal_seed=0)


# ═══════════════════════════════════════════════════════════════════════
# D1: pump → inversion → threshold chain
# ═══════════════════════════════════════════════════════════════════════


class TestPumpInversionChain:
    """Verify that laser power feeds through to cooperativity and SNR."""

    def test_higher_pump_increases_cooperativity(self) -> None:
        """More pump power → higher effective efficiency → higher cooperativity."""
        env_low = _make_env(laser_power_w=0.5)
        env_high = _make_env(laser_power_w=4.0)

        net_low = env_low.step(t=0.0)
        net_high = env_high.step(t=0.0)

        m_low = env_low.compute_uniformity_metric(net_low)
        m_high = env_high.compute_uniformity_metric(net_high)

        assert m_high["cooperativity"] > m_low["cooperativity"]

    def test_higher_pump_increases_snr(self) -> None:
        """More pump power → more emission → higher SNR."""
        env_low = _make_env(laser_power_w=0.5)
        env_high = _make_env(laser_power_w=4.0)

        net_low = env_low.step(t=0.0)
        net_high = env_high.step(t=0.0)

        m_low = env_low.compute_uniformity_metric(net_low)
        m_high = env_high.compute_uniformity_metric(net_high)

        assert m_high["snr_db"] > m_low["snr_db"]

    def test_effective_pump_efficiency_varies_with_power(self) -> None:
        """Pump at different powers gives different effective efficiencies."""
        env_low = _make_env(laser_power_w=0.1)
        env_high = _make_env(laser_power_w=10.0)

        net_low = env_low.step(t=0.0)
        net_high = env_high.step(t=0.0)

        m_low = env_low.compute_uniformity_metric(net_low)
        m_high = env_high.compute_uniformity_metric(net_high)

        # Low power → far from saturation, high power → near saturation 2/3
        assert m_low["effective_pump_efficiency"] < m_high["effective_pump_efficiency"]
        assert m_high["effective_pump_efficiency"] <= 2.0 / 3.0 + 1e-6

    def test_zero_pump_gives_zero_cooperativity(self) -> None:
        """No pump → no inversion → cooperativity collapses."""
        env = _make_env(laser_power_w=0.0)
        net = env.step(t=0.0)
        m = env.compute_uniformity_metric(net)

        assert m["effective_pump_efficiency"] == 0.0
        assert m["cooperativity"] == pytest.approx(0.0, abs=1e-12)

    def test_pump_efficiency_feeds_into_signal_chain(self) -> None:
        """Signal chain power should scale with effective pump efficiency."""
        env_low = _make_env(laser_power_w=0.1)
        env_high = _make_env(laser_power_w=10.0)

        net_low = env_low.step(t=0.0)
        net_high = env_high.step(t=0.0)

        m_low = env_low.compute_uniformity_metric(net_low)
        m_high = env_high.compute_uniformity_metric(net_high)

        assert m_high["received_power_w"] > m_low["received_power_w"]

    def test_n_effective_scales_with_pump(self) -> None:
        """n_effective (inverted spins) should increase with pump power."""
        env_low = _make_env(laser_power_w=0.2)
        env_high = _make_env(laser_power_w=5.0)

        net_low = env_low.step(t=0.0)
        net_high = env_high.step(t=0.0)

        m_low = env_low.compute_uniformity_metric(net_low)
        m_high = env_high.compute_uniformity_metric(net_high)

        assert m_high["n_effective"] > m_low["n_effective"]


# ═══════════════════════════════════════════════════════════════════════
# D2: pump → thermal → T2* → gain chain
# ═══════════════════════════════════════════════════════════════════════


class TestPumpThermalChain:
    """Verify that pump heating feeds through to temperature and T2*."""

    def test_pump_heats_diamond(self) -> None:
        """Nonzero laser → higher temperature than zero laser."""
        env_heated = _make_env(laser_power_w=4.0)
        env_cold = _make_env(laser_power_w=0.0)

        env_heated.step(t=0.0)
        env_cold.step(t=0.0)

        t_heated = env_heated.thermal_state.temperature_c
        t_cold = env_cold.thermal_state.temperature_c

        assert t_heated > t_cold

    def test_zero_pump_no_extra_heat(self) -> None:
        """Zero laser power → temperature equals ambient (no extra heat)."""
        env = _make_env(laser_power_w=0.0)
        env.step(t=0.0)

        assert env.thermal_state.temperature_c == pytest.approx(25.0, abs=0.01)

    def test_pump_heat_proportional_to_power(self) -> None:
        """More pump power → proportionally more heating."""
        env_1w = _make_env(laser_power_w=1.0)
        env_4w = _make_env(laser_power_w=4.0)

        env_1w.step(t=0.0)
        env_4w.step(t=0.0)

        # Temperature rise should be roughly proportional
        t_ref = 25.0  # ambient
        rise_1w = env_1w.thermal_state.temperature_c - t_ref
        rise_4w = env_4w.thermal_state.temperature_c - t_ref

        # Not exactly 4× due to Beer-Lambert absorption saturation,
        # but should be significantly more
        assert rise_4w > rise_1w * 2.0

    def test_pump_heating_degrades_t2_star(self) -> None:
        """Higher temperature from pump → shorter effective T2*."""
        env_heated = _make_env(laser_power_w=8.0)
        env_cold = _make_env(laser_power_w=0.0)

        env_heated.step(t=0.0)
        env_cold.step(t=0.0)

        assert env_heated.thermal_state.effective_t2_star_us < \
               env_cold.thermal_state.effective_t2_star_us

    def test_pump_heating_degrades_cavity_q(self) -> None:
        """Higher temperature from pump → lower effective cavity Q."""
        env_heated = _make_env(laser_power_w=8.0)
        env_cold = _make_env(laser_power_w=0.0)

        env_heated.step(t=0.0)
        env_cold.step(t=0.0)

        assert env_heated.thermal_state.effective_cavity_q < \
               env_cold.thermal_state.effective_cavity_q

    def test_thermal_load_matches_pump_thermal_load(self) -> None:
        """The ThermalConfig.external_heat_w should match pump thermal_load_w."""
        from nv_maser.physics.optical_pump import compute_pump_state

        cfg = SimConfig(optical_pump={"laser_power_w": 3.0})
        pump = compute_pump_state(cfg.optical_pump, cfg.nv)

        env = FieldEnvironment(cfg, thermal_seed=0)
        # The environment should have injected the pump heat
        assert env.thermal_model.config.external_heat_w == pytest.approx(
            pump.thermal_load_w, rel=1e-6
        )

    def test_explicit_external_heat_respected(self) -> None:
        """If user explicitly sets external_heat_w, pump doesn't override it."""
        cfg = SimConfig(
            thermal={"external_heat_w": 99.0},
            optical_pump={"laser_power_w": 2.0},
        )
        env = FieldEnvironment(cfg, thermal_seed=0)
        # User-provided value should be preserved (pump doesn't override nonzero)
        assert env.thermal_model.config.external_heat_w == 99.0


# ═══════════════════════════════════════════════════════════════════════
# Integration: end-to-end pump → inversion → thermal → gain budget
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEndChain:
    """Verify the full chain: pump → (inversion + thermal) → metrics."""

    def test_high_pump_improves_gain_budget(self) -> None:
        """More pump power → higher gain budget (despite thermal degradation)."""
        env_low = _make_env(laser_power_w=0.5)
        env_high = _make_env(laser_power_w=3.0)

        net_low = env_low.step(t=0.0)
        net_high = env_high.step(t=0.0)

        m_low = env_low.compute_uniformity_metric(net_low)
        m_high = env_high.compute_uniformity_metric(net_high)

        # At moderate powers, inversion benefit dominates thermal degradation
        assert m_high["gain_budget"] > m_low["gain_budget"]

    def test_metrics_self_consistent(self) -> None:
        """All returned metrics are finite and within physical bounds."""
        env = _make_env(laser_power_w=2.0)
        net = env.step(t=0.0)
        m = env.compute_uniformity_metric(net)

        assert np.isfinite(m["snr_db"])
        assert np.isfinite(m["cooperativity"])
        assert m["cooperativity"] >= 0
        assert 0 <= m["effective_pump_efficiency"] <= 1.0
        assert m["gain_budget"] > 0
        assert m["thermal_load_w"] >= 0
        assert m["pump_saturation"] >= 0
        assert m["pump_saturation"] <= 1.0

    def test_pump_saturation_bounded(self) -> None:
        """Even at extreme power, saturation stays in [0, 1]."""
        env = _make_env(laser_power_w=100.0)
        net = env.step(t=0.0)
        m = env.compute_uniformity_metric(net)

        assert 0.0 <= m["pump_saturation"] <= 1.0
        assert 0.0 <= m["effective_pump_efficiency"] <= 2.0 / 3.0 + 1e-6


# ═══════════════════════════════════════════════════════════════════════
# PhysicsInformedLoss integration
# ═══════════════════════════════════════════════════════════════════════


class TestPhysicsInformedLoss:
    """Verify the PhysicsInformedLoss computes and returns physics metrics."""

    def test_physics_loss_includes_physics_metrics(self) -> None:
        """PhysicsInformedLoss output includes gain_budget and cooperativity."""
        cfg = SimConfig(grid={"size": 16}, disturbance={"seed": 42})
        env = FieldEnvironment(cfg, thermal_seed=0)
        mask = torch.tensor(env.grid.active_zone_mask, dtype=torch.bool)
        loss_fn = PhysicsInformedLoss(mask, env, gain_budget_weight=1e-5)

        net = torch.randn(2, 16, 16)
        currents = torch.randn(2, cfg.coils.num_coils)
        loss, metrics = loss_fn(net, currents)

        assert "gain_budget" in metrics
        assert "cooperativity" in metrics
        assert "physics_penalty" in metrics
        assert loss.item() > 0

    def test_physics_loss_ge_base_loss(self) -> None:
        """Physics loss is always ≥ base FieldUniformityLoss (penalties are non-negative)."""
        from nv_maser.model.loss import FieldUniformityLoss

        cfg = SimConfig(grid={"size": 16}, disturbance={"seed": 42})
        env = FieldEnvironment(cfg, thermal_seed=0)
        mask = torch.tensor(env.grid.active_zone_mask, dtype=torch.bool)

        base_fn = FieldUniformityLoss(mask, 1e-6)
        phys_fn = PhysicsInformedLoss(mask, env, 1e-6, 1e-5, 1e-5)

        net = torch.randn(2, 16, 16)
        currents = torch.randn(2, cfg.coils.num_coils)

        base_loss, _ = base_fn(net, currents)
        phys_loss, _ = phys_fn(net, currents)

        assert phys_loss.item() >= base_loss.item() - 1e-6

    def test_physics_penalty_same_order_as_variance(self) -> None:
        """With default weights, physics penalty should be within 100x of field variance."""
        cfg = SimConfig(grid={"size": 16}, disturbance={"seed": 42})
        env = FieldEnvironment(cfg, thermal_seed=0)
        mask = torch.tensor(env.grid.active_zone_mask, dtype=torch.bool)
        loss_fn = PhysicsInformedLoss(mask, env)

        # Use the actual distorted field (realistic input)
        net_field = env.step(t=0.0)
        net = torch.tensor(net_field, dtype=torch.float32).unsqueeze(0)
        currents = torch.zeros(1, cfg.coils.num_coils)
        _, metrics = loss_fn(net, currents)

        fv = metrics["field_variance"]
        pp = metrics["physics_penalty"]
        # Physics penalty should be within 2 orders of magnitude of field variance
        # (not 4+ orders like the old 1/gain_budget formulation)
        assert pp < fv * 100, f"physics_penalty {pp:.4e} >> field_variance {fv:.4e}"
        assert pp > 0, "physics_penalty should be positive"


# ═══════════════════════════════════════════════════════════════════════
# RL reward shaping integration
# ═══════════════════════════════════════════════════════════════════════


class TestRLRewardShaping:
    """Verify RL reward shaping produces physics info in step() output."""

    def test_reward_shaping_adds_physics_info(self) -> None:
        """With reward_shaping=True, step info includes gain_budget."""
        cfg = SimConfig(
            grid={"size": 16},
            disturbance={"seed": 42},
            training={"reward_shaping": True, "reward_shaping_weight": 0.1},
        )
        env = ShimmingEnv(cfg)
        env.reset(seed=0)
        action = np.zeros(cfg.coils.num_coils, dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert "gain_budget" in info
        assert "cooperativity" in info
        assert "snr_db" in info

    def test_reward_shaping_disabled_no_physics(self) -> None:
        """With reward_shaping=False (default), info has only variance."""
        cfg = SimConfig(
            grid={"size": 16},
            disturbance={"seed": 42},
        )
        env = ShimmingEnv(cfg)
        env.reset(seed=0)
        action = np.zeros(cfg.coils.num_coils, dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert "variance" in info
        assert "gain_budget" not in info

    def test_shaped_reward_higher_than_unshaped(self) -> None:
        """Shaped reward includes gain_budget bonus, so typically higher."""
        cfg_shaped = SimConfig(
            grid={"size": 16},
            disturbance={"seed": 42},
            training={"reward_shaping": True, "reward_shaping_weight": 0.5},
        )
        cfg_plain = SimConfig(
            grid={"size": 16},
            disturbance={"seed": 42},
        )
        env_shaped = ShimmingEnv(cfg_shaped)
        env_plain = ShimmingEnv(cfg_plain)

        env_shaped.reset(seed=0)
        env_plain.reset(seed=0)

        action = np.zeros(cfg_shaped.coils.num_coils, dtype=np.float32)
        _, r_shaped, _, _, _ = env_shaped.step(action)
        _, r_plain, _, _, _ = env_plain.step(action)

        # Shaped reward = variance_delta + γ × gain_budget
        # gain_budget > 0, so shaped reward should be higher
        assert r_shaped > r_plain
