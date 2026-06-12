"""Tests for ProbeShimmingEnv — probe-aware RL shimming environment (SS12)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nv_maser.physics.disturbance import ImagingMagnetDisturbanceConfig
from nv_maser.rl.probe_env import ProbeShimmingConfig, ProbeShimmingEnv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def env() -> ProbeShimmingEnv:
    return ProbeShimmingEnv()


@pytest.fixture()
def reset_env(env: ProbeShimmingEnv) -> tuple[ProbeShimmingEnv, np.ndarray]:
    obs, _ = env.reset(seed=0)
    return env, obs


# ---------------------------------------------------------------------------
# TestProbeShimmingConfig
# ---------------------------------------------------------------------------


class TestProbeShimmingConfig:
    def test_defaults_have_imaging_magnet(self) -> None:
        cfg = ProbeShimmingConfig()
        assert isinstance(cfg.imaging_magnet, ImagingMagnetDisturbanceConfig)

    def test_default_probe_snr_weight_positive(self) -> None:
        cfg = ProbeShimmingConfig()
        assert cfg.probe_snr_weight > 0.0

    def test_default_use_probe_reward_false(self) -> None:
        """use_probe_reward defaults to False for backward-compatible reward scaling."""
        cfg = ProbeShimmingConfig()
        assert cfg.use_probe_reward is False

    def test_custom_params_accepted(self) -> None:
        cfg = ProbeShimmingConfig(probe_snr_weight=0.5, use_probe_reward=True)
        assert cfg.probe_snr_weight == pytest.approx(0.5)
        assert cfg.use_probe_reward is True


# ---------------------------------------------------------------------------
# TestProbeShimmingEnvConstruction
# ---------------------------------------------------------------------------


class TestProbeShimmingEnvConstruction:
    def test_constructs_with_defaults(self) -> None:
        e = ProbeShimmingEnv()
        assert e is not None

    def test_constructs_with_custom_probe_config(self) -> None:
        pcfg = ProbeShimmingConfig(
            imaging_magnet=ImagingMagnetDisturbanceConfig(offset_x_mm=60.0),
            probe_snr_weight=0.2,
            use_probe_reward=True,
        )
        e = ProbeShimmingEnv(probe_config=pcfg)
        assert e is not None

    def test_stray_field_rms_nonzero(self) -> None:
        """Default imaging magnet should produce a non-zero RMS stray field."""
        e = ProbeShimmingEnv()
        assert e._stray_field_rms_mt > 0.0

    def test_stray_field_rms_finite(self) -> None:
        """Stray field RMS must be a finite number."""
        e = ProbeShimmingEnv()
        assert math.isfinite(e._stray_field_rms_mt)

    def test_probe_config_stored(self) -> None:
        pcfg = ProbeShimmingConfig(probe_snr_weight=0.3)
        e = ProbeShimmingEnv(probe_config=pcfg)
        assert e.probe_config.probe_snr_weight == pytest.approx(0.3)

    def test_stray_field_rms_property(self) -> None:
        e = ProbeShimmingEnv()
        assert e.stray_field_rms_mt == pytest.approx(e._stray_field_rms_mt)


# ---------------------------------------------------------------------------
# TestProbeShimmingEnvReset
# ---------------------------------------------------------------------------


class TestProbeShimmingEnvReset:
    def test_reset_returns_correct_obs_shape(self, env: ProbeShimmingEnv) -> None:
        obs, info = env.reset()
        assert obs.shape == (1, 64, 64)

    def test_reset_returns_float32_obs(self, env: ProbeShimmingEnv) -> None:
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_reset_returns_dict_info(self, env: ProbeShimmingEnv) -> None:
        _, info = env.reset()
        assert isinstance(info, dict)

    def test_reset_with_seed_reproducible(self, env: ProbeShimmingEnv) -> None:
        obs1, _ = env.reset(seed=99)
        obs2, _ = env.reset(seed=99)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_multiple_times_no_error(self, env: ProbeShimmingEnv) -> None:
        for i in range(3):
            env.reset(seed=i)


# ---------------------------------------------------------------------------
# TestProbeShimmingEnvStep
# ---------------------------------------------------------------------------


class TestProbeShimmingEnvStep:
    def test_step_obs_shape_and_dtype(
        self, reset_env: tuple[ProbeShimmingEnv, np.ndarray]
    ) -> None:
        env, _ = reset_env
        obs, *_ = env.step(np.zeros(env.num_coils, dtype=np.float32))
        assert obs.shape == (1, 64, 64)
        assert obs.dtype == np.float32

    def test_step_reward_is_float(
        self, reset_env: tuple[ProbeShimmingEnv, np.ndarray]
    ) -> None:
        env, _ = reset_env
        _, reward, *_ = env.step(np.zeros(env.num_coils, dtype=np.float32))
        assert isinstance(reward, float)

    def test_step_info_has_maser_noise_temp(
        self, reset_env: tuple[ProbeShimmingEnv, np.ndarray]
    ) -> None:
        env, _ = reset_env
        _, _, _, _, info = env.step(np.zeros(env.num_coils, dtype=np.float32))
        assert "maser_noise_temp_k" in info
        assert math.isfinite(info["maser_noise_temp_k"])

    def test_step_info_has_probe_snr_db(
        self, reset_env: tuple[ProbeShimmingEnv, np.ndarray]
    ) -> None:
        env, _ = reset_env
        _, _, _, _, info = env.step(np.zeros(env.num_coils, dtype=np.float32))
        assert "probe_snr_db" in info
        assert math.isfinite(info["probe_snr_db"])

    def test_step_info_has_stray_field_rms_mt(
        self, reset_env: tuple[ProbeShimmingEnv, np.ndarray]
    ) -> None:
        env, _ = reset_env
        _, _, _, _, info = env.step(np.zeros(env.num_coils, dtype=np.float32))
        assert "stray_field_rms_mt" in info
        assert info["stray_field_rms_mt"] >= 0.0

    def test_step_returns_five_tuple(
        self, reset_env: tuple[ProbeShimmingEnv, np.ndarray]
    ) -> None:
        env, _ = reset_env
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.num_coils, dtype=np.float32)
        )
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_stray_field_rms_constant_across_steps(
        self, reset_env: tuple[ProbeShimmingEnv, np.ndarray]
    ) -> None:
        """The stray field RMS should be the same every step (static magnet)."""
        env, _ = reset_env
        action = np.zeros(env.num_coils, dtype=np.float32)
        values = []
        for _ in range(3):
            _, _, _, _, info = env.step(action)
            values.append(info["stray_field_rms_mt"])
        assert values[0] == pytest.approx(values[1]) == pytest.approx(values[2])


# ---------------------------------------------------------------------------
# TestProbeRewardShaping
# ---------------------------------------------------------------------------


class TestProbeRewardShaping:
    def test_probe_reward_disabled_by_default(self) -> None:
        """With use_probe_reward=False, two envs (same seed) give same reward."""
        e1 = ProbeShimmingEnv(probe_config=ProbeShimmingConfig(use_probe_reward=False))
        e2 = ProbeShimmingEnv(
            probe_config=ProbeShimmingConfig(use_probe_reward=False, probe_snr_weight=100.0)
        )
        e1.reset(seed=0)
        e2.reset(seed=0)
        action = np.zeros(e1.num_coils, dtype=np.float32)
        _, r1, _, _, _ = e1.step(action)
        _, r2, _, _, _ = e2.step(action)
        # Neither uses probe reward → rewards equal despite different probe_snr_weight
        assert r1 == pytest.approx(r2)

    def test_probe_reward_enabled_adds_snr_term(self) -> None:
        """Reward with use_probe_reward=True equals reward_off + weight * snr_db."""
        e_off = ProbeShimmingEnv(
            probe_config=ProbeShimmingConfig(use_probe_reward=False, probe_snr_weight=1.0)
        )
        e_on = ProbeShimmingEnv(
            probe_config=ProbeShimmingConfig(use_probe_reward=True, probe_snr_weight=1.0)
        )
        e_off.reset(seed=5)
        e_on.reset(seed=5)
        action = np.zeros(e_off.num_coils, dtype=np.float32)
        _, reward_off, _, _, _ = e_off.step(action)
        _, reward_on, _, _, info_on = e_on.step(action)
        expected_delta = 1.0 * info_on["probe_snr_db"]
        assert reward_on == pytest.approx(reward_off + expected_delta, rel=1e-5)

    def test_higher_probe_weight_scales_reward(self) -> None:
        """Doubling probe_snr_weight doubles the probe reward contribution."""
        e_w1 = ProbeShimmingEnv(
            probe_config=ProbeShimmingConfig(use_probe_reward=True, probe_snr_weight=0.1)
        )
        e_w2 = ProbeShimmingEnv(
            probe_config=ProbeShimmingConfig(use_probe_reward=True, probe_snr_weight=0.2)
        )
        e_w1.reset(seed=7)
        e_w2.reset(seed=7)
        action = np.zeros(e_w1.num_coils, dtype=np.float32)
        _, r1, _, _, info1 = e_w1.step(action)
        _, r2, _, _, info2 = e_w2.step(action)
        snr = info1["probe_snr_db"]  # same field → same SNR
        # r2 - r1 ≈ (0.2 - 0.1) * snr_db
        assert r2 - r1 == pytest.approx(0.1 * snr, rel=1e-5)


# ---------------------------------------------------------------------------
# TestSpaceCompatibility
# ---------------------------------------------------------------------------


class TestSpaceCompatibility:
    def test_observation_space_unchanged(self, env: ProbeShimmingEnv) -> None:
        """ProbeShimmingEnv must have the same obs space as ShimmingEnv."""
        from nv_maser.rl.env import ShimmingEnv

        base = ShimmingEnv()
        assert env.observation_space == base.observation_space

    def test_action_space_unchanged(self, env: ProbeShimmingEnv) -> None:
        from nv_maser.rl.env import ShimmingEnv

        base = ShimmingEnv()
        assert env.action_space == base.action_space

    def test_num_coils_unchanged(self, env: ProbeShimmingEnv) -> None:
        from nv_maser.rl.env import ShimmingEnv

        base = ShimmingEnv()
        assert env.num_coils == base.num_coils

    def test_max_steps_unchanged(self, env: ProbeShimmingEnv) -> None:
        from nv_maser.rl.env import ShimmingEnv

        base = ShimmingEnv()
        assert env.max_steps == base.max_steps


# ---------------------------------------------------------------------------
# TestEpisodeMechanics
# ---------------------------------------------------------------------------


class TestEpisodeMechanics:
    def test_action_clipping_no_crash(self, env: ProbeShimmingEnv) -> None:
        env.reset(seed=0)
        action = np.full(env.num_coils, 9999.0, dtype=np.float32)
        obs, reward, _, _, info = env.step(action)
        assert math.isfinite(reward)
        assert obs.shape == (1, 64, 64)
        assert math.isfinite(info["maser_noise_temp_k"])

    def test_episode_terminates_at_max_steps(self, env: ProbeShimmingEnv) -> None:
        env.reset(seed=0)
        action = np.zeros(env.num_coils, dtype=np.float32)
        terminated = False
        for _ in range(env.max_steps):
            _, _, terminated, _, _ = env.step(action)
        assert terminated

    def test_zero_action_no_crash(self, env: ProbeShimmingEnv) -> None:
        env.reset(seed=0)
        obs, reward, _, _, info = env.step(np.zeros(env.num_coils, dtype=np.float32))
        assert obs.shape == (1, 64, 64)
        assert math.isfinite(reward)
        assert "probe_snr_db" in info


# ---------------------------------------------------------------------------
# TestPhysicsCorrectness
# ---------------------------------------------------------------------------


class TestPhysicsCorrectness:
    def test_maser_noise_temp_positive(
        self, reset_env: tuple[ProbeShimmingEnv, np.ndarray]
    ) -> None:
        """Maser noise temperature is a positive physical quantity."""
        env, _ = reset_env
        _, _, _, _, info = env.step(np.zeros(env.num_coils, dtype=np.float32))
        assert info["maser_noise_temp_k"] > 0.0

    def test_stray_field_increases_with_larger_magnet(self) -> None:
        """Larger imaging magnet → larger stray field RMS on maser grid."""
        e_small = ProbeShimmingEnv(
            probe_config=ProbeShimmingConfig(
                imaging_magnet=ImagingMagnetDisturbanceConfig(magnet_volume_m3=1e-6)
            )
        )
        e_large = ProbeShimmingEnv(
            probe_config=ProbeShimmingConfig(
                imaging_magnet=ImagingMagnetDisturbanceConfig(magnet_volume_m3=4e-5)
            )
        )
        assert e_large._stray_field_rms_mt > e_small._stray_field_rms_mt

    def test_shielding_reduces_stray_field(self) -> None:
        """Mu-metal shielding (shield_attenuation_db > 0) reduces stray field RMS."""
        e_unshielded = ProbeShimmingEnv(
            probe_config=ProbeShimmingConfig(
                imaging_magnet=ImagingMagnetDisturbanceConfig(shield_attenuation_db=0.0)
            )
        )
        e_shielded = ProbeShimmingEnv(
            probe_config=ProbeShimmingConfig(
                imaging_magnet=ImagingMagnetDisturbanceConfig(shield_attenuation_db=40.0)
            )
        )
        assert e_shielded._stray_field_rms_mt < e_unshielded._stray_field_rms_mt

    def test_stray_field_rms_in_mt_range(self) -> None:
        """Default magnet (2e-5 m³ NdFeB at 40 mm) → RMS stray field in mT range."""
        e = ProbeShimmingEnv()
        # 20 mL NdFeB at 40 mm gives ~10–100 mT on the near field; accept > 0.1 mT
        assert e._stray_field_rms_mt > 0.1

    def test_imaging_magnet_field_property(self) -> None:
        """DisturbanceGenerator.imaging_magnet_field is non-None after construction."""
        e = ProbeShimmingEnv()
        imf = e._disturbance.imaging_magnet_field
        assert imf is not None
        assert imf.shape == (64, 64)
        assert imf.dtype == np.float32
