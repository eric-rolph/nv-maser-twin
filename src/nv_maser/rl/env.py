"""
Gym-compatible RL environment for magnetic field shimming.

Implements the ShimmingEnv without requiring gymnasium/gym as a dependency.
Observation: disturbance field (size, size) reshaped to (1, size, size).
Action:       coil current vector (num_coils,) in Amps.
Reward:       improvement in field variance over active zone.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import SimConfig
from ..physics.disturbance import DisturbanceGenerator
from ..physics.environment import FieldEnvironment


class ShimmingEnv:
    """
    RL environment for learning a magnetic shimming policy.

    The agent observes the current disturbance field and must output coil
    currents that cancel it — minimizing field variance over the active zone.

    No gym/gymnasium dependency: observation_space and action_space are
    plain dicts describing shape/dtype/bounds.
    """

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config if config is not None else SimConfig()
        self._env = FieldEnvironment(self.config)
        self._disturbance = DisturbanceGenerator(
            self._env.grid, self.config.disturbance
        )

        self.grid_size: int = self.config.grid.size
        self.num_coils: int = self.config.coils.num_coils
        self.max_current: float = self.config.coils.max_current_amps
        self.max_steps: int = 200

        self.observation_space: dict = {
            "shape": (1, self.grid_size, self.grid_size),
            "dtype": "float32",
        }
        self.action_space: dict = {
            "shape": (self.num_coils,),
            "low": -self.max_current,
            "high": self.max_current,
            "dtype": "float32",
        }

        # Episode state (initialised on reset)
        self._current_field: NDArray[np.float32] | None = None
        self._target_field: NDArray[np.float32] | None = None
        self._step_count: int = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> tuple[NDArray[np.float32], dict]:
        """
        Reset the environment with a fresh disturbance realisation.

        Args:
            seed: Optional RNG seed for reproducibility.

        Returns:
            (obs, info) where obs has shape (1, grid_size, grid_size).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            # Seed the disturbance generator so reset(seed=x) is reproducible
            self._disturbance.rng = np.random.default_rng(seed)

        self._disturbance.randomize()
        self._current_field = self._disturbance.generate(t=0.0)
        # Target is the undistorted base field (ideal B₀)
        self._target_field = self._env.base_field
        self._step_count = 0

        obs = self._current_field[np.newaxis, :, :].astype(np.float32)
        return obs, {}

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        """
        Apply coil currents and advance by one step.

        Args:
            action: (num_coils,) coil current array in Amps.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        assert self._current_field is not None, "Call reset() before step()"

        action = np.clip(
            np.asarray(action, dtype=np.float32),
            -self.max_current,
            self.max_current,
        )

        correction = self._env.coils.compute_field(action)
        corrected = self._current_field + correction

        mask = self._env.grid.active_zone_mask
        distorted_var = float(np.var(self._current_field[mask]))
        corrected_var = float(np.var(corrected[mask]))
        # Positive reward when variance decreases; negative when it worsens.
        reward = distorted_var - corrected_var

        # Optional physics reward shaping: add gain_budget improvement
        info: dict[str, float] = {"variance": corrected_var}
        if self.config.training.reward_shaping:
            phys = self._env.compute_uniformity_metric(corrected)
            gain_budget = phys.get("gain_budget", 0.0)
            info["gain_budget"] = gain_budget
            info["cooperativity"] = phys.get("cooperativity", 0.0)
            info["snr_db"] = phys.get("snr_db", 0.0)
            # Reward shaping: bonus for improving gain budget
            reward += self.config.training.reward_shaping_weight * gain_budget

        self._step_count += 1
        terminated = self._step_count >= self.max_steps
        truncated = False

        self._current_field = corrected
        obs = corrected[np.newaxis, :, :].astype(np.float32)

        return obs, reward, terminated, truncated, info

    def render(self) -> None:  # noqa: D102
        pass

    def close(self) -> None:  # noqa: D102
        pass
