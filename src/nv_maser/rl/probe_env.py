"""
Probe-aware shimming environment.

Extends :class:`ShimmingEnv` with an imaging magnet as a persistent
stray-field disturbance source and adds probe imaging metrics (maser noise
temperature, chain SNR) to the per-step reward / info dict.

Physical scenario
─────────────────
A handheld NMR/MRI probe is positioned next to a patient.  The imaging
magnet (single-sided permanent magnet, ~0.5–1 kg NdFeB) produces stray
fringe fields at the maser module location.  The RL shim controller must
cancel both the usual background disturbance *and* the quasi-static imaging
magnet stray field to keep the maser within its gain bandwidth.

The coupling between shimming quality and imaging performance is:

    field uniformity → T₂* → Q_m → maser noise temperature (T_a)  →  probe SNR

Better shimming → longer T₂* → lower T_a → better probe SNR.

Reward shaping
──────────────
When ``ProbeShimmingConfig.use_probe_reward`` is True the per-step reward
includes a term proportional to the maser chain SNR in dB (from
``FieldEnvironment.compute_uniformity_metric``).  This steers the agent
toward shimmed states that simultaneously maximise imaging SNR — not just
field-variance minimisation.

Usage
─────
::

    from nv_maser.rl.probe_env import ProbeShimmingConfig, ProbeShimmingEnv
    from nv_maser.physics.disturbance import ImagingMagnetDisturbanceConfig

    pcfg = ProbeShimmingConfig(
        imaging_magnet=ImagingMagnetDisturbanceConfig(
            offset_x_mm=50.0,
            shield_attenuation_db=50.0,
        ),
        probe_snr_weight=0.1,
        use_probe_reward=True,
    )
    env = ProbeShimmingEnv(probe_config=pcfg)
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(action)
    print(info["probe_snr_db"], info["maser_noise_temp_k"])
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..config import SimConfig
from ..physics.disturbance import ImagingMagnetDisturbanceConfig
from .env import ShimmingEnv


@dataclass
class ProbeShimmingConfig:
    """Configuration for the probe-aware shimming environment.

    Attributes
    ----------
    imaging_magnet:
        Description of the imaging magnet that produces stray fields at the
        maser location.  Defaults to a ``ImagingMagnetDisturbanceConfig()``
        instance (2×10⁻⁵ m³ NdFeB magnet at 40 mm lateral offset, no
        shielding).
    probe_snr_weight:
        Reward multiplier applied to the maser chain SNR (dB) term when
        ``use_probe_reward`` is True.  Keep small relative to the variance
        reduction term (default 0.1).
    use_probe_reward:
        If True, add ``probe_snr_weight × snr_db`` to the per-step reward.
        Defaults to False to preserve the reward scaling of the base
        :class:`ShimmingEnv`.
    """

    imaging_magnet: ImagingMagnetDisturbanceConfig = field(
        default_factory=ImagingMagnetDisturbanceConfig
    )
    probe_snr_weight: float = 0.1
    use_probe_reward: bool = False


class ProbeShimmingEnv(ShimmingEnv):
    """RL shimming environment with imaging magnet disturbance and probe SNR reward.

    The imaging magnet stray field is registered once at construction time
    via :py:meth:`DisturbanceGenerator.add_imaging_magnet` and persists
    across all episodes.  ``DisturbanceGenerator.randomize()`` deliberately
    does *not* clear it (see disturbance module docs), so RL episodes always
    include the static imaging magnet contribution.

    Each ``step()`` augments the ``info`` dict with three additional keys:

    ``maser_noise_temp_k``
        Maser amplifier noise temperature on the corrected field (K).
        Lower is better; computed from Wang (2024) Eq. 4 via
        :py:func:`~nv_maser.physics.signal_chain.compute_maser_noise_temperature`.

    ``probe_snr_db``
        Full maser chain signal-to-noise ratio (dB) from the corrected
        field, forwarded from
        :py:meth:`~nv_maser.physics.environment.FieldEnvironment.compute_uniformity_metric`.

    ``stray_field_rms_mt``
        RMS of the imaging-magnet stray field over the maser grid (mT).
        Constant across steps; cached at construction time.

    Args:
        config:       :class:`~nv_maser.config.SimConfig` for the base
                      shimming environment.  ``None`` → default config.
        probe_config: :class:`ProbeShimmingConfig` controlling the imaging
                      magnet and probe reward shaping.  ``None`` →
                      ``ProbeShimmingConfig()`` (40 mm magnet, no probe
                      reward shaping).
    """

    def __init__(
        self,
        config: SimConfig | None = None,
        probe_config: ProbeShimmingConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._probe_config: ProbeShimmingConfig = (
            probe_config if probe_config is not None else ProbeShimmingConfig()
        )

        # Register imaging magnet as a static stray-field disturbance source.
        # It persists across all episodes — randomize() does NOT clear it.
        self._disturbance.add_imaging_magnet(self._probe_config.imaging_magnet)

        # Cache the RMS stray field (mT) on the maser grid.  Constant until
        # clear_imaging_magnets() is called on the underlying generator.
        imf = self._disturbance.imaging_magnet_field
        if imf is not None and imf.size > 0:
            self._stray_field_rms_mt: float = float(
                np.sqrt(np.mean(imf**2))
            ) * 1e3  # T → mT
        else:
            self._stray_field_rms_mt = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def probe_config(self) -> ProbeShimmingConfig:
        """The probe shimming configuration."""
        return self._probe_config

    @property
    def stray_field_rms_mt(self) -> float:
        """RMS imaging-magnet stray field on the maser grid (mT)."""
        return self._stray_field_rms_mt

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        """Apply coil currents, step the environment, and return probe metrics.

        Calls the parent :py:meth:`ShimmingEnv.step` first, then evaluates
        maser physics on the corrected field to populate probe metric keys.

        Args:
            action: ``(num_coils,)`` coil current array in Amps.

        Returns:
            ``(obs, reward, terminated, truncated, info)`` where ``info``
            contains the standard parent keys plus:
            ``maser_noise_temp_k``, ``probe_snr_db``, ``stray_field_rms_mt``.
        """
        obs, reward, terminated, truncated, info = super().step(action)

        # Evaluate maser + probe metrics on the corrected field.
        # self._current_field has already been updated by the parent step().
        phys = self._env.compute_uniformity_metric(self._current_field)

        noise_temp_k = phys.get("maser_noise_temperature_k", math.nan)
        snr_db = phys.get("snr_db", math.nan)

        info["maser_noise_temp_k"] = float(noise_temp_k)
        info["probe_snr_db"] = float(snr_db)
        info["stray_field_rms_mt"] = self._stray_field_rms_mt

        if self._probe_config.use_probe_reward:
            reward += self._probe_config.probe_snr_weight * float(snr_db)

        return obs, reward, terminated, truncated, info
