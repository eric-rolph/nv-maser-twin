"""
Generates realistic magnetic field disturbances.

Models environmental interference as a superposition of 2D spatial harmonics
with random amplitudes, frequencies, and phases. Supports temporal evolution.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import DisturbanceConfig
from .grid import SpatialGrid


class DisturbanceGenerator:
    """
    Generates spatially smooth, physically plausible field disturbances.

    Each disturbance is a superposition of 2D sinusoidal modes::

        δB(x,y) = Σᵢ Aᵢ · sin(kx_i·x + ky_i·y + φᵢ)

    This models the slowly-varying gradients from nearby ferromagnetic objects,
    Earth's field components, and thermal drift.
    """

    def __init__(self, grid: SpatialGrid, config: DisturbanceConfig) -> None:
        self.grid = grid
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Transient state: list of (onset_time, amplitude, spatial_pattern)
        self._active_transients: list[
            tuple[float, float, NDArray[np.float32]]
        ] = []

        # Precompute mode parameters for temporal evolution
        self._init_modes()

    def _init_modes(self) -> None:
        """Initialize random spatial frequency modes."""
        n = self.config.num_modes
        freq_range = (self.config.min_spatial_freq, self.config.max_spatial_freq)

        self.amplitudes = self.rng.uniform(
            0, self.config.max_amplitude_tesla, size=n
        ).astype(np.float32)
        self.kx = self.rng.uniform(*freq_range, size=n).astype(np.float32)
        self.ky = self.rng.uniform(*freq_range, size=n).astype(np.float32)
        self.phases = self.rng.uniform(0, 2 * np.pi, size=n).astype(np.float32)
        # Temporal drift frequencies
        self.drift_freqs = self.rng.uniform(
            0, self.config.temporal_drift_rate, size=n
        ).astype(np.float32)

    def generate(self, t: float = 0.0) -> NDArray[np.float32]:
        """
        Generate a disturbance field at time t.

        Includes spatial harmonics, mains hum, transient spikes, and DC drift
        when enabled in config.

        Args:
            t: Time in seconds. At t=0, phases are as initialized.

        Returns:
            (size, size) disturbance field in Tesla.
        """
        # Vectorized over all modes simultaneously
        # Expand grid coords: (size, size) → (1, size, size)
        x = self.grid.x[np.newaxis, :, :]  # (1, N, N)
        y = self.grid.y[np.newaxis, :, :]  # (1, N, N)

        # Mode params: (num_modes,) → (num_modes, 1, 1)
        A = self.amplitudes[:, np.newaxis, np.newaxis]
        kx = self.kx[:, np.newaxis, np.newaxis]
        ky = self.ky[:, np.newaxis, np.newaxis]
        phi = self.phases[:, np.newaxis, np.newaxis]
        drift = self.drift_freqs[:, np.newaxis, np.newaxis]

        # Superposition of all modes: (num_modes, N, N) → sum → (N, N)
        spatial = kx * x + ky * y + phi + 2.0 * np.pi * drift * t
        disturbance = np.sum(A * np.sin(spatial), axis=0)

        # ── Mains hum (50/60 Hz sinusoidal interference) ─────────
        cfg = self.config
        if cfg.mains_hum_enabled and cfg.mains_hum_amplitude_tesla > 0:
            # Spatially uniform oscillation at mains frequency
            hum = cfg.mains_hum_amplitude_tesla * np.sin(
                2.0 * np.pi * cfg.mains_hum_frequency_hz * t
            )
            disturbance = disturbance + hum

        # ── Transient spikes (Poisson process, exponential decay) ─
        if cfg.transient_enabled and cfg.transient_rate_hz > 0 and t > 0:
            self._maybe_spawn_transient(t)
            for onset, amp, pattern in self._active_transients:
                dt = t - onset
                if dt >= 0:
                    envelope = amp * np.exp(-dt / cfg.transient_decay_time_s)
                    disturbance = disturbance + envelope * pattern
            # Prune decayed transients (< 1% of peak)
            self._active_transients = [
                (onset, amp, pat)
                for onset, amp, pat in self._active_transients
                if amp * np.exp(-(t - onset) / cfg.transient_decay_time_s) > 0.01 * amp
            ]

        # ── DC drift (slow linear ramp) ──────────────────────────
        if cfg.dc_drift_enabled and cfg.dc_drift_rate_tesla_per_s > 0:
            disturbance = disturbance + cfg.dc_drift_rate_tesla_per_s * t

        return disturbance.astype(np.float32)

    def _maybe_spawn_transient(self, t: float) -> None:
        """Probabilistically spawn a new transient event."""
        cfg = self.config
        # Expected number of events per generate() call interval
        # Use a small dt approximation (assume generate called ~every ms)
        dt_approx = 1e-3
        p = cfg.transient_rate_hz * dt_approx
        if self.rng.random() < p:
            pattern = self.rng.standard_normal(
                (self.grid.size, self.grid.size)
            ).astype(np.float32)
            # Smooth the pattern spatially to make it physically plausible
            pattern = pattern / (np.abs(pattern).max() + 1e-20)
            self._active_transients.append(
                (t, cfg.transient_amplitude_tesla, pattern)
            )

    def generate_batch(self, batch_size: int) -> NDArray[np.float32]:
        """
        Generate a batch of independent random disturbances for training.

        Each sample gets freshly randomized mode parameters.

        Returns:
            (batch_size, size, size) array of disturbance fields.
        """
        N = self.grid.size
        batch = np.empty((batch_size, N, N), dtype=np.float32)

        for i in range(batch_size):
            self._init_modes()
            batch[i] = self.generate(t=0.0)

        return batch

    def randomize(self) -> None:
        """Re-randomize mode parameters (public interface for RL env reset)."""
        self._init_modes()
        self._active_transients.clear()
