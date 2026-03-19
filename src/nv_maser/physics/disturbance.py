"""
Generates realistic magnetic field disturbances.

Models environmental interference as a superposition of 2D spatial harmonics
with random amplitudes, frequencies, and phases. Supports temporal evolution.

Imaging-magnet stray field
──────────────────────────
The handheld probe architecture places a single-sided imaging magnet adjacent
to the maser module.  Its fringe field is a quasi-static spatially varying
disturbance.  ``ImagingMagnetDisturbanceConfig`` + ``compute_imaging_magnet_stray_field()``
model this via a magnetic dipole approximation; ``DisturbanceGenerator.add_imaging_magnet()``
superimposes it onto the dynamic disturbance so that the shimming controller
can quantify the required compensation headroom.

Reference: Handheld probe architecture doc §11.1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import DisturbanceConfig
from .grid import SpatialGrid

# ── Physical constants ────────────────────────────────────────────
_MU0 = 4.0 * math.pi * 1e-7  # T·m/A


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Imaging-magnet stray field model                                ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class ImagingMagnetDisturbanceConfig:
    """Configuration for the stray field from the single-sided imaging magnet.

    The imaging magnet is positioned at (offset_x_mm, offset_y_mm) in the
    plane of the maser diamond slab (the xy-plane), at height offset_z_mm
    above or below the slab.  A positive x-offset places the magnet to the
    right of the maser module — the typical handheld probe geometry where the
    two modules sit side-by-side.

    The magnet is modelled as a point magnetic dipole with moment:

        m = Br × V / µ₀   (A·m²)

    oriented along the z-axis (normal to the maser diamond slab).

    Args:
        offset_x_mm:
            Lateral separation of the imaging magnet from the maser centre (mm).
            Typical value: 40–60 mm (probe width).
        offset_y_mm:
            Forward offset along the probe face (mm). 0 = coaxial.
        offset_z_mm:
            Height offset; 0 = magnet in the same plane as the diamond slab.
        magnet_volume_m3:
            Effective magnet volume (m³). Default ≈ 150 g NdFeB at 7.5 g/cm³.
        remanence_tesla:
            NdFeB remanence Br used to compute dipole moment.
            1.3 T is conservative for N48 grade.
        effective_dipole_moment_am2:
            Override the computed dipole moment (A·m²). None = use
            ``remanence_tesla × magnet_volume_m3 / µ₀``.
        shield_attenuation_db:
            Mu-metal shielding attenuation factor (dB).  The architecture doc
            targets 50 dB (factor ≈ 316×) to reduce ~30 mT to < 100 µT at
            40 mm separation.  0 = no shielding.
    """

    offset_x_mm: float = 40.0
    offset_y_mm: float = 0.0
    offset_z_mm: float = 0.0
    magnet_volume_m3: float = 2.0e-5   # ≈ 150 g × (1/7500 m³/kg)
    remanence_tesla: float = 1.3
    effective_dipole_moment_am2: float | None = None
    shield_attenuation_db: float = 0.0


def compute_imaging_magnet_stray_field(
    grid: SpatialGrid,
    config: ImagingMagnetDisturbanceConfig,
) -> NDArray[np.float32]:
    """Compute the z-component of the stray field from the imaging magnet.

    Models the imaging magnet as a magnetic dipole with moment
    ``m = Br × V / µ₀`` oriented along the z-axis (perpendicular to the
    maser diamond slab).  The total field magnitude at positions in the slab
    plane is used because the slab NV centres respond to the total projected
    field component along the NV axis.

    For a z-oriented magnetic dipole at position (x₀, y₀, z₀), the
    Biot–Savart dipole field at grid point (x, y, 0) is:

        Bz = (µ₀/4π) m (3dz² − r²) / r⁵

    where dz = −z₀ (grid at z = 0), r² = (x−x₀)² + (y−y₀)² + z₀².

    Args:
        grid: SpatialGrid representing the NV diamond slab.
        config: ImagingMagnetDisturbanceConfig.

    Returns:
        Float32 array of shape (size, size) in Tesla.
    """
    if config.effective_dipole_moment_am2 is not None:
        m = config.effective_dipole_moment_am2
    else:
        m = config.remanence_tesla * config.magnet_volume_m3 / _MU0

    # Grid coordinates in metres (float32 → float64 for precision)
    gx = grid.x.astype(np.float64) * 1e-3
    gy = grid.y.astype(np.float64) * 1e-3

    x0 = config.offset_x_mm * 1e-3
    y0 = config.offset_y_mm * 1e-3
    z0 = config.offset_z_mm * 1e-3

    dx = gx - x0   # (ny, nx) metres
    dy = gy - y0   # (ny, nx) metres
    dz = -z0       # scalar: grid is at z=0, dipole at z=z0

    r2 = dx**2 + dy**2 + dz**2
    r = np.sqrt(r2)

    # Guard against zero distance (should never occur for a properly offset magnet)
    safe = r > 1e-9

    # B_z = (µ₀/4π) m (3dz² − r²) / r⁵
    r5 = np.where(safe, r**5, 1.0)
    bz = np.where(
        safe,
        (_MU0 / (4.0 * math.pi)) * m * (3.0 * dz**2 - r2) / r5,
        0.0,
    )

    if config.shield_attenuation_db > 0.0:
        attenuation = 10.0 ** (-config.shield_attenuation_db / 20.0)
        bz = bz * attenuation

    return bz.astype(np.float32)


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

        # Imaging-magnet stray field (static, set via add_imaging_magnet())
        self._imaging_magnet_field: NDArray[np.float32] | None = None

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

        # ── Imaging-magnet stray field (static) ───────────────────
        if self._imaging_magnet_field is not None:
            disturbance = disturbance + self._imaging_magnet_field

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
        # Note: imaging-magnet stray field is NOT cleared on randomize —
        # it is a fixed physical disturbance that persists across episodes.
        # Use clear_imaging_magnets() to remove it explicitly.

    def add_imaging_magnet(
        self,
        config: ImagingMagnetDisturbanceConfig,
    ) -> None:
        """Register an imaging magnet as a static stray-field disturbance source.

        The stray field is computed once from the dipole model and added to
        every subsequent ``generate()`` call.  Multiple calls accumulate
        (e.g. two adjacent magnets).

        This implements the §11.1 architecture doc requirement:
        "Add stray field from imaging magnet as disturbance source."

        Args:
            config: ImagingMagnetDisturbanceConfig describing the adjacent
                    imaging magnet's position, size, and shielding.
        """
        field = compute_imaging_magnet_stray_field(self.grid, config)
        if self._imaging_magnet_field is None:
            self._imaging_magnet_field = field
        else:
            self._imaging_magnet_field = self._imaging_magnet_field + field

    def clear_imaging_magnets(self) -> None:
        """Remove all registered imaging-magnet stray fields.

        After calling this, ``generate()`` no longer adds any static
        imaging-magnet contribution.
        """
        self._imaging_magnet_field = None

    @property
    def imaging_magnet_field(self) -> NDArray[np.float32] | None:
        """Read-only access to the accumulated imaging-magnet stray field (T).

        Returns ``None`` if no imaging magnet has been registered via
        ``add_imaging_magnet()``.  The array shape is ``(size, size)``.
        """
        return self._imaging_magnet_field
