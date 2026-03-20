"""physics/gain_bandwidth_match.py — Maser gain-bandwidth vs NMR readout matching (R2).

Background
----------
The NV-maser cavity has a Lorentzian gain profile centred at *f0* with
full-width half-maximum (FWHM):

    BW_maser = f0 / Q_loaded                                     [Hz]

The NMR readout bandwidth (gradient-encoded spatial information) spans a
window of width *BW_readout* around the Larmor frequency.  After up-
conversion the entire NMR readout band must sit inside the maser gain
envelope for distortion-free amplification.

Overlap condition (necessary for distortion-free gain)
------------------------------------------------------
    BW_readout + 2 |Δf_centre| ≤ BW_maser

where *Δf_centre* is the displacement of the NMR-band centre from the
maser gain centre (e.g. caused by a static B₀ drift).

A static B₀ drift *ΔB* shifts the proton Larmor frequency by

    Δf_Larmor = γ_p · ΔB

which is the dominant source of centre misalignment when the LO is fixed.

Maximum tolerable one-sided drift
----------------------------------
    ΔB_max = (BW_maser − BW_readout) / 2 / γ_p                  [T]
    ΔB_ppm = ΔB_max / B₀ × 10⁶                                  [ppm]

Architecture context
--------------------
From ``docs/research/handheld-maser-probe-architecture.md`` §7, §12, §13-R2:

* Maser centre: f0 = 1.4699 GHz
* Stated gain BW ≈ 50 kHz  →  loaded Q ≈ f0 / BW ≈ 30 000
  (The architecture doc quotes "Q = 10 000" as a representative *unloaded*
  Q value; the *loaded* Q that produces the stated 50 kHz gain BW at
  f0 = 1.4699 GHz is ≈ 29 400, rounded to 30 000 here.)
* NMR readout BW: ±10–25 kHz full-width; nominal 20 kHz used here
* Sweet-spot B₀ = 50 mT

Risk register R2 (Medium): "Maser gain bandwidth too narrow for readout →
truncated signal → artefacts.  Mitigation: match maser Q to readout BW;
use lower Q cavity or frequency-tracking."

References
----------
* Architecture doc §7 (The Critical Bandwidth Problem)
* Architecture doc §12 (Parameters table, Critical point note)
* Architecture doc §13 (Risk register, R2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# ---------------------------------------------------------------------------
# Physical constant
# ---------------------------------------------------------------------------

#: Proton gyromagnetic ratio / 2π  [Hz/T]  (CODATA 2018)
PROTON_GYRO_HZ_T: float = 42.5774e6


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GainBandwidthConfig:
    """Maser–readout bandwidth matching configuration.

    Parameters
    ----------
    cavity_q : float
        Loaded quality factor of the maser cavity (dimensionless).
        Default 30 000 yields BW_maser ≈ 49 kHz at f0 = 1.4699 GHz,
        matching the architecture-doc stated nominal of ~50 kHz.
    f0_hz : float
        Maser cavity centre frequency [Hz].  Default = 1.4699 GHz.
    readout_bw_hz : float
        NMR readout full bandwidth [Hz].  Default = 20 kHz (mid-range).
    b0_tesla : float
        Static main field at the sweet spot [T].  Default = 50 mT.
    gyro_ratio_hz_t : float
        Proton gyromagnetic ratio / 2π  [Hz/T].
    """

    cavity_q: float = 30_000.0
    f0_hz: float = 1.4699e9
    readout_bw_hz: float = 20_000.0
    b0_tesla: float = 0.050
    gyro_ratio_hz_t: float = PROTON_GYRO_HZ_T


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandwidthMatchResult:
    """Summary of the maser–readout bandwidth matching analysis.

    Attributes
    ----------
    maser_gain_bw_hz : float
        FWHM of the maser Lorentzian gain profile = f0 / Q  [Hz].
    readout_bw_hz : float
        NMR readout full bandwidth echoed from the config  [Hz].
    frequency_margin_hz : float
        One-sided slack = (BW_maser − BW_readout) / 2  [Hz].
        Negative when BW_readout > BW_maser.
    overlap_fraction : float
        1 − BW_readout / BW_maser, clamped to [0, 1].
        Equals 1 for zero-BW readout; equals 0 when readout fills gain band.
    b0_drift_tolerance_t : float
        Maximum B₀ drift before the Larmor frequency exits the gain margin [T].
        Zero when there is no positive margin.
    b0_drift_tolerance_ppm : float
        Same limit expressed as parts-per-million of B₀.
    passes_criterion : bool
        True iff BW_readout ≤ BW_maser (readout fits inside gain band).
    """

    maser_gain_bw_hz: float
    readout_bw_hz: float
    frequency_margin_hz: float
    overlap_fraction: float
    b0_drift_tolerance_t: float
    b0_drift_tolerance_ppm: float
    passes_criterion: bool


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_maser_gain_bandwidth(cavity_q: float, f0_hz: float) -> float:
    """FWHM of the maser Lorentzian gain profile.

    BW_maser = f0 / Q_loaded

    Parameters
    ----------
    cavity_q : float
        Loaded Q factor (must be > 0).
    f0_hz : float
        Maser centre frequency [Hz] (must be > 0).

    Returns
    -------
    float
        Gain bandwidth [Hz].
    """
    if cavity_q <= 0:
        raise ValueError(f"cavity_q must be positive, got {cavity_q}")
    if f0_hz <= 0:
        raise ValueError(f"f0_hz must be positive, got {f0_hz}")
    return f0_hz / cavity_q


def compute_b0_drift_tolerance(
    config: GainBandwidthConfig,
) -> tuple[float, float]:
    """Maximum B₀ drift before Larmor exits the one-sided frequency margin.

    Derivation::

        margin_hz = (BW_maser − BW_readout) / 2
        ΔB_max    = margin_hz / γ_p            [T]
        ΔB_ppm    = ΔB_max / B₀ × 10⁶         [ppm]

    Returns ``(0.0, 0.0)`` when BW_readout ≥ BW_maser (no positive margin).

    Returns
    -------
    tuple[float, float]
        ``(tolerance_tesla, tolerance_ppm)``
    """
    bw_maser = compute_maser_gain_bandwidth(config.cavity_q, config.f0_hz)
    margin_hz = (bw_maser - config.readout_bw_hz) / 2.0
    if margin_hz <= 0.0:
        return (0.0, 0.0)
    tol_t = margin_hz / config.gyro_ratio_hz_t
    tol_ppm = tol_t / config.b0_tesla * 1.0e6
    return (tol_t, tol_ppm)


def compute_bandwidth_match(config: GainBandwidthConfig) -> BandwidthMatchResult:
    """Primary entry point: full bandwidth-matching summary.

    Parameters
    ----------
    config : GainBandwidthConfig
        Maser + readout parameters.

    Returns
    -------
    BandwidthMatchResult
    """
    bw_maser = compute_maser_gain_bandwidth(config.cavity_q, config.f0_hz)
    margin_hz = (bw_maser - config.readout_bw_hz) / 2.0
    overlap = max(0.0, 1.0 - config.readout_bw_hz / bw_maser)
    tol_t, tol_ppm = compute_b0_drift_tolerance(config)
    return BandwidthMatchResult(
        maser_gain_bw_hz=bw_maser,
        readout_bw_hz=config.readout_bw_hz,
        frequency_margin_hz=margin_hz,
        overlap_fraction=overlap,
        b0_drift_tolerance_t=tol_t,
        b0_drift_tolerance_ppm=tol_ppm,
        passes_criterion=(config.readout_bw_hz <= bw_maser),
    )


# ---------------------------------------------------------------------------
# Sweep utilities
# ---------------------------------------------------------------------------

def sweep_q_vs_gain_bandwidth(
    q_values: Sequence[float],
    f0_hz: float = 1.4699e9,
) -> list[float]:
    """Return gain bandwidths [Hz] for each Q value in *q_values*.

    Parameters
    ----------
    q_values : sequence of float
        Q values to evaluate.
    f0_hz : float
        Centre frequency [Hz] (shared across all Q values).

    Returns
    -------
    list[float]
        Gain bandwidth [Hz] for each entry in *q_values*  (same order).
    """
    return [compute_maser_gain_bandwidth(q, f0_hz) for q in q_values]


def sweep_b0_drift_vs_overlap(
    drift_values_ut: Sequence[float],
    config: GainBandwidthConfig,
) -> list[float]:
    """Return overlap fraction for each B₀ drift value (µT).

    The overlap fraction after a drift *δB* [µT] is::

        remaining = BW_maser − BW_readout − 2 · |δB × γ_p × 10⁻⁶|
        overlap   = max(0, remaining / BW_maser)

    Zero means the readout band clips the maser gain edge.

    Parameters
    ----------
    drift_values_ut : sequence of float
        B₀ drift values in micro-tesla [µT].  Negative values are treated
        as equal-magnitude positive drifts (symmetric gain profile).
    config : GainBandwidthConfig

    Returns
    -------
    list[float]
        Overlap fraction in [0, 1] for each drift entry.
    """
    bw_maser = compute_maser_gain_bandwidth(config.cavity_q, config.f0_hz)
    out: list[float] = []
    for drift_ut in drift_values_ut:
        delta_f = abs(float(drift_ut) * 1.0e-6) * config.gyro_ratio_hz_t
        remaining = bw_maser - config.readout_bw_hz - 2.0 * delta_f
        out.append(max(0.0, remaining / bw_maser))
    return out
