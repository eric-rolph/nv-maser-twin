"""physics/rf_rejection.py — RF interference rejection model (R6).

Background
----------
The NV-maser handheld probe is intended for use in *unshielded*
environments — emergency wards, field hospitals, ambulances — where
broadband RF interference from Wi-Fi (2.4/5 GHz), Bluetooth (2.4 GHz),
cellular (700 MHz – 3.5 GHz), and broadcast (FM/DAB) sources is present.
Conventional low-field MRI scanners require Faraday-cage rooms to exclude
this interference.  The maser's intrinsic narrowband gain spectrum acts as
a highly selective bandpass filter that provides the same rejection without
any enclosure.

Physical mechanism
------------------
The maser cavity supports stimulated emission only within its Lorentzian
gain envelope.  A signal at frequency *f* experiences gain:

    G(f) ∝  1 / [ 1 + (2(f − f0) / BW)² ]           (Lorentzian)

where *f0* is the maser centre frequency (≈ 1.4699 GHz, set by the NV
zero-field splitting minus the electron-Zeeman shift at 50 mT) and *BW*
is the gain FWHM (≈ 49 kHz, set by the loaded cavity Q ≈ 30 000).

An interferer at frequency *f_int* is attenuated relative to the on-
resonance gain by the factor:

    A(f_int) = G(f_int) / G(f0) = 1 / [ 1 + (2 Δf / BW)² ]

where Δf = |f_int − f0|.  In decibels the out-of-band rejection is:

    OOB_dB(f_int) = −10 log₁₀ A(f_int)
                  = 10 log₁₀[ 1 + (2 Δf / BW)² ]

For Δf ≫ BW/2 this simplifies to ≈ 20 log₁₀(2 Δf / BW).

Down-conversion and image-band analysis
----------------------------------------
After the maser the signal is down-converted to baseband by mixing with
the same LO used for up-conversion:

    f_bb = |f_maser_out − f_LO| ≈ f_NMR ≈ 2.13 MHz

An interferer that somehow leaks through the maser from centre offset Δf
maps to a baseband frequency:

    f_bb_int = |f_int − f_LO|

which, after a final low-pass filter of width *BW_readout* (≈ 20 kHz),
survives only if f_bb_int < BW_readout / 2.  The boolean flag
``in_readout_band`` captures this condition.

The residual interference power at the maser *output* (after attenuation)
is:

    P_residual_dBm = P_int_dBm − OOB_dB

Risk register R6 context
-------------------------
Architecture doc §13 risk register:
    "RF interference in unshielded environment → Image artifacts —
     Medium impact — Maser's natural bandpass helps; apply real-time
     interference cancellation."

This module quantifies the "natural bandpass" contribution so that the
team can judge whether residual interference warrants additional active
cancellation hardware.

References
----------
* Architecture doc §2 (table, Narrow instantaneous bandwidth)
* Architecture doc §3.5 (gap table, RF interference row)
* Architecture doc §5.3 (Frequency Plan, Maser gain bandwidth = 50 kHz)
* Architecture doc §13 (Risk register R6)
* Pozar — "Microwave Engineering", 4th ed. (2012), §10.4–10.5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ── Default interference sources modelled ────────────────────────────────────
# These represent a realistic hospital / urban unshielded environment.
# Power levels are conservative (pessimistic) figures in dBm received
# at the probe with a 1 m separation from the interfering device.

_DEFAULT_INTERFERERS: tuple[tuple, ...] = (
    # (name,               center_hz,  bandwidth_hz, power_dbm)
    ("WiFi 2.4 GHz",      2_412e6,     40e6,         -30.0),
    ("WiFi 5 GHz",        5_180e6,     80e6,         -30.0),
    ("Bluetooth 2.4 GHz", 2_441e6,     80e6,         -50.0),  # FHSS avg
    ("LTE 700 MHz",        746e6,      10e6,         -60.0),
    ("LTE 1800 MHz",      1_800e6,     20e6,         -60.0),
    ("LTE 2600 MHz",      2_600e6,     20e6,         -60.0),
    ("Hospital WiFi 2.4", 2_437e6,     40e6,         -40.0),
    ("Broadcast FM",       98e6,       200e3,        -20.0),
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Data classes                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class InterfererSpec:
    """Specification of a single RF interference source.

    Attributes
    ----------
    name:
        Human-readable label (e.g. ``"WiFi 2.4 GHz"``).
    center_freq_hz:
        Centre frequency of the interferer (Hz).
    bandwidth_hz:
        Emission bandwidth of the interferer (Hz, informational only).
        The rejection calculation uses *center_freq_hz*.
    power_dbm:
        Received power at the probe input (dBm).  Use a conservative
        (high) estimate for worst-case analysis.
    """

    name: str
    center_freq_hz: float
    bandwidth_hz: float
    power_dbm: float

    def __post_init__(self) -> None:
        if self.center_freq_hz <= 0:
            raise ValueError(
                f"center_freq_hz must be positive, got {self.center_freq_hz}"
            )
        if self.bandwidth_hz <= 0:
            raise ValueError(
                f"bandwidth_hz must be positive, got {self.bandwidth_hz}"
            )


@dataclass(frozen=True)
class RFRejectionConfig:
    """Configuration for the RF rejection model.

    Attributes
    ----------
    maser_center_hz:
        Maser gain-profile centre frequency (Hz).  Default matches the NV
        ν₋ transition at 50 mT sweet-spot field: 1.4699 GHz.
    maser_gain_bw_hz:
        Maser gain FWHM (Hz).  Default 49 000 Hz (loaded Q ≈ 30 000 at
        1.4699 GHz), consistent with ``GainBandwidthConfig`` defaults.
    readout_bw_hz:
        NMR readout bandwidth after final down-conversion and low-pass
        filter (Hz, full width).  Default 20 000 Hz.
    lo_freq_hz:
        Local-oscillator frequency used for down-conversion (Hz).
        Default = maser_center_hz − 2.129 MHz (standard up-conversion
        plan: NMR at 2.129 MHz → USB at 1.4699 GHz).
    interferers:
        Tuple of :class:`InterfererSpec` objects to evaluate.  Defaults
        to a built-in set representing a realistic hospital environment.
    """

    maser_center_hz: float = 1.4699e9
    maser_gain_bw_hz: float = 49_000.0
    readout_bw_hz: float = 20_000.0
    lo_freq_hz: float | None = None  # resolved in __post_init__
    interferers: tuple[InterfererSpec, ...] = field(
        default_factory=lambda: tuple(
            InterfererSpec(name=n, center_freq_hz=f, bandwidth_hz=bw, power_dbm=p)
            for n, f, bw, p in _DEFAULT_INTERFERERS
        )
    )

    def __post_init__(self) -> None:
        if self.maser_center_hz <= 0:
            raise ValueError(
                f"maser_center_hz must be positive, got {self.maser_center_hz}"
            )
        if self.maser_gain_bw_hz <= 0:
            raise ValueError(
                f"maser_gain_bw_hz must be positive, got {self.maser_gain_bw_hz}"
            )
        if self.readout_bw_hz <= 0:
            raise ValueError(
                f"readout_bw_hz must be positive, got {self.readout_bw_hz}"
            )
        # Resolve the LO frequency (bypass frozen by using object.__setattr__)
        if self.lo_freq_hz is None:
            object.__setattr__(
                self,
                "lo_freq_hz",
                self.maser_center_hz - 2.129e6,
            )
        if self.lo_freq_hz <= 0:
            raise ValueError(
                f"lo_freq_hz must be positive, got {self.lo_freq_hz}"
            )


@dataclass(frozen=True)
class InterfererResult:
    """Rejection analysis result for a single interferer.

    Attributes
    ----------
    interferer:
        Reference to the input :class:`InterfererSpec`.
    freq_offset_hz:
        Absolute frequency offset from the maser centre: |f_int − f0| (Hz).
    attenuation_db:
        Out-of-band attenuation provided by the maser's Lorentzian bandpass
        (dB, positive = suppression).
    residual_power_dbm:
        Interference power after attenuation, at the maser output (dBm).
    baseband_freq_hz:
        Absolute baseband frequency after down-conversion: |f_int − f_LO| (Hz).
    in_readout_band:
        ``True`` if the down-converted residual falls within the NMR readout
        bandwidth (i.e. it could overlay on the NMR signal and cause artefacts).
    """

    interferer: InterfererSpec
    freq_offset_hz: float
    attenuation_db: float
    residual_power_dbm: float
    baseband_freq_hz: float
    in_readout_band: bool


@dataclass(frozen=True)
class RFRejectionResult:
    """Aggregate RF rejection analysis.

    Attributes
    ----------
    interferer_results:
        Per-interferer analysis, one entry per input :class:`InterfererSpec`.
    worst_case_residual_dbm:
        Highest residual power (dBm) across all interferers — the
        worst-case threat to the receiving chain.
    worst_case_name:
        Name of the interferer with the highest residual power.
    any_in_readout_band:
        ``True`` if any attenuated interferer lands inside the NMR
        readout bandwidth (potential image artefact source).
    maser_fractional_bw:
        Dimensionless: maser_gain_bw_hz / maser_center_hz.  The key
        figure-of-merit for intrinsic rejection capability.
    min_attenuation_db:
        Smallest per-interferer attenuation across the set (worst case).
    max_attenuation_db:
        Largest per-interferer attenuation (best case).
    """

    interferer_results: tuple[InterfererResult, ...]
    worst_case_residual_dbm: float
    worst_case_name: str
    any_in_readout_band: bool
    maser_fractional_bw: float
    min_attenuation_db: float
    max_attenuation_db: float


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Core computation functions                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def compute_lorentzian_attenuation(
    freq_hz: float,
    center_hz: float,
    bw_hz: float,
) -> float:
    """Maser Lorentzian bandpass attenuation at *freq_hz* (dB).

    Returns the out-of-band rejection provided by the maser's Lorentzian
    gain profile, i.e. how many dB below the on-resonance gain the signal
    at *freq_hz* experiences:

        OOB_dB = 10 log₁₀[ 1 + (2 Δf / BW)² ]

    where Δf = |freq_hz − center_hz| and BW = bw_hz (FWHM).

    Parameters
    ----------
    freq_hz:
        Frequency of the signal to evaluate (Hz).
    center_hz:
        Maser gain-profile centre frequency (Hz).
    bw_hz:
        Maser gain FWHM (Hz).

    Returns
    -------
    float
        Attenuation in dB (positive = suppression, 0 = on-resonance).
    """
    delta_f = abs(freq_hz - center_hz)
    lorentzian = 1.0 / (1.0 + (2.0 * delta_f / bw_hz) ** 2)
    return -10.0 * math.log10(lorentzian)


def compute_fractional_bandwidth(config: RFRejectionConfig) -> float:
    """Maser fractional bandwidth: BW / f₀  (dimensionless).

    A smaller value means a more selective bandpass, i.e. greater intrinsic
    out-of-band rejection.  The NV maser at 1.47 GHz with 49 kHz FWHM
    achieves ≈ 3.33 × 10⁻⁵ — far narrower than any room-temperature LNA.

    Parameters
    ----------
    config:
        :class:`RFRejectionConfig` instance.

    Returns
    -------
    float
        Dimensionless fractional bandwidth.
    """
    return config.maser_gain_bw_hz / config.maser_center_hz


def compute_interferer_rejection(
    spec: InterfererSpec,
    config: RFRejectionConfig,
) -> InterfererResult:
    """Compute the rejection of a single interferer.

    Parameters
    ----------
    spec:
        Interferer specification.
    config:
        RF rejection configuration (maser and readout parameters).

    Returns
    -------
    InterfererResult
        Full per-interferer analysis.
    """
    freq_offset_hz = abs(spec.center_freq_hz - config.maser_center_hz)
    attenuation_db = compute_lorentzian_attenuation(
        freq_hz=spec.center_freq_hz,
        center_hz=config.maser_center_hz,
        bw_hz=config.maser_gain_bw_hz,
    )
    residual_power_dbm = spec.power_dbm - attenuation_db
    baseband_freq_hz = abs(spec.center_freq_hz - config.lo_freq_hz)  # type: ignore[operator]
    in_readout_band = baseband_freq_hz < config.readout_bw_hz / 2.0

    return InterfererResult(
        interferer=spec,
        freq_offset_hz=freq_offset_hz,
        attenuation_db=attenuation_db,
        residual_power_dbm=residual_power_dbm,
        baseband_freq_hz=baseband_freq_hz,
        in_readout_band=in_readout_band,
    )


def compute_rf_rejection(
    config: RFRejectionConfig | None = None,
) -> RFRejectionResult:
    """Compute the maser's RF interference rejection for all configured sources.

    This is the primary entry point.  It evaluates every interferer in
    *config.interferers*, then aggregates the results into an
    :class:`RFRejectionResult` summary.

    Parameters
    ----------
    config:
        RF rejection configuration.  If ``None``, the default
        :class:`RFRejectionConfig` (8 standard hospital-environment
        interferers) is used.

    Returns
    -------
    RFRejectionResult
        Aggregate rejection analysis.
    """
    if config is None:
        config = RFRejectionConfig()

    results: list[InterfererResult] = [
        compute_interferer_rejection(spec, config) for spec in config.interferers
    ]

    if not results:
        # Edge case: empty interferer list
        return RFRejectionResult(
            interferer_results=(),
            worst_case_residual_dbm=float("-inf"),
            worst_case_name="(none)",
            any_in_readout_band=False,
            maser_fractional_bw=compute_fractional_bandwidth(config),
            min_attenuation_db=0.0,
            max_attenuation_db=0.0,
        )

    worst = max(results, key=lambda r: r.residual_power_dbm)
    best_att = max(results, key=lambda r: r.attenuation_db)
    worst_att = min(results, key=lambda r: r.attenuation_db)

    return RFRejectionResult(
        interferer_results=tuple(results),
        worst_case_residual_dbm=worst.residual_power_dbm,
        worst_case_name=worst.interferer.name,
        any_in_readout_band=any(r.in_readout_band for r in results),
        maser_fractional_bw=compute_fractional_bandwidth(config),
        min_attenuation_db=worst_att.attenuation_db,
        max_attenuation_db=best_att.attenuation_db,
    )
