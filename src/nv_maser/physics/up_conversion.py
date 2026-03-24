"""
RF up-conversion mixer model: NMR frequency → maser frequency.

In the handheld probe the NMR receive signal originates at the Larmor
frequency (f_NMR ≈ 2.13 MHz at 50 mT) and must be placed inside the
maser's gain band (~1.47 GHz) before amplification.  A single-sideband
(SSB) or image-reject upconverter achieves this by mixing with a local
oscillator (LO):

    f_LO ≈ f_maser − f_NMR          (USB convention)
    f_USB = f_LO + f_NMR ≈ f_maser  (upper sideband)
    f_LSB = f_LO − f_NMR            (lower sideband, rejected in SSB)

Noise model
───────────
A passive double-balanced mixer (DBM) has a conversion loss L (dB)
and adds noise proportional to that loss.  For an image-reject (SSB)
mixer, the noise figure equals the conversion loss plus ≈ 0 dB (ideal)
or a few dB of excess noise.  In the SSB convention used here:

    NF_mixer = L_dB + NF_excess_dB      (≥ L_dB ≥ 0)

The corresponding noise temperature (referred to the mixer input) is:

    T_mixer = T₀ · (10^(NF/10) − 1)    with T₀ = 290 K

For the Friis cascade (coil → mixer → maser → LNA) the system noise
temperature referred to the coil input is:

    T_sys = T_coil + T_mixer / G_coil + T_maser / (G_coil · G_mixer) + …

Since the coil is passive (G_coil = 1, T_coil = T_phys × (1 − 1) = 0
for a lossless coil) the dominant contribution is usually the coil
thermal noise plus the mixer noise.

References
──────────
Pozar — "Microwave Engineering", 4th ed. (2012), §10.4–10.5.
Friis — "Noise Figures of Radio Receivers", Proc. IRE 32, 419 (1944).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ── Physical constants ────────────────────────────────────────────
_T0 = 290.0  # IEEE reference temperature (K)
from .constants import KB as _KB


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Data classes                                                    ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class MixerSpec:
    """RF mixer specification.

    Attributes
    ----------
    conversion_loss_db:
        Double-sideband (DSB) conversion loss (dB, positive number).
        Typical passive DBM: 5–8 dB.
    noise_figure_ssb_db:
        Single-sideband (SSB) noise figure (dB).  For an ideal passive
        mixer NF_SSB = L_dB (no excess noise); practical mixers add 1–3 dB.
        Defaults to ``conversion_loss_db`` (ideal zero excess noise).
    lo_leakage_db:
        LO-to-RF port isolation (dB, negative convention, e.g. −40 dB).
        Used for spurious signal estimation; does not affect the noise model.
    ip3_dbm:
        Input-referred 3rd-order intercept point (dBm).  Used for
        linearity / dynamic-range checks.
    """

    conversion_loss_db: float = 6.5
    noise_figure_ssb_db: float | None = None
    lo_leakage_db: float = -40.0
    ip3_dbm: float = 0.0

    def __post_init__(self) -> None:
        if self.conversion_loss_db < 0:
            raise ValueError(
                f"conversion_loss_db must be ≥ 0 (a loss), got {self.conversion_loss_db}"
            )
        # If SSB NF not specified, default to conversion loss (ideal mixer)
        if self.noise_figure_ssb_db is not None and self.noise_figure_ssb_db < 0:
            raise ValueError(
                f"noise_figure_ssb_db must be ≥ 0, got {self.noise_figure_ssb_db}"
            )

    @property
    def effective_nf_db(self) -> float:
        """Effective SSB noise figure (dB)."""
        if self.noise_figure_ssb_db is not None:
            return self.noise_figure_ssb_db
        return self.conversion_loss_db  # ideal passive mixer


# Default mixer representative of a miniature image-reject MMIC mixer
DEFAULT_MIXER = MixerSpec(
    conversion_loss_db=6.5,
    noise_figure_ssb_db=8.0,
    lo_leakage_db=-40.0,
    ip3_dbm=5.0,
)


@dataclass(frozen=True)
class UpConversionResult:
    """Complete up-conversion chain result.

    Attributes
    ----------
    nmr_frequency_hz:
        Input NMR (Larmor) frequency at the mixer RF port (Hz).
    lo_frequency_ghz:
        LO frequency used: f_LO = f_maser − f_NMR (GHz).
    usb_frequency_ghz:
        Upper sideband output: f_USB = f_LO + f_NMR ≈ f_maser (GHz).
    lsb_frequency_ghz:
        Lower sideband: f_LSB = f_LO − f_NMR (GHz).
    conversion_loss_db:
        Signal power loss through the mixer (dB).
    noise_figure_db:
        Effective SSB noise figure (dB).
    noise_temperature_k:
        Equivalent input noise temperature T_mixer (K).
    bandwidth_utilization:
        Fraction of the maser 3-dB gain bandwidth occupied by the NMR
        readout bandwidth: nmr_bandwidth_hz / maser_bandwidth_hz.
        Values > 1 indicate the NMR bandwidth exceeds the maser gain band.
    maser_frequency_ghz:
        Target maser centre frequency (GHz).
    nmr_bandwidth_hz:
        NMR readout bandwidth (Hz).
    maser_bandwidth_hz:
        Maser 3-dB gain bandwidth (Hz).
    image_frequency_ghz:
        Image frequency for a single-sideband conversion (GHz);
        f_image = f_LO − f_NMR = f_maser − 2·f_NMR.
    """

    nmr_frequency_hz: float
    lo_frequency_ghz: float
    usb_frequency_ghz: float
    lsb_frequency_ghz: float
    conversion_loss_db: float
    noise_figure_db: float
    noise_temperature_k: float
    bandwidth_utilization: float
    maser_frequency_ghz: float
    nmr_bandwidth_hz: float
    maser_bandwidth_hz: float
    image_frequency_ghz: float


@dataclass(frozen=True)
class UpConversionNoiseContribution:
    """Noise powers contributed by the up-conversion stage.

    All powers in Watts, referred to the mixer *input*.

    Attributes
    ----------
    thermal_noise_w:
        Available thermal noise at the mixer input: kB · T₀ · BW (W).
    mixer_added_noise_w:
        Noise added by mixer, referred to input:
        kB · T_mixer · BW (W), where T_mixer = T₀ · (10^(NF/10) − 1).
    total_input_referred_noise_w:
        Sum of thermal + mixer noise at the mixer input.
    bandwidth_hz:
        Noise bandwidth used for the calculation.
    noise_temperature_k:
        Mixer equivalent input noise temperature (K).
    """

    thermal_noise_w: float
    mixer_added_noise_w: float
    total_input_referred_noise_w: float
    bandwidth_hz: float
    noise_temperature_k: float


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Public API                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝


def compute_up_conversion(
    nmr_frequency_hz: float,
    maser_frequency_ghz: float,
    nmr_bandwidth_hz: float,
    maser_bandwidth_hz: float,
    mixer: MixerSpec | None = None,
) -> UpConversionResult:
    """Compute the up-conversion chain from NMR to maser frequency.

    The LO is set to place the NMR signal exactly at the maser centre:

        f_LO = f_maser − f_NMR         (upper-sideband convention)
        f_USB = f_NMR + f_LO = f_maser  (in-band)

    Args:
        nmr_frequency_hz:
            NMR Larmor frequency at the mixer input (Hz).
        maser_frequency_ghz:
            Maser centre frequency — target for the upconverted signal (GHz).
        nmr_bandwidth_hz:
            NMR readout bandwidth (Hz); checked against maser gain bandwidth.
        maser_bandwidth_hz:
            Maser 3-dB gain bandwidth (Hz); used for BW utilisation calc.
        mixer:
            MixerSpec; defaults to ``DEFAULT_MIXER`` if None.

    Returns:
        UpConversionResult containing frequencies, noise, and BW utilisation.

    Raises:
        ValueError: If any frequency or bandwidth is non-positive.
    """
    if nmr_frequency_hz <= 0:
        raise ValueError(f"nmr_frequency_hz must be positive, got {nmr_frequency_hz}")
    if maser_frequency_ghz <= 0:
        raise ValueError(f"maser_frequency_ghz must be positive, got {maser_frequency_ghz}")
    if nmr_bandwidth_hz <= 0:
        raise ValueError(f"nmr_bandwidth_hz must be positive, got {nmr_bandwidth_hz}")
    if maser_bandwidth_hz <= 0:
        raise ValueError(f"maser_bandwidth_hz must be positive, got {maser_bandwidth_hz}")

    if mixer is None:
        mixer = DEFAULT_MIXER

    nmr_ghz = nmr_frequency_hz / 1e9
    f_maser = maser_frequency_ghz

    # LO frequency (upper-sideband convention)
    lo_ghz = f_maser - nmr_ghz  # f_LO = f_maser − f_NMR

    # Sideband frequencies
    usb_ghz = lo_ghz + nmr_ghz  # = f_maser (by construction)
    lsb_ghz = lo_ghz - nmr_ghz  # = f_maser − 2·f_NMR
    image_ghz = lsb_ghz        # For USB conversion, image is at LSB

    # Mixer noise temperature
    nf_db = mixer.effective_nf_db
    t_mixer_k = _T0 * (10.0 ** (nf_db / 10.0) - 1.0)

    # Bandwidth utilisation
    bw_util = nmr_bandwidth_hz / maser_bandwidth_hz

    return UpConversionResult(
        nmr_frequency_hz=nmr_frequency_hz,
        lo_frequency_ghz=lo_ghz,
        usb_frequency_ghz=usb_ghz,
        lsb_frequency_ghz=lsb_ghz,
        conversion_loss_db=mixer.conversion_loss_db,
        noise_figure_db=nf_db,
        noise_temperature_k=t_mixer_k,
        bandwidth_utilization=bw_util,
        maser_frequency_ghz=maser_frequency_ghz,
        nmr_bandwidth_hz=nmr_bandwidth_hz,
        maser_bandwidth_hz=maser_bandwidth_hz,
        image_frequency_ghz=image_ghz,
    )


def compute_mixer_noise_contribution(
    mixer: MixerSpec,
    bandwidth_hz: float,
    physical_temperature_k: float = _T0,
) -> UpConversionNoiseContribution:
    """Noise power contributed by the up-conversion mixer stage.

    The noise is referred to the mixer RF (input) port.

    Thermal noise available at input:
        P_thermal = kB · T_phys · BW

    Mixer adds equivalent noise corresponding to its noise temperature:
        P_mixer = kB · T_mixer · BW    where T_mixer = T₀·(10^(NF/10)−1)

    Args:
        mixer: MixerSpec with noise figure.
        bandwidth_hz: Noise bandwidth (Hz).
        physical_temperature_k: Physical temperature of the input (K).

    Returns:
        UpConversionNoiseContribution with all noise terms.

    Raises:
        ValueError: If bandwidth is non-positive.
    """
    if bandwidth_hz <= 0:
        raise ValueError(f"bandwidth_hz must be positive, got {bandwidth_hz}")

    nf_db = mixer.effective_nf_db
    t_mixer_k = _T0 * (10.0 ** (nf_db / 10.0) - 1.0)

    p_thermal = _KB * physical_temperature_k * bandwidth_hz
    p_mixer = _KB * t_mixer_k * bandwidth_hz
    p_total = p_thermal + p_mixer

    return UpConversionNoiseContribution(
        thermal_noise_w=p_thermal,
        mixer_added_noise_w=p_mixer,
        total_input_referred_noise_w=p_total,
        bandwidth_hz=bandwidth_hz,
        noise_temperature_k=t_mixer_k,
    )


def compute_lo_frequency_ghz(
    nmr_frequency_hz: float,
    maser_frequency_ghz: float,
) -> float:
    """Local oscillator frequency for upper-sideband up-conversion (GHz).

    f_LO = f_maser − f_NMR

    Args:
        nmr_frequency_hz: NMR Larmor frequency (Hz).
        maser_frequency_ghz: Target maser frequency (GHz).

    Returns:
        LO frequency (GHz).
    """
    return maser_frequency_ghz - nmr_frequency_hz / 1e9


def compute_bandwidth_utilization(
    nmr_bandwidth_hz: float,
    maser_bandwidth_hz: float,
) -> float:
    """Fraction of the maser gain bandwidth occupied by the NMR readout.

    Values > 1 mean the NMR readout bandwidth exceeds the maser gain
    band and signal will be distorted.

    Args:
        nmr_bandwidth_hz: NMR readout bandwidth (Hz).
        maser_bandwidth_hz: Maser 3-dB gain bandwidth (Hz).

    Returns:
        Bandwidth utilisation ratio (dimensionless).

    Raises:
        ValueError: If either bandwidth is non-positive.
    """
    if nmr_bandwidth_hz <= 0 or maser_bandwidth_hz <= 0:
        raise ValueError("Both bandwidths must be positive")
    return nmr_bandwidth_hz / maser_bandwidth_hz


def friis_system_temperature_with_mixer(
    coil_temperature_k: float,
    mixer: MixerSpec,
    maser_noise_temperature_k: float,
    maser_gain_db: float,
    lna_noise_temperature_k: float,
) -> float:
    """Friis cascade noise temperature: coil → mixer → maser → LNA.

    Referred to the coil (first-stage) input:

        T_sys = T_coil + T_mixer / G_coil + T_maser / (G_coil · G_mixer)
                + T_LNA / (G_coil · G_mixer · G_maser)

    For a lossless coil (G_coil = 1) and no coil insertion loss:

        T_sys = T_coil + T_mixer + T_maser / G_mixer + T_LNA / (G_mixer · G_maser)

    The mixer is passive so G_mixer = 1 / L_mixer (L_mixer > 1 is the
    linear conversion loss).

    Args:
        coil_temperature_k:
            Physical temperature of the coil conductor (K).  Represents
            kB·T_coil·BW thermal noise power available at the coil output.
        mixer:
            MixerSpec for the up-conversion mixer.
        maser_noise_temperature_k:
            Equivalent input noise temperature of the maser amplifier (K).
        maser_gain_db:
            Maser available gain (dB).  Positive for amplification.
        lna_noise_temperature_k:
            Equivalent noise temperature of the post-maser LNA (K).

    Returns:
        System noise temperature referred to the coil input (K).
    """
    # Mixer gain = −conversion_loss (it attenuates)
    g_mixer_linear = 10.0 ** (-mixer.conversion_loss_db / 10.0)  # < 1
    g_maser_linear = 10.0 ** (maser_gain_db / 10.0)

    t_mixer_k = _T0 * (10.0 ** (mixer.effective_nf_db / 10.0) - 1.0)

    t_sys = (
        coil_temperature_k
        + t_mixer_k                                        # mixer referred to coil input
        + maser_noise_temperature_k / g_mixer_linear       # maser referred to coil input
        + lna_noise_temperature_k / (g_mixer_linear * g_maser_linear)
    )
    return t_sys
