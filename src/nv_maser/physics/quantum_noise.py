"""
Quantum Langevin noise model for the NV diamond maser.

Classical Maxwell-Bloch ODEs are deterministic: they predict the mean
field and inversion, but give no information about the fundamental noise
floor of the maser output.  This module adds the quantum fluctuation
layer derived from the quantum Langevin equations (input-output theory).

Physical picture
────────────────
A maser operating above threshold generates a coherent microwave field
whose noise is dominated by *quantum phase diffusion*, not technical
noise.  The key results are:

1. **Schawlow-Townes linewidth** Δν_ST (Hz)
   The linewidth of the maser output is far narrower than the cavity
   linewidth κ_c/(2π) because the coherent photon number N̄ provides
   "inertia" against phase kicks:

       Δν_ST = (κ_c/2π) × n_sp / (2 N̄)

   For the NV maser at 50 mT (κ_c/(2π) ≈ 147 kHz, n_sp ≈ 1.5,
   N̄ ≈ 100–10 000): Δν_ST ≈ 10 Hz – 1 kHz.

2. **Population-inversion factor** n_sp
   Describes how much "excess" stimulated emission occurs relative to
   net gain.  For a uniformly pumped two-level system:

       n_sp = f_e / (f_e − f_g) = (1 + η) / (2η)

   where η = η_pump is the pump efficiency (fractional inversion).
   η → 1 ⇒ n_sp → 1 (ideal); η → 0 ⇒ n_sp → ∞ (all noise).

3. **Added noise** (Caves 1982 theorem)
   A phase-insensitive linear amplifier must add at least n_sp ≥ 1
   noise photons per bandwidth per second:

       n_add = n_sp

   This sets the quantum noise temperature:

       T_noise = ℏω n_sp / k_B

   For ν = 1.47 GHz and n_sp = 1.5:  T_noise ≈ 0.11 K — well below
   room temperature, explaining why masers are uniquely low-noise
   microwave amplifiers.

4. **Phase noise PSD** (single-sideband)
   Quantum phase diffusion gives a 1/f² spectrum:

       S_φ(f) = Δν_ST / (π f²)   [rad²/Hz]
       L(f)  = 10 log₁₀(S_φ(f)/2) [dBc/Hz]  (SSB convention)

5. **RIN spectrum** (relative intensity noise)
   Photon-number fluctuations are filtered by the cavity:

       RIN(f) = (2 n_sp / N̄) / (1 + (f / κ_c_hz)²)  [1/Hz]

   The DC plateau 2 n_sp / N̄ is the quantum shot-noise limit; the
   cavity acts as a low-pass filter with corner frequency κ_c_hz.

References
──────────
Schawlow, A. L., Townes, C. H. (1958) Phys. Rev. 112, 1940.
Caves, C. M. (1982) Phys. Rev. D 26, 1817.
Lax, M. (1966) Phys. Rev. 145, 110-129.
Yamamoto, Y. & Haus, H. A. (1986) Rev. Mod. Phys. 58, 1001.
Wang et al. (2024) Advanced Science, PMC11425272 — NV maser.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import MaserConfig, NVConfig
from .cavity import CavityProperties
from .maxwell_bloch import MaxwellBlochResult

# ── Physical constants ─────────────────────────────────────────────
_HBAR = 1.054571817e-34   # J·s
_KB   = 1.380649e-23      # J/K


# ── Result containers ──────────────────────────────────────────────

@dataclass(frozen=True)
class MaserNoiseResult:
    """Complete quantum noise characterisation of the NV diamond maser.

    All quantities derived from quantum Langevin theory (mean-field limit).
    """

    # Population inversion / amplifier noise factor
    population_inversion_factor: float
    """n_sp = (1 + η_pump) / (2 η_pump).  Minimum = 1.0 (full inversion)."""

    added_noise_number: float
    """n_add = n_sp — minimum added photons per mode per second (Caves)."""

    # Linewidth
    schawlow_townes_linewidth_hz: float
    """Δν_ST = (κ_c/2π) × n_sp / (2 N̄).  Fundamental maser linewidth (Hz)."""

    # Noise temperature
    noise_temperature_k: float
    """T_noise = ℏω n_sp / k_B.  Quantum noise temperature (K)."""

    # Photon statistics (from Maxwell-Bloch steady state)
    steady_state_photons: float
    """N̄ — mean intracavity photon number."""

    output_power_w: float
    """P_out — maser output power (W)."""

    # Reference noise metrics
    phase_noise_1hz_dbc_hz: float
    """L(1 Hz) = 10 log₁₀(Δν_ST / (2π)) dBc/Hz — phase noise at 1 Hz offset."""

    rin_floor_per_hz: float
    """2 n_sp / N̄  [1/Hz] — quantum shot-noise RIN plateau."""

    rin_floor_dbc_hz: float
    """10 log₁₀(RIN_floor) dBc/Hz."""

    # Cavity parameters stored for convenience
    cavity_linewidth_hz: float
    """κ_c / (2π) — cavity half-bandwidth (Hz)."""

    cavity_frequency_hz: float
    """ν_c — cavity resonance frequency (Hz)."""


@dataclass(frozen=True)
class PhaseNoiseSpectrum:
    """Single-sideband phase noise spectrum of the maser output.

    Models quantum phase diffusion from spontaneous emission events:

        S_φ(f) = Δν_ST / (π f²)   [rad²/Hz]
        L(f)  = 10 log₁₀(Δν_ST / (2π f²))  [dBc/Hz]

    The classic signature is a −20 dB/decade (1/f²) slope that gives
    the fundamental linewidth when extrapolated to the 0 dB intercept.
    """

    freq_offsets_hz: NDArray[np.float64]
    """Frequency offsets from carrier (Hz). Must be > 0."""

    psd_rad2_per_hz: NDArray[np.float64]
    """S_φ(f) = Δν_ST / (π f²) in rad²/Hz."""

    psd_dbc_hz: NDArray[np.float64]
    """L(f) = 10 log₁₀(S_φ(f)/2) in dBc/Hz (SSB convention)."""

    schawlow_townes_linewidth_hz: float
    """Δν_ST used for this spectrum."""


@dataclass(frozen=True)
class RINSpectrum:
    """Relative intensity noise (RIN) spectrum.

    Photon-number fluctuations in the maser cavity have a Lorentzian
    power spectral density filtered by the cavity decay:

        RIN(f) = (2 n_sp / N̄) / (1 + (f / κ_c_hz)²)   [1/Hz]

    The ``rin_floor_per_hz`` is the DC plateau value 2 n_sp / N̄, which
    represents the quantum shot-noise limited RIN.  Above the cavity
    linewidth (corner frequency), photon-number fluctuations are
    quenched by the cavity filter.
    """

    freq_offsets_hz: NDArray[np.float64]
    """Frequency offsets (Hz). May include 0."""

    rin_per_hz: NDArray[np.float64]
    """RIN(f) = (2 n_sp / N̄) / (1 + (f/κ_c)²) in 1/Hz."""

    rin_dbc_hz: NDArray[np.float64]
    """10 log₁₀(RIN(f)) in dBc/Hz."""

    cavity_linewidth_hz: float
    """κ_c/(2π) — corner frequency of the Lorentzian roll-off (Hz)."""

    rin_floor_per_hz: float
    """DC plateau = 2 n_sp / N̄ (1/Hz). The quantum shot-noise limit."""


# ── Core physics functions ─────────────────────────────────────────


def compute_population_inversion_factor(pump_efficiency: float) -> float:
    """Compute the population inversion (noise) factor n_sp.

    For a homogeneously pumped two-level system with fractional inversion
    η (pump_efficiency):

        n_sp = f_e / (f_e − f_g) = (1 + η) / (2η)

    Interpretation:
    - η = 1 (complete inversion, ground state empty) → n_sp = 1 (ideal).
    - η = 0.5 (equal populations, barely above threshold) → n_sp = 1.5.
    - η → 0 (no inversion) → n_sp → ∞ (maximum noise).

    Args:
        pump_efficiency: Fractional spin inversion η ∈ (0, 1].

    Returns:
        n_sp ∈ [1, ∞).

    Raises:
        ValueError: if pump_efficiency ≤ 0 or > 1.
    """
    if pump_efficiency <= 0.0 or pump_efficiency > 1.0:
        raise ValueError(
            f"pump_efficiency must be in (0, 1], got {pump_efficiency}."
        )
    return (1.0 + pump_efficiency) / (2.0 * pump_efficiency)


def compute_schawlow_townes_linewidth(
    cavity_linewidth_hz: float,
    steady_state_photons: float,
    n_sp: float,
) -> float:
    """Schawlow-Townes quantum phase-diffusion linewidth (Hz).

    The maser output linewidth is reduced below the cavity linewidth
    by the coherent photon number N̄, which suppresses phase diffusion:

        Δν_ST = (κ_c/2π) × n_sp / (2 N̄)

    where κ_c/(2π) = cavity_linewidth_hz.

    For typical NV maser parameters (κ_c/(2π) ≈ 147 kHz, N̄ ≈ 100):
        Δν_ST ≈ 1.1 kHz  (≪ 147 kHz cavity linewidth).

    Args:
        cavity_linewidth_hz: κ_c/(2π) — cavity half-bandwidth in Hz.
        steady_state_photons: N̄ — mean intracavity photon number.
        n_sp: Population inversion factor (see compute_population_inversion_factor).

    Returns:
        Schawlow-Townes linewidth Δν_ST in Hz.

    Raises:
        ValueError: if any argument is non-positive.
    """
    if cavity_linewidth_hz <= 0.0:
        raise ValueError(f"cavity_linewidth_hz must be > 0, got {cavity_linewidth_hz}.")
    if n_sp < 1.0:
        raise ValueError(f"n_sp must be ≥ 1, got {n_sp}.")
    if steady_state_photons <= 0.0:
        # Below threshold or trivially small: return cavity linewidth itself
        # (linewidth can't be narrower than cavity in absence of photons)
        return cavity_linewidth_hz
    return cavity_linewidth_hz * n_sp / (2.0 * steady_state_photons)


def compute_added_noise(n_sp: float) -> float:
    """Added noise photon number (Caves 1982 theorem).

    A phase-insensitive linear amplifier must add at least n_sp noise
    photons per mode bandwidth — this is the quantum noise floor set
    by stimulated and spontaneous emission:

        n_add = n_sp

    The Caves lower bound n_add ≥ 1 is achieved only for complete
    inversion (n_sp = 1, η_pump = 1).

    Args:
        n_sp: Population inversion factor (≥ 1).

    Returns:
        n_add = n_sp (added noise photon number).

    Raises:
        ValueError: if n_sp < 1.
    """
    if n_sp < 1.0:
        raise ValueError(f"n_sp must be ≥ 1 (Caves theorem), got {n_sp}.")
    return float(n_sp)


def compute_noise_temperature(frequency_hz: float, n_sp: float) -> float:
    """Quantum noise temperature of the maser amplifier (K).

    The minimum noise temperature for a maser/laser amplifier arises
    from the quantum vacuum fluctuations and stimulated emission noise:

        T_noise = ℏω n_sp / k_B

    For the NV maser at 1.47 GHz with n_sp = 1.5:
        T_noise ≈ 0.11 K  (far below room temperature).

    This explains why room-temperature masers are among the lowest-noise
    microwave amplifiers available.

    Args:
        frequency_hz: Cavity/signal frequency (Hz).
        n_sp: Population inversion factor (≥ 1).

    Returns:
        Noise temperature T_noise (K).

    Raises:
        ValueError: if frequency_hz ≤ 0 or n_sp < 1.
    """
    if frequency_hz <= 0.0:
        raise ValueError(f"frequency_hz must be > 0, got {frequency_hz}.")
    if n_sp < 1.0:
        raise ValueError(f"n_sp must be ≥ 1, got {n_sp}.")
    omega = 2.0 * math.pi * frequency_hz
    return (_HBAR * omega * n_sp) / _KB


def compute_phase_noise_spectrum(
    st_linewidth_hz: float,
    freq_offsets_hz: NDArray[np.float64],
) -> PhaseNoiseSpectrum:
    """Single-sideband phase noise spectrum from quantum phase diffusion.

    The phase of the maser field undergoes a random walk due to
    spontaneous emission events.  This gives a 1/f² (−20 dB/decade)
    phase noise PSD:

        S_φ(f) = Δν_ST / (π f²)   [rad²/Hz]
        L(f)  = 10 log₁₀(S_φ(f)/2) [dBc/Hz]  (SSB, single-sided)

    The Schawlow-Townes linewidth Δν_ST is the 1/f² corner: the output
    spectrum is a Lorentzian of HWHM = Δν_ST.

    Args:
        st_linewidth_hz: Δν_ST — Schawlow-Townes linewidth (Hz).
        freq_offsets_hz: Array of positive frequency offsets (Hz).

    Returns:
        PhaseNoiseSpectrum with PSD in rad²/Hz and dBc/Hz.

    Raises:
        ValueError: if any freq_offset ≤ 0 or st_linewidth_hz ≤ 0.
    """
    if st_linewidth_hz <= 0.0:
        raise ValueError(f"st_linewidth_hz must be > 0, got {st_linewidth_hz}.")
    freq = np.asarray(freq_offsets_hz, dtype=np.float64)
    if np.any(freq <= 0.0):
        raise ValueError("All freq_offsets_hz must be > 0 (phase noise is undefined at DC).")

    # S_φ(f) = Δν_ST / (π f²)
    psd = st_linewidth_hz / (math.pi * freq**2)

    # SSB phase noise in dBc/Hz: L(f) = 10 log10(S_φ(f) / 2)
    psd_dbc = 10.0 * np.log10(psd / 2.0)

    return PhaseNoiseSpectrum(
        freq_offsets_hz=freq,
        psd_rad2_per_hz=psd,
        psd_dbc_hz=psd_dbc,
        schawlow_townes_linewidth_hz=st_linewidth_hz,
    )


def compute_rin_spectrum(
    cavity_linewidth_hz: float,
    steady_state_photons: float,
    n_sp: float,
    freq_offsets_hz: NDArray[np.float64],
) -> RINSpectrum:
    """Quantum-limited RIN spectrum for the maser output.

    Photon-number fluctuations in the cavity are described by a
    Lorentzian spectrum with corner frequency equal to the cavity
    linewidth κ_c/(2π):

        RIN(f) = (2 n_sp / N̄) / (1 + (f / κ_c_hz)²)   [1/Hz]

    Physical interpretation:
    - For f ≪ κ_c_hz: RIN ≈ 2 n_sp / N̄  (quantum shot-noise plateau).
    - For f ≫ κ_c_hz: RIN ∝ 1/f² (cavity filters fast fluctuations).
    - The corner frequency is the cavity half-bandwidth.

    Note: To observe the flat shot-noise floor, the detection bandwidth
    must exceed the cavity linewidth.

    Args:
        cavity_linewidth_hz: κ_c/(2π) — cavity half-bandwidth (Hz).
        steady_state_photons: N̄ — mean intracavity photon number.
        n_sp: Population inversion factor (≥ 1).
        freq_offsets_hz: Array of frequency offsets (Hz); may include 0.

    Returns:
        RINSpectrum with linear and dB RIN values.

    Raises:
        ValueError: if cavity_linewidth_hz ≤ 0, n_sp < 1, or N̄ ≤ 0.
    """
    if cavity_linewidth_hz <= 0.0:
        raise ValueError(f"cavity_linewidth_hz must be > 0, got {cavity_linewidth_hz}.")
    if n_sp < 1.0:
        raise ValueError(f"n_sp must be ≥ 1, got {n_sp}.")
    if steady_state_photons <= 0.0:
        raise ValueError(f"steady_state_photons must be > 0, got {steady_state_photons}.")

    freq = np.asarray(freq_offsets_hz, dtype=np.float64)
    rin_floor = 2.0 * n_sp / steady_state_photons
    rin = rin_floor / (1.0 + (freq / cavity_linewidth_hz) ** 2)

    return RINSpectrum(
        freq_offsets_hz=freq,
        rin_per_hz=rin,
        rin_dbc_hz=10.0 * np.log10(np.where(rin > 0, rin, 1e-300)),
        cavity_linewidth_hz=cavity_linewidth_hz,
        rin_floor_per_hz=rin_floor,
    )


# ── Top-level entry point ──────────────────────────────────────────


def compute_maser_noise(
    cavity_props: CavityProperties,
    mb_result: MaxwellBlochResult,
    nv_config: NVConfig,
    maser_config: MaserConfig,
) -> MaserNoiseResult:
    """Complete quantum noise characterisation of the NV maser.

    Combines results from the cavity model (CavityProperties) and the
    semiclassical Maxwell-Bloch solver (MaxwellBlochResult) to compute
    the full quantum noise budget:

    - Schawlow-Townes linewidth (quantum phase diffusion)
    - Population inversion factor n_sp
    - Added noise number (Caves theorem)
    - Quantum noise temperature
    - Phase noise at 1 Hz offset
    - RIN floor

    Args:
        cavity_props: Pre-computed cavity properties (from cavity.py).
        mb_result: Result of solve_maxwell_bloch (mean-field dynamics).
        nv_config: NV center parameters (pump_efficiency used for n_sp).
        maser_config: Cavity parameters (frequency for T_noise).

    Returns:
        MaserNoiseResult with all quantum noise metrics.

    Notes:
        Below threshold (steady_state_photons ≈ 0), Schawlow-Townes
        linewidth defaults to the cavity linewidth — the maser is not
        coherent and the concept of a carrier phase does not apply.
        This is flagged by Δν_ST ≈ κ_c/(2π) in the output.
    """
    kappa_hz = cavity_props.cavity_linewidth_hz
    freq_hz = maser_config.cavity_frequency_ghz * 1e9
    eta = nv_config.pump_efficiency

    n_sp = compute_population_inversion_factor(eta)
    n_add = compute_added_noise(n_sp)
    n_bar = mb_result.steady_state_photons
    t_noise = compute_noise_temperature(freq_hz, n_sp)

    delta_nu = compute_schawlow_townes_linewidth(kappa_hz, n_bar, n_sp)

    # Phase noise at 1 Hz offset: L(1 Hz) = 10 log10(Δν_ST / (2π))
    if n_bar > 0:
        phase_noise_1hz = 10.0 * math.log10(delta_nu / (2.0 * math.pi))
    else:
        phase_noise_1hz = float("nan")

    rin_floor = 2.0 * n_sp / n_bar if n_bar > 0 else float("inf")
    rin_floor_dbc = 10.0 * math.log10(rin_floor) if math.isfinite(rin_floor) else float("nan")

    return MaserNoiseResult(
        population_inversion_factor=n_sp,
        added_noise_number=n_add,
        schawlow_townes_linewidth_hz=delta_nu,
        noise_temperature_k=t_noise,
        steady_state_photons=n_bar,
        output_power_w=mb_result.output_power_w,
        phase_noise_1hz_dbc_hz=phase_noise_1hz,
        rin_floor_per_hz=rin_floor,
        rin_floor_dbc_hz=rin_floor_dbc,
        cavity_linewidth_hz=kappa_hz,
        cavity_frequency_hz=freq_hz,
    )
