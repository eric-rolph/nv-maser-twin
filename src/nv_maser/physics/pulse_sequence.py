"""
Analytical NMR pulse sequence simulator.

Implements closed-form (Ernst-equation) signal models for common MRI pulse
sequences: spin echo (SE), CPMG multi-echo, gradient echo (GRE), and
inversion recovery (IR).  All results are expressed as *normalised* signal
M_xy / M₀ so that they can be scaled by any absolute magnetisation.

Sequence | Formula
---------|-------------------------------------------------
SE       | sin α · exp(-TE/T₂) · (1 − 2·E₁·exp(+(TE/2)/T₁) + E₁)
CPMG     | sin α · (1 − E₁) · exp(−t_n / T₂)  for each echo n
GRE      | sin α · (1 − E₁) / (1 − cos α · E₁) · exp(−TE/T₂*)
IR       | |1 − 2·exp(−TI/T₁) + E₁|

where E₁ = exp(−TR/T₁), E₂ = exp(−TE/T₂), E₂* = exp(−TE/T₂*).

References
──────────
Haacke, Brown, Thompson, Venkatesan — "Magnetic Resonance Imaging:
Physical Principles and Sequence Design", 2nd ed. (2014).

Bernstein, King, Zhou — "Handbook of MRI Pulse Sequences" (2004).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ── Physical constants ────────────────────────────────────────────


def _deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Result dataclasses                                              ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class SpinEchoResult:
    """Analytical spin-echo signal (perfect 180° refocus).

    Attributes
    ----------
    signal_normalized:
        Transverse magnetisation M_xy / M₀ at echo centre.
    te_ms, tr_ms, flip_angle_deg: sequence parameters.
    t1_ms, t2_ms: tissue relaxation times.
    e1, e2: decay factors E₁ = exp(−TR/T₁), E₂ = exp(−TE/T₂).
    t1_saturation_factor: fraction of equilibrium Mz recovered.
    t2_decay_factor: fraction of signal remaining at TE.
    """

    signal_normalized: float
    te_ms: float
    tr_ms: float
    flip_angle_deg: float
    t1_ms: float
    t2_ms: float
    e1: float
    e2: float
    t1_saturation_factor: float
    t2_decay_factor: float


@dataclass(frozen=True)
class CPMGResult:
    """CPMG multi-echo train result.

    Attributes
    ----------
    echo_amplitudes:
        Normalised echo amplitudes M_xy(t_n) / M₀ for each echo.
    echo_times_ms:
        Centre time of each echo (ms).
    esp_ms:
        Echo spacing (ms); time between consecutive echo centres.
    n_echoes:
        Number of echoes in the train.
    te_first_ms:
        Time of the first echo centre.
    t2_eff_ms:
        Effective T₂ estimated from monoexponential fit through
        the echo-amplitude train.
    t1_ms, t2_ms: tissue relaxation times.
    """

    echo_amplitudes: tuple[float, ...]
    echo_times_ms: tuple[float, ...]
    esp_ms: float
    n_echoes: int
    te_first_ms: float
    t2_eff_ms: float
    t1_ms: float
    t2_ms: float


@dataclass(frozen=True)
class GREResult:
    """Steady-state gradient-echo (Ernst-equation) result.

    The GRE sequence uses a small flip angle so that steady-state
    transverse magnetisation is maximised for a given TR/T₁.

    Attributes
    ----------
    signal_normalized:
        M_xy / M₀ at echo centre (uses T₂* for TE decay).
    te_ms, tr_ms, flip_angle_deg: sequence parameters.
    t1_ms, t2_star_ms: tissue relaxation times.
    e1: exp(−TR/T₁).
    e2_star: exp(−TE/T₂*).
    ernst_angle_deg:
        Optimal flip angle for this TR/T₁ combination.
    """

    signal_normalized: float
    te_ms: float
    tr_ms: float
    flip_angle_deg: float
    t1_ms: float
    t2_star_ms: float
    e1: float
    e2_star: float
    ernst_angle_deg: float


@dataclass(frozen=True)
class InversionRecoveryResult:
    """Inversion-recovery (T₁ preparation) result.

    Attributes
    ----------
    signal_normalized:
        Longitudinal magnetisation Mz / M₀ at time TI.
    ti_ms: inversion time.
    tr_ms: repetition time.
    t1_ms: tissue T₁.
    null_point_ms:
        TI at which the recovered signal crosses zero.
    """

    signal_normalized: float
    ti_ms: float
    tr_ms: float
    t1_ms: float
    null_point_ms: float


@dataclass(frozen=True)
class SNREfficiency:
    """SNR efficiency for sequence parameter optimisation.

    Attributes
    ----------
    signal_normalized:
        M_xy / M₀ for the chosen sequence and parameters.
    scan_time_per_slice_s:
        TR (the time cost per phase-encoding step).
    snr_per_sqrt_scan_time:
        signal_normalized / sqrt(TR_s); proportional to SNR / sqrt(total_scan).
    ernst_angle_deg:
        Optimal flip angle maximising SNR/√time for this TR/T₁.
    """

    signal_normalized: float
    scan_time_per_slice_s: float
    snr_per_sqrt_scan_time: float
    ernst_angle_deg: float


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Public API                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝


def simulate_spin_echo(
    t1_ms: float,
    t2_ms: float,
    tr_ms: float,
    te_ms: float,
    flip_angle_deg: float = 90.0,
) -> SpinEchoResult:
    """Analytical spin-echo signal with perfect 90° excitation / 180° refocus.

    For an arbitrary excitation flip angle α the formula is:

        S_SE = sin(α) · exp(−TE / T₂) · (1 − 2·exp(−(TR − TE/2) / T₁) + E₁)

    where E₁ = exp(−TR / T₁).  With α = 90° and TR ≫ TE this
    simplifies to M₀ · E₂ · (1 − E₁).

    The refocusing pulse is assumed to be a perfect 180° pulse
    (field-inhomogeneity dephasing is fully reversed at TE).

    Args:
        t1_ms: Longitudinal relaxation time (ms).
        t2_ms: Transverse relaxation time (ms).
        tr_ms: Repetition time (ms); must satisfy TR > TE.
        te_ms: Echo time (ms).
        flip_angle_deg: Excitation flip angle (degrees).

    Returns:
        SpinEchoResult with normalised signal and intermediate factors.

    Raises:
        ValueError: If TE ≥ TR, or if any time is non-positive.
    """
    if te_ms >= tr_ms:
        raise ValueError(f"TE ({te_ms} ms) must be less than TR ({tr_ms} ms)")
    if t1_ms <= 0 or t2_ms <= 0 or te_ms <= 0 or tr_ms <= 0:
        raise ValueError("All relaxation times and sequence timings must be positive")

    alpha_rad = _deg_to_rad(flip_angle_deg)
    e1 = math.exp(-tr_ms / t1_ms)
    e2 = math.exp(-te_ms / t2_ms)

    # Haacke eq. 18.13: include partial recovery between 180° and next excitation
    e1_half = math.exp(-(tr_ms - te_ms / 2.0) / t1_ms)
    sat = 1.0 - 2.0 * e1_half + e1

    signal = math.sin(alpha_rad) * e2 * sat

    return SpinEchoResult(
        signal_normalized=signal,
        te_ms=te_ms,
        tr_ms=tr_ms,
        flip_angle_deg=flip_angle_deg,
        t1_ms=t1_ms,
        t2_ms=t2_ms,
        e1=e1,
        e2=e2,
        t1_saturation_factor=sat,
        t2_decay_factor=e2,
    )


def simulate_cpmg(
    t1_ms: float,
    t2_ms: float,
    tr_ms: float,
    esp_ms: float,
    n_echoes: int,
    te_first_ms: float | None = None,
    flip_angle_deg: float = 90.0,
) -> CPMGResult:
    """CPMG multi-echo train signal (analytical model).

    The CPMG sequence consists of a 90° excitation followed by a train of
    180° refocusing pulses with inter-pulse spacing ESP.  For an ideal
    180° refocus train, the echo amplitudes decay as:

        E_n = sin(α) · (1 − E₁) · exp(−t_n / T₂)

    where t_n = TE_first + (n − 1) × ESP and E₁ = exp(−TR / T₁).

    The *effective* T₂ is back-calculated from a monoexponential fit
    through the echo-amplitude series; for the ideal case it equals the
    true T₂.

    Args:
        t1_ms: Longitudinal relaxation time (ms).
        t2_ms: Transverse relaxation time (ms).
        tr_ms: Repetition time (ms).
        esp_ms: Echo spacing — time between consecutive echo centres (ms).
        n_echoes: Number of echoes to generate (≥ 1).
        te_first_ms: Time of first echo (ms). Defaults to ``esp_ms``.
        flip_angle_deg: Excitation flip angle (degrees; 90° is optimal).

    Returns:
        CPMGResult with echo amplitudes and times, and fitted T₂_eff.

    Raises:
        ValueError: If ``n_echoes < 1`` or timing parameters are non-positive.
    """
    if n_echoes < 1:
        raise ValueError(f"n_echoes must be >= 1, got {n_echoes}")
    if t1_ms <= 0 or t2_ms <= 0 or tr_ms <= 0 or esp_ms <= 0:
        raise ValueError("Relaxation times and sequence timings must be positive")

    if te_first_ms is None:
        te_first_ms = esp_ms

    alpha_rad = _deg_to_rad(flip_angle_deg)
    e1 = math.exp(-tr_ms / t1_ms)
    sat = 1.0 - e1  # (1 − exp(−TR/T₁))

    times: list[float] = []
    amps: list[float] = []

    for n in range(1, n_echoes + 1):
        t_n = te_first_ms + (n - 1) * esp_ms
        e2_n = math.exp(-t_n / t2_ms)
        amp = math.sin(alpha_rad) * sat * e2_n
        times.append(t_n)
        amps.append(amp)

    # Estimate T2_eff from log-linear fit of amplitudes vs. time
    if n_echoes == 1:
        t2_eff_ms = t2_ms  # can't fit from one point
    else:
        log_amps = np.log(np.abs(amps))
        t_arr = np.array(times)
        # Linear regression: log(S) = log(S₀) − t / T₂_eff
        # slope = −1/T₂_eff
        coeffs = np.polyfit(t_arr, log_amps, 1)
        slope = coeffs[0]
        t2_eff_ms = float(-1.0 / slope) if slope < 0 else t2_ms

    return CPMGResult(
        echo_amplitudes=tuple(amps),
        echo_times_ms=tuple(times),
        esp_ms=esp_ms,
        n_echoes=n_echoes,
        te_first_ms=te_first_ms,
        t2_eff_ms=t2_eff_ms,
        t1_ms=t1_ms,
        t2_ms=t2_ms,
    )


def simulate_gre(
    t1_ms: float,
    t2_star_ms: float,
    tr_ms: float,
    te_ms: float,
    flip_angle_deg: float = 30.0,
) -> GREResult:
    """Steady-state gradient-echo signal (Ernst equation).

    The GRE (FLASH/SPGR) steady-state signal is:

        S_GRE = sin(α) · (1 − E₁) / (1 − cos(α) · E₁) · exp(−TE / T₂*)

    where E₁ = exp(−TR / T₁).

    At the Ernst angle θ_E = arccos(E₁) the signal is maximised.

    Note: GRE uses *T₂** (includes susceptibility dephasing), unlike
    spin echo which uses T₂ (susceptibility refocused).

    Args:
        t1_ms: Longitudinal relaxation time (ms).
        t2_star_ms: Effective transverse relaxation time T₂* (ms).
        tr_ms: Repetition time (ms).
        te_ms: Echo time (ms).
        flip_angle_deg: Excitation flip angle (degrees).

    Returns:
        GREResult with normalised signal and Ernst-angle calculation.

    Raises:
        ValueError: If TE ≥ TR or any time is non-positive.
    """
    if te_ms >= tr_ms:
        raise ValueError(f"TE ({te_ms} ms) must be less than TR ({tr_ms} ms)")
    if t1_ms <= 0 or t2_star_ms <= 0 or te_ms <= 0 or tr_ms <= 0:
        raise ValueError("All relaxation times and sequence timings must be positive")

    alpha_rad = _deg_to_rad(flip_angle_deg)
    e1 = math.exp(-tr_ms / t1_ms)
    e2_star = math.exp(-te_ms / t2_star_ms)

    denom = 1.0 - math.cos(alpha_rad) * e1
    if abs(denom) < 1e-20:
        # Numerically singular (cos α ≈ E₁); limit approaches zero
        signal = 0.0
    else:
        signal = math.sin(alpha_rad) * (1.0 - e1) / denom * e2_star

    # Ernst angle: cos(θ_E) = E₁ → signal maximised for GRE
    ernst_rad = math.acos(e1)
    ernst_deg = math.degrees(ernst_rad)

    return GREResult(
        signal_normalized=signal,
        te_ms=te_ms,
        tr_ms=tr_ms,
        flip_angle_deg=flip_angle_deg,
        t1_ms=t1_ms,
        t2_star_ms=t2_star_ms,
        e1=e1,
        e2_star=e2_star,
        ernst_angle_deg=ernst_deg,
    )


def simulate_inversion_recovery(
    t1_ms: float,
    tr_ms: float,
    ti_ms: float,
) -> InversionRecoveryResult:
    """Inversion-recovery (IR) longitudinal magnetisation at time TI.

    Mz(TI) / M₀ = 1 − 2·exp(−TI / T₁) + exp(−TR / T₁)

    The signal null point where Mz = 0 satisfies:

        TI_null = T₁ · ln(2 / (1 + exp(−TR / T₁)))

    When TR ≫ T₁ this approaches the familiar T₁ · ln(2).

    Args:
        t1_ms: Longitudinal relaxation time (ms).
        tr_ms: Repetition time (ms); must satisfy TR ≥ TI.
        ti_ms: Inversion time (ms).

    Returns:
        InversionRecoveryResult with normalised Mz at TI and null point.

    Raises:
        ValueError: If TI > TR or any time is non-positive.
    """
    if ti_ms > tr_ms:
        raise ValueError(f"TI ({ti_ms} ms) must be ≤ TR ({tr_ms} ms)")
    if t1_ms <= 0 or ti_ms <= 0 or tr_ms <= 0:
        raise ValueError("All times must be positive")

    e1 = math.exp(-tr_ms / t1_ms)
    signal = 1.0 - 2.0 * math.exp(-ti_ms / t1_ms) + e1

    # Null point: TI_null = T₁ ln(2 / (1 + E₁))
    denom_log = 1.0 + e1
    if denom_log <= 0:
        ti_null_ms = float("nan")
    else:
        ti_null_ms = t1_ms * math.log(2.0 / denom_log)

    return InversionRecoveryResult(
        signal_normalized=signal,
        ti_ms=ti_ms,
        tr_ms=tr_ms,
        t1_ms=t1_ms,
        null_point_ms=ti_null_ms,
    )


def ernst_angle(t1_ms: float, tr_ms: float) -> float:
    """Optimal GRE flip angle maximising M_xy (Ernst angle).

    θ_E = arccos(exp(−TR / T₁))

    At the Ernst angle the GRE signal S_GRE = M₀ · (1 − E₁) / (1 + E₁)
    · exp(−TE / T₂*) is maximum.

    Args:
        t1_ms: Longitudinal relaxation time (ms).
        tr_ms: Repetition time (ms).

    Returns:
        Ernst angle in degrees.
    """
    if t1_ms <= 0 or tr_ms <= 0:
        raise ValueError("t1_ms and tr_ms must be positive")
    e1 = math.exp(-tr_ms / t1_ms)
    return math.degrees(math.acos(e1))


def optimal_te_for_contrast(
    t2_a_ms: float,
    t2_b_ms: float,
) -> float:
    """TE that maximises contrast between two tissues with T₂_a and T₂_b.

    For a spin echo, contrast |S_a − S_b| is maximised at:

        TE_opt = ln(T₂_a / T₂_b) / (1/T₂_b − 1/T₂_a)
                = T₂_a · T₂_b · ln(T₂_a / T₂_b) / (T₂_a − T₂_b)

    Args:
        t2_a_ms: T₂ of tissue A (ms). Must differ from t2_b_ms.
        t2_b_ms: T₂ of tissue B (ms).

    Returns:
        Optimal TE (ms) for maximum T₂ contrast.

    Raises:
        ValueError: If T₂_a == T₂_b (no contrast possible).
    """
    if abs(t2_a_ms - t2_b_ms) < 1e-12:
        raise ValueError("t2_a_ms and t2_b_ms must differ for contrast optimisation")
    if t2_a_ms <= 0 or t2_b_ms <= 0:
        raise ValueError("T₂ values must be positive")

    return t2_a_ms * t2_b_ms * math.log(t2_a_ms / t2_b_ms) / (t2_a_ms - t2_b_ms)


def snr_efficiency(
    t1_ms: float,
    t2_ms: float,
    tr_ms: float,
    te_ms: float,
    flip_angle_deg: float = 90.0,
) -> SNREfficiency:
    """SNR efficiency for spin-echo sequence parameter optimisation.

    SNR per unit sqrt(scan_time) ∝ S / √TR (for fixed FOV and matrix).
    The Ernst angle for GRE is also returned for reference.

    Args:
        t1_ms: Longitudinal relaxation time (ms).
        t2_ms: Transverse relaxation time (ms).
        tr_ms: Repetition time (ms).
        te_ms: Echo time (ms).
        flip_angle_deg: Excitation flip angle (degrees).

    Returns:
        SNREfficiency with signal, scan time, and SNR/√time metric.
    """
    se = simulate_spin_echo(t1_ms, t2_ms, tr_ms, te_ms, flip_angle_deg)
    tr_s = tr_ms * 1e-3
    snr_sqrt_time = se.signal_normalized / math.sqrt(tr_s)

    return SNREfficiency(
        signal_normalized=se.signal_normalized,
        scan_time_per_slice_s=tr_s,
        snr_per_sqrt_scan_time=snr_sqrt_time,
        ernst_angle_deg=ernst_angle(t1_ms, tr_ms),
    )
