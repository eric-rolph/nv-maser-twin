"""Frequency-resolved maser gain curve, bandwidth, and probe-power saturation.

Extends the single-frequency (resonance-only) model in :mod:`.amplifier` to a
complete gain *curve* G(f) across an arbitrary frequency grid.  The central
formula comes directly from one-port coupled-mode theory (CMT) applied to the
maser cavity with spin gain.

Physics
───────
At arbitrary probe frequency f, the reflection amplitude for a one-port
maser cavity is (CMT, linearised about resonance):

    S₁₁(f) = [N  +  2jQ_L δ] / [D  +  2jQ_L δ]

where

    δ   = (f − f_c) / f_c      fractional detuning
    N   = (β−1)/[Q_L(1+β)] + 1/Q_m   numerator (same as ``_gain_voltage`` in amplifier.py)
    D   = 1/Q_L − 1/Q_m              denominator (net effective loss; > 0 below threshold)

Power gain:

    G(f) = |S₁₁(f)|² = (N² + [2Q_L δ]²) / (D² + [2Q_L δ]²)

At resonance (δ = 0) this recovers the :func:`.amplifier.compute_maser_gain`
result exactly.  Far off-resonance (δ → ∞) the gain approaches 1 (passive
reflection with no net amplification).

The 3 dB bandwidth (half-power points) follows the parametric-narrowing
formula already stored in :attr:`.amplifier.MaserGainResult.bandwidth_hz`:

    B_3dB ≈ f_c (Q_m − Q_L) / (Q_L Q_m)

:func:`compute_bandwidth_3db` evaluates this numerically from the gain
array as an independent cross-check.

Probe-power saturation
──────────────────────
At finite input power P_in the intracavity photon number builds up and
partially depletes the spin inversion, increasing the effective magnetic Q:

    Q_m_eff = Q_m · (1 + P_in / P_sat)

where the saturation power P_sat is the input power at which the intracavity
photon number equals the critical photon number n_crit:

    n_crit = (γ_⊥ / (2 g₀))² / N_eff

    γ_⊥  — spin linewidth FWHM (ordinary Hz)
    g₀   — single-spin vacuum coupling (ordinary Hz)
    N_eff — effective number of inverted spins

    κ_ext — output-port decay rate = ω_c β / [Q_L (1+β)]  (rad/s)

    P_sat = n_crit · κ_ext · ℏ · ω_c                      (W)

Pass ``probe_power_w`` to :func:`compute_gain_curve` together with the
saturation parameters to obtain the compressed gain curve at finite power.

References
──────────
Wang et al., Advanced Science (2024), PMC11425272.
    Eq. 7: G = (Q_L / (Q_m − Q_L))² for critical coupling (β = 1).
    Validated: G ≈ 14.5 dB, B ≈ 340 kHz at Q_m = 1589, Q_L = 1337, f_c = 2.87 GHz.

Breeze et al., Nature 555, 493 (2018).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..config import CavityConfig, MaserConfig, NVConfig
from .cavity import compute_cavity_properties, compute_effective_q, compute_n_effective

if TYPE_CHECKING:
    from .spectral_maxwell_bloch import SpectralMBResult  # pragma: no cover


# ── Physical constants ────────────────────────────────────────────
from .constants import HBAR as _HBAR

# ── Result dataclasses ────────────────────────────────────────────

@dataclass(frozen=True)
class GainCurveResult:
    """Frequency-resolved maser gain curve and derived metrics.

    The gain G(f) = |S₁₁(f)|² is the power reflection gain as a function of
    probe frequency.  At resonance it equals the small-signal gain computed by
    :func:`.amplifier.compute_maser_gain`.
    """

    frequencies_hz: NDArray[np.float64]
    """Probe frequency grid (Hz), shape ``(n_points,)``."""

    gain_linear: NDArray[np.float64]
    """Power gain G(f) = |S₁₁(f)|², shape ``(n_points,)``.

    Values ≥ 1 throughout the amplifier regime (device is a net amplifier
    at all frequencies in the gain window).  Approaches 1 far off-resonance.
    """

    peak_gain_linear: float
    """Maximum gain on the grid (linear scale, ≥ 1 for an amplifier)."""

    peak_gain_db: float
    """Maximum gain in dB: 10 log₁₀(peak_gain_linear)."""

    peak_frequency_hz: float
    """Frequency at which the gain peaks (Hz).

    Equals *f_c* for a symmetric frequency grid centred on resonance.
    """

    bandwidth_3db_hz: float
    """Numerically extracted 3 dB bandwidth of the gain curve (Hz).

    Defined as the width of the region where
    G(f) ≥ 1 + (G_max − 1) / 2
    (i.e., half of the gain *above unity*).
    This definition correctly ignores passive off-resonance reflection.
    """

    bandwidth_analytical_hz: float
    """Analytical parametric-narrowing bandwidth (Hz).

    B = f_c (Q_m_eff − Q_L) / (Q_L Q_m_eff).
    Independent of the frequency grid; serves as a cross-check for
    ``bandwidth_3db_hz``.
    """

    gain_bandwidth_hz: float
    """Gain-voltage × bandwidth product √G_max × B_3dB (Hz).

    For critically coupled maser (β = 1) this equals f_c / Q_m —
    the spin-gain linewidth — regardless of how close Q_m is to Q_L.
    """

    below_threshold: bool
    """True when Q_m_eff > Q_L (the device is a linear amplifier)."""

    saturation_power_w: float
    """Input-referred saturation power P_sat (W).

    NaN if saturation parameters were not provided.
    """

    saturation_power_dbm: float
    """P_sat expressed in dBm.  NaN if P_sat unavailable or zero."""

    effective_q_m: float
    """Effective magnetic Q after probe-power saturation.

    Equals *q_m* when ``probe_power_w = 0`` (small-signal limit).
    """


# ── Core frequency-domain gain formula ────────────────────────────

def _s11_power_gain_at_detuning(
    q_m: float,
    q_l: float,
    coupling_beta: float,
    u: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Vectorised power gain G(f) = |S₁₁(f)|² for array *u*.

    CMT derivation (angular-frequency domain, mapped to normalised detuning
    u = 2 Q_L δ with δ = (f − f_c)/f_c):

    .. math::

        G(f) = \frac{\tilde{N}^2 + u^2}{\tilde{D}^2 + u^2}

    where the *Q_L-normalised* amplitudes are

    .. math::

        \tilde{N} = \frac{\beta-1}{1+\beta} + \frac{Q_L}{Q_m},\quad
        \tilde{D} = 1 - \frac{Q_L}{Q_m} = \frac{Q_m - Q_L}{Q_m}

    At resonance (u = 0): G = (\tilde{N}/\tilde{D})² equals the result from
    ``_gain_voltage`` in ``amplifier.py`` (both use the same physics).

    Note: The amplifier-module amplitudes N, D are the un-scaled versions
    (N_amp = \tilde{N}/Q_L, D_amp = \tilde{D}/Q_L).  The on-resonance ratio
    N_amp/D_amp = \tilde{N}/\tilde{D} is identical, but the frequency-resolved
    formula requires the Q_L-scaled forms to obtain the correct bandwidth.

    Args:
        q_m:           Effective magnetic quality factor.
        q_l:           Loaded cavity quality factor.
        coupling_beta: Coupling coefficient β.
        u:             Normalised detuning array 2 Q_L δ.

    Returns:
        Power gain array, same shape as *u*.
    """
    # Q_L-normalised numerator and denominator amplitudes
    n_tilde = (coupling_beta - 1.0) / (1.0 + coupling_beta) + q_l / q_m
    d_tilde = 1.0 - q_l / q_m   # = (q_m − q_l)/q_m  > 0 when q_m > q_l
    return (n_tilde**2 + u**2) / (d_tilde**2 + u**2)


# ── Public API ────────────────────────────────────────────────────

def compute_gain_curve(  # noqa: PLR0913
    cavity_frequency_hz: float,
    q_m: float,
    q_l: float,
    coupling_beta: float,
    *,
    freq_grid_hz: NDArray[np.float64] | None = None,
    n_points: int = 1001,
    span_bandwidth_factor: float = 20.0,
    probe_power_w: float = 0.0,
    n_effective: float | None = None,
    single_spin_coupling_hz: float | None = None,
    spin_linewidth_hz: float | None = None,
) -> GainCurveResult:
    r"""Compute the frequency-resolved maser gain curve G(f) = |S₁₁(f)|².

    The gain curve is evaluated on *freq_grid_hz* (or an auto-generated grid
    centred on *cavity_frequency_hz* spanning ``span_bandwidth_factor`` times
    the analytical 3 dB bandwidth).

    Probe-power saturation
    ~~~~~~~~~~~~~~~~~~~~~~
    If ``probe_power_w > 0``, the effective magnetic Q increases:

    .. math::

        Q_m^\text{eff} = Q_m \cdot \left(1 + \frac{P_\text{in}}{P_\text{sat}}\right)

    and the curve is computed at the saturated operating point.  When
    saturation parameters are provided they are also used to populate
    ``saturation_power_w`` in the result at zero probe power.

    Args:
        cavity_frequency_hz:     Cavity resonance f_c (Hz).
        q_m:                     Magnetic quality factor (small-signal).
        q_l:                     Loaded cavity quality factor Q_L.
        coupling_beta:           Coupling coefficient β (1.0 = critical coupling
                                 in the Wang convention).
        freq_grid_hz:            Optional explicit frequency grid (Hz).
                                 Overrides auto-generation.
        n_points:                Points in auto-generated frequency grid.
        span_bandwidth_factor:   Auto-grid half-span = factor × B_analytical / 2.
        probe_power_w:           Probe input power (W).  0 → small-signal.
        n_effective:             Effective spin count N_eff.  Required when
                                 ``probe_power_w > 0``.
        single_spin_coupling_hz: Single-spin vacuum coupling g₀ (Hz).  Required
                                 when ``probe_power_w > 0``.
        spin_linewidth_hz:       Spin linewidth FWHM γ_⊥ (Hz).  Required when
                                 ``probe_power_w > 0``.

    Returns:
        :class:`GainCurveResult` with the gain array and all derived metrics.

    Raises:
        ValueError: If *q_m* ≤ *q_l* (device is at or above oscillation
                    threshold — the gain formula is singular there).
        ValueError: If ``probe_power_w > 0`` but saturation parameters are
                    absent.
        ValueError: If probe-power saturation drives Q_m_eff ≤ Q_L (the device
                    enters the oscillating regime under the applied probe).
    """
    if q_m <= q_l:
        raise ValueError(
            f"q_m={q_m:.6g} ≤ q_l={q_l:.6g}: device is at or above oscillation "
            "threshold.  Gain curve is undefined in the oscillating regime."
        )
    if probe_power_w > 0 and (
        n_effective is None
        or single_spin_coupling_hz is None
        or spin_linewidth_hz is None
    ):
        raise ValueError(
            "probe_power_w > 0 requires n_effective, single_spin_coupling_hz, "
            "and spin_linewidth_hz to be provided."
        )

    # ── Saturation power (computed whenever parameters are available) ──
    p_sat_w: float = float("nan")
    p_sat_dbm: float = float("nan")

    _has_sat_params = (
        n_effective is not None
        and single_spin_coupling_hz is not None
        and spin_linewidth_hz is not None
    )
    if _has_sat_params:
        p_sat_w = compute_saturation_power(
            cavity_frequency_hz=cavity_frequency_hz,
            q_l=q_l,
            coupling_beta=coupling_beta,
            n_effective=n_effective,  # type: ignore[arg-type]
            single_spin_coupling_hz=single_spin_coupling_hz,  # type: ignore[arg-type]
            spin_linewidth_hz=spin_linewidth_hz,  # type: ignore[arg-type]
        )
        if math.isfinite(p_sat_w) and p_sat_w > 0:
            p_sat_dbm = 10.0 * math.log10(p_sat_w / 1e-3)

    # ── Apply probe-power saturation to Q_m ───────────────────────
    q_m_eff = q_m
    if probe_power_w > 0 and math.isfinite(p_sat_w) and p_sat_w > 0:
        q_m_eff = q_m * (1.0 + probe_power_w / p_sat_w)

    # ── Analytical bandwidth (used for auto grid) ─────────────────
    bw_analytical = bandwidth_analytical(cavity_frequency_hz, q_m_eff, q_l)

    # ── Frequency grid ────────────────────────────────────────────
    if freq_grid_hz is not None:
        freq = np.asarray(freq_grid_hz, dtype=np.float64)
    else:
        half_span = span_bandwidth_factor * bw_analytical / 2.0
        freq = np.linspace(
            cavity_frequency_hz - half_span,
            cavity_frequency_hz + half_span,
            n_points,
        )

    # ── Gain curve G(f) ──────────────────────────────────────────
    delta = (freq - cavity_frequency_hz) / cavity_frequency_hz  # (f − f_c)/f_c
    u = 2.0 * q_l * delta  # normalised detuning

    gain_linear = _s11_power_gain_at_detuning(q_m_eff, q_l, coupling_beta, u)

    # ── Peak metrics ──────────────────────────────────────────────
    peak_idx = int(np.argmax(gain_linear))
    peak_gain = float(gain_linear[peak_idx])
    peak_freq = float(freq[peak_idx])
    peak_db = 10.0 * math.log10(peak_gain) if peak_gain > 0 else float("nan")

    # ── 3 dB bandwidth (numerical, independent cross-check) ───────
    bw_3db = compute_bandwidth_3db(gain_linear, freq)

    # ── Gain–bandwidth product ────────────────────────────────────
    gbp = compute_gain_bandwidth_product(peak_gain, bw_3db)

    return GainCurveResult(
        frequencies_hz=freq,
        gain_linear=gain_linear,
        peak_gain_linear=peak_gain,
        peak_gain_db=peak_db,
        peak_frequency_hz=peak_freq,
        bandwidth_3db_hz=bw_3db,
        bandwidth_analytical_hz=bw_analytical,
        gain_bandwidth_hz=gbp,
        below_threshold=True,
        saturation_power_w=p_sat_w,
        saturation_power_dbm=p_sat_dbm,
        effective_q_m=q_m_eff,
    )


def compute_bandwidth_3db(
    gain_linear: NDArray[np.float64],
    freq_grid_hz: NDArray[np.float64],
) -> float:
    """Numerically extract the 3 dB bandwidth from a gain curve.

    The 3 dB threshold is defined as half the gain *above unity*:

        G_threshold = 1 + (G_max − 1) / 2

    This definition correctly handles the fact that the maser gain
    curve approaches 1 (passive reflection) far from resonance rather
    than 0 as for a conventional bandpass filter.

    Bandwidth is the total width of the frequency interval where
    G(f) ≥ G_threshold.  Zero crossings are located by linear
    interpolation between adjacent grid points.

    Args:
        gain_linear:   Power gain array G(f), shape ``(n_points,)``.
        freq_grid_hz:  Frequency grid (Hz), same shape as *gain_linear*.

    Returns:
        3 dB bandwidth in Hz.  Returns the full grid span if the gain
        is above threshold at every point (e.g. very narrow grid).
        Returns 0 if the peak gain is ≤ 1 (no amplification).
    """
    gain = np.asarray(gain_linear, dtype=np.float64)
    freq = np.asarray(freq_grid_hz, dtype=np.float64)

    if gain.size == 0:
        return 0.0

    g_max = float(np.max(gain))
    if g_max <= 1.0:
        return 0.0

    # Threshold: halfway between unity and the peak
    g_threshold = 1.0 + (g_max - 1.0) / 2.0

    above = gain - g_threshold  # positive where gain > threshold

    # Find all sign changes (zero crossings of *above*)
    crossings: list[float] = []
    for i in range(len(above) - 1):
        if above[i] * above[i + 1] <= 0.0:
            x0, x1 = float(freq[i]), float(freq[i + 1])
            y0, y1 = float(above[i]), float(above[i + 1])
            if y1 != y0:
                x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
            else:
                x_cross = (x0 + x1) * 0.5
            crossings.append(x_cross)

    if len(crossings) < 2:
        # Gain above threshold across the entire grid → return full span
        return float(freq[-1] - freq[0])

    # Bandwidth = distance between outermost crossings
    return float(crossings[-1] - crossings[0])


def bandwidth_analytical(
    cavity_frequency_hz: float,
    q_m: float,
    q_l: float,
) -> float:
    """Analytical parametric-narrowing bandwidth (Hz).

    .. math::

        B = f_c \\cdot \\frac{Q_m - Q_L}{Q_L \\cdot Q_m}

    Args:
        cavity_frequency_hz: Cavity resonance f_c (Hz).
        q_m:                 Magnetic quality factor.
        q_l:                 Loaded quality factor Q_L.

    Returns:
        Bandwidth B (Hz).  Zero when q_m ≤ q_l.
    """
    if q_m <= q_l or q_l <= 0.0 or q_m <= 0.0:
        return 0.0
    return cavity_frequency_hz * (q_m - q_l) / (q_l * q_m)


def compute_gain_bandwidth_product(
    peak_gain_linear: float,
    bandwidth_hz: float,
) -> float:
    r"""Gain-voltage × bandwidth product (Hz).

    Computes :math:`\sqrt{G_\text{max}} \times B`.

    For critically coupled maser (β = 1) this equals *f_c / Q_m* —
    the spin-gain linewidth — and is invariant with respect to how
    close Q_m is to Q_L.

    Args:
        peak_gain_linear: Peak power gain G_max (linear, ≥ 1).
        bandwidth_hz:     3 dB bandwidth B (Hz).

    Returns:
        √G_max × B in Hz.  Zero if either input is non-positive.
    """
    if peak_gain_linear <= 0.0 or bandwidth_hz <= 0.0:
        return 0.0
    return math.sqrt(peak_gain_linear) * bandwidth_hz


# ── Probe-power saturation ────────────────────────────────────────

def compute_saturation_power(
    cavity_frequency_hz: float,
    q_l: float,
    coupling_beta: float,
    n_effective: float,
    single_spin_coupling_hz: float,
    spin_linewidth_hz: float,
) -> float:
    r"""Input-referred saturation power P_sat (W).

    At input power P_sat the intracavity photon number equals the critical
    photon number n_crit.  Above this level the spin inversion is partially
    depleted and the gain is compressed.

    Derivation
    ~~~~~~~~~~

    Critical photon number (collective ensemble):

    .. math::

        n_\text{crit} = \frac{(\gamma_\perp / 2 g_0)^2}{N_\text{eff}}

    Output-port decay rate:

    .. math::

        \kappa_\text{ext} = \frac{\omega_c \beta}{Q_L (1+\beta)}

    Saturation power:

    .. math::

        P_\text{sat} = n_\text{crit} \cdot \kappa_\text{ext} \cdot \hbar \omega_c

    Args:
        cavity_frequency_hz:     Cavity resonance f_c (Hz).
        q_l:                     Loaded quality factor Q_L.
        coupling_beta:           Coupling coefficient β.
        n_effective:             Effective number of inverted spins N_eff.
        single_spin_coupling_hz: Single-spin vacuum coupling g₀ (Hz).
        spin_linewidth_hz:       Spin linewidth FWHM γ_⊥ (Hz).

    Returns:
        P_sat in Watts.  Zero if any parameter is non-positive.
    """
    if (
        n_effective <= 0.0
        or single_spin_coupling_hz <= 0.0
        or spin_linewidth_hz <= 0.0
        or q_l <= 0.0
        or coupling_beta <= 0.0
    ):
        return 0.0

    # Critical photon number (dimensionless)
    n_crit = (spin_linewidth_hz / (2.0 * single_spin_coupling_hz)) ** 2 / n_effective

    # Output-port decay rate κ_ext (rad/s)
    omega_c = 2.0 * math.pi * cavity_frequency_hz
    kappa_ext = omega_c * coupling_beta / (q_l * (1.0 + coupling_beta))

    return n_crit * kappa_ext * _HBAR * omega_c


# ── Wiring with SpectralMBResult ──────────────────────────────────

def gain_curve_from_mb_result(
    mb_result: SpectralMBResult,
    nv_config: NVConfig,
    maser_config: MaserConfig,
    cavity_config: CavityConfig,
    *,
    freq_grid_hz: NDArray[np.float64] | None = None,
    n_points: int = 1001,
    span_bandwidth_factor: float = 20.0,
) -> GainCurveResult:
    """Derive the gain curve implied by a spectral Maxwell–Bloch simulation.

    Computes the Wang 2024 magnetic quality factor Q_m from the NV / cavity
    configuration and the loaded Q_L (after Q-boost), then delegates to
    :func:`compute_gain_curve`.

    The magnetic Q_m comes from :func:`.amplifier.compute_magnetic_q` (Wang
    2024, Eq. 1), which is self-consistent with
    :func:`.amplifier.compute_amplifier_properties`.  The spectral solver's
    cooperativity field (which uses a different normalisation) is *not* used
    for the Q_m derivation.

    Args:
        mb_result:             Completed spectral MB simulation result.
        nv_config:             NV spin parameters.
        maser_config:          Cavity Q, frequency, and coupling coefficient.
        cavity_config:         Cavity mode volume and fill factor.
        freq_grid_hz:          Optional explicit frequency grid (Hz).
        n_points:              Points in auto-generated grid.
        span_bandwidth_factor: Auto-grid half-span in units of B_analytical.

    Returns:
        :class:`GainCurveResult` consistent with the simulation parameters.

    Raises:
        ValueError: If Q_m ≤ Q_L (high-cooperativity regime where the device
                    oscillates rather than amplifies in steady state).
    """
    # Avoid circular import: amplifier → cavity (fine), maser_gain_frequency → amplifier
    from .amplifier import compute_magnetic_q as _compute_q_m_wang

    q_m = _compute_q_m_wang(nv_config, cavity_config, maser_config)
    q_l = compute_effective_q(maser_config)

    # Cavity properties for saturation parameters
    cavity_props = compute_cavity_properties(maser_config, cavity_config)
    g0_hz = cavity_props.single_spin_coupling_hz

    # Spin linewidth FWHM from T₂* using Lorentzian relation γ_⊥=1/(π T₂*)
    spin_lw_hz = 1.0 / (math.pi * nv_config.t2_star_us * 1e-6)

    # Effective spin count (full gain budget = 1.0; spectral narrowing accounted
    # for by the spectral solver, not by de-rating N_eff here)
    n_eff = compute_n_effective(nv_config, cavity_config, gain_budget=1.0)

    return compute_gain_curve(
        cavity_frequency_hz=maser_config.cavity_frequency_ghz * 1e9,
        q_m=q_m,
        q_l=q_l,
        coupling_beta=maser_config.coupling_beta,
        freq_grid_hz=freq_grid_hz,
        n_points=n_points,
        span_bandwidth_factor=span_bandwidth_factor,
        probe_power_w=0.0,
        n_effective=n_eff,
        single_spin_coupling_hz=g0_hz,
        spin_linewidth_hz=spin_lw_hz,
    )
