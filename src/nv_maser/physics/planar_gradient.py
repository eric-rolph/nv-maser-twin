"""
Planar gradient coil model for lateral spatial encoding.

The handheld probe uses flat (planar) gradient coils mounted on the face
of the single-sided magnet to phase/frequency-encode the two lateral
dimensions (x, y) for 2D imaging.

Physical picture
────────────────
Two orthogonal gradient channels (Gx, Gy) use figure-8 (Golay-style)
coils.  Each channel consists of two rectangular or circular loop halves
carrying current in opposite directions to produce ∂Bz/∂x or ∂Bz/∂y.
The gradient field is evaluated on the imaging plane at the sweet-spot
depth.

For a figure-8 coil of loop radius *a* with loop centres offset ±d from
the coil axis (total coil width 2d), the gradient efficiency referred to
the centre is:

    η = μ₀ N × f_golay(a, d)   [T m⁻¹ A⁻¹]

where the Golay geometry factor is:

    f_golay(a, d) = 3 μ₀ a² d / (a² + d²)^(5/2)

In the default configuration (a = d = R/2 for outer radius R) this
simplifies to η ≈ 0.63 μ₀ N / R².

Gradient pulse timing
─────────────────────
Trapezoidal waveforms:

    rise phase  t ∈ [0, t_r)            G(t) = G_flat × t / t_r
    flat phase  t ∈ [t_r, t_r + t_f]    G(t) = G_flat
    fall phase  t ∈ (t_r + t_f, 2t_r + t_f]  G(t) = G_flat × (1 − …)

k-space encoding
────────────────
The spin phase at position r under gradient G(t) is:

    φ(r, t) = γ r ∫₀ᵗ G(τ) dτ

The k-space coordinate:

    k(t) = (γ / 2π) ∫₀ᵗ G(τ) dτ       [m⁻¹]

Phase-encoding: each line uses the same gradient waveform but a
different pre-phasing amplitude G_PE ∈ [−G_max, +G_max] covering FOV.

References
──────────
Turner & Bowley, "Passive screening of switched magnetic field
gradients", J. Phys. E 19, 876 (1986).

Vlaardingerbroek & den Boer, "Magnetic Resonance Imaging", Springer
(2003), §7.2–7.4.

Handheld probe architecture document, §6.2 and §9.2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

# ── Physical constants ────────────────────────────────────────────
_MU0: float = 4.0 * math.pi * 1e-7   # Vacuum permeability (T·m/A)
_GAMMA_P: float = 2.0 * math.pi * 42.577e6  # Proton gyromagnetic ratio (rad/s/T)

# Default probe-level gradient limits (from architecture doc §6.2)
_DEFAULT_MAX_GRADIENT_T_PER_M: float = 0.050      # 50 mT/m
_DEFAULT_MAX_SLEW_RATE: float = 150.0              # T/m/s
_DEFAULT_MAX_CURRENT_A: float = 10.0               # A per channel


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Data classes                                                    ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class GradientCoilSpec:
    """Specification for one planar gradient coil channel.

    Models a figure-8 (Golay-style) coil lying in the xy-plane.

    Attributes
    ----------
    loop_radius_mm:
        Radius of each loop half (mm).  The two halves are centred at
        x = ±offset_mm from the coil axis.
    offset_mm:
        Centre-to-centre half-separation of the two loop halves (mm).
        Optimal gradient uniformity near offset ≈ loop_radius / √2.
    n_turns:
        Number of turns per loop half.
    wire_diameter_mm:
        Copper wire diameter (mm), used for resistance calculation.
    resistivity:
        Copper resistivity (Ω·m).  Default 1.7×10⁻⁸ at 20 °C.
    axis:
        Which spatial gradient this coil produces: ``"x"`` or ``"y"``.
    max_current_a:
        Peak current rating (A).
    """

    loop_radius_mm: float = 20.0
    offset_mm: float = 14.14     # ≈ 20/√2 mm — optimal for uniformity
    n_turns: int = 5
    wire_diameter_mm: float = 1.0
    resistivity: float = 1.72e-8  # Ω·m, copper at 20 °C
    axis: str = "x"
    max_current_a: float = _DEFAULT_MAX_CURRENT_A

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got '{self.axis}'")
        if self.loop_radius_mm <= 0:
            raise ValueError("loop_radius_mm must be positive")
        if self.offset_mm <= 0:
            raise ValueError("offset_mm must be positive")
        if self.n_turns < 1:
            raise ValueError("n_turns must be ≥ 1")


@dataclass(frozen=True)
class GradientWaveform:
    """Trapezoidal gradient waveform for one axis.

    Attributes
    ----------
    amplitude_t_per_m:
        Flat-top gradient amplitude (T/m).  May be negative (polarity flip).
    rise_time_us:
        Ramp duration from 0 to ``amplitude_t_per_m`` (µs).
    flat_time_us:
        Duration of the flat-top plateau (µs).
    delay_us:
        Start delay relative to the sequence trigger (µs).
    """

    amplitude_t_per_m: float = 0.0
    rise_time_us: float = 200.0
    flat_time_us: float = 500.0
    delay_us: float = 0.0

    @property
    def duration_us(self) -> float:
        """Total waveform duration including both ramps (µs)."""
        return 2.0 * self.rise_time_us + self.flat_time_us

    @property
    def slew_rate_t_per_m_per_s(self) -> float:
        """Slew rate = |ΔG / Δt| (T/m/s).  Zero if rise_time_us = 0."""
        if self.rise_time_us == 0.0:
            return 0.0 if self.amplitude_t_per_m == 0.0 else float("inf")
        return abs(self.amplitude_t_per_m) / (self.rise_time_us * 1e-6)

    @property
    def area_t_per_m_us(self) -> float:
        """Gradient-time area (T/m·µs) = k-space step × 2π/γ × 1e6."""
        return self.amplitude_t_per_m * (self.rise_time_us + self.flat_time_us)


@dataclass(frozen=True)
class GradientPulseResult:
    """Result of evaluating a gradient waveform.

    Attributes
    ----------
    k_position_per_m:
        k-space position at readout centre (m⁻¹).
    phase_rad_per_mm:
        Phase per mm of spatial offset at readout centre (rad/mm).
    power_w:
        Instantaneous electrical power during flat top (W).
    peak_current_a:
        Peak current in the coil (A).
    slew_rate_t_per_m_per_s:
        Achieved slew rate (T/m/s).
    meets_limits:
        True if both gradient amplitude and slew rate are within limits.
    """

    k_position_per_m: float
    phase_rad_per_mm: float
    power_w: float
    peak_current_a: float
    slew_rate_t_per_m_per_s: float
    meets_limits: bool


@dataclass(frozen=True)
class PhaseEncodeScheme:
    """Phase-encoding scheme for 2D imaging.

    Attributes
    ----------
    n_lines:
        Number of phase-encode lines (must be even).
    fov_m:
        Field of view in the phase-encode direction (m).
    k_step_per_m:
        k-space step between adjacent lines (m⁻¹).
    k_max_per_m:
        Maximum k-space extent; determines resolution.
    resolution_mm:
        Pixel size in phase-encode direction (mm).
    gradient_amplitudes_t_per_m:
        Array of gradient amplitudes for each phase-encode step (T/m).
        Shape: (n_lines,).  Symmetric about zero; step = 1/FOV.
    scan_time_s:
        Estimated total acquisition time (seconds).
    """

    n_lines: int
    fov_m: float
    k_step_per_m: float
    k_max_per_m: float
    resolution_mm: float
    gradient_amplitudes_t_per_m: NDArray
    scan_time_s: float


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Coil physics                                                    ║
# ╚══════════════════════════════════════════════════════════════════╝


def compute_gradient_efficiency(spec: GradientCoilSpec) -> float:
    """Compute gradient efficiency η [T/m/A] at the coil centre.

    Uses the Golay figure-8 formula for a circular loop pair:

        η = μ₀ N × 3 a² d / (a² + d²)^(5/2)

    where *a* = loop radius and *d* = offset (half-separation).

    Parameters
    ----------
    spec:
        Gradient coil specification.

    Returns
    -------
    float
        Gradient efficiency (T·m⁻¹·A⁻¹).
    """
    a = spec.loop_radius_mm * 1e-3   # m
    d = spec.offset_mm * 1e-3        # m
    n = spec.n_turns
    numerator = 3.0 * a**2 * d
    denominator = (a**2 + d**2) ** 2.5
    return _MU0 * n * numerator / denominator


def compute_coil_resistance(spec: GradientCoilSpec) -> float:
    """Estimate DC resistance of one gradient coil channel (Ω).

    Each figure-8 channel consists of 2 loop halves × n_turns turns.
    Wire length per turn ≈ 2π × loop_radius.  Total wire length:

        L_wire = 2 × n_turns × 2π × a

    R = ρ L / A   with A = π (d_wire/2)²

    Parameters
    ----------
    spec:
        Gradient coil specification.

    Returns
    -------
    float
        DC resistance (Ω).
    """
    a = spec.loop_radius_mm * 1e-3          # loop radius (m)
    r_wire = spec.wire_diameter_mm * 0.5e-3  # wire radius (m)
    wire_length = 2.0 * spec.n_turns * 2.0 * math.pi * a  # m
    cross_section = math.pi * r_wire**2                    # m²
    return spec.resistivity * wire_length / cross_section


def compute_power_dissipation(spec: GradientCoilSpec, current_a: float) -> float:
    """Instantaneous resistive power dissipation P = I² R (W).

    Parameters
    ----------
    spec:
        Gradient coil specification.
    current_a:
        Coil current (A).

    Returns
    -------
    float
        Power dissipated (W).
    """
    r = compute_coil_resistance(spec)
    return current_a**2 * r


def compute_inductance(spec: GradientCoilSpec) -> float:
    """Estimate self-inductance of one gradient coil channel (H).

    Approximation for N-turn circular coil using Nagaoka-style formula
    for a flat (pancake) coil:

        L ≈ μ₀ N² a / 2 × [ln(8a/w) − 2]

    where w = wire diameter (Neumann formula for single loop) ×
    geometry factor 2 (figure-8 doubles the inductance).

    Parameters
    ----------
    spec:
        Gradient coil specification.

    Returns
    -------
    float
        Self-inductance (H).
    """
    a = spec.loop_radius_mm * 1e-3
    w = spec.wire_diameter_mm * 1e-3
    # Single-loop inductance (Lorenz approximation for thin wire)
    l_single = _MU0 * a * (math.log(8.0 * a / w) - 2.0)
    # N-turn figure-8 (both halves in series, mutual cancels largely)
    return l_single * spec.n_turns**2


def compute_max_gradient(spec: GradientCoilSpec) -> float:
    """Maximum achievable gradient at rated current (T/m).

    Parameters
    ----------
    spec:
        Gradient coil specification.

    Returns
    -------
    float
        Peak gradient amplitude (T/m).
    """
    return compute_gradient_efficiency(spec) * spec.max_current_a


def current_for_gradient(spec: GradientCoilSpec, gradient_t_per_m: float) -> float:
    """Required current to achieve a given gradient amplitude (A).

    Parameters
    ----------
    spec:
        Gradient coil specification.
    gradient_t_per_m:
        Desired gradient amplitude (T/m).

    Returns
    -------
    float
        Required current (A).

    Raises
    ------
    ValueError
        If the required current exceeds the coil's rated maximum.
    """
    eta = compute_gradient_efficiency(spec)
    if eta == 0.0:
        raise ValueError("Gradient efficiency is zero; cannot compute current.")
    current = abs(gradient_t_per_m) / eta
    if current > spec.max_current_a:
        raise ValueError(
            f"Required current {current:.2f} A exceeds max {spec.max_current_a} A"
            f" for gradient {gradient_t_per_m*1e3:.1f} mT/m."
        )
    return math.copysign(current, gradient_t_per_m)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  k-space and waveform functions                                  ║
# ╚══════════════════════════════════════════════════════════════════╝


def compute_k_position(
    waveform: GradientWaveform,
    time_us: float,
) -> float:
    """Evaluate the k-space position at time *time_us* (µs).

    Integrates the trapezoidal waveform:
        k(t) = (γ/2π) ∫₀ᵗ G(τ) dτ   [m⁻¹]

    Parameters
    ----------
    waveform:
        Gradient waveform to integrate.
    time_us:
        Evaluation time relative to waveform start (µs).

    Returns
    -------
    float
        k-space position (m⁻¹).
    """
    t = time_us - waveform.delay_us
    if t <= 0.0:
        return 0.0

    t_r = waveform.rise_time_us
    t_f = waveform.flat_time_us
    g = waveform.amplitude_t_per_m

    # Compute integrated area (T/m · µs → T/m · s using 1e-6 factor)
    if t <= t_r:
        # Rising ramp: triangle
        area_us = 0.5 * g * t**2 / t_r
    elif t <= t_r + t_f:
        # Flat top
        area_us = g * (0.5 * t_r + (t - t_r))
    elif t <= 2.0 * t_r + t_f:
        # Falling ramp
        t_fall = t - t_r - t_f
        area_us = g * (0.5 * t_r + t_f + t_fall - 0.5 * t_fall**2 / t_r)
    else:
        # After waveform ends
        area_us = g * (t_r + t_f)

    return (_GAMMA_P / (2.0 * math.pi)) * area_us * 1e-6  # m⁻¹


def compute_k_trajectory(
    waveform: GradientWaveform,
    n_samples: int,
    *,
    readout_start_us: float | None = None,
    readout_duration_us: float = 500.0,
) -> tuple[NDArray, NDArray]:
    """Sample the k-space trajectory during a readout window.

    Parameters
    ----------
    waveform:
        Gradient waveform defining the frequency-encode axis.
    n_samples:
        Number of ADC samples.
    readout_start_us:
        Start of ADC readout (µs).  Defaults to end of ramp-up.
    readout_duration_us:
        Duration of ADC window (µs).

    Returns
    -------
    times_us:
        Sample times (µs), shape (n_samples,).
    k_values_per_m:
        k-space positions (m⁻¹), shape (n_samples,).
    """
    if readout_start_us is None:
        readout_start_us = waveform.delay_us + waveform.rise_time_us

    times = np.linspace(
        readout_start_us,
        readout_start_us + readout_duration_us,
        n_samples,
    )
    k_values = np.array([compute_k_position(waveform, float(t)) for t in times])
    return times, k_values


def evaluate_waveform(
    waveform: GradientWaveform,
    spec: GradientCoilSpec,
    *,
    max_gradient_t_per_m: float = _DEFAULT_MAX_GRADIENT_T_PER_M,
    max_slew_t_per_m_per_s: float = _DEFAULT_MAX_SLEW_RATE,
) -> GradientPulseResult:
    """Evaluate a gradient waveform against coil limits.

    Parameters
    ----------
    waveform:
        Trapezoidal gradient waveform.
    spec:
        Coil specification (for resistance and current computation).
    max_gradient_t_per_m:
        System gradient amplitude limit (T/m).
    max_slew_t_per_m_per_s:
        System slew rate limit (T/m/s).

    Returns
    -------
    GradientPulseResult
    """
    eta = compute_gradient_efficiency(spec)
    peak_current = abs(waveform.amplitude_t_per_m) / eta if eta > 0 else 0.0
    power = compute_power_dissipation(spec, peak_current)
    slew = waveform.slew_rate_t_per_m_per_s

    # k-position at readout centre (mid-flat-top)
    readout_centre_us = (
        waveform.delay_us
        + waveform.rise_time_us
        + waveform.flat_time_us / 2.0
    )
    k_pos = compute_k_position(waveform, readout_centre_us)

    meets = (
        abs(waveform.amplitude_t_per_m) <= max_gradient_t_per_m
        and peak_current <= spec.max_current_a
        and slew <= max_slew_t_per_m_per_s
    )

    return GradientPulseResult(
        k_position_per_m=k_pos,
        phase_rad_per_mm=2.0 * math.pi * k_pos * 1e-3,
        power_w=power,
        peak_current_a=peak_current,
        slew_rate_t_per_m_per_s=slew,
        meets_limits=meets,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Phase-encoding scheme                                           ║
# ╚══════════════════════════════════════════════════════════════════╝


def build_phase_encode_scheme(
    spec: GradientCoilSpec,
    *,
    n_lines: int = 64,
    fov_m: float = 0.08,
    tr_ms: float = 100.0,
    rise_time_us: float = 200.0,
    flat_time_us: float = 300.0,
) -> PhaseEncodeScheme:
    """Compute the phase-encode gradient table for a 2D scan.

    Covers k-space from −k_max to +k_max with uniform step 1/FOV.

    Parameters
    ----------
    spec:
        Gradient coil for the phase-encode axis.
    n_lines:
        Number of k-space lines (even number).
    fov_m:
        Field of view (m).  Resolution = FOV / n_lines.
    tr_ms:
        Repetition time (ms).
    rise_time_us:
        Ramp time for each phase-encode gradient (µs).
    flat_time_us:
        Flat-top duration of each phase-encode gradient (µs).

    Returns
    -------
    PhaseEncodeScheme
    """
    if n_lines % 2 != 0:
        raise ValueError("n_lines must be even")

    k_step = 1.0 / fov_m                           # m⁻¹
    k_max = k_step * (n_lines / 2)                 # m⁻¹
    resolution_mm = fov_m / n_lines * 1e3          # mm

    # Required area per step (T/m·s)
    area_per_step_t_per_m_s = k_step * (2.0 * math.pi / _GAMMA_P)

    # Gradient amplitude for a given flat-top duration
    flat_s = flat_time_us * 1e-6
    rise_s = rise_time_us * 1e-6
    # area = G × (rise_s + flat_s)  for trapezoid up to flat-top end
    # We normalise so that the maximum step uses fraction ≤ max_gradient
    # For phase encoding the same trapezoid is played at scaled amplitudes.
    g_per_step = area_per_step_t_per_m_s / (rise_s + flat_s)
    g_max_needed = g_per_step * (n_lines / 2)

    # Warn if beyond coil capability, but don't raise — caller can adapt
    g_coil_max = compute_max_gradient(spec)
    if g_max_needed > g_coil_max * 1.05:
        # Accept up to 5% margin for rounding
        raise ValueError(
            f"Required peak gradient {g_max_needed*1e3:.1f} mT/m exceeds "
            f"coil max {g_coil_max*1e3:.1f} mT/m.  Reduce n_lines or increase FOV."
        )

    # Symmetric table: −N/2 … 0 … +N/2−1 steps
    line_indices = np.arange(-n_lines // 2, n_lines // 2)
    amplitudes = line_indices * g_per_step

    scan_time_s = n_lines * tr_ms * 1e-3

    return PhaseEncodeScheme(
        n_lines=n_lines,
        fov_m=fov_m,
        k_step_per_m=k_step,
        k_max_per_m=k_max,
        resolution_mm=resolution_mm,
        gradient_amplitudes_t_per_m=amplitudes,
        scan_time_s=scan_time_s,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Field map utilities                                             ║
# ╚══════════════════════════════════════════════════════════════════╝


def gradient_field_1d(
    spec: GradientCoilSpec,
    positions_mm: NDArray,
    *,
    current_a: float = 1.0,
    depth_mm: float = 15.0,
) -> NDArray:
    """Evaluate the Bz gradient field along the coil's sensitive axis.

    This uses the Golay figure-8 dipole approximation at a fixed depth.
    The field along axis *x* at depth *z = depth_mm* due to the figure-8 is:

        Bz(x, z) ≈ I × η × x          (linear near origin)

    with corrections for off-axis non-linearity:

        Bz(x, z) = η × I × x × (1 − 3x²/(2(a²+d²+z²)) + …)

    Parameters
    ----------
    spec:
        Gradient coil specification.
    positions_mm:
        Positions along the gradient axis (mm).  Shape (N,).
    current_a:
        Coil current (A).
    depth_mm:
        Depth below the coil plane (mm).

    Returns
    -------
    NDArray
        Bz field values (T) at each position.  Shape (N,).
    """
    x = np.asarray(positions_mm, dtype=float) * 1e-3  # m
    z = depth_mm * 1e-3                                # m
    a = spec.loop_radius_mm * 1e-3
    d = spec.offset_mm * 1e-3
    eta = compute_gradient_efficiency(spec)

    # Linear + third-order correction (Golay 1958 expansion)
    r2 = a**2 + d**2 + z**2
    bz = eta * current_a * x * (1.0 - 1.5 * x**2 / r2)
    return bz


def linearity_error(
    spec: GradientCoilSpec,
    positions_mm: NDArray,
    *,
    depth_mm: float = 15.0,
) -> NDArray:
    """Fractional linearity error relative to ideal linear gradient (%).

    Parameters
    ----------
    spec:
        Gradient coil specification.
    positions_mm:
        Positions (mm).
    depth_mm:
        Depth (mm).

    Returns
    -------
    NDArray
        Fractional error (%) at each position.
    """
    x = np.asarray(positions_mm, dtype=float) * 1e-3
    a = spec.loop_radius_mm * 1e-3
    d = spec.offset_mm * 1e-3
    z = depth_mm * 1e-3
    r2 = a**2 + d**2 + z**2
    # Third-order term only; higher terms neglected
    error_fraction = -1.5 * x**2 / r2
    return error_fraction * 100.0  # percent


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Parametric sweeps                                               ║
# ╚══════════════════════════════════════════════════════════════════╝


def sweep_efficiency_vs_radius(
    radii_mm: Sequence[float],
    *,
    offset_fraction: float = 0.707,
    n_turns: int = 5,
) -> NDArray:
    """Gradient efficiency as a function of loop radius.

    Parameters
    ----------
    radii_mm:
        Loop radii to sweep (mm).
    offset_fraction:
        offset = offset_fraction × loop_radius (default 1/√2 ≈ 0.707).
    n_turns:
        Turns per loop half.

    Returns
    -------
    NDArray
        Gradient efficiency (T/m/A) for each radius.  Shape (len(radii_mm),).
    """
    results = []
    for r in radii_mm:
        spec = GradientCoilSpec(
            loop_radius_mm=r,
            offset_mm=r * offset_fraction,
            n_turns=n_turns,
        )
        results.append(compute_gradient_efficiency(spec))
    return np.array(results)


def sweep_max_gradient_vs_current(
    currents_a: Sequence[float],
    spec: GradientCoilSpec,
) -> NDArray:
    """Peak gradient amplitude as a function of drive current.

    Parameters
    ----------
    currents_a:
        Drive currents (A).
    spec:
        Gradient coil specification.

    Returns
    -------
    NDArray
        Gradient amplitude (T/m) for each current.  Shape (len(currents_a),).
    """
    eta = compute_gradient_efficiency(spec)
    return np.array([eta * i for i in currents_a])


def sweep_k_max_vs_fov(
    fov_values_m: Sequence[float],
    n_lines: int = 64,
) -> NDArray:
    """Maximum k-space extent as a function of field of view.

    k_max = n_lines / (2 × FOV)

    Parameters
    ----------
    fov_values_m:
        FOV values (m).
    n_lines:
        Number of phase-encode lines.

    Returns
    -------
    NDArray
        k_max values (m⁻¹).
    """
    return np.array([n_lines / (2.0 * fov) for fov in fov_values_m])


def sweep_resolution_vs_n_lines(
    n_lines_array: Sequence[int],
    fov_m: float = 0.08,
) -> NDArray:
    """In-plane resolution as a function of number of phase-encode lines.

    Parameters
    ----------
    n_lines_array:
        Array of n_lines values.
    fov_m:
        Field of view (m).

    Returns
    -------
    NDArray
        Resolution (mm) for each n_lines.
    """
    return np.array([fov_m / nl * 1e3 for nl in n_lines_array])


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Default probe gradient array                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

#: Default Gx coil matching the probe face geometry (30 mm radius, 5 turns)
DEFAULT_GX = GradientCoilSpec(
    loop_radius_mm=15.0,
    offset_mm=10.6,    # 15/√2 ≈ 10.6 mm
    n_turns=5,
    axis="x",
)

#: Default Gy coil — same geometry, orthogonal axis
DEFAULT_GY = GradientCoilSpec(
    loop_radius_mm=15.0,
    offset_mm=10.6,
    n_turns=5,
    axis="y",
)
