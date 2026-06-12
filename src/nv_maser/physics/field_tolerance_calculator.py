"""
physics/field_tolerance_calculator.py — B₀ field strength and homogeneity
tolerance analysis for the single-sided magnet (R1 closure).

Two independent effects are modelled:

1. **B₀ strength sensitivity** — How SNR scales with the absolute field at
   the sweet spot.  Signal ∝ ω₀ × M₀ ∝ B₀², so a magnet that delivers
   only 45 mT instead of 50 mT yields (45/50)² = 0.81 of nominal SNR
   (−1.84 dB).

2. **Field inhomogeneity (ppm) effects** — Non-uniformity across the imaging
   volume causes two measurable penalties:

   a. **FID T2* dephasing** — spins at different B₀_local precess at slightly
      different Larmor frequencies, dephasing the free-induction decay.
      This matters for single-pulse (FID) acquisitions but is *refocused* by
      spin-echo / CPMG sequences—the primary acquisition mode for this device.

   b. **Signal spectral bandwidth** — the Larmor frequency spread of the
      sweet-spot volume equals Δν = γ̄ × ΔB₀.  This must fit within the
      maser gain bandwidth (≈ 49 kHz) for distortion-free amplification.

Closure of R1
─────────────
The Phase-2 milestone (arch doc §12.2) specifies B₀ = 50 ± 5 mT at 20 mm
depth and < 500 ppm over a 10 mm sphere.  This module quantifies:

* Within the ±5 mT tolerance band [45, 55] mT, SNR loss is bounded to
  −1.84 dB (worst case at 45 mT), requiring ≤ 53 % more signal averages
  — well within scan-time budget.

* At 500 ppm inhomogeneity and TE = 100 µs, FID amplitude loss ≈ 3 dB,
  but the device uses CPMG (T2 measurement), which refocuses static field
  inhomogeneity entirely, so this is not the operating constraint.

* The 500 ppm signal spectral bandwidth (≈ 1 064 Hz at 50 mT) is far
  below the maser gain bandwidth (49 kHz), providing > 46 kHz of margin.

R1 is confirmed closed at V1 specification provided the magnet assembly
delivers 45–55 mT at 20 mm depth and < 500 ppm over 10 mm diameter.

References
──────────
Hoult & Richards, "The signal-to-noise ratio of the NMR experiment",
J. Magn. Reson. 24, 71 (1976).

Handheld probe architecture document §8, §12.2, §13-R1.

ADR-020 (gain_bandwidth_match.py): maser loaded Q = 30 000, BW ≈ 49 kHz.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ── Physical constants ────────────────────────────────────────────
_GAMMA_BAR_P = 42.577e6   # proton γ/(2π) in Hz/T
_PI = math.pi


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Configuration                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class FieldToleranceConfig:
    """Configuration for the B₀ field tolerance sweep.

    All defaults are the V1 Phase-2 architecture targets from §12.2.
    """

    # ── Nominal operating point ───────────────────────────────────
    b0_nominal_t: float = 0.050
    """Target sweet-spot field strength (T). Default 50 mT."""

    # ── B₀ strength sweep ─────────────────────────────────────────
    b0_sweep_min_t: float = 0.030
    """Lower bound of B₀ sweep (T). Default 30 mT."""

    b0_sweep_max_t: float = 0.070
    """Upper bound of B₀ sweep (T). Default 70 mT."""

    n_b0_sweep: int = 21
    """Number of B₀ sweep points (inclusive of min and max)."""

    # ── Homogeneity sweep ─────────────────────────────────────────
    uniformity_sweep_min_ppm: float = 10.0
    """Lower bound of uniformity sweep (ppm)."""

    uniformity_sweep_max_ppm: float = 5_000.0
    """Upper bound of uniformity sweep (ppm)."""

    n_uniformity_sweep: int = 20
    """Number of uniformity sweep points."""

    # ── T2* / FID parameters ──────────────────────────────────────
    te_fid_us: float = 100.0
    """FID echo time for T2* dephasing calculation (µs). Default 100 µs."""

    t2_tissue_ms: float = 50.0
    """Representative tissue T2 (ms). Default 50 ms (brain white matter)."""

    # ── Maser / receiver bandwidth ────────────────────────────────
    maser_bandwidth_hz: float = 49_000.0
    """Maser gain FWHM bandwidth (Hz). From ADR-020: f₀/Q_loaded ≈ 49 kHz."""

    # ── V1 milestone acceptance criteria (arch doc §12.2) ─────────
    v1_b0_tolerance_t: float = 0.005
    """Symmetric B₀ tolerance around nominal (T). Default ±5 mT."""

    v1_uniformity_ppm: float = 500.0
    """Maximum peak-to-peak field uniformity over 10 mm sphere (ppm)."""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Result dataclasses                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class B0SensitivityPoint:
    """SNR penalty at a given absolute sweet-spot field strength.

    Physical model
    ──────────────
    The NMR signal amplitude is proportional to both:
    * the equilibrium magnetisation M₀ ∝ B₀  (Boltzmann polarisation)
    * the detection frequency ω₀ = γ B₀       (Faraday induction EMF)

    Assuming thermal-noise-dominated detection (coil/body noise fixed):

        SNR ∝ ω₀ × M₀ ∝ B₀²

    The SNR factor is therefore (B₀_actual / B₀_nominal)², and the
    corresponding loss in dB (using the amplitude-ratio convention,
    20 log₁₀) is 40 log₁₀(B₀_nominal / B₀_actual).

    Note: both M₀ and ω₀ contribute one power of B₀ each, so the
    combined sensitivity exponent is 2, not the Hoult–Richards value
    of 7/4 which applies to optimally-scaled coils at low field.  This
    module uses the simpler B₀² model consistent with the fixed-coil
    geometry of the handheld probe.
    """

    b0_tesla: float
    """Actual sweet-spot field strength (T)."""

    b0_mT: float
    """Actual sweet-spot field strength (mT)."""

    b0_deviation_pct: float
    """Signed deviation from nominal: 100 × (B₀ − B₀_nom) / B₀_nom (%)."""

    polarization_factor: float
    """M₀(B₀_actual) / M₀(B₀_nominal) = B₀_actual / B₀_nominal."""

    signal_frequency_factor: float
    """ω₀(B₀_actual) / ω₀(B₀_nominal) = B₀_actual / B₀_nominal."""

    snr_factor: float
    """SNR_actual / SNR_nominal = (B₀_actual / B₀_nominal)²."""

    snr_loss_db: float
    """SNR loss relative to nominal (dB, positive = reduced SNR).

    snr_loss_db = −20 log₁₀(snr_factor) = 40 log₁₀(B₀_nom / B₀_actual).
    """

    larmor_frequency_hz: float
    """Proton Larmor frequency at B₀_actual: γ̄ × B₀_actual (Hz)."""


@dataclass(frozen=True)
class HomogeneityPoint:
    """Signal quality penalty at a given field uniformity within the voxel.

    Physical model
    ──────────────
    The peak-to-peak field variation ΔB₀ across the sweet-spot volume
    maps to a Larmor frequency spread:

        Δν = γ̄ × ΔB₀  [Hz]

    This produces two measurable effects:

    1. **FID T2* dephasing** — For a rectangular field distribution the
       characteristic dephasing time is T2*_inhom = 1 / (π × Δν).  The
       effective T2* combines with tissue T2:
           1/T2*_eff = 1/T2 + 1/T2*_inhom
       The FID amplitude at echo time TE is reduced by the factor:
           exp(−TE / T2*_eff) / exp(−TE / T2) = exp(−TE / T2*_inhom)
       CPMG sequences refocus static inhomogeneity (the 180° pulses
       re-phase dephased spins), so this loss is relevant only for
       single-pulse (FID) acquisitions.

    2. **Spectral bandwidth** — The NMR signal occupies a passband of
       width Δν around the Larmor carrier.  For amplification without
       distortion, this must satisfy Δν ≤ BW_maser / 2.
    """

    uniformity_ppm: float
    """Peak-to-peak field variation over the evaluation volume (ppm)."""

    delta_b0_mt: float
    """Field spread ΔB₀ in mT: uniformity_ppm × 1e−6 × B₀_nominal × 1e3."""

    delta_frequency_hz: float
    """Larmor frequency spread: γ̄ × ΔB₀ (Hz)."""

    t2star_inhom_ms: float
    """T2* contribution from inhomogeneity alone: 1/(π × Δν) (ms).

    Convention: T2*_inhom = 1/(π × δν) where δν = γ̄ × ΔB₀ is the
    full-width Larmor spread (rectangular distribution).
    """

    t2star_eff_ms: float
    """Combined T2*: 1/T2*_eff = 1/T2_tissue + 1/T2*_inhom (ms)."""

    snr_loss_fid_factor: float
    """FID amplitude ratio: exp(−TE / T2*_eff) / exp(−TE / T2_tissue).

    Equals exp(−TE_µs × 1e−3 / T2*_inhom_ms).  Unity when inhomogeneity
    is zero; decreases toward 0 as inhomogeneity grows.
    """

    snr_loss_fid_db: float
    """FID SNR loss in dB (positive = amplitude reduced).

    snr_loss_fid_db = −20 log₁₀(snr_loss_fid_factor).
    """

    within_maser_bandwidth: bool
    """True if spectral spread Δν ≤ BW_maser / 2 (signal fits in passband)."""


@dataclass(frozen=True)
class FieldToleranceResult:
    """Complete B₀ field tolerance sweep — strength and homogeneity effects.

    Produced by :func:`compute_field_tolerance`.
    """

    config: FieldToleranceConfig
    """Input configuration used to produce this result."""

    b0_sensitivity: tuple[B0SensitivityPoint, ...]
    """SNR vs B₀ sweep from b0_sweep_min_t to b0_sweep_max_t."""

    homogeneity: tuple[HomogeneityPoint, ...]
    """FID T2* and spectral-BW vs uniformity sweep."""

    # ── Key threshold values (analytically derived) ───────────────

    b0_3db_loss_t: float
    """B₀ at which SNR loss = 3 dB: B₀_nom × 10^(−3/40) (T)."""

    b0_1db_loss_t: float
    """B₀ at which SNR loss = 1 dB: B₀_nom × 10^(−1/40) (T)."""

    uniformity_3db_fid_loss_ppm: float
    """Uniformity for 3 dB FID amplitude loss at the configured TE (ppm)."""

    uniformity_1db_fid_loss_ppm: float
    """Uniformity for 1 dB FID amplitude loss at the configured TE (ppm)."""

    uniformity_maser_limit_ppm: float
    """Maximum uniformity before spectral spread exceeds BW_maser / 2 (ppm)."""

    # ── V1 milestone assessment ───────────────────────────────────

    v1_snr_loss_at_b0_min_db: float
    """SNR loss at the lowest V1-acceptable B₀ (b0_nominal − v1_b0_tolerance).

    This is the *worst-case* strength penalty within the V1 tolerance band.
    """

    v1_spectral_bandwidth_at_spec_hz: float
    """NMR signal bandwidth at v1_uniformity_ppm (Hz)."""

    v1_maser_bandwidth_margin_hz: float
    """Margin between maser half-BW and V1 spectral bandwidth (Hz).

    Positive means signal fits inside maser passband.
    """

    r1_risk_closed: bool
    """True when V1 spec is achievable without SNR compromise.

    Condition: both of the following hold:
    * v1_snr_loss_at_b0_min_db < 3 dB  (strength loss bounded)
    * v1_maser_bandwidth_margin_hz > 0  (signal fits in maser passband)
    """


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Core calculation functions                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

def compute_b0_sensitivity_point(
    b0_nominal_t: float,
    b0_actual_t: float,
) -> B0SensitivityPoint:
    """Compute SNR penalty at a given sweet-spot field strength.

    Args:
        b0_nominal_t: Design-target B₀ (T).  Typically 0.050 T.
        b0_actual_t:  Actual sweet-spot field delivered by the magnet (T).

    Returns:
        :class:`B0SensitivityPoint` with SNR factor and loss.

    Raises:
        ValueError: if ``b0_nominal_t`` or ``b0_actual_t`` are ≤ 0.
    """
    if b0_nominal_t <= 0:
        raise ValueError(f"b0_nominal_t must be positive, got {b0_nominal_t}")
    if b0_actual_t <= 0:
        raise ValueError(f"b0_actual_t must be positive, got {b0_actual_t}")

    ratio = b0_actual_t / b0_nominal_t
    snr_factor = ratio ** 2  # polarisation × signal_frequency ∝ B₀²
    snr_loss_db = -20.0 * math.log10(max(snr_factor, 1e-30))

    return B0SensitivityPoint(
        b0_tesla=b0_actual_t,
        b0_mT=b0_actual_t * 1e3,
        b0_deviation_pct=100.0 * (b0_actual_t - b0_nominal_t) / b0_nominal_t,
        polarization_factor=ratio,
        signal_frequency_factor=ratio,
        snr_factor=snr_factor,
        snr_loss_db=snr_loss_db,
        larmor_frequency_hz=_GAMMA_BAR_P * b0_actual_t,
    )


def compute_homogeneity_point(
    config: FieldToleranceConfig,
    uniformity_ppm: float,
) -> HomogeneityPoint:
    """Compute FID T2* dephasing and spectral-bandwidth penalty.

    Args:
        config:          Field tolerance configuration.
        uniformity_ppm:  Peak-to-peak B₀ variation over the sweet-spot
                         evaluation volume (ppm).

    Returns:
        :class:`HomogeneityPoint` with T2*, FID loss, and BW assessment.

    Raises:
        ValueError: if ``uniformity_ppm`` < 0.
    """
    if uniformity_ppm < 0:
        raise ValueError(
            f"uniformity_ppm must be ≥ 0, got {uniformity_ppm}"
        )

    # ΔB₀ (T) from ppm value
    delta_b0_t = config.b0_nominal_t * uniformity_ppm * 1e-6
    delta_b0_mt = delta_b0_t * 1e3

    # Larmor frequency spread (Hz) — rectangular distribution full-width
    # Δν = γ̄ × ΔB₀
    delta_frequency_hz = _GAMMA_BAR_P * delta_b0_t

    # T2*_inhom = 1/(π × Δν) using the rectangular-distribution convention
    # If Δν = 0 (perfect homogeneity), T2*_inhom → ∞
    if delta_frequency_hz > 0:
        t2star_inhom_ms = 1.0 / (_PI * delta_frequency_hz) * 1e3  # → ms
    else:
        t2star_inhom_ms = float("inf")

    # Combined T2*: 1/T2*_eff = 1/T2_tissue + 1/T2*_inhom
    inv_t2 = 1.0 / config.t2_tissue_ms
    if math.isfinite(t2star_inhom_ms):
        inv_t2star_inhom = 1.0 / t2star_inhom_ms
    else:
        inv_t2star_inhom = 0.0
    inv_t2star_eff = inv_t2 + inv_t2star_inhom
    t2star_eff_ms = 1.0 / inv_t2star_eff

    # FID amplitude at TE, normalised to ideal (no inhomogeneity) FID
    # = exp(-TE / T2*_eff) / exp(-TE / T2) = exp(-TE × 1/T2*_inhom)
    te_ms = config.te_fid_us * 1e-3  # µs → ms
    if math.isfinite(t2star_inhom_ms) and t2star_inhom_ms > 0:
        snr_loss_fid_factor = math.exp(-te_ms / t2star_inhom_ms)
    else:
        snr_loss_fid_factor = 1.0  # zero inhomogeneity → no loss

    snr_loss_fid_factor = max(snr_loss_fid_factor, 1e-30)
    snr_loss_fid_db = -20.0 * math.log10(snr_loss_fid_factor)

    # Spectral bandwidth check vs maser gain bandwidth
    within_maser_bandwidth = delta_frequency_hz <= config.maser_bandwidth_hz / 2

    return HomogeneityPoint(
        uniformity_ppm=uniformity_ppm,
        delta_b0_mt=delta_b0_mt,
        delta_frequency_hz=delta_frequency_hz,
        t2star_inhom_ms=t2star_inhom_ms,
        t2star_eff_ms=t2star_eff_ms,
        snr_loss_fid_factor=snr_loss_fid_factor,
        snr_loss_fid_db=snr_loss_fid_db,
        within_maser_bandwidth=within_maser_bandwidth,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Sweep and summary functions                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

def sweep_b0_sensitivity(
    config: FieldToleranceConfig,
) -> tuple[B0SensitivityPoint, ...]:
    """Sweep B₀ from min to max, computing SNR sensitivity at each point.

    Args:
        config: Field tolerance configuration.

    Returns:
        Tuple of :class:`B0SensitivityPoint` in ascending B₀ order.
    """
    if config.n_b0_sweep < 2:
        raise ValueError("n_b0_sweep must be ≥ 2")
    if config.b0_sweep_min_t >= config.b0_sweep_max_t:
        raise ValueError("b0_sweep_min_t must be < b0_sweep_max_t")

    step = (
        (config.b0_sweep_max_t - config.b0_sweep_min_t)
        / (config.n_b0_sweep - 1)
    )
    return tuple(
        compute_b0_sensitivity_point(
            config.b0_nominal_t,
            config.b0_sweep_min_t + i * step,
        )
        for i in range(config.n_b0_sweep)
    )


def sweep_homogeneity(
    config: FieldToleranceConfig,
) -> tuple[HomogeneityPoint, ...]:
    """Sweep field uniformity from min to max ppm.

    Args:
        config: Field tolerance configuration.

    Returns:
        Tuple of :class:`HomogeneityPoint` in ascending uniformity order.
    """
    if config.n_uniformity_sweep < 2:
        raise ValueError("n_uniformity_sweep must be ≥ 2")
    if config.uniformity_sweep_min_ppm >= config.uniformity_sweep_max_ppm:
        raise ValueError(
            "uniformity_sweep_min_ppm must be < uniformity_sweep_max_ppm"
        )

    step = (
        (config.uniformity_sweep_max_ppm - config.uniformity_sweep_min_ppm)
        / (config.n_uniformity_sweep - 1)
    )
    return tuple(
        compute_homogeneity_point(
            config,
            config.uniformity_sweep_min_ppm + i * step,
        )
        for i in range(config.n_uniformity_sweep)
    )


def _b0_for_snr_loss_db(b0_nominal_t: float, loss_db: float) -> float:
    """Return the B₀ at which SNR loss equals ``loss_db`` dB.

    Derived analytically from SNR ∝ B₀²:
        snr_factor = (B₀/B₀_nom)² = 10^(−loss_db/20)
        B₀ = B₀_nom × 10^(−loss_db/40)
    """
    return b0_nominal_t * 10.0 ** (-loss_db / 40.0)


def _uniformity_for_fid_loss_db(
    config: FieldToleranceConfig,
    loss_db: float,
) -> float:
    """Return uniformity (ppm) at which FID loss equals ``loss_db`` dB.

    Derived from snr_loss_fid_factor = exp(−TE / T2*_inhom) = 10^(−loss_db/20):
        TE / T2*_inhom = loss_db × ln(10) / 20
        T2*_inhom = TE / (loss_db × ln(10) / 20)  [ms]
        Δν = 1 / (π × T2*_inhom)               [kHz if ms used]
        ΔB₀ = Δν / γ̄
        ppm = ΔB₀ / B₀_nom × 1e6

    Returns infinity if TE = 0 (no dephasing at zero echo time).
    """
    if config.te_fid_us <= 0 or loss_db <= 0:
        return float("inf")

    te_ms = config.te_fid_us * 1e-3
    # From exp(−TE / T2*_inhom) = 10^(−loss/20):
    t2star_inhom_ms = te_ms / (loss_db * math.log(10) / 20.0)
    delta_freq_hz = 1.0 / (_PI * t2star_inhom_ms * 1e-3)  # T2* ms → s
    delta_b0_t = delta_freq_hz / _GAMMA_BAR_P
    return (delta_b0_t / config.b0_nominal_t) * 1e6


def _maser_limit_ppm(config: FieldToleranceConfig) -> float:
    """Return uniformity (ppm) at which Δν = BW_maser / 2.

    Above this limit the NMR signal spectral spread exceeds the one-sided
    maser gain bandwidth and signal power is partially lost.
    """
    delta_freq_limit_hz = config.maser_bandwidth_hz / 2.0
    delta_b0_limit_t = delta_freq_limit_hz / _GAMMA_BAR_P
    return (delta_b0_limit_t / config.b0_nominal_t) * 1e6


def compute_field_tolerance(
    config: FieldToleranceConfig | None = None,
) -> FieldToleranceResult:
    """Compute the complete B₀ field tolerance sweep.

    Performs both the B₀ strength sweep and the field uniformity sweep,
    then derives analytical threshold values and the V1 R1 closure verdict.

    Args:
        config: Field tolerance configuration.  Uses default V1 parameters
                if not supplied.

    Returns:
        :class:`FieldToleranceResult` with sweeps and summary metrics.
    """
    if config is None:
        config = FieldToleranceConfig()

    b0_sensitivity = sweep_b0_sensitivity(config)
    homogeneity = sweep_homogeneity(config)

    # ── Analytical threshold values ────────────────────────────────
    b0_3db = _b0_for_snr_loss_db(config.b0_nominal_t, 3.0)
    b0_1db = _b0_for_snr_loss_db(config.b0_nominal_t, 1.0)

    unif_3db = _uniformity_for_fid_loss_db(config, 3.0)
    unif_1db = _uniformity_for_fid_loss_db(config, 1.0)
    unif_maser = _maser_limit_ppm(config)

    # ── V1 milestone worst-case assessment ────────────────────────
    b0_min_v1 = config.b0_nominal_t - config.v1_b0_tolerance_t
    worst_snr_pt = compute_b0_sensitivity_point(config.b0_nominal_t, b0_min_v1)
    v1_snr_loss_db = worst_snr_pt.snr_loss_db

    v1_spec_bw_hz = compute_homogeneity_point(
        config, config.v1_uniformity_ppm
    ).delta_frequency_hz
    v1_maser_margin_hz = config.maser_bandwidth_hz / 2.0 - v1_spec_bw_hz

    r1_risk_closed = (v1_snr_loss_db < 3.0) and (v1_maser_margin_hz > 0.0)

    return FieldToleranceResult(
        config=config,
        b0_sensitivity=b0_sensitivity,
        homogeneity=homogeneity,
        b0_3db_loss_t=b0_3db,
        b0_1db_loss_t=b0_1db,
        uniformity_3db_fid_loss_ppm=unif_3db,
        uniformity_1db_fid_loss_ppm=unif_1db,
        uniformity_maser_limit_ppm=unif_maser,
        v1_snr_loss_at_b0_min_db=v1_snr_loss_db,
        v1_spectral_bandwidth_at_spec_hz=v1_spec_bw_hz,
        v1_maser_bandwidth_margin_hz=v1_maser_margin_hz,
        r1_risk_closed=r1_risk_closed,
    )
