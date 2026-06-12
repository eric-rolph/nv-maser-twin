"""
Phase-1 maser oscillation milestone validator.

Architecture doc §12.2 (handheld-maser-probe-architecture.md) defines the
first critical milestone for **Phase 1: Maser Module**:

    "Maser oscillation — Detectable stimulated emission at 1.47 GHz"

This module provides a formal digital-twin validation using the existing
physics stack:

    NVConfig + CavityConfig + MaserConfig
    ├── compute_amplifier_properties  →  Q_m vs Q_L (Wang 2024 threshold)
    ├── compute_full_threshold        →  Cooperativity C (Breeze formulation)
    └── compute_output_power          →  CW output power (W / dBm)

Milestone success criteria (digital twin)
──────────────────────────────────────────
1. **Oscillation threshold met**: the inverted NV spin ensemble provides
   sufficient gain to overcome cavity loss, assessed via *two* independent
   criteria:

   a. Wang 2024 magnetic-Q criterion:  Q_m ≤ Q_L (spin Q ≤ loaded cavity Q).
   b. Cooperativity criterion (Breeze):  C = 4g_N² / (κ γ⊥) > 1.

   Both must be satisfied for ``threshold_met = True``.

2. **Frequency criterion**: the cavity resonant frequency lies within
   ``frequency_tolerance_mhz`` of ``target_frequency_ghz`` (default 1.47 GHz,
   the NV zero-field |ms=0⟩ → |ms=−1⟩ splitting at 50 mT).

3. **Output power criterion**: CW steady-state output power above the
   detection floor ``min_output_power_dbm`` (default −100 dBm, well above
   the room-temperature thermal noise floor of ≈ −174 dBm/Hz).

The default NV/cavity/maser configuration produces:

    Q_m = 7 788  <  Q_L = 10 000  →  above_threshold = True  ✓
    Cooperativity C ≈ 2.57  >  1      →  masing = True         ✓
    Frequency = 1.47 GHz, deviation = 0 MHz  (< 50 MHz)        ✓
    Output power ≈ −69.5 dBm  >  −100 dBm                      ✓
    phase1_milestone_closed = True                              ✓

Reference
─────────
handheld-maser-probe-architecture.md §12.1–12.2.
Wang et al., Advanced Science (2024), PMC11425272.  Eq. 1 (Q_m), Eq. 4 (T_a).
Breeze et al., Nature 555, 493 (2018).  Cooperativity threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import CavityConfig, MaserConfig, NVConfig
from .amplifier import (
    AmplifierProperties,
    compute_amplifier_properties,
    compute_output_power,
)
from .cavity import compute_cavity_properties, compute_full_threshold

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Configuration                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class Phase1Config:
    """Acceptance thresholds for the Phase-1 maser oscillation milestone.

    All default values are chosen so that the reference NV diamond maser
    (T₂* = 1 µs, η_fill = 0.01, ρ_NV = 10¹⁷ /cm³, Q_L = 10 000)
    passes with comfortable margin.

    Args:
        target_frequency_ghz:
            Target oscillation frequency (GHz).  Default 1.47 GHz matches
            the NV zero-field splitting of 2.87 GHz shifted to the lower
            branch at 50 mT static field (ν = D − γ_e · B₀ ≈ 1.47 GHz).
        frequency_tolerance_mhz:
            Maximum allowed deviation from ``target_frequency_ghz`` (MHz).
            Default 50 MHz accommodates small variations in static field.
        min_output_power_dbm:
            Minimum detectable CW output power (dBm).  Default −100 dBm
            is 74 dB above the room-temperature 1 Hz noise floor
            (k_B T ≈ −174 dBm/Hz at 300 K), representing a realistic
            detection threshold for a narrowband receiver.
        gain_budget:
            Spectral overlap fraction between the NV inhomogeneous
            distribution and the cavity linewidth.  This is passed to
            ``compute_full_threshold`` as its ``gain_budget`` argument
            and determines the effective number of contributing spins.
            Default 0.5 matches ``MaserConfig.min_gain_budget``.
    """

    target_frequency_ghz: float = 1.47
    frequency_tolerance_mhz: float = 50.0
    min_output_power_dbm: float = -100.0
    gain_budget: float = 0.5

    def __post_init__(self) -> None:
        if self.target_frequency_ghz <= 0:
            raise ValueError("target_frequency_ghz must be positive")
        if self.frequency_tolerance_mhz < 0:
            raise ValueError("frequency_tolerance_mhz must be non-negative")
        if self.gain_budget <= 0 or self.gain_budget > 1:
            raise ValueError("gain_budget must be in (0, 1]")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Intermediate result types                                       ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class OscillationThresholdResult:
    """Combined Wang-2024 and cooperativity threshold assessment.

    Two independent physical approaches are evaluated:

    * **Wang 2024 (Q_m criterion)**: Q_m is the magnetic quality factor of
      the inverted spin ensemble.  Oscillation occurs when Q_m ≤ Q_L
      (spin gain ≥ total cavity loss).

    * **Breeze cooperativity**: C = 4g_N² / (κ γ⊥) where g_N is the
      collective spin–photon coupling, κ is the cavity decay rate, and
      γ⊥ is the spin decoherence rate.  Oscillation when C > 1.

    Both criteria give equivalent conditions at threshold but differ in
    how they parametrize gain and loss; agreement between the two
    increases confidence in the result.
    """

    # Wang 2024 Q-factor criterion
    magnetic_q: float
    """Q_m — spin gain quality factor (Wang 2024 Eq. 1)."""

    loaded_q: float
    """Q_L — loaded cavity quality factor."""

    q_ratio: float
    """Q_m / Q_L.  Values < 1 indicate oscillation (Q_m ≤ Q_L)."""

    above_threshold_wang: bool
    """True when Q_m ≤ Q_L (Wang 2024 oscillation condition)."""

    # Breeze cooperativity criterion
    cooperativity: float
    """Ensemble cooperativity C = 4g_N² / (κ γ⊥) (Breeze 2018)."""

    threshold_margin: float
    """C − 1.  Positive for masing; negative for sub-threshold."""

    masing: bool
    """True when C > 1 (Breeze cooperativity condition)."""

    # Spin temperature (diagnostic)
    spin_temperature_k: float
    """T_s — effective spin temperature of the inverted ensemble (K)."""


@dataclass(frozen=True)
class Phase1MilestoneResult:
    """Complete Phase-1 maser oscillation milestone assessment.

    Aggregates the three sub-criteria (threshold, frequency, power)
    and records the final pass/fail verdict.
    """

    # ── Oscillation threshold ──────────────────────────────────────
    oscillation: OscillationThresholdResult
    """Threshold status from Wang 2024 + cooperativity criterion."""

    # ── Frequency ─────────────────────────────────────────────────
    frequency_ghz: float
    """Actual cavity resonant frequency (GHz)."""

    target_frequency_ghz: float
    """Target oscillation frequency (GHz); default 1.47."""

    frequency_deviation_mhz: float
    """Absolute deviation |f_cavity − f_target| in MHz."""

    frequency_tolerance_mhz: float
    """Maximum allowed deviation (MHz) from ``Phase1Config``."""

    # ── Output power ──────────────────────────────────────────────
    output_power_w: float
    """CW steady-state output power (W); 0 if below threshold."""

    output_power_dbm: float
    """CW output power in dBm (−999 dBm if below threshold)."""

    min_output_power_dbm: float
    """Minimum detectable power threshold (dBm) from ``Phase1Config``."""

    # ── Pass / Fail ────────────────────────────────────────────────
    threshold_met: bool
    """True when both Wang-Q and cooperativity conditions are satisfied."""

    frequency_met: bool
    """True when |f_cavity − f_target| ≤ frequency_tolerance_mhz."""

    power_met: bool
    """True when output_power_dbm ≥ min_output_power_dbm."""

    phase1_milestone_closed: bool
    """True when all three sub-criteria pass (milestone achieved)."""

    closing_message: str
    """Human-readable summary of the milestone verdict."""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Computation helpers                                             ║
# ╚══════════════════════════════════════════════════════════════════╝


def _spin_linewidth_hz(nv_config: NVConfig) -> float:
    """Lorentzian FWHM linewidth from T₂* (Hz).

    Γ_eff = 1 / (π T₂*)

    This is the dephasing-limited spin linewidth used in the
    cooperativity calculation.  It represents the full width at half
    maximum of the NV spin resonance in the frequency domain.
    """
    return 1.0 / (math.pi * nv_config.t2_star_us * 1e-6)


def _evaluate_oscillation(
    nv_config: NVConfig,
    cavity_config: CavityConfig,
    maser_config: MaserConfig,
    gain_budget: float,
) -> OscillationThresholdResult:
    """Assess oscillation threshold via Wang-Q and cooperativity."""
    # Wang 2024: amplifier properties include Q_m vs Q_L comparison
    amp_props: AmplifierProperties = compute_amplifier_properties(
        nv_config, cavity_config, maser_config
    )

    # Breeze cooperativity: separate calculation using spin linewidth
    spin_lw = _spin_linewidth_hz(nv_config)
    thresh = compute_full_threshold(
        nv_config, maser_config, cavity_config, gain_budget, spin_lw
    )

    q_ratio = (
        amp_props.magnetic_q / amp_props.loaded_q
        if amp_props.loaded_q > 0
        else float("inf")
    )

    return OscillationThresholdResult(
        magnetic_q=amp_props.magnetic_q,
        loaded_q=amp_props.loaded_q,
        q_ratio=q_ratio,
        above_threshold_wang=amp_props.above_threshold,
        cooperativity=thresh.cooperativity,
        threshold_margin=thresh.threshold_margin,
        masing=thresh.masing,
        spin_temperature_k=amp_props.spin_temperature_k,
    )


def _evaluate_frequency(
    maser_config: MaserConfig,
    config: Phase1Config,
) -> tuple[float, float, float, bool]:
    """Return (freq_ghz, target_ghz, deviation_mhz, met)."""
    f = maser_config.cavity_frequency_ghz
    deviation_mhz = abs(f - config.target_frequency_ghz) * 1e3
    met = deviation_mhz <= config.frequency_tolerance_mhz
    return f, config.target_frequency_ghz, deviation_mhz, met


def _evaluate_output_power(
    nv_config: NVConfig,
    cavity_config: CavityConfig,
    maser_config: MaserConfig,
    gain_budget: float,
    config: Phase1Config,
) -> tuple[float, float, bool]:
    """Return (power_w, power_dbm, met)."""
    cprops = compute_cavity_properties(maser_config, cavity_config)
    spin_lw = _spin_linewidth_hz(nv_config)
    thresh = compute_full_threshold(
        nv_config, maser_config, cavity_config, gain_budget, spin_lw
    )
    pwr = compute_output_power(cprops, thresh, nv_config, maser_config)
    met = pwr.output_power_dbm >= config.min_output_power_dbm
    return pwr.output_power_w, pwr.output_power_dbm, met


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Public API                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝


def validate_phase1_milestone(
    nv_config: NVConfig | None = None,
    cavity_config: CavityConfig | None = None,
    maser_config: MaserConfig | None = None,
    config: Phase1Config | None = None,
) -> Phase1MilestoneResult:
    """Run the Phase-1 maser oscillation milestone validation.

    Evaluates three criteria against the digital-twin physics stack:

    1. **Threshold**: Q_m ≤ Q_L (Wang 2024) AND C > 1 (Breeze cooperativity).
    2. **Frequency**: |f_cavity − 1.47 GHz| ≤ 50 MHz.
    3. **Power**: CW output power ≥ −100 dBm.

    All three must pass for ``phase1_milestone_closed = True``.

    Args:
        nv_config:    NV spin parameters.  ``None`` → ``NVConfig()`` defaults.
        cavity_config: Cavity geometry.   ``None`` → ``CavityConfig()`` defaults.
        maser_config: Cavity Q and freq.  ``None`` → ``MaserConfig()`` defaults.
        config:       Acceptance thresholds.  ``None`` → ``Phase1Config()`` defaults.

    Returns:
        :class:`Phase1MilestoneResult` with full assessment.
    """
    if nv_config is None:
        nv_config = NVConfig()
    if cavity_config is None:
        cavity_config = CavityConfig()
    if maser_config is None:
        maser_config = MaserConfig()
    if config is None:
        config = Phase1Config()

    # ── Criterion 1: Oscillation threshold ────────────────────────
    oscillation = _evaluate_oscillation(
        nv_config, cavity_config, maser_config, config.gain_budget
    )
    threshold_met = oscillation.above_threshold_wang and oscillation.masing

    # ── Criterion 2: Frequency ─────────────────────────────────────
    freq_ghz, target_ghz, deviation_mhz, frequency_met = _evaluate_frequency(
        maser_config, config
    )

    # ── Criterion 3: Output power ──────────────────────────────────
    power_w, power_dbm, power_met = _evaluate_output_power(
        nv_config, cavity_config, maser_config, config.gain_budget, config
    )

    # ── Verdict ────────────────────────────────────────────────────
    milestone_closed = threshold_met and frequency_met and power_met

    if milestone_closed:
        msg = (
            f"Phase-1 CLOSED: Q_m/Q_L = {oscillation.q_ratio:.3f} < 1, "
            f"C = {oscillation.cooperativity:.2f} > 1, "
            f"f = {freq_ghz:.3f} GHz (Δ = {deviation_mhz:.1f} MHz), "
            f"P_out = {power_dbm:.1f} dBm."
        )
    else:
        failures = []
        if not threshold_met:
            failures.append(
                f"threshold not met (Q_m/Q_L = {oscillation.q_ratio:.3f}, "
                f"C = {oscillation.cooperativity:.2f})"
            )
        if not frequency_met:
            failures.append(
                f"frequency off by {deviation_mhz:.1f} MHz "
                f"(limit {config.frequency_tolerance_mhz:.0f} MHz)"
            )
        if not power_met:
            failures.append(
                f"output power {power_dbm:.1f} dBm < "
                f"{config.min_output_power_dbm:.0f} dBm"
            )
        msg = "Phase-1 OPEN: " + "; ".join(failures) + "."

    return Phase1MilestoneResult(
        oscillation=oscillation,
        frequency_ghz=freq_ghz,
        target_frequency_ghz=target_ghz,
        frequency_deviation_mhz=deviation_mhz,
        frequency_tolerance_mhz=config.frequency_tolerance_mhz,
        output_power_w=power_w,
        output_power_dbm=power_dbm,
        min_output_power_dbm=config.min_output_power_dbm,
        threshold_met=threshold_met,
        frequency_met=frequency_met,
        power_met=power_met,
        phase1_milestone_closed=milestone_closed,
        closing_message=msg,
    )
