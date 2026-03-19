"""
Pulsed pump sequence optimizer for the NV diamond maser.

Optimizes the pulse width and duty cycle of a pulsed optical pump
to maximise population inversion subject to a peak-power constraint,
and compares pulsed versus CW operation.

Background
──────────
Long et al. (2025) demonstrated a room-temperature maser using an LED
pumped at 7–200 µs pulses with ~130 W peak power.  Because LEDs are
cheaper and more compact than CW lasers, optimising the pulsed sequence
for a given power budget is important for portable hardware.

The key trade-offs are:

* **Pulse width** — longer pulses let inversion build toward the CW
  steady-state, but also let it decay more between pulses if T₁ is short.
* **Duty cycle** — higher duty increases mean power (and thus approaches
  CW), but peak power and driver cost set an upper limit.
* **Peak power** — for LEDs the peak power is limited by junction heating;
  a separate power budget constraint is enforced.

Pulsed maser threshold
──────────────────────
In the pulsed regime the effective pump efficiency used in the threshold
calculation is the *time-averaged* inversion over a burst sequence,
rather than the CW steady-state value.  The cooperativity is therefore:

    C_pulsed = C_CW × (η_mean_pulsed / η_cw_ss)

where η_cw_ss = s × 2/3 at CW saturation.

References
──────────
Long et al., Commun. Eng. 4, 67 (2025), PMC12241473.
Breeze et al., Nature 555, 493 (2018).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence as TSequence

import numpy as np

from ..config import CavityConfig, MaserConfig, NVConfig, OpticalPumpConfig
from .cavity import (
    ThresholdResult,
    compute_cavity_properties,
    compute_maser_threshold,
    compute_n_effective,
)
from .optical_pump import compute_pump_state
from .pulsed_pump import PulsedPumpResult, compute_pulsed_inversion


# ── Public dataclasses ────────────────────────────────────────────


@dataclass(frozen=True)
class PulseCandidate:
    """A single (pulse_width, duty_cycle) candidate evaluated during search.

    Attributes:
        pulse_duration_us: Pump-ON duration per cycle (µs).
        duty_cycle: τ_ON / τ_period ∈ (0, 1].
        peak_inversion: Maximum inversion achieved during any pulse.
        mean_inversion: Time-averaged inversion over the evaluated window.
        equivalent_cw_power_w: Peak power × duty cycle (W).
        pulsed_result: Full time-evolution from the ODE solver.
    """

    pulse_duration_us: float
    duty_cycle: float
    peak_inversion: float
    mean_inversion: float
    equivalent_cw_power_w: float
    pulsed_result: PulsedPumpResult


@dataclass(frozen=True)
class OptimizedSequence:
    """Best pulse sequence for a given power budget.

    Attributes:
        pulse_duration_us: Recommended pulse width (µs).
        duty_cycle: Recommended duty cycle.
        pulse_period_us: Recommended repetition period (µs).
        peak_power_w: Required peak power.
        mean_inversion: Time-averaged inversion of the optimal sequence.
        peak_inversion: Maximum inversion during a single pulse.
        equivalent_cw_power_w: Power-budget-equivalent CW power.
        all_candidates: All (width, duty) candidates evaluated.
    """

    pulse_duration_us: float
    duty_cycle: float
    pulse_period_us: float
    peak_power_w: float
    mean_inversion: float
    peak_inversion: float
    equivalent_cw_power_w: float
    all_candidates: list[PulseCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class PulsedThresholdResult:
    """Maser threshold evaluation for a pulsed sequence.

    Attributes:
        cw_threshold: Threshold evaluated using the CW pump efficiency.
        pulsed_threshold: Threshold using the pulsed mean inversion.
        mean_inversion_fraction: η_mean / η_cw_ss.
        sequence: The pulse sequence that was evaluated.
    """

    cw_threshold: ThresholdResult
    pulsed_threshold: ThresholdResult
    mean_inversion_fraction: float
    sequence: OptimizedSequence


@dataclass(frozen=True)
class CWvsPulsedReport:
    """Comparison of CW and best pulsed operation.

    Attributes:
        cw_inversion: Steady-state CW inversion (dimensionless).
        pulsed_inversion: Mean inversion for best pulsed sequence.
        inversion_ratio: pulsed / CW.
        cw_power_w: CW laser power.
        pulsed_equivalent_cw_power_w: Pulsed sequence equivalent CW power.
        power_reduction_fraction: (cw − pulsed_eq) / cw.
        cw_cooperativity: C under CW.
        pulsed_cooperativity: C under best pulsed sequence.
        recommended: "cw" or "pulsed" — which mode is better overall.
        optimal_sequence: Best pulsed sequence found.
    """

    cw_inversion: float
    pulsed_inversion: float
    inversion_ratio: float
    cw_power_w: float
    pulsed_equivalent_cw_power_w: float
    power_reduction_fraction: float
    cw_cooperativity: float
    pulsed_cooperativity: float
    recommended: str  # "cw" or "pulsed"
    optimal_sequence: OptimizedSequence


# ── Core optimizer ────────────────────────────────────────────────


def optimize_pulse_sequence(
    pump_config: OpticalPumpConfig,
    nv_config: NVConfig,
    *,
    target_inversion: float = 0.3,
    max_peak_power_w: float | None = None,
    pulse_durations_us: TSequence[float] | None = None,
    duty_cycles: TSequence[float] | None = None,
    n_cycles: int = 6,
) -> OptimizedSequence:
    """
    Grid-search optimal pulse width and duty cycle.

    Evaluates each (pulse_width, duty_cycle) combination by running the
    pulsed-inversion ODE and selects the candidate that minimises the
    distance to *target_inversion* while staying within the peak-power
    budget.

    If *target_inversion* is already achieved by many candidates the one
    with the lowest equivalent CW power (i.e. lowest energy per cycle)
    is returned.

    Args:
        pump_config: Base pump config.  ``laser_power_w`` is used as the
            peak power.  ``pulsed``, ``pulse_duration_us``, and
            ``pulse_period_us`` are overridden during the search.
        nv_config: NV parameters (T₁).
        target_inversion: Desired mean inversion (0–2/3).  Defaults to 0.3.
        max_peak_power_w: Maximum allowed peak power (W).  When ``None``
            the ``pump_config.laser_power_w`` value is used.
        pulse_durations_us: Pulse widths to try (µs).  Defaults to a
            log-spaced grid from 1 µs to 500 µs (20 points).
        duty_cycles: Duty cycles to evaluate.  Defaults to
            [0.1, 0.2, 0.3, 0.5, 0.7, 1.0].
        n_cycles: ODE cycles per candidate evaluation.

    Returns:
        OptimizedSequence with recommended pulse parameters.

    Raises:
        ValueError: If target_inversion is outside (0, 2/3].
        ValueError: If max_peak_power_w ≤ 0.
    """
    if not (0 < target_inversion <= 2.0 / 3.0):
        raise ValueError(
            f"target_inversion must be in (0, 2/3], got {target_inversion}"
        )

    p_peak = max_peak_power_w if max_peak_power_w is not None else pump_config.laser_power_w
    if p_peak <= 0:
        raise ValueError(f"max_peak_power_w must be > 0, got {p_peak}")

    # ── Build search grids ────────────────────────────────────────
    if pulse_durations_us is None:
        pulse_durations_us = list(
            np.logspace(np.log10(1.0), np.log10(500.0), 20)
        )
    if duty_cycles is None:
        duty_cycles = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    candidates: list[PulseCandidate] = []

    for pw_us in pulse_durations_us:
        for duty in duty_cycles:
            period_us = pw_us / duty  # τ_period = τ_pulse / duty

            # Build pulsed OpticalPumpConfig with this (width, period)
            trial_config = pump_config.model_copy(
                update={
                    "pulsed": True,
                    "laser_power_w": p_peak,
                    "pulse_duration_us": pw_us,
                    "pulse_period_us": period_us,
                }
            )

            try:
                result = compute_pulsed_inversion(
                    trial_config, nv_config, n_cycles=n_cycles
                )
            except (ValueError, RuntimeError):
                # Skip invalid combinations (shouldn't happen for valid grids)
                continue

            cand = PulseCandidate(
                pulse_duration_us=pw_us,
                duty_cycle=duty,
                peak_inversion=result.peak_inversion,
                mean_inversion=result.mean_inversion,
                equivalent_cw_power_w=p_peak * duty,
                pulsed_result=result,
            )
            candidates.append(cand)

    if not candidates:
        raise RuntimeError("No valid candidates found during pulse optimisation.")

    # ── Select best candidate ─────────────────────────────────────
    # Primary key: minimise |η_mean − target|
    # Tie-break: minimise equivalent CW power (lower duty)
    def score(c: PulseCandidate) -> tuple[float, float]:
        return abs(c.mean_inversion - target_inversion), c.equivalent_cw_power_w

    best = min(candidates, key=score)

    return OptimizedSequence(
        pulse_duration_us=best.pulse_duration_us,
        duty_cycle=best.duty_cycle,
        pulse_period_us=best.pulse_duration_us / best.duty_cycle,
        peak_power_w=p_peak,
        mean_inversion=best.mean_inversion,
        peak_inversion=best.peak_inversion,
        equivalent_cw_power_w=best.equivalent_cw_power_w,
        all_candidates=candidates,
    )


# ── Pulsed threshold evaluator ─────────────────────────────────────


def compute_pulsed_threshold(
    sequence: OptimizedSequence,
    pump_config: OpticalPumpConfig,
    nv_config: NVConfig,
    maser_config: MaserConfig,
    cavity_config: CavityConfig,
    gain_budget: float = 1.0,
) -> PulsedThresholdResult:
    """
    Compare maser threshold for CW vs the given pulsed sequence.

    The cooperativity is computed twice:

    1. **CW** — using the steady-state pump efficiency from
       ``compute_pump_state()`` (η_eff = s × 2/3).
    2. **Pulsed** — using the time-averaged inversion from the sequence,
       which serves as an effective pump efficiency.

    Args:
        sequence: Optimal or candidate pulse sequence.
        pump_config: Base pump configuration (CW reference).
        nv_config: NV center parameters.
        maser_config: Cavity Q and frequency.
        cavity_config: Mode volume and fill factor.
        gain_budget: Spectral overlap fraction (0–1).

    Returns:
        PulsedThresholdResult comparing CW and pulsed cooperativity.
    """
    spin_lw_hz = 1.0 / (math.pi * nv_config.t2_star_us * 1e-6)
    cavity_props = compute_cavity_properties(maser_config, cavity_config)

    # ── CW threshold ──────────────────────────────────────────────
    cw_n_eff = compute_n_effective(nv_config, cavity_config, gain_budget)
    cw_threshold = compute_maser_threshold(cavity_props, cw_n_eff, spin_lw_hz)

    # ── Pulsed threshold ──────────────────────────────────────────
    # Replace pump_efficiency with the mean inversion from the pulsed sequence
    # (normalized to 2/3 since N_eff uses pump_efficiency ∈ [0, η_max]).
    cw_pump = compute_pump_state(pump_config, nv_config)
    cw_eta_ss = cw_pump.effective_pump_efficiency  # η_cw = s × 2/3

    if cw_eta_ss > 0:
        scale = sequence.mean_inversion / cw_eta_ss
    else:
        scale = 0.0

    pulsed_nv = nv_config.model_copy(
        update={"pump_efficiency": nv_config.pump_efficiency * scale}
    )
    pulsed_n_eff = compute_n_effective(pulsed_nv, cavity_config, gain_budget)
    pulsed_threshold = compute_maser_threshold(
        cavity_props, pulsed_n_eff, spin_lw_hz
    )

    return PulsedThresholdResult(
        cw_threshold=cw_threshold,
        pulsed_threshold=pulsed_threshold,
        mean_inversion_fraction=scale,
        sequence=sequence,
    )


# ── CW vs pulsed comparison ────────────────────────────────────────


def compare_cw_vs_pulsed(
    pump_config: OpticalPumpConfig,
    nv_config: NVConfig,
    maser_config: MaserConfig,
    cavity_config: CavityConfig,
    *,
    max_peak_power_w: float | None = None,
    target_inversion: float = 0.3,
    gain_budget: float = 1.0,
) -> CWvsPulsedReport:
    """
    Compare CW laser pumping against the best pulsed sequence.

    Runs ``optimize_pulse_sequence()`` to find the best pulsed strategy,
    then evaluates both modes on inversion, power, and cooperativity.

    The recommended mode is ``"pulsed"`` when pulsed achieves ≥ 90% of
    the CW cooperativity with ≤ 70% of the equivalent CW power (i.e. a
    meaningful energy saving without too much performance dip).

    Args:
        pump_config: Pump config used for both CW and pulsed runs.
        nv_config: NV center parameters.
        maser_config: Cavity Q and frequency.
        cavity_config: Mode volume and fill factor.
        max_peak_power_w: Maximum peak power for pulsed run.  Defaults to
            ``pump_config.laser_power_w``.
        target_inversion: Inversion target passed to the optimizer.
        gain_budget: Spectral overlap fraction.

    Returns:
        CWvsPulsedReport with side-by-side metrics and a recommendation.
    """
    spin_lw_hz = 1.0 / (math.pi * nv_config.t2_star_us * 1e-6)
    cavity_props = compute_cavity_properties(maser_config, cavity_config)

    # ── CW evaluation ─────────────────────────────────────────────
    cw_pump = compute_pump_state(pump_config, nv_config)
    cw_inv = cw_pump.effective_pump_efficiency  # η_eff = s × 2/3

    cw_n_eff = compute_n_effective(nv_config, cavity_config, gain_budget)
    cw_thr = compute_maser_threshold(cavity_props, cw_n_eff, spin_lw_hz)

    # ── Pulsed evaluation ─────────────────────────────────────────
    opt_seq = optimize_pulse_sequence(
        pump_config,
        nv_config,
        target_inversion=target_inversion,
        max_peak_power_w=max_peak_power_w,
    )

    pulsed_inv = opt_seq.mean_inversion
    scale = pulsed_inv / cw_inv if cw_inv > 0 else 0.0
    pulsed_nv = nv_config.model_copy(
        update={"pump_efficiency": nv_config.pump_efficiency * scale}
    )
    pulsed_n_eff = compute_n_effective(pulsed_nv, cavity_config, gain_budget)
    pulsed_thr = compute_maser_threshold(cavity_props, pulsed_n_eff, spin_lw_hz)

    # ── Power metrics ─────────────────────────────────────────────
    p_cw = pump_config.laser_power_w
    p_equiv = opt_seq.equivalent_cw_power_w
    p_reduction = (p_cw - p_equiv) / p_cw if p_cw > 0 else 0.0

    # ── Recommend pulsed when energy saving > 30% & C ≥ 90% CW ──
    c_ratio = (
        pulsed_thr.cooperativity / cw_thr.cooperativity
        if cw_thr.cooperativity > 0
        else 0.0
    )
    recommended = "pulsed" if (c_ratio >= 0.9 and p_reduction >= 0.3) else "cw"

    return CWvsPulsedReport(
        cw_inversion=cw_inv,
        pulsed_inversion=pulsed_inv,
        inversion_ratio=pulsed_inv / cw_inv if cw_inv > 0 else 0.0,
        cw_power_w=p_cw,
        pulsed_equivalent_cw_power_w=p_equiv,
        power_reduction_fraction=p_reduction,
        cw_cooperativity=cw_thr.cooperativity,
        pulsed_cooperativity=pulsed_thr.cooperativity,
        recommended=recommended,
        optimal_sequence=opt_seq,
    )
