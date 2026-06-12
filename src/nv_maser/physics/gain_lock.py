"""Gain-lock PI control loop for maser threshold stabilisation (R10 mitigation).

Risk R10 — *Maser regenerative mode unstable*: "Digital twin models threshold
precisely; implement gain-lock control loop."

Architecture reference: handheld-maser-probe-architecture.md §13 (Risk Register)

Physics
───────
The NV maser enters CW oscillation when the ensemble cooperativity C ≥ 1:

    C = 4 g_N² / (κ · γ_⊥)

where

    g_N  = g₀ √N_eff            ensemble vacuum Rabi coupling   (Hz)
    κ    = ω_c / Q_L            cavity field decay rate          (Hz)
    γ_⊥  = 1 / (π T₂*)          spin decoherence rate (HWHM)    (Hz)
    N_eff = N × η_orient × η_pump  effective inverted spin count

The pump efficiency η_pump grows monotonically with laser power P through the
saturation curve from ``optical_pump.compute_pump_state()``:

    Γ_pump(P) = σ_abs × I₀(P) / (ℏ ω_L)
    η_pump(P) = [Γ_pump / (Γ_pump + 1/T₁)] × (2/3)

Substituting gives C(P): a monotonically increasing function of laser power.
Near threshold C ≈ 1, the gain is extremely sensitivity to P (∂G/∂P → ∞ right
above threshold for a CW oscillator), making active stabilisation essential.

Gain-lock PI controller
────────────────────────
Observable  ε  = C(P) − 1     signed cooperativity excess
Setpoint    ε* = C_target − 1  (e.g. C_target = 1.10 → 10 % above threshold)
Error        e  = ε* − ε

Discrete-time PI update (Euler forward, step Δt):
    integral ← integral + e × Δt
    ΔP       = Kp × e + Ki × integral
    P_new    = clip(P + ΔP, P_min, P_max)

Convergence: with Kp ≈ 0.05 W and Ki ≈ Kp / (10 × T₁_ms × 1e-3), the loop
settles within O(10 × T₁) ≈ O(60 ms) for the default NVConfig (T₁ = 6 ms).

Disturbance model (optional)
──────────────────────────────
Additive white Gaussian noise on the cooperativity observable mimics laser
shot noise, pointing jitter, and vibration-induced cavity detuning.  Pass
``coop_noise_sigma`` in ``GainLockConfig`` to enable.

References
──────────
Risk register: Handheld probe architecture §13 R10.
Breeze et al. (2018) Nature 555, 493.
Siegman, Lasers (1986) — gain saturation and CW laser stabilisation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import CavityConfig, MaserConfig, NVConfig, OpticalPumpConfig
from .cavity import compute_full_threshold
from .optical_pump import compute_pump_state

# ── Physical constant ─────────────────────────────────────────────
_PI = math.pi


# ══════════════════════════════════════════════════════════════════
# Public data classes
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GainLockConfig:
    """Configuration for the gain-lock PI control loop.

    Attributes:
        target_cooperativity: Desired C setpoint (> 1 for above-threshold
            operation).  Default 1.10 = 10 % above threshold.
        kp: Proportional gain (W per unit cooperativity error).
        ki: Integral gain (W s⁻¹ per unit cooperativity error).
        dt_us: Control step in microseconds.
        min_pump_power_w: Lower saturation limit for laser power.
        max_pump_power_w: Upper saturation limit for laser power.
        coop_noise_sigma: Standard deviation of additive Gaussian noise on the
            cooperativity observable (0.0 disables noise).
        lock_tolerance: Cooperativity |error| below which loop is considered
            "locked".
    """

    target_cooperativity: float = 1.10
    kp: float = 0.001
    ki: float = 1.0
    dt_us: float = 10.0
    min_pump_power_w: float = 1e-3
    max_pump_power_w: float = 20.0
    coop_noise_sigma: float = 0.0
    lock_tolerance: float = 0.05


@dataclass(frozen=True)
class GainLockStep:
    """Snapshot of the gain-lock loop at one control step.

    Attributes:
        step: Zero-based step index.
        time_us: Elapsed time in microseconds.
        pump_power_w: Commanded laser power at this step (W).
        cooperativity: Ensemble cooperativity C computed from current pump.
        error: Signed cooperativity error (target − current).
        integral: Accumulated PI integral at this step.
        locked: True when |error| < ``lock_tolerance``.
    """

    step: int
    time_us: float
    pump_power_w: float
    cooperativity: float
    error: float
    integral: float
    locked: bool


@dataclass(frozen=True)
class GainLockResult:
    """Summary of a completed gain-lock simulation.

    Attributes:
        steps: Time-ordered list of per-step snapshots.
        locked_at_step: First step index with ``locked = True``, or -1.
        locked_at_us: Elapsed time (μs) when first locked, or NaN.
        final_cooperativity: C at the last step.
        final_pump_power_w: Laser power at the last step (W).
        converged: True when the loop was locked at the final step.
    """

    steps: list[GainLockStep]
    locked_at_step: int
    locked_at_us: float
    final_cooperativity: float
    final_pump_power_w: float
    converged: bool

    @property
    def n_steps(self) -> int:
        """Total number of simulated steps."""
        return len(self.steps)

    @property
    def lock_time_t1_units(self) -> float:
        """Lock time expressed as multiples of T₁ (if nv_config available).

        Returns NaN when never locked.
        """
        return float("nan") if math.isnan(self.locked_at_us) else self.locked_at_us


# ══════════════════════════════════════════════════════════════════
# Core physics helper
# ══════════════════════════════════════════════════════════════════


def compute_cooperativity(
    pump_power_w: float,
    nv_config: NVConfig,
    maser_config: MaserConfig,
    cavity_config: CavityConfig,
    pump_config_template: OpticalPumpConfig,
    *,
    gain_budget: float = 1.0,
) -> float:
    """Compute ensemble cooperativity C for a given laser power.

    Maps ``pump_power_w`` → η_pump → updated NVConfig → C via
    ``compute_full_threshold()``.  The caller's NVConfig and
    OpticalPumpConfig are not mutated.

    Args:
        pump_power_w: Laser power to evaluate (W).
        nv_config: Base NV centre parameters (pump_efficiency is overridden).
        maser_config: Cavity Q and resonance frequency.
        cavity_config: Mode volume and fill factor.
        pump_config_template: Pump laser geometry (wavelength, beam waist,
            absorption cross-section).  Only ``laser_power_w`` is overridden.
        gain_budget: Spectral overlap fraction ∈ (0, 1] accounting for field
            inhomogeneous broadening.  Default 1.0 (perfect shimming).

    Returns:
        Dimensionless cooperativity C.  Zero if the pump is off.
    """
    if pump_power_w <= 0.0:
        return 0.0

    # Build pump config at this power level
    pump_cfg = pump_config_template.model_copy(update={"laser_power_w": pump_power_w})

    # η_pump from saturation curve
    pump_state = compute_pump_state(pump_cfg, nv_config)
    eta_pump = pump_state.effective_pump_efficiency

    # Substitute into NVConfig (pump_efficiency = η_eff)
    nv_pumped = nv_config.model_copy(update={"pump_efficiency": eta_pump})

    # Spin linewidth γ_⊥ = 1 / (π T₂*) [Hz]
    spin_lw_hz = 1.0 / (_PI * nv_config.t2_star_us * 1e-6)

    threshold = compute_full_threshold(
        nv_pumped, maser_config, cavity_config, gain_budget, spin_lw_hz
    )
    return threshold.cooperativity


def find_threshold_pump_power(
    nv_config: NVConfig,
    maser_config: MaserConfig,
    cavity_config: CavityConfig,
    pump_config_template: OpticalPumpConfig,
    *,
    gain_budget: float = 1.0,
    p_lo: float = 1e-4,
    p_hi: float = 50.0,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> float:
    """Find the laser power at which C = 1 (masing threshold).

    Uses bisection on compute_cooperativity().  Raises ValueError if
    C < 1 even at ``p_hi`` (threshold unreachable in the bracket) or
    C > 1 already at ``p_lo``.

    Args:
        nv_config: NV parameters.
        maser_config: Cavity configuration.
        cavity_config: Mode volume.
        pump_config_template: Pump geometry template.
        gain_budget: Field quality factor (0–1).
        p_lo: Lower bracket (W).  Must have C(p_lo) < 1.
        p_hi: Upper bracket (W).  Must have C(p_hi) > 1.
        tol: Convergence tolerance on laser power (W).
        max_iter: Safety limit on bisection iterations.

    Returns:
        Threshold pump power P_th in Watts.

    Raises:
        ValueError: If the bracket does not straddle threshold.
    """
    c_lo = compute_cooperativity(
        p_lo, nv_config, maser_config, cavity_config, pump_config_template,
        gain_budget=gain_budget,
    )
    c_hi = compute_cooperativity(
        p_hi, nv_config, maser_config, cavity_config, pump_config_template,
        gain_budget=gain_budget,
    )

    if c_lo >= 1.0:
        raise ValueError(
            f"C({p_lo:.4f} W) = {c_lo:.4f} ≥ 1 — already above threshold at "
            f"lower bracket.  Decrease p_lo."
        )
    if c_hi < 1.0:
        raise ValueError(
            f"C({p_hi:.2f} W) = {c_hi:.4f} < 1 — threshold unreachable in "
            f"[{p_lo}, {p_hi}] W.  Increase p_hi or relax gain_budget."
        )

    for _ in range(max_iter):
        p_mid = 0.5 * (p_lo + p_hi)
        if (p_hi - p_lo) < tol:
            return p_mid
        c_mid = compute_cooperativity(
            p_mid, nv_config, maser_config, cavity_config, pump_config_template,
            gain_budget=gain_budget,
        )
        if c_mid < 1.0:
            p_lo = p_mid
        else:
            p_hi = p_mid

    return 0.5 * (p_lo + p_hi)


# ══════════════════════════════════════════════════════════════════
# Gain-lock simulation
# ══════════════════════════════════════════════════════════════════


def run_gain_lock_simulation(
    n_steps: int,
    nv_config: NVConfig,
    maser_config: MaserConfig,
    cavity_config: CavityConfig,
    pump_config_template: OpticalPumpConfig,
    lock_config: GainLockConfig,
    *,
    initial_pump_power_w: float | None = None,
    gain_budget: float = 1.0,
    rng: np.random.Generator | None = None,
) -> GainLockResult:
    """Simulate the gain-lock PI control loop for ``n_steps`` steps.

    The loop measures the ensemble cooperativity C at each step,
    computes the error against the target setpoint, and applies a PI
    correction to the laser power for the next step.

    Optionally injects additive Gaussian noise onto the cooperativity
    measurement to model laser shot noise and pointing jitter.

    Args:
        n_steps: Number of control steps to simulate (≥ 1).
        nv_config: NV centre parameters.
        maser_config: Cavity configuration.
        cavity_config: Mode volume / fill factor.
        pump_config_template: Pump laser geometry template.
        lock_config: PI gains, setpoint, power limits, noise.
        initial_pump_power_w: Starting laser power (W).  Defaults to
            ``pump_config_template.laser_power_w``.
        gain_budget: Field quality factor ∈ (0, 1].
        rng: Optional NumPy random generator for reproducibility.

    Returns:
        :class:`GainLockResult` with per-step snapshots and summary.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be ≥ 1, got {n_steps}")

    if rng is None:
        rng = np.random.default_rng(seed=0)

    dt_s = lock_config.dt_us * 1e-6
    target = lock_config.target_cooperativity

    pump_power = (
        pump_config_template.laser_power_w
        if initial_pump_power_w is None
        else float(initial_pump_power_w)
    )
    pump_power = float(
        np.clip(pump_power, lock_config.min_pump_power_w, lock_config.max_pump_power_w)
    )

    integral = 0.0
    steps: list[GainLockStep] = []
    locked_at_step = -1
    locked_at_us = float("nan")

    for i in range(n_steps):
        # ── Measure cooperativity ──────────────────────────────────
        coop = compute_cooperativity(
            pump_power, nv_config, maser_config, cavity_config,
            pump_config_template, gain_budget=gain_budget,
        )
        if lock_config.coop_noise_sigma > 0.0:
            coop += float(rng.normal(0.0, lock_config.coop_noise_sigma))

        # ── PI update ──────────────────────────────────────────────
        error = target - coop
        integral += error * dt_s
        delta_p = lock_config.kp * error + lock_config.ki * integral
        pump_power = float(
            np.clip(
                pump_power + delta_p,
                lock_config.min_pump_power_w,
                lock_config.max_pump_power_w,
            )
        )

        locked = abs(error) < lock_config.lock_tolerance
        time_us = (i + 1) * lock_config.dt_us

        step = GainLockStep(
            step=i,
            time_us=time_us,
            pump_power_w=pump_power,
            cooperativity=coop,
            error=error,
            integral=integral,
            locked=locked,
        )
        steps.append(step)

        if locked_at_step < 0 and locked:
            locked_at_step = i
            locked_at_us = time_us

    return GainLockResult(
        steps=steps,
        locked_at_step=locked_at_step,
        locked_at_us=locked_at_us if locked_at_step >= 0 else float("nan"),
        final_cooperativity=steps[-1].cooperativity,
        final_pump_power_w=steps[-1].pump_power_w,
        converged=steps[-1].locked,
    )
