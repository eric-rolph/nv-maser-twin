"""Scan-time-gated depth-limit model (R7 — SNR at depth).

Answers the clinical question: for a given scan-time budget and target SNR,
what is the maximum voxel depth achievable with the configured hardware?

Core formula::

    required_averages(d) = ceil( (target_snr / snr_per_shot(d))² )
    scan_time(d)         = required_averages(d) × TR_ms / 1000  [s]
    feasible(d)          ⟺ scan_time(d) ≤ scan_time_budget_s
    max_depth_mm         = max{ d : feasible(d) }

Architecture context (§8.3–8.4 of the probe architecture document):
  - Single-shot SNR at 20 mm depth: ~10⁻⁶ (coil-noise dominated at 300 K).
  - Maser advantage over conventional LNA: ~1.11× (~1 dB); coil noise dominates.
  - V1 design intent: 5–15 mm operating range (Risk R7 mitigation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .depth_profile import TissueLayer
from .single_sided_magnet import SingleSidedMagnet
from .snr_calculator import compute_snr_budget
from .surface_coil import SurfaceCoil


@dataclass(frozen=True)
class DepthLimitConfig:
    """Configuration for the scan-time-gated depth-limit sweep.

    Attributes:
        target_snr: Minimum acceptable SNR (linear).  Default 5.0 is a
            conservative diagnostic threshold.
        scan_time_budget_s: Maximum allowed scan time in seconds.  Default
            120 s corresponds to a 2-minute clinical constraint.
        voxel_size_mm: Isotropic voxel side length (mm).
        tr_ms: Repetition time (ms).
        te_ms: Echo time (ms).
        sequence: Pulse sequence type passed to ``compute_snr_budget``.
        bandwidth_hz: Receiver bandwidth (Hz).
        depth_step_mm: Depth increment between evaluated voxels (mm).
        min_depth_mm: First depth to evaluate (mm).
        max_depth_mm: Last depth to evaluate (mm); must be > min_depth_mm.
    """

    target_snr: float = 5.0
    scan_time_budget_s: float = 120.0
    voxel_size_mm: float = 3.0
    tr_ms: float = 100.0
    te_ms: float = 10.0
    sequence: str = "spin_echo"
    bandwidth_hz: float = 10_000.0
    depth_step_mm: float = 1.0
    min_depth_mm: float = 1.0
    max_depth_mm: float = 30.0

    def __post_init__(self) -> None:
        if self.target_snr <= 0:
            raise ValueError(
                f"target_snr must be positive, got {self.target_snr}"
            )
        if self.scan_time_budget_s <= 0:
            raise ValueError(
                f"scan_time_budget_s must be positive, got {self.scan_time_budget_s}"
            )
        if self.voxel_size_mm <= 0:
            raise ValueError(
                f"voxel_size_mm must be positive, got {self.voxel_size_mm}"
            )
        if self.tr_ms <= 0:
            raise ValueError(
                f"tr_ms must be positive, got {self.tr_ms}"
            )
        if self.depth_step_mm <= 0:
            raise ValueError(
                f"depth_step_mm must be positive, got {self.depth_step_mm}"
            )
        if self.min_depth_mm >= self.max_depth_mm:
            raise ValueError(
                f"min_depth_mm ({self.min_depth_mm}) must be less than "
                f"max_depth_mm ({self.max_depth_mm})"
            )


@dataclass(frozen=True)
class DepthPoint:
    """Depth-limit analysis result for a single voxel depth.

    Attributes:
        depth_mm: Voxel depth from coil surface (mm).
        snr_per_shot: Single-acquisition SNR (linear, << 1 at clinical depths).
        required_averages: Number of signal averages needed to reach
            ``target_snr``; computed as ``ceil((target_snr / snr_per_shot)²)``.
        scan_time_s: Total scan time = ``required_averages × TR_ms / 1000`` (s).
        within_budget: True iff ``scan_time_s ≤ scan_time_budget_s``.
    """

    depth_mm: float
    snr_per_shot: float
    required_averages: int
    scan_time_s: float
    within_budget: bool


@dataclass(frozen=True)
class DepthLimitResult:
    """Result of a scan-time-gated depth-limit sweep.

    Attributes:
        depth_profile: Tuple of :class:`DepthPoint` for each evaluated depth,
            ordered from shallowest to deepest.
        max_depth_mm: Deepest voxel depth within the scan-time budget (mm).
            Zero when no depth is feasible (``any_feasible`` is False).
        n_depths_evaluated: Number of depths in the sweep.
        scan_time_budget_s: Budget used for this sweep (echoed from config).
        target_snr: Target SNR used for this sweep (echoed from config).
        snr_per_shot_at_limit: Per-shot SNR at ``max_depth_mm``.  When
            ``any_feasible`` is False this reflects the shallowest (least
            bad) depth in the profile.
        required_averages_at_limit: Averages required at the limit depth.
        scan_time_at_limit_s: Scan time required at the limit depth (s).
        any_feasible: True iff at least one depth in the profile is within
            the scan-time budget.
        v1_depth_range_confirmed: True iff all depths in [5 mm, 15 mm]
            present in the profile are within the scan-time budget.  This
            directly validates the V1 design intent (R7 mitigation).
        snr_per_shot_at_5mm: Per-shot SNR at 5 mm depth (None if 5 mm is not
            in the evaluated profile).
        snr_per_shot_at_10mm: Per-shot SNR at 10 mm depth (None if absent).
        snr_per_shot_at_15mm: Per-shot SNR at 15 mm depth (None if absent).
    """

    depth_profile: tuple[DepthPoint, ...]
    max_depth_mm: float
    n_depths_evaluated: int
    scan_time_budget_s: float
    target_snr: float
    snr_per_shot_at_limit: float
    required_averages_at_limit: int
    scan_time_at_limit_s: float
    any_feasible: bool
    v1_depth_range_confirmed: bool
    snr_per_shot_at_5mm: float | None
    snr_per_shot_at_10mm: float | None
    snr_per_shot_at_15mm: float | None


# ──────────────────────────────────────────────────────────────────


def compute_depth_point(
    depth_mm: float,
    config: DepthLimitConfig,
    *,
    coil: SurfaceCoil,
    magnet: SingleSidedMagnet,
    tissue: TissueLayer,
) -> DepthPoint:
    """Evaluate the scan-time-gated depth limit at a single voxel depth.

    Calls :func:`~.snr_calculator.compute_snr_budget` with ``n_averages=1``
    to obtain the per-shot SNR, then applies

    .. math::

        N_\\text{avg} = \\left\\lceil
            \\left(\\frac{\\text{target\\_snr}}{\\text{snr\\_per\\_shot}}\\right)^2
        \\right\\rceil

    and checks whether the resulting scan time fits within the budget.

    Args:
        depth_mm: Voxel depth from the coil surface (mm).
        config: Sweep configuration.
        coil: Surface coil model.
        magnet: Single-sided magnet model.
        tissue: Tissue layer model.

    Returns:
        :class:`DepthPoint` with all metrics for this depth.
    """
    budget = compute_snr_budget(
        depth_mm,
        config.voxel_size_mm,
        coil=coil,
        magnet=magnet,
        tissue=tissue,
        tr_ms=config.tr_ms,
        te_ms=config.te_ms,
        sequence=config.sequence,
        bandwidth_hz=config.bandwidth_hz,
        n_averages=1,
    )
    snr_per_shot = budget.snr_per_shot

    if snr_per_shot <= 0.0:
        required_averages = int(1e9)  # effectively infinite
    else:
        ratio = config.target_snr / snr_per_shot
        required_averages = max(1, math.ceil(ratio**2))

    scan_time_s = required_averages * config.tr_ms / 1000.0
    within_budget = scan_time_s <= config.scan_time_budget_s

    return DepthPoint(
        depth_mm=depth_mm,
        snr_per_shot=snr_per_shot,
        required_averages=required_averages,
        scan_time_s=scan_time_s,
        within_budget=within_budget,
    )


def compute_depth_limit(
    config: DepthLimitConfig | None = None,
    *,
    coil: SurfaceCoil,
    magnet: SingleSidedMagnet,
    tissue: TissueLayer,
) -> DepthLimitResult:
    """Scan-time-gated depth-limit sweep over a range of voxel depths.

    Evaluates :class:`DepthPoint` at each depth in the range
    ``[config.min_depth_mm, config.max_depth_mm]`` in steps of
    ``config.depth_step_mm``, then identifies the deepest feasible voxel.

    Args:
        config: Sweep configuration.  Uses :class:`DepthLimitConfig` defaults
            when *None*.
        coil: Surface coil model.
        magnet: Single-sided magnet model.
        tissue: Tissue layer model.

    Returns:
        :class:`DepthLimitResult` summarising the full depth profile and the
        scan-time-budget-limited maximum depth.
    """
    if config is None:
        config = DepthLimitConfig()

    # Build depth sweep with integer-step indexing to avoid float accumulation.
    n_steps = round(
        (config.max_depth_mm - config.min_depth_mm) / config.depth_step_mm
    )
    profile: list[DepthPoint] = []
    for i in range(n_steps + 1):
        d = config.min_depth_mm + i * config.depth_step_mm
        profile.append(
            compute_depth_point(d, config, coil=coil, magnet=magnet, tissue=tissue)
        )

    depth_profile = tuple(profile)

    feasible = [p for p in depth_profile if p.within_budget]
    any_feasible = bool(feasible)
    max_depth_mm = max((p.depth_mm for p in feasible), default=0.0)

    # Limit-point metrics: deepest feasible, or shallowest when none feasible.
    limit = (
        max(feasible, key=lambda p: p.depth_mm)
        if any_feasible
        else depth_profile[0]
    )

    # V1 design validation: every depth in [5, 15] mm must be within budget.
    v1_points = [p for p in depth_profile if 5.0 <= p.depth_mm <= 15.0]
    v1_depth_range_confirmed = bool(
        v1_points and all(p.within_budget for p in v1_points)
    )

    # Reference per-shot SNR at three clinically relevant depths.
    snr_map: dict[float, float] = {p.depth_mm: p.snr_per_shot for p in depth_profile}
    snr_per_shot_at_5mm: float | None = snr_map.get(5.0)
    snr_per_shot_at_10mm: float | None = snr_map.get(10.0)
    snr_per_shot_at_15mm: float | None = snr_map.get(15.0)

    return DepthLimitResult(
        depth_profile=depth_profile,
        max_depth_mm=max_depth_mm,
        n_depths_evaluated=len(depth_profile),
        scan_time_budget_s=config.scan_time_budget_s,
        target_snr=config.target_snr,
        snr_per_shot_at_limit=limit.snr_per_shot,
        required_averages_at_limit=limit.required_averages,
        scan_time_at_limit_s=limit.scan_time_s,
        any_feasible=any_feasible,
        v1_depth_range_confirmed=v1_depth_range_confirmed,
        snr_per_shot_at_5mm=snr_per_shot_at_5mm,
        snr_per_shot_at_10mm=snr_per_shot_at_10mm,
        snr_per_shot_at_15mm=snr_per_shot_at_15mm,
    )
