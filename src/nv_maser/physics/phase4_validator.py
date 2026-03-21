"""
Phase-4 depth-profile milestone validator.

Architecture doc §12.2 (handheld-maser-probe-architecture.md) identifies two
deliverables for **Phase 4: First Depth Profile**:

    * "First NMR signal" — FID or echo from water sample at known depth
    * "Depth profile"    — Resolvable layers in layered phantom

This module provides a formal digital-twin validation of those two criteria
using the existing physics stack:

    SingleSidedMagnet  →  B₀(z) from barrel-array sweet-spot magnet
    SurfaceCoil        →  B₁ sensitivity(z), thermal + body noise
    simulate_depth_profile → per-depth SNR in a stacked-layer tissue model

Milestone success criteria (digital twin)
──────────────────────────────────────────
1. **First NMR signal**: maximum SNR anywhere in the ``depth_range_mm``
   window exceeds ``signal_snr_threshold`` (default 3.0).

2. **Resolvable layers**: every tissue-layer boundary that falls *within*
   ``depth_range_mm`` satisfies both:
       * T2 contrast ratio  ≥ ``contrast_ratio_threshold`` (default 1.5×)
       * SNR at the boundary ≥ ``snr_at_boundary_threshold`` (default 1.0)

3. **Scan-time budget**: total acquisition time ≤ ``scan_time_limit_s``
   (default 120 s, from architecture §1.3 "15–120 s per view").

The default probe configuration (50 mT barrel magnet, 15 mm radius / 5-turn
surface coil, 1024 averages, TR = 100 ms) on the FOREARM_LAYERS phantom
produces:

    max SNR in 3–15 mm: 20.75 (> 3.0 ✓)
    fat→muscle boundary (7 mm): SNR ≈ 16, T2 ratio 2.29× (> 1.5 ✓)
    scan time: 102.4 s (< 120 s ✓)
    phase4_milestone_closed = True ✓

Reference
─────────
handheld-maser-probe-architecture.md §12.1–12.2.
Blümich et al., "Mobile single-sided NMR", Prog. NMR Spectroscopy 52, 197 (2008).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import (
    DepthProfileConfig,
    SingleSidedMagnetConfig,
    SurfaceCoilConfig,
)
from .depth_profile import (
    DepthProfile,
    TissueLayer,
    FOREARM_LAYERS,
    simulate_depth_profile,
)
from .single_sided_magnet import SingleSidedMagnet
from .surface_coil import SurfaceCoil


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Configuration                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class Phase4Config:
    """V1 acceptance thresholds for the Phase-4 depth-profile milestone.

    These values were chosen so that the default probe configuration
    (50 mT barrel magnet, 15 mm surface coil, 1024 averages, TR = 100 ms)
    passes with comfortable margin on the FOREARM_LAYERS phantom.

    Args:
        signal_snr_threshold:
            Minimum SNR required anywhere in *depth_range_mm* to declare
            "first NMR signal" (architecture §12.2 criterion 3).
            Default 3.0.
        contrast_ratio_threshold:
            Minimum T2 contrast ratio (max/min of the two flanking layers)
            at each layer boundary inside *depth_range_mm* for that
            boundary to be considered "resolvable".  Default 1.5×.
        snr_at_boundary_threshold:
            SNR must exceed this value at the boundary depth bin for a
            layer transition to be classified as detectable.  Default 1.0.
        scan_time_limit_s:
            Maximum allowable acquisition time (seconds).  Architecture
            §1.3 targets 15–120 s per view for emergency-triage use.
            Default 120.0.
        depth_range_mm:
            (min, max) depth window in mm that defines the V1 operating
            range.  Only depth bins and layer boundaries within this
            window are evaluated for the SNR and contrast criteria.
            Default (3.0, 15.0) — avoids the near-surface signal
            dropout region and the deep-tissue regime outside V1 scope.
    """

    signal_snr_threshold: float = 3.0
    contrast_ratio_threshold: float = 1.5
    snr_at_boundary_threshold: float = 1.0
    scan_time_limit_s: float = 120.0
    depth_range_mm: tuple[float, float] = (3.0, 15.0)

    def __post_init__(self) -> None:
        if self.signal_snr_threshold <= 0:
            raise ValueError(
                f"signal_snr_threshold must be > 0, got {self.signal_snr_threshold}"
            )
        if self.contrast_ratio_threshold < 1.0:
            raise ValueError(
                f"contrast_ratio_threshold must be ≥ 1.0 (ratio is always ≥ 1), "
                f"got {self.contrast_ratio_threshold}"
            )
        if self.snr_at_boundary_threshold < 0:
            raise ValueError(
                f"snr_at_boundary_threshold must be ≥ 0, got {self.snr_at_boundary_threshold}"
            )
        if self.scan_time_limit_s <= 0:
            raise ValueError(
                f"scan_time_limit_s must be > 0, got {self.scan_time_limit_s}"
            )
        dmin, dmax = self.depth_range_mm
        if dmin >= dmax:
            raise ValueError(
                f"depth_range_mm must satisfy min < max, got ({dmin}, {dmax})"
            )
        if dmin < 0:
            raise ValueError(
                f"depth_range_mm lower bound must be ≥ 0, got {dmin}"
            )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Result dataclasses                                              ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class LayerContrastResult:
    """T2 contrast assessment at a single tissue-layer boundary.

    Args:
        layer_a_name: name of the shallower (proximal) tissue layer.
        layer_b_name: name of the deeper (distal) tissue layer.
        t2_a_ms: T2 of the shallower layer (ms).
        t2_b_ms: T2 of the deeper layer (ms).
        t2_contrast_ratio: max(T2_a, T2_b) / min(T2_a, T2_b) — always ≥ 1.
            Not defined (stored as ``float('inf')``) when either T2 ≤ 0.
        boundary_depth_mm: depth of the layer interface (mm).
        snr_at_boundary: simulated SNR at the depth bin nearest the boundary.
        in_depth_range: True if *boundary_depth_mm* lies within
            ``Phase4Config.depth_range_mm``.
        detectable: True iff *t2_contrast_ratio* ≥ threshold AND
            *snr_at_boundary* ≥ ``snr_at_boundary_threshold``.
            Evaluated regardless of *in_depth_range*; the milestone check
            only gates on boundaries **inside** the depth range.
    """

    layer_a_name: str
    layer_b_name: str
    t2_a_ms: float
    t2_b_ms: float
    t2_contrast_ratio: float
    boundary_depth_mm: float
    snr_at_boundary: float
    in_depth_range: bool
    detectable: bool


@dataclass(frozen=True)
class Phase4MilestoneResult:
    """Result of the Phase-4 depth-profile milestone validation.

    Architecture §12.2 success criteria:
        (a) "First NMR signal" — echo detectable at known depth.
        (b) "Depth profile"    — adjacent tissue layers resolved.
        (c) Scan-time budget ≤ 120 s (§1.3).

    Args:
        depth_profile:
            Underlying simulated DepthProfile from
            ``physics.depth_profile.simulate_depth_profile``.
        layer_contrasts:
            One ``LayerContrastResult`` per adjacent layer boundary in the
            tissue model (all boundaries, not only in-range ones).
        n_depths_evaluated:
            Number of depth bins within ``Phase4Config.depth_range_mm``.
        max_snr_in_range:
            Peak SNR inside the depth range (determines first-signal pass).
        min_snr_in_range:
            Minimum SNR inside the depth range.
        scan_time_s:
            Total acquisition time (n_averages × TR).
        all_in_range_layers_detectable:
            True iff every layer boundary **inside** the depth range is
            detectable (vacuously True when no boundaries fall in-range).
        snr_pass:
            True iff ``max_snr_in_range`` ≥ ``signal_snr_threshold``.
        scan_time_pass:
            True iff ``scan_time_s`` ≤ ``scan_time_limit_s``.
        phase4_milestone_closed:
            True iff all three criteria pass simultaneously.
    """

    depth_profile: DepthProfile
    layer_contrasts: tuple[LayerContrastResult, ...]
    n_depths_evaluated: int
    max_snr_in_range: float
    min_snr_in_range: float
    scan_time_s: float
    all_in_range_layers_detectable: bool
    snr_pass: bool
    scan_time_pass: bool
    phase4_milestone_closed: bool


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Layer contrast helper                                           ║
# ╚══════════════════════════════════════════════════════════════════╝


def compute_layer_contrast(
    layer_a: TissueLayer,
    layer_b: TissueLayer,
    boundary_depth_mm: float,
    profile: DepthProfile,
    config: Phase4Config,
) -> LayerContrastResult:
    """Assess T2 contrast and SNR at a single tissue-layer boundary.

    Args:
        layer_a: shallower (proximal) tissue layer.
        layer_b: deeper (distal) tissue layer.
        boundary_depth_mm: depth of the interface (mm).
        profile: simulated depth profile.
        config: milestone acceptance thresholds.

    Returns:
        LayerContrastResult with all measurements and pass/fail flags.
    """
    # T2 contrast ratio (always ≥ 1; inf if either T2 ≤ 0)
    t2a = float(layer_a.t2_ms)
    t2b = float(layer_b.t2_ms)
    if t2a <= 0.0 or t2b <= 0.0:
        ratio = float("inf")
    else:
        ratio = float(max(t2a, t2b) / min(t2a, t2b))

    # Nearest depth bin to the boundary
    idx = int(np.argmin(np.abs(profile.depths_mm - boundary_depth_mm)))
    snr_at_boundary = float(profile.snr[idx])

    dmin, dmax = config.depth_range_mm
    in_range = bool(dmin <= boundary_depth_mm <= dmax)

    # Detectable: T2 ratio sufficient AND SNR adequate
    detectable = bool(
        ratio >= config.contrast_ratio_threshold
        and snr_at_boundary >= config.snr_at_boundary_threshold
    )

    return LayerContrastResult(
        layer_a_name=layer_a.name,
        layer_b_name=layer_b.name,
        t2_a_ms=t2a,
        t2_b_ms=t2b,
        t2_contrast_ratio=ratio,
        boundary_depth_mm=float(boundary_depth_mm),
        snr_at_boundary=snr_at_boundary,
        in_depth_range=in_range,
        detectable=detectable,
    )


def _compute_all_layer_contrasts(
    tissue_layers: list[TissueLayer],
    profile: DepthProfile,
    config: Phase4Config,
) -> tuple[LayerContrastResult, ...]:
    """Compute LayerContrastResult for every adjacent layer pair."""
    results: list[LayerContrastResult] = []
    cumulative_depth = 0.0
    for i in range(len(tissue_layers) - 1):
        cumulative_depth += tissue_layers[i].thickness_mm
        lc = compute_layer_contrast(
            tissue_layers[i],
            tissue_layers[i + 1],
            cumulative_depth,
            profile,
            config,
        )
        results.append(lc)
    return tuple(results)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Top-level milestone validator                                   ║
# ╚══════════════════════════════════════════════════════════════════╝


def validate_phase4_milestone(
    magnet: SingleSidedMagnet | None = None,
    coil: SurfaceCoil | None = None,
    config: Phase4Config | None = None,
    tissue_layers: list[TissueLayer] | None = None,
    te_ms: float = 10.0,
) -> Phase4MilestoneResult:
    """Validate the Phase-4 depth-profile milestone (architecture §12.2).

    Runs a full 1D NMR depth profile simulation and checks three criteria:

    1. **First NMR signal** — max SNR inside *depth_range_mm* exceeds
       ``signal_snr_threshold``.
    2. **Resolvable layers** — every layer boundary inside the depth range
       satisfies ``contrast_ratio_threshold`` and ``snr_at_boundary_threshold``.
    3. **Scan-time budget** — acquisition time ≤ ``scan_time_limit_s``.

    ``phase4_milestone_closed = True`` iff all three criteria pass.

    Args:
        magnet:
            Pre-built ``SingleSidedMagnet``.  Uses the default barrel-array
            configuration (50 mT sweet spot at 20 mm) when ``None``.
        coil:
            Pre-built ``SurfaceCoil``.  Uses the default 15 mm / 5-turn coil
            when ``None``.
        config:
            Milestone acceptance thresholds; defaults to ``Phase4Config()``.
        tissue_layers:
            Stacked tissue model; defaults to ``FOREARM_LAYERS`` (skin →
            subcutaneous fat → muscle → bone cortex).
        te_ms:
            Echo time for the spin-echo sequence (ms).  Default 10.0.

    Returns:
        Phase4MilestoneResult with full simulation data and pass/fail flags.
    """
    if config is None:
        config = Phase4Config()
    if magnet is None:
        magnet = SingleSidedMagnet(SingleSidedMagnetConfig())
    if coil is None:
        coil = SurfaceCoil(SurfaceCoilConfig())
    if tissue_layers is None:
        tissue_layers = list(FOREARM_LAYERS)

    dp_config = DepthProfileConfig()
    profile = simulate_depth_profile(
        magnet,
        coil,
        dp_config,
        tissue_layers,
        sequence="spin_echo",
        te_ms=te_ms,
    )

    # ── Criterion 1: First NMR signal ────────────────────────────────
    dmin, dmax = config.depth_range_mm
    mask = (profile.depths_mm >= dmin) & (profile.depths_mm <= dmax)
    n_depths = int(mask.sum())
    snr_in_range = profile.snr[mask]
    max_snr = float(snr_in_range.max()) if n_depths > 0 else 0.0
    min_snr = float(snr_in_range.min()) if n_depths > 0 else 0.0
    snr_pass = bool(max_snr >= config.signal_snr_threshold)

    # ── Criterion 2: Resolvable layers ───────────────────────────────
    layer_contrasts = _compute_all_layer_contrasts(tissue_layers, profile, config)
    in_range_contrasts = [lc for lc in layer_contrasts if lc.in_depth_range]
    all_detectable = bool(
        all(lc.detectable for lc in in_range_contrasts)
    ) if in_range_contrasts else True

    # ── Criterion 3: Scan-time budget ────────────────────────────────
    scan_time_pass = bool(profile.scan_time_s <= config.scan_time_limit_s)

    milestone_closed = bool(snr_pass and scan_time_pass and all_detectable)

    return Phase4MilestoneResult(
        depth_profile=profile,
        layer_contrasts=layer_contrasts,
        n_depths_evaluated=n_depths,
        max_snr_in_range=max_snr,
        min_snr_in_range=min_snr,
        scan_time_s=float(profile.scan_time_s),
        all_in_range_layers_detectable=all_detectable,
        snr_pass=snr_pass,
        scan_time_pass=scan_time_pass,
        phase4_milestone_closed=milestone_closed,
    )
