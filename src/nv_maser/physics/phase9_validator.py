"""Phase-9 tissue-contrast milestone validator (architecture §12.2).

Criterion (architecture §12.2, Phase 9)
----------------------------------------
    "Tissue contrast — T2 difference visible between fat and muscle"

The NV-maser probe must produce T2-weighted depth profiles in which
subcutaneous fat and skeletal muscle appear as distinguishably different
grey levels, demonstrating clinical-grade tissue discrimination without a
static magnetic-field enclosure.

Success criteria
----------------
Three criteria must all pass simultaneously:

1. **T2-weighted contrast visible** — the ratio
       fat_signal_long / muscle_signal_long ≥ ``t2_contrast_ratio_threshold``
   where *fat_signal_long* and *muscle_signal_long* are the simulated NMR
   signals at the fat-layer centre and muscle-layer centre respectively, both
   measured at the T2-weighting echo time ``te_long_ms``.

   Physical basis: at ``te_long_ms`` ≈ T2_fat = 80 ms the fat signal is
   attenuated to ≈ exp(−1) ≈ 37 % of its TE=0 amplitude while muscle
   (T2 = 35 ms) is attenuated to ≈ exp(−2.3) ≈ 10 %.  The resulting
   signal ratio exceeds 3× comfortably, well above the 1.5× threshold.

2. **Adequate SNR in both tissue compartments** — SNR at the fat-layer centre
   AND SNR at the muscle-layer centre must both exceed ``snr_threshold``
   in the *short-TE* reference acquisition (``te_short_ms`` = 10 ms).
   This verifies that both tissues are imageable before T2 weighting is
   applied, which is the physically correct precondition for claiming that
   the contrast is genuinely "visible" rather than noise-dominated.

3. **Scan-time budget** — the single-acquisition scan time must not exceed
   ``scan_time_limit_s`` (default: 120 s, matching the architecture §1.3
   emergency-triage target).  Only the long-TE acquisition governs this
   criterion; the short-TE scan is the same duration in hardware but is
   treated as a calibration step.

``phase9_milestone_closed = True`` iff all three criteria pass.

References
----------
* Architecture doc §12.2 milestone table.
* Architecture doc §1.3 (scan-time target 15–120 s).
* Blümich et al., "Mobile single-sided NMR", Prog. NMR Spectroscopy 52, 197 (2008).
* Brownstein & Tarr, "Importance of classical diffusion in NMR studies of
  water in biological cells", Phys. Rev. A 19, 2446 (1979).
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
    FOREARM_LAYERS,
    DepthProfile,
    TissueLayer,
    simulate_depth_profile,
)
from .single_sided_magnet import SingleSidedMagnet
from .surface_coil import SurfaceCoil

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Configuration                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class Phase9Config:
    """Acceptance thresholds for the Phase-9 tissue-contrast milestone.

    These values were chosen so that the default probe configuration (50 mT
    barrel magnet, 15 mm/5-turn surface coil, 1024 averages, TR = 100 ms)
    passes with comfortable margin on the FOREARM_LAYERS phantom
    (skin → subcutaneous fat → muscle → bone cortex).

    Args:
        te_short_ms:
            Echo time for the reference (short-TE) acquisition (ms).  Both
            fat and muscle should appear bright.  Default 10.0 ms.
        te_long_ms:
            Echo time for the T2-weighted acquisition (ms).  Fat
            (T2 ≈ 80 ms) remains relatively bright; muscle (T2 ≈ 35 ms)
            is substantially attenuated.  Default 80.0 ms.
        t2_contrast_ratio_threshold:
            Minimum fat/muscle signal ratio at ``te_long_ms`` required to
            declare T2 contrast "visible".  Default 1.5×.
        snr_threshold:
            Minimum SNR (linear ratio, same units as ``DepthProfile.snr``)
            required at both the fat-layer centre and the muscle-layer
            centre in the *short-TE* profile.  Default 3.0.
        scan_time_limit_s:
            Maximum scan time for the long-TE acquisition (seconds).
            Architecture §1.3 targets 15–120 s.  Default 120.0.
        fat_center_depth_mm:
            Nominal depth of the fat-layer centre in the FOREARM phantom
            (mm).  Default 4.5 mm  (skin = 2 mm → fat = 5 mm; centre at
            2 + 2.5 = 4.5 mm).
        muscle_center_depth_mm:
            Nominal depth inside the muscle layer used for the contrast
            measurement (mm).  Default 10.0 mm  (muscle begins at 7 mm;
            10.0 mm is 3 mm past the fat–muscle boundary, well within the
            muscle compartment and still within the probe's sweet-spot
            sensitivity depth range).
    """

    te_short_ms: float = 10.0
    te_long_ms: float = 80.0
    t2_contrast_ratio_threshold: float = 1.5
    snr_threshold: float = 3.0
    scan_time_limit_s: float = 120.0
    fat_center_depth_mm: float = 4.5
    muscle_center_depth_mm: float = 10.0

    def __post_init__(self) -> None:
        if self.te_short_ms <= 0:
            raise ValueError(
                f"te_short_ms must be > 0, got {self.te_short_ms}"
            )
        if self.te_long_ms <= self.te_short_ms:
            raise ValueError(
                f"te_long_ms ({self.te_long_ms}) must be > te_short_ms "
                f"({self.te_short_ms})"
            )
        if self.t2_contrast_ratio_threshold < 1.0:
            raise ValueError(
                f"t2_contrast_ratio_threshold must be ≥ 1.0, "
                f"got {self.t2_contrast_ratio_threshold}"
            )
        if self.snr_threshold <= 0:
            raise ValueError(
                f"snr_threshold must be > 0, got {self.snr_threshold}"
            )
        if self.scan_time_limit_s <= 0:
            raise ValueError(
                f"scan_time_limit_s must be > 0, got {self.scan_time_limit_s}"
            )
        if self.fat_center_depth_mm <= 0:
            raise ValueError(
                f"fat_center_depth_mm must be > 0, got {self.fat_center_depth_mm}"
            )
        if self.muscle_center_depth_mm <= self.fat_center_depth_mm:
            raise ValueError(
                f"muscle_center_depth_mm ({self.muscle_center_depth_mm}) must be "
                f"> fat_center_depth_mm ({self.fat_center_depth_mm})"
            )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Result dataclasses                                              ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class T2ContrastResult:
    """T2-weighted contrast measurement between fat and muscle.

    Args:
        fat_depth_mm:
            Actual depth bin selected for the fat measurement (mm).
        muscle_depth_mm:
            Actual depth bin selected for the muscle measurement (mm).
        fat_signal_short:
            Simulated NMR signal at the fat depth at ``te_short_ms``.
        muscle_signal_short:
            Simulated NMR signal at the muscle depth at ``te_short_ms``.
        fat_signal_long:
            Simulated NMR signal at the fat depth at ``te_long_ms``.
        muscle_signal_long:
            Simulated NMR signal at the muscle depth at ``te_long_ms``.
        t2_contrast_ratio:
            ``fat_signal_long / muscle_signal_long``.  Always ≥ 0; stored
            as ``float('inf')`` when muscle signal is zero or negative.
        passes:
            ``True`` iff ``t2_contrast_ratio`` ≥ the configured threshold.
    """

    fat_depth_mm: float
    muscle_depth_mm: float
    fat_signal_short: float
    muscle_signal_short: float
    fat_signal_long: float
    muscle_signal_long: float
    t2_contrast_ratio: float
    passes: bool


@dataclass(frozen=True)
class Phase9MilestoneResult:
    """Result of the Phase-9 tissue-contrast milestone validation.

    Architecture §12.2 success criterion:
        "T2 difference visible between fat and muscle."

    Three sub-criteria must pass simultaneously:
        (a) T2-weighted signal ratio ≥ ``t2_contrast_ratio_threshold``.
        (b) SNR ≥ ``snr_threshold`` at both tissue depths (short-TE image).
        (c) Scan-time budget ≤ ``scan_time_limit_s``.

    Args:
        config:
            Acceptance thresholds used for this validation run.
        profile_short:
            Depth profile simulated at ``te_short_ms`` (reference image).
        profile_long:
            Depth profile simulated at ``te_long_ms`` (T2-weighted image).
        t2_contrast:
            Per-compartment signal measurements and the T2 contrast result.
        fat_snr:
            SNR at the fat depth bin in the *short-TE* profile (linear).
        muscle_snr:
            SNR at the muscle depth bin in the *short-TE* profile (linear).
        contrast_pass:
            ``True`` iff ``t2_contrast.passes`` is True.
        snr_pass:
            ``True`` iff both ``fat_snr`` and ``muscle_snr`` ≥ ``snr_threshold``.
        scan_time_s:
            Long-TE acquisition scan time (seconds).
        scan_time_pass:
            ``True`` iff ``scan_time_s`` ≤ ``scan_time_limit_s``.
        phase9_milestone_closed:
            ``True`` iff all three criteria pass simultaneously.
    """

    config: Phase9Config
    profile_short: DepthProfile
    profile_long: DepthProfile
    t2_contrast: T2ContrastResult
    fat_snr: float
    muscle_snr: float
    contrast_pass: bool
    snr_pass: bool
    scan_time_s: float
    scan_time_pass: bool
    phase9_milestone_closed: bool


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Top-level milestone validator                                   ║
# ╚══════════════════════════════════════════════════════════════════╝


def validate_phase9_milestone(
    magnet: SingleSidedMagnet | None = None,
    coil: SurfaceCoil | None = None,
    config: Phase9Config | None = None,
    tissue_layers: list[TissueLayer] | None = None,
) -> Phase9MilestoneResult:
    """Validate the Phase-9 tissue-contrast milestone (architecture §12.2).

    Runs two 1D NMR depth-profile simulations (short TE and long TE) and
    checks three criteria:

    1. **T2 contrast visible** — the fat/muscle signal ratio at ``te_long_ms``
       exceeds ``t2_contrast_ratio_threshold`` (default 1.5×).
    2. **Adequate SNR** — SNR at both the fat-layer centre and the
       muscle-layer centre in the *short-TE* profile exceeds
       ``snr_threshold`` (default 3.0).
    3. **Scan-time budget** — the long-TE acquisition time does not exceed
       ``scan_time_limit_s`` (default 120 s).

    ``phase9_milestone_closed = True`` iff all three criteria pass.

    Args:
        magnet:
            Pre-built ``SingleSidedMagnet``.  Uses the default barrel-array
            configuration (50 mT sweet spot at 20 mm) when ``None``.
        coil:
            Pre-built ``SurfaceCoil``.  Uses the default 15 mm / 5-turn coil
            when ``None``.
        config:
            Milestone acceptance thresholds; defaults to ``Phase9Config()``.
        tissue_layers:
            Stacked tissue phantom; defaults to ``FOREARM_LAYERS``
            (skin → subcutaneous fat → muscle → bone cortex).

    Returns:
        Phase9MilestoneResult with all simulation data and pass/fail flags.
    """
    if config is None:
        config = Phase9Config()
    if magnet is None:
        magnet = SingleSidedMagnet(SingleSidedMagnetConfig())
    if coil is None:
        coil = SurfaceCoil(SurfaceCoilConfig())
    if tissue_layers is None:
        tissue_layers = list(FOREARM_LAYERS)

    dp_cfg = DepthProfileConfig()

    # ── Short-TE reference: both fat and muscle should be bright ─────
    profile_short = simulate_depth_profile(
        magnet,
        coil,
        dp_cfg,
        tissue_layers,
        sequence="spin_echo",
        te_ms=config.te_short_ms,
    )

    # ── Long-TE T2-weighted: fat bright, muscle suppressed ───────────
    profile_long = simulate_depth_profile(
        magnet,
        coil,
        dp_cfg,
        tissue_layers,
        sequence="spin_echo",
        te_ms=config.te_long_ms,
    )

    # ── Locate nearest depth bins to nominal tissue centres ──────────
    fat_idx = int(np.argmin(np.abs(profile_long.depths_mm - config.fat_center_depth_mm)))
    muscle_idx = int(np.argmin(np.abs(profile_long.depths_mm - config.muscle_center_depth_mm)))

    fat_depth_mm = float(profile_long.depths_mm[fat_idx])
    muscle_depth_mm = float(profile_long.depths_mm[muscle_idx])

    # ── Extract signals at both TEs ──────────────────────────────────
    fat_signal_short = float(profile_short.signal[fat_idx])
    muscle_signal_short = float(profile_short.signal[muscle_idx])
    fat_signal_long = float(profile_long.signal[fat_idx])
    muscle_signal_long = float(profile_long.signal[muscle_idx])

    # ── Criterion 1: T2 contrast ratio ───────────────────────────────
    if muscle_signal_long > 0.0:
        ratio = fat_signal_long / muscle_signal_long
    else:
        ratio = float("inf")
    contrast_passes = bool(ratio >= config.t2_contrast_ratio_threshold)

    t2_contrast = T2ContrastResult(
        fat_depth_mm=fat_depth_mm,
        muscle_depth_mm=muscle_depth_mm,
        fat_signal_short=fat_signal_short,
        muscle_signal_short=muscle_signal_short,
        fat_signal_long=fat_signal_long,
        muscle_signal_long=muscle_signal_long,
        t2_contrast_ratio=ratio,
        passes=contrast_passes,
    )

    # ── Criterion 2: SNR at both tissue depths (short-TE) ────────────
    fat_snr = float(profile_short.snr[fat_idx])
    muscle_snr = float(profile_short.snr[muscle_idx])
    snr_pass = bool(
        fat_snr >= config.snr_threshold and muscle_snr >= config.snr_threshold
    )

    # ── Criterion 3: Scan-time budget ────────────────────────────────
    scan_time_s = float(profile_long.scan_time_s)
    scan_time_pass = bool(scan_time_s <= config.scan_time_limit_s)

    milestone_closed = bool(contrast_passes and snr_pass and scan_time_pass)

    return Phase9MilestoneResult(
        config=config,
        profile_short=profile_short,
        profile_long=profile_long,
        t2_contrast=t2_contrast,
        fat_snr=fat_snr,
        muscle_snr=muscle_snr,
        contrast_pass=contrast_passes,
        snr_pass=snr_pass,
        scan_time_s=scan_time_s,
        scan_time_pass=scan_time_pass,
        phase9_milestone_closed=milestone_closed,
    )
