"""
Handheld NMR/MRI probe — integrated simulation.

Combines all probe subsystems into a unified model:

    SingleSidedMagnet  →  B₀ field + sweet spot
    SurfaceCoil        →  B₁ sensitivity + noise
    DepthProfile       →  1D tissue profiling (A-mode equivalent)
    PlanarGradient     →  lateral phase encoding
    Reconstruction     →  k-space → 2D image
    SNRCalculator      →  end-to-end SNR budget + parametric sweeps

Stray-field shielding
─────────────────────
The imaging magnet (0.5–1 kg NdFeB) produces a fringe field at the maser
module location.  ``compute_stray_field_rms()`` quantifies this so that
required shim headroom can be verified.  The architecture doc targets a
residual stray field of < 100 µT (≈ 0.1 mT) on the maser grid after
50 dB mu-metal shielding.

    Without shielding at 40 mm:  ~30 mT   (dipole at 40 mm broadside)
    After 50 dB shielding:       ~0.1 mT  ← architecture target

``sweep_stray_field_vs_separation()`` maps this vs. probe width to help
choose the mechanical layout.

References
──────────
Handheld probe architecture document, §11.1–11.3, §12.
Blümich et al., "Mobile single-sided NMR", Prog. NMR Spectroscopy 52, 197 (2008).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..config import (
    SingleSidedMagnetConfig,
    SurfaceCoilConfig,
    DepthProfileConfig,
)
from .single_sided_magnet import SingleSidedMagnet
from .surface_coil import SurfaceCoil, sensitivity_on_axis
from .depth_profile import (
    TissueLayer,
    DepthProfile,
    simulate_depth_profile,
    FOREARM_LAYERS,
)
from .planar_gradient import (
    GradientCoilSpec,
    DEFAULT_GX,
    DEFAULT_GY,
    build_phase_encode_scheme,
    PhaseEncodeScheme,
)
from .snr_calculator import (
    SNRBudget,
    compute_snr_budget,
    snr_vs_depth,
    snr_vs_averages,
)

# ── Physical constants ────────────────────────────────────────────
from .constants import MU0 as _MU0
_GAMMA_P_HZ_PER_T = 42.577e6    # proton gyromagnetic ratio (Hz/T)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Configuration                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class ProbeConfig:
    """Configuration for the complete handheld NMR/MRI probe.

    Aggregates component configs and scan-protocol parameters into one
    object that can be passed to all probe simulation functions.

    Args:
        magnet:
            Single-sided magnet configuration.  Default is a 4-ring barrel
            array targeting a 20 mm sweet spot at 50 mT.
        coil:
            Surface coil configuration.  Default is a 15 mm radius, 5-turn
            flat spiral at room temperature.
        gradient_x:
            Gradient coil spec for the x (phase-encode) direction.
        gradient_y:
            Gradient coil spec for the y (frequency-encode) direction.
        pulse_sequence:
            Excitation / echo sequence: ``"spin_echo"`` or ``"gre"``.
        tr_ms:
            Repetition time (ms).
        te_ms:
            Echo time (ms).  Must satisfy TE < TR.
        bandwidth_hz:
            Readout detection bandwidth (Hz).
        n_averages:
            Number of signal averages (NEX).  SNR scales as √NEX.
        n_phase_lines:
            Number of k-space phase-encoding lines for 2D imaging.
        fov_m:
            Lateral field of view (m).  Determines lateral resolution.
        target_depth_mm:
            Nominal imaging depth (mm); used for SNR and resolution estimates.
        maser_separation_mm:
            Lateral distance from the imaging magnet surface to the maser
            module centre (mm).  Used for stray-field calculation.
        shield_attenuation_db:
            Mu-metal shielding attenuation (dB) applied to stray field.
            Architecture doc target: 50 dB.
    """

    magnet: SingleSidedMagnetConfig = field(
        default_factory=SingleSidedMagnetConfig
    )
    coil: SurfaceCoilConfig = field(
        default_factory=SurfaceCoilConfig
    )
    gradient_x: GradientCoilSpec = field(
        default_factory=lambda: DEFAULT_GX
    )
    gradient_y: GradientCoilSpec = field(
        default_factory=lambda: DEFAULT_GY
    )
    pulse_sequence: str = "spin_echo"
    tr_ms: float = 500.0
    te_ms: float = 10.0
    bandwidth_hz: float = 10_000.0
    n_averages: int = 4
    n_phase_lines: int = 64
    fov_m: float = 0.08         # 8 cm FOV
    target_depth_mm: float = 20.0
    maser_separation_mm: float = 50.0
    shield_attenuation_db: float = 50.0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Result dataclass                                                ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class ProbePerformanceReport:
    """Comprehensive performance summary of the handheld probe.

    Produced by :func:`compute_probe_performance` for a given
    :class:`ProbeConfig` and representative tissue layer.

    Attributes
    ----------
    sweet_spot_depth_mm:
        Depth of the B₀ saddle point (mm).
    sweet_spot_b0_tesla:
        B₀ field strength at the sweet spot (T).
    sweet_spot_larmor_mhz:
        Proton Larmor frequency at the sweet spot (MHz).
    sweet_spot_uniformity_ppm:
        Field uniformity over a 10 mm sphere at the sweet spot (ppm).
    coil_sensitivity_t_per_a:
        Surface coil B₁/I at the target depth (T/A).
    depth_gradient_t_per_m:
        |∂B₀/∂z| at the sweet spot (T/m) — drives depth resolution.
    depth_resolution_mm:
        Depth resolution: δz = BW / (γ |∂B/∂z|) (mm).
    lateral_resolution_mm:
        Lateral resolution imposed by the phase-encode scheme (mm).
    scan_time_s:
        Total 2D scan time = n_phase_lines × n_averages × TR (s).
    snr_at_target_db:
        SNR (dB) at the target depth with n_averages averages.
    snr_budget:
        Full SNRBudget breakdown from snr_calculator.
    stray_field_rms_mt:
        RMS stray field from the imaging magnet on the maser grid (mT),
        after applying the configured shielding attenuation.
    """

    sweet_spot_depth_mm: float
    sweet_spot_b0_tesla: float
    sweet_spot_larmor_mhz: float
    sweet_spot_uniformity_ppm: float
    coil_sensitivity_t_per_a: float
    depth_gradient_t_per_m: float
    depth_resolution_mm: float
    lateral_resolution_mm: float
    scan_time_s: float
    snr_at_target_db: float
    snr_budget: SNRBudget
    stray_field_rms_mt: float


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Stray-field utility                                             ║
# ╚══════════════════════════════════════════════════════════════════╝


def compute_stray_field_rms(
    magnet: SingleSidedMagnet,
    maser_separation_mm: float,
    *,
    shield_attenuation_db: float = 0.0,
    grid_extent_mm: float = 10.0,
    grid_size: int = 16,
) -> float:
    """Compute the RMS stray field from the imaging magnet at the maser module.

    The imaging magnet is offset laterally by ``maser_separation_mm`` from
    the maser module centre (i.e. they are side by side in the probe).  The
    maser module occupies a small square grid of side ``grid_extent_mm``.

    **Dipole model:**  For distances ≫ magnet size the field falls as a
    magnetic dipole:

        |B| = (µ₀/4π) m / r³   (broadside, dz = 0)

    with m = Br × V / µ₀ where V is the total magnet ring volume.

    **Physical values:**
    - Default magnet (4 rings, Br = 1.45 T):  V ≈ 1.8 × 10⁻⁴ m³, m ≈ 166 A·m²
    - At 50 mm separation:  |B| ≈ (1e-7 × 166) / (0.05³) ≈ 1.33 T — note
      the dipole approximation overestimates at close range where the magnet
      is no longer point-like.  Use ``grid_extent_mm`` ≪ ``maser_separation_mm``
      for accuracy.

    Args:
        magnet: :class:`SingleSidedMagnet` instance.  Its ring geometry
            determines the effective magnetic volume.
        maser_separation_mm: Lateral distance between probe face and maser
            module centre (mm).
        shield_attenuation_db: Mu-metal shielding attenuation (dB).
            50 dB ≈ factor 316 field reduction.  0 = no shield.
        grid_extent_mm: Side length of the maser grid for RMS averaging (mm).
            Should be ≪ maser_separation_mm.
        grid_size: Grid points per axis for the RMS integral.

    Returns:
        RMS stray field in milli-Tesla (mT) at the maser module grid.
    """
    # ── Estimate dipole moment from ring geometry ─────────────────
    c = magnet.config
    total_volume_m3 = 0.0
    for i in range(c.num_rings):
        r_in = c.ring_inner_radii_mm[i] * 1e-3
        r_out = c.ring_outer_radii_mm[i] * 1e-3
        h = c.ring_heights_mm[i] * 1e-3
        total_volume_m3 += math.pi * (r_out**2 - r_in**2) * h

    m = c.remanence_tesla * total_volume_m3 / _MU0   # A·m²

    # ── Small grid at the maser location ─────────────────────────
    half = grid_extent_mm / 2.0 * 1e-3  # metres
    axis = np.linspace(-half, half, grid_size)
    gx, gy = np.meshgrid(axis, axis, indexing="xy")

    # Maser is at origin; imaging magnet dipole at (x0, 0, 0)
    x0 = maser_separation_mm * 1e-3  # metres
    dx = gx - x0
    dy = gy
    dz = 0.0   # same plane (conservative: maximises stray field)

    r2 = dx**2 + dy**2 + dz**2
    r = np.sqrt(r2)
    safe = r > 1e-9

    # Total |B| for a z-oriented dipole at broadside (dz=0):
    #   |B| = (µ₀/4π) m / r³  (exact for dz=0)
    # For dz ≠ 0 the general expression is used.
    r3 = np.where(safe, r**3, 1.0)
    b_total = np.where(safe, (_MU0 / (4.0 * math.pi)) * m / r3, 0.0)

    if shield_attenuation_db > 0.0:
        attenuation = 10.0 ** (-shield_attenuation_db / 20.0)
        b_total = b_total * attenuation

    rms_tesla = float(np.sqrt(np.mean(b_total**2)))
    return rms_tesla * 1e3   # → mT


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Core probe performance function                                 ║
# ╚══════════════════════════════════════════════════════════════════╝


def compute_probe_performance(
    probe_config: ProbeConfig | None = None,
    *,
    tissue: TissueLayer | None = None,
) -> ProbePerformanceReport:
    """Compute the complete performance envelope of the handheld probe.

    Evaluates all key metrics:
    - B₀ sweet spot (depth, strength, uniformity)
    - Surface-coil B₁ sensitivity at target depth
    - Depth resolution from the B₀ gradient
    - Lateral resolution from the phase-encode scheme
    - Total 2D scan time
    - End-to-end SNR at target depth
    - Stray field on the maser module

    Args:
        probe_config:
            :class:`ProbeConfig` to evaluate.  Defaults to ``ProbeConfig()``.
        tissue:
            Representative tissue layer for SNR calculation.
            Defaults to muscle (T1=600 ms, T2=35 ms, ρ=1.0).

    Returns:
        :class:`ProbePerformanceReport` with all performance metrics.
    """
    if probe_config is None:
        probe_config = ProbeConfig()
    if tissue is None:
        tissue = TissueLayer("muscle", thickness_mm=20.0, t1_ms=600, t2_ms=35)

    magnet = SingleSidedMagnet(probe_config.magnet)
    coil = SurfaceCoil(probe_config.coil)

    # ── Sweet spot ────────────────────────────────────────────────
    sweet = magnet.sweet_spot()

    # ── Depth gradient at the imaging target depth ─────────────────
    # NOTE: The sweet spot is a saddle point where dBz/dz ≡ 0 by design.
    # For depth-slice resolution we need the gradient at the *imaging* depth
    # (target_depth_mm), which is typically on the flank of the field profile.
    dz_step = 0.5   # mm, half-step for central-difference derivative
    bz_for_grad = magnet.field_on_axis(
        np.array([
            probe_config.target_depth_mm - dz_step,
            probe_config.target_depth_mm + dz_step,
        ])
    )
    depth_gradient_t_per_m = float(
        np.abs((bz_for_grad[1] - bz_for_grad[0]) / (2.0 * dz_step * 1e-3))
    )

    # ── Depth resolution: δz = BW / (γ |∂B/∂z|) ─────────────────
    if depth_gradient_t_per_m > 1e-6:
        depth_res_mm = (
            probe_config.bandwidth_hz
            / (_GAMMA_P_HZ_PER_T * depth_gradient_t_per_m)
            * 1e3   # m → mm
        )
    else:
        depth_res_mm = float("inf")

    # ── Lateral resolution ────────────────────────────────────────
    scheme: PhaseEncodeScheme = build_phase_encode_scheme(
        probe_config.gradient_x,
        n_lines=probe_config.n_phase_lines,
        fov_m=probe_config.fov_m,
    )
    lateral_res_mm = scheme.resolution_mm

    # ── Total 2D scan time ────────────────────────────────────────
    scan_time_s = (
        probe_config.n_phase_lines
        * probe_config.n_averages
        * probe_config.tr_ms
        * 1e-3
    )

    # ── Coil sensitivity at target depth ─────────────────────────
    sens = float(
        sensitivity_on_axis(
            coil.config, np.array([probe_config.target_depth_mm])
        )[0]
    )

    # ── SNR at target depth ───────────────────────────────────────
    budget = compute_snr_budget(
        depth_mm=probe_config.target_depth_mm,
        voxel_size_mm=3.0,
        coil=coil,
        magnet=magnet,
        tissue=tissue,
        tr_ms=probe_config.tr_ms,
        te_ms=probe_config.te_ms,
        sequence=probe_config.pulse_sequence,
        bandwidth_hz=probe_config.bandwidth_hz,
        n_averages=probe_config.n_averages,
    )

    # ── Stray field on maser module ───────────────────────────────
    stray_rms_mt = compute_stray_field_rms(
        magnet,
        maser_separation_mm=probe_config.maser_separation_mm,
        shield_attenuation_db=probe_config.shield_attenuation_db,
    )

    # ── Larmor frequency at sweet spot ───────────────────────────
    larmor_mhz = abs(sweet.b0_tesla) * _GAMMA_P_HZ_PER_T * 1e-6

    # ── SNR in dB after n_averages ────────────────────────────────
    snr_averaged_db = (
        20.0 * math.log10(budget.snr_after_averaging)
        if budget.snr_after_averaging > 0
        else float("-inf")
    )

    return ProbePerformanceReport(
        sweet_spot_depth_mm=sweet.depth_mm,
        sweet_spot_b0_tesla=sweet.b0_tesla,
        sweet_spot_larmor_mhz=larmor_mhz,
        sweet_spot_uniformity_ppm=sweet.uniformity_ppm,
        coil_sensitivity_t_per_a=sens,
        depth_gradient_t_per_m=depth_gradient_t_per_m,
        depth_resolution_mm=depth_res_mm,
        lateral_resolution_mm=lateral_res_mm,
        scan_time_s=scan_time_s,
        snr_at_target_db=snr_averaged_db,
        snr_budget=budget,
        stray_field_rms_mt=stray_rms_mt,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  HandheldProbe class — OO interface                              ║
# ╚══════════════════════════════════════════════════════════════════╝


class HandheldProbe:
    """Object-oriented interface to the full handheld NMR/MRI probe simulation.

    Wraps :class:`SingleSidedMagnet`, :class:`SurfaceCoil`, and the complete
    signal-chain behind a single API.

    Example::

        probe = HandheldProbe(ProbeConfig(target_depth_mm=15.0))
        report = probe.performance_report()
        print(f"SNR @ 15 mm: {report.snr_at_target_db:.1f} dB")
        profile = probe.depth_scan()
        print(f"Peak signal depth: {profile.depths_mm[profile.signal.argmax()]:.1f} mm")
    """

    def __init__(self, config: ProbeConfig | None = None) -> None:
        self.config = config if config is not None else ProbeConfig()
        self._magnet = SingleSidedMagnet(self.config.magnet)
        self._coil = SurfaceCoil(self.config.coil)

    # ── Component access ──────────────────────────────────────────

    @property
    def magnet(self) -> SingleSidedMagnet:
        """Underlying :class:`SingleSidedMagnet` instance."""
        return self._magnet

    @property
    def coil(self) -> SurfaceCoil:
        """Underlying :class:`SurfaceCoil` instance."""
        return self._coil

    # ── High-level API ────────────────────────────────────────────

    def performance_report(
        self,
        tissue: TissueLayer | None = None,
    ) -> ProbePerformanceReport:
        """Compute the full performance report for this probe.

        Args:
            tissue: Representative tissue layer.  Defaults to muscle.

        Returns:
            :class:`ProbePerformanceReport`.
        """
        return compute_probe_performance(self.config, tissue=tissue)

    def depth_scan(
        self,
        tissue_layers: list[TissueLayer] | None = None,
        depth_config: DepthProfileConfig | None = None,
    ) -> DepthProfile:
        """Run 1D depth profiling (A-mode equivalent) through tissue layers.

        Args:
            tissue_layers: Stacked tissue layers in order from coil surface
                inward.  Defaults to the ``FOREARM_LAYERS`` model (skin/fat/
                muscle/bone).
            depth_config: :class:`DepthProfileConfig`; if None, uses a
                default config derived from the probe's TR and n_averages.

        Returns:
            :class:`DepthProfile` containing signal vs. depth, SNR, and
            tissue-layer labels.
        """
        if depth_config is None:
            depth_config = DepthProfileConfig(
                max_depth_mm=35.0,
                n_averages=self.config.n_averages,
                repetition_time_ms=self.config.tr_ms,
                readout_bandwidth_hz=self.config.bandwidth_hz,
            )
        return simulate_depth_profile(
            self._magnet,
            self._coil,
            depth_config,
            tissue_layers=tissue_layers,
        )

    def stray_field_rms_on_maser(
        self,
        maser_separation_mm: float | None = None,
        shield_attenuation_db: float | None = None,
    ) -> float:
        """Compute the RMS stray field from this probe's imaging magnet on the maser.

        Args:
            maser_separation_mm: Override the probe-config value (mm).
            shield_attenuation_db: Override the probe-config shielding (dB).

        Returns:
            RMS stray field in milli-Tesla (mT).
        """
        sep = (
            maser_separation_mm
            if maser_separation_mm is not None
            else self.config.maser_separation_mm
        )
        atten = (
            shield_attenuation_db
            if shield_attenuation_db is not None
            else self.config.shield_attenuation_db
        )
        return compute_stray_field_rms(
            self._magnet,
            maser_separation_mm=sep,
            shield_attenuation_db=atten,
        )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Parametric sweeps                                               ║
# ╚══════════════════════════════════════════════════════════════════╝


def sweep_snr_vs_depth(
    probe_config: ProbeConfig | None = None,
    depths_mm: NDArray[np.float64] | None = None,
    *,
    tissue: TissueLayer | None = None,
    voxel_size_mm: float = 3.0,
) -> NDArray[np.float64]:
    """SNR (dB) versus voxel depth for a fixed voxel size.

    Uses the full end-to-end SNR model from :func:`snr_vs_depth`, which
    accounts for coil sensitivity fall-off, B₀ field vs. depth, and the
    Larmor-frequency dependence of the signal EMF.

    Args:
        probe_config: probe configuration; defaults to :class:`ProbeConfig`.
        depths_mm: depth values to evaluate (mm).  Defaults to 5–30 mm,
            10 points.
        tissue: tissue layer; defaults to muscle.
        voxel_size_mm: isotropic voxel size for the SNR calculation (mm).

    Returns:
        ``(N,)`` array of SNR values in dB.  ``-inf`` where SNR ≤ 0.
    """
    if probe_config is None:
        probe_config = ProbeConfig()
    if depths_mm is None:
        depths_mm = np.linspace(5.0, 30.0, 10)
    if tissue is None:
        tissue = TissueLayer("muscle", thickness_mm=20.0, t1_ms=600, t2_ms=35)

    magnet = SingleSidedMagnet(probe_config.magnet)
    coil = SurfaceCoil(probe_config.coil)

    snrs_linear = snr_vs_depth(
        depths_mm,
        voxel_size_mm,
        coil=coil,
        magnet=magnet,
        tissue=tissue,
        tr_ms=probe_config.tr_ms,
        te_ms=probe_config.te_ms,
        sequence=probe_config.pulse_sequence,
        bandwidth_hz=probe_config.bandwidth_hz,
        n_averages=probe_config.n_averages,
    )

    result = np.empty_like(snrs_linear)
    for i, s in enumerate(snrs_linear):
        result[i] = 20.0 * math.log10(s) if s > 0 else float("-inf")
    return result


def sweep_snr_vs_averages(
    probe_config: ProbeConfig | None = None,
    n_averages_array: NDArray[np.int_] | None = None,
    *,
    depth_mm: float = 20.0,
    tissue: TissueLayer | None = None,
    voxel_size_mm: float = 3.0,
) -> NDArray[np.float64]:
    """SNR (dB) versus number of signal averages at a fixed depth.

    ``SNR ∝ √NEX`` — this function confirms that scaling with the full
    physics model.

    Args:
        probe_config: probe configuration; defaults to :class:`ProbeConfig`.
        n_averages_array: average counts to evaluate.  Defaults to
            ``[1, 4, 16, 64, 256]``.
        depth_mm: fixed voxel depth (mm).
        tissue: tissue layer; defaults to muscle.
        voxel_size_mm: voxel side length (mm).

    Returns:
        ``(N,)`` array of post-averaging SNR values in dB.
    """
    if probe_config is None:
        probe_config = ProbeConfig()
    if n_averages_array is None:
        n_averages_array = np.array([1, 4, 16, 64, 256], dtype=int)
    if tissue is None:
        tissue = TissueLayer("muscle", thickness_mm=20.0, t1_ms=600, t2_ms=35)

    magnet = SingleSidedMagnet(probe_config.magnet)
    coil = SurfaceCoil(probe_config.coil)

    snrs_linear = snr_vs_averages(
        n_averages_array,
        depth_mm,
        voxel_size_mm,
        coil=coil,
        magnet=magnet,
        tissue=tissue,
        tr_ms=probe_config.tr_ms,
        te_ms=probe_config.te_ms,
        sequence=probe_config.pulse_sequence,
        bandwidth_hz=probe_config.bandwidth_hz,
    )

    result = np.empty_like(snrs_linear)
    for i, s in enumerate(snrs_linear):
        result[i] = 20.0 * math.log10(s) if s > 0 else float("-inf")
    return result


def sweep_lateral_resolution_vs_n_lines(
    n_lines_values: NDArray[np.int_] | None = None,
    *,
    fov_m: float = 0.08,
    gradient_spec: GradientCoilSpec | None = None,
) -> NDArray[np.float64]:
    """Lateral resolution (mm) versus number of phase-encoding lines.

    Resolution = FOV / n_lines.  This is a pure geometry calculation — it
    confirms that finer encoding grids translate to higher resolution.

    Args:
        n_lines_values: phase-line counts.  Defaults to ``[16, 32, 64, 128, 256]``.
        fov_m: field of view (m).
        gradient_spec: gradient coil spec; defaults to :data:`DEFAULT_GX`.

    Returns:
        ``(N,)`` array of lateral resolutions in mm.
    """
    if n_lines_values is None:
        n_lines_values = np.array([16, 32, 64, 128, 256], dtype=int)
    if gradient_spec is None:
        gradient_spec = DEFAULT_GX

    result = np.empty(len(n_lines_values))
    for i, n in enumerate(n_lines_values):
        scheme = build_phase_encode_scheme(
            gradient_spec, n_lines=int(n), fov_m=fov_m
        )
        result[i] = scheme.resolution_mm
    return result


def sweep_stray_field_vs_separation(
    magnet: SingleSidedMagnet | None = None,
    separations_mm: NDArray[np.float64] | None = None,
    *,
    shield_attenuation_db: float = 0.0,
    grid_extent_mm: float = 10.0,
) -> NDArray[np.float64]:
    """RMS stray field (mT) versus lateral probe–maser separation.

    Useful for selecting the mechanical layout of the probe so that the
    maser module experiences an acceptable residual disturbance field even
    before mu-metal shielding is applied.

    The stray field falls as 1/r³ (dipole), so doubling the separation
    reduces it by ≈ 8×.

    Args:
        magnet: imaging magnet; defaults to ``SingleSidedMagnet(SingleSidedMagnetConfig())``.
        separations_mm: separation distances (mm).  Defaults to 20–100 mm,
            10 points.
        shield_attenuation_db: shielding factor.  0 = no shielding. 50 = target.
        grid_extent_mm: maser grid extent for RMS averaging (mm).

    Returns:
        ``(N,)`` array of RMS stray fields in milli-Tesla.
    """
    if magnet is None:
        magnet = SingleSidedMagnet(SingleSidedMagnetConfig())
    if separations_mm is None:
        separations_mm = np.linspace(20.0, 100.0, 10)

    result = np.empty(len(separations_mm))
    for i, sep in enumerate(separations_mm):
        result[i] = compute_stray_field_rms(
            magnet,
            maser_separation_mm=float(sep),
            shield_attenuation_db=shield_attenuation_db,
            grid_extent_mm=grid_extent_mm,
        )
    return result


def sweep_depth_resolution_vs_bandwidth(
    bandwidths_hz: NDArray[np.float64] | None = None,
    *,
    magnet: SingleSidedMagnet | None = None,
) -> NDArray[np.float64]:
    """Depth resolution (mm) versus readout bandwidth.

    δz = BW / (γ |∂B/∂z|) at the sweet spot.  A wider readout bandwidth
    reduces depth resolution (trades SNR for resolution).

    Args:
        bandwidths_hz: bandwidths to evaluate (Hz).  Defaults to
            ``[1000, 2000, 5000, 10000, 20000, 50000]`` Hz.
        magnet: imaging magnet; defaults to default barrel config.

    Returns:
        ``(N,)`` array of depth resolutions in mm.
    """
    if bandwidths_hz is None:
        bandwidths_hz = np.array(
            [1_000, 2_000, 5_000, 10_000, 20_000, 50_000], dtype=float
        )
    if magnet is None:
        magnet = SingleSidedMagnet(SingleSidedMagnetConfig())

    # Use the gradient at a representative imaging depth (20 mm), not the sweet
    # spot where dBz/dz = 0 by definition.  The depth-slice bandwidth formula
    # δz = BW / (γ |∂B/∂z|) is most meaningful on the field gradient flank.
    eval_depth_mm = 20.0
    dz_step = 0.5   # mm
    bz_for_grad = magnet.field_on_axis(
        np.array([eval_depth_mm - dz_step, eval_depth_mm + dz_step])
    )
    grad_t_per_m = float(
        np.abs((bz_for_grad[1] - bz_for_grad[0]) / (2.0 * dz_step * 1e-3))
    )

    result = np.empty(len(bandwidths_hz))
    for i, bw in enumerate(bandwidths_hz):
        if grad_t_per_m > 1e-6:
            result[i] = bw / (_GAMMA_P_HZ_PER_T * grad_t_per_m) * 1e3  # mm
        else:
            result[i] = float("inf")
    return result
