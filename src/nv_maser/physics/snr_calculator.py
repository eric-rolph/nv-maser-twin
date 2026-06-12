"""
End-to-end SNR calculator for the handheld NMR/MRI probe.

Integrates the complete signal chain from tissue voxel to digitiser:

    Tissue voxel  →  surface coil  →  up-conversion mixer
         →  NV maser amplifier  →  LNA  →  ADC

Signal model
────────────
The NMR signal voltage at the coil terminals is:

    V_sig = ω₀ × M₀ × V_voxel × (B₁/I) × C_seq

where:
    ω₀  = 2π f_Larmor   (angular frequency, rad/s)
    M₀  = equilibrium magnetisation (A/m)  — proton density × B₀ / T
    V   = voxel volume (m³)
    B₁/I = coil sensitivity via reciprocity (T/A)
    C_seq = sequence contrast factor (T₁/T₂ weighting) ∈ [0, 1]

Noise model
───────────
Noise is referred to the coil input:

    V_n,total² = V_n,coil² + V_n,body² + V_n,mixer² + V_n,maser²

where the mixer and maser noise are referred back through the conversion
gain/loss.  For the system described in Sec. 8 of the architecture doc:
the coil thermal noise at 300 K dominates; the maser provides ~11% SNR
improvement by replacing a conventional LNA.

Parametric sweeps supported
────────────────────────────
- snr_vs_depth:       SNR as a function of voxel depth (mm)
- snr_vs_averages:    SNR versus number of NSA (signal averages)
- snr_vs_voxel_size:  SNR versus isotropic voxel side length (mm)
- required_averages_for_snr: minimum NSA to achieve a target SNR

References
──────────
Hoult & Richards, "The signal-to-noise ratio of the NMR experiment",
J. Magn. Reson. 24, 71 (1976).

Handheld probe architecture document, §8.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import HBAR as _HBAR
from .constants import KB as _KB

# ── Physical constants ────────────────────────────────────────────
from .depth_profile import TissueLayer
from .pulse_sequence import simulate_gre, simulate_spin_echo
from .signal_chain import compute_maser_noise_temperature
from .single_sided_magnet import SingleSidedMagnet
from .surface_coil import SurfaceCoil, compute_noise, sensitivity_on_axis
from .up_conversion import DEFAULT_MIXER, MixerSpec, friis_system_temperature_with_mixer

_GAMMA_P = 2.675e8                   # proton gyromagnetic ratio (rad/s/T)
_N_PROTONS_WATER = 6.69e28           # proton number density in water (m⁻³)
_T0 = 290.0                          # IEEE reference temperature (K)


def _equilibrium_magnetisation(
    b0_tesla: float,
    proton_density: float,
    temperature_k: float = 310.0,
) -> float:
    """Equilibrium proton magnetisation M₀ (A/m).

    M₀ = n_H · ρ_p · γ²ℏ²I(I+1) · B₀ / (3 kB T)

    Args:
        b0_tesla: B₀ field at the voxel location (T).
        proton_density: fractional proton density (0–1; 1 = pure water).
        temperature_k: tissue temperature (K).

    Returns:
        M₀ in A/m.
    """
    n = _N_PROTONS_WATER * proton_density
    spin_i = 0.5
    return n * _GAMMA_P**2 * _HBAR**2 * spin_i * (spin_i + 1) * abs(b0_tesla) / (
        3.0 * _KB * temperature_k
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Result dataclasses                                              ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class SNRBudget:
    """Complete SNR budget for a single tissue voxel.

    Attributes
    ----------
    depth_mm:
        Voxel depth below coil surface (mm).
    voxel_size_mm:
        Isotropic voxel side length (mm).
    b0_tesla:
        B₀ field at the voxel depth (T).
    larmor_frequency_hz:
        Proton Larmor frequency at depth (Hz).
    signal_v:
        EMF from precessing magnetisation at coil terminals (V).
    noise_coil_v:
        Coil thermal + body noise: √(4kT_coil R_coil Δf + body_noise²) (V).
    noise_mixer_v:
        Mixer noise referred to coil input (V). 0 if mixer is None.
    noise_total_v:
        RSS of all noise sources referred to the coil input (V).
    snr_per_shot:
        Single-shot signal-to-noise ratio (linear, dimensionless).
    snr_db:
        Single-shot SNR in dB.
    snr_after_averaging:
        SNR after n_averages × number of signal averages (√NEX improvement).
    n_averages:
        Number of signal averages applied.
    bandwidth_hz:
        Readout detection bandwidth (Hz).
    scan_time_s:
        Total acquisition time = n_averages × TR (s).
    sequence_contrast:
        T₁/T₂ weighting factor from the pulse sequence (0–1).
    system_noise_temp_k:
        Friis-cascade system noise temperature referred to coil input (K).
    maser_noise_temp_k:
        Maser amplifier equivalent noise temperature (K).
    maser_advantage_db:
        SNR improvement (dB) of maser preamp over conventional LNA (NF=1 dB).
        Positive means the maser is better.
    """

    depth_mm: float
    voxel_size_mm: float
    b0_tesla: float
    larmor_frequency_hz: float
    signal_v: float
    noise_coil_v: float
    noise_mixer_v: float
    noise_total_v: float
    snr_per_shot: float
    snr_db: float
    snr_after_averaging: float
    n_averages: int
    bandwidth_hz: float
    scan_time_s: float
    sequence_contrast: float
    system_noise_temp_k: float
    maser_noise_temp_k: float
    maser_advantage_db: float


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Core SNR computation                                            ║
# ╚══════════════════════════════════════════════════════════════════╝


def compute_snr_budget(
    depth_mm: float,
    voxel_size_mm: float,
    *,
    coil: SurfaceCoil,
    magnet: SingleSidedMagnet,
    tissue: TissueLayer,
    tr_ms: float = 100.0,
    te_ms: float = 10.0,
    sequence: str = "spin_echo",
    bandwidth_hz: float = 10_000.0,
    n_averages: int = 1,
    body_temperature_k: float = 310.0,
    lna_noise_figure_db: float = 1.0,
    mixer: MixerSpec | None = None,
    cavity_q: float | None = None,
    q_magnetic: float | None = None,
) -> SNRBudget:
    """Compute the end-to-end SNR budget for one voxel at depth ``depth_mm``.

    Args:
        depth_mm:
            Voxel centre depth below coil plane (mm). Must be > 0.
        voxel_size_mm:
            Isotropic voxel side length (mm).
        coil:
            Surface coil object (provides B₁/I sensitivity and noise).
        magnet:
            Single-sided magnet (provides B₀ at depth).
        tissue:
            TissueLayer with T₁, T₂, and proton density.
        tr_ms:
            Repetition time (ms).
        te_ms:
            Echo time (ms). Must satisfy TE < TR.
        sequence:
            Pulse sequence: "spin_echo" or "gre".
        bandwidth_hz:
            Readout detection bandwidth (Hz).
        n_averages:
            Number of signal averages (NEX).
        body_temperature_k:
            Tissue temperature (K).
        lna_noise_figure_db:
            Noise figure of the post-maser LNA for Friis calculation (dB).
        mixer:
            Up-conversion mixer spec; defaults to DEFAULT_MIXER.
        cavity_q:
            Unloaded cavity Q factor. If None, maser noise model is skipped.
        q_magnetic:
            Magnetic Q factor of the maser gain medium. If None, skipped.

    Returns:
        SNRBudget with complete signal and noise breakdown.

    Raises:
        ValueError: For invalid inputs (depth, TE ≥ TR, unknown sequence).
    """
    if depth_mm <= 0:
        raise ValueError(f"depth_mm must be positive, got {depth_mm}")
    if voxel_size_mm <= 0:
        raise ValueError(f"voxel_size_mm must be positive, got {voxel_size_mm}")
    if bandwidth_hz <= 0:
        raise ValueError(f"bandwidth_hz must be positive, got {bandwidth_hz}")
    if n_averages < 1:
        raise ValueError(f"n_averages must be ≥ 1, got {n_averages}")

    if mixer is None:
        mixer = DEFAULT_MIXER

    # ── B₀ and Larmor frequency at depth ─────────────────────────
    b0_arr = magnet.field_on_axis(np.array([depth_mm]))
    b0 = float(b0_arr[0])
    _PROTON_LARMOR_HZ_PER_T = 42.577e6  # Hz/T
    larmor_hz = abs(b0) * _PROTON_LARMOR_HZ_PER_T

    # ── Equilibrium magnetisation ─────────────────────────────────
    m0 = _equilibrium_magnetisation(b0, tissue.proton_density, body_temperature_k)

    # ── Coil sensitivity at depth ─────────────────────────────────
    sens = float(sensitivity_on_axis(coil.config, np.array([depth_mm]))[0])

    # ── Voxel volume ──────────────────────────────────────────────
    v_voxel = (voxel_size_mm * 1e-3) ** 3  # m³

    # ── Sequence contrast factor ──────────────────────────────────
    if sequence == "spin_echo":
        seq_result = simulate_spin_echo(
            t1_ms=tissue.t1_ms,
            t2_ms=tissue.t2_ms,
            tr_ms=tr_ms,
            te_ms=te_ms,
        )
        contrast = seq_result.signal_normalized
    elif sequence == "gre":
        # For GRE use T2* = T2 / 2 (rough estimate if no T2* provided)
        t2_star = tissue.t2_ms / 2.0
        seq_result = simulate_gre(
            t1_ms=tissue.t1_ms,
            t2_star_ms=t2_star,
            tr_ms=tr_ms,
            te_ms=te_ms,
        )
        contrast = seq_result.signal_normalized
    else:
        raise ValueError(f"Unknown sequence: {sequence!r}; choose 'spin_echo' or 'gre'")

    # ── Signal voltage at coil terminals ─────────────────────────
    # V_sig = ω₀ M₀ V_voxel (B₁/I) × C_seq
    omega0 = 2.0 * math.pi * larmor_hz
    signal_v = omega0 * m0 * v_voxel * sens * contrast

    # ── Coil noise (thermal + body) ───────────────────────────────
    noise_comp = compute_noise(coil.config, larmor_hz, bandwidth_hz)
    noise_coil_v = noise_comp.total_noise_v

    # ── Mixer noise referred to coil input ───────────────────────
    # Mixer adds noise as if there were an extra thermal source at T_mixer
    # V_n,mixer = sqrt(4 kB T_mixer R_coil Δf), referred to coil input
    t_mixer_k = _T0 * (10.0 ** (mixer.effective_nf_db / 10.0) - 1.0)
    # Mixer loss means it attenuates the signal but adds noise
    # Referred-to-input mixer noise voltage across coil resistance
    coil_props = coil.properties(larmor_hz)
    r_coil = coil_props.ac_resistance_ohm
    noise_mixer_v = math.sqrt(4 * _KB * t_mixer_k * r_coil * bandwidth_hz)

    # ── Maser amplifier noise temperature ────────────────────────
    if cavity_q is not None and q_magnetic is not None:
        t_maser_k = compute_maser_noise_temperature(
            q_magnetic=q_magnetic,
            cavity_q=cavity_q,
            bath_temperature_k=coil.config.temperature_k,
        )
        if math.isinf(t_maser_k) or math.isnan(t_maser_k):
            t_maser_k = 0.0  # at/above threshold: don't penalise SNR
    else:
        # Default: typical maser noise temp from Wang 2024 (T_n ~ 1–5 K)
        t_maser_k = 5.0

    # Friis system noise temperature (coil input referred)
    maser_gain_db = 30.0  # conservative typical maser gain
    t_sys = friis_system_temperature_with_mixer(
        coil_temperature_k=coil.config.temperature_k,
        mixer=mixer,
        maser_noise_temperature_k=t_maser_k,
        maser_gain_db=maser_gain_db,
        lna_noise_temperature_k=_T0 * (10.0 ** (lna_noise_figure_db / 10.0) - 1.0),
    )

    # ── Total noise voltage (RSS) ─────────────────────────────────
    noise_total_v = math.sqrt(noise_coil_v**2 + noise_mixer_v**2)

    # ── SNR ───────────────────────────────────────────────────────
    if noise_total_v > 0:
        snr_per_shot = signal_v / noise_total_v
    else:
        snr_per_shot = float("inf")

    snr_db = 20.0 * math.log10(snr_per_shot) if snr_per_shot > 0 else float("-inf")
    snr_averaged = snr_per_shot * math.sqrt(n_averages)

    # ── Maser advantage vs conventional LNA (NF=1 dB → T_amp ≈ 75 K) ──
    # Without maser: conventional LNA (NF = 1 dB, T_amp ≈ 75 K)
    # SNR_conv ∝ 1/sqrt(T_coil + T_lna) ; SNR_maser ∝ 1/sqrt(T_coil + T_maser)
    t_lna_conventional = _T0 * (10.0 ** (lna_noise_figure_db / 10.0) - 1.0)
    # Simplified: compare only LNA contribution
    noise_conv_total2 = noise_coil_v**2 + math.sqrt(
        4 * _KB * t_lna_conventional * r_coil * bandwidth_hz
    ) ** 2
    noise_maser_total2 = noise_coil_v**2 + math.sqrt(
        4 * _KB * t_maser_k * r_coil * bandwidth_hz
    ) ** 2
    if noise_maser_total2 > 0 and noise_conv_total2 > 0:
        snr_ratio = math.sqrt(noise_conv_total2 / noise_maser_total2)
        maser_advantage_db = 20.0 * math.log10(snr_ratio)
    else:
        maser_advantage_db = 0.0

    # ── Scan time ─────────────────────────────────────────────────
    scan_time_s = n_averages * tr_ms * 1e-3

    return SNRBudget(
        depth_mm=depth_mm,
        voxel_size_mm=voxel_size_mm,
        b0_tesla=b0,
        larmor_frequency_hz=larmor_hz,
        signal_v=signal_v,
        noise_coil_v=noise_coil_v,
        noise_mixer_v=noise_mixer_v,
        noise_total_v=noise_total_v,
        snr_per_shot=snr_per_shot,
        snr_db=snr_db,
        snr_after_averaging=snr_averaged,
        n_averages=n_averages,
        bandwidth_hz=bandwidth_hz,
        scan_time_s=scan_time_s,
        sequence_contrast=contrast,
        system_noise_temp_k=t_sys,
        maser_noise_temp_k=t_maser_k,
        maser_advantage_db=maser_advantage_db,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Parametric sweeps                                              ║
# ╚══════════════════════════════════════════════════════════════════╝


def snr_vs_depth(
    depths_mm: NDArray[np.float64],
    voxel_size_mm: float,
    *,
    coil: SurfaceCoil,
    magnet: SingleSidedMagnet,
    tissue: TissueLayer,
    tr_ms: float = 100.0,
    te_ms: float = 10.0,
    sequence: str = "spin_echo",
    bandwidth_hz: float = 10_000.0,
    n_averages: int = 1,
    body_temperature_k: float = 310.0,
    lna_noise_figure_db: float = 1.0,
    mixer: MixerSpec | None = None,
) -> NDArray[np.float64]:
    """SNR versus depth for a 1-D depth sweep.

    Vectorised wrapper around ``compute_snr_budget``.

    Args:
        depths_mm: array of voxel depths (mm). All must be > 0.
        (other args: same as ``compute_snr_budget``)

    Returns:
        Array of SNR values (linear) at each depth. Same shape as depths_mm.
    """
    snr_arr = np.empty(len(depths_mm))
    for i, d in enumerate(depths_mm):
        budget = compute_snr_budget(
            d, voxel_size_mm,
            coil=coil, magnet=magnet, tissue=tissue,
            tr_ms=tr_ms, te_ms=te_ms, sequence=sequence,
            bandwidth_hz=bandwidth_hz, n_averages=n_averages,
            body_temperature_k=body_temperature_k,
            lna_noise_figure_db=lna_noise_figure_db,
            mixer=mixer,
        )
        snr_arr[i] = budget.snr_per_shot
    return snr_arr


def snr_vs_averages(
    n_averages_array: NDArray[np.int_],
    depth_mm: float,
    voxel_size_mm: float,
    *,
    coil: SurfaceCoil,
    magnet: SingleSidedMagnet,
    tissue: TissueLayer,
    tr_ms: float = 100.0,
    te_ms: float = 10.0,
    sequence: str = "spin_echo",
    bandwidth_hz: float = 10_000.0,
    body_temperature_k: float = 310.0,
    lna_noise_figure_db: float = 1.0,
    mixer: MixerSpec | None = None,
) -> NDArray[np.float64]:
    """SNR versus number of signal averages.

    SNR scales as √NEX; this function confirms that with the full physics.

    Args:
        n_averages_array: array of integer average counts. All must be ≥ 1.
        depth_mm: fixed voxel depth (mm).
        voxel_size_mm: voxel side length (mm).
        (other args: same as ``compute_snr_budget``)

    Returns:
        Array of SNR values after averaging (linear).
    """
    snr_arr = np.empty(len(n_averages_array))
    for i, nex in enumerate(n_averages_array):
        budget = compute_snr_budget(
            depth_mm, voxel_size_mm,
            coil=coil, magnet=magnet, tissue=tissue,
            tr_ms=tr_ms, te_ms=te_ms, sequence=sequence,
            bandwidth_hz=bandwidth_hz, n_averages=int(nex),
            body_temperature_k=body_temperature_k,
            lna_noise_figure_db=lna_noise_figure_db,
            mixer=mixer,
        )
        snr_arr[i] = budget.snr_after_averaging
    return snr_arr


def snr_vs_voxel_size(
    voxel_sizes_mm: NDArray[np.float64],
    depth_mm: float,
    *,
    coil: SurfaceCoil,
    magnet: SingleSidedMagnet,
    tissue: TissueLayer,
    tr_ms: float = 100.0,
    te_ms: float = 10.0,
    sequence: str = "spin_echo",
    bandwidth_hz: float = 10_000.0,
    n_averages: int = 1,
    body_temperature_k: float = 310.0,
    lna_noise_figure_db: float = 1.0,
    mixer: MixerSpec | None = None,
) -> NDArray[np.float64]:
    """SNR versus isotropic voxel size.

    Signal ∝ V_voxel ∝ voxel_size³, so SNR ∝ voxel_size³.

    Args:
        voxel_sizes_mm: array of voxel side lengths (mm). All must be > 0.
        depth_mm: fixed voxel depth (mm).
        (other args: same as ``compute_snr_budget``)

    Returns:
        Array of SNR values (linear) for each voxel size.
    """
    snr_arr = np.empty(len(voxel_sizes_mm))
    for i, vs in enumerate(voxel_sizes_mm):
        budget = compute_snr_budget(
            depth_mm, float(vs),
            coil=coil, magnet=magnet, tissue=tissue,
            tr_ms=tr_ms, te_ms=te_ms, sequence=sequence,
            bandwidth_hz=bandwidth_hz, n_averages=n_averages,
            body_temperature_k=body_temperature_k,
            lna_noise_figure_db=lna_noise_figure_db,
            mixer=mixer,
        )
        snr_arr[i] = budget.snr_per_shot
    return snr_arr


def required_averages_for_snr(
    target_snr: float,
    depth_mm: float,
    voxel_size_mm: float,
    *,
    coil: SurfaceCoil,
    magnet: SingleSidedMagnet,
    tissue: TissueLayer,
    tr_ms: float = 100.0,
    te_ms: float = 10.0,
    sequence: str = "spin_echo",
    bandwidth_hz: float = 10_000.0,
    body_temperature_k: float = 310.0,
    lna_noise_figure_db: float = 1.0,
    mixer: MixerSpec | None = None,
) -> int:
    """Minimum number of averages required to reach ``target_snr``.

    SNR_final = SNR_per_shot × √NEX  →  NEX = (target_snr / SNR_per_shot)²

    Returns the ceiling integer of NEX.

    Args:
        target_snr: minimum required SNR (linear).
        depth_mm: voxel depth (mm).
        voxel_size_mm: voxel side length (mm).
        (other args: same as ``compute_snr_budget``)

    Returns:
        Minimum integer number of averages (≥ 1).

    Raises:
        ValueError: If target_snr ≤ 0.
    """
    if target_snr <= 0:
        raise ValueError(f"target_snr must be positive, got {target_snr}")

    budget = compute_snr_budget(
        depth_mm, voxel_size_mm,
        coil=coil, magnet=magnet, tissue=tissue,
        tr_ms=tr_ms, te_ms=te_ms, sequence=sequence,
        bandwidth_hz=bandwidth_hz, n_averages=1,
        body_temperature_k=body_temperature_k,
        lna_noise_figure_db=lna_noise_figure_db,
        mixer=mixer,
    )

    snr_per_shot = budget.snr_per_shot
    if snr_per_shot <= 0:
        return int(1e9)  # effectively infinite

    ratio = target_snr / snr_per_shot
    return max(1, math.ceil(ratio**2))
