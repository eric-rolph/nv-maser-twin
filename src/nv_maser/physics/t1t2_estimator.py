"""
T1/T2 tissue parameter estimation from EPG echo-train data.

After an EPG CPMG simulation (``epg_cpmg``) or multiple spin-echo
acquisitions (``epg_signal`` at varying TR), this module provides:

* **T2 mapping** — monoexponential fit to CPMG echo trains per depth layer.
* **T1 mapping** — saturation-recovery fit to SE signals at multiple TRs.
* **Joint T1/T2 map** — combines both into one depth-resolved result.
* **Abnormality detector** — flags depth layers whose fitted T1/T2 deviate
  significantly from a supplied reference map.
* **Cross-validation** — compares two independent T1/T2 maps.

Clinical context (handheld probe, 50 mT)
─────────────────────────────────────────
At 50 mT (low field) T1 values are shorter than at clinical 1.5 T because
T1 ∝ B₀^0.3 approximately.  T2 is relatively field-independent.  Reference
values used here follow De Graaf (2007) and the ``depth_profile.py`` tissue
model embedded in this codebase.

Key diagnostic markers detectable by the probe:

* **Subacute haemorrhage** — prolonged T2_eff (early, free water stage)
  versus expected muscle T2.  Later stages exhibit shortened T2 (deoxy-Hb/
  met-Hb).
* **Acute oedema** — both T1 and T2 prolonged (increased free water).
* **Subcutaneous fat** — short T1 ≈ 250 ms, long T2 ≈ 80 ms at low field.
* **Cortical bone** — T2 < 1 ms, essentially invisible in spin-echo.

Algorithm
─────────
1. **T2**: run ``epg_cpmg`` at each tissue layer → echo train
   S(n) = S₀ exp(−n·ESP/T₂).  Fit with scipy curve_fit.
2. **T1**: run ``epg_signal`` at N_TR repetition-time values, collect
   S(TR) ≈ A (1 − exp(−TR/T₁)).  Fit with curve_fit (absorbs TE factor
   into amplitude A = M₀ exp(−TE/T₂)).
3. **Joint map**: run both passes and merge into a ``T1T2Map``.
4. **Abnormality detection**: per-depth comparison of fitted T2 (or T1)
   against a reference map generated from the healthy tissue model.

References
──────────
De Graaf, R. A. (2007) *In Vivo NMR Spectroscopy*, 2nd ed., Wiley.
Bottomley et al. (1984) "A review of normal tissue hydrogen NMR relaxation
times and relaxation mechanisms from 1–100 MHz", Med. Phys. 11(4):425.
Weigel M (2015) "Extended phase graphs", JMRI 41(2):266–295.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeWarning, curve_fit

from .depth_profile import TissueLayer, _assign_layers
from .epg_adapter import epg_cpmg, epg_signal

# ── Fitting result containers ─────────────────────────────────────


@dataclass(frozen=True)
class T2FitResult:
    """Result of a single-layer monoexponential T2 fit."""

    t2_s: float          # fitted T2 (seconds)
    s0: float            # fitted steady-state echo amplitude
    r_squared: float     # R² goodness-of-fit
    residuals_rms: float # RMS residual between model and data
    converged: bool      # False if curve_fit did not converge


@dataclass(frozen=True)
class T1FitResult:
    """Result of a single-layer saturation-recovery T1 fit."""

    t1_s: float
    amplitude: float     # A = M₀ × exp(−TE/T2) (absorbed TE factor)
    r_squared: float
    residuals_rms: float
    converged: bool


# ── Depth-map containers ──────────────────────────────────────────


@dataclass(frozen=True)
class T2MapResult:
    """Depth-resolved T2 map from EPG CPMG simulation + fitting."""

    depths_mm: NDArray[np.float64]        # depth positions (mm)
    t2_fitted_s: NDArray[np.float64]      # fitted T2 per depth (s)
    t2_reference_s: NDArray[np.float64]   # ground-truth T2 from TissueLayer (s)
    s0: NDArray[np.float64]               # fitted steady-state amplitude
    r_squared: NDArray[np.float64]        # R² per depth
    residuals_rms: NDArray[np.float64]    # RMS residual per depth
    tissue_labels: list[str]              # tissue name at each depth
    esp_s: float                          # echo spacing used (s)
    n_echoes: int                         # number of echoes used
    tr_s: float                           # repetition time used (s)


@dataclass(frozen=True)
class T1MapResult:
    """Depth-resolved T1 map from EPG saturation-recovery simulation + fitting."""

    depths_mm: NDArray[np.float64]
    t1_fitted_s: NDArray[np.float64]
    t1_reference_s: NDArray[np.float64]
    amplitude: NDArray[np.float64]        # fitted A = M₀ exp(−TE/T2)
    r_squared: NDArray[np.float64]
    residuals_rms: NDArray[np.float64]
    tissue_labels: list[str]
    tr_values_s: NDArray[np.float64]      # TR values used for fit
    te_s: float                           # echo time used (s)


@dataclass(frozen=True)
class T1T2Map:
    """Combined depth-resolved T1 and T2 map."""

    depths_mm: NDArray[np.float64]
    t1_s: NDArray[np.float64]             # fitted T1 per depth
    t2_s: NDArray[np.float64]             # fitted T2 per depth
    t1_reference_s: NDArray[np.float64]   # ground-truth T1 from TissueLayer
    t2_reference_s: NDArray[np.float64]   # ground-truth T2 from TissueLayer
    t1_r_squared: NDArray[np.float64]
    t2_r_squared: NDArray[np.float64]
    tissue_labels: list[str]
    t1t2_ratio: NDArray[np.float64]       # T1/T2 tissue contrast ratio
    r1_s_inv: NDArray[np.float64]         # longitudinal relaxation rate R1 = 1/T1
    r2_s_inv: NDArray[np.float64]         # transverse relaxation rate R2 = 1/T2


@dataclass(frozen=True)
class AbnormalityFlag:
    """A single depth location flagged as deviating from reference."""

    depth_mm: float
    tissue_label: str
    parameter: str           # "T2" or "T1"
    fitted_value_s: float
    reference_value_s: float
    deviation_fraction: float  # (fitted − reference) / reference
    flag_type: str             # "prolonged" | "shortened"


@dataclass(frozen=True)
class T1T2CrossValidation:
    """Cross-validation metrics between two T1/T2 maps."""

    t1_correlation: float          # Pearson r for T1 arrays
    t2_correlation: float          # Pearson r for T2 arrays
    t1_max_relative_error: float   # max |T1_a − T1_b| / T1_a
    t2_max_relative_error: float
    t1_mean_relative_error: float  # mean |T1_a − T1_b| / T1_a
    t2_mean_relative_error: float
    n_depths: int                  # number of shared depth points compared


# ── Core fitting functions ────────────────────────────────────────


def fit_t2_monoexponential(
    echo_amplitudes: NDArray[np.float64],
    echo_times_s: NDArray[np.float64],
) -> T2FitResult:
    """Fit S(t) = S₀ × exp(−t / T₂) to a CPMG echo train.

    Uses ``scipy.optimize.curve_fit`` with bounds to ensure physical
    parameter values (S₀ > 0, T₂ > 0).  If the fit fails to converge,
    returns a result with ``converged=False`` and a fallback estimate.

    Args:
        echo_amplitudes: array of echo magnitudes |F⁺[0]| at each echo time.
        echo_times_s: echo centre times in seconds, same length as amplitudes.

    Returns:
        T2FitResult with fitted T₂, S₀, R², RMS residual, and convergence flag.

    Raises:
        ValueError: if arrays have fewer than 2 non-zero elements.
    """
    if len(echo_amplitudes) < 2:
        raise ValueError(
            f"Need at least 2 echo amplitudes, got {len(echo_amplitudes)}."
        )
    if len(echo_amplitudes) != len(echo_times_s):
        raise ValueError("echo_amplitudes and echo_times_s must have equal length.")

    # Guard: if all amplitudes are negligibly small (bone cortex edge case),
    # return a degenerate result without crashing curve_fit.
    if np.max(echo_amplitudes) < 1e-10:
        return T2FitResult(
            t2_s=1e-6,
            s0=float(np.max(echo_amplitudes)),
            r_squared=0.0,
            residuals_rms=0.0,
            converged=False,
        )

    def _model(t: NDArray, s0: float, t2: float) -> NDArray:
        return s0 * np.exp(-t / t2)

    # Robust initial guess from log-linear slope.
    s0_init = float(echo_amplitudes[0])
    # Estimate T2 from the time to drop to e^{-1} of peak, clipped.
    safe_s0 = s0_init if s0_init > 1e-15 else 1e-15
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vals = np.where(
            echo_amplitudes > 1e-15,
            np.log(echo_amplitudes / safe_s0),
            -100.0,
        )
    # Linear fit to log data for initial T2 estimate
    valid = echo_amplitudes > 0
    if valid.sum() >= 2:
        slope, _ = np.polyfit(echo_times_s[valid], log_vals[valid], 1)
        t2_init = max(-1.0 / (slope - 1e-12), 1e-4)
    else:
        t2_init = float(echo_times_s[-1]) * 0.5

    # Bounds: S0 in (0, 10], T2 in (1 μs, 100 s)
    bounds = ([0.0, 1e-6], [10.0, 100.0])
    t2_init = float(np.clip(t2_init, 1e-5, 99.0))

    converged = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                _model,
                echo_times_s,
                echo_amplitudes,
                p0=[s0_init, t2_init],
                bounds=bounds,
                maxfev=10_000,
            )
        s0_fit, t2_fit = float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        # Fallback: log-linear estimate
        converged = False
        s0_fit = s0_init
        t2_fit = t2_init

    fitted = _model(echo_times_s, s0_fit, t2_fit)
    ss_res = float(np.sum((echo_amplitudes - fitted) ** 2))
    ss_tot = float(np.sum((echo_amplitudes - np.mean(echo_amplitudes)) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0
    rms_res = float(np.sqrt(np.mean((echo_amplitudes - fitted) ** 2)))

    return T2FitResult(
        t2_s=t2_fit,
        s0=s0_fit,
        r_squared=float(np.clip(r_sq, -1.0, 1.0)),
        residuals_rms=rms_res,
        converged=converged,
    )


def fit_t1_saturation_recovery(
    signals: NDArray[np.float64],
    tr_values_s: NDArray[np.float64],
) -> T1FitResult:
    """Fit S(TR) = A × (1 − exp(−TR / T₁)) to saturation-recovery data.

    The amplitude A absorbs any TE-dependent factor (A = M₀ × exp(−TE/T₂)),
    so caller does not need to provide TE explicitly.

    Args:
        signals: signal amplitude at each TR value.
        tr_values_s: TR values in seconds, same length as signals.

    Returns:
        T1FitResult with fitted T₁, amplitude, R², RMS residual, convergence flag.
    """
    if len(signals) < 2:
        raise ValueError(f"Need at least 2 TR values, got {len(signals)}.")
    if len(signals) != len(tr_values_s):
        raise ValueError("signals and tr_values_s must have equal length.")

    def _model(tr: NDArray, amplitude: float, t1: float) -> NDArray:
        return amplitude * (1.0 - np.exp(-tr / t1))

    # Initial guess: A ≈ max signal, T1 ≈ TR where signal reaches 63%
    a_init = float(np.max(signals))
    idx_63 = int(np.argmin(np.abs(signals - 0.63 * a_init)))
    t1_init = max(float(tr_values_s[idx_63]), 1e-3)

    bounds = ([0.0, 1e-4], [10.0, 100.0])  # A in (0,10], T1 in (0.1 ms, 100 s)
    t1_init = float(np.clip(t1_init, 1e-3, 99.0))

    converged = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                _model,
                tr_values_s,
                signals,
                p0=[a_init, t1_init],
                bounds=bounds,
                maxfev=10_000,
            )
        a_fit, t1_fit = float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        converged = False
        a_fit = a_init
        t1_fit = t1_init

    fitted = _model(tr_values_s, a_fit, t1_fit)
    ss_res = float(np.sum((signals - fitted) ** 2))
    ss_tot = float(np.sum((signals - np.mean(signals)) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0
    rms_res = float(np.sqrt(np.mean((signals - fitted) ** 2)))

    return T1FitResult(
        t1_s=t1_fit,
        amplitude=a_fit,
        r_squared=float(np.clip(r_sq, -1.0, 1.0)),
        residuals_rms=rms_res,
        converged=converged,
    )


# ── Depth-profile mapping ─────────────────────────────────────────


def map_t2_from_cpmg(
    tissue_layers: list[TissueLayer],
    esp_ms: float = 10.0,
    n_echoes: int = 16,
    tr_ms: float = 1000.0,
    depth_step_mm: float = 0.5,
    refocus_angle_deg: float = 180.0,
    n_states: int = 64,
) -> T2MapResult:
    """Map T₂ across stacked tissue layers via EPG CPMG simulation.

    For each depth slice the corresponding tissue properties (T₁, T₂) are
    looked up via ``_assign_layers``.  An EPG CPMG echo train is simulated
    and then fitted with a monoexponential decay to extract T₂.  Results
    are cached per unique (T₁, T₂) pair so each tissue type is only
    simulated once.

    Args:
        tissue_layers: ordered list of tissue layers (surface → depth).
        esp_ms: echo spacing in milliseconds.
        n_echoes: number of CPMG echoes.
        tr_ms: repetition time in milliseconds.
        depth_step_mm: depth sampling interval in mm.
        refocus_angle_deg: refocusing flip angle (180° = ideal CPMG).
        n_states: EPG configuration states to track.

    Returns:
        T2MapResult with depth-resolved fitted and reference T₂ values.
    """
    esp_s = esp_ms * 1e-3
    tr_s = tr_ms * 1e-3

    depths_mm = _build_depth_grid(tissue_layers, depth_step_mm)
    layer_props = _assign_layers(depths_mm, tissue_layers)

    echo_times_s = np.arange(1, n_echoes + 1, dtype=np.float64) * esp_s

    # Cache: (t1_key, t2_key) → T2FitResult
    _cache: dict[tuple[float, float], T2FitResult] = {}

    t2_fitted_list: list[float] = []
    t2_ref_list: list[float] = []
    s0_list: list[float] = []
    r2_list: list[float] = []
    res_list: list[float] = []
    labels: list[str] = []

    for props in layer_props:
        t1_s = props["t1_ms"] * 1e-3
        t2_s = props["t2_ms"] * 1e-3
        key = (_round6(t1_s), _round6(t2_s))

        if key not in _cache:
            echoes = epg_cpmg(
                t1_s, t2_s, esp_s, n_echoes,
                tr_s=tr_s,
                refocus_angle_deg=refocus_angle_deg,
                n_states=n_states,
            )
            _cache[key] = fit_t2_monoexponential(echoes, echo_times_s)

        fit = _cache[key]
        t2_fitted_list.append(fit.t2_s)
        t2_ref_list.append(t2_s)
        s0_list.append(fit.s0)
        r2_list.append(fit.r_squared)
        res_list.append(fit.residuals_rms)
        labels.append(props["name"])

    return T2MapResult(
        depths_mm=depths_mm,
        t2_fitted_s=np.array(t2_fitted_list),
        t2_reference_s=np.array(t2_ref_list),
        s0=np.array(s0_list),
        r_squared=np.array(r2_list),
        residuals_rms=np.array(res_list),
        tissue_labels=labels,
        esp_s=esp_s,
        n_echoes=n_echoes,
        tr_s=tr_s,
    )


def map_t1_from_saturation_recovery(
    tissue_layers: list[TissueLayer],
    tr_values_ms: NDArray[np.float64] | list[float] | None = None,
    te_ms: float = 10.0,
    depth_step_mm: float = 0.5,
    flip_angle_deg: float = 90.0,
    refocus_angle_deg: float = 180.0,
    n_states: int = 32,
) -> T1MapResult:
    """Map T₁ across stacked tissue layers via EPG saturation-recovery.

    At each depth, spin-echo signals are computed at multiple TR values via
    ``epg_signal``.  The resulting S(TR) curve is fitted with
    S(TR) = A (1 − exp(−TR/T₁)) to extract T₁.

    Args:
        tissue_layers: ordered list of tissue layers.
        tr_values_ms: TR values in milliseconds to sample.  Default spans
            from 0.5 × min(T₁) to 5 × max(T₁) with 8 logarithmically
            spaced points.
        te_ms: echo time in milliseconds.
        depth_step_mm: depth sampling interval in mm.
        flip_angle_deg: excitation flip angle.
        refocus_angle_deg: refocusing flip angle.
        n_states: EPG configuration states.

    Returns:
        T1MapResult with depth-resolved fitted and reference T₁ values.
    """
    te_s = te_ms * 1e-3

    # Auto-generate TR grid if not provided
    if tr_values_ms is None:
        t1_vals = [lay.t1_ms for lay in tissue_layers]
        t1_min_s = min(t1_vals) * 1e-3
        t1_max_s = max(t1_vals) * 1e-3
        tr_log = np.logspace(
            math.log10(max(0.5 * t1_min_s, te_s + 1e-4)),
            math.log10(5.0 * t1_max_s),
            num=10,
        )
        tr_values_s = tr_log
    else:
        tr_values_s = np.asarray(tr_values_ms, dtype=np.float64) * 1e-3

    depths_mm = _build_depth_grid(tissue_layers, depth_step_mm)
    layer_props = _assign_layers(depths_mm, tissue_layers)

    # Cache: (t1_key, t2_key) → T1FitResult
    _cache: dict[tuple[float, float], T1FitResult] = {}

    t1_fitted_list: list[float] = []
    t1_ref_list: list[float] = []
    amp_list: list[float] = []
    r2_list: list[float] = []
    res_list: list[float] = []
    labels: list[str] = []

    for props in layer_props:
        t1_s = props["t1_ms"] * 1e-3
        t2_s = props["t2_ms"] * 1e-3
        key = (_round6(t1_s), _round6(t2_s))

        if key not in _cache:
            signals = np.array([
                epg_signal(t1_s, t2_s, te_s, tr_i,
                           flip_angle_deg=flip_angle_deg,
                           refocus_angle_deg=refocus_angle_deg,
                           n_states=n_states)
                for tr_i in tr_values_s
            ])
            _cache[key] = fit_t1_saturation_recovery(signals, tr_values_s)

        fit = _cache[key]
        t1_fitted_list.append(fit.t1_s)
        t1_ref_list.append(t1_s)
        amp_list.append(fit.amplitude)
        r2_list.append(fit.r_squared)
        res_list.append(fit.residuals_rms)
        labels.append(props["name"])

    return T1MapResult(
        depths_mm=depths_mm,
        t1_fitted_s=np.array(t1_fitted_list),
        t1_reference_s=np.array(t1_ref_list),
        amplitude=np.array(amp_list),
        r_squared=np.array(r2_list),
        residuals_rms=np.array(res_list),
        tissue_labels=labels,
        tr_values_s=tr_values_s,
        te_s=te_s,
    )


def build_t1t2_map(
    tissue_layers: list[TissueLayer],
    esp_ms: float = 10.0,
    n_echoes: int = 16,
    tr_ms: float = 1000.0,
    tr_values_ms: NDArray[np.float64] | list[float] | None = None,
    te_ms: float = 10.0,
    depth_step_mm: float = 0.5,
    refocus_angle_deg: float = 180.0,
) -> T1T2Map:
    """Build a combined depth-resolved T₁ and T₂ map.

    Runs both ``map_t2_from_cpmg`` (CPMG echo-train T₂) and
    ``map_t1_from_saturation_recovery`` (multiple-TR T₁), then merges
    the results into a single ``T1T2Map``.

    Args:
        tissue_layers: ordered list of tissue layers.
        esp_ms: echo spacing for CPMG (ms).
        n_echoes: number CPMG echoes for T₂ mapping.
        tr_ms: CPMG repetition time (ms).
        tr_values_ms: TR values for T₁ saturation recovery (ms).
        te_ms: echo time for T₁ acquisition (ms).
        depth_step_mm: depth sampling step (mm).
        refocus_angle_deg: refocusing flip angle.

    Returns:
        T1T2Map with T₁, T₂, derived rates R₁=1/T₁, R₂=1/T₂, T₁/T₂ ratio.
    """
    t2_map = map_t2_from_cpmg(
        tissue_layers,
        esp_ms=esp_ms,
        n_echoes=n_echoes,
        tr_ms=tr_ms,
        depth_step_mm=depth_step_mm,
        refocus_angle_deg=refocus_angle_deg,
    )
    t1_map = map_t1_from_saturation_recovery(
        tissue_layers,
        tr_values_ms=tr_values_ms,
        te_ms=te_ms,
        depth_step_mm=depth_step_mm,
        refocus_angle_deg=refocus_angle_deg,
    )

    # Both maps share the same depth grid (same tissue_layers + depth_step_mm)
    t1_s = t1_map.t1_fitted_s
    t2_s = t2_map.t2_fitted_s

    # Guard against T2 = 0 (bone cortex edge case)
    t2_safe = np.where(t2_s > 0, t2_s, 1e-6)
    t1_safe = np.where(t1_s > 0, t1_s, 1e-6)

    return T1T2Map(
        depths_mm=t2_map.depths_mm,
        t1_s=t1_s,
        t2_s=t2_s,
        t1_reference_s=t1_map.t1_reference_s,
        t2_reference_s=t2_map.t2_reference_s,
        t1_r_squared=t1_map.r_squared,
        t2_r_squared=t2_map.r_squared,
        tissue_labels=t2_map.tissue_labels,
        t1t2_ratio=t1_safe / t2_safe,
        r1_s_inv=1.0 / t1_safe,
        r2_s_inv=1.0 / t2_safe,
    )


# ── Abnormality detection ─────────────────────────────────────────


def detect_tissue_abnormalities(
    observed_map: T2MapResult,
    reference_map: T2MapResult,
    t2_threshold: float = 0.25,
) -> list[AbnormalityFlag]:
    """Flag depth positions where observed T₂ deviates from a reference map.

    Depths are compared by position: for each depth in observed_map, the
    nearest depth in reference_map is found and the T₂ values are compared.
    Flags are raised when |observed_T₂ − reference_T₂| / reference_T₂
    exceeds *t2_threshold*.

    Args:
        observed_map: T₂ map for the tissue under investigation.
        reference_map: T₂ map for the healthy reference model.
        t2_threshold: fractional deviation threshold (default 0.25 = 25%).

    Returns:
        List of AbnormalityFlag instances, one per flagged depth.
    """
    flags: list[AbnormalityFlag] = []

    for i, depth in enumerate(observed_map.depths_mm):
        obs_t2 = float(observed_map.t2_fitted_s[i])

        # Find nearest depth in reference map
        j = int(np.argmin(np.abs(reference_map.depths_mm - depth)))
        ref_t2 = float(reference_map.t2_fitted_s[j])

        if ref_t2 <= 0:
            continue

        deviation = (obs_t2 - ref_t2) / ref_t2
        if abs(deviation) > t2_threshold:
            flag_type = "prolonged" if deviation > 0 else "shortened"
            flags.append(
                AbnormalityFlag(
                    depth_mm=float(depth),
                    tissue_label=observed_map.tissue_labels[i],
                    parameter="T2",
                    fitted_value_s=obs_t2,
                    reference_value_s=ref_t2,
                    deviation_fraction=float(deviation),
                    flag_type=flag_type,
                )
            )

    return flags


# ── Cross-validation ──────────────────────────────────────────────


def cross_validate_t1t2(
    map_a: T1T2Map,
    map_b: T1T2Map,
) -> T1T2CrossValidation:
    """Compare two T1T2Map instances at their shared depth range.

    Both maps must be generated with the same depth_step_mm so depth
    arrays align.  The comparison uses the shorter depth array (min length
    over both maps) to avoid boundary effects.

    Args:
        map_a: first T₁/T₂ map (e.g., ground truth).
        map_b: second T₁/T₂ map (e.g., fitted estimate).

    Returns:
        T1T2CrossValidation with correlation and relative-error statistics.
    """
    n = min(len(map_a.depths_mm), len(map_b.depths_mm))
    if n == 0:
        return T1T2CrossValidation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    t1_a = map_a.t1_s[:n]
    t1_b = map_b.t1_s[:n]
    t2_a = map_a.t2_s[:n]
    t2_b = map_b.t2_s[:n]

    t1_corr = _safe_pearson(t1_a, t1_b)
    t2_corr = _safe_pearson(t2_a, t2_b)

    t1_rel = np.abs(t1_a - t1_b) / np.where(t1_a > 0, t1_a, 1.0)
    t2_rel = np.abs(t2_a - t2_b) / np.where(t2_a > 0, t2_a, 1.0)

    return T1T2CrossValidation(
        t1_correlation=float(t1_corr),
        t2_correlation=float(t2_corr),
        t1_max_relative_error=float(np.max(t1_rel)),
        t2_max_relative_error=float(np.max(t2_rel)),
        t1_mean_relative_error=float(np.mean(t1_rel)),
        t2_mean_relative_error=float(np.mean(t2_rel)),
        n_depths=n,
    )


# ── Helpers ───────────────────────────────────────────────────────


def _build_depth_grid(
    tissue_layers: list[TissueLayer],
    depth_step_mm: float,
) -> NDArray[np.float64]:
    """Build a 1D depth array from surface to end of last layer."""
    total_mm = sum(lay.thickness_mm for lay in tissue_layers)
    return np.arange(depth_step_mm, total_mm + depth_step_mm * 0.5, depth_step_mm)


def _round6(x: float) -> float:
    """Round to 6 significant figures for cache key."""
    return round(x, 6)


def _safe_pearson(a: NDArray, b: NDArray) -> float:
    """Pearson correlation, returns 1.0 if any array has zero variance."""
    if np.std(a) < 1e-20 or np.std(b) < 1e-20:
        return 1.0 if np.allclose(a, b, rtol=1e-4) else 0.0
    return float(np.corrcoef(a, b)[0, 1])
