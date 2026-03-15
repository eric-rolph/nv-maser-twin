"""
Maser gain model: single-pass gain and threshold margin.

This module answers the central question: **given the B₀ field
uniformity produced by the shimming system, can the NV maser lase?**

Key concept — *gain budget*:

    gain_budget = Γ_h / Γ_eff = Γ_h / (Γ_h + γe · σ(B))

This is the fraction of ideal (homogeneously-broadened) gain that
survives the inhomogeneous broadening introduced by B₀ non-uniformity.

    1.0  →  perfect field  →  shimming does not limit gain
    0.0  →  infinite broadening  →  no maser possible

When the gain budget drops below ``min_gain_budget`` (set by cavity
loss, NV density, and pump power), masing ceases.

The **maser margin** = gain_budget / min_gain_budget − 1 gives the
safety factor: positive means masing, negative means below threshold.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import NVConfig, MaserConfig
from .nv_spin import (
    effective_linewidth_ghz,
    homogeneous_linewidth_ghz,
    transition_frequencies,
)


def compute_gain_budget(
    b_field: NDArray[np.float32],
    active_mask: NDArray[np.bool_],
    nv_config: NVConfig,
) -> float:
    """
    Fraction of peak gain retained given B₀ non-uniformity.

    gain_budget = Γ_h / (Γ_h + Γ_inh)  ∈ (0, 1]

    Args:
        b_field:     (H, W) net magnetic field in Tesla.
        active_mask: (H, W) boolean mask for diamond active zone.
        nv_config:   NV center parameters.

    Returns:
        Gain budget factor in (0, 1].
    """
    gamma_eff, gamma_h, _ = effective_linewidth_ghz(
        b_field, active_mask, nv_config
    )
    if gamma_eff <= 0:
        return 0.0
    return gamma_h / gamma_eff


def compute_maser_metrics(
    b_field: NDArray[np.float32],
    active_mask: NDArray[np.bool_],
    nv_config: NVConfig,
    maser_config: MaserConfig,
) -> dict[str, float]:
    """
    Comprehensive maser performance metrics from a field map.

    Returns dict with:
        gain_budget              Γ_h / Γ_eff  (0–1)
        gamma_h_ghz             homogeneous linewidth
        gamma_inh_ghz           inhomogeneous linewidth from B₀
        gamma_eff_ghz           total effective linewidth
        transition_freq_mean_ghz  mean ν− over active zone
        transition_freq_spread_ghz  std(ν−) over active zone
        b_std_tesla             σ(B) over active zone
        b_ptp_tesla             peak-to-peak B variation
        maser_margin            (gain_budget / min_budget) − 1
        masing                  bool — above threshold?
    """
    gamma_eff, gamma_h, gamma_inh = effective_linewidth_ghz(
        b_field, active_mask, nv_config
    )

    gain_budget = gamma_h / gamma_eff if gamma_eff > 0 else 0.0

    # Transition frequencies across diamond (lower branch = maser transition)
    _, nu_minus = transition_frequencies(b_field, nv_config)
    active_nu = nu_minus[active_mask]

    # B field statistics over active zone
    active_b = b_field[active_mask]
    b_std = float(np.std(active_b))
    b_ptp = float(np.ptp(active_b))

    # Maser margin: how far above/below threshold
    min_budget = maser_config.min_gain_budget
    margin = (gain_budget / min_budget - 1.0) if min_budget > 0 else float("inf")

    return {
        "gain_budget": gain_budget,
        "gamma_h_ghz": gamma_h,
        "gamma_inh_ghz": gamma_inh,
        "gamma_eff_ghz": gamma_eff,
        "transition_freq_mean_ghz": float(np.mean(active_nu)),
        "transition_freq_spread_ghz": float(np.std(active_nu)),
        "b_std_tesla": b_std,
        "b_ptp_tesla": b_ptp,
        "maser_margin": margin,
        "masing": gain_budget >= min_budget,
    }


def max_tolerable_b_std(nv_config: NVConfig, maser_config: MaserConfig) -> float:
    """
    Maximum tolerable σ(B) for maser operation.

    At threshold:  Γ_h / (Γ_h + γe · σ_B) = min_gain_budget
    Solving:       σ_B = Γ_h · (1/budget − 1) / γe

    Returns:
        Maximum σ(B) in Tesla.
    """
    gamma_h = homogeneous_linewidth_ghz(nv_config.t2_star_us)
    budget = maser_config.min_gain_budget

    if budget >= 1.0:
        return 0.0
    if budget <= 0.0:
        return float("inf")

    return gamma_h * (1.0 / budget - 1.0) / nv_config.gamma_e_ghz_per_t
