"""
NV center spin physics: ground-state Hamiltonian, energy levels, transition frequencies.

Models the nitrogen-vacancy center ground-state triplet (³A₂) with spin-1
states |0⟩, |+1⟩, |−1⟩.

Hamiltonian (axial approximation, B along NV axis):

    H = D·Sz² + γe·B·Sz

where D = 2.87 GHz is the zero-field splitting and γe = 28.025 GHz/T is
the electron gyromagnetic ratio for NV⁻ in diamond.

Energy levels:
    E(|0⟩)  = 0
    E(|+1⟩) = D + γe·B
    E(|−1⟩) = D − γe·B

This module computes spatially-resolved transition frequencies from a B₀ field
map, and the linewidth contributions that determine maser gain.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import NVConfig


def transition_frequencies(
    b_field: NDArray[np.float32],
    config: NVConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Compute NV transition frequencies at each spatial point.

    For field B along the NV axis:
        ν+ = D + γe·B   (|0⟩ → |+1⟩)
        ν− = D − γe·B   (|0⟩ → |−1⟩)

    Args:
        b_field: (...) magnetic field magnitude in Tesla.
        config:  NV center configuration.

    Returns:
        (nu_plus, nu_minus): each same shape as b_field, in GHz.
    """
    D = config.zero_field_splitting_ghz
    gamma = config.gamma_e_ghz_per_t

    nu_plus = (D + gamma * b_field).astype(np.float32)
    nu_minus = (D - gamma * b_field).astype(np.float32)

    return nu_plus, nu_minus


def homogeneous_linewidth_ghz(t2_star_us: float) -> float:
    """
    Compute the homogeneous linewidth from T2*.

    Γ_h = 1 / (π · T2*)

    Args:
        t2_star_us: T2* dephasing time in microseconds.

    Returns:
        Linewidth in GHz.
    """
    t2_star_s = t2_star_us * 1e-6
    return 1.0 / (np.pi * t2_star_s) / 1e9


def inhomogeneous_linewidth_ghz(
    b_field: NDArray[np.float32],
    active_mask: NDArray[np.bool_],
    config: NVConfig,
) -> float:
    """
    Compute inhomogeneous linewidth due to B₀ non-uniformity.

    Γ_inh = γe · σ(B)

    where σ(B) is the standard deviation of B over the active zone.

    Args:
        b_field:     (H, W) magnetic field in Tesla.
        active_mask: (H, W) boolean mask for diamond active zone.
        config:      NV center configuration.

    Returns:
        Inhomogeneous linewidth in GHz.
    """
    active_field = b_field[active_mask]
    b_std = float(np.std(active_field))
    return config.gamma_e_ghz_per_t * b_std


def effective_linewidth_ghz(
    b_field: NDArray[np.float32],
    active_mask: NDArray[np.bool_],
    config: NVConfig,
) -> tuple[float, float, float]:
    """
    Compute total effective linewidth: homogeneous + inhomogeneous.

    For Lorentzian lineshapes the widths add directly:
        Γ_eff = Γ_h + Γ_inh

    Returns:
        (gamma_eff, gamma_h, gamma_inh) all in GHz.
    """
    gamma_h = homogeneous_linewidth_ghz(config.t2_star_us)
    gamma_inh = inhomogeneous_linewidth_ghz(b_field, active_mask, config)
    gamma_eff = gamma_h + gamma_inh
    return gamma_eff, gamma_h, gamma_inh
