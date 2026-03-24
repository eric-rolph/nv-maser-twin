"""
Fundamental physical constants used throughout the NV maser physics stack.

Centralises values that were previously duplicated across 15+ modules.
All values use SI units.

References:
    CODATA 2018 recommended values (NIST SP 961, 2019).
"""
from __future__ import annotations

import math

# ── Quantum / statistical mechanics ──────────────────────────────
HBAR: float = 1.054571817e-34
"""Reduced Planck constant ℏ (J·s)."""

KB: float = 1.380649e-23
"""Boltzmann constant k_B (J/K)."""

# ── Electromagnetism ─────────────────────────────────────────────
MU0: float = 4.0 * math.pi * 1e-7
"""Vacuum permeability μ₀ (H/m = T·m/A)."""
