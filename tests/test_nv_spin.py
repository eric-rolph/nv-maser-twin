"""Tests for NV center spin physics module."""
import numpy as np
import pytest

from nv_maser.config import NVConfig
from nv_maser.physics.nv_spin import (
    effective_linewidth_ghz,
    homogeneous_linewidth_ghz,
    inhomogeneous_linewidth_ghz,
    transition_frequencies,
)


@pytest.fixture
def nv_config() -> NVConfig:
    return NVConfig()


# ── transition_frequencies ─────────────────────────────────────────


def test_zero_field_degeneracy(nv_config: NVConfig) -> None:
    """At B=0 both transitions equal the zero-field splitting D."""
    b = np.zeros((4, 4), dtype=np.float32)
    nu_plus, nu_minus = transition_frequencies(b, nv_config)
    assert np.allclose(nu_plus, 2.87, atol=1e-6)
    assert np.allclose(nu_minus, 2.87, atol=1e-6)


def test_zeeman_splitting(nv_config: NVConfig) -> None:
    """At 50 mT the two branches split by 2·γe·B."""
    b = np.full((4, 4), 0.05, dtype=np.float32)
    nu_plus, nu_minus = transition_frequencies(b, nv_config)
    expected_split = 2 * 28.025 * 0.05  # 2·γe·B
    actual_split = float(np.mean(nu_plus - nu_minus))
    assert abs(actual_split - expected_split) < 1e-4


def test_transition_frequencies_at_50mt(nv_config: NVConfig) -> None:
    """Verify specific values at the default B₀ = 50 mT."""
    b = np.full((2, 2), 0.05, dtype=np.float32)
    nu_plus, nu_minus = transition_frequencies(b, nv_config)
    # ν+ = 2.87 + 28.025×0.05 = 4.27125
    # ν- = 2.87 - 28.025×0.05 = 1.46875
    assert abs(float(nu_plus[0, 0]) - 4.27125) < 1e-4
    assert abs(float(nu_minus[0, 0]) - 1.46875) < 1e-4


def test_transition_freq_shape() -> None:
    """Output shape matches input."""
    b = np.ones((16, 16), dtype=np.float32) * 0.01
    nu_p, nu_m = transition_frequencies(b, NVConfig())
    assert nu_p.shape == (16, 16)
    assert nu_m.shape == (16, 16)


# ── homogeneous_linewidth ──────────────────────────────────────────


def test_homogeneous_linewidth_1us() -> None:
    """Γ_h at T2* = 1 μs ≈ 0.318 MHz ≈ 3.18e-4 GHz."""
    lw = homogeneous_linewidth_ghz(1.0)
    expected = 1.0 / (np.pi * 1e-6) / 1e9  # Hz → GHz
    assert abs(lw - expected) < 1e-8


def test_homogeneous_linewidth_10us() -> None:
    """Longer T2* → narrower linewidth."""
    lw1 = homogeneous_linewidth_ghz(1.0)
    lw10 = homogeneous_linewidth_ghz(10.0)
    assert lw10 < lw1
    assert abs(lw10 * 10 - lw1) < 1e-8  # should scale as 1/T2*


# ── inhomogeneous_linewidth ────────────────────────────────────────


def test_uniform_field_zero_inh(nv_config: NVConfig) -> None:
    """Perfectly uniform field → zero inhomogeneous broadening."""
    b = np.full((8, 8), 0.05, dtype=np.float32)
    mask = np.ones((8, 8), dtype=bool)
    lw = inhomogeneous_linewidth_ghz(b, mask, nv_config)
    assert lw == pytest.approx(0.0, abs=1e-5)  # float32 has ~7 digits


def test_nonuniform_field_positive_inh(nv_config: NVConfig) -> None:
    """Non-uniform field → positive inhomogeneous broadening."""
    rng = np.random.default_rng(42)
    b = np.full((8, 8), 0.05, dtype=np.float32)
    b += rng.normal(0, 1e-6, size=(8, 8)).astype(np.float32)  # 1 μT noise
    mask = np.ones((8, 8), dtype=bool)
    lw = inhomogeneous_linewidth_ghz(b, mask, nv_config)
    assert lw > 0
    # Expected: γe · σ(B) ≈ 28.025 × 1e-6 ≈ 2.8e-5 GHz
    assert lw == pytest.approx(28.025 * float(np.std(b)), rel=1e-3)


# ── effective_linewidth ────────────────────────────────────────────


def test_effective_linewidth_uniform(nv_config: NVConfig) -> None:
    """Uniform field → effective = homogeneous only."""
    b = np.full((8, 8), 0.05, dtype=np.float32)
    mask = np.ones((8, 8), dtype=bool)
    g_eff, g_h, g_inh = effective_linewidth_ghz(b, mask, nv_config)
    assert g_inh == pytest.approx(0.0, abs=1e-5)  # float32 precision
    assert g_eff == pytest.approx(g_h, abs=1e-5)


def test_effective_linewidth_sum(nv_config: NVConfig) -> None:
    """Γ_eff = Γ_h + Γ_inh (Lorentzian convolution)."""
    rng = np.random.default_rng(42)
    b = np.full((16, 16), 0.05, dtype=np.float32) + rng.normal(
        0, 1e-5, size=(16, 16)
    ).astype(np.float32)
    mask = np.ones((16, 16), dtype=bool)
    g_eff, g_h, g_inh = effective_linewidth_ghz(b, mask, nv_config)
    assert g_eff == pytest.approx(g_h + g_inh, abs=1e-12)
