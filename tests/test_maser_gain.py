"""Tests for maser gain model."""
import numpy as np
import pytest

from nv_maser.config import NVConfig, MaserConfig
from nv_maser.physics.maser_gain import (
    compute_gain_budget,
    compute_maser_metrics,
    max_tolerable_b_std,
)


@pytest.fixture
def nv_config() -> NVConfig:
    return NVConfig()


@pytest.fixture
def maser_config() -> MaserConfig:
    return MaserConfig()


# ── compute_gain_budget ────────────────────────────────────────────


def test_gain_budget_uniform_field(nv_config: NVConfig) -> None:
    """Perfectly uniform field → gain budget = 1.0."""
    b = np.full((8, 8), 0.05, dtype=np.float32)
    mask = np.ones((8, 8), dtype=bool)
    gb = compute_gain_budget(b, mask, nv_config)
    assert gb == pytest.approx(1.0, abs=1e-3)  # float32 precision


def test_gain_budget_noisy_field(nv_config: NVConfig) -> None:
    """Non-uniform field → gain budget < 1.0."""
    rng = np.random.default_rng(42)
    b = np.full((32, 32), 0.05, dtype=np.float32)
    b += rng.normal(0, 1e-5, size=(32, 32)).astype(np.float32)
    mask = np.ones((32, 32), dtype=bool)
    gb = compute_gain_budget(b, mask, nv_config)
    assert 0 < gb < 1.0


def test_gain_budget_decreases_with_noise(nv_config: NVConfig) -> None:
    """More noise → lower gain budget."""
    mask = np.ones((32, 32), dtype=bool)
    rng = np.random.default_rng(42)

    b_low = np.full((32, 32), 0.05, dtype=np.float32)
    b_low += rng.normal(0, 1e-7, size=(32, 32)).astype(np.float32)

    b_high = np.full((32, 32), 0.05, dtype=np.float32)
    b_high += rng.normal(0, 1e-4, size=(32, 32)).astype(np.float32)

    gb_low = compute_gain_budget(b_low, mask, nv_config)
    gb_high = compute_gain_budget(b_high, mask, nv_config)
    assert gb_low > gb_high


# ── compute_maser_metrics ──────────────────────────────────────────


def test_maser_metrics_keys(
    nv_config: NVConfig, maser_config: MaserConfig
) -> None:
    """metrics dict has all expected keys."""
    b = np.full((8, 8), 0.05, dtype=np.float32)
    mask = np.ones((8, 8), dtype=bool)
    m = compute_maser_metrics(b, mask, nv_config, maser_config)

    expected_keys = {
        "gain_budget",
        "gamma_h_ghz",
        "gamma_inh_ghz",
        "gamma_eff_ghz",
        "transition_freq_mean_ghz",
        "transition_freq_spread_ghz",
        "b_std_tesla",
        "b_ptp_tesla",
        "maser_margin",
        "masing",
    }
    assert set(m.keys()) == expected_keys


def test_uniform_field_is_masing(
    nv_config: NVConfig, maser_config: MaserConfig
) -> None:
    """Perfectly uniform field should be above maser threshold."""
    b = np.full((16, 16), 0.05, dtype=np.float32)
    mask = np.ones((16, 16), dtype=bool)
    m = compute_maser_metrics(b, mask, nv_config, maser_config)
    assert m["masing"] is True
    assert m["gain_budget"] == pytest.approx(1.0, abs=1e-3)  # float32
    assert m["maser_margin"] > 0


def test_very_noisy_field_not_masing(
    nv_config: NVConfig, maser_config: MaserConfig
) -> None:
    """Very noisy field should be below maser threshold."""
    rng = np.random.default_rng(99)
    b = np.full((16, 16), 0.05, dtype=np.float32)
    b += rng.normal(0, 1e-3, size=(16, 16)).astype(np.float32)  # 1 mT noise
    mask = np.ones((16, 16), dtype=bool)
    m = compute_maser_metrics(b, mask, nv_config, maser_config)
    assert m["masing"] is False
    assert m["maser_margin"] < 0
    assert m["gain_budget"] < 0.1


def test_transition_freq_at_50mt(
    nv_config: NVConfig, maser_config: MaserConfig
) -> None:
    """Mean transition frequency at 50 mT should be ~1.469 GHz (lower branch)."""
    b = np.full((8, 8), 0.05, dtype=np.float32)
    mask = np.ones((8, 8), dtype=bool)
    m = compute_maser_metrics(b, mask, nv_config, maser_config)
    assert m["transition_freq_mean_ghz"] == pytest.approx(1.46875, abs=0.01)


# ── max_tolerable_b_std ────────────────────────────────────────────


def test_max_b_std_positive(nv_config: NVConfig, maser_config: MaserConfig) -> None:
    """The tolerance is positive and finite."""
    sigma = max_tolerable_b_std(nv_config, maser_config)
    assert sigma > 0
    assert np.isfinite(sigma)


def test_max_b_std_formula(nv_config: NVConfig, maser_config: MaserConfig) -> None:
    """Verify against hand calculation.

    T2* = 1 μs → Γ_h = 1/(π·1e-6) Hz = 318,310 Hz = 3.183e-4 GHz
    min_budget = 0.5 → σ_B = Γ_h · (1/0.5 - 1) / γe = Γ_h / γe
    = 3.183e-4 / 28.025 ≈ 1.136e-5 T ≈ 11.36 μT
    """
    sigma = max_tolerable_b_std(nv_config, maser_config)
    expected = (1.0 / (np.pi * 1e-6) / 1e9) / 28.025  # Γ_h/γe (budget=0.5 → ×1)
    assert sigma == pytest.approx(expected, rel=1e-6)


def test_tighter_budget_reduces_tolerance(nv_config: NVConfig) -> None:
    """Higher min_gain_budget → stricter tolerance on σ(B)."""
    loose = MaserConfig(min_gain_budget=0.3)
    tight = MaserConfig(min_gain_budget=0.8)
    assert max_tolerable_b_std(nv_config, tight) < max_tolerable_b_std(
        nv_config, loose
    )


def test_better_diamond_reduces_tolerance() -> None:
    """Longer T2* → narrower Γ_h → LESS tolerance for B non-uniformity.

    σ_B_max = Γ_h·(1/budget - 1)/γe, and Γ_h ∝ 1/T2*, so longer T2*
    means the maser is more sensitive to field inhomogeneity.
    """
    maser = MaserConfig()
    nv_short = NVConfig(t2_star_us=1.0)
    nv_long = NVConfig(t2_star_us=10.0)
    assert max_tolerable_b_std(nv_long, maser) < max_tolerable_b_std(
        nv_short, maser
    )


def test_budget_1_returns_zero(nv_config: NVConfig) -> None:
    """min_gain_budget=1.0 → only perfectly uniform field is acceptable."""
    maser = MaserConfig(min_gain_budget=1.0)
    assert max_tolerable_b_std(nv_config, maser) == 0.0


# ── integration: gain_budget consistent with max_tolerable_b_std ──


def test_at_max_tolerance_budget_matches_threshold(
    nv_config: NVConfig, maser_config: MaserConfig
) -> None:
    """Field with σ(B) = max_tolerable → gain_budget ≈ min_gain_budget."""
    sigma_max = max_tolerable_b_std(nv_config, maser_config)

    # Create a field with exactly that standard deviation
    rng = np.random.default_rng(0)
    b = np.full((128, 128), 0.05, dtype=np.float32)
    noise = rng.normal(0, 1, size=(128, 128)).astype(np.float32)
    noise = noise / float(np.std(noise)) * sigma_max  # exact σ = sigma_max
    b += noise

    mask = np.ones((128, 128), dtype=bool)
    gb = compute_gain_budget(b, mask, nv_config)
    assert gb == pytest.approx(maser_config.min_gain_budget, rel=0.01)
