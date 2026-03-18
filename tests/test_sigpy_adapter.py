"""Tests for sigpy reconstruction adapter.

Tests cover:
- Encoding geometry construction
- Forward signal simulation (round-trip)
- Least-squares reconstruction
- L1-wavelet compressed sensing reconstruction
- Total-variation reconstruction
- RF pulse design
- Bloch simulation of excitation profiles
"""
import math

import numpy as np
import pytest

from nv_maser.config import DepthProfileConfig, SingleSidedMagnetConfig

try:
    import sigpy  # noqa: F401

    HAS_SIGPY = True
except ImportError:
    HAS_SIGPY = False

pytestmark = pytest.mark.skipif(not HAS_SIGPY, reason="sigpy not installed")

from nv_maser.physics.sigpy_adapter import (  # noqa: E402
    SIGPY_AVAILABLE,
    EncodingInfo,
    ReconResult,
    RFPulseResult,
    bloch_simulate_pulse,
    build_encoding_info,
    build_encoding_matrix,
    design_excitation_pulse,
    reconstruct_l1_wavelet,
    reconstruct_least_squares,
    reconstruct_total_variation,
    simulate_signal,
)
from nv_maser.physics.single_sided_magnet import SingleSidedMagnet  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def magnet_config() -> SingleSidedMagnetConfig:
    """Default barrel magnet configuration."""
    return SingleSidedMagnetConfig()


@pytest.fixture
def depth_config() -> DepthProfileConfig:
    """Depth profile config with coarse resolution for fast tests."""
    return DepthProfileConfig(
        max_depth_mm=20.0,
        depth_resolution_mm=1.0,
    )


@pytest.fixture
def magnet(magnet_config: SingleSidedMagnetConfig) -> SingleSidedMagnet:
    return SingleSidedMagnet(magnet_config)


@pytest.fixture
def encoding(
    magnet: SingleSidedMagnet, depth_config: DepthProfileConfig
) -> EncodingInfo:
    return build_encoding_info(magnet, depth_config)


# ── TestEncodingGeometry ──────────────────────────────────────────


class TestEncodingGeometry:
    def test_encoding_info_shapes(self, encoding: EncodingInfo):
        n = len(encoding.depths_mm)
        assert n > 0
        assert encoding.b0_tesla.shape == (n,)
        assert encoding.larmor_hz.shape == (n,)
        assert encoding.gradient_t_per_m.shape == (n,)

    def test_larmor_positive(self, encoding: EncodingInfo):
        """Larmor frequencies must be positive."""
        assert np.all(encoding.larmor_hz > 0)

    def test_larmor_consistent_with_b0(self, encoding: EncodingInfo):
        """f = γ |B₀| within rounding tolerance."""
        expected = np.abs(encoding.b0_tesla) * 42.577e6
        np.testing.assert_allclose(encoding.larmor_hz, expected, rtol=1e-10)

    def test_encoding_matrix_shape(self, encoding: EncodingInfo):
        n_time = 64
        dwell = 1e-5
        E = build_encoding_matrix(encoding, n_time, dwell)
        assert E.shape == (n_time, len(encoding.depths_mm))
        assert E.dtype == np.complex128

    def test_encoding_matrix_unit_norm_columns(self, encoding: EncodingInfo):
        """At t=0, all columns should be 1 (exp(0)=1)."""
        n_time = 64
        dwell = 1e-5
        E = build_encoding_matrix(encoding, n_time, dwell)
        # First row (t=0): all should be exp(0) = 1
        np.testing.assert_allclose(np.abs(E[0, :]), 1.0, atol=1e-12)


# ── TestForwardSimulation ─────────────────────────────────────────


class TestForwardSimulation:
    def test_zero_profile_gives_zero_signal(self, encoding: EncodingInfo):
        profile = np.zeros(len(encoding.depths_mm))
        sig = simulate_signal(encoding, profile, 32, 1e-5)
        np.testing.assert_allclose(np.abs(sig), 0.0, atol=1e-15)

    def test_uniform_profile_nonzero_signal(self, encoding: EncodingInfo):
        profile = np.ones(len(encoding.depths_mm))
        sig = simulate_signal(encoding, profile, 32, 1e-5)
        assert np.max(np.abs(sig)) > 0

    def test_noise_adds_variance(self, encoding: EncodingInfo):
        profile = np.ones(len(encoding.depths_mm))
        sig_clean = simulate_signal(encoding, profile, 64, 1e-5, noise_std=0.0)
        sig_noisy = simulate_signal(
            encoding, profile, 64, 1e-5, noise_std=0.1,
            rng=np.random.default_rng(42),
        )
        diff = np.abs(sig_noisy - sig_clean)
        assert np.mean(diff) > 0.01

    def test_signal_length(self, encoding: EncodingInfo):
        sig = simulate_signal(encoding, np.ones(len(encoding.depths_mm)), 128, 1e-5)
        assert len(sig) == 128


# ── TestLeastSquaresRecon ─────────────────────────────────────────


class TestLeastSquaresRecon:
    def test_round_trip_clean(self, encoding: EncodingInfo):
        """Recon of clean forward signal should approximate the original."""
        n_depth = len(encoding.depths_mm)
        # Simple profile: higher in middle
        profile_true = np.zeros(n_depth)
        profile_true[n_depth // 4 : 3 * n_depth // 4] = 1.0

        # Oversample in time for a well-conditioned problem
        n_time = n_depth * 4
        dwell = 1e-5
        sig = simulate_signal(encoding, profile_true, n_time, dwell)

        result = reconstruct_least_squares(sig, encoding, dwell, max_iter=200)
        assert isinstance(result, ReconResult)
        assert result.method == "least_squares"
        # Correlation with true profile should be high
        corr = np.corrcoef(result.profile, profile_true)[0, 1]
        assert corr > 0.8, f"Correlation too low: {corr:.3f}"

    def test_result_fields(self, encoding: EncodingInfo):
        sig = simulate_signal(encoding, np.ones(len(encoding.depths_mm)), 64, 1e-5)
        result = reconstruct_least_squares(sig, encoding, 1e-5, max_iter=10)
        assert result.depths_mm.shape == encoding.depths_mm.shape
        assert result.profile.shape == encoding.depths_mm.shape
        assert result.residual_norm >= 0
        assert result.iterations > 0


# ── TestL1WaveletRecon ────────────────────────────────────────────


class TestL1WaveletRecon:
    def test_sparse_profile_recovery(self, encoding: EncodingInfo):
        """L1-wavelet should recover sparse (step) profiles well."""
        n_depth = len(encoding.depths_mm)
        profile_true = np.zeros(n_depth)
        profile_true[n_depth // 3 : 2 * n_depth // 3] = 1.0

        n_time = n_depth * 3
        dwell = 1e-5
        sig = simulate_signal(encoding, profile_true, n_time, dwell)

        result = reconstruct_l1_wavelet(
            sig, encoding, dwell, lamda=0.001, max_iter=100
        )
        assert result.method == "l1_wavelet"
        assert result.residual_norm >= 0
        # Profile should be non-negative and have reasonable structure
        assert np.max(result.profile) > 0

    def test_noisy_signal(self, encoding: EncodingInfo):
        """L1-wavelet should still produce a reasonable result with noise."""
        n_depth = len(encoding.depths_mm)
        profile_true = np.zeros(n_depth)
        profile_true[5:15] = 1.0

        n_time = n_depth * 3
        dwell = 1e-5
        sig = simulate_signal(
            encoding, profile_true, n_time, dwell, noise_std=0.05,
            rng=np.random.default_rng(99),
        )
        result = reconstruct_l1_wavelet(
            sig, encoding, dwell, lamda=0.01, max_iter=100
        )
        assert np.max(result.profile) > 0


# ── TestTotalVariationRecon ───────────────────────────────────────


class TestTotalVariationRecon:
    def test_step_profile(self, encoding: EncodingInfo):
        """TV recon should recover step-function profiles."""
        n_depth = len(encoding.depths_mm)
        profile_true = np.zeros(n_depth)
        profile_true[n_depth // 4 : 3 * n_depth // 4] = 1.0

        n_time = n_depth * 3
        dwell = 1e-5
        sig = simulate_signal(encoding, profile_true, n_time, dwell)

        result = reconstruct_total_variation(
            sig, encoding, dwell, lamda=0.001, max_iter=100
        )
        assert result.method == "total_variation"
        assert np.max(result.profile) > 0

    def test_result_shape(self, encoding: EncodingInfo):
        sig = simulate_signal(encoding, np.ones(len(encoding.depths_mm)), 64, 1e-5)
        result = reconstruct_total_variation(sig, encoding, 1e-5, max_iter=10)
        assert result.depths_mm.shape == encoding.depths_mm.shape
        assert result.profile.shape == encoding.depths_mm.shape


# ── TestRFPulseDesign ─────────────────────────────────────────────


class TestRFPulseDesign:
    def test_sinc_pulse_basic(self):
        result = design_excitation_pulse(
            bandwidth_hz=5000, duration_us=2000, n_points=128
        )
        assert isinstance(result, RFPulseResult)
        assert len(result.pulse) == 128
        assert len(result.time_us) == 128
        assert result.duration_us == 2000.0
        assert result.bandwidth_hz == 5000
        assert result.time_bandwidth == pytest.approx(10.0, rel=1e-6)

    def test_adiabatic_pulse(self):
        result = design_excitation_pulse(
            bandwidth_hz=10000,
            duration_us=5000,
            n_points=512,
            pulse_type="adiabatic",
        )
        assert isinstance(result, RFPulseResult)
        assert len(result.pulse) >= 512
        assert result.duration_us == 5000.0

    def test_invalid_pulse_type(self):
        with pytest.raises(ValueError, match="Unknown pulse type"):
            design_excitation_pulse(1000, pulse_type="invalid")

    def test_pulse_has_energy(self):
        result = design_excitation_pulse(bandwidth_hz=5000, n_points=128)
        energy = np.sum(np.abs(result.pulse) ** 2)
        assert energy > 0


# ── TestBlochSimulation ───────────────────────────────────────────


class TestBlochSimulation:
    def test_on_resonance_larger_than_off(self):
        """On-resonance excitation should produce more Mxy than far off-resonance."""
        pulse_result = design_excitation_pulse(
            bandwidth_hz=5000, duration_us=2000, n_points=128
        )
        dt_s = pulse_result.duration_us * 1e-6 / len(pulse_result.pulse)
        offsets = np.array([0.0, 50000.0])  # on-resonance vs far off

        mxy = bloch_simulate_pulse(pulse_result.pulse, dt_s, offsets)
        assert np.abs(mxy[0]) > np.abs(mxy[1])

    def test_excitation_profile_shape(self):
        pulse_result = design_excitation_pulse(
            bandwidth_hz=5000, duration_us=2000, n_points=128
        )
        dt_s = pulse_result.duration_us * 1e-6 / len(pulse_result.pulse)
        offsets = np.linspace(-20000, 20000, 101)

        mxy = bloch_simulate_pulse(pulse_result.pulse, dt_s, offsets)
        assert mxy.shape == (101,)
        # Maximum should be near centre (on-resonance)
        peak_idx = np.argmax(np.abs(mxy))
        assert abs(peak_idx - 50) < 15  # within ±15 bins of centre


# ── TestSigpyAvailability ────────────────────────────────────────


class TestSigpyAvailability:
    def test_flag_is_true(self):
        assert SIGPY_AVAILABLE is True
