"""Tests for the 1D NMR depth profiling simulator."""
import math

import numpy as np
import pytest

from nv_maser.config import (
    DepthProfileConfig,
    SingleSidedMagnetConfig,
    SurfaceCoilConfig,
)
from nv_maser.physics.depth_profile import (
    TissueLayer,
    DepthProfile,
    FOREARM_LAYERS,
    HEMORRHAGE_LAYERS,
    _equilibrium_magnetisation,
    _assign_layers,
    simulate_depth_profile,
    add_noise,
)
from nv_maser.physics.single_sided_magnet import SingleSidedMagnet
from nv_maser.physics.surface_coil import SurfaceCoil


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def magnet() -> SingleSidedMagnet:
    return SingleSidedMagnet(SingleSidedMagnetConfig())


@pytest.fixture
def coil() -> SurfaceCoil:
    return SurfaceCoil(SurfaceCoilConfig())


@pytest.fixture
def dp_config() -> DepthProfileConfig:
    return DepthProfileConfig()


# ── TissueLayer ───────────────────────────────────────────────────


class TestTissueLayer:
    def test_valid_layer(self) -> None:
        layer = TissueLayer("muscle", thickness_mm=10.0, proton_density=1.0)
        assert layer.name == "muscle"

    def test_zero_thickness_raises(self) -> None:
        with pytest.raises(ValueError, match="Thickness"):
            TissueLayer("bad", thickness_mm=0.0)

    def test_negative_thickness_raises(self) -> None:
        with pytest.raises(ValueError, match="Thickness"):
            TissueLayer("bad", thickness_mm=-1.0)

    def test_negative_density_raises(self) -> None:
        with pytest.raises(ValueError, match="Proton density"):
            TissueLayer("bad", thickness_mm=1.0, proton_density=-0.5)

    def test_zero_density_allowed(self) -> None:
        layer = TissueLayer("void", thickness_mm=1.0, proton_density=0.0)
        assert layer.proton_density == 0.0


# ── Preset tissue models ─────────────────────────────────────────


class TestPresetLayers:
    def test_forearm_has_expected_count(self) -> None:
        assert len(FOREARM_LAYERS) == 4

    def test_hemorrhage_has_expected_count(self) -> None:
        assert len(HEMORRHAGE_LAYERS) == 4

    def test_forearm_layer_names(self) -> None:
        names = [l.name for l in FOREARM_LAYERS]
        assert "skin" in names
        assert "muscle" in names
        assert "bone_cortex" in names

    def test_hemorrhage_layer_names(self) -> None:
        names = [l.name for l in HEMORRHAGE_LAYERS]
        assert "hemorrhage" in names

    def test_hemorrhage_t2_longer_than_surrounding(self) -> None:
        """Hemorrhage has high T2 (~150 ms) vs. muscle (~35 ms)."""
        hem = next(l for l in HEMORRHAGE_LAYERS if l.name == "hemorrhage")
        muscle = next(l for l in HEMORRHAGE_LAYERS if l.name == "muscle")
        assert hem.t2_ms > muscle.t2_ms

    def test_all_thicknesses_positive(self) -> None:
        for layer in FOREARM_LAYERS + HEMORRHAGE_LAYERS:
            assert layer.thickness_mm > 0


# ── Equilibrium magnetisation ─────────────────────────────────────


class TestEquilibriumMag:
    def test_positive(self) -> None:
        m0 = _equilibrium_magnetisation(0.05, 1.0)
        assert m0 > 0

    def test_scales_with_b0(self) -> None:
        m1 = _equilibrium_magnetisation(0.05, 1.0)
        m2 = _equilibrium_magnetisation(0.10, 1.0)
        assert math.isclose(m2 / m1, 2.0, rel_tol=1e-10)

    def test_scales_with_density(self) -> None:
        m_half = _equilibrium_magnetisation(0.05, 0.5)
        m_full = _equilibrium_magnetisation(0.05, 1.0)
        assert math.isclose(m_full / m_half, 2.0, rel_tol=1e-10)

    def test_decreases_with_temperature(self) -> None:
        m_cold = _equilibrium_magnetisation(0.05, 1.0, temperature_k=250.0)
        m_hot = _equilibrium_magnetisation(0.05, 1.0, temperature_k=350.0)
        assert m_cold > m_hot

    def test_zero_field(self) -> None:
        assert _equilibrium_magnetisation(0.0, 1.0) == 0.0


# ── Layer assignment ──────────────────────────────────────────────


class TestAssignLayers:
    def test_correct_count(self) -> None:
        depths = np.array([1.0, 3.0, 10.0, 25.0])
        result = _assign_layers(depths, FOREARM_LAYERS)
        assert len(result) == 4

    def test_skin_at_surface(self) -> None:
        depths = np.array([1.0])
        result = _assign_layers(depths, FOREARM_LAYERS)
        assert result[0]["name"] == "skin"

    def test_muscle_at_depth(self) -> None:
        # skin=2mm, fat=5mm → muscle starts at 7mm
        depths = np.array([10.0])
        result = _assign_layers(depths, FOREARM_LAYERS)
        assert result[0]["name"] == "muscle"

    def test_beyond_all_layers(self) -> None:
        """Depths beyond total thickness use last layer."""
        total = sum(l.thickness_mm for l in FOREARM_LAYERS)
        depths = np.array([total + 5.0])
        result = _assign_layers(depths, FOREARM_LAYERS)
        assert result[0]["name"] == FOREARM_LAYERS[-1].name


# ── Depth profile simulation ─────────────────────────────────────


class TestSimulateProfile:
    def test_returns_depth_profile(
        self,
        magnet: SingleSidedMagnet,
        coil: SurfaceCoil,
        dp_config: DepthProfileConfig,
    ) -> None:
        result = simulate_depth_profile(magnet, coil, dp_config)
        assert isinstance(result, DepthProfile)

    def test_arrays_consistent_length(
        self,
        magnet: SingleSidedMagnet,
        coil: SurfaceCoil,
        dp_config: DepthProfileConfig,
    ) -> None:
        result = simulate_depth_profile(magnet, coil, dp_config)
        n = len(result.depths_mm)
        assert len(result.b0_tesla) == n
        assert len(result.larmor_mhz) == n
        assert len(result.signal) == n
        assert len(result.snr) == n
        assert len(result.tissue_labels) == n

    def test_signal_non_negative(
        self,
        magnet: SingleSidedMagnet,
        coil: SurfaceCoil,
        dp_config: DepthProfileConfig,
    ) -> None:
        result = simulate_depth_profile(magnet, coil, dp_config)
        assert np.all(result.signal >= 0)

    def test_scan_time_correct(
        self,
        magnet: SingleSidedMagnet,
        coil: SurfaceCoil,
    ) -> None:
        cfg = DepthProfileConfig(n_averages=100, repetition_time_ms=50.0)
        result = simulate_depth_profile(magnet, coil, cfg)
        expected = 100 * 50.0 * 1e-3  # 5.0 seconds
        assert math.isclose(result.scan_time_s, expected, rel_tol=1e-10)

    def test_more_averages_more_signal(
        self,
        magnet: SingleSidedMagnet,
        coil: SurfaceCoil,
    ) -> None:
        cfg1 = DepthProfileConfig(n_averages=1)
        cfg16 = DepthProfileConfig(n_averages=16)
        r1 = simulate_depth_profile(magnet, coil, cfg1)
        r16 = simulate_depth_profile(magnet, coil, cfg16)
        # Signal scales as √N, so ratio at every depth should be ~4
        ratio = r16.signal / np.where(r1.signal > 0, r1.signal, 1.0)
        assert np.allclose(ratio[r1.signal > 0], 4.0, rtol=1e-6)

    def test_cpmg_sequence(
        self,
        magnet: SingleSidedMagnet,
        coil: SurfaceCoil,
        dp_config: DepthProfileConfig,
    ) -> None:
        result = simulate_depth_profile(magnet, coil, dp_config, sequence="cpmg")
        assert isinstance(result, DepthProfile)
        assert np.all(result.signal >= 0)

    def test_unknown_sequence_raises(
        self,
        magnet: SingleSidedMagnet,
        coil: SurfaceCoil,
        dp_config: DepthProfileConfig,
    ) -> None:
        with pytest.raises(ValueError, match="Unknown sequence"):
            simulate_depth_profile(magnet, coil, dp_config, sequence="inversion_recovery")

    def test_hemorrhage_model(
        self,
        magnet: SingleSidedMagnet,
        coil: SurfaceCoil,
        dp_config: DepthProfileConfig,
    ) -> None:
        result = simulate_depth_profile(
            magnet, coil, dp_config, tissue_layers=HEMORRHAGE_LAYERS
        )
        assert "hemorrhage" in result.tissue_labels


# ── add_noise ─────────────────────────────────────────────────────


class TestAddNoise:
    def test_noisy_shape_matches(
        self,
        magnet: SingleSidedMagnet,
        coil: SurfaceCoil,
        dp_config: DepthProfileConfig,
    ) -> None:
        clean = simulate_depth_profile(magnet, coil, dp_config)
        noisy = add_noise(clean, rng=np.random.default_rng(42))
        assert noisy.signal.shape == clean.signal.shape

    def test_depths_unchanged(
        self,
        magnet: SingleSidedMagnet,
        coil: SurfaceCoil,
        dp_config: DepthProfileConfig,
    ) -> None:
        clean = simulate_depth_profile(magnet, coil, dp_config)
        noisy = add_noise(clean, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(noisy.depths_mm, clean.depths_mm)

    def test_noise_is_gaussian(
        self,
        magnet: SingleSidedMagnet,
        coil: SurfaceCoil,
        dp_config: DepthProfileConfig,
    ) -> None:
        """Statistical check: mean of (noisy − clean) ≈ 0 over many realisations."""
        clean = simulate_depth_profile(magnet, coil, dp_config)
        rng = np.random.default_rng(123)
        n_trials = 500
        diffs = np.zeros((n_trials, len(clean.signal)))
        for i in range(n_trials):
            noisy = add_noise(clean, rng=rng)
            diffs[i] = noisy.signal - clean.signal
        mean_diff = np.mean(diffs, axis=0)
        # Mean should be within ~3σ/√N of zero
        tol = 3 * clean.noise_v / math.sqrt(n_trials)
        assert np.all(np.abs(mean_diff) < tol)

    def test_reproducible_with_seed(
        self,
        magnet: SingleSidedMagnet,
        coil: SurfaceCoil,
        dp_config: DepthProfileConfig,
    ) -> None:
        clean = simulate_depth_profile(magnet, coil, dp_config)
        n1 = add_noise(clean, rng=np.random.default_rng(99))
        n2 = add_noise(clean, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(n1.signal, n2.signal)
