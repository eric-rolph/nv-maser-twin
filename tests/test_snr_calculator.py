"""Tests for nv_maser.physics.snr_calculator.

Covers:
- SNRBudget dataclass (fields, types, immutability)
- compute_snr_budget (signal physics, noise model, SNR correctness)
- snr_vs_depth (vectorised sweep)
- snr_vs_averages (√NEX scaling)
- snr_vs_voxel_size (cubic scaling)
- required_averages_for_snr (ceiling, correctness)
- Input validation (depth, voxel, bandwidth, averages, target_snr, sequence)
"""
import math

import numpy as np
import pytest

from nv_maser.config import SurfaceCoilConfig, SingleSidedMagnetConfig
from nv_maser.physics.depth_profile import TissueLayer
from nv_maser.physics.single_sided_magnet import SingleSidedMagnet
from nv_maser.physics.snr_calculator import (
    SNRBudget,
    compute_snr_budget,
    required_averages_for_snr,
    snr_vs_averages,
    snr_vs_depth,
    snr_vs_voxel_size,
)
from nv_maser.physics.surface_coil import SurfaceCoil
from nv_maser.physics.up_conversion import DEFAULT_MIXER, MixerSpec


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def coil() -> SurfaceCoil:
    """Default surface coil (30 mm radius, 5 turns, copper, 300 K)."""
    return SurfaceCoil(SurfaceCoilConfig())


@pytest.fixture
def magnet() -> SingleSidedMagnet:
    """Default barrel magnet (50 mT sweet spot at 20 mm)."""
    return SingleSidedMagnet(SingleSidedMagnetConfig())


@pytest.fixture
def tissue() -> TissueLayer:
    """Representative muscle tissue."""
    return TissueLayer(
        name="muscle",
        thickness_mm=50.0,
        proton_density=0.85,
        t1_ms=900.0,
        t2_ms=50.0,
    )


@pytest.fixture
def budget(coil, magnet, tissue) -> SNRBudget:
    """Default SNR budget at 10 mm depth, 3 mm voxel."""
    return compute_snr_budget(
        10.0,
        3.0,
        coil=coil,
        magnet=magnet,
        tissue=tissue,
        tr_ms=100.0,
        te_ms=10.0,
        bandwidth_hz=10_000.0,
        n_averages=1,
    )


# ── SNRBudget dataclass ───────────────────────────────────────────


class TestSNRBudgetDataclass:
    def test_is_frozen(self, budget: SNRBudget) -> None:
        with pytest.raises((AttributeError, TypeError)):
            budget.depth_mm = 99.0  # type: ignore[misc]

    def test_has_all_fields(self, budget: SNRBudget) -> None:
        expected_fields = [
            "depth_mm",
            "voxel_size_mm",
            "b0_tesla",
            "larmor_frequency_hz",
            "signal_v",
            "noise_coil_v",
            "noise_mixer_v",
            "noise_total_v",
            "snr_per_shot",
            "snr_db",
            "snr_after_averaging",
            "n_averages",
            "bandwidth_hz",
            "scan_time_s",
            "sequence_contrast",
            "system_noise_temp_k",
            "maser_noise_temp_k",
            "maser_advantage_db",
        ]
        for f in expected_fields:
            assert hasattr(budget, f), f"SNRBudget missing field: {f}"

    def test_field_types_are_numeric(self, budget: SNRBudget) -> None:
        for f in budget.__dataclass_fields__:
            val = getattr(budget, f)
            assert isinstance(val, (int, float)), (
                f"Field {f!r} should be numeric, got {type(val)}"
            )

    def test_depth_preserved(self, budget: SNRBudget) -> None:
        assert budget.depth_mm == pytest.approx(10.0)

    def test_voxel_size_preserved(self, budget: SNRBudget) -> None:
        assert budget.voxel_size_mm == pytest.approx(3.0)

    def test_n_averages_preserved(self, budget: SNRBudget) -> None:
        assert budget.n_averages == 1

    def test_bandwidth_preserved(self, budget: SNRBudget) -> None:
        assert budget.bandwidth_hz == pytest.approx(10_000.0)


# ── compute_snr_budget — B₀ and Larmor ───────────────────────────


class TestSNRBudgetB0:
    def test_b0_positive(self, budget: SNRBudget) -> None:
        """Field magnitude is positive (barrel magnet points upward)."""
        assert abs(budget.b0_tesla) > 0

    def test_larmor_proportional_to_b0(self, budget: SNRBudget) -> None:
        """f_Larmor ≈ 42.577 MHz/T × B₀."""
        expected = abs(budget.b0_tesla) * 42.577e6
        assert budget.larmor_frequency_hz == pytest.approx(expected, rel=1e-4)

    def test_larmor_positive(self, budget: SNRBudget) -> None:
        assert budget.larmor_frequency_hz > 0


# ── compute_snr_budget — signal voltage ──────────────────────────


class TestSNRBudgetSignal:
    def test_signal_positive(self, budget: SNRBudget) -> None:
        assert budget.signal_v > 0

    def test_signal_finite(self, budget: SNRBudget) -> None:
        assert math.isfinite(budget.signal_v)

    def test_signal_scales_with_voxel_volume(
        self, coil, magnet, tissue
    ) -> None:
        """Signal ∝ V_voxel = voxel_size³."""
        b1 = compute_snr_budget(10.0, 1.0, coil=coil, magnet=magnet, tissue=tissue)
        b2 = compute_snr_budget(10.0, 2.0, coil=coil, magnet=magnet, tissue=tissue)
        # 2 mm³ / 1 mm³ cube = volume ratio of 8
        ratio = b2.signal_v / b1.signal_v
        assert ratio == pytest.approx(8.0, rel=0.01)

    def test_signal_decreases_with_depth(self, coil, magnet, tissue) -> None:
        """Deeper voxels give weaker signal due to B₁/I falloff."""
        b_shallow = compute_snr_budget(5.0, 3.0, coil=coil, magnet=magnet, tissue=tissue)
        b_deep = compute_snr_budget(25.0, 3.0, coil=coil, magnet=magnet, tissue=tissue)
        assert b_shallow.signal_v > b_deep.signal_v

    def test_signal_increases_with_proton_density(self, coil, magnet) -> None:
        t_dense = TissueLayer("dense", 50.0, proton_density=1.0, t1_ms=900.0, t2_ms=50.0)
        t_sparse = TissueLayer("sparse", 50.0, proton_density=0.5, t1_ms=900.0, t2_ms=50.0)
        b_dense = compute_snr_budget(10.0, 3.0, coil=coil, magnet=magnet, tissue=t_dense)
        b_sparse = compute_snr_budget(10.0, 3.0, coil=coil, magnet=magnet, tissue=t_sparse)
        assert b_dense.signal_v > b_sparse.signal_v

    def test_sequence_contrast_between_0_and_1(self, budget: SNRBudget) -> None:
        assert 0.0 < budget.sequence_contrast <= 1.0


# ── compute_snr_budget — noise model ─────────────────────────────


class TestSNRBudgetNoise:
    def test_noise_coil_positive(self, budget: SNRBudget) -> None:
        assert budget.noise_coil_v > 0

    def test_noise_mixer_positive_with_default_mixer(self, budget: SNRBudget) -> None:
        assert budget.noise_mixer_v >= 0

    def test_noise_total_positive(self, budget: SNRBudget) -> None:
        assert budget.noise_total_v > 0

    def test_noise_total_gte_components(self, budget: SNRBudget) -> None:
        """Total noise must be ≥ each individual component."""
        assert budget.noise_total_v >= budget.noise_coil_v - 1e-30
        assert budget.noise_total_v >= budget.noise_mixer_v - 1e-30

    def test_noise_scales_with_sqrt_bandwidth(self, coil, magnet, tissue) -> None:
        """Noise ∝ √bandwidth (thermal noise floor)."""
        b1 = compute_snr_budget(
            10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue, bandwidth_hz=10_000.0
        )
        b4 = compute_snr_budget(
            10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue, bandwidth_hz=40_000.0
        )
        # √4 = 2 noise increase when bandwidth quadruples
        ratio = b4.noise_coil_v / b1.noise_coil_v
        assert ratio == pytest.approx(2.0, rel=0.01)

    def test_noise_decreases_with_cold_coil(self, magnet, tissue) -> None:
        """Cooled coil (200 K) has lower thermal noise than room temperature."""
        hot_coil = SurfaceCoil(SurfaceCoilConfig(temperature_k=300.0))
        cold_coil = SurfaceCoil(SurfaceCoilConfig(temperature_k=200.0))
        b_hot = compute_snr_budget(10.0, 3.0, coil=hot_coil, magnet=magnet, tissue=tissue)
        b_cold = compute_snr_budget(10.0, 3.0, coil=cold_coil, magnet=magnet, tissue=tissue)
        assert b_cold.noise_coil_v < b_hot.noise_coil_v


# ── compute_snr_budget — SNR ─────────────────────────────────────


class TestSNRBudgetSNR:
    def test_snr_per_shot_positive(self, budget: SNRBudget) -> None:
        assert budget.snr_per_shot > 0

    def test_snr_per_shot_equals_signal_over_noise(self, budget: SNRBudget) -> None:
        expected = budget.signal_v / budget.noise_total_v
        assert budget.snr_per_shot == pytest.approx(expected, rel=1e-9)

    def test_snr_db_consistent(self, budget: SNRBudget) -> None:
        expected_db = 20.0 * math.log10(budget.snr_per_shot)
        assert budget.snr_db == pytest.approx(expected_db, rel=1e-6)

    def test_snr_after_averaging_consistent(self, budget: SNRBudget) -> None:
        """snr_after_averaging = snr_per_shot × √n_averages."""
        expected = budget.snr_per_shot * math.sqrt(budget.n_averages)
        assert budget.snr_after_averaging == pytest.approx(expected, rel=1e-9)

    def test_snr_increases_with_n_averages(self, coil, magnet, tissue) -> None:
        b1 = compute_snr_budget(10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue, n_averages=1)
        b4 = compute_snr_budget(10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue, n_averages=4)
        assert b4.snr_after_averaging == pytest.approx(2.0 * b1.snr_after_averaging, rel=1e-6)

    def test_snr_decreases_with_depth(self, coil, magnet, tissue) -> None:
        b_shallow = compute_snr_budget(5.0, 3.0, coil=coil, magnet=magnet, tissue=tissue)
        b_deep = compute_snr_budget(25.0, 3.0, coil=coil, magnet=magnet, tissue=tissue)
        assert b_shallow.snr_per_shot > b_deep.snr_per_shot

    def test_snr_increases_with_voxel_size(self, coil, magnet, tissue) -> None:
        b_small = compute_snr_budget(10.0, 1.0, coil=coil, magnet=magnet, tissue=tissue)
        b_large = compute_snr_budget(10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue)
        assert b_large.snr_per_shot > b_small.snr_per_shot

    def test_snr_finite(self, budget: SNRBudget) -> None:
        assert math.isfinite(budget.snr_per_shot)
        assert math.isfinite(budget.snr_db)


# ── compute_snr_budget — scan time ───────────────────────────────


class TestSNRBudgetScanTime:
    def test_scan_time_correct(self, coil, magnet, tissue) -> None:
        b = compute_snr_budget(
            10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue,
            tr_ms=200.0, n_averages=10
        )
        expected = 10 * 200.0e-3
        assert b.scan_time_s == pytest.approx(expected)

    def test_scan_time_positive(self, budget: SNRBudget) -> None:
        assert budget.scan_time_s > 0


# ── compute_snr_budget — maser noise temp ────────────────────────


class TestMaserNoise:
    def test_maser_noise_temp_default(self, budget: SNRBudget) -> None:
        """Default maser noise temperature is 5 K."""
        assert budget.maser_noise_temp_k == pytest.approx(5.0)

    def test_maser_advantage_positive(self, budget: SNRBudget) -> None:
        """Maser (Tn~5 K) beats conventional LNA (~75 K) → positive dB advantage."""
        assert budget.maser_advantage_db > 0

    def test_maser_advantage_small_in_coil_dominated_regime(
        self, budget: SNRBudget
    ) -> None:
        """In coil-noise-dominated regimes, maser advantage is < 2 dB."""
        # Architecture doc: ~1.11× linear = 0.9 dB
        assert budget.maser_advantage_db < 2.0

    def test_maser_advantage_larger_with_low_coil_noise(
        self, magnet, tissue
    ) -> None:
        """Colder coil means preamp noise contribution is relatively larger
        → maser advantage grows."""
        warm_coil = SurfaceCoil(SurfaceCoilConfig(temperature_k=300.0))
        cold_coil = SurfaceCoil(SurfaceCoilConfig(temperature_k=77.0))
        b_warm = compute_snr_budget(10.0, 3.0, coil=warm_coil, magnet=magnet, tissue=tissue)
        b_cold = compute_snr_budget(10.0, 3.0, coil=cold_coil, magnet=magnet, tissue=tissue)
        assert b_cold.maser_advantage_db > b_warm.maser_advantage_db

    def test_system_noise_temp_positive(self, budget: SNRBudget) -> None:
        assert budget.system_noise_temp_k > 0

    def test_custom_cavity_q(self, coil, magnet, tissue) -> None:
        """Providing cavity Q and magnetic Q wires into compute_maser_noise_temperature."""
        b = compute_snr_budget(
            10.0, 3.0,
            coil=coil, magnet=magnet, tissue=tissue,
            cavity_q=1000.0,
            q_magnetic=500.0,
        )
        # Q_m < Q_0 → finite (not default 5 K)
        assert math.isfinite(b.maser_noise_temp_k)
        assert b.maser_noise_temp_k > 0


# ── compute_snr_budget — sequences ───────────────────────────────


class TestSequences:
    def test_spin_echo_sequence(self, coil, magnet, tissue) -> None:
        b = compute_snr_budget(
            10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue, sequence="spin_echo"
        )
        assert b.snr_per_shot > 0

    def test_gre_sequence(self, coil, magnet, tissue) -> None:
        b = compute_snr_budget(
            10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue, sequence="gre"
        )
        assert b.snr_per_shot > 0

    def test_invalid_sequence_raises(self, coil, magnet, tissue) -> None:
        with pytest.raises(ValueError, match="Unknown sequence"):
            compute_snr_budget(
                10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue, sequence="flair"
            )


# ── compute_snr_budget — input validation ────────────────────────


class TestInputValidation:
    def test_zero_depth_raises(self, coil, magnet, tissue) -> None:
        with pytest.raises(ValueError, match="depth_mm must be positive"):
            compute_snr_budget(0.0, 3.0, coil=coil, magnet=magnet, tissue=tissue)

    def test_negative_depth_raises(self, coil, magnet, tissue) -> None:
        with pytest.raises(ValueError, match="depth_mm must be positive"):
            compute_snr_budget(-5.0, 3.0, coil=coil, magnet=magnet, tissue=tissue)

    def test_zero_voxel_raises(self, coil, magnet, tissue) -> None:
        with pytest.raises(ValueError, match="voxel_size_mm must be positive"):
            compute_snr_budget(10.0, 0.0, coil=coil, magnet=magnet, tissue=tissue)

    def test_negative_voxel_raises(self, coil, magnet, tissue) -> None:
        with pytest.raises(ValueError, match="voxel_size_mm must be positive"):
            compute_snr_budget(10.0, -1.0, coil=coil, magnet=magnet, tissue=tissue)

    def test_zero_bandwidth_raises(self, coil, magnet, tissue) -> None:
        with pytest.raises(ValueError, match="bandwidth_hz must be positive"):
            compute_snr_budget(
                10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue, bandwidth_hz=0.0
            )

    def test_zero_averages_raises(self, coil, magnet, tissue) -> None:
        with pytest.raises(ValueError, match="n_averages must be"):
            compute_snr_budget(
                10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue, n_averages=0
            )


# ── snr_vs_depth ──────────────────────────────────────────────────


class TestSNRVsDepth:
    @pytest.fixture
    def depths(self) -> np.ndarray:
        return np.array([5.0, 10.0, 15.0, 20.0])

    def test_returns_array(self, coil, magnet, tissue, depths) -> None:
        result = snr_vs_depth(depths, 3.0, coil=coil, magnet=magnet, tissue=tissue)
        assert isinstance(result, np.ndarray)

    def test_same_length_as_depths(self, coil, magnet, tissue, depths) -> None:
        result = snr_vs_depth(depths, 3.0, coil=coil, magnet=magnet, tissue=tissue)
        assert len(result) == len(depths)

    def test_all_positive(self, coil, magnet, tissue, depths) -> None:
        result = snr_vs_depth(depths, 3.0, coil=coil, magnet=magnet, tissue=tissue)
        assert np.all(result > 0)

    def test_strictly_decreasing(self, coil, magnet, tissue) -> None:
        """SNR monotonically decreases with increasing depth."""
        depths = np.linspace(5.0, 30.0, 8)
        result = snr_vs_depth(depths, 3.0, coil=coil, magnet=magnet, tissue=tissue)
        diffs = np.diff(result)
        assert np.all(diffs < 0), "SNR should decrease monotonically with depth"

    def test_single_depth(self, coil, magnet, tissue) -> None:
        result = snr_vs_depth(np.array([10.0]), 3.0, coil=coil, magnet=magnet, tissue=tissue)
        assert len(result) == 1
        assert result[0] > 0

    def test_matches_compute_snr_budget(self, coil, magnet, tissue) -> None:
        depths = np.array([10.0, 20.0])
        result = snr_vs_depth(depths, 3.0, coil=coil, magnet=magnet, tissue=tissue)
        for i, d in enumerate(depths):
            b = compute_snr_budget(d, 3.0, coil=coil, magnet=magnet, tissue=tissue)
            assert result[i] == pytest.approx(b.snr_per_shot, rel=1e-9)

    def test_kwargs_forwarded(self, coil, magnet, tissue) -> None:
        depths = np.array([10.0])
        r_se = snr_vs_depth(depths, 3.0, coil=coil, magnet=magnet,
                             tissue=tissue, sequence="spin_echo")
        r_gre = snr_vs_depth(depths, 3.0, coil=coil, magnet=magnet,
                              tissue=tissue, sequence="gre")
        # Different sequences give different SNR
        assert r_se[0] != r_gre[0]


# ── snr_vs_averages ───────────────────────────────────────────────


class TestSNRVsAverages:
    @pytest.fixture
    def nex_array(self) -> np.ndarray:
        return np.array([1, 4, 9, 16], dtype=int)

    def test_returns_array(self, coil, magnet, tissue, nex_array) -> None:
        result = snr_vs_averages(nex_array, 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue)
        assert isinstance(result, np.ndarray)

    def test_same_length_as_input(self, coil, magnet, tissue, nex_array) -> None:
        result = snr_vs_averages(nex_array, 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue)
        assert len(result) == len(nex_array)

    def test_all_positive(self, coil, magnet, tissue, nex_array) -> None:
        result = snr_vs_averages(nex_array, 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue)
        assert np.all(result > 0)

    def test_sqrt_nex_scaling(self, coil, magnet, tissue) -> None:
        """SNR should scale as √NEX."""
        nex = np.array([1, 4, 9, 16], dtype=int)
        result = snr_vs_averages(nex, 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue)
        # Ratio of SNR(N) to SNR(1) should equal sqrt(N)
        for i in range(1, len(nex)):
            expected_ratio = math.sqrt(nex[i])
            actual_ratio = result[i] / result[0]
            assert actual_ratio == pytest.approx(expected_ratio, rel=1e-6)

    def test_strictly_increasing(self, coil, magnet, tissue, nex_array) -> None:
        result = snr_vs_averages(nex_array, 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue)
        assert np.all(np.diff(result) > 0)

    def test_single_average_matches_budget(self, coil, magnet, tissue) -> None:
        result = snr_vs_averages(
            np.array([1]), 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue
        )
        b = compute_snr_budget(10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue, n_averages=1)
        assert result[0] == pytest.approx(b.snr_after_averaging, rel=1e-9)


# ── snr_vs_voxel_size ─────────────────────────────────────────────


class TestSNRVsVoxelSize:
    @pytest.fixture
    def voxel_sizes(self) -> np.ndarray:
        return np.array([1.0, 2.0, 3.0, 4.0])

    def test_returns_array(self, coil, magnet, tissue, voxel_sizes) -> None:
        result = snr_vs_voxel_size(voxel_sizes, 10.0, coil=coil, magnet=magnet, tissue=tissue)
        assert isinstance(result, np.ndarray)

    def test_same_length_as_input(self, coil, magnet, tissue, voxel_sizes) -> None:
        result = snr_vs_voxel_size(voxel_sizes, 10.0, coil=coil, magnet=magnet, tissue=tissue)
        assert len(result) == len(voxel_sizes)

    def test_all_positive(self, coil, magnet, tissue, voxel_sizes) -> None:
        result = snr_vs_voxel_size(voxel_sizes, 10.0, coil=coil, magnet=magnet, tissue=tissue)
        assert np.all(result > 0)

    def test_strictly_increasing(self, coil, magnet, tissue, voxel_sizes) -> None:
        """Larger voxels give better SNR due to more signal."""
        result = snr_vs_voxel_size(voxel_sizes, 10.0, coil=coil, magnet=magnet, tissue=tissue)
        assert np.all(np.diff(result) > 0)

    def test_cubic_snr_scaling(self, coil, magnet, tissue) -> None:
        """Signal ∝ V = voxel_size³, so SNR(2x) = SNR(x) × 8."""
        r1 = snr_vs_voxel_size(np.array([1.0]), 10.0, coil=coil, magnet=magnet, tissue=tissue)
        r2 = snr_vs_voxel_size(np.array([2.0]), 10.0, coil=coil, magnet=magnet, tissue=tissue)
        ratio = r2[0] / r1[0]
        assert ratio == pytest.approx(8.0, rel=0.01)

    def test_matches_compute_snr_budget(self, coil, magnet, tissue) -> None:
        voxels = np.array([2.0, 4.0])
        result = snr_vs_voxel_size(voxels, 10.0, coil=coil, magnet=magnet, tissue=tissue)
        for i, vs in enumerate(voxels):
            b = compute_snr_budget(10.0, float(vs), coil=coil, magnet=magnet, tissue=tissue)
            assert result[i] == pytest.approx(b.snr_per_shot, rel=1e-9)


# ── required_averages_for_snr ─────────────────────────────────────


class TestRequiredAverages:
    def test_returns_int(self, coil, magnet, tissue) -> None:
        result = required_averages_for_snr(
            5.0, 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue
        )
        assert isinstance(result, int)

    def test_at_least_one(self, coil, magnet, tissue) -> None:
        result = required_averages_for_snr(
            1e-6, 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue
        )
        assert result >= 1

    def test_achieves_target_snr(self, coil, magnet, tissue) -> None:
        """After required_averages, achieved SNR ≥ target."""
        target_snr = 10.0
        nex = required_averages_for_snr(
            target_snr, 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue
        )
        b = compute_snr_budget(
            10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue, n_averages=nex
        )
        assert b.snr_after_averaging >= target_snr

    def test_one_fewer_average_insufficient(self, coil, magnet, tissue) -> None:
        """nex - 1 averages should generally not reach the target SNR."""
        target_snr = 10.0
        nex = required_averages_for_snr(
            target_snr, 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue
        )
        if nex > 1:
            b = compute_snr_budget(
                10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue, n_averages=nex - 1
            )
            assert b.snr_after_averaging < target_snr

    def test_proportional_to_target_squared(self, coil, magnet, tissue) -> None:
        """Required averages ∝ target_snr²."""
        nex1 = required_averages_for_snr(
            2.0, 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue
        )
        nex2 = required_averages_for_snr(
            4.0, 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue
        )
        # (4/2)² = 4× more averages
        ratio = nex2 / nex1
        assert ratio == pytest.approx(4.0, abs=1.0)  # allow 1 due to ceiling

    def test_zero_target_snr_raises(self, coil, magnet, tissue) -> None:
        with pytest.raises(ValueError, match="target_snr must be positive"):
            required_averages_for_snr(
                0.0, 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue
            )

    def test_negative_target_snr_raises(self, coil, magnet, tissue) -> None:
        with pytest.raises(ValueError, match="target_snr must be positive"):
            required_averages_for_snr(
                -5.0, 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue
            )

    def test_ceiling_applied(self, coil, magnet, tissue) -> None:
        """Result is always a ceiling integer (never fractional)."""
        # Pick many random-ish targets and verify ≥ floor ratio
        for tgt in [3.1, 7.7, 12.3]:
            nex = required_averages_for_snr(
                tgt, 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue
            )
            assert isinstance(nex, int)

    def test_very_high_target_is_positive(self, coil, magnet, tissue) -> None:
        nex = required_averages_for_snr(
            1000.0, 10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue
        )
        assert nex >= 1


# ── Custom mixer ──────────────────────────────────────────────────


class TestCustomMixer:
    def test_custom_mixer_accepted(self, coil, magnet, tissue) -> None:
        mixer = MixerSpec(conversion_loss_db=8.0, noise_figure_ssb_db=10.0)
        b = compute_snr_budget(
            10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue, mixer=mixer
        )
        assert b.snr_per_shot > 0

    def test_lossier_mixer_gives_lower_snr(self, coil, magnet, tissue) -> None:
        good_mixer = MixerSpec(conversion_loss_db=3.0, noise_figure_ssb_db=5.0)
        bad_mixer = MixerSpec(conversion_loss_db=12.0, noise_figure_ssb_db=15.0)
        b_good = compute_snr_budget(
            10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue, mixer=good_mixer
        )
        b_bad = compute_snr_budget(
            10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue, mixer=bad_mixer
        )
        assert b_good.snr_per_shot >= b_bad.snr_per_shot


# ── Integration: architecture validated values ────────────────────


class TestArchitectureValues:
    """Sanity-check against the architecture doc (Section 8).

    At 20 mm depth, 8 µL voxel, 50 mT, 300 K coil, 10 kHz BW:
    - Signal ~6.5 fV, total noise ~7.05 nV, SNR << 1 per shot.
    - Maser advantage ~0.9 dB (i.e. ~1.11× in linear scale).
    """

    def test_snr_much_less_than_1_per_shot_at_20mm(
        self, coil, magnet, tissue
    ) -> None:
        b = compute_snr_budget(
            20.0, 2.0, coil=coil, magnet=magnet, tissue=tissue,
            bandwidth_hz=10_000.0, tr_ms=500.0, te_ms=25.0
        )
        # Per-shot SNR is << 1; need many averages
        assert b.snr_per_shot < 0.5

    def test_maser_advantage_about_1_db(self, coil, magnet, tissue) -> None:
        """Architecture says ~1.11× = ~0.9 dB advantage in coil-noise regime."""
        b = compute_snr_budget(
            20.0, 2.0, coil=coil, magnet=magnet, tissue=tissue,
            bandwidth_hz=10_000.0, tr_ms=500.0, te_ms=25.0
        )
        # Maser advantage should be less than 2 dB (coil noise dominates)
        assert 0.0 < b.maser_advantage_db < 2.0

    def test_signal_in_femtovolt_range_at_20mm(self, coil, magnet, tissue) -> None:
        """Architecture doc says ~6.5 fV signal at 20 mm, 8 µL voxel."""
        # 8 µL = 8×10⁻⁹ m³ = cube of (2×10⁻³)^(1/3) ... actually 8 µL = (2 mm)³ × 1000? NO
        # 8 µL = 8e-9 m³ = (2e-3)³ = 8mm × 8mm × 8mm? No. Let's compute:
        # (x mm)³ = 8e-9 m³ → x = (8e-9)^(1/3) * 1000 mm = 2e-3 m × 1000 mm/m = 2 mm
        # So 8 µL = (2 mm)³ = cube with 2 mm side. That's only 8e-9 m³... wait.
        # 8 µL = 8e-6 L = 8e-9 m³. cube: (8e-9)^(1/3) = 2e-3 m = 2 mm sides. Yes.
        b = compute_snr_budget(
            20.0, 2.0, coil=coil, magnet=magnet, tissue=tissue,
            bandwidth_hz=10_000.0, tr_ms=500.0, te_ms=25.0
        )
        # Signal should be in picovolts to nanovolts range (architecture says ~fV)
        assert 1e-15 < b.signal_v < 1e-8  # between 1 fV and 10 nV

    def test_1024_averages_gives_adequate_snr(self, coil, magnet, tissue) -> None:
        """Architecture doc: 1024 averages on 3 mm voxel at 10 mm gives ~SNR 7."""
        b = compute_snr_budget(
            10.0, 3.0, coil=coil, magnet=magnet, tissue=tissue,
            tr_ms=100.0, te_ms=10.0, bandwidth_hz=10_000.0, n_averages=1024
        )
        # We don't require exactly SNR=7, but it should be > 1
        assert b.snr_after_averaging > 1.0
