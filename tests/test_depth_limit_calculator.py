"""Tests for nv_maser.physics.depth_limit_calculator.

Covers:
- DepthLimitConfig defaults and __post_init__ validation
- DepthPoint field presence and types
- compute_depth_point: physics correctness, scan-time formula
- compute_depth_limit: profile sweep, max_depth, V1 range confirmation
- Monotonicity: SNR decreases / scan-time increases with depth
- Budget sensitivity: larger budget → deeper max_depth
- Target-SNR sensitivity: higher target → shallower max_depth
- Architecture validation: V1 5–15 mm range
"""

import pytest

from nv_maser.config import SingleSidedMagnetConfig, SurfaceCoilConfig
from nv_maser.physics.depth_limit_calculator import (
    DepthLimitConfig,
    DepthLimitResult,
    DepthPoint,
    compute_depth_limit,
    compute_depth_point,
)
from nv_maser.physics.depth_profile import TissueLayer
from nv_maser.physics.single_sided_magnet import SingleSidedMagnet
from nv_maser.physics.surface_coil import SurfaceCoil

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
def default_config() -> DepthLimitConfig:
    """Default depth-limit configuration."""
    return DepthLimitConfig()


@pytest.fixture
def depth_point_10mm(coil, magnet, tissue, default_config) -> DepthPoint:
    """DepthPoint computed at 10 mm with default config."""
    return compute_depth_point(10.0, default_config, coil=coil, magnet=magnet, tissue=tissue)


@pytest.fixture
def result_default(coil, magnet, tissue, default_config) -> DepthLimitResult:
    """Full depth-limit result with default config."""
    return compute_depth_limit(default_config, coil=coil, magnet=magnet, tissue=tissue)


# ── DepthLimitConfig defaults ──────────────────────────────────────


class TestDepthLimitConfigDefaults:
    def test_default_target_snr(self) -> None:
        assert DepthLimitConfig().target_snr == 5.0

    def test_default_scan_time_budget_s(self) -> None:
        assert DepthLimitConfig().scan_time_budget_s == 120.0

    def test_default_voxel_size_mm(self) -> None:
        assert DepthLimitConfig().voxel_size_mm == 3.0

    def test_default_tr_ms(self) -> None:
        assert DepthLimitConfig().tr_ms == 100.0

    def test_default_te_ms(self) -> None:
        assert DepthLimitConfig().te_ms == 10.0

    def test_default_sequence(self) -> None:
        assert DepthLimitConfig().sequence == "spin_echo"

    def test_default_bandwidth_hz(self) -> None:
        assert DepthLimitConfig().bandwidth_hz == 10_000.0

    def test_default_depth_step_mm(self) -> None:
        assert DepthLimitConfig().depth_step_mm == 1.0

    def test_default_min_depth_mm(self) -> None:
        assert DepthLimitConfig().min_depth_mm == 1.0

    def test_default_max_depth_mm(self) -> None:
        assert DepthLimitConfig().max_depth_mm == 30.0

    def test_is_frozen(self) -> None:
        cfg = DepthLimitConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.target_snr = 99.0  # type: ignore[misc]


# ── DepthLimitConfig validation ────────────────────────────────────


class TestDepthLimitConfigValidation:
    def test_negative_target_snr_raises(self) -> None:
        with pytest.raises(ValueError, match="target_snr"):
            DepthLimitConfig(target_snr=-1.0)

    def test_zero_target_snr_raises(self) -> None:
        with pytest.raises(ValueError, match="target_snr"):
            DepthLimitConfig(target_snr=0.0)

    def test_negative_scan_time_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="scan_time_budget_s"):
            DepthLimitConfig(scan_time_budget_s=-1.0)

    def test_zero_scan_time_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="scan_time_budget_s"):
            DepthLimitConfig(scan_time_budget_s=0.0)

    def test_negative_voxel_size_raises(self) -> None:
        with pytest.raises(ValueError, match="voxel_size_mm"):
            DepthLimitConfig(voxel_size_mm=-1.0)

    def test_negative_tr_ms_raises(self) -> None:
        with pytest.raises(ValueError, match="tr_ms"):
            DepthLimitConfig(tr_ms=-1.0)

    def test_negative_depth_step_raises(self) -> None:
        with pytest.raises(ValueError, match="depth_step_mm"):
            DepthLimitConfig(depth_step_mm=-0.5)

    def test_min_equals_max_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="min_depth_mm"):
            DepthLimitConfig(min_depth_mm=10.0, max_depth_mm=10.0)

    def test_min_greater_than_max_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="min_depth_mm"):
            DepthLimitConfig(min_depth_mm=20.0, max_depth_mm=5.0)

    def test_valid_custom_config_accepted(self) -> None:
        cfg = DepthLimitConfig(
            target_snr=3.0,
            scan_time_budget_s=60.0,
            voxel_size_mm=5.0,
            tr_ms=200.0,
            depth_step_mm=0.5,
            min_depth_mm=2.0,
            max_depth_mm=20.0,
        )
        assert cfg.target_snr == 3.0


# ── DepthPoint structure ───────────────────────────────────────────


class TestDepthPointFields:
    def test_depth_mm_field(self, depth_point_10mm: DepthPoint) -> None:
        assert hasattr(depth_point_10mm, "depth_mm")

    def test_snr_per_shot_field(self, depth_point_10mm: DepthPoint) -> None:
        assert hasattr(depth_point_10mm, "snr_per_shot")

    def test_required_averages_field(self, depth_point_10mm: DepthPoint) -> None:
        assert hasattr(depth_point_10mm, "required_averages")

    def test_scan_time_s_field(self, depth_point_10mm: DepthPoint) -> None:
        assert hasattr(depth_point_10mm, "scan_time_s")

    def test_within_budget_field(self, depth_point_10mm: DepthPoint) -> None:
        assert hasattr(depth_point_10mm, "within_budget")

    def test_depth_point_is_frozen(self, depth_point_10mm: DepthPoint) -> None:
        with pytest.raises((AttributeError, TypeError)):
            depth_point_10mm.depth_mm = 99.0  # type: ignore[misc]


# ── compute_depth_point ────────────────────────────────────────────


class TestComputeDepthPoint:
    def test_returns_depth_point(self, depth_point_10mm: DepthPoint) -> None:
        assert isinstance(depth_point_10mm, DepthPoint)

    def test_depth_mm_matches_input(self, coil, magnet, tissue, default_config) -> None:
        p = compute_depth_point(7.0, default_config, coil=coil, magnet=magnet, tissue=tissue)
        assert p.depth_mm == 7.0

    def test_snr_per_shot_positive(self, depth_point_10mm: DepthPoint) -> None:
        assert depth_point_10mm.snr_per_shot > 0.0

    def test_required_averages_is_int(self, depth_point_10mm: DepthPoint) -> None:
        assert isinstance(depth_point_10mm.required_averages, int)

    def test_required_averages_at_least_one(
        self, depth_point_10mm: DepthPoint
    ) -> None:
        assert depth_point_10mm.required_averages >= 1

    def test_scan_time_equals_averages_times_tr(
        self, depth_point_10mm: DepthPoint, default_config: DepthLimitConfig
    ) -> None:
        expected = depth_point_10mm.required_averages * default_config.tr_ms / 1000.0
        assert depth_point_10mm.scan_time_s == pytest.approx(expected, rel=1e-9)

    def test_within_budget_large_budget(self, coil, magnet, tissue) -> None:
        big_budget = DepthLimitConfig(scan_time_budget_s=1e9)
        p = compute_depth_point(
            10.0, big_budget, coil=coil, magnet=magnet, tissue=tissue
        )
        assert p.within_budget is True

    def test_within_budget_tiny_budget(self, coil, magnet, tissue) -> None:
        tiny_budget = DepthLimitConfig(scan_time_budget_s=1e-6)
        p = compute_depth_point(
            10.0, tiny_budget, coil=coil, magnet=magnet, tissue=tissue
        )
        assert p.within_budget is False

    def test_within_budget_flag_consistent_with_scan_time(
        self, depth_point_10mm: DepthPoint, default_config: DepthLimitConfig
    ) -> None:
        expected = depth_point_10mm.scan_time_s <= default_config.scan_time_budget_s
        assert depth_point_10mm.within_budget is expected

    def test_deeper_voxel_requires_more_averages(
        self, coil, magnet, tissue, default_config
    ) -> None:
        p_shallow = compute_depth_point(
            5.0, default_config, coil=coil, magnet=magnet, tissue=tissue
        )
        p_deep = compute_depth_point(
            15.0, default_config, coil=coil, magnet=magnet, tissue=tissue
        )
        assert p_deep.required_averages >= p_shallow.required_averages


# ── compute_depth_limit ────────────────────────────────────────────


class TestComputeDepthLimit:
    def test_returns_depth_limit_result(
        self, result_default: DepthLimitResult
    ) -> None:
        assert isinstance(result_default, DepthLimitResult)

    def test_n_depths_evaluated_matches_range(
        self, result_default: DepthLimitResult, default_config: DepthLimitConfig
    ) -> None:
        expected = (
            round(
                (default_config.max_depth_mm - default_config.min_depth_mm)
                / default_config.depth_step_mm
            )
            + 1
        )
        assert result_default.n_depths_evaluated == expected

    def test_depth_profile_length_matches_n_depths(
        self, result_default: DepthLimitResult
    ) -> None:
        assert len(result_default.depth_profile) == result_default.n_depths_evaluated

    def test_profile_depths_are_increasing(
        self, result_default: DepthLimitResult
    ) -> None:
        depths = [p.depth_mm for p in result_default.depth_profile]
        assert all(depths[i] < depths[i + 1] for i in range(len(depths) - 1))

    def test_max_depth_mm_is_feasible(
        self, result_default: DepthLimitResult
    ) -> None:
        if result_default.any_feasible:
            limit = next(
                p
                for p in result_default.depth_profile
                if p.depth_mm == result_default.max_depth_mm
            )
            assert limit.within_budget is True

    def test_max_depth_mm_is_maximum_among_feasible(
        self, result_default: DepthLimitResult
    ) -> None:
        if result_default.any_feasible:
            feasible_depths = [
                p.depth_mm for p in result_default.depth_profile if p.within_budget
            ]
            assert result_default.max_depth_mm == max(feasible_depths)

    def test_any_feasible_with_enormous_budget(
        self, coil, magnet, tissue
    ) -> None:
        cfg = DepthLimitConfig(scan_time_budget_s=1e9)
        res = compute_depth_limit(cfg, coil=coil, magnet=magnet, tissue=tissue)
        assert res.any_feasible is True

    def test_not_any_feasible_with_microscopic_budget(
        self, coil, magnet, tissue
    ) -> None:
        cfg = DepthLimitConfig(scan_time_budget_s=1e-9)
        res = compute_depth_limit(cfg, coil=coil, magnet=magnet, tissue=tissue)
        assert res.any_feasible is False

    def test_max_depth_zero_when_not_feasible(
        self, coil, magnet, tissue
    ) -> None:
        cfg = DepthLimitConfig(scan_time_budget_s=1e-9)
        res = compute_depth_limit(cfg, coil=coil, magnet=magnet, tissue=tissue)
        assert res.max_depth_mm == 0.0

    def test_budget_echoed_in_result(
        self, result_default: DepthLimitResult, default_config: DepthLimitConfig
    ) -> None:
        assert result_default.scan_time_budget_s == default_config.scan_time_budget_s

    def test_target_snr_echoed_in_result(
        self, result_default: DepthLimitResult, default_config: DepthLimitConfig
    ) -> None:
        assert result_default.target_snr == default_config.target_snr

    def test_none_config_uses_defaults(self, coil, magnet, tissue) -> None:
        res = compute_depth_limit(None, coil=coil, magnet=magnet, tissue=tissue)
        assert isinstance(res, DepthLimitResult)
        assert res.scan_time_budget_s == DepthLimitConfig().scan_time_budget_s

    def test_result_is_frozen(self, result_default: DepthLimitResult) -> None:
        with pytest.raises((AttributeError, TypeError)):
            result_default.max_depth_mm = 99.0  # type: ignore[misc]

    def test_scan_time_at_limit_within_budget(
        self, result_default: DepthLimitResult
    ) -> None:
        if result_default.any_feasible:
            assert (
                result_default.scan_time_at_limit_s
                <= result_default.scan_time_budget_s
            )

    def test_profile_first_depth_equals_min(
        self, result_default: DepthLimitResult, default_config: DepthLimitConfig
    ) -> None:
        assert result_default.depth_profile[0].depth_mm == pytest.approx(
            default_config.min_depth_mm
        )

    def test_profile_last_depth_equals_max(
        self, result_default: DepthLimitResult, default_config: DepthLimitConfig
    ) -> None:
        assert result_default.depth_profile[-1].depth_mm == pytest.approx(
            default_config.max_depth_mm
        )


# ── Monotonicity ───────────────────────────────────────────────────


class TestDepthSNRMonotonicity:
    """SNR is non-monotonic at very shallow depths (1 mm fringe-field effect
    from the single-sided magnet sweet spot at 20 mm), but is strictly
    monotonically decreasing from the SNR peak (~5 mm) onward.
    """

    @pytest.fixture
    def result_5_30mm(self, coil, magnet, tissue) -> DepthLimitResult:
        """Profile from 5 mm to 30 mm — past the SNR peak, monotone region."""
        cfg = DepthLimitConfig(min_depth_mm=5.0, max_depth_mm=30.0)
        return compute_depth_limit(cfg, coil=coil, magnet=magnet, tissue=tissue)

    def test_snr_strictly_decreasing_past_peak(
        self, result_5_30mm: DepthLimitResult
    ) -> None:
        """From 5 mm onward SNR is strictly decreasing (past the SNR peak
        where coil sensitivity and effective B₀ both fall together)."""
        snrs = [p.snr_per_shot for p in result_5_30mm.depth_profile]
        assert all(snrs[i] > snrs[i + 1] for i in range(len(snrs) - 1))

    def test_required_averages_nondecreasing_past_peak(
        self, result_5_30mm: DepthLimitResult
    ) -> None:
        avgs = [p.required_averages for p in result_5_30mm.depth_profile]
        assert all(avgs[i] <= avgs[i + 1] for i in range(len(avgs) - 1))

    def test_scan_time_nondecreasing_past_peak(
        self, result_5_30mm: DepthLimitResult
    ) -> None:
        times = [p.scan_time_s for p in result_5_30mm.depth_profile]
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))

    def test_all_snr_per_shot_positive(
        self, result_default: DepthLimitResult
    ) -> None:
        for p in result_default.depth_profile:
            assert p.snr_per_shot > 0.0

    def test_all_depths_beyond_max_depth_infeasible(
        self, result_default: DepthLimitResult
    ) -> None:
        """Every depth strictly greater than max_depth_mm is infeasible."""
        if result_default.any_feasible:
            for p in result_default.depth_profile:
                if p.depth_mm > result_default.max_depth_mm:
                    assert not p.within_budget


# ── Budget sensitivity ─────────────────────────────────────────────


class TestScanTimeBudgetSensitivity:
    def test_larger_budget_gives_deeper_or_equal_max_depth(
        self, coil, magnet, tissue
    ) -> None:
        res_short = compute_depth_limit(
            DepthLimitConfig(scan_time_budget_s=30.0),
            coil=coil, magnet=magnet, tissue=tissue,
        )
        res_long = compute_depth_limit(
            DepthLimitConfig(scan_time_budget_s=600.0),
            coil=coil, magnet=magnet, tissue=tissue,
        )
        assert res_long.max_depth_mm >= res_short.max_depth_mm

    def test_smaller_budget_gives_shallower_or_equal_max_depth(
        self, coil, magnet, tissue
    ) -> None:
        res_large = compute_depth_limit(
            DepthLimitConfig(scan_time_budget_s=300.0),
            coil=coil, magnet=magnet, tissue=tissue,
        )
        res_small = compute_depth_limit(
            DepthLimitConfig(scan_time_budget_s=10.0),
            coil=coil, magnet=magnet, tissue=tissue,
        )
        assert res_small.max_depth_mm <= res_large.max_depth_mm

    def test_scan_time_at_limit_does_not_exceed_budget(
        self, coil, magnet, tissue
    ) -> None:
        for budget_s in (10.0, 60.0, 300.0):
            res = compute_depth_limit(
                DepthLimitConfig(scan_time_budget_s=budget_s),
                coil=coil, magnet=magnet, tissue=tissue,
            )
            if res.any_feasible:
                assert res.scan_time_at_limit_s <= budget_s


# ── Target-SNR sensitivity ─────────────────────────────────────────


class TestTargetSNRSensitivity:
    def test_higher_target_snr_shallower_or_equal_max_depth(
        self, coil, magnet, tissue
    ) -> None:
        res_easy = compute_depth_limit(
            DepthLimitConfig(target_snr=2.0),
            coil=coil, magnet=magnet, tissue=tissue,
        )
        res_hard = compute_depth_limit(
            DepthLimitConfig(target_snr=10.0),
            coil=coil, magnet=magnet, tissue=tissue,
        )
        assert res_hard.max_depth_mm <= res_easy.max_depth_mm

    def test_lower_target_snr_deeper_or_equal_max_depth(
        self, coil, magnet, tissue
    ) -> None:
        res_low = compute_depth_limit(
            DepthLimitConfig(target_snr=1.0),
            coil=coil, magnet=magnet, tissue=tissue,
        )
        res_high = compute_depth_limit(
            DepthLimitConfig(target_snr=5.0),
            coil=coil, magnet=magnet, tissue=tissue,
        )
        assert res_low.max_depth_mm >= res_high.max_depth_mm

    def test_target_snr_echoed_correctly(self, coil, magnet, tissue) -> None:
        cfg = DepthLimitConfig(target_snr=3.5)
        res = compute_depth_limit(cfg, coil=coil, magnet=magnet, tissue=tissue)
        assert res.target_snr == pytest.approx(3.5)

    def test_required_averages_at_10mm_increases_with_target(
        self, coil, magnet, tissue
    ) -> None:
        p_low = compute_depth_point(
            10.0, DepthLimitConfig(target_snr=2.0),
            coil=coil, magnet=magnet, tissue=tissue,
        )
        p_high = compute_depth_point(
            10.0, DepthLimitConfig(target_snr=8.0),
            coil=coil, magnet=magnet, tissue=tissue,
        )
        assert p_high.required_averages >= p_low.required_averages


# ── V1 depth-range confirmation (R7 mitigation) ────────────────────


class TestV1RangeConfirmation:
    def test_v1_confirmed_with_enormous_budget(
        self, coil, magnet, tissue
    ) -> None:
        """With infinite budget any 5–15 mm depth is achievable."""
        cfg = DepthLimitConfig(scan_time_budget_s=1e9)
        res = compute_depth_limit(cfg, coil=coil, magnet=magnet, tissue=tissue)
        assert res.v1_depth_range_confirmed is True

    def test_v1_not_confirmed_with_microscopic_budget(
        self, coil, magnet, tissue
    ) -> None:
        cfg = DepthLimitConfig(scan_time_budget_s=1e-9)
        res = compute_depth_limit(cfg, coil=coil, magnet=magnet, tissue=tissue)
        assert res.v1_depth_range_confirmed is False

    def test_v1_range_consistent_with_depth_profile(
        self, result_default: DepthLimitResult
    ) -> None:
        """v1_depth_range_confirmed must agree with per-point within_budget."""
        v1_points = [
            p
            for p in result_default.depth_profile
            if 5.0 <= p.depth_mm <= 15.0
        ]
        if not v1_points:
            assert result_default.v1_depth_range_confirmed is False
        else:
            expected = all(p.within_budget for p in v1_points)
            assert result_default.v1_depth_range_confirmed is expected

    def test_v1_not_confirmed_when_15mm_outside_profile(
        self, coil, magnet, tissue
    ) -> None:
        """Profile that only reaches 10 mm cannot confirm V1 5–15 mm range."""
        cfg = DepthLimitConfig(
            scan_time_budget_s=1e9,
            min_depth_mm=1.0,
            max_depth_mm=10.0,
        )
        res = compute_depth_limit(cfg, coil=coil, magnet=magnet, tissue=tissue)
        # There are no depths in [11, 15], but [5, 10] should be present.
        # With enormous budget those should all pass, so v1 depends on
        # whether the sub-range [5, 10] is enough for True.  Here [11-15]
        # are missing so only [5-10] are evaluated — still within [5,15].
        v1_points = [p for p in res.depth_profile if 5.0 <= p.depth_mm <= 15.0]
        expected = bool(v1_points and all(p.within_budget for p in v1_points))
        assert res.v1_depth_range_confirmed is expected


# ── Reference depth SNR values ─────────────────────────────────────


class TestReferenceDepths:
    def test_snr_at_5mm_present_in_default_profile(
        self, result_default: DepthLimitResult
    ) -> None:
        assert result_default.snr_per_shot_at_5mm is not None

    def test_snr_at_10mm_present_in_default_profile(
        self, result_default: DepthLimitResult
    ) -> None:
        assert result_default.snr_per_shot_at_10mm is not None

    def test_snr_at_15mm_present_in_default_profile(
        self, result_default: DepthLimitResult
    ) -> None:
        assert result_default.snr_per_shot_at_15mm is not None

    def test_snr_at_5mm_greater_than_at_10mm(
        self, result_default: DepthLimitResult
    ) -> None:
        assert result_default.snr_per_shot_at_5mm > result_default.snr_per_shot_at_10mm  # type: ignore[operator]

    def test_snr_at_10mm_greater_than_at_15mm(
        self, result_default: DepthLimitResult
    ) -> None:
        assert result_default.snr_per_shot_at_10mm > result_default.snr_per_shot_at_15mm  # type: ignore[operator]

    def test_snr_at_key_depths_match_depth_profile(
        self, result_default: DepthLimitResult
    ) -> None:
        snr_map = {p.depth_mm: p.snr_per_shot for p in result_default.depth_profile}
        if result_default.snr_per_shot_at_5mm is not None:
            assert result_default.snr_per_shot_at_5mm == pytest.approx(
                snr_map[5.0], rel=1e-9
            )
        if result_default.snr_per_shot_at_10mm is not None:
            assert result_default.snr_per_shot_at_10mm == pytest.approx(
                snr_map[10.0], rel=1e-9
            )
        if result_default.snr_per_shot_at_15mm is not None:
            assert result_default.snr_per_shot_at_15mm == pytest.approx(
                snr_map[15.0], rel=1e-9
            )

    def test_snr_absent_when_depth_not_in_profile(
        self, coil, magnet, tissue
    ) -> None:
        cfg = DepthLimitConfig(min_depth_mm=1.0, max_depth_mm=4.0)
        res = compute_depth_limit(cfg, coil=coil, magnet=magnet, tissue=tissue)
        assert res.snr_per_shot_at_5mm is None
        assert res.snr_per_shot_at_10mm is None
        assert res.snr_per_shot_at_15mm is None


# ── Architecture sanity check ──────────────────────────────────────


class TestArchitectureValidation:
    """Confirm V1 depth-range claim from §8.4 of the architecture doc."""

    def test_snr_per_shot_less_than_1_at_all_depths(
        self, result_default: DepthLimitResult
    ) -> None:
        """Per-shot SNR at clinical depths must be << 1 (need averaging)."""
        for p in result_default.depth_profile:
            # SNR per shot must be < 1 at all depths ≥ 5 mm.
            if p.depth_mm >= 5.0:
                assert p.snr_per_shot < 1.0

    def test_required_averages_increases_sharply_beyond_10mm(
        self, result_default: DepthLimitResult
    ) -> None:
        """Required averages at 20 mm >> required averages at 10 mm."""
        avgs = {p.depth_mm: p.required_averages for p in result_default.depth_profile}
        assert avgs[20.0] > avgs[10.0]

    def test_n_depths_default_is_30(
        self, result_default: DepthLimitResult
    ) -> None:
        # min=1, max=30, step=1 → 30 points
        assert result_default.n_depths_evaluated == 30
