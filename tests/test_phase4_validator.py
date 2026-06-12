"""Tests for the Phase-4 depth-profile milestone validator.

Coverage
────────
TestPhase4Config               (13) — defaults, validation errors, frozen
TestLayerContrastResultFields   (7) — fields, types, Python native types
TestPhase4MilestoneResultFields (8) — fields, frozen, type assertions
TestComputeLayerContrast       (14) — T2 ratio formulas, detectability,
                                      in-range detection, edge cases
TestValidatePhase4MilestoneDefault (12) — default config passes milestone,
                                          measured values within spec
TestPhase4SNRCriteria           (5) — tight threshold fails, loose passes
TestPhase4ContrastCriteria      (5) — identical T2 → ratio=1, extreme ratios
TestPhase4ScanTimeCriteria      (3) — budget pass/fail logic
TestPhase4MilestoneClosedLogic  (4) — all three sub-criteria required
"""
from __future__ import annotations

import math

import pytest

from nv_maser.config import (
    DepthProfileConfig,
    SingleSidedMagnetConfig,
    SurfaceCoilConfig,
)
from nv_maser.physics.depth_profile import (
    FOREARM_LAYERS,
    DepthProfile,
    TissueLayer,
    simulate_depth_profile,
)
from nv_maser.physics.phase4_validator import (
    LayerContrastResult,
    Phase4Config,
    Phase4MilestoneResult,
    compute_layer_contrast,
    validate_phase4_milestone,
)
from nv_maser.physics.single_sided_magnet import SingleSidedMagnet
from nv_maser.physics.surface_coil import SurfaceCoil

# ── Shared fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def default_result() -> Phase4MilestoneResult:
    """Run the milestone validator once for the whole module (slow)."""
    return validate_phase4_milestone()


@pytest.fixture
def simple_layers() -> list[TissueLayer]:
    """Two-layer model: fat (T2=80 ms) over muscle (T2=35 ms)."""
    return [
        TissueLayer("fat", thickness_mm=5.0, proton_density=0.9, t1_ms=250, t2_ms=80),
        TissueLayer("muscle", thickness_mm=20.0, proton_density=1.0, t1_ms=600, t2_ms=35),
    ]


@pytest.fixture(scope="module")
def default_profile() -> DepthProfile:
    magnet = SingleSidedMagnet(SingleSidedMagnetConfig())
    coil = SurfaceCoil(SurfaceCoilConfig())
    cfg = DepthProfileConfig()
    return simulate_depth_profile(magnet, coil, cfg, list(FOREARM_LAYERS))


# ── TestPhase4Config ──────────────────────────────────────────────


class TestPhase4Config:
    def test_default_signal_threshold(self) -> None:
        cfg = Phase4Config()
        assert cfg.signal_snr_threshold == 3.0

    def test_default_contrast_ratio_threshold(self) -> None:
        cfg = Phase4Config()
        assert cfg.contrast_ratio_threshold == 1.5

    def test_default_snr_at_boundary(self) -> None:
        cfg = Phase4Config()
        assert cfg.snr_at_boundary_threshold == 1.0

    def test_default_scan_time_limit(self) -> None:
        cfg = Phase4Config()
        assert cfg.scan_time_limit_s == 120.0

    def test_default_depth_range(self) -> None:
        cfg = Phase4Config()
        assert cfg.depth_range_mm == (3.0, 15.0)

    def test_frozen(self) -> None:
        cfg = Phase4Config()
        with pytest.raises((AttributeError, TypeError)):
            cfg.signal_snr_threshold = 10.0  # type: ignore[misc]

    def test_negative_snr_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="signal_snr_threshold"):
            Phase4Config(signal_snr_threshold=-1.0)

    def test_zero_snr_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="signal_snr_threshold"):
            Phase4Config(signal_snr_threshold=0.0)

    def test_contrast_ratio_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="contrast_ratio_threshold"):
            Phase4Config(contrast_ratio_threshold=0.5)

    def test_negative_scan_time_raises(self) -> None:
        with pytest.raises(ValueError, match="scan_time_limit_s"):
            Phase4Config(scan_time_limit_s=-10.0)

    def test_invalid_depth_range_raises(self) -> None:
        with pytest.raises(ValueError, match="depth_range_mm"):
            Phase4Config(depth_range_mm=(10.0, 5.0))

    def test_equal_depth_range_raises(self) -> None:
        with pytest.raises(ValueError, match="depth_range_mm"):
            Phase4Config(depth_range_mm=(5.0, 5.0))

    def test_negative_depth_range_raises(self) -> None:
        with pytest.raises(ValueError, match="depth_range_mm"):
            Phase4Config(depth_range_mm=(-1.0, 10.0))


# ── TestLayerContrastResultFields ────────────────────────────────


class TestLayerContrastResultFields:
    def test_construction(self) -> None:
        lc = LayerContrastResult(
            layer_a_name="fat",
            layer_b_name="muscle",
            t2_a_ms=80.0,
            t2_b_ms=35.0,
            t2_contrast_ratio=2.286,
            boundary_depth_mm=5.0,
            snr_at_boundary=12.0,
            in_depth_range=True,
            detectable=True,
        )
        assert lc.layer_a_name == "fat"
        assert lc.layer_b_name == "muscle"

    def test_frozen(self) -> None:
        lc = LayerContrastResult(
            layer_a_name="a", layer_b_name="b",
            t2_a_ms=50.0, t2_b_ms=100.0, t2_contrast_ratio=2.0,
            boundary_depth_mm=3.0, snr_at_boundary=5.0,
            in_depth_range=True, detectable=True,
        )
        with pytest.raises((AttributeError, TypeError)):
            lc.detectable = False  # type: ignore[misc]

    def test_in_depth_range_is_bool(self) -> None:
        lc = LayerContrastResult(
            layer_a_name="a", layer_b_name="b",
            t2_a_ms=50.0, t2_b_ms=100.0, t2_contrast_ratio=2.0,
            boundary_depth_mm=3.0, snr_at_boundary=5.0,
            in_depth_range=True, detectable=True,
        )
        assert isinstance(lc.in_depth_range, bool)

    def test_detectable_is_bool(self) -> None:
        lc = LayerContrastResult(
            layer_a_name="a", layer_b_name="b",
            t2_a_ms=50.0, t2_b_ms=100.0, t2_contrast_ratio=2.0,
            boundary_depth_mm=3.0, snr_at_boundary=5.0,
            in_depth_range=False, detectable=False,
        )
        assert isinstance(lc.detectable, bool)

    def test_all_float_fields_are_floats(self) -> None:
        lc = LayerContrastResult(
            layer_a_name="a", layer_b_name="b",
            t2_a_ms=50.0, t2_b_ms=100.0, t2_contrast_ratio=2.0,
            boundary_depth_mm=3.0, snr_at_boundary=5.0,
            in_depth_range=True, detectable=True,
        )
        assert isinstance(lc.t2_contrast_ratio, float)
        assert isinstance(lc.boundary_depth_mm, float)
        assert isinstance(lc.snr_at_boundary, float)

    def test_inf_t2_contrast_allowed(self) -> None:
        lc = LayerContrastResult(
            layer_a_name="x", layer_b_name="y",
            t2_a_ms=0.0, t2_b_ms=50.0, t2_contrast_ratio=float("inf"),
            boundary_depth_mm=2.0, snr_at_boundary=8.0,
            in_depth_range=True, detectable=True,
        )
        assert math.isinf(lc.t2_contrast_ratio)

    def test_nine_fields(self) -> None:
        lc = LayerContrastResult(
            layer_a_name="skin", layer_b_name="fat",
            t2_a_ms=30.0, t2_b_ms=80.0, t2_contrast_ratio=2.67,
            boundary_depth_mm=2.0, snr_at_boundary=2.0,
            in_depth_range=False, detectable=True,
        )
        assert lc.boundary_depth_mm == 2.0
        assert lc.snr_at_boundary == 2.0


# ── TestPhase4MilestoneResultFields ─────────────────────────────


class TestPhase4MilestoneResultFields:
    def test_phase4_milestone_closed_is_bool(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        assert isinstance(default_result.phase4_milestone_closed, bool)

    def test_snr_pass_is_bool(self, default_result: Phase4MilestoneResult) -> None:
        assert isinstance(default_result.snr_pass, bool)

    def test_scan_time_pass_is_bool(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        assert isinstance(default_result.scan_time_pass, bool)

    def test_all_in_range_layers_detectable_is_bool(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        assert isinstance(default_result.all_in_range_layers_detectable, bool)

    def test_layer_contrasts_is_tuple(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        assert isinstance(default_result.layer_contrasts, tuple)

    def test_depth_profile_present(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        assert isinstance(default_result.depth_profile, DepthProfile)

    def test_frozen(self, default_result: Phase4MilestoneResult) -> None:
        with pytest.raises((AttributeError, TypeError)):
            default_result.phase4_milestone_closed = False  # type: ignore[misc]

    def test_max_snr_geq_min_snr(self, default_result: Phase4MilestoneResult) -> None:
        assert default_result.max_snr_in_range >= default_result.min_snr_in_range


# ── TestComputeLayerContrast ──────────────────────────────────────


class TestComputeLayerContrast:
    def test_fat_muscle_t2_ratio(
        self,
        default_profile: DepthProfile,
    ) -> None:
        fat = TissueLayer("fat", thickness_mm=5.0, t1_ms=250, t2_ms=80)
        muscle = TissueLayer("muscle", thickness_mm=20.0, t1_ms=600, t2_ms=35)
        cfg = Phase4Config()
        lc = compute_layer_contrast(fat, muscle, 5.0, default_profile, cfg)
        # max(80,35)/min(80,35) = 80/35
        assert abs(lc.t2_contrast_ratio - 80.0 / 35.0) < 1e-9

    def test_equal_t2_ratio_is_one(
        self, default_profile: DepthProfile
    ) -> None:
        a = TissueLayer("a", thickness_mm=5.0, t2_ms=50)
        b = TissueLayer("b", thickness_mm=5.0, t2_ms=50)
        cfg = Phase4Config()
        lc = compute_layer_contrast(a, b, 5.0, default_profile, cfg)
        assert abs(lc.t2_contrast_ratio - 1.0) < 1e-9

    def test_zero_t2_gives_inf_ratio(
        self, default_profile: DepthProfile
    ) -> None:
        a = TissueLayer("a", thickness_mm=5.0, t2_ms=0.0)
        b = TissueLayer("b", thickness_mm=5.0, t2_ms=50)
        cfg = Phase4Config()
        lc = compute_layer_contrast(a, b, 5.0, default_profile, cfg)
        assert math.isinf(lc.t2_contrast_ratio)

    def test_ratio_is_always_geq_one(
        self, default_profile: DepthProfile
    ) -> None:
        a = TissueLayer("a", thickness_mm=2.0, t2_ms=20)
        b = TissueLayer("b", thickness_mm=5.0, t2_ms=100)
        cfg = Phase4Config()
        lc = compute_layer_contrast(a, b, 2.0, default_profile, cfg)
        assert lc.t2_contrast_ratio >= 1.0

    def test_in_range_flag_true(
        self, default_profile: DepthProfile
    ) -> None:
        a = TissueLayer("a", thickness_mm=5.0, t2_ms=80)
        b = TissueLayer("b", thickness_mm=5.0, t2_ms=35)
        cfg = Phase4Config()  # depth_range=(3.0, 15.0)
        lc = compute_layer_contrast(a, b, 7.0, default_profile, cfg)
        assert lc.in_depth_range is True

    def test_in_range_flag_false_shallow(
        self, default_profile: DepthProfile
    ) -> None:
        a = TissueLayer("a", thickness_mm=2.0, t2_ms=30)
        b = TissueLayer("b", thickness_mm=5.0, t2_ms=80)
        cfg = Phase4Config()  # depth_range=(3.0, 15.0)
        # boundary at 2mm is below 3mm lower bound
        lc = compute_layer_contrast(a, b, 2.0, default_profile, cfg)
        assert lc.in_depth_range is False

    def test_in_range_flag_false_deep(
        self, default_profile: DepthProfile
    ) -> None:
        a = TissueLayer("a", thickness_mm=20.0, t2_ms=35)
        b = TissueLayer("b", thickness_mm=5.0, t2_ms=1)
        cfg = Phase4Config()  # depth_range=(3.0, 15.0)
        # boundary at 27mm is beyond 15mm upper bound
        lc = compute_layer_contrast(a, b, 27.0, default_profile, cfg)
        assert lc.in_depth_range is False

    def test_detectable_true_when_criteria_met(
        self, default_profile: DepthProfile
    ) -> None:
        # fat-muscle boundary at 7mm: T2 ratio 2.29 > 1.5, SNR ~16 > 1.0
        fat = TissueLayer("fat", thickness_mm=7.0, t2_ms=80)
        muscle = TissueLayer("muscle", thickness_mm=10.0, t2_ms=35)
        cfg = Phase4Config()
        lc = compute_layer_contrast(fat, muscle, 7.0, default_profile, cfg)
        assert lc.detectable is True

    def test_detectable_false_when_snr_too_low(
        self, default_profile: DepthProfile
    ) -> None:
        # boundary at 27mm has SNR ~0; raise threshold to 100 so it fails
        a = TissueLayer("a", thickness_mm=27.0, t2_ms=35)
        b = TissueLayer("b", thickness_mm=3.0, t2_ms=1)
        cfg = Phase4Config(snr_at_boundary_threshold=100.0)
        lc = compute_layer_contrast(a, b, 27.0, default_profile, cfg)
        assert lc.detectable is False

    def test_detectable_false_low_t2_contrast(
        self, default_profile: DepthProfile
    ) -> None:
        # identical T2 → ratio=1.0 < 1.5 threshold
        a = TissueLayer("a", thickness_mm=5.0, t2_ms=50)
        b = TissueLayer("b", thickness_mm=5.0, t2_ms=50)
        cfg = Phase4Config()
        lc = compute_layer_contrast(a, b, 5.0, default_profile, cfg)
        assert lc.detectable is False

    def test_layer_names_echoed(
        self, default_profile: DepthProfile
    ) -> None:
        a = TissueLayer("alpha", thickness_mm=5.0, t2_ms=80)
        b = TissueLayer("beta", thickness_mm=5.0, t2_ms=40)
        cfg = Phase4Config()
        lc = compute_layer_contrast(a, b, 5.0, default_profile, cfg)
        assert lc.layer_a_name == "alpha"
        assert lc.layer_b_name == "beta"

    def test_t2_values_echoed(
        self, default_profile: DepthProfile
    ) -> None:
        a = TissueLayer("a", thickness_mm=5.0, t2_ms=80.0)
        b = TissueLayer("b", thickness_mm=5.0, t2_ms=40.0)
        cfg = Phase4Config()
        lc = compute_layer_contrast(a, b, 5.0, default_profile, cfg)
        assert lc.t2_a_ms == 80.0
        assert lc.t2_b_ms == 40.0

    def test_boundary_depth_echoed(
        self, default_profile: DepthProfile
    ) -> None:
        a = TissueLayer("a", thickness_mm=6.0, t2_ms=80)
        b = TissueLayer("b", thickness_mm=5.0, t2_ms=40)
        cfg = Phase4Config()
        lc = compute_layer_contrast(a, b, 6.0, default_profile, cfg)
        assert lc.boundary_depth_mm == 6.0

    def test_snr_at_boundary_non_negative(
        self, default_profile: DepthProfile
    ) -> None:
        a = TissueLayer("a", thickness_mm=5.0, t2_ms=80)
        b = TissueLayer("b", thickness_mm=5.0, t2_ms=40)
        cfg = Phase4Config()
        lc = compute_layer_contrast(a, b, 5.0, default_profile, cfg)
        assert lc.snr_at_boundary >= 0.0


# ── TestValidatePhase4MilestoneDefault ───────────────────────────


class TestValidatePhase4MilestoneDefault:
    def test_milestone_closed(self, default_result: Phase4MilestoneResult) -> None:
        assert default_result.phase4_milestone_closed is True

    def test_snr_pass(self, default_result: Phase4MilestoneResult) -> None:
        assert default_result.snr_pass is True

    def test_scan_time_pass(self, default_result: Phase4MilestoneResult) -> None:
        assert default_result.scan_time_pass is True

    def test_all_in_range_layers_detectable(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        assert default_result.all_in_range_layers_detectable is True

    def test_max_snr_exceeds_threshold(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        assert default_result.max_snr_in_range >= 3.0

    def test_scan_time_within_budget(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        assert default_result.scan_time_s <= 120.0

    def test_n_depths_positive(self, default_result: Phase4MilestoneResult) -> None:
        assert default_result.n_depths_evaluated > 0

    def test_forearm_has_three_layer_contrasts(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        # FOREARM_LAYERS has 4 layers → 3 adjacent boundaries
        assert len(default_result.layer_contrasts) == 3

    def test_fat_muscle_boundary_in_range(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        # fat→muscle boundary is at 7 mm, within [3, 15]
        fat_muscle = next(
            lc for lc in default_result.layer_contrasts
            if lc.layer_a_name == "subcutaneous_fat"
        )
        assert fat_muscle.in_depth_range is True

    def test_fat_muscle_boundary_detectable(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        fat_muscle = next(
            lc for lc in default_result.layer_contrasts
            if lc.layer_a_name == "subcutaneous_fat"
        )
        assert fat_muscle.detectable is True

    def test_fat_muscle_t2_ratio_above_threshold(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        fat_muscle = next(
            lc for lc in default_result.layer_contrasts
            if lc.layer_a_name == "subcutaneous_fat"
        )
        # T2_fat=80ms, T2_muscle=35ms → ratio = 80/35 ≈ 2.29
        assert fat_muscle.t2_contrast_ratio > 1.5

    def test_depth_profile_has_correct_type(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        assert isinstance(default_result.depth_profile, DepthProfile)


# ── TestPhase4SNRCriteria ────────────────────────────────────────


class TestPhase4SNRCriteria:
    def test_very_high_threshold_fails_snr(self) -> None:
        result = validate_phase4_milestone(
            config=Phase4Config(signal_snr_threshold=1000.0)
        )
        assert result.snr_pass is False

    def test_very_high_threshold_closes_milestone_false(self) -> None:
        result = validate_phase4_milestone(
            config=Phase4Config(signal_snr_threshold=1000.0)
        )
        assert result.phase4_milestone_closed is False

    def test_threshold_below_max_snr_passes(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        # Use a threshold well below the observed max (20.75)
        result = validate_phase4_milestone(
            config=Phase4Config(signal_snr_threshold=0.1)
        )
        assert result.snr_pass is True

    def test_max_snr_geq_threshold_means_pass(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        # max_snr_in_range ≈ 20.75; threshold of 3.0 → pass
        assert default_result.max_snr_in_range >= 3.0
        assert default_result.snr_pass is True

    def test_none_config_uses_defaults(self) -> None:
        result = validate_phase4_milestone(config=None)
        assert isinstance(result, Phase4MilestoneResult)


# ── TestPhase4ContrastCriteria ────────────────────────────────────


class TestPhase4ContrastCriteria:
    def test_identical_t2_layers_not_detectable(
        self, default_profile: DepthProfile
    ) -> None:
        cfg = Phase4Config()
        a = TissueLayer("a", thickness_mm=3.0, t2_ms=50)
        b = TissueLayer("b", thickness_mm=3.0, t2_ms=50)
        lc = compute_layer_contrast(a, b, 3.0, default_profile, cfg)
        assert lc.t2_contrast_ratio == 1.0
        assert lc.detectable is False

    def test_extreme_t2_ratio_is_inf_when_zero(
        self, default_profile: DepthProfile
    ) -> None:
        cfg = Phase4Config()
        a = TissueLayer("bone", thickness_mm=3.0, t2_ms=0.5)
        b = TissueLayer("void", thickness_mm=3.0, t2_ms=0.0)
        lc = compute_layer_contrast(a, b, 3.0, default_profile, cfg)
        assert math.isinf(lc.t2_contrast_ratio)

    def test_large_ratio_always_detectable_when_snr_ok(
        self, default_profile: DepthProfile
    ) -> None:
        # boundary at 5mm has SNR ~20; ratio 100/1 >> 1.5
        cfg = Phase4Config()
        a = TissueLayer("a", thickness_mm=5.0, t2_ms=100)
        b = TissueLayer("b", thickness_mm=5.0, t2_ms=1)
        lc = compute_layer_contrast(a, b, 5.0, default_profile, cfg)
        assert lc.t2_contrast_ratio > 1.5
        assert lc.detectable is True

    def test_contrast_threshold_one_always_passes(
        self, default_profile: DepthProfile
    ) -> None:
        # contrast_ratio_threshold=1.0 allows ratio ≥ 1 (always)
        cfg = Phase4Config(contrast_ratio_threshold=1.0)
        a = TissueLayer("a", thickness_mm=5.0, t2_ms=50)
        b = TissueLayer("b", thickness_mm=5.0, t2_ms=51)
        lc = compute_layer_contrast(a, b, 5.0, default_profile, cfg)
        # ratio = 51/50 = 1.02 ≥ 1.0
        assert lc.detectable is True

    def test_in_range_contrast_drives_all_detectable(self) -> None:
        # If the only in-range boundary has very high threshold, milestone fails
        result = validate_phase4_milestone(
            config=Phase4Config(
                contrast_ratio_threshold=1000.0,
                depth_range_mm=(3.0, 15.0),
            )
        )
        assert result.all_in_range_layers_detectable is False
        assert result.phase4_milestone_closed is False


# ── TestPhase4ScanTimeCriteria ────────────────────────────────────


class TestPhase4ScanTimeCriteria:
    def test_tight_scan_time_limit_fails(self) -> None:
        result = validate_phase4_milestone(
            config=Phase4Config(scan_time_limit_s=1.0)
        )
        assert result.scan_time_pass is False

    def test_generous_scan_time_limit_passes(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        assert default_result.scan_time_pass is True

    def test_scan_time_s_is_float(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        assert isinstance(default_result.scan_time_s, float)


# ── TestPhase4MilestoneClosedLogic ───────────────────────────────


class TestPhase4MilestoneClosedLogic:
    def test_all_pass_implies_closed(
        self, default_result: Phase4MilestoneResult
    ) -> None:
        assert (
            default_result.snr_pass
            and default_result.scan_time_pass
            and default_result.all_in_range_layers_detectable
        )
        assert default_result.phase4_milestone_closed is True

    def test_snr_fail_opens_milestone(self) -> None:
        result = validate_phase4_milestone(
            config=Phase4Config(signal_snr_threshold=9999.0)
        )
        assert result.snr_pass is False
        assert result.phase4_milestone_closed is False

    def test_scan_time_fail_opens_milestone(self) -> None:
        result = validate_phase4_milestone(
            config=Phase4Config(scan_time_limit_s=0.001)
        )
        assert result.scan_time_pass is False
        assert result.phase4_milestone_closed is False

    def test_contrast_fail_opens_milestone(self) -> None:
        result = validate_phase4_milestone(
            config=Phase4Config(contrast_ratio_threshold=999.0)
        )
        assert result.all_in_range_layers_detectable is False
        assert result.phase4_milestone_closed is False
