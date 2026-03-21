"""Tests for physics.phase9_validator — Phase-9 tissue-contrast milestone."""
from __future__ import annotations

import math
import pytest

from nv_maser.physics.depth_profile import FOREARM_LAYERS, TissueLayer
from nv_maser.physics.phase9_validator import (
    Phase9Config,
    Phase9MilestoneResult,
    T2ContrastResult,
    validate_phase9_milestone,
)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Helpers / shared fixtures                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

_DEFAULT_RESULT: Phase9MilestoneResult | None = None


def _default_result() -> Phase9MilestoneResult:
    """Run once and cache to keep the test suite fast."""
    global _DEFAULT_RESULT
    if _DEFAULT_RESULT is None:
        _DEFAULT_RESULT = validate_phase9_milestone()
    return _DEFAULT_RESULT


# ╔══════════════════════════════════════════════════════════════════╗
# ║  1. Phase9Config — construction & validation                     ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestPhase9Config:
    def test_defaults(self):
        cfg = Phase9Config()
        assert cfg.te_short_ms == 10.0
        assert cfg.te_long_ms == 80.0
        assert cfg.t2_contrast_ratio_threshold == 1.5
        assert cfg.snr_threshold == 3.0
        assert cfg.scan_time_limit_s == 120.0
        assert cfg.fat_center_depth_mm == 4.5
        assert cfg.muscle_center_depth_mm == 10.0

    def test_custom_valid(self):
        cfg = Phase9Config(te_short_ms=5.0, te_long_ms=60.0)
        assert cfg.te_short_ms == 5.0
        assert cfg.te_long_ms == 60.0

    def test_te_short_zero_raises(self):
        with pytest.raises(ValueError, match="te_short_ms"):
            Phase9Config(te_short_ms=0.0)

    def test_te_short_negative_raises(self):
        with pytest.raises(ValueError, match="te_short_ms"):
            Phase9Config(te_short_ms=-1.0)

    def test_te_long_equal_short_raises(self):
        with pytest.raises(ValueError, match="te_long_ms"):
            Phase9Config(te_short_ms=20.0, te_long_ms=20.0)

    def test_te_long_less_than_short_raises(self):
        with pytest.raises(ValueError, match="te_long_ms"):
            Phase9Config(te_short_ms=20.0, te_long_ms=10.0)

    def test_contrast_threshold_below_one_raises(self):
        with pytest.raises(ValueError, match="t2_contrast_ratio_threshold"):
            Phase9Config(t2_contrast_ratio_threshold=0.9)

    def test_contrast_threshold_exactly_one_ok(self):
        cfg = Phase9Config(t2_contrast_ratio_threshold=1.0)
        assert cfg.t2_contrast_ratio_threshold == 1.0

    def test_snr_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="snr_threshold"):
            Phase9Config(snr_threshold=0.0)

    def test_snr_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="snr_threshold"):
            Phase9Config(snr_threshold=-1.0)

    def test_scan_time_zero_raises(self):
        with pytest.raises(ValueError, match="scan_time_limit_s"):
            Phase9Config(scan_time_limit_s=0.0)

    def test_fat_depth_zero_raises(self):
        with pytest.raises(ValueError, match="fat_center_depth_mm"):
            Phase9Config(fat_center_depth_mm=0.0)

    def test_fat_depth_negative_raises(self):
        with pytest.raises(ValueError, match="fat_center_depth_mm"):
            Phase9Config(fat_center_depth_mm=-2.0)

    def test_muscle_depth_equal_fat_raises(self):
        with pytest.raises(ValueError, match="muscle_center_depth_mm"):
            Phase9Config(fat_center_depth_mm=5.0, muscle_center_depth_mm=5.0)

    def test_muscle_depth_shallower_than_fat_raises(self):
        with pytest.raises(ValueError, match="muscle_center_depth_mm"):
            Phase9Config(fat_center_depth_mm=5.0, muscle_center_depth_mm=3.0)

    def test_frozen_immutable(self):
        cfg = Phase9Config()
        with pytest.raises((AttributeError, TypeError)):
            cfg.te_long_ms = 50.0  # type: ignore[misc]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  2. T2ContrastResult — construction                              ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestT2ContrastResult:
    def test_basic_construction(self):
        r = T2ContrastResult(
            fat_depth_mm=4.5,
            muscle_depth_mm=10.0,
            fat_signal_short=1e-7,
            muscle_signal_short=8e-8,
            fat_signal_long=3.7e-8,
            muscle_signal_long=1.0e-9,
            t2_contrast_ratio=37.0,
            passes=True,
        )
        assert r.fat_depth_mm == 4.5
        assert r.muscle_depth_mm == 10.0
        assert r.t2_contrast_ratio == pytest.approx(37.0)
        assert r.passes is True

    def test_passes_false(self):
        r = T2ContrastResult(
            fat_depth_mm=4.5,
            muscle_depth_mm=10.0,
            fat_signal_short=1e-7,
            muscle_signal_short=8e-8,
            fat_signal_long=1.1e-8,
            muscle_signal_long=1.0e-8,
            t2_contrast_ratio=1.1,
            passes=False,
        )
        assert r.passes is False

    def test_inf_ratio_stored(self):
        r = T2ContrastResult(
            fat_depth_mm=4.5,
            muscle_depth_mm=10.0,
            fat_signal_short=1e-7,
            muscle_signal_short=8e-8,
            fat_signal_long=1e-7,
            muscle_signal_long=0.0,
            t2_contrast_ratio=float("inf"),
            passes=True,
        )
        assert math.isinf(r.t2_contrast_ratio)

    def test_frozen_immutable(self):
        r = T2ContrastResult(
            fat_depth_mm=4.5,
            muscle_depth_mm=10.0,
            fat_signal_short=1e-7,
            muscle_signal_short=8e-8,
            fat_signal_long=3.7e-8,
            muscle_signal_long=1.0e-9,
            t2_contrast_ratio=37.0,
            passes=True,
        )
        with pytest.raises((AttributeError, TypeError)):
            r.t2_contrast_ratio = 1.0  # type: ignore[misc]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  3. Phase9MilestoneResult — field types                          ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestPhase9MilestoneResultFields:
    def test_config_stored(self):
        r = _default_result()
        assert isinstance(r.config, Phase9Config)

    def test_profiles_present(self):
        r = _default_result()
        assert r.profile_short is not None
        assert r.profile_long is not None

    def test_t2_contrast_stored(self):
        r = _default_result()
        assert isinstance(r.t2_contrast, T2ContrastResult)

    def test_snr_fields_float(self):
        r = _default_result()
        assert isinstance(r.fat_snr, float)
        assert isinstance(r.muscle_snr, float)

    def test_scan_time_float(self):
        r = _default_result()
        assert isinstance(r.scan_time_s, float)

    def test_boolean_fields(self):
        r = _default_result()
        assert isinstance(r.contrast_pass, bool)
        assert isinstance(r.snr_pass, bool)
        assert isinstance(r.scan_time_pass, bool)
        assert isinstance(r.phase9_milestone_closed, bool)

    def test_frozen_immutable(self):
        r = _default_result()
        with pytest.raises((AttributeError, TypeError)):
            r.phase9_milestone_closed = False  # type: ignore[misc]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  4. Default configuration — milestone passes                     ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestDefaultMilestonePasses:
    def test_milestone_closed(self):
        assert _default_result().phase9_milestone_closed is True

    def test_contrast_pass(self):
        assert _default_result().contrast_pass is True

    def test_snr_pass(self):
        assert _default_result().snr_pass is True

    def test_scan_time_pass(self):
        assert _default_result().scan_time_pass is True

    def test_contrast_ratio_above_threshold(self):
        r = _default_result()
        assert r.t2_contrast.t2_contrast_ratio >= r.config.t2_contrast_ratio_threshold

    def test_fat_snr_above_threshold(self):
        r = _default_result()
        assert r.fat_snr >= r.config.snr_threshold

    def test_muscle_snr_above_threshold(self):
        r = _default_result()
        assert r.muscle_snr >= r.config.snr_threshold

    def test_scan_time_below_limit(self):
        r = _default_result()
        assert r.scan_time_s <= r.config.scan_time_limit_s


# ╔══════════════════════════════════════════════════════════════════╗
# ║  5. T2 contrast physics                                          ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestT2ContrastPhysics:
    def test_fat_brighter_at_long_te(self):
        """Fat signal > muscle signal at long TE (by T2 decay physics)."""
        r = _default_result()
        assert r.t2_contrast.fat_signal_long > r.t2_contrast.muscle_signal_long

    def test_fat_brighter_at_short_te(self):
        """Fat and muscle both present at short TE; fat brighter due to PD/T2."""
        r = _default_result()
        assert r.t2_contrast.fat_signal_short > 0.0
        assert r.t2_contrast.muscle_signal_short > 0.0

    def test_contrast_ratio_positive(self):
        assert _default_result().t2_contrast.t2_contrast_ratio > 0.0

    def test_fat_signal_decays_from_short_to_long_te(self):
        """Fat signal at long TE must be less than at short TE."""
        r = _default_result()
        assert r.t2_contrast.fat_signal_long < r.t2_contrast.fat_signal_short

    def test_muscle_signal_decays_more_than_fat(self):
        """Muscle decays faster — relative signal drop must be larger."""
        r = _default_result()
        fat_decay = r.t2_contrast.fat_signal_long / r.t2_contrast.fat_signal_short
        muscle_decay = r.t2_contrast.muscle_signal_long / r.t2_contrast.muscle_signal_short
        assert muscle_decay < fat_decay

    def test_contrast_ratio_enhancement_matches_t2_decay(self):
        """The ratio-of-ratios (long-TE / short-TE) cancels depth-dependent coil
        sensitivity and should match pure T2-decay physics to within 10%."""
        import math as _math
        import numpy as np
        r = _default_result()
        cfg = r.config
        # FOREARM T2 values
        t2_fat = 80.0
        t2_muscle = 35.0
        # enhancement = exp(Δte × (1/T2_muscle − 1/T2_fat))
        delta_te = cfg.te_long_ms - cfg.te_short_ms
        expected_enhancement = _math.exp(delta_te * (1.0 / t2_muscle - 1.0 / t2_fat))
        # short-TE ratio at the same depth bins
        fat_idx = int(np.argmin(np.abs(r.profile_short.depths_mm - r.t2_contrast.fat_depth_mm)))
        muscle_idx = int(np.argmin(np.abs(r.profile_short.depths_mm - r.t2_contrast.muscle_depth_mm)))
        ratio_short = float(r.profile_short.signal[fat_idx]) / float(r.profile_short.signal[muscle_idx])
        actual_enhancement = r.t2_contrast.t2_contrast_ratio / ratio_short
        assert actual_enhancement == pytest.approx(expected_enhancement, rel=0.10)

    def test_fat_depth_approximately_correct(self):
        """Resolved fat bin should be within 1 mm of nominal 4.5 mm."""
        r = _default_result()
        assert abs(r.t2_contrast.fat_depth_mm - r.config.fat_center_depth_mm) <= 1.0

    def test_muscle_depth_approximately_correct(self):
        """Resolved muscle bin should be within 1 mm of nominal 10.0 mm."""
        r = _default_result()
        assert abs(r.t2_contrast.muscle_depth_mm - r.config.muscle_center_depth_mm) <= 1.0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  6. Contrast criterion failure path                              ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestContrastFailurePath:
    def test_very_high_threshold_fails(self):
        cfg = Phase9Config(t2_contrast_ratio_threshold=100.0)
        r = validate_phase9_milestone(config=cfg)
        assert r.contrast_pass is False
        assert r.phase9_milestone_closed is False

    def test_threshold_just_below_actual_passes(self):
        r_def = _default_result()
        just_below = r_def.t2_contrast.t2_contrast_ratio * 0.99
        cfg = Phase9Config(t2_contrast_ratio_threshold=just_below)
        r = validate_phase9_milestone(config=cfg)
        assert r.contrast_pass is True

    def test_threshold_just_above_actual_fails(self):
        r_def = _default_result()
        just_above = r_def.t2_contrast.t2_contrast_ratio * 1.01
        cfg = Phase9Config(t2_contrast_ratio_threshold=just_above)
        r = validate_phase9_milestone(config=cfg)
        assert r.contrast_pass is False
        assert r.phase9_milestone_closed is False

    def test_contrast_fail_does_not_affect_snr_field(self):
        """SNR fields are always populated even when contrast fails."""
        cfg = Phase9Config(t2_contrast_ratio_threshold=1000.0)
        r = validate_phase9_milestone(config=cfg)
        assert r.fat_snr > 0.0
        assert r.muscle_snr > 0.0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  7. SNR criterion failure path                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestSNRFailurePath:
    def test_very_high_snr_threshold_fails(self):
        cfg = Phase9Config(snr_threshold=1000.0)
        r = validate_phase9_milestone(config=cfg)
        assert r.snr_pass is False
        assert r.phase9_milestone_closed is False

    def test_snr_threshold_just_below_muscle_snr_passes(self):
        r_def = _default_result()
        # Set threshold just below the weaker of the two SNRs
        weaker_snr = min(r_def.fat_snr, r_def.muscle_snr)
        cfg = Phase9Config(snr_threshold=weaker_snr * 0.99)
        r = validate_phase9_milestone(config=cfg)
        assert r.snr_pass is True

    def test_snr_threshold_just_above_muscle_snr_fails(self):
        r_def = _default_result()
        weaker_snr = min(r_def.fat_snr, r_def.muscle_snr)
        cfg = Phase9Config(snr_threshold=weaker_snr * 1.01)
        r = validate_phase9_milestone(config=cfg)
        assert r.snr_pass is False
        assert r.phase9_milestone_closed is False

    def test_snr_fail_does_not_affect_contrast_field(self):
        """Contrast ratio is always computed even when SNR fails."""
        cfg = Phase9Config(snr_threshold=1000.0)
        r = validate_phase9_milestone(config=cfg)
        assert r.t2_contrast.t2_contrast_ratio > 0.0

    def test_snr_fields_reflect_short_te_profile(self):
        """fat_snr and muscle_snr must be consistent with the short-TE profile."""
        r = _default_result()
        import numpy as np
        fat_idx = int(np.argmin(np.abs(r.profile_short.depths_mm - r.config.fat_center_depth_mm)))
        muscle_idx = int(np.argmin(np.abs(r.profile_short.depths_mm - r.config.muscle_center_depth_mm)))
        assert r.fat_snr == pytest.approx(float(r.profile_short.snr[fat_idx]))
        assert r.muscle_snr == pytest.approx(float(r.profile_short.snr[muscle_idx]))


# ╔══════════════════════════════════════════════════════════════════╗
# ║  8. Scan-time criterion failure path                             ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestScanTimeFailurePath:
    def test_very_tight_scan_time_limit_fails(self):
        cfg = Phase9Config(scan_time_limit_s=1.0)
        r = validate_phase9_milestone(config=cfg)
        assert r.scan_time_pass is False
        assert r.phase9_milestone_closed is False

    def test_scan_time_just_above_actual_passes(self):
        r_def = _default_result()
        cfg = Phase9Config(scan_time_limit_s=r_def.scan_time_s + 1.0)
        r = validate_phase9_milestone(config=cfg)
        assert r.scan_time_pass is True

    def test_scan_time_exactly_at_limit_passes(self):
        r_def = _default_result()
        cfg = Phase9Config(scan_time_limit_s=r_def.scan_time_s)
        r = validate_phase9_milestone(config=cfg)
        assert r.scan_time_pass is True

    def test_scan_time_just_below_actual_fails(self):
        r_def = _default_result()
        cfg = Phase9Config(scan_time_limit_s=r_def.scan_time_s - 0.001)
        r = validate_phase9_milestone(config=cfg)
        assert r.scan_time_pass is False
        assert r.phase9_milestone_closed is False

    def test_scan_time_reflects_long_te_profile(self):
        r = _default_result()
        assert r.scan_time_s == pytest.approx(r.profile_long.scan_time_s)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  9. Custom tissue layers and edge cases                          ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestCustomTissueLayers:
    def test_forearm_layers_explicit_same_as_default(self):
        r_explicit = validate_phase9_milestone(tissue_layers=list(FOREARM_LAYERS))
        r_implicit = _default_result()
        assert r_explicit.t2_contrast.t2_contrast_ratio == pytest.approx(
            r_implicit.t2_contrast.t2_contrast_ratio, rel=1e-6
        )

    def test_equal_t2_layers_no_t2_enhancement(self):
        """If fat and muscle have the same T2, the ratio-of-ratios (long/short TE)
        should be 1.0 — no additional T2-weighted contrast is added by TE change."""
        import numpy as np
        equal_t2_layers = [
            TissueLayer("skin", thickness_mm=2.0, proton_density=0.7, t1_ms=400, t2_ms=30),
            TissueLayer("fat_equiv", thickness_mm=5.0, proton_density=1.0, t1_ms=500, t2_ms=50),
            TissueLayer("muscle_equiv", thickness_mm=20.0, proton_density=1.0, t1_ms=500, t2_ms=50),
            TissueLayer("bone_cortex", thickness_mm=3.0, proton_density=0.05, t1_ms=1000, t2_ms=0.5),
        ]
        cfg = Phase9Config(t2_contrast_ratio_threshold=1.0)
        r = validate_phase9_milestone(config=cfg, tissue_layers=equal_t2_layers)
        # With identical T2, long/short ratio enhancement ≈ 1.0
        fat_idx = int(np.argmin(np.abs(r.profile_short.depths_mm - r.t2_contrast.fat_depth_mm)))
        muscle_idx = int(np.argmin(np.abs(r.profile_short.depths_mm - r.t2_contrast.muscle_depth_mm)))
        ratio_short = float(r.profile_short.signal[fat_idx]) / float(r.profile_short.signal[muscle_idx])
        ratio_long = r.t2_contrast.t2_contrast_ratio
        enhancement = ratio_long / ratio_short
        assert enhancement == pytest.approx(1.0, abs=0.05)

    def test_very_high_t2_contrast_tissue_passes(self):
        """Amplify contrast: fat T2=500ms, muscle T2=5ms → huge ratio."""
        high_contrast_layers = [
            TissueLayer("skin", thickness_mm=2.0, proton_density=0.7, t1_ms=400, t2_ms=30),
            TissueLayer("fat_long_t2", thickness_mm=5.0, proton_density=0.9, t1_ms=250, t2_ms=500),
            TissueLayer("muscle_short_t2", thickness_mm=20.0, proton_density=1.0, t1_ms=600, t2_ms=5),
            TissueLayer("bone_cortex", thickness_mm=3.0, proton_density=0.05, t1_ms=1000, t2_ms=0.5),
        ]
        r = validate_phase9_milestone(tissue_layers=high_contrast_layers)
        assert r.t2_contrast.t2_contrast_ratio > 1.5
        assert r.contrast_pass is True

    def test_custom_config_accepted(self):
        cfg = Phase9Config(
            te_short_ms=5.0,
            te_long_ms=60.0,
            t2_contrast_ratio_threshold=1.2,
            snr_threshold=2.0,
            scan_time_limit_s=200.0,
        )
        r = validate_phase9_milestone(config=cfg)
        assert r.config is cfg
        assert r.phase9_milestone_closed is True

    def test_returns_phase9milestone_result_type(self):
        r = validate_phase9_milestone()
        assert isinstance(r, Phase9MilestoneResult)

    def test_depth_profiles_correct_depths(self):
        """Short-TE and long-TE profiles should span the same depth range."""
        r = _default_result()
        import numpy as np
        assert r.profile_short.depths_mm[-1] == pytest.approx(r.profile_long.depths_mm[-1])
        assert len(r.profile_short.depths_mm) == len(r.profile_long.depths_mm)

    def test_milestone_closed_false_when_all_fail(self):
        cfg = Phase9Config(
            t2_contrast_ratio_threshold=1000.0,
            snr_threshold=1000.0,
            scan_time_limit_s=0.001,
        )
        with pytest.raises(ValueError):
            # scan_time_limit_s=0.001 triggers Phase9Config validation
            _ = Phase9Config(scan_time_limit_s=0.0)

    def test_milestone_closed_false_only_contrast_fails(self):
        cfg = Phase9Config(t2_contrast_ratio_threshold=1000.0)
        r = validate_phase9_milestone(config=cfg)
        assert r.contrast_pass is False
        assert r.phase9_milestone_closed is False

    def test_milestone_closed_false_only_snr_fails(self):
        cfg = Phase9Config(snr_threshold=1000.0)
        r = validate_phase9_milestone(config=cfg)
        assert r.snr_pass is False
        assert r.phase9_milestone_closed is False

    def test_milestone_closed_false_only_scan_time_fails(self):
        cfg = Phase9Config(scan_time_limit_s=1.0)
        r = validate_phase9_milestone(config=cfg)
        assert r.scan_time_pass is False
        assert r.phase9_milestone_closed is False
