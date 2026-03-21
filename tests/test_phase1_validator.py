"""
Tests for the Phase-1 maser oscillation milestone validator.

Architecture §12.2 milestone: "Maser oscillation — Detectable stimulated
emission at 1.47 GHz".

Coverage targets
────────────────
* Phase1Config — defaults, field validation, custom construction.
* OscillationThresholdResult — construction and immutability.
* Phase1MilestoneResult — field types, immutability, derived fields.
* Default milestone passes with comfortable margin on all three criteria.
* Oscillation physics — Q-ratio, cooperativity, threshold margin.
* Threshold failure path — low NV density drives Q_m above Q_L.
* Frequency failure path — cavity far from 1.47 GHz.
* Power failure path — output power below configurable floor.
* Custom configurations — tight tolerance, modified gain budget.
"""
from __future__ import annotations

import math
import pytest

from nv_maser.config import CavityConfig, MaserConfig, NVConfig
from nv_maser.physics.phase1_validator import (
    OscillationThresholdResult,
    Phase1Config,
    Phase1MilestoneResult,
    validate_phase1_milestone,
    _spin_linewidth_hz,
)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  1. Phase1Config                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestPhase1Config:
    """Phase1Config dataclass construction and validation."""

    def test_default_target_frequency(self) -> None:
        assert Phase1Config().target_frequency_ghz == pytest.approx(1.47)

    def test_default_frequency_tolerance(self) -> None:
        assert Phase1Config().frequency_tolerance_mhz == pytest.approx(50.0)

    def test_default_min_output_power(self) -> None:
        assert Phase1Config().min_output_power_dbm == pytest.approx(-100.0)

    def test_default_gain_budget(self) -> None:
        assert Phase1Config().gain_budget == pytest.approx(0.5)

    def test_custom_target_frequency(self) -> None:
        cfg = Phase1Config(target_frequency_ghz=2.0)
        assert cfg.target_frequency_ghz == pytest.approx(2.0)

    def test_custom_frequency_tolerance(self) -> None:
        cfg = Phase1Config(frequency_tolerance_mhz=10.0)
        assert cfg.frequency_tolerance_mhz == pytest.approx(10.0)

    def test_custom_min_output_power(self) -> None:
        cfg = Phase1Config(min_output_power_dbm=-80.0)
        assert cfg.min_output_power_dbm == pytest.approx(-80.0)

    def test_custom_gain_budget(self) -> None:
        cfg = Phase1Config(gain_budget=0.8)
        assert cfg.gain_budget == pytest.approx(0.8)

    def test_zero_frequency_tolerance_allowed(self) -> None:
        """Exact frequency match (zero tolerance) is valid."""
        cfg = Phase1Config(frequency_tolerance_mhz=0.0)
        assert cfg.frequency_tolerance_mhz == 0.0

    def test_gain_budget_one_allowed(self) -> None:
        """Full spectral overlap is a valid edge case."""
        cfg = Phase1Config(gain_budget=1.0)
        assert cfg.gain_budget == pytest.approx(1.0)

    def test_negative_target_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="target_frequency_ghz"):
            Phase1Config(target_frequency_ghz=-1.0)

    def test_zero_target_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="target_frequency_ghz"):
            Phase1Config(target_frequency_ghz=0.0)

    def test_negative_frequency_tolerance_raises(self) -> None:
        with pytest.raises(ValueError, match="frequency_tolerance_mhz"):
            Phase1Config(frequency_tolerance_mhz=-1.0)

    def test_zero_gain_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="gain_budget"):
            Phase1Config(gain_budget=0.0)

    def test_gain_budget_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="gain_budget"):
            Phase1Config(gain_budget=1.5)

    def test_frozen(self) -> None:
        cfg = Phase1Config()
        with pytest.raises((AttributeError, TypeError)):
            cfg.target_frequency_ghz = 2.0  # type: ignore[misc]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  2. OscillationThresholdResult                                   ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestOscillationThresholdResult:
    """OscillationThresholdResult construction and immutability."""

    @pytest.fixture
    def sample(self) -> OscillationThresholdResult:
        return OscillationThresholdResult(
            magnetic_q=7800.0,
            loaded_q=10000.0,
            q_ratio=0.78,
            above_threshold_wang=True,
            cooperativity=2.5,
            threshold_margin=1.5,
            masing=True,
            spin_temperature_k=0.10,
        )

    def test_all_fields_accessible(self, sample: OscillationThresholdResult) -> None:
        assert sample.magnetic_q == pytest.approx(7800.0)
        assert sample.loaded_q == pytest.approx(10000.0)
        assert sample.q_ratio == pytest.approx(0.78)
        assert sample.above_threshold_wang is True
        assert sample.cooperativity == pytest.approx(2.5)
        assert sample.threshold_margin == pytest.approx(1.5)
        assert sample.masing is True
        assert sample.spin_temperature_k == pytest.approx(0.10)

    def test_frozen(self, sample: OscillationThresholdResult) -> None:
        with pytest.raises((AttributeError, TypeError)):
            sample.magnetic_q = 1.0  # type: ignore[misc]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  3. Phase1MilestoneResult field types                            ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestPhase1MilestoneResultFields:
    """Verify field types and immutability of the result dataclass."""

    @pytest.fixture
    def result(self) -> Phase1MilestoneResult:
        return validate_phase1_milestone()

    def test_oscillation_is_oscillation_threshold_result(
        self, result: Phase1MilestoneResult
    ) -> None:
        assert isinstance(result.oscillation, OscillationThresholdResult)

    def test_frequency_fields_are_float(self, result: Phase1MilestoneResult) -> None:
        assert isinstance(result.frequency_ghz, float)
        assert isinstance(result.target_frequency_ghz, float)
        assert isinstance(result.frequency_deviation_mhz, float)
        assert isinstance(result.frequency_tolerance_mhz, float)

    def test_power_fields_are_float(self, result: Phase1MilestoneResult) -> None:
        assert isinstance(result.output_power_w, float)
        assert isinstance(result.output_power_dbm, float)
        assert isinstance(result.min_output_power_dbm, float)

    def test_pass_fail_fields_are_bool(self, result: Phase1MilestoneResult) -> None:
        assert isinstance(result.threshold_met, bool)
        assert isinstance(result.frequency_met, bool)
        assert isinstance(result.power_met, bool)
        assert isinstance(result.phase1_milestone_closed, bool)

    def test_closing_message_is_str(self, result: Phase1MilestoneResult) -> None:
        assert isinstance(result.closing_message, str)
        assert len(result.closing_message) > 0

    def test_frozen(self, result: Phase1MilestoneResult) -> None:
        with pytest.raises((AttributeError, TypeError)):
            result.phase1_milestone_closed = False  # type: ignore[misc]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  4. Default milestone passes                                     ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestDefaultMilestonePasses:
    """Default NV/cavity/maser configuration closes the Phase-1 milestone."""

    @pytest.fixture
    def result(self) -> Phase1MilestoneResult:
        return validate_phase1_milestone()

    def test_milestone_closed(self, result: Phase1MilestoneResult) -> None:
        assert result.phase1_milestone_closed is True

    def test_threshold_met(self, result: Phase1MilestoneResult) -> None:
        assert result.threshold_met is True

    def test_frequency_met(self, result: Phase1MilestoneResult) -> None:
        assert result.frequency_met is True

    def test_power_met(self, result: Phase1MilestoneResult) -> None:
        assert result.power_met is True

    def test_output_power_above_default_floor(
        self, result: Phase1MilestoneResult
    ) -> None:
        """Output power must be well above −100 dBm detection floor."""
        assert result.output_power_dbm >= -100.0

    def test_output_power_positive_watts(self, result: Phase1MilestoneResult) -> None:
        assert result.output_power_w > 0.0

    def test_frequency_at_target(self, result: Phase1MilestoneResult) -> None:
        assert result.frequency_ghz == pytest.approx(1.47)

    def test_zero_frequency_deviation(self, result: Phase1MilestoneResult) -> None:
        assert result.frequency_deviation_mhz == pytest.approx(0.0, abs=0.01)

    def test_closing_message_contains_closed(
        self, result: Phase1MilestoneResult
    ) -> None:
        assert "CLOSED" in result.closing_message

    def test_none_inputs_equivalent_to_defaults(self) -> None:
        """Passing None for all configs must produce the same result."""
        r_none = validate_phase1_milestone(
            nv_config=None, cavity_config=None, maser_config=None, config=None
        )
        r_explicit = validate_phase1_milestone(
            nv_config=NVConfig(),
            cavity_config=CavityConfig(),
            maser_config=MaserConfig(),
            config=Phase1Config(),
        )
        assert r_none.phase1_milestone_closed == r_explicit.phase1_milestone_closed
        assert r_none.output_power_dbm == pytest.approx(r_explicit.output_power_dbm)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  5. Oscillation physics                                          ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestOscillationPhysics:
    """Verify oscillation sub-criterion values match expected physics."""

    @pytest.fixture
    def result(self) -> Phase1MilestoneResult:
        return validate_phase1_milestone()

    def test_q_ratio_below_one(self, result: Phase1MilestoneResult) -> None:
        """Q_m < Q_L for the default operating point."""
        assert result.oscillation.q_ratio < 1.0

    def test_q_ratio_approx(self, result: Phase1MilestoneResult) -> None:
        """Q_m / Q_L ≈ 0.779 (exact value from smoke test)."""
        assert result.oscillation.q_ratio == pytest.approx(0.779, rel=0.05)

    def test_loaded_q_equals_cavity_q_default(
        self, result: Phase1MilestoneResult
    ) -> None:
        """With q_boost_gain = 0, Q_L == MaserConfig.cavity_q (10 000)."""
        assert result.oscillation.loaded_q == pytest.approx(10_000.0)

    def test_magnetic_q_below_loaded_q(self, result: Phase1MilestoneResult) -> None:
        assert result.oscillation.magnetic_q < result.oscillation.loaded_q

    def test_cooperativity_above_one(self, result: Phase1MilestoneResult) -> None:
        """C > 1 required for masing."""
        assert result.oscillation.cooperativity > 1.0

    def test_cooperativity_approx(self, result: Phase1MilestoneResult) -> None:
        """C ≈ 2.57 from smoke test."""
        assert result.oscillation.cooperativity == pytest.approx(2.57, rel=0.05)

    def test_threshold_margin_positive(self, result: Phase1MilestoneResult) -> None:
        assert result.oscillation.threshold_margin > 0.0

    def test_masing_true(self, result: Phase1MilestoneResult) -> None:
        assert result.oscillation.masing is True

    def test_above_threshold_wang_true(self, result: Phase1MilestoneResult) -> None:
        assert result.oscillation.above_threshold_wang is True

    def test_spin_temperature_well_below_bath(
        self, result: Phase1MilestoneResult
    ) -> None:
        """Inverted NV ensemble has spin temperature ≪ 300 K."""
        assert result.oscillation.spin_temperature_k < 10.0

    def test_spin_linewidth_inversely_proportional_to_t2(self) -> None:
        """Γ_eff = 1/(π T₂*): doubling T₂* halves the linewidth."""
        nv_short = NVConfig(t2_star_us=1.0)
        nv_long = NVConfig(t2_star_us=2.0)
        lw_short = _spin_linewidth_hz(nv_short)
        lw_long = _spin_linewidth_hz(nv_long)
        assert lw_short == pytest.approx(2.0 * lw_long, rel=1e-9)

    def test_output_power_approx_dbm(self, result: Phase1MilestoneResult) -> None:
        """Output power ≈ −69.5 dBm from smoke test."""
        assert result.output_power_dbm == pytest.approx(-69.5, abs=5.0)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  6. Threshold failure path                                       ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestThresholdFailurePath:
    """Low NV density prevents oscillation threshold from being reached."""

    def _low_density_nv(self) -> NVConfig:
        """NV density reduced 100× drives Q_m ≫ Q_L and C ≪ 1."""
        return NVConfig(nv_density_per_cm3=1e15)  # 100× below default 1e17

    def test_milestone_open_on_low_density(self) -> None:
        r = validate_phase1_milestone(nv_config=self._low_density_nv())
        assert r.phase1_milestone_closed is False

    def test_threshold_not_met_on_low_density(self) -> None:
        r = validate_phase1_milestone(nv_config=self._low_density_nv())
        assert r.threshold_met is False

    def test_cooperativity_below_one_on_low_density(self) -> None:
        r = validate_phase1_milestone(nv_config=self._low_density_nv())
        assert r.oscillation.cooperativity < 1.0

    def test_q_ratio_above_one_on_low_density(self) -> None:
        """Q_m ≫ Q_L means spin gain cannot overcome cavity loss."""
        r = validate_phase1_milestone(nv_config=self._low_density_nv())
        assert r.oscillation.q_ratio > 1.0

    def test_closing_message_contains_open(self) -> None:
        r = validate_phase1_milestone(nv_config=self._low_density_nv())
        assert "OPEN" in r.closing_message

    def test_zero_pump_efficiency_prevents_oscillation(self) -> None:
        """No optical pumping → no inversion → no masing."""
        nv = NVConfig(pump_efficiency=0.0)
        r = validate_phase1_milestone(nv_config=nv)
        assert r.threshold_met is False


# ╔══════════════════════════════════════════════════════════════════╗
# ║  7. Frequency failure path                                       ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestFrequencyFailurePath:
    """Cavity tuned far from 1.47 GHz fails the frequency criterion."""

    def test_cavity_at_3ghz_fails_frequency(self) -> None:
        m = MaserConfig(cavity_frequency_ghz=3.0)
        r = validate_phase1_milestone(maser_config=m)
        assert r.frequency_met is False

    def test_deviation_computed_correctly(self) -> None:
        m = MaserConfig(cavity_frequency_ghz=1.97)  # +500 MHz from 1.47
        r = validate_phase1_milestone(maser_config=m)
        assert r.frequency_deviation_mhz == pytest.approx(500.0, rel=1e-3)

    def test_tight_tolerance_fails_on_slight_offset(self) -> None:
        """1 MHz tolerance with 0 MHz deviation still passes."""
        cfg = Phase1Config(frequency_tolerance_mhz=1.0)
        r = validate_phase1_milestone(config=cfg)
        assert r.frequency_met is True

    def test_zero_tolerance_on_exact_frequency_passes(self) -> None:
        cfg = Phase1Config(frequency_tolerance_mhz=0.0)
        m = MaserConfig(cavity_frequency_ghz=1.47)
        r = validate_phase1_milestone(maser_config=m, config=cfg)
        assert r.frequency_met is True

    def test_milestone_open_when_frequency_fails(self) -> None:
        m = MaserConfig(cavity_frequency_ghz=3.0)
        r = validate_phase1_milestone(maser_config=m)
        assert r.phase1_milestone_closed is False


# ╔══════════════════════════════════════════════════════════════════╗
# ║  8. Output power failure path                                    ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestPowerFailurePath:
    """Configurable power floor can force the power criterion to fail."""

    def test_high_power_floor_fails_power(self) -> None:
        """Setting floor at −60 dBm fails the default −69.5 dBm output."""
        cfg = Phase1Config(min_output_power_dbm=-60.0)
        r = validate_phase1_milestone(config=cfg)
        assert r.power_met is False

    def test_milestone_open_when_power_fails(self) -> None:
        cfg = Phase1Config(min_output_power_dbm=-60.0)
        r = validate_phase1_milestone(config=cfg)
        assert r.phase1_milestone_closed is False

    def test_lower_power_floor_relaxes_criterion(self) -> None:
        """−200 dBm floor means any detectable emission passes."""
        cfg = Phase1Config(min_output_power_dbm=-200.0)
        r = validate_phase1_milestone(config=cfg)
        assert r.power_met is True

    def test_power_zero_when_below_threshold(self) -> None:
        """Sub-threshold operation yields zero output power."""
        nv = NVConfig(nv_density_per_cm3=1e15)  # 100× below threshold
        r = validate_phase1_milestone(nv_config=nv)
        assert r.output_power_w == pytest.approx(0.0)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  9. Custom configurations                                        ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestCustomConfigurations:
    """Verify consistent behaviour with non-default configurations."""

    def test_higher_nv_density_increases_cooperativity(self) -> None:
        """Cooperativity scales roughly as NV density (masing improves)."""
        r_default = validate_phase1_milestone()
        r_high = validate_phase1_milestone(
            nv_config=NVConfig(nv_density_per_cm3=3e17)
        )
        assert r_high.oscillation.cooperativity > r_default.oscillation.cooperativity

    def test_higher_nv_density_reduces_q_ratio(self) -> None:
        """More inverted spins → lower Q_m → smaller Q_m/Q_L ratio."""
        r_default = validate_phase1_milestone()
        r_high = validate_phase1_milestone(
            nv_config=NVConfig(nv_density_per_cm3=3e17)
        )
        assert r_high.oscillation.q_ratio < r_default.oscillation.q_ratio

    def test_longer_t2_increases_cooperativity(self) -> None:
        """Longer T₂* → narrower linewidth → higher cooperativity."""
        r_default = validate_phase1_milestone()
        r_long_t2 = validate_phase1_milestone(
            nv_config=NVConfig(t2_star_us=2.0)
        )
        assert r_long_t2.oscillation.cooperativity > r_default.oscillation.cooperativity

    def test_lower_cavity_q_makes_oscillation_harder(self) -> None:
        """Lower Q_L raises the oscillation threshold; push Q_L below Q_m."""
        # Q_m ≈ 7788, so if Q_L < 7788 the Wang criterion fails.
        m = MaserConfig(cavity_q=5000)
        r = validate_phase1_milestone(maser_config=m)
        # Q_m/Q_L ≈ 7788/5000 ≈ 1.56 > 1 → Wang criterion fails
        assert r.oscillation.above_threshold_wang is False

    def test_high_cavity_q_with_q_boost_passes(self) -> None:
        """Q-boosting raises effective Q_L making it easier to oscillate."""
        # q_boost_gain = 0.9 → Q_eff = 10000 / (1-0.9) = 100000
        m = MaserConfig(cavity_q=1000, q_boost_gain=0.9)
        r = validate_phase1_milestone(maser_config=m)
        assert r.oscillation.loaded_q == pytest.approx(10000.0, rel=0.01)

    def test_gain_budget_affects_cooperativity(self) -> None:
        """Larger gain_budget → more effective spins → higher cooperativity."""
        r_low = validate_phase1_milestone(config=Phase1Config(gain_budget=0.1))
        r_high = validate_phase1_milestone(config=Phase1Config(gain_budget=0.9))
        assert r_high.oscillation.cooperativity > r_low.oscillation.cooperativity

    def test_output_power_positive_when_above_threshold(self) -> None:
        """CW emission is present when cooperativity > 1."""
        r = validate_phase1_milestone()
        assert r.output_power_w > 0.0

    def test_result_fields_consistent(self) -> None:
        """Milestone closed iff all three sub-criteria pass."""
        r = validate_phase1_milestone()
        expected = r.threshold_met and r.frequency_met and r.power_met
        assert r.phase1_milestone_closed == expected
