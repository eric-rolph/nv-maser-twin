"""Tests for the handheld probe integration module (physics/probe.py).

Covers:
- ProbeConfig dataclass
- compute_stray_field_rms()
- compute_probe_performance()
- HandheldProbe class
- All four parametric sweep functions
- Physics correctness (depth resolution formula, dipole stray-field scaling)
- Integration with snr_calculator
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nv_maser.config import SingleSidedMagnetConfig, SurfaceCoilConfig
from nv_maser.physics.depth_profile import TissueLayer
from nv_maser.physics.probe import (
    HandheldProbe,
    ProbeConfig,
    ProbePerformanceReport,
    compute_probe_performance,
    compute_stray_field_rms,
    sweep_depth_resolution_vs_bandwidth,
    sweep_lateral_resolution_vs_n_lines,
    sweep_snr_vs_averages,
    sweep_snr_vs_depth,
    sweep_stray_field_vs_separation,
)
from nv_maser.physics.single_sided_magnet import SingleSidedMagnet
from nv_maser.physics.snr_calculator import compute_snr_budget
from nv_maser.physics.surface_coil import SurfaceCoil

_MUSCLE = TissueLayer("muscle", thickness_mm=20.0, t1_ms=600, t2_ms=35)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ProbeConfig                                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestProbeConfig:
    def test_default_construction(self) -> None:
        """ProbeConfig() produces a valid config with sensible defaults."""
        cfg = ProbeConfig()
        assert cfg.pulse_sequence == "spin_echo"
        assert cfg.tr_ms > 0
        assert cfg.te_ms > 0
        assert cfg.te_ms < cfg.tr_ms
        assert cfg.bandwidth_hz > 0
        assert cfg.n_averages >= 1
        assert cfg.n_phase_lines > 0
        assert cfg.fov_m > 0
        assert cfg.target_depth_mm > 0
        assert cfg.maser_separation_mm > 0
        assert cfg.shield_attenuation_db >= 0

    def test_custom_pulse_sequence(self) -> None:
        """ProbeConfig accepts custom pulse_sequence."""
        cfg = ProbeConfig(pulse_sequence="gre", te_ms=5.0)
        assert cfg.pulse_sequence == "gre"

    def test_subsystem_configs_are_set(self) -> None:
        """Magnet and coil configs default to SingleSidedMagnetConfig and SurfaceCoilConfig."""
        cfg = ProbeConfig()
        assert isinstance(cfg.magnet, SingleSidedMagnetConfig)
        assert isinstance(cfg.coil, SurfaceCoilConfig)

    def test_frozen(self) -> None:
        """ProbeConfig is a frozen dataclass — mutation raises."""
        cfg = ProbeConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.tr_ms = 999.0  # type: ignore[misc]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  compute_stray_field_rms                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestComputeStrayFieldRms:
    @pytest.fixture
    def magnet(self) -> SingleSidedMagnet:
        return SingleSidedMagnet(SingleSidedMagnetConfig())

    def test_returns_finite_positive(self, magnet: SingleSidedMagnet) -> None:
        """Stray field RMS is a positive finite float."""
        rms = compute_stray_field_rms(magnet, maser_separation_mm=50.0)
        assert math.isfinite(rms)
        assert rms > 0.0

    def test_returns_mT_units_reasonable(self, magnet: SingleSidedMagnet) -> None:
        """At 50 mm with no shielding, stray field should be in the mT range (not µT or T)."""
        rms = compute_stray_field_rms(magnet, maser_separation_mm=50.0)
        # Typical dipole field: expect between 0.1 mT and 1000 mT at 50 mm
        assert 0.1 < rms < 1000.0

    def test_decreases_with_separation(self, magnet: SingleSidedMagnet) -> None:
        """Stray field must decrease monotonically as separation increases."""
        seps = [20.0, 40.0, 60.0, 80.0, 100.0]
        rms_vals = [compute_stray_field_rms(magnet, maser_separation_mm=s) for s in seps]
        for a, b in zip(rms_vals, rms_vals[1:]):
            assert a > b, f"Expected monotone decrease: {rms_vals}"

    def test_shielding_reduces_field(self, magnet: SingleSidedMagnet) -> None:
        """Shield attenuation must reduce the stray field."""
        rms_bare = compute_stray_field_rms(
            magnet, maser_separation_mm=50.0, shield_attenuation_db=0.0
        )
        rms_shielded = compute_stray_field_rms(
            magnet, maser_separation_mm=50.0, shield_attenuation_db=50.0
        )
        assert rms_shielded < rms_bare

    def test_50dB_shielding_factor(self, magnet: SingleSidedMagnet) -> None:
        """50 dB ≈ factor 316 in field amplitude; ratio between bare and shielded ≈ 316."""
        rms_bare = compute_stray_field_rms(
            magnet, maser_separation_mm=50.0, shield_attenuation_db=0.0
        )
        rms_50db = compute_stray_field_rms(
            magnet, maser_separation_mm=50.0, shield_attenuation_db=50.0
        )
        ratio = rms_bare / rms_50db
        assert 200 < ratio < 500, f"Expected ~316×, got {ratio:.1f}×"

    def test_grid_size_kwarg_accepted(self, magnet: SingleSidedMagnet) -> None:
        """grid_size kwarg is accepted and produces a finite result."""
        rms = compute_stray_field_rms(
            magnet, maser_separation_mm=60.0, grid_size=8
        )
        assert math.isfinite(rms)
        assert rms > 0.0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  compute_probe_performance                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestComputeProbePerformance:
    @pytest.fixture
    def report(self) -> ProbePerformanceReport:
        return compute_probe_performance(ProbeConfig(), tissue=_MUSCLE)

    def test_returns_report_type(self, report: ProbePerformanceReport) -> None:
        assert isinstance(report, ProbePerformanceReport)

    def test_sweet_spot_finite(self, report: ProbePerformanceReport) -> None:
        """Sweet-spot depth and B₀ are finite."""
        assert math.isfinite(report.sweet_spot_depth_mm)
        assert math.isfinite(report.sweet_spot_b0_tesla)

    def test_larmor_frequency_positive(self, report: ProbePerformanceReport) -> None:
        assert report.sweet_spot_larmor_mhz > 0.0

    def test_depth_resolution_positive(self, report: ProbePerformanceReport) -> None:
        assert report.depth_resolution_mm > 0.0

    def test_lateral_resolution_positive(self, report: ProbePerformanceReport) -> None:
        assert report.lateral_resolution_mm > 0.0

    def test_scan_time_positive(self, report: ProbePerformanceReport) -> None:
        assert report.scan_time_s > 0.0

    def test_snr_finite(self, report: ProbePerformanceReport) -> None:
        assert math.isfinite(report.snr_at_target_db)

    def test_stray_field_finite_positive(self, report: ProbePerformanceReport) -> None:
        assert math.isfinite(report.stray_field_rms_mt)
        assert report.stray_field_rms_mt > 0.0

    def test_defaults_accepted(self) -> None:
        """compute_probe_performance() with no args uses defaults and doesn't raise."""
        report = compute_probe_performance()
        assert isinstance(report, ProbePerformanceReport)

    def test_scan_time_formula(self) -> None:
        """scan_time = n_phase_lines × n_averages × TR."""
        cfg = ProbeConfig(n_phase_lines=32, n_averages=8, tr_ms=200.0)
        report = compute_probe_performance(cfg, tissue=_MUSCLE)
        expected_s = 32 * 8 * 200e-3
        assert report.scan_time_s == pytest.approx(expected_s, rel=1e-6)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  HandheldProbe class                                             ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestHandheldProbeClass:
    def test_default_construction(self) -> None:
        """HandheldProbe() constructs with default ProbeConfig."""
        probe = HandheldProbe()
        assert isinstance(probe.config, ProbeConfig)

    def test_custom_config(self) -> None:
        cfg = ProbeConfig(target_depth_mm=15.0)
        probe = HandheldProbe(cfg)
        assert probe.config.target_depth_mm == 15.0

    def test_magnet_property(self) -> None:
        probe = HandheldProbe()
        assert isinstance(probe.magnet, SingleSidedMagnet)

    def test_coil_property(self) -> None:
        probe = HandheldProbe()
        assert isinstance(probe.coil, SurfaceCoil)

    def test_performance_report(self) -> None:
        probe = HandheldProbe()
        report = probe.performance_report(tissue=_MUSCLE)
        assert isinstance(report, ProbePerformanceReport)

    def test_depth_scan_returns_depth_profile(self) -> None:
        from nv_maser.physics.depth_profile import DepthProfile
        probe = HandheldProbe()
        profile = probe.depth_scan()
        assert isinstance(profile, DepthProfile)

    def test_stray_field_rms_on_maser_positive(self) -> None:
        probe = HandheldProbe()
        rms = probe.stray_field_rms_on_maser()
        assert rms > 0.0
        assert math.isfinite(rms)

    def test_stray_field_rms_override_separation(self) -> None:
        """Explicitly passing a separation overrides the config value."""
        probe = HandheldProbe(ProbeConfig(maser_separation_mm=50.0))
        rms_far = probe.stray_field_rms_on_maser(maser_separation_mm=100.0)
        rms_near = probe.stray_field_rms_on_maser(maser_separation_mm=40.0)
        assert rms_near > rms_far


# ╔══════════════════════════════════════════════════════════════════╗
# ║  sweep_snr_vs_depth                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestSweepSnrVsDepth:
    def test_output_length(self) -> None:
        depths = np.linspace(5.0, 25.0, 8)
        snrs_db = sweep_snr_vs_depth(depths_mm=depths, tissue=_MUSCLE)
        assert snrs_db.shape == (8,)

    def test_all_finite(self) -> None:
        snrs_db = sweep_snr_vs_depth(tissue=_MUSCLE)
        assert np.all(np.isfinite(snrs_db))

    def test_generally_decreases_with_depth(self) -> None:
        """SNR should generally decrease as depth increases (physics constraint)."""
        depths = np.linspace(5.0, 30.0, 6)
        snrs_db = sweep_snr_vs_depth(depths_mm=depths, tissue=_MUSCLE)
        # Allow one non-monotone point, but total trend must be decreasing
        total_change = snrs_db[-1] - snrs_db[0]
        assert total_change < 0, f"SNR should decrease overall, got Δ={total_change:.1f} dB"

    def test_defaults_run(self) -> None:
        """sweep_snr_vs_depth() with default args completes without error."""
        snrs_db = sweep_snr_vs_depth()
        assert len(snrs_db) > 0

    def test_gre_sequence_accepted(self) -> None:
        cfg = ProbeConfig(pulse_sequence="gre", te_ms=5.0)
        snrs_db = sweep_snr_vs_depth(probe_config=cfg, tissue=_MUSCLE)
        assert np.all(np.isfinite(snrs_db))


# ╔══════════════════════════════════════════════════════════════════╗
# ║  sweep_snr_vs_averages                                           ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestSweepSnrVsAverages:
    def test_output_length(self) -> None:
        n_arr = np.array([1, 4, 16, 64], dtype=int)
        snrs_db = sweep_snr_vs_averages(n_averages_array=n_arr, tissue=_MUSCLE)
        assert snrs_db.shape == (4,)

    def test_all_finite(self) -> None:
        snrs_db = sweep_snr_vs_averages(tissue=_MUSCLE)
        assert np.all(np.isfinite(snrs_db))

    def test_monotone_increasing(self) -> None:
        """More averages → higher SNR (√NEX scaling)."""
        n_arr = np.array([1, 4, 16, 64, 256], dtype=int)
        snrs_db = sweep_snr_vs_averages(n_averages_array=n_arr, tissue=_MUSCLE)
        for a, b in zip(snrs_db, snrs_db[1:]):
            assert a < b, f"SNR must increase with n_averages: {snrs_db}"

    def test_sqrt_nex_scaling(self) -> None:
        """Doubling NEX should increase SNR by ≈ 3 dB (factor √2)."""
        n_arr = np.array([4, 16], dtype=int)
        snrs_db = sweep_snr_vs_averages(n_averages_array=n_arr, tissue=_MUSCLE)
        # 16 = 4 × 4, so expect SNR gain ≈ 20*log10(2) ≈ 6 dB
        gain = snrs_db[1] - snrs_db[0]
        assert 4.0 < gain < 8.0, f"Expected ~6 dB for 4× averages, got {gain:.2f} dB"

    def test_defaults_run(self) -> None:
        snrs_db = sweep_snr_vs_averages()
        assert len(snrs_db) > 0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  sweep_lateral_resolution_vs_n_lines                             ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestSweepLateralResolution:
    def test_output_length(self) -> None:
        ns = np.array([16, 32, 64, 128], dtype=int)
        res = sweep_lateral_resolution_vs_n_lines(n_lines_values=ns)
        assert res.shape == (4,)

    def test_all_positive(self) -> None:
        res = sweep_lateral_resolution_vs_n_lines()
        assert np.all(res > 0.0)

    def test_monotone_decreasing(self) -> None:
        """More phase lines → finer (smaller) lateral resolution."""
        ns = np.array([16, 32, 64, 128, 256], dtype=int)
        res = sweep_lateral_resolution_vs_n_lines(n_lines_values=ns)
        for a, b in zip(res, res[1:]):
            assert a > b, f"Resolution must decrease with n_lines: {res}"

    def test_resolution_formula(self) -> None:
        """Lateral resolution = FOV / n_lines (in mm)."""
        fov_m = 0.08
        n_lines = np.array([64], dtype=int)
        res = sweep_lateral_resolution_vs_n_lines(n_lines_values=n_lines, fov_m=fov_m)
        expected_mm = fov_m / 64 * 1e3
        assert res[0] == pytest.approx(expected_mm, rel=1e-4)

    def test_defaults_run(self) -> None:
        res = sweep_lateral_resolution_vs_n_lines()
        assert len(res) > 0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  sweep_stray_field_vs_separation                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestSweepStrayFieldVsSeparation:
    def test_output_length(self) -> None:
        seps = np.linspace(20.0, 80.0, 7)
        rms = sweep_stray_field_vs_separation(separations_mm=seps)
        assert rms.shape == (7,)

    def test_all_positive(self) -> None:
        rms = sweep_stray_field_vs_separation()
        assert np.all(rms > 0.0)

    def test_monotone_decreasing(self) -> None:
        """Stray field decreases as probe–maser separation increases."""
        seps = np.linspace(30.0, 100.0, 8)
        rms = sweep_stray_field_vs_separation(separations_mm=seps)
        for a, b in zip(rms, rms[1:]):
            assert a > b, f"Stray field must decrease with separation: {rms}"

    def test_shielding_reduces_sweep(self) -> None:
        """50 dB shielding sweeps produce consistently lower values."""
        seps = np.linspace(30.0, 80.0, 5)
        rms_bare = sweep_stray_field_vs_separation(
            separations_mm=seps, shield_attenuation_db=0.0
        )
        rms_shielded = sweep_stray_field_vs_separation(
            separations_mm=seps, shield_attenuation_db=50.0
        )
        assert np.all(rms_shielded < rms_bare)

    def test_defaults_run(self) -> None:
        rms = sweep_stray_field_vs_separation()
        assert len(rms) > 0


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Physics correctness                                             ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestDepthResolutionFormula:
    """Verify δz = BW / (γ |∂B/∂z|) formula implementation."""

    def test_resolution_increases_with_bandwidth(self) -> None:
        """Wider bandwidth → coarser depth resolution."""
        bws = np.array([1000.0, 10000.0, 50000.0])
        res = sweep_depth_resolution_vs_bandwidth(bandwidths_hz=bws)
        for a, b in zip(res, res[1:]):
            assert a < b, f"Resolution must increase (get worse) with BW: {res}"

    def test_resolution_finite(self) -> None:
        res = sweep_depth_resolution_vs_bandwidth()
        assert np.all(np.isfinite(res))
        assert np.all(res > 0.0)

    def test_defaults_run(self) -> None:
        res = sweep_depth_resolution_vs_bandwidth()
        assert len(res) > 0


class TestStrayFieldPhysics:
    """Verify stray-field dipole physics is correctly implemented."""

    @pytest.fixture
    def magnet(self) -> SingleSidedMagnet:
        return SingleSidedMagnet(SingleSidedMagnetConfig())

    def test_1_over_r3_scaling(self, magnet: SingleSidedMagnet) -> None:
        """Dipole field falls as 1/r³; doubling distance reduces field ~8×."""
        rms_near = compute_stray_field_rms(magnet, maser_separation_mm=40.0)
        rms_far = compute_stray_field_rms(magnet, maser_separation_mm=80.0)
        # Exact factor: (80/40)³ = 8; allow ±30% for grid averaging effects
        ratio = rms_near / rms_far
        assert 4.0 < ratio < 12.0, f"Expected ~8× for 2× separation, got {ratio:.2f}×"

    def test_shielding_is_linear_in_field(self, magnet: SingleSidedMagnet) -> None:
        """20 dB shielding gives exactly 10× field reduction."""
        rms_bare = compute_stray_field_rms(
            magnet, maser_separation_mm=60.0, shield_attenuation_db=0.0
        )
        rms_20db = compute_stray_field_rms(
            magnet, maser_separation_mm=60.0, shield_attenuation_db=20.0
        )
        ratio = rms_bare / rms_20db
        assert ratio == pytest.approx(10.0, rel=0.01)

    def test_larger_magnet_gives_larger_field(self) -> None:
        """A larger magnet (bigger rings) produces a stronger stray field."""
        cfg_small = SingleSidedMagnetConfig(
            ring_inner_radii_mm=[5.0],
            ring_outer_radii_mm=[10.0],
            ring_heights_mm=[10.0],
            ring_polarities=[1.0],
            num_rings=1,
        )
        cfg_large = SingleSidedMagnetConfig(
            ring_inner_radii_mm=[5.0],
            ring_outer_radii_mm=[30.0],
            ring_heights_mm=[40.0],
            ring_polarities=[1.0],
            num_rings=1,
        )
        magnet_small = SingleSidedMagnet(cfg_small)
        magnet_large = SingleSidedMagnet(cfg_large)
        rms_small = compute_stray_field_rms(magnet_small, maser_separation_mm=80.0)
        rms_large = compute_stray_field_rms(magnet_large, maser_separation_mm=80.0)
        assert rms_large > rms_small


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Integration: probe vs. snr_calculator                           ║
# ╚══════════════════════════════════════════════════════════════════╝

class TestProbeIntegration:
    """End-to-end probe performance agrees with direct snr_calculator calls."""

    def test_snr_agrees_with_direct_budget(self) -> None:
        """ProbePerformanceReport.snr_at_target_db should be consistent with
        compute_snr_budget called with the same parameters."""
        cfg = ProbeConfig(
            target_depth_mm=20.0,
            tr_ms=500.0,
            te_ms=10.0,
            bandwidth_hz=10_000.0,
            n_averages=4,
        )
        report = compute_probe_performance(cfg, tissue=_MUSCLE)

        magnet = SingleSidedMagnet(cfg.magnet)
        coil = SurfaceCoil(cfg.coil)
        budget = compute_snr_budget(
            depth_mm=cfg.target_depth_mm,
            voxel_size_mm=3.0,
            coil=coil,
            magnet=magnet,
            tissue=_MUSCLE,
            tr_ms=cfg.tr_ms,
            te_ms=cfg.te_ms,
            sequence=cfg.pulse_sequence,
            bandwidth_hz=cfg.bandwidth_hz,
            n_averages=cfg.n_averages,
        )
        expected_db = (
            20.0 * math.log10(budget.snr_after_averaging)
            if budget.snr_after_averaging > 0
            else float("-inf")
        )
        assert report.snr_at_target_db == pytest.approx(expected_db, rel=1e-6)

    def test_stray_field_agrees_with_direct_call(self) -> None:
        """HandheldProbe.stray_field_rms_on_maser matches compute_stray_field_rms."""
        cfg = ProbeConfig(
            maser_separation_mm=60.0,
            shield_attenuation_db=30.0,
        )
        probe = HandheldProbe(cfg)
        from_probe = probe.stray_field_rms_on_maser()
        direct = compute_stray_field_rms(
            probe.magnet,
            maser_separation_mm=60.0,
            shield_attenuation_db=30.0,
        )
        assert from_probe == pytest.approx(direct, rel=1e-9)

    def test_scan_time_consistent_with_config(self) -> None:
        cfg = ProbeConfig(n_phase_lines=64, n_averages=16, tr_ms=300.0)
        probe = HandheldProbe(cfg)
        rpt = probe.performance_report(tissue=_MUSCLE)
        expected = 64 * 16 * 0.3  # TR in seconds
        assert rpt.scan_time_s == pytest.approx(expected, rel=1e-6)

    def test_larmor_frequency_from_b0(self) -> None:
        """Larmor MHz ≈ B₀[T] × 42.577 MHz/T."""
        cfg = ProbeConfig()
        rpt = compute_probe_performance(cfg, tissue=_MUSCLE)
        gamma = 42.577  # MHz/T
        expected_mhz = abs(rpt.sweet_spot_b0_tesla) * gamma
        assert rpt.sweet_spot_larmor_mhz == pytest.approx(expected_mhz, rel=1e-4)

    def test_depth_scan_signal_positive(self) -> None:
        """depth_scan() produces a DepthProfile whose signal has positive values."""
        probe = HandheldProbe()
        profile = probe.depth_scan()
        assert np.any(profile.signal > 0.0)

    def test_performance_report_snr_budget_type(self) -> None:
        """report.snr_budget is an SNRBudget instance."""
        from nv_maser.physics.snr_calculator import SNRBudget
        probe = HandheldProbe()
        rpt = probe.performance_report(tissue=_MUSCLE)
        assert isinstance(rpt.snr_budget, SNRBudget)
