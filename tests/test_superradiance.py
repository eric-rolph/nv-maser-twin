"""
Tests for superradiance.py — analytic Dicke SR vs CW masing regime model.

Coverage:
  TestComputeCollectiveCoupling  — algebraic scaling, edge cases
  TestDetermineRegime            — all three regimes, boundary conditions
  TestSuperradiantPulseDuration  — formula check, error handling
  TestSuperradiantDelay          — log-N scaling, edge cases
  TestSuperradiantPeak           — N_eff/4 photons, energy, power formula
  TestComputeSuperradiance       — integration with real configs
  TestPhysicsScaling             — N and Q scaling laws
  TestCrossValidation            — self-consistency relations
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from nv_maser.config import CavityConfig, MaserConfig, NVConfig
from nv_maser.physics.cavity import (
    compute_cavity_properties,
    compute_maser_threshold,
    compute_n_effective,
)
from nv_maser.physics.superradiance import (
    BELOW_THRESHOLD,
    MASING,
    SUPERRADIANT,
    SuperradianceResult,
    _HBAR,
    compute_collective_coupling,
    compute_superradiance,
    compute_superradiant_delay,
    compute_superradiant_peak,
    compute_superradiant_pulse_duration,
    determine_regime,
)


# ── Helpers ────────────────────────────────────────────────────────


def _default_setup(nv_density: float | None = None, t2_star_us: float | None = None):
    """Return (cavity_props, threshold_result, nv_config, maser_config) defaults."""
    kwargs: dict = {}
    if nv_density is not None:
        kwargs["nv_density_per_cm3"] = nv_density
    if t2_star_us is not None:
        kwargs["t2_star_us"] = t2_star_us
    nv = NVConfig(**kwargs)
    ms = MaserConfig()
    cc = CavityConfig()
    cp = compute_cavity_properties(ms, cc)
    n_eff = compute_n_effective(nv, cc, gain_budget=1.0)
    tr = compute_maser_threshold(cp, n_eff, spin_linewidth_hz=1e6)
    return cp, tr, nv, ms


# ── TestComputeCollectiveCoupling ──────────────────────────────────


class TestComputeCollectiveCoupling:
    def test_exact_value(self):
        # g_eff = g0 × √N, e.g. g0=50, N=16 → 50×4 = 200
        assert compute_collective_coupling(50.0, 16.0) == pytest.approx(200.0, rel=1e-9)

    def test_scales_as_sqrt_n(self):
        # g_eff(4N) = 2 × g_eff(N) — hallmark of collective enhancement
        g0 = 100.0
        g_n = compute_collective_coupling(g0, 100.0)
        g_4n = compute_collective_coupling(g0, 400.0)
        assert g_4n == pytest.approx(2 * g_n, rel=1e-9)

    def test_zero_spins_returns_zero(self):
        assert compute_collective_coupling(100.0, 0.0) == 0.0

    def test_negative_spins_returns_zero(self):
        assert compute_collective_coupling(100.0, -1.0) == 0.0

    def test_single_spin(self):
        # N=1 → g_eff = g0
        g0 = 37.5
        assert compute_collective_coupling(g0, 1.0) == pytest.approx(g0, rel=1e-9)

    def test_large_n(self):
        g0 = 1.0
        n = 1e12
        assert compute_collective_coupling(g0, n) == pytest.approx(1e6, rel=1e-9)


# ── TestDetermineRegime ────────────────────────────────────────────


class TestDetermineRegime:
    def test_superradiant_regime(self):
        # g_eff=1 MHz > κ/2=50 kHz, C=5>1, τ_SR=159 ns < T2*=1 µs
        g_eff = 1e6
        kappa = 1e5
        t2_star = 1e-6
        assert determine_regime(g_eff, kappa, cooperativity=5.0, t2_star_s=t2_star) == SUPERRADIANT

    def test_masing_weak_coupling(self):
        # g_eff < κ/2 → bad-cavity / CW masing
        g_eff = 1e4       # 10 kHz
        kappa = 1e5       # 100 kHz → κ/2 = 50 kHz > g_eff
        t2_star = 1e-6
        assert determine_regime(g_eff, kappa, cooperativity=5.0, t2_star_s=t2_star) == MASING

    def test_below_threshold_when_coop_le_1(self):
        # C ≤ 1 → below threshold regardless of coupling
        g_eff = 1e6
        kappa = 1e3
        assert determine_regime(g_eff, kappa, cooperativity=1.0, t2_star_s=1e-6) == BELOW_THRESHOLD

    def test_below_threshold_strictly(self):
        assert determine_regime(1e6, 1e3, cooperativity=0.5, t2_star_s=1e-6) == BELOW_THRESHOLD

    def test_masing_when_dephasing_too_fast(self):
        # g_eff > κ/2, C > 1, but τ_SR > T₂* → burst can't complete coherently
        g_eff = 1e6       # τ_SR = 1/(2π×1e6) ≈ 159 ns
        kappa = 1e3       # κ/2 = 500 Hz << g_eff ✓
        t2_star = 1e-10   # 0.1 ns << 159 ns → coherence fails
        assert determine_regime(g_eff, kappa, cooperativity=100.0, t2_star_s=t2_star) == MASING

    def test_exactly_at_strong_coupling_boundary_is_masing(self):
        # g_eff == κ/2 exactly → criterion is strictly >, boundary → masing
        kappa = 1e5
        g_eff = kappa / 2.0
        assert determine_regime(g_eff, kappa, cooperativity=5.0, t2_star_s=1e-6) == MASING

    def test_just_above_boundary_is_superradiant(self):
        kappa = 1e5
        g_eff = kappa / 2.0 + 1.0   # just above κ/2
        t2_star = 1e-3               # very long T₂* → coherence easily satisfied
        assert determine_regime(g_eff, kappa, cooperativity=5.0, t2_star_s=t2_star) == SUPERRADIANT


# ── TestSuperradiantPulseDuration ─────────────────────────────────


class TestSuperradiantPulseDuration:
    def test_exact_formula(self):
        g_eff = 1e6
        expected = 1.0 / (2.0 * math.pi * 1e6)
        assert compute_superradiant_pulse_duration(g_eff) == pytest.approx(expected, rel=1e-10)

    def test_doubling_coupling_halves_duration(self):
        tau1 = compute_superradiant_pulse_duration(1e6)
        tau2 = compute_superradiant_pulse_duration(2e6)
        assert tau2 == pytest.approx(tau1 / 2.0, rel=1e-9)

    def test_zero_coupling_raises_value_error(self):
        with pytest.raises(ValueError):
            compute_superradiant_pulse_duration(0.0)

    def test_negative_coupling_raises_value_error(self):
        with pytest.raises(ValueError):
            compute_superradiant_pulse_duration(-1.0)

    def test_duration_positive(self):
        assert compute_superradiant_pulse_duration(500e3) > 0.0


# ── TestSuperradiantDelay ──────────────────────────────────────────


class TestSuperradiantDelay:
    def test_exact_formula(self):
        g_eff = 1e6
        n = 100.0
        tau_sr = compute_superradiant_pulse_duration(g_eff)
        expected = tau_sr * math.log(100.0) / 2.0
        assert compute_superradiant_delay(g_eff, n) == pytest.approx(expected, rel=1e-9)

    def test_scales_as_log_n(self):
        # t_D ∝ ln(N): doubling ln(N) doubles t_D
        # ln(1e12) / ln(1e6) = 2  → t_D(1e12) / t_D(1e6) = 2
        g_eff = 1e6
        t_d1 = compute_superradiant_delay(g_eff, 1e6)
        t_d2 = compute_superradiant_delay(g_eff, 1e12)
        assert t_d2 / t_d1 == pytest.approx(2.0, rel=1e-9)

    def test_positive_for_n_gt_1(self):
        assert compute_superradiant_delay(1e6, 100.0) > 0.0

    def test_zero_spins_returns_zero(self):
        assert compute_superradiant_delay(1e6, 0.0) == pytest.approx(0.0)

    def test_zero_coupling_returns_zero(self):
        assert compute_superradiant_delay(0.0, 1e6) == pytest.approx(0.0)

    def test_n_equals_1_gives_zero_delay(self):
        # ln(1) = 0
        assert compute_superradiant_delay(1e6, 1.0) == pytest.approx(0.0, abs=1e-30)


# ── TestSuperradiantPeak ───────────────────────────────────────────


class TestSuperradiantPeak:
    def _peak(self, n=1e6, kappa=1e5, f=1.47e9):
        return compute_superradiant_peak(1e6, n, kappa, f)

    def test_peak_photons_is_n_over_4(self):
        r = self._peak(n=4e6)
        assert r["peak_photons"] == pytest.approx(1e6, rel=1e-9)

    def test_peak_photons_quarter_n(self):
        n = 1e8
        r = self._peak(n=n)
        assert r["peak_photons"] == pytest.approx(n / 4.0, rel=1e-9)

    def test_peak_power_positive(self):
        assert self._peak()["peak_power_w"] > 0.0

    def test_pulse_energy_equals_n_hbar_omega(self):
        n = 1e6
        f = 1.47e9
        omega = 2 * math.pi * f
        r = self._peak(n=n, f=f)
        assert r["pulse_energy_j"] == pytest.approx(n * _HBAR * omega, rel=1e-9)

    def test_peak_power_formula(self):
        n = 1e6
        kappa = 1e5
        f = 1.47e9
        omega = 2.0 * math.pi * f
        kappa_rad = 2.0 * math.pi * kappa
        expected = _HBAR * omega * kappa_rad * (n / 4.0)
        r = self._peak(n=n, kappa=kappa, f=f)
        assert r["peak_power_w"] == pytest.approx(expected, rel=1e-9)

    def test_n_times_4_gives_proportional_power(self):
        r1 = self._peak(n=1e6)
        r4 = self._peak(n=4e6)
        assert r4["peak_photons"] == pytest.approx(4 * r1["peak_photons"], rel=1e-9)
        assert r4["peak_power_w"] == pytest.approx(4 * r1["peak_power_w"], rel=1e-9)


# ── TestComputeSuperradiance ───────────────────────────────────────


class TestComputeSuperradiance:
    def test_returns_correct_type(self):
        cp, tr, nv, ms = _default_setup()
        result = compute_superradiance(cp, tr, nv, ms)
        assert isinstance(result, SuperradianceResult)

    def test_regime_is_valid_string(self):
        cp, tr, nv, ms = _default_setup()
        result = compute_superradiance(cp, tr, nv, ms)
        assert result.regime in (BELOW_THRESHOLD, MASING, SUPERRADIANT)

    def test_is_superradiant_consistent_with_regime(self):
        cp, tr, nv, ms = _default_setup()
        result = compute_superradiance(cp, tr, nv, ms)
        assert result.is_superradiant == (result.regime == SUPERRADIANT)

    def test_cooperativity_matches_threshold(self):
        cp, tr, nv, ms = _default_setup()
        result = compute_superradiance(cp, tr, nv, ms)
        assert result.cooperativity == pytest.approx(tr.cooperativity, rel=1e-9)

    def test_collective_coupling_matches_threshold(self):
        cp, tr, nv, ms = _default_setup()
        result = compute_superradiance(cp, tr, nv, ms)
        assert result.collective_coupling_hz == pytest.approx(tr.ensemble_coupling_hz, rel=1e-9)

    def test_cavity_linewidth_matches_props(self):
        cp, tr, nv, ms = _default_setup()
        result = compute_superradiance(cp, tr, nv, ms)
        assert result.cavity_linewidth_hz == pytest.approx(cp.cavity_linewidth_hz, rel=1e-9)

    def test_threshold_coupling_is_half_kappa(self):
        cp, tr, nv, ms = _default_setup()
        result = compute_superradiance(cp, tr, nv, ms)
        assert result.superradiant_threshold_coupling_hz == pytest.approx(
            result.cavity_linewidth_hz / 2.0, rel=1e-9
        )

    def test_no_mb_result_gives_nan(self):
        cp, tr, nv, ms = _default_setup()
        result = compute_superradiance(cp, tr, nv, ms, mb_result=None)
        assert math.isnan(result.cw_power_w)
        assert math.isnan(result.power_enhancement)

    def test_with_mock_mb_result(self):
        cp, tr, nv, ms = _default_setup(nv_density=1e18)
        mock_mb = MagicMock()
        mock_mb.output_power_w = 1e-9
        result = compute_superradiance(cp, tr, nv, ms, mb_result=mock_mb)
        assert result.cw_power_w == pytest.approx(1e-9)
        assert not math.isnan(result.power_enhancement)
        assert result.power_enhancement > 0.0

    def test_high_density_is_superradiant(self):
        # nv_density=1e18 → large N_eff → g_eff >> κ/2
        cp, tr, nv, ms = _default_setup(nv_density=1e18, t2_star_us=10.0)
        result = compute_superradiance(cp, tr, nv, ms)
        assert result.is_superradiant

    def test_low_density_not_superradiant(self):
        # nv_density=1e10 → tiny N_eff → g_eff < κ/2
        cp, tr, nv, ms = _default_setup(nv_density=1e10)
        result = compute_superradiance(cp, tr, nv, ms)
        assert not result.is_superradiant

    def test_pulse_duration_positive(self):
        cp, tr, nv, ms = _default_setup()
        result = compute_superradiance(cp, tr, nv, ms)
        assert result.pulse_duration_s > 0.0

    def test_peak_photons_is_n_over_4(self):
        cp, tr, nv, ms = _default_setup()
        result = compute_superradiance(cp, tr, nv, ms)
        assert result.peak_photons == pytest.approx(tr.n_effective / 4.0, rel=1e-9)


# ── TestPhysicsScaling ─────────────────────────────────────────────


class TestPhysicsScaling:
    def test_higher_q_lowers_kappa_threshold(self):
        ms1 = MaserConfig(cavity_q=10_000)
        ms2 = MaserConfig(cavity_q=20_000)
        cc = CavityConfig()
        cp1 = compute_cavity_properties(ms1, cc)
        cp2 = compute_cavity_properties(ms2, cc)
        # κ = f/Q → Q×2 → κ/2
        assert cp2.cavity_linewidth_hz == pytest.approx(cp1.cavity_linewidth_hz / 2.0, rel=1e-9)
        # SR threshold = κ/2 → also halved
        nv = NVConfig(nv_density_per_cm3=1e10)
        n_eff = compute_n_effective(nv, cc, 1.0)
        tr1 = compute_maser_threshold(cp1, n_eff, 1e6)
        tr2 = compute_maser_threshold(cp2, n_eff, 1e6)
        r1 = compute_superradiance(cp1, tr1, nv, ms1)
        r2 = compute_superradiance(cp2, tr2, nv, ms2)
        assert r2.superradiant_threshold_coupling_hz == pytest.approx(
            r1.superradiant_threshold_coupling_hz / 2.0, rel=1e-9
        )

    def test_n_times_4_gives_peak_power_times_4(self):
        cc = CavityConfig()
        ms = MaserConfig()
        cp = compute_cavity_properties(ms, cc)
        nv1 = NVConfig(nv_density_per_cm3=1e17)
        nv4 = NVConfig(nv_density_per_cm3=4e17)
        tr1 = compute_maser_threshold(cp, compute_n_effective(nv1, cc, 1.0), 1e6)
        tr4 = compute_maser_threshold(cp, compute_n_effective(nv4, cc, 1.0), 1e6)
        r1 = compute_superradiance(cp, tr1, nv1, ms)
        r4 = compute_superradiance(cp, tr4, nv4, ms)
        # N×4 → peak_photons×4, peak_power×4
        assert r4.peak_photons == pytest.approx(4 * r1.peak_photons, rel=0.01)
        assert r4.peak_power_w == pytest.approx(4 * r1.peak_power_w, rel=0.01)

    def test_more_spins_shorter_sr_pulse(self):
        cc = CavityConfig()
        ms = MaserConfig()
        cp = compute_cavity_properties(ms, cc)
        nv1 = NVConfig(nv_density_per_cm3=1e17)
        nv4 = NVConfig(nv_density_per_cm3=4e17)
        tr1 = compute_maser_threshold(cp, compute_n_effective(nv1, cc, 1.0), 1e6)
        tr4 = compute_maser_threshold(cp, compute_n_effective(nv4, cc, 1.0), 1e6)
        r1 = compute_superradiance(cp, tr1, nv1, ms)
        r4 = compute_superradiance(cp, tr4, nv4, ms)
        # g_eff ~ √N → τ_SR ~ 1/√N; N×4 → τ_SR/2
        assert r4.pulse_duration_s < r1.pulse_duration_s

    def test_g_eff_scales_sqrt_n(self):
        cc = CavityConfig()
        ms = MaserConfig()
        cp = compute_cavity_properties(ms, cc)
        nv1 = NVConfig(nv_density_per_cm3=1e17)
        nv4 = NVConfig(nv_density_per_cm3=4e17)
        n1 = compute_n_effective(nv1, cc, 1.0)
        n4 = compute_n_effective(nv4, cc, 1.0)
        tr1 = compute_maser_threshold(cp, n1, 1e6)
        tr4 = compute_maser_threshold(cp, n4, 1e6)
        r1 = compute_superradiance(cp, tr1, nv1, ms)
        r4 = compute_superradiance(cp, tr4, nv4, ms)
        # N×4 → g_eff×2
        assert r4.collective_coupling_hz == pytest.approx(2 * r1.collective_coupling_hz, rel=0.01)

    def test_regime_boundary_transition(self):
        """Verify regime flips from MASING to SUPERRADIANT as N crosses κ²/(4g₀²)."""
        ms = MaserConfig()
        cc = CavityConfig()
        cp = compute_cavity_properties(ms, cc)
        kappa = cp.cavity_linewidth_hz
        g0 = cp.single_spin_coupling_hz
        # N such that g_eff = κ/2 exactly: N_boundary = (κ / (2 g₀))²
        n_boundary = (kappa / (2.0 * g0)) ** 2
        # Below boundary → masing (use tiny γ⊥ to ensure C > 1)
        tr_below = compute_maser_threshold(cp, n_boundary * 0.99, 1.0)
        r_below = compute_superradiance(cp, tr_below, NVConfig(t2_star_us=1000.0), ms)
        assert r_below.collective_coupling_hz < kappa / 2.0
        assert not r_below.is_superradiant
        # Above boundary → superradiant (long T₂* to satisfy coherence check)
        tr_above = compute_maser_threshold(cp, n_boundary * 1.01, 1.0)
        r_above = compute_superradiance(cp, tr_above, NVConfig(t2_star_us=1000.0), ms)
        assert r_above.collective_coupling_hz > kappa / 2.0
        assert r_above.is_superradiant


# ── TestCrossValidation ────────────────────────────────────────────


class TestCrossValidation:
    def test_energy_per_spin_equals_hbar_omega(self):
        cp, tr, nv, ms = _default_setup()
        result = compute_superradiance(cp, tr, nv, ms)
        omega = 2.0 * math.pi * ms.cavity_frequency_ghz * 1e9
        expected_per_spin = _HBAR * omega
        actual_per_spin = result.pulse_energy_j / result.n_effective
        assert actual_per_spin == pytest.approx(expected_per_spin, rel=1e-9)

    def test_collective_coupling_from_g0_sqrt_n(self):
        # g_eff = g0 × √N_eff exactly
        cp, tr, nv, ms = _default_setup()
        result = compute_superradiance(cp, tr, nv, ms)
        expected = result.single_spin_coupling_hz * math.sqrt(result.n_effective)
        assert result.collective_coupling_hz == pytest.approx(expected, rel=1e-9)

    def test_sr_pulse_faster_than_t2star_in_sr_regime(self):
        # Invariant: if regime is SUPERRADIANT then τ_SR < T₂* (by construction)
        cp, tr, nv, ms = _default_setup(nv_density=1e18, t2_star_us=10.0)
        result = compute_superradiance(cp, tr, nv, ms)
        if result.is_superradiant:
            t2_star_s = nv.t2_star_us * 1e-6
            assert result.pulse_duration_s < t2_star_s

    def test_pulse_duration_inverse_of_coupling(self):
        # τ_SR = 1/(2π g_eff) must hold
        cp, tr, nv, ms = _default_setup()
        result = compute_superradiance(cp, tr, nv, ms)
        expected_tau = 1.0 / (2.0 * math.pi * result.collective_coupling_hz)
        assert result.pulse_duration_s == pytest.approx(expected_tau, rel=1e-9)

    def test_n_effective_stored_correctly(self):
        cp, tr, nv, ms = _default_setup()
        result = compute_superradiance(cp, tr, nv, ms)
        assert result.n_effective == pytest.approx(tr.n_effective, rel=1e-9)

    def test_cavity_frequency_hz_stored(self):
        cp, tr, nv, ms = _default_setup()
        result = compute_superradiance(cp, tr, nv, ms)
        assert result.cavity_frequency_hz == pytest.approx(
            ms.cavity_frequency_ghz * 1e9, rel=1e-9
        )
