"""Tests for maser_gain_frequency — frequency-resolved gain curve.

Reference values from Wang et al., Advanced Science (2024), PMC11425272:
    G ≈ 14.5 dB, B ≈ 340 kHz at Q_m = 1589, Q_L = 1337,
    coupling_beta = 1.0 (critical coupling, Wang convention),
    f_c = 2.87 GHz.

These reference parameters are used throughout to anchor the numerical
tests to experiment.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nv_maser.config import CavityConfig, MaserConfig, NVConfig
from nv_maser.physics.amplifier import compute_maser_gain
from nv_maser.physics.maser_gain_frequency import (
    GainCurveResult,
    bandwidth_analytical,
    compute_bandwidth_3db,
    compute_gain_bandwidth_product,
    compute_gain_curve,
    compute_saturation_power,
    gain_curve_from_mb_result,
)

# ── Wang 2024 reference parameters ───────────────────────────────
_FC = 2.87e9          # Hz  (NV maser cavity frequency)
_QM = 1589.0          # magnetic Q (small-signal)
_QL = 1337.0          # loaded Q
_BETA = 1.0           # critical coupling (Wang convention)
_G0_DB = 14.5         # expected peak gain (dB)
_BW_HZ = 340e3        # expected 3 dB bandwidth


def _wang_result() -> GainCurveResult:
    """Gain curve at Wang 2024 reference parameters."""
    return compute_gain_curve(_FC, _QM, _QL, _BETA)


# ── Standard NV configs (matching previous test patterns) ────────

def _nv_cfg(**kw: float) -> NVConfig:
    return NVConfig(
        nv_density_per_cm3=1e17,
        pump_efficiency=0.5,
        orientation_fraction=0.25,
        t2_star_us=1.0,
        t1_ms=5.0,
        **kw,
    )


def _maser_cfg(**kw: float) -> MaserConfig:
    defaults: dict = dict(cavity_q=1337, cavity_frequency_ghz=2.87, coupling_beta=1.0)
    defaults.update(kw)
    return MaserConfig(**defaults)


def _cavity_cfg(**kw: float) -> CavityConfig:
    return CavityConfig(mode_volume_cm3=0.5, fill_factor=0.01, **kw)


# ═══════════════════════════════════════════════════════════════════
# 1. TestComputeGainCurve
# ═══════════════════════════════════════════════════════════════════

class TestComputeGainCurve:
    """Tests for compute_gain_curve()."""

    # ── Return type and shape ──────────────────────────────────────

    def test_returns_gain_curve_result(self) -> None:
        result = _wang_result()
        assert isinstance(result, GainCurveResult)

    def test_gain_array_shape_matches_default_n_points(self) -> None:
        result = compute_gain_curve(_FC, _QM, _QL, _BETA, n_points=501)
        assert result.gain_linear.shape == (501,)
        assert result.frequencies_hz.shape == (501,)

    def test_gain_array_shape_matches_provided_grid(self) -> None:
        grid = np.linspace(_FC - 1e6, _FC + 1e6, 200)
        result = compute_gain_curve(_FC, _QM, _QL, _BETA, freq_grid_hz=grid)
        assert result.gain_linear.shape == (200,)

    def test_gain_array_shape_default_1001_points(self) -> None:
        result = _wang_result()
        assert result.gain_linear.shape == (1001,)

    # ── Peak gain against Wang 2024 reference ─────────────────────

    def test_peak_gain_db_matches_wang2024_within_5pct(self) -> None:
        result = _wang_result()
        # 5% relative tolerance on the dB value
        assert abs(result.peak_gain_db - _G0_DB) / _G0_DB < 0.05, (
            f"peak_gain_db={result.peak_gain_db:.2f} dB, expected ≈{_G0_DB} dB"
        )

    def test_peak_gain_db_matches_compute_maser_gain(self) -> None:
        """Peak of gain curve must match the on-resonance MaserGainResult."""
        analytical = compute_maser_gain(_FC, _QM, _QL, _BETA)
        result = _wang_result()
        assert abs(result.peak_gain_db - analytical.gain_db) < 0.01, (
            f"gain curve peak {result.peak_gain_db:.3f} dB vs "
            f"analytical {analytical.gain_db:.3f} dB"
        )

    def test_peak_frequency_near_resonance(self) -> None:
        """Peak should appear at the cavity resonance for a symmetric grid."""
        result = _wang_result()
        assert abs(result.peak_frequency_hz - _FC) / _FC < 1e-3, (
            f"peak at {result.peak_frequency_hz:.6g} Hz, "
            f"resonance at {_FC:.6g} Hz"
        )

    # ── Bandwidth ─────────────────────────────────────────────────

    def test_analytical_bandwidth_matches_wang2024_within_5pct(self) -> None:
        result = _wang_result()
        assert abs(result.bandwidth_analytical_hz - _BW_HZ) / _BW_HZ < 0.05, (
            f"B_analytical={result.bandwidth_analytical_hz/1e3:.1f} kHz, "
            f"expected ≈{_BW_HZ/1e3:.0f} kHz"
        )

    def test_analytical_bandwidth_matches_amplifier_result(self) -> None:
        analytical = compute_maser_gain(_FC, _QM, _QL, _BETA)
        result = _wang_result()
        assert abs(result.bandwidth_analytical_hz - analytical.bandwidth_hz) < 1.0, (
            "bandwidth_analytical_hz should exactly match MaserGainResult.bandwidth_hz"
        )

    def test_numerical_bandwidth_within_10pct_of_analytical(self) -> None:
        result = _wang_result()
        bw_a = result.bandwidth_analytical_hz
        bw_n = result.bandwidth_3db_hz
        # Allow 10% since numerical BW depends on grid resolution
        assert abs(bw_n - bw_a) / bw_a < 0.10, (
            f"numerical BW={bw_n/1e3:.2f} kHz vs analytical={bw_a/1e3:.2f} kHz"
        )

    def test_gain_at_half_bandwidth_points_near_half_maximum(self) -> None:
        """G(f_c ± B/2) ≈ G₀/2 in terms of excess above unity."""
        result = _wang_result()
        bw = result.bandwidth_analytical_hz
        g0 = result.peak_gain_linear
        threshold = 1.0 + (g0 - 1.0) / 2.0

        # Evaluate gain at f = f_c ± B_analytical / 2 using the corrected
        # Q_L-normalised formula: n_tilde = (β-1)/(1+β) + Q_L/Q_m,
        #                        d_tilde = 1 - Q_L/Q_m
        n_tilde = (_BETA - 1.0) / (1.0 + _BETA) + _QL / _QM
        d_tilde = 1.0 - _QL / _QM
        for f_probe in (_FC - bw / 2, _FC + bw / 2):
            delta = (f_probe - _FC) / _FC
            u = 2.0 * _QL * delta
            gain_at_halfbw = (n_tilde**2 + u**2) / (d_tilde**2 + u**2)
            # Should be within 10% of the threshold for high gain
            rel = abs(gain_at_halfbw - threshold) / threshold
            assert rel < 0.10, (
                f"G(f_c ± B/2) = {gain_at_halfbw:.4g}, "
                f"threshold = {threshold:.4g}, relative err = {rel:.3f}"
            )

    # ── Gain curve shape properties ───────────────────────────────

    def test_all_gain_values_gte_one(self) -> None:
        """Amplifier gain ≥ 1 everywhere on the grid (net amplification)."""
        result = _wang_result()
        assert np.all(result.gain_linear >= 1.0 - 1e-10), (
            "Some gain values below 1 — device should be a net amplifier"
        )

    def test_gain_approaches_one_far_off_resonance(self) -> None:
        """Far detuned, gain → 1 (passive reflection)."""
        far_off = np.array([_FC - 100e6, _FC + 100e6])  # ±100 MHz from resonance
        result = compute_gain_curve(_FC, _QM, _QL, _BETA, freq_grid_hz=far_off)
        assert np.allclose(result.gain_linear, 1.0, atol=0.02), (
            f"Far off-resonance gain should ≈ 1, got {result.gain_linear}"
        )

    def test_gain_curve_is_symmetric_for_symmetric_grid(self) -> None:
        """Gain curve should be symmetric about f_c for a symmetric grid."""
        result = _wang_result()
        # Default grid is symmetric about f_c
        g = result.gain_linear
        n = len(g)
        left = g[: n // 2]
        right = g[n - n // 2 :][::-1]  # reversed right half
        np.testing.assert_allclose(left, right, rtol=1e-10)

    # ── Gain–bandwidth product ────────────────────────────────────

    def test_gain_bandwidth_product_near_wang_invariant(self) -> None:
        """√G × B ≈ f_c / Q_m for β = 1 (Wang invariant)."""
        result = _wang_result()
        gbp_expected = _FC / _QM  # ≈ 1.806 MHz
        gbp_actual = result.gain_bandwidth_hz
        assert abs(gbp_actual - gbp_expected) / gbp_expected < 0.10, (
            f"GBP = {gbp_actual/1e6:.3f} MHz, expected ≈{gbp_expected/1e6:.3f} MHz"
        )

    # ── below_threshold flag ─────────────────────────────────────

    def test_below_threshold_is_true(self) -> None:
        result = _wang_result()
        assert result.below_threshold is True

    # ── Error handling ────────────────────────────────────────────

    def test_raises_if_q_m_equals_q_l(self) -> None:
        """At threshold gain diverges — must raise ValueError."""
        with pytest.raises(ValueError, match="q_m"):
            compute_gain_curve(_FC, 1337.0, 1337.0, _BETA)

    def test_raises_if_q_m_below_q_l(self) -> None:
        """Above threshold (oscillating) — must raise ValueError."""
        with pytest.raises(ValueError, match="q_m"):
            compute_gain_curve(_FC, 1000.0, 1337.0, _BETA)

    def test_raises_if_probe_power_without_saturation_params(self) -> None:
        with pytest.raises(ValueError, match="probe_power_w"):
            compute_gain_curve(_FC, _QM, _QL, _BETA, probe_power_w=1e-12)

    # ── effective_q_m ─────────────────────────────────────────────

    def test_effective_q_m_equals_q_m_at_zero_probe_power(self) -> None:
        result = _wang_result()
        assert result.effective_q_m == _QM

    # ── Very safe amplifier (Q_m >> Q_L) ─────────────────────────

    def test_large_qm_gives_low_gain_and_wide_bandwidth(self) -> None:
        """Q_m = 10 × Q_L: low gain, wide BW."""
        result = compute_gain_curve(_FC, 13370.0, 1337.0, 1.0)
        # Low gain (close to 1 in linear) because device is far from threshold
        assert result.peak_gain_linear < 2.0
        # Wide bandwidth — much wider than the Wang reference
        assert result.bandwidth_analytical_hz > _BW_HZ * 5

    # ── Saturation power populated when params provided ───────────

    def test_saturation_power_populated_when_params_given(self) -> None:
        result = compute_gain_curve(
            _FC, _QM, _QL, _BETA,
            n_effective=1e10,
            single_spin_coupling_hz=200.0,
            spin_linewidth_hz=318e3,
        )
        assert math.isfinite(result.saturation_power_w)
        assert result.saturation_power_w > 0

    def test_saturation_power_is_nan_without_params(self) -> None:
        result = _wang_result()  # no saturation params provided
        assert math.isnan(result.saturation_power_w)


# ═══════════════════════════════════════════════════════════════════
# 2. TestComputeBandwidth3db
# ═══════════════════════════════════════════════════════════════════

class TestComputeBandwidth3db:
    """Tests for compute_bandwidth_3db()."""

    def _lorentzian_gain_curve(
        self,
        f_c: float,
        g0: float,
        bw: float,
        n: int = 2001,
        span: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ideal Lorentzian gain curve G(f)=(G_0-1)/(1+(Δf/(B/2))²)+1."""
        freq = np.linspace(f_c - span * bw / 2, f_c + span * bw / 2, n)
        gain = 1.0 + (g0 - 1.0) / (1.0 + ((freq - f_c) / (bw / 2)) ** 2)
        return freq, gain

    def test_lorentzian_bandwidth_recovered(self) -> None:
        """Numerical BW from a perfect Lorentzian should match its BW parameter."""
        f0 = 2.87e9
        bw_true = 400e3
        g0_linear = 100.0  # 20 dB
        freq, gain = self._lorentzian_gain_curve(f0, g0_linear, bw_true)
        bw_extracted = compute_bandwidth_3db(gain, freq)
        # Should be within 0.1% of the true BW (fine grid)
        assert abs(bw_extracted - bw_true) / bw_true < 0.001, (
            f"Extracted BW = {bw_extracted/1e3:.2f} kHz, true = {bw_true/1e3:.2f} kHz"
        )

    def test_returns_zero_for_flat_gain_curve(self) -> None:
        """Flat curve at exactly 1 — no amplification, BW = 0."""
        freq = np.linspace(1e9, 2e9, 100)
        gain = np.ones_like(freq)
        assert compute_bandwidth_3db(gain, freq) == 0.0

    def test_returns_zero_for_gain_below_one(self) -> None:
        """Lossy device (gain < 1) should return 0."""
        freq = np.linspace(1e9, 2e9, 100)
        gain = np.full_like(freq, 0.5)
        assert compute_bandwidth_3db(gain, freq) == 0.0

    def test_empty_array_returns_zero(self) -> None:
        assert compute_bandwidth_3db(np.array([]), np.array([])) == 0.0

    def test_returns_full_span_when_gain_above_threshold_everywhere(self) -> None:
        """Gain above threshold at every point → returns full grid span."""
        freq = np.linspace(1.0e9, 2.0e9, 100)
        # Set gain to 1000 everywhere: threshold is 500.5, well below 1000
        gain = np.full_like(freq, 1000.0)
        bw = compute_bandwidth_3db(gain, freq)
        assert abs(bw - (freq[-1] - freq[0])) < 1e3

    def test_bandwidth_increases_with_closer_threshold_approach(self) -> None:
        """Closer to threshold → narrower BW (parametric narrowing)."""
        # Q_m close to Q_L vs Q_m far from Q_L
        r_close = compute_gain_curve(_FC, 1350.0, 1337.0, 1.0)   # just below threshold
        r_far = compute_gain_curve(_FC, 2000.0, 1337.0, 1.0)    # further away
        # Close to threshold: narrow BW; far away: wider BW
        assert r_close.bandwidth_3db_hz < r_far.bandwidth_3db_hz


# ═══════════════════════════════════════════════════════════════════
# 3. TestBandwidthAnalytical
# ═══════════════════════════════════════════════════════════════════

class TestBandwidthAnalytical:
    """Tests for bandwidth_analytical()."""

    def test_returns_correct_wang_bandwidth(self) -> None:
        bw = bandwidth_analytical(_FC, _QM, _QL)
        assert abs(bw - _BW_HZ) / _BW_HZ < 0.01

    def test_returns_zero_at_threshold(self) -> None:
        assert bandwidth_analytical(_FC, _QL, _QL) == 0.0

    def test_returns_zero_above_threshold(self) -> None:
        assert bandwidth_analytical(_FC, 1000.0, 1337.0) == 0.0

    def test_increases_with_qm_for_fixed_ql(self) -> None:
        """Further from threshold (higher Q_m–Q_L) → wider BW."""
        bw1 = bandwidth_analytical(_FC, 1400.0, 1337.0)
        bw2 = bandwidth_analytical(_FC, 2000.0, 1337.0)
        assert bw2 > bw1


# ═══════════════════════════════════════════════════════════════════
# 4. TestComputeGainBandwidthProduct
# ═══════════════════════════════════════════════════════════════════

class TestComputeGainBandwidthProduct:
    """Tests for compute_gain_bandwidth_product()."""

    def test_formula_is_sqrt_g_times_bw(self) -> None:
        g = 100.0  # 20 dB gain
        b = 500e3  # 500 kHz
        gbp = compute_gain_bandwidth_product(g, b)
        assert abs(gbp - math.sqrt(g) * b) < 1e-6

    def test_wang_gbp_near_fc_over_qm(self) -> None:
        """For β=1, GBP ≈ f_c/Q_m (Wang invariant)."""
        result = _wang_result()
        expected = _FC / _QM
        # Allow 10% tolerance (numerical BW introduces small error)
        assert abs(result.gain_bandwidth_hz - expected) / expected < 0.10

    def test_returns_zero_for_zero_gain(self) -> None:
        assert compute_gain_bandwidth_product(0.0, 500e3) == 0.0

    def test_returns_zero_for_zero_bandwidth(self) -> None:
        assert compute_gain_bandwidth_product(100.0, 0.0) == 0.0

    def test_returns_zero_for_negative_inputs(self) -> None:
        assert compute_gain_bandwidth_product(-1.0, 500e3) == 0.0


# ═══════════════════════════════════════════════════════════════════
# 5. TestComputeSaturationPower
# ═══════════════════════════════════════════════════════════════════

class TestComputeSaturationPower:
    """Tests for compute_saturation_power()."""

    # Reference: NV center parameters giving finite P_sat
    _F_C = 2.87e9
    _Q_L = 1337.0
    _BETA = 1.0
    _N_EFF = 1e10
    _G0 = 200.0      # Hz  (single-spin coupling)
    _G_PERP = 320e3  # Hz  (spin linewidth ~ 1/(πT₂*) for T₂*=1μs)

    def _p_sat(self, **kw) -> float:
        params = dict(
            cavity_frequency_hz=self._F_C,
            q_l=self._Q_L,
            coupling_beta=self._BETA,
            n_effective=self._N_EFF,
            single_spin_coupling_hz=self._G0,
            spin_linewidth_hz=self._G_PERP,
        )
        params.update(kw)
        return compute_saturation_power(**params)

    def test_returns_positive_finite_value(self) -> None:
        p = self._p_sat()
        assert math.isfinite(p) and p > 0

    def test_returns_zero_for_zero_n_effective(self) -> None:
        assert self._p_sat(n_effective=0.0) == 0.0

    def test_returns_zero_for_zero_coupling(self) -> None:
        assert self._p_sat(single_spin_coupling_hz=0.0) == 0.0

    def test_returns_zero_for_zero_spin_linewidth(self) -> None:
        assert self._p_sat(spin_linewidth_hz=0.0) == 0.0

    def test_smaller_n_eff_gives_higher_p_sat(self) -> None:
        """Fewer spins → each photon has higher relative effect → saturates at higher power."""
        p_large = self._p_sat(n_effective=1e13)
        p_small = self._p_sat(n_effective=1e10)
        # n_crit ∝ 1/N → P_sat ∝ 1/N → smaller N → smaller P_sat
        # Wait: n_crit = (γ/(2g))²/N, so larger N → smaller n_crit → smaller P_sat
        assert p_large < p_small

    def test_larger_coupling_g0_gives_lower_p_sat(self) -> None:
        """Larger g₀ → smaller n_crit → saturates at lower power."""
        p_weak = self._p_sat(single_spin_coupling_hz=50.0)
        p_strong = self._p_sat(single_spin_coupling_hz=500.0)
        assert p_strong < p_weak

    def test_p_sat_scales_correctly_with_n_eff(self) -> None:
        """P_sat ∝ 1/N_eff (n_crit ∝ 1/N_eff, rest constant)."""
        p1 = self._p_sat(n_effective=1e10)
        p2 = self._p_sat(n_effective=2e10)
        assert abs(p1 / p2 - 2.0) < 0.01, f"P_sat ratio = {p1/p2:.4f}, expected 2.0"


# ═══════════════════════════════════════════════════════════════════
# 6. TestProbePowerSaturation
# ═══════════════════════════════════════════════════════════════════

class TestProbePowerSaturation:
    """Tests for saturation effects in compute_gain_curve()."""

    _SAT_PARAMS = dict(
        n_effective=1e10,
        single_spin_coupling_hz=200.0,
        spin_linewidth_hz=320e3,
    )

    def _sat_result(self, probe_power_w: float) -> GainCurveResult:
        return compute_gain_curve(
            _FC, _QM, _QL, _BETA,
            probe_power_w=probe_power_w,
            **self._SAT_PARAMS,
        )

    def test_zero_probe_power_matches_small_signal(self) -> None:
        r0 = _wang_result()
        r_sat = self._sat_result(probe_power_w=0.0)
        np.testing.assert_allclose(r0.gain_linear, r_sat.gain_linear, rtol=1e-10)

    def test_large_probe_power_compresses_gain(self) -> None:
        """High P_in → Q_m_eff >> Q_L → on-resonance gain collapses."""
        p_sat = compute_saturation_power(
            cavity_frequency_hz=_FC, q_l=_QL, coupling_beta=_BETA,
            **self._SAT_PARAMS,
        )
        # Apply 1000× saturation: evaluate only at resonance
        resonance_grid = np.array([_FC])
        r_compressed = compute_gain_curve(
            _FC, _QM, _QL, _BETA,
            freq_grid_hz=resonance_grid,
            probe_power_w=1000.0 * p_sat,
            **self._SAT_PARAMS,
        )
        r_small = compute_gain_curve(
            _FC, _QM, _QL, _BETA, freq_grid_hz=resonance_grid
        )
        # On-resonance gain should be strongly compressed
        assert float(r_compressed.gain_linear[0]) < float(r_small.gain_linear[0]) * 0.01

    def test_probe_at_p_sat_increases_effective_q_m(self) -> None:
        """At P_in = P_sat, Q_m_eff should be 2 × Q_m."""
        p_sat = compute_saturation_power(
            cavity_frequency_hz=_FC, q_l=_QL, coupling_beta=_BETA,
            **self._SAT_PARAMS,
        )
        r = self._sat_result(probe_power_w=p_sat)
        assert abs(r.effective_q_m / _QM - 2.0) < 1e-6

    def test_bandwidth_widens_with_increasing_probe_power(self) -> None:
        """Higher P_in → Q_m_eff larger → further from threshold → wider analytical BW."""
        p_sat = compute_saturation_power(
            cavity_frequency_hz=_FC, q_l=_QL, coupling_beta=_BETA,
            **self._SAT_PARAMS,
        )
        r_small = self._sat_result(probe_power_w=0.0)
        r_sat = self._sat_result(probe_power_w=p_sat)
        # Parametric formula B = f_c(Q_m_eff−Q_L)/(Q_L Q_m_eff) increases with Q_m_eff
        assert r_sat.bandwidth_analytical_hz > r_small.bandwidth_analytical_hz

    def test_effective_q_m_always_gte_small_signal_q_m(self) -> None:
        """Saturation can only increase Q_m_eff, never reduce it below Q_m."""
        p_sat = compute_saturation_power(
            cavity_frequency_hz=_FC, q_l=_QL, coupling_beta=_BETA,
            **self._SAT_PARAMS,
        )
        for power_factor in (0.0, 0.1, 1.0, 10.0, 1000.0):
            r = self._sat_result(probe_power_w=power_factor * p_sat)
            assert r.effective_q_m >= _QM - 1e-9, (
                f"Q_m_eff={r.effective_q_m} < Q_m={_QM} at power_factor={power_factor}"
            )


# ═══════════════════════════════════════════════════════════════════
# 7. TestGainCurveFromMBResult
# ═══════════════════════════════════════════════════════════════════

class TestGainCurveFromMBResult:
    """Tests for gain_curve_from_mb_result() — wiring with SpectralMBResult."""

    def _make_mock_mb_result(self):
        """Build a minimal SpectralMBResult using the real solver (above-threshold)."""
        from nv_maser.config import (
            DipolarConfig,
            MaxwellBlochConfig,
            SpectralConfig,
        )
        from nv_maser.physics.spectral_maxwell_bloch import solve_spectral_maxwell_bloch

        nv = _nv_cfg()
        maser = _maser_cfg(cavity_q=10_000, cavity_frequency_ghz=1.47)
        cavity = _cavity_cfg()

        mb_cfg = MaxwellBlochConfig(t_max_us=50.0)
        sp_cfg = SpectralConfig(
            inhomogeneous_linewidth_mhz=1.0,
            n_freq_bins=11,
        )
        dip_cfg = DipolarConfig(enable=False)

        return (
            solve_spectral_maxwell_bloch(
                nv, maser, cavity, mb_cfg, sp_cfg, dip_cfg
            ),
            nv,
            maser,
            cavity,
        )

    def _make_below_threshold_mb_result(self):
        """Return a stub (mb_result, nv, maser, cavity) with Q_m > Q_L.

        Q_m from NV params ≈ 7788 (depends only on spin density/T₂/fill).
        cavity_q = 7000 → Q_L = 7000 < 7788, so device is below oscillation
        threshold and the gain curve is valid.
        """
        import numpy as _np

        from nv_maser.physics.spectral_maxwell_bloch import SpectralMBResult

        nv = _nv_cfg()
        maser = _maser_cfg(cavity_q=7_000, cavity_frequency_ghz=1.47)
        cavity = _cavity_cfg()

        # gain_curve_from_mb_result does not use any field of mb_result;
        # supply a minimal valid stub to satisfy the type annotation.
        _z = _np.zeros(1)
        stub = SpectralMBResult(
            time_s=_z, cavity_re=_z, cavity_im=_z, photon_number=_z,
            delta_hz=_z, inversion_profile=_z, on_resonance_inversion=_z,
            steady_state_photons=0.0, steady_state_on_res_inversion=0.0,
            output_power_w=0.0, cooperativity=0.5, n_bursts=0,
        )
        return stub, nv, maser, cavity

    def test_returns_gain_curve_result(self) -> None:
        mb, nv, maser, cavity = self._make_below_threshold_mb_result()
        result = gain_curve_from_mb_result(mb, nv, maser, cavity)
        assert isinstance(result, GainCurveResult)

    def test_gain_curve_has_correct_shape(self) -> None:
        mb, nv, maser, cavity = self._make_below_threshold_mb_result()
        result = gain_curve_from_mb_result(mb, nv, maser, cavity, n_points=201)
        assert result.gain_linear.shape == (201,)

    def test_saturation_power_is_finite(self) -> None:
        """Saturation params are always derived → P_sat should be finite."""
        mb, nv, maser, cavity = self._make_below_threshold_mb_result()
        result = gain_curve_from_mb_result(mb, nv, maser, cavity)
        assert math.isfinite(result.saturation_power_w)
        assert result.saturation_power_w > 0

    def test_peak_frequency_near_cavity_resonance(self) -> None:
        mb, nv, maser, cavity = self._make_below_threshold_mb_result()
        result = gain_curve_from_mb_result(mb, nv, maser, cavity)
        f_c = maser.cavity_frequency_ghz * 1e9
        # Peak should be within 0.1% of resonance
        assert abs(result.peak_frequency_hz - f_c) / f_c < 1e-3

    def test_raises_when_above_threshold(self) -> None:
        """If Q_m ≤ Q_L (oscillating), gain curve is undefined."""
        mb, nv_base, maser_base, cavity_base = self._make_mock_mb_result()

        # Increase density and pump to drive Q_m below Q_L
        nv_high = NVConfig(
            nv_density_per_cm3=1e22,   # very high density
            pump_efficiency=0.95,
            orientation_fraction=1.0,
            t2_star_us=100.0,
            t1_ms=5.0,
        )
        cavity_large = CavityConfig(mode_volume_cm3=0.001, fill_factor=0.99)
        with pytest.raises(ValueError):
            gain_curve_from_mb_result(mb, nv_high, maser_base, cavity_large)


# ═══════════════════════════════════════════════════════════════════
# 8. TestGainCurveResultImmutability
# ═══════════════════════════════════════════════════════════════════

class TestGainCurveResultImmutability:
    """GainCurveResult is a frozen dataclass."""

    def test_is_frozen(self) -> None:
        result = _wang_result()
        with pytest.raises((AttributeError, TypeError)):
            result.peak_gain_db = 99.0  # type: ignore[misc]

    def test_gain_linear_array_shape_consistent_with_frequencies(self) -> None:
        result = _wang_result()
        assert result.gain_linear.shape == result.frequencies_hz.shape


# ═══════════════════════════════════════════════════════════════════
# 9. TestMaserConfigProbeField
# ═══════════════════════════════════════════════════════════════════

class TestMaserConfigProbeField:
    """Verify MaserConfig.maser_probe_power_dbm was added correctly."""

    def test_default_probe_power_dbm(self) -> None:
        cfg = MaserConfig()
        assert cfg.maser_probe_power_dbm == -120.0

    def test_custom_probe_power_accepted(self) -> None:
        cfg = MaserConfig(maser_probe_power_dbm=-60.0)
        assert cfg.maser_probe_power_dbm == -60.0

    def test_probe_power_to_watts_conversion(self) -> None:
        cfg = MaserConfig(maser_probe_power_dbm=-30.0)  # 1 μW
        p_w = 1e-3 * 10 ** (cfg.maser_probe_power_dbm / 10)
        assert abs(p_w - 1e-6) / 1e-6 < 1e-6
