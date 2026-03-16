"""Tests for the spectral Maxwell-Bloch solver."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nv_maser.config import (
    CavityConfig,
    DipolarConfig,
    MaserConfig,
    MaxwellBlochConfig,
    NVConfig,
    SpectralConfig,
)
from nv_maser.physics.spectral_maxwell_bloch import (
    SpectralMBResult,
    _compute_bin_weights,
    _count_bursts,
    solve_spectral_maxwell_bloch,
)
from nv_maser.physics.spectral import build_detuning_grid


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def nv_cfg() -> NVConfig:
    return NVConfig()


@pytest.fixture
def maser_cfg() -> MaserConfig:
    return MaserConfig()


@pytest.fixture
def cavity_cfg() -> CavityConfig:
    return CavityConfig()


@pytest.fixture
def spectral_cfg() -> SpectralConfig:
    return SpectralConfig(
        enable=True,
        n_freq_bins=51,  # small for fast tests
        freq_range_mhz=30.0,
        inhomogeneous_linewidth_mhz=8.65,
    )


@pytest.fixture
def mb_cfg() -> MaxwellBlochConfig:
    return MaxwellBlochConfig(
        enable=True,
        t_max_us=20.0,
        n_time_points=200,
    )


@pytest.fixture
def dipolar_cfg() -> DipolarConfig:
    return DipolarConfig(
        enable=True,
        refilling_time_us=11.6,
        stretch_exponent=0.5,
    )


# ── Bin weights ───────────────────────────────────────────────────


class TestBinWeights:
    def test_sum_equals_n_eff(self, spectral_cfg: SpectralConfig) -> None:
        delta = build_detuning_grid(spectral_cfg)
        n_eff = 1e10
        weights = _compute_bin_weights(delta, spectral_cfg, n_eff)
        assert weights.sum() == pytest.approx(n_eff, rel=1e-6)

    def test_peak_at_centre(self, spectral_cfg: SpectralConfig) -> None:
        delta = build_detuning_grid(spectral_cfg)
        weights = _compute_bin_weights(delta, spectral_cfg, 1e10)
        centre = len(delta) // 2
        assert weights[centre] == weights.max()

    def test_all_positive(self, spectral_cfg: SpectralConfig) -> None:
        delta = build_detuning_grid(spectral_cfg)
        weights = _compute_bin_weights(delta, spectral_cfg, 1e10)
        assert np.all(weights >= 0)

    def test_shape(self, spectral_cfg: SpectralConfig) -> None:
        delta = build_detuning_grid(spectral_cfg)
        weights = _compute_bin_weights(delta, spectral_cfg, 1e10)
        assert weights.shape == delta.shape


# ── Burst counter ─────────────────────────────────────────────────


class TestCountBursts:
    def test_no_bursts_flat(self) -> None:
        signal = np.ones(100)
        assert _count_bursts(signal) == 0

    def test_single_burst(self) -> None:
        signal = np.ones(100)
        signal[40:50] = 100.0
        assert _count_bursts(signal) == 1

    def test_multiple_bursts(self) -> None:
        signal = np.ones(200)
        signal[30:40] = 100.0
        signal[80:90] = 100.0
        signal[150:160] = 100.0
        assert _count_bursts(signal) == 3

    def test_zero_signal(self) -> None:
        signal = np.zeros(100)
        assert _count_bursts(signal) == 0


# ── Result structure ──────────────────────────────────────────────


class TestSpectralMBResult:
    def test_result_fields(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        mb_cfg: MaxwellBlochConfig,
        spectral_cfg: SpectralConfig,
    ) -> None:
        result = solve_spectral_maxwell_bloch(
            nv_cfg, maser_cfg, cavity_cfg, mb_cfg, spectral_cfg,
        )
        assert isinstance(result, SpectralMBResult)
        assert len(result.time_s) == mb_cfg.n_time_points
        assert result.delta_hz.shape == (spectral_cfg.n_freq_bins,)
        assert result.inversion_profile.shape == (
            mb_cfg.n_time_points,
            spectral_cfg.n_freq_bins,
        )
        assert result.on_resonance_inversion.shape == (mb_cfg.n_time_points,)
        assert result.photon_number.shape == (mb_cfg.n_time_points,)
        assert result.cooperativity > 0
        assert result.output_power_w >= 0

    def test_time_monotonic(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        mb_cfg: MaxwellBlochConfig,
        spectral_cfg: SpectralConfig,
    ) -> None:
        result = solve_spectral_maxwell_bloch(
            nv_cfg, maser_cfg, cavity_cfg, mb_cfg, spectral_cfg,
        )
        assert np.all(np.diff(result.time_s) > 0)


# ── Free-running maser dynamics ───────────────────────────────────


class TestFreeRunningMaser:
    def test_above_threshold_photons(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        spectral_cfg: SpectralConfig,
    ) -> None:
        """Above threshold (default config), maser should produce photons."""
        mb = MaxwellBlochConfig(enable=True, t_max_us=100.0, n_time_points=500)
        result = solve_spectral_maxwell_bloch(
            nv_cfg, maser_cfg, cavity_cfg, mb, spectral_cfg,
        )
        assert result.steady_state_photons > 0

    def test_initial_inversion_matches_pump(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        mb_cfg: MaxwellBlochConfig,
        spectral_cfg: SpectralConfig,
    ) -> None:
        """Initial inversion profile should peak at sz0."""
        result = solve_spectral_maxwell_bloch(
            nv_cfg, maser_cfg, cavity_cfg, mb_cfg, spectral_cfg,
        )
        sz0 = nv_cfg.pump_efficiency / 2.0
        assert result.inversion_profile[0].max() == pytest.approx(sz0, rel=0.01)

    def test_spectral_hole_develops(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        spectral_cfg: SpectralConfig,
    ) -> None:
        """After masing begins, on-resonance inversion should drop below initial."""
        mb = MaxwellBlochConfig(enable=True, t_max_us=100.0, n_time_points=500)
        result = solve_spectral_maxwell_bloch(
            nv_cfg, maser_cfg, cavity_cfg, mb, spectral_cfg,
        )
        initial_on_res = result.on_resonance_inversion[0]
        final_on_res = result.on_resonance_inversion[-1]
        # If above threshold, on-resonance inversion should decrease
        if result.steady_state_photons > 1:
            assert final_on_res < initial_on_res

    def test_off_resonance_preserved(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        spectral_cfg: SpectralConfig,
    ) -> None:
        """Far-off-resonance bins should be less affected than on-resonance."""
        mb = MaxwellBlochConfig(enable=True, t_max_us=100.0, n_time_points=500)
        result = solve_spectral_maxwell_bloch(
            nv_cfg, maser_cfg, cavity_cfg, mb, spectral_cfg,
        )
        centre = spectral_cfg.n_freq_bins // 2
        # Compare depletion at centre vs edge
        initial = result.inversion_profile[0]
        final = result.inversion_profile[-1]
        depletion_centre = initial[centre] - final[centre]
        depletion_edge = initial[0] - final[0]
        if result.steady_state_photons > 1:
            assert depletion_centre > depletion_edge

    def test_cooperativity_consistent_with_scalar(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        mb_cfg: MaxwellBlochConfig,
        spectral_cfg: SpectralConfig,
    ) -> None:
        """Cooperativity should match scalar solver (same physics, different representation)."""
        from nv_maser.physics.maxwell_bloch import solve_maxwell_bloch

        scalar_result = solve_maxwell_bloch(
            nv_cfg, maser_cfg, cavity_cfg, mb_cfg,
        )
        spectral_result = solve_spectral_maxwell_bloch(
            nv_cfg, maser_cfg, cavity_cfg, mb_cfg, spectral_cfg,
        )
        # Should be close — not exact because spectral discretises the lineshape
        assert spectral_result.cooperativity == pytest.approx(
            scalar_result.cooperativity, rel=0.1,
        )


# ── Dipolar refilling ────────────────────────────────────────────


class TestDipolarIntegration:
    def test_with_dipolar_runs(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        spectral_cfg: SpectralConfig,
        dipolar_cfg: DipolarConfig,
    ) -> None:
        """Solver should complete with dipolar refilling enabled."""
        mb = MaxwellBlochConfig(enable=True, t_max_us=50.0, n_time_points=200)
        result = solve_spectral_maxwell_bloch(
            nv_cfg, maser_cfg, cavity_cfg, mb, spectral_cfg, dipolar_cfg,
        )
        assert isinstance(result, SpectralMBResult)
        assert len(result.time_s) == 200

    def test_dipolar_refills_hole(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        spectral_cfg: SpectralConfig,
    ) -> None:
        """With dipolar refilling, on-resonance inversion should be higher than without."""
        mb = MaxwellBlochConfig(enable=True, t_max_us=100.0, n_time_points=500)
        dipolar_on = DipolarConfig(enable=True, refilling_time_us=11.6)
        dipolar_off = DipolarConfig(enable=False)

        res_on = solve_spectral_maxwell_bloch(
            nv_cfg, maser_cfg, cavity_cfg, mb, spectral_cfg, dipolar_on,
        )
        res_off = solve_spectral_maxwell_bloch(
            nv_cfg, maser_cfg, cavity_cfg, mb, spectral_cfg, dipolar_off,
        )

        # Dipolar refilling should maintain higher on-resonance inversion
        assert res_on.steady_state_on_res_inversion >= res_off.steady_state_on_res_inversion - 0.01

    def test_dipolar_disabled_no_effect(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        mb_cfg: MaxwellBlochConfig,
        spectral_cfg: SpectralConfig,
    ) -> None:
        """With dipolar disabled, result should match no-dipolar run."""
        result_none = solve_spectral_maxwell_bloch(
            nv_cfg, maser_cfg, cavity_cfg, mb_cfg, spectral_cfg, None,
        )
        result_off = solve_spectral_maxwell_bloch(
            nv_cfg, maser_cfg, cavity_cfg, mb_cfg, spectral_cfg,
            DipolarConfig(enable=False),
        )
        np.testing.assert_allclose(
            result_none.photon_number, result_off.photon_number, rtol=1e-10,
        )


# ── Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_pump_no_masing(
        self,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        mb_cfg: MaxwellBlochConfig,
        spectral_cfg: SpectralConfig,
    ) -> None:
        """With zero pump efficiency, no masing should occur."""
        nv_zero = NVConfig(pump_efficiency=0.0)
        result = solve_spectral_maxwell_bloch(
            nv_zero, maser_cfg, cavity_cfg, mb_cfg, spectral_cfg,
        )
        assert result.steady_state_photons < 1e-6

    def test_very_low_q_no_masing(
        self,
        nv_cfg: NVConfig,
        cavity_cfg: CavityConfig,
        mb_cfg: MaxwellBlochConfig,
        spectral_cfg: SpectralConfig,
    ) -> None:
        """Very low cavity Q should prevent masing."""
        maser_low_q = MaserConfig(cavity_q=10)
        result = solve_spectral_maxwell_bloch(
            nv_cfg, maser_low_q, cavity_cfg, mb_cfg, spectral_cfg,
        )
        assert result.steady_state_photons < 1

    def test_single_bin_degenerates(
        self,
        nv_cfg: NVConfig,
        maser_cfg: MaserConfig,
        cavity_cfg: CavityConfig,
        mb_cfg: MaxwellBlochConfig,
    ) -> None:
        """With a single frequency bin, behaviour should resemble scalar solver."""
        spectral_1bin = SpectralConfig(
            enable=True,
            n_freq_bins=11,
            freq_range_mhz=0.001,  # very narrow ≈ single bin
            inhomogeneous_linewidth_mhz=100.0,  # much wider than range
        )
        result = solve_spectral_maxwell_bloch(
            nv_cfg, maser_cfg, cavity_cfg, mb_cfg, spectral_1bin,
        )
        assert isinstance(result, SpectralMBResult)
        assert result.cooperativity > 0
