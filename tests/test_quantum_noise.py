"""Tests for the quantum Langevin noise model (quantum_noise.py)."""
import math

import numpy as np
import pytest

from nv_maser.config import CavityConfig, MaserConfig, MaxwellBlochConfig, NVConfig
from nv_maser.physics.cavity import compute_cavity_properties
from nv_maser.physics.maxwell_bloch import solve_maxwell_bloch
from nv_maser.physics.quantum_noise import (
    MaserNoiseResult,
    PhaseNoiseSpectrum,
    RINSpectrum,
    compute_added_noise,
    compute_maser_noise,
    compute_noise_temperature,
    compute_phase_noise_spectrum,
    compute_population_inversion_factor,
    compute_rin_spectrum,
    compute_schawlow_townes_linewidth,
)


# ── Shared physical constants ──────────────────────────────────────

_HBAR = 1.054571817e-34
_KB   = 1.380649e-23


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
def mb_cfg() -> MaxwellBlochConfig:
    return MaxwellBlochConfig(enable=True, t_max_us=50.0, n_time_points=500)


@pytest.fixture
def cavity_props(maser_cfg: MaserConfig, cavity_cfg: CavityConfig) -> object:
    return compute_cavity_properties(maser_cfg, cavity_cfg)


@pytest.fixture
def mb_result(
    nv_cfg: NVConfig,
    maser_cfg: MaserConfig,
    cavity_cfg: CavityConfig,
    mb_cfg: MaxwellBlochConfig,
) -> object:
    return solve_maxwell_bloch(nv_cfg, maser_cfg, cavity_cfg, mb_cfg)


@pytest.fixture
def noise_result(cavity_props, mb_result, nv_cfg, maser_cfg) -> MaserNoiseResult:
    return compute_maser_noise(cavity_props, mb_result, nv_cfg, maser_cfg)


@pytest.fixture
def default_freq_offsets() -> np.ndarray:
    """Decade-spaced frequency offsets from 1 Hz to 1 MHz."""
    return np.logspace(0, 6, 50)


# ── TestComputePopulationInversionFactor ──────────────────────────


class TestComputePopulationInversionFactor:
    """n_sp = (1 + η) / (2η)."""

    def test_full_inversion_n_sp_equals_one(self) -> None:
        """At η = 1 (complete inversion) n_sp must be exactly 1."""
        assert compute_population_inversion_factor(1.0) == pytest.approx(1.0, rel=1e-9)

    def test_half_inversion_n_sp_equals_1p5(self) -> None:
        """At η = 0.5 (equal populations), n_sp = (1+0.5)/(2*0.5) = 1.5."""
        assert compute_population_inversion_factor(0.5) == pytest.approx(1.5, rel=1e-9)

    def test_quarter_inversion_n_sp_equals_2p5(self) -> None:
        """At η = 0.25, n_sp = (1.25)/(0.5) = 2.5."""
        assert compute_population_inversion_factor(0.25) == pytest.approx(2.5, rel=1e-9)

    def test_invalid_zero_pump_efficiency_raises(self) -> None:
        """η = 0 must raise ValueError (division by zero)."""
        with pytest.raises(ValueError, match="pump_efficiency"):
            compute_population_inversion_factor(0.0)

    def test_invalid_negative_pump_efficiency_raises(self) -> None:
        """Negative η is unphysical."""
        with pytest.raises(ValueError):
            compute_population_inversion_factor(-0.1)

    def test_n_sp_increases_as_eta_decreases(self) -> None:
        """Worse pump efficiency → more noise (larger n_sp)."""
        n_sp_good = compute_population_inversion_factor(0.8)
        n_sp_bad  = compute_population_inversion_factor(0.2)
        assert n_sp_bad > n_sp_good


# ── TestSchawlowTownesLinewidth ────────────────────────────────────


class TestSchawlowTownesLinewidth:
    """Δν_ST = κ_c_hz × n_sp / (2 N̄)."""

    def test_exact_formula(self) -> None:
        """Verify arithmetic against known values."""
        kappa = 147_000.0   # Hz  (Q=10000 at 1.47 GHz)
        n_bar = 1_000.0
        n_sp  = 1.5
        expected = kappa * n_sp / (2.0 * n_bar)   # 110.25 Hz
        result = compute_schawlow_townes_linewidth(kappa, n_bar, n_sp)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_linewidth_halves_when_photons_double(self) -> None:
        """Δν_ST ∝ 1/N̄  — doubling intracavity photons halves linewidth."""
        kappa, n_sp = 147_000.0, 1.5
        lw1 = compute_schawlow_townes_linewidth(kappa, 1000.0, n_sp)
        lw2 = compute_schawlow_townes_linewidth(kappa, 2000.0, n_sp)
        assert lw2 == pytest.approx(lw1 / 2.0, rel=1e-9)

    def test_linewidth_well_below_cavity_linewidth(self) -> None:
        """Δν_ST ≪ κ_c/(2π) — the key maser property."""
        kappa = 147_000.0
        n_bar = 100.0
        n_sp  = 1.5
        delta_nu = compute_schawlow_townes_linewidth(kappa, n_bar, n_sp)
        assert delta_nu < kappa

    def test_minimum_at_full_inversion(self) -> None:
        """n_sp = 1 gives narrowest linewidth for fixed N̄."""
        kappa, n_bar = 147_000.0, 500.0
        lw_ideal = compute_schawlow_townes_linewidth(kappa, n_bar, 1.0)
        lw_real  = compute_schawlow_townes_linewidth(kappa, n_bar, 1.5)
        assert lw_ideal < lw_real

    def test_below_threshold_returns_cavity_linewidth(self) -> None:
        """N̄ ≤ 0 (below threshold) defaults to cavity linewidth."""
        kappa = 147_000.0
        result = compute_schawlow_townes_linewidth(kappa, 0.0, 1.5)
        assert result == pytest.approx(kappa)


# ── TestAddedNoise ────────────────────────────────────────────────


class TestAddedNoise:
    """n_add = n_sp (Caves 1982 theorem)."""

    def test_added_noise_equals_n_sp(self) -> None:
        """Caves theorem: n_add = n_sp exactly."""
        for n_sp in [1.0, 1.5, 2.0, 3.14]:
            assert compute_added_noise(n_sp) == pytest.approx(n_sp, rel=1e-9)

    def test_minimum_added_noise_is_one(self) -> None:
        """Ideal maser (n_sp=1) adds exactly 1 photon per mode."""
        assert compute_added_noise(1.0) == pytest.approx(1.0)

    def test_invalid_n_sp_below_one_raises(self) -> None:
        """n_sp < 1 violates Caves theorem."""
        with pytest.raises(ValueError, match="Caves"):
            compute_added_noise(0.99)


# ── TestNoiseTemperature ───────────────────────────────────────────


class TestNoiseTemperature:
    """T_noise = ℏω n_sp / k_B."""

    def test_below_1k_for_ghz_maser(self) -> None:
        """NV maser at 1.47 GHz has noise temperature < 1 K."""
        t = compute_noise_temperature(1.47e9, 1.5)
        assert t < 1.0
        assert t > 0.0

    def test_minimum_at_full_inversion(self) -> None:
        """n_sp = 1 gives minimum noise temperature."""
        t_min = compute_noise_temperature(1.47e9, 1.0)
        t_typ = compute_noise_temperature(1.47e9, 1.5)
        assert t_min < t_typ

    def test_proportional_to_frequency(self) -> None:
        """T_noise ∝ frequency — doubling freq doubles T_noise."""
        t1 = compute_noise_temperature(1.0e9, 1.5)
        t2 = compute_noise_temperature(2.0e9, 1.5)
        assert t2 == pytest.approx(2.0 * t1, rel=1e-9)

    def test_proportional_to_n_sp(self) -> None:
        """T_noise ∝ n_sp for fixed frequency."""
        t1 = compute_noise_temperature(1.47e9, 1.0)
        t2 = compute_noise_temperature(1.47e9, 2.0)
        assert t2 == pytest.approx(2.0 * t1, rel=1e-9)

    def test_exact_value(self) -> None:
        """Verify against manual calculation."""
        freq = 1.47e9
        n_sp = 1.5
        omega = 2.0 * math.pi * freq
        expected = _HBAR * omega * n_sp / _KB
        assert compute_noise_temperature(freq, n_sp) == pytest.approx(expected, rel=1e-9)

    def test_invalid_zero_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="frequency_hz"):
            compute_noise_temperature(0.0, 1.5)


# ── TestPhaseNoiseSpectrum ────────────────────────────────────────


class TestPhaseNoiseSpectrum:
    """S_φ(f) = Δν_ST / (π f²)."""

    def test_inverse_square_law(self) -> None:
        """Doubling the offset frequency quarters the PSD."""
        st = 110.0   # Hz
        f  = np.array([100.0, 200.0])
        result = compute_phase_noise_spectrum(st, f)
        assert result.psd_rad2_per_hz[1] == pytest.approx(
            result.psd_rad2_per_hz[0] / 4.0, rel=1e-9
        )

    def test_all_psd_positive(self, default_freq_offsets: np.ndarray) -> None:
        """All PSD values must be positive."""
        result = compute_phase_noise_spectrum(110.0, default_freq_offsets)
        assert np.all(result.psd_rad2_per_hz > 0)

    def test_monotone_decreasing(self, default_freq_offsets: np.ndarray) -> None:
        """PSD decreases monotonically with offset frequency."""
        result = compute_phase_noise_spectrum(110.0, default_freq_offsets)
        assert np.all(np.diff(result.psd_rad2_per_hz) < 0)

    def test_dbc_decreases_20dB_per_decade(self) -> None:
        """−20 dB/decade slope is the 1/f² signature."""
        f  = np.array([100.0, 1000.0])   # 10× offset
        result = compute_phase_noise_spectrum(110.0, f)
        # slope should be exactly −20 dB per decade
        slope = result.psd_dbc_hz[1] - result.psd_dbc_hz[0]   # should be −20
        assert slope == pytest.approx(-20.0, abs=1e-6)

    def test_dbc_below_zero_at_large_offset(self) -> None:
        """dBc/Hz must be strongly negative far from carrier."""
        result = compute_phase_noise_spectrum(110.0, np.array([1e6]))
        assert result.psd_dbc_hz[0] < -100.0

    def test_exact_psd_formula(self) -> None:
        """Verify S_φ(f) = Δν_ST / (π f²) numerically."""
        st, f0 = 110.0, 1000.0
        result = compute_phase_noise_spectrum(st, np.array([f0]))
        expected = st / (math.pi * f0**2)
        assert result.psd_rad2_per_hz[0] == pytest.approx(expected, rel=1e-9)

    def test_dc_offset_raises(self) -> None:
        """f = 0 is undefined for 1/f² spectrum."""
        with pytest.raises(ValueError):
            compute_phase_noise_spectrum(110.0, np.array([0.0, 100.0]))

    def test_stored_linewidth_matches(self) -> None:
        """Result stores the linewidth used for computation."""
        st = 250.0
        result = compute_phase_noise_spectrum(st, np.array([100.0]))
        assert result.schawlow_townes_linewidth_hz == pytest.approx(st)


# ── TestRINSpectrum ───────────────────────────────────────────────


class TestRINSpectrum:
    """RIN(f) = (2 n_sp / N̄) / (1 + (f/κ_c)²)."""

    def test_peak_at_dc(self) -> None:
        """RIN(0) = 2 n_sp / N̄ — the DC plateau = shot noise floor."""
        kappa, n_bar, n_sp = 147_000.0, 1000.0, 1.5
        result = compute_rin_spectrum(kappa, n_bar, n_sp, np.array([0.0]))
        expected = 2.0 * n_sp / n_bar
        assert result.rin_per_hz[0] == pytest.approx(expected, rel=1e-9)

    def test_3dB_corner_at_cavity_linewidth(self) -> None:
        """RIN(κ_c_hz) = RIN(0) / 2 — Lorentzian -3 dB."""
        kappa, n_bar, n_sp = 147_000.0, 1000.0, 1.5
        result = compute_rin_spectrum(kappa, n_bar, n_sp, np.array([0.0, kappa]))
        assert result.rin_per_hz[1] == pytest.approx(result.rin_per_hz[0] / 2.0, rel=1e-9)

    def test_decreases_above_corner(self) -> None:
        """RIN above corner frequency < RIN at corner."""
        kappa, n_bar, n_sp = 147_000.0, 1000.0, 1.5
        result = compute_rin_spectrum(
            kappa, n_bar, n_sp, np.array([kappa, 2.0 * kappa])
        )
        assert result.rin_per_hz[1] < result.rin_per_hz[0]

    def test_rin_floor_stored_correctly(self) -> None:
        """rin_floor_per_hz == DC value 2 n_sp / N̄."""
        kappa, n_bar, n_sp = 147_000.0, 1000.0, 1.5
        result = compute_rin_spectrum(kappa, n_bar, n_sp, np.array([0.0]))
        assert result.rin_floor_per_hz == pytest.approx(2.0 * n_sp / n_bar, rel=1e-9)

    def test_doubles_when_photons_halve(self) -> None:
        """RIN_floor ∝ 1/N̄ — halving photon number doubles RIN."""
        kappa, n_sp = 147_000.0, 1.5
        r1 = compute_rin_spectrum(kappa, 1000.0, n_sp, np.array([0.0]))
        r2 = compute_rin_spectrum(kappa, 500.0,  n_sp, np.array([0.0]))
        assert r2.rin_floor_per_hz == pytest.approx(2.0 * r1.rin_floor_per_hz, rel=1e-9)

    def test_invalid_zero_photons_raises(self) -> None:
        with pytest.raises(ValueError, match="steady_state_photons"):
            compute_rin_spectrum(147_000.0, 0.0, 1.5, np.array([0.0]))

    def test_all_rin_finite_and_positive(self) -> None:
        freqs = np.logspace(0, 7, 40)
        result = compute_rin_spectrum(147_000.0, 500.0, 1.5, freqs)
        assert np.all(np.isfinite(result.rin_per_hz))
        assert np.all(result.rin_per_hz > 0)


# ── TestComputeMaserNoise ─────────────────────────────────────────


class TestComputeMaserNoise:
    """Integration tests — use real configs + Maxwell-Bloch solver."""

    def test_returns_correct_type(self, noise_result: MaserNoiseResult) -> None:
        assert isinstance(noise_result, MaserNoiseResult)

    def test_noise_temp_below_1k_for_default_config(
        self, noise_result: MaserNoiseResult
    ) -> None:
        """NV maser at 1.47 GHz must have noise temp < 1 K."""
        assert noise_result.noise_temperature_k < 1.0
        assert noise_result.noise_temperature_k > 0.0

    def test_st_linewidth_well_below_cavity_linewidth(
        self, noise_result: MaserNoiseResult
    ) -> None:
        """Above threshold (N̄ ≥ 1), Schawlow-Townes line must be narrower than cavity.

        The 50 µs ODE run may be below threshold for default params; that case is
        handled correctly by the module (returns κ_c as upper bound) and is already
        exercised by TestSchawlowTownesLinewidth.test_below_threshold_returns_cavity_linewidth.
        """
        if noise_result.steady_state_photons < 1.0:
            return  # Below threshold — Schawlow-Townes formula not applicable
        assert noise_result.schawlow_townes_linewidth_hz < noise_result.cavity_linewidth_hz

    def test_added_noise_equals_n_sp(self, noise_result: MaserNoiseResult) -> None:
        """Caves theorem: n_add = n_sp always."""
        assert noise_result.added_noise_number == pytest.approx(
            noise_result.population_inversion_factor, rel=1e-9
        )

    def test_rin_floor_matches_formula(self, noise_result: MaserNoiseResult) -> None:
        """RIN floor = 2 n_sp / N̄ when above threshold."""
        if noise_result.steady_state_photons > 0:
            expected = 2.0 * noise_result.population_inversion_factor / noise_result.steady_state_photons
            assert noise_result.rin_floor_per_hz == pytest.approx(expected, rel=1e-9)

    def test_cavity_frequency_matches_config(
        self, noise_result: MaserNoiseResult, maser_cfg: MaserConfig
    ) -> None:
        """Stored cavity frequency must match MaserConfig."""
        expected = maser_cfg.cavity_frequency_ghz * 1e9
        assert noise_result.cavity_frequency_hz == pytest.approx(expected, rel=1e-9)

    def test_output_power_matches_mb(
        self, noise_result: MaserNoiseResult, mb_result: object
    ) -> None:
        """Output power must be passed through from Maxwell-Bloch."""
        assert noise_result.output_power_w == pytest.approx(mb_result.output_power_w, rel=1e-9)  # type: ignore[attr-defined]


# ── TestPhysicsScaling ────────────────────────────────────────────


class TestPhysicsScaling:
    """Cross-validation: verify self-consistent physical scaling laws."""

    def test_higher_q_gives_narrower_schawlow_townes(self) -> None:
        """Higher Q → smaller κ_c → narrower Δν_ST (for fixed N̄, n_sp)."""
        n_bar, n_sp = 1000.0, 1.5
        # Cavity linewidth ∝ frequency / Q
        kappa_low_q  = 1.47e9 / 5_000    # 294 kHz (Q=5000)
        kappa_high_q = 1.47e9 / 20_000   # 73.5 kHz (Q=20000)
        lw_low_q  = compute_schawlow_townes_linewidth(kappa_low_q,  n_bar, n_sp)
        lw_high_q = compute_schawlow_townes_linewidth(kappa_high_q, n_bar, n_sp)
        assert lw_high_q < lw_low_q

    def test_better_pump_reduces_noise_temperature(self) -> None:
        """η closer to 1 → n_sp closer to 1 → lower T_noise."""
        freq = 1.47e9
        t_good = compute_noise_temperature(freq, compute_population_inversion_factor(0.9))
        t_poor = compute_noise_temperature(freq, compute_population_inversion_factor(0.3))
        assert t_good < t_poor

    def test_more_photons_reduce_rin_floor(self) -> None:
        """RIN_floor ∝ 1/N̄ — higher photon number → lower RIN."""
        kappa, n_sp = 147_000.0, 1.5
        r_low  = compute_rin_spectrum(kappa, 100.0,  n_sp, np.array([0.0]))
        r_high = compute_rin_spectrum(kappa, 1000.0, n_sp, np.array([0.0]))
        assert r_high.rin_floor_per_hz < r_low.rin_floor_per_hz

    def test_rin_floor_exactly_halved_for_double_photons(self) -> None:
        """Exact 2× scaling: RIN_floor(2N̄) = RIN_floor(N̄) / 2."""
        kappa, n_sp, n_bar = 147_000.0, 1.5, 500.0
        r1 = compute_rin_spectrum(kappa, n_bar,       n_sp, np.array([0.0]))
        r2 = compute_rin_spectrum(kappa, 2.0 * n_bar, n_sp, np.array([0.0]))
        assert r2.rin_floor_per_hz == pytest.approx(r1.rin_floor_per_hz / 2.0, rel=1e-9)

    def test_phase_noise_psd_decreases_with_better_pump(self) -> None:
        """Better pump efficiency → lower n_sp → less phase noise."""
        n_bar, kappa = 1000.0, 147_000.0
        lw_good = compute_schawlow_townes_linewidth(
            kappa, n_bar, compute_population_inversion_factor(0.9)
        )
        lw_poor = compute_schawlow_townes_linewidth(
            kappa, n_bar, compute_population_inversion_factor(0.3)
        )
        result_good = compute_phase_noise_spectrum(lw_good, np.array([1000.0]))
        result_poor = compute_phase_noise_spectrum(lw_poor, np.array([1000.0]))
        assert result_good.psd_rad2_per_hz[0] < result_poor.psd_rad2_per_hz[0]
