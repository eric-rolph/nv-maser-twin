"""
Maser-as-amplifier properties: magnetic Q factor, noise temperature,
and CW output power.

Implements the Wang et al. (Advanced Science, 2024) model for an NV
diamond maser operating as a low-noise microwave amplifier.  The same
physics extends to the CW oscillator regime once threshold is exceeded.

Physical picture
────────────────
The NV ensemble in the microwave cavity acts simultaneously as:
  (a) A gain medium — inverted population provides stimulated emission.
  (b) A reactive element — susceptibility shifts the cavity frequency.

The spin gain is characterised by the *magnetic quality factor* Q_m,
which quantifies how strongly the inverted spin ensemble drives the
cavity field.  The threshold condition is:

    Q_m ≤ Q_L           (loaded cavity Q)

Below this threshold (Q_m > Q_L), the device is a parametric or maser
amplifier.  Above this threshold (Q_m < Q_L), it self-oscillates.

Magnetic quality factor (Wang 2024, Eq. 1)
──────────────────────────────────────────
    Q_m^{-1} = μ₀ ℏ γ_e² σ_2 η Δn T₂ / 2

where
  μ₀  = vacuum permeability (4π × 10⁻⁷ H/m)
  ℏ   = reduced Planck constant
  γ_e = electron gyromagnetic ratio in rad/s/T
  σ_2 = 0.5 — transition matrix element for S=1 driven by π-polarised B₁
  η   = cavity filling factor (V_diamond / V_mode)
  Δn  = inverted spin density (m⁻³)
  T₂  = spin coherence time (s)

Noise temperature (Wang 2024, Eq. 4)
─────────────────────────────────────
    T_a = Q_m / (Q₀ − Q_m) × T_bath  +  Q₀ / (Q₀ − Q_m) × T_s

where
  Q₀  = unloaded cavity Q (derived from loaded Q and coupling factor β)
        Q₀ = Q_L × (1 + β)
  T_s = spin temperature = ℏ ω_c / (k_B ln(p_upper / p_lower))
  T_a → very small when T_s ≪ T_bath and Q_m ≪ Q₀

The Standard Quantum Limit for a linear phase-preserving amplifier
(Caves 1982) sets a minimum for T_a:
    T_SQL = ℏ ω_c / (2 k_B)

CW output power
───────────────
For a CW oscillator above threshold, the output power is:

    P_out = (ℏ ω_c / 2T₁) × N_eff × (p₀ − p_th) × β / (1 + β)

where p_th = 1/C is the threshold inversion fraction and β = Q₀/Q_e
is the coupling coefficient.  This reflects that each spin above
threshold contributes spin-flip energy at rate 1/T₁, and fraction
β/(1+β) leaves as useful output.

References
──────────
Wang et al., Advanced Science (2024), PMC11425272.
  - Eq. 1: magnetic Q_m formula
  - Eq. 4: amplifier noise temperature
Caves, C.M. (1982). Phys. Rev. D 26, 1817.
Breeze et al., Nature 555, 493 (2018).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import CavityConfig, MaserConfig, NVConfig
from .cavity import (
    CavityProperties,
    ThresholdResult,
    compute_cavity_properties,
    compute_full_threshold,
)


# ── Physical constants ────────────────────────────────────────────
_HBAR = 1.054571817e-34   # J·s
_KB = 1.380649e-23         # J/K
_MU0 = 1.2566370614e-6    # H/m  (= 4π × 10⁻⁷)
_GAMMA_E_RAD = 2 * math.pi * 28.025e9  # rad/s/T  (angular gyromagnetic ratio)

# Transition matrix element for S=1 driven by linearly polarized B₁ ⊥ B₀.
# Derived from ⟨S=1, ms=0 | S_x | S=1, ms=-1⟩² = 1/2 (normalised).
# Wang et al. (2024) use σ₂ = 0.5.
SIGMA_2: float = 0.5


# ── Result dataclasses ────────────────────────────────────────────

@dataclass(frozen=True)
class AmplifierProperties:
    """Wang et al. (2024) maser-as-amplifier characterisation.

    All temperatures in Kelvin, quality factors dimensionless.
    """

    # Magnetic quality factor
    magnetic_q: float
    """Q_m — spin-gain quality factor.  Lower Q_m = higher spin gain."""

    # Cavity Q factors
    loaded_q: float
    """Q_L = cavity_q from MaserConfig (includes internal + coupling loss)."""

    unloaded_q: float
    """Q₀ = Q_L × (1 + β) — internal cavity Q only."""

    # Threshold
    above_threshold: bool
    """True when Q_m ≤ Q_L (spin gain ≥ total cavity loss → masing)."""

    # Temperatures
    spin_temperature_k: float
    """T_s — spin temperature of the inverted NV ensemble (K).
    Very low (≪ 300 K) for high population inversion.
    Returns float('nan') if population fractions are unphysical.
    """

    noise_temperature_k: float
    """T_a — effective noise temperature of the maser amplifier (K).
    Computed from Wang Eq. 4 when below oscillation threshold.
    Returns float('nan') if Q_m ≈ Q₀ (at oscillation threshold).
    """

    sql_noise_temp_k: float
    """Standard Quantum Limit: T_SQL = ℏω_c / (2k_B).
    Sets the minimum achievable T_a for any phase-preserving amplifier.
    """

    below_sql: bool
    """True when T_a ≤ T_SQL (quantum-limited amplification achieved)."""


@dataclass(frozen=True)
class OutputPowerResult:
    """CW maser output power above oscillation threshold."""

    intracavity_photon_number: float
    """Estimated steady-state intracavity photon number n_ss."""

    output_power_w: float
    """P_out (W).  Zero if below threshold."""

    output_power_dbm: float
    """P_out in dBm.  −inf (represented as −999.0) if P_out = 0."""


# ── Core functions ────────────────────────────────────────────────

def compute_magnetic_q(
    nv_config: NVConfig,
    cavity_config: CavityConfig,
    maser_config: MaserConfig,
) -> float:
    r"""Compute the magnetic quality factor Q_m.

    Implements Wang et al. (2024), Eq. 1:

    .. math::

        Q_m^{-1} = \frac{\mu_0 \hbar \gamma_e^2 \sigma_2 \eta \Delta n T_2}{2}

    Args:
        nv_config:     NV spin parameters (density, pump efficiency, T₂*).
        cavity_config: Cavity geometry (fill factor).
        maser_config:  Cavity Q (not used directly — Q_m is compared to it).

    Returns:
        Magnetic quality factor Q_m (dimensionless, positive).
        Returns float('inf') if spin gain is zero.
    """
    # Inverted spin density: Δn (m⁻³)
    nv_density_m3 = nv_config.nv_density_per_cm3 * 1e6
    delta_n = (
        nv_density_m3
        * nv_config.orientation_fraction
        * nv_config.pump_efficiency
    )
    if delta_n <= 0:
        return float("inf")

    # Spin coherence time T₂ (s)
    t2_s = nv_config.t2_star_us * 1e-6

    # Cavity filling factor η (dimensionless)
    eta = cavity_config.fill_factor

    # Q_m^{-1} = μ₀ ℏ γ_e² σ₂ η Δn T₂ / 2
    inv_q_m = (
        _MU0 * _HBAR * _GAMMA_E_RAD**2 * SIGMA_2 * eta * delta_n * t2_s / 2.0
    )
    if inv_q_m <= 0:
        return float("inf")

    return 1.0 / inv_q_m


def compute_spin_temperature(
    p_upper: float,
    p_lower: float,
    cavity_frequency_ghz: float,
) -> float:
    r"""Compute the spin temperature of the NV ensemble.

    .. math::

        T_s = \frac{\hbar \omega_c}{k_B \ln(p_\text{upper} / p_\text{lower})}

    For the NV maser transition (ms=0 as upper, ms=+1 as lower):
    - Well-pumped (p_upper ≫ p_lower): T_s ≪ T_bath → near-ideal gain.
    - Thermal equilibrium (p_upper ≈ p_lower): T_s → very large.
    - No inversion (p_upper < p_lower): T_s < 0 (absorbing medium).

    Args:
        p_upper:              Population fraction in ms=0 (upper lasing state).
        p_lower:              Population fraction in ms=+1 (lower lasing state).
        cavity_frequency_ghz: Cavity (transition) frequency in GHz.

    Returns:
        Spin temperature T_s in Kelvin.
        Returns float('nan') if populations are non-positive or equal.
    """
    if p_upper <= 0.0 or p_lower <= 0.0:
        return float("nan")

    ratio = p_upper / p_lower
    if ratio == 1.0:
        return float("nan")

    omega_c = 2.0 * math.pi * cavity_frequency_ghz * 1e9  # rad/s
    t_s = _HBAR * omega_c / (_KB * math.log(ratio))
    return t_s


def compute_noise_temperature(
    magnetic_q: float,
    unloaded_q: float,
    spin_temperature_k: float,
    bath_temperature_k: float = 300.0,
) -> float:
    r"""Compute the maser amplifier noise temperature T_a.

    Implements Wang et al. (2024), Eq. 4:

    .. math::

        T_a = \frac{Q_m}{Q_0 - Q_m} T_\text{bath}
            + \frac{Q_0}{Q_0 - Q_m} T_s

    Valid when Q_m < Q₀ (amplifier regime, below oscillation threshold).
    As Q_m → Q₀, T_a diverges (onset of oscillation).

    For an ideal well-pumped maser (T_s ≪ T_bath, Q_m ≪ Q₀):
        T_a ≈ (Q_m / Q₀) × T_bath

    Args:
        magnetic_q:          Q_m from :func:`compute_magnetic_q`.
        unloaded_q:          Q₀ (unloaded cavity Q, including Q-boost).
        spin_temperature_k:  T_s (K) from :func:`compute_spin_temperature`.
        bath_temperature_k:  Physical temperature of the cavity (K).

    Returns:
        Noise temperature T_a in Kelvin.
        Returns float('nan') if Q_m ≥ Q₀ (at or above oscillation threshold).
    """
    if math.isnan(spin_temperature_k):
        return float("nan")

    denominator = unloaded_q - magnetic_q
    if denominator <= 0:
        return float("nan")  # at oscillation threshold, T_a diverges

    t_a = (
        (magnetic_q / denominator) * bath_temperature_k
        + (unloaded_q / denominator) * spin_temperature_k
    )
    return t_a


def compute_sql_noise_temperature(cavity_frequency_ghz: float) -> float:
    r"""Standard Quantum Limit for a linear phase-preserving amplifier.

    .. math::

        T_\text{SQL} = \frac{\hbar \omega_c}{2 k_B}

    Any linear amplifier satisfies T_a ≥ T_SQL (Caves 1982).

    Args:
        cavity_frequency_ghz: Cavity frequency in GHz.

    Returns:
        T_SQL in Kelvin.
    """
    omega_c = 2.0 * math.pi * cavity_frequency_ghz * 1e9
    return _HBAR * omega_c / (2.0 * _KB)


def compute_output_power(
    cavity_props: CavityProperties,
    threshold_result: ThresholdResult,
    nv_config: NVConfig,
    maser_config: MaserConfig,
) -> OutputPowerResult:
    r"""Estimate CW maser output power above oscillation threshold.

    Uses the energy-balance result for a CW two-level maser:

    .. math::

        P_\text{out} = \frac{\hbar \omega_c}{2 T_1}
            \times N_\text{eff} \times (p_0 - p_\text{th})
            \times \frac{\beta}{1 + \beta}

    where
      - :math:`p_0 = \text{pump\_efficiency}` is the steady-state inversion
        fraction,
      - :math:`p_\text{th} = 1/C` is the threshold inversion fraction,
      - :math:`\beta = \text{coupling\_beta}` is the cavity coupling coefficient.

    The factor :math:`\beta/(1+\beta)` represents the fraction of
    generated spin-flip power that exits through the output port rather
    than being dissipated internally.

    Args:
        cavity_props:     Pre-computed cavity mode quantities.
        threshold_result: Cooperativity and N_eff (from threshold calculation).
        nv_config:        NV parameters (T₁, pump efficiency).
        maser_config:     Cavity Q and coupling coefficient.

    Returns:
        :class:`OutputPowerResult`.  ``output_power_w = 0`` if below threshold.
    """
    if not threshold_result.masing:
        return OutputPowerResult(
            intracavity_photon_number=0.0,
            output_power_w=0.0,
            output_power_dbm=-999.0,
        )

    omega_c = 2.0 * math.pi * maser_config.cavity_frequency_ghz * 1e9  # rad/s

    # Threshold inversion fraction p_th = 1/C
    cooperativity = threshold_result.cooperativity
    p_threshold = 1.0 / cooperativity if cooperativity > 0 else 1.0

    # Operating inversion fraction p₀ = pump_efficiency
    p0 = nv_config.pump_efficiency
    inversion_above_threshold = max(0.0, p0 - p_threshold)

    # T₁ in seconds
    t1_s = nv_config.t1_ms * 1e-3

    # Coupling coefficient β and output fraction
    beta = maser_config.coupling_beta
    output_fraction = beta / (1.0 + beta) if (1.0 + beta) > 0 else 0.0

    # CW output power (W)
    n_eff = threshold_result.n_effective
    if t1_s <= 0 or n_eff <= 0:
        p_out_w = 0.0
    else:
        p_out_w = (
            (_HBAR * omega_c / (2.0 * t1_s))
            * n_eff
            * inversion_above_threshold
            * output_fraction
        )

    # Intracavity photon number: n_ss ≈ P_out / (ℏω_c × κ_ext)
    # κ_ext = ω_c × β / (Q_L × (1+β)) = ω_c / Q₀ × β/(1+β) ... wait
    # Actually κ_ext = coupling portion of κ_total:
    #   κ_total = ω_c / Q_L
    #   κ_ext   = κ_total × β/(1+β)
    kappa_total = omega_c / maser_config.cavity_q
    kappa_ext = kappa_total * output_fraction
    if kappa_ext > 0 and p_out_w > 0:
        n_ss = p_out_w / (_HBAR * omega_c * kappa_ext)
    else:
        n_ss = 0.0

    # Convert to dBm
    if p_out_w > 0:
        p_dbm = 10.0 * math.log10(p_out_w / 1e-3)
    else:
        p_dbm = -999.0

    return OutputPowerResult(
        intracavity_photon_number=n_ss,
        output_power_w=p_out_w,
        output_power_dbm=p_dbm,
    )


# ── Convenience helpers ───────────────────────────────────────────

def _derive_population_fractions(nv_config: NVConfig) -> tuple[float, float]:
    """Derive upper- and lower-level population fractions from NV config.

    For the NV maser transition (ms=0 upper, ms=+1 lower):
    - Under optical pumping: ms=0 accumulates with efficiency η_pump.
    - The remaining population is distributed among ms=+1 and ms=−1.

    Returns:
        (p_upper, p_lower) as fractions summing to ≤ 1.
    """
    pump_eff = nv_config.pump_efficiency
    # Upper state (ms=0) fraction
    p_upper = pump_eff
    # Lower state (ms=+1) gets half of non-pumped population
    p_lower = (1.0 - pump_eff) / 2.0
    return p_upper, p_lower


def compute_amplifier_properties(
    nv_config: NVConfig,
    cavity_config: CavityConfig,
    maser_config: MaserConfig,
    bath_temperature_k: float = 300.0,
) -> AmplifierProperties:
    """Full Wang et al. (2024) maser-as-amplifier characterisation.

    Computes Q_m, spin temperature, noise temperature, SQL comparison,
    and oscillation threshold status from the NV and cavity configuration.

    Args:
        nv_config:          NV spin parameters.
        cavity_config:      Cavity geometry.
        maser_config:       Cavity Q, coupling coefficient β.
        bath_temperature_k: Physical temperature (K).

    Returns:
        :class:`AmplifierProperties` with all amplifier metrics.
    """
    # ── Q factors ──────────────────────────────────────────────────
    q_loaded = maser_config.cavity_q            # Q_L (loaded)
    beta = maser_config.coupling_beta
    q_unloaded = q_loaded * (1.0 + beta)         # Q₀ = Q_L (1+β)

    # ── Magnetic quality factor ────────────────────────────────────
    q_m = compute_magnetic_q(nv_config, cavity_config, maser_config)

    # Oscillation threshold: Q_m ≤ Q_L
    above_threshold = (q_m <= q_loaded) and not math.isinf(q_m)

    # ── Spin temperature ───────────────────────────────────────────
    p_upper, p_lower = _derive_population_fractions(nv_config)
    t_spin = compute_spin_temperature(
        p_upper, p_lower, maser_config.cavity_frequency_ghz
    )

    # ── Standard Quantum Limit ─────────────────────────────────────
    t_sql = compute_sql_noise_temperature(maser_config.cavity_frequency_ghz)

    # ── Noise temperature ──────────────────────────────────────────
    # Wang Eq. 4 valid in amplifier regime (Q_m > Q_L → Q_m < Q₀).
    t_noise = compute_noise_temperature(
        q_m, q_unloaded, t_spin, bath_temperature_k
    )

    # ── SQL comparison ─────────────────────────────────────────────
    if math.isnan(t_noise):
        below_sql = False
    else:
        below_sql = t_noise <= t_sql

    return AmplifierProperties(
        magnetic_q=q_m,
        loaded_q=q_loaded,
        unloaded_q=q_unloaded,
        above_threshold=above_threshold,
        spin_temperature_k=t_spin,
        noise_temperature_k=t_noise,
        sql_noise_temp_k=t_sql,
        below_sql=below_sql,
    )
