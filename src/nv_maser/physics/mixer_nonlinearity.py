"""physics/mixer_nonlinearity.py — Mixer IMD3 intermodulation distortion model (R9).

Background
----------
The NV-maser up-conversion chain (§7.1, step 5) mixes the NMR signal
(~2.13 MHz) with a local oscillator (~1.4678 GHz) to produce an upper
sideband at the maser centre (~1.4699 GHz).  Any RF interference present
at the coil — Wi-Fi, LTE, Bluetooth — travels through the same signal
path and enters the mixer's RF port.

While the maser cavity (step 6) provides > 80 dB Lorentzian bandpass
rejection of *individual* OOB signals (quantified in ``rf_rejection.py``),
a *pair* of strong OOB signals can interact inside the mixer's nonlinear
transfer function to produce **third-order intermodulation distortion
(IMD3)** products at new frequencies that may coincidentally land *inside*
the maser gain band — bypassing the passive bandpass protection entirely.

IMD3 mechanic
-------------
For a weakly-nonlinear system y = a₁x + a₃x³, two input tones at f₁ and
f₂ produce spurious output products at:

    f_IM3⁺ = 2f₁ − f₂          (lower IM3 sideband)
    f_IM3⁻ = 2f₂ − f₁          (upper IM3 sideband)

The power of the 2f₁−f₂ product (referred to the mixer input) is:

    P_IM3(2f₁−f₂) [dBm]  =  2·P₁  +  P₂  −  2·IIP3         (1)

And symmetrically for 2f₂−f₁:

    P_IM3(2f₂−f₁) [dBm]  =  P₁   +  2·P₂  −  2·IIP3        (2)

where P₁, P₂ are the input powers (dBm) and IIP3 is the input-referred
third-order intercept point of the mixer (dBm).  For equal tones
(P₁ = P₂ = P) both formulae reduce to the standard result:
P_IM3 = 3P − 2·IIP3.

Reference: Pozar, "Microwave Engineering", 4th ed., §10.3;
           Razavi, "Design of Analog CMOS Circuits", §B.3.

IMD3 in-band condition
-----------------------
An IMD3 product is *dangerous* when its frequency falls inside the maser
gain window:

    |f_IM3 − f_maser| < BW_maser / 2

Because the maser amplifies in-band signals by 30–60 dB before the
downstream chain, an in-band IMD3 product at power P would emerge from
the maser at P + G_maser — regardless of the maser's Lorentzian rejection
of the *original* interferers.

Practical finding (default 8-interferer hospital environment)
-------------------------------------------------------------
With the default interferer set from ``rf_rejection.py`` — all far from
1.4699 GHz — no unordered pair produces an IMD3 product within the maser
49 kHz gain band.  The closest approach is ~75 MHz offset.  However, if
any future interferer or spurious LO harmonic places two tones near the
maser centre, this model will flag the in-band threat and quantify the
power.

Risk register R9 context
------------------------
Architecture doc §13:
    "Up-conversion mixer adds noise / IMD → corrupts NMR signal — Medium
    impact — Use low-noise mixer; IMD characterisation required."

The companion module ``up_conversion.py`` covers the Friis thermal-noise
contribution (R9 partial).  This module covers the nonlinear IMD3
contribution, completing R9 coverage.

References
----------
* Architecture doc §7.1 (step 5, up-conversion), §13 (Risk R9)
* Pozar — "Microwave Engineering", 4th ed. (2012), §10.3.
* Razavi — "Design of Analog CMOS Circuits", 2nd ed. (2001).
* ``up_conversion.py`` — DEFAULT_MIXER with ip3_dbm = 5.0 dBm used as default.
* ``rf_rejection.py`` — InterfererSpec and default 8-interferer set reused.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from .rf_rejection import InterfererSpec, _DEFAULT_INTERFERERS


# ── defaults from up_conversion.DEFAULT_MIXER (avoid circular import) ─────────
_DEFAULT_IIP3_DBM: float = 5.0   # default from DEFAULT_MIXER.ip3_dbm = 5.0

# Product type tags (2f1-f2 = "lower IM3"; 2f2-f1 = "upper IM3")
PRODUCT_2F1_MINUS_F2 = "2f1-f2"
PRODUCT_2F2_MINUS_F1 = "2f2-f1"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Data classes                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class IMD3Product:
    """IMD3 spurious product from a single interferer pair.

    Attributes
    ----------
    interferer_1:
        First input tone (f₁, P₁).
    interferer_2:
        Second input tone (f₂, P₂).
    product_type:
        ``"2f1-f2"`` or ``"2f2-f1"`` — which mixing product this is.
    product_freq_hz:
        Frequency of the IMD3 product (Hz).  May be negative for
        (2f₁−f₂) when f₂ > 2f₁; such products are physically absent and
        flagged via ``is_physical = False``.
    is_physical:
        ``True`` if ``product_freq_hz > 0``.  Negative results indicate
        the mixing product would fall at a negative (non-physical) frequency
        and cannot appear in band.
    product_power_dbm:
        IMD3 output power (dBm), computed from Eq. (1) or (2).
        Only meaningful when ``is_physical`` is ``True``.
    freq_offset_from_maser_hz:
        Absolute frequency offset of this product from the maser centre.
    in_maser_band:
        ``True`` if this product is physical AND its frequency lies within
        the maser gain passband (±BW_maser/2 of centre).
    """

    interferer_1: InterfererSpec
    interferer_2: InterfererSpec
    product_type: str
    product_freq_hz: float
    is_physical: bool
    product_power_dbm: float
    freq_offset_from_maser_hz: float
    in_maser_band: bool


@dataclass(frozen=True)
class MixerNonlinearityConfig:
    """Configuration for the mixer IMD3 nonlinearity model.

    Attributes
    ----------
    iip3_dbm:
        Mixer input-referred third-order intercept point (dBm).
        Default 5.0 dBm matches ``DEFAULT_MIXER.ip3_dbm`` in
        ``up_conversion.py``.
    maser_center_hz:
        Maser gain-profile centre frequency (Hz).  Default 1.4699 GHz.
    maser_gain_bw_hz:
        Maser gain FWHM (Hz).  Default 49 000 Hz (loaded Q ≈ 30 000).
        A product within ±BW/2 of centre is marked ``in_maser_band``.
    interferers:
        Tuple of :class:`~rf_rejection.InterfererSpec` objects.
        Default: the same 8-source hospital-environment set from
        ``rf_rejection.py``.
    """

    iip3_dbm: float = _DEFAULT_IIP3_DBM
    maser_center_hz: float = 1.4699e9
    maser_gain_bw_hz: float = 49_000.0
    interferers: tuple[InterfererSpec, ...] = field(
        default_factory=lambda: tuple(
            InterfererSpec(name=n, center_freq_hz=f, bandwidth_hz=bw, power_dbm=p)
            for n, f, bw, p in _DEFAULT_INTERFERERS
        )
    )

    def __post_init__(self) -> None:
        if self.maser_center_hz <= 0:
            raise ValueError(
                f"maser_center_hz must be positive, got {self.maser_center_hz}"
            )
        if self.maser_gain_bw_hz <= 0:
            raise ValueError(
                f"maser_gain_bw_hz must be positive, got {self.maser_gain_bw_hz}"
            )


@dataclass(frozen=True)
class MixerNonlinearityResult:
    """Aggregate IMD3 analysis result.

    Attributes
    ----------
    imd3_products:
        All IMD3 products evaluated (physical and non-physical).
        Length = n_pairs × 2.
    n_pairs_evaluated:
        Number of unique (unordered) interferer pairs evaluated.
        For N interferers this is C(N,2) = N(N−1)/2.
    n_products_evaluated:
        Total number of IMD3 products (2 per pair).
    any_in_band:
        ``True`` if any physical IMD3 product falls inside the maser
        gain passband.
    worst_in_band_power_dbm:
        Power of the strongest in-band IMD3 product (dBm), or
        ``−∞`` if ``any_in_band`` is ``False``.
    in_band_products:
        Tuple of all in-band IMD3 products (empty if none).
    max_imd3_power_dbm:
        Highest IMD3 product power across ALL physical products
        (in-band or not).  Useful as a mixer linearity figure of merit.
    """

    imd3_products: tuple[IMD3Product, ...]
    n_pairs_evaluated: int
    n_products_evaluated: int
    any_in_band: bool
    worst_in_band_power_dbm: float
    in_band_products: tuple[IMD3Product, ...]
    max_imd3_power_dbm: float


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Core computation functions                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def compute_imd3_power_dbm(
    p1_dbm: float,
    p2_dbm: float,
    iip3_dbm: float,
    product_type: str = PRODUCT_2F1_MINUS_F2,
) -> float:
    """IMD3 output power for a two-tone input (dBm).

    Applies the standard RF two-tone IMD3 formula:

    * ``"2f1-f2"`` product:  P_IM3 = 2·P₁ + P₂ − 2·IIP3
    * ``"2f2-f1"`` product:  P_IM3 = P₁ + 2·P₂ − 2·IIP3

    All quantities in dBm.  For equal tones (P₁ = P₂ = P) both
    reduce to the equal-tone result P_IM3 = 3P − 2·IIP3.

    Parameters
    ----------
    p1_dbm:
        Power of tone 1 at the mixer input (dBm).
    p2_dbm:
        Power of tone 2 at the mixer input (dBm).
    iip3_dbm:
        Mixer input-referred third-order intercept (dBm).
    product_type:
        Which product to compute.  Must be ``PRODUCT_2F1_MINUS_F2``
        or ``PRODUCT_2F2_MINUS_F1``.

    Returns
    -------
    float
        IMD3 product power (dBm).

    Raises
    ------
    ValueError
        If *product_type* is not recognized.
    """
    if product_type == PRODUCT_2F1_MINUS_F2:
        return 2.0 * p1_dbm + p2_dbm - 2.0 * iip3_dbm
    if product_type == PRODUCT_2F2_MINUS_F1:
        return p1_dbm + 2.0 * p2_dbm - 2.0 * iip3_dbm
    raise ValueError(
        f"product_type must be '{PRODUCT_2F1_MINUS_F2}' or "
        f"'{PRODUCT_2F2_MINUS_F1}', got {product_type!r}"
    )


def compute_imd3_frequency_hz(
    f1_hz: float,
    f2_hz: float,
    product_type: str = PRODUCT_2F1_MINUS_F2,
) -> float:
    """IMD3 product frequency for a two-tone input (Hz).

    * ``"2f1-f2"`` product: f_IM3 = 2·f₁ − f₂
    * ``"2f2-f1"`` product: f_IM3 = 2·f₂ − f₁

    Parameters
    ----------
    f1_hz:
        Frequency of tone 1 (Hz).
    f2_hz:
        Frequency of tone 2 (Hz).
    product_type:
        Which product to compute.

    Returns
    -------
    float
        IMD3 product frequency (Hz).  May be negative; callers should
        check ``is_physical`` on the returned :class:`IMD3Product`.

    Raises
    ------
    ValueError
        If *product_type* is not recognized.
    """
    if product_type == PRODUCT_2F1_MINUS_F2:
        return 2.0 * f1_hz - f2_hz
    if product_type == PRODUCT_2F2_MINUS_F1:
        return 2.0 * f2_hz - f1_hz
    raise ValueError(
        f"product_type must be '{PRODUCT_2F1_MINUS_F2}' or "
        f"'{PRODUCT_2F2_MINUS_F1}', got {product_type!r}"
    )


def compute_imd3_pair(
    spec1: InterfererSpec,
    spec2: InterfererSpec,
    config: MixerNonlinearityConfig,
) -> tuple[IMD3Product, IMD3Product]:
    """Compute both IMD3 products from an ordered/unordered interferer pair.

    Evaluates the ``"2f1-f2"`` and ``"2f2-f1"`` mixing products and
    determines whether each falls inside the maser gain passband.

    Parameters
    ----------
    spec1:
        First interferer (plays the role of f₁ in the formulae).
    spec2:
        Second interferer (plays the role of f₂).
    config:
        Maser and mixer parameters.

    Returns
    -------
    tuple[IMD3Product, IMD3Product]
        Two-element tuple: ``(product_2f1_f2, product_2f2_f1)``.
    """
    half_bw = config.maser_gain_bw_hz / 2.0

    def _make(ptype: str) -> IMD3Product:
        f_prod = compute_imd3_frequency_hz(
            spec1.center_freq_hz, spec2.center_freq_hz, ptype
        )
        # p1_dbm = power of tone at f1 (spec1); p2_dbm = power of tone at f2 (spec2).
        # compute_imd3_power_dbm routes the formula by product_type — no swap needed.
        p_prod = compute_imd3_power_dbm(
            p1_dbm=spec1.power_dbm,
            p2_dbm=spec2.power_dbm,
            iip3_dbm=config.iip3_dbm,
            product_type=ptype,
        )
        is_phys = f_prod > 0.0
        offset = abs(f_prod - config.maser_center_hz) if is_phys else float("inf")
        in_band = is_phys and (offset < half_bw)
        return IMD3Product(
            interferer_1=spec1,
            interferer_2=spec2,
            product_type=ptype,
            product_freq_hz=f_prod,
            is_physical=is_phys,
            product_power_dbm=p_prod,
            freq_offset_from_maser_hz=offset,
            in_maser_band=in_band,
        )

    prod_a = _make(PRODUCT_2F1_MINUS_F2)
    prod_b = _make(PRODUCT_2F2_MINUS_F1)
    return prod_a, prod_b


def compute_mixer_nonlinearity(
    config: MixerNonlinearityConfig | None = None,
) -> MixerNonlinearityResult:
    """Compute IMD3 products for all interferer pairs.

    Iterates over every unique (unordered) pair of interferers in
    *config.interferers*, computes both IMD3 products per pair, and
    aggregates the results.

    Parameters
    ----------
    config:
        Mixer nonlinearity configuration.  If ``None``, the default
        :class:`MixerNonlinearityConfig` (8 hospital-environment
        interferers, IIP3 = 5 dBm) is used.

    Returns
    -------
    MixerNonlinearityResult
        Aggregate IMD3 analysis.
    """
    if config is None:
        config = MixerNonlinearityConfig()

    interferers = list(config.interferers)
    all_products: list[IMD3Product] = []

    for s1, s2 in itertools.combinations(interferers, 2):
        pa, pb = compute_imd3_pair(s1, s2, config)
        all_products.extend([pa, pb])

    n_pairs = len(interferers) * (len(interferers) - 1) // 2
    in_band = [p for p in all_products if p.in_maser_band]
    physical = [p for p in all_products if p.is_physical]

    worst_in_band_power = (
        max(p.product_power_dbm for p in in_band) if in_band else float("-inf")
    )
    max_imd3_power = (
        max(p.product_power_dbm for p in physical) if physical else float("-inf")
    )

    return MixerNonlinearityResult(
        imd3_products=tuple(all_products),
        n_pairs_evaluated=n_pairs,
        n_products_evaluated=len(all_products),
        any_in_band=bool(in_band),
        worst_in_band_power_dbm=worst_in_band_power,
        in_band_products=tuple(in_band),
        max_imd3_power_dbm=max_imd3_power,
    )
