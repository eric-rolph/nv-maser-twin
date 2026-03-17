# Portable NV-Maser MRI: Systems Engineering Design Review

**Date**: 2026-03-17  
**Author**: Digital Twin + Hardware Engineering Review  
**Status**: DRAFT — requires expert physics review before hardware commitment  
**Scope**: First-principles analysis of a $20–40K portable MRI device enabled by room-temperature NV-diamond maser detection

---

## Table of Contents

1. [Mission Definition](#1-mission-definition)
2. [How MRI Actually Works (What We Must Build)](#2-how-mri-actually-works)
3. [The NV Maser Value Proposition](#3-the-nv-maser-value-proposition)
4. [System Architecture](#4-system-architecture)
5. [Critical Signal Chain: Tissue → Image](#5-critical-signal-chain)
6. [Subsystem Requirements Flowdown](#6-subsystem-requirements-flowdown)
7. [Gap Analysis: Current Twin vs. MRI Requirements](#7-gap-analysis)
8. [Design Trade Studies](#8-design-trade-studies)
9. [BOM & Cost Analysis](#9-bom--cost-analysis)
10. [Risk Register](#10-risk-register)
11. [Regulatory & Safety](#11-regulatory--safety)
12. [Recommended Development Path](#12-recommended-development-path)

---

## 1. Mission Definition

### 1.1 Product Intent

A portable, room-temperature MRI device for **extremity imaging** (hand, wrist, knee, ankle, foot) in settings where conventional MRI is unavailable:

| Setting | Constraint | Implication |
|---------|-----------|-------------|
| **EMS vehicle** | Moving platform, limited power (12V/24V DC + inverter), vibration | Vibration isolation, compact, low peak power |
| **Small clinic** | No RF-shielded room, limited floor space, no cryogenics | Self-shielded magnet, integrated Faraday cage |
| **Field hospital / military** | Air-transportable, ruggedised, rapid setup | < 50 kg, pelican-case form factor |
| **Veterinary** | Animal extremities, varying sizes | Adjustable bore or single-sided geometry |

### 1.2 Performance Targets

| Parameter | Target | Rationale |
|-----------|--------|-----------|
| **Retail price** | $20,000–40,000 | 10–50× below conventional MRI ($500K–3M); competitive with ultrasound high-end |
| **Weight** | < 50 kg total system (< 25 kg sensor head) | One-person lift for sensor head; cart for electronics |
| **Image resolution** | 1–2 mm in-plane, 3–5 mm slice | Adequate for fracture, soft-tissue injury, joint assessment |
| **FOV** | ≥ 12 cm (extremity cross-section) | Knee = ~15 cm, wrist = ~8 cm |
| **Scan time** | 2–10 minutes per sequence | Tolerable for cooperative patient; competitive with POC ultrasound workflow |
| **Power** | < 2 kW peak, < 500 W average | Compatible with standard 15A/120V outlet or vehicle inverter |
| **Setup time** | < 5 minutes | Place extremity → scan → image |
| **Image contrast** | T1, T2, proton density (basic sequences) | Spin echo, gradient echo minimum |

### 1.3 Comparison: Existing Low-Field MRI

| Device | B₀ | Weight | Price | FOV | Resolution | FDA |
|--------|----|--------|-------|-----|------------|-----|
| **Hyperfine Swoop** | 64 mT | 640 kg | ~$50K (lease) | Brain | 1.5×1.5×5 mm | 510(k) cleared |
| **Promaxo** | 64 mT | ~200 kg | ~$250K | Prostate | ~2 mm | 510(k) cleared |
| **Our target** | 50–80 mT | < 50 kg | $20–40K | Extremity | 1–2 mm | 510(k) TBD |

The key differentiation: **maser-enhanced SNR** allows comparable image quality at lower field strength with a much smaller, lighter, cheaper magnet. If the maser delivers 10–20 dB SNR improvement over conventional receive, we can radically shrink the hardware.

---

## 2. How MRI Actually Works (What We Must Build)

### 2.1 The Five Essential Subsystems of ANY MRI

An MRI—regardless of field strength—requires all five of these to produce an image. Missing any one means no image.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MRI SIGNAL PATH                                     │
│                                                                              │
│  ① B₀ MAGNET          ② RF EXCITATION       ③ GRADIENT ENCODING             │
│  Aligns proton         Tilts proton spin      Spatially encodes              │
│  magnetization         away from B₀           the NMR signal                 │
│  (Boltzmann            (Larmor freq RF        (3-axis linear                 │
│   equilibrium)          pulse at ω₀)           gradients for                 │
│                                                slice/phase/freq)             │
│          ↓                    ↓                       ↓                      │
│  ┌───────────────────────────────────────────────────────┐                   │
│  │  TISSUE: Protons precess → relax → emit NMR signal   │                   │
│  │  Signal ∝ M₀ × sin(α) × exp(-TE/T2) × exp(-TR/T1)   │                   │
│  └───────────────────────────────────────────────────────┘                   │
│          ↓                                                                   │
│  ④ RF RECEIVE (+ MASER PREAMPLIFIER)                                        │
│  Detect the weak NMR                                                         │
│  signal (~nV to µV)                                                          │
│          ↓                                                                   │
│  ⑤ RECONSTRUCTION                                                            │
│  k-space → image                                                             │
│  (FFT + model-based)                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 What the Current Digital Twin Models vs. What's Missing

| MRI Subsystem | Current Twin Coverage | Status |
|---------------|----------------------|--------|
| ① **B₀ magnet** (Halbach) | ✅ Full multipole model, manufacturing tolerances | Modeled, but 14 mm bore |
| ② **RF excitation** | ❌ **NOT MODELED** | No transmit coil, no pulse sequences |
| ③ **Gradient encoding** | ❌ **NOT MODELED** (shimming ≠ imaging gradients) | Shimming coils correct static errors; imaging gradients must switch dynamically |
| ④ **RF receive + maser** | ✅ Maser gain, signal chain, SNR | Core strength of twin |
| ⑤ **Reconstruction** | ❌ **NOT MODELED** | No k-space, no image formation |

**The current twin models the maser sensor physics.** It does NOT model the complete MRI imaging chain. This is the largest gap.

### 2.3 The MRI Signal Equation (What Determines Image Quality)

For a single voxel at position **r**, the detected NMR signal is:

$$S(t) = \int_V M_0(\mathbf{r}) \cdot \sin\alpha \cdot e^{-TE/T_2(\mathbf{r})} \cdot e^{-i\gamma \mathbf{G}\cdot\mathbf{r}\,t} \, d^3\mathbf{r}$$

**Key dependencies**:

| Factor | Controls | Our Lever |
|--------|----------|-----------|
| $M_0 \propto \rho_H B_0 / T$ | Equilibrium magnetization | Higher B₀ = more signal; $M_0$ at 50 mT is **30× weaker** than at 1.5T |
| $\sin\alpha$ | Flip angle (RF pulse) | RF coil + amplifier design |
| $e^{-TE/T_2}$ | T2 contrast weighting | Pulse sequence timing |
| $e^{-i\gamma \mathbf{G}\cdot\mathbf{r}\,t}$ | Spatial encoding | Gradient coil performance |
| **SNR** ∝ $B_0 \cdot V_{voxel} \cdot \sqrt{N_{avg}} / \sqrt{T_{noise}}$ | Image quality | **Maser drops $T_{noise}$; compensates low $B_0$** |

### 2.4 The SNR Crisis at Low Field — And Why the Maser Matters

SNR in conventional MRI scales roughly as:

$$\text{SNR} \propto B_0^{7/4} \cdot V_{voxel} \cdot \sqrt{N_{avg} \cdot BW^{-1}} \cdot \frac{1}{\sqrt{R_{coil} + R_{body}}}$$

At **50 mT** vs. **1.5T**:

$$\frac{\text{SNR}_{50\text{mT}}}{\text{SNR}_{1.5\text{T}}} = \left(\frac{0.05}{1.5}\right)^{7/4} = \left(\frac{1}{30}\right)^{1.75} \approx \frac{1}{480}$$

**Conventional approach to recover 480× SNR loss:**  
- Average 480² ≈ 230,000 times → scan time: years (impossible)  
- Increase voxel volume 480× → resolution: ~15 mm (useless)

**Maser approach:**  
A room-temperature maser with noise temperature $T_n \approx 1$ K (vs. $T_{body} \approx 300$ K) provides:

$$\text{SNR gain} = \sqrt{\frac{T_{conventional}}{T_{maser}}} = \sqrt{\frac{300}{1}} \approx 17 \times \quad (24 \text{ dB})$$

Remaining gap after maser: $480 / 17 \approx 28×$  
Recovered by: larger voxels (2 mm vs. 0.5 mm → 64×) + moderate averaging (4–16 averages)

**This is the fundamental physics argument that makes maser-MRI viable.** Without the maser, 50 mT MRI at clinical resolution requires either:
- Cryogenic SQUID detectors (expensive, defeats portable goal)
- Impractically long scan times (hours)
- Unacceptably coarse resolution (>5 mm)

---

## 3. The NV Maser Value Proposition

### 3.1 Operating Principle as MRI Preamplifier

The NV maser does NOT directly amplify the 2 MHz NMR signal. Instead:

```
Tissue NMR precession (oscillating B-field at ~2.1 MHz)
        ↓
   Pickup coil (tuned to ~2.1 MHz) converts to voltage
        ↓
   [Option A: Direct detection]  OR  [Option B: Maser-enhanced detection]
```

**Option A (conventional):** Coil → room-temp LNA → ADC  
Noise temp ≈ 300 K (body noise) + 75 K (amplifier) = 375 K

**Option B (maser):** Coil → frequency up-conversion mixer → maser amplifier at 1.47 GHz → ADC  
Noise temp ≈ body noise (unavoidable) + maser noise (~1–10 K)

Wait — there's a subtlety. **At 2 MHz (low frequency), the dominant noise source is the body itself** (conductive tissue acts as a lossy antenna). The coil noise is secondary. So the maser cannot reduce body noise.

### 3.2 Honest SNR Assessment (Critical)

At **low frequencies** (< 10 MHz, i.e., low-field MRI), the noise is **body-dominated**, not **amplifier-dominated**:

$$\text{SNR} = \frac{V_{signal}}{\sqrt{4 k_B T_{eff} R_{eff} \Delta f}}$$

where $R_{eff} = R_{coil} + R_{body}$

| B₀ | Larmor freq | $R_{coil}$ | $R_{body}$ | Dominant noise | Maser benefit |
|----|-------------|------------|-----------|---------------|---------------|
| 1.5 T | 64 MHz | 0.5 Ω | 3–10 Ω | **Body** | Minimal (body dominates) |
| 64 mT | 2.7 MHz | 0.3 Ω | 0.01–0.1 Ω | **Coil** | **SIGNIFICANT** |
| 50 mT | 2.1 MHz | 0.2 Ω | 0.005–0.05 Ω | **Coil** | **VERY SIGNIFICANT** |

**KEY INSIGHT**: At ultra-low field (< 100 mT), body noise is negligible because $R_{body} \propto \omega^2$ (skin depth is very large). The dominant noise is the **receive coil itself** and the **preamplifier**. This is exactly where a quantum-limited preamplifier (maser) provides maximum benefit.

### 3.3 Revised SNR Gain Estimate

At 50 mT, assuming coil-dominated noise regime:

| Noise source | Conventional | Maser-enhanced |
|-------------|-------------|----------------|
| Coil thermal (300 K, R=0.2 Ω) | 0.2 × 4kT = 3.3×10⁻²¹ W/Hz | Same coil noise (unavoidable) |
| Preamp noise temp | 75 K (NF=1 dB LNA) | 1–10 K (maser) |
| **System noise temp** | **375 K** | **301–310 K** |
| SNR improvement | baseline | **~1.1× (only 10%)** |

**Wait — this is concerning.** If the coil is at 300 K and the maser only replaces the preamp noise, the improvement is modest (~10%) because $T_{coil} = 300$ K >> $T_{preamp}$.

### 3.4 The Real Maser Advantage: Parametric Amplification & Noise Squeezing

The above analysis assumes the maser is a simple low-noise amplifier. But a **maser in the stimulated emission regime** can do more:

1. **Near-quantum-limited noise**: The added noise of an ideal maser amplifier is $n_{add} = 0.5$ photons (quantum limit). At 1.47 GHz: $T_{quantum} = h\nu/k_B = 0.07$ K. This is far below the coil temperature.

2. **The real trick: Couple the maser directly to the NMR receive coil.** If the maser crystal is positioned so that the NV spins are magnetically coupled to the same mode as the MRI receive coil, the maser provides **coherent gain** to the NMR signal before thermal noise is added.

3. **Alternatively: Use the NV ensemble as the receive element itself.** Instead of a traditional inductive coil, use the NV spins as a quantum sensor. The NMR signal from tissue modulates the NV transition frequency → detected via ODMR or maser frequency shift.

**Architecture B (direct NV sensing):**
```
Tissue NMR signal (oscillating B at ~2.1 MHz at sensor location)
        ↓
   NV ensemble (magnetically sensitive, positioned near tissue)
        ↓
   Maser frequency shift (FM modulation on 1.47 GHz carrier)
        ↓
   FM demodulator → NMR signal recovery
        ↓
   Sensitivity: < 1 pT/√Hz at room temperature (published NV magnetometry)
```

**Published NV magnetometry sensitivity** (Balasubramanian et al., Barry et al.):
- DC: ~1 pT/√Hz for mm³ diamond volumes
- AC (narrowband near specific freq): ~10 fT/√Hz achievable with T₂* ≈ 1 µs ensembles

**Required NMR signal strength for detectable image** at 50 mT:
- Voxel: 2×2×5 mm³ = 20 µL of water
- Proton density: 6.7×10²⁸ m⁻³
- Equilibrium magnetization: $M_0 = \chi_0 B_0 / \mu_0 \approx 1.7 \times 10^{-8}$ A/m (at 50 mT)
- Precessing signal at 1 cm distance: $B_{signal} \sim \mu_0 M_0 V / (4\pi r^3) \approx 0.3$ fT

**This is at the edge of NV ensemble sensitivity.** With a 1 mm³ diamond at 10 fT/√Hz, detecting 0.3 fT requires:
$$t_{avg} = \left(\frac{\text{noise floor}}{\text{signal}}\right)^2 = \left(\frac{10}{0.3}\right)^2 \approx 1100 \text{ seconds}$$

**Per voxel.** For a 64×64 image: prohibitive.

### 3.5 Revised Architecture: Inductive Detection + Maser Preamp

The direct NV sensing approach is too slow for imaging. The viable path uses **conventional inductive detection** (pickup coil) with the maser providing parametric gain:

```
┌──────────────────────────────────────────────────────────────────┐
│                     VIABLE MRI ARCHITECTURE                       │
│                                                                   │
│  Tissue → RF pickup coil (tuned ~2 MHz) → matching network       │
│               ↓                                                   │
│  Up-conversion mixer (2 MHz → 1.47 GHz using local oscillator)   │
│               ↓                                                   │
│  NV maser amplifier (1.47 GHz, gain ~20 dB, T_n ~ 1 K)          │
│               ↓                                                   │
│  Down-conversion → baseband ADC → k-space → image                │
│                                                                   │
│  KEY: The up-conversion places the NMR signal inside the maser   │
│  bandwidth where quantum-limited amplification occurs BEFORE      │
│  thermal noise from downstream electronics is added.              │
└──────────────────────────────────────────────────────────────────┘
```

**Effective system noise temperature with up-conversion + maser:**

$$T_{sys} = T_{coil} + T_{mixer}/G_{mixer} + T_{maser}/G_{mixer} \approx T_{coil} + T_{maser}$$

If we **cool the receive coil** (even to liquid nitrogen, 77 K — far cheaper than liquid helium):

$$T_{sys} = 77 + 1 = 78 \text{ K} \quad \text{(vs. 375 K conventional)}$$

$$\text{SNR gain} = \sqrt{375/78} \approx 2.2 \times \quad (7 \text{ dB})$$

With a **superconducting receive coil at LN₂ temp** ($R_{coil} \to 0$, $T_{coil} \to$ a few K):

$$T_{sys} = 5 + 1 = 6 \text{ K} \quad \text{SNR gain} = \sqrt{375/6} \approx 7.9 \times \quad (18 \text{ dB})$$

But using LN₂ compromises the "portable" goal somewhat. **HTS (high-temperature superconductor) coils at 77 K** are a realistic middle ground — LN₂ is cheap, safe, and requires only a small dewar.

### 3.6 Sensitivity Budget Summary

| Configuration | $T_{sys}$ (K) | SNR gain vs. RT conventional | Scan time reduction |
|--------------|---------------|------------------------------|---------------------|
| Room-temp coil + room-temp LNA | 375 | 1.0× (baseline) | 1× |
| Room-temp coil + maser preamp | 301 | 1.12× | 0.80× |
| **LN₂-cooled Cu coil + maser** | **78** | **2.2×** | **0.21×** |
| **HTS coil (77K) + maser** | **6** | **7.9×** | **0.016×** |
| SQUID (4K, liquid He) | 0.1 | 61× | impractical cost |

**Recommendation**: The **LN₂-cooled copper coil + maser** configuration provides the best cost/benefit ratio for a portable device. It gives ~2× SNR gain, which translates to ~5× scan time reduction or ~1.5× resolution improvement. The HTS option gives much more but adds $5–15K to BOM.

---

## 4. System Architecture

### 4.1 Proposed Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PORTABLE NV-MASER MRI SYSTEM                          │
│                                                                          │
│  ┌──────────────── SENSOR HEAD (< 25 kg) ─────────────────┐             │
│  │                                                          │             │
│  │  ┌──────────────────────────────────┐                    │             │
│  │  │     HALBACH ARRAY (B₀ = 80 mT)  │ ← Permanent magnets│             │
│  │  │     Bore: 16 cm (extremity)      │   Self-shielded    │             │
│  │  │                                  │                    │             │
│  │  │  ┌────── Inside Bore ──────┐     │                    │             │
│  │  │  │                         │     │                    │             │
│  │  │  │  3-axis GRADIENT COILS  │     │                    │             │
│  │  │  │  (10-30 mT/m)          │     │                    │             │
│  │  │  │                         │     │                    │             │
│  │  │  │  RF TX/RX COIL (Birdcage│     │                    │             │
│  │  │  │  or solenoid, ~3.4 MHz) │     │                    │             │
│  │  │  │                         │     │                    │             │
│  │  │  │  PATIENT EXTREMITY ──── │ ←── │ ── 16 cm bore      │             │
│  │  │  │                         │     │                    │             │
│  │  │  └─────────────────────────┘     │                    │             │
│  │  └──────────────────────────────────┘                    │             │
│  │                                                          │             │
│  │  ┌─────────── MASER MODULE (shielded) ─────────────┐    │             │
│  │  │  Mini Halbach (50 mT, 14mm bore)                 │    │             │
│  │  │  NV-diamond + microwave cavity                   │    │             │
│  │  │  532 nm laser + optics                           │    │             │
│  │  │  Shimming coils + control MCU                    │    │             │
│  │  │  RF input: up-converted NMR signal               │    │             │
│  │  │  RF output: amplified signal to electronics      │    │             │
│  │  └─────────────────────────────────────────────────┘    │             │
│  └──────────────────────────────────────────────────────────┘            │
│                           │                                              │
│                      Cable bundle                                        │
│                    (RF, power, gradient)                                  │
│                           │                                              │
│  ┌──────────── ELECTRONICS CART (< 25 kg) ──────────────┐               │
│  │  Power supplies (gradient amps, laser driver)         │               │
│  │  RF transmitter + T/R switch                          │               │
│  │  Up-conversion mixer + local oscillator               │               │
│  │  Maser control electronics                            │               │
│  │  ADC (16-bit, 10 MS/s)                                │               │
│  │  SBC/FPGA (sequence controller + recon)               │               │
│  │  Display + UI                                         │               │
│  └───────────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Architecture Rationale

**Why a separate imaging Halbach and maser Halbach?**

The imaging patient-bore magnet (B₀ = 80 mT, 16 cm bore) and the maser's own magnet (50 mT, 14 mm bore) serve completely different purposes:

| | Imaging Halbach | Maser Halbach |
|-|----------------|---------------|
| **Purpose** | Polarise patient tissue | Set NV transition frequency |
| **Bore** | 16 cm (fits knee/wrist) | 14 mm (fits 3mm diamond) |
| **B₀** | 80 mT (→ Larmor 3.4 MHz) | 50 mT (→ NV ν₋ = 1.47 GHz) |
| **Homogeneity** | < 100 ppm over 12 cm DSV | < 50 ppm over 6 mm |
| **Weight** | ~15–30 kg | ~0.3 kg |
| **Shimming** | Yes (imaging gradients + B₀ shims) | Yes (active 8-coil) |

**Why 80 mT instead of 50 mT for imaging?** Higher B₀ → more signal. The Hyperfine Swoop uses 64 mT; we go slightly higher because the maser's SNR advantage lets us use a smaller, less homogeneous magnet while still producing acceptable images. 80 mT is achievable with a N52 Halbach of reasonable size.

### 4.3 Frequency Plan

| Signal | Frequency | Origin |
|--------|-----------|--------|
| Proton Larmor (imaging) | $\gamma_p \times 0.080 = 3.406$ MHz | B₀ of imaging magnet |
| NV maser carrier | 1.4699 GHz | NV ν₋ = D − γ_e × 0.050 T |
| Up-conversion LO | 1.4699 GHz − 3.406 MHz = 1.46649 GHz | Mixer reference |
| Gradient bandwidth | DC – 10 kHz | Imaging sequence timing |
| Shimming control | DC – 1 kHz | Maser field stabilisation |

---

## 5. Critical Signal Chain: Tissue → Image

### 5.1 MRI Pulse Sequence (Spin Echo Example)

```
RF:     ┌──┐         ┌────┐
        │90°│         │180°│
        └──┘         └────┘
             TE/2          TE/2
             ├──────────────┤
Gslice: ┌───┐         ┌───┐
        │ + │         │ + │
        └───┘         └───┘
Gphase: ┌───┐
        │var│  (stepped each TR)
        └───┘
Gfreq:                      ┌─────────┐
                             │readout  │
                             └─────────┘
Signal:                           ╱╲
                                 ╱  ╲  ← spin echo
                                ╱    ╲
ADC:                        ┌──────────┐
                            │  sample  │
                            └──────────┘
        ├─────── TR (50-500 ms) ─────────┤
```

**Sequence parameters at 80 mT:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| TE | 15–60 ms | T2-weighted; T2 of tissue ~40–200 ms |
| TR | 200–2000 ms | T1-weighted; T1 at 80 mT ~200–600 ms (shorter than 1.5T) |
| Readout BW | 5–50 kHz | Determines frequency-encode resolution |
| Matrix | 64×64 to 128×128 | Trading resolution vs. scan time |
| Averages | 4–32 | More at lower SNR |
| Scan time (64×64, NEX=16) | 64 × 500ms × 16 = 8.5 min | Acceptable for extremity |

### 5.2 Expected NMR Signal Amplitude

**Equilibrium magnetisation at 80 mT:**
$$M_0 = \frac{n_H \gamma^2 \hbar^2 I(I+1)}{3 k_B T} B_0$$

For water ($n_H = 6.7 \times 10^{28}$ m⁻³, I = 1/2):
$$M_0 = \frac{6.7 \times 10^{28} \times (2.675 \times 10^8)^2 \times (1.055 \times 10^{-34})^2 \times 0.75}{3 \times 1.38 \times 10^{-23} \times 310} \times 0.08$$
$$M_0 \approx 2.7 \times 10^{-8} \text{ A/m}$$

**EMF in receive coil** (solenoid, filling factor η ≈ 0.3, coil sensitivity B₁/I ≈ 50 µT/A):
$$\mathcal{E} = -\frac{d\Phi}{dt} = \omega_0 M_0 V_{voxel} \times \frac{B_1}{I} \times \eta$$

For a 2×2×5 mm³ voxel:
$$\mathcal{E} = 2\pi \times 3.4 \times 10^6 \times 2.7 \times 10^{-8} \times 20 \times 10^{-9} \times 50 \times 10^{-6} \times 0.3$$
$$\mathcal{E} \approx 17 \text{ pV (picovolts) per voxel}$$

**Coil thermal noise** (R = 0.3 Ω, BW = 20 kHz, T = 300 K):
$$V_n = \sqrt{4 k_B T R \Delta f} = \sqrt{4 \times 1.38 \times 10^{-23} \times 300 \times 0.3 \times 20000} \approx 10 \text{ nV}$$

**Raw single-shot SNR per voxel:**
$$\text{SNR}_1 = \frac{17 \text{ pV}}{10 \text{ nV}} = 0.0017$$

This is terrible. We need averaging:
$$\text{SNR}_{final} = \text{SNR}_1 \times \sqrt{N_{avg}} = 0.0017 \times \sqrt{N_{avg}}$$

For SNR = 10: $N_{avg} = (10/0.0017)^2 \approx 35 \text{ million}$ averages. **Impossible with conventional detection.**

### 5.3 How the Maser Closes the Gap (Revised)

The above calculation reveals the brutal reality of ultra-low-field MRI. Let's see what the maser architecture actually provides:

**The maser doesn't just reduce noise — it provides coherent gain.**

In the up-conversion architecture:
1. The 3.4 MHz NMR signal is mixed up to 1.47 GHz
2. The maser amplifies with gain G ≈ 20–30 dB (100–1000×)
3. This amplification happens **before** significant noise is added

But the coil thermal noise gets amplified equally with the signal. The maser only helps relative to **downstream** noise sources.

**Real improvement comes from three combined strategies:**

| Strategy | Factor | Mechanism |
|----------|--------|-----------|
| **Cooled receive coil (77K)** | 2× SNR | $\sqrt{300/77} = 1.97$ |
| **Maser preamp (vs. room-temp LNA)** | 1.1× | Marginal if coil noise dominates |
| **Higher B₀ (80 mT vs. 50 mT)** | 2× | Signal ∝ B₀^(7/4) → (80/50)^1.75 = 2.2 |
| **Larger voxels (2 mm vs. 1 mm)** | 8× | Volume scales as L³ |
| **Optimised low-field sequences** | 2–3× | Longer T2 at low field, steady-state free precession |
| **Advanced reconstruction (compressed sensing)** | 3–5× | Under-sampling + model priors |
| **Combined** | **~200–500×** | Bridges most of the 480× gap |

With all optimisations: $\text{SNR}_{final} \approx 0.0017 \times 400 \times \sqrt{16} \approx 2.7$ per voxel per average, with 16 averages → **SNR ≈ 11.** Marginally acceptable for clinical utility.

### 5.4 Honest Assessment

The physics is **tight but feasible** at 80 mT with aggressive engineering. The maser's contribution is meaningful but not transformative when used as a simple preamp. The maser becomes **game-changing** under two conditions:

1. **If the maser operates as a regenerative amplifier** (gain >> 30 dB near threshold), it can provide extraordinary sensitivity. Operating just below maser oscillation threshold gives very high gain with near-quantum-limited noise. This requires exquisite field uniformity (< 10 ppm).

2. **If the NV ensemble is used for Overhauser-enhanced dynamic nuclear polarisation (DNP)**. The 532 nm laser polarises NV electron spins, which can transfer polarisation to nearby ¹H nuclei via the Overhauser effect, enhancing $M_0$ by 100–1000×. This would be the real breakthrough: hyperpolarised tissue without cryogens.

**Priority recommendation**: Investigate NV-driven Overhauser DNP as the primary value proposition, with the maser as a secondary SNR-enhancement tool.

---

## 6. Subsystem Requirements Flowdown

### 6.1 From Image Quality to Subsystem Specs

Starting from the target image and working backwards:

```
Target: 2mm resolution, 12cm FOV, SNR≥10, scan ≤10 min
    ↓
Matrix: 64×64, voxel 2×2×5mm, NEX=16, TR=500ms
    ↓
┌──────────────────────────────────────────────────────────────┐
│ B₀ MAGNET                                                     │
│ • Strength: ≥ 80 mT (Larmor ≥ 3.4 MHz)                      │
│ • Homogeneity: < 100 ppm over 12cm DSV (before shimming)     │
│ • Bore: ≥ 16 cm clear inner diameter                          │
│ • Temporal stability: < 1 ppm/min (thermal drift controlled)  │
│ • Self-shielded: 5-gauss line < 1m from bore center           │
│ • Weight: < 20 kg                                             │
│ • Material: N52 NdFeB, 16+ segments, K=2 Halbach              │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ GRADIENT COILS (3-axis)                                       │
│ • Max gradient: ≥ 20 mT/m per axis                            │
│ • Linearity: < 5% over 12 cm DSV                              │
│ • Rise time: < 500 µs (to 90% of max)                         │
│ • Duty cycle: ≥ 50% at max amplitude                          │
│ • Eddy-current compensation: active or passive shielding       │
│ • Acoustic noise: < 80 dB(A) at 1m                            │
│ • Cooling: forced air or water (if >10W dissipation)           │
│ • Weight: < 3 kg (embedded in bore liner)                      │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ RF TRANSMIT / RECEIVE                                         │
│ • Frequency: 3.406 MHz ± 50 kHz (tracking B₀ drift)          │
│ • Transmit power: 50–200 W peak (SAR-limited for extremity)   │
│ • Receive coil: solenoid or birdcage, Q > 50 (loaded)         │
│ • T/R isolation: > 60 dB                                      │
│ • Receive noise figure: system NF < 1 dB (with maser)         │
│ • B₁ uniformity: < 20% variation over FOV                     │
│ • SAR limit: < 4 W/kg extremity (IEC 60601-2-33)              │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ NV MASER PREAMPLIFIER MODULE                                  │
│ • Operating frequency: 1.47 GHz                               │
│ • Gain: ≥ 20 dB (regenerative, near-threshold)                │
│ • Noise temperature: < 10 K (quantum-limited target: 0.07 K)  │
│ • Bandwidth: ≥ 50 kHz (covers NMR readout BW)                 │
│ • 1 dB compression: > -60 dBm (NMR signal is very weak)       │
│ • B₀ uniformity (maser magnet): < 50 ppm over diamond         │
│ • Cavity Q (effective): ≥ 50,000 (Q-boosted)                  │
│ • Optical pump: 2W CW 532 nm, thermally managed               │
│ • Active shimming: 8-coil, 1 kHz loop, < 100 µs latency       │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│ DIGITAL BACKEND                                               │
│ • ADC: 16-bit, ≥ 1 MS/s (Nyquist for 50 kHz readout BW)     │
│ • Sequence controller: FPGA or real-time MCU, µs timing        │
│ • Reconstruction: 2D FFT + NUFFT, iterative if needed          │
│ • Compute: embedded GPU (Jetson class) or FPGA                 │
│ • Display: 10" touch screen, DICOM export                      │
│ • Storage: SSD for raw k-space + images                        │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Imaging Halbach Requirements (NEW — not in current twin)

This is the largest missing subsystem. The current twin's 14 mm bore Halbach is only for the maser's own diamond. The imaging magnet is a separate, much larger structure.

**Imaging Halbach Design Constraints:**

| Parameter | Requirement | Challenge |
|-----------|-------------|-----------|
| Bore ID | ≥ 16 cm | Large N52 segments; ~20–40 magnets |
| B₀ | 80 mT | Feasible with N52 at this bore size |
| Homogeneity | < 100 ppm (12 cm DSV) | Very challenging — passive shimming + supershims |
| 5-gauss line | < 1 m | Self-shielded Halbach (add outer counter-rotating ring) |
| Weight | < 20 kg | Possible with optimised geometry |
| Length | ~25 cm (for 5 mm slice) | Short magnet → homogeneity degrades axially |
| Temperature coefficient | -0.12%/°C (NdFeB) | Must temp-stabilise to ±0.5°C or actively correct |
| Assembly forces | ~500 N between adjacent segments | Requires non-magnetic jig; SAFETY CRITICAL |

**Estimated imaging Halbach dimensions (preliminary):**
- Inner radius: 85 mm (16 cm bore + liner)
- Outer radius: 150 mm (for B₀ ≈ 80 mT with N52)
- Length: 250 mm
- Mass: ~15 kg NdFeB + 3 kg housing
- Segments: 16 × 2 rings = 32 magnets
- Cost: ~$2,000–4,000 (custom arc segments)

---

## 7. Gap Analysis: Current Twin vs. MRI Requirements

### 7.1 What the Twin Models Well (Keep)

| Module | Relevance to MRI | Status |
|--------|-----------------|--------|
| `halbach.py` | Maser-magnet multipole model | ✅ Accurate for maser module |
| `nv_spin.py` | NV transition physics, linewidth | ✅ Core maser physics |
| `maser.py` | Maser gain, threshold | ✅ Critical for preamp performance |
| `cavity.py` | Cavity QED, cooperativity | ✅ Maser module design |
| `optical_pump.py` | 532 nm inversion | ✅ Maser module design |
| `signal_chain.py` | RF detection, SNR budget | ⚠️ Models maser output, not NMR signal |
| `coils.py` | Active shimming | ✅ Maser field stabilisation |
| `closed_loop.py` | Real-time shimming control | ✅ Maser field stabilisation |
| `thermal.py` | Tempco, drift | ✅ Both magnets |
| `calibration/` | Field map I/O | ✅ Both magnets |

### 7.2 What's Missing (Must Build)

| Module | Purpose | Priority | Complexity |
|--------|---------|----------|------------|
| **`imaging_magnet.py`** | Large-bore Halbach field model (separate from maser Halbach) | **CRITICAL** | Medium |
| **`gradient_coils.py`** | 3-axis imaging gradient design, eddy currents, linearity | **CRITICAL** | High |
| **`rf_system.py`** | Transmit coil model, SAR calculation, B₁ map | **CRITICAL** | High |
| **`pulse_sequence.py`** | Spin echo, gradient echo, SSFP sequence timing | **CRITICAL** | High |
| **`nmr_signal.py`** | Bloch equation simulation of tissue NMR response | **CRITICAL** | Medium |
| **`reconstruction.py`** | k-space → image (FFT, NUFFT, compressed sensing) | **CRITICAL** | High |
| **`up_conversion.py`** | Mixer model: NMR freq → maser freq, noise analysis | **HIGH** | Medium |
| **`snr_calculator.py`** | End-to-end SNR budget: tissue → final image | **HIGH** | Medium |
| **`sar_calculator.py`** | Specific absorption rate for safety compliance | **HIGH** | Low |
| **`b0_shim.py`** | Passive + active shimming for imaging magnet | **HIGH** | Medium |

### 7.3 What Must Change in Existing Modules

| Module | Change Needed | Reason |
|--------|---------------|--------|
| `signal_chain.py` | Add NMR-to-maser frequency conversion | Currently models maser output only |
| `config/default.yaml` | Add imaging sections (magnet, gradient, RF, sequence) | Current config is maser-only |
| `halbach.py` | Generalise for large-bore geometry | Current model assumes small bore |
| `coils.py` | Distinguish shimming coils from imaging gradient coils | Different design requirements |

---

## 8. Design Trade Studies

### 8.1 Trade: Bore Geometry

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Cylindrical Halbach (16 cm bore)** | Standard MRI geometry; full 2D/3D encoding possible; good homogeneity | Heavy (15–25 kg); can't image large joints; patient comfort | **BASELINE** |
| **C-shaped / Yokeless** | Open access; patient comfort; can be lighter | Much worse homogeneity; weaker B₀; complex gradients | Not for v1 |
| **Single-sided (NMR-MOUSE style)** | Ultra-portable (<5 kg); no bore limitation; surface profiling | Very limited depth (1–3 cm); 1D profiles only; low SNR | Future product |

### 8.2 Trade: Maser Integration Strategy

| Option | SNR Gain | Complexity | Cost | Verdict |
|--------|----------|------------|------|---------|
| **No maser (conventional LNA)** | 1× | Lowest | -$3K | Not competitive |
| **Maser as stand-alone preamp module** | 1.1–2× | Medium | +$3K | **Phase 1 target** |
| **Maser with LN₂-cooled coil** | 2–8× | Medium-High | +$5K | **Phase 2 target** |
| **Maser + Overhauser DNP** | 100–1000× | Very High | +$5K | Research phase |

### 8.3 Trade: Target B₀ vs. Magnet Weight

| B₀ (mT) | Larmor (MHz) | Halbach Mass (est.) | SNR relative | Homogeneity challenge |
|----------|-------------|---------------------|-------------|----------------------|
| 50 | 2.1 | 8 kg | 1.0× | Moderate |
| 64 | 2.7 | 12 kg | 1.5× | Moderate-High |
| **80** | **3.4** | **18 kg** | **2.2×** | **High** |
| 100 | 4.3 | 28 kg | 3.2× | Very High |
| 150 | 6.4 | 50 kg | 5.8× | Extreme |

**Recommendation**: 80 mT balances SNR, weight, and homogeneity. This is the same order as Hyperfine (64 mT) with slightly better signal.

---

## 9. BOM & Cost Analysis

### 9.1 Estimated BOM at Production (100+ units)

| Subsystem | Components | Est. Unit Cost | Notes |
|-----------|-----------|---------------|-------|
| **Imaging Halbach array** | 32× N52 arc segments, housing, shim rings | $2,500–4,000 | Custom magnets from Ningbo/Arnold |
| **Gradient coils** | 3-axis wire-wound on bore liner, potted | $800–1,500 | In-house winding or Resonance Research |
| **Gradient amplifiers** | 3× class-D, ±20A, ±48V, 1 kHz BW | $1,500–3,000 | Custom PCB; GPA-FHDO open-source |
| **RF coil** | Birdcage or solenoid, tune/match network | $200–500 | Copper tube + capacitors |
| **RF transmitter** | 200W class-D PA at 3.4 MHz + T/R switch | $500–1,000 | RFPA from Mini-Circuits + PIN diode |
| **NV-diamond** | 3×3×0.5 mm CVD, >1 ppm NV | $1,000–3,000 | Element Six or Delaware Diamond |
| **Maser cavity** | OFHC Cu, CNC machined, coupling loop | $500–1,000 | Local machine shop |
| **532 nm laser module** | 2W DPSS + optics + fiber coupling | $800–2,000 | Coherent, Thorlabs, Lasertack |
| **Maser shimming PCB** | 8-ch DAC, H-bridge drivers, MCU | $150–300 | Custom KiCad design |
| **Maser Halbach** | 8–16× N52 cylinders, 3D-printed holder | $100–200 | Small, well-characterised |
| **Digital backend** | FPGA (Xilinx Artix-7) or SBC (Jetson Orin Nano) | $300–800 | Sequence control + recon |
| **ADC board** | 16-bit, 10 MS/s, 4-channel | $200–500 | LTC2387 or AD9467 |
| **Power supply** | Medical-grade SMPS, 48V/500W + 12V/100W | $300–600 | Mean Well or TDK-Lambda |
| **Enclosure** | IP54 aluminium + RF shielding | $500–1,500 | Sheet metal + EMI gaskets |
| **Display** | 10" medical-grade touch LCD | $200–400 | Advantech or Kontron |
| **Cabling & connectors** | RF, gradient, power, Ethernet | $200–400 | Medical-grade |
| **Assembly, test, calibration** | Per-unit labour | $1,000–2,000 | 8–16 hours technician time |
| **Software (per-unit amort.)** | Embedded firmware + recon + UI | $500–1,000 | Amortised over production run |
| | | | |
| **TOTAL COGS** | | **$11,350–23,700** | |
| **Retail (2× markup)** | | **$22,700–47,400** | |
| **Target retail** | | **$20,000–40,000** | **ACHIEVABLE at 2× on low-end BOM** |

### 9.2 Cost Sensitivities

| Component | If we get lucky | If we get unlucky |
|-----------|----------------|-------------------|
| NV-diamond | $500 (bulk CVD) | $5,000 (custom isotopically enriched ¹²C) |
| Imaging Halbach | $2,000 (standard arc) | $6,000 (custom tolerance-graded) |
| Gradient amps | $1,200 (GPA-FHDO) | $4,000 (commercial) |
| 532 nm laser | $500 (Chinese DPSS) | $3,000 (Coherent Verdi) |

**The diamond and magnet are the swing items.** Focus sourcing effort here.

---

## 10. Risk Register

### 10.1 Technical Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| **T1** | **Maser SNR gain insufficient for clinical imaging** | HIGH | CRITICAL | Validate with phantom before patient imaging; prepare fallback to conventional LNA; pursue Overhauser DNP as multiplier |
| **T2** | **Imaging Halbach homogeneity > 100 ppm** | MEDIUM | HIGH | Passive shimming iron rings + active B₀ shim coils; accept lower homogeneity with model-based recon |
| **T3** | **Maser field uniformity unstable in EM-noisy environment** | MEDIUM | HIGH | RF shielding (mu-metal + copper); active shimming proven in twin; increase sensor density |
| **T4** | **Gradient-induced eddy currents degrade B₀** | HIGH | MEDIUM | Pre-emphasis compensation in sequence controller; gradient shielding |
| **T5** | **Up-conversion mixer adds unacceptable noise** | MEDIUM | HIGH | Use low-noise MMIC mixer (e.g., LTC5549); measure noise contribution empirically |
| **T6** | **Thermal drift causes image artefacts** | MEDIUM | MEDIUM | Insulate magnet; run temp compensation loop; allow 15-min warm-up |
| **T7** | **2W 532 nm laser eye safety in clinical setting** | LOW | CRITICAL | Fully enclosed optical path; interlocked housing; Class 1 enclosure per IEC 60825-1 |
| **T8** | **Halbach assembly forces cause injury** | LOW | CRITICAL | Non-magnetic assembly jig; written SOP; gauss-rated gloves; >2 person procedure |

### 10.2 Commercial Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| **C1** | **Image quality insufficient for reimbursement** | HIGH | CRITICAL | Target sports medicine / MSK where lower fidelity is acceptable; seek FDA 510(k) vs. predicate (Hyperfine) |
| **C2** | **NV diamond supply chain** | MEDIUM | HIGH | Dual-source: Element Six + Delaware Diamond; maintain 6-month buffer |
| **C3** | **Patent freedom-to-operate** | MEDIUM | HIGH | FTO search on NV-maser MRI; Wang/Breeze patents; Halbach MRI patents (O'Reilly, Cooley) |
| **C4** | **Hyperfine dominates market at lower price** | MEDIUM | MEDIUM | Differentiate on weight/portability (50 kg vs. 640 kg) and price ($30K vs. $50K lease) |

---

## 11. Regulatory & Safety

### 11.1 Applicable Standards

| Standard | Scope | Impact |
|----------|-------|--------|
| **IEC 60601-1** | General medical electrical safety | Enclosure, insulation, grounding |
| **IEC 60601-1-2** | Electromagnetic compatibility (EMC) | Emissions & immunity testing |
| **IEC 60601-2-33** | MRI-specific safety | B₀ fringe field, SAR limits, gradient dB/dt, acoustic noise |
| **IEC 60825-1** | Laser safety | Class 1 enclosure for 532 nm laser |
| **IEC 62304** | Medical device software lifecycle | Software QMS, SOUP management |
| **ISO 14971** | Risk management | FMEA for all subsystems |
| **ISO 13485** | Quality management system | Required for CE/FDA |
| **FDA 21 CFR 892.1000** | MRI classification (Class II) | 510(k) pathway; predicate: Hyperfine Swoop |
| **EU MDR 2017/745** | CE marking | Class IIa medical device |

### 11.2 SAR Safety (Patient Heating)

At 80 mT (3.4 MHz), SAR is inherently low:

$$\text{SAR} = \frac{\sigma E^2}{2\rho}$$

At 3.4 MHz, the wavelength in tissue is ~8.8 m >> body dimension → quasi-static regime → negligible E-field heating. SAR calculations are trivial at this frequency.

**IEC 60601-2-33 limits:**
- Whole body: 4 W/kg (normal mode)
- Extremity: 12 W/kg (normal mode)
- Our expected SAR: << 0.1 W/kg (not a concern at 80 mT)

### 11.3 Gradient dB/dt Safety (Peripheral Nerve Stimulation)

IEC limit: dB/dt < 20 T/s (normal mode) for non-cardiac

At 20 mT/m gradient, 500 µs rise time, over 20 cm:
$$\frac{dB}{dt} = 20 \times 10^{-3} \text{ T/m} \times 0.2 \text{ m} / 500 \times 10^{-6} \text{ s} = 8 \text{ T/s}$$

Well within limits.

### 11.4 Fringe Field (5-Gauss Line)

For self-shielded Halbach (inner dipole + outer counter-dipole):
- 80 mT at bore center
- 5-gauss (0.5 mT) line: < 0.5 m from bore center (achievable with optimised shielding ring)
- Safe for pacemaker patients at > 1 m

---

## 12. Recommended Development Path

### 12.1 Phased Approach

**Phase 0: Validate Physics (3–6 months)**
> "Does the maser-enhanced receive chain produce measurably better NMR signal?"

| Task | Deliverable |
|------|-------------|
| Build maser module (existing twin scope) | Working maser with measured gain + noise temp |
| Bench NMR experiment (no imaging) | Detect water proton FID at 80 mT with and without maser |
| Measure SNR improvement | Published comparison: conventional LNA vs. maser preamp |
| Decide: maser-only vs. maser + cooled coil vs. Overhauser DNP | Architecture freeze memo |

**Phase 1: Imaging Prototype (6–12 months)**
> "Can we form an image?"

| Task | Deliverable |
|------|-------------|
| Design + build imaging Halbach (80 mT, 16 cm bore) | Measured field map, shimmed to < 200 ppm |
| Design + build 3-axis gradient coils | Gradient maps, linearity characterised |
| Implement spin echo sequence on FPGA | Working pulse sequence with timing verified |
| First phantom image (water bottle) | 64×64 image, resolution verified |
| Integrate maser module | Before/after SNR comparison on phantom |

**Phase 2: Clinical Prototype (6–12 months)**
> "Can this produce clinically useful images?"

| Task | Deliverable |
|------|-------------|
| Optimise sequences (multi-contrast) | T1, T2, PD-weighted extremity images |
| Advanced reconstruction (compressed sensing, deep learning) | Improved image quality from under-sampled data |
| Build integrated housing + electronics | Portable prototype, < 50 kg |
| Ex-vivo tissue imaging | Cadaveric extremity images for clinical review |
| IRB-approved volunteer study | First human images |

**Phase 3: Product (12–18 months)**
> "Can we sell this?"

| Task | Deliverable |
|------|-------------|
| IEC 60601 compliance testing | Test reports (accredited lab) |
| FDA 510(k) submission | Substantial equivalence to Hyperfine Swoop |
| Design for manufacturing | Production BOM, assembly procedures |
| Clinical validation study | Comparison vs. conventional MRI for target indication |
| First sales | $20–40K extremity MRI |

### 12.2 What to Build Next in the Digital Twin

Based on this analysis, the twin development should shift from pure maser physics to **end-to-end MRI simulation**:

| Priority | Module | Purpose | Depends on |
|----------|--------|---------|------------|
| **1** | `nmr_signal.py` | Bloch equation: tissue response to RF pulse at given B₀ | None |
| **2** | `imaging_magnet.py` | Large-bore Halbach model (generalise existing halbach.py) | None |
| **3** | `gradient_coils.py` | 3-axis gradient field calculation | imaging_magnet.py |
| **4** | `rf_system.py` | Transmit B₁ field, SAR, receive sensitivity | None |
| **5** | `pulse_sequence.py` | Spin echo / gradient echo timing engine | nmr_signal.py |
| **6** | `up_conversion.py` | NMR-to-maser frequency mixing, noise | maser.py |
| **7** | `snr_calculator.py` | End-to-end SNR: tissue → coil → maser → ADC → image | All above |
| **8** | `reconstruction.py` | k-space → image (FFT, iterative) | pulse_sequence.py |
| **9** | `phantom_sim.py` | Virtual imaging experiment (water bottle, Shepp-Logan) | All above |

### 12.3 What to Build Next in Hardware

| Priority | Task | Cost | Timeline |
|----------|------|------|----------|
| **1** | Complete maser module (Phase 0 scope) | $3–5K | 2–4 months |
| **2** | Bench-top NMR magnet (80 mT, small bore for flask) | $1–2K | 1–2 months |
| **3** | NMR probe coil + preamp | $500 | 2 weeks |
| **4** | Prove maser improves NMR SNR on bench | $0 (integration) | 1 month |
| **5** | Decision gate: proceed to imaging or pivot | — | 1 week review |

---

## Appendix A: Key Literature

| Ref | Relevance |
|-----|-----------|
| Wang et al. (2024) "Room-temperature maser" | NV maser gain, Q-boosting, noise temperature |
| Breeze et al. (2018) "Continuous-wave room-temperature diamond maser" | First NV maser demonstration |
| Sarracanie & Salameh (2020) "Low-Field MRI: How Low Can We Go?" | SNR analysis at ultra-low field |
| O'Reilly et al. (2021) "In vivo 3D brain / knee at 50 mT Halbach" | Extremity imaging at our target field |
| Cooley et al. (2021) "A portable scanner for MRI of the brain" | Halbach design for portable MRI |
| Hyperfine Swoop (2023) FDA 510(k) K200373 | Predicate device for regulatory pathway |
| Barry et al. (2020) "Sensitivity optimization for NV-diamond magnetometry" | NV ensemble sensitivity limits |

## Appendix B: Nomenclature

| Symbol | Meaning | Unit |
|--------|---------|------|
| B₀ | Static main magnetic field | T |
| γ_p | Proton gyromagnetic ratio (42.577 MHz/T) | MHz/T |
| γ_e | Electron gyromagnetic ratio (28.025 GHz/T) | GHz/T |
| D | NV zero-field splitting (2.87 GHz) | GHz |
| T₁ | Spin-lattice relaxation time | s |
| T₂ | Spin-spin relaxation time | s |
| T₂* | Effective transverse relaxation (includes inhomogeneity) | s |
| SAR | Specific Absorption Rate | W/kg |
| DSV | Diameter of Spherical Volume (homogeneity region) | cm |
| FOV | Field of View | cm |
| NEX | Number of Excitations (signal averages) | — |
| SNR | Signal-to-Noise Ratio | — |
| SSFP | Steady-State Free Precession | — |
| DNP | Dynamic Nuclear Polarisation | — |
