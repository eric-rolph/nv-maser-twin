# Handheld NV-Maser MRI Probe: Architecture Design

**Date**: 2025-07-17  
**Author**: Digital Twin + Hardware Engineering Review  
**Status**: DRAFT — requires expert physics review  
**Scope**: Architecture for a handheld, maser-enhanced medical imaging probe for emergency triage  
**Supersedes**: Bore-based extremity scanner concept (see `portable-mri-design-review.md`)

---

## Table of Contents

1. [Mission Redefinition](#1-mission-redefinition)
2. [Why Handheld Changes Everything](#2-why-handheld-changes-everything)
3. [Single-Sided NMR/MRI: Prior Art](#3-single-sided-nmrmri-prior-art)
4. [The Maser Advantage in Handheld Geometry](#4-the-maser-advantage-in-handheld-geometry)
5. [System Architecture](#5-system-architecture)
6. [Probe Head Design](#6-probe-head-design)
7. [Signal Chain: Tissue → Screen](#7-signal-chain-tissue--screen)
8. [SNR Budget (Single-Sided Geometry)](#8-snr-budget-single-sided-geometry)
9. [Imaging Modes & Clinical Use Cases](#9-imaging-modes--clinical-use-cases)
10. [Compute Architecture (Tablet + Cloud)](#10-compute-architecture-tablet--cloud)
11. [Digital Twin Module Plan](#11-digital-twin-module-plan)
12. [Hardware Development Path](#12-hardware-development-path)
13. [Risk Register](#13-risk-register)
14. [Regulatory Considerations](#14-regulatory-considerations)

---

## 1. Mission Redefinition

### 1.1 From Bore Scanner to Handheld Probe

The previous design review explored a **bore-based extremity scanner** (~50 kg, 16 cm bore Halbach, cart-mounted). That concept is viable but doesn't fully leverage the maser's unique properties. We're pivoting to a fundamentally different form factor.

**New product intent**: A **handheld medical imaging probe** — like an ultrasound transducer in size and usage — that produces MRI-contrast images (T1, T2, proton density) by exploiting the NV diamond maser's room-temperature quantum-limited sensitivity.

| Attribute | Bore Scanner (old) | Handheld Probe (new) |
|-----------|-------------------|----------------------|
| Form factor | Cart-mounted cylinder | Ultrasound-transducer-like |
| Weight (probe) | ~25 kg (sensor head) | < 1.5 kg |
| Patient interface | Extremity inserted in bore | Probe placed on skin surface |
| Magnet geometry | Cylindrical Halbach (enclosed) | **Single-sided** (open, one-sided) |
| B₀ homogeneity need | < 100 ppm over 12 cm FOV | Controlled inhomogeneity (gradient used for encoding) |
| FOV | 12+ cm cross-section | 3–8 cm lateral, 1–5 cm depth |
| Use case | Scheduled extremity scan | **Immediate emergency triage** |
| Setup time | 5 min (position patient) | **Seconds** (place probe on body) |
| Operator skill | Trained technician | Paramedic / ER nurse |
| Mobility | Wheeled cart | **Truly portable** (one hand) |

### 1.2 Product Vision

**"MRI contrast in the palm of your hand."**

The operator places the probe on the patient's body — like an ultrasound transducer — and within 30–120 seconds sees a tissue-contrast image on a connected tablet. No bore, no patient repositioning, no shielded room.

### 1.3 Performance Targets

| Parameter | Target | Rationale |
|-----------|--------|-----------|
| **Probe weight** | < 1.5 kg | One-hand operation (cf. ultrasound probe ~0.3 kg) |
| **Total system** | < 10 kg (probe + base unit if needed) | Carry in backpack |
| **Depth penetration** | 1–5 cm into tissue | Covers subcutaneous tissue, tendons, shallow organs |
| **Lateral FOV** | 3–8 cm | Region of clinical interest |
| **Resolution** | 1–3 mm in-plane, 1–5 mm depth | Adequate for hemorrhage, fracture, edema |
| **Acquisition time** | 15–120 seconds per view | Emergency triage pace |
| **Image contrast** | T1, T2, proton density (fluid vs. solid) | Critical for hemorrhage/edema detection |
| **Power** | < 100 W total (battery-operable with base unit) | Field-deployable |
| **Connectivity** | USB-C/Thunderbolt to tablet, WiFi to cloud | Standard clinical tablet |
| **No cryogenics** | Entirely room-temperature | Maser advantage: no LN₂ or LHe |

### 1.4 Comparison: Existing Handheld / Point-of-Care Devices

| Device | Modality | Weight | Depth | Resolution | Price | FDA |
|--------|----------|--------|-------|------------|-------|-----|
| **Butterfly iQ+** | Ultrasound | 0.31 kg | 30 cm | ~0.5 mm | $2,500 | 510(k) |
| **Clarius C3** | Ultrasound | 0.2 kg | 30 cm | ~0.3 mm | $5,000 | 510(k) |
| **NMR-MOUSE** | 1D NMR profiling | ~1 kg (probe) | 2.5 cm | 0.02 mm (depth) | ~$50K | Research |
| **Hyperfine Swoop** | Full MRI (bore) | 640 kg | N/A | 1.5 mm | ~$50K/yr | 510(k) |
| **Our target** | MRI-contrast imaging | < 1.5 kg (probe) | 5 cm | 1–3 mm | TBD | 510(k) TBD |

Our device occupies an **uncontested niche**: MRI tissue contrast in an ultrasound form factor. Ultrasound gives structural imaging but cannot distinguish hemorrhage from edema (both are hypoechoic). MRI contrast can.

---

## 2. Why Handheld Changes Everything

### 2.1 Single-Sided Geometry Advantages

A bore-based MRI requires the patient to be **inside** a magnet. A handheld probe uses a **single-sided** (or "unilateral") magnet — all magnets are on one side, and the sensitive region extends outward into the tissue.

**Key insight**: In single-sided geometry, the B₀ field is **inherently inhomogeneous** — it falls off with distance from the magnet surface. In bore-based MRI, this inhomogeneity is a defect to be corrected. In single-sided NMR, **it is the gradient** — the built-in field gradient provides spatial encoding along the depth axis for free.

```
     BORE-BASED                          SINGLE-SIDED
     ┌─────────┐                         ┌──────────┐
     │ Magnet  │                         │  Magnet  │ ← magnets on ONE side
     │ ┌─────┐ │                         │  array   │
     │ │     │ │ ← patient inside        └────┬─────┘
     │ │tissue│ │                              │  ← field extends outward
     │ │     │ │    B₀ uniform                 │
     │ └─────┘ │    inside bore           ═════╪═════ ← patient surface (skin)
     │ Magnet  │                              │
     └─────────┘                         ░░░░░│░░░░░ ← tissue
                                         ░░░ sensitive ░░░
                                         ░░░  volume  ░░░
                                         ░░░░░░░░░░░░░░░
```

### 2.2 Why the Maser Is MORE Valuable in Handheld Than in Bore

The bore-based design review found that the maser provides only ~1.1× SNR improvement as a simple preamp (Section 3.3 of previous review). For the handheld, the physics shifts significantly in the maser's favor:

| Factor | Bore-Based | Handheld (Single-Sided) |
|--------|------------|------------------------|
| **Receive coil distance** | 8 cm (bore radius) | **< 1 cm** (on skin) |
| **Coil filling factor** | ~0.3 (large bore, small tissue) | **> 0.5** (coil matched to ROI) |
| **Sensitive volume** | 12 cm × 12 cm × 5 mm (large) | 4 cm × 4 cm × 2 mm (small) |
| **Body noise** ($R_{body} \propto \omega^2$) | Moderate (bigger volume → more lossy tissue) | **Very low** (small region, low freq) |
| **Coil noise dominance** | Coil 300K + body ~50K | **Coil 300K + body ~5K** |
| **Maser preamp benefit** | ~1.1× (body noise significant) | **~1.2×** (almost pure coil noise) |
| **Proximity gain** | None (8 cm to tissue) | **10–30× signal boost** (1/r³ coupling) |

But the **real game-changer** is proximity. With the receive coil < 1 cm from tissue, the signal-per-voxel is vastly stronger:

$$V_{signal} \propto \frac{1}{r^3} \times M_0 \times V_{voxel}$$

At r = 0.5 cm vs. r = 8 cm: signal is $(8/0.5)^3 = 4096×$ stronger for the same voxel.

This transforms the SNR equation. The bore scanner had a per-voxel signal of ~17 pV. The handheld probe at 1 cm depth gets:

$$V_{signal,handheld} \sim 17 \text{ pV} \times \left(\frac{8}{1}\right)^3 \times \frac{\text{filling factor ratio}}{\text{volume ratio}} \approx \text{several nV}$$

(The exact value depends on coil geometry, filling factor, and voxel placement — see Section 8 for a rigorous calculation.)

### 2.3 Leveraging Unique Maser Properties

Beyond simple low-noise preamplification, the NV maser offers properties no other room-temperature detector has:

| Unique Property | How It Helps Handheld Imaging |
|----------------|------------------------------|
| **Room-temperature quantum-limited noise** | No cryogenics → handheld form factor possible |
| **Narrow instantaneous bandwidth** (~50 kHz around 1.47 GHz) | Natural bandpass filter; rejects out-of-band interference (no RF shielded room needed) |
| **Frequency-selective detection** | Can be tuned to specific B₀ values → depth-selective detection in a gradient field |
| **Regenerative gain** (near-threshold operation) | 30–60 dB gain before noise floor, far exceeding any room-temp LNA |
| **Continuous operation** (stimulated emission) | No dead time, no duty cycle limits |
| **Optical readout** (ODMR/fluorescence) | Secondary/backup detection channel for calibration |
| **Self-contained field reference** | The NV zero-field splitting D = 2.87 GHz is a fundamental constant — temperature-compensated frequency reference |

**The narrow bandwidth is especially valuable for handheld use without an RF-shielded room.** Conventional MRI requires Faraday cages to reject RF interference (cell phones, Wi-Fi, broadcast radio). The maser's ~50 kHz bandwidth at 1.47 GHz (fractional bandwidth ~3×10⁻⁵) naturally rejects interference. After down-conversion, only signals within the NMR readout bandwidth survive.

---

## 3. Single-Sided NMR/MRI: Prior Art

### 3.1 NMR-MOUSE (Mobile Universal Surface Explorer)

Invented by Bernhard Blümich (RWTH Aachen), the NMR-MOUSE is the canonical single-sided NMR device.

**Key parameters:**
- Magnet: U-shaped permanent magnet with anti-Helmholtz arrangement
- B₀: ~0.3–0.5 T at surface, falls off rapidly with depth
- Built-in gradient: ~15–25 T/m (very strong)
- Sensitive slice: ~0.01–0.1 mm thick at any given depth
- Depth range: 0–25 mm
- Weight: ~1 kg (probe head)
- Produces: 1D depth profiles (T1, T2, diffusion vs. depth)
- Limitations: **No lateral imaging** (1D only), very thin sensitive slice

**What we can learn:**
- U-shaped magnet geometry is proven and manufacturable
- The strong built-in gradient provides micron-level depth resolution
- 1D profiling is clinically useful (skin layers, cartilage, bone interface)
- Acquisition time: seconds to minutes for a profile

### 3.2 GARField (Gradient At Right-angles to Field)

A variant where B₀ and its gradient are perpendicular, allowing thicker slices.

- Better SNR per acquisition (thicker slice = more signal)
- Used for thin-film and coating analysis
- Less depth resolution but faster

### 3.3 Sweet-Spot Magnets

A magnet designed so that B₀ has a local **sweet spot** — a small region where the field is relatively homogeneous — at a specific depth outside the magnet.

```
Field profile vs. depth:

B₀  ┤
    │    ╱╲
    │   ╱  ╲ ← sweet spot (quasi-homogeneous region)
    │  ╱    ╲
    │ ╱      ╲
    │╱        ╲──────
    ├──────────────────→ depth from surface
    magnet     1 cm      3 cm
    surface
```

**Advantage**: In the sweet spot, the field is uniform enough for multi-echo sequences (T2 measurements) and even rudimentary 2D imaging with lateral gradients.

**Key designs:**
- Prado et al. (2003): Cylindrical magnet array with sweet spot at ~1 cm
- Marble et al. (2007): Optimized single-sided magnet for NMR relaxometry
- Manz et al. (2006): Single-sided MRI with 2D resolution using lateral gradient coils

### 3.4 Single-Sided MRI (Lateral Imaging)

Several groups have demonstrated 2D and even 3D imaging with single-sided geometries:

| Group | Method | Resolution | Depth | Time |
|-------|--------|------------|-------|------|
| Blümich (RWTH) | NMR-MOUSE + mechanical scanning | 0.05 mm depth | 25 mm | Minutes per profile |
| Casanova (RWTH) | Planar gradient coils on MOUSE | 3 mm lateral | 10 mm | ~15 min for 2D |
| Perlo (Magritek) | MOUSE + phase encoding | 2 mm lateral | 10 mm | ~10 min |
| Cooley et al. (2021) | Rotating single-sided array | 2 mm | 50 mm | ~5 min |

**These resolutions and times are in the right ballpark for our emergency triage target.**

### 3.5 Gap: Why No One Has Done This at Scale

Existing single-sided MRI has limitations that the maser can address:

| Limitation | Current Approach | Maser Solution |
|-----------|-----------------|----------------|
| **Low SNR** (weak field, small volume) | Long averaging (10+ min) | Maser preamp reduces noise → faster scans |
| **RF interference** (no Faraday cage) | Shielding box around probe | Maser's narrow bandwidth is a natural filter |
| **Limited depth** (field falls off quickly) | Strong magnets → heavy | Maser detects weaker signals → works at greater depth |
| **Coarse resolution** (SNR-limited) | Large voxels | Better SNR → smaller voxels feasible |
| **No clinical adoption** (research-only) | Custom, fragile hardware | Tablet-connected, app-guided, user-friendly |

---

## 4. The Maser Advantage in Handheld Geometry

### 4.1 Noise Temperature Budget (Handheld)

At the probe surface, with a small surface coil closely coupled to tissue:

**Coil parameters:**
- Coil diameter: 3 cm (matched to ROI)
- Coil turns: 4–8 (flat spiral on PCB)
- Coil resistance: R_coil ≈ 0.1–0.5 Ω (depending on frequency and construction)
- Q_loaded ≈ 30–80

**Body noise at 2 MHz (50 mT Larmor):**

$$R_{body} \approx \frac{\omega^2 \mu_0^2 \sigma_{tissue} V_{eff}}{20}$$

For a 3 cm coil at 2 MHz, with tissue conductivity σ ≈ 0.5 S/m and effective volume ~30 cm³:

$$R_{body} \approx \frac{(2\pi \times 2 \times 10^6)^2 \times (4\pi \times 10^{-7})^2 \times 0.5 \times 30 \times 10^{-6}}{20} \approx 0.001 \text{ Ω}$$

**$R_{body} \ll R_{coil}$** at 2 MHz — confirmed: the system is coil-noise-dominated.

**Noise temperature comparison:**

| Component | Conventional | With Maser |
|-----------|-------------|------------|
| Coil (300K, R=0.3 Ω) | 300 K | 300 K (unchanged) |
| Body (~2K effective) | 2 K | 2 K (unchanged) |
| Preamp | 75 K (1 dB NF LNA) | **1–5 K** (maser at quantum limit) |
| Mixer (if used) | 50 K × (1/G_preamp) | Negligible (after maser gain) |
| **System total** | **~377 K** | **~307 K** |
| **SNR ratio** | baseline | **1.11×** |

This is a disappointingly modest improvement — same ~10% as the bore case. The coil at 300 K still dominates.

### 4.2 Breaking the Coil-Noise Barrier

To make the maser truly transformative, we must address the coil temperature. Three strategies:

#### Strategy A: Cryogenic Coil (LN₂ or Thermoelectric)

Cool the receive coil to reduce its thermal noise:

| Cooling method | T_coil | T_sys | SNR gain | Practical? |
|---------------|--------|-------|----------|------------|
| Room temp (300 K) | 300 K | 307 K | 1.0× | ✅ Baseline |
| Thermoelectric (200 K) | 200 K | 207 K | 1.35× | ✅ Feasible in probe |
| LN₂ (77 K) | 77 K | 82 K | 2.14× | ⚠️ Adds complexity |
| HTS coil at 77 K | ~5 K | 10 K | 6.1× | ❌ Not handheld |

**Thermoelectric (Peltier) cooling to ~200 K is realistic in a handheld probe** — a small Peltier element behind the coil, heat sunk to the probe housing. This gives 1.35× SNR = 1.8× scan time reduction *on top of* the maser benefit.

#### Strategy B: Regenerative Maser (Near-Threshold Operation)

The maser operated just below oscillation threshold provides **very high gain** (potentially 40–60 dB) with near-quantum-limited noise. This is distinct from simple linear amplification — it's a **parametric amplification** regime.

In this regime, the maser doesn't just reduce additive noise — it provides coherent gain that amplifies the signal before any thermal noise contaminates it. The effective system noise temperature becomes:

$$T_{sys,regen} = T_{coil}/G_{regen} + T_{maser} + T_{downstream}/G_{regen}$$

If $G_{regen} = 10,000$ (40 dB):

$$T_{sys,regen} = 300/10000 + 5 + 75/10000 \approx 5.03 \text{ K}$$

$$\text{SNR gain} = \sqrt{377/5.03} \approx 8.7\times \quad (19 \text{ dB})$$

**This is the regime where the maser becomes truly game-changing.** 

But there's a critical caveat: the maser gain bandwidth is ~50 kHz (set by Q_cavity / f_0). The NMR readout bandwidth must fit within this. At 50 mT with a 20 kHz readout bandwidth, this is fine. But the maser must be exquisitely frequency-stable and the B₀ must not drift the Larmor frequency outside the gain band.

#### Strategy C: Overhauser Dynamic Nuclear Polarization (DNP)

The most radical option: use the NV electron spin polarization to enhance proton polarization via the Overhauser effect.

If the maser diamond (or a separate NV-rich diamond layer) is placed close to tissue, the polarized NV electrons can transfer spin angular momentum to nearby protons through cross-relaxation:

$$\text{Enhancement} = 1 - \frac{\gamma_e}{\gamma_H} \times f \times s \leq 1 + |\gamma_e/\gamma_H| \approx 660$$

In practice, with coupling factor f ~0.1 and saturation factor s ~0.5: enhancement ~30×.

**This would increase the signal by 30× before detection.** Combined with maser detection:

$$\text{Total SNR gain} \approx 30 \times 8.7 \approx 260\times$$

This is speculative and requires the NV centers to be in direct magnetic contact with tissue protons (distance < 1 nm), which is challenging. However, it's worth investigating as a long-term unique selling point.

### 4.3 Composite Strategy Assessment

| Strategy | SNR Gain | Feasibility | Timeline |
|----------|----------|-------------|----------|
| Room-temp coil + maser (linear) | 1.1× | ✅ Immediate | MVP |
| Peltier-cooled coil + maser (linear) | 1.5× | ✅ Near-term | V1 |
| Room-temp coil + maser (regenerative) | **8.7×** | ⚠️ Requires precision | V2 |
| Peltier coil + regenerative maser | **12×** | ⚠️ Precision + cooling | V2+ |
| + Overhauser DNP | **200+×** | ❓ Research required | V3 (speculative) |

**The regenerative maser regime (Strategy B) is the primary value proposition.** This is what makes the device unique — no other room-temperature detector can achieve 40+ dB coherent gain at GHz frequencies. It requires the maser field to be stabilized to < 10 ppm (already within our twin's shimming capability), and the NMR frequency to remain within the maser gain bandwidth.

---

## 5. System Architecture

### 5.1 System Block Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                     HANDHELD NV-MASER MRI PROBE                       │
│                                                                       │
│  ┌──────────────── PROBE HEAD (< 1.5 kg) ────────────────────┐       │
│  │                                                             │       │
│  │  ┌─── Patient-Facing Side ───┐                              │       │
│  │  │  Surface RF coil (Tx/Rx)  │ ← flat spiral, 3-4 cm dia   │       │
│  │  │  Acoustic coupling pad    │ ← ultrasound gel equivalent  │       │
│  │  │  (optional Peltier cooler)│                              │       │
│  │  └───────────────────────────┘                              │       │
│  │             ↕ (< 5 mm gap)                                  │       │
│  │  ┌─── Single-Sided Magnet ──────────────────────────────┐   │       │
│  │  │  Sweet-spot permanent magnet array                    │   │       │
│  │  │  B₀ = 50-80 mT at sweet spot (1-3 cm into tissue)    │   │       │
│  │  │  Built-in gradient ~5-15 T/m (depth encoding)         │   │       │
│  │  │  Lateral gradient coils (2D in-plane encoding)        │   │       │
│  │  │  Weight: ~0.5-1.0 kg (NdFeB magnets)                 │   │       │
│  │  └──────────────────────────────────────────────────────┘   │       │
│  │             ↕ (mechanically coupled)                        │       │
│  │  ┌─── Maser Module ────────────────────────────────────┐    │       │
│  │  │  Mini Halbach (50 mT, 14 mm bore)                    │    │       │
│  │  │  NV diamond + TE₀₁₁ cavity                           │    │       │
│  │  │  532 nm laser (miniature DPSS, 2W)                    │    │       │
│  │  │  8 shimming coils + MCU                               │    │       │
│  │  │  Up-conversion mixer (NMR freq → 1.47 GHz)           │    │       │
│  │  │  Volume: ~50 mm cube                                  │    │       │
│  │  └──────────────────────────────────────────────────────┘   │       │
│  │                                                             │       │
│  │  ┌─── Probe Electronics ───────────────────────────────┐    │       │
│  │  │  T/R switch                                          │    │       │
│  │  │  RF transmitter (small PA, ~10W peak)                │    │       │
│  │  │  Maser control MCU (shimming, laser, temp)           │    │       │
│  │  │  Pre-ADC conditioning                                │    │       │
│  │  └──────────────────────────────────────────────────────┘   │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                           │                                            │
│                    Cable (USB-C / Thunderbolt)                          │
│    Power + Data + Gradient drive + RF/LO reference                     │
│                           │                                            │
│  ┌──────────── BASE UNIT (optional, < 5 kg) ──────────────────┐       │
│  │  Gradient amplifiers (3 channels, ~5A each)                 │       │
│  │  RF power amplifier (if > 10W needed)                       │       │
│  │  Power supply (battery or AC adapter)                       │       │
│  │  High-speed ADC (16-bit, 10 MS/s)                           │       │
│  │  FPGA/MCU sequence controller                               │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                           │                                            │
│                    USB-C / WiFi / Bluetooth                             │
│                           │                                            │
│  ┌──────────── TABLET / DISPLAY ──────────────────────────────┐       │
│  │  App: Scan planning, real-time preview, AI-enhanced recon   │       │
│  │  Cloud upload for advanced reconstruction (optional)         │       │
│  │  DICOM export, EHR integration                              │       │
│  │  Guided scanning workflow (for non-expert operators)         │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                           │ (optional)                                 │
│                    HIPAA-compliant cloud                                │
│  ┌──────────── CLOUD SERVICES ────────────────────────────────┐       │
│  │  ML-enhanced reconstruction (compressed sensing, denoising) │       │
│  │  AI diagnostic assist ("possible hemorrhage detected")      │       │
│  │  Remote radiologist review                                  │       │
│  │  Fleet management, firmware OTA updates                     │       │
│  └─────────────────────────────────────────────────────────────┘       │
└───────────────────────────────────────────────────────────────────────┘
```

### 5.2 Probe vs. Base Unit Split

The probe contains everything needed for the physics:
- Magnets (B₀), gradients (encoding), RF coil (Tx/Rx), maser (detection)

The base unit handles power-hungry electronics that don't need to be at the probe:
- Gradient power amplifiers (~50W per channel)
- High-speed ADC
- Sequence controller
- Battery

**Alternative: All-in-probe design.** If the gradient requirements are modest (relying mainly on the built-in B₀ gradient for depth encoding), the base unit could be eliminated entirely, with just a cable to the tablet carrying power and data. This would be the ultimate simplicity target.

### 5.3 Frequency Plan (50 mT Imaging Field)

| Signal | Frequency | Origin |
|--------|-----------|--------|
| Proton Larmor at sweet spot | $\gamma_p \times 0.050 = 2.129$ MHz | B₀ at sweet spot |
| NMR readout bandwidth | ±10–25 kHz around Larmor | Gradient-encoded spatial info |
| NV maser carrier | 1.4699 GHz | NV ν₋ = D − γ_e × 0.050 T |
| Up-conversion LO | 1.4699 GHz − 2.129 MHz = 1.46777 GHz | Mixer reference |
| Maser gain bandwidth | ~50 kHz around 1.4699 GHz | Set by Q = 10,000 |
| Gradient update rate | DC – 10 kHz | Sequence timing |

**Critical point**: The maser's 50 kHz gain bandwidth at 1.47 GHz perfectly matches the NMR readout bandwidth. This is not a coincidence — we can tune the maser Q and field to match the imaging requirements. Higher Q (narrower bandwidth) = lower noise but less room for field drift.

---

## 6. Probe Head Design

### 6.1 Single-Sided Magnet Options

#### Option A: U-Shaped Magnet (NMR-MOUSE Style)

```
Cross-section (side view):

    ┌────────┐    ┌────────┐
    │   N    │    │    S   │ ← NdFeB blocks (anti-parallel)
    │  ↑ ↑   │    │  ↓ ↓   │
    └────┬───┘    └───┬────┘
         │   RF coil  │
         │  ╔════════╗│
         └──╣ surface ╠┘
            ╚════════╝  ← skin contact
    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ← tissue surface
    ░░░░░░░░░░░░░░░░░░░░░░
    ░░░ sensitive slice ░░░ ← flat plane at specific depth
    ░░░░░░░░░░░░░░░░░░░░░░
```

- **Pros**: Simple, proven, strong gradient (~20 T/m)
- **Cons**: Very thin sensitive slice (~0.1 mm), 1D profiling only (without lateral gradients)
- **Best for**: Skin/tissue boundary detection, moisture profiling

#### Option B: Sweet-Spot Barrel Magnet

```
Cross-section (side view):

    ┌──────────────────────┐
    │  Magnet ring array   │ ← concentric NdFeB rings
    │  ┌────────────────┐  │    optimized for sweet spot
    │  │    Gap for      │  │
    │  │    RF coil      │  │
    │  └────────────────┘  │
    └──────────┬───────────┘
               │
    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ← tissue surface
    ░░░░░░░░░░░░░░░░░░░░░░
    ░░░   sweet spot    ░░░ ← quasi-homogeneous region at ~2 cm depth
    ░░░ (B₀ ~ uniform)  ░░░
    ░░░░░░░░░░░░░░░░░░░░░░
```

- **Pros**: Homogeneous region → multi-echo sequences → true T2 maps
- **Cons**: More complex magnet design, weaker inherent gradient (need separate gradient coils)
- **Best for**: Quantitative relaxometry, 2D imaging with lateral gradients

#### Option C: Rotating Magnet Array (Cooley/Wald 2021 Style)

A recent innovation: a cylindrical single-sided array that can be rotated to change the encoding direction, enabling back-projection reconstruction.

- **Pros**: 3D-like encoding without gradient coils
- **Cons**: Moving parts, slower acquisition, mechanical complexity
- **Best for**: Research-grade 3D imaging

### 6.2 Recommended Design: Hybrid Sweet-Spot with Lateral Gradients

**Option B is the best fit for our application.** Rationale:

1. **Sweet spot provides T2 contrast** — essential for hemorrhage detection (blood has distinct T2)
2. **Lateral gradient coils enable 2D imaging** — slap a pair of planar gradient coils on the magnet face
3. **Depth encoding from B₀ gradient** — the residual gradient outside the sweet spot encodes depth
4. **Compatible with maser bandwidth** — the sweet spot's field uniformity keeps the NMR signal within the maser's gain band

**Target sweet-spot parameters:**
- B₀ at sweet spot: 50 mT (matches maser's NV transition at 1.47 GHz)
- Sweet spot depth: 15–25 mm from probe surface
- Sweet spot size: ~10 mm diameter × 5 mm deep (quasi-homogeneous to < 500 ppm)
- Residual gradient: ~2–5 T/m (provides ~1 mm depth resolution)
- Lateral gradient coils: switchable, ≤ 50 mT/m

### 6.3 Magnet Weight and Size Estimate

For a 50 mT sweet spot at 2 cm depth:

Using the Halbach-inspired barrel design (concentric rings, N52 NdFeB):
- Outer diameter: ~6–8 cm
- Height: ~5–7 cm
- Magnet mass: ~0.3–0.6 kg (NdFeB density: 7.5 g/cm³)
- Iron yoke (if needed): ~0.1–0.3 kg

**Total magnet assembly: ~0.5–1.0 kg** — feasible for handheld.

### 6.4 Complete Probe Head Layout

```
Side view (cutaway):

    ┌──────────── 8 cm ────────────┐
    │                               │
    │  ┌───── Maser Module ─────┐  │  ← ~5 cm
    │  │  Mini Halbach + diamond │  │
    │  │  Cavity + laser + shims │  │
    │  └────────────────────────┘  │
    │                               │  ← ~5 cm
    │  ┌── Single-Sided Magnet ──┐  │
    │  │  Barrel array (NdFeB)   │  │
    │  │  Lateral gradient coils │  │
    │  └────────────────────────┘  │
    │                               │
    │  ┌── RF Coil Assembly ────┐  │  ← ~1 cm
    │  │  Surface coil (spiral) │  │
    │  │  (Peltier cooler)      │  │
    │  │  Coupling pad          │  │
    │  └────────────────────────┘  │
    └───────────────────────────────┘
                   │
              skin contact

    Total height: ~11 cm
    Width: ~8 cm
    Weight: ~1.0 kg (magnet) + 0.3 kg (maser) + 0.2 kg (electronics/housing)
           = ~1.5 kg total
```

### 6.5 Magnetic Isolation Between Imaging Magnet and Maser Magnet

**Critical design challenge**: The imaging magnet (0.5–1.0 kg of NdFeB) creates stray fields that will affect the maser module's small Halbach (50 mT, 14 mm bore). These must be magnetically isolated.

**Solutions:**
1. **Distance**: Separate by > 5 cm (stray field from imaging magnet falls off as ~1/r³)
2. **Shielding**: Mu-metal shell around the maser module (~50 dB attenuation)
3. **Active compensation**: The maser's 8 shimming coils can correct residual stray fields (this is exactly what the twin models)
4. **Field orientation**: Align the maser Halbach axis perpendicular to the imaging magnet's dominant flux direction

The digital twin's `closed_loop.py` and `disturbance.py` modules can model the stray field from the imaging magnet as a disturbance source and verify that the shimming controller can compensate.

---

## 7. Signal Chain: Tissue → Screen

### 7.1 Complete Signal Path

```
1. B₀ POLARIZATION
   Tissue protons align with imaging magnet's B₀ (~50 mT at sweet spot)
   Equilibrium magnetization: M₀ ∝ B₀/T

2. RF EXCITATION
   Surface Tx coil sends 90° or 180° pulse at f_Larmor (~2.13 MHz)
   Flip angle controlled by pulse amplitude × duration

3. SPATIAL ENCODING
   Depth: Built-in B₀ gradient (protons at different depths → different frequencies)
   Lateral: Switchable planar gradient coils (phase/frequency encoding)
   Slice selection: Not needed (B₀ gradient naturally selects a depth band)

4. NMR SIGNAL
   Precessing magnetization induces EMF in surface Rx coil
   Signal at ~2.13 MHz ± gradient-encoded bandwidth

5. UP-CONVERSION
   Mixer: NMR signal (2.13 MHz) + LO (1.46777 GHz) → USB at 1.4699 GHz
   This places the NMR signal inside the maser's gain band

6. MASER AMPLIFICATION
   NV maser (regenerative, near-threshold): 30–60 dB gain, T_n ~1–5 K
   Output: amplified NMR signal at 1.47 GHz

7. DOWN-CONVERSION + DIGITIZATION
   Mixer: 1.47 GHz → baseband (DC – 50 kHz)
   ADC: 16-bit, 200 kHz sampling

8. CABLE TRANSFER
   Digital data stream over USB-C to tablet
   Raw k-space data: ~100 kB per acquisition

9. RECONSTRUCTION (Tablet or Cloud)
   NUFFT (non-uniform FFT) for non-Cartesian k-space
   Compressed sensing reconstruction (exploit sparsity)
   ML denoising (trained on paired low-SNR/high-SNR data)
   → Final image: 2D slice with MRI contrast

10. DISPLAY + AI ASSIST
    Real-time preview during acquisition
    AI overlay: "possible fluid collection", "tissue boundary"
    Guided scanning: "move probe 2 cm left"
```

### 7.2 Timing: Emergency Triage Workflow

| Phase | Time | Description |
|-------|------|-------------|
| 1. Place probe | 5 s | Operator positions probe on patient |
| 2. Auto-tune | 3 s | Maser lock, frequency calibration, field survey |
| 3. Quick survey | 10 s | 1D depth profile at probe location (like A-mode ultrasound) |
| 4. 2D scan | 30–90 s | Phase-encoded 2D imaging (like B-mode equivalent) |
| 5. AI analysis | 2 s | Cloud ML processes image, returns annotations |
| **Total** | **~50–110 s** | Faster than CT (requires transport), comparable to ultrasound |

---

## 8. SNR Budget (Single-Sided Geometry)

### 8.1 Signal Calculation

**Setup:**
- B₀ at sweet spot: 50 mT (f_Larmor = 2.129 MHz)
- Sweet spot depth: 20 mm from coil plane
- Voxel size: 2 × 2 × 2 mm³ = 8 µL
- Surface coil: 30 mm diameter flat spiral, 5 turns

**Equilibrium magnetization (50 mT, body temp 310K):**
$$M_0 = \frac{n_H \gamma^2 \hbar^2 I(I+1)}{3 k_B T} B_0 = 1.69 \times 10^{-8} \text{ A/m}$$

**Surface coil sensitivity at depth d:**

For a circular coil of radius a at depth d on-axis:
$$\frac{B_1}{I} = \frac{\mu_0 N a^2}{2(a^2 + d^2)^{3/2}}$$

With a = 15 mm, N = 5, d = 20 mm:
$$\frac{B_1}{I} = \frac{4\pi \times 10^{-7} \times 5 \times (0.015)^2}{2 \times ((0.015)^2 + (0.020)^2)^{3/2}} = \frac{1.414 \times 10^{-10}}{2 \times (6.25 \times 10^{-4})^{3/2}}$$

$$= \frac{1.414 \times 10^{-10}}{2 \times 1.563 \times 10^{-5}} = 4.52 \times 10^{-6} \text{ T/A} = 4.52 \text{ µT/A}$$

**EMF per voxel:**
$$\mathcal{E}_{voxel} = \omega_0 \times M_0 \times V_{voxel} \times \frac{B_1}{I}$$

$$= 2\pi \times 2.129 \times 10^6 \times 1.69 \times 10^{-8} \times 8 \times 10^{-9} \times 4.52 \times 10^{-6}$$

$$\mathcal{E}_{voxel} \approx 6.5 \times 10^{-15} \text{ V} = 6.5 \text{ fV}$$

**Note**: This is for a single on-axis voxel at 20 mm depth. At shallower depth (10 mm), the signal is ~8× stronger because $B_1/I \propto 1/(a^2+d^2)^{3/2}$.

### 8.2 Noise Calculation

**Coil thermal noise:**
- Coil resistance at 2 MHz: R_coil ≈ 0.3 Ω (including skin effect)
- Temperature: 300 K
- Bandwidth: 10 kHz (readout window)

$$V_{n,coil} = \sqrt{4 k_B T R \Delta f} = \sqrt{4 \times 1.38 \times 10^{-23} \times 300 \times 0.3 \times 10000} = 7.05 \text{ nV}$$

**Preamp noise (conventional LNA, NF = 1 dB):**
- Noise temperature: 75 K
- $V_{n,preamp} = \sqrt{4 k_B \times 75 \times 0.3 \times 10000} = 3.53$ nV

**Preamp noise (maser, regenerative, T_n = 5 K):**
- $V_{n,maser} = \sqrt{4 k_B \times 5 \times 0.3 \times 10000} = 0.91$ nV

### 8.3 SNR Comparison

**Single-shot SNR per voxel at 20 mm depth:**

| Configuration | V_signal | V_noise_total | SNR₁ |
|--------------|----------|---------------|------|
| Conventional (RT coil + LNA) | 6.5 fV | $\sqrt{7.05^2 + 3.53^2}$ = 7.88 nV | 8.2 × 10⁻⁷ |
| Maser linear (RT coil + maser) | 6.5 fV | $\sqrt{7.05^2 + 0.91^2}$ = 7.11 nV | 9.1 × 10⁻⁷ |
| Maser regenerative (coherent gain G=10⁴) | 6.5 fV × 10⁴ = 65 pV | $\sqrt{(7.05/10^4)^2 + 0.91^2}$ nV ≈ 0.91 nV | **7.1 × 10⁻⁵** |

Wait — the regenerative case needs careful analysis. When the maser provides coherent gain G before noise is imposed:

The signal gets amplified by G: $V_{sig,out} = G \times V_{sig,in}$  
The coil noise also gets amplified: $V_{n,coil,out} = G \times V_{n,coil,in}$  
The maser adds its own noise: $V_{n,maser}$  

$$\text{SNR}_{regen} = \frac{G \times V_{sig}}{G \times V_{n,coil} + V_{n,maser}} \approx \frac{V_{sig}}{V_{n,coil} + V_{n,maser}/G}$$

For large G: $\text{SNR}_{regen} \to V_{sig}/V_{n,coil}$ — the maser noise becomes negligible relative to the amplified coil noise.

**This means even regenerative gain cannot beat the coil thermal noise.** The coil noise is fundamental.

**Correction**: The regenerative advantage comes from making downstream electronics noise irrelevant, NOT from beating the coil noise. The real benefit is:

$$\text{SNR}_{maser} = \frac{V_{sig}}{\sqrt{V_{n,coil}^2 + (V_{n,maser}/1)^2}} = \frac{V_{sig}}{\sqrt{V_{n,coil}^2 + V_{n,maser}^2}}$$

vs. conventional:

$$\text{SNR}_{conv} = \frac{V_{sig}}{\sqrt{V_{n,coil}^2 + V_{n,preamp}^2}}$$

$$\frac{\text{SNR}_{maser}}{\text{SNR}_{conv}} = \frac{\sqrt{V_{n,coil}^2 + V_{n,preamp}^2}}{\sqrt{V_{n,coil}^2 + V_{n,maser}^2}} = \frac{\sqrt{7.05^2 + 3.53^2}}{\sqrt{7.05^2 + 0.91^2}} = \frac{7.88}{7.11} = 1.11$$

**Same result: ~11% improvement.** The coil noise at 300 K dominates.

### 8.4 The Path to Useful SNR

Single-shot SNR of ~10⁻⁶ per voxel is unusable. We need to combine every available tool:

| Tool | Gain Factor | Mechanism |
|------|-------------|-----------|
| **Averaging** (1024 NEX) | 32× | $\sqrt{N}$ |
| **TR = 100 ms** (fast steady-state) | 10× more averages/min | Short T1 at low field |
| **Larger voxels** (3×3×3 mm = 27 µL) | 3.4× | Volume ∝ L³ |
| **Closer depth** (10 mm vs 20 mm) | 5.6× | $(a^2+d_2^2)^{3/2}/(a^2+d_1^2)^{3/2}$ |
| **Cooled coil** (200 K Peltier) | 1.2× | $\sqrt{300/200}$ |
| **Maser preamp** | 1.1× | Replaces 75K → 5K preamp |
| **Tuned coil Q** (Q=80 matched) | 2× | Resonant voltage enhancement |
| **ML denoising** | 3–5× | Trained DnCNN or similar |
| **Compressed sensing** | 2–3× | Under-sample k-space, reconstruct |
| **Combined** | **~2,400–12,000×** | Product of all factors |

With all optimizations at 10 mm depth:

$$\text{SNR}_{final} = 9.1 \times 10^{-7} \times 32 \times 3.4 \times 5.6 \times 1.2 \times 1.1 \times 2 \times 4 \times 2.5 \approx 6.6$$

**SNR ≈ 7 at 10 mm depth** with 1024 averages and 3 mm voxels.

At 100 ms TR with 1024 averages: scan time = 1024 × 0.1 = **~100 seconds** (within our 2-minute target).

At 5 mm depth: signal is ~23× stronger → SNR ≈ 50 (excellent).

### 8.5 Depth-dependent SNR Profile

| Depth (mm) | Relative B₁/I | SNR (3mm voxel, 100s scan, all optimizations) | Clinical Utility |
|------------|---------------|-----------------------------------------------|-----------------|
| 5 | 1.0 (reference) | ~50 | Excellent — skin, superficial tissue |
| 10 | 0.18 | ~9 | Good — subcutaneous, tendons |
| 15 | 0.057 | ~3 | Marginal — deep tissue |
| 20 | 0.023 | ~1.2 | Barely detectable |
| 30 | 0.006 | ~0.3 | Not useful |

**Practical imaging depth: 0–15 mm with diagnostic quality, 15–25 mm with averaging.**

This matches the clinical use cases: subcutaneous hemorrhage, tendon injury, fracture detection (cortical bone is < 10 mm from skin at most extremity sites), muscle edema.

---

## 9. Imaging Modes & Clinical Use Cases

### 9.1 Imaging Modes

#### Mode 1: Depth Profile (A-mode equivalent)
- **Sequence**: CPMG (multi-echo) or saturation recovery
- **Encoding**: Built-in B₀ gradient provides depth encoding
- **Output**: 1D profile of T2 (or T1) vs. depth
- **Time**: 10–30 seconds
- **Resolution**: ~0.5–2 mm depth resolution
- **Clinical value**: Fluid detection (long T2 = bright), tissue interfaces

```
Display:

Depth (mm)  0   5   10   15   20
            |---|---|---|---|---|
T2 signal:  ████████░░░░████░░░
            skin  tissue  fluid  bone
                        ↑
                   hemorrhage?
```

#### Mode 2: 2D Slice (B-mode equivalent)
- **Sequence**: Spin echo or gradient echo
- **Encoding**: Depth from B₀ gradient + lateral from gradient coils (phase encoding)
- **Output**: 2D cross-sectional image with T1/T2 contrast
- **Time**: 30–120 seconds
- **Resolution**: 2–3 mm in-plane, 2–5 mm depth
- **Clinical value**: Spatial extent of injury, tissue morphology

#### Mode 3: Tissue Characterization Map
- **Sequence**: Multi-echo CPMG → T2 map; inversion recovery → T1 map
- **Output**: Quantitative relaxation time maps (not just contrast-weighted images)
- **Time**: 60–180 seconds
- **Clinical value**: Pathology-specific (T2 of hemorrhage vs. edema vs. normal tissue)
- **Advantage over ultrasound**: Quantitative tissue typing, not just structural

### 9.2 Clinical Use Cases

| Clinical Scenario | Mode | What MRI Sees That Ultrasound Can't | Time |
|-------------------|------|--------------------------------------|------|
| **Subcutaneous hemorrhage** | Profile + 2D | T2 distinguishes fresh blood (short T2) from old (long T2) | 30s |
| **Compartment syndrome** | Profile | Fluid pressure → fascial plane fluid (long T2, bright) | 15s |
| **Fracture assessment** | 2D | Bone marrow edema (T2 bright) confirms occult fractures | 60s |
| **Soft tissue swelling** | T2 map | Quantitative edema measurement, track over time | 90s |
| **Pneumothorax (exploratory)** | Profile | Air vs. tissue (zero signal vs. tissue signal) | 10s |
| **Burns depth assessment** | Profile | Viable tissue T2 vs. necrotic (different relaxation) | 20s |
| **Tendon/ligament injury** | 2D | Partial tears show edema pattern (T2 bright) | 60s |
| **Abscess vs. cellulitis** | T2 map | Abscess = fluid collection (very long T2); cellulitis = diffuse edema | 60s |

### 9.3 Key Advantage: MRI Contrast Without MRI Size

Ultrasound shows **structural** boundaries (acoustic impedance differences).  
MRI shows **tissue type** (water content, mobility, chemical environment).

For emergency medicine, the tissue-typing ability is critical:
- **"Is there bleeding?"** → T2 signature change
- **"Is the tissue alive?"** → Perfusion-weighted or T2 mapping
- **"How deep is the injury?"** → Quantitative depth profile

No other handheld device provides this information.

---

## 10. Compute Architecture (Tablet + Cloud)

### 10.1 Data Flow

```
Probe → Base Unit → Tablet → Cloud (optional)
 │         │          │          │
 │    ADC + FPGA     App      ML inference
 │    Raw data      Preview   Advanced recon
 │    ~2 MB/scan    Display   AI diagnostics
```

### 10.2 Tablet App Requirements

| Feature | Implementation | Rationale |
|---------|---------------|-----------|
| Real-time preview | Streaming FFT display during acquisition | Operator feedback, positioning guide |
| Guided scanning | AR overlay with body landmarks | Non-expert operators in emergency |
| On-device reconstruction | Basic FFT + simple denoising | Works offline (no WiFi needed) |
| Cloud reconstruction | Upload raw k-space, receive ML-enhanced image | 10× better image quality |
| AI diagnostic assist | Highlight regions of concern | "Possible fluid collection at 12 mm depth" |
| DICOM/HL7 export | Standard medical imaging format | EHR integration |
| Report generation | Template + measurements | Clinical documentation |
| Multi-scan comparison | Overlay time-series scans | Track treatment response |

### 10.3 Compute Requirements

| Level | Hardware | Capability | Latency |
|-------|----------|------------|---------|
| **On-probe** | MCU (ESP32-S3 or STM32H7) | Sequence control, shimming, basic diagnostics | < 1 ms |
| **Base unit** | FPGA (Artix-7) or RPi CM4 | ADC acquisition, pulse sequence execution, raw data buffering | < 10 µs timing |
| **Tablet** | iPad Pro / Android tablet | Basic 2D FFT reconstruction, display, UI | ~1 s |
| **Cloud** | GPU instance (A100/T4) | Compressed sensing, ML denoising, AI diagnostics | 2–5 s (with upload) |

### 10.4 ML/AI Integration

| Model | Purpose | Training Data | Architecture |
|-------|---------|---------------|--------------|
| **Denoiser** | Enhance low-SNR images | Paired low/high-SNR knee/wrist MRI | DnCNN or U-Net |
| **Compressed sensing recon** | Reconstruct from under-sampled k-space | Retrospectively under-sampled clinical MRI | Unrolled ADMM / MoDL |
| **Tissue classifier** | Segment hemorrhage/edema/normal | Labeled clinical data | 2D U-Net / nnU-Net |
| **Positioning guide** | Guide operator to optimal probe position | Body landmark + tissue quality feedback | CNN on depth profiles |

---

## 11. Digital Twin Module Plan

### 11.1 Current Twin vs. Handheld Requirements

| Existing Module | Reusable? | Notes |
|----------------|-----------|-------|
| `halbach.py` | ✅ Partially | Maser's internal Halbach is unchanged. Need new single-sided imaging magnet model |
| `nv_spin.py` | ✅ Fully | NV physics is geometry-independent |
| `maser_gain.py` | ✅ Fully | Maser gain model unchanged |
| `cavity.py` | ✅ Fully | Cavity QED unchanged |
| `optical_pump.py` | ✅ Fully | 532 nm pump is geometry-independent |
| `signal_chain.py` | ⚠️ Partially | Need surface coil model instead of solenoid; add up-conversion mixer |
| `coils.py` | ✅ Partially | Shimming coils for maser. Need new planar gradient coils for imaging |
| `closed_loop.py` | ✅ Fully | Shimming controller is geometry-independent |
| `thermal.py` | ✅ Fully | Thermal model applies to maser module |
| `disturbance.py` | ⚠️ Needs extension | Add stray field from imaging magnet as disturbance source |
| `calibration/` | ✅ Fully | Field maps, uniformity analysis |

### 11.2 New Modules Required

| New Module | Purpose | Priority |
|------------|---------|----------|
| **`single_sided_magnet.py`** | Model B₀ field from barrel/U-shaped single-sided magnet array | **P0 — Critical** |
| **`surface_coil.py`** | Surface coil B₁ sensitivity vs. position, impedance, noise model | **P0 — Critical** |
| **`depth_profile.py`** | 1D NMR depth profiling through B₀ gradient, CPMG simulation | **P0 — Critical** |
| **`up_conversion.py`** | Mixer model: NMR frequency → maser frequency, noise contribution | **P1 — Important** |
| **`pulse_sequence.py`** | Basic spin echo / CPMG / GRE sequence simulator | **P1 — Important** |
| **`planar_gradient.py`** | Flat gradient coil model for lateral encoding | **P2 — Later** |
| **`reconstruction.py`** | k-space → image (FFT, NUFFT, compressed sensing) | **P2 — Later** |
| **`snr_calculator.py`** | End-to-end SNR from tissue to image, parametric sweeps | **P1 — Important** |

### 11.3 Module Specifications

#### `single_sided_magnet.py` (P0)

```python
# Key classes and functions:

class SingleSidedMagnet:
    """Model for barrel/U-shaped single-sided permanent magnet array.
    
    Computes B₀ field in the half-space above the magnet surface,
    including sweet-spot optimization and spatial gradient characterization.
    """
    def __init__(self, magnet_type: str, magnet_params: dict): ...
    def field_at(self, positions: np.ndarray) -> np.ndarray: ...
    def sweet_spot_location(self) -> tuple[float, float, float]: ...
    def gradient_at(self, position: np.ndarray) -> np.ndarray: ...
    def field_map_2d(self, plane: str, depth: float, extent: float, resolution: int) -> FieldMap: ...
    def optimize_sweet_spot(self, target_depth: float, target_b0: float) -> dict: ...

# Magnet types:
# - "u_shaped": NMR-MOUSE style with two anti-parallel blocks
# - "barrel": Concentric annular rings optimized for sweet spot
# - "halbach_single_sided": One-sided Halbach array
```

#### `surface_coil.py` (P0)

```python
class SurfaceCoil:
    """Flat spiral surface coil for single-sided NMR Tx/Rx.
    
    Models B₁ field, sensitivity profile, impedance, and noise
    as functions of position relative to the coil plane.
    """
    def __init__(self, radius_mm: float, n_turns: int, wire_gauge: int): ...
    def b1_per_amp(self, positions: np.ndarray) -> np.ndarray: ...
    def sensitivity(self, positions: np.ndarray) -> np.ndarray: ...
    def impedance(self, frequency_hz: float, temperature_k: float) -> complex: ...
    def thermal_noise(self, bandwidth_hz: float, temperature_k: float) -> float: ...
    def body_noise(self, frequency_hz: float, tissue_conductivity: float, volume_m3: float) -> float: ...
    def snr_per_voxel(self, position: np.ndarray, voxel_size_m: float, m0: float, bandwidth_hz: float) -> float: ...
```

#### `depth_profile.py` (P0)

```python
class DepthProfileSimulator:
    """Simulate 1D NMR depth profiling through single-sided magnet gradient.
    
    Given a magnet field profile and tissue model, simulates the NMR signal
    vs. frequency (which maps to depth via the gradient).
    """
    def __init__(self, magnet: SingleSidedMagnet, coil: SurfaceCoil): ...
    def tissue_model(self, layers: list[TissueLayer]) -> None: ...
    def simulate_profile(self, sequence: str, params: dict) -> DepthProfile: ...
    def add_noise(self, profile: DepthProfile, snr_config: dict) -> DepthProfile: ...
```

---

## 12. Hardware Development Path

### 12.1 Revised Build Phases

The hardware build phases shift from "maser module only" to "complete handheld probe":

| Phase | Scope | Deliverable | Dependencies |
|-------|-------|-------------|--------------|
| **Phase 0: Research & Simulation** | Literature, simulation, magnet optimization | Validated sweet-spot magnet design (in twin) | None |
| **Phase 1: Maser Module** | Build the NV maser (unchanged from original plan) | Working maser oscillation | Phase 0 |
| **Phase 2: Single-Sided Magnet** | Design, order NdFeB, 3D-print jig, assemble, map | Measured field profile with sweet spot | Phase 0 |
| **Phase 3: Surface Coil & Receive** | PCB coil, T/R switch, up-conversion mixer, maser integration | NMR signal detection at bench | Phases 1, 2 |
| **Phase 4: First Depth Profile** | Combine magnet + coil + maser, acquire from phantom | 1D T2 depth profile of test sample | Phase 3 |
| **Phase 5: Lateral Gradients** | Design/build planar gradient coils, pulse sequencer | 2D encoded raw data | Phase 4 |
| **Phase 6: Reconstruction** | NUFFT + compressed sensing on tablet | First 2D image on screen | Phases 4, 5 |
| **Phase 7: Probe Integration** | 3D-printed housing, cable, thermal management | Handheld prototype | Phase 6 |
| **Phase 8: Phantom Validation** | Tissue-mimicking phantoms, SNR/resolution measurements | Validated performance specs | Phase 7 |
| **Phase 9: Tissue Imaging** | Ex vivo tissue (chicken, pork), then in vivo (forearm skin) | First human tissue image | Phase 8 |

### 12.2 Critical Milestones

| Milestone | Success Criterion | Phase |
|-----------|-------------------|-------|
| **Maser oscillation** | Detectable stimulated emission at 1.47 GHz | 1 |
| **Sweet-spot field quality** | B₀ = 50 ± 5 mT at 20 mm depth, < 500 ppm over 10 mm | 2 |
| **First NMR signal** | FID or echo from water sample at known depth | 4 |
| **Depth profile** | Resolvable layers in layered phantom | 4 |
| **First 2D image** | Grid phantom resolved at 3 mm | 6 |
| **Tissue contrast** | T2 difference visible between fat and muscle | 9 |

---

## 13. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|-----------|------------|
| R1 | Sweet-spot magnet field too weak at target depth | No signal | Medium | Optimize magnet geometry in simulation before building; increase magnet mass if needed |
| R2 | Maser gain bandwidth too narrow for readout | Truncated signal → artifacts | Medium | Match maser Q to readout BW; use lower Q cavity or frequency-tracking |
| R3 | Stray field from imaging magnet destabilizes maser | Maser fails | High | Mu-metal shielding, distance, active compensation (twin models this) |
| R4 | Peltier cooler adds too much weight/heat to probe | Can't achieve < 1.5 kg | Low | Defer cooling to V2; run RT coil for MVP |
| R5 | Regulatory delay (510(k) process) | Delayed market entry | High | Start pre-submission meeting with FDA early (Phase 6+) |
| R6 | RF interference in unshielded environment | Image artifacts | Medium | Maser's natural bandpass helps; apply real-time interference cancellation |
| R7 | SNR insufficient at 15+ mm depth | Limited clinical utility | Medium | Accept depth limitation; focus on 5–15 mm use cases for V1 |
| R8 | Reconstruction artifacts from non-Cartesian k-space | Poor image quality | Low | Use proven NUFFT libraries; train ML denoiser |
| R9 | Up-conversion mixer adds unacceptable noise | SNR gain lost | Medium | Use high-quality passive mixer or parametric frequency conversion |
| R10 | Maser regenerative mode unstable | Oscillation or no gain | High | Digital twin models threshold precisely; implement gain-lock control loop |

---

## 14. Regulatory Considerations

### 14.1 FDA Classification

**Most likely pathway: 510(k) — Class II medical device**

Predicate devices:
- Hyperfine Swoop (K200352) — portable MRI, 510(k) cleared Dec 2020
- NMR-MOUSE derivatives used in clinical research
- Point-of-care ultrasound devices (for form factor comparison)

**Product code**: LNH (Magnetic resonance imaging system) or QKO (Imaging, NMR)

### 14.2 Key Standards

| Standard | Scope | Applicability |
|----------|-------|---------------|
| IEC 60601-1 | General medical device safety | ✅ Required |
| IEC 60601-2-33 | MRI-specific safety | ✅ Required (SAR limits, field limits, noise) |
| IEC 62570 | MR safety labeling | ✅ Required (for use near other devices) |
| FDA guidance on AI/ML | AI-enabled diagnostic assist | ✅ If AI overlay is used |
| HIPAA | Patient data privacy | ✅ For cloud features |

### 14.3 Safety Considerations Specific to Handheld

| Concern | Risk Level | Mitigation |
|---------|-----------|------------|
| Laser exposure (532 nm, 2W) | ⚠️ Class 4 | Fully enclosed in probe housing; interlock switches |
| RF exposure (SAR) | Low (< 1 W/kg at these powers) | Calculate SAR per IEC 60601-2-33 |
| Magnetic projectile risk | Low (50 mT, self-shielded) | 5-gauss line well within probe housing |
| Thermal (probe surface temp) | Medium | Temperature sensor + shutdown; max 41°C skin contact |
| EMI to other devices | Low | Emissions testing per IEC 60601-1-2 |

---

## Appendix A: Key Physics Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $\gamma_p$ | 42.577 MHz/T | Proton gyromagnetic ratio |
| $\gamma_e$ | 28.025 GHz/T | Electron gyromagnetic ratio |
| D | 2.87 GHz | NV zero-field splitting |
| $k_B$ | 1.381 × 10⁻²³ J/K | Boltzmann constant |
| $\mu_0$ | 4π × 10⁻⁷ T·m/A | Vacuum permeability |
| $\hbar$ | 1.055 × 10⁻³⁴ J·s | Reduced Planck constant |

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **NMR-MOUSE** | Mobile Universal Surface Explorer — single-sided NMR device |
| **Sweet spot** | Region outside single-sided magnet where B₀ is locally homogeneous |
| **CPMG** | Carr-Purcell-Meiboom-Gill — multi-echo pulse sequence for T2 measurement |
| **Surface coil** | Flat RF coil placed on body surface for local Tx/Rx |
| **Regenerative amplifier** | Amplifier operating near oscillation threshold for very high gain |
| **Compressed sensing** | Reconstruction from under-sampled data using sparsity priors |
| **NUFFT** | Non-uniform Fast Fourier Transform — for non-Cartesian k-space data |
| **510(k)** | FDA premarket notification pathway for Class II medical devices |
