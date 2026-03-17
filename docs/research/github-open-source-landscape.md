# Open-Source GitHub Landscape: Handheld NV-Diamond Maser MRI Probe

**Date**: 2026-03-17  
**Purpose**: Survey of existing open-source codebases that can accelerate development of the handheld NV-diamond maser MRI probe  
**Method**: Systematic GitHub search across 40+ query combinations covering permanent magnet simulation, NV center physics, low-field MRI hardware/software, pulse sequence design, MRI reconstruction, and open-source consoles

---

## Executive Summary

The open-source MRI ecosystem is surprisingly mature, with multiple complete systems (OCRA, MRI4ALL), validated physics libraries (magpylib, SigPy, KomaMRI), and an excellent curated resource list (Ultra-Low-Field-MRI). However, **no existing project combines NV-diamond maser amplification with single-sided MRI** — our unique contribution. We can leverage substantial existing work for magnet design, coil optimization, pulse sequence simulation, and image reconstruction while focusing our novel efforts on the maser preamplifier integration and up-conversion chain.

**Key finding**: The NV center simulation space is nascent (0-1 stars per repo), while the low-field MRI space is well-established. Our competitive advantage lies in bridging these two domains.

---

## Tier 1: DIRECTLY INTEGRATE — Can be used as dependencies or code in our twin/hardware

### 1. magpylib/magpylib ⭐ 340
**URL**: https://github.com/magpylib/magpylib  
**Language**: Python | **License**: BSD-2 | **Last updated**: 2026-03-15  
**What it does**: Computes static magnetic fields of permanent magnets, currents, and moments using analytical expressions in vectorized form. Supports cuboid, cylinder, sphere magnets, current loops, dipoles. Full 3D visualization via Matplotlib/Plotly/Pyvista.

**Why it matters for us**:
- **Can REPLACE our analytical `single_sided_magnet.py` barrel magnet model** with a validated, optimized 3D calculation
- Supports NdFeB-class magnet parameters directly
- Handles complex magnet assemblies (segmented rings, Halbach arrays)
- SI units (Version 5), extremely fast vectorized computation
- Python 3.11+, NumPy/SciPy stack — drops right into our twin

**Integration plan**:
```python
import magpylib as magpy
# Model our barrel magnet as a magnetized cylinder
magnet = magpy.magnet.Cylinder(
    polarization=(0, 0, 1.2),  # NdFeB N42, ~1.2 T remnance
    dimension=(0.06, 0.04),     # 60mm dia, 40mm height
)
# Compute sweet-spot field at probe depth
B = magpy.getB(magnet, observers=[(0, 0, 0.020)])  # 20mm depth
```

**Effort**: LOW — `pip install magpylib`, create adapter functions to feed into our existing `SingleSidedMagnetConfig`

---

### 2. mikgroup/sigpy ⭐ 335
**URL**: https://github.com/mikgroup/sigpy  
**Language**: Python | **License**: BSD-3 | **Last updated**: 2026-03-11  
**What it does**: Signal processing with emphasis on iterative methods. Includes dedicated `sigpy.mri` submodule for MRI reconstruction and `sigpy.mri.rf` for RF pulse design. GPU-accelerated via CuPy.

**Why it matters for us**:
- **MRI reconstruction algorithms**: compressed sensing, NUFFT, iterative methods
- **RF pulse design tools**: critical for optimizing our surface coil excitation at 2.13 MHz
- PyTorch interoperability for future ML reconstruction
- Compressed sensing reconstruction essential for our SNR-limited single-sided regime

**Integration plan**: Use `sigpy.mri` for 1D depth profile reconstruction; use `sigpy.mri.rf` to design optimized RF pulses for inhomogeneous B₀ sweet-spot

**Effort**: LOW-MEDIUM — requires learning API, but well-documented with tutorials

---

### 3. MRsources/MRzero-Core ⭐ 67
**URL**: https://github.com/MRsources/MRzero-Core  
**Language**: Python + Rust | **License**: Unknown | **Last updated**: 2026-03-11  
**What it does**: MRI sequence building, simulation, and reconstruction. Simulates Bloch equations with GPU acceleration via PyTorch. **Directly imports Pulseq .seq files** for one-line simulation.

**Why it matters for us**:
- **Bloch equation simulation** validates our `depth_profile.py` analytical model
- Can simulate realistic NMR signals under inhomogeneous B₀ (exactly our scenario)
- PyTorch backend enables gradient-based sequence optimization
- Pulseq-compatible — industry standard format

**Integration plan**: Use MRzero to validate our analytical depth profiling against full Bloch simulation. Design optimal pulse sequences for the sweet-spot geometry.

**Effort**: MEDIUM — PyTorch dependency, Rust compilation, but powerful

---

### 4. shoham-b/NVision ⭐ 0 (brand new, actively developed)
**URL**: https://github.com/shoham-b/NVision  
**Language**: Python | **Last updated**: 2026-03-17 (TODAY)  
**What it does**: Framework for NV center simulation. Includes experiment generators, noise models, peak-finding locator strategies, and Bayesian visualization. Uses Polars for data, Plotly for visualization.

**Why it matters for us**:
- **Most directly relevant to our NV physics** — NV center signal generation, noise modeling
- Modular architecture with Locator protocol for peak-finding (useful for ODMR analysis)
- Active development, modern Python stack (uv, Polars, Typer CLI)
- Could cross-validate our NV maser gain/noise models

**Integration plan**: Study their NV center generator implementation. Potentially borrow noise model architecture. Compare their ODMR simulation with our maser gain calculations.

**Effort**: MEDIUM — needs evaluation of actual physics depth vs our existing twin

---

## Tier 2: VALIDATE AGAINST — Help verify our physics models and architecture

### 5. JuliaHealth/KomaMRI.jl ⭐ 189
**URL**: https://github.com/JuliaHealth/KomaMRI.jl  
**Language**: Julia | **License**: MIT | **Last updated**: 2026-03-13  
**What it does**: Pulseq-compatible MRI acquisition simulator. GPU-accelerated Bloch equations. Full sequence → signal → image pipeline. Supports diffusion, cardiac, general scenarios.

**Why it matters for us**:
- Gold-standard MRI simulation for validating our simplified depth profile model
- Can simulate realistic acquisitions in inhomogeneous fields
- Julia performance but requires Julia installation

**Use case**: Cross-validate our Python depth profile SNR predictions against KomaMRI's full Bloch simulation. If numbers agree within 10%, our analytical model is trustworthy.

**Effort**: MEDIUM-HIGH — Julia ecosystem, but worth it for validation

---

### 6. bretglun/BlochBuster ⭐ 54
**URL**: https://github.com/bretglun/BlochBuster  
**Language**: Python | **Last updated**: Active  
**What it does**: Graphical NMR Bloch equation simulator. Visualizes magnetization dynamics under RF pulses, gradients, and relaxation.

**Why it matters for us**:
- Educational validation of our Bloch dynamics assumptions
- Quick visual debugging of pulse sequence behavior in the sweet-spot
- Verify T1/T2 relaxation at 50 mT (our operating field)

**Effort**: LOW — standalone educational tool

---

### 7. samuel-gythia/nv-diamond-magnetometry ⭐ 1
**URL**: https://github.com/samuel-gythia/nv-diamond-magnetometry  
**Language**: Jupyter Notebook  
**What it does**: Ramsey interferometry simulation for NV center quantum sensing. Vector magnetometry using diamond probe design.

**Why it matters for us**:
- Validates our NV center frequency splitting calculations (D = 2.87 GHz)
- Ramsey protocol implementation we could adapt for maser threshold calibration
- Small but directly relevant physics

**Effort**: LOW — Jupyter notebooks, quick to review

---

### 8. samuel-gythia/nv-quantum-field-mapping ⭐ 0
**URL**: https://github.com/samuel-gythia/nv-quantum-field-mapping  
**Language**: Python (QuTiP-based)  
**What it does**: QuTiP-based quantum simulation of NV center Ramsey field mapping.

**Why it matters for us**:
- QuTiP integration pattern for quantum-level NV simulation
- Could enhance our twin with quantum-accurate noise modeling
- Field mapping relevant to understanding maser behavior in inhomogeneous B₀

**Effort**: MEDIUM — requires QuTiP dependency

---

### 9. abhimanyumagapu/NVCentre-ODMR ⭐ 1
**URL**: https://github.com/abhimanyumagapu/NVCentre-ODMR  
**Language**: Python  
**What it does**: Python simulation of NV centre ODMR calculation in diamond.

**Why it matters for us**:
- ODMR spectrum simulation directly relevant to maser frequency characterization
- Validates our NV energy level calculations

**Effort**: LOW — straightforward Python

---

## Tier 3: LEARN FROM — Architecture, hardware reference, domain knowledge

### 10. OpenMRI/ocra ⭐ 48
**URL**: https://github.com/OpenMRI/ocra  
**Language**: Python | **Last updated**: 2026-02-25  
**What it does**: Open Source Console for Realtime Acquisitions. Complete MRI console using Red Pitaya FPGA board. Includes pulse sequencer, ADC, shim control. Supports Eclypse-Z7 and Snickerdoodle boards.

**Why it matters for us**:
- **Hardware reference architecture** for our console electronics
- Red Pitaya approach could work for our 2.13 MHz acquisition
- Pulse sequence control, ADC sampling, and reconstruction pipeline
- Could adapt their FPGA pulse sequencer for our simpler 1D profiling use case

**Learn**: Console architecture, FPGA control, real-time data acquisition at low frequencies

---

### 11. mri4all/console ⭐ 20
**URL**: https://github.com/mri4all/console  
**Language**: Python/PyQt5 | **Last updated**: 2026-03-10  
**What it does**: Complete MRI console software for the Zeugmatron Z1 scanner. Python 3, PyQt5 GUI, Ubuntu. Includes custom sequences and reconstruction integration.

**Why it matters for us**:
- **Full MRI software stack** from sequence → acquisition → reconstruction → display
- Architecture patterns for our tablet/cloud UI
- Open-source, well-documented from hackathon
- Integration patterns for custom acquisition hardware

**Learn**: Software architecture for portable MRI console, UI patterns

---

### 12. schote/nexus-console ⭐ 22
**URL**: https://github.com/schote/nexus-console  
**Language**: Python  
**What it does**: MRI console using pypulseq + Spectrum Instrumentation data acquisition cards. Bridges sequence design to hardware acquisition.

**Why it matters for us**:
- pypulseq → hardware pipeline we need to replicate
- Shows how to bridge software pulse sequences to real DAC/ADC hardware
- Spectrum cards could be an alternative to Red Pitaya for our console

**Learn**: pypulseq integration with real hardware

---

### 13. Jagent-x/Ultra-Low-Field-MRI ⭐ (curated list)
**URL**: https://github.com/Jagent-x/Ultra-Low-Field-MRI  
**What it does**: **GOLDMINE** — Comprehensive curated list of ALL low-field MRI resources: papers, videos, open-source projects, companies, system designs, B₀ design, gradient coils, RF coils, EMI mitigation, reconstruction, deep learning approaches.

**Why it matters for us**:
- Links to ALL relevant open-source projects (see below)
- 100+ papers organized by subsystem
- Companies to watch: Hyperfine, Promaxo, Neuro42, Multiwave
- Conference links (ISMRM workshops, MRI4ALL hackathon)

**Critical open-source projects referenced**:
| Project | URL | Purpose |
|---------|-----|---------|
| LUMC-LowFieldMRI/HalbachOptimisation | github.com/LUMC-LowFieldMRI/HalbachOptimisation | Genetic algorithm for Halbach array homogeneity |
| menkueclab/HalbachMRIDesigner | github.com/menkueclab/HalbachMRIDesigner | Parameterized Halbach cylinder → OpenSCAD + FEM |
| kev-m/pyCoilGen | github.com/kev-m/pyCoilGen | Gradient coil winding layout generator (Python port) |
| Philipp-MR/CoilGen | github.com/Philipp-MR/CoilGen | Original MATLAB gradient coil design |
| LUMC-LowFieldMRI/GradientDesignTool | github.com/LUMC-LowFieldMRI/GradientDesignTool | Gradient coil design for low-field |
| opensourceimaging/cosi-transmit | github.com/opensourceimaging/cosi-transmit | RF coil design toolkit |
| OCRA | openmri.github.io/ocra/ | Open MRI console (see #10) |
| OSI² | gitlab.com/osii | Open Source Imaging Initiative (GitLab) |

---

### 14. LUMC-LowFieldMRI/HalbachOptimisation
**URL**: https://github.com/LUMC-LowFieldMRI/HalbachOptimisation  
**What it does**: Genetic algorithm to optimize homogeneity of Halbach arrays by varying ring diameters.

**Why it matters for us**: While our probe is single-sided (not a Halbach cylinder), the GA optimization approach could be adapted for our barrel magnet sweet-spot optimization.

---

### 15. menkueclab/HalbachMRIDesigner  
**URL**: https://github.com/menkueclab/HalbachMRIDesigner  
**Language**: Python  
**What it does**: Parameterized Halbach cylinder design → OpenSCAD geometry → FEM simulation with Gmsh + GetDP. Full mechanical design pipeline.

**Why it matters for us**: The magnet → CAD → FEM pipeline is exactly what we need for hardware prototyping. Even though our geometry differs, the toolchain (OpenSCAD, Gmsh, FEM) applies directly.

---

### 16. kev-m/pyCoilGen ⭐ (Python)
**URL**: https://github.com/kev-m/pyCoilGen  
**Language**: Python  
**What it does**: Generates gradient coil winding layouts using boundary element methods. Produces non-overlapping wire tracks on 3D support structures.

**Why it matters for us**: While we don't use traditional gradient coils (frequency-encoding in stray field), the coil design methodology could help optimize our surface coil geometry and any shim coils needed.

---

### 17. zaccharieramzi/fastmri-reproducible-benchmark ⭐ 161
**URL**: https://github.com/zaccharieramzi/fastmri-reproducible-benchmark  
**Language**: Python/TensorFlow  
**What it does**: Reproducible benchmark for MRI reconstruction methods on the fastMRI dataset. Includes XPDNet (2020 fastMRI challenge runner-up) and other unrolled reconstruction algorithms.

**Why it matters for us**: Contains state-of-the-art deep learning reconstruction algorithms. Our low-SNR regime will likely benefit from learned reconstruction methods.

---

### 18. khammernik/sigmanet ⭐ 53
**URL**: https://github.com/khammernik/sigmanet  
**Language**: Python  
**What it does**: Iterative deep neural network for parallel MRI reconstruction (Σ-net). Learning-based approach to accelerated MRI.

**Why it matters for us**: Architecture reference for our future ML reconstruction module.

---

### 19. mrphysics-bonn/python-ismrmrd-reco ⭐ 27
**URL**: https://github.com/mrphysics-bonn/python-ismrmrd-reco  
**Language**: Python  
**What it does**: MRI reconstruction pipeline using Python + ISMRMRD raw data format + BART toolbox integration.

**Why it matters for us**: Shows how to integrate BART reconstruction toolbox with Python pipeline. ISMRMRD format is the standard for raw MRI data.

---

### 20. ovedtal1/ProjectC ⭐ 0
**URL**: https://github.com/ovedtal1/ProjectC  
**Language**: Python  
**What it does**: Deep learning for low-field MRI reconstruction specifically.

**Why it matters for us**: Directly addresses our low-field reconstruction challenge. Worth reviewing their architecture choices for SNR-limited regimes.

---

## Gap Analysis: What DOESN'T Exist

| Domain | Status | Our Opportunity |
|--------|--------|----------------|
| **NV-diamond maser amplifier simulation** | **NOTHING** (0 repos) | We are building the first open-source maser digital twin |
| **Maser + MRI integration** | **NOTHING** | Completely novel combination |
| **Single-sided NMR depth profiling simulation** | **Very little** | Our `depth_profile.py` fills a gap |
| **Up-conversion receiver chain simulation** | **NOTHING** | 2.13 MHz → 1.47 GHz chain is unique |
| **Handheld MRI probe (non-bore)** | **Limited** | Promaxo/Neuro42 are commercial, not open |
| **Low-field MRI + quantum amplifier** | **NOTHING** | Our entire value proposition |

---

## Recommended Integration Priority

### Phase 1 (Immediate — P0 enhancement)
1. **`pip install magpylib`** → Replace analytical magnet model with validated 3D field computation
2. **Review NVision** → Compare NV simulation approaches, borrow noise model patterns
3. **Review NVCentre-ODMR** → Validate ODMR spectrum calculations

### Phase 2 (P1 — Reconstruction & Sequences)
4. **`pip install sigpy`** → MRI reconstruction algorithms, RF pulse design
5. **`pip install MRzeroCore`** → Bloch equation validation of depth profiling
6. **Review OCRA/MRI4ALL** → Console architecture for hardware phase

### Phase 3 (P2 — Hardware Design)
7. **Clone HalbachMRIDesigner** → Adapt CAD/FEM pipeline for barrel magnet
8. **Clone pyCoilGen** → Surface coil optimization
9. **Study Ultra-Low-Field-MRI paper list** → B₀ design papers for single-sided geometry

### Phase 4 (Future — ML Reconstruction)
10. **fastmri-reproducible-benchmark** → Benchmark reconstruction approaches
11. **sigmanet** → Learned iterative reconstruction
12. **ProjectC** → Low-field-specific DL reconstruction

---

## Known Important Repos NOT Found via GitHub Search

These repos are known to exist but didn't surface in our queries (may be in different orgs, use different names, or require direct URL access):

| Repo | Expected Location | Purpose |
|------|-------------------|---------|
| **fastMRI** | facebookresearch/fastMRI | Main MRI reconstruction benchmark dataset + models |
| **pypulseq** | imr-framework/pypulseq | Python Pulseq sequence design |
| **BART** | mrirecon/bart | Berkeley Advanced Reconstruction Toolbox (C) |
| **QuTiP** | qutip/qutip | Quantum Toolbox in Python (5000+ stars) |
| **OSI²ONE** | gitlab.com/osii | Open Source Imaging scanner (on GitLab, not GitHub) |
| **gr-MRI** | opensourceimaging.org | GNU Radio MRI spectrometer (SDR-based) |

---

## Summary Statistics

| Category | Repos Found | Most Relevant |
|----------|-------------|---------------|
| Permanent Magnet Simulation | 3 | magpylib (340⭐) |
| NV Center / Diamond Physics | 4 | NVision, NVCentre-ODMR |
| Low-Field MRI Hardware | 5+ | OCRA (48⭐), MRI4ALL (20⭐) |
| MRI Pulse Sequence Simulation | 4 | KomaMRI (189⭐), MRzero (67⭐) |
| MRI Reconstruction | 6 | SigPy (335⭐), fastmri-benchmark (161⭐) |
| Coil Design | 4 | pyCoilGen, cosi-transmit |
| Bloch Equation NMR | 3 | BlochBuster (54⭐), MRzero |
| Curated Resource Lists | 1 | Ultra-Low-Field-MRI (GOLDMINE) |
| **Maser / Quantum Amplifier** | **0** | **WE ARE FIRST** |

**Total unique relevant repos cataloged**: ~25  
**Directly integrable as Python dependencies**: 4 (magpylib, SigPy, MRzero, BlochBuster)  
**Hardware reference architecture**: 3 (OCRA, MRI4ALL, nexus-console)  
**Novel gap we fill**: NV-diamond maser MRI integration (no existing open-source work)
