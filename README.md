# NV Maser "Tricorder" Digital Twin

A Python-based real-time simulation (**Digital Twin**) of an active magnetic shimming system for a **Nitrogen-Vacancy (NV) center diamond maser**.

## What it does

1. Simulates a base magnetic field from a permanent Halbach array (50 mT)
2. Introduces realistic spatial disturbances (environmental interference)
3. Uses a PyTorch neural network (CNN or MLP) to compute corrective coil currents
4. Drives simulated micro-coils to cancel distortions in real-time
5. Visualizes the entire pipeline as a live oscilloscope-style dashboard

## Architecture (mental model)

| Audio analogy | Magnetic equivalent | Code role |
|---|---|---|
| Pure sine wave | Base field B₀ (Halbach array) | `FieldEnvironment.base_field` |
| Noise / modulation | External interference | `DisturbanceGenerator` |
| Auto-Tune plugin | AI shimming controller | `ShimController` (PyTorch) |
| Inverted control signal | Corrective coil currents | Model output → coil superposition |
| Flat output waveform | Uniform net field | Loss → 0 |

## Quick start

```bash
# 1. Install (Python 3.10+, CUDA optional)
pip install -e ".[dev]"

# 2. Train the shimming controller (10K samples, 50 epochs)
python -m nv_maser train

# 3. Launch the real-time dashboard
python -m nv_maser demo

# 4. Evaluate on 500 test disturbances
python -m nv_maser evaluate

# 5. Visualise coil influence maps
python -m nv_maser visualize-coils
```

## CLI options

```
python -m nv_maser train --arch mlp --epochs 100 --samples 20000
python -m nv_maser train --device cpu
python -m nv_maser demo  --device cuda
```

## Project structure

```
nv-maser-twin/
├── src/nv_maser/
│   ├── config.py          # Pydantic settings (all tunable params)
│   ├── physics/
│   │   ├── grid.py        # 2D spatial grid (64×64, 10 mm)
│   │   ├── base_field.py  # Halbach B₀ field
│   │   ├── disturbance.py # Spatial harmonic interference generator
│   │   ├── coils.py       # Biot-Savart shim coil model
│   │   └── environment.py # Full simulation compositor
│   ├── model/
│   │   ├── controller.py  # CNN & MLP PyTorch controllers
│   │   ├── loss.py        # Field uniformity loss
│   │   └── training.py    # Training loop + checkpointing
│   ├── viz/
│   │   ├── dashboard.py   # PyQtGraph real-time dashboard
│   │   └── plots.py       # Matplotlib static plots
│   └── main.py            # CLI entry point
├── tests/                 # pytest test suite (Phase 1 + 2)
├── config/default.yaml    # Default simulation parameters
└── checkpoints/           # Saved model weights (git-ignored)
```

## Performance targets

| Metric | Target |
|---|---|
| Training (10K samples, 50 epochs, RTX 5090) | < 2 minutes |
| Inference (single sample, GPU) | < 1 ms |
| Field uniformity improvement | ≥ 10× reduction in active-zone std |
| PPM homogeneity after correction | < 100 ppm |
| Dashboard frame rate | ≥ 20 FPS |

## Running tests

```bash
pytest                  # All tests
pytest tests/test_grid.py -v   # Single module
pytest --cov=nv_maser   # With coverage
```
