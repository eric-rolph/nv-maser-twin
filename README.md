# NV Maser "Tricorder" Digital Twin

`Python 3.10+` `PyTorch` `FastAPI` `Tests: 73 passed`

Real-time digital twin of an active magnetic shimming system for a **Nitrogen-Vacancy (NV) center diamond maser** — simulates, trains, serves, and reinforcement-learns a shimming policy that keeps the B₀ field uniform to < 100 ppm.

---

## What it does

1. Simulates a base magnetic field from a permanent Halbach array (50 mT)
2. Introduces realistic spatial disturbances (environmental interference, thermal drift)
3. Runs a PyTorch neural controller (CNN / MLP / LSTM) to compute corrective coil currents
4. Drives simulated micro-coils to cancel distortions in real-time
5. Exposes a FastAPI REST server so external hardware can query the controller over HTTP
6. Provides a gym-compatible RL environment and a REINFORCE baseline for policy learning
7. Visualizes the entire pipeline as a live oscilloscope-style dashboard

### Mental model (audio analogy)

| Audio analogy | Magnetic equivalent | Code role |
|---|---|---|
| Pure sine wave | Base field B₀ (Halbach array) | `FieldEnvironment.base_field` |
| Noise / modulation | External interference | `DisturbanceGenerator` |
| Auto-Tune plugin | Supervised shimming controller | `ShimController` (CNN/MLP/LSTM) |
| Inverted control signal | Corrective coil currents | Model output → coil superposition |
| Flat output waveform | Uniform net field | `FieldUniformityLoss` → 0 |
| Game environment | RL shimming environment | `ShimmingEnv` (gym-compatible) |
| Self-learning mixer | REINFORCE policy gradient | `StochasticShimmingPolicy` |

---

## Architecture

```
src/nv_maser/
├── config.py               Pydantic SimConfig — all tunable parameters
├── physics/
│   ├── grid.py             2D spatial grid (64×64, 10 mm span)
│   ├── base_field.py       Halbach B₀ generator (50 mT)
│   ├── disturbance.py      Spatial harmonic interference + randomize()
│   ├── coils.py            Biot-Savart shim coil model
│   └── environment.py      Unified physics environment compositor
├── model/
│   ├── controller.py       CNN / MLP / LSTM controllers + build_controller()
│   ├── training.py         Supervised training loop + checkpointing
│   └── loss.py             Field uniformity loss + current penalty
├── api/
│   └── server.py           FastAPI inference server (/health, /shim)
├── rl/
│   └── env.py              ShimmingEnv — gym-compatible RL environment
└── main.py                 CLI entry point (train / demo / evaluate / serve)

scripts/
└── train_rl.py             REINFORCE policy-gradient baseline

benchmarks/
└── benchmark_inference.py  Multi-arch latency benchmark (CNN, MLP, LSTM)

docs/adr/                   Architecture Decision Records (ADR-001 – ADR-004)
config/default.yaml         Default simulation parameters (YAML, deep-merge)
tests/                      pytest test suite — 73 passed, 1 skipped (CUDA)
checkpoints/                Saved model weights (git-ignored)
```

---

## Quick start

```bash
# Install core + dev dependencies
pip install -e ".[dev]"

# Supervised training — CNN by default
python -m nv_maser train
python -m nv_maser train --arch mlp --epochs 100 --samples 20000
python -m nv_maser train --arch lstm --epochs 100

# Real-time field dashboard
python -m nv_maser demo
python -m nv_maser demo --device cuda

# Evaluate on 500 test disturbances
python -m nv_maser evaluate

# Visualize coil influence maps
python -m nv_maser visualize-coils
```

### REST inference server

```bash
pip install -e ".[api]"
uvicorn nv_maser.api.server:app --reload

# Liveness probe
# GET /health → {"status": "ok", "model_loaded": true, "grid_size": 64}

# Inference
# POST /shim  body: {"distorted_field": [[...64 rows of 64 floats...]]}
#             → {"currents": [...8 floats...],
#                "corrected_field_variance": 1.4e-8,
#                "distorted_field_variance": 3.2e-6,
#                "improvement_factor": 228.5,
#                "inference_ms": 0.31}
```

### RL training (REINFORCE baseline)

```bash
# Default: CNN backbone, 500 episodes, 50 steps each
python scripts/train_rl.py

# With options
python scripts/train_rl.py --episodes 500 --arch cnn --steps 50 --seed 42
python scripts/train_rl.py --episodes 200 --arch lstm --steps 100
python scripts/train_rl.py --episodes 300 --arch mlp --lr 3e-4
```

### Benchmarks and tests

```bash
# Multi-arch latency benchmark
python benchmarks/benchmark_inference.py

# Full test suite
pytest                               # 73 passed, 1 skipped (CUDA)
pytest tests/test_api.py             # API endpoint tests (requires httpx)
pytest tests/test_rl_train.py        # RL policy shape + smoke tests
pytest --cov=nv_maser                # With coverage report
```

---

## Performance

Measured on CPU (PyTorch 2.x, no CUDA), 300 reps per architecture after 20 warm-up passes.

| Architecture | Params | B=1 Median | B=1 P95 | Spec |
|---|---|---|---|---|
| CNN | 32,872 | 0.28 ms | 1.35 ms | ✅ PASS |
| MLP | 2,262,920 | 0.14 ms | 0.27 ms | ✅ PASS |
| LSTM | 881,640 | 0.46 ms | 1.62 ms | ✅ PASS |

**Spec**: B=1 median inference < 1.0 ms (CPU, PyTorch 2.x)

RL environment step latency: ~0.030 ms mean (100-step benchmark, `ShimmingEnv.step()`).

---

## API reference

### `GET /health`

Returns server liveness and model status.

```json
{
  "status": "ok",
  "model_loaded": true,
  "grid_size": 64
}
```

### `POST /shim`

**Request body**

```json
{
  "distorted_field": [[...64 rows, each 64 floats (Tesla)...]]
}
```

**Response** (`ShimResponse`)

| Field | Type | Description |
|---|---|---|
| `currents` | `float[8]` | Corrective coil currents in Amps |
| `corrected_field_variance` | `float` | Residual field variance after correction |
| `distorted_field_variance` | `float` | Input field variance before correction |
| `improvement_factor` | `float` | Variance ratio (before / after) |
| `inference_ms` | `float` | Model forward-pass wall time |

Returns `503` if no checkpoint is loaded; `422` if field shape is not 64×64.

---

## Configuration

All simulation parameters are defined in `SimConfig` (Pydantic). The defaults load from `config/default.yaml`. Override any sub-tree with your own YAML file — deep-merge is applied:

```bash
python -m nv_maser train --config my_config.yaml
```

Example override file (only changed keys are needed):

```yaml
model:
  architecture: lstm
training:
  epochs: 100
  learning_rate: 5e-4
coils:
  num_coils: 16
```

---

## Architecture Decision Records

| ADR | Title | Decision |
|---|---|---|
| [ADR-001](docs/adr/ADR-001-controller-architecture.md) | Controller Architecture — CNN vs MLP | CNN is the default; MLP available as `--arch mlp` |
| [ADR-002](docs/adr/ADR-002-api-design.md) | FastAPI REST Inference Server Design | FastAPI + uvicorn; `/health` and `/shim` endpoints |
| [ADR-003](docs/adr/ADR-003-rl-environment-design.md) | RL Environment Design | Standalone gym-compatible `ShimmingEnv` (no gymnasium dep) |
| [ADR-004](docs/adr/ADR-004-temporal-controller-lstm.md) | Temporal Controller Architecture — LSTM | CNN feature extractor → LSTM → linear head for temporal disturbances |

---

## Development

```bash
# Install all extras
pip install -e ".[dev,api]"

# Tests
pytest                                       # 73 passed, 1 skipped (CUDA)
pytest tests/test_api.py -v                  # API endpoint tests
pytest tests/test_rl_train.py -v             # RL smoke tests
pytest --cov=nv_maser --cov-report=term      # Coverage

# Benchmarks
python benchmarks/benchmark_inference.py     # Multi-arch latency table

# Linting / formatting (if configured)
ruff check src tests
```

### Optional dependencies

| Extra | Packages | Required for |
|---|---|---|
| `dev` | pytest, pytest-cov, httpx, jupyter | Testing and notebooks |
| `api` | fastapi, uvicorn[standard] | REST inference server |

---

## License

MIT
