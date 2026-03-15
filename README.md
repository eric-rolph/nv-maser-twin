# NV Maser "Tricorder" Digital Twin

`Python 3.10+` `PyTorch` `FastAPI` `Tests: 119 passed`

Real-time digital twin of an active magnetic shimming system for a **Nitrogen-Vacancy (NV) center diamond maser** — simulates, trains, serves, and reinforcement-learns a shimming policy that keeps the B₀ field uniform to < 100 ppm.

---

## What it does

1. Simulates a base magnetic field from a permanent Halbach array (50 mT)
2. Introduces realistic spatial disturbances (environmental interference, thermal drift)
3. Runs a PyTorch neural controller (CNN / MLP / LSTM) to compute corrective coil currents
4. Drives simulated micro-coils to cancel distortions in real-time
5. Exposes a FastAPI REST server so external hardware can query the controller over HTTP
6. Provides a gym-compatible RL environment and a REINFORCE baseline for policy learning
7. Exports trained models to ONNX for hardware-accelerated or cross-runtime deployment
8. Tracks all training runs in a local SQLite experiment database
9. Visualizes the entire pipeline as a live oscilloscope-style dashboard

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
│   ├── training.py         Supervised training loop, checkpointing, tracker integration
│   └── loss.py             Field uniformity loss + current penalty
├── api/
│   └── server.py           FastAPI server (/health, /shim, /metrics, /reload, /info, /ui)
├── rl/
│   └── env.py              ShimmingEnv — gym-compatible RL environment
├── export/
│   └── onnx_export.py      ONNX export via torch.onnx.export
├── tracking/
│   └── tracker.py          SQLite experiment tracker (stdlib sqlite3)
├── data/
│   └── dataset.py          Content-addressed .npz dataset cache (SHA-256)
└── main.py                 CLI: train / demo / evaluate / dataset / export / serve

scripts/
├── train_rl.py             REINFORCE policy-gradient baseline
├── run_sweep.py            Hyperparameter grid search (lr × arch)
├── export_onnx.py          ONNX export CLI
├── build_dataset.py        Dataset pre-build CLI
└── show_experiments.py     Inspect SQLite experiment history

benchmarks/
└── benchmark_inference.py  Multi-arch latency benchmark (CNN, MLP, LSTM)

docs/adr/                   Architecture Decision Records (ADR-001 – ADR-004)
config/default.yaml         Default simulation parameters (YAML, deep-merge)
experiments/                SQLite runs.db — auto-created on first train
tests/                      119 passed, 2 skipped (CUDA + onnxruntime)
checkpoints/                Saved model weights + optional model.onnx (git-ignored)
```

---

## Quick start

```bash
# Install core + dev + API dependencies
pip install -e ".[dev,api]"

# Or use make
make install
```

### Training

```bash
# Supervised training — CNN by default
python -m nv_maser train
python -m nv_maser train --arch mlp --epochs 100 --samples 20000
python -m nv_maser train --arch lstm --config my_config.yaml

# Hyperparameter sweep across lr × arch (9 combos, 5 epochs each)
python scripts/run_sweep.py --epochs 5

# After training, view run history
python scripts/show_experiments.py
python scripts/show_experiments.py --run-id 1
```

### Inference server

```bash
pip install -e ".[api]"
python -m nv_maser serve                          # starts on 127.0.0.1:8000
python -m nv_maser serve --host 0.0.0.0 --port 9000

# OR with make
make serve
```

Open **http://localhost:8000/ui** for the live status dashboard.

### ONNX export

```bash
# Export after training
python -m nv_maser export
python -m nv_maser export --output checkpoints/model.onnx --arch cnn --opset 17

# OR
make export
```

### Dataset pre-build

```bash
python -m nv_maser dataset --num-samples 50000 --cache-dir dataset_cache/
```

### RL training

```bash
python scripts/train_rl.py --episodes 500 --arch cnn --steps 50 --seed 42
python scripts/train_rl.py --episodes 200 --arch lstm --config my_config.yaml
```

### Tests and benchmarks

```bash
make test                               # 119 passed, 2 skipped
make test-cov                           # HTML coverage report in htmlcov/
make lint                               # ruff check
make benchmark                          # multi-arch latency table
pytest tests/test_api.py -v             # API endpoint tests (23 tests)
```

---

## Performance

Measured on CPU (PyTorch 2.x, no CUDA), 300 reps per architecture after 20 warm-up passes.

| Architecture | Params | B=1 Median | B=1 P95 | Spec |
|---|---|---|---|---|
| CNN | 32,872 | 0.28 ms | 1.35 ms | ✅ PASS |
| MLP | 2,262,920 | 0.14 ms | 0.27 ms | ✅ PASS |
| LSTM | 881,640 | 0.46 ms | 1.62 ms | ✅ PASS |

**Spec**: B=1 median inference < 1.0 ms on CPU (PyTorch 2.x)

RL environment step latency: ~0.030 ms mean (100-step benchmark, `ShimmingEnv.step()`).

---

## API reference

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "grid_size": 64,
  "arch": "cnn",
  "coils": 8,
  "uptime_s": 42.1
}
```

### `GET /info`

Static model/server info — no side-effects.

```json
{
  "version": "1.0.0",
  "arch": "cnn",
  "grid_size": 64,
  "num_coils": 8,
  "onnx_available": false
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
| `improvement_factor` | `float` | Variance ratio (before/after) |
| `inference_ms` | `float` | Model forward-pass wall time |

Returns `503` if no checkpoint is loaded, `422` if field is not 64×64 or contains NaN/Inf.

### `POST /reload`

Hot-reloads `checkpoints/best.pt` without restarting the server. Returns `404` if no checkpoint exists.

### `GET /metrics`

Prometheus plain-text exposition format:

```
nv_maser_shim_requests_total 100
nv_maser_shim_errors_total 2
nv_maser_shim_latency_ms_avg 0.3100
nv_maser_arch{arch="cnn"} 1
nv_maser_coils 8
```

### `GET /ui`

Browser-based status dashboard — auto-refreshes every 5 seconds using `fetch` against `/health` and `/metrics`. No dependencies, works fully offline.

---

## Experiment tracking

Every `python -m nv_maser train` run is automatically logged to `experiments/runs.db` (created on first use, stdlib `sqlite3`). Use the CLI to inspect:

```bash
python scripts/show_experiments.py                # table of all runs
python scripts/show_experiments.py --run-id 3     # epoch-by-epoch metrics for run 3
```

Programmatic access:

```python
from nv_maser.tracking import ExperimentTracker

tracker = ExperimentTracker()
for run in tracker.list_runs():
    print(run["arch"], run["best_val_loss"])
```

---

## Configuration

All simulation parameters live in `SimConfig` (Pydantic v2). Defaults load from `config/default.yaml`. Override any sub-tree with your own YAML — deep-merge is applied, so only changed keys are needed:

```bash
python -m nv_maser train --config my_config.yaml
python scripts/train_rl.py --config my_config.yaml
```

Example override:

```yaml
model:
  architecture: lstm
training:
  epochs: 100
  learning_rate: 5e-4
  auto_export_onnx: true        # export ONNX automatically after training
coils:
  num_coils: 16
```

---

## Architecture Decision Records

| ADR | Title | Decision |
|---|---|---|
| [ADR-001](docs/adr/ADR-001-controller-architecture.md) | Controller Architecture — CNN vs MLP | CNN is default; MLP/LSTM via `--arch` |
| [ADR-002](docs/adr/ADR-002-api-design.md) | FastAPI REST Inference Server | FastAPI + uvicorn; hardened with CORS, body guard, security headers |
| [ADR-003](docs/adr/ADR-003-rl-environment-design.md) | RL Environment Design | Standalone `ShimmingEnv`, no gymnasium dependency |
| [ADR-004](docs/adr/ADR-004-temporal-controller-lstm.md) | Temporal Controller — LSTM | CNN extractor → 2-layer LSTM → linear head |

---

## Development

```bash
# Install everything
make install-all          # dev + api + onnx extras

# Common tasks
make test                 # pytest -q
make test-cov             # pytest + HTML coverage
make lint                 # ruff check
make format               # ruff --fix
make benchmark            # inference latency table
make sweep                # 9-combo hyperparameter sweep
make export               # ONNX export

# Pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Optional dependencies

| Extra | Packages | Required for |
|---|---|---|
| `dev` | pytest, pytest-cov, httpx, jupyter, pre-commit | Testing and notebooks |
| `api` | fastapi, uvicorn[standard] | REST inference server |
| `onnx` | onnxscript, onnxruntime | ONNX export + runtime verification |

### Structured logging

Set `LOG_FORMAT=json` to emit newline-delimited JSON log records (useful with log aggregators):

```bash
LOG_FORMAT=json python -m nv_maser serve
```

---

## License

MIT
