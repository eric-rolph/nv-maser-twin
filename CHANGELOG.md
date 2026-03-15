# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.7.0] - 2025-07-XX

### Added

- Multi-stage `Dockerfile` (`python:3.11-slim` builder + runtime layers) and `docker-compose.yml` with healthcheck and `checkpoints/` volume mount
- `/metrics` endpoint in Prometheus exposition format; thread-safe `_Metrics` class tracking `shim_requests_total`, error count, and rolling average latency
- Content-addressed dataset caching in `src/nv_maser/data/dataset.py` — SHA-256 hash of `grid + disturbance + field + coils` config, stored as `.npz`; `force_rebuild` flag; progress logging every 10 %
- `scripts/build_dataset.py` CLI entry point for offline dataset generation
- CI matrix expanded to Python 3.10 / 3.11 / 3.12 with `actions/cache` for pip; separate `lint` job (ruff) gating the `test` matrix
- `config/default.yaml` documenting all `SimConfig` defaults with inline comments
- 3 new metrics tests in `tests/test_api.py` (16 tests total)
- `tests/test_dataset.py` — 4 tests: shape validation, cache hit, hash invalidation, `force_rebuild`
- `.dockerignore` to exclude venv, checkpoints cache, and notebook output from build context

## [0.6.0] - 2025-07-XX

### Added

- Full README rewrite with architecture diagram (module tree), performance table, and FastAPI endpoint reference

### Fixed

- FastAPI security hardening: 1 MB request-body guard, dimension bounds check (`[2, 512]` each), NaN/Inf sanitisation before inference, CORS restricted to `localhost` origins only, `X-Content-Type-Options` / `X-Frame-Options` security-headers middleware
- Dead code `max_current * 0` expression removed from `scripts/train_rl.py`

### Security

- CORS `allow_origins` restricted to `["http://localhost", "http://127.0.0.1"]`; wildcard origin removed

## [0.5.0] - 2025-07-XX

### Added

- `benchmarks/benchmark_inference.py` — multi-architecture latency comparison for CNN, MLP, and LSTM controllers (all < 1 ms median on CPU)
- `scripts/train_rl.py` — REINFORCE policy-gradient baseline with `StochasticShimmingPolicy`, gradient clipping (`max_norm=1.0`), and episode logging
- `tests/test_rl_train.py` — 2 tests: single episode runs without error, policy gradient step updates parameters
- ADR-003: RL environment design (episode termination, reward shaping, action scale)
- ADR-004: Temporal controller design — rationale for LSTM over GRU/Transformer for real-time latency budget

## [0.4.0] - 2025-07-XX

### Added

- `LSTMController` architecture: CNN spatial extractor feeding a two-layer LSTM, hidden-state carry across inference calls
- `ShimmingEnv` — gymnasium-compatible RL environment without a hard `gymnasium` dependency; `reset()` / `step()` / `render()` interface
- `DisturbanceGenerator.randomize()` helper for stochastic episode initialisation
- `tests/test_api.py` — 10 endpoint tests using `httpx.AsyncClient` (health, shim valid input, shim error paths)
- LSTM checkpoint key extraction (`model_state`) aligned with supervised training output format

### Fixed

- `server.py` duplicate `/shim` route definition removed (caused silent shadowing)
- Checkpoint loading now extracts `model_state` sub-key from training dict; previously failed silently on new-format checkpoints

## [0.3.0] - 2025-07-XX

### Added

- `src/nv_maser/api/server.py` — FastAPI inference server with `/health` and `/shim` endpoints
- `uvicorn` entry point: `python -m nv_maser serve`
- Inference latency benchmark: 0.27 ms median end-to-end (CNN, batch-1, CPU)
- ADR-001: Controller architecture overview (CNN chosen as default; MLP as ablation; LSTM for temporal)
- ADR-002: Single-server deployment rationale (no reverse proxy for v0; revisit at production scale)

## [0.2.0] - 2025-07-XX

### Added

- GitHub Actions CI workflow (`.github/workflows/ci.yml`) running `pytest` + `ruff` on push and pull request to `main`
- `notebooks/exploration.ipynb` — interactive walkthrough of physics simulation, controller training, and field visualisation
- YAML deep-merge config overrides: `python -m nv_maser train --config my_overrides.yaml` merges on top of `config/default.yaml`
- `src/nv_maser/__main__.py` enabling `python -m nv_maser` entry point

## [0.1.0] - 2025-07-XX

### Added

- Core physics simulation: `NVCentreHamiltonian`, `FieldEnvironment`, `DisturbanceGenerator`, `ShimCoilArray` (Biot-Savart)
- `CNNController` and `MLPController` neural shimming controllers
- `Trainer` with AdamW optimiser, cosine/step LR scheduler, early stopping on validation loss plateau
- `FieldUniformityLoss` + current-penalty regulariser
- `Pydantic`-based `SimConfig` for all tunable simulation parameters
- PyQtGraph real-time field dashboard (`python -m nv_maser demo`)
- 51 passing pytest tests across physics, model, and training modules
