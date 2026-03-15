# ADR-002: FastAPI REST Inference Server Design

**Status:** Accepted  
**Date:** 2026-03-14  
**Deciders:** AI Engineer, Software Architect

---

## Context

Laboratory instrumentation systems require a lightweight, language-agnostic interface to the trained shimming controller so that:
- External measurement hardware (e.g., NMR console, field camera systems) can submit field maps and receive correction currents via HTTP.
- The controller can be deployed as a microservice alongside data acquisition pipelines.
- Multiple simultaneous clients can query the model without blocking each other.

---

## Decision

**Deploy a FastAPI REST server** exposing two endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe: returns model-loaded status and grid size |
| `/shim` | POST | Accept a 64×64 field array, return correction currents + quality metrics |

The server is launched via `python -m nv_maser serve [--host HOST] [--port PORT]` using `uvicorn` as the ASGI runtime.

---

## Consequences

### Positive
- **Zero-dependency clients**: Any HTTP client (Python `requests`, curl, LabVIEW web services, MATLAB `webwrite`) can interact with the API.
- **Automatic OpenAPI docs**: FastAPI generates `/docs` (Swagger UI) and `/redoc` automatically from Pydantic models — zero boilerplate for API documentation.
- **Async-ready**: `uvicorn` runs the FastAPI app in an async event loop; long-running background tasks (e.g., periodic model reloading) can be added as `asyncio` coroutines.
- **Pydantic validation**: Input field shape is validated before tensor construction, returning clear 422 errors instead of cryptic PyTorch shape exceptions.

### Negative / Trade-offs
- **JSON serialisation overhead**: A 64×64 float32 field serialised as nested JSON lists is ~40 KB. For very high sample rates (> 1 kHz) this becomes the bottleneck; a binary Protocol Buffers / msgpack transport would be preferable.
- **Stateless per-request inference**: Each `/shim` call is independent; there is no built-in session concept for temporally correlated disturbances. Feedback controllers requiring state must maintain it client-side.

---

## Stateless vs Stateful Design

A **stateless** design was chosen deliberately:

| Criterion | Stateless (chosen) | Stateful |
|-----------|-------------------|---------|
| Horizontal scaling | Trivial — add replicas behind a load balancer | Requires sticky sessions or distributed state store |
| Crash recovery | Restart resumes immediately; no session hydration | Must serialise and restore per-client state |
| Client complexity | Client manages temporal context if needed | Server manages feedback history |
| Fit for use case | ✅ Sufficient — coil commands are computed per-frame | ❌ Over-engineered for single-node lab deployment |

---

## Inference Latency Rationale

**CPU inference is acceptable for real-time use at laboratory field-update rates:**

- Typical NMR shim update cycles: 1–100 Hz (10–1000 ms per frame)
- Measured CPU inference median: < 2 ms on modern x86 (see benchmark results)
- GPU inference median: < 0.1 ms (on CUDA-capable hardware)

The < 1 ms GPU target stated in the design spec is met with any discrete GPU. CPU-only deployments (e.g., Raspberry Pi 5 field-deployed controller) accept the relaxed constraint of < 10 ms.

The server loads a checkpoint from `checkpoints/best.pt` at startup; if absent, the randomly-initialised model is used (suitable for integration testing without a trained checkpoint).

---

## JSON Field Serialisation Trade-offs

| Format | Size (64×64 f32) | Human-readable | Cross-language | Latency |
|--------|-----------------|----------------|----------------|---------|
| JSON nested lists | ~40 KB | ✅ | ✅ | ~0.5 ms parse |
| Base64-encoded bytes | ~22 KB | ❌ | ✅ | ~0.2 ms |
| Protocol Buffers | ~16 KB | ❌ | ✅ | ~0.05 ms |
| Raw TCP binary | ~16 KB | ❌ | Limited | ~0.01 ms |

JSON was chosen for the v1.0.0 API for maximum interoperability. A future `Content-Type: application/octet-stream` endpoint can be added for high-throughput use cases without breaking the existing JSON interface.

---

## References
- `src/nv_maser/api/server.py` — FastAPI application
- `src/nv_maser/main.py` — `serve` subcommand (uvicorn launcher)
- `pyproject.toml` — `[project.optional-dependencies] api`
- `benchmarks/benchmark_inference.py` — latency measurements
- FastAPI docs: https://fastapi.tiangolo.com
- Uvicorn docs: https://www.uvicorn.org
