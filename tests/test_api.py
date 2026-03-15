"""Comprehensive API tests for the NV Maser shim FastAPI server.

Sprint 4 — covers health endpoint, shim endpoint (functional, shape,
range, schema), wrong-shape 422 error, and random-field correctness.
"""
import numpy as np
import pytest
from starlette.testclient import TestClient

from nv_maser.api.server import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_field(rows: int = 64, cols: int = 64, value: float = 0.0) -> dict:
    """Return a JSON-serialisable payload with a rows×cols float field."""
    return {"distorted_field": np.full((rows, cols), value).tolist()}


# ---------------------------------------------------------------------------
# Shared client fixture (module scope → single lifespan startup)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health tests
# ---------------------------------------------------------------------------

def test_health_status_ok(client):
    """GET /health returns HTTP 200 and status == 'ok'."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_model_loaded(client):
    """GET /health reports model_loaded == True after lifespan startup."""
    r = client.get("/health")
    assert r.json()["model_loaded"] is True


def test_health_grid_size(client):
    """GET /health reports grid_size == 64 (SimConfig default)."""
    r = client.get("/health")
    assert r.json()["grid_size"] == 64


# ---------------------------------------------------------------------------
# /shim tests — happy path
# ---------------------------------------------------------------------------

def test_shim_valid_request(client):
    """POST /shim with a 64×64 zero field returns HTTP 200."""
    r = client.post("/shim", json=_make_field())
    assert r.status_code == 200
    body = r.json()
    assert "currents" in body
    assert "inference_ms" in body


def test_shim_currents_shape(client):
    """POST /shim returns exactly 8 current values (num_coils default)."""
    r = client.post("/shim", json=_make_field())
    assert r.status_code == 200
    assert len(r.json()["currents"]) == 8


def test_shim_currents_range(client):
    """Each current must be within [-1.1, 1.1] (tanh output ≤ 1.0)."""
    r = client.post("/shim", json=_make_field())
    assert r.status_code == 200
    for amp in r.json()["currents"]:
        assert -1.1 <= amp <= 1.1, f"Current {amp} out of expected range"


def test_shim_inference_ms_positive(client):
    """inference_ms must be a positive number."""
    r = client.post("/shim", json=_make_field())
    assert r.status_code == 200
    assert r.json()["inference_ms"] > 0


def test_shim_response_schema(client):
    """All expected keys are present in the ShimResponse."""
    expected_keys = {
        "currents",
        "corrected_field_variance",
        "distorted_field_variance",
        "improvement_factor",
        "inference_ms",
    }
    r = client.post("/shim", json=_make_field())
    assert r.status_code == 200
    assert expected_keys <= set(r.json().keys())


def test_shim_random_field(client):
    """POST /shim with random noise field returns 200 and improvement_factor >= 0."""
    rng = np.random.default_rng(seed=42)
    noisy = rng.normal(loc=0.0, scale=0.1, size=(64, 64)).astype(np.float32)
    payload = {"distorted_field": noisy.tolist()}
    r = client.post("/shim", json=payload)
    assert r.status_code == 200
    assert r.json()["improvement_factor"] >= 0.0


# ---------------------------------------------------------------------------
# /shim tests — error handling
# ---------------------------------------------------------------------------

def test_shim_wrong_shape_422(client):
    """POST /shim with a 32×32 field returns HTTP 422 (wrong dimensions)."""
    r = client.post("/shim", json=_make_field(rows=32, cols=32))
    assert r.status_code == 422
