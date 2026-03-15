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


# ---------------------------------------------------------------------------
# Security hardening tests
# ---------------------------------------------------------------------------

def test_oversized_field_dimension_422(client):
    """POST /shim with a 65×64 field (wrong outer dimension) → 422."""
    r = client.post("/shim", json=_make_field(rows=65, cols=64))
    assert r.status_code == 422


def test_nan_field_422(client):
    """POST /shim with a 64×64 field containing NaN → 422.

    Uses allow_nan=True + raw content because Python's json.dumps raises
    ValueError for NaN by default, but json.loads (server-side) accepts the
    bare NaN token, so our server-side isfinite check must catch it.
    """
    import json as _json

    field = np.full((64, 64), 0.0).tolist()
    field[0][0] = float("nan")
    raw = _json.dumps({"distorted_field": field}, allow_nan=True).encode()
    r = client.post(
        "/shim", content=raw, headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 422


def test_security_headers_present(client):
    """GET /health response includes X-Content-Type-Options: nosniff."""
    r = client.get("/health")
    assert r.headers.get("x-content-type-options") == "nosniff"


# ---------------------------------------------------------------------------
# /metrics tests
# ---------------------------------------------------------------------------

def test_metrics_endpoint_ok(client):
    """GET /metrics returns HTTP 200."""
    r = client.get("/metrics")
    assert r.status_code == 200


def test_metrics_format(client):
    """GET /metrics response contains Prometheus-style nv_maser_shim_requests_total line."""
    r = client.get("/metrics")
    assert "nv_maser_shim_requests_total" in r.text


def test_metrics_counts_requests(client):
    """After a valid /shim call, shim_requests_total in /metrics is >= 1.

    The client fixture is module-scoped so counts accumulate — use >= not ==.
    """
    client.post("/shim", json=_make_field())
    r = client.get("/metrics")
    assert r.status_code == 200
    for line in r.text.splitlines():
        if line.startswith("nv_maser_shim_requests_total "):
            value = int(line.split()[-1])
            assert value >= 1
            break
    else:
        pytest.fail("nv_maser_shim_requests_total not found in /metrics output")


# ---------------------------------------------------------------------------
# /reload tests
# ---------------------------------------------------------------------------

def test_reload_no_checkpoint(client):
    """POST /reload returns 404 when no checkpoint file exists in test env."""
    from unittest.mock import patch
    with patch("pathlib.Path.exists", return_value=False):
        r = client.post("/reload")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /ui tests
# ---------------------------------------------------------------------------

def test_ui_returns_html(client):
    """GET /ui returns HTTP 200 with text/html content-type and NV Maser title."""
    r = client.get("/ui")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "NV Maser" in r.text


def test_ui_has_refresh_script(client):
    """GET /ui response contains setInterval (confirms JS auto-refresh is present)."""
    r = client.get("/ui")
    assert r.status_code == 200
    assert "setInterval" in r.text


# ---------------------------------------------------------------------------
# /info tests
# ---------------------------------------------------------------------------

def test_info_endpoint(client):
    """GET /info returns HTTP 200 with expected JSON keys."""
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    for key in ("version", "arch", "grid_size", "num_coils", "onnx_available"):
        assert key in body, f"Missing key: {key}"


def test_info_arch_matches_health(client):
    """GET /info arch field matches GET /health arch field."""
    info = client.get("/info").json()
    health = client.get("/health").json()
    assert info["arch"] == health["arch"]


# ---------------------------------------------------------------------------
# /health enriched field tests
# ---------------------------------------------------------------------------

def test_health_has_arch_field(client):
    """GET /health response includes 'arch' key."""
    r = client.get("/health")
    assert r.status_code == 200
    assert "arch" in r.json()


# ---------------------------------------------------------------------------
# /metrics enriched label tests
# ---------------------------------------------------------------------------

def test_metrics_has_arch_label(client):
    """GET /metrics response includes nv_maser_arch label line."""
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "nv_maser_arch" in r.text
