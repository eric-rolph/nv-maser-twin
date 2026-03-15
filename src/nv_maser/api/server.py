"""FastAPI inference server for real-time field shimming."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse
import logging
import threading
import numpy as np
import torch
import time
from ..config import SimConfig
from ..physics.environment import FieldEnvironment
from ..model.controller import build_controller
import json as _json
import os as _os


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return _json.dumps(payload)


def _configure_logging() -> None:
    """Configure root logging. JSON format when LOG_FORMAT=json env var is set."""
    fmt = _os.environ.get("LOG_FORMAT", "text").lower()
    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
        )
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(logging.INFO)


_configure_logging()

logger = logging.getLogger("nv_maser.api")


class _Metrics:
    """Thread-safe request counters and latency accumulator."""

    def __init__(self):
        self._lock = threading.Lock()
        self.shim_requests_total: int = 0
        self.shim_errors_total: int = 0
        self.shim_latency_ms_sum: float = 0.0
        self.shim_latency_ms_count: int = 0

    def record_shim(self, latency_ms: float, error: bool = False) -> None:
        with self._lock:
            self.shim_requests_total += 1
            if error:
                self.shim_errors_total += 1
            else:
                self.shim_latency_ms_sum += latency_ms
                self.shim_latency_ms_count += 1

    def avg_latency_ms(self) -> float:
        with self._lock:
            if self.shim_latency_ms_count == 0:
                return 0.0
            return self.shim_latency_ms_sum / self.shim_latency_ms_count


_metrics = _Metrics()
_start_time = time.time()


# Global state
_env: FieldEnvironment | None = None
_model = None
_config: SimConfig | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env, _model, _config
    _config = SimConfig()
    _env = FieldEnvironment(_config)
    _model = build_controller(_config.grid.size, _config.model, _config.coils)
    # Try to load checkpoint
    import pathlib

    ckpt = pathlib.Path("checkpoints/best.pt")
    if ckpt.exists():
        saved = torch.load(ckpt, map_location="cpu")
        # Support both bare state-dicts and training checkpoints
        # (training checkpoints have keys: epoch, model_state, optimizer_state, val_loss)
        state_dict = saved.get("model_state", saved) if isinstance(saved, dict) else saved
        _model.load_state_dict(state_dict)
    _model.eval()
    logger.info(
        "NV Maser API ready | grid=%dx%d | coils=%d | arch=%s | checkpoint=%s",
        _config.grid.size,
        _config.grid.size,
        _config.coils.num_coils,
        _config.model.architecture.value,
        "loaded" if ckpt.exists() else "random-init",
    )
    yield
    # Teardown (nothing needed)


app = FastAPI(title="NV Maser Shim API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

MAX_BODY_BYTES = 1 * 1024 * 1024  # 1 MB


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Request body too large"})
    return await call_next(request)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    return response


class FieldRequest(BaseModel):
    distorted_field: list[list[float]] = Field(
        ..., description="64x64 distorted field values"
    )


class ShimResponse(BaseModel):
    currents: list[float]
    corrected_field_variance: float
    distorted_field_variance: float
    improvement_factor: float
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    grid_size: int
    arch: str
    coils: int
    uptime_s: float


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        grid_size=_config.grid.size if _config else 0,
        arch=_config.model.architecture.value if _config else "unknown",
        coils=_config.coils.num_coils if _config else 0,
        uptime_s=round(time.time() - _start_time, 2),
    )


@app.post("/shim", response_model=ShimResponse)
async def shim(req: FieldRequest):
    if _model is None:
        _metrics.record_shim(0.0, error=True)
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(req.distorted_field) != _config.grid.size:
        _metrics.record_shim(0.0, error=True)
        raise HTTPException(
            status_code=422,
            detail=f"Field must be {_config.grid.size}\u00d7{_config.grid.size}",
        )
    if any(len(row) != _config.grid.size for row in req.distorted_field):
        _metrics.record_shim(0.0, error=True)
        raise HTTPException(
            status_code=422,
            detail=f"All rows must have {_config.grid.size} columns",
        )

    field = np.array(req.distorted_field, dtype=np.float32)
    if not np.isfinite(field).all():
        _metrics.record_shim(0.0, error=True)
        raise HTTPException(status_code=422, detail="Field contains NaN or Inf values")
    if field.shape != (_config.grid.size, _config.grid.size):
        _metrics.record_shim(0.0, error=True)
        raise HTTPException(
            status_code=422,
            detail=f"Field must be {_config.grid.size}x{_config.grid.size}",
        )

    x = torch.from_numpy(field).unsqueeze(0).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        currents = _model(x).squeeze(0).numpy()
    inference_ms = (time.perf_counter() - t0) * 1000
    _metrics.record_shim(inference_ms)

    # Compute corrected field
    coil_field = _env.coils.compute_field(currents)
    corrected = field + coil_field
    mask = _env.grid.active_zone_mask

    return ShimResponse(
        currents=currents.tolist(),
        corrected_field_variance=float(np.var(corrected[mask])),
        distorted_field_variance=float(np.var(field[mask])),
        improvement_factor=float(
            np.var(field[mask]) / max(np.var(corrected[mask]), 1e-12)
        ),
        inference_ms=inference_ms,
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Prometheus-compatible plain-text metrics."""
    avg_lat = _metrics.avg_latency_ms()
    lines = [
        "# HELP nv_maser_shim_requests_total Total /shim requests",
        "# TYPE nv_maser_shim_requests_total counter",
        f"nv_maser_shim_requests_total {_metrics.shim_requests_total}",
        "# HELP nv_maser_shim_errors_total Total /shim errors (503+422)",
        "# TYPE nv_maser_shim_errors_total counter",
        f"nv_maser_shim_errors_total {_metrics.shim_errors_total}",
        "# HELP nv_maser_shim_latency_ms_avg Average /shim inference latency (ms)",
        "# TYPE nv_maser_shim_latency_ms_avg gauge",
        f"nv_maser_shim_latency_ms_avg {avg_lat:.4f}",
        "# HELP nv_maser_arch Architecture in use",
        "# TYPE nv_maser_arch gauge",
        f'nv_maser_arch{{arch="{_config.model.architecture.value if _config else "unknown"}"}}1',
        "# HELP nv_maser_coils Number of shimming coils",
        "# TYPE nv_maser_coils gauge",
        f"nv_maser_coils {_config.coils.num_coils if _config else 0}",
    ]
    return "\n".join(lines) + "\n"


class ReloadResponse(BaseModel):
    status: str
    checkpoint: str
    arch: str


_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>NV Maser Status</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: monospace; background: #0d1117; color: #e6edf3; margin: 0; padding: 24px; }
    h1 { color: #58a6ff; margin-bottom: 4px; }
    .subtitle { color: #8b949e; font-size: 12px; margin-bottom: 24px; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 12px 0; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }
    .ok { background: #238636; color: #fff; }
    .err { background: #da3633; color: #fff; }
    .label { color: #8b949e; font-size: 12px; }
    .val { font-size: 22px; font-weight: bold; color: #e6edf3; }
    .row { display: flex; gap: 24px; flex-wrap: wrap; }
    .metric { min-width: 140px; }
    .bar-bg { background: #21262d; border-radius: 4px; height: 8px; margin-top: 8px; }
    .bar-fill { background: #238636; height: 8px; border-radius: 4px; transition: width 0.3s; }
    .ts { color: #8b949e; font-size: 11px; text-align: right; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>NV Maser Digital Twin</h1>
  <div class="subtitle">Real-time inference server status</div>

  <div class="card">
    <div class="row" style="align-items:center;justify-content:space-between">
      <div>Status: <span id="status-badge" class="badge ok">OK</span></div>
      <div class="ts" id="last-update">Connecting...</div>
    </div>
  </div>

  <div class="card">
    <div class="label" style="margin-bottom:10px">Model Info</div>
    <div class="row">
      <div class="metric"><div class="label">Architecture</div><div class="val" id="arch">-</div></div>
      <div class="metric"><div class="label">Grid Size</div><div class="val" id="grid">-</div></div>
      <div class="metric"><div class="label">Coils</div><div class="val" id="coils">-</div></div>
      <div class="metric"><div class="label">Uptime</div><div class="val" id="uptime">-</div></div>
    </div>
  </div>

  <div class="card">
    <div class="label" style="margin-bottom:10px">Shim Requests</div>
    <div class="row">
      <div class="metric"><div class="label">Total</div><div class="val" id="req-total">-</div></div>
      <div class="metric"><div class="label">Errors</div><div class="val" id="req-errors">-</div></div>
      <div class="metric"><div class="label">Avg Latency</div><div class="val" id="req-latency">-</div></div>
    </div>
    <div class="label" style="margin-top:12px">Success Rate</div>
    <div class="bar-bg"><div class="bar-fill" id="success-bar" style="width:100%"></div></div>
  </div>

  <script>
    function parseProm(text) {
      const out = {};
      text.split('\\n').forEach(line => {
        if (line.startsWith('#') || !line.trim()) return;
        const m = line.match(/^(\\S+?)(?:\\{[^}]*\\})?\\s+([\\d.]+)/);
        if (m) out[m[1]] = parseFloat(m[2]);
        const lm = line.match(/^\\S+\\{arch="([^"]+)"\\}/);
        if (lm) out['_arch'] = lm[1];
      });
      return out;
    }
    async function refresh() {
      try {
        const [h, m] = await Promise.all([
          fetch('/health').then(r => r.json()),
          fetch('/metrics').then(r => r.text())
        ]);
        const p = parseProm(m);
        document.getElementById('status-badge').textContent = h.status.toUpperCase();
        document.getElementById('status-badge').className = 'badge ' + (h.status === 'ok' ? 'ok' : 'err');
        document.getElementById('arch').textContent = h.arch || p['_arch'] || '-';
        document.getElementById('grid').textContent = h.grid_size ? h.grid_size + 'x' + h.grid_size : '-';
        document.getElementById('coils').textContent = h.coils ?? '-';
        document.getElementById('uptime').textContent = h.uptime_s != null ? h.uptime_s + 's' : '-';
        const total = p['nv_maser_shim_requests_total'] ?? 0;
        const errors = p['nv_maser_shim_errors_total'] ?? 0;
        const lat = p['nv_maser_shim_latency_ms_avg'] ?? 0;
        document.getElementById('req-total').textContent = total;
        document.getElementById('req-errors').textContent = errors;
        document.getElementById('req-latency').textContent = lat.toFixed(3) + ' ms';
        const pct = total > 0 ? Math.round(((total - errors) / total) * 100) : 100;
        document.getElementById('success-bar').style.width = pct + '%';
        document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();
      } catch(e) {
        document.getElementById('status-badge').textContent = 'UNREACHABLE';
        document.getElementById('status-badge').className = 'badge err';
      }
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>"""


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Minimal HTML status dashboard (auto-refreshes every 5 s)."""
    return HTMLResponse(content=_DASHBOARD_HTML)


class InfoResponse(BaseModel):
    version: str
    arch: str
    grid_size: int
    num_coils: int
    onnx_available: bool


@app.get("/info", response_model=InfoResponse)
async def info():
    """Static model/server info."""
    try:
        import onnxruntime  # noqa: F401
        onnx_ok = True
    except ImportError:
        onnx_ok = False
    return InfoResponse(
        version="1.0.0",
        arch=_config.model.architecture.value if _config else "unknown",
        grid_size=_config.grid.size if _config else 0,
        num_coils=_config.coils.num_coils if _config else 0,
        onnx_available=onnx_ok,
    )


@app.post("/reload", response_model=ReloadResponse)
async def reload_model():
    """Reload the model checkpoint from disk (hot-reload without server restart)."""
    global _model
    if _model is None or _config is None:
        raise HTTPException(status_code=503, detail="Server not initialised")
    import pathlib
    ckpt = pathlib.Path("checkpoints/best.pt")
    if not ckpt.exists():
        raise HTTPException(status_code=404, detail="No checkpoint found at checkpoints/best.pt")
    saved = torch.load(ckpt, map_location="cpu")
    state_dict = saved.get("model_state", saved) if isinstance(saved, dict) else saved
    _model.load_state_dict(state_dict)
    _model.eval()
    logger.info("Model reloaded from checkpoint: %s", ckpt)
    return ReloadResponse(
        status="reloaded",
        checkpoint=str(ckpt),
        arch=_config.model.architecture.value,
    )
