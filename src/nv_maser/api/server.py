"""FastAPI inference server for real-time field shimming."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import torch
import time
from ..config import SimConfig
from ..physics.environment import FieldEnvironment
from ..model.controller import build_controller

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
    yield
    # Teardown (nothing needed)


app = FastAPI(title="NV Maser Shim API", version="1.0.0", lifespan=lifespan)


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


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        grid_size=_config.grid.size if _config else 0,
    )


@app.post("/shim", response_model=ShimResponse)
async def shim(req: FieldRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    field = np.array(req.distorted_field, dtype=np.float32)
    if field.shape != (_config.grid.size, _config.grid.size):
        raise HTTPException(
            status_code=422,
            detail=f"Field must be {_config.grid.size}x{_config.grid.size}",
        )

    x = torch.from_numpy(field).unsqueeze(0).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        currents = _model(x).squeeze(0).numpy()
    inference_ms = (time.perf_counter() - t0) * 1000

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
