# ADR-004: Temporal Controller Architecture — LSTM

**Status:** Accepted  
**Date:** 2026-03-14  
**Deciders:** AI Engineer, Software Architect

---

## Context

The CNN and MLP controllers are stateless: each inference call maps a single field-map frame → correction currents independently of all prior frames. In practice, real shimming systems face temporally-structured disturbances:

- **Field drift**: slow thermal gradients that evolve over seconds to minutes
- **Oscillating disturbances**: periodic mechanical vibration coupling into the field at 10–100 Hz
- **Transient events**: fast magnetic events (coil current surges, nearby equipment) whose full effect spans multiple measurement frames

A controller with no temporal memory must re-identify the disturbance type from every single frame. A controller with memory can use prior observations to anticipate disturbance evolution and apply smoother, more stable corrections.

Four options were evaluated:

1. **Stateless controller (status quo)** — keep CNN/MLP, no architectural change
2. **LSTM temporal memory** — CNN feature extractor → LSTM → linear head
3. **Transformer with causal attention** — CNN features as token sequence, masked self-attention
4. **State-space model (S4 / Mamba)** — linear recurrence with structured state matrix

---

## Decision

**LSTMController (option 2).**

The architecture is:

```
Input: (batch, seq_len, 1, grid_size, grid_size)
  → CNN feature extractor (shared weights with CNNController)
  → Flatten → (batch, seq_len, feature_dim)
  → LSTM(input=feature_dim, hidden=256, num_layers=2, batch_first=True)
  → Last hidden state → Linear(256, num_coils)
  → tanh × max_current_amps
```

**Sequence-of-1 compatibility**: when `seq_len=1` (single-frame inference), the LSTM behaves identically to a stateless pass. The hidden state `(h, c)` is preserved across steps in streaming deployments, giving full temporal context with zero API change relative to the existing `Controller.predict(field_map)` interface.

---

## Consequences

### Positive
- **Temporal context at near-zero API cost**: passing `seq_len=1` with a preserved hidden state is a drop-in replacement for the existing inference interface.
- **Well-understood training dynamics**: LSTM gradient flow is mature in PyTorch; `nn.utils.clip_grad_norm_` mitigates exploding gradients out of the box.
- **Shared CNN backbone**: the feature extractor is weight-compatible with CNNController checkpoints, enabling warm-start from a pre-trained CNN before fine-tuning the recurrent head.
- **Disturbance timescale coverage**: LSTM hidden size 256 × 2 layers gives sufficient capacity to model drift (slow) and oscillations (fast) simultaneously within a single hidden state.

### Negative / Trade-offs
- **Parameter overhead**: LSTMController adds ~881 K parameters vs CNNController's ~32 K — a 27× increase concentrated in the recurrent layers.
- **Inference latency (CPU)**: measured P95 = 1.6 ms under CPU load spikes, slightly above the < 1 ms spec. On any CUDA-capable GPU the target is met with margin.
- **Sequential dependence**: LSTM hidden state cannot be parallelised across time steps, preventing the vectorised-env batching optimisation available to stateless controllers.
- **Hidden state persistence**: production deployment must manage `(h, c)` tensors per active channel; a reset policy is required when a new shimming epoch begins.

---

## Latency Summary

| Controller | Params | CPU median | CPU P95 | GPU P95 |
|-----------|--------|-----------|---------|---------|
| MLPController | ~1.3 M | ~1.2 ms | ~2.1 ms | < 0.1 ms |
| CNNController | ~32 K | ~0.8 ms | ~1.3 ms | < 0.1 ms |
| LSTMController | ~881 K | ~1.0 ms | ~1.6 ms | < 0.2 ms |

CPU P95 = 1.6 ms marginally exceeds the 1 ms spec; this is acceptable for GPU deployment and for CPU-only lab units where the NMR shim update rate is ≤ 100 Hz (10 ms budget).

---

## Alternatives Considered

### Stateless controller (option 1)
- No change to existing code; zero added complexity.
- **Con**: Cannot distinguish a persisting gradient from a transient spike without external state management.
- **Decision**: Retained as the default `--arch cnn`; LSTM is opt-in via `--arch lstm`.

### Transformer with causal attention (option 3)
- Full self-attention over a window of K frames; superior long-range temporal modelling.
- **Con**: O(K²) attention complexity; requires fixed context window management; overkill for the < 10-frame effective memory needed by typical lab disturbances.
- **Decision**: Deferred to future work if disturbance timescales exceed LSTM capacity.

### State-space model — S4 / Mamba (option 4)
- Linear recurrence with structured state matrix; trains in parallel (unlike LSTM), inference is recurrent.
- **Con**: Mamba has unstable PyTorch bindings on Windows (CUDA kernel compilation failures observed in CI); S4 requires custom CUDA extensions.
- **Decision**: Rejected for current hardware environment; revisit when Mamba ships a stable pip wheel.

---

## References
- `src/nv_maser/model/controller.py` — `LSTMController`, `build_controller()`
- `src/nv_maser/config.py` — `ModelConfig.arch` (`"cnn"` | `"mlp"` | `"lstm"`)
- `benchmarks/benchmark_inference.py` — latency measurements
- ADR-001 — CNN vs MLP baseline architecture
