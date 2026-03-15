# ADR-001: Controller Architecture — CNN vs MLP

**Status:** Accepted  
**Date:** 2026-03-14  
**Deciders:** AI Engineer, Software Architect

---

## Context

The NV-center maser active magnetic shimming system requires a real-time neural controller that takes a 64×64 measured magnetic field map as input and outputs correction currents for 8 shim coils. Two principal architectures were considered: a Convolutional Neural Network (CNNController) and a Multi-Layer Perceptron (MLPController).

The controller must:
- Run inference in < 1 ms on GPU (target); < 10 ms on CPU acceptable
- Generalise across diverse disturbance morphologies (smooth gradients, ripple modes, localised hotspots)
- Produce stable, bounded correction currents (tanh-scaled to ±max_current_amps)
- Be trainable end-to-end with the `FieldUniformityLoss`

---

## Decision

**CNNController is the default architecture.**

The CNN processes the full 64×64 field heatmap through a stack of `Conv2d → BatchNorm2d → Activation → MaxPool2d` layers, followed by `AdaptiveAvgPool2d(1)` and a linear head. The MLP flattens the same input and passes it through fully-connected layers.

---

## Consequences

### Positive
- **Local spatial receptivity**: Conv filters learn spatially-local correlations between field gradients and optimal coil responses. A hotspot at position (x, y) activates nearby filters without requiring the network to learn which of 4096 input pixels is relevant via dense weights.
- **Translation equivariance**: Disturbance patterns shifted relative to the coil array are handled naturally by shared filter weights, improving generalisation.
- **Parameter efficiency**: A 3-layer CNN with channels [32, 64, 128] uses ~250 K parameters vs ~1.3 M for a comparable MLP, reducing overfitting risk and inference cost.
- **BatchNorm stability**: Per-channel normalisation in the CNN prevents gradient pathologies during training, yielding faster convergence.

### Negative / Trade-offs
- **Global context limitation**: Each conv layer has a bounded receptive field. Very low-frequency (DC offset) disturbances spanning the full grid are better captured by global pooling — partially mitigated by `AdaptiveAvgPool2d(1)` at the bottleneck.
- **Added complexity**: CNNs require careful hyperparameter tuning of kernel sizes, channels, and pool strides; MLPs are simpler to configure and debug.

---

## Alternatives Considered

### MLPController
- Flattens the 64×64 field to a 4096-vector and passes through configurable hidden layers.
- **Pros**: Unrestricted access to all pixel pairs; simpler architecture; faster to iterate.
- **Pros in practice**: For the current 8-coil, 64×64 setting the MLP achieves competitive loss when trained longer.
- **Cons**: Dense weight matrices couple every pixel to every hidden unit, leading to high variance on unseen disturbance modes and a ~5× parameter overhead.
- **Decision**: Available as `--arch mlp` for ablation studies; not the default.

### Graph Neural Network (not implemented)
- Model coils and field points as a bipartite graph for explicit spatial reasoning.
- Rejected due to implementation complexity disproportionate to the 8-coil problem scale.

### Transformer / ViT (not implemented)
- Self-attention over 64×64 patch tokens for long-range field dependencies.
- Attractive for future work with larger arrays (32+ coils, 128×128 grids); currently over-parameterised.

---

## References
- `src/nv_maser/model/controller.py` — CNNController, MLPController, build_controller()
- `src/nv_maser/model/loss.py` — FieldUniformityLoss
- `src/nv_maser/config.py` — ModelConfig (cnn_channels, mlp_hidden_dims, activation)
