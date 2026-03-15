"""
Matplotlib static plots for analysis and export.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure

from ..physics.grid import SpatialGrid
from ..physics.coils import ShimCoilArray


def plot_training_history(
    history: dict[str, list[float]], save_path: str | None = None
) -> Figure:
    """Plot training/validation loss curves over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].semilogy(epochs, history["train_loss"], label="Train", color="royalblue")
    axes[0].semilogy(epochs, history["val_loss"], label="Validation", color="tomato")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Improvement ratio per epoch
    ratio = np.array(history["val_loss"]) / history["val_loss"][0]
    axes[1].plot(epochs, ratio, color="seagreen")
    axes[1].axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val loss / initial loss")
    axes[1].set_title("Relative Improvement")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"[plots] Saved training history → {save_path}")
    return fig


def plot_field_snapshot(
    distorted: np.ndarray,
    correction: np.ndarray,
    net: np.ndarray,
    grid: SpatialGrid,
    coil_array: ShimCoilArray,
    save_path: str | None = None,
    colormap: str = "RdBu_r",
    field_range: tuple[float, float] | None = None,
) -> Figure:
    """Three-panel heatmap snapshot (Matplotlib version for export/notebooks)."""
    vmin, vmax = field_range or (
        float(min(distorted.min(), net.min())),
        float(max(distorted.max(), net.max())),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Distorted Field (B₀ + noise)", "Correction Field (coils)", "Net Field"]
    fields = [distorted, correction, net]
    extent = [-grid.extent / 2, grid.extent / 2, -grid.extent / 2, grid.extent / 2]

    for ax, title, field in zip(axes, titles, fields):
        im = ax.imshow(
            field,
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap=colormap,
            aspect="equal",
        )
        plt.colorbar(im, ax=ax, label="B (T)")

        # Active zone boundary
        az_half = grid.extent * grid.active_fraction / 2
        rect = patches.Rectangle(
            (-az_half, -az_half),
            2 * az_half,
            2 * az_half,
            linewidth=1.5,
            edgecolor="lime",
            facecolor="none",
            linestyle="--",
            label="Active zone",
        )
        ax.add_patch(rect)

        # Coil positions
        ax.scatter(
            coil_array.coil_x,
            coil_array.coil_y,
            color="yellow",
            edgecolors="black",
            s=80,
            zorder=5,
            label="Coils",
        )

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(title)

    axes[0].legend(fontsize=8, loc="lower right")
    plt.suptitle("NV Maser Field Snapshot", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"[plots] Saved snapshot → {save_path}")
    return fig


def plot_coil_influence(
    coil_array: ShimCoilArray, save_path: str | None = None
) -> Figure:
    """Visualize the influence matrix of each coil as a subplot grid."""
    n = coil_array.num_coils
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for i in range(n):
        ax = axes_flat[i]
        influence = coil_array.influence_matrix[i]
        im = ax.imshow(influence, origin="lower", cmap="hot", aspect="equal")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Coil {i}")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for i in range(n, rows * cols):
        axes_flat[i].set_visible(False)

    plt.suptitle("Shim Coil Influence Matrix", fontsize=13)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"[plots] Saved coil influence → {save_path}")
    return fig


def plot_disturbance_spectrum(
    disturbance: np.ndarray, save_path: str | None = None
) -> Figure:
    """2D FFT power spectrum of a disturbance field (verify low-freq content)."""
    fft = np.fft.fft2(disturbance)
    power = np.abs(np.fft.fftshift(fft)) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(disturbance, origin="lower", cmap="RdBu_r")
    axes[0].set_title("Disturbance Field")

    im = axes[1].imshow(np.log1p(power), origin="lower", cmap="inferno")
    plt.colorbar(im, ax=axes[1], label="log(1+power)")
    axes[1].set_title("2D Power Spectrum (log scale)")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"[plots] Saved spectrum → {save_path}")
    return fig
