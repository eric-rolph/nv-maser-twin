"""
Real-time PyQtGraph dashboard for the NV Maser shimming simulation.

Layout::

    ┌──────────────────────────────────────────────────────────┐
    │                  NV Maser Shimming Dashboard              │
    ├───────────────────┬──────────────────┬───────────────────┤
    │  DISTORTED FIELD  │ CORRECTION FIELD │    NET FIELD      │
    │   [heatmap]       │   [heatmap]      │   [heatmap]       │
    ├───────────────────┴──────────────────┴───────────────────┤
    │ Status | coil currents bar chart                         │
    └──────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import time

import numpy as np
import torch

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ..config import SimConfig
from ..physics.environment import FieldEnvironment


class NVMaserDashboard(QtWidgets.QMainWindow):
    """Main dashboard window."""

    def __init__(
        self,
        env: FieldEnvironment,
        model: "torch.nn.Module",
        influence_tensor: "torch.Tensor",
        config: SimConfig,
        device: str,
    ) -> None:
        super().__init__()
        self.env = env
        self.model = model
        self.influence_tensor = influence_tensor
        self.config = config
        self.device = device
        self._t = 0.0
        self._paused = False
        self._amplitude_scale = 1.0

        self.setWindowTitle("NV Maser Active Shimming — Digital Twin")
        self.resize(1200, 700)

        self._build_ui()
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._update_frame)
        self._timer.start(config.viz.update_interval_ms)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # Top row: three heatmaps
        heatmap_row = QtWidgets.QHBoxLayout()
        self._heatmaps: list[pg.ImageView] = []
        titles = ["Distorted Field (B₀ + noise)", "Correction Field (coils)", "Net Field"]
        for title in titles:
            container = QtWidgets.QGroupBox(title)
            box_layout = QtWidgets.QVBoxLayout(container)
            iv = pg.ImageView()
            iv.ui.roiBtn.hide()
            iv.ui.menuBtn.hide()
            iv.setColorMap(pg.colormap.get(self.config.viz.colormap))
            vmin, vmax = self.config.viz.field_range_tesla
            iv.setLevels(vmin, vmax)
            box_layout.addWidget(iv)
            heatmap_row.addWidget(container)
            self._heatmaps.append(iv)
        main_layout.addLayout(heatmap_row, stretch=4)

        # Bottom row: controls + metrics + coil bar chart
        bottom_row = QtWidgets.QHBoxLayout()

        # Controls
        ctrl_box = QtWidgets.QGroupBox("Controls")
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_box)

        self._pause_btn = QtWidgets.QPushButton("⏸ Pause")
        self._pause_btn.setCheckable(True)
        self._pause_btn.toggled.connect(self._on_pause_toggle)
        ctrl_layout.addWidget(self._pause_btn)

        ctrl_layout.addWidget(QtWidgets.QLabel("Disturbance Intensity:"))
        self._intensity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._intensity_slider.setRange(0, 200)
        self._intensity_slider.setValue(100)
        self._intensity_slider.valueChanged.connect(
            lambda v: setattr(self, "_amplitude_scale", v / 100.0)
        )
        ctrl_layout.addWidget(self._intensity_slider)

        bottom_row.addWidget(ctrl_box, stretch=1)

        # Metrics display
        metrics_box = QtWidgets.QGroupBox("Metrics")
        metrics_layout = QtWidgets.QFormLayout(metrics_box)
        self._metric_labels: dict[str, QtWidgets.QLabel] = {}
        for key in ("Field Std (μT)", "PPM", "Inference (ms)", "FPS"):
            lbl = QtWidgets.QLabel("—")
            lbl.setStyleSheet("font-family: monospace; font-size: 14px;")
            metrics_layout.addRow(key + ":", lbl)
            self._metric_labels[key] = lbl
        bottom_row.addWidget(metrics_box, stretch=1)

        # Coil current bar chart
        coil_box = QtWidgets.QGroupBox("Coil Currents (A)")
        coil_layout = QtWidgets.QVBoxLayout(coil_box)
        self._coil_plot = pg.PlotWidget()
        self._coil_plot.setYRange(
            -self.config.coils.max_current_amps,
            self.config.coils.max_current_amps,
        )
        self._coil_plot.setFixedHeight(120)
        n = self.config.coils.num_coils
        self._coil_bars = pg.BarGraphItem(
            x=np.arange(n),
            height=np.zeros(n),
            width=0.7,
            brush="steelblue",
        )
        self._coil_plot.addItem(self._coil_bars)
        self._coil_plot.setLabel("bottom", "Coil index")
        self._coil_plot.setLabel("left", "Current (A)")
        coil_layout.addWidget(self._coil_plot)
        bottom_row.addWidget(coil_box, stretch=2)

        main_layout.addLayout(bottom_row, stretch=1)

    # ── Frame update ───────────────────────────────────────────────────────

    def _update_frame(self) -> None:
        if self._paused:
            return

        # Step environment
        distorted = self.env.step(self._t)
        self._t += self.config.viz.update_interval_ms / 1000.0

        # Apply intensity scale to the stored disturbance
        if self._amplitude_scale != 1.0 and self.env._current_disturbance is not None:
            scaled = self.env._current_disturbance * self._amplitude_scale
            self.env._current_disturbance = scaled
            distorted = self.env.distorted_field

        # Inference
        t0 = time.perf_counter()
        X = torch.tensor(
            distorted[np.newaxis, np.newaxis, :, :], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            currents = self.model(X)
            coil_field = torch.einsum("bc,cij->bij", currents, self.influence_tensor)

        infer_ms = (time.perf_counter() - t0) * 1000

        correction_np = coil_field[0].cpu().numpy()
        net_np = distorted + correction_np
        currents_np = currents[0].cpu().numpy()

        # Update heatmaps
        for iv, field in zip(
            self._heatmaps, [distorted, correction_np, net_np]
        ):
            iv.setImage(field.T, autoLevels=False)

        # Compute metrics
        metrics = self.env.compute_uniformity_metric(net_np)
        fps = 1000.0 / max(self.config.viz.update_interval_ms, 1)
        self._metric_labels["Field Std (μT)"].setText(f"{metrics['std']*1e6:.2f}")
        self._metric_labels["PPM"].setText(f"{metrics['ppm']:.1f}")
        self._metric_labels["Inference (ms)"].setText(f"{infer_ms:.2f}")
        self._metric_labels["FPS"].setText(f"{fps:.0f}")

        # Update coil bar chart
        self._coil_bars.setOpts(height=currents_np)

    def _on_pause_toggle(self, paused: bool) -> None:
        self._paused = paused
        self._pause_btn.setText("▶ Resume" if paused else "⏸ Pause")


def run_dashboard(
    env: FieldEnvironment,
    model: "torch.nn.Module",
    influence_tensor: "torch.Tensor",
    config: SimConfig,
) -> None:
    """Launch the PyQtGraph dashboard. Blocks until window is closed."""
    device = config.resolve_device()
    app = pg.mkQApp("NV Maser Digital Twin")
    pg.setConfigOptions(antialias=True)

    window = NVMaserDashboard(env, model, influence_tensor, config, device)
    window.show()
    app.exec()
