"""
Composes the full field environment: B₀ + disturbance + coil corrections.
This is the main interface for the training loop and visualization.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import SimConfig, NVConfig, MaserConfig
from .grid import SpatialGrid
from .base_field import compute_base_field
from .disturbance import DisturbanceGenerator
from .coils import ShimCoilArray
from .maser_gain import compute_maser_metrics, max_tolerable_b_std
from .thermal import ThermalModel, ThermalState, compute_thermal_state
from .cavity import (
    compute_cavity_properties,
    compute_effective_q,
    compute_full_threshold,
    compute_magnetic_q,
    compute_spectral_overlap,
)
from .signal_chain import compute_signal_chain_budget, compute_maser_noise_temperature
from .optical_pump import compute_pump_state, compute_depth_resolved_pump
from .pulsed_pump import compute_pulsed_inversion, compute_equivalent_cw_power
from .maxwell_bloch import solve_maxwell_bloch, compute_steady_state_power
from .spectral import (
    build_detuning_grid,
    build_initial_inversion,
    compute_on_resonance_inversion,
    spectral_overlap_weights,
)
from .dipolar import estimate_dipolar_coupling_hz, estimate_refilling_time_us
from .spectral_maxwell_bloch import solve_spectral_maxwell_bloch


class FieldEnvironment:
    """
    Complete simulation environment.

    Manages the spatial grid, base field, disturbance generator, and coil array.
    Provides the interface for:

    - Generating distorted field observations (for the AI controller input)
    - Applying coil corrections and computing the net field
    - Evaluating field uniformity (for the loss function)
    """

    def __init__(self, config: SimConfig, thermal_seed: int | None = None) -> None:
        self.config = config
        self.grid = SpatialGrid(config.grid)
        self.base_field = compute_base_field(self.grid, config.field, config.halbach)
        self.disturbance_gen = DisturbanceGenerator(self.grid, config.disturbance)
        self.coils = ShimCoilArray(self.grid, config.coils)

        # Close pump → thermal loop: compute pump thermal load and inject
        # into ThermalConfig before building the ThermalModel.
        pump = compute_pump_state(config.optical_pump, config.nv)
        thermal_cfg = config.thermal
        if pump.thermal_load_w > 0 and thermal_cfg.external_heat_w == 0.0:
            thermal_cfg = thermal_cfg.model_copy(
                update={"external_heat_w": pump.thermal_load_w}
            )

        # Thermal model (always created; at default 25°C → zero offset)
        self.thermal_model = ThermalModel(thermal_cfg, seed=thermal_seed)
        self._thermal_state: ThermalState | None = None

        # Current state
        self._current_disturbance: NDArray[np.float32] | None = None

    @property
    def thermal_state(self) -> ThermalState | None:
        """Current thermal state, if step() has been called."""
        return self._thermal_state

    @property
    def effective_base_field(self) -> NDArray[np.float32]:
        """B₀ adjusted for thermal drift."""
        if self._thermal_state is not None:
            return self.base_field + np.float32(self._thermal_state.b0_shift_tesla)
        return self.base_field

    @property
    def distorted_field(self) -> NDArray[np.float32]:
        """B₀ (thermally shifted) + current disturbance (before correction)."""
        if self._current_disturbance is None:
            self._current_disturbance = self.disturbance_gen.generate()
        return self.effective_base_field + self._current_disturbance

    def step(self, t: float = 0.0) -> NDArray[np.float32]:
        """
        Advance the environment: generate a new disturbance at time t.
        Also updates the thermal state.

        Returns the distorted field (without correction).
        """
        self._current_disturbance = self.disturbance_gen.generate(t)
        self._thermal_state = self.thermal_model.state_at(
            t, self.config.field, self.config.nv,
            self.config.maser, self.config.feedback,
        )
        return self.distorted_field

    def apply_correction(
        self, currents: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Apply coil currents and return the net (corrected) field.

        Args:
            currents: (num_coils,) coil current array.

        Returns:
            (size, size) net field = B₀ + disturbance + coil_field.
        """
        coil_field = self.coils.compute_field(currents)
        return self.distorted_field + coil_field

    def compute_uniformity_metric(
        self, net_field: NDArray[np.float32]
    ) -> dict[str, float]:
        """
        Compute field uniformity metrics over the active zone.

        When thermal state is available, uses temperature-adjusted T2* and Q.

        Returns dict with:
            - variance: Var(B) over active zone (the primary loss target)
            - std: std(B) over active zone in Tesla
            - ppm: peak-to-peak homogeneity in ppm
            - max_deviation: max |B - B₀| over active zone
            - temperature_c: current temperature (if thermal active)
        """
        mask = self.grid.active_zone_mask
        active = net_field[mask]

        mean_b = float(np.mean(active))
        var_b = float(np.var(active))
        std_b = float(np.std(active))
        min_b = float(np.min(active))
        max_b = float(np.max(active))

        ppm = ((max_b - min_b) / mean_b * 1e6) if mean_b > 0 else float("inf")
        max_dev = float(np.max(np.abs(active - self.config.field.b0_tesla)))

        # Use thermally-adjusted NV/maser params if available
        nv_config = self.config.nv
        maser_config = self.config.maser
        if self._thermal_state is not None:
            nv_config = nv_config.model_copy(
                update={"t2_star_us": self._thermal_state.effective_t2_star_us}
            )
            maser_config = maser_config.model_copy(
                update={"cavity_q": self._thermal_state.effective_cavity_q}
            )

        # Q-boosting via electronic feedback (Wang 2024)
        if maser_config.q_boost_gain > 0:
            q_eff = compute_effective_q(maser_config)
            maser_config = maser_config.model_copy(
                update={"cavity_q": q_eff}
            )

        maser = compute_maser_metrics(net_field, mask, nv_config, maser_config)

        result = {
            "variance": var_b,
            "std": std_b,
            "ppm": ppm,
            "max_deviation": max_dev,
            **maser,
        }

        if self._thermal_state is not None:
            result["temperature_c"] = self._thermal_state.temperature_c
            result["b0_shift_tesla"] = self._thermal_state.b0_shift_tesla

        # Q-boost reporting
        if self.config.maser.q_boost_gain > 0:
            result["q_boost_effective_q"] = maser_config.cavity_q
            result["q_boost_gain"] = self.config.maser.q_boost_gain

        # Optical pump — compute FIRST so effective efficiency feeds
        # into signal_chain and cavity calculations (closes pump→inversion loop).
        pump = compute_pump_state(self.config.optical_pump, nv_config)
        result["pump_rate_hz"] = pump.pump_rate_hz
        result["pump_saturation"] = pump.pump_saturation
        result["effective_pump_efficiency"] = pump.effective_pump_efficiency
        result["thermal_load_w"] = pump.thermal_load_w

        # Depth-resolved pump (when n_depth_slices > 1)
        pump_cfg = self.config.optical_pump
        effective_eta = pump.effective_pump_efficiency
        if pump_cfg.n_depth_slices > 1:
            dr_pump = compute_depth_resolved_pump(
                pump_cfg, nv_config, pump_cfg.n_depth_slices,
            )
            effective_eta = dr_pump.effective_pump_efficiency
            result["effective_pump_efficiency"] = effective_eta
            result["pump_front_back_ratio"] = dr_pump.front_back_ratio

        # Pulsed pump metrics (config-gated)
        if pump_cfg.pulsed:
            pulsed = compute_pulsed_inversion(pump_cfg, nv_config)
            result["pulsed_peak_inversion"] = pulsed.peak_inversion
            result["pulsed_mean_inversion"] = pulsed.mean_inversion
            result["pulsed_duty_cycle"] = pulsed.duty_cycle
            result["pulsed_equivalent_cw_power_w"] = compute_equivalent_cw_power(
                pump_cfg
            )

        # Override pump_efficiency with the dynamic value from optical pump
        nv_config = nv_config.model_copy(
            update={"pump_efficiency": effective_eta}
        )

        # Signal chain SNR budget (now uses dynamic pump efficiency)
        gain_budget = maser["gain_budget"]
        snr_budget = compute_signal_chain_budget(
            nv_config, maser_config, self.config.signal_chain, gain_budget
        )
        result["snr_db"] = snr_budget.snr_db
        result["received_power_w"] = snr_budget.received_power_w
        result["total_noise_w"] = snr_budget.total_noise_w
        result["system_noise_temperature_k"] = snr_budget.system_noise_temperature_k

        # Cavity QED threshold (now uses dynamic pump efficiency)
        gamma_eff_hz = maser["gamma_eff_ghz"] * 1e9  # GHz → Hz
        threshold = compute_full_threshold(
            nv_config, maser_config, self.config.cavity,
            gain_budget, gamma_eff_hz,
        )
        result["cooperativity"] = threshold.cooperativity
        result["threshold_margin"] = threshold.threshold_margin
        result["n_effective"] = threshold.n_effective
        result["ensemble_coupling_hz"] = threshold.ensemble_coupling_hz

        # Magnetic Q, spectral overlap, maser noise temperature
        cavity_props = compute_cavity_properties(maser_config, self.config.cavity)
        mag_q = compute_magnetic_q(nv_config, self.config.cavity)
        result["q_magnetic"] = mag_q.q_magnetic

        result["spectral_overlap_R"] = compute_spectral_overlap(
            cavity_props, gamma_eff_hz,
        )

        result["maser_noise_temperature_k"] = compute_maser_noise_temperature(
            mag_q.q_magnetic,
            maser_config.cavity_q,
            self.config.signal_chain.physical_temperature_k,
        )

        # Maxwell-Bloch time-domain metrics (config-gated)
        mb_config = self.config.maxwell_bloch
        if mb_config.enable:
            mb_result = solve_maxwell_bloch(
                nv_config, maser_config, self.config.cavity,
                mb_config, gain_budget,
            )
            result["mb_output_power_w"] = mb_result.output_power_w
            result["mb_steady_state_photons"] = mb_result.steady_state_photons
            result["mb_cooperativity"] = mb_result.cooperativity
            if mb_result.gain_db is not None:
                result["mb_gain_db"] = mb_result.gain_db
            result["mb_analytical_power_w"] = compute_steady_state_power(
                nv_config, maser_config, self.config.cavity, gain_budget,
            )

        # Spectral dynamics metrics (config-gated)
        if self.config.spectral.enable:
            delta_hz, p_delta = build_initial_inversion(
                nv_config, self.config.spectral,
            )
            cavity_bw_hz = (
                maser_config.cavity_frequency_ghz * 1e9 / maser_config.cavity_q
            )
            sz_on_res = compute_on_resonance_inversion(
                p_delta, delta_hz, cavity_bw_hz,
            )
            result["spectral_on_resonance_inversion"] = sz_on_res
            result["spectral_inhomogeneous_linewidth_mhz"] = (
                self.config.spectral.inhomogeneous_linewidth_mhz
            )

        # Dipolar interaction metrics (config-gated)
        if self.config.dipolar.enable:
            n_nv = nv_config.nv_density_m3
            result["dipolar_coupling_hz"] = estimate_dipolar_coupling_hz(n_nv)
            result["dipolar_refilling_time_us"] = estimate_refilling_time_us(n_nv)

        # Spectral Maxwell-Bloch solver (requires both MB and spectral enabled)
        if mb_config.enable and self.config.spectral.enable:
            smb_result = solve_spectral_maxwell_bloch(
                nv_config, maser_config, self.config.cavity,
                mb_config, self.config.spectral,
                self.config.dipolar if self.config.dipolar.enable else None,
                gain_budget,
            )
            result["smb_output_power_w"] = smb_result.output_power_w
            result["smb_steady_state_photons"] = smb_result.steady_state_photons
            result["smb_on_res_inversion"] = smb_result.steady_state_on_res_inversion
            result["smb_cooperativity"] = smb_result.cooperativity
            result["smb_n_bursts"] = smb_result.n_bursts

        return result

    def generate_training_data(
        self, num_samples: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Generate a dataset of distorted fields for training.

        Returns:
            distorted_fields: (num_samples, size, size) — model inputs
            disturbances:     (num_samples, size, size) — ground-truth disturbances
        """
        disturbances = self.disturbance_gen.generate_batch(num_samples)
        distorted = self.base_field[np.newaxis, :, :] + disturbances
        return distorted, disturbances
