"""
Classical least-squares shimming baseline.

The coil field is linear in the currents (``B_coil = Σ I_c · P_c``), so the
training objective ``Var(B_net over active zone) + λ·mean(I²)`` has a
closed-form minimiser: a ridge-regularised least-squares projection of the
distorted field onto the coil basis. This is how production NMR/MRI shim
systems actually compute shim currents.

The baseline serves as the null hypothesis for the neural controllers: any
learned model must beat (or at least match) this solver to justify itself
on the clean shimming task. The learned models earn their keep, if at all,
in the closed-loop setting (sensor noise, DAC quantisation, settling,
latency) — compare there with ``scripts/eval_baseline.py``.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..physics.environment import FieldEnvironment


class LeastSquaresShimmer:
    """Closed-form optimal shim currents for the linear coil model.

    Minimises exactly the supervised training loss

        L(I) = Var(d + Σ I_c·P_c over active zone) + λ·mean(I²)

    via the normal equations. The variance term is invariant to the field
    mean, so both the disturbed field and the coil patterns are mean-centred
    over the active zone before solving. Currents are clipped to the
    physical ±max_current bound after the solve (real shim supplies do the
    same); use :meth:`solve_raw` for the unconstrained solution.
    """

    def __init__(
        self,
        influence_matrix: NDArray[np.float32],
        active_mask: NDArray[np.bool_],
        max_current_amps: float,
        ridge: float = 1e-6,
    ) -> None:
        """
        Args:
            influence_matrix: (num_coils, H, W) coil patterns in Tesla/Amp.
            active_mask:      (H, W) boolean active-zone mask.
            max_current_amps: Physical current bound per coil.
            ridge:            λ in the loss above. Match
                              ``training.current_penalty_weight`` for an
                              apples-to-apples comparison with the NN.
        """
        self.max_current = float(max_current_amps)
        self.mask = active_mask

        num_coils = influence_matrix.shape[0]
        # A: (n_active, num_coils) coil patterns over the active zone
        a = influence_matrix[:, active_mask].astype(np.float64).T
        n_active = a.shape[0]

        # Mean-centre the columns (variance ignores the DC component)
        self._a_centered = a - a.mean(axis=0, keepdims=True)

        # Normal equations for (1/n)‖d̃ + ÃI‖² + (λ/C)‖I‖²:
        #   (ÃᵀÃ/n + (λ/C)·Id) I = −Ãᵀd̃/n
        gram = self._a_centered.T @ self._a_centered / n_active
        gram += (ridge / num_coils) * np.eye(num_coils)
        # Pre-fold everything into one linear map: I = M @ d_active
        # (centring of d is absorbed because Ã's columns are zero-mean,
        #  so Ãᵀd̃ = Ãᵀd)
        self._solve_map = -np.linalg.solve(
            gram, self._a_centered.T / n_active
        )  # (num_coils, n_active)

    @classmethod
    def from_environment(
        cls, env: FieldEnvironment, ridge: float | None = None
    ) -> "LeastSquaresShimmer":
        """Build from a FieldEnvironment, defaulting λ to the training value."""
        if ridge is None:
            ridge = env.config.training.current_penalty_weight
        return cls(
            env.coils.influence_matrix,
            env.grid.active_zone_mask,
            env.config.coils.max_current_amps,
            ridge=ridge,
        )

    def solve_raw(
        self, fields: NDArray[np.float32]
    ) -> NDArray[np.float64]:
        """Unconstrained optimal currents for (H, W) or (B, H, W) fields."""
        if fields.ndim == 2:
            return self._solve_map @ fields[self.mask].astype(np.float64)
        # Batch: (B, n_active) @ (n_active, C)ᵀ → (B, C)
        d = fields[:, self.mask].astype(np.float64)
        return d @ self._solve_map.T

    def solve(self, fields: NDArray[np.float32]) -> NDArray[np.float32]:
        """Optimal currents clipped to the physical bound.

        Args:
            fields: (H, W) single field or (B, H, W) batch, Tesla.

        Returns:
            (num_coils,) or (B, num_coils) currents in Amps, float32.
        """
        raw = self.solve_raw(fields)
        return np.clip(raw, -self.max_current, self.max_current).astype(
            np.float32
        )

    def __call__(
        self, field: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """controller_fn protocol for ClosedLoopSimulator."""
        return self.solve(field)
