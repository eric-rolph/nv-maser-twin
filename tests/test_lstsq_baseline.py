"""Tests for the least-squares shimming baseline."""
from __future__ import annotations

import numpy as np
import pytest

from nv_maser.config import SimConfig
from nv_maser.model.lstsq_baseline import LeastSquaresShimmer
from nv_maser.physics.environment import FieldEnvironment


@pytest.fixture(scope="module")
def env() -> FieldEnvironment:
    return FieldEnvironment(SimConfig())


@pytest.fixture(scope="module")
def shimmer(env: FieldEnvironment) -> LeastSquaresShimmer:
    return LeastSquaresShimmer.from_environment(env)


def _active_std(env: FieldEnvironment, field: np.ndarray) -> float:
    return float(np.std(field[env.grid.active_zone_mask]))


class TestExactCancellation:
    def test_in_span_disturbance_cancelled(self, env):
        """With ridge=0, a disturbance built from the coil basis is
        cancelled exactly. (The training-default λ=1e-6 is NOT negligible
        at Tesla² scales and deliberately undershoots by a few percent.)"""
        shimmer = LeastSquaresShimmer(
            env.coils.influence_matrix,
            env.grid.active_zone_mask,
            env.config.coils.max_current_amps,
            ridge=0.0,
        )
        rng = np.random.default_rng(0)
        true_currents = rng.uniform(-0.5, 0.5, env.coils.num_coils).astype(
            np.float32
        )
        disturbance = env.coils.compute_field(true_currents)
        distorted = env.base_field + disturbance

        currents = shimmer.solve(distorted)
        net = distorted.astype(np.float64) + env.coils.compute_field(
            currents
        )

        # Residual std should be tiny vs the 2.5 mT disturbance amplitude
        assert _active_std(env, net) < 1e-8
        # And the solver should recover the negated generating currents
        np.testing.assert_allclose(currents, -true_currents, atol=1e-4)

    def test_zero_disturbance_gives_zero_currents(self, env, shimmer):
        currents = shimmer.solve(env.base_field)
        np.testing.assert_allclose(currents, 0.0, atol=1e-6)


class TestOptimality:
    def test_minimises_training_loss(self, env, shimmer):
        """No single-coil perturbation of the solution improves the loss.

        Loss evaluated in float64: float32 variance of ~0.05 T fields is
        too noisy to resolve optimality at the stationary point.
        """
        lam = env.config.training.current_penalty_weight
        env.disturbance_gen.randomize()
        distorted = (
            env.base_field + env.disturbance_gen.generate()
        ).astype(np.float64)
        influence = env.coils.influence_matrix.astype(np.float64)
        mask = env.grid.active_zone_mask

        def loss(currents: np.ndarray) -> float:
            net = distorted + np.einsum("c,cij->ij", currents, influence)
            return float(np.var(net[mask])) + lam * float(
                np.mean(currents**2)
            )

        star = shimmer.solve_raw(distorted.astype(np.float32))
        base = loss(star)
        for c in range(env.coils.num_coils):
            for eps in (1e-3, -1e-3):
                perturbed = star.copy()
                perturbed[c] += eps
                assert loss(perturbed) >= base - abs(base) * 1e-9

    def test_never_worse_than_no_correction(self, env, shimmer):
        """I=0 is feasible, so the optimum can't increase the variance."""
        rng = np.random.default_rng(3)
        # Out-of-span content: spatial frequency beyond the harmonic basis
        xx, yy = env.grid.x, env.grid.y
        high_freq = 1e-4 * np.sin(2 * np.pi * xx) * np.cos(
            2 * np.pi * 1.3 * yy
        )
        distorted = (env.base_field + high_freq).astype(np.float32)
        distorted += rng.normal(0, 1e-5, distorted.shape).astype(np.float32)

        net = distorted + env.coils.compute_field(shimmer.solve(distorted))
        assert _active_std(env, net) <= _active_std(env, distorted) + 1e-12


class TestConstraints:
    def test_currents_clipped_to_bound(self, env, shimmer):
        # A huge disturbance forces the unconstrained solution out of bounds
        huge = env.base_field + 100 * env.coils.compute_field(
            np.ones(env.coils.num_coils, dtype=np.float32)
        )
        currents = shimmer.solve(huge.astype(np.float32))
        bound = env.config.coils.max_current_amps
        assert np.all(np.abs(currents) <= bound + 1e-9)
        # And the raw solution really was out of bounds
        assert np.any(np.abs(shimmer.solve_raw(huge)) > bound)


class TestBatch:
    def test_batch_matches_single(self, env, shimmer):
        distorted, _ = env.generate_training_data(5)
        batch = shimmer.solve(distorted)
        assert batch.shape == (5, env.coils.num_coils)
        for i in range(5):
            np.testing.assert_allclose(
                batch[i], shimmer.solve(distorted[i]), atol=1e-7
            )


class TestEndToEnd:
    def test_improvement_on_default_disturbances(self, env, shimmer):
        """Documents a measured property of the default benchmark: the
        8-harmonic coil basis captures only ~1/3 of the synthetic
        disturbance energy, so even the OPTIMAL linear correction yields
        just ~1.25x improvement (and bounds every learned controller).
        See scripts/eval_baseline.py for the full comparison."""
        distorted, _ = env.generate_training_data(50)
        currents = shimmer.solve(distorted)
        net = distorted + env.coils.compute_field(currents)

        mask = env.grid.active_zone_mask
        before = np.std(distorted[:, mask], axis=1)
        after = np.std(net[:, mask], axis=1)
        improvement = float(np.mean(before) / np.mean(after))
        assert improvement > 1.15, f"improvement only {improvement:.2f}x"
        # If this ever jumps well past ~1.3x, the disturbance model or coil
        # basis changed — revisit the README baseline table.
        assert improvement < 2.0, (
            f"improvement {improvement:.2f}x — benchmark changed, "
            "update README baselines"
        )

    def test_controller_fn_protocol(self, env, shimmer):
        """Callable form matches solve() — required by ClosedLoopSimulator."""
        env.disturbance_gen.randomize()
        field = (env.base_field + env.disturbance_gen.generate()).astype(
            np.float32
        )
        np.testing.assert_array_equal(shimmer(field), shimmer.solve(field))
