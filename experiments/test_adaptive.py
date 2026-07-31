"""Correctness tests for the strongly adaptive experiment implementation."""

from __future__ import annotations

import math
import unittest

import numpy as np

from .algorithms import StronglyAdaptiveDefensiveBooster
from .weak_learners import (
    DyadicAdaptMLProd,
    DyadicFiniteSecondOrderLinearOracle,
    DyadicLinearSecondOrderOracle,
    DyadicScalarAdaptiveOGD,
    FiniteSecondOrderLinearOracle,
    LinearSecondOrderOracle,
    ScalarAdaptiveOGD,
)


class _ReferenceDyadicLinear:
    """Object-based reference used to audit the vectorized implementation."""

    def __init__(self, dim: int, horizon: int) -> None:
        self.dim = dim
        self.meta = DyadicAdaptMLProd(horizon)
        self.learners = [LinearSecondOrderOracle(dim) for _ in self.meta.lengths]
        self.x: np.ndarray | None = None

    def predict(self, x: np.ndarray) -> float:
        reset = self.meta.start_round()
        for index in np.flatnonzero(reset):
            self.learners[int(index)] = LinearSecondOrderOracle(self.dim)
        predictions = np.asarray([learner.predict(x) for learner in self.learners])
        self.x = x
        return self.meta.aggregate(predictions)

    def update_gain(self, x: np.ndarray, coeff: float) -> None:
        for learner in self.learners:
            learner.update_gain(x, coeff)
        self.meta.update(coeff)
        self.x = None


class _ReferenceDyadicScalar:
    def __init__(self, horizon: int) -> None:
        self.meta = DyadicAdaptMLProd(horizon)
        self.learners = [ScalarAdaptiveOGD() for _ in self.meta.lengths]

    def predict(self) -> float:
        reset = self.meta.start_round()
        for index in np.flatnonzero(reset):
            self.learners[int(index)] = ScalarAdaptiveOGD()
        return self.meta.aggregate(np.asarray([learner.predict() for learner in self.learners]))

    def update_gain(self, coeff: float) -> None:
        for learner in self.learners:
            learner.update_gain(coeff)
        self.meta.update(coeff)


class _ReferenceDyadicFinite:
    def __init__(self, base_dim: int, horizon: int) -> None:
        self.base_dim = base_dim
        self.meta = DyadicAdaptMLProd(horizon)
        self.learners = [
            FiniteSecondOrderLinearOracle(base_dim) for _ in self.meta.lengths
        ]

    def predict(self, x: np.ndarray) -> float:
        reset = self.meta.start_round()
        for index in np.flatnonzero(reset):
            self.learners[int(index)] = FiniteSecondOrderLinearOracle(self.base_dim)
        predictions = np.asarray([learner.predict(x) for learner in self.learners])
        return self.meta.aggregate(predictions)

    def update_gain(self, x: np.ndarray, coeff: float) -> None:
        for learner in self.learners:
            learner.update_gain(x, coeff)
        self.meta.update(coeff)


class DyadicAdaptMLProdTest(unittest.TestCase):
    def test_first_update_matches_algorithm_two(self) -> None:
        meta = DyadicAdaptMLProd(horizon=8)
        reset = meta.start_round()
        self.assertTrue(np.all(reset))
        predictions = np.linspace(-1.0, 1.0, meta.num_active_specialists)
        mixture = meta.aggregate(predictions)
        self.assertAlmostEqual(mixture, float(np.mean(predictions)))

        old_log_weights = meta.log_weights.copy()
        old_rates = meta._learning_rates.copy()
        coeff = 1.6
        excess = coeff * (predictions - mixture) / 4.0
        new_variances = excess * excess
        new_rates = meta._rates(new_variances)
        expected = (new_rates / old_rates) * (
            old_log_weights + np.log1p(old_rates * excess)
        )
        meta.update(coeff)
        np.testing.assert_allclose(meta.log_weights, expected, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(meta.excess_variances, new_variances, rtol=0.0, atol=1e-14)

        reset = meta.start_round()
        np.testing.assert_array_equal(reset, np.array([True, False, False, False]))

    def test_vectorized_linear_matches_object_reference(self) -> None:
        rng = np.random.default_rng(7)
        horizon = 32
        vectorized = DyadicLinearSecondOrderOracle(dim=5, horizon=horizon)
        reference = _ReferenceDyadicLinear(dim=5, horizon=horizon)
        for _ in range(horizon):
            x = rng.normal(size=5)
            x /= max(float(np.linalg.norm(x)), 1.0)
            coeff = float(rng.uniform(-2.0, 2.0))
            self.assertAlmostEqual(vectorized.predict(x), reference.predict(x), places=13)
            vectorized.update_gain(x, coeff)
            reference.update_gain(x, coeff)
            np.testing.assert_allclose(vectorized.w, np.vstack([learner.w for learner in reference.learners]), atol=1e-13)
            np.testing.assert_allclose(vectorized.meta.log_weights, reference.meta.log_weights, atol=1e-13)

    def test_vectorized_scalar_matches_object_reference(self) -> None:
        rng = np.random.default_rng(11)
        horizon = 32
        vectorized = DyadicScalarAdaptiveOGD(horizon)
        reference = _ReferenceDyadicScalar(horizon)
        for _ in range(horizon):
            coeff = float(rng.uniform(-2.0, 2.0))
            self.assertAlmostEqual(vectorized.predict(), reference.predict(), places=14)
            vectorized.update_gain(coeff)
            reference.update_gain(coeff)
            np.testing.assert_allclose(
                vectorized.values,
                np.asarray([learner.value for learner in reference.learners]),
                atol=1e-14,
            )
            np.testing.assert_allclose(vectorized.meta.log_weights, reference.meta.log_weights, atol=1e-13)

    def test_vectorized_finite_matches_object_reference(self) -> None:
        rng = np.random.default_rng(13)
        horizon = 32
        base_dim = 7
        vectorized = DyadicFiniteSecondOrderLinearOracle(base_dim, horizon)
        reference = _ReferenceDyadicFinite(base_dim, horizon)
        for _ in range(horizon):
            x = rng.uniform(-1.0, 1.0, size=base_dim)
            coeff = float(rng.uniform(-2.0, 2.0))
            self.assertAlmostEqual(vectorized.predict(x), reference.predict(x), places=13)
            vectorized.update_gain(x, coeff)
            reference.update_gain(x, coeff)
            np.testing.assert_allclose(
                vectorized.cumulative_gains,
                np.vstack([learner.cumulative_gains for learner in reference.learners]),
                atol=1e-13,
            )
            np.testing.assert_allclose(
                vectorized.meta.log_weights,
                reference.meta.log_weights,
                atol=1e-13,
            )

    def test_booster_predictions_remain_bounded(self) -> None:
        rng = np.random.default_rng(19)
        horizon = 64
        weak = DyadicLinearSecondOrderOracle(dim=4, horizon=horizon)
        booster = StronglyAdaptiveDefensiveBooster(weak=weak, horizon=horizon)
        for _ in range(horizon):
            x = rng.normal(size=4)
            x /= max(float(np.linalg.norm(x)), 1.0)
            y = 1 if rng.random() >= 0.5 else -1
            result = booster.step(x, y)
            self.assertTrue(math.isfinite(result["score"]))
            self.assertLessEqual(abs(result["score"]), 1.0)


if __name__ == "__main__":
    unittest.main()
