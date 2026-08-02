"""Focused tests for the main forecaster and experiment conventions."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from .algorithms import (
    AdaBoostOL,
    DefensiveBooster,
    OnlineBBM,
    OnlineGradientBoosting,
    OnlineSmoothBoost,
)
from .metrics import (
    brier_aggregate_traces,
    hard_core_edge,
    offline_span_diagnostics,
    trace_from_probability_scores,
)
from .real_streams import _hashed_frame, _online_standardize_matrix
from .run import build_algorithms
from .streams import binary_aggregation, group_subset_heterogeneous_margin


class ConstantGainOracle:
    def predict(self, x: np.ndarray) -> float:
        return 0.0

    def update_gain(self, x: np.ndarray, coeff: float) -> None:
        return None


class RootRuleTest(unittest.TestCase):
    def test_prefers_an_available_root(self) -> None:
        self.assertEqual(DefensiveBooster._root(0.0, 0.0), 0.0)
        self.assertAlmostEqual(DefensiveBooster._root(0.2, 0.5), -0.4)

    def test_sign_property(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(1000):
            a, b = rng.uniform(-2.0, 2.0, size=2)
            mu = DefensiveBooster._root(float(a), float(b))
            for sigma in (-1.0, 1.0):
                self.assertLessEqual((a + b * mu) * (sigma - mu), 1e-12)


class BaselineConventionTest(unittest.TestCase):
    def test_ogb_step_size(self) -> None:
        one = OnlineGradientBoosting(ConstantGainOracle, n_learners=1)
        many = OnlineGradientBoosting(ConstantGainOracle, n_learners=100)
        self.assertEqual(one.eta, 1.0)
        self.assertAlmostEqual(many.eta, math.log(100.0) / 100.0)

    def test_synthetic_edge_uses_each_papers_parameterization(self) -> None:
        stream = group_subset_heterogeneous_margin(T=80, low_margin=0.5, seed=0)
        algorithms = build_algorithms(
            stream,
            ogb_learners=2,
            bbm_learners=2,
            adaboost_learners=2,
            osb_learners=2,
            osb_gamma=0.1,
            bbm_gamma=0.1,
            gain_oracle="second_order",
        )
        bbm = next(algo for algo in algorithms if isinstance(algo, OnlineBBM))
        osboost = next(algo for algo in algorithms if isinstance(algo, OnlineSmoothBoost))
        self.assertAlmostEqual(bbm.gamma, stream.classification_advantage_hint)
        self.assertAlmostEqual(osboost.gamma, stream.classification_advantage_hint)

    def test_adaboost_ol_probability_and_updates(self) -> None:
        stream = group_subset_heterogeneous_margin(T=40, low_margin=0.5, seed=2)
        algorithms = build_algorithms(
            stream,
            ogb_learners=2,
            bbm_learners=2,
            adaboost_learners=3,
            osb_learners=2,
            osb_gamma=0.1,
            bbm_gamma=0.1,
            gain_oracle="second_order",
        )
        adaboost = next(algo for algo in algorithms if isinstance(algo, AdaBoostOL))
        first = adaboost.step(stream.X[0], int(stream.y[0]))
        self.assertGreaterEqual(float(first["score"]), -1.0)
        self.assertLessEqual(float(first["score"]), 1.0)
        self.assertTrue(np.all(np.abs(adaboost.alpha) <= 2.0))
        self.assertTrue(np.any(np.abs(adaboost.alpha) > 0.0))

    def test_real_valued_and_classification_edges_are_distinct(self) -> None:
        stream = group_subset_heterogeneous_margin(T=80, low_margin=0.6, seed=0)
        self.assertAlmostEqual(stream.correlation_edge_hint, 0.12)
        self.assertAlmostEqual(stream.classification_advantage_hint, 0.1)


class BrierAggregatorTest(unittest.TestCase):
    def _trace(self, stream, name: str, probabilities: np.ndarray):
        return trace_from_probability_scores(
            stream,
            algorithm=name,
            seed=0,
            probabilities=probabilities,
        )

    def test_can_strictly_outperform_every_component(self) -> None:
        stream = group_subset_heterogeneous_margin(T=80, low_margin=0.5, seed=0)
        zero = self._trace(stream, "zero", np.zeros(stream.T))
        one = self._trace(stream, "one", np.ones(stream.T))
        aggregate = brier_aggregate_traces(
            [zero, one],
            algorithm="aggregate",
            eta=0.5,
        )
        self.assertLess(np.mean(aggregate.brier_losses), 0.5)
        self.assertAlmostEqual(np.mean(zero.brier_losses), 0.5)
        self.assertAlmostEqual(np.mean(one.brier_losses), 0.5)

    def test_mixability_bound(self) -> None:
        stream = group_subset_heterogeneous_margin(T=80, low_margin=0.5, seed=1)
        labels = (stream.y.astype(float) + 1.0) / 2.0
        components = [
            self._trace(stream, "constant-zero", np.zeros(stream.T)),
            self._trace(stream, "constant-half", np.full(stream.T, 0.5)),
            self._trace(stream, "perfect", labels),
        ]
        aggregate = brier_aggregate_traces(
            components,
            algorithm="aggregate",
            eta=0.5,
        )
        aggregate_loss = float(np.sum(aggregate.brier_losses))
        best_loss = min(float(np.sum(trace.brier_losses)) for trace in components)
        self.assertLessEqual(aggregate_loss, best_loss + 2.0 * math.log(3.0) + 1e-12)


class StreamConstructionTest(unittest.TestCase):
    def test_online_standardization_does_not_use_future_rows(self) -> None:
        prefix = np.array([[1.0, 5.0], [2.0, 4.0], [4.0, 3.0], [8.0, 2.0]])
        first = np.vstack([prefix, [[16.0, 1.0]]])
        second = np.vstack([prefix, [[-1000.0, 1000.0]]])
        np.testing.assert_allclose(
            _online_standardize_matrix(first)[: len(prefix)],
            _online_standardize_matrix(second)[: len(prefix)],
        )

    def test_hashed_prefix_does_not_use_future_covariates(self) -> None:
        prefix = pd.DataFrame(
            {"value": [1.0, 2.0, 4.0, 8.0], "kind": ["a", "b", "a", "b"], "label": [0, 1, 0, 1]}
        )
        first = pd.concat(
            [prefix, pd.DataFrame({"value": [16.0], "kind": ["a"], "label": [1]})],
            ignore_index=True,
        )
        second = pd.concat(
            [prefix, pd.DataFrame({"value": [-1000.0], "kind": ["z"], "label": [0]})],
            ignore_index=True,
        )
        X_first, _ = _hashed_frame(
            first,
            label_col="label",
            positive=1,
            numeric_cols=["value"],
            dim=32,
            max_rows=None,
        )
        X_second, _ = _hashed_frame(
            second,
            label_col="label",
            positive=1,
            numeric_cols=["value"],
            dim=32,
            max_rows=None,
        )
        np.testing.assert_allclose(X_first[: len(prefix)], X_second[: len(prefix)])

    def test_group_subset_edge_and_span_obstruction(self) -> None:
        delta = 0.5
        stream = group_subset_heterogeneous_margin(T=400, low_margin=delta, seed=3)
        rng = np.random.default_rng(11)
        for _ in range(50):
            weights = rng.uniform(0.0, 1.0, size=stream.T)
            self.assertGreaterEqual(hard_core_edge(stream, weights), 0.1 - 1e-12)

        diagnostics = offline_span_diagnostics(stream)
        expected = (1.0 - delta) ** 2 / (8.0 * (1.0 + delta**2))
        self.assertAlmostEqual(diagnostics["span_squared_loss"], expected, places=10)

    def test_binary_aggregation_edge_and_span_obstruction(self) -> None:
        stream = binary_aggregation(
            T=400,
            n_pairs=20,
            correct_orientations=12,
            seed=5,
        )
        signed_values = stream.y[:, None] * stream.X
        edge = 0.2

        self.assertTrue(np.all(np.isin(stream.X, [-1.0, 1.0])))
        np.testing.assert_allclose(np.mean(signed_values, axis=1), 0.0, atol=1e-12)
        self.assertTrue(np.all(np.any(signed_values < 0.0, axis=0)))
        uniform_vote = np.where(np.mean(stream.X, axis=1) >= 0.0, 1, -1)
        self.assertAlmostEqual(np.mean(uniform_vote != stream.y), 0.5)

        rng = np.random.default_rng(17)
        for _ in range(50):
            weights = rng.uniform(0.0, 1.0, size=stream.T)
            self.assertGreaterEqual(hard_core_edge(stream, weights), edge - 1e-12)

        diagnostics = offline_span_diagnostics(stream)
        expected = (1.0 - edge) ** 2 / (8.0 * (1.0 + edge**2))
        self.assertAlmostEqual(diagnostics["span_squared_loss"], expected, places=10)


if __name__ == "__main__":
    unittest.main()
