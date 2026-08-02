"""Focused tests for the bounded-regression experiment path."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from .algorithms import DefensiveBooster, OnlineGradientBoosting
from .plots import plot_real_summary
from .regression_streams import _build_stream
from .weak_learners import LinearSecondOrderOracle


class RegressionStreamTest(unittest.TestCase):
    def _toy_stream(self, future_offset: float = 0.0):
        n = 300
        timestamps = pd.date_range("2020-01-01", periods=n, freq="h")
        target = np.linspace(0.0, 10.0, n)
        target[270:] += future_offset
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "target": target,
                "context": np.sin(np.arange(n) / 12.0),
                "kind": np.where(np.arange(n) % 2 == 0, "a", "b"),
            }
        )
        return _build_stream(
            name="toy",
            frame=frame,
            timestamp_col="timestamp",
            target_col="target",
            numeric_cols=["context"],
            categorical_cols=["kind"],
            lag_deltas=[pd.Timedelta(hours=1)],
            dim=32,
            initialization_fraction=0.1,
            target_unit="units",
            target_bounds=None,
            description="toy",
        )

    def test_target_scale_uses_only_initialization_prefix(self) -> None:
        first = self._toy_stream(future_offset=0.0)
        second = self._toy_stream(future_offset=1000.0)
        self.assertEqual(first.target_min, second.target_min)
        self.assertEqual(first.target_max, second.target_max)
        np.testing.assert_allclose(first.X_init, second.X_init)
        np.testing.assert_allclose(first.y_init, second.y_init)

    def test_continuous_outcomes_run_through_both_algorithms(self) -> None:
        stream = self._toy_stream()

        def factory() -> LinearSecondOrderOracle:
            return LinearSecondOrderOracle(dim=stream.dim)

        algorithms = [
            DefensiveBooster(factory),
            OnlineGradientBoosting(factory, n_learners=3),
        ]
        for algorithm in algorithms:
            scores = []
            for x_t, y_t in zip(stream.X[:20], stream.y[:20]):
                result = algorithm.step(x_t, 2.0 * float(y_t) - 1.0)
                scores.append(float(result["score"]))
            self.assertTrue(np.all(np.isfinite(scores)))
            self.assertTrue(np.all(np.abs(scores) <= 1.0))

    def test_combined_introductory_figure_renders(self) -> None:
        reference_dir = Path(__file__).resolve().parent / "reference"
        with (reference_dir / "real.json").open() as handle:
            real = json.load(handle)
        with (reference_dir / "regression.json").open() as handle:
            regression = json.load(handle)
        with tempfile.TemporaryDirectory() as directory:
            path = plot_real_summary(real, Path(directory), regression)
            self.assertIsNotNone(path)
            assert path is not None
            self.assertGreater(path.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
