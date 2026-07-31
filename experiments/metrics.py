"""Metrics and offline diagnostics for online boosting experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .algorithms import OnlineAlgorithm
from .streams import StreamData


def _row_dot(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.einsum("ij,j->i", X, beta, optimize=True)


def _weighted_feature_sum(weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.einsum("i,ij->j", weights, X, optimize=True)


@dataclass
class RunTrace:
    stream: StreamData
    algorithm: str
    seed: int
    scores: np.ndarray
    predictions: np.ndarray
    mistakes: np.ndarray
    brier_losses: np.ndarray
    randomized_errors: np.ndarray
    weak_residual_corrs: np.ndarray
    self_residual_corrs: np.ndarray
    hard_core_edges: np.ndarray
    hard_core_densities: np.ndarray
    auxiliary: Dict[str, np.ndarray] = field(default_factory=dict)


def run_online_algorithm(algo: OnlineAlgorithm, stream: StreamData, seed: int) -> RunTrace:
    scores: List[float] = []
    predictions: List[float] = []
    mistakes: List[float] = []
    brier_losses: List[float] = []
    randomized_errors: List[float] = []
    auxiliary: Dict[str, List[float]] = {}
    for x_t, y_t in zip(stream.X, stream.y):
        result = algo.step(x_t, int(y_t))
        scores.append(result["score"])
        predictions.append(result["prediction"])
        mistakes.append(result["mistake"])
        brier_losses.append(result["brier_loss"])
        randomized_errors.append(result["randomized_error"])
        for key, value in result.get("extra", {}).items():
            auxiliary.setdefault(key, []).append(float(value))
    scores_arr = np.asarray(scores, dtype=float)
    cert = certificate_curves(stream, scores_arr)
    return RunTrace(
        stream=stream,
        algorithm=algo.name,
        seed=seed,
        scores=scores_arr,
        predictions=np.asarray(predictions, dtype=float),
        mistakes=np.asarray(mistakes, dtype=float),
        brier_losses=np.asarray(brier_losses, dtype=float),
        randomized_errors=np.asarray(randomized_errors, dtype=float),
        weak_residual_corrs=cert["weak_residual_corr"],
        self_residual_corrs=cert["self_residual_corr"],
        hard_core_edges=cert["hard_core_edge"],
        hard_core_densities=cert["hard_core_density"],
        auxiliary={key: np.asarray(values, dtype=float) for key, values in auxiliary.items()},
    )


def trace_from_probability_scores(
    stream: StreamData,
    *,
    algorithm: str,
    seed: int,
    probabilities: np.ndarray,
    auxiliary: Dict[str, np.ndarray] | None = None,
) -> RunTrace:
    """Build a trace from causal probability forecasts in [0,1]."""

    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != (stream.T,):
        raise ValueError(f"expected {stream.T} probabilities, got {probs.shape}")
    if np.any((probs < -1e-12) | (probs > 1.0 + 1e-12)):
        raise ValueError("probabilities must lie in [0,1]")
    probs = np.clip(probs, 0.0, 1.0)
    scores = 2.0 * probs - 1.0
    predictions = np.where(scores >= 0.0, 1.0, -1.0)
    labels = (stream.y.astype(float) + 1.0) / 2.0
    cert = certificate_curves(stream, scores)
    return RunTrace(
        stream=stream,
        algorithm=algorithm,
        seed=seed,
        scores=scores,
        predictions=predictions,
        mistakes=(predictions != stream.y).astype(float),
        brier_losses=(labels - probs) ** 2,
        randomized_errors=np.abs(labels - probs),
        weak_residual_corrs=cert["weak_residual_corr"],
        self_residual_corrs=cert["self_residual_corr"],
        hard_core_edges=cert["hard_core_edge"],
        hard_core_densities=cert["hard_core_density"],
        auxiliary={} if auxiliary is None else auxiliary,
    )


def brier_aggregate_traces(
    component_traces: Sequence[RunTrace],
    *,
    algorithm: str,
    eta: float = 0.5,
) -> RunTrace:
    """Causally aggregate probability forecasts by exponential Brier weights.

    For eta <= 1/2, the weighted-average forecast has cumulative Brier loss at
    most the best component's loss plus log(K)/eta.
    """

    if not component_traces:
        raise ValueError("at least one component trace is required")
    if not 0.0 < eta <= 0.5:
        raise ValueError("the Brier aggregation guarantee requires eta in (0, 1/2]")
    first = component_traces[0]
    if any(trace.stream.name != first.stream.name for trace in component_traces):
        raise ValueError("all component traces must use the same stream")
    if any(trace.seed != first.seed for trace in component_traces):
        raise ValueError("all component traces must use the same seed")

    component_probs = np.vstack([(trace.scores + 1.0) / 2.0 for trace in component_traces])
    labels = (first.stream.y.astype(float) + 1.0) / 2.0
    log_weights = np.zeros(len(component_traces), dtype=float)
    forecasts = np.empty(first.stream.T, dtype=float)
    weights_history = np.empty((first.stream.T, len(component_traces)), dtype=float)
    for t in range(first.stream.T):
        stabilized = log_weights - np.max(log_weights)
        weights = np.exp(stabilized)
        weights /= np.sum(weights)
        weights_history[t] = weights
        forecasts[t] = float(weights @ component_probs[:, t])
        log_weights -= eta * (labels[t] - component_probs[:, t]) ** 2

    return trace_from_probability_scores(
        first.stream,
        algorithm=algorithm,
        seed=first.seed,
        probabilities=forecasts,
        auxiliary={"component_weights": weights_history},
    )


def summarize_trace(trace: RunTrace) -> Dict[str, float | str | int]:
    stream = trace.stream
    weights = (1.0 - stream.y.astype(float) * trace.scores) / 2.0
    edge = hard_core_edge(stream, weights)
    comparator = offline_span_diagnostics(stream)
    labels = (stream.y.astype(float) + 1.0) / 2.0
    prevalence = float(np.mean(labels))
    constant_brier = float(np.mean((labels - prevalence) ** 2))
    return {
        "stream": stream.name,
        "weak_type": stream.weak_type,
        "algorithm": trace.algorithm,
        "seed": trace.seed,
        "T": stream.T,
        "positive_rate": prevalence,
        "classification_error": float(np.mean(trace.mistakes)),
        "brier_loss": float(np.mean(trace.brier_losses)),
        "constant_brier_loss": constant_brier,
        "randomized_error": float(np.mean(trace.randomized_errors)),
        "hard_core_density": float(np.mean(weights)),
        "hard_core_edge": float(edge),
        "weak_residual_corr": float(trace.weak_residual_corrs[-1]),
        "self_residual_corr": float(trace.self_residual_corrs[-1]),
        "offline_span_squared_loss": comparator["span_squared_loss"],
        "offline_span_representation_norm": comparator["span_representation_norm"],
        "offline_span_classification_error": comparator["span_classification_error"],
        "max_single_edge": comparator["max_single_edge"],
    }


def certificate_curves(stream: StreamData, scores: np.ndarray) -> Dict[str, np.ndarray]:
    y = stream.y.astype(float)
    residual = y - scores
    rounds = np.arange(1, stream.T + 1, dtype=float)

    self_residual_corr = np.abs(np.cumsum(scores * residual)) / rounds

    weak_prefix = np.cumsum(residual[:, None] * stream.X, axis=0)
    weights = (1.0 - y * scores) / 2.0
    density = np.cumsum(weights) / rounds
    hard_core_prefix = np.cumsum((weights * y)[:, None] * stream.X, axis=0)
    denom = np.cumsum(weights)

    if stream.weak_type == "finite":
        weak_residual_corr = np.max(np.abs(weak_prefix), axis=1) / rounds
        hard_core_num = np.max(np.abs(hard_core_prefix), axis=1)
    elif stream.weak_type == "linear":
        weak_residual_corr = np.linalg.norm(weak_prefix, axis=1) / rounds
        hard_core_num = np.linalg.norm(hard_core_prefix, axis=1)
    else:
        raise ValueError(f"unknown weak_type: {stream.weak_type}")

    hard_core_edge = np.divide(
        hard_core_num,
        denom,
        out=np.zeros_like(hard_core_num, dtype=float),
        where=denom > 1e-12,
    )
    return {
        "weak_residual_corr": weak_residual_corr,
        "self_residual_corr": self_residual_corr,
        "hard_core_edge": hard_core_edge,
        "hard_core_density": density,
    }


def hard_core_edge(stream: StreamData, weights: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if denom <= 1e-12:
        return 0.0
    signed = weights * stream.y.astype(float)
    if stream.weak_type == "finite":
        corr = _weighted_feature_sum(signed, stream.X)
        return float(np.max(np.abs(corr)) / denom)
    if stream.weak_type == "linear":
        vec = _weighted_feature_sum(signed, stream.X)
        return float(np.linalg.norm(vec) / denom)
    raise ValueError(f"unknown weak_type: {stream.weak_type}")


def offline_span_diagnostics(stream: StreamData) -> Dict[str, float]:
    X = stream.X
    y = stream.y.astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    scores = _row_dot(X, beta)
    preds = np.where(scores >= 0.0, 1.0, -1.0)
    if stream.weak_type == "finite":
        representation_norm = float(np.linalg.norm(beta, ord=1))
    elif stream.weak_type == "linear":
        representation_norm = float(np.linalg.norm(beta, ord=2))
    else:
        raise ValueError(f"unknown weak_type: {stream.weak_type}")
    if stream.weak_type == "finite":
        single_edges = np.abs(_weighted_feature_sum(y, X)) / stream.T
        max_single_edge = float(np.max(single_edges))
    else:
        max_single_edge = float(np.linalg.norm(_weighted_feature_sum(y, X)) / stream.T)
    return {
        "span_squared_loss": float(np.mean((y - scores) ** 2) / 4.0),
        "span_representation_norm": representation_norm,
        "span_classification_error": float(np.mean(preds != y)),
        "max_single_edge": max_single_edge,
    }


def cumulative(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return np.cumsum(arr) / np.arange(1, arr.size + 1)
