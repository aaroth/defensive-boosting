"""Synthetic binary streams for the boosting experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List
import itertools

import numpy as np


@dataclass
class StreamData:
    name: str
    X: np.ndarray
    y: np.ndarray
    weak_type: str
    description: str
    correlation_edge_hint: float | None = None
    classification_advantage_hint: float | None = None

    @property
    def T(self) -> int:
        return int(self.y.shape[0])

    @property
    def dim(self) -> int:
        return int(self.X.shape[1])


def _row_dot(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Stable row-wise dot product that avoids noisy BLAS warning flags."""

    return np.einsum("ij,j->i", X, beta, optimize=True)


def heterogeneous_margin(T: int, gamma: float = 0.12, seed: int = 0) -> StreamData:
    """Intro separation example: sign(h) is perfect but scalar span loss is large."""

    rng = np.random.default_rng(seed)
    y = rng.choice(np.array([-1, 1], dtype=int), size=T)
    mask = rng.random(T) < 0.5
    s = np.where(mask, 1.0, gamma)
    X = (s * y).reshape(T, 1)
    return StreamData(
        name=f"heterogeneous_margin_gamma={gamma:g}",
        X=X,
        y=y,
        weak_type="finite",
        correlation_edge_hint=gamma,
        classification_advantage_hint=0.1,
        description=(
            "One finite weak feature has the correct sign on every round but "
            "heterogeneous magnitude. Smooth weak learning holds, while the "
            "one-dimensional real span has constant squared-loss error for "
            "small gamma."
        ),
    )


def group_subset_positive(
    T: int,
    n_groups: int = 10,
    subset_size: int = 6,
    seed: int = 0,
) -> StreamData:
    """Large finite-class regime where smooth weak learning holds.

    Abstractly, each context is a group and an observed sign feature, and the
    label equals the sign.  There is one weak hypothesis for each subset A of
    `subset_size` groups.  On groups in A it returns the sign; outside A it
    returns the opposite sign.  StreamData.X stores the resulting vector of
    weak-hypothesis values.  For any reweighting, the subset containing the
    heaviest `subset_size` groups has edge at least 2*subset_size/n_groups - 1.
    """

    if not (0 < subset_size < n_groups):
        raise ValueError("subset_size must be between 1 and n_groups - 1")
    if 2 * subset_size <= n_groups:
        raise ValueError("subset_size must be larger than n_groups/2 for positive edge")

    rng = np.random.default_rng(seed)
    groups = rng.integers(0, n_groups, size=T)
    y = rng.choice(np.array([-1, 1], dtype=int), size=T)
    subsets = list(itertools.combinations(range(n_groups), subset_size))
    membership = np.zeros((n_groups, len(subsets)), dtype=float)
    for j, subset in enumerate(subsets):
        membership[list(subset), j] = 1.0
    weak_values_by_group = 2.0 * membership - 1.0
    X = weak_values_by_group[groups] * y[:, None]
    edge = 2.0 * subset_size / n_groups - 1.0
    return StreamData(
        name=f"group_subset_positive_m={n_groups}_k={subset_size}",
        X=X,
        y=y,
        weak_type="finite",
        correlation_edge_hint=edge,
        classification_advantage_hint=edge / 2.0,
        description=(
            "Each context consists abstractly of a group and an observed sign "
            "feature equal to the label. A large finite weak class satisfies "
            "smooth weak learning, but the useful group-subset rule changes "
            "with the mistake distribution."
        ),
    )


def group_subset_heterogeneous_margin(
    T: int,
    n_groups: int = 10,
    subset_size: int = 6,
    low_margin: float = 0.5,
    seed: int = 0,
) -> StreamData:
    """Aggregation regime separating sign boosting from span prediction.

    Each context contains a group, an observed sign, and a magnitude in
    {low_margin, 1}.  The label is the sign.  For every subset A of the given
    size, one weak rule returns magnitude times the sign on A and its negation
    off A.  The best subset depends on the reweighting, and every reweighting
    has correlation edge at least
    low_margin * (2 * subset_size / n_groups - 1).  The heterogeneous
    magnitudes prevent one affine span score from fitting both magnitude
    levels exactly.
    """

    if not (0 < subset_size < n_groups):
        raise ValueError("subset_size must be between 1 and n_groups - 1")
    if 2 * subset_size <= n_groups:
        raise ValueError("subset_size must be larger than n_groups/2 for positive edge")
    if not 0.0 < low_margin < 1.0:
        raise ValueError("low_margin must lie in (0, 1)")
    if T % (4 * n_groups) != 0:
        raise ValueError("T must be divisible by 4 * n_groups for exact balance")

    rng = np.random.default_rng(seed)
    groups = np.arange(T, dtype=int) % n_groups
    signs = np.where((np.arange(T) // n_groups) % 2 == 0, 1, -1).astype(int)
    magnitudes = np.where((np.arange(T) // (2 * n_groups)) % 2 == 0, 1.0, low_margin)
    order = rng.permutation(T)
    groups = groups[order]
    y = signs[order]
    magnitudes = magnitudes[order]

    subsets = list(itertools.combinations(range(n_groups), subset_size))
    membership = np.zeros((n_groups, len(subsets)), dtype=float)
    for j, subset in enumerate(subsets):
        membership[list(subset), j] = 1.0
    orientations = 2.0 * membership - 1.0
    X = magnitudes[:, None] * orientations[groups] * y[:, None]
    edge = low_margin * (2.0 * subset_size / n_groups - 1.0)
    return StreamData(
        name=(
            f"group_subset_heterogeneous_m={n_groups}_k={subset_size}"
            f"_delta={low_margin:g}"
        ),
        X=X,
        y=y,
        weak_type="finite",
        correlation_edge_hint=edge,
        classification_advantage_hint=(2.0 * subset_size / n_groups - 1.0) / 2.0,
        description=(
            "The latent generator uses a group, a sign, and a magnitude in "
            "{delta, 1}; algorithms receive only the resulting vector of weak-rule "
            "values. A large finite weak class "
            "satisfies smooth weak learning, but the useful group-subset rule "
            "changes with the reweighting and heterogeneous magnitudes obstruct "
            "a single accurate affine span score."
        ),
    )


def planted_decoy_margin(
    T: int,
    d: int = 200,
    gamma: float = 0.12,
    seed: int = 0,
) -> StreamData:
    """Weak-to-strong favorable regime with one useful rule among decoys.

    The first weak feature always has the correct sign, but with heterogeneous
    margin in {gamma,1}; the remaining weak features are independent decoys.
    Thus smooth weak learning holds at edge gamma, the class is much larger than
    the one-rule sanity check, and sign-based boosting has an advantage over
    squared-loss span fitting.
    """

    if d < 2:
        raise ValueError("d must be at least 2")
    rng = np.random.default_rng(seed)
    y = rng.choice(np.array([-1, 1], dtype=int), size=T)
    scale = np.where(rng.random(T) < 0.5, 1.0, gamma)
    X = np.empty((T, d), dtype=float)
    X[:, 0] = scale * y
    X[:, 1:] = rng.choice(np.array([-1.0, 1.0]), size=(T, d - 1))
    return StreamData(
        name=f"planted_decoy_margin_d={d}_gamma={gamma:g}",
        X=X,
        y=y,
        weak_type="finite",
        correlation_edge_hint=gamma,
        classification_advantage_hint=0.1,
        description=(
            "A large finite class contains one sign-perfect heterogeneous-margin "
            "weak rule and many decoys.  Smooth weak learning holds, while the "
            "real-valued span view is penalized by the weak rule's small margins."
        ),
    )


def many_weak_features(
    T: int,
    d: int = 80,
    margin_noise: float = 0.15,
    seed: int = 0,
) -> StreamData:
    """Span-regret regime: many tiny coordinates combine into a good rule."""

    rng = np.random.default_rng(seed)
    X = rng.choice(np.array([-1.0, 1.0]), size=(T, d))
    beta = rng.normal(size=d)
    beta /= np.linalg.norm(beta)
    raw_margin = _row_dot(X, beta) / np.sqrt(d)
    noise = margin_noise * rng.normal(size=T)
    y = np.where(raw_margin + noise >= 0.0, 1, -1).astype(int)
    return StreamData(
        name=f"many_weak_features_d={d}",
        X=X,
        y=y,
        weak_type="finite",
        correlation_edge_hint=None,
        classification_advantage_hint=None,
        description=(
            "Each coordinate has only small edge, so a constant-edge smooth weak "
            "learning condition is a poor description. A linear combination of "
            "coordinates is predictive, testing the squared-loss span guarantee."
        ),
    )


def linear_span_fallback(
    T: int,
    d: int = 40,
    margin_noise: float = 0.02,
    seed: int = 0,
) -> StreamData:
    """Infinite-linear-class regime: the span is predictive but smooth edge fails.

    The contexts are normalized, so any fixed unit vector has small raw margin
    on many examples.  Smooth reweightings can focus on the near-margin rounds,
    where no constant edge is available.  A scaled linear predictor is still a
    good span comparator.
    """

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, d))
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    beta = rng.normal(size=d)
    beta /= np.linalg.norm(beta)
    margin = _row_dot(X, beta)
    y = np.where(margin + margin_noise * rng.normal(size=T) >= 0.0, 1, -1).astype(int)
    return StreamData(
        name=f"linear_span_fallback_d={d}",
        X=X,
        y=y,
        weak_type="linear",
        correlation_edge_hint=None,
        classification_advantage_hint=None,
        description=(
            "The weak class is the infinite Euclidean linear ball. Smooth weak "
            "learning with a constant edge fails on near-margin subsets, "
            "but a scaled linear span predictor is highly accurate."
        ),
    )


def mixed_linear_random_label_mixture(
    T: int,
    d: int = 30,
    random_fraction: float = 0.35,
    margin_noise: float = 0.05,
    seed: int = 0,
) -> StreamData:
    """Linear-span regime with independently randomized labels."""

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, d))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)
    beta = rng.normal(size=d)
    beta /= np.linalg.norm(beta)
    structured_y = np.where(_row_dot(X, beta) + margin_noise * rng.normal(size=T) >= 0.0, 1, -1)
    random_y = rng.choice(np.array([-1, 1], dtype=int), size=T)
    random_mask = rng.random(T) < random_fraction
    y = np.where(random_mask, random_y, structured_y).astype(int)
    return StreamData(
        name=f"mixed_linear_random_label_mixture_p={random_fraction:g}",
        X=X,
        y=y,
        weak_type="linear",
        correlation_edge_hint=None,
        classification_advantage_hint=None,
        description=(
            "Labels are independently randomized on a fixed fraction of rounds. "
            "Those rounds give a smooth low-edge weighting, while the best linear "
            "span predictor still captures the structured component."
        ),
    )
def random_labels(T: int, d: int = 30, seed: int = 0) -> StreamData:
    """Negative control: neither smooth weak learning nor span prediction helps."""

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, d))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)
    y = rng.choice(np.array([-1, 1], dtype=int), size=T)
    return StreamData(
        name="random_labels",
        X=X,
        y=y,
        weak_type="linear",
        correlation_edge_hint=None,
        classification_advantage_hint=None,
        description=(
            "Pure random labels. This is a sanity check: the hard-core edge "
            "certificate should be small and no method should beat chance by much."
        ),
    )


def default_streams(T: int, seed: int) -> List[StreamData]:
    return [
        planted_decoy_margin(T=T, d=200, gamma=0.12, seed=seed),
        group_subset_heterogeneous_margin(
            T=T,
            n_groups=10,
            subset_size=6,
            low_margin=0.6,
            seed=seed,
        ),
        linear_span_fallback(T=T, d=40, margin_noise=0.02, seed=seed),
        mixed_linear_random_label_mixture(T=T, d=30, random_fraction=0.35, seed=seed),
        random_labels(T=T, d=30, seed=seed),
    ]


STREAM_BUILDERS: Dict[str, Callable[..., StreamData]] = {
    "heterogeneous_margin": heterogeneous_margin,
    "group_subset_positive": group_subset_positive,
    "group_subset_heterogeneous_margin": group_subset_heterogeneous_margin,
    "planted_decoy_margin": planted_decoy_margin,
    "many_weak_features": many_weak_features,
    "linear_span_fallback": linear_span_fallback,
    "mixed_linear_random_label_mixture": mixed_linear_random_label_mixture,
    "random_labels": random_labels,
}
