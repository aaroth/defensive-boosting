"""Online weak learners and small optimization helpers.

The experiments use two weak-class models.

* Finite classes: the context vector is interpreted as the vector of weak
  predictions h_j(x).  We symmetrize internally by adding -h_j.
* Linear balls: the weak class is {x -> <u,x> : ||u||_2 <= radius}; online
  projected gradient ascent implements the linear-loss oracle.

The classes below are intentionally lightweight.  They expose the operations
needed by the three boosters without depending on a particular dataset format.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np


def sign_pm(value: float) -> int:
    """Return the paper's deterministic sign convention in {-1, 1}."""

    return 1 if value >= 0 else -1


def clip_unit(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def project_l2_ball(w: np.ndarray, radius: float) -> np.ndarray:
    norm = float(np.linalg.norm(w))
    if norm > radius and norm > 0:
        return w * (radius / norm)
    return w


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""

    if v.size == 1:
        return np.ones_like(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u * np.arange(1, v.size + 1) > (cssv - 1.0)
    if not np.any(rho_candidates):
        return np.ones_like(v) / v.size
    rho = int(np.nonzero(rho_candidates)[0][-1])
    theta = (cssv[rho] - 1.0) / (rho + 1)
    return np.maximum(v - theta, 0.0)


class LinearGainOracle(Protocol):
    def predict(self, x: np.ndarray) -> float:
        ...

    def update_gain(self, x: np.ndarray, coeff: float) -> None:
        ...


class WeightedClassifier(Protocol):
    def predict(self, x: np.ndarray) -> int:
        ...

    def update_weighted(self, x: np.ndarray, y: int, weight: float) -> None:
        ...


@dataclass
class HedgeGains:
    """Exponentiated weights for maximizing expert gains."""

    n: int
    eta: float = 0.25

    def __post_init__(self) -> None:
        self.logw = np.zeros(self.n, dtype=float)

    @property
    def probs(self) -> np.ndarray:
        shifted = self.logw - np.max(self.logw)
        w = np.exp(shifted)
        return w / np.sum(w)

    def update(self, gains: np.ndarray) -> None:
        gains = np.asarray(gains, dtype=float)
        self.logw += self.eta * np.clip(gains, -4.0, 4.0)
        self.logw -= np.max(self.logw)


@dataclass
class FiniteHedgeLinearOracle:
    """First-order linear-loss oracle retained for ablation comparisons."""

    base_dim: int
    eta: float = 0.2

    def __post_init__(self) -> None:
        self.hedge = HedgeGains(2 * self.base_dim, eta=self.eta)

    def values(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.concatenate([x, -x])

    def predict(self, x: np.ndarray) -> float:
        return clip_unit(float(self.hedge.probs @ self.values(x)))

    def update_gain(self, x: np.ndarray, coeff: float) -> None:
        self.hedge.update(float(coeff) * self.values(x))


@dataclass
class FiniteSecondOrderLinearOracle:
    """Entropy-FTRL oracle with a coefficient-energy adaptive rate.

    On round t the learning rate is proportional to
    sqrt(log(n) / (V_0 + sum_{s<t} c_s^2)).  Standard adaptive-FTRL analysis
    gives regret O(sqrt(log(n) sum_t c_t^2) + log(n)) because every expert
    gain has magnitude at most |c_t|.
    """

    base_dim: int
    initial_variance: float = 4.0
    max_eta: float = 0.25

    def __post_init__(self) -> None:
        if self.base_dim < 1:
            raise ValueError("base_dim must be positive")
        if self.initial_variance <= 0.0:
            raise ValueError("initial_variance must be positive")
        self.n = 2 * self.base_dim
        self.cumulative_gains = np.zeros(self.n, dtype=float)
        self.variance = float(self.initial_variance)
        self.log_n = math.log(self.n)

    def values(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.concatenate([x, -x])

    @property
    def eta(self) -> float:
        return min(self.max_eta, math.sqrt(self.log_n / self.variance))

    @property
    def probs(self) -> np.ndarray:
        logits = self.eta * self.cumulative_gains
        logits -= np.max(logits)
        weights = np.exp(logits)
        return weights / np.sum(weights)

    def predict(self, x: np.ndarray) -> float:
        return clip_unit(float(self.probs @ self.values(x)))

    def update_gain(self, x: np.ndarray, coeff: float) -> None:
        coeff = float(coeff)
        if abs(coeff) > 2.0 + 1e-12:
            raise ValueError("FiniteSecondOrderLinearOracle requires coefficients in [-2, 2]")
        self.cumulative_gains += coeff * self.values(x)
        self.variance += coeff * coeff


@dataclass
class FiniteHedgeClassifier:
    """Weighted online classifier over signs of a finite class."""

    base_dim: int
    eta: float = 0.5
    symmetric: bool = False

    def __post_init__(self) -> None:
        n_experts = 2 * self.base_dim if self.symmetric else self.base_dim
        self.logw = np.zeros(n_experts, dtype=float)

    def signed_values(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        vals = np.concatenate([x, -x]) if self.symmetric else x
        return np.where(vals >= 0.0, 1.0, -1.0)

    @property
    def probs(self) -> np.ndarray:
        shifted = self.logw - np.max(self.logw)
        w = np.exp(shifted)
        return w / np.sum(w)

    def predict(self, x: np.ndarray) -> int:
        return sign_pm(float(self.probs @ self.signed_values(x)))

    def update_weighted(self, x: np.ndarray, y: int, weight: float) -> None:
        if weight <= 0:
            return
        preds = self.signed_values(x)
        losses = (preds != y).astype(float)
        self.logw -= self.eta * float(weight) * losses
        self.logw -= np.max(self.logw)


@dataclass
class LinearBallOracle:
    """First-order projected-gradient oracle retained for ablations."""

    dim: int
    radius: float = 1.0
    eta: float = 0.6

    def __post_init__(self) -> None:
        self.w = np.zeros(self.dim, dtype=float)
        self.t = 0

    def predict(self, x: np.ndarray) -> float:
        return clip_unit(float(self.w @ x))

    def update_gain(self, x: np.ndarray, coeff: float) -> None:
        self.t += 1
        lr = self.eta / math.sqrt(self.t)
        self.w = project_l2_ball(self.w + lr * float(coeff) * x, self.radius)


@dataclass
class LinearSecondOrderOracle:
    """Adaptive projected OGD over a Euclidean ball.

    The step size is based on cumulative squared gradient norm, yielding
    O(radius * sqrt(sum_t ||c_t x_t||_2^2)) regret up to an additive constant.
    """

    dim: int
    radius: float = 1.0
    initial_variance: float = 4.0
    eta_scale: float | None = None

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("dim must be positive")
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        if self.initial_variance <= 0.0:
            raise ValueError("initial_variance must be positive")
        self.w = np.zeros(self.dim, dtype=float)
        self.variance = float(self.initial_variance)
        if self.eta_scale is None:
            self.eta_scale = 0.5 * self.radius

    def predict(self, x: np.ndarray) -> float:
        return clip_unit(float(self.w @ x))

    def update_gain(self, x: np.ndarray, coeff: float) -> None:
        coeff = float(coeff)
        if abs(coeff) > 2.0 + 1e-12:
            raise ValueError("LinearSecondOrderOracle requires coefficients in [-2, 2]")
        x = np.asarray(x, dtype=float)
        gradient = coeff * x
        lr = float(self.eta_scale) / math.sqrt(self.variance)
        self.w = project_l2_ball(self.w + lr * gradient, self.radius)
        self.variance += float(gradient @ gradient)


@dataclass
class LinearPerceptronClassifier:
    """Weighted projected perceptron used by OSBoost on linear classes."""

    dim: int
    radius: float = 1.0
    eta: float = 1.0
    margin: float = 1.0

    def __post_init__(self) -> None:
        self.w = np.zeros(self.dim, dtype=float)
        self.t = 0

    def predict(self, x: np.ndarray) -> int:
        return sign_pm(float(self.w @ x))

    def update_weighted(self, x: np.ndarray, y: int, weight: float) -> None:
        if weight <= 0:
            return
        self.t += 1
        score = float(y) * float(self.w @ x)
        if score < self.margin:
            lr = self.eta * float(weight) / math.sqrt(self.t)
            self.w = project_l2_ball(self.w + lr * float(y) * x, self.radius)


class ScalarAdaptiveOGD:
    """Projected one-dimensional adaptive OGD from the paper."""

    def __init__(self) -> None:
        self.value = 0.0
        self.variance = 4.0

    def predict(self) -> float:
        return self.value

    def update_gain(self, coeff: float) -> None:
        coeff = float(coeff)
        if abs(coeff) > 2.0 + 1e-12:
            raise ValueError("ScalarAdaptiveOGD requires coefficients in [-2, 2]")
        self.value = clip_unit(self.value + coeff / math.sqrt(self.variance))
        self.variance += coeff * coeff


class DyadicAdaptMLProd:
    """Adapt-ML-Prod aggregation for the canonical dyadic specialists.

    This is Algorithm 2 and Corollary 4 of Gaillard, Stoltz, and van Erven
    (2014), combined with their confidence reduction.  At every round exactly
    one interval at each dyadic scale is awake.  Sleeping specialists have zero
    modified excess loss, so they need not be represented explicitly.

    Linear gains are mapped to losses by ell(z) = (2 - c z) / 4.  Therefore an
    awake specialist's modified excess loss is c (z_j - z_mix) / 4 in [-1, 1].
    The uniform prior is over the complete dyadic family after padding the
    horizon to a power of two, including specialists that never wake before T.
    """

    def __init__(self, horizon: int) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        self.padded_horizon = 1 << (self.horizon - 1).bit_length()
        self.lengths = 1 << np.arange(self.padded_horizon.bit_length(), dtype=np.int64)
        self.num_total_specialists = 2 * self.padded_horizon - 1
        self.log_num_specialists = math.log(self.num_total_specialists)
        self.log_prior = -self.log_num_specialists
        self.log_weights = np.full(self.lengths.size, self.log_prior, dtype=float)
        self.excess_variances = np.zeros(self.lengths.size, dtype=float)
        self.round = 0
        self.last_reset_mask = np.ones(self.lengths.size, dtype=bool)
        self.last_mixture_weights = np.ones(self.lengths.size, dtype=float) / self.lengths.size
        self._predictions: np.ndarray | None = None
        self._mixture = 0.0
        self._learning_rates: np.ndarray | None = None

    @property
    def num_active_specialists(self) -> int:
        return int(self.lengths.size)

    def _rates(self, variances: np.ndarray) -> np.ndarray:
        if self.num_total_specialists == 1:
            return np.full_like(variances, 0.5)
        return np.minimum(0.5, np.sqrt(self.log_num_specialists / (1.0 + variances)))

    def start_round(self) -> np.ndarray:
        if self._predictions is not None:
            raise RuntimeError("previous dyadic round was not updated")
        self.round += 1
        if self.round > self.horizon:
            raise RuntimeError("dyadic wrapper used beyond its declared horizon")
        self.last_reset_mask = ((self.round - 1) % self.lengths) == 0
        self.log_weights[self.last_reset_mask] = self.log_prior
        self.excess_variances[self.last_reset_mask] = 0.0
        return self.last_reset_mask

    def aggregate(self, predictions: np.ndarray) -> float:
        predictions = np.asarray(predictions, dtype=float)
        if predictions.shape != self.log_weights.shape:
            raise ValueError("one prediction is required for every active dyadic scale")
        if np.any(np.abs(predictions) > 1.0 + 1e-12):
            raise ValueError("dyadic specialist predictions must lie in [-1, 1]")
        learning_rates = self._rates(self.excess_variances)
        log_masses = np.log(learning_rates) + self.log_weights
        log_masses -= np.max(log_masses)
        masses = np.exp(log_masses)
        mixture_weights = masses / np.sum(masses)
        mixture = clip_unit(float(mixture_weights @ predictions))
        self.last_mixture_weights = mixture_weights
        self._predictions = predictions.copy()
        self._mixture = mixture
        self._learning_rates = learning_rates
        return mixture

    def update(self, coeff: float) -> None:
        coeff = float(coeff)
        if abs(coeff) > 2.0 + 1e-12:
            raise ValueError("dyadic Adapt-ML-Prod requires coefficients in [-2, 2]")
        if self._predictions is None or self._learning_rates is None:
            raise RuntimeError("aggregate must be called before update")

        excess_losses = np.clip(
            coeff * (self._predictions - self._mixture) / 4.0,
            -1.0,
            1.0,
        )
        new_variances = self.excess_variances + excess_losses * excess_losses
        new_rates = self._rates(new_variances)
        update_factors = 1.0 + self._learning_rates * excess_losses
        if np.any(update_factors <= 0.0):
            raise RuntimeError("invalid Adapt-ML-Prod multiplicative update")
        self.log_weights = (new_rates / self._learning_rates) * (
            self.log_weights + np.log(update_factors)
        )
        self.excess_variances = new_variances
        self._predictions = None
        self._learning_rates = None


@dataclass
class DyadicLinearSecondOrderOracle:
    """Vectorized dyadic copies of ``LinearSecondOrderOracle``."""

    dim: int
    horizon: int
    radius: float = 1.0
    initial_variance: float = 4.0
    eta_scale: float | None = None

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("dim must be positive")
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        if self.initial_variance <= 0.0:
            raise ValueError("initial_variance must be positive")
        if self.eta_scale is None:
            self.eta_scale = 0.5 * self.radius
        self.meta = DyadicAdaptMLProd(self.horizon)
        copies = self.meta.num_active_specialists
        self.w = np.zeros((copies, self.dim), dtype=float)
        self.variance = np.full(copies, self.initial_variance, dtype=float)
        self._x: np.ndarray | None = None

    def predict(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,):
            raise ValueError(f"expected context shape {(self.dim,)}, got {x.shape}")
        reset = self.meta.start_round()
        self.w[reset] = 0.0
        self.variance[reset] = self.initial_variance
        predictions = np.clip(
            np.sum(self.w * x[None, :], axis=1),
            -1.0,
            1.0,
        )
        self._x = x
        return self.meta.aggregate(predictions)

    def update_gain(self, x: np.ndarray, coeff: float) -> None:
        if self._x is None:
            raise RuntimeError("predict must be called before update_gain")
        x = np.asarray(x, dtype=float)
        if not np.array_equal(x, self._x):
            raise ValueError("update context must match the prediction context")
        coeff = float(coeff)
        if abs(coeff) > 2.0 + 1e-12:
            raise ValueError("DyadicLinearSecondOrderOracle requires coefficients in [-2, 2]")
        gradient = coeff * x
        learning_rates = float(self.eta_scale) / np.sqrt(self.variance)
        self.w += learning_rates[:, None] * gradient[None, :]
        norms = np.linalg.norm(self.w, axis=1)
        outside = norms > self.radius
        if np.any(outside):
            self.w[outside] *= (self.radius / norms[outside])[:, None]
        self.variance += float(gradient @ gradient)
        self.meta.update(coeff)
        self._x = None


@dataclass
class DyadicFiniteSecondOrderLinearOracle:
    """Vectorized dyadic copies of ``FiniteSecondOrderLinearOracle``."""

    base_dim: int
    horizon: int
    initial_variance: float = 4.0
    max_eta: float = 0.25

    def __post_init__(self) -> None:
        if self.base_dim < 1:
            raise ValueError("base_dim must be positive")
        if self.initial_variance <= 0.0:
            raise ValueError("initial_variance must be positive")
        self.n = 2 * self.base_dim
        self.log_n = math.log(self.n)
        self.meta = DyadicAdaptMLProd(self.horizon)
        copies = self.meta.num_active_specialists
        self.cumulative_gains = np.zeros((copies, self.n), dtype=float)
        self.variance = np.full(copies, self.initial_variance, dtype=float)
        self._values: np.ndarray | None = None

    def predict(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.base_dim,):
            raise ValueError(f"expected context shape {(self.base_dim,)}, got {x.shape}")
        reset = self.meta.start_round()
        self.cumulative_gains[reset] = 0.0
        self.variance[reset] = self.initial_variance
        values = np.concatenate([x, -x])
        learning_rates = np.minimum(self.max_eta, np.sqrt(self.log_n / self.variance))
        logits = learning_rates[:, None] * self.cumulative_gains
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=1, keepdims=True)
        predictions = np.clip(
            np.sum(weights * values[None, :], axis=1),
            -1.0,
            1.0,
        )
        self._values = values
        return self.meta.aggregate(predictions)

    def update_gain(self, x: np.ndarray, coeff: float) -> None:
        del x
        if self._values is None:
            raise RuntimeError("predict must be called before update_gain")
        coeff = float(coeff)
        if abs(coeff) > 2.0 + 1e-12:
            raise ValueError("DyadicFiniteSecondOrderLinearOracle requires coefficients in [-2, 2]")
        self.cumulative_gains += coeff * self._values[None, :]
        self.variance += coeff * coeff
        self.meta.update(coeff)
        self._values = None


class DyadicScalarAdaptiveOGD:
    """Vectorized dyadic copies of ``ScalarAdaptiveOGD``."""

    def __init__(self, horizon: int) -> None:
        self.meta = DyadicAdaptMLProd(horizon)
        copies = self.meta.num_active_specialists
        self.values = np.zeros(copies, dtype=float)
        self.variance = np.full(copies, 4.0, dtype=float)
        self._pending = False

    def predict(self) -> float:
        reset = self.meta.start_round()
        self.values[reset] = 0.0
        self.variance[reset] = 4.0
        self._pending = True
        return self.meta.aggregate(self.values)

    def update_gain(self, coeff: float) -> None:
        if not self._pending:
            raise RuntimeError("predict must be called before update_gain")
        coeff = float(coeff)
        if abs(coeff) > 2.0 + 1e-12:
            raise ValueError("DyadicScalarAdaptiveOGD requires coefficients in [-2, 2]")
        self.values = np.clip(self.values + coeff / np.sqrt(self.variance), -1.0, 1.0)
        self.variance += coeff * coeff
        self.meta.update(coeff)
        self._pending = False
