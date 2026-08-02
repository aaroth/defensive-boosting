"""Online algorithms compared in the experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Union

import numpy as np

from .weak_learners import (
    DyadicScalarAdaptiveOGD,
    LinearGainOracle,
    ScalarAdaptiveOGD,
    WeightedClassifier,
    clip_unit,
    project_simplex,
    sign_pm,
)


RoundResult = Dict[str, Union[float, str, Dict[str, float]]]
GainOracleFactory = Callable[[], LinearGainOracle]
ClassifierFactory = Callable[[], WeightedClassifier]


class OnlineAlgorithm:
    name: str

    def step(self, x: np.ndarray, y: int) -> RoundResult:
        raise NotImplementedError


@dataclass
class DefensiveBooster(OnlineAlgorithm):
    """The Defensive Booster from boosting.tex."""

    weak_factory: GainOracleFactory
    name: str = "defensive"

    def __post_init__(self) -> None:
        self.weak = self.weak_factory()
        self.scalar = ScalarAdaptiveOGD()
        self.aggregation = ScalarAdaptiveOGD()

    @staticmethod
    def _root(a: float, b: float) -> float:
        if abs(b) < 1e-12:
            if abs(a) < 1e-12:
                return 0.0
            return 1.0 if a > 0.0 else -1.0
        root = -a / b
        if -1.0 <= root <= 1.0:
            return float(root)
        return 1.0 if a > 0.0 else -1.0

    def step(self, x: np.ndarray, y: int) -> RoundResult:
        h_hat = self.weak.predict(x)
        theta = self.scalar.predict()
        lambda_t = self.aggregation.predict()
        q_h = 0.5 * (1.0 + lambda_t)
        q_p = 0.5 * (1.0 - lambda_t)
        p = self._root(float(q_h * h_hat), float(q_p * theta))
        residual = float(y) - p

        z_h = h_hat * residual
        z_p = theta * p * residual
        self.aggregation.update_gain(0.5 * (z_h - z_p))
        self.weak.update_gain(x, residual)
        self.scalar.update_gain(p * residual)

        return _round_result(
            self.name,
            p,
            y,
            extra={"h_hat": h_hat, "lambda": lambda_t},
        )


@dataclass
class StronglyAdaptiveDefensiveBooster(OnlineAlgorithm):
    """The interval-adaptive Defensive Booster from Section 5."""

    weak: LinearGainOracle
    horizon: int
    name: str = "adaptive_defensive"

    def __post_init__(self) -> None:
        self.scalar = DyadicScalarAdaptiveOGD(self.horizon)
        self.aggregation = DyadicScalarAdaptiveOGD(self.horizon)

    def step(self, x: np.ndarray, y: int) -> RoundResult:
        h_hat = self.weak.predict(x)
        theta = self.scalar.predict()
        lambda_t = self.aggregation.predict()
        q_h = 0.5 * (1.0 + lambda_t)
        q_p = 0.5 * (1.0 - lambda_t)
        p = DefensiveBooster._root(float(q_h * h_hat), float(q_p * theta))
        residual = float(y) - p

        z_h = h_hat * residual
        z_p = theta * p * residual
        self.aggregation.update_gain(0.5 * (z_h - z_p))
        self.weak.update_gain(x, residual)
        self.scalar.update_gain(p * residual)

        return _round_result(
            self.name,
            p,
            y,
            extra={
                "h_hat": h_hat,
                "lambda": lambda_t,
                "active_copies": float(self.scalar.meta.num_active_specialists),
            },
        )


@dataclass
class OnlineGradientBoosting(OnlineAlgorithm):
    """Beygelzimer-Hazan-Kale-Luo online gradient boosting for squared loss.

    This is Algorithm 1 specialized to one-dimensional squared loss with
    labels and weak predictions in [-1,1].  Thus B=1 and L_B=4.
    """

    weak_factory: GainOracleFactory
    n_learners: int = 20
    eta: float | None = None
    name: str = "ogb"

    def __post_init__(self) -> None:
        if self.n_learners < 1:
            raise ValueError("OnlineGradientBoosting requires at least one weak learner")
        self.learners = [self.weak_factory() for _ in range(self.n_learners)]
        if self.eta is None:
            self.eta = 1.0 if self.n_learners == 1 else math.log(self.n_learners) / self.n_learners
        self.sigma = np.zeros(self.n_learners, dtype=float)
        self.t = 0
        self.b_radius = 1.0
        self.lipschitz = 4.0

    def step(self, x: np.ndarray, y: int) -> RoundResult:
        self.t += 1
        partials: List[float] = [0.0]
        weak_preds = []
        current = 0.0
        for i, learner in enumerate(self.learners):
            a_i = learner.predict(x)
            weak_preds.append(a_i)
            current = clip_unit((1.0 - self.sigma[i] * self.eta) * current + self.eta * a_i)
            partials.append(current)
        p = current

        for i, learner in enumerate(self.learners):
            y_prev = partials[i]
            grad = 2.0 * (y_prev - float(y))
            learner.update_gain(x, -grad / self.lipschitz)
            alpha_t = 1.0 / (self.lipschitz * self.b_radius * math.sqrt(self.t))
            self.sigma[i] = float(np.clip(self.sigma[i] + alpha_t * grad * y_prev, 0.0, 1.0))

        return _round_result(self.name, p, y, extra={"mean_weak_pred": float(np.mean(weak_preds))})


@dataclass
class OnlineSquaredLossRegressor(OnlineAlgorithm):
    """One-copy online squared-loss learner over the same weak model class."""

    weak_factory: GainOracleFactory
    name: str = "unboosted"

    def __post_init__(self) -> None:
        self.learner = self.weak_factory()

    def step(self, x: np.ndarray, y: int) -> RoundResult:
        score = self.learner.predict(x)
        self.learner.update_gain(x, float(y) - score)
        return _round_result(self.name, score, y)


@dataclass
class OnlineUnboostedClassifier(OnlineAlgorithm):
    """One-copy online classification learner used by the boosters."""

    classifier_factory: ClassifierFactory
    name: str = "unboosted_cls"

    def __post_init__(self) -> None:
        self.learner = self.classifier_factory()

    def step(self, x: np.ndarray, y: int) -> RoundResult:
        score = float(self.learner.predict(x))
        self.learner.update_weighted(x, y, 1.0)
        return _round_result(self.name, score, y)


@dataclass
class OnlineBBM(OnlineAlgorithm):
    """Beygelzimer-Kale-Luo Online BBM classification booster.

    This is the importance-weighted version of Online BBM.  The booster returns
    a hard majority-vote label, so its Brier loss is the Brier loss of the
    induced binary probability forecast.
    """

    classifier_factory: ClassifierFactory
    n_learners: int = 40
    gamma: float = 0.1
    name: str = "bbm"

    def __post_init__(self) -> None:
        if self.n_learners < 1:
            raise ValueError("Online BBM requires at least one weak learner")
        if not 0.0 < self.gamma < 0.5:
            raise ValueError("Online BBM requires gamma in (0, 1/2)")
        self.learners = [self.classifier_factory() for _ in range(self.n_learners)]
        self._log_pmf: List[np.ndarray] = []
        self._log_pmf_max: List[float] = []
        # Beygelzimer--Kale--Luo define gamma by weak error 1/2-gamma.
        p = 0.5 + float(self.gamma)
        q = 1.0 - p
        for remaining in range(self.n_learners):
            logs = np.empty(remaining + 1, dtype=float)
            for k in range(remaining + 1):
                logs[k] = math.lgamma(remaining + 1) - math.lgamma(k + 1) - math.lgamma(remaining - k + 1)
                logs[k] += k * math.log(p) + (remaining - k) * math.log(q)
            self._log_pmf.append(logs)
            self._log_pmf_max.append(float(np.max(logs)))

    def _bbm_weight(self, remaining: int, signed_partial_sum: float) -> float:
        k = math.floor((remaining - signed_partial_sum + 1.0) / 2.0)
        if k < 0 or k > remaining:
            return 0.0
        log_weight = self._log_pmf[remaining][k]
        return float(math.exp(log_weight - self._log_pmf_max[remaining]))

    def step(self, x: np.ndarray, y: int) -> RoundResult:
        h = np.array([learner.predict(x) for learner in self.learners], dtype=float)
        score = float(sign_pm(float(np.sum(h))))

        signed_partial_sum = 0.0
        total_weight = 0.0
        for i, learner in enumerate(self.learners):
            remaining = self.n_learners - i - 1
            weight = self._bbm_weight(remaining, signed_partial_sum)
            learner.update_weighted(x, y, weight)
            total_weight += weight
            signed_partial_sum += float(y) * h[i]

        return _round_result(
            self.name,
            score,
            y,
            extra={
                "mean_bbm_weight": total_weight / self.n_learners,
                "bbm_margin": float(np.sum(h)) / self.n_learners,
            },
        )


@dataclass
class AdaBoostOL(OnlineAlgorithm):
    """Importance-weighted AdaBoost.OL.W of Beygelzimer, Kale, and Luo.

    Each partial ensemble is an expert.  Hedge combines their hard
    predictions, while projected OGD learns the logistic-loss coefficient of
    each weak learner.  We return Hedge's probability of predicting +1 as a
    signed probability score.  Thus randomized error is the expected 0/1
    loss of AdaBoost.OL's randomized classifier, and Brier loss scores that
    same randomization probability as a probability forecast.
    """

    classifier_factory: ClassifierFactory
    n_learners: int = 40
    name: str = "adaboost_ol"

    def __post_init__(self) -> None:
        if self.n_learners < 1:
            raise ValueError("AdaBoost.OL requires at least one weak learner")
        self.learners = [self.classifier_factory() for _ in range(self.n_learners)]
        self.alpha = np.zeros(self.n_learners, dtype=float)
        self.log_expert_weights = np.zeros(self.n_learners, dtype=float)
        self.t = 0

    @staticmethod
    def _logistic_weight(margin: float) -> float:
        """Return 1 / (1 + exp(margin)) without overflow."""

        if margin >= 0.0:
            exp_neg = math.exp(-margin)
            return exp_neg / (1.0 + exp_neg)
        exp_pos = math.exp(margin)
        return 1.0 / (1.0 + exp_pos)

    def step(self, x: np.ndarray, y: int) -> RoundResult:
        self.t += 1
        h = np.array([learner.predict(x) for learner in self.learners], dtype=float)
        prefix_scores = np.cumsum(self.alpha * h)
        expert_predictions = np.where(prefix_scores >= 0.0, 1.0, -1.0)

        shifted = self.log_expert_weights - np.max(self.log_expert_weights)
        expert_weights = np.exp(shifted)
        expert_weights /= np.sum(expert_weights)
        score = float(expert_weights @ expert_predictions)

        signed_prefix_margin = 0.0
        total_importance = 0.0
        eta_t = 4.0 / math.sqrt(self.t)
        for i, learner in enumerate(self.learners):
            importance = self._logistic_weight(signed_prefix_margin)
            learner.update_weighted(x, y, importance)
            total_importance += importance

            z_i = float(y) * h[i]
            signed_prefix_margin += self.alpha[i] * z_i
            gradient_weight = self._logistic_weight(signed_prefix_margin)
            self.alpha[i] = float(
                np.clip(self.alpha[i] + eta_t * z_i * gradient_weight, -2.0, 2.0)
            )

        self.log_expert_weights -= (expert_predictions != float(y)).astype(float)
        self.log_expert_weights -= np.max(self.log_expert_weights)

        return _round_result(
            self.name,
            score,
            y,
            extra={
                "mean_importance_weight": total_importance / self.n_learners,
                "mean_abs_alpha": float(np.mean(np.abs(self.alpha))),
            },
        )


@dataclass
class OnlineSmoothBoost(OnlineAlgorithm):
    """Chen-Lin-Lu online SmoothBoost baseline with OCP combiner.

    The implementation uses importance-weighted updates, as the paper reports
    doing in experiments.  Weak learners receive the SmoothBoost weight
    w_t^(i); the OCP combiner projects alpha onto the simplex.
    """

    classifier_factory: ClassifierFactory
    n_learners: int = 40
    gamma: float = 0.1
    theta: float | None = None
    alpha_lr: float = 0.5
    combiner: str = "ocp"
    name: str = "osboost"

    def __post_init__(self) -> None:
        self.learners = [self.classifier_factory() for _ in range(self.n_learners)]
        self.alpha = np.ones(self.n_learners, dtype=float) / self.n_learners
        self.t = 0
        if self.theta is None:
            self.theta = self.gamma / (2.0 + self.gamma)

    def step(self, x: np.ndarray, y: int) -> RoundResult:
        self.t += 1
        h = np.array([learner.predict(x) for learner in self.learners], dtype=float)
        if self.combiner == "uniform":
            alpha_for_pred = np.ones(self.n_learners, dtype=float) / self.n_learners
        else:
            alpha_for_pred = self.alpha
        score = clip_unit(float(alpha_for_pred @ h))

        if self.combiner == "ocp" and float(y) * score < float(self.theta):
            lr = self.alpha_lr / math.sqrt(self.t)
            self.alpha = project_simplex(self.alpha + lr * float(y) * h)

        z = 0.0
        smooth_weight = 1.0
        total_weight = 0.0
        for i, learner in enumerate(self.learners):
            learner.update_weighted(x, y, smooth_weight)
            total_weight += smooth_weight
            z += float(y) * h[i] - float(self.theta)
            smooth_weight = min((1.0 - self.gamma) ** (z / 2.0), 1.0)

        return _round_result(
            self.name,
            score,
            y,
            extra={"mean_smooth_weight": total_weight / self.n_learners},
        )


def _round_result(name: str, score: float, y: int, extra: Dict[str, float] | None = None) -> RoundResult:
    score = clip_unit(score)
    prob = (score + 1.0) / 2.0
    label = (float(y) + 1.0) / 2.0
    pred = sign_pm(score)
    result: RoundResult = {
        "algorithm": name,
        "score": score,
        "prediction": float(pred),
        "mistake": float(pred != y),
        "brier_loss": float((label - prob) ** 2),
        "signed_squared_loss": float((float(y) - score) ** 2),
        "randomized_error": float((1.0 - float(y) * score) / 2.0),
    }
    if extra:
        result["extra"] = {k: float(v) for k, v in extra.items()}
    return result
