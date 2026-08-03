"""Conformal prediction sets with abstention, and adaptive coverage under shift.

Deliberately implemented in NumPy rather than PyTorch. Conformal calibration is a
post-hoc operation on a probability matrix, it belongs to the decision layer rather than
the network, and keeping it framework-free means the paper's headline uncertainty
machinery can be tested and reproduced without a GPU or a deep-learning install.

Why this is in the paper at all
-------------------------------
Everything else here produces a *point* forecast that is then sized. Conformal
prediction produces a **set** with a finite-sample coverage guarantee: under
exchangeability, the true class lies in ``C_alpha(x)`` with probability at least
``1 - alpha``, with no assumption whatsoever about the model being correct. Trading only
when the set is a singleton turns that guarantee into an abstention rule, and abstention
is a position.

The guarantee is the interesting part precisely because a daily equity model's point
forecasts are weak. A statement of the form "we are 90% covered and we decline to trade
on 60% of days" is far more defensible than "our accuracy is 57%".

Distribution shift
------------------
Financial data is not exchangeable across regimes, so split conformal's guarantee
degrades in 2020 and 2022 exactly when it matters. ``AdaptiveConformal`` implements the
online update of Gibbs & Candes (2021),

    alpha[t+1] = alpha[t] + gamma * (alpha_target - 1{y_t not in C_t})

which restores long-run coverage without any exchangeability assumption. Reporting both
the split and adaptive variants, and showing where the split version breaks, strengthens
the paper rather than weakening it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "aps_scores",
    "lac_scores",
    "conformal_quantile",
    "SplitConformal",
    "AdaptiveConformal",
    "coverage_report",
]

_EPS = 1e-12


def aps_scores(probs: np.ndarray, labels: np.ndarray, randomized: bool = True,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Adaptive Prediction Set nonconformity scores (Romano, Sesia & Candes, 2020).

    The score is the probability mass of classes ranked at or above the true one. Small
    scores mean the model ranked the truth highly. APS adapts set size to difficulty --
    easy samples get singletons, hard ones get larger sets -- which is what an abstention
    rule wants.

    ``randomized`` defaults to True and is not cosmetic. With three classes the
    non-randomized score is a coarse discrete variable, and the resulting sets
    systematically **over-cover**: in testing, a nominal 90% predictor delivered 100%
    coverage by simply including every class. Subtracting ``u * p_y`` smooths the score
    and restores exact coverage. The randomisation is seeded, so decisions remain
    reproducible.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int).ravel()

    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    cumulative = np.cumsum(sorted_p, axis=1)

    rank_of_true = np.argmax(order == y[:, None], axis=1)
    rows = np.arange(y.size)
    scores = cumulative[rows, rank_of_true]

    if randomized:
        rng = rng or np.random.default_rng(0)
        scores = scores - rng.random(y.size) * sorted_p[rows, rank_of_true]
    return scores


def lac_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Least-ambiguous set score, ``1 - p_y``. Smaller sets, worse conditional coverage."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int).ravel()
    return 1.0 - p[np.arange(y.size), y]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample corrected quantile ``ceil((n+1)(1-alpha)) / n``.

    The ``(n+1)`` correction is what makes the coverage guarantee exact rather than
    asymptotic; using the plain empirical quantile under-covers by roughly ``1/n``.
    """
    s = np.asarray(scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return 1.0
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(s, level, method="higher"))


@dataclass
class SplitConformal:
    """Split conformal predictor fitted on a held-out calibration fold.

    ``fit`` must be called on the **inner-validation** block of a walk-forward fold, never
    on the test block. That is the same discipline as the temperature fit, and for the
    same reason: a quantile chosen on the evaluation data is not a guarantee, it is a
    fitted parameter.
    """

    alpha: float = 0.10
    score: str = "aps"
    randomized: bool = True
    seed: int = 0
    q_hat: float = field(default=np.nan)
    n_calibration: int = 0

    def _scores(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if self.score == "aps":
            return aps_scores(probs, labels, self.randomized, np.random.default_rng(self.seed))
        if self.score == "lac":
            return lac_scores(probs, labels)
        raise ValueError(f"unknown conformal score {self.score!r}")

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "SplitConformal":
        s = self._scores(probs, labels)
        self.q_hat = conformal_quantile(s, self.alpha)
        self.n_calibration = int(np.asarray(labels).size)
        return self

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        """Boolean ``(n, n_classes)`` membership matrix.

        For APS, classes are admitted in descending probability until the cumulative
        mass reaches ``q_hat``; the final admitted class is then dropped with probability
        ``(C_J - q_hat) / p_(J)``, which is the randomisation that makes coverage exact
        rather than conservative. The draw is seeded, so a given model and calibration
        set always produce the same trading decisions.

        The top-1 class is always retained, so a set is never empty: an empty set is a
        degenerate abstention that says nothing about which directions remain plausible.
        """
        if not np.isfinite(self.q_hat):
            raise RuntimeError("SplitConformal.fit must be called before predict_sets")

        p = np.asarray(probs, dtype=float)
        order = np.argsort(-p, axis=1)
        sorted_p = np.take_along_axis(p, order, axis=1)
        cumulative = np.cumsum(sorted_p, axis=1)

        if self.score == "aps":
            include_sorted = cumulative - sorted_p < self.q_hat
            if self.randomized:
                rng = np.random.default_rng(self.seed + 1)
                # Index of the last admitted class per row.
                last = include_sorted.sum(axis=1) - 1
                rows = np.arange(p.shape[0])
                excess = cumulative[rows, last] - self.q_hat
                width = np.maximum(sorted_p[rows, last], _EPS)
                drop = rng.random(p.shape[0]) < np.clip(excess / width, 0.0, 1.0)
                include_sorted[rows[drop], last[drop]] = False
        else:
            include_sorted = sorted_p >= 1.0 - self.q_hat

        include_sorted[:, 0] = True
        sets = np.zeros_like(p, dtype=bool)
        np.put_along_axis(sets, order, include_sorted, axis=1)
        return sets

    def singleton_mask(self, probs: np.ndarray) -> np.ndarray:
        """True where the model commits to exactly one direction -- the trade gate."""
        return self.predict_sets(probs).sum(axis=1) == 1

    def evaluate(self, probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        sets = self.predict_sets(probs)
        y = np.asarray(labels, dtype=int).ravel()
        covered = sets[np.arange(y.size), y]
        sizes = sets.sum(axis=1)
        return {
            "alpha": self.alpha,
            "nominal_coverage": 1.0 - self.alpha,
            "empirical_coverage": float(covered.mean()),
            "coverage_gap": float(covered.mean() - (1.0 - self.alpha)),
            "mean_set_size": float(sizes.mean()),
            "singleton_rate": float((sizes == 1).mean()),
            "abstention_rate": float((sizes > 1).mean()),
            "q_hat": float(self.q_hat),
            "n": int(y.size),
        }


@dataclass
class AdaptiveConformal:
    """Online coverage control under distribution shift (Gibbs & Candes, 2021).

    ``alpha_t`` is nudged after every observation so that long-run coverage tracks the
    target even when exchangeability fails. ``gamma`` trades adaptation speed against
    stability; 0.005 recovers from a regime break in roughly thirty sessions, which is
    the right order for equity data.
    """

    alpha_target: float = 0.10
    gamma: float = 0.005
    score: str = "aps"
    alpha_t: float = field(init=False)
    history: list[float] = field(default_factory=list, repr=False)
    _calibration_scores: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    def __post_init__(self) -> None:
        self.alpha_t = float(self.alpha_target)

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "AdaptiveConformal":
        self._calibration_scores = (
            aps_scores(probs, labels) if self.score == "aps" else lac_scores(probs, labels)
        )
        return self

    def _quantile(self) -> float:
        return conformal_quantile(self._calibration_scores, float(np.clip(self.alpha_t, 1e-3, 0.5)))

    def step(self, probs_row: np.ndarray, label: int | None = None) -> np.ndarray:
        """Emit a prediction set for one observation, then update ``alpha_t``.

        ``label`` is the realised outcome, available only *after* the decision. Passing
        it before acting on the set would be a leak; the ordering here mirrors live use.
        """
        predictor = SplitConformal(alpha=float(np.clip(self.alpha_t, 1e-3, 0.5)),
                                   score=self.score)
        predictor.q_hat = self._quantile()
        predictor.n_calibration = int(self._calibration_scores.size)
        sets = predictor.predict_sets(np.atleast_2d(probs_row))

        if label is not None:
            covered = bool(sets[0, int(label)])
            self.alpha_t = float(
                np.clip(
                    self.alpha_t + self.gamma * (self.alpha_target - (0.0 if covered else 1.0)),
                    1e-3,
                    0.5,
                )
            )
            self.history.append(self.alpha_t)
        return sets[0]

    def run(self, probs: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray | float]:
        """Sequential pass over a test block, returning sets and realised coverage."""
        p = np.asarray(probs, dtype=float)
        y = np.asarray(labels, dtype=int).ravel()

        sets = np.zeros_like(p, dtype=bool)
        for i in range(y.size):
            sets[i] = self.step(p[i], int(y[i]))

        covered = sets[np.arange(y.size), y]
        sizes = sets.sum(axis=1)
        return {
            "sets": sets,
            "empirical_coverage": float(covered.mean()),
            "target_coverage": 1.0 - self.alpha_target,
            "final_alpha": float(self.alpha_t),
            "mean_set_size": float(sizes.mean()),
            "singleton_rate": float((sizes == 1).mean()),
            "alpha_path": np.asarray(self.history, dtype=float),
        }


def coverage_report(
    probs_cal: np.ndarray,
    labels_cal: np.ndarray,
    probs_test: np.ndarray,
    labels_test: np.ndarray,
    alphas: tuple[float, ...] = (0.05, 0.10, 0.20),
    score: str = "aps",
) -> list[dict[str, float]]:
    """Coverage and set size across nominal levels -- the paper's Figure F9."""
    rows = []
    for alpha in alphas:
        predictor = SplitConformal(alpha=alpha, score=score).fit(probs_cal, labels_cal)
        rows.append(predictor.evaluate(probs_test, labels_test))
    return rows
