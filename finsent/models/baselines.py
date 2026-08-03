"""The baselines the paper must beat, or honestly report that it does not.

These are deliberately humiliating. If a five-parameter logistic regression ties a
450,000-parameter cross-modal network at daily-to-weekly frequency -- and at this
frequency it very often does -- the author needs to know that before a referee does.
Reporting it is not a weakness in the paper; omitting it is.

Implemented in NumPy so they run, and are tested, without PyTorch. The deep baselines
that do need a framework (DLinear, PatchTST, StockNet) live in ``baselines_torch`` and
are wired into the same interface.

========================  ==============================================================
baseline                  what it tests
========================  ==============================================================
``BuyAndHold``            is there any edge over simply being long?
``TimeSeriesMomentum``    is the signal just 12-1 momentum in disguise?
``LogisticFive``          does 450k parameters buy anything over 5?
``LexiconSentiment``      does FinBERT beat a 2011 word count (Loughran-McDonald)?
``CrossSectionalMean``    the true null: a signal with no information at all
========================  ==============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from finsent.config import LABEL_DOWN, LABEL_NEUTRAL, LABEL_UP

__all__ = [
    "BaselineResult",
    "BuyAndHold",
    "TimeSeriesMomentum",
    "LogisticFive",
    "LexiconSentiment",
    "CrossSectionalMean",
    "BASELINES",
]

_EPS = 1e-12


@dataclass
class BaselineResult:
    """A baseline's output in the same schema the model produces.

    Sharing the schema is what lets ``finsent.eval.report`` treat baselines and the model
    identically, so no comparison can quietly use a different backtest.
    """

    score: np.ndarray
    probs: np.ndarray
    name: str

    def to_frame(self, dates, tickers, y_true, fwd_ret) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": dates,
                "ticker": tickers,
                "score": self.score,
                "mu_hat": self.score,
                "sigma2": np.full(self.score.shape, np.var(fwd_ret) + _EPS),
                "p_down": self.probs[:, LABEL_DOWN],
                "p_neutral": self.probs[:, LABEL_NEUTRAL],
                "p_up": self.probs[:, LABEL_UP],
                "y_true": y_true,
                "fwd_ret": fwd_ret,
                "tradeable": True,
            }
        )


def _scores_to_probs(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Map a scalar directional score to a 3-class distribution.

    A wide NEUTRAL band is assigned deliberately: a baseline that emits a scalar has no
    opinion about the magnitude threshold, and pretending otherwise would flatter it.
    """
    s = np.asarray(scores, dtype=float).ravel()
    s = np.nan_to_num(s / max(temperature, _EPS), nan=0.0)
    z = np.column_stack([-s, np.zeros_like(s), s])
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


@dataclass
class BuyAndHold:
    """Always long, equally weighted. The benchmark everybody forgets to report."""

    name: str = "buy_and_hold"

    def predict(self, features: pd.DataFrame) -> BaselineResult:
        n = len(features)
        score = np.ones(n)
        return BaselineResult(score=score, probs=_scores_to_probs(score, 1.0), name=self.name)


@dataclass
class TimeSeriesMomentum:
    """Sign of trailing 12-1 momentum (Moskowitz, Ooi & Pedersen, 2012).

    If the model's ranking correlates with this at 0.8, the "novel cross-modal signal"
    is momentum with extra steps, and the factor regression in ``eval.attribution`` will
    say so out loud.
    """

    column: str = "mom_12_1"
    name: str = "tsmom"

    def predict(self, features: pd.DataFrame) -> BaselineResult:
        raw = features[self.column].to_numpy(dtype=float)
        score = np.nan_to_num(raw, nan=0.0)
        return BaselineResult(score=score, probs=_scores_to_probs(score * 3.0), name=self.name)


@dataclass
class LogisticFive:
    """Multinomial logistic regression on five features. The important baseline.

    Fitted with plain gradient descent on the softmax cross-entropy so there is no
    dependency and no hidden regularisation to argue about. Five features, three classes,
    eighteen parameters.
    """

    columns: tuple[str, ...] = (
        "sentiment", "ret_5", "rsi_14", "ewma_vol_20", "mom_12_1",
    )
    lr: float = 0.5
    epochs: int = 400
    l2: float = 1e-4
    name: str = "logit5"
    coef_: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), repr=False)
    mean_: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    scale_: np.ndarray = field(default_factory=lambda: np.ones(0), repr=False)

    def _design(self, features: pd.DataFrame, fit: bool) -> np.ndarray:
        cols = [c for c in self.columns if c in features.columns]
        if not cols:
            raise ValueError(f"none of {self.columns} present in the feature frame")
        X = features[cols].to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            self.scale_[self.scale_ < _EPS] = 1.0
        X = (X - self.mean_) / self.scale_
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> "LogisticFive":
        X = self._design(features, fit=True)
        y = np.asarray(labels, dtype=int).ravel()
        n, d = X.shape
        k = 3

        onehot = np.zeros((n, k))
        onehot[np.arange(n), y] = 1.0
        w = np.zeros((d, k))

        for _ in range(self.epochs):
            z = X @ w
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            grad = X.T @ (p - onehot) / n + self.l2 * w
            w -= self.lr * grad

        self.coef_ = w
        return self

    def predict(self, features: pd.DataFrame) -> BaselineResult:
        if self.coef_.size == 0:
            raise RuntimeError("LogisticFive.fit must be called before predict")
        X = self._design(features, fit=False)
        z = X @ self.coef_
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return BaselineResult(score=p[:, LABEL_UP] - p[:, LABEL_DOWN], probs=p, name=self.name)


@dataclass
class LexiconSentiment:
    """Loughran-McDonald style net negativity -- the 2011 word count.

    Tetlock (2007) established that a dictionary count of negative words in financial
    text predicts returns. Any claim that a 110M-parameter encoder is necessary has to
    clear this bar first, and the paper reports the comparison whichever way it falls.
    """

    name: str = "lm_lexicon"
    positive: frozenset[str] = frozenset(
        {"gain", "gains", "beat", "beats", "surge", "surged", "profit", "profits",
         "growth", "upgrade", "upgraded", "strong", "record", "rally", "outperform"}
    )
    negative: frozenset[str] = frozenset(
        {"loss", "losses", "miss", "missed", "plunge", "plunged", "decline", "declined",
         "downgrade", "downgraded", "weak", "lawsuit", "probe", "fraud", "underperform"}
    )

    def score_text(self, headlines: list[str]) -> float:
        """Net negativity, normalised by token count; zero when there is no text."""
        if not headlines:
            return 0.0
        tokens = " ".join(headlines).lower().replace(",", " ").replace(".", " ").split()
        if not tokens:
            return 0.0
        pos = sum(1 for t in tokens if t in self.positive)
        neg = sum(1 for t in tokens if t in self.negative)
        return (pos - neg) / len(tokens)

    def predict(self, features: pd.DataFrame) -> BaselineResult:
        if "sentiment" not in features.columns:
            raise ValueError("LexiconSentiment expects a precomputed 'sentiment' column")
        score = np.nan_to_num(features["sentiment"].to_numpy(dtype=float), nan=0.0)
        return BaselineResult(score=score, probs=_scores_to_probs(score * 5.0), name=self.name)


@dataclass
class CrossSectionalMean:
    """A signal with no information. The floor every other number is measured against.

    Included because a table in which every method beats "the market" but none is
    compared to noise cannot distinguish skill from a shared bias in the backtest.
    """

    name: str = "null_signal"
    seed: int = 0

    def predict(self, features: pd.DataFrame) -> BaselineResult:
        rng = np.random.default_rng(self.seed)
        score = rng.standard_normal(len(features)) * 1e-6
        return BaselineResult(score=score, probs=_scores_to_probs(score), name=self.name)


BASELINES: dict[str, type] = {
    "buy_and_hold": BuyAndHold,
    "tsmom": TimeSeriesMomentum,
    "logit5": LogisticFive,
    "lm_lexicon": LexiconSentiment,
    "null_signal": CrossSectionalMean,
}
