"""Position sizing rules -- the paper's headline experiment lives here.

Four rules are implemented, all consuming **the same fixed predictions** so that the
comparison is paired and its variance is dominated by the sizing rule alone. That is
why this experiment has statistical power where an alpha test on ~2,300 daily returns
does not.

======  ==========================================  =================================
Rule    Name                                        What it demonstrates
======  ==========================================  =================================
(1)     ``equal_weight``                            the naive benchmark
(2)     ``raw_softmax_kelly``                       uncalibrated confidence -> overbet
(3)     ``calibrated_kelly``                        the proposed rule
(4)     ``conformal_gated_kelly``                   (3) plus abstention
======  ==========================================  =================================

Rule (2) being *worse* than rule (1) is the expected and most quotable finding: Kelly is
convex in overbetting, and Proposition 2 in ``growth_theory`` shows that underestimating
variance by a factor of two destroys all growth. Uncalibrated confidence plus Kelly is
worse than not using confidence at all.

Two V2 defects are fixed here and must not return:

* **No ``x confidence`` multiplier.** A calibrated probability already encodes
  confidence; multiplying by a separate scalar double-counts the same information.
* **No barrier odds fed a terminal-horizon probability.** ``f = kappa * mu / sigma^2``
  is a statement about the same object the model predicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from finsent.config import LABEL_DOWN, LABEL_UP
from finsent.decision.growth_theory import kelly_fraction_continuous

__all__ = [
    "SizingConfig",
    "score_from_probs",
    "kelly_continuous",
    "equal_weight_sizing",
    "raw_softmax_kelly_sizing",
    "calibrated_kelly_sizing",
    "conformal_gated_kelly_sizing",
    "SIZING_RULES",
    "cross_sectional_weights",
]

_EPS = 1e-12


@dataclass(frozen=True)
class SizingConfig:
    """Sizing hyperparameters (mirrors ``decision.kelly`` in the configuration)."""

    kappa: float = 0.25
    f_max: float = 0.10
    gross_leverage_max: float = 2.0
    dollar_neutral: bool = True
    long_short_quantile: float = 0.10


def score_from_probs(
    probs: np.ndarray,
    mode: Literal["p_up_minus_p_down", "p_up"] = "p_up_minus_p_down",
) -> np.ndarray:
    """Directional score from a 3-class probability vector.

    ``p_up`` alone is prohibited by SPEC.md 5.1: it assigns identical actions to
    ``(0.35, 0.60, 0.05)`` and ``(0.35, 0.05, 0.60)``, which carry opposite risk. The
    option exists only so the ablation can quantify how much that defect costs.
    """
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] < 3:
        raise ValueError(f"probs must be (n, 3); got {p.shape}")
    if mode == "p_up":
        return p[:, LABEL_UP]
    return p[:, LABEL_UP] - p[:, LABEL_DOWN]


def kelly_continuous(
    mu: np.ndarray,
    sigma2: np.ndarray,
    kappa: float = 0.25,
    f_max: float = 0.10,
) -> np.ndarray:
    """Fractional growth-optimal sizing, ``clip(kappa * mu / sigma^2, -f_max, f_max)``.

    ``kappa < 1`` (fractional Kelly) is not timidity: full Kelly assumes the parameters
    are known exactly, and Proposition 2 shows the penalty for overestimating the edge
    is quadratic then catastrophic. Quarter-Kelly retains roughly 75% of the growth for
    about half the volatility.
    """
    f = kappa * kelly_fraction_continuous(mu, sigma2)
    return np.clip(f, -abs(f_max), abs(f_max))


# --------------------------------------------------------------------------------------
# The four rules of the headline experiment
# --------------------------------------------------------------------------------------
def equal_weight_sizing(scores: np.ndarray, cfg: SizingConfig) -> np.ndarray:
    """Rule (1): unit exposure in the direction of the score, magnitude ignored."""
    return np.sign(np.asarray(scores, dtype=float)) * cfg.f_max


def raw_softmax_kelly_sizing(
    probs_raw: np.ndarray,
    mu: np.ndarray,
    cfg: SizingConfig,
) -> np.ndarray:
    """Rule (2): Kelly sized from *uncalibrated* top-class confidence.

    The variance proxy is taken from the softmax confidence itself, which is what a
    practitioner does when the model has no variance head: high confidence implies low
    assumed variance. Overconfident probabilities therefore shrink the denominator and
    the position is overbet -- the mechanism Proposition 2 quantifies.
    """
    p = np.asarray(probs_raw, dtype=float)
    conf = p.max(axis=1)
    implied_sigma2 = np.maximum((1.0 - conf) ** 2, 1e-4)
    return kelly_continuous(np.asarray(mu, dtype=float), implied_sigma2, cfg.kappa, cfg.f_max)


def calibrated_kelly_sizing(
    mu: np.ndarray,
    sigma2: np.ndarray,
    cfg: SizingConfig,
) -> np.ndarray:
    """Rule (3): Kelly from the heteroscedastic head's ``mu_hat`` and ``sigma2_hat``."""
    return kelly_continuous(mu, sigma2, cfg.kappa, cfg.f_max)


def conformal_gated_kelly_sizing(
    mu: np.ndarray,
    sigma2: np.ndarray,
    singleton_mask: np.ndarray,
    cfg: SizingConfig,
) -> np.ndarray:
    """Rule (4): rule (3), but only where the conformal prediction set is a singleton.

    Abstention is a position. Setting the weight to zero when the model's uncertainty
    set contains more than one direction is what converts a coverage guarantee into a
    risk reduction.
    """
    f = calibrated_kelly_sizing(mu, sigma2, cfg)
    return np.where(np.asarray(singleton_mask, dtype=bool), f, 0.0)


SIZING_RULES: tuple[str, ...] = (
    "equal_weight",
    "raw_softmax_kelly",
    "calibrated_kelly",
    "conformal_gated_kelly",
)


# --------------------------------------------------------------------------------------
# Cross-sectional assembly
# --------------------------------------------------------------------------------------
def cross_sectional_weights(
    raw_sizes: pd.Series,
    dates: pd.Series | np.ndarray,
    tickers: pd.Series | np.ndarray,
    cfg: SizingConfig,
) -> pd.DataFrame:
    """Turn per-name target fractions into a dated weight matrix.

    Applies, in order: dollar neutrality (subtract the cross-sectional mean), the
    per-name box constraint, and the gross-leverage cap. Every one of these is a real
    constraint a desk imposes, and stating them is what makes the backtest reviewable.
    """
    frame = pd.DataFrame(
        {
            "date": pd.Series(np.asarray(dates)).to_numpy(),
            "ticker": pd.Series(np.asarray(tickers)).to_numpy(),
            "size": np.asarray(raw_sizes, dtype=float),
        }
    ).dropna()

    wide = frame.pivot_table(index="date", columns="ticker", values="size", aggfunc="mean")
    wide = wide.sort_index().fillna(0.0)

    if cfg.dollar_neutral:
        active = (wide != 0.0).sum(axis=1).replace(0, np.nan)
        wide = wide.sub(wide.sum(axis=1) / active, axis=0).fillna(0.0)
        wide = wide.where(wide.abs() > _EPS, 0.0)

    wide = wide.clip(lower=-cfg.f_max, upper=cfg.f_max)

    gross = wide.abs().sum(axis=1)
    scale = (cfg.gross_leverage_max / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(0.0)
    return wide.mul(scale, axis=0)
