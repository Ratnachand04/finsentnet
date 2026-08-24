"""Cross-sectional backtester with an explicitly stated trading rule.

A backtest whose trading rule is unstated is unreviewable, and the V2 draft never said
what happened on a NEUTRAL prediction, whether SELL signals were shorted (they were not
-- the optimiser clipped to long-only, so half the signal vocabulary was unusable), or
what the turnover was. ``TradingRule`` below is the complete specification, it is
printed into the paper's methods section by ``describe()``, and nothing in this module
reads anything that is not in it.

The rule, in words
------------------
1. On each **rebalance date** (every ``rebalance_days`` sessions) the model emits a
   score per name in that day's point-in-time universe.
2. Names whose conformal prediction set is not a singleton are dropped when
   ``gate_on_singleton`` is set. Abstention is a position.
3. Surviving names are sorted; the top and bottom ``quantile`` fractions are held long
   and short respectively (``decile_long_short``), or sized by growth-optimal Kelly
   (``kelly``).
4. Weights are made dollar-neutral, clipped to ``f_max`` per name, and scaled to at
   most ``gross_leverage_max`` gross.
5. Weights are **held** until the next rebalance date. Returns accrue daily on the held
   book; costs are charged on the date weights change.

Everything is decided with information available at the decision timestamp: the score at
date ``t`` earns the return from ``t`` to ``t+1``, never the return that ends at ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Literal

import numpy as np
import pandas as pd

from finsent.decision.costs import CostModel, break_even_cost_bps
from finsent.decision.sizing import SizingConfig, kelly_continuous
from finsent.eval import metrics as M

__all__ = ["TradingRule", "BacktestResult", "build_weights", "run_backtest", "cost_sweep"]

_EPS = 1e-12


@dataclass(frozen=True)
class TradingRule:
    """The complete, printable specification of how signals become positions."""

    scheme: Literal["decile_long_short", "kelly", "equal_weight_long_short"] = "decile_long_short"
    quantile: float = 0.10
    rebalance_days: int = 5
    gate_on_singleton: bool = True
    f_max: float = 0.10
    gross_leverage_max: float = 2.0
    dollar_neutral: bool = True
    kappa: float = 0.25
    neutral_action: Literal["flat", "hold"] = "flat"

    def describe(self) -> str:
        """Prose description, pasted verbatim into the paper's methods section."""
        legs = f"top/bottom {self.quantile:.0%}"
        gate = "conformal singleton sets only" if self.gate_on_singleton else "all names"
        neutral = "flat on NEUTRAL" if self.neutral_action == "flat" else "hold through NEUTRAL"
        return (
            f"Rebalance every {self.rebalance_days} sessions. Trade {gate}. "
            f"Scheme: {self.scheme} ({legs}). {neutral}. "
            f"Per-name cap {self.f_max:.0%}; gross leverage <= {self.gross_leverage_max:.1f}x; "
            f"{'dollar-neutral' if self.dollar_neutral else 'directional'}. "
            f"Kelly multiplier kappa = {self.kappa}. Weights held between rebalances; "
            f"costs charged on weight changes."
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class BacktestResult:
    """Everything Tables 5 and 6 of the paper need, and nothing that is not measured."""

    gross_returns: pd.Series
    net_returns: pd.Series
    costs: pd.Series
    weights: pd.DataFrame
    equity: pd.Series
    rule: TradingRule
    summary_gross: dict[str, float] = field(default_factory=dict)
    summary_net: dict[str, float] = field(default_factory=dict)
    turnover: float = np.nan
    break_even_bps: float = np.nan
    n_trades: int = 0

    def as_row(self, name: str) -> dict[str, object]:
        return {
            "method": name,
            "sharpe_gross": self.summary_gross.get("sharpe", np.nan),
            "sharpe_net": self.summary_net.get("sharpe", np.nan),
            "sortino_net": self.summary_net.get("sortino", np.nan),
            "ann_return_net": self.summary_net.get("ann_return", np.nan),
            "max_drawdown": self.summary_net.get("max_drawdown", np.nan),
            "hit_rate": self.summary_net.get("hit_rate", np.nan),
            "turnover_annualised": self.turnover * 252 if np.isfinite(self.turnover) else np.nan,
            "break_even_bps": self.break_even_bps,
            "n_periods": self.summary_net.get("n_periods", 0),
        }


def _rebalance_dates(dates: pd.DatetimeIndex, every: int) -> pd.DatetimeIndex:
    if every <= 1:
        return dates
    return dates[:: int(every)]


def build_weights(
    signals: pd.DataFrame,
    rule: TradingRule,
    sizing: SizingConfig | None = None,
) -> pd.DataFrame:
    """Turn a tidy signal frame into a dated, held weight matrix.

    ``signals`` must have columns ``date``, ``ticker``, ``score`` and may have
    ``sigma2`` (required by the ``kelly`` scheme) and ``tradeable`` (the conformal
    singleton mask). Any other column is ignored, deliberately: the backtester must not
    be able to see a realised return.
    """
    required = {"date", "ticker", "score"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"signals is missing required columns: {sorted(missing)}")
    if rule.scheme == "kelly" and "sigma2" not in signals.columns:
        raise ValueError("scheme='kelly' requires a 'sigma2' column from the variance head")

    df = signals.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["score"]).sort_values(["date", "ticker"])

    if rule.gate_on_singleton and "tradeable" in df.columns:
        df = df[df["tradeable"].astype(bool)]

    all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(signals["date"]).unique()))
    rebal = set(_rebalance_dates(all_dates, rule.rebalance_days))

    tickers = sorted(signals["ticker"].astype(str).unique())
    rows: dict[pd.Timestamp, pd.Series] = {}
    current = pd.Series(0.0, index=tickers, dtype=float)

    for date in all_dates:
        if date in rebal:
            grp = df[df["date"] == date]
            current = _weights_for_date(grp, rule, sizing, tickers)
        rows[date] = current.copy()

    return pd.DataFrame(rows).T.reindex(columns=tickers).fillna(0.0)


def _weights_for_date(
    grp: pd.DataFrame,
    rule: TradingRule,
    sizing: SizingConfig | None,
    tickers: list[str],
) -> pd.Series:
    w = pd.Series(0.0, index=tickers, dtype=float)
    if grp.empty:
        return w

    names = grp["ticker"].astype(str).to_numpy()
    scores = grp["score"].to_numpy(dtype=float)

    if rule.scheme == "kelly":
        cfg = sizing or SizingConfig(kappa=rule.kappa, f_max=rule.f_max)
        sigma2 = grp["sigma2"].to_numpy(dtype=float)
        mu = scores
        if rule.dollar_neutral and mu.size > 1:
            # A dollar-neutral book cannot express a view on the market level, so the
            # common component of mu must be removed BEFORE sizing, not after. Sizing
            # the raw level and demeaning the resulting weights is not equivalent and
            # fails badly: on this panel the label distribution drifts up (UP 34.3%
            # against DOWN 27.1%), so every mu_hat was positive, every Kelly fraction
            # saturated at +f_max, and demeaning the saturated weights collapsed the
            # entire book to zero on 100% of dates. Demeaning the edge first keeps the
            # cross-sectional information that the book is actually able to trade.
            mu = mu - mu.mean()
        raw = kelly_continuous(mu, sigma2, cfg.kappa, cfg.f_max)
    else:
        n = scores.size
        k = max(int(np.floor(n * rule.quantile)), 1)
        if n < 2 * k:
            return w
        order = np.argsort(scores)
        raw = np.zeros(n)
        raw[order[-k:]] = rule.f_max
        raw[order[:k]] = -rule.f_max

    if rule.dollar_neutral and np.any(raw != 0.0):
        active = raw != 0.0
        raw = raw - raw[active].mean() * active

    raw = np.clip(raw, -rule.f_max, rule.f_max)
    gross = np.abs(raw).sum()
    if gross > rule.gross_leverage_max and gross > _EPS:
        raw = raw * (rule.gross_leverage_max / gross)

    w.loc[list(names)] = raw
    return w


def run_backtest(
    weights: pd.DataFrame,
    forward_returns: pd.DataFrame,
    rule: TradingRule,
    cost_model: CostModel | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Accrue returns on a held book and charge costs on weight changes.

    ``forward_returns[t, i]`` must be the return **earned** by holding name ``i`` from
    ``t`` to ``t+1``. Constructing it that way, rather than shifting inside the
    backtester, keeps the causality contract visible at the call site where a reviewer
    can check it.
    """
    cost_model = cost_model or CostModel()

    idx = weights.index.intersection(forward_returns.index)
    cols = weights.columns.intersection(forward_returns.columns)
    if len(idx) == 0 or len(cols) == 0:
        raise ValueError("weights and forward_returns do not overlap on dates/tickers")

    w = weights.loc[idx, cols].fillna(0.0)
    r = forward_returns.loc[idx, cols]

    gross = (w * r.fillna(0.0)).sum(axis=1)

    traded = w.diff().abs()
    traded.iloc[0] = w.iloc[0].abs()
    cost = traded.sum(axis=1) * (cost_model.spread_bps * 1e-4)
    net = gross - cost

    turn = float(traded.sum(axis=1).mean())
    equity = (1.0 + net).cumprod()

    return BacktestResult(
        gross_returns=gross,
        net_returns=net,
        costs=cost,
        weights=w,
        equity=equity,
        rule=rule,
        summary_gross=M.performance_summary(gross.to_numpy(), periods_per_year, risk_free_rate),
        summary_net=M.performance_summary(
            net.to_numpy(), periods_per_year, risk_free_rate, weights=w.to_numpy()
        ),
        turnover=turn,
        break_even_bps=break_even_cost_bps(gross, turn),
        n_trades=int((traded.to_numpy() > _EPS).sum()),
    )


def cost_sweep(
    weights: pd.DataFrame,
    forward_returns: pd.DataFrame,
    rule: TradingRule,
    bps_levels: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0),
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Net Sharpe as a function of round-trip cost -- the paper's Figure F7.

    Presenting a single cost assumption invites the reviewer to guess which one flatters
    the result. Presenting the curve, with the break-even level marked, forecloses that.
    """
    rows = []
    for bps in bps_levels:
        res = run_backtest(
            weights, forward_returns, rule, CostModel(spread_bps=bps), periods_per_year
        )
        rows.append(
            {
                "cost_bps": bps,
                "sharpe_net": res.summary_net.get("sharpe", np.nan),
                "ann_return_net": res.summary_net.get("ann_return", np.nan),
                "max_drawdown": res.summary_net.get("max_drawdown", np.nan),
                "turnover_annualised": res.turnover * periods_per_year,
            }
        )
    return pd.DataFrame(rows)
