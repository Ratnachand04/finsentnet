"""Transaction costs: spread plus square-root market impact.

A flat "5 bps commission + 2 bps slippage" is not defensible for a paper, because it is
insensitive to the two things that actually determine cost -- how wide the name trades
and how much of its daily volume you are taking. The model here is

    c = 0.5 * spread + eta * sigma * sqrt(Q / ADV)                             (1)

the square-root impact law in the Almgren-Chriss lineage, with ``eta ~ 0.5`` a standard
calibration (Almgren et al., 2005; Frazzini, Israel & Moskowitz, 2018).

The paper reports **net Sharpe as a curve over 0-25 bps**, not a single number, and
marks the break-even cost. That curve is the honest way to present a daily-to-weekly
equity strategy, because the arithmetic below decides whether it is investable at all:

    decile long-short gross ~= IC * sigma_cs * 2.25
    IC = 0.02, sigma_cs = 2.0%  ->  ~9 bps/day gross
    daily rebalancing turnover ~= 100%/day  ->  10 bps/day cost at 10 bps round trip
    ==> net is negative. Hence SPEC.md sets the primary horizon to h = 5 (weekly).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "CostModel",
    "corwin_schultz_spread",
    "apply_costs",
    "break_even_cost_bps",
]

_EPS = 1e-12


@dataclass(frozen=True)
class CostModel:
    """Participation-aware cost model.

    Parameters
    ----------
    spread_bps
        Half-spread floor in basis points, used when no per-name spread is supplied.
    impact_eta
        Coefficient of the square-root impact term. 0.5 is the conventional value.
    participation_cap
        Fraction of ADV above which the impact term is treated as prohibitive; used by
        the capacity estimate rather than to reject trades.
    """

    spread_bps: float = 10.0
    impact_eta: float = 0.5
    participation_cap: float = 0.10

    def cost_rate(
        self,
        traded_notional: np.ndarray | float,
        adv: np.ndarray | float | None = None,
        volatility: np.ndarray | float | None = None,
        spread: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Cost as a fraction of traded notional (equation 1).

        With ``adv`` or ``volatility`` omitted the model degenerates to a flat
        half-spread, which is exactly the V2 assumption -- retained only so that the
        cost-sensitivity curve has a comparable left-hand endpoint.
        """
        q = np.abs(np.asarray(traded_notional, dtype=float))
        half_spread = (
            0.5 * np.asarray(spread, dtype=float)
            if spread is not None
            else np.full_like(q, self.spread_bps * 1e-4)
        )
        if adv is None or volatility is None:
            return half_spread

        adv_arr = np.maximum(np.asarray(adv, dtype=float), _EPS)
        vol = np.asarray(volatility, dtype=float)
        impact = self.impact_eta * vol * np.sqrt(q / adv_arr)
        return half_spread + impact

    def flat_bps(self, bps: float) -> "CostModel":
        """A copy of this model with the spread floor replaced, for the cost sweep."""
        return CostModel(
            spread_bps=bps, impact_eta=self.impact_eta, participation_cap=self.participation_cap
        )

    def capacity_usd(
        self,
        adv_usd: np.ndarray | float,
        turnover_per_period: float,
    ) -> float:
        """Crude AUM capacity: the size at which participation hits the cap.

        Deliberately crude, and labelled as such in the paper. A reviewer wants to see
        that the author knows capacity is finite, not a precise number nobody can verify.
        """
        adv = np.asarray(adv_usd, dtype=float)
        adv = adv[np.isfinite(adv)]
        if adv.size == 0 or turnover_per_period <= _EPS:
            return np.nan
        return float(adv.sum() * self.participation_cap / turnover_per_period)


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Corwin & Schultz (2012) high-low bid-ask spread estimator.

    Useful because it needs only daily OHLC, which every free data source provides,
    where true quoted spreads generally cost money. Negative estimates are set to zero,
    as the original paper prescribes.
    """
    h = pd.Series(high, dtype=float)
    l = pd.Series(low, dtype=float)

    hl = np.log(h / l.replace(0.0, np.nan)) ** 2
    beta = hl + hl.shift(1)

    h2 = h.rolling(2).max()
    l2 = l.rolling(2).min()
    gamma = np.log(h2 / l2.replace(0.0, np.nan)) ** 2

    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.clip(lower=0.0)


def apply_costs(
    gross_returns: pd.Series,
    weights: pd.DataFrame,
    cost_model: CostModel,
    adv: pd.DataFrame | None = None,
    volatility: pd.DataFrame | None = None,
    gross_notional: float = 1.0,
) -> tuple[pd.Series, pd.Series]:
    """Subtract trading costs from a gross return stream.

    Returns ``(net_returns, cost_series)``. Costs are charged on the date the weight
    changes, on the absolute change in weight, which is the standard convention and the
    one the paper states explicitly.
    """
    w = weights.fillna(0.0)
    traded = w.diff().abs()
    traded.iloc[0] = w.iloc[0].abs()

    if adv is None or volatility is None:
        rate = cost_model.spread_bps * 1e-4
        cost = traded.sum(axis=1) * rate
    else:
        adv_al = adv.reindex_like(w).ffill()
        vol_al = volatility.reindex_like(w).ffill()
        rates = cost_model.cost_rate(
            traded_notional=traded.to_numpy() * gross_notional,
            adv=adv_al.to_numpy(),
            volatility=vol_al.to_numpy(),
        )
        cost = pd.Series((traded.to_numpy() * rates).sum(axis=1), index=w.index)

    cost = cost.reindex(gross_returns.index).fillna(0.0)
    return gross_returns - cost, cost


def break_even_cost_bps(
    gross_returns: pd.Series,
    turnover_per_period: float,
) -> float:
    """Round-trip cost, in basis points, at which mean net return reaches zero.

    The single most useful number in the economic section: a strategy whose break-even
    cost is below realistic execution costs does not exist, however good its gross
    Sharpe looks.
    """
    r = pd.Series(gross_returns).dropna()
    if r.empty or turnover_per_period <= _EPS:
        return np.nan
    return float(r.mean() / turnover_per_period * 1e4)
