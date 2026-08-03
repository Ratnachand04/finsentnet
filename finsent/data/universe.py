"""Point-in-time universe construction — the first thing a referee checks.

Pulling "S&P 500 constituents" from a current source and applying that list to
2016-2023, as V2 did, is survivorship bias. Roughly 25-30 names leave the index each
year and the leavers are disproportionately the losers, so a backtest on today's members
is a backtest on companies that were selected for having survived. The effect is large
enough on its own to manufacture a respectable Sharpe ratio out of nothing.

Two defensible modes are provided:

``pit_index``
    Reconstruct membership from a dated changes file (additions and deletions with
    effective dates). Use this when you have CRSP/Compustat or a curated changes table.

``pit_liquidity``  (default)
    Top ``n_names`` by trailing median dollar volume, rebalanced monthly, with a price
    floor. Fully reproducible from free daily data, contains no forward information, and
    is honest: the paper says "a liquidity-screened universe", not "the S&P 500".

Either way the acceptance test is the same and lives in ``tests/test_universe.py``: the
universe on an early date must contain names absent from the final date. If it does not,
survivorship bias is still present.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["UniverseConfig", "build_liquidity_universe", "build_index_universe", "membership_stats"]


@dataclass(frozen=True)
class UniverseConfig:
    n_names: int = 400
    min_price: float = 5.0
    min_adv_usd: float = 5.0e6
    lookback_days: int = 60
    rebalance: str = "monthly"


def _rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq == "daily":
        return dates
    period = {"monthly": "M", "quarterly": "Q", "weekly": "W"}.get(freq, "M")
    series = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(series.groupby(dates.to_period(period)).last().to_numpy())


def build_liquidity_universe(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    cfg: UniverseConfig | None = None,
) -> pd.DataFrame:
    """Monthly-rebalanced liquidity screen, evaluated causally.

    ``close`` and ``volume`` are wide frames indexed by session date with one column per
    ticker; delisted names simply stop having data, which is the behaviour that makes
    the screen point-in-time. The dollar-volume statistic at a rebalance date uses the
    trailing ``lookback_days`` window and **not** the rebalance day itself, so no
    same-day information enters the selection.

    Returns a long frame ``[date, ticker, in_universe]`` covering every session.
    """
    cfg = cfg or UniverseConfig()
    close = close.sort_index()
    volume = volume.reindex_like(close)

    dollar_volume = (close * volume).shift(1)
    adv = dollar_volume.rolling(cfg.lookback_days, min_periods=cfg.lookback_days // 2).median()
    price_ok = close.shift(1) >= cfg.min_price

    rebal = _rebalance_dates(close.index, cfg.rebalance)
    selections: dict[pd.Timestamp, set[str]] = {}
    for date in rebal:
        row = adv.loc[date]
        eligible = row[(row >= cfg.min_adv_usd) & price_ok.loc[date].fillna(False)]
        selections[date] = set(eligible.nlargest(cfg.n_names).index)

    rows = []
    current: set[str] = set()
    rebal_set = set(rebal)
    for date in close.index:
        if date in rebal_set:
            current = selections[date]
        for ticker in current:
            rows.append((date, ticker, True))

    return pd.DataFrame(rows, columns=["date", "ticker", "in_universe"])


def build_index_universe(
    changes: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    initial_members: list[str] | None = None,
) -> pd.DataFrame:
    """Reconstruct index membership from a dated additions/deletions table.

    ``changes`` needs columns ``effective_date``, ``ticker`` and ``action`` with values
    ``add`` or ``remove``. Membership is applied from the *effective* date forward, and a
    name removed on date ``d`` is a member through ``d - 1`` -- the convention that
    keeps a deleted name's final, usually poor, returns inside the sample where they
    belong.
    """
    required = {"effective_date", "ticker", "action"}
    missing = required - set(changes.columns)
    if missing:
        raise ValueError(f"changes table missing columns: {sorted(missing)}")

    ch = changes.copy()
    ch["effective_date"] = pd.to_datetime(ch["effective_date"])
    ch = ch.sort_values("effective_date")

    sessions = pd.DatetimeIndex(pd.to_datetime(sessions)).sort_values()
    members = set(initial_members or [])

    rows = []
    pointer = 0
    events = list(ch.itertuples(index=False))
    for date in sessions:
        while pointer < len(events) and events[pointer].effective_date <= date:
            ev = events[pointer]
            if str(ev.action).lower() == "add":
                members.add(str(ev.ticker))
            else:
                members.discard(str(ev.ticker))
            pointer += 1
        for ticker in members:
            rows.append((date, ticker, True))

    return pd.DataFrame(rows, columns=["date", "ticker", "in_universe"])


def membership_stats(universe: pd.DataFrame) -> dict[str, object]:
    """Turnover statistics plus the survivorship check the paper must report.

    ``survivorship_check_passed`` is False when every early member is still present at
    the end, which is the signature of a retroactively applied constituent list.
    """
    if universe.empty:
        return {"n_dates": 0, "survivorship_check_passed": False}

    u = universe.copy()
    u["date"] = pd.to_datetime(u["date"])
    by_date = u.groupby("date")["ticker"].apply(set)

    first, last = by_date.iloc[0], by_date.iloc[-1]
    dropped = first - last

    sizes = by_date.map(len)
    entries = [
        len(by_date.iloc[i] - by_date.iloc[i - 1]) for i in range(1, len(by_date))
    ]

    return {
        "n_dates": int(len(by_date)),
        "n_unique_tickers": int(u["ticker"].nunique()),
        "mean_size": float(sizes.mean()),
        "min_size": int(sizes.min()),
        "max_size": int(sizes.max()),
        "n_dropped_from_first_date": int(len(dropped)),
        "mean_daily_entries": float(np.mean(entries)) if entries else 0.0,
        "survivorship_check_passed": bool(len(dropped) >= 5),
    }
