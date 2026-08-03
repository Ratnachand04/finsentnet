"""News-to-decision timestamp alignment — where most hidden leaks live.

The contract, stated once and enforced everywhere
-------------------------------------------------
A headline published at ``pub_ts`` may inform the decision executed at the open of
session ``t`` if and only if::

    open_ts(t) >= pub_ts + min_lag_hours                                       (1)

and returns are measured **open-to-open**, from ``open(t)`` to ``open(t + h)``.

Three worked examples, all with ``min_lag_hours = 24`` and a 09:30 US/Eastern open:

===================================  ==================  ==========================
publication (US/Eastern)             decision open       return window
===================================  ==================  ==========================
Tue 09:00 (before Tuesday's open)    **Wed 09:30**       Wed open -> Wed+h open
Tue 16:30 (after Tuesday's close)    **Thu 09:30**       Thu open -> Thu+h open
Sat 03:00 (weekend)                  **Mon 09:30**       Mon open -> Mon+h open
===================================  ==================  ==========================

Note the second row: a 16:30 headline is *not* usable at Wednesday's 09:30 open, because
only 17 hours have elapsed. The 24-hour lag is deliberately conservative; it buys
immunity to timestamp errors in the news vendor's feed, which are common and which
silently manufacture predictive power when the lag is tight.

Why open-to-open and not close-to-close
---------------------------------------
Labelling close-to-close while nominally "deciding at t" gives the model the last few
hours of the decision day, a window in which the news it just read is already being
priced. That is the single most common hidden leak in this literature, and it inflates
reported accuracy by several points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "AlignmentConfig",
    "decision_session",
    "align_news_to_sessions",
    "assert_no_lookahead",
    "news_coverage_stats",
]

DEFAULT_TZ = "America/New_York"
DEFAULT_OPEN = "09:30"


@dataclass(frozen=True)
class AlignmentConfig:
    """Parameters of the alignment contract (mirrors ``data.news`` in the config)."""

    min_lag_hours: int = 24
    lookback_hours: int = 72
    max_headlines: int = 8
    exchange_tz: str = DEFAULT_TZ
    open_time: str = DEFAULT_OPEN

    def session_opens(self, sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Timezone-aware opening timestamps for a set of session dates."""
        dates = pd.DatetimeIndex(pd.to_datetime(sessions)).tz_localize(None).normalize()
        hh, mm = (int(v) for v in self.open_time.split(":"))
        naive = dates + pd.Timedelta(hours=hh, minutes=mm)
        return naive.tz_localize(self.exchange_tz, nonexistent="shift_forward",
                                 ambiguous=True)


def decision_session(
    pub_ts: pd.Series | pd.DatetimeIndex,
    sessions: pd.DatetimeIndex,
    cfg: AlignmentConfig | None = None,
) -> pd.Series:
    """First session whose open satisfies inequality (1), per headline.

    Returns a Series of session dates (``NaT`` when the headline is later than the last
    available session). This function is the *only* place the lag rule is implemented;
    every other module calls it.
    """
    cfg = cfg or AlignmentConfig()
    ts = pd.DatetimeIndex(pd.to_datetime(pd.Series(pub_ts), utc=True))
    opens = cfg.session_opens(sessions).tz_convert("UTC")

    earliest = ts + pd.Timedelta(hours=cfg.min_lag_hours)
    pos = np.searchsorted(opens.to_numpy(), earliest.to_numpy(), side="left")

    out = np.full(len(ts), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = pos < len(sessions)
    session_dates = pd.DatetimeIndex(pd.to_datetime(sessions)).normalize().to_numpy()
    out[valid] = session_dates[pos[valid]]
    return pd.Series(out, name="decision_date")


def align_news_to_sessions(
    news: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    cfg: AlignmentConfig | None = None,
    ts_column: str = "published_at",
    ticker_column: str = "ticker",
) -> pd.DataFrame:
    """Attach a decision date and a recency to every headline, then cap per (name, day).

    The returned frame adds:

    ``decision_date``  session whose open first satisfies (1)
    ``lag_hours``      hours from publication to that open (always ``>= min_lag_hours``)
    ``rank``           recency rank within the (ticker, decision_date) group, 0 = newest

    Headlines older than ``lookback_hours`` at the decision open are dropped, and only
    the ``max_headlines`` most recent survive -- ``K`` in the model's text branch.
    """
    cfg = cfg or AlignmentConfig()
    if news.empty:
        return news.assign(decision_date=pd.NaT, lag_hours=np.nan, rank=np.nan)

    df = news.copy()
    df[ts_column] = pd.to_datetime(df[ts_column], utc=True)
    df["decision_date"] = decision_session(df[ts_column], sessions, cfg).to_numpy()
    df = df[df["decision_date"].notna()].copy()
    if df.empty:
        return df.assign(lag_hours=np.nan, rank=np.nan)

    opens = cfg.session_opens(pd.DatetimeIndex(df["decision_date"])).tz_convert("UTC")
    df["lag_hours"] = (opens - pd.DatetimeIndex(df[ts_column])).total_seconds() / 3600.0

    df = df[df["lag_hours"] >= cfg.min_lag_hours - 1e-9]
    df = df[df["lag_hours"] <= cfg.lookback_hours]

    df = df.sort_values([ticker_column, "decision_date", "lag_hours"])
    df["rank"] = df.groupby([ticker_column, "decision_date"]).cumcount()
    return df[df["rank"] < cfg.max_headlines].reset_index(drop=True)


def assert_no_lookahead(
    aligned: pd.DataFrame,
    cfg: AlignmentConfig | None = None,
    ts_column: str = "published_at",
) -> None:
    """Raise if any aligned headline violates inequality (1).

    Called by ``tests/test_align.py`` and by the dataset builder. Cheap enough to run
    every time, and it catches the class of bug that no metric will reveal.
    """
    cfg = cfg or AlignmentConfig()
    if aligned.empty:
        return
    bad = aligned["lag_hours"] < cfg.min_lag_hours - 1e-6
    if bad.any():
        first = aligned[bad].iloc[0]
        raise AssertionError(
            f"LOOKAHEAD: headline at {first[ts_column]} assigned to decision "
            f"{first['decision_date']} with lag {first['lag_hours']:.2f}h "
            f"< min_lag_hours={cfg.min_lag_hours}"
        )


def news_coverage_stats(
    aligned: pd.DataFrame,
    panel_keys: pd.DataFrame,
    ticker_column: str = "ticker",
) -> dict[str, float]:
    """Coverage of the (ticker, day) panel by news — a mandatory Table 1 row.

    Most ticker-days carry no relevant headline. A reviewer will assume you silently
    dropped those days unless you report this, and dropping them would be a selection
    bias, because news days are systematically higher-volatility.
    """
    keys = panel_keys[[ticker_column, "date"]].drop_duplicates()
    if aligned.empty:
        return {"coverage": 0.0, "mean_headlines": 0.0, "panel_rows": int(len(keys))}

    counts = (
        aligned.groupby([ticker_column, "decision_date"]).size().rename("n_headlines").reset_index()
    )
    merged = keys.merge(
        counts, left_on=[ticker_column, "date"], right_on=[ticker_column, "decision_date"],
        how="left",
    )
    n = merged["n_headlines"].fillna(0.0)
    return {
        "coverage": float((n > 0).mean()),
        "mean_headlines": float(n.mean()),
        "mean_headlines_when_present": float(n[n > 0].mean()) if (n > 0).any() else 0.0,
        "max_headlines": float(n.max()),
        "panel_rows": int(len(keys)),
    }
