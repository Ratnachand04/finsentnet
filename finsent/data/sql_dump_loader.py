"""Loader for the exported SQL price dumps.

The repository ships daily OHLCV for 227 unique US tickers as SQL ``INSERT``
statements, one file per (ticker, market). This module parses them into a tidy
panel and caches the result as parquet, so the walk-forward study reads a 40 MB
file instead of re-parsing 5.2 GB.

Scope decisions, made once and recorded here rather than buried in a script
------------------------------------------------------------------------------
**US markets only.** The dumps also contain NSE and BSE listings, crypto and
commodities. Indian venues have a different settlement regime, lot sizes,
circuit breakers and an STT/stamp-duty cost structure that the cost model in
``finsent.decision.costs`` does not represent; crypto trades continuously, which
breaks the daily-bar convention, the news lag and the 252-day annualisation
simultaneously. Mixing them into one panel would make the reported statistics
uninterpretable, so they are excluded and their exclusion is stated.

**One row per (ticker, date).** A ticker may appear in several market files ---
AAPL is present as NASDAQ, NYSE and SP500 --- with identical bars. We keep the
listing with the most observations and drop the rest, so the cross-section is
not silently triple-weighted on the mega-caps.

**Prices are adjusted.** The dumps carry dividend- and split-adjusted closes.
This is what makes returns comparable across corporate actions; it also means
the series is back-adjusted, so an indicator computed on it embeds knowledge of
later corporate actions. That is a known limitation of adjusted series and is
recorded in the manuscript rather than papered over.

**The news tables are unusable.** Each dump has a ``news_articles`` table, but
of 5{,}391 rows across the corpus only 72 are non-empty, and those 72 are
generated template strings from a single synthetic source bearing one identical
timestamp. ``sentiment_summary`` is uniformly zero with ``count = 0``. There is
therefore no text modality available from this data, and none is fabricated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["DumpConfig", "parse_dump", "load_price_panel", "news_audit"]

US_MARKETS = ("NYSE", "NASDAQ", "SP500")

_OHLCV = re.compile(
    r"""INSERT\s+INTO\s+"ohlcv_daily".*?VALUES\s*\(\s*
        '(?P<ticker>[^']*)'\s*,\s*
        '(?P<market>[^']*)'\s*,\s*
        '(?P<date>\d{4}-\d{2}-\d{2})'\s*,\s*
        (?P<open>-?[\d.]+(?:[eE][-+]?\d+)?)\s*,\s*
        (?P<high>-?[\d.]+(?:[eE][-+]?\d+)?)\s*,\s*
        (?P<low>-?[\d.]+(?:[eE][-+]?\d+)?)\s*,\s*
        (?P<close>-?[\d.]+(?:[eE][-+]?\d+)?)\s*,\s*
        (?P<volume>-?[\d.]+(?:[eE][-+]?\d+)?)\s*\)""",
    re.VERBOSE,
)

_NEWS = re.compile(
    r"""INSERT\s+INTO\s+"news_articles".*?VALUES\s*\(\s*
        '(?P<ticker>[^']*)'\s*,\s*'(?P<market>[^']*)'\s*,\s*
        '(?P<published_at>[^']*)'\s*,\s*'(?P<source>[^']*)'\s*,\s*
        '(?P<title>[^']*)'""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class DumpConfig:
    root: Path = Path("finsentnet_pro/backend/data/sql_training_companies")
    markets: tuple[str, ...] = US_MARKETS
    start: str = "2014-01-01"      # early enough to warm up 252-day features
    end: str = "2025-12-31"        # last complete calendar year in the dumps
    min_sessions: int = 1500       # ~6 years; shorter series cannot span folds
    cache: Path = Path("data/cache/price_panel.parquet")


def parse_dump(path: Path) -> pd.DataFrame:
    """Parse one SQL file's ``ohlcv_daily`` rows into a frame."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = _OHLCV.findall(text)
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(
        rows, columns=["ticker", "market", "date", "open", "high", "low", "close", "volume"]
    )
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def news_audit(cfg: DumpConfig | None = None) -> dict[str, int]:
    """Count usable headlines. Reported in the manuscript, not assumed.

    A text modality is only admissible if this returns a non-trivial number of
    distinct, non-empty, differently-timestamped headlines. On this corpus it
    does not.
    """
    cfg = cfg or DumpConfig()
    total = nonempty = 0
    titles: set[str] = set()
    stamps: set[str] = set()

    for path in sorted(cfg.root.glob("*.sql")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in _NEWS.finditer(text):
            total += 1
            title = m.group("title").strip()
            if title:
                nonempty += 1
                titles.add(title)
                stamps.add(m.group("published_at"))

    return {
        "rows": total,
        "non_empty": nonempty,
        "distinct_titles": len(titles),
        "distinct_timestamps": len(stamps),
    }


def load_price_panel(cfg: DumpConfig | None = None, rebuild: bool = False) -> pd.DataFrame:
    """Tidy ``[date, ticker, open, high, low, close, volume]`` panel, cached.

    Deduplicates a ticker that appears under several markets by keeping the
    listing with the most observations, so mega-caps present in three files do
    not enter the cross-section three times.
    """
    cfg = cfg or DumpConfig()
    if cfg.cache.exists() and not rebuild:
        return pd.read_parquet(cfg.cache)

    files = [p for p in sorted(cfg.root.glob("*.sql"))
             if p.stem.rsplit("_", 1)[-1] in cfg.markets]
    if not files:
        raise FileNotFoundError(f"no dumps under {cfg.root} for markets {cfg.markets}")

    frames = []
    for i, path in enumerate(files, 1):
        frame = parse_dump(path)
        if not frame.empty:
            frames.append(frame)
        if i % 50 == 0:
            print(f"  parsed {i}/{len(files)} files", flush=True)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["date", "close"])
    panel = panel[(panel["date"] >= cfg.start) & (panel["date"] <= cfg.end)]
    panel = panel[(panel[["open", "high", "low", "close"]] > 0).all(axis=1)]
    panel = panel[panel["volume"] >= 0]

    # Keep, per ticker, the market listing with the most observations.
    counts = panel.groupby(["ticker", "market"]).size().rename("n").reset_index()
    best = counts.sort_values("n", ascending=False).drop_duplicates("ticker")[["ticker", "market"]]
    panel = panel.merge(best, on=["ticker", "market"], how="inner")

    panel = panel.drop_duplicates(subset=["ticker", "date"], keep="first")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Drop series too short to span the walk-forward protocol.
    lengths = panel.groupby("ticker")["date"].size()
    keep = lengths[lengths >= cfg.min_sessions].index
    panel = panel[panel["ticker"].isin(keep)].reset_index(drop=True)

    cfg.cache.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cfg.cache, index=False)
    return panel


def panel_to_wide(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    """Wide ``date x ticker`` frame for one field."""
    return panel.pivot_table(index="date", columns="ticker", values=field, aggfunc="last")


def panel_to_ohlcv_dict(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-ticker OHLCV frames indexed by date, as the feature builder expects."""
    out: dict[str, pd.DataFrame] = {}
    for ticker, grp in panel.groupby("ticker", sort=True):
        frame = grp.set_index("date")[["open", "high", "low", "close", "volume"]]
        out[str(ticker)] = frame.sort_index()
    return out
