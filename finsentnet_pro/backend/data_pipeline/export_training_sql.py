"""
FINSENT NET PRO - Training Data SQL Exporter

Exports training-ready data into SQL text files (one file per company).
Each file contains:
  - company_metadata
  - ohlcv_daily
  - technical_indicators
  - news_articles
  - sentiment_summary

Data sources:
  - MarketDataFetcher (FMP -> yfinance -> synthetic fallback)
  - TechnicalIndicators
  - LiveDataService news providers
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import yfinance as yf

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from data_pipeline.market_data_fetcher import MarketDataFetcher
from data_pipeline.technical_indicators import TechnicalIndicators
from data_pipeline.live_data_service import LiveDataService
from data_pipeline.news_sentiment_engine import NewsSentimentEngine


ALL_MARKETS = ["SP500", "NASDAQ", "NYSE", "NSE", "BSE", "CRYPTO", "COMMODITIES"]


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _sql_value(value):
    if value is None:
        return "NULL"
    if isinstance(value, float):
        if pd.isna(value):
            return "NULL"
        return repr(float(value))
    if isinstance(value, int):
        return str(value)
    if hasattr(value, "isoformat"):
        return "'" + str(value.isoformat()).replace("'", "''") + "'"
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def _create_table_sql(table_name: str, columns: List[Tuple[str, str]]) -> str:
    col_defs = ",\n  ".join([f'"{name}" {dtype}' for name, dtype in columns])
    return f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n  {col_defs}\n);\n'


def _insert_sql(table_name: str, row: Dict) -> str:
    columns = [f'"{c}"' for c in row.keys()]
    values = [_sql_value(v) for v in row.values()]
    return (
        f'INSERT INTO "{table_name}" ({", ".join(columns)}) '
        f'VALUES ({", ".join(values)});\n'
    )


def _build_market_universe(fetcher: MarketDataFetcher, markets: Iterable[str], limit_per_market: int) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for market in markets:
        if market not in ALL_MARKETS:
            continue

        # 0 means "all available" from the fetcher list for that market.
        request_limit = 10000 if limit_per_market == 0 else limit_per_market
        tickers = fetcher.get_market_components(market, limit=request_limit)

        # Deduplicate preserving order.
        seen = set()
        for ticker in tickers:
            if ticker not in seen:
                pairs.append((ticker, market))
                seen.add(ticker)
    return pairs


def _fetch_company_profile(ticker: str, market: str) -> Dict:
    suffix_map = {"NSE": ".NS", "BSE": ".BO", "CRYPTO": "-USD"}
    full_ticker = ticker
    suffix = suffix_map.get(market, "")
    if suffix and not full_ticker.endswith(suffix):
        full_ticker = f"{full_ticker}{suffix}"

    profile = {
        "company_name": ticker,
        "exchange": market,
        "sector": None,
        "industry": None,
        "country": None,
        "currency": None,
        "market_cap": None,
        "employees": None,
        "website": None,
        "source": "yfinance",
    }

    try:
        info = yf.Ticker(full_ticker).info or {}
        profile.update({
            "company_name": info.get("longName") or info.get("shortName") or ticker,
            "exchange": info.get("exchange") or market,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "currency": info.get("currency"),
            "market_cap": info.get("marketCap"),
            "employees": info.get("fullTimeEmployees"),
            "website": info.get("website"),
        })
    except Exception:
        pass
    return profile


def _write_company_sql(
    output_dir: Path,
    ticker: str,
    market: str,
    period: str,
    ohlcv_df: pd.DataFrame,
    features_df: pd.DataFrame,
    news_payload: Dict,
    sentiment_engine: NewsSentimentEngine,
    company_profile: Dict,
):
    now_iso = datetime.now(timezone.utc).isoformat()
    fname = _safe_filename(f"{ticker}_{market}")
    sql_path = output_dir / f"{fname}.sql"

    titles = []
    for article in news_payload.get("articles", []):
        title = str(article.get("title", "")).strip()
        summary = str(article.get("summary", "")).strip()
        text = (title + " " + summary).strip()
        if text:
            titles.append(text)

    sentiment_scores = sentiment_engine.score_batch(titles) if titles else []
    sentiment_summary = sentiment_engine.aggregate_sentiment(sentiment_scores)

    ohlcv_cols = ["ticker", "market", "trade_date", "open", "high", "low", "close", "volume"]

    feature_cols = ["ticker", "market", "trade_date"] + [str(c) for c in features_df.columns]

    metadata_cols = [
        ("ticker", "TEXT"),
        ("market", "TEXT"),
        ("period", "TEXT"),
        ("exported_at_utc", "TEXT"),
        ("ohlcv_rows", "INTEGER"),
        ("feature_rows", "INTEGER"),
        ("news_rows", "INTEGER"),
        ("news_source", "TEXT"),
    ]
    ohlcv_schema = [
        ("ticker", "TEXT"),
        ("market", "TEXT"),
        ("trade_date", "TEXT"),
        ("open", "REAL"),
        ("high", "REAL"),
        ("low", "REAL"),
        ("close", "REAL"),
        ("volume", "REAL"),
    ]
    feature_schema = [("ticker", "TEXT"), ("market", "TEXT"), ("trade_date", "TEXT")] + [
        (str(c), "REAL") for c in features_df.columns
    ]
    news_schema = [
        ("ticker", "TEXT"),
        ("market", "TEXT"),
        ("published_at", "TEXT"),
        ("source", "TEXT"),
        ("title", "TEXT"),
        ("summary", "TEXT"),
        ("url", "TEXT"),
    ]
    sentiment_schema = [
        ("ticker", "TEXT"),
        ("market", "TEXT"),
        ("score", "REAL"),
        ("label", "TEXT"),
        ("confidence", "REAL"),
        ("count", "INTEGER"),
        ("positive_pct", "REAL"),
        ("negative_pct", "REAL"),
        ("exported_at_utc", "TEXT"),
    ]
    profile_schema = [
        ("ticker", "TEXT"),
        ("market", "TEXT"),
        ("company_name", "TEXT"),
        ("exchange", "TEXT"),
        ("sector", "TEXT"),
        ("industry", "TEXT"),
        ("country", "TEXT"),
        ("currency", "TEXT"),
        ("market_cap", "REAL"),
        ("employees", "INTEGER"),
        ("website", "TEXT"),
        ("source", "TEXT"),
        ("exported_at_utc", "TEXT"),
    ]

    lines: List[str] = []
    lines.append("BEGIN TRANSACTION;\n")
    lines.append(_create_table_sql("company_metadata", metadata_cols))
    lines.append(_create_table_sql("ohlcv_daily", ohlcv_schema))
    lines.append(_create_table_sql("technical_indicators", feature_schema))
    lines.append(_create_table_sql("news_articles", news_schema))
    lines.append(_create_table_sql("sentiment_summary", sentiment_schema))
    lines.append(_create_table_sql("company_profile", profile_schema))

    lines.append(
        _insert_sql(
            "company_metadata",
            {
                "ticker": ticker,
                "market": market,
                "period": period,
                "exported_at_utc": now_iso,
                "ohlcv_rows": int(len(ohlcv_df)),
                "feature_rows": int(len(features_df)),
                "news_rows": int(len(news_payload.get("articles", []))),
                "news_source": str(news_payload.get("articles", [{}])[0].get("source", "")) if news_payload.get("articles") else "",
            },
        )
    )

    for idx, row in ohlcv_df.iterrows():
        payload = {
            "ticker": ticker,
            "market": market,
            "trade_date": pd.Timestamp(idx).date().isoformat(),
            "open": float(row.get("Open", 0.0)),
            "high": float(row.get("High", 0.0)),
            "low": float(row.get("Low", 0.0)),
            "close": float(row.get("Close", 0.0)),
            "volume": float(row.get("Volume", 0.0)),
        }
        lines.append(_insert_sql("ohlcv_daily", payload))

    for idx, row in features_df.iterrows():
        payload = {
            "ticker": ticker,
            "market": market,
            "trade_date": pd.Timestamp(idx).date().isoformat(),
        }
        for col in features_df.columns:
            payload[str(col)] = row.get(col)
        lines.append(_insert_sql("technical_indicators", payload))

    for article in news_payload.get("articles", []):
        lines.append(
            _insert_sql(
                "news_articles",
                {
                    "ticker": ticker,
                    "market": market,
                    "published_at": article.get("published_at"),
                    "source": article.get("source"),
                    "title": article.get("title"),
                    "summary": article.get("summary"),
                    "url": article.get("url"),
                },
            )
        )

    lines.append(
        _insert_sql(
            "sentiment_summary",
            {
                "ticker": ticker,
                "market": market,
                "score": sentiment_summary.get("score", 0.0),
                "label": sentiment_summary.get("label", "neutral"),
                "confidence": sentiment_summary.get("confidence", 0.0),
                "count": sentiment_summary.get("count", 0),
                "positive_pct": sentiment_summary.get("positive_pct", 0.0),
                "negative_pct": sentiment_summary.get("negative_pct", 0.0),
                "exported_at_utc": now_iso,
            },
        )
    )

    lines.append(
        _insert_sql(
            "company_profile",
            {
                "ticker": ticker,
                "market": market,
                "company_name": company_profile.get("company_name"),
                "exchange": company_profile.get("exchange"),
                "sector": company_profile.get("sector"),
                "industry": company_profile.get("industry"),
                "country": company_profile.get("country"),
                "currency": company_profile.get("currency"),
                "market_cap": company_profile.get("market_cap"),
                "employees": company_profile.get("employees"),
                "website": company_profile.get("website"),
                "source": company_profile.get("source", "yfinance"),
                "exported_at_utc": now_iso,
            },
        )
    )

    lines.append("COMMIT;\n")
    sql_path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Export training data to SQL files (one per company).")
    parser.add_argument("--markets", nargs="+", default=ALL_MARKETS, help="Markets to export")
    parser.add_argument("--period", default="max", help="Historical period (default: max)")
    parser.add_argument(
        "--output-dir",
        default="finsentnet_pro/backend/data/sql_training_companies",
        help="Output folder for SQL files",
    )
    parser.add_argument(
        "--limit-per-market",
        type=int,
        default=0,
        help="Per-market ticker limit (0 means all available)",
    )
    parser.add_argument(
        "--max-news",
        type=int,
        default=50,
        help="Maximum news records per company",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fetcher = MarketDataFetcher()
    indicators = TechnicalIndicators()
    live_data = LiveDataService()
    sentiment = NewsSentimentEngine(use_finbert=False)

    universe = _build_market_universe(fetcher, args.markets, args.limit_per_market)
    if not universe:
        print("[EXPORT] No tickers found for selected markets.")
        return

    print(f"[EXPORT] Output directory: {output_dir}")
    print(f"[EXPORT] Total companies to export: {len(universe)}")

    success = 0
    failed = 0

    for idx, (ticker, market) in enumerate(universe, start=1):
        try:
            print(f"[{idx}/{len(universe)}] Exporting {ticker} ({market}) ...")

            ohlcv_df = fetcher.fetch_ohlcv(ticker=ticker, market=market, period=args.period, interval="1d")
            if ohlcv_df is None or ohlcv_df.empty:
                raise ValueError("No OHLCV data returned")

            features_df = indicators.compute_all(ohlcv_df.copy())
            if features_df is None or features_df.empty:
                raise ValueError("No technical indicators generated")

            news_payload = live_data.get_live_news(ticker=ticker, market=market)
            news_payload["articles"] = (news_payload.get("articles") or [])[: args.max_news]
            company_profile = _fetch_company_profile(ticker=ticker, market=market)

            _write_company_sql(
                output_dir=output_dir,
                ticker=ticker,
                market=market,
                period=args.period,
                ohlcv_df=ohlcv_df,
                features_df=features_df,
                news_payload=news_payload,
                sentiment_engine=sentiment,
                company_profile=company_profile,
            )

            success += 1
        except Exception as exc:
            failed += 1
            print(f"[WARN] Failed {ticker} ({market}): {exc}")

    print("\n[EXPORT] Completed")
    print(f"[EXPORT] Success: {success}")
    print(f"[EXPORT] Failed: {failed}")
    print(f"[EXPORT] SQL files stored in: {output_dir}")


if __name__ == "__main__":
    main()
