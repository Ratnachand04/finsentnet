"""
FINSENT NET PRO — Market Data Routes
/api/market/* endpoints for price data, search, and trending stocks.
"""

from fastapi import APIRouter, HTTPException, Query
import logging
import time

router = APIRouter(prefix="/api/market", tags=["Market Data"])
logger = logging.getLogger("finsent.routes.market")

# Singletons are injected from main.py via app.state
_fetcher = None
_indicators = None


def init(fetcher, indicators):
    """Initialize route with shared service instances."""
    global _fetcher, _indicators
    _fetcher = fetcher
    _indicators = indicators


@router.get("/search")
async def search_stocks(q: str = Query(..., min_length=1), limit: int = 20):
    """Search for tickers across all supported markets."""
    results = []
    query = q.upper().strip()
    for market, cfg in _fetcher.MARKET_CONFIG.items():
        components = _fetcher.get_market_components(market)
        for ticker in components:
            if query in ticker.upper():
                results.append({
                    "ticker": ticker,
                    "market": market,
                    "currency": cfg.get("currency", "USD"),
                })
                if len(results) >= limit:
                    return {"results": results, "query": q}
    return {"results": results, "query": q}


@router.get("/quote/{ticker}")
async def get_quote(ticker: str, market: str = "SP500"):
    """Get current quote for a ticker."""
    try:
        price_data = _fetcher.fetch_ohlcv(ticker, market=market, period="5d")
        if price_data.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        latest = price_data.iloc[-1]
        prev = price_data.iloc[-2] if len(price_data) > 1 else latest
        change = latest["Close"] - prev["Close"]
        change_pct = (change / prev["Close"]) * 100 if prev["Close"] > 0 else 0

        return {
            "ticker": ticker,
            "market": market,
            "price": round(float(latest["Close"]), 2),
            "change": round(float(change), 2),
            "change_pct": round(float(change_pct), 2),
            "volume": int(latest.get("Volume", 0)),
            "high": round(float(latest["High"]), 2),
            "low": round(float(latest["Low"]), 2),
            "open": round(float(latest["Open"]), 2),
            "week_52_high": round(float(price_data["High"].max()), 2),
            "week_52_low": round(float(price_data["Low"].min()), 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quote error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chart/{ticker}")
async def get_chart_data(
    ticker: str,
    market: str = "SP500",
    period: str = "6mo",
    interval: str = "1d",
):
    """Get OHLCV data with technical indicators for charting."""
    try:
        df = _fetcher.fetch_ohlcv(ticker, market=market, period=period, interval=interval)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No chart data for {ticker}")

        df_ind = _indicators.compute_all(df)

        candles = []
        for idx, row in df_ind.iterrows():
            ts = int(idx.timestamp()) if hasattr(idx, "timestamp") else int(time.time())
            candles.append({
                "time": ts,
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row.get("Volume", 0)),
                "rsi": round(float(row.get("RSI_14", 50)), 2),
                "macd": round(float(row.get("MACD", 0)), 4),
                "macd_signal": round(float(row.get("MACD_Signal", 0)), 4),
                "macd_hist": round(float(row.get("MACD_Hist", 0)), 4),
                "bb_upper": round(float(row.get("BB_Upper", row["Close"])), 2),
                "bb_lower": round(float(row.get("BB_Lower", row["Close"])), 2),
                "ema_20": round(float(row.get("EMA_20", row["Close"])), 2),
                "ema_50": round(float(row.get("EMA_50", row["Close"])), 2),
            })

        return {
            "ticker": ticker,
            "market": market,
            "candles": candles[-200:],
            "period": period,
            "interval": interval,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trending")
async def get_trending(market: str = "SP500", limit: int = 10):
    """Get trending / most active stocks with live prices."""
    components = _fetcher.get_market_components(market)[:limit]
    trending_list = []
    for ticker in components:
        try:
            df = _fetcher.fetch_ohlcv(ticker, market=market, period="5d")
            if df.empty:
                continue
            latest, prev = df.iloc[-1], (df.iloc[-2] if len(df) > 1 else df.iloc[-1])
            change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
            trending_list.append({
                "ticker": ticker,
                "price": round(float(latest["Close"]), 2),
                "change_pct": round(float(change_pct), 2),
            })
        except Exception:
            continue
    return {"market": market, "trending": trending_list}
