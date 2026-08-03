"""
FINSENT NET PRO — Live Data & Prediction Routes
/api/live/* — Real-time market data, candle charts, news, and predictions.
"""

import logging
import os
import time
import asyncio
import requests
from typing import Optional, List, Dict, Union
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from api.websocket_handler import get_tunnel_status

router = APIRouter(prefix="/api/live", tags=["Live Data"])
logger = logging.getLogger("finsent.routes.live")

# ── Singletons (injected from main.py) ─────────────
_live_service = None
_predictor = None
_trainer = None
_fetcher = None
_snapshot_cache: Dict[str, Dict] = {}
_snapshot_ttl_seconds = 20
_snapshot_max_workers = max(4, int(os.getenv("LIVE_SNAPSHOT_MAX_WORKERS", "12")))
_snapshot_executor = ThreadPoolExecutor(max_workers=_snapshot_max_workers)
_news_timeout_seconds = max(1.0, float(os.getenv("LIVE_NEWS_ENDPOINT_TIMEOUT_SECONDS", "4")))
_news_executor = ThreadPoolExecutor(max_workers=max(2, int(os.getenv("LIVE_NEWS_MAX_WORKERS", "6"))))


def init(live_service, predictor, trainer, fetcher=None):
    """Initialize route with shared services."""
    global _live_service, _predictor, _trainer, _fetcher
    _live_service = live_service
    _predictor = predictor
    _trainer = trainer
    _fetcher = fetcher


# ═══════════════════════════════════════════════════════
#  Request Models
# ═══════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    ticker: str
    market: str = "SP500"
    capital: float = Field(default=100000, ge=1000)
    risk_tolerance: float = Field(default=0.5, ge=0.1, le=1.0)


class ApiKeyRequest(BaseModel):
    fmp_keys: Optional[List[str]] = None
    alpha_vantage: Optional[str] = None
    finnhub: Optional[str] = None
    news_api: Optional[str] = None
    providers: Optional[Dict[str, Union[str, List[str]]]] = None


def _parse_bool_env(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(value: Optional[str], default: Optional[int]) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _grafana_reachable(base_url: str, timeout_s: float = 1.2) -> bool:
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        return False
    try:
        resp = requests.get(f"{url}/api/health", timeout=timeout_s)
        return resp.status_code == 200
    except Exception:
        return False


# ═══════════════════════════════════════════════════════
#  REAL-TIME QUOTE
# ═══════════════════════════════════════════════════════

@router.get("/quote/{ticker}")
async def get_realtime_quote(ticker: str, market: str = "SP500"):
    """
    Get real-time price quote for a ticker.
    Uses Finnhub (if API key set) → yfinance fallback.
    """
    if not _live_service:
        raise HTTPException(status_code=503, detail="Live data service not ready")

    try:
        quote = _live_service.get_realtime_quote(ticker.upper(), market)
        return {"status": "success", **quote}
    except Exception as e:
        logger.error(f"Quote error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-snapshot")
async def get_market_snapshot(
    markets: str = Query(default="SP500,NASDAQ,NYSE,NSE,BSE,CRYPTO,COMMODITIES"),
    per_market_limit: int = Query(default=8, ge=1, le=30),
):
    """
    Get live market snapshots with advance/decline ratios.

    Returns per-market stats and a list of quote samples suitable for ticker-strip UIs.
    Response is cached briefly to reduce API-key burn while keeping near-real-time updates.
    """
    if not _live_service or not _fetcher:
        raise HTTPException(status_code=503, detail="Live market services not ready")

    normalized_markets = [m.strip().upper() for m in markets.split(",") if m.strip()]
    cache_key = f"{','.join(normalized_markets)}|{per_market_limit}"
    now = time.time()
    cached = _snapshot_cache.get(cache_key)
    if cached and (now - cached.get("_ts", 0) <= _snapshot_ttl_seconds):
        return cached["payload"]

    valid_markets = [m for m in normalized_markets if m in _fetcher.MARKET_CONFIG]
    if not valid_markets:
        raise HTTPException(status_code=400, detail="No valid markets requested")

    market_rows = []
    quotes = []
    total_advancers = 0
    total_decliners = 0
    total_flat = 0

    def _fetch_quote_safe(ticker: str, market_name: str) -> Optional[Dict]:
        try:
            quote = _live_service.get_realtime_quote(ticker=ticker, market=market_name)
            pct = float(quote.get("change_pct", 0.0) or 0.0)
            return {
                "ticker": quote.get("ticker", ticker),
                "market": market_name,
                "price": quote.get("price"),
                "change_pct": round(pct, 2),
                "source": quote.get("source", "unknown"),
            }
        except Exception:
            return None

    for market in valid_markets:
        tickers = _fetcher.get_market_components(market, limit=per_market_limit)
        advancers = 0
        decliners = 0
        flat = 0
        sampled = []

        loop = asyncio.get_running_loop()
        jobs = [
            loop.run_in_executor(_snapshot_executor, _fetch_quote_safe, ticker, market)
            for ticker in tickers
        ]
        job_results = await asyncio.gather(*jobs, return_exceptions=True)
        for item in job_results:
            if isinstance(item, Exception) or item is None:
                continue
            pct = float(item.get("change_pct", 0.0) or 0.0)
            if pct > 0:
                advancers += 1
            elif pct < 0:
                decliners += 1
            else:
                flat += 1
            sampled.append(item)

        total = max(1, len(sampled))
        up_ratio = advancers / total
        down_ratio = decliners / total
        flat_ratio = flat / total
        avg_change_pct = (
            sum(float(q.get("change_pct", 0.0) or 0.0) for q in sampled) / total
            if sampled else 0.0
        )

        total_advancers += advancers
        total_decliners += decliners
        total_flat += flat

        market_rows.append({
            "market": market,
            "sample_size": len(sampled),
            "advancers": advancers,
            "decliners": decliners,
            "flat": flat,
            "advance_ratio": round(up_ratio, 4),
            "decline_ratio": round(down_ratio, 4),
            "flat_ratio": round(flat_ratio, 4),
            "avg_change_pct": round(avg_change_pct, 3),
        })

        top_movers = sorted(sampled, key=lambda q: abs(float(q.get("change_pct", 0.0) or 0.0)), reverse=True)[:3]
        quotes.extend(top_movers)

    quotes = sorted(quotes, key=lambda q: abs(float(q.get("change_pct", 0.0) or 0.0)), reverse=True)

    total_count = max(1, total_advancers + total_decliners + total_flat)
    payload = {
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "markets": market_rows,
        "quotes": quotes,
        "global_ratio": {
            "advancers": total_advancers,
            "decliners": total_decliners,
            "flat": total_flat,
            "advance_ratio": round(total_advancers / total_count, 4),
            "decline_ratio": round(total_decliners / total_count, 4),
            "flat_ratio": round(total_flat / total_count, 4),
        },
    }

    _snapshot_cache[cache_key] = {
        "_ts": now,
        "payload": payload,
    }
    return payload


# ═══════════════════════════════════════════════════════
#  CANDLE DATA (for TradingView Lightweight Charts)
# ═══════════════════════════════════════════════════════

@router.get("/candles/{ticker}")
async def get_intraday_candles(
    ticker: str,
    market: str = "SP500",
    interval: str = Query(default="5m", description="1m, 5m, 15m, 1h"),
    period: str = Query(default="1d", description="1d, 5d"),
):
    """
    Fetch intraday candle data for live candlestick charts.
    Returns OHLCV data formatted for TradingView Lightweight Charts.
    """
    if not _predictor:
        raise HTTPException(status_code=503, detail="Predictor not ready")

    try:
        data = _predictor.get_live_candles(
            ticker=ticker.upper(),
            market=market,
            interval=interval,
            period=period,
        )
        return data
    except Exception as e:
        logger.error(f"Candle error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily/{ticker}")
async def get_daily_candles(
    ticker: str,
    market: str = "SP500",
    period: str = Query(default="6mo", description="3mo, 6mo, 1y, 2y"),
):
    """
    Fetch daily candle data with indicator overlays (SMA, BB).
    Returns data for the main chart display with volume histogram.
    """
    if not _predictor:
        raise HTTPException(status_code=503, detail="Predictor not ready")

    try:
        data = _predictor.get_daily_candles(
            ticker=ticker.upper(),
            market=market,
            period=period,
        )
        return data
    except Exception as e:
        logger.error(f"Daily candle error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════
#  NEWS
# ═══════════════════════════════════════════════════════

@router.get("/news/{ticker}")
async def get_live_news(ticker: str, market: str = "SP500"):
    """
    Fetch latest news headlines for a ticker.
    Sources: Finnhub → NewsAPI → yfinance → demo fallback.
    """
    if not _live_service:
        raise HTTPException(status_code=503, detail="Live data service not ready")

    try:
        symbol = ticker.upper().strip()
        loop = asyncio.get_running_loop()
        try:
            data = await asyncio.wait_for(
                loop.run_in_executor(_news_executor, _live_service.get_live_news, symbol, market),
                timeout=_news_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("News fetch timeout for %s (market=%s), returning cached/demo fallback", symbol, market)
            data = _live_service.get_cached_or_demo_news(symbol)
        return {"status": "success", **data}
    except Exception as e:
        logger.error(f"News error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════
#  PREDICTION (using trained model)
# ═══════════════════════════════════════════════════════

@router.post("/predict")
async def predict_signal(request: PredictRequest):
    """
    Generate a real-time prediction/signal using the trained model.
    The model must be trained first via /api/train/start.
    Returns direction, confidence, entry/target/stop-loss prices.
    """
    if not _predictor:
        raise HTTPException(status_code=503, detail="Predictor not ready")

    ticker = request.ticker.upper().strip()

    result = _predictor.predict(
        ticker=ticker,
        market=request.market,
        total_capital=request.capital,
        risk_tolerance=request.risk_tolerance,
    )

    if result.get("status") == "not_trained":
        raise HTTPException(
            status_code=400,
            detail=f"Model not trained for {ticker}. POST /api/train/start first.",
        )

    return result


@router.get("/predict/{ticker}")
async def predict_signal_get(
    ticker: str,
    market: str = "SP500",
    capital: float = 100000,
    risk_tolerance: float = 0.5,
):
    """
    GET variant of prediction endpoint (for easy polling).
    """
    if not _predictor:
        raise HTTPException(status_code=503, detail="Predictor not ready")

    ticker = ticker.upper().strip()

    result = _predictor.predict(
        ticker=ticker,
        market=market,
        total_capital=capital,
        risk_tolerance=risk_tolerance,
    )

    if result.get("status") == "not_trained":
        raise HTTPException(
            status_code=400,
            detail=f"Model not trained for {ticker}. POST /api/train/start first.",
        )

    return result


@router.post("/predict/batch")
async def batch_predict(
    tickers: list,
    market: str = "SP500",
    capital: float = 100000,
    risk_tolerance: float = 0.5,
):
    """Generate predictions for multiple tickers."""
    if not _predictor:
        raise HTTPException(status_code=503, detail="Predictor not ready")

    results = _predictor.batch_predict(
        tickers=[t.upper().strip() for t in tickers],
        market=market,
        total_capital=capital,
        risk_tolerance=risk_tolerance,
    )
    return {"results": results, "count": len(results)}


# ═══════════════════════════════════════════════════════
#  API KEY CONFIGURATION
# ═══════════════════════════════════════════════════════

@router.post("/settings/api-keys")
async def configure_api_keys(request: ApiKeyRequest):
    """
    Set API keys at runtime for enhanced data sources.
    Keys are stored in memory only (not persisted).
    """
    if not _live_service:
        raise HTTPException(status_code=503, detail="Live data service not ready")

    _live_service.configure_api_keys(
        fmp_keys=request.fmp_keys,
        alpha_vantage=request.alpha_vantage,
        finnhub=request.finnhub,
        news_api=request.news_api,
        provider_keys=request.providers,
    )

    # Keep historical OHLCV fetcher in sync with runtime FMP key updates used by analysis.
    if _fetcher is not None and hasattr(_fetcher, "set_fmp_keys"):
        try:
            _fetcher.set_fmp_keys(list(_live_service.provider_keys.get("fmp", [])))
        except Exception as exc:
            logger.warning(f"Failed to sync FMP keys to market fetcher: {exc}")

    return {
        "status": "ok",
        "message": "API keys updated",
        "sources": {
            "fmp": _live_service.fmp.available,
            "fmp_keys_count": _live_service.fmp.total_keys,
            "fmp_daily_budget": _live_service.fmp.total_daily_budget,
            "alpha_vantage": bool(_live_service.alpha_vantage_key),
            "finnhub": bool(_live_service.finnhub_key),
            "news_api": bool(_live_service.news_api_key),
            "providers": _live_service.get_provider_key_status(),
        },
    }


@router.get("/settings/api-keys")
async def get_api_key_status():
    """Check which API keys are configured (does NOT reveal keys)."""
    if not _live_service:
        return {"sources": {}}

    return {
        "sources": {
            "fmp": _live_service.fmp.available,
            "fmp_keys_count": _live_service.fmp.total_keys,
            "fmp_daily_budget": _live_service.fmp.total_daily_budget,
            "fmp_status": _live_service.get_fmp_key_status(),
            "alpha_vantage": bool(_live_service.alpha_vantage_key),
            "finnhub": bool(_live_service.finnhub_key),
            "news_api": bool(_live_service.news_api_key),
            "providers": _live_service.get_provider_key_status(),
            "yfinance": True,  # always available
        },
    }


# ═══════════════════════════════════════════════════════
#  GRAFANA EMBED CONFIG
# ═══════════════════════════════════════════════════════

@router.get("/grafana/embed-config")
async def get_grafana_embed_config():
    """
    Return Grafana panel embedding configuration for true server-side panel embeds.
    Configure with environment variables:
      - GRAFANA_EMBED_ENABLED=true|false
      - GRAFANA_URL=https://grafana.example.com
      - GRAFANA_DASHBOARD_UID=abc123xyz
      - GRAFANA_DASHBOARD_SLUG=market-live
      - GRAFANA_PANEL_PRICE_ID=12
      - GRAFANA_PANEL_VOLUME_ID=13 (optional)
      - GRAFANA_ORG_ID=1
      - GRAFANA_THEME=dark|light
      - GRAFANA_REFRESH=5s
      - GRAFANA_TICKER_VAR=ticker
      - GRAFANA_MARKET_VAR=market
      - GRAFANA_TIMEFRAME_VAR=timeframe
    """
    enabled = _parse_bool_env(os.getenv("GRAFANA_EMBED_ENABLED"), default=True)

    config = {
        "enabled": enabled,
        "grafana_url": (os.getenv("GRAFANA_URL") or "http://localhost:3000").rstrip("/"),
        "dashboard_uid": (os.getenv("GRAFANA_DASHBOARD_UID") or "finsent-live").strip(),
        "dashboard_slug": (os.getenv("GRAFANA_DASHBOARD_SLUG") or "finsent-live-tickers").strip(),
        "org_id": _parse_int_env(os.getenv("GRAFANA_ORG_ID"), 1),
        "price_panel_id": _parse_int_env(os.getenv("GRAFANA_PANEL_PRICE_ID"), 1),
        "volume_panel_id": _parse_int_env(os.getenv("GRAFANA_PANEL_VOLUME_ID"), 3),
        "theme": (os.getenv("GRAFANA_THEME") or "light").strip().lower(),
        "refresh": (os.getenv("GRAFANA_REFRESH") or "5s").strip(),
        "ticker_var": (os.getenv("GRAFANA_TICKER_VAR") or "ticker").strip(),
        "market_var": (os.getenv("GRAFANA_MARKET_VAR") or "market").strip(),
        "timeframe_var": (os.getenv("GRAFANA_TIMEFRAME_VAR") or "timeframe").strip(),
    }

    reachable = _grafana_reachable(config["grafana_url"]) if enabled else False

    ready = (
        enabled
        and bool(config["grafana_url"])
        and bool(config["dashboard_uid"])
        and isinstance(config["price_panel_id"], int)
        and reachable
    )

    return {
        "status": "ok",
        "ready": ready,
        "reachable": reachable,
        "config": config,
    }


# ═══════════════════════════════════════════════════════
#  DUAL TUNNEL STATUS
# ═══════════════════════════════════════════════════════

@router.get("/tunnels/status")
async def get_dual_tunnel_status():
    """Inspect runtime status of both websocket tunnels."""
    return {
        "status": "ok",
        "tunnels": get_tunnel_status(),
    }
