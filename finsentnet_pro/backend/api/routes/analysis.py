"""
FINSENT NET PRO — Analysis Routes
/api/analyze endpoint — the main analysis pipeline.
"""

import asyncio
import copy
import hashlib
import json
import os
import time
import logging
import threading
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from observability.live_metrics import register_selected_tickers, build_grafana_live_url

router = APIRouter(prefix="/api", tags=["Analysis"])
logger = logging.getLogger("finsent.routes.analysis")

# Singletons injected from main.py
_fetcher = None
_indicators = None
_sentiment = None
_regime = None
_aligner = None
_live_service = None
_model = None
_signal_gen = None
_optimizer = None
_allocator = None
_backtester = None
_risk_engine = None
_model_device = torch.device("cpu")

_ALLOW_SYNTHETIC_FALLBACK = str(
    os.getenv("ALLOW_SYNTHETIC_FALLBACK", "0")
).strip().lower() in {"1", "true", "yes", "on"}
_ANALYSIS_CACHE_TTL_SECONDS = max(1, int(os.getenv("ANALYSIS_CACHE_TTL_SECONDS", "20")))
_ANALYSIS_CACHE_MAX_SIZE = max(16, int(os.getenv("ANALYSIS_CACHE_MAX_SIZE", "256")))
_ANALYSIS_MAX_WORKERS = max(1, int(os.getenv("ANALYSIS_MAX_WORKERS", "4")))
_ANALYSIS_EXECUTOR = ThreadPoolExecutor(max_workers=_ANALYSIS_MAX_WORKERS)
_ANALYSIS_LOOKBACK_PERIOD = str(os.getenv("ANALYSIS_LOOKBACK_PERIOD", "6mo") or "6mo").strip()
_ANALYSIS_HARD_TIMEOUT_SECONDS = max(1.0, float(os.getenv("ANALYSIS_HARD_TIMEOUT_SECONDS", "10")))
_ANALYSIS_TEXT_TOKEN_LENGTH = max(16, int(os.getenv("ANALYSIS_TEXT_TOKEN_LENGTH", "50")))
_ANALYSIS_LIVE_ENRICH_ENABLED = str(os.getenv("ANALYSIS_LIVE_ENRICH_ENABLED", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
_ANALYSIS_LIVE_ENRICH_TIMEOUT_SECONDS = max(0.1, float(os.getenv("ANALYSIS_LIVE_ENRICH_TIMEOUT_SECONDS", "1.2")))
_ANALYSIS_LIVE_ENRICH_MAX_WORKERS = max(2, int(os.getenv("ANALYSIS_LIVE_ENRICH_MAX_WORKERS", "8")))
_ANALYSIS_CACHE: Dict[str, Dict] = {}
_ANALYSIS_CACHE_LOCK = threading.Lock()
_MODEL_INFER_LOCK = threading.Lock()
_ANALYSIS_LIVE_EXECUTOR = ThreadPoolExecutor(max_workers=_ANALYSIS_LIVE_ENRICH_MAX_WORKERS)


def init(fetcher, indicators, sentiment, regime, aligner, model,
         signal_gen, optimizer, allocator, backtester, risk_engine, live_service=None):
    """Initialize route with shared service instances."""
    global _fetcher, _indicators, _sentiment, _regime, _aligner, _live_service
    global _model, _signal_gen, _optimizer, _allocator, _backtester, _risk_engine
    global _model_device
    _fetcher = fetcher
    _indicators = indicators
    _sentiment = sentiment
    _regime = regime
    _aligner = aligner
    _live_service = live_service
    _model = model
    _signal_gen = signal_gen
    _optimizer = optimizer
    _allocator = allocator
    _backtester = backtester
    _risk_engine = risk_engine

    try:
        first_param = next(_model.parameters())
        _model_device = first_param.device
    except StopIteration:
        _model_device = torch.device("cpu")


def _build_cache_key(request: "AnalysisRequest") -> str:
    ticker_markets = {
        str(k).upper().strip(): str(v).upper().strip()
        for k, v in (request.ticker_markets or {}).items()
        if str(k).strip() and str(v).strip()
    }
    payload = {
        "tickers": sorted(
            {str(t).upper().strip() for t in request.tickers[:10] if str(t).strip()}
        ),
        "market": str(request.market).upper().strip(),
        "markets": sorted({str(m).upper().strip() for m in request.markets if str(m).strip()}),
        "ticker_markets": ticker_markets,
        "capital": round(float(request.total_capital), 2),
        "risk": round(float(request.risk_tolerance), 4),
        "currency": str(request.currency).upper().strip(),
        "horizon": str(request.horizon).upper().strip(),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _prune_cache_locked() -> None:
    if len(_ANALYSIS_CACHE) <= _ANALYSIS_CACHE_MAX_SIZE:
        return

    overflow = len(_ANALYSIS_CACHE) - _ANALYSIS_CACHE_MAX_SIZE
    oldest = sorted(
        _ANALYSIS_CACHE.items(),
        key=lambda item: item[1].get("cached_at", 0.0),
    )[:overflow]
    for key, _ in oldest:
        _ANALYSIS_CACHE.pop(key, None)


def _get_cached_response(cache_key: str) -> Optional[Dict]:
    now = time.time()
    with _ANALYSIS_CACHE_LOCK:
        cached = _ANALYSIS_CACHE.get(cache_key)
        if not cached:
            return None

        age = now - cached.get("cached_at", 0.0)
        if age > _ANALYSIS_CACHE_TTL_SECONDS:
            _ANALYSIS_CACHE.pop(cache_key, None)
            return None

        response = copy.deepcopy(cached["payload"])
        response["cache_hit"] = True
        response["cache_age_seconds"] = round(age, 2)
        return response


def _set_cached_response(cache_key: str, payload: Dict) -> None:
    with _ANALYSIS_CACHE_LOCK:
        _ANALYSIS_CACHE[cache_key] = {
            "payload": copy.deepcopy(payload),
            "cached_at": time.time(),
        }
        _prune_cache_locked()


def _build_stage_metrics(stage_times: Dict[str, float], total_elapsed: float) -> Dict[str, float]:
    out = {k: round(float(v), 4) for k, v in stage_times.items()}
    out["total_s"] = round(float(total_elapsed), 4)
    return out


def _raise_if_deadline_exceeded(start: float, stage: str, stage_times: Dict[str, float]) -> None:
    elapsed = time.time() - start
    if elapsed <= _ANALYSIS_HARD_TIMEOUT_SECONDS:
        return

    metrics = _build_stage_metrics(stage_times, elapsed)
    logger.warning(
        "Analysis hard-timeout at stage=%s elapsed=%.3fs limit=%.3fs metrics=%s",
        stage,
        elapsed,
        _ANALYSIS_HARD_TIMEOUT_SECONDS,
        metrics,
    )
    raise HTTPException(
        status_code=504,
        detail={
            "message": (
                f"Analysis exceeded hard timeout of {_ANALYSIS_HARD_TIMEOUT_SECONDS:.1f}s "
                f"at stage '{stage}'"
            ),
            "stage": stage,
            "timings": metrics,
            "timeout_seconds": _ANALYSIS_HARD_TIMEOUT_SECONDS,
        },
    )


def _prepare_price_tensor(df) -> torch.Tensor:
    if len(df) < 30:
        raise ValueError(f"Insufficient rows for inference: {len(df)}")

    price_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Upper", "BB_Lower", "BB_Position", "ATR_14",
        "OBV", "Volume_Ratio", "EMA_20", "EMA_50",
        "Log_Return", "Volatility_20", "Stoch_K",
    ]
    available_cols = [c for c in price_cols if c in df.columns]
    if not available_cols:
        raise ValueError("No model-compatible feature columns found")

    price_window = df[available_cols].iloc[-30:].values

    # Causal z-score normalization from the lookback window.
    price_mean = price_window.mean(axis=0, keepdims=True)
    price_std = price_window.std(axis=0, keepdims=True) + 1e-8
    price_norm = (price_window - price_mean) / price_std

    if price_norm.shape[1] < 20:
        pad = np.zeros((price_norm.shape[0], 20 - price_norm.shape[1]))
        price_norm = np.concatenate([price_norm, pad], axis=1)

    return torch.as_tensor(price_norm, dtype=torch.float32, device=_model_device).unsqueeze(0)


def _build_text_token_batch(prepared_inputs: List[Dict]) -> torch.Tensor:
    """Build deterministic token ids from sentiment to avoid random-noise inference drift."""
    batch_size = len(prepared_inputs)
    tokens = torch.full(
        (batch_size, _ANALYSIS_TEXT_TOKEN_LENGTH),
        101,
        dtype=torch.long,
        device=_model_device,
    )
    if _ANALYSIS_TEXT_TOKEN_LENGTH >= 2:
        tokens[:, -1] = 102

    for idx, item in enumerate(prepared_inputs):
        sentiment = (item.get("detail") or {}).get("sentiment") or {}
        sent_score = float(sentiment.get("score", 0.0) or 0.0)
        sent_bucket = int(max(-1.0, min(1.0, sent_score)) * 300 + 500)
        sent_bucket = max(100, min(999, sent_bucket))
        tokens[idx, 1:min(6, _ANALYSIS_TEXT_TOKEN_LENGTH - 1)] = sent_bucket
    return tokens


def _build_chart(df) -> List[Dict]:
    chart_df = df.tail(60)
    return [
        {
            "time": int(idx.timestamp()) if hasattr(idx, "timestamp") else i,
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
        }
        for i, (idx, row) in enumerate(chart_df.iterrows())
    ]


def _build_stock_detail(df, ticker: str, sentiment: Dict, regime_result: Dict, synthetic_used: bool) -> Dict:
    latest = df.iloc[-1]
    return {
        "ticker": ticker,
        "price": round(float(latest["Close"]), 2),
        "rsi": round(float(latest.get("RSI_14", 50)), 1),
        "macd": round(float(latest.get("MACD", 0)), 4),
        "sma_50": round(float(latest.get("SMA_50", latest["Close"])), 2),
        "sma_200": round(float(latest.get("SMA_200", latest["Close"])), 2),
        "volume": int(latest.get("Volume", 0)),
        "atr": round(float(latest.get("ATR_14", 0)), 2),
        "bb_position": round(float(latest.get("BB_Position", 0.5)), 2),
        "sentiment": sentiment,
        "regime": regime_result.get("detail", "UNKNOWN"),
        "data_source": "synthetic" if synthetic_used else "real",
    }


def _collect_live_enrichment(ticker: str, market: str) -> Tuple[Dict, List[Dict]]:
    """Fetch quote/news with strict per-call time budgets and timeout-safe fallbacks."""
    if _live_service is None or not _ANALYSIS_LIVE_ENRICH_ENABLED:
        return {}, []

    timeout_s = max(0.1, float(_ANALYSIS_LIVE_ENRICH_TIMEOUT_SECONDS))
    live_quote: Dict = {}
    live_news: List[Dict] = []

    quote_future = _ANALYSIS_LIVE_EXECUTOR.submit(_live_service.get_realtime_quote, ticker, market)
    news_future = _ANALYSIS_LIVE_EXECUTOR.submit(_live_service.get_live_news, ticker, market)

    try:
        quote_payload = quote_future.result(timeout=timeout_s)
        if isinstance(quote_payload, dict):
            live_quote = quote_payload
    except FuturesTimeoutError:
        quote_future.cancel()
        logger.debug("Live quote enrichment timeout for %s after %.2fs", ticker, timeout_s)
    except Exception as exc:
        logger.debug("Live quote enrichment failed for %s: %s", ticker, exc)

    try:
        news_payload = news_future.result(timeout=timeout_s)
        if isinstance(news_payload, dict):
            live_news = list((news_payload or {}).get("articles", [])[:5])
    except FuturesTimeoutError:
        news_future.cancel()
        logger.debug("Live news enrichment timeout for %s after %.2fs", ticker, timeout_s)
        try:
            cached_news_payload = _live_service.get_cached_or_demo_news(ticker)
            if isinstance(cached_news_payload, dict):
                live_news = list((cached_news_payload or {}).get("articles", [])[:5])
        except Exception:
            live_news = []
    except Exception as exc:
        logger.debug("Live news enrichment failed for %s: %s", ticker, exc)

    return live_quote, live_news


def _collect_ticker_inputs(
    ticker: str,
    market: str,
) -> Optional[Dict]:
    ticker = str(ticker).upper().strip()
    if not ticker:
        return None

    one_start = time.time()
    synthetic_used = False

    stage_times = {
        "fetch_s": 0.0,
        "indicators_s": 0.0,
        "live_data_s": 0.0,
    }

    # 1) Fetch OHLCV with strict real-data preference by default.
    t_fetch = time.time()
    df = _fetcher.fetch_ohlcv(ticker, market=market, period=_ANALYSIS_LOOKBACK_PERIOD)
    if df.empty:
        if _ALLOW_SYNTHETIC_FALLBACK:
            df = _fetcher._fallback_synthetic(ticker, _ANALYSIS_LOOKBACK_PERIOD)
            synthetic_used = True
        else:
            raise ValueError("No real market data available")

    # Some providers may return a non-empty frame with too few rows for model windows.
    if len(df) < 30:
        if _ALLOW_SYNTHETIC_FALLBACK:
            df = _fetcher._fallback_synthetic(ticker, _ANALYSIS_LOOKBACK_PERIOD)
            synthetic_used = True
        else:
            raise ValueError(f"Insufficient real market data rows: {len(df)}")
    stage_times["fetch_s"] = time.time() - t_fetch

    # 2) Technicals + higher-level context.
    t_indicators = time.time()
    df = _indicators.compute_all(df)

    # If indicator warmup drops too many rows, retry with a longer history.
    if len(df) < 30 and not synthetic_used and _ANALYSIS_LOOKBACK_PERIOD.lower() != "1y":
        df_retry = _fetcher.fetch_ohlcv(ticker, market=market, period="1y")
        if (df_retry is not None) and (not df_retry.empty):
            df = _indicators.compute_all(df_retry)

    # Final fallback for constrained provider responses.
    if len(df) < 30 and _ALLOW_SYNTHETIC_FALLBACK:
        synthetic_used = True
        df = _indicators.compute_all(_fetcher._fallback_synthetic(ticker, "1y"))

    if len(df) < 30:
        raise ValueError(f"Insufficient rows for inference: {len(df)}")

    sentiment = _sentiment.generate_demo_sentiment(ticker)
    regime_result = _regime.detect(df)
    stage_times["indicators_s"] = time.time() - t_indicators

    t_live = time.time()
    live_quote, live_news = _collect_live_enrichment(ticker, market)
    stage_times["live_data_s"] = time.time() - t_live

    # 3) Prepare per-ticker model input for batched GPU inference.
    price_tensor = _prepare_price_tensor(df)
    return {
        "ticker": ticker,
        "df": df,
        "price_tensor": price_tensor,
        "chart": _build_chart(df),
        "detail": {
            **_build_stock_detail(df, ticker, sentiment, regime_result, synthetic_used),
            "live_quote": live_quote,
            "top_live_news": live_news,
        },
        "stage_times": stage_times,
        "elapsed_s": (time.time() - one_start),
    }


def _slice_model_output(batch_output: Dict[str, torch.Tensor], idx: int) -> Dict[str, torch.Tensor]:
    sliced = {}
    for key, value in batch_output.items():
        if not torch.is_tensor(value):
            continue
        if value.ndim == 0:
            sliced[key] = value
        elif value.shape[0] > idx:
            sliced[key] = value[idx:idx + 1]
    return sliced


class AnalysisRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1, max_length=10)
    market: str = "SP500"
    markets: List[str] = Field(default_factory=list)
    ticker_markets: Dict[str, str] = Field(default_factory=dict)
    total_capital: float = Field(default=100000, alias="investment_amount", ge=1000)
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    currency: str = "USD"
    horizon: str = "1M"
    force_refresh: bool = False

    class Config:
        populate_by_name = True


@router.post("/analyze")
async def analyze_stocks(request: AnalysisRequest):
    """
    Complete multi-stock analysis pipeline.
    Fetches data → computes indicators → runs FINSENT → generates signals →
    optimizes portfolio → runs backtest → computes risk metrics.
    """
    start = time.time()
    deadline = start + _ANALYSIS_HARD_TIMEOUT_SECONDS
    stage_times = {
        "fetch_s": 0.0,
        "indicators_s": 0.0,
        "live_data_s": 0.0,
        "inference_s": 0.0,
        "signal_s": 0.0,
        "portfolio_s": 0.0,
    }
    normalized_tickers = [str(t).upper().strip() for t in request.tickers[:10] if str(t).strip()]
    if not normalized_tickers:
        raise HTTPException(status_code=400, detail="At least one valid ticker is required")

    logger.info(
        f"Analysis: {normalized_tickers} | Market: {request.market} "
        f"| Capital: {request.total_capital}"
    )

    cache_key = _build_cache_key(request)
    use_cache = not bool(request.force_refresh)
    if use_cache:
        cached = _get_cached_response(cache_key)
        if cached is not None:
            logger.info(f"Analysis cache hit for {len(normalized_tickers)} ticker(s)")
            return cached

    signals = []
    charts = {}
    stock_details = []

    loop = asyncio.get_running_loop()
    ticker_market_map = {
        str(k).upper().strip(): str(v).upper().strip()
        for k, v in (request.ticker_markets or {}).items()
        if str(k).strip() and str(v).strip()
    }
    register_selected_tickers(normalized_tickers, ticker_market_map, request.market)

    tasks_by_ticker = {
        ticker: loop.run_in_executor(
            _ANALYSIS_EXECUTOR,
            _collect_ticker_inputs,
            ticker,
            ticker_market_map.get(ticker, request.market),
        )
        for ticker in normalized_tickers
    }

    remaining_before_collect = deadline - time.time()
    if remaining_before_collect <= 0:
        _raise_if_deadline_exceeded(start, "fetch", stage_times)

    done, pending = await asyncio.wait(
        list(tasks_by_ticker.values()),
        timeout=remaining_before_collect,
        return_when=asyncio.ALL_COMPLETED,
    )

    timed_out_tickers = [
        ticker
        for ticker, task in tasks_by_ticker.items()
        if task in pending
    ]
    if timed_out_tickers:
        logger.warning(
            "Analysis fetch collection timed out for tickers=%s (completed=%d/%d)",
            timed_out_tickers,
            len(done),
            len(tasks_by_ticker),
        )
        for task in pending:
            task.cancel()

    _raise_if_deadline_exceeded(start, "fetch", stage_times)

    ticker_times = {}
    prepared_inputs = []
    for ticker in normalized_tickers:
        task = tasks_by_ticker[ticker]
        if task not in done:
            continue

        try:
            result = task.result()
        except Exception as exc:
            logger.error(f"Error analyzing {ticker}: {exc}")
            continue

        if result is None:
            continue

        prepared_inputs.append(result)
        item_stage = result.get("stage_times", {})
        stage_times["fetch_s"] += float(item_stage.get("fetch_s", 0.0))
        stage_times["indicators_s"] += float(item_stage.get("indicators_s", 0.0))
        stage_times["live_data_s"] += float(item_stage.get("live_data_s", 0.0))

    _raise_if_deadline_exceeded(start, "indicators", stage_times)

    if not prepared_inputs:
        if timed_out_tickers:
            elapsed = time.time() - start
            metrics = _build_stage_metrics(stage_times, elapsed)
            raise HTTPException(
                status_code=504,
                detail={
                    "message": "All ticker fetch tasks timed out before analysis could complete",
                    "stage": "fetch",
                    "timings": metrics,
                    "timeout_seconds": _ANALYSIS_HARD_TIMEOUT_SECONDS,
                    "timed_out_tickers": timed_out_tickers,
                },
            )

        detail = "No valid signals could be generated from real market data"
        if _ALLOW_SYNTHETIC_FALLBACK:
            detail = "No valid signals could be generated"
        raise HTTPException(status_code=400, detail=detail)

    # Run a single batched forward pass to maximize GPU utilization.
    t_infer = time.time()
    price_batch = torch.cat([item["price_tensor"] for item in prepared_inputs], dim=0)
    text_tokens = _build_text_token_batch(prepared_inputs)

    with _MODEL_INFER_LOCK:
        _model.eval()
        with torch.inference_mode():
            if _model_device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    batched_model_output = _model(text_tokens, price_batch)
            else:
                batched_model_output = _model(text_tokens, price_batch)
    stage_times["inference_s"] = time.time() - t_infer

    _raise_if_deadline_exceeded(start, "inference", stage_times)

    t_signal = time.time()
    for idx, item in enumerate(prepared_inputs):
        ticker = item["ticker"]
        per_ticker_output = _slice_model_output(batched_model_output, idx)

        signal = _signal_gen.generate_signal(
            ticker=ticker,
            market=ticker_market_map.get(ticker, request.market),
            model_output=per_ticker_output,
            price_data=item["df"],
            total_capital=request.total_capital,
            risk_tolerance=request.risk_tolerance,
        )

        signals.append(signal)
        charts[ticker] = item["chart"]
        stock_details.append(item["detail"])
        ticker_times[ticker] = round(item["elapsed_s"], 3)
    stage_times["signal_s"] = time.time() - t_signal

    _raise_if_deadline_exceeded(start, "signal", stage_times)

    if not signals:
        detail = "No valid signals could be generated from real market data"
        if _ALLOW_SYNTHETIC_FALLBACK:
            detail = "No valid signals could be generated"
        raise HTTPException(status_code=400, detail=detail)

    # ── Portfolio Optimization ──
    t_portfolio = time.time()
    n = len(signals)
    expected_rets = np.array([s.predicted_return / 100 for s in signals])
    cov = np.eye(n) * 0.02 ** 2
    opt = _optimizer.optimize_weights(expected_rets, cov, method="max_sharpe")

    # ── Allocation ──
    allocation = _allocator.allocate(signals, opt, request.total_capital)

    # ── Backtest (demo) ──
    bt = _backtester.generate_demo_backtest(request.total_capital)

    # ── Risk metrics ──
    equity = np.array(bt["equity_curve"])
    risk_report = _risk_engine.full_risk_report(
        np.diff(equity) / equity[:-1], equity,
    )
    stage_times["portfolio_s"] = time.time() - t_portfolio

    _raise_if_deadline_exceeded(start, "portfolio", stage_times)

    elapsed = time.time() - start
    stage_metrics = _build_stage_metrics(stage_times, elapsed)
    logger.info(
        "Analysis complete in %.2fs for %d stocks | stage_metrics=%s",
        elapsed,
        len(signals),
        stage_metrics,
    )

    response = {
        "status": "success",
        "analysis_time_seconds": round(elapsed, 2),
        "cache_hit": False,
        "compute_device": str(_model_device),
        "grafana_live_url": build_grafana_live_url(normalized_tickers),
        "ticker_markets": {
            ticker: ticker_market_map.get(ticker, request.market)
            for ticker in normalized_tickers
        },
        "timeout_seconds": _ANALYSIS_HARD_TIMEOUT_SECONDS,
        "stage_metrics": stage_metrics,
        "synthetic_fallback_enabled": _ALLOW_SYNTHETIC_FALLBACK,
        "timed_out_tickers": timed_out_tickers,
        "ticker_times_seconds": ticker_times,
        "signals": [
            {
                "ticker": s.ticker,
                "direction": s.direction.value,
                "confidence": s.confidence,
                "predicted_return": s.predicted_return,
                "predicted_downside": s.predicted_downside,
                "entry_price": s.entry_price,
                "target_price": s.target_price,
                "stop_loss": s.stop_loss,
                "risk_reward": s.risk_reward_ratio,
                "kelly_fraction": s.kelly_fraction,
                "quantity": s.recommended_quantity,
                "capital_required": s.capital_required,
                "time_horizon": s.time_horizon,
                "reasoning": s.reasoning,
                "regime": s.regime,
                "sentiment_score": s.sentiment_score,
                "technical_score": s.technical_score,
            }
            for s in signals
        ],
        "stock_details": stock_details,
        "charts": charts,
        "portfolio": {
            "optimization": opt,
            "allocation": allocation,
            "orders": [
                {
                    "ticker": s.ticker,
                    "action": (
                        "BUY" if str(getattr(s.direction, "value", "")).upper() in {"BUY", "UP"}
                        else "SELL" if str(getattr(s.direction, "value", "")).upper() in {"SELL", "DOWN"}
                        else "HOLD"
                    ),
                    "quantity": int(max(0, s.recommended_quantity)),
                    "entry_price": float(s.entry_price),
                    "target_price": float(s.target_price),
                    "stop_loss": float(s.stop_loss),
                    "capital_required": float(s.capital_required),
                    "confidence": float(s.confidence),
                    "predicted_return": float(s.predicted_return),
                }
                for s in signals
            ],
        },
        "backtest": bt,
        "risk": risk_report,
    }

    if use_cache:
        _set_cached_response(cache_key, response)
    return response
