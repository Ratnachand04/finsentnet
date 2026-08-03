"""
Prometheus live metrics bridge for Grafana dashboards.
Tracks selected tickers and refreshes quote metrics from live providers.
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Dict, List, Tuple
from urllib.parse import urlencode

from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

_live_service = None
_metrics_lock = threading.Lock()
_selected_tickers: Dict[str, str] = {}
_last_refresh_ts = 0.0

_REFRESH_SECONDS = max(1.0, float(os.getenv("FINSENT_METRICS_REFRESH_SECONDS", "4")))
_MAX_TICKERS = max(1, int(os.getenv("FINSENT_METRICS_MAX_TICKERS", "30")))
_MAX_WORKERS = max(1, int(os.getenv("FINSENT_METRICS_MAX_WORKERS", "8")))
_SCRAPE_TIMEOUT_SECONDS = max(0.2, float(os.getenv("FINSENT_METRICS_SCRAPE_TIMEOUT_SECONDS", "1.2")))

LIVE_PRICE = Gauge(
    "finsent_live_price",
    "Latest live price from provider",
    labelnames=("ticker", "market", "source"),
)
LIVE_CHANGE_PCT = Gauge(
    "finsent_live_change_pct",
    "Latest live percent change",
    labelnames=("ticker", "market", "source"),
)
LIVE_VOLUME = Gauge(
    "finsent_live_volume",
    "Latest live volume",
    labelnames=("ticker", "market", "source"),
)
LIVE_LAST_UPDATE_EPOCH = Gauge(
    "finsent_live_last_update_epoch",
    "Epoch timestamp for last quote update",
    labelnames=("ticker", "market", "source"),
)
LIVE_SCRAPE_SUCCESS = Gauge(
    "finsent_live_scrape_success",
    "Whether quote refresh succeeded (1) or failed (0)",
    labelnames=("ticker", "market"),
)


def configure_live_metrics(live_service) -> None:
    global _live_service
    _live_service = live_service


def register_selected_tickers(tickers: List[str], ticker_markets: Dict[str, str], default_market: str) -> None:
    normalized_default = str(default_market or "SP500").upper().strip() or "SP500"
    selected: Dict[str, str] = {}
    for ticker in tickers[:_MAX_TICKERS]:
        t = str(ticker or "").upper().strip()
        if not t:
            continue
        m = str(ticker_markets.get(t) or normalized_default).upper().strip() or normalized_default
        selected[t] = m

    with _metrics_lock:
        _selected_tickers.clear()
        _selected_tickers.update(selected)


def _snapshot_selection() -> List[Tuple[str, str]]:
    with _metrics_lock:
        return list(_selected_tickers.items())


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _update_one_quote(ticker: str, market: str) -> None:
    if _live_service is None:
        LIVE_SCRAPE_SUCCESS.labels(ticker=ticker, market=market).set(0)
        return

    try:
        quote = _live_service.get_realtime_quote(ticker=ticker, market=market)
        source = str(quote.get("source") or "unknown")
        price = _safe_float(quote.get("price"), 0.0)
        change_pct = _safe_float(quote.get("change_pct"), 0.0)
        volume = _safe_float(quote.get("volume"), 0.0)

        LIVE_PRICE.labels(ticker=ticker, market=market, source=source).set(price)
        LIVE_CHANGE_PCT.labels(ticker=ticker, market=market, source=source).set(change_pct)
        LIVE_VOLUME.labels(ticker=ticker, market=market, source=source).set(volume)
        LIVE_LAST_UPDATE_EPOCH.labels(ticker=ticker, market=market, source=source).set(time.time())
        LIVE_SCRAPE_SUCCESS.labels(ticker=ticker, market=market).set(1)
    except Exception:
        LIVE_SCRAPE_SUCCESS.labels(ticker=ticker, market=market).set(0)


def refresh_selected_ticker_metrics(force: bool = False) -> None:
    global _last_refresh_ts

    now = time.time()
    if not force and (now - _last_refresh_ts) < _REFRESH_SECONDS:
        return

    selected = _snapshot_selection()
    if not selected:
        return

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(selected))) as pool:
        futures = [pool.submit(_update_one_quote, ticker, market) for ticker, market in selected]
        done, pending = wait(futures, timeout=_SCRAPE_TIMEOUT_SECONDS)
        for f in done:
            try:
                f.result()
            except Exception:
                pass
        for f in pending:
            f.cancel()

    _last_refresh_ts = now


def render_prometheus_metrics() -> Tuple[bytes, str]:
    # Pull fresh provider-backed quotes just before scrape for live Grafana panels.
    refresh_selected_ticker_metrics(force=False)
    return generate_latest(), CONTENT_TYPE_LATEST


def build_grafana_live_url(tickers: List[str]) -> str:
    base = str(
        os.getenv(
            "GRAFANA_DASHBOARD_BASE_URL",
            "http://localhost:3000/d/finsent-live/finsent-live-tickers",
        )
    ).strip()
    normalized = [str(t).upper().strip() for t in tickers if str(t).strip()]
    params = {"refresh": "5s"}
    if normalized:
        params["var-ticker"] = normalized
    qs = urlencode(params, doseq=True)
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}{qs}" if qs else base
