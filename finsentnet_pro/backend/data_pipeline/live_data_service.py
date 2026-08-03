"""
FINSENT NET PRO — Live Data Service
Multi-source real-time market data and news fetching.

Data Sources (priority order):
  1. FMP (Financial Modeling Prep) — primary source (free: 250 calls/day/key)
     Supports multiple API keys with automatic rotation.
  2. Yahoo Finance (yfinance) — free, no API key needed
  3. Alpha Vantage — intraday OHLCV (free key: 25 req/day)
  4. Finnhub — real-time quotes + news (free key: 60 req/min)
  5. NewsAPI — financial headlines (free key: 100 req/day)

API keys are loaded from environment variables:
  FMP_API_KEYS           (comma-separated list of FMP keys)
  ALPHA_VANTAGE_API_KEY
  FINNHUB_API_KEY
  NEWS_API_KEY

FMP Free Tier Limits:
    - 250 API calls per day per key
    - Multiple keys supported for round-robin rotation
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
from requests.adapters import HTTPAdapter
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta, date

logger = logging.getLogger("finsent.live_data")

FMP_BASE = "https://financialmodelingprep.com/api/v3"
FMP_DAILY_LIMIT = 250  # free tier limit per key


class FMPKeyManager:
    """
    Manages multiple FMP API keys with round-robin rotation
    and per-key daily call tracking (250 calls/day free tier).
    """

    def __init__(self, keys: List[str]):
        self.keys = [k.strip() for k in keys if k.strip()]
        self._call_counts: Dict[str, Dict[str, int]] = {}  # key -> {date_str: count}
        self._current_index = 0
        self._all_exhausted_on: Optional[str] = None

    @property
    def available(self) -> bool:
        return len(self.keys) > 0

    @property
    def total_keys(self) -> int:
        return len(self.keys)

    @property
    def total_daily_budget(self) -> int:
        return len(self.keys) * FMP_DAILY_LIMIT

    def get_key(self) -> Optional[str]:
        """
        Get the next available FMP key that hasn't exceeded daily limit.
        Uses round-robin with automatic rotation on limit.
        """
        if not self.keys:
            return None

        today = date.today().isoformat()
        if self._all_exhausted_on == today:
            return None

        tried = 0

        while tried < len(self.keys):
            key = self.keys[self._current_index]
            count = self._get_count(key, today)

            if count < FMP_DAILY_LIMIT:
                self._increment(key, today)
                self._current_index = (self._current_index + 1) % len(self.keys)
                return key

            # This key exhausted, try next
            logger.warning(
                f"FMP key #{self._current_index + 1} exhausted "
                f"({count}/{FMP_DAILY_LIMIT}), rotating..."
            )
            self._current_index = (self._current_index + 1) % len(self.keys)
            tried += 1

        self._all_exhausted_on = today
        logger.error("All FMP API keys exhausted for today!")
        return None

    def get_status(self) -> Dict:
        """Return status of all keys (call counts, remaining)."""
        today = date.today().isoformat()
        statuses = []
        for i, key in enumerate(self.keys):
            count = self._get_count(key, today)
            statuses.append({
                "key_index": i + 1,
                "calls_used": count,
                "calls_remaining": max(0, FMP_DAILY_LIMIT - count),
                "exhausted": count >= FMP_DAILY_LIMIT,
            })
        total_used = sum(s["calls_used"] for s in statuses)
        return {
            "total_keys": len(self.keys),
            "total_budget": self.total_daily_budget,
            "total_used": total_used,
            "total_remaining": self.total_daily_budget - total_used,
            "keys": statuses,
        }

    def _get_count(self, key: str, date_str: str) -> int:
        return self._call_counts.get(key, {}).get(date_str, 0)

    def _increment(self, key: str, date_str: str):
        if key not in self._call_counts:
            self._call_counts[key] = {}
        # Reset old dates to save memory
        self._call_counts[key] = {
            d: c for d, c in self._call_counts[key].items()
            if d == date_str
        }
        self._call_counts[key][date_str] = self._call_counts[key].get(date_str, 0) + 1

    def add_keys(self, new_keys: List[str]):
        """Add new FMP keys at runtime."""
        added = 0
        for k in new_keys:
            k = k.strip()
            if k and k not in self.keys:
                self.keys.append(k)
                added += 1
        if added > 0:
            self._all_exhausted_on = None
        logger.info(f"FMP keys updated: {len(self.keys)} total")

    def set_keys(self, keys: List[str]):
        """Replace FMP keys and reset rotation/exhaustion state."""
        self.keys = [k.strip() for k in keys if k.strip()]
        self._current_index = 0
        self._all_exhausted_on = None
        self._call_counts = {
            key: counts
            for key, counts in self._call_counts.items()
            if key in self.keys
        }


class LiveDataService:
    """
    Unified live data provider for FINSENT NET PRO.
    FMP (Financial Modeling Prep) is the primary data source.
    Falls back to yfinance (free, no key) when FMP keys are exhausted.
    """

    def __init__(self):
        # Load FMP keys from environment (comma-separated)
        fmp_keys_raw = os.environ.get("FMP_API_KEYS", "")
        if not str(fmp_keys_raw).strip():
            fmp_keys_raw = os.environ.get("FMP_API_KEY", "")
        fmp_keys = [k for k in fmp_keys_raw.split(",") if k.strip()]
        self.fmp = FMPKeyManager(fmp_keys)

        # Legacy API keys
        self.alpha_vantage_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self.finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
        self.news_api_key = os.environ.get("NEWS_API_KEY", "")

        # Provider-agnostic runtime key store.
        self.provider_keys: Dict[str, List[str]] = self._load_provider_keys_from_env()
        self._set_provider_keys("fmp", fmp_keys)
        self._set_provider_keys("alpha_vantage", self.alpha_vantage_key)
        self._set_provider_keys("finnhub", self.finnhub_key)
        self._set_provider_keys("news_api", self.news_api_key)

        # Rate limiting
        self._last_request_time: Dict[str, float] = {}
        self._min_interval = {
            "fmp": 0.5, "alpha_vantage": 12, "finnhub": 1, "newsapi": 1,
        }

        # Cache
        self._quote_cache: Dict[str, Dict] = {}
        self._quote_cache_ttl = max(1, int(os.getenv("LIVE_QUOTE_CACHE_TTL_SECONDS", "2")))
        self._intraday_cache: Dict[str, Dict] = {}
        self._intraday_cache_ttl = max(2, int(os.getenv("LIVE_INTRADAY_CACHE_TTL_SECONDS", "12")))
        self._news_cache: Dict[str, Dict] = {}
        self._news_cache_ttl = 300  # 5 min

        # Keep-alive session reduces TLS/connect overhead on rapid polling.
        self._http = requests.Session()
        self._http.mount("https://", HTTPAdapter(pool_connections=32, pool_maxsize=64, max_retries=1))
        self._http.mount("http://", HTTPAdapter(pool_connections=32, pool_maxsize=64, max_retries=1))

        logger.info(
            f"LiveDataService initialized | "
            f"FMP: {self.fmp.total_keys} key(s) ({self.fmp.total_daily_budget} calls/day) | "
            f"Alpha Vantage: {'✓' if self.alpha_vantage_key else '✗'} | "
            f"Finnhub: {'✓' if self.finnhub_key else '✗'} | "
            f"NewsAPI: {'✓' if self.news_api_key else '✗'}"
        )

    def configure_api_keys(
        self,
        fmp_keys: Optional[List[str]] = None,
        alpha_vantage: Optional[str] = None,
        finnhub: Optional[str] = None,
        news_api: Optional[str] = None,
        provider_keys: Optional[Dict[str, Union[str, List[str]]]] = None,
    ):
        """Set API keys at runtime."""
        if fmp_keys:
            self.fmp.add_keys(fmp_keys)
            self._set_provider_keys("fmp", fmp_keys)
        if alpha_vantage:
            self.alpha_vantage_key = alpha_vantage
            self._set_provider_keys("alpha_vantage", alpha_vantage)
        if finnhub:
            self.finnhub_key = finnhub
            self._set_provider_keys("finnhub", finnhub)
        if news_api:
            self.news_api_key = news_api
            self._set_provider_keys("news_api", news_api)

        if provider_keys:
            for provider, keys in provider_keys.items():
                normalized = self._normalize_provider_name(provider)
                self._set_provider_keys(normalized, keys)
                if normalized == "fmp":
                    fmp_list = self.provider_keys.get("fmp", [])
                    self.fmp.set_keys(list(dict.fromkeys([k.strip() for k in fmp_list if k.strip()])))
                elif normalized == "alpha_vantage":
                    keys_list = self.provider_keys.get("alpha_vantage", [])
                    self.alpha_vantage_key = keys_list[0] if keys_list else ""
                elif normalized == "finnhub":
                    keys_list = self.provider_keys.get("finnhub", [])
                    self.finnhub_key = keys_list[0] if keys_list else ""
                elif normalized == "news_api":
                    keys_list = self.provider_keys.get("news_api", [])
                    self.news_api_key = keys_list[0] if keys_list else ""
        logger.info("API keys updated")

    def get_provider_key_status(self) -> Dict:
        providers: Dict[str, Dict[str, Union[bool, int]]] = {}
        for provider, keys in self.provider_keys.items():
            providers[provider] = {
                "configured": len(keys) > 0,
                "keys_count": len(keys),
            }
        return providers

    def _normalize_provider_name(self, provider: str) -> str:
        return provider.strip().lower().replace("-", "_").replace(" ", "_")

    def _set_provider_keys(self, provider: str, keys: Union[str, List[str], None]):
        normalized = self._normalize_provider_name(provider)
        if keys is None:
            return
        if isinstance(keys, str):
            key_list = [k.strip() for k in keys.split(",") if k.strip()]
        else:
            key_list = [str(k).strip() for k in keys if str(k).strip()]
        self.provider_keys[normalized] = list(dict.fromkeys(key_list))

    def _load_provider_keys_from_env(self) -> Dict[str, List[str]]:
        providers: Dict[str, List[str]] = {}
        for env_name, env_value in os.environ.items():
            if not env_value:
                continue
            if env_name.endswith("_API_KEY") or env_name.endswith("_API_KEYS"):
                provider = env_name.replace("_API_KEYS", "").replace("_API_KEY", "")
                provider = self._normalize_provider_name(provider)
                keys = [k.strip() for k in env_value.split(",") if k.strip()]
                if keys:
                    providers[provider] = keys
        return providers

    # ═══════════════════════════════════════════════════════
    #  REAL-TIME PRICE
    # ═══════════════════════════════════════════════════════

    def get_realtime_quote(self, ticker: str, market: str = "SP500") -> Dict:
        """
        Get real-time price quote.
        Priority: FMP → Finnhub → yfinance → synthetic.
        """
        cache_key = f"{ticker.upper().strip()}|{market.upper().strip()}"
        cached = self._quote_cache.get(cache_key)
        if cached and (time.time() - cached.get("_ts", 0) < self._quote_cache_ttl):
            return cached["payload"]

        # 1) FMP (primary)
        if self.fmp.available:
            try:
                result = self._fmp_quote(ticker)
                if result and result.get("price", 0) > 0:
                    self._quote_cache[cache_key] = {"payload": result, "_ts": time.time()}
                    return result
            except Exception as e:
                logger.debug(f"FMP quote failed for {ticker}: {e}")

        # 2) Finnhub
        if self.finnhub_key:
            try:
                result = self._finnhub_quote(ticker)
                if result and result.get("price", 0) > 0:
                    self._quote_cache[cache_key] = {"payload": result, "_ts": time.time()}
                    return result
            except Exception as e:
                logger.debug(f"Finnhub quote failed for {ticker}: {e}")

        # 3) yfinance
        try:
            result = self._yfinance_quote(ticker, market)
            self._quote_cache[cache_key] = {"payload": result, "_ts": time.time()}
            return result
        except Exception as e:
            logger.error(f"All quote sources failed for {ticker}: {e}")
            result = self._synthetic_quote(ticker)
            self._quote_cache[cache_key] = {"payload": result, "_ts": time.time()}
            return result

    def _fmp_quote(self, ticker: str) -> Dict:
        """Fetch real-time quote from FMP."""
        api_key = self.fmp.get_key()
        if not api_key:
            return {}
        self._rate_limit("fmp")
        url = f"{FMP_BASE}/quote/{ticker}?apikey={api_key}"
        resp = self._http.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        if not data or not isinstance(data, list) or len(data) == 0:
            return {}

        q = data[0]
        price = q.get("price", 0)
        prev = q.get("previousClose", price)

        return {
            "ticker": ticker,
            "price": round(price, 2),
            "open": round(q.get("open", price), 2),
            "high": round(q.get("dayHigh", price), 2),
            "low": round(q.get("dayLow", price), 2),
            "prev_close": round(prev, 2),
            "change": round(q.get("change", 0), 2),
            "change_pct": round(q.get("changesPercentage", 0), 2),
            "volume": q.get("volume", 0),
            "market_cap": q.get("marketCap"),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "fmp",
        }

    def _finnhub_quote(self, ticker: str) -> Dict:
        """Fetch real-time quote from Finnhub."""
        self._rate_limit("finnhub")
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.finnhub_key}"
        resp = self._http.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        if data.get("c", 0) == 0:
            return {}

        return {
            "ticker": ticker,
            "price": round(data["c"], 2),
            "open": round(data["o"], 2),
            "high": round(data["h"], 2),
            "low": round(data["l"], 2),
            "prev_close": round(data["pc"], 2),
            "change": round(data["c"] - data["pc"], 2),
            "change_pct": round((data["c"] - data["pc"]) / data["pc"] * 100, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "finnhub",
        }

    def _yfinance_quote(self, ticker: str, market: str) -> Dict:
        """Fetch quote using yfinance."""
        import yfinance as yf

        suffix_map = {
            "BSE": ".BO", "NSE": ".NS", "CRYPTO": "-USD",
        }
        suffix = suffix_map.get(market, "")
        full = f"{ticker}{suffix}" if suffix and not ticker.endswith(suffix) else ticker

        stock = yf.Ticker(full)
        info = stock.fast_info

        price = float(info.last_price)
        prev = float(info.previous_close)

        return {
            "ticker": ticker,
            "price": round(price, 2),
            "open": round(float(info.open) if hasattr(info, "open") else price, 2),
            "high": round(float(info.day_high) if hasattr(info, "day_high") else price, 2),
            "low": round(float(info.day_low) if hasattr(info, "day_low") else price, 2),
            "prev_close": round(prev, 2),
            "change": round(price - prev, 2),
            "change_pct": round((price - prev) / prev * 100, 2) if prev else 0,
            "volume": int(info.last_volume) if hasattr(info, "last_volume") else 0,
            "market_cap": float(info.market_cap) if hasattr(info, "market_cap") and info.market_cap else None,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "yfinance",
        }

    def _synthetic_quote(self, ticker: str) -> Dict:
        """Generate synthetic quote for demo."""
        np.random.seed(abs(hash(ticker)) % 2**31)
        price = round(np.random.uniform(50, 500), 2)
        change_pct = round(np.random.uniform(-3, 3), 2)
        return {
            "ticker": ticker,
            "price": price,
            "prev_close": round(price / (1 + change_pct / 100), 2),
            "change": round(price * change_pct / 100, 2),
            "change_pct": change_pct,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "synthetic",
        }

    # ═══════════════════════════════════════════════════════
    #  INTRADAY CANDLES
    # ═══════════════════════════════════════════════════════

    def get_intraday_candles(
        self,
        ticker: str,
        market: str = "SP500",
        interval: str = "5min",
        output_size: str = "compact",
    ) -> List[Dict]:
        """
        Fetch intraday OHLCV candles.
        Priority: FMP → Alpha Vantage → yfinance.
        """
        cache_key = f"{ticker.upper().strip()}|{market.upper().strip()}|{interval}|{output_size}"
        cached = self._intraday_cache.get(cache_key)
        if cached and (time.time() - cached.get("_ts", 0) < self._intraday_cache_ttl):
            return cached["payload"]

        # 1) FMP intraday
        if self.fmp.available:
            try:
                candles = self._fmp_intraday(ticker, interval)
                if candles:
                    self._intraday_cache[cache_key] = {"payload": candles, "_ts": time.time()}
                    return candles
            except Exception as e:
                logger.debug(f"FMP intraday failed: {e}")

        # 2) Alpha Vantage
        if self.alpha_vantage_key:
            try:
                candles = self._alpha_vantage_intraday(ticker, interval)
                if candles:
                    self._intraday_cache[cache_key] = {"payload": candles, "_ts": time.time()}
                    return candles
            except Exception as e:
                logger.debug(f"Alpha Vantage intraday failed: {e}")

        # 3) yfinance
        try:
            candles = self._yfinance_intraday(ticker, market, interval)
            if candles:
                self._intraday_cache[cache_key] = {"payload": candles, "_ts": time.time()}
            return candles
        except Exception as e:
            logger.error(f"All intraday sources failed for {ticker}: {e}")
            return []

    def _fmp_intraday(self, ticker: str, interval: str) -> List[Dict]:
        """Fetch intraday candles from FMP."""
        api_key = self.fmp.get_key()
        if not api_key:
            return []
        self._rate_limit("fmp")

        # Map intervals to FMP format
        fmp_interval = {
            "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1hour",
            "1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min", "60min": "1hour",
        }.get(interval, "5min")

        url = f"{FMP_BASE}/historical-chart/{fmp_interval}/{ticker}?apikey={api_key}"
        resp = self._http.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data or not isinstance(data, list):
            return []

        candles = []
        for item in data:
            try:
                ts = datetime.strptime(item["date"], "%Y-%m-%d %H:%M:%S")
                candles.append({
                    "time": int(ts.timestamp()),
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": int(item.get("volume", 0)),
                })
            except (KeyError, ValueError):
                continue

        candles.sort(key=lambda x: x["time"])
        return candles

        # Fallback: yfinance intraday
        try:
            return self._yfinance_intraday(ticker, market, interval)
        except Exception as e:
            logger.error(f"All intraday sources failed for {ticker}: {e}")
            return []

    def _alpha_vantage_intraday(self, ticker: str, interval: str) -> List[Dict]:
        """Fetch intraday data from Alpha Vantage."""
        self._rate_limit("alpha_vantage")

        av_interval = {
            "1m": "1min", "5m": "5min", "15m": "15min",
            "30m": "30min", "1h": "60min",
            "1min": "1min", "5min": "5min", "15min": "15min",
            "30min": "30min", "60min": "60min",
        }.get(interval, "5min")

        url = (
            f"https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_INTRADAY"
            f"&symbol={ticker}&interval={av_interval}"
            f"&outputsize=compact&apikey={self.alpha_vantage_key}"
        )
        resp = self._http.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        ts_key = f"Time Series ({av_interval})"
        if ts_key not in data:
            return []

        candles = []
        for timestamp_str, values in data[ts_key].items():
            ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            candles.append({
                "time": int(ts.timestamp()),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": int(values["5. volume"]),
            })

        candles.sort(key=lambda x: x["time"])
        return candles

    def _yfinance_intraday(self, ticker: str, market: str, interval: str) -> List[Dict]:
        """Fetch intraday candles from yfinance."""
        import yfinance as yf

        suffix_map = {"BSE": ".BO", "NSE": ".NS", "CRYPTO": "-USD"}
        suffix = suffix_map.get(market, "")
        full = f"{ticker}{suffix}" if suffix and not ticker.endswith(suffix) else ticker

        # yfinance interval format
        yf_interval = {
            "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "60min": "1h",
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h",
        }.get(interval, "5m")

        # yfinance limits: 1m data only for 7 days, 5m for 60 days
        period = "1d" if yf_interval == "1m" else "5d"

        stock = yf.Ticker(full)
        df = stock.history(period=period, interval=yf_interval)

        if df.empty:
            return []

        candles = []
        for idx, row in df.iterrows():
            candles.append({
                "time": int(idx.timestamp()),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row.get("Volume", 0)),
            })

        return candles

    # ═══════════════════════════════════════════════════════
    #  NEWS
    # ═══════════════════════════════════════════════════════

    def get_live_news(self, ticker: str, market: str = "SP500") -> Dict:
        """
        Fetch latest news headlines for a ticker.
        Priority: FMP → Finnhub → NewsAPI → yfinance → Demo fallback.
        """
        cache_key = f"news_{ticker}"
        if cache_key in self._news_cache:
            cached = self._news_cache[cache_key]
            if time.time() - cached.get("_ts", 0) < self._news_cache_ttl:
                return cached

        articles = []

        # 1) FMP stock news
        if self.fmp.available:
            try:
                articles = self._fmp_news(ticker)
            except Exception:
                pass

        # 2) Finnhub company news
        if not articles and self.finnhub_key:
            try:
                articles = self._finnhub_news(ticker)
            except Exception:
                pass

        # 3) NewsAPI
        if not articles and self.news_api_key:
            try:
                articles = self._newsapi_search(ticker)
            except Exception:
                pass

        # 4) yfinance news
        if not articles:
            try:
                articles = self._yfinance_news(ticker, market)
            except Exception:
                pass

        # 5) Fallback demo
        if not articles:
            articles = self._demo_news(ticker)

        result = {
            "ticker": ticker,
            "article_count": len(articles),
            "articles": articles[:20],
            "timestamp": datetime.utcnow().isoformat(),
            "_ts": time.time(),
        }
        self._news_cache[cache_key] = result
        return result

    def get_cached_or_demo_news(self, ticker: str) -> Dict:
        """Return cached news instantly or demo fallback for timeout-safe API responses."""
        cache_key = f"news_{ticker}"
        cached = self._news_cache.get(cache_key)
        if cached and (time.time() - cached.get("_ts", 0) < self._news_cache_ttl):
            return cached

        articles = self._demo_news(ticker)
        result = {
            "ticker": ticker,
            "article_count": len(articles),
            "articles": articles[:20],
            "timestamp": datetime.utcnow().isoformat(),
            "_ts": time.time(),
        }
        self._news_cache[cache_key] = result
        return result

    def _fmp_news(self, ticker: str) -> List[Dict]:
        """Fetch stock news from FMP."""
        api_key = self.fmp.get_key()
        if not api_key:
            return []
        self._rate_limit("fmp")
        url = f"{FMP_BASE}/stock_news?tickers={ticker}&limit=20&apikey={api_key}"
        resp = self._http.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data or not isinstance(data, list):
            return []

        return [
            {
                "title": item.get("title", ""),
                "summary": item.get("text", ""),
                "source": item.get("site", ""),
                "url": item.get("url", ""),
                "published_at": item.get("publishedDate", ""),
                "image": item.get("image", ""),
            }
            for item in data[:20]
        ]

        result = {
            "ticker": ticker,
            "article_count": len(articles),
            "articles": articles[:20],
            "timestamp": datetime.utcnow().isoformat(),
            "_ts": time.time(),
        }
        self._news_cache[cache_key] = result
        return result

    def _finnhub_news(self, ticker: str) -> List[Dict]:
        """Fetch company news from Finnhub."""
        self._rate_limit("finnhub")
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}&from={week_ago}&to={today}"
            f"&token={self.finnhub_key}"
        )
        resp = self._http.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        return [
            {
                "title": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "published_at": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
            }
            for item in data[:20]
        ]

    def _newsapi_search(self, ticker: str) -> List[Dict]:
        """Search for news via NewsAPI."""
        self._rate_limit("newsapi")
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={ticker}+stock&language=en"
            f"&sortBy=publishedAt&pageSize=20"
            f"&apiKey={self.news_api_key}"
        )
        resp = self._http.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        return [
            {
                "title": art.get("title", ""),
                "summary": art.get("description", ""),
                "source": art.get("source", {}).get("name", ""),
                "url": art.get("url", ""),
                "published_at": art.get("publishedAt", ""),
            }
            for art in data.get("articles", [])
        ]

    def _yfinance_news(self, ticker: str, market: str) -> List[Dict]:
        """Fetch news from yfinance."""
        import yfinance as yf

        suffix_map = {"BSE": ".BO", "NSE": ".NS", "CRYPTO": "-USD"}
        suffix = suffix_map.get(market, "")
        full = f"{ticker}{suffix}" if suffix and not ticker.endswith(suffix) else ticker

        stock = yf.Ticker(full)
        news = stock.news if hasattr(stock, "news") else []

        return [
            {
                "title": item.get("title", ""),
                "summary": item.get("title", ""),
                "source": item.get("publisher", ""),
                "url": item.get("link", ""),
                "published_at": datetime.fromtimestamp(
                    item.get("providerPublishTime", 0)
                ).isoformat() if item.get("providerPublishTime") else "",
            }
            for item in (news or [])[:20]
        ]

    def _demo_news(self, ticker: str) -> List[Dict]:
        """Generate demo news for display."""
        headlines = [
            f"{ticker} reports strong quarterly earnings, beating Wall Street estimates",
            f"Analysts upgrade {ticker} citing robust growth pipeline and market momentum",
            f"{ticker} announces strategic acquisition, expanding market presence",
            f"Market outlook for {ticker}: Institutional investors increase positions",
            f"{ticker} unveils new product lineup, shares respond positively",
            f"Technical analysis: {ticker} breaks above key resistance level",
            f"Options activity surges for {ticker} ahead of earnings",
            f"{ticker} CEO outlines 5-year growth strategy at investor conference",
        ]
        return [
            {
                "title": h,
                "summary": h,
                "source": "FINSENT Research",
                "url": "#",
                "published_at": datetime.utcnow().isoformat(),
            }
            for h in headlines
        ]

    # ═══════════════════════════════════════════════════════
    #  FMP HISTORICAL DATA
    # ═══════════════════════════════════════════════════════

    def get_fmp_historical(
        self, ticker: str, period: str = "5y"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily historical OHLCV from FMP.
        Returns DataFrame with columns: [Open, High, Low, Close, Volume]
        """
        api_key = self.fmp.get_key()
        if not api_key:
            return None
        self._rate_limit("fmp")

        # Map period string to approximate date range
        period_days = {
            "1mo": 30, "3mo": 90, "6mo": 180,
            "1y": 365, "2y": 730, "3y": 1095,
            "5y": 1825, "10y": 3650, "max": 36500,
        }
        days = period_days.get(period, 1825)
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")

        url = (
            f"{FMP_BASE}/historical-price-full/{ticker}"
            f"?from={from_date}&to={to_date}&apikey={api_key}"
        )
        resp = self._http.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        historical = data.get("historical", [])
        if not historical:
            return None

        df = pd.DataFrame(historical)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # Rename to standard OHLCV columns
        col_map = {
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        }
        df = df.rename(columns=col_map)

        required = ["Open", "High", "Low", "Close", "Volume"]
        for c in required:
            if c not in df.columns:
                return None

        return df[required]

    def get_fmp_key_status(self) -> Dict:
        """Return FMP key rotation status with call counts."""
        return self.fmp.get_status()

    # ═══════════════════════════════════════════════════════
    #  UTILITIES
    # ═══════════════════════════════════════════════════════

    def _rate_limit(self, source: str):
        """Simple rate limiting per source."""
        now = time.time()
        min_gap = self._min_interval.get(source, 1)
        last = self._last_request_time.get(source, 0)
        wait = min_gap - (now - last)
        if wait > 0:
            time.sleep(wait)
        self._last_request_time[source] = time.time()
