"""
FINSENT NET PRO — Market Data Fetcher
Fetches OHLCV data from multiple sources with intelligent fallback.
Supports: S&P500, NASDAQ, NYSE, BSE, NSE, Commodities, Crypto

Data Sources (priority):
    1. FMP (Financial Modeling Prep) — primary, 250 calls/day/key
  2. yfinance — free fallback, no key needed
  3. Synthetic — demo fallback when all APIs fail
"""

import os
import logging
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datetime import date
from typing import Optional, Dict, List
import time

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

logger = logging.getLogger("finsent.market_data")

FMP_BASE = "https://financialmodelingprep.com/api/v3"
FMP_DAILY_LIMIT = 250


class MarketDataFetcher:
    """
    Unified data fetcher supporting all global markets.
    FMP (Financial Modeling Prep) is the primary source.
    Falls back to yfinance when FMP keys are exhausted.
    Implements strict temporal integrity — no look-ahead contamination.
    """

    MARKET_CONFIG = {
        "SP500": {
            "index_ticker": "^GSPC",
            "suffix": "",
            "currency": "USD",
            "timezone": "America/New_York",
            "trading_hours": {"open": "09:30", "close": "16:00"},
        },
        "NASDAQ": {
            "index_ticker": "^IXIC",
            "suffix": "",
            "currency": "USD",
            "timezone": "America/New_York",
            "trading_hours": {"open": "09:30", "close": "16:00"},
        },
        "NYSE": {
            "index_ticker": "^NYA",
            "suffix": "",
            "currency": "USD",
            "timezone": "America/New_York",
            "trading_hours": {"open": "09:30", "close": "16:00"},
        },
        "BSE": {
            "index_ticker": "^BSESN",
            "suffix": ".BO",
            "currency": "INR",
            "timezone": "Asia/Kolkata",
            "trading_hours": {"open": "09:15", "close": "15:30"},
        },
        "NSE": {
            "index_ticker": "^NSEI",
            "suffix": ".NS",
            "currency": "INR",
            "timezone": "Asia/Kolkata",
            "trading_hours": {"open": "09:15", "close": "15:30"},
        },
        "COMMODITIES": {
            "tickers": {
                "GOLD": "GC=F",
                "SILVER": "SI=F",
                "CRUDE_OIL": "CL=F",
                "NATURAL_GAS": "NG=F",
                "COPPER": "HG=F",
                "WHEAT": "ZW=F",
                "CORN": "ZC=F",
                "PLATINUM": "PL=F",
            },
            "suffix": "",
            "currency": "USD",
        },
        "CRYPTO": {
            "suffix": "-USD",
            "top_symbols": [
                "BTC", "ETH", "BNB", "SOL", "ADA", "AVAX",
                "DOT", "MATIC", "LINK", "UNI", "XRP", "DOGE",
            ],
            "currency": "USD",
        },
    }

    # Hard-coded component lists for reliability
    SENSEX_TICKERS = [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "BHARTIARTL",
        "SBIN", "INFY", "HINDUNILVR", "ITC", "LT",
        "KOTAKBANK", "AXISBANK", "BAJFINANCE", "MARUTI", "SUNPHARMA",
        "WIPRO", "ULTRACEMCO", "NESTLEIND", "POWERGRID", "TITAN",
        "HCLTECH", "ADANIENT", "ONGC", "NTPC", "COALINDIA",
        "M&M", "BAJAJFINSV", "ASIANPAINT", "TATAMOTORS", "JSWSTEEL",
    ]

    NIFTY100_EXTRA_TICKERS = [
        "ADANIPORTS", "BAJAJ-AUTO", "BEL", "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL",
        "DLF", "DMART", "GAIL", "GODREJCP", "HAL", "HAVELLS", "HINDPETRO", "IDFCFIRSTB",
        "IGL", "INDIGO", "IOC", "JINDALSTEL", "JSWENERGY", "LICI", "LODHA", "MOTHERSON",
        "NAUKRI", "NMDC", "OBEROIRLTY", "OFSS", "PAGEIND", "PEL", "PFC", "PIDILITIND",
        "PNB", "POLYCAB", "RECLTD", "SAIL", "SIEMENS", "SRF", "TATAPOWER", "TORNTPHARM",
        "TVSMOTOR", "VBL", "ZYDUSLIFE", "AUROPHARMA", "ASHOKLEY", "BHEL", "CONCOR", "CUMMINSIND",
    ]

    NIFTY50_TICKERS = [
        "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK",
        "SBIN", "INFY", "HINDUNILVR", "ITC", "LT",
        "KOTAKBANK", "AXISBANK", "BAJFINANCE", "MARUTI", "SUNPHARMA",
        "WIPRO", "ULTRACEMCO", "NESTLEIND", "POWERGRID", "TITAN",
        "HCLTECH", "ADANIENT", "ONGC", "NTPC", "COALINDIA",
        "M&M", "BAJAJFINSV", "ASIANPAINT", "TATAMOTORS", "JSWSTEEL",
        "TATASTEEL", "TECHM", "CIPLA", "BPCL", "SHREECEM",
        "BRITANNIA", "DRREDDY", "EICHERMOT", "HEROMOTOCO", "HINDALCO",
        "APOLLOHOSP", "DIVISLAB", "GRASIM", "INDUSINDBK", "SBILIFE",
        "HDFCLIFE", "TATACONSUM", "UPL", "VEDL", "DABUR",
    ]

    TOP_SP500 = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B",
        "LLY", "TSM", "AVGO", "TSLA", "JPM", "V", "WMT", "XOM",
        "ORCL", "MA", "UNH", "JNJ", "PG", "HD", "COST", "ABBV",
        "CRM", "BAC", "NFLX", "KO", "MRK", "AMD", "PEP", "CVX", "ACN", "ADBE",
        "DIS", "CSCO", "DHR", "MCD", "ABT", "TMO", "NKE", "VZ", "CMCSA", "QCOM",
        "TXN", "LIN", "NEE", "HON", "PM", "RTX", "UNP", "LOW", "UPS", "IBM",
        "SCHW", "AMGN", "SBUX", "INTU", "BKNG", "GS", "BLK", "SPGI", "ISRG", "CAT",
        "DE", "GE", "MDT", "SYK", "BA", "LMT", "PLD", "CI", "MU", "ADP",
        "ELV", "AXP", "CB", "TGT", "MMC", "GILD", "MO", "VRTX", "PGR", "SO",
        "DUK", "EQIX", "TMUS", "REGN", "PNC", "AON", "USB", "AMT", "C", "WFC",
        "FIS", "PAYX", "ITW", "CSX", "NSC", "EMR", "ECL", "KLAC", "APH", "MSI",
        "MAR", "HCA", "EW", "ADM", "SHW", "ZTS", "ETN", "CDNS", "ORLY", "ADSK",
        "CMG", "ROST", "AEP", "F", "GM", "MPC", "PSX", "OXY", "SLB", "PXD",
    ]

    NASDAQ100 = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "AVGO", "GOOGL",
        "TSLA", "COST", "NFLX", "AMD", "QCOM", "ADBE", "PEP",
        "INTC", "CSCO", "CMCSA", "INTU", "AMGN", "TMUS",
        "AMAT", "TXN", "ISRG", "MU", "LRCX", "BKNG", "MDLZ",
        "ADI", "REGN", "SNPS", "PANW", "MELI", "CDNS", "KLAC", "CRWD", "MAR", "ORLY",
        "FTNT", "MRVL", "ABNB", "ADSK", "WDAY", "CSX", "ASML", "NXPI", "PCAR", "PYPL",
        "CHTR", "MNST", "KDP", "CTAS", "IDXX", "TEAM", "DDOG", "ROP", "ROST", "EA",
        "CTSH", "BIIB", "FAST", "XEL", "AEP", "CPRT", "ODFL", "VRSK", "EXC", "PAYX",
        "DLTR", "GEHC", "SIRI", "BKR", "WBD", "JD", "PDD", "BIDU", "NTES", "LULU",
        "ZS", "MDB", "DOCU", "ANSS", "MCHP", "ILMN", "KHC", "GFS", "ON", "WBA",
        "DXCM", "CEG", "FANG", "CSGP", "MRNA", "AAL", "UAL", "NDAQ", "TTWO", "VRSN",
    ]

    NYSE_LARGE_CAP = [
        "BRK-B", "JPM", "V", "WMT", "XOM", "MA", "UNH", "JNJ", "PG", "HD",
        "BAC", "KO", "MRK", "ABBV", "CVX", "MCD", "DHR", "LIN", "PFE", "TMO",
        "ABT", "NKE", "TXN", "VZ", "PM", "BMY", "UNP", "NEE", "LOW", "UPS",
        "AXP", "GS", "MS", "BLK", "SCHW", "C", "WFC", "USB", "PNC", "BK",
        "SPGI", "ICE", "MMC", "AON", "CB", "TRV", "AIG", "MET", "PRU", "ALL",
        "RTX", "LMT", "NOC", "GD", "HWM", "GE", "BA", "CAT", "DE", "EMR",
        "HON", "MMM", "ETN", "PH", "ROP", "ITW", "OTIS", "CMI", "PCAR", "NSC",
        "CSX", "UNP", "FDX", "DOW", "DD", "APD", "ECL", "SHW", "NEM", "FCX",
        "OXY", "SLB", "HAL", "MPC", "PSX", "VLO", "KMI", "OKE", "WMB", "EOG",
        "SO", "DUK", "AEP", "D", "EXC", "XEL", "SRE", "AFL", "HIG", "AMP",
        "MO", "PM", "KMB", "CL", "GIS", "K", "EL", "TGT", "COST", "DG",
    ]

    def __init__(self):
        self._session_cache: Dict[str, pd.DataFrame] = {}

        # Load FMP keys from environment (comma-separated)
        fmp_keys_raw = os.environ.get("FMP_API_KEYS", "")
        if not str(fmp_keys_raw).strip():
            fmp_keys_raw = os.environ.get("FMP_API_KEY", "")
        self._fmp_keys = [k.strip() for k in fmp_keys_raw.split(",") if k.strip()]
        self._fmp_call_counts: Dict[str, Dict[str, int]] = {}
        self._fmp_disabled_keys: set = set()
        self._fmp_index = 0
        self._fmp_exhausted_on: Optional[str] = None

        self.synthetic_fallback_enabled = str(
            os.environ.get("ALLOW_SYNTHETIC_FALLBACK", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}

        if self._fmp_keys:
            logger.info(
                f"MarketDataFetcher: {len(self._fmp_keys)} FMP key(s) loaded "
                f"({len(self._fmp_keys) * FMP_DAILY_LIMIT} calls/day)"
            )

        # Optional Zerodha Kite fallback (mainly for NSE/BSE daily candles).
        self.kite_api_key = os.environ.get("KITE_API_KEY", "").strip()
        self.kite_access_token = os.environ.get("KITE_ACCESS_TOKEN", "").strip()
        self._kite = None
        self._kite_instruments: Dict[str, int] = {}

        if self.kite_api_key and self.kite_access_token and KiteConnect is not None:
            try:
                self._kite = KiteConnect(api_key=self.kite_api_key)
                self._kite.set_access_token(self.kite_access_token)
                self._load_kite_instruments()
                logger.info(
                    f"MarketDataFetcher: Kite enabled with {len(self._kite_instruments)} instruments"
                )
            except Exception as exc:
                logger.warning(f"Kite initialization failed: {exc}")
                self._kite = None
        elif self.kite_api_key and KiteConnect is None:
            logger.warning("KITE_API_KEY is set but kiteconnect package is not installed")
        elif self.kite_api_key and not self.kite_access_token:
            logger.warning("KITE_API_KEY is set but KITE_ACCESS_TOKEN is missing; Kite historical fallback is unavailable")

    def _get_fmp_key(self) -> Optional[str]:
        """Get next available FMP key with round-robin rotation."""
        if not self._fmp_keys:
            return None

        today = date.today().isoformat()

        # If all keys are known exhausted for the current day, skip repeated scans.
        if self._fmp_exhausted_on == today:
            return None

        tried = 0
        while tried < len(self._fmp_keys):
            key = self._fmp_keys[self._fmp_index]
            if key in self._fmp_disabled_keys:
                self._fmp_index = (self._fmp_index + 1) % len(self._fmp_keys)
                tried += 1
                continue
            counts = self._fmp_call_counts.get(key, {})
            used = counts.get(today, 0)
            if used < FMP_DAILY_LIMIT:
                return key
            self._fmp_index = (self._fmp_index + 1) % len(self._fmp_keys)
            tried += 1

        self._fmp_exhausted_on = today
        logger.warning("All FMP keys exhausted for today")
        return None

    def _mark_fmp_key_used(self, key: str):
        today = date.today().isoformat()
        counts = self._fmp_call_counts.get(key, {})
        used = counts.get(today, 0)
        self._fmp_call_counts[key] = {today: used + 1}

    def _disable_fmp_key(self, key: str):
        self._fmp_disabled_keys.add(key)
        logger.warning("Disabled FMP key due to authorization failure")

    def set_fmp_keys(self, keys: List[str]):
        """Set FMP keys at runtime."""
        self._fmp_keys = [k.strip() for k in keys if k.strip()]
        self._fmp_call_counts = {}
        self._fmp_disabled_keys = set()
        self._fmp_index = 0
        self._fmp_exhausted_on = None
        logger.info(f"FMP keys updated: {len(self._fmp_keys)} total")

    def fetch_ohlcv(
        self,
        ticker: str,
        market: str,
        period: str = "5y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data. Tries FMP first, falls back to yfinance.

        Returns:
            DataFrame with columns: [Open, High, Low, Close, Volume]
            Index: DatetimeIndex
        """
        config = self.MARKET_CONFIG.get(market, {})
        suffix = config.get("suffix", "")

        # For commodities, use the predefined ticker mapping
        if market == "COMMODITIES":
            commodity_map = config.get("tickers", {})
            full_ticker = commodity_map.get(ticker.upper(), ticker)
        elif market == "CRYPTO" and not ticker.endswith("-USD"):
            full_ticker = f"{ticker}-USD"
        else:
            full_ticker = f"{ticker}{suffix}" if suffix and not ticker.endswith(suffix) else ticker

        cache_key = f"{full_ticker}_{period}_{interval}"
        if cache_key in self._session_cache:
            return self._session_cache[cache_key].copy()

        # If FMP budget is exhausted for the day, prefer Kite immediately for NSE/BSE.
        fmp_exhausted_today = self._fmp_exhausted_on == date.today().isoformat()
        prefer_kite = fmp_exhausted_today and market in {"NSE", "BSE"}

        # 1) Try FMP (primary source for daily data), unless Kite-first mode is active.
        if self._fmp_keys and interval == "1d" and not prefer_kite:
            try:
                df = self._fetch_fmp_historical(ticker, period)
                if df is not None and not df.empty:
                    df = self._validate_and_clean_ohlcv(df, ticker)
                    self._session_cache[cache_key] = df
                    logger.info(f"FMP: fetched {len(df)} bars for {ticker}")
                    return df.copy()
            except Exception as e:
                logger.debug(f"FMP historical failed for {ticker}: {e}")

        # 1.5) Optional Kite fallback for India exchanges.
        if self._kite and interval == "1d" and market in {"NSE", "BSE"}:
            try:
                df = self._fetch_kite_historical(ticker, market, period)
                if df is not None and not df.empty:
                    df = self._validate_and_clean_ohlcv(df, ticker)
                    self._session_cache[cache_key] = df
                    logger.info(f"Kite: fetched {len(df)} bars for {ticker}")
                    return df.copy()
            except Exception as e:
                logger.debug(f"Kite historical failed for {ticker}: {e}")

        # 2) Fallback: yfinance
        try:
            stock = yf.Ticker(full_ticker)
            df = stock.history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                raise ValueError(f"No data returned for {full_ticker}")

            df = self._validate_and_clean_ohlcv(df, full_ticker)
            self._session_cache[cache_key] = df
            return df.copy()

        except Exception as e:
            logger.warning(f"All fetch sources failed for {full_ticker}: {e}")
            if self.synthetic_fallback_enabled:
                return self._fallback_synthetic(ticker, period)
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    def _fetch_fmp_historical(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV from FMP historical-price-full endpoint."""
        api_key = self._get_fmp_key()
        if not api_key:
            return None

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
        resp = requests.get(url, timeout=15)
        if resp.status_code in (401, 403):
            self._disable_fmp_key(api_key)
            return None
        resp.raise_for_status()
        data = resp.json()

        historical = data.get("historical", [])
        if not historical:
            return None

        df = pd.DataFrame(historical)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })

        required = ["Open", "High", "Low", "Close", "Volume"]
        for c in required:
            if c not in df.columns:
                return None

        self._mark_fmp_key_used(api_key)
        return df[required]

    def _validate_and_clean_ohlcv(
        self, df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """Validates and cleans OHLCV data."""
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # OHLC consistency: High >= Low
        invalid_ohlc = df["High"] < df["Low"]
        if invalid_ohlc.sum() > 0:
            print(f"[WARNING] {invalid_ohlc.sum()} OHLC inconsistencies in {ticker}")
            df = df[~invalid_ohlc]

        # Remove zero-volume days
        df = df[df["Volume"] > 0]

        # Forward fill small gaps (holidays)
        df = df.ffill(limit=5).dropna()

        # Detect extreme single-day moves (>50% = likely data error)
        daily_returns = df["Close"].pct_change().abs()
        suspect = daily_returns > 0.50
        if suspect.sum() > 0:
            print(f"[ALERT] {suspect.sum()} extreme price moves in {ticker}")

        return df

    def get_live_price(self, ticker: str, market: str) -> Dict:
        """Returns real-time price data for a single ticker."""
        config = self.MARKET_CONFIG.get(market, {})
        suffix = config.get("suffix", "")

        if market == "COMMODITIES":
            commodity_map = config.get("tickers", {})
            full_ticker = commodity_map.get(ticker.upper(), ticker)
        elif market == "CRYPTO" and not ticker.endswith("-USD"):
            full_ticker = f"{ticker}-USD"
        else:
            full_ticker = f"{ticker}{suffix}" if suffix and not ticker.endswith(suffix) else ticker

        try:
            stock = yf.Ticker(full_ticker)
            info = stock.fast_info

            last_price = float(info.last_price)
            prev_close = float(info.previous_close)
            change = last_price - prev_close
            change_pct = (change / prev_close * 100) if prev_close != 0 else 0

            return {
                "ticker": ticker,
                "market": market,
                "price": round(last_price, 2),
                "prev_close": round(prev_close, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "volume": int(info.last_volume) if hasattr(info, "last_volume") else 0,
                "market_cap": float(info.market_cap) if hasattr(info, "market_cap") and info.market_cap else None,
                "currency": config.get("currency", "USD"),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            # Fallback with synthetic data for demo
            np.random.seed(abs(hash(ticker)) % 2**31)
            base = np.random.uniform(50, 500)
            change_pct = np.random.uniform(-5, 5)
            return {
                "ticker": ticker,
                "market": market,
                "price": round(base, 2),
                "prev_close": round(base / (1 + change_pct / 100), 2),
                "change": round(base * change_pct / 100, 2),
                "change_pct": round(change_pct, 2),
                "volume": int(np.random.uniform(1e6, 50e6)),
                "market_cap": None,
                "currency": config.get("currency", "USD"),
                "timestamp": datetime.now().isoformat(),
            }

    def get_market_components(self, market: str, limit: int = 100) -> List[str]:
        """Returns list of ticker symbols for a given market."""
        if market == "SP500":
            tickers = self._get_sp500_tickers()
        elif market == "NASDAQ":
            tickers = self._get_nasdaq_tickers()
        elif market == "NYSE":
            tickers = self._get_nyse_tickers()
        elif market == "NSE":
            tickers = self._get_india_exchange_tickers(exchange="NSE")
        elif market == "BSE":
            tickers = self._get_india_exchange_tickers(exchange="BSE")
        elif market == "CRYPTO":
            symbols = self.MARKET_CONFIG["CRYPTO"]["top_symbols"]
            tickers = [f"{s}-USD" for s in symbols]
        elif market == "COMMODITIES":
            tickers = list(self.MARKET_CONFIG["COMMODITIES"]["tickers"].values())
        else:
            tickers = []
        return tickers[:limit]

    def _get_sp500_tickers(self) -> List[str]:
        """Fetches S&P 500 components from Wikipedia with local fallback."""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        except Exception:
            return self.TOP_SP500[:]

    def _get_nasdaq_tickers(self) -> List[str]:
        """Returns an expanded NASDAQ universe, preferring FMP exchange screener."""
        listed = self._get_nasdaq_trader_symbols(market="NASDAQ")
        if listed:
            return self._merge_unique(listed, self.NASDAQ100)

        dynamic = self._fetch_fmp_exchange_symbols(exchange="NASDAQ", max_count=1500)
        if dynamic:
            return self._merge_unique(dynamic, self.NASDAQ100)
        return self.NASDAQ100[:]

    def _get_nyse_tickers(self) -> List[str]:
        """Returns an expanded NYSE universe, preferring FMP exchange screener."""
        listed = self._get_nasdaq_trader_symbols(market="NYSE")
        if listed:
            return self._merge_unique(listed, self.NYSE_LARGE_CAP)

        dynamic = self._fetch_fmp_exchange_symbols(exchange="NYSE", max_count=1500)
        if dynamic:
            return self._merge_unique(dynamic, self.NYSE_LARGE_CAP)
        return self._merge_unique(self.NYSE_LARGE_CAP, self.TOP_SP500)

    BSE_EXTRA_TICKERS = [
        "TATAELXSI", "BANKBARODA", "DIXON", "IRCTC", "NHPC", "MUTHOOTFIN", "GMRINFRA",
        "TATACHEM", "ESCORTS", "GLENMARK", "BIOCON", "ABB", "ATGL", "MPHASIS", "MCDOWELL-N",
        "PETRONET", "MFSL", "LTTS", "BANDHANBNK", "MANAPPURAM", "EXIDEIND", "AUBANK", "DEEPAKNTR",
        "INDUSTOWER", "JUBLFOOD", "PAYTM", "COFORGE", "SONACOMS", "SUPREMEIND", "ACC", "AMBUJACEM",
        "HONAUT", "BALKRISIND", "TATACOMM", "BERGEPAINT", "DALBHARAT", "HINDZINC", "TRENT", "VOLTAS",
        "UBL", "ZEEL", "WELCORP", "TORNTPOWER", "SUNTV", "LUPIN", "M&MFIN", "MARICO",
    ]

    def _get_india_exchange_tickers(self, exchange: str) -> List[str]:
        """Returns an expanded India exchange universe with stable fallbacks."""
        exchange = exchange.upper()
        suffix = ".NS" if exchange == "NSE" else ".BO"

        # If Kite is authenticated, use the full instrument list from the broker feed.
        if self._kite_instruments:
            kite_list = []
            prefix = f"{exchange}:"
            for key in self._kite_instruments.keys():
                if key.startswith(prefix):
                    sym = key.split(":", 1)[1]
                    kite_list.append(f"{sym}{suffix}")
            if kite_list:
                base_fallback = self.NIFTY50_TICKERS if exchange == "NSE" else self.SENSEX_TICKERS
                extra = self.NIFTY100_EXTRA_TICKERS if exchange == "NSE" else self.BSE_EXTRA_TICKERS
                fallback = [f"{t}{suffix}" for t in (base_fallback + extra)]
                return self._merge_unique(kite_list, fallback)

        dynamic = self._fetch_fmp_exchange_symbols(exchange=exchange, max_count=2000)
        if dynamic:
            normalized = [t if t.endswith(suffix) else f"{t}{suffix}" for t in dynamic]
            fallback_base = self.NIFTY50_TICKERS if exchange == "NSE" else self.SENSEX_TICKERS
            fallback_extra = self.NIFTY100_EXTRA_TICKERS if exchange == "NSE" else self.BSE_EXTRA_TICKERS
            fallback = fallback_base + fallback_extra
            fallback = [f"{t}{suffix}" for t in fallback]
            return self._merge_unique(normalized, fallback)

        fallback = self.NIFTY50_TICKERS if exchange == "NSE" else self.SENSEX_TICKERS
        fallback = fallback + (self.NIFTY100_EXTRA_TICKERS if exchange == "NSE" else self.BSE_EXTRA_TICKERS)
        return [f"{t}{suffix}" for t in fallback]

    def _get_nasdaq_trader_symbols(self, market: str) -> List[str]:
        """
        Fetch broad US listed symbols from Nasdaq Trader reference files.
        market: NASDAQ or NYSE
        """
        market = market.upper()
        try:
            if market == "NASDAQ":
                # Pipe-delimited file with Symbol column.
                df = pd.read_csv(
                    "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                    sep="|",
                    dtype=str,
                )
                if "Symbol" not in df.columns:
                    return []
                symbols = [s.strip() for s in df["Symbol"].tolist() if isinstance(s, str) and s.strip()]
                symbols = [s for s in symbols if "File Creation Time" not in s]
                return self._merge_unique(symbols)

            if market == "NYSE":
                df = pd.read_csv(
                    "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                    sep="|",
                    dtype=str,
                )
                if "ACT Symbol" not in df.columns or "Exchange" not in df.columns:
                    return []
                nyse = df[df["Exchange"] == "N"]
                symbols = [s.strip() for s in nyse["ACT Symbol"].tolist() if isinstance(s, str) and s.strip()]
                symbols = [s for s in symbols if "File Creation Time" not in s]
                return self._merge_unique(symbols)
        except Exception as exc:
            logger.debug(f"Nasdaq Trader symbol fetch failed for {market}: {exc}")
        return []

    def _fetch_fmp_exchange_symbols(self, exchange: str, max_count: int = 1000) -> List[str]:
        """
        Fetch a broad exchange symbol list from FMP stock screener.
        Falls back gracefully when endpoint/rate-limit/auth is unavailable.
        """
        api_key = self._get_fmp_key()
        if not api_key:
            return []

        symbols: List[str] = []
        page = 0
        page_size = 1000

        while len(symbols) < max_count and page < 5:
            try:
                remaining = max_count - len(symbols)
                limit = page_size if remaining > page_size else remaining
                url = (
                    f"{FMP_BASE}/stock-screener?exchange={exchange}"
                    f"&isEtf=false&isFund=false&limit={limit}&page={page}&apikey={api_key}"
                )
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list) or not data:
                    break

                added = 0
                for row in data:
                    symbol = str(row.get("symbol", "")).strip()
                    if not symbol:
                        continue
                    symbol = symbol.replace(".", "-") if exchange in {"NYSE", "NASDAQ"} else symbol
                    symbols.append(symbol)
                    added += 1

                if added == 0:
                    break
                page += 1
                time.sleep(0.25)
            except Exception as exc:
                logger.debug(f"FMP exchange screener failed for {exchange}: {exc}")
                break

        return self._merge_unique(symbols)

    def _load_kite_instruments(self):
        """Load NSE/BSE instrument-token maps for Kite historical API."""
        if not self._kite:
            return

        mapping: Dict[str, int] = {}
        for exchange in ("NSE", "BSE"):
            try:
                rows = self._kite.instruments(exchange)
                for item in rows:
                    symbol = str(item.get("tradingsymbol", "")).strip().upper()
                    token = item.get("instrument_token")
                    if not symbol or token is None:
                        continue
                    mapping[f"{exchange}:{symbol}"] = int(token)
            except Exception as exc:
                logger.debug(f"Kite instrument load failed for {exchange}: {exc}")
        self._kite_instruments = mapping

    def _fetch_kite_historical(self, ticker: str, market: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV using Zerodha Kite historical API for NSE/BSE."""
        if not self._kite:
            return None

        market = market.upper()
        base_ticker = ticker.replace(".NS", "").replace(".BO", "").upper()
        instrument_token = self._kite_instruments.get(f"{market}:{base_ticker}")
        if not instrument_token:
            return None

        period_days = {
            "1mo": 30, "3mo": 90, "6mo": 180,
            "1y": 365, "2y": 730, "3y": 1095,
            "5y": 1825, "10y": 3650, "max": 36500,
        }
        days = period_days.get(period, 1825)
        from_date = (datetime.now() - timedelta(days=days)).date()
        to_date = datetime.now().date()

        # Pull data in chunks to improve reliability for long periods (e.g., max).
        all_rows = []
        cursor = from_date
        while cursor <= to_date:
            chunk_end = min(cursor + timedelta(days=365), to_date)
            chunk = self._kite.historical_data(
                instrument_token=instrument_token,
                from_date=cursor,
                to_date=chunk_end,
                interval="day",
                continuous=False,
                oi=False,
            )
            if chunk:
                all_rows.extend(chunk)
            cursor = chunk_end + timedelta(days=1)

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        df = df.drop_duplicates(subset=["date"], keep="last")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })

        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                return None
        return df[required]

    def _merge_unique(self, *lists: List[str]) -> List[str]:
        """Merge multiple symbol lists while preserving order and uniqueness."""
        seen = set()
        merged: List[str] = []
        for items in lists:
            for item in items:
                sym = str(item).strip()
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                merged.append(sym)
        return merged

    def _fallback_synthetic(self, ticker: str, period: str) -> pd.DataFrame:
        """Generate synthetic OHLCV for demo/fallback when API fails."""
        period_days = {
            "1mo": 21, "3mo": 63, "6mo": 126, "1y": 252,
            "2y": 504, "5y": 1260, "10y": 2520, "max": 7560,
        }
        n_days = period_days.get(period, 252)
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq="B")
        np.random.seed(abs(hash(ticker)) % 2**31)
        prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.015, n_days))
        df = pd.DataFrame(
            {
                "Open": prices * (1 - np.random.uniform(0, 0.01, n_days)),
                "High": prices * (1 + np.random.uniform(0, 0.02, n_days)),
                "Low": prices * (1 - np.random.uniform(0, 0.02, n_days)),
                "Close": prices,
                "Volume": np.random.randint(1_000_000, 50_000_000, n_days),
            },
            index=dates,
        )
        return df
