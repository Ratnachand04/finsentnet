"""
FINSENT NET PRO — FastAPI Backend
REST API serving all analysis endpoints.
AI-Powered Quantitative Trading Intelligence.
"""

import os
import sys
import logging
import torch
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# ── Fix imports ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _bootstrap_runtime_env() -> None:
    """Load .env files and normalize provider env aliases used across services."""
    if load_dotenv is not None:
        base_path = Path(__file__).resolve()
        candidates = [
            base_path.parents[3] / ".env",  # workspace root
            base_path.parents[2] / ".env",  # finsentnet_pro/
            base_path.parents[1] / ".env",  # backend/
        ]
        for env_path in candidates:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=False)

    # FMP supports both singular and comma-separated key env vars.
    if not os.getenv("FMP_API_KEYS"):
        single_fmp = str(os.getenv("FMP_API_KEY") or "").strip()
        if single_fmp:
            os.environ["FMP_API_KEYS"] = single_fmp

    # Normalize Kite variable names if the environment uses CONNECT-prefixed names.
    if not os.getenv("KITE_API_KEY"):
        kite_key = str(os.getenv("KITE_CONNECT_API_KEY") or "").strip()
        if kite_key:
            os.environ["KITE_API_KEY"] = kite_key

    if not os.getenv("KITE_ACCESS_TOKEN"):
        kite_token = str(os.getenv("KITE_CONNECT_ACCESS_TOKEN") or "").strip()
        if kite_token:
            os.environ["KITE_ACCESS_TOKEN"] = kite_token

    if not os.getenv("KITE_API_SECRET"):
        kite_secret = str(
            os.getenv("KITE_CONNECT_SECRET_KEY")
            or os.getenv("KITE_CONNECT_API_SECRET")
            or ""
        ).strip()
        if kite_secret:
            os.environ["KITE_API_SECRET"] = kite_secret


def _select_torch_device() -> torch.device:
    """Select compute device with optional override via FINSENT_DEVICE."""
    requested = str(os.getenv("FINSENT_DEVICE", "auto") or "auto").strip().lower()

    if requested in {"cpu"}:
        return torch.device("cpu")

    if requested in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        logging.getLogger("finsent").warning(
            "FINSENT_DEVICE requested CUDA but no CUDA device is available. Falling back to CPU."
        )
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


_bootstrap_runtime_env()

from data_pipeline.market_data_fetcher import MarketDataFetcher
from data_pipeline.technical_indicators import TechnicalIndicators
from data_pipeline.news_sentiment_engine import NewsSentimentEngine
from data_pipeline.regime_detector import RegimeDetector
from data_pipeline.data_aligner import DataAligner
from models.finsentnet_core import FinSentNetCore
from models.model_registry import ModelRegistry
from models.signal_generator import SignalGenerator
from portfolio.kelly_sizer import KellySizer
from portfolio.portfolio_optimizer import PortfolioOptimizer
from portfolio.risk_engine import RiskEngine
from portfolio.allocation_engine import AllocationEngine
from backtesting.backtest_engine import BacktestEngine
from training.trainer import ModelTrainer
from training.predictor import LivePredictor
from data_pipeline.live_data_service import LiveDataService
from observability.live_metrics import configure_live_metrics, render_prometheus_metrics

# Route modules
from api.routes import market_data as market_routes
from api.routes import analysis as analysis_routes
from api.routes import signals as signal_routes
from api.routes import portfolio as portfolio_routes
from api.routes import training as training_routes
from api.routes import live as live_routes
from api.routes import system as system_routes
from api.websocket_handler import websocket_endpoint, tunnel_coordinator

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finsent")

COMPUTE_DEVICE = _select_torch_device()
if COMPUTE_DEVICE.type == "cuda":
    try:
        matmul_precision = str(os.getenv("FINSENT_MATMUL_PRECISION", "high") or "high").strip()
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = _env_flag("FINSENT_CUDNN_BENCHMARK", True)
    torch.backends.cuda.matmul.allow_tf32 = _env_flag("FINSENT_ALLOW_TF32", True)
    torch.backends.cudnn.allow_tf32 = _env_flag("FINSENT_ALLOW_TF32", True)

    # High throughput for repeated inference workloads on laptop GPUs.
    if _env_flag("FINSENT_FORCE_MAX_GPU_CLOCKS_HINT", False):
        logger.info("FINSENT_FORCE_MAX_GPU_CLOCKS_HINT is enabled; ensure OS power mode is set to Best Performance.")

logger.info(f"Backend compute device: {COMPUTE_DEVICE}")
if COMPUTE_DEVICE.type == "cuda":
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

# ── App ──────────────────────────────────────────────────
app = FastAPI(
    title="FINSENT NET PRO API",
    description="AI-Powered Quantitative Trading Intelligence",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ── Singletons ───────────────────────────────────────────
fetcher = MarketDataFetcher()
indicators = TechnicalIndicators()
sentiment_engine = NewsSentimentEngine(use_finbert=False)
regime_detector = RegimeDetector()
aligner = DataAligner()
model = FinSentNetCore().to(COMPUTE_DEVICE)
if COMPUTE_DEVICE.type == "cuda" and _env_flag("FINSENT_TORCH_COMPILE", False) and hasattr(torch, "compile"):
    try:
        compile_mode = str(os.getenv("FINSENT_TORCH_COMPILE_MODE", "reduce-overhead") or "reduce-overhead").strip()
        model = torch.compile(model, mode=compile_mode, dynamic=True)
        logger.info(f"torch.compile enabled for model (mode={compile_mode}).")
    except Exception as e:
        logger.warning(f"torch.compile unavailable or failed; continuing without compile. Reason: {e}")
model.eval()
model_registry = ModelRegistry()
signal_gen = SignalGenerator()
kelly = KellySizer()
optimizer = PortfolioOptimizer()
risk_engine = RiskEngine()
allocator = AllocationEngine()
backtester = BacktestEngine()
trainer = ModelTrainer(model, fetcher, indicators, device=str(COMPUTE_DEVICE))
predictor = LivePredictor(model, fetcher, indicators, signal_gen, trainer)
live_data_service = LiveDataService()
configure_live_metrics(live_data_service)

# ── Initialize route modules with shared services ────────
market_routes.init(fetcher, indicators)
analysis_routes.init(
    fetcher, indicators, sentiment_engine, regime_detector, aligner,
    model, signal_gen, optimizer, allocator, backtester, risk_engine,
    live_service=live_data_service,
)
signal_routes.init(signal_gen, fetcher, indicators)
portfolio_routes.init(optimizer, risk_engine, allocator, kelly)
training_routes.init(trainer, predictor)
live_routes.init(live_data_service, predictor, trainer, fetcher=fetcher)
system_routes.init(model_registry)

# ── Register routers ─────────────────────────────────────
app.include_router(market_routes.router)
app.include_router(analysis_routes.router)
app.include_router(signal_routes.router)
app.include_router(portfolio_routes.router)
app.include_router(training_routes.router)
app.include_router(live_routes.router)
app.include_router(system_routes.router)


@app.on_event("startup")
async def startup_tunnel_runtime():
    """Optionally start websocket tunnels at startup for distributed deployments."""
    tunnel_coordinator.configure(
        fetcher=fetcher,
        live_service=live_data_service,
        predictor=predictor,
        trainer=trainer,
    )

    start_on_boot_env = str(os.getenv("WS_START_TUNNELS_ON_STARTUP") or "").strip().lower()
    start_on_boot = start_on_boot_env in {"1", "true", "yes", "on"}
    if tunnel_coordinator.redis_enabled or start_on_boot:
        await tunnel_coordinator.ensure_started()


@app.on_event("shutdown")
async def shutdown_tunnel_runtime():
    await tunnel_coordinator.stop()


# ── Core endpoints ───────────────────────────────────────
@app.get("/")
async def root():
    """Serve frontend."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "FINSENT NET PRO API v2.0 — Use /docs for API reference.",
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "FINSENT NET PRO is operational",
        "version": "2.0.0",
        "model": "FINSENT Core v1.0",
        "compute": {
            "device": str(COMPUTE_DEVICE),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "active_model_profile": model_registry.get_active().profile_id,
        "timestamp": datetime.utcnow().isoformat(),
        "modules": {
            "data_fetcher": True,
            "technical_indicators": True,
            "sentiment_engine": True,
            "regime_detector": True,
            "finsentnet_model": True,
            "signal_generator": True,
            "portfolio_optimizer": True,
            "risk_engine": True,
            "backtester": True,
            "model_trainer": True,
            "live_predictor": True,
            "live_data_service": True,
        },
    }


@app.get("/metrics")
async def prometheus_metrics():
    payload, content_type = render_prometheus_metrics()
    return Response(content=payload, media_type=content_type)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time price streaming."""
    await websocket_endpoint(
        websocket,
        fetcher=fetcher,
        live_service=live_data_service,
        predictor=predictor,
        trainer=trainer,
    )


# ── Run ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
