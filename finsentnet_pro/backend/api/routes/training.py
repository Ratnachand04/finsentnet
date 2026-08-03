"""
FINSENT NET PRO — Training Routes
/api/train/* endpoints for per-company model training and status monitoring.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/train", tags=["Training"])
logger = logging.getLogger("finsent.routes.training")

# ── Singletons (injected from main.py) ─────────────
_trainer = None
_predictor = None


def init(trainer, predictor):
    """Initialize route with shared services."""
    global _trainer, _predictor
    _trainer = trainer
    _predictor = predictor


# ═══════════════════════════════════════════════════════
#  Request / Response Models
# ═══════════════════════════════════════════════════════

class TrainRequest(BaseModel):
    ticker: str
    market: str = "SP500"
    epochs: int = Field(default=50, ge=5, le=500)
    batch_size: int = Field(default=32, ge=8, le=256)
    learning_rate: float = Field(default=1e-3, ge=1e-6, le=1e-1)
    window_size: int = Field(default=30, ge=10, le=120)
    horizon: int = Field(default=5, ge=1, le=30)
    period: str = Field(default="5y", description="Data period: 1y, 2y, 5y (FMP free tier max: 5y)")


class BatchTrainRequest(BaseModel):
    tickers: list
    market: str = "SP500"
    epochs: int = 50
    period: str = "5y"


# ═══════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════

@router.post("/start")
async def start_training(request: TrainRequest):
    """
    Start training the FINSENT model on a specific company's data.

    The model is trained on 2 years of historical OHLCV + technical indicators.
    Training runs in a background task so the API remains responsive.
    Poll /api/train/status/{ticker} for progress.
    """
    if not _trainer:
        raise HTTPException(status_code=503, detail="Trainer not initialized")

    ticker = request.ticker.upper().strip()
    market = request.market

    # Check if already training
    progress = _trainer.get_progress(ticker)
    if progress and progress["status"] == "training":
        return {
            "status": "already_training",
            "ticker": ticker,
            "message": f"Training already in progress for {ticker}",
            "progress": progress,
        }

    # Launch training in background (non-blocking)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        lambda: _trainer.train(
            ticker=ticker,
            market=market,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            window_size=request.window_size,
            horizon=request.horizon,
            period=request.period,
        ),
    )

    return {
        "status": "started",
        "ticker": ticker,
        "market": market,
        "message": f"Training started for {ticker} ({request.epochs} epochs)",
        "config": {
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            "window_size": request.window_size,
            "horizon": request.horizon,
            "period": request.period,
        },
    }


@router.get("/status/{ticker}")
async def get_training_status(ticker: str):
    """
    Get training progress for a ticker.
    Returns epoch, loss, accuracy, and progress percentage.
    """
    if not _trainer:
        raise HTTPException(status_code=503, detail="Trainer not initialized")

    ticker = ticker.upper().strip()
    progress = _trainer.get_progress(ticker)

    if progress:
        return {"status": "ok", "progress": progress}

    # Check if model is already trained (completed before)
    meta = _trainer.get_checkpoint_meta(ticker)
    if meta:
        return {
            "status": "ok",
            "progress": {
                "ticker": ticker,
                "status": "completed",
                "message": f"Model trained on {meta.get('trained_at', 'unknown')}",
                "current_epoch": meta.get("epochs_trained", 0),
                "total_epochs": meta.get("epochs_trained", 0),
                "progress_pct": 100,
                "val_accuracy": meta.get("best_val_accuracy", 0),
                "val_loss": meta.get("best_val_loss", 0),
                "training_time_seconds": meta.get("training_time_seconds", 0),
            },
        }

    return {
        "status": "ok",
        "progress": {
            "ticker": ticker,
            "status": "not_started",
            "message": f"No training found for {ticker}",
        },
    }


@router.get("/check/{ticker}")
async def check_trained(ticker: str):
    """Check if a trained model exists for this ticker."""
    if not _trainer:
        raise HTTPException(status_code=503, detail="Trainer not initialized")

    ticker = ticker.upper().strip()
    is_trained = _trainer.is_model_trained(ticker)
    meta = _trainer.get_checkpoint_meta(ticker) if is_trained else None

    return {
        "ticker": ticker,
        "is_trained": is_trained,
        "metadata": meta,
    }


@router.post("/batch")
async def batch_train(request: BatchTrainRequest):
    """Start training for multiple tickers."""
    if not _trainer:
        raise HTTPException(status_code=503, detail="Trainer not initialized")

    results = []
    loop = asyncio.get_event_loop()

    for ticker in request.tickers:
        t = ticker.upper().strip()
        # Skip if already trained unless forced
        if _trainer.is_model_trained(t):
            results.append({
                "ticker": t,
                "status": "already_trained",
                "message": f"Model already exists for {t}",
            })
            continue

        # Launch in background
        loop.run_in_executor(
            None,
            lambda t=t: _trainer.train(
                ticker=t,
                market=request.market,
                epochs=request.epochs,
                period=request.period,
            ),
        )
        results.append({
            "ticker": t,
            "status": "started",
            "message": f"Training queued for {t}",
        })

    return {"status": "ok", "results": results}


@router.get("/models")
async def list_trained_models():
    """List all trained model checkpoints."""
    import os
    import json

    if not _trainer:
        raise HTTPException(status_code=503, detail="Trainer not initialized")

    models = []
    ckpt_dir = _trainer.checkpoint_dir

    if os.path.isdir(ckpt_dir):
        for fname in os.listdir(ckpt_dir):
            if fname.endswith("_meta.json"):
                try:
                    with open(os.path.join(ckpt_dir, fname)) as f:
                        meta = json.load(f)
                    models.append(meta)
                except Exception:
                    pass

    return {"models": models, "count": len(models)}
