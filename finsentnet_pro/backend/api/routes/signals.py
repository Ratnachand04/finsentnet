"""
FINSENT NET PRO — Signal Routes
/api/signals endpoints for standalone signal generation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

router = APIRouter(prefix="/api/signals", tags=["Signals"])
logger = logging.getLogger("finsent.routes.signals")

_signal_gen = None
_fetcher = None
_indicators = None


def init(signal_gen, fetcher, indicators):
    global _signal_gen, _fetcher, _indicators
    _signal_gen = signal_gen
    _fetcher = fetcher
    _indicators = indicators


class SignalRequest(BaseModel):
    ticker: str
    market: str = "SP500"
    capital: float = 100000
    risk_tolerance: float = 0.5


@router.post("/generate")
async def generate_signal(request: SignalRequest):
    """Generate a trade signal for a single ticker."""
    try:
        import numpy as np

        df = _fetcher.fetch_ohlcv(request.ticker, market=request.market, period="1y")
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {request.ticker}")

        df = _indicators.compute_all(df)

        # Mock model output (demo mode)
        np.random.seed(abs(hash(request.ticker)) % 2**31)
        p_up = np.random.uniform(0.3, 0.85)
        p_down = np.random.uniform(0.05, 1 - p_up)
        p_neutral = 1 - p_up - p_down

        model_output = {
            "direction_probs": np.array([p_up, p_neutral, p_down]),
            "return_pred": np.array([np.random.uniform(-0.08, 0.12)]),
        }

        signal = _signal_gen.generate_signal(
            ticker=request.ticker,
            market=request.market,
            model_output=model_output,
            price_data=df,
            total_capital=request.capital,
            risk_tolerance=request.risk_tolerance,
        )

        return {
            "ticker": signal.ticker,
            "direction": signal.direction.value,
            "confidence": signal.confidence,
            "entry_price": signal.entry_price,
            "target_price": signal.target_price,
            "stop_loss": signal.stop_loss,
            "risk_reward": signal.risk_reward_ratio,
            "kelly_fraction": signal.kelly_fraction,
            "quantity": signal.recommended_quantity,
            "capital_required": signal.capital_required,
            "predicted_return": signal.predicted_return,
            "time_horizon": signal.time_horizon,
            "reasoning": signal.reasoning,
            "regime": signal.regime,
            "sentiment_score": signal.sentiment_score,
            "technical_score": signal.technical_score,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signal generation error for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
