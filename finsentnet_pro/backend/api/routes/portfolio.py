"""
FINSENT NET PRO — Portfolio Routes
/api/portfolio endpoints for allocation and risk analytics.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import logging

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])
logger = logging.getLogger("finsent.routes.portfolio")

_optimizer = None
_risk_engine = None
_allocator = None
_kelly = None


def init(optimizer, risk_engine, allocator, kelly):
    global _optimizer, _risk_engine, _allocator, _kelly
    _optimizer = optimizer
    _risk_engine = risk_engine
    _allocator = allocator
    _kelly = kelly


class PortfolioOptRequest(BaseModel):
    expected_returns: List[float]
    method: str = "max_sharpe"


class KellyRequest(BaseModel):
    capital: float
    price: float
    win_prob: float
    reward_risk_ratio: float
    risk_tolerance: float = 0.5


@router.post("/optimize")
async def optimize_portfolio(request: PortfolioOptRequest):
    """Run portfolio optimization on expected returns."""
    try:
        n = len(request.expected_returns)
        returns = np.array(request.expected_returns)
        cov = np.eye(n) * 0.02 ** 2
        result = _optimizer.optimize_weights(returns, cov, method=request.method)
        return result
    except Exception as e:
        logger.error(f"Portfolio optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kelly")
async def kelly_sizing(request: KellyRequest):
    """Compute Kelly Criterion position sizing."""
    try:
        return _kelly.compute_position(
            capital=request.capital,
            price=request.price,
            win_prob=request.win_prob,
            reward_risk_ratio=request.reward_risk_ratio,
            risk_tolerance=request.risk_tolerance,
        )
    except Exception as e:
        logger.error(f"Kelly sizing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk")
async def risk_report(equity_curve: List[float]):
    """Compute full risk report from an equity curve."""
    try:
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        return _risk_engine.full_risk_report(returns, equity)
    except Exception as e:
        logger.error(f"Risk report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
