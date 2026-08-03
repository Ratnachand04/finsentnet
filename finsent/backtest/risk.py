"""
Risk management engine.
========================

Real-time risk limits that would halt or reduce trading:
  - Maximum drawdown limit
  - Daily VaR limit
  - Position concentration limits
  - Correlation limits
  - Volatility regime detection

In production: these fire BEFORE execution, not after.
In backtesting: they modify positions retroactively.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class RiskManager:
    """Portfolio risk management layer.
    
    Sits between signal generation and execution.
    Enforces hard risk limits and adjusts position sizes.
    """
    
    def __init__(
        self,
        max_drawdown_pct: float = 0.15,
        daily_var_limit: float = 0.02,
        max_position_pct: float = 0.10,
        max_leverage: float = 1.0,
        correlation_limit: float = 0.7,
        vol_lookback: int = 21,
    ):
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_var_limit = daily_var_limit
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage
        self.correlation_limit = correlation_limit
        self.vol_lookback = vol_lookback
        
        # State
        self.peak_equity = 0.0
        self.is_halted = False
        self.halt_reason = ""
    
    def check_drawdown(
        self,
        current_equity: float,
    ) -> Tuple[bool, float]:
        """Check if drawdown exceeds limit.
        
        Returns: (is_breach, current_drawdown)
        """
        self.peak_equity = max(self.peak_equity, current_equity)
        
        if self.peak_equity > 0:
            drawdown = (current_equity - self.peak_equity) / self.peak_equity
        else:
            drawdown = 0.0
        
        is_breach = drawdown < -self.max_drawdown_pct
        
        if is_breach and not self.is_halted:
            self.is_halted = True
            self.halt_reason = f"Max drawdown breached: {drawdown:.2%}"
        
        return is_breach, drawdown
    
    def check_var_limit(
        self,
        recent_returns: np.ndarray,
        proposed_position: float,
    ) -> Tuple[bool, float]:
        """Check if proposed position would violate daily VaR limit.
        
        Uses historical VaR at 95% confidence.
        
        Returns: (is_breach, estimated_var)
        """
        if len(recent_returns) < 10:
            return False, 0.0
        
        vol = np.std(recent_returns) * np.sqrt(1)  # daily vol
        estimated_var = 1.645 * vol * abs(proposed_position)  # 95% VaR
        
        return estimated_var > self.daily_var_limit, estimated_var
    
    def adjust_position(
        self,
        raw_position: float,
        current_equity: float,
        recent_returns: np.ndarray,
    ) -> float:
        """Adjust position size based on risk limits.
        
        Sequential checks:
        1. Is the strategy halted (max drawdown)?
        2. Does position exceed max single-position limit?
        3. Does position violate VaR?
        4. Is total leverage within limits?
        
        Returns: adjusted position size
        """
        # Check 1: Halt check
        is_dd_breach, _ = self.check_drawdown(current_equity)
        if is_dd_breach:
            return 0.0  # flat everything
        
        adjusted = raw_position
        
        # Check 2: Max position
        adjusted = np.clip(adjusted, -self.max_position_pct, self.max_position_pct)
        
        # Check 3: VaR limit
        if len(recent_returns) >= 10:
            vol = np.std(recent_returns)
            if vol > 0:
                max_pos_by_var = self.daily_var_limit / (1.645 * vol)
                adjusted = np.clip(adjusted, -max_pos_by_var, max_pos_by_var)
        
        # Check 4: Leverage
        adjusted = np.clip(adjusted, -self.max_leverage, self.max_leverage)
        
        return float(adjusted)
    
    def compute_portfolio_var(
        self,
        positions: np.ndarray,
        return_matrix: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """Portfolio VaR using variance-covariance method.
        
        VaR_p = z_α × σ_p × √Δt
        σ_p² = w' Σ w
        
        where Σ is the covariance matrix of asset returns.
        """
        if return_matrix.shape[0] < 10:
            return 0.0
        
        cov_matrix = np.cov(return_matrix, rowvar=False)
        portfolio_var = np.sqrt(positions @ cov_matrix @ positions)
        
        z_score = 1.645 if confidence == 0.95 else 2.326
        return z_score * portfolio_var
    
    def check_correlation_risk(
        self,
        return_matrix: np.ndarray,
        position_mask: np.ndarray,
    ) -> Tuple[bool, float]:
        """Check if active positions are too highly correlated.
        
        High correlation means positions move together →
        diversification benefit is illusory.
        """
        active_returns = return_matrix[:, position_mask.astype(bool)]
        
        if active_returns.shape[1] < 2:
            return False, 0.0
        
        corr_matrix = np.corrcoef(active_returns, rowvar=False)
        
        # Max off-diagonal correlation
        np.fill_diagonal(corr_matrix, 0)
        max_corr = np.max(np.abs(corr_matrix))
        
        return max_corr > self.correlation_limit, max_corr
    
    def reset(self) -> None:
        """Reset risk manager state (for new backtest run)."""
        self.peak_equity = 0.0
        self.is_halted = False
        self.halt_reason = ""
    
    def get_status(self) -> Dict:
        """Current risk status."""
        return {
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "peak_equity": self.peak_equity,
        }
