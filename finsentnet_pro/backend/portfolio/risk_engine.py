"""
Portfolio Risk Engine
Calculates VaR, CVaR, drawdowns, and risk-adjusted metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class RiskEngine:
    """Comprehensive portfolio risk analytics."""

    def __init__(self, trading_days: int = 252, risk_free_rate: float = 0.05):
        self.trading_days = trading_days
        self.rf = risk_free_rate

    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95,
                      method: str = "historical") -> float:
        """Value at Risk (VaR) using historical or parametric method."""
        if len(returns) < 10:
            return 0.0
        if method == "parametric":
            from scipy.stats import norm
            z = norm.ppf(1 - confidence)
            return float(-(returns.mean() + z * returns.std()))
        else:
            return float(-np.percentile(returns, (1 - confidence) * 100))

    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall)."""
        var = self.calculate_var(returns, confidence)
        tail = returns[returns <= -var]
        return float(-tail.mean()) if len(tail) > 0 else var

    def max_drawdown(self, equity_curve: np.ndarray) -> Dict:
        """Compute max drawdown and its duration."""
        peak = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - peak) / np.where(peak > 0, peak, 1)
        max_dd = float(dd.min())
        max_dd_idx = int(dd.argmin())
        peak_idx = int(np.argmax(equity_curve[:max_dd_idx + 1])) if max_dd_idx > 0 else 0
        return {
            "max_drawdown": round(max_dd * 100, 2),
            "peak_idx": peak_idx,
            "trough_idx": max_dd_idx,
            "duration": max_dd_idx - peak_idx,
        }

    def sharpe_ratio(self, returns: np.ndarray) -> float:
        """Annualized Sharpe Ratio."""
        if len(returns) < 5 or returns.std() == 0:
            return 0.0
        excess = returns.mean() - self.rf / self.trading_days
        return float(excess / returns.std() * np.sqrt(self.trading_days))

    def sortino_ratio(self, returns: np.ndarray) -> float:
        """Annualized Sortino Ratio (downside risk only)."""
        downside = returns[returns < 0]
        if len(downside) < 2:
            return 0.0
        downside_std = downside.std()
        if downside_std == 0:
            return 0.0
        excess = returns.mean() - self.rf / self.trading_days
        return float(excess / downside_std * np.sqrt(self.trading_days))

    def calmar_ratio(self, returns: np.ndarray, equity: np.ndarray) -> float:
        """Calmar Ratio = annualized return / max drawdown."""
        dd = self.max_drawdown(equity)
        mdd = abs(dd["max_drawdown"]) / 100
        if mdd < 0.001:
            return 0.0
        ann_return = returns.mean() * self.trading_days
        return float(ann_return / mdd)

    def full_risk_report(self, returns: np.ndarray, equity: np.ndarray) -> Dict:
        """Comprehensive risk report."""
        return {
            "var_95": round(self.calculate_var(returns, 0.95) * 100, 2),
            "cvar_95": round(self.calculate_cvar(returns, 0.95) * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio(returns), 2),
            "sortino_ratio": round(self.sortino_ratio(returns), 2),
            "calmar_ratio": round(self.calmar_ratio(returns, equity), 2),
            **self.max_drawdown(equity),
            "annualized_return": round(returns.mean() * self.trading_days * 100, 2),
            "annualized_volatility": round(returns.std() * np.sqrt(self.trading_days) * 100, 2),
            "win_rate": round(np.mean(returns > 0) * 100, 1) if len(returns) > 0 else 0,
        }
