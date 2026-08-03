"""
Performance Metrics — After-the-fact evaluation of strategies and portfolios.
"""

import numpy as np
from typing import Dict


class PerformanceMetrics:
    """Calculate strategy performance statistics."""

    @staticmethod
    def compute(equity_curve: np.ndarray, risk_free_rate: float = 0.05,
                trading_days: int = 252) -> Dict:
        """Compute comprehensive performance metrics from an equity curve."""
        if len(equity_curve) < 2:
            return {"error": "Insufficient data"}

        returns = np.diff(equity_curve) / equity_curve[:-1]
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        ann_return = (1 + total_return) ** (trading_days / max(len(returns), 1)) - 1
        ann_vol = returns.std() * np.sqrt(trading_days)

        # Sharpe
        daily_rf = risk_free_rate / trading_days
        sharpe = ((returns.mean() - daily_rf) / max(returns.std(), 1e-8)
                  * np.sqrt(trading_days))

        # Sortino
        downside = returns[returns < 0]
        sortino = ((returns.mean() - daily_rf) / max(downside.std(), 1e-8)
                   * np.sqrt(trading_days)) if len(downside) > 0 else 0

        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - peak) / np.where(peak > 0, peak, 1)
        max_dd = float(dd.min())

        # Calmar
        calmar = ann_return / max(abs(max_dd), 1e-8)

        # Win rate
        win_rate = float(np.mean(returns > 0) * 100) if len(returns) > 0 else 0

        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / max(losses, 1e-8)

        return {
            "total_return_pct": round(total_return * 100, 2),
            "annualized_return_pct": round(ann_return * 100, 2),
            "annualized_volatility_pct": round(ann_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "calmar_ratio": round(calmar, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "best_day_pct": round(float(returns.max()) * 100, 2),
            "worst_day_pct": round(float(returns.min()) * 100, 2),
            "avg_daily_return_pct": round(float(returns.mean()) * 100, 3),
            "total_trading_days": len(returns),
        }
