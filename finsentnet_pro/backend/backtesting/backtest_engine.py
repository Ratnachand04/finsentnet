"""
Backtest Engine — Walk-forward simulation with realistic execution modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class BacktestEngine:
    """Walk-forward backtester with slippage, commissions, and position tracking."""

    def __init__(self, initial_capital: float = 1_000_000,
                 commission: float = 0.001, slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(self, signals: list, price_data: Dict[str, pd.DataFrame],
            holding_period: int = 5) -> Dict:
        """
        Run backtest over historical signals.
        signals: list of TradeSignal objects
        price_data: {ticker: DataFrame with 'Close'}
        """
        capital = self.initial_capital
        equity_curve = [capital]
        trades = []
        daily_returns = []

        for signal in signals:
            ticker = signal.ticker
            if ticker not in price_data or price_data[ticker].empty:
                continue

            df = price_data[ticker]
            if len(df) < holding_period + 1:
                continue

            entry_price = signal.entry_price * (1 + self.slippage)
            commission_cost = entry_price * signal.recommended_quantity * self.commission
            cost = entry_price * signal.recommended_quantity + commission_cost

            if cost > capital:
                continue

            exit_idx = min(len(df) - 1, holding_period)
            exit_price = float(df["Close"].iloc[exit_idx]) * (1 - self.slippage)

            if signal.direction.value in ("STRONG BUY", "BUY"):
                pnl = (exit_price - entry_price) * signal.recommended_quantity
            elif signal.direction.value in ("STRONG SELL", "SELL"):
                pnl = (entry_price - exit_price) * signal.recommended_quantity
            else:
                pnl = 0

            pnl -= commission_cost * 2

            capital += pnl
            equity_curve.append(capital)
            daily_ret = pnl / max(capital - pnl, 1)
            daily_returns.append(daily_ret)

            trades.append({
                "ticker": ticker,
                "direction": signal.direction.value,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "quantity": signal.recommended_quantity,
                "pnl": round(pnl, 2),
                "return_pct": round(daily_ret * 100, 2),
            })

        equity_np = np.array(equity_curve)
        returns_np = np.array(daily_returns) if daily_returns else np.array([0])

        peak = np.maximum.accumulate(equity_np)
        dd = (equity_np - peak) / np.where(peak > 0, peak, 1)

        total_return = (capital - self.initial_capital) / self.initial_capital
        win_trades = [t for t in trades if t["pnl"] > 0]

        return {
            "initial_capital": self.initial_capital,
            "final_capital": round(capital, 2),
            "total_return_pct": round(total_return * 100, 2),
            "total_trades": len(trades),
            "winning_trades": len(win_trades),
            "losing_trades": len(trades) - len(win_trades),
            "win_rate": round(len(win_trades) / max(len(trades), 1) * 100, 1),
            "max_drawdown_pct": round(float(dd.min()) * 100, 2),
            "sharpe_ratio": round(
                returns_np.mean() / max(returns_np.std(), 1e-8) * np.sqrt(252), 2
            ) if len(returns_np) > 1 else 0,
            "avg_trade_return": round(returns_np.mean() * 100, 2),
            "equity_curve": equity_np.tolist(),
            "trades": trades[:50],
        }

    def generate_demo_backtest(self, total_capital: float = 1_000_000) -> Dict:
        """Generate synthetic backtest results for demo mode."""
        np.random.seed(42)
        n_days = 252
        daily_rets = np.random.normal(0.0008, 0.012, n_days)
        equity = total_capital * np.cumprod(1 + daily_rets)
        equity = np.insert(equity, 0, total_capital)

        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / np.where(peak > 0, peak, 1)

        final = equity[-1]
        total_return = (final - total_capital) / total_capital

        return {
            "initial_capital": total_capital,
            "final_capital": round(final, 2),
            "total_return_pct": round(total_return * 100, 2),
            "total_trades": 147,
            "winning_trades": 89,
            "losing_trades": 58,
            "win_rate": 60.5,
            "max_drawdown_pct": round(float(dd.min()) * 100, 2),
            "sharpe_ratio": round(
                daily_rets.mean() / max(daily_rets.std(), 1e-8) * np.sqrt(252), 2
            ),
            "avg_trade_return": round(daily_rets.mean() * 100, 3),
            "equity_curve": equity.tolist(),
            "trades": [],
        }
