"""
Event-driven backtesting engine.
==================================

Simulates order execution with realistic frictions:
  - Commission costs (configurable bps)
  - Slippage modeling
  - Position sizing via Kelly / confidence scaling
  - Risk management limits
  - Daily P&L tracking
  
Design: Walk through time sequentially. At each timestep:
  1. Model generates prediction (direction + confidence)
  2. Position sizer computes target allocation
  3. Risk manager adjusts/blocks the allocation
  4. Execution simulator fills the order with costs
  5. P&L is computed and portfolio is updated

CRITICAL: The backtest engine NEVER touches future data.
Model predictions at time t use only data from times ≤ t.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field

from finsent.backtest.metrics import compute_all_metrics, print_metrics
from finsent.backtest.position_sizing import confidence_scaled_kelly
from finsent.backtest.risk import RiskManager


@dataclass
class Trade:
    """Record of a single trade."""
    date: pd.Timestamp
    direction: str        # "long", "short", "flat"
    position_pct: float   # fraction of portfolio
    confidence: float     # model confidence
    entry_price: float
    exit_price: float = 0.0
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    equity_curve: pd.Series
    returns: pd.Series
    drawdown: pd.Series
    trades: List[Trade]
    daily_positions: pd.Series
    metrics: Dict[str, float]
    predictions: np.ndarray
    confidences: np.ndarray


class BacktestEngine:
    """Event-driven backtesting engine for FinSentNet.
    
    Simulates live trading as closely as possible.
    
    Flow:
        for each day t in test set:
            prediction = model.predict(data[:t])
            target_position = position_sizer(prediction)
            adj_position = risk_manager.adjust(target_position)
            portfolio.execute(adj_position, costs)
            portfolio.update_pnl()
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_bps: float = 5.0,
        slippage_bps: float = 2.0,
        max_position_pct: float = 0.10,
        kelly_fraction: float = 0.25,
        risk_free_rate: float = 0.05,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_bps / 10000
        self.slippage_rate = slippage_bps / 10000
        self.max_position_pct = max_position_pct
        self.kelly_fraction_mult = kelly_fraction
        self.risk_free_rate = risk_free_rate
        
        self.risk_manager = RiskManager(
            max_position_pct=max_position_pct,
        )
    
    def run(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        test_dates: List[pd.Timestamp],
        test_close_prices: np.ndarray,
        device: torch.device,
    ) -> BacktestResult:
        """Run full backtest on test set.
        
        Args:
            model: Trained FinSentNet model.
            test_loader: DataLoader for test set.
            test_dates: Dates corresponding to test samples.
            test_close_prices: Raw close prices for P&L computation.
            device: Compute device.
        
        Returns:
            BacktestResult with full performance analysis.
        """
        model.eval()
        self.risk_manager.reset()
        
        # ─── Step 1: Generate All Predictions ─────────────────────
        all_predictions = []
        all_confidences = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                price = batch["price"].to(device)
                text_ids = batch["text_ids"].to(device)
                text_mask = batch["text_mask"].to(device)
                
                outputs = model(price, text_ids, text_mask)
                
                preds = outputs["direction_logits"].argmax(dim=-1).cpu().numpy()
                confs = outputs["confidence"].squeeze(-1).cpu().numpy()
                probs = outputs["direction_probs"].cpu().numpy()
                
                all_predictions.append(preds)
                all_confidences.append(confs)
                all_probs.append(probs)
        
        predictions = np.concatenate(all_predictions)
        confidences = np.concatenate(all_confidences)
        probs = np.concatenate(all_probs)
        
        n_samples = min(len(predictions), len(test_dates), len(test_close_prices))
        predictions = predictions[:n_samples]
        confidences = confidences[:n_samples]
        test_dates = test_dates[:n_samples]
        test_close_prices = test_close_prices[:n_samples]
        
        # ─── Step 2: Simulate Trading ─────────────────────────────
        equity = np.zeros(n_samples)
        equity[0] = self.initial_capital
        daily_returns = np.zeros(n_samples)
        positions = np.zeros(n_samples)
        trades = []
        
        # Running estimates for Kelly (updated online)
        win_returns = []
        loss_returns = []
        
        for t in range(1, n_samples):
            pred = predictions[t - 1]  # prediction made at t-1 for t
            conf = confidences[t - 1]
            
            # Actual return at time t
            actual_return = (test_close_prices[t] - test_close_prices[t - 1]) / test_close_prices[t - 1]
            
            # ─── Position Sizing ──────────────────────────────────
            if len(win_returns) >= 10 and len(loss_returns) >= 10:
                avg_win = np.mean(win_returns[-252:])
                avg_loss = np.mean(np.abs(loss_returns[-252:]))
                win_prob = len(win_returns) / (len(win_returns) + len(loss_returns))
            else:
                avg_win = 0.01
                avg_loss = 0.01
                win_prob = 0.5
            
            raw_position = confidence_scaled_kelly(
                confidence=conf,
                win_prob=win_prob,
                avg_win=avg_win,
                avg_loss=avg_loss,
                kelly_multiplier=self.kelly_fraction_mult,
                max_position=self.max_position_pct,
            )
            
            # Direction
            if pred == 2:  # Up
                direction = "long"
                position_sign = 1.0
            elif pred == 0:  # Down
                direction = "short"
                position_sign = -1.0
            else:  # Neutral
                direction = "flat"
                position_sign = 0.0
                raw_position = 0.0
            
            signed_position = position_sign * raw_position
            
            # ─── Risk Management ──────────────────────────────────
            recent_rets = daily_returns[max(0, t - 21):t]
            adj_position = self.risk_manager.adjust_position(
                signed_position, equity[t - 1], recent_rets
            )
            
            positions[t] = adj_position
            
            # ─── Execution with Costs ─────────────────────────────
            position_change = abs(adj_position - positions[t - 1])
            commission = position_change * equity[t - 1] * self.commission_rate
            slippage = position_change * equity[t - 1] * self.slippage_rate
            total_costs = commission + slippage
            
            # P&L
            gross_pnl = adj_position * actual_return * equity[t - 1]
            net_pnl = gross_pnl - total_costs
            
            equity[t] = equity[t - 1] + net_pnl
            daily_returns[t] = net_pnl / equity[t - 1] if equity[t - 1] > 0 else 0
            
            # Update running stats
            strategy_return = adj_position * actual_return
            if strategy_return > 0:
                win_returns.append(strategy_return)
            elif strategy_return < 0:
                loss_returns.append(strategy_return)
            
            # Record trade
            if position_change > 1e-6:
                trades.append(Trade(
                    date=test_dates[t],
                    direction=direction,
                    position_pct=abs(adj_position),
                    confidence=conf,
                    entry_price=test_close_prices[t - 1],
                    exit_price=test_close_prices[t],
                    pnl=net_pnl,
                    commission=commission,
                    slippage=slippage,
                ))
        
        # ─── Step 3: Compute Metrics ──────────────────────────────
        equity_series = pd.Series(equity, index=test_dates)
        returns_series = pd.Series(daily_returns, index=test_dates)
        
        # Drawdown
        running_max = equity_series.cummax()
        drawdown_series = (equity_series - running_max) / running_max
        
        # Benchmark: buy-and-hold
        benchmark_returns = np.diff(test_close_prices) / test_close_prices[:-1]
        benchmark_returns = np.insert(benchmark_returns, 0, 0.0)
        
        # Prediction scores for IC
        pred_scores = predictions.astype(float) - 1.0
        actual_rets = np.diff(test_close_prices) / test_close_prices[:-1]
        actual_rets = np.insert(actual_rets, 0, 0.0)
        
        metrics = compute_all_metrics(
            returns=daily_returns[1:],
            benchmark_returns=benchmark_returns[1:],
            predictions=pred_scores[:-1],
            actual_returns=actual_rets[1:],
            risk_free_rate=self.risk_free_rate,
        )
        
        # Additional trade metrics
        metrics["total_trades"] = len(trades)
        metrics["total_commission"] = sum(t.commission for t in trades)
        metrics["total_slippage"] = sum(t.slippage for t in trades)
        metrics["avg_position_size"] = np.mean(np.abs(positions[positions != 0])) if (positions != 0).any() else 0
        
        result = BacktestResult(
            equity_curve=equity_series,
            returns=returns_series,
            drawdown=drawdown_series,
            trades=trades,
            daily_positions=pd.Series(positions, index=test_dates),
            metrics=metrics,
            predictions=predictions,
            confidences=confidences,
        )
        
        return result
    
    def print_report(self, result: BacktestResult) -> None:
        """Print comprehensive backtest report."""
        print_metrics(result.metrics, "FinSentNet Backtest Results")
        
        print(f"\n  Trading Activity:")
        print(f"    Total Trades:        {result.metrics['total_trades']:>10d}")
        print(f"    Total Commission:    ${result.metrics['total_commission']:>10,.2f}")
        print(f"    Total Slippage:      ${result.metrics['total_slippage']:>10,.2f}")
        print(f"    Avg Position Size:   {result.metrics['avg_position_size']:>10.2%}")
        
        print(f"\n  Capital:")
        print(f"    Initial:             ${self.initial_capital:>14,.2f}")
        print(f"    Final:               ${result.equity_curve.iloc[-1]:>14,.2f}")
        print(f"    Net P&L:             ${result.equity_curve.iloc[-1] - self.initial_capital:>14,.2f}")
