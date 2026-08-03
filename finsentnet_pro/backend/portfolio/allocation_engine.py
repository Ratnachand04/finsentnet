"""
Allocation Engine — Converts signals + optimizer weights into executable allocations.
"""

from typing import Dict, List
import numpy as np


class AllocationEngine:
    """Generates final allocation table from signals and optimizer output."""

    def __init__(self, broker_fee: float = 0.001, min_lot: int = 1):
        self.broker_fee = broker_fee
        self.min_lot = min_lot

    def allocate(self, signals: list, optimizer_result: dict,
                 total_capital: float) -> Dict:
        """
        Combine trade signals with portfolio optimization weights
        to produce a final allocation table.
        """
        weights = optimizer_result.get("weights", [])
        tickers = [s.ticker for s in signals]
        n = len(signals)
        if len(weights) < n:
            weights = [1 / n] * n

        allocations = []
        total_deployed = 0
        for i, signal in enumerate(signals):
            w = weights[i]
            capital = total_capital * w
            price = signal.entry_price
            quantity = max(self.min_lot, int(capital / price)) if price > 0 else 0
            cost = quantity * price * (1 + self.broker_fee)
            total_deployed += cost

            allocations.append({
                "ticker": signal.ticker,
                "market": signal.market,
                "direction": signal.direction.value,
                "confidence": signal.confidence,
                "weight": round(w * 100, 2),
                "quantity": quantity,
                "entry_price": signal.entry_price,
                "target_price": signal.target_price,
                "stop_loss": signal.stop_loss,
                "capital_deployed": round(cost, 2),
                "risk_reward": signal.risk_reward_ratio,
                "predicted_return": signal.predicted_return,
            })

        cash_remaining = total_capital - total_deployed
        return {
            "allocations": allocations,
            "total_deployed": round(total_deployed, 2),
            "cash_remaining": round(max(0, cash_remaining), 2),
            "num_positions": n,
            "portfolio_utilization": round(total_deployed / total_capital * 100, 1) if total_capital > 0 else 0,
        }
