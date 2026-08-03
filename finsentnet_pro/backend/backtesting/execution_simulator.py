"""
FINSENT NET PRO — Execution Simulator
Models realistic trade execution: slippage, commission, and market impact.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of a simulated order execution."""
    ticker: str
    side: str                   # "BUY" or "SELL"
    requested_price: float
    executed_price: float
    slippage_cost: float
    commission_cost: float
    market_impact_cost: float
    total_cost: float
    quantity: int
    net_cost: float             # total capital spent including all costs


class ExecutionSimulator:
    """
    Simulates realistic trade execution with:
    - Slippage: price movement between order and fill
    - Commission: brokerage fees
    - Market impact: price impact of large orders on thin books
    """

    def __init__(
        self,
        commission_rate: float = 0.001,      # 0.1% per trade
        slippage_bps: float = 5.0,           # 5 basis points (0.05%)
        impact_coefficient: float = 0.1,     # market impact scaling
        min_commission: float = 1.0,         # minimum $1 commission
    ):
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.impact_coefficient = impact_coefficient
        self.min_commission = min_commission

    def simulate_execution(
        self,
        ticker: str,
        side: str,
        price: float,
        quantity: int,
        avg_daily_volume: float = 1_000_000,
        volatility: float = 0.02,
    ) -> ExecutionResult:
        """
        Simulate order execution with realistic friction costs.

        Args:
            ticker: Symbol
            side: "BUY" or "SELL"
            price: Intended execution price
            quantity: Number of shares
            avg_daily_volume: Average daily volume for impact estimation
            volatility: Daily return volatility

        Returns:
            ExecutionResult with all cost components
        """
        # 1. Slippage — random adverse price movement
        slippage_pct = self.slippage_bps / 10000
        if side.upper() == "BUY":
            # Buying: price slips upward
            slippage = price * slippage_pct * abs(np.random.normal(1.0, 0.3))
            executed_price = price + slippage
        else:
            # Selling: price slips downward
            slippage = price * slippage_pct * abs(np.random.normal(1.0, 0.3))
            executed_price = price - slippage

        slippage_cost = abs(slippage) * quantity

        # 2. Commission
        commission = max(
            self.min_commission,
            executed_price * quantity * self.commission_rate,
        )

        # 3. Market impact — proportional to order size relative to volume
        participation_rate = quantity / max(avg_daily_volume, 1)
        market_impact = (
            self.impact_coefficient
            * volatility
            * np.sqrt(participation_rate)
            * executed_price
            * quantity
        )

        total_cost = slippage_cost + commission + market_impact
        net_cost = executed_price * quantity + total_cost

        return ExecutionResult(
            ticker=ticker,
            side=side.upper(),
            requested_price=round(price, 2),
            executed_price=round(executed_price, 4),
            slippage_cost=round(slippage_cost, 2),
            commission_cost=round(commission, 2),
            market_impact_cost=round(market_impact, 2),
            total_cost=round(total_cost, 2),
            quantity=quantity,
            net_cost=round(net_cost, 2),
        )

    def estimate_round_trip_cost(
        self,
        price: float,
        quantity: int,
        avg_daily_volume: float = 1_000_000,
        volatility: float = 0.02,
    ) -> Dict:
        """Estimate total round-trip (buy + sell) transaction costs."""
        buy = self.simulate_execution("", "BUY", price, quantity, avg_daily_volume, volatility)
        sell = self.simulate_execution("", "SELL", price, quantity, avg_daily_volume, volatility)
        total = buy.total_cost + sell.total_cost
        return {
            "buy_side_cost": buy.total_cost,
            "sell_side_cost": sell.total_cost,
            "total_round_trip_cost": round(total, 2),
            "cost_as_pct_of_trade": round(total / (price * quantity) * 100, 4),
        }
