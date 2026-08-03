"""
Kelly Criterion Position Sizer
Optimal bankroll fraction: f* = (bp - q) / b
"""

import numpy as np
from typing import Dict, Optional


class KellySizer:
    """Calculates optimal position sizes using Kelly Criterion."""

    def __init__(self, max_fraction: float = 0.25, min_fraction: float = 0.01):
        self.max_fraction = max_fraction
        self.min_fraction = min_fraction

    def full_kelly(self, win_prob: float, reward_risk_ratio: float) -> float:
        """Full Kelly fraction: f* = (b*p - q) / b."""
        p = np.clip(win_prob, 0.01, 0.99)
        q = 1 - p
        b = max(reward_risk_ratio, 0.001)
        f = (b * p - q) / b
        return float(np.clip(f, 0, self.max_fraction))

    def half_kelly(self, win_prob: float, reward_risk_ratio: float) -> float:
        """Half-Kelly — a more conservative approach."""
        return self.full_kelly(win_prob, reward_risk_ratio) * 0.5

    def fractional_kelly(self, win_prob: float, reward_risk_ratio: float,
                         fraction: float = 0.5) -> float:
        """Fractional Kelly for configurable risk aversion."""
        return self.full_kelly(win_prob, reward_risk_ratio) * np.clip(fraction, 0.1, 1.0)

    def compute_position(self, capital: float, price: float,
                         win_prob: float, reward_risk_ratio: float,
                         risk_tolerance: float = 0.5) -> Dict:
        """Compute full position sizing details."""
        kelly = self.fractional_kelly(win_prob, reward_risk_ratio, risk_tolerance)
        deploy = capital * kelly
        quantity = max(1, int(deploy / price)) if price > 0 else 0
        actual = quantity * price
        return {
            "kelly_fraction": round(kelly, 4),
            "capital_deployed": round(actual, 2),
            "quantity": quantity,
            "portfolio_pct": round(actual / capital * 100, 2) if capital > 0 else 0,
        }
