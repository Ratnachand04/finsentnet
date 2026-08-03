"""
Position sizing methods.
=========================

Implements multiple position sizing strategies:
  - Kelly Criterion (optimal growth rate maximization)
  - Equal Weight
  - Volatility Parity (inverse vol weighting)
  - Confidence-Scaled Kelly (our primary method)

Kelly Criterion Mathematics:
    f* = p/a - q/b
    
    where:
        f* = optimal fraction of capital to wager
        p  = probability of winning
        q  = 1 - p = probability of losing
        a  = net odds received on a loss (loss per unit wagered)
        b  = net odds received on a win (win per unit wagered)
    
    In practice, we use fractional Kelly (f*/4) because:
    1. Kelly assumes exact probabilities (we have estimates)
    2. Full Kelly is very aggressive with high drawdowns
    3. 1/4 Kelly provides ~75% of the growth with ~50% of the volatility
"""

import numpy as np
from typing import Optional


def kelly_fraction(
    win_prob: float,
    avg_win: float,
    avg_loss: float,
    kelly_multiplier: float = 0.25,  # quarter-Kelly
) -> float:
    """Kelly Criterion position sizing.
    
    Args:
        win_prob: Estimated probability of positive return
        avg_win: Average winning return (positive)
        avg_loss: Average losing return (negative, stored as positive magnitude)
        kelly_multiplier: Fraction of full Kelly to use
    
    Returns:
        Optimal position size as fraction of capital [0, 1]
    """
    if avg_loss < 1e-10 or avg_win < 1e-10:
        return 0.0
    
    q = 1 - win_prob
    
    # Kelly formula
    f_star = (win_prob / avg_loss) - (q / avg_win)
    
    # Apply fractional Kelly
    position = f_star * kelly_multiplier
    
    # Clamp to [0, 1] (no negative positions via Kelly, no leverage)
    return float(np.clip(position, 0.0, 1.0))


def confidence_scaled_kelly(
    confidence: float,
    win_prob: float,
    avg_win: float,
    avg_loss: float,
    kelly_multiplier: float = 0.25,
    max_position: float = 0.10,
) -> float:
    """Kelly fraction scaled by model confidence.
    
    position_size = min(kelly_fraction * confidence, max_position)
    
    This is the primary position sizing method for FinSentNet.
    The calibrated confidence output directly modulates allocation.
    
    Args:
        confidence: Model's calibrated confidence [0, 1]
        max_position: Maximum position as fraction of portfolio
    """
    base_kelly = kelly_fraction(win_prob, avg_win, avg_loss, kelly_multiplier)
    scaled = base_kelly * confidence
    return float(np.clip(scaled, 0.0, max_position))


def equal_weight(
    n_positions: int,
    max_position: float = 0.10,
) -> float:
    """Equal weight: 1/N allocation."""
    if n_positions <= 0:
        return 0.0
    return float(min(1.0 / n_positions, max_position))


def volatility_parity(
    volatilities: np.ndarray,
    target_vol: float = 0.15,  # 15% annual
    max_position: float = 0.10,
) -> np.ndarray:
    """Inverse-volatility position sizing.
    
    Each position is sized inversely proportional to its volatility,
    so each contributes equally to portfolio risk.
    
    w_i = (1/σ_i) / Σ(1/σ_j) × target_leverage
    
    Args:
        volatilities: Array of asset volatilities (annualized)
        target_vol: Target portfolio volatility
        max_position: Maximum single-position weight
    """
    inv_vol = 1.0 / np.maximum(volatilities, 1e-8)
    weights = inv_vol / inv_vol.sum()
    
    # Scale to target volatility
    portfolio_vol = np.sqrt(np.sum((weights * volatilities) ** 2))
    if portfolio_vol > 1e-8:
        leverage = target_vol / portfolio_vol
        weights *= leverage
    
    # Cap individual positions
    weights = np.minimum(weights, max_position)
    
    return weights
