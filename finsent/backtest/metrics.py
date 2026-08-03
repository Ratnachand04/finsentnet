"""
Financial performance metrics.
================================

Comprehensive metrics for evaluating trading strategies.
All computations are vectorized for efficiency.

Metrics implemented:
  - Sharpe Ratio (annualized)
  - Sortino Ratio (downside deviation only)
  - Maximum Drawdown and Duration
  - Calmar Ratio (return / max drawdown)
  - Information Coefficient (IC) and IR
  - Win Rate and Profit Factor
  - Value at Risk (VaR) and CVaR
  - Turnover
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy import stats


def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.05,
    annualization: int = 252,
) -> float:
    """Annualized Sharpe Ratio.
    
    SR = (E[R] - R_f) / σ(R) × √252
    
    where R_f is the daily risk-free rate.
    """
    daily_rf = (1 + risk_free_rate) ** (1 / annualization) - 1
    excess = returns - daily_rf
    
    if np.std(excess) < 1e-10:
        return 0.0
    
    return np.mean(excess) / np.std(excess) * np.sqrt(annualization)


def compute_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.05,
    annualization: int = 252,
) -> float:
    """Sortino Ratio — penalizes only downside volatility.
    
    Sortino = (E[R] - R_f) / σ_down(R) × √252
    
    Better than Sharpe for asymmetric return distributions
    (which is typical for directional strategies).
    """
    daily_rf = (1 + risk_free_rate) ** (1 / annualization) - 1
    excess = returns - daily_rf
    
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("inf") if np.mean(excess) > 0 else 0.0
    
    downside_std = np.std(downside)
    if downside_std < 1e-10:
        return float("inf") if np.mean(excess) > 0 else 0.0
    
    return np.mean(excess) / downside_std * np.sqrt(annualization)


def compute_max_drawdown(
    returns: np.ndarray,
) -> Tuple[float, int, int, int]:
    """Maximum drawdown and its timing.
    
    Returns:
        (max_drawdown, peak_idx, trough_idx, recovery_idx)
        
    max_drawdown is negative (e.g., -0.15 = 15% drawdown).
    """
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative / running_max - 1
    
    max_dd = np.min(drawdown)
    trough_idx = np.argmin(drawdown)
    peak_idx = np.argmax(cumulative[:trough_idx + 1]) if trough_idx > 0 else 0
    
    # Recovery: first time cumulative exceeds the peak after trough
    recovery_idx = len(returns) - 1  # default: never recovered
    peak_value = running_max[trough_idx]
    for i in range(trough_idx, len(cumulative)):
        if cumulative[i] >= peak_value:
            recovery_idx = i
            break
    
    return max_dd, peak_idx, trough_idx, recovery_idx


def compute_calmar_ratio(
    returns: np.ndarray,
    annualization: int = 252,
) -> float:
    """Calmar Ratio = Annualized Return / |Max Drawdown|.
    
    Measures return per unit of max drawdown risk.
    """
    ann_return = (np.prod(1 + returns) ** (annualization / len(returns))) - 1
    max_dd, _, _, _ = compute_max_drawdown(returns)
    
    if abs(max_dd) < 1e-10:
        return float("inf") if ann_return > 0 else 0.0
    
    return ann_return / abs(max_dd)


def compute_information_coefficient(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
) -> Tuple[float, float]:
    """Information Coefficient = Spearman rank correlation.
    
    IC measures the predictive power of the model's signals.
    IC > 0.05 is considered meaningful in quantitative finance.
    IC > 0.10 is excellent.
    
    Returns: (IC, p-value)
    """
    mask = ~(np.isnan(predictions) | np.isnan(actual_returns))
    if mask.sum() < 10:
        return 0.0, 1.0
    
    ic, pvalue = stats.spearmanr(predictions[mask], actual_returns[mask])
    return float(ic), float(pvalue)


def compute_information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    annualization: int = 252,
) -> float:
    """Information Ratio = mean(active_return) / tracking_error × √252.
    
    Measures alpha per unit of active risk.
    """
    active_return = returns - benchmark_returns
    tracking_error = np.std(active_return)
    
    if tracking_error < 1e-10:
        return 0.0
    
    return np.mean(active_return) / tracking_error * np.sqrt(annualization)


def compute_var(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Value at Risk (historical method).
    
    VaR_α = -quantile(returns, 1-α)
    
    Positive VaR means the potential loss at the given confidence level.
    """
    return -np.percentile(returns, (1 - confidence) * 100)


def compute_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Conditional Value at Risk (Expected Shortfall).
    
    CVaR = -E[R | R < -VaR]
    
    Average loss in the worst (1-α) scenarios.
    More informative than VaR for tail risk.
    """
    var = compute_var(returns, confidence)
    tail_returns = returns[returns < -var]
    
    if len(tail_returns) == 0:
        return var
    
    return -np.mean(tail_returns)


def compute_win_rate(returns: np.ndarray) -> float:
    """Fraction of positive-return trades."""
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return 0.0
    return np.mean(nonzero > 0)


def compute_profit_factor(returns: np.ndarray) -> float:
    """Profit Factor = sum(gains) / |sum(losses)|.
    
    PF > 1 means profitable. PF > 2 is strong.
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses < 1e-10:
        return float("inf") if gains > 0 else 0.0
    
    return gains / losses


def compute_all_metrics(
    returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    predictions: Optional[np.ndarray] = None,
    actual_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.05,
) -> Dict[str, float]:
    """Compute comprehensive performance metrics.
    
    Returns dict of all financial metrics.
    """
    metrics = {}
    
    # Return metrics
    ann_return = (np.prod(1 + returns) ** (252 / max(len(returns), 1))) - 1
    metrics["annualized_return"] = ann_return
    metrics["cumulative_return"] = np.prod(1 + returns) - 1
    metrics["volatility"] = np.std(returns) * np.sqrt(252)
    
    # Risk-adjusted metrics
    metrics["sharpe_ratio"] = compute_sharpe_ratio(returns, risk_free_rate)
    metrics["sortino_ratio"] = compute_sortino_ratio(returns, risk_free_rate)
    
    # Drawdown
    max_dd, peak_idx, trough_idx, recovery_idx = compute_max_drawdown(returns)
    metrics["max_drawdown"] = max_dd
    metrics["drawdown_duration"] = trough_idx - peak_idx
    metrics["recovery_duration"] = recovery_idx - trough_idx
    metrics["calmar_ratio"] = compute_calmar_ratio(returns)
    
    # Risk metrics
    metrics["var_95"] = compute_var(returns, 0.95)
    metrics["cvar_95"] = compute_cvar(returns, 0.95)
    metrics["var_99"] = compute_var(returns, 0.99)
    
    # Trade metrics
    metrics["win_rate"] = compute_win_rate(returns)
    metrics["profit_factor"] = compute_profit_factor(returns)
    metrics["avg_win"] = np.mean(returns[returns > 0]) if (returns > 0).any() else 0
    metrics["avg_loss"] = np.mean(returns[returns < 0]) if (returns < 0).any() else 0
    metrics["win_loss_ratio"] = (
        abs(metrics["avg_win"] / metrics["avg_loss"])
        if abs(metrics["avg_loss"]) > 1e-10 else float("inf")
    )
    
    # Distribution metrics
    metrics["skewness"] = float(stats.skew(returns))
    metrics["kurtosis"] = float(stats.kurtosis(returns))
    
    # Benchmark comparison
    if benchmark_returns is not None:
        metrics["information_ratio"] = compute_information_ratio(returns, benchmark_returns)
        metrics["alpha"] = ann_return - (
            (np.prod(1 + benchmark_returns) ** (252 / max(len(benchmark_returns), 1))) - 1
        )
    
    # Predictive power
    if predictions is not None and actual_returns is not None:
        ic, ic_p = compute_information_coefficient(predictions, actual_returns)
        metrics["information_coefficient"] = ic
        metrics["ic_pvalue"] = ic_p
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Strategy Performance") -> None:
    """Pretty-print performance metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    
    sections = {
        "Returns": ["annualized_return", "cumulative_return", "volatility"],
        "Risk-Adjusted": ["sharpe_ratio", "sortino_ratio", "calmar_ratio"],
        "Drawdown": ["max_drawdown", "drawdown_duration", "recovery_duration"],
        "Risk": ["var_95", "cvar_95", "var_99"],
        "Trade Quality": ["win_rate", "profit_factor", "win_loss_ratio"],
        "Distribution": ["skewness", "kurtosis"],
        "Alpha": ["information_ratio", "information_coefficient", "alpha"],
    }
    
    format_pct = {"annualized_return", "cumulative_return", "volatility",
                  "max_drawdown", "var_95", "cvar_95", "var_99", "win_rate",
                  "avg_win", "avg_loss", "alpha"}
    
    for section, keys in sections.items():
        available = [k for k in keys if k in metrics]
        if not available:
            continue
        
        print(f"\n  {section}:")
        for key in available:
            val = metrics[key]
            name = key.replace("_", " ").title()
            if key in format_pct:
                print(f"    {name:30s}: {val:>10.2%}")
            elif isinstance(val, float):
                print(f"    {name:30s}: {val:>10.4f}")
            else:
                print(f"    {name:30s}: {val:>10d}")
    
    print(f"\n{'=' * 60}")
