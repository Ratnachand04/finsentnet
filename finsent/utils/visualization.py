"""
Visualization utilities for FinSentNet.
Financial-grade plotting for model diagnostics and backtesting results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Dict, List


plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "profit": "#2ecc71",
    "loss": "#e74c3c",
    "neutral": "#95a5a6",
    "primary": "#3498db",
    "secondary": "#e67e22",
    "benchmark": "#7f8c8d",
}


def plot_equity_curve(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    drawdown: Optional[pd.Series] = None,
    title: str = "Strategy Equity Curve",
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot equity curve with optional benchmark and drawdown overlay.
    
    Args:
        equity: Strategy cumulative returns / NAV indexed by date.
        benchmark: Buy-and-hold benchmark for comparison.
        drawdown: Drawdown series (negative values).
        title: Plot title.
    """
    if drawdown is not None:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None

    # Equity curve
    ax1.plot(equity.index, equity.values, color=COLORS["primary"],
             linewidth=1.5, label="Strategy")
    if benchmark is not None:
        ax1.plot(benchmark.index, benchmark.values, color=COLORS["benchmark"],
                 linewidth=1.0, linestyle="--", alpha=0.7, label="Benchmark")
    
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Drawdown
    if ax2 is not None and drawdown is not None:
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                         color=COLORS["loss"], alpha=0.4)
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot returns histogram with normal overlay and key statistics."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram
    n, bins, patches = ax.hist(returns, bins=80, density=True, alpha=0.7,
                                color=COLORS["primary"], edgecolor="white")
    
    # Color positive/negative
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0:
            patch.set_facecolor(COLORS["loss"])
        else:
            patch.set_facecolor(COLORS["profit"])
    
    # Normal overlay
    from scipy.stats import norm
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 200)
    ax.plot(x, norm.pdf(x, mu, sigma), color="black", linewidth=2,
            linestyle="--", label=f"Normal(μ={mu:.4f}, σ={sigma:.4f})")
    
    # Stats annotation
    skew = returns.skew()
    kurt = returns.kurtosis()
    stats_text = (
        f"Mean: {mu:.4f}\nStd: {sigma:.4f}\n"
        f"Skew: {skew:.2f}\nKurtosis: {kurt:.2f}\n"
        f"Min: {returns.min():.4f}\nMax: {returns.max():.4f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_attention_weights(
    attention_matrix: np.ndarray,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Cross-Modal Attention Weights",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize attention matrix as heatmap.
    
    Useful for interpreting which news tokens attend to which price features.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attention_matrix, aspect="auto", cmap="YlOrRd")
    
    if x_labels:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    if y_labels:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
    
    plt.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_regime_transitions(
    prices: pd.Series,
    regimes: pd.Series,
    regime_names: Dict[int, str] = None,
    title: str = "Market Regime Detection",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot price series colored by detected market regime."""
    if regime_names is None:
        regime_names = {0: "Bull", 1: "Bear", 2: "Crisis", 3: "High Vol", 4: "Low Vol"}
    
    regime_colors = {0: "#2ecc71", 1: "#e74c3c", 2: "#8e44ad", 3: "#e67e22", 4: "#3498db"}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for regime_id in regimes.unique():
        mask = regimes == regime_id
        label = regime_names.get(regime_id, f"Regime {regime_id}")
        color = regime_colors.get(regime_id, "#95a5a6")
        ax.scatter(prices.index[mask], prices.values[mask],
                   c=color, s=2, alpha=0.6, label=label)
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(markerscale=5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
