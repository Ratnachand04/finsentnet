"""
Portfolio Optimizer — Mean-Variance (Markowitz) + Risk Parity
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class PortfolioOptimizer:
    """Markowitz-style portfolio optimization with optional risk parity."""

    def __init__(self, risk_free_rate: float = 0.05, trading_days: int = 252):
        self.rf = risk_free_rate
        self.td = trading_days

    def optimize_weights(self, expected_returns: np.ndarray,
                         cov_matrix: np.ndarray,
                         method: str = "max_sharpe",
                         target_return: Optional[float] = None
                         ) -> Dict:
        """
        Optimize portfolio weights.
        Methods: max_sharpe, min_variance, risk_parity, equal_weight
        """
        n = len(expected_returns)
        if n == 0:
            return {"weights": [], "method": method}

        if method == "max_sharpe":
            weights = self._max_sharpe(expected_returns, cov_matrix)
        elif method == "min_variance":
            weights = self._min_variance(cov_matrix)
        elif method == "risk_parity":
            weights = self._risk_parity(cov_matrix)
        else:
            weights = np.ones(n) / n

        port_return = float(np.dot(weights, expected_returns) * self.td)
        port_vol = float(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)) * self.td))
        sharpe = (port_return - self.rf) / max(port_vol, 1e-8)

        return {
            "weights": weights.tolist(),
            "method": method,
            "expected_return": round(port_return * 100, 2),
            "expected_volatility": round(port_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
        }

    def _max_sharpe(self, returns: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Analytical max Sharpe for long-only (approximate via inverse-variance tilt)."""
        n = len(returns)
        try:
            inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-8)
            excess = returns - self.rf / self.td
            raw = inv_cov @ excess
            raw = np.maximum(raw, 0)
            if raw.sum() > 0:
                return raw / raw.sum()
        except np.linalg.LinAlgError:
            pass
        return np.ones(n) / n

    def _min_variance(self, cov: np.ndarray) -> np.ndarray:
        """Global minimum variance portfolio."""
        n = cov.shape[0]
        try:
            inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-8)
            ones = np.ones(n)
            raw = inv_cov @ ones
            raw = np.maximum(raw, 0)
            if raw.sum() > 0:
                return raw / raw.sum()
        except np.linalg.LinAlgError:
            pass
        return np.ones(n) / n

    def _risk_parity(self, cov: np.ndarray, tol: float = 1e-6, max_iter: int = 200) -> np.ndarray:
        """Risk parity via iterative reweighting."""
        n = cov.shape[0]
        w = np.ones(n) / n
        for _ in range(max_iter):
            sigma_p = np.sqrt(w @ cov @ w)
            mc = (cov @ w) / max(sigma_p, 1e-10)
            rc = w * mc
            target_rc = sigma_p / n
            w_new = w * (target_rc / np.maximum(rc, 1e-10))
            w_new = np.maximum(w_new, 0)
            w_new /= w_new.sum()
            if np.max(np.abs(w_new - w)) < tol:
                break
            w = w_new
        return w
