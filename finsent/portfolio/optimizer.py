"""
Portfolio optimization engine.
================================

Implements multiple portfolio construction methods:
  - Mean-Variance Optimization (Markowitz)
  - Risk Parity
  - Black-Litterman (incorporating model views)
  - Ledoit-Wolf Shrinkage Covariance

Mathematical Framework:

Mean-Variance:
    max_w  w'μ - (λ/2) w'Σw
    s.t.   w'1 = 1, w_i ≥ 0
    
    Closed form (unconstrained): w* = (1/λ) Σ⁻¹ μ
    
Risk Parity:
    Find w such that: w_i × (Σw)_i = 1/N × w'Σw  for all i
    Each asset contributes equally to portfolio risk.
    
Black-Litterman:
    Posterior return: μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]
    
    where:
        π = equilibrium excess returns (from market cap weights)
        P = pick matrix (which assets model has views on)
        Q = model's return forecasts
        Ω = uncertainty in model's views
        τ = scalar (confidence in prior)
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional, Tuple


def ledoit_wolf_shrinkage(
    returns: np.ndarray,
) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance estimator.
    
    Shrinks sample covariance toward a structured target (identity × avg variance).
    This dramatically improves covariance estimation when T/N is small.
    
    Σ_shrunk = δ × F + (1-δ) × S
    
    where F is the structured target and δ is the optimal shrinkage intensity.
    """
    T, N = returns.shape
    
    # Sample covariance
    mean_returns = returns.mean(axis=0)
    X = returns - mean_returns
    S = (X.T @ X) / T
    
    # Target: scaled identity
    trace_S = np.trace(S)
    mu = trace_S / N
    F = mu * np.eye(N)
    
    # Optimal shrinkage intensity
    delta_sq = 0.0
    for t in range(T):
        x_t = X[t:t+1].T  # column vector
        delta_sq += np.sum((x_t @ x_t.T - S) ** 2)
    delta_sq /= T ** 2
    
    # Numerator of shrinkage formula
    numerator = delta_sq
    denominator = np.sum((S - F) ** 2)
    
    if denominator < 1e-10:
        shrinkage = 1.0
    else:
        shrinkage = min(numerator / denominator, 1.0)
    
    return shrinkage * F + (1 - shrinkage) * S


class PortfolioOptimizer:
    """Multi-method portfolio optimizer.
    
    Takes model predictions and constructs optimal portfolios.
    """
    
    def __init__(
        self,
        method: str = "mean_variance",
        risk_free_rate: float = 0.05,
        target_volatility: float = 0.15,
        max_position: float = 0.10,
        use_shrinkage: bool = True,
    ):
        self.method = method
        self.risk_free_rate = risk_free_rate / 252  # daily
        self.target_volatility = target_volatility / np.sqrt(252)  # daily
        self.max_position = max_position
        self.use_shrinkage = use_shrinkage
    
    def optimize(
        self,
        expected_returns: np.ndarray,   # (N,) expected returns per asset
        return_history: np.ndarray,      # (T, N) historical return matrix
        confidence: Optional[np.ndarray] = None,  # (N,) model confidence per asset
    ) -> np.ndarray:
        """Compute optimal portfolio weights.
        
        Args:
            expected_returns: Model-predicted expected returns
            return_history: Historical returns for covariance estimation
            confidence: Per-asset confidence (scales uncertainty)
        
        Returns:
            weights: (N,) optimal portfolio weights
        """
        if self.method == "mean_variance":
            return self._mean_variance(expected_returns, return_history)
        elif self.method == "risk_parity":
            return self._risk_parity(return_history)
        elif self.method == "black_litterman":
            return self._black_litterman(expected_returns, return_history, confidence)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _mean_variance(
        self,
        expected_returns: np.ndarray,
        return_history: np.ndarray,
    ) -> np.ndarray:
        """Mean-Variance Optimization (Markowitz, 1952).
        
        Maximize Sharpe ratio subject to constraints.
        
        max_w  (w'μ - r_f) / √(w'Σw)
        s.t.   Σw_i = 1, 0 ≤ w_i ≤ max_position
        """
        N = len(expected_returns)
        
        # Covariance estimation
        if self.use_shrinkage:
            cov_matrix = ledoit_wolf_shrinkage(return_history)
        else:
            cov_matrix = np.cov(return_history, rowvar=False)
        
        # Objective: negative Sharpe (minimize)
        def neg_sharpe(w):
            port_return = w @ expected_returns - self.risk_free_rate
            port_vol = np.sqrt(w @ cov_matrix @ w)
            if port_vol < 1e-10:
                return 0.0
            return -port_return / port_vol
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # weights sum to 1
        ]
        
        bounds = [(0, self.max_position) for _ in range(N)]
        
        # Initial guess: equal weight
        w0 = np.ones(N) / N
        
        result = minimize(
            neg_sharpe, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )
        
        return result.x if result.success else w0
    
    def _risk_parity(
        self,
        return_history: np.ndarray,
    ) -> np.ndarray:
        """Risk Parity optimization.
        
        Find weights where each asset contributes equally to portfolio risk.
        
        RC_i = w_i × (Σw)_i / σ_p
        Objective: minimize Σ(RC_i - target)²
        """
        N = return_history.shape[1]
        
        if self.use_shrinkage:
            cov_matrix = ledoit_wolf_shrinkage(return_history)
        else:
            cov_matrix = np.cov(return_history, rowvar=False)
        
        target_risk = 1.0 / N
        
        def risk_parity_objective(w):
            port_vol = np.sqrt(w @ cov_matrix @ w)
            if port_vol < 1e-10:
                return 1e6
            
            # Marginal risk contribution
            mrc = cov_matrix @ w / port_vol
            rc = w * mrc  # risk contribution
            rc_pct = rc / port_vol  # percentage risk contribution
            
            # Minimize deviation from equal risk
            return np.sum((rc_pct - target_risk) ** 2)
        
        bounds = [(0.001, self.max_position) for _ in range(N)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        w0 = np.ones(N) / N
        
        result = minimize(
            risk_parity_objective, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        return result.x if result.success else w0
    
    def _black_litterman(
        self,
        model_returns: np.ndarray,
        return_history: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        tau: float = 0.05,
    ) -> np.ndarray:
        """Black-Litterman model incorporating model views.
        
        The model's predictions are treated as "views" on expected returns.
        Confidence calibration directly maps to view uncertainty (Ω).
        
        Higher model confidence → lower Ω_ii → view weighted more heavily.
        """
        N = len(model_returns)
        
        if self.use_shrinkage:
            Sigma = ledoit_wolf_shrinkage(return_history)
        else:
            Sigma = np.cov(return_history, rowvar=False)
        
        # Equilibrium returns (from equal-weight portfolio as proxy)
        w_eq = np.ones(N) / N
        pi = 2.5 * Sigma @ w_eq  # risk aversion λ = 2.5 (typical)
        
        # View matrix (identity — one view per asset)
        P = np.eye(N)
        Q = model_returns
        
        # View uncertainty
        if confidence is not None:
            # Higher confidence → lower uncertainty
            omega_diag = (1 - confidence.clip(0.1, 0.9)) * np.diag(tau * Sigma)
        else:
            omega_diag = np.diag(tau * Sigma)
        
        Omega = np.diag(omega_diag)
        
        # Black-Litterman posterior
        tau_Sigma = tau * Sigma
        tau_Sigma_inv = np.linalg.inv(tau_Sigma)
        Omega_inv = np.linalg.inv(Omega)
        
        # Posterior expected returns
        M_inv = np.linalg.inv(tau_Sigma_inv + P.T @ Omega_inv @ P)
        mu_BL = M_inv @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q)
        
        # Optimal weights from posterior
        lambda_risk = 2.5
        w_BL = (1 / lambda_risk) * np.linalg.inv(Sigma) @ mu_BL
        
        # Normalize and constrain
        w_BL = np.clip(w_BL, 0, self.max_position)
        if w_BL.sum() > 0:
            w_BL /= w_BL.sum()
        else:
            w_BL = np.ones(N) / N
        
        return w_BL
