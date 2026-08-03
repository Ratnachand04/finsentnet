"""Factor attribution: is the "alpha" just repackaged beta and momentum?

A referee at any finance-adjacent venue asks this within the first page of the results
section, and the honest answer for most equity ML strategies is "substantially yes".
This module runs the regression that settles it:

    r_strategy(t) = alpha + b_mkt*MKT + b_smb*SMB + b_hml*HML
                          + b_rmw*RMW + b_cma*CMA + b_mom*MOM + b_str*STR + e(t)

Standard errors are Newey-West, because strategy returns are autocorrelated and OLS
standard errors would overstate the significance of alpha.

Factor data: supply Ken French / AQR style daily factors as a CSV via
``load_factors_csv``. The module never fabricates factors -- if none are available it
says so and the report prints "factor attribution unavailable" rather than a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from finsent.eval.stats import newey_west_lrv

__all__ = ["FactorRegression", "regress_on_factors", "load_factors_csv", "FACTOR_COLUMNS"]

FACTOR_COLUMNS = ("MKT", "SMB", "HML", "RMW", "CMA", "MOM", "STR")
_EPS = 1e-12


@dataclass
class FactorRegression:
    """Result of a factor regression, in the form the paper's Table 6 needs."""

    alpha_per_period: float
    alpha_annualised: float
    alpha_tstat: float
    alpha_pvalue: float
    betas: dict[str, float]
    beta_tstats: dict[str, float]
    r_squared: float
    adj_r_squared: float
    n_obs: int
    factors_used: tuple[str, ...]
    residuals: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def summary(self) -> str:
        lines = [
            f"n = {self.n_obs}   R^2 = {self.r_squared:.4f} (adj {self.adj_r_squared:.4f})",
            f"alpha = {self.alpha_annualised:.4%}/yr   t = {self.alpha_tstat:.2f}"
            f"   p = {self.alpha_pvalue:.4f}",
        ]
        for name in self.factors_used:
            lines.append(f"  {name:<5s} beta = {self.betas[name]:+.4f}"
                         f"   t = {self.beta_tstats[name]:+.2f}")
        if np.isfinite(self.alpha_tstat) and abs(self.alpha_tstat) < 3.0:
            lines.append(
                "  NOTE: |t| < 3.0, the threshold Harvey, Liu & Zhu (2016) advocate for a "
                "newly discovered factor. Do not claim a novel risk premium."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "alpha_per_period": self.alpha_per_period,
            "alpha_annualised": self.alpha_annualised,
            "alpha_tstat": self.alpha_tstat,
            "alpha_pvalue": self.alpha_pvalue,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "n_obs": self.n_obs,
            **{f"beta_{k}": v for k, v in self.betas.items()},
            **{f"t_{k}": v for k, v in self.beta_tstats.items()},
        }


def load_factors_csv(path: str | Path, date_column: str = "date") -> pd.DataFrame:
    """Load a daily factor file.

    Expected layout: a date column plus any subset of ``FACTOR_COLUMNS``, with returns
    expressed as decimals (0.0031, not 0.31). Ken French files ship percentages, so
    divide by 100 before saving.
    """
    df = pd.read_csv(path)
    if date_column not in df.columns:
        raise ValueError(f"factor file {path} has no '{date_column}' column")
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column).sort_index()

    keep = [c for c in df.columns if c.upper() in FACTOR_COLUMNS]
    df = df[keep]
    df.columns = [c.upper() for c in df.columns]

    if (df.abs() > 1.0).to_numpy().mean() > 0.5:
        raise ValueError(
            f"{path}: factor values look like percentages. Divide by 100 so that a 31 bp "
            "day is 0.0031; otherwise every beta is off by two orders of magnitude."
        )
    return df


def regress_on_factors(
    strategy_returns: pd.Series,
    factors: pd.DataFrame,
    periods_per_year: int = 252,
    hac_lags: int | None = 10,
) -> FactorRegression:
    """OLS with Newey-West standard errors, aligned on the intersection of dates."""
    y_ser = pd.Series(strategy_returns).dropna()
    joined = pd.concat([y_ser.rename("y"), factors], axis=1, join="inner").dropna()
    if len(joined) < 30:
        raise ValueError(
            f"only {len(joined)} overlapping observations between strategy and factors; "
            "refusing to report an attribution that noisy"
        )

    names = tuple(c for c in joined.columns if c != "y")
    y = joined["y"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(joined)), joined[list(names)].to_numpy(dtype=float)])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > _EPS else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if np.isfinite(r2) and n > k else np.nan

    # Newey-West covariance: (X'X)^-1 S (X'X)^-1 with S the HAC meat matrix.
    xtx_inv = np.linalg.pinv(X.T @ X)
    lags = hac_lags if hac_lags is not None else max(1, int(4 * (n / 100) ** (2 / 9)))
    u = X * resid[:, None]
    S = u.T @ u
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = u[lag:].T @ u[:-lag]
        S += w * (gamma + gamma.T)
    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), _EPS))

    tstats = beta / se
    pvals = 2.0 * (1.0 - sps.t.cdf(np.abs(tstats), df=max(n - k, 1)))

    return FactorRegression(
        alpha_per_period=float(beta[0]),
        alpha_annualised=float(beta[0] * periods_per_year),
        alpha_tstat=float(tstats[0]),
        alpha_pvalue=float(pvals[0]),
        betas={name: float(beta[i + 1]) for i, name in enumerate(names)},
        beta_tstats={name: float(tstats[i + 1]) for i, name in enumerate(names)},
        r_squared=float(r2),
        adj_r_squared=float(adj_r2),
        n_obs=int(n),
        factors_used=names,
        residuals=resid,
    )
