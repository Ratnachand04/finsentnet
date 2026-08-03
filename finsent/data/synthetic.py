"""Synthetic panels for testing the machinery, including deliberate leaks.

Nothing here ever produces a number for the manuscript. Its only job is to give the test
suite data whose ground truth is known, so that claims about the pipeline can be
*verified* rather than asserted:

* ``make_ohlcv_panel`` — geometric Brownian motion with volatility clustering and a
  cross-sectional factor, so cross-sectional ranking and covariance shrinkage have
  something realistic to act on.
* ``make_leaky_panel`` — a panel in which one feature at ``t`` is literally the label at
  ``t+h``. ``tests/test_no_leakage.py`` asserts that a naive random split learns this
  perfectly while the purged walk-forward splitter learns nothing from it. A splitter
  that cannot catch a planted leak cannot be trusted to catch an accidental one.
* ``make_signal_panel`` — scores and probabilities with a *known* information
  coefficient and a *known* calibration error, used to check that the metrics recover
  the planted values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "make_ohlcv_panel",
    "make_leaky_panel",
    "make_signal_panel",
    "make_news_frame",
]


def make_ohlcv_panel(
    n_names: int = 20,
    n_days: int = 800,
    seed: int = 0,
    start: str = "2018-01-02",
    annual_vol: float = 0.30,
) -> dict[str, pd.DataFrame]:
    """OHLCV frames with a common factor, idiosyncratic noise and volatility clustering.

    Includes a market factor so that cross-sectional demeaning has an effect worth
    measuring, and a GARCH-like variance so that the volatility-scaled label band is
    exercised rather than being effectively constant.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    daily_vol = annual_vol / np.sqrt(252.0)

    market = rng.standard_normal(n_days) * daily_vol * 0.6
    out: dict[str, pd.DataFrame] = {}

    for i in range(n_names):
        beta = 0.6 + 0.8 * rng.random()
        sigma = np.empty(n_days)
        sigma[0] = daily_vol
        shock = rng.standard_normal(n_days)
        for t in range(1, n_days):  # simple GARCH(1,1)-style persistence
            sigma[t] = np.sqrt(
                0.02 * daily_vol**2
                + 0.90 * sigma[t - 1] ** 2
                + 0.08 * (sigma[t - 1] * shock[t - 1]) ** 2
            )
        ret = beta * market + sigma * shock

        close = 50.0 * np.exp(np.cumsum(ret))

        # The open follows the PREVIOUS close, never the current one. Splitting the
        # daily move into an overnight gap and an intraday leg is what makes the
        # overnight-gap feature a distinct quantity rather than a copy of the return;
        # generating the open around the same day's close (as an earlier revision did)
        # collapses that distinction and silently invalidates any test that relies on it.
        overnight_share = 0.4
        overnight = overnight_share * ret + sigma * rng.standard_normal(n_days) * 0.5
        prev_close = np.concatenate([[close[0] / np.exp(ret[0])], close[:-1]])
        open_ = prev_close * np.exp(overnight)

        wick = np.abs(rng.standard_normal(n_days)) * sigma * close * 0.6
        high = np.maximum(open_, close) + wick
        low = np.minimum(open_, close) - wick
        volume = np.exp(13.0 + rng.standard_normal(n_days) * 0.4) * (1.0 + 3.0 * np.abs(ret))

        out[f"SYN{i:03d}"] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
    return out


def make_leaky_panel(
    n_names: int = 12,
    n_days: int = 700,
    horizon: int = 5,
    seed: int = 0,
    leak_strength: float = 1.0,
) -> pd.DataFrame:
    """Panel whose feature ``leak`` at ``t`` contains the forward return at ``t+h``.

    ``leak_strength = 1.0`` plants a perfect leak; anything below 1.0 mixes in noise so
    the test can also check a partial leak. Returns a tidy frame with columns
    ``date, ticker, leak, noise_0..noise_3, fwd_ret, y``.
    """
    rng = np.random.default_rng(seed)
    prices = make_ohlcv_panel(n_names=n_names, n_days=n_days, seed=seed)

    frames = []
    for ticker, ohlcv in prices.items():
        close = ohlcv["close"]
        fwd = np.log(close.shift(-horizon) / close)

        noise = rng.standard_normal((len(close), 4))
        leak = leak_strength * fwd.to_numpy() + (1.0 - leak_strength) * rng.standard_normal(
            len(close)
        ) * float(np.nanstd(fwd.to_numpy()))

        sigma = close.pct_change().abs().ewm(span=60, adjust=False).mean().shift(1)
        theta = 0.6 * sigma * np.sqrt(horizon)
        y = np.where(fwd > theta, 2, np.where(fwd < -theta, 0, 1)).astype(float)
        y[~np.isfinite(fwd.to_numpy())] = np.nan

        frames.append(
            pd.DataFrame(
                {
                    "date": close.index,
                    "ticker": ticker,
                    "leak": leak,
                    **{f"noise_{j}": noise[:, j] for j in range(4)},
                    "fwd_ret": fwd.to_numpy(),
                    "y": y,
                }
            )
        )

    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def make_signal_panel(
    n_dates: int = 400,
    n_names: int = 50,
    seed: int = 0,
    target_ic: float = 0.03,
    overconfidence: float = 0.0,
    start: str = "2020-01-02",
) -> pd.DataFrame:
    """Scores and probabilities with a planted IC and a planted calibration error.

    ``overconfidence`` sharpens the probability vector without changing which class is
    argmax, so the accuracy is unchanged while ECE rises. That separation is exactly
    what the paper claims matters, and it lets the tests verify that the calibration
    metrics detect it.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_dates)
    tickers = [f"SYN{i:03d}" for i in range(n_names)]
    rho = float(np.clip(target_ic, -0.99, 0.99))

    rows = []
    for date in dates:
        z = rng.standard_normal(n_names)
        eps = rng.standard_normal(n_names)
        sigma = 0.02 * np.exp(0.25 * rng.standard_normal(n_names))
        fwd = sigma * (rho * z + np.sqrt(1.0 - rho**2) * eps)

        theta = 0.6 * sigma
        y = np.where(fwd > theta, 2, np.where(fwd < -theta, 0, 1))

        logits = np.column_stack([-z, np.zeros(n_names), z]) * (1.0 + overconfidence)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)

        rows.append(
            pd.DataFrame(
                {
                    "date": date,
                    "ticker": tickers,
                    # The score always varies, even when target_ic is zero: a constant
                    # score has undefined cross-sectional correlation, so a degenerate
                    # null panel would silently produce an empty IC series rather than
                    # an IC of zero. mu_hat carries the correctly scaled conditional
                    # mean, which is what the sizing rules consume.
                    "score": z * sigma,
                    "mu_hat": z * sigma * rho,
                    "sigma2": sigma**2,
                    "p_down": probs[:, 0],
                    "p_neutral": probs[:, 1],
                    "p_up": probs[:, 2],
                    "y_true": y,
                    "fwd_ret": fwd,
                    "tradeable": probs.max(axis=1) > 0.40,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def make_news_frame(
    tickers: list[str],
    sessions: pd.DatetimeIndex,
    seed: int = 0,
    coverage: float = 0.35,
    max_per_day: int = 4,
) -> pd.DataFrame:
    """Headlines with realistic publication timestamps, including overnight and weekend.

    ``coverage`` is the fraction of ticker-days that receive at least one headline;
    the default of 0.35 reflects the fact that most ticker-days have no relevant news,
    which the model's null-modality path exists to handle.
    """
    rng = np.random.default_rng(seed)
    rows = []
    sources = ["reuters", "bloomberg", "ap", "wsj", "pr_newswire"]

    for ticker in tickers:
        for session in sessions:
            if rng.random() > coverage:
                continue
            for _ in range(rng.integers(1, max_per_day + 1)):
                # Publish 24-96 hours before the session open, at any hour of the day.
                offset = pd.Timedelta(hours=float(rng.uniform(24.0, 96.0)))
                ts = pd.Timestamp(session).tz_localize("America/New_York") + pd.Timedelta(
                    hours=9, minutes=30
                ) - offset
                rows.append(
                    {
                        "ticker": ticker,
                        "published_at": ts.tz_convert("UTC"),
                        "source": sources[int(rng.integers(0, len(sources)))],
                        "headline": f"{ticker} synthetic headline {rng.integers(0, 10**6)}",
                    }
                )

    return pd.DataFrame(rows)
