"""The causality contract, checked against the real cached panel rather than a fixture.

Every other leakage test in this suite runs on synthetic data whose ground truth we
control. That is the right way to test the machinery, but it cannot catch a mistake made
while *building* the study panel -- an off-by-one in the lag, a label joined to the wrong
session, a traded return that starts a day early. Those defects live in the data, not in
the code under test, so they need the data to detect.

These tests skip when the cache is absent, so a fresh clone still passes; they are
meaningful exactly when someone has built a panel and is about to train on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

PRICES = Path("data/cache/price_panel.parquet")
STUDY = Path("data/cache/study_panel.parquet")

pytestmark = pytest.mark.skipif(
    not (PRICES.exists() and STUDY.exists()),
    reason="no cached panel; run experiments/01_build_dataset.py first",
)


@pytest.fixture(scope="module")
def panels():
    px = pd.read_parquet(PRICES)
    sp = pd.read_parquet(STUDY)
    px["date"] = pd.to_datetime(px["date"])
    sp["date"] = pd.to_datetime(sp["date"])
    return px, sp


def _sample_tickers(sp, n=5):
    return sorted(sp["ticker"].unique())[:n]


def test_a_feature_at_t_knows_yesterday_and_not_today(panels):
    """``ret_1`` at date t must describe the session ending at t-1, not at t.

    A model decides at the open of t, so a feature built from the close of t is
    unknowable. This is the defect that inflated a trailing-return rank IC from 0.004
    to 0.108 on driftless data, and it is invisible to any train/test splitter because
    nothing crosses the split boundary.
    """
    px, sp = panels
    for ticker in _sample_tickers(sp):
        a = px[px["ticker"] == ticker].sort_values("date").set_index("date")
        b = sp[sp["ticker"] == ticker].sort_values("date").set_index("date")
        r = a["close"].pct_change()
        j = b.join(pd.DataFrame({"ends_t": r, "ends_tm1": r.shift(1)}),
                   how="inner").dropna(subset=["ret_1", "ends_t", "ends_tm1"])
        if len(j) < 200:
            continue

        lagged = abs(stats.spearmanr(j["ret_1"], j["ends_tm1"])[0])
        same_day = abs(stats.spearmanr(j["ret_1"], j["ends_t"])[0])
        assert lagged > 0.5, (
            f"{ticker}: ret_1 does not track the previous session (rho={lagged:.3f}); "
            "the feature panel may be lagged by more than one session, which throws "
            "away information rather than leaking it"
        )
        assert same_day < 0.15, (
            f"{ticker}: ret_1 tracks the *same* session (rho={same_day:.3f}); the "
            "execution lag is missing and the panel leaks"
        )
        assert lagged > same_day * 3, f"{ticker}: lag direction is ambiguous"


def test_the_traded_return_starts_at_the_decision_open(panels):
    """``next_ret`` at t must be exactly open(t+1)/open(t) - 1.

    The backtester accrues this on the book it held from t. If it were instead the
    return *ending* at t, every reported Sharpe would be a look-ahead artefact.
    """
    px, sp = panels
    checked = 0
    for ticker in _sample_tickers(sp):
        a = px[px["ticker"] == ticker].sort_values("date").set_index("date")
        b = sp[sp["ticker"] == ticker].sort_values("date").set_index("date")
        expected = a["open"].shift(-1) / a["open"] - 1.0
        j = b.join(expected.rename("expected"), how="inner").dropna(
            subset=["next_ret", "expected"])
        if len(j) < 200:
            continue
        worst = float((j["next_ret"] - j["expected"]).abs().max())
        assert worst < 1e-9, (
            f"{ticker}: next_ret differs from open(t+1)/open(t)-1 by up to {worst:.2e}"
        )
        checked += 1
    assert checked, "no ticker had enough overlapping rows to check"


def test_no_feature_column_correlates_with_the_return_it_is_used_to_predict(panels):
    """A blunt sweep for the leak we have not thought of.

    Any feature whose cross-sectional rank IC against the traded return is implausibly
    large for daily equities is far more likely to be a timestamp defect than a
    discovery, so this test asserts an upper bound rather than a lower one. The bound
    is SPEC.md's audit threshold.
    """
    _, sp = panels
    exclude = {"date", "ticker", "y_dir", "y_ret", "fwd_ret", "next_ret",
               "weight", "in_universe"}
    features = [c for c in sp.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(sp[c])]
    assert features, "no feature columns found in the study panel"

    d = sp.dropna(subset=["next_ret"])
    suspicious = {}
    for col in features:
        sub = d.dropna(subset=[col])
        ic = sub.groupby("date").apply(
            lambda g: stats.spearmanr(g[col], g["next_ret"])[0] if len(g) > 5 else np.nan,
            include_groups=False,
        ).dropna()
        if len(ic) > 100 and abs(float(ic.mean())) > 0.06:
            suspicious[col] = float(ic.mean())
    assert not suspicious, (
        f"features with implausible rank IC against the traded return: {suspicious}. "
        "At daily frequency on liquid US equities this is a timestamp defect until "
        "proven otherwise."
    )
