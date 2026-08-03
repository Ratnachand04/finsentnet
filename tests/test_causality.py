"""The causality contract: nothing at ``t`` may read anything after ``t``.

``test_no_leakage.py`` shows that a splitter cannot rescue a contaminated feature, so
this contract has to be enforced where the features are built. The test is mechanical:
perturb the input at position ``k``, recompute, and assert that every output strictly
before ``k`` is bit-identical.

It runs over every indicator primitive and over the full twelve-feature set, so a new
feature cannot be added without either satisfying the contract or failing here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsent.data import indicators as ind
from finsent.data.features_causal import FEATURE_NAMES, compute_features
from finsent.data.synthetic import make_ohlcv_panel

PERTURB_AT = 200


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return make_ohlcv_panel(n_names=1, n_days=400, seed=5)["SYN000"]


def _assert_prefix_unchanged(before: pd.Series | pd.DataFrame, after, k: int, what: str) -> None:
    b = np.asarray(before)[:k]
    a = np.asarray(after)[:k]
    same = np.isclose(b, a, rtol=0, atol=0, equal_nan=True)
    if not same.all():
        first = int(np.argmax(~same))
        raise AssertionError(
            f"{what} is NOT causal: perturbing input {k} changed output {first} "
            f"({b.ravel()[first]!r} -> {a.ravel()[first]!r})"
        )


@pytest.mark.parametrize(
    "name,fn",
    [
        ("sma", lambda d: ind.sma(d["close"], 20)),
        ("ema", lambda d: ind.ema(d["close"], 12)),
        ("wilder_smooth", lambda d: ind.wilder_smooth(d["close"], 14)),
        ("rsi", lambda d: ind.rsi(d["close"], 14)),
        ("macd_hist", lambda d: ind.macd(d["close"])[2]),
        ("true_range", lambda d: ind.true_range(d["high"], d["low"], d["close"])),
        ("atr", lambda d: ind.atr(d["high"], d["low"], d["close"], 14)),
        ("bollinger_pctb", lambda d: ind.bollinger_pctb(d["close"], 20)),
        ("ewma_volatility", lambda d: ind.ewma_volatility(np.log(d["close"]).diff(), 20)),
        (
            "garman_klass",
            lambda d: ind.garman_klass_volatility(d["open"], d["high"], d["low"], d["close"], 20),
        ),
        ("parkinson", lambda d: ind.parkinson_volatility(d["high"], d["low"], 20)),
        ("obv", lambda d: ind.obv(d["close"], d["volume"])),
    ],
)
def test_indicator_is_causal(ohlcv, name, fn):
    base = fn(ohlcv)

    shocked = ohlcv.copy()
    for col in ("open", "high", "low", "close"):
        shocked.iloc[PERTURB_AT:, shocked.columns.get_loc(col)] *= 1.5
    shocked.iloc[PERTURB_AT:, shocked.columns.get_loc("volume")] *= 3.0

    _assert_prefix_unchanged(base, fn(shocked), PERTURB_AT, name)


def test_full_feature_set_is_causal(ohlcv):
    base = compute_features(ohlcv)

    shocked = ohlcv.copy()
    for col in ("open", "high", "low", "close"):
        shocked.iloc[PERTURB_AT:, shocked.columns.get_loc(col)] *= 1.5
    shocked.iloc[PERTURB_AT:, shocked.columns.get_loc("volume")] *= 3.0
    after = compute_features(shocked)

    for name in FEATURE_NAMES:
        _assert_prefix_unchanged(base[name], after[name], PERTURB_AT, f"feature {name}")


def test_feature_set_has_exactly_the_declared_columns(ohlcv):
    from finsent.config import load_config

    feats = compute_features(ohlcv)
    assert tuple(feats.columns) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == load_config().data.n_features == 12


def test_rev_5_is_not_collinear_with_ret_5(ohlcv):
    """``rev_5`` is an overnight-gap sum, not ``-ret_5``.

    Defining short-term reversal as the negation of the five-day return would make two
    of the twelve features perfectly collinear, which is the kind of thing a referee
    notices in a correlation matrix.
    """
    feats = compute_features(ohlcv).dropna()
    corr = float(np.corrcoef(feats["rev_5"], feats["ret_5"])[0, 1])
    assert abs(corr) < 0.95, f"rev_5 and ret_5 correlate at {corr:.3f}: effectively collinear"


def test_features_are_not_correlated_with_the_future():
    """No feature may predict the forward return on data that contains no signal.

    Pooled across many independent names, because a single 400-day path gives roughly
    150 usable observations of a highly persistent variable such as ``mom_12_1``, and a
    spurious correlation of 0.3 on that sample is entirely ordinary. Testing one path
    would either fail at random or force a threshold so loose it detects nothing --
    which is the same small-sample trap the paper's own evaluation section is about.
    """
    panel = make_ohlcv_panel(n_names=25, n_days=900, seed=17)

    pooled = []
    for ohlcv in panel.values():
        feats = compute_features(ohlcv)
        fwd = np.log(ohlcv["close"].shift(-5) / ohlcv["close"])
        pooled.append(feats.assign(_fwd=fwd).dropna())
    joined = pd.concat(pooled, ignore_index=True)
    assert len(joined) > 10_000, f"only {len(joined)} pooled observations"

    for name in FEATURE_NAMES:
        corr = float(np.corrcoef(joined[name], joined["_fwd"])[0, 1])
        assert abs(corr) < 0.10, (
            f"feature {name} correlates {corr:.3f} with the forward 5-day return over "
            f"{len(joined)} observations of driftless data; that indicates a leak, not "
            "sampling noise"
        )


def test_features_must_be_lagged_for_an_open_executed_decision():
    """Regression test for a real look-ahead bug found by the SPEC.md audit threshold.

    A feature dated ``t`` uses the close of day ``t``; an open-to-open label starting at
    ``t`` begins *before* that close, so the two share day ``t``'s return. On pure
    random-walk data that alone produced a cross-sectional rank IC of **0.108** for
    ``ret_5`` -- five times a realistic value, entirely spurious, and invisible to any
    splitter. After lagging one session it is 0.004.

    Both directions are asserted, because a test that only checks the fixed version
    cannot tell you whether the fix is doing anything.
    """
    from finsent.data.features_causal import build_feature_panel, lag_for_open_execution
    from finsent.data.labeling import forward_log_return
    from finsent.eval.metrics import information_coefficient

    panel = make_ohlcv_panel(n_names=25, n_days=900, seed=23)
    horizon = 5

    labels = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": d.index,
                    "ticker": t,
                    "fwd_ret": forward_log_return(d["open"], horizon).to_numpy(),
                }
            )
            for t, d in panel.items()
        ],
        ignore_index=True,
    )

    unlagged = build_feature_panel(panel, cross_sectional=True, lag_sessions=0)
    lagged = lag_for_open_execution(unlagged, sessions=1)

    def ic_of(frame: pd.DataFrame, column: str) -> float:
        merged = frame.merge(labels, on=["date", "ticker"], how="inner").dropna(
            subset=[column, "fwd_ret"]
        )
        return float(
            information_coefficient(merged[column], merged["fwd_ret"], merged["date"]).mean()
        )

    ic_unlagged = ic_of(unlagged, "ret_5")
    ic_lagged = ic_of(lagged, "ret_5")

    assert ic_unlagged > 0.05, (
        f"the unlagged bug should be clearly visible; measured IC {ic_unlagged:.4f}. "
        "If this fails, the fixture no longer reproduces the failure it guards against."
    )
    assert abs(ic_lagged) < 0.02, (
        f"after lagging one session, ret_5 must carry no forward information on "
        f"random-walk data; measured IC {ic_lagged:.4f}"
    )


def test_build_feature_panel_lags_by_default():
    """The safe behaviour must be the default, not an option the caller remembers."""
    from finsent.data.features_causal import build_feature_panel

    panel = make_ohlcv_panel(n_names=6, n_days=400, seed=24)
    default = build_feature_panel(panel, cross_sectional=False)
    explicit = build_feature_panel(panel, cross_sectional=False, lag_sessions=0)

    merged = default.merge(explicit, on=["date", "ticker"], suffixes=("_lag", "_raw"))
    one_ticker = merged[merged["ticker"] == "SYN000"].sort_values("date")

    lagged = one_ticker["ret_1_lag"].to_numpy()[1:]
    raw = one_ticker["ret_1_raw"].to_numpy()[:-1]
    both = np.isfinite(lagged) & np.isfinite(raw)
    assert np.allclose(lagged[both], raw[both]), (
        "the default panel must equal the raw panel shifted by exactly one session"
    )


def test_cross_sectional_ranking_uses_only_same_day_information():
    """Ranking must be within a date, never pooled across the sample."""
    from finsent.data.features_causal import build_feature_panel

    panel_data = make_ohlcv_panel(n_names=8, n_days=300, seed=2)
    ranked = build_feature_panel(panel_data, cross_sectional=True)

    per_day = ranked.dropna(subset=["ret_5"]).groupby("date")["ret_5"]
    means = per_day.mean()
    assert means.abs().max() < 0.05, (
        "cross-sectionally ranked features must be approximately centred within each "
        f"day; observed max |mean| = {means.abs().max():.3f}"
    )
    assert ranked["ret_5"].dropna().between(-0.5, 0.5).all(), "ranks must lie in [-0.5, 0.5]"
