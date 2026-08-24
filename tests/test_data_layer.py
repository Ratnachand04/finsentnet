"""Universe construction, news alignment, and label uniqueness.

The universe test is the survivorship check a referee performs first; the alignment
tests pin the timestamp contract that decides whether the whole study is valid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsent.data.align import (
    AlignmentConfig,
    align_news_to_sessions,
    assert_no_lookahead,
    decision_session,
    news_coverage_stats,
)
from finsent.data.synthetic import make_news_frame, make_ohlcv_panel
from finsent.data.uniqueness import (
    average_uniqueness,
    concurrency,
    effective_sample_size,
    sequential_bootstrap,
    uniqueness_from_horizon,
)
from finsent.data.universe import (
    UniverseConfig,
    build_index_universe,
    build_liquidity_universe,
    membership_stats,
)


# --------------------------------------------------------------------------------------
# Timestamp alignment — the contract in SPEC.md 2.3
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sessions() -> pd.DatetimeIndex:
    return pd.bdate_range("2022-03-01", periods=40)


def test_the_three_worked_examples_from_the_specification(sessions):
    """Tuesday 09:00, Tuesday 16:30 and Saturday 03:00, with a 24-hour lag.

    The middle case is the one people get wrong: a 16:30 headline is *not* usable at the
    next morning's 09:30 open, because only 17 hours have passed.
    """
    cfg = AlignmentConfig(min_lag_hours=24)
    tz = "America/New_York"

    cases = [
        (pd.Timestamp("2022-03-08 09:00", tz=tz), pd.Timestamp("2022-03-09")),
        (pd.Timestamp("2022-03-08 16:30", tz=tz), pd.Timestamp("2022-03-10")),
        (pd.Timestamp("2022-03-05 03:00", tz=tz), pd.Timestamp("2022-03-07")),
    ]
    published = pd.Series([c[0] for c in cases])
    decisions = decision_session(published, sessions, cfg)

    for (ts, expected), got in zip(cases, decisions):
        assert pd.Timestamp(got) == expected, (
            f"headline at {ts} should inform the {expected.date()} open, got {got}"
        )


def test_every_aligned_headline_satisfies_the_lag_inequality(sessions):
    news = make_news_frame(["AAA", "BBB"], sessions, seed=0, coverage=0.6)
    cfg = AlignmentConfig(min_lag_hours=24, lookback_hours=72, max_headlines=8)

    aligned = align_news_to_sessions(news, sessions, cfg)
    assert not aligned.empty
    assert (aligned["lag_hours"] >= 24 - 1e-9).all()
    assert (aligned["lag_hours"] <= 72 + 1e-9).all()
    assert_no_lookahead(aligned, cfg)


def test_lookahead_assertion_actually_fires(sessions):
    """A guard that cannot fail is not a guard."""
    bad = pd.DataFrame(
        {
            "published_at": [pd.Timestamp("2022-03-09 09:00", tz="UTC")],
            "lag_hours": [2.0],
            "decision_date": [pd.Timestamp("2022-03-09")],
        }
    )
    with pytest.raises(AssertionError, match="LOOKAHEAD"):
        assert_no_lookahead(bad, AlignmentConfig(min_lag_hours=24))


def test_headline_cap_keeps_the_most_recent(sessions):
    cfg = AlignmentConfig(min_lag_hours=24, lookback_hours=96, max_headlines=3)
    news = make_news_frame(["AAA"], sessions, seed=1, coverage=1.0, max_per_day=8)

    aligned = align_news_to_sessions(news, sessions, cfg)
    counts = aligned.groupby(["ticker", "decision_date"]).size()
    assert counts.max() <= 3, "the per-day headline cap was not applied"

    for (_ticker, _date), grp in aligned.groupby(["ticker", "decision_date"]):
        assert list(grp["rank"]) == sorted(grp["rank"]), "ranks must be recency-ordered"
        assert grp["lag_hours"].is_monotonic_increasing, "rank 0 must be the newest"


def test_news_coverage_is_reported_rather_than_silently_dropped(sessions):
    """Most ticker-days have no news, and Table 1 must say so.

    Dropping them would be a selection bias: news days are systematically more volatile,
    so a model evaluated only on them is evaluated on an easier and unrepresentative
    sample.
    """
    tickers = ["AAA", "BBB", "CCC"]
    news = make_news_frame(tickers, sessions, seed=2, coverage=0.3)
    aligned = align_news_to_sessions(news, sessions)

    keys = pd.DataFrame(
        [(t, d) for t in tickers for d in sessions], columns=["ticker", "date"]
    )
    stats = news_coverage_stats(aligned, keys)

    assert 0.0 < stats["coverage"] < 1.0
    assert stats["panel_rows"] == len(tickers) * len(sessions)
    assert stats["mean_headlines"] < stats["mean_headlines_when_present"]


# --------------------------------------------------------------------------------------
# Universe — the survivorship check
# --------------------------------------------------------------------------------------
def test_liquidity_universe_turns_over_and_passes_the_survivorship_check():
    panel = make_ohlcv_panel(n_names=40, n_days=600, seed=3)
    close = pd.DataFrame({k: v["close"] for k, v in panel.items()})
    volume = pd.DataFrame({k: v["volume"] for k, v in panel.items()})

    universe = build_liquidity_universe(
        close, volume, UniverseConfig(n_names=20, min_adv_usd=0.0, min_price=0.0)
    )
    stats = membership_stats(universe)

    assert stats["mean_size"] == pytest.approx(20, abs=1)
    assert stats["survivorship_check_passed"], (
        "no name present on the first date has left by the last: this is the signature "
        "of a retroactively applied constituent list"
    )
    assert stats["n_dropped_from_first_date"] >= 5


def test_universe_selection_uses_no_same_day_information():
    """The screen at a rebalance date must not read that day's price or volume."""
    panel = make_ohlcv_panel(n_names=15, n_days=400, seed=4)
    close = pd.DataFrame({k: v["close"] for k, v in panel.items()})
    volume = pd.DataFrame({k: v["volume"] for k, v in panel.items()})
    cfg = UniverseConfig(n_names=8, min_adv_usd=0.0, min_price=0.0, lookback_days=30)

    base = build_liquidity_universe(close, volume, cfg)

    shocked_volume = volume.copy()
    cut = 300
    shocked_volume.iloc[cut:] *= 100.0
    after = build_liquidity_universe(close, shocked_volume, cfg)

    early_base = base[base["date"] < close.index[cut]]
    early_after = after[after["date"] < close.index[cut]]
    assert len(early_base) == len(early_after)
    assert set(map(tuple, early_base.to_numpy())) == set(map(tuple, early_after.to_numpy())), (
        "membership before the shock changed: the screen is reading future volume"
    )


def test_index_universe_applies_changes_from_the_effective_date():
    sessions = pd.bdate_range("2020-01-01", periods=30)
    changes = pd.DataFrame(
        {
            "effective_date": [sessions[10], sessions[20]],
            "ticker": ["NEW", "OLD"],
            "action": ["add", "remove"],
        }
    )
    universe = build_index_universe(changes, sessions, initial_members=["OLD", "KEEP"])

    def members(day):
        return set(universe[universe["date"] == day]["ticker"])

    assert members(sessions[5]) == {"OLD", "KEEP"}
    assert members(sessions[15]) == {"OLD", "KEEP", "NEW"}
    assert members(sessions[25]) == {"KEEP", "NEW"}, "a removed name must leave the universe"


# --------------------------------------------------------------------------------------
# Uniqueness — overlapping labels
# --------------------------------------------------------------------------------------
def test_non_overlapping_labels_have_uniqueness_one():
    t1 = np.array([0.0, 1.0, 2.0, 3.0])  # each label resolves on its own bar
    u = average_uniqueness(t1)
    assert np.allclose(u, 1.0)


def test_overlapping_labels_have_uniqueness_near_one_over_h():
    """With h = 5 on a daily panel, uniqueness should sit around 0.2.

    That factor is exactly how much a naive standard error overstates the evidence, and
    it is why the paper reports an *effective* sample size beside the row count.
    """
    n, h = 500, 5
    u = average_uniqueness(uniqueness_from_horizon(n, h))
    interior = u[50 : n - h - 50]
    assert np.nanmean(interior) == pytest.approx(1.0 / (h + 1), abs=0.05), (
        f"mean uniqueness {np.nanmean(interior):.3f} for horizon {h}"
    )


def test_effective_sample_size_is_far_below_the_row_count():
    n, h = 1000, 5
    u = average_uniqueness(uniqueness_from_horizon(n, h))
    ess = effective_sample_size(u)
    assert ess < n / 3, (
        f"effective sample size {ess:.0f} out of {n} rows; if this were near n the "
        "overlap correction would be doing nothing"
    )


def test_concurrency_counts_live_labels_correctly():
    # Labels [0,2], [1,3], [2,4] -> period 2 has all three live.
    t1 = np.array([2.0, 3.0, 4.0])
    conc = concurrency(t1, n_periods=5)
    assert conc[0] == 1
    assert conc[1] == 2
    assert conc[2] == 3
    assert conc[4] == 1


def test_sequential_bootstrap_raises_average_uniqueness():
    """The point of the scheme: a plain bootstrap oversamples redundant observations."""
    n, h = 300, 5
    t1 = uniqueness_from_horizon(n, h)

    picks = sequential_bootstrap(t1, size=120, seed=0)
    assert picks.size > 0

    gaps = np.diff(np.sort(np.unique(picks)))
    assert float(np.mean(gaps)) > 1.5, (
        "sequentially bootstrapped draws should be spread out rather than clustered; "
        f"mean gap {np.mean(gaps):.2f}"
    )


# --------------------------------------------------------------------------------------
# Panel window gathering
# --------------------------------------------------------------------------------------
def test_vectorised_windows_match_the_scalar_slicer():
    """The fast path must produce exactly what the readable one does.

    Window gathering is the hot loop of training, so it is vectorised; that is only
    safe if it is bit-identical to the per-row slice it replaces.
    """
    from finsent.data.panel import PanelDataset
    from finsent.data.features_causal import FEATURE_NAMES

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2019-01-01", periods=400)
    tickers = [f"T{i}" for i in range(6)]
    rows = []
    for t in tickers:
        rows.append(pd.DataFrame({
            "date": dates, "ticker": t,
            **{f: rng.standard_normal(len(dates)) for f in FEATURE_NAMES},
        }))
    feats = pd.concat(rows, ignore_index=True)
    labs = feats[["date", "ticker"]].copy()
    labs["y_dir"] = rng.integers(0, 3, len(labs))
    labs["y_ret"] = rng.standard_normal(len(labs)) * 0.01
    labs["weight"] = 1.0

    ds = PanelDataset(feats, labs, lookback=60)
    idx = ds.index.iloc[np.sort(rng.choice(len(ds), 200, replace=False))]
    dp = idx["date_pos"].to_numpy(dtype=int)
    tp = idx["ticker_pos"].to_numpy(dtype=int)

    fast = ds.windows(dp, tp)
    slow = np.stack([ds.window(int(a), int(b)) for a, b in zip(dp, tp)]).astype(np.float32)

    assert fast.shape == slow.shape == (200, 60, len(FEATURE_NAMES))
    assert np.array_equal(fast, slow), "vectorised gather diverges from the scalar slicer"


def test_windows_never_reach_forward():
    """Perturbing a bar must not change any window that ends before it."""
    from finsent.data.panel import PanelDataset
    from finsent.data.features_causal import FEATURE_NAMES

    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2019-01-01", periods=300)
    feats = pd.DataFrame({
        "date": dates, "ticker": "AAA",
        **{f: rng.standard_normal(len(dates)) for f in FEATURE_NAMES},
    })
    labs = feats[["date", "ticker"]].copy()
    labs["y_dir"] = 1
    labs["y_ret"] = 0.0
    labs["weight"] = 1.0

    ds = PanelDataset(feats.copy(), labs, lookback=60)
    base = ds.windows(np.array([150]), np.array([0]))

    shocked = feats.copy()
    shocked.loc[shocked.index[151:], FEATURE_NAMES] += 99.0
    ds2 = PanelDataset(shocked, labs, lookback=60)
    after = ds2.windows(np.array([150]), np.array([0]))

    assert np.array_equal(base, after), "a window ending at t changed when t+1 changed"
