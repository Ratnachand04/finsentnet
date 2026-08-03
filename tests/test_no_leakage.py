"""The most important tests in the repository.

There are **two distinct kinds of leakage**, they have different causes, and only one of
them is a splitter's job to prevent. Conflating them is why so many papers claim
"we used a temporal split" and still report impossible numbers.

Kind 1 — feature contamination
    A feature at ``t`` reads a price after ``t``. No splitter can fix this: the
    contaminated value travels inside the row, so it is present in train and test alike
    and the relationship generalises perfectly. This is caught by the *causality*
    contract on the feature builder (``test_causality.py``), and the control test below
    demonstrates that purging does nothing about it.

Kind 2 — overlapping-label contamination
    Features are perfectly causal, but a training sample's *label* is computed from a
    price window that reaches into the evaluation block. This is a boundary effect: with
    horizon ``h``, only the ``h`` samples either side of the boundary are affected. It is
    exactly what purging and embargoing exist to remove, and it is what these tests
    measure.

The quantitative test below measures the leak the way it actually works: for each test
sample, take the label of the nearest available *training* sample and correlate it with
the truth. Without purging, the nearest training label shares ``h-1`` of its ``h`` return
terms with the test label; with purging it shares none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsent.data.splits import PurgedWalkForward, purge_indices, embargo_indices
from finsent.data.synthetic import make_leaky_panel

HORIZON = 5


# --------------------------------------------------------------------------------------
# Kind 2: overlapping labels — the leak purging is responsible for
# --------------------------------------------------------------------------------------
def _overlapping_labels(n: int, horizon: int, seed: int = 0) -> np.ndarray:
    """``label[t] = sum(r[t+1 : t+1+h])`` over i.i.d. returns.

    Adjacent labels share ``h-1`` of their ``h`` terms, which is precisely the structure
    that makes a naive temporal split leak at the boundary.
    """
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n + horizon + 1)
    return np.array([r[t + 1 : t + 1 + horizon].sum() for t in range(n)])


def _nearest_neighbour_leak_r2(
    labels: np.ndarray, train_hi: int, test_lo: int, test_hi: int
) -> float:
    """R^2 from predicting each test label with the nearest available training label.

    The nearest training sample is always ``train_hi`` for every test position, because
    training stops there. This is the sharpest possible probe of boundary contamination
    and needs no fitted model at all.
    """
    truth = labels[test_lo:test_hi]
    pred = np.full(truth.shape, labels[train_hi])
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def test_control_a_naive_boundary_leaks_overlapping_labels():
    """Control: without purging, the label immediately before the test block leaks."""
    n = 4000
    labels = _overlapping_labels(n, HORIZON, seed=1)
    test_lo = 3000

    corr = np.corrcoef(labels[test_lo : test_lo + HORIZON], labels[test_lo - 1])[0, 1] \
        if False else None
    _ = corr

    # Correlation between the last training label and each of the first h test labels.
    last_train = labels[test_lo - 1]
    overlaps = []
    for j in range(HORIZON):
        window = labels[test_lo + j]
        overlaps.append((last_train, window))

    # Measured over many independent draws so the boundary correlation is estimable.
    sims = []
    for seed in range(200):
        lab = _overlapping_labels(n, HORIZON, seed=seed)
        sims.append((lab[test_lo - 1], lab[test_lo]))
    a, b = np.array(sims).T
    boundary_corr = float(np.corrcoef(a, b)[0, 1])

    expected = (HORIZON - 1) / HORIZON
    assert boundary_corr > 0.5, (
        f"the synthetic labels do not actually overlap (corr={boundary_corr:.3f}); "
        "the test harness would then prove nothing"
    )
    assert abs(boundary_corr - expected) < 0.15, (
        f"boundary correlation {boundary_corr:.3f} should be near (h-1)/h = {expected:.2f}"
    )
    _ = overlaps


def test_purging_removes_the_boundary_correlation():
    """The real assertion: after purging, the last training label shares no terms."""
    n = 4000
    test_lo = 3000

    naive, purged = [], []
    for seed in range(200):
        lab = _overlapping_labels(n, HORIZON, seed=seed)
        naive.append((lab[test_lo - 1], lab[test_lo]))
        purged.append((lab[test_lo - 1 - HORIZON], lab[test_lo]))

    naive_corr = float(np.corrcoef(*np.array(naive).T)[0, 1])
    purged_corr = float(np.corrcoef(*np.array(purged).T)[0, 1])

    assert naive_corr > 0.5, "control failed: no leak to remove"
    assert abs(purged_corr) < 0.15, (
        f"after purging {HORIZON} sessions the boundary correlation is "
        f"{purged_corr:.3f}; it should be indistinguishable from zero"
    )


def test_walk_forward_folds_leave_no_overlapping_label_windows():
    """Structural guarantee, checked exhaustively rather than argued.

    For every fold, every training sample's label window ``[p, p+h]`` must end strictly
    before the first evaluation position. This is what "purged" means, and it is a
    property that can be verified exactly.
    """
    dates = pd.bdate_range("2016-01-04", periods=2400)
    splitter = PurgedWalkForward(dates=dates, horizon=HORIZON, embargo_pct=0.01)
    folds = splitter.folds()
    assert folds, "splitter produced no folds"

    for fold in folds:
        train_positions = np.arange(*fold.train)
        val_lo = fold.inner_val[0]
        test_lo = fold.test[0]

        assert (train_positions + fold.purge_days < val_lo).all(), (
            f"fold {fold.index}: a training label window reaches into the inner-val block"
        )
        assert (train_positions + fold.purge_days < test_lo).all(), (
            f"fold {fold.index}: a training label window reaches into the test block"
        )
        val_positions = np.arange(*fold.inner_val)
        assert (val_positions + fold.purge_days < test_lo).all(), (
            f"fold {fold.index}: an inner-val label window reaches into the test block"
        )
        assert test_lo - fold.inner_val[1] >= fold.purge_days + fold.embargo_days, (
            f"fold {fold.index}: embargo not applied before the test block"
        )


def test_purge_removes_exactly_the_overlapping_positions():
    """Boundary arithmetic: a sample at p is contaminated iff p + h >= evaluation start."""
    train = np.arange(0, 100)
    test = np.arange(100, 150)

    kept = purge_indices(train, test, horizon=5)
    assert kept.max() == 94, f"expected last kept position 94, got {kept.max()}"
    assert 95 not in kept, "position 95 has a label window reaching position 100"


def test_embargo_removes_positions_after_the_evaluation_block():
    train = np.concatenate([np.arange(0, 100), np.arange(150, 200)])
    test = np.arange(100, 150)

    kept = embargo_indices(train, test, embargo=10)
    after = kept[kept > 149]
    assert after.min() == 160, (
        f"embargo of 10 should resume training at 160, got {after.min()}"
    )


def test_no_fold_overlaps_its_own_blocks():
    dates = pd.bdate_range("2016-01-04", periods=2400)
    splitter = PurgedWalkForward(dates=dates, horizon=5, embargo_pct=0.01)

    for fold in splitter:
        tr = set(range(*fold.train))
        va = set(range(*fold.inner_val))
        te = set(range(*fold.test))
        assert not tr & va and not tr & te and not va & te, f"fold {fold.index} overlaps"
        assert max(tr) < min(va) < max(va) < min(te), f"fold {fold.index} is out of order"


def test_configured_protocol_produces_the_documented_number_of_folds():
    """SPEC.md 2.5 documents folds "in the teens". Verify, do not assume."""
    from finsent.config import load_config

    cfg = load_config()
    dates = pd.bdate_range(cfg.data.start, cfg.data.end)
    splitter = PurgedWalkForward(
        dates=dates,
        horizon=cfg.data.horizon_h,
        embargo_pct=cfg.data.embargo_pct,
        train_min_years=cfg.data.train_min_years,
        inner_val_months=cfg.data.inner_val_months,
        test_months=cfg.data.test_months,
        refit_every_months=cfg.data.refit_every_months,
        train_mode=cfg.data.train_mode,
    )
    n = splitter.n_folds()
    assert 8 <= n <= 20, (
        f"protocol produced {n} folds; SPEC.md documents a number in the teens. "
        "If this is intentional, update SPEC.md rather than this test."
    )


# --------------------------------------------------------------------------------------
# Kind 1: feature contamination — documented here precisely because purging cannot fix it
# --------------------------------------------------------------------------------------
def _ridge_r2(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> float:
    tr = train.dropna(subset=features + ["fwd_ret"])
    te = test.dropna(subset=features + ["fwd_ret"])
    if len(tr) < 50 or len(te) < 50:
        return np.nan
    X = np.column_stack([np.ones(len(tr)), tr[features].to_numpy(dtype=float)])
    y = tr["fwd_ret"].to_numpy(dtype=float)
    beta = np.linalg.solve(X.T @ X + 1e-8 * np.eye(X.shape[1]), X.T @ y)
    Xt = np.column_stack([np.ones(len(te)), te[features].to_numpy(dtype=float)])
    yt = te["fwd_ret"].to_numpy(dtype=float)
    ss_res = float(((yt - Xt @ beta) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def test_a_non_causal_feature_survives_purging_which_is_the_point():
    """A contaminated *feature* is immune to any splitter. Demonstrated, not asserted.

    This is why the repository has a separate causality contract on the feature builder.
    A reader who sees only "we used purged walk-forward" and concludes the pipeline is
    safe has drawn the wrong conclusion, and this test exists to make that explicit.
    """
    panel = make_leaky_panel(n_names=8, n_days=800, horizon=HORIZON, seed=7)
    features = ["leak", "noise_0", "noise_1"]

    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    splitter = PurgedWalkForward(
        dates=dates, horizon=HORIZON, embargo_pct=0.01, train_min_years=1,
        inner_val_months=3, test_months=3, refit_every_months=3,
    )

    scores = [
        _ridge_r2(train, test, features)
        for _fold, train, _val, test in splitter.split_frame(panel)
    ]
    scores = [s for s in scores if np.isfinite(s)]
    assert scores, "no fold produced a usable probe"
    assert float(np.mean(scores)) > 0.90, (
        "a perfectly contaminated feature should still be learnable after purging; "
        "if this fails, the demonstration no longer makes its point"
    )


def test_a_causal_feature_set_is_not_predictive_of_pure_noise():
    """Sanity floor: with only noise features, no split should find signal."""
    panel = make_leaky_panel(n_names=8, n_days=800, horizon=HORIZON, seed=11)
    features = ["noise_0", "noise_1", "noise_2", "noise_3"]

    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    splitter = PurgedWalkForward(
        dates=dates, horizon=HORIZON, embargo_pct=0.01, train_min_years=1,
        inner_val_months=3, test_months=3, refit_every_months=3,
    )
    scores = [
        _ridge_r2(train, test, features)
        for _fold, train, _val, test in splitter.split_frame(panel)
    ]
    scores = [s for s in scores if np.isfinite(s)]
    assert scores
    assert float(np.mean(scores)) < 0.02, (
        f"noise features produced mean out-of-sample R^2={np.mean(scores):.4f}; "
        "something in the pipeline is manufacturing signal"
    )
