"""Significance machinery: correct size under the null, power under a real effect.

A test that never rejects is useless and a test that always rejects is worse. Both
directions are checked here, because the whole point of this module is to stop the paper
from reporting "57.4 beats 54.8" as a finding.
"""

from __future__ import annotations

import numpy as np
import pytest

from finsent.eval import metrics as M
from finsent.eval import stats as S


def test_newey_west_lrv_matches_the_iid_variance_when_there_is_no_autocorrelation():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    lrv = S.newey_west_lrv(x, lags=0)
    assert lrv == pytest.approx(float(np.var(x)), rel=1e-9)


def test_newey_west_lrv_exceeds_the_iid_variance_under_positive_autocorrelation():
    rng = np.random.default_rng(1)
    n = 5000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.standard_normal()

    naive = float(np.var(x))
    lrv = S.newey_west_lrv(x, lags=20)
    assert lrv > 1.5 * naive, (
        f"HAC variance {lrv:.3f} should be well above the naive {naive:.3f} for an "
        "AR(1) with rho=0.6"
    )


def test_diebold_mariano_has_correct_size_under_the_null():
    """Two identical forecasters must not be declared different more than ~5% of the time."""
    rejections = 0
    trials = 300
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        loss_a = rng.standard_normal(400) ** 2
        loss_b = rng.standard_normal(400) ** 2
        if S.diebold_mariano(loss_a, loss_b).p_value < 0.05:
            rejections += 1

    rate = rejections / trials
    assert rate < 0.12, f"empirical size {rate:.3f} at a nominal 5% level"


def test_diebold_mariano_detects_a_genuinely_better_forecaster():
    rng = np.random.default_rng(2)
    truth = rng.standard_normal(600)
    good = truth + rng.standard_normal(600) * 0.5
    bad = truth + rng.standard_normal(600) * 1.5

    result = S.diebold_mariano((truth - good) ** 2, (truth - bad) ** 2, labels=("good", "bad"))
    assert result.p_value < 0.01
    assert result.better == "good"
    assert result.statistic < 0, "a negative statistic must favour model A"


def test_stationary_bootstrap_indices_have_the_right_shape_and_range():
    idx = S.stationary_bootstrap_indices(n=100, block_mean=10, n_resamples=50)
    assert idx.shape == (50, 100)
    assert idx.min() >= 0 and idx.max() < 100


def test_stationary_bootstrap_preserves_serial_dependence():
    """An i.i.d. bootstrap would destroy autocorrelation; the block version must not."""
    rng = np.random.default_rng(3)
    n = 2000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + rng.standard_normal()

    idx = S.stationary_bootstrap_indices(n, block_mean=25, n_resamples=40, rng=rng)
    acfs = []
    for row in idx:
        s = x[row]
        acfs.append(float(np.corrcoef(s[:-1], s[1:])[0, 1]))

    assert float(np.mean(acfs)) > 0.4, (
        f"resampled lag-1 autocorrelation {np.mean(acfs):.3f} is far below the original; "
        "the block structure is not being preserved"
    )


def test_paired_bootstrap_detects_a_sizing_improvement_that_unpaired_would_miss():
    """The reason the sizing experiment is paired.

    Two strategies share most of their variation and differ by a small consistent
    amount. The paired comparison should find it; an unpaired comparison of the two
    Sharpe ratios' confidence intervals would not.
    """
    rng = np.random.default_rng(4)
    common = rng.standard_normal(1500) * 0.01
    a = common + 0.0004
    b = common

    paired = S.paired_bootstrap_diff(
        a, b, lambda x: M.sharpe_ratio(x), block_mean=10, n_resamples=1000
    )
    assert paired["diff"] > 0
    assert paired["p_value"] < 0.05, f"paired p={paired['p_value']:.3f}"
    assert paired["lo"] > 0, "the paired confidence interval should exclude zero"

    ci_a = S.block_bootstrap_ci(a, lambda x: M.sharpe_ratio(x), n_resamples=1000)
    ci_b = S.block_bootstrap_ci(b, lambda x: M.sharpe_ratio(x), n_resamples=1000)
    assert ci_a["lo"] < ci_b["hi"], (
        "the unpaired intervals overlap, which is exactly why the paper pairs them"
    )


def test_whites_reality_check_controls_for_the_best_of_many_null_models():
    """Twenty coin-flip models: the best one looks good, and the test must not be fooled."""
    rng = np.random.default_rng(5)
    n = 500
    bench = rng.standard_normal(n) ** 2
    models = rng.standard_normal((20, n)) ** 2

    result = S.whites_reality_check(bench, models, n_resamples=500)
    assert result["p_value"] > 0.05, (
        f"Reality Check rejected at p={result['p_value']:.3f} on pure noise"
    )


def test_hansen_spa_detects_a_genuinely_superior_model_in_a_zoo():
    rng = np.random.default_rng(6)
    n = 800
    truth = rng.standard_normal(n)
    bench = (truth - rng.standard_normal(n) * 1.5) ** 2

    models = np.stack(
        [(truth - rng.standard_normal(n) * 1.5) ** 2 for _ in range(9)]
        + [(truth - rng.standard_normal(n) * 0.4) ** 2]
    )

    result = S.hansen_spa(bench, models, n_resamples=500)
    assert result["p_value"] < 0.10, f"SPA missed a clearly superior model, p={result['p_value']}"
    assert result["best_model"] == 9


def test_hansen_spa_does_not_reject_on_pure_noise():
    rng = np.random.default_rng(7)
    n = 600
    bench = rng.standard_normal(n) ** 2
    models = rng.standard_normal((15, n)) ** 2
    assert S.hansen_spa(bench, models, n_resamples=500)["p_value"] > 0.05
