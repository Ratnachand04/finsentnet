"""Metrics must recover known quantities, not merely run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsent.data.synthetic import make_signal_panel
from finsent.eval import metrics as M


def test_information_coefficient_recovers_a_planted_value():
    panel = make_signal_panel(n_dates=600, n_names=80, seed=0, target_ic=0.05)
    ic = M.information_coefficient(panel["score"], panel["fwd_ret"], panel["date"])
    summary = M.ic_summary(ic)

    assert summary.mean == pytest.approx(0.05, abs=0.015), (
        f"planted IC 0.05, recovered {summary.mean:.4f}"
    )
    assert summary.n_periods > 500
    assert summary.t_stat > 3.0, "a genuine 0.05 IC over 600 days should be significant"


def test_information_coefficient_is_near_zero_on_a_null_signal():
    panel = make_signal_panel(n_dates=600, n_names=80, seed=1, target_ic=0.0)
    summary = M.ic_summary(
        M.information_coefficient(panel["score"], panel["fwd_ret"], panel["date"])
    )
    assert summary.n_periods > 500, "a varying null score must still yield an IC series"
    assert abs(summary.mean) < 0.02
    assert abs(summary.t_stat) < 3.0, "a null signal must not appear significant"


def test_constant_score_yields_no_ic_rather_than_a_spurious_zero():
    """A degenerate score has undefined cross-sectional correlation.

    Returning an empty series is the honest behaviour: reporting 0.0 would imply the
    quantity was measured, when in fact it was not defined.
    """
    dates = pd.bdate_range("2020-01-01", periods=50).repeat(20)
    ic = M.information_coefficient(
        pd.Series(np.ones(len(dates))),
        pd.Series(np.random.default_rng(0).standard_normal(len(dates))),
        dates,
    )
    assert ic.empty
    assert M.ic_summary(ic).n_periods == 0


def test_hac_tstat_is_more_conservative_than_the_naive_one():
    """IC is autocorrelated; a plain t-statistic overstates significance."""
    rng = np.random.default_rng(0)
    n = 800
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.standard_normal() * 0.01
    x += 0.004

    naive = float(x.mean() / (x.std(ddof=1) / np.sqrt(n)))
    summary = M.ic_summary(pd.Series(x), hac_lags=10)
    assert abs(summary.t_stat) < abs(naive), (
        f"HAC t={summary.t_stat:.2f} should be smaller than naive t={naive:.2f} under "
        "positive autocorrelation"
    )


def test_quantile_spread_is_monotone_for_a_real_signal():
    panel = make_signal_panel(n_dates=400, n_names=100, seed=2, target_ic=0.08)
    spread, per_quantile = M.quantile_spread(
        panel["score"], panel["fwd_ret"], panel["date"], n_quantiles=5
    )
    means = per_quantile.mean()
    assert spread.mean() > 0
    assert means.iloc[-1] > means.iloc[0], "top quantile must out-return the bottom"
    increasing = sum(means.iloc[i] < means.iloc[i + 1] for i in range(len(means) - 1))
    assert increasing >= 3, f"quantile means are not broadly monotone: {means.to_dict()}"


def test_classification_metrics_against_a_hand_built_confusion_matrix():
    y_true = np.array([0, 0, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 2])

    m = M.classification_metrics(y_true, y_pred)
    assert m["n"] == 7
    assert m["accuracy"] == pytest.approx(5 / 7)
    assert m["majority_baseline"] == pytest.approx(3 / 7)

    cm = np.array(m["confusion_matrix"])
    assert cm.sum() == 7
    assert cm[0, 0] == 1 and cm[0, 1] == 1
    assert cm[1, 1] == 2 and cm[1, 2] == 1
    assert cm[2, 2] == 2

    # Directional accuracy excludes true NEUTRAL rows: {0,0,2,2} vs {0,1,2,2}.
    assert m["binary_accuracy"] == pytest.approx(3 / 4)


def test_mcc_is_zero_for_a_constant_predictor():
    y_true = np.array([0, 1, 2] * 30)
    y_pred = np.ones_like(y_true)
    assert M.classification_metrics(y_true, y_pred)["mcc"] == pytest.approx(0.0, abs=1e-9)


def test_mcc_is_one_for_a_perfect_predictor():
    y = np.array([0, 1, 2] * 30)
    assert M.classification_metrics(y, y)["mcc"] == pytest.approx(1.0)


def test_sharpe_ratio_matches_a_hand_computed_value():
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert M.sharpe_ratio(r) == pytest.approx(expected)


def test_max_drawdown_matches_a_hand_computed_path():
    # 1.00 -> 1.10 -> 0.88 -> 0.968 ; trough at 0.88 against a peak of 1.10
    r = np.array([0.10, -0.20, 0.10])
    assert M.max_drawdown(r) == pytest.approx(0.20, abs=1e-9)


def test_turnover_counts_both_legs():
    w = pd.DataFrame([[0.5, -0.5], [0.0, 0.0], [0.5, -0.5]]).to_numpy()
    # Each transition moves 1.0 of gross exposure.
    assert M.turnover(w) == pytest.approx(1.0)


def test_ece_bins_partition_the_sample():
    panel = make_signal_panel(n_dates=200, n_names=50, seed=3)
    probs = panel[["p_down", "p_neutral", "p_up"]].to_numpy()
    stats = M.expected_calibration_error(probs, panel["y_true"].to_numpy(), n_bins=10)

    assert stats["bins"]["count"].sum() == len(panel)
    assert 0.0 <= stats["ece"] <= 1.0
    assert stats["mce"] >= stats["ece"], "max gap cannot be below the weighted mean gap"


def test_performance_summary_reports_every_field_table_six_needs():
    rng = np.random.default_rng(4)
    r = rng.standard_normal(500) * 0.01
    summary = M.performance_summary(r, weights=np.tile([0.5, -0.5], (500, 1)))
    for key in ("sharpe", "sortino", "max_drawdown", "calmar", "hit_rate", "turnover"):
        assert key in summary, f"performance_summary is missing {key}"
