"""The backtest's trading rule and the report's completeness.

The most important assertion here is ``test_backtest_cannot_see_the_return_it_trades``:
a backtester that shifts returns internally is one refactor away from shifting them the
wrong way, and nobody would notice except through an implausibly good Sharpe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finsent.config import load_config
from finsent.data.synthetic import make_signal_panel
from finsent.decision.costs import CostModel
from finsent.eval import metrics as M
from finsent.eval.backtest import TradingRule, build_weights, cost_sweep, run_backtest
from finsent.eval.report import ReportBuilder, make_random_signals


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def panel():
    return make_signal_panel(n_dates=300, n_names=40, seed=0, target_ic=0.06)


@pytest.fixture(scope="module")
def forward(panel):
    return panel.pivot_table(
        index="date", columns="ticker", values="fwd_ret", aggfunc="mean"
    ).sort_index()


# --------------------------------------------------------------------------------------
# Trading rule
# --------------------------------------------------------------------------------------
def test_trading_rule_describes_itself_completely():
    """A backtest whose rule is unstated is unreviewable; this is the stated rule."""
    text = TradingRule(rebalance_days=5).describe()
    for token in ("Rebalance every 5", "top/bottom", "NEUTRAL", "leverage", "costs"):
        assert token.lower() in text.lower(), f"the rule description omits {token!r}"


def test_weights_are_neutral_capped_and_held_between_rebalances(panel):
    rule = TradingRule(scheme="decile_long_short", quantile=0.2, rebalance_days=5,
                       gate_on_singleton=False, f_max=0.05, gross_leverage_max=1.0)
    w = build_weights(panel, rule)

    assert np.allclose(w.sum(axis=1), 0.0, atol=1e-9), "weights are not dollar neutral"
    assert (w.abs() <= 0.05 + 1e-12).all().all()
    assert (w.abs().sum(axis=1) <= 1.0 + 1e-9).all()

    changed = (w.diff().abs().sum(axis=1) > 1e-12).to_numpy()[1:]
    assert changed.mean() < 0.35, (
        f"weights changed on {changed.mean():.0%} of sessions under a 5-day rebalance; "
        "they are not being held"
    )


def test_conformal_gate_reduces_the_number_of_positions(panel):
    gated = TradingRule(scheme="kelly", rebalance_days=5, gate_on_singleton=True)
    ungated = TradingRule(scheme="kelly", rebalance_days=5, gate_on_singleton=False)

    n_gated = int((build_weights(panel, gated) != 0).to_numpy().sum())
    n_ungated = int((build_weights(panel, ungated) != 0).to_numpy().sum())
    assert n_gated < n_ungated, "the conformal gate is not gating anything"


def test_kelly_scheme_requires_a_variance_column(panel):
    rule = TradingRule(scheme="kelly")
    with pytest.raises(ValueError, match="sigma2"):
        build_weights(panel.drop(columns=["sigma2"]), rule)


def test_backtest_cannot_see_the_return_it_trades(panel, forward):
    """The signal frame handed to the weight builder must contain no realised return.

    Enforced structurally: ``build_weights`` reads only ``date``, ``ticker``, ``score``,
    ``sigma2`` and ``tradeable``, so deleting the return column cannot change a weight.
    """
    rule = TradingRule(scheme="decile_long_short", rebalance_days=5, gate_on_singleton=False)

    with_returns = build_weights(panel, rule)
    without = build_weights(panel.drop(columns=["fwd_ret", "y_true"]), rule)

    pd.testing.assert_frame_equal(with_returns, without)


def test_a_real_signal_makes_money_gross_and_a_null_one_does_not():
    rule = TradingRule(scheme="decile_long_short", rebalance_days=5, gate_on_singleton=False)

    real = make_signal_panel(n_dates=500, n_names=60, seed=1, target_ic=0.10)
    null = make_signal_panel(n_dates=500, n_names=60, seed=2, target_ic=0.0)

    results = []
    for p in (real, null):
        fwd = p.pivot_table(index="date", columns="ticker", values="fwd_ret",
                            aggfunc="mean").sort_index()
        results.append(run_backtest(build_weights(p, rule), fwd, rule, CostModel(0.0)))

    assert results[0].summary_gross["sharpe"] > results[1].summary_gross["sharpe"]
    assert results[0].summary_gross["sharpe"] > 0.5
    assert abs(results[1].summary_gross["sharpe"]) < 1.0, (
        "a null signal produced a large Sharpe; the backtest is manufacturing returns"
    )


def test_costs_monotonically_reduce_net_sharpe(panel, forward):
    rule = TradingRule(scheme="decile_long_short", rebalance_days=5, gate_on_singleton=False)
    sweep = cost_sweep(build_weights(panel, rule), forward, rule,
                       bps_levels=(0.0, 5.0, 10.0, 20.0, 40.0))

    sharpes = sweep["sharpe_net"].to_numpy()
    assert np.all(np.diff(sharpes) <= 1e-9), f"net Sharpe must fall with cost: {sharpes}"
    assert sweep["turnover_annualised"].nunique() == 1, "turnover must not depend on cost"


def test_break_even_cost_is_reported(panel, forward):
    rule = TradingRule(scheme="decile_long_short", rebalance_days=5, gate_on_singleton=False)
    result = run_backtest(build_weights(panel, rule), forward, rule, CostModel(10.0))
    assert np.isfinite(result.break_even_bps)
    assert result.turnover > 0


def test_daily_rebalancing_costs_more_than_weekly(panel, forward):
    """The arithmetic behind SPEC.md's choice of a five-day primary horizon."""
    daily = TradingRule(rebalance_days=1, gate_on_singleton=False)
    weekly = TradingRule(rebalance_days=5, gate_on_singleton=False)

    r_daily = run_backtest(build_weights(panel, daily), forward, daily, CostModel(10.0))
    r_weekly = run_backtest(build_weights(panel, weekly), forward, weekly, CostModel(10.0))

    assert r_daily.turnover > r_weekly.turnover * 1.5, (
        f"daily turnover {r_daily.turnover:.3f} vs weekly {r_weekly.turnover:.3f}"
    )
    assert r_daily.costs.mean() > r_weekly.costs.mean()


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def test_report_emits_every_table_and_figure(cfg, tmp_path: Path):
    """The Phase-1 acceptance criterion: the full artifact set renders on noise."""
    noise = make_random_signals(n_dates=180, n_names=30, seed=0, signal_strength=0.0)
    builder = ReportBuilder(cfg, noise, tmp_path, seeds=(0,), label="test")
    tables = builder.build_all()

    assert set(tables) == {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"}
    for tid in tables:
        assert (tmp_path / "tables" / f"{tid}_test.csv").exists()
        assert (tmp_path / "tables" / f"{tid}_test.tex").exists()

    figures = list((tmp_path / "figures").glob("*.png"))
    assert len(figures) >= 6, f"only {len(figures)} figures rendered"
    assert (tmp_path / "manifest_test.json").exists()


def test_every_table_carries_a_provenance_footnote(cfg, tmp_path: Path):
    """SPEC.md 6.1: no number reaches the manuscript without its provenance."""
    noise = make_random_signals(n_dates=120, n_names=25, seed=1)
    builder = ReportBuilder(cfg, noise, tmp_path, seeds=(1,), label="prov")
    builder.build_all()

    tex = (tmp_path / "tables" / "T3_prov.tex").read_text(encoding="utf-8")
    assert "Provenance" in tex
    assert builder.provenance.config_hash in tex
    assert builder.provenance.data_hash in tex


def test_sizing_table_compares_four_rules_on_identical_predictions(cfg, tmp_path: Path):
    """The headline experiment must vary only the sizing rule."""
    panel = make_signal_panel(n_dates=260, n_names=40, seed=3, target_ic=0.06)
    builder = ReportBuilder(cfg, panel, tmp_path, seeds=(3,), label="sizing")
    t5 = builder.table_t5_sizing()

    assert list(t5["method"]) == [
        "equal_weight", "raw_softmax_kelly", "calibrated_kelly", "conformal_gated_kelly"
    ]
    assert "paired_p_vs_equal_weight" in t5.columns
    assert np.isnan(t5.loc[0, "paired_p_vs_equal_weight"]), "the benchmark has no p-value"
    assert t5["turnover_annualised"].notna().all()


def test_economic_table_declares_the_trial_budget(cfg, tmp_path: Path):
    noise = make_random_signals(n_dates=300, n_names=30, seed=4)
    builder = ReportBuilder(cfg, noise, tmp_path, seeds=(4,), label="econ")
    t6 = builder.table_t6_economic()

    assert (t6["n_trials_declared"] == cfg.eval.n_configs_evaluated).all()
    assert "deflated_sharpe" in t6.columns and "dsr_verdict" in t6.columns
    assert set(t6["cost_bps"]) == set(cfg.decision.sweep_bps)
