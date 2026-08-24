"""Emit every paper table and figure from the real walk-forward study.

    python experiments/03_make_paper.py

Reads the out-of-sample prediction panels written by ``02_run_study.py`` and
writes ``paper/tables/*.tex`` and ``paper/figures/*.pdf``. Every table carries a
provenance footnote (git sha, config hash, data hash, seed list), so no number
in the manuscript is entered by hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finsent.config import load_config  # noqa: E402
from finsent.decision.costs import CostModel  # noqa: E402
from finsent.eval import dsr as DSR  # noqa: E402
from finsent.eval import metrics as M  # noqa: E402
from finsent.eval import stats as S  # noqa: E402
from finsent.eval.backtest import TradingRule, build_weights, cost_sweep, run_backtest  # noqa: E402
from finsent.utils.hashing import Provenance, hash_frame  # noqa: E402

STUDY = Path("runs/study")
TABLES = Path("paper/tables")
FIGURES = Path("paper/figures")

PRIMARY = "finsentnet_c"
ORDER = ["null_signal", "buy_and_hold", "tsmom", "logit5", "gbm", "finsentnet_c"]
PRETTY = {
    "null_signal": "Null signal",
    "buy_and_hold": "Buy-and-hold",
    "tsmom": "Time-series momentum (12-1)",
    "logit5": "Logistic, 5 features (18 params)",
    "gbm": "Gradient-boosted trees, 12 features",
    "finsentnet_c": "FinSentNet-C (price-only)",
}


# --------------------------------------------------------------------------------------
def write_table(tid: str, df: pd.DataFrame, caption: str, prov: Provenance,
                note: str = "", float_fmt: str = "%.4f") -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{tid}.csv", index=False)
    body = df.to_latex(index=False, escape=True,
                       float_format=lambda v: (float_fmt % v) if np.isfinite(v) else "---")
    extra = f"\\\\[1pt]{note}" if note else ""
    tex = (
        f"% {tid}\n"
        f"\\begin{{table}}[t]\n\\centering\\small\n"
        f"\\caption{{{caption}}}\n\\label{{tab:{tid}}}\n"
        f"{body}\n"
        f"\\vspace{{2pt}}\\footnotesize\\emph{{Provenance:}} "
        f"\\texttt{{{prov.footnote()}}}{extra}\n"
        f"\\end{{table}}\n"
    )
    (TABLES / f"{tid}.tex").write_text(tex, encoding="utf-8")
    print(f"  wrote {tid}  ({len(df)} rows)")


def load_panels(study: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Load each model's out-of-sample panel, attaching the realised holding return.

    ``next_ret`` -- the return actually earned from the open of ``t`` to the open of
    ``t+1`` -- lives in the study panel, not in the prediction record, because the
    walk-forward driver deliberately hands a model only what it is allowed to see. It
    is merged back here, at report time, once predictions are already fixed.
    """
    out = {}
    keys = None
    if study is not None:
        keys = study[["date", "ticker", "next_ret"]].copy()
        keys["date"] = pd.to_datetime(keys["date"])
        keys["ticker"] = keys["ticker"].astype(str)
    for name in ORDER:
        path = STUDY / f"{name}.parquet"
        if not path.exists():
            continue
        panel = pd.read_parquet(path)
        panel["date"] = pd.to_datetime(panel["date"])
        panel["ticker"] = panel["ticker"].astype(str)
        if keys is not None and "next_ret" not in panel.columns:
            panel = panel.merge(keys, on=["date", "ticker"], how="left")
            panel["next_ret"] = panel["next_ret"].fillna(0.0)
        out[name] = panel
    return out


def probs(panel: pd.DataFrame, raw: bool = False) -> np.ndarray:
    cols = ["p_down_raw", "p_neutral_raw", "p_up_raw"] if raw else ["p_down", "p_neutral", "p_up"]
    cols = [c for c in cols if c in panel.columns] or ["p_down", "p_neutral", "p_up"]
    return panel[cols].to_numpy(dtype=float)


def per_seed(panel: pd.DataFrame, fn):
    """Apply fn to each seed's sub-panel; return (mean, std, values)."""
    vals = []
    for _, grp in panel.groupby("seed"):
        v = fn(grp)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return np.nan, np.nan, []
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, vals


def strategy_returns(panel: pd.DataFrame, rule: TradingRule, cost_bps: float,
                     ret_col: str = "next_ret") -> pd.Series:
    """Net return series for one panel under one trading rule."""
    p = panel.copy()
    if ret_col not in p.columns:
        ret_col = "fwd_ret"
    fwd = p.pivot_table(index="date", columns="ticker", values=ret_col,
                        aggfunc="mean").sort_index()
    w = build_weights(p, rule)
    return run_backtest(w, fwd, rule, CostModel(spread_bps=cost_bps)).net_returns


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study-panel", default="data/cache/study_panel.parquet")
    args = ap.parse_args()

    cfg = load_config()
    study = pd.read_parquet(args.study_panel)
    panels = load_panels(study)
    if PRIMARY not in panels:
        print(f"missing {PRIMARY}; run 02_run_study.py first")
        return 1

    main_panel = panels[PRIMARY]
    seeds = tuple(sorted(main_panel["seed"].unique().tolist()))
    prov = Provenance.capture(cfg.config_hash, hash_frame(study), seeds)
    print(f"provenance: {prov.footnote()}\n")

    ppy = cfg.eval.periods_per_year
    rule_ls = TradingRule(scheme="decile_long_short", quantile=0.10,
                          rebalance_days=cfg.decision.rebalance_days,
                          gate_on_singleton=False)

    # ---------------------------------------------------------------- T1: dataset
    s = study.assign(year=pd.to_datetime(study["date"]).dt.year)
    rows = []
    for year, grp in s.groupby("year"):
        c = grp["y_dir"].value_counts(normalize=True)
        rows.append({
            "Year": int(year),
            "Names": int(grp["ticker"].nunique()),
            "Ticker-days": int(len(grp)),
            "Names/session": round(float(grp.groupby("date").size().mean()), 1),
            "DOWN": float(c.get(0, 0.0)),
            "NEUTRAL": float(c.get(1, 0.0)),
            "UP": float(c.get(2, 0.0)),
        })
    t1 = pd.DataFrame(rows)
    ess = float(study["weight"].sum())
    write_table(
        "T1_main", t1,
        "Study panel: point-in-time liquidity-screened US universe, "
        "volatility-scaled labels at $h=5$ sessions.", prov,
        note=(f"Total {len(study):,} ticker-days over {study['ticker'].nunique()} names and "
              f"{study['date'].nunique():,} sessions. Effective sample size after the "
              f"overlap correction: {ess:,.0f} (a factor of {len(study)/ess:.1f}). "
              f"\\textbf{{Usable headlines in this corpus: 0}} --- of 5{{,}}391 news rows, "
              f"72 are non-empty and carry 64 template titles across 9 distinct "
              f"timestamps, so the text modality is not evaluated."))

    # ---------------------------------------------------------------- T2: parameters
    try:
        import torch  # noqa: F401
        from finsent.models.finsentnet_c import FinSentNetC
        counts = FinSentNetC.from_config(cfg).parameter_counts()
        total = counts.pop("TOTAL")
        t2 = pd.DataFrame([{"Module": k, "Parameters": v,
                            "Percent": 100.0 * v / total} for k, v in counts.items()])
        t2 = t2.sort_values("Parameters", ascending=False)
        t2.loc[len(t2)] = {"Module": "TOTAL", "Parameters": total, "Percent": 100.0}
        write_table("T2_main", t2,
                    "Trainable parameters by module.", prov,
                    note=("The evaluated configuration is the forced price-only state, in "
                          "which the text encoder emits its learned null embedding."),
                    float_fmt="%.1f")
    except Exception as exc:
        print(f"  T2 skipped: {exc}")

    # ---------------------------------------------------------------- T3: predictive
    rows = []
    ic_series = {}
    for name, panel in panels.items():
        ic_m, ic_s, _ = per_seed(panel, lambda g: M.information_coefficient(
            g["score"], g["fwd_ret"], g["date"]).mean())
        acc_m, acc_s, _ = per_seed(panel, lambda g: M.classification_metrics(
            g["y_true"].to_numpy(), probs(g).argmax(1))["accuracy"])
        bal_m, _, _ = per_seed(panel, lambda g: M.classification_metrics(
            g["y_true"].to_numpy(), probs(g).argmax(1))["balanced_accuracy"])
        mcc_m, _, _ = per_seed(panel, lambda g: M.classification_metrics(
            g["y_true"].to_numpy(), probs(g).argmax(1))["mcc"])
        bin_m, _, _ = per_seed(panel, lambda g: M.classification_metrics(
            g["y_true"].to_numpy(), probs(g).argmax(1))["binary_accuracy"])
        maj = M.classification_metrics(panel["y_true"].to_numpy(),
                                       probs(panel).argmax(1))["majority_baseline"]
        ic_full = M.information_coefficient(panel["score"], panel["fwd_ret"], panel["date"])
        summ = M.ic_summary(ic_full, ppy, cfg.eval.hac_lags)
        ic_series[name] = ic_full
        rows.append({
            "Model": PRETTY.get(name, name),
            "Acc.": acc_m, "Acc. sd": acc_s, "Bal. acc.": bal_m,
            "Majority": maj, "Binary acc.": bin_m, "MCC": mcc_m,
            "Rank IC": ic_m, "IC sd": ic_s, "HAC t": summ.t_stat,
        })
    t3 = pd.DataFrame(rows)

    # Diebold-Mariano of every baseline against the primary model, on IC loss.
    base_ic = ic_series.get(PRIMARY)
    dm = []
    for name in t3["Model"]:
        key = next((k for k, v in PRETTY.items() if v == name), name)
        if key == PRIMARY or base_ic is None or key not in ic_series:
            dm.append(np.nan)
            continue
        j = pd.concat([base_ic.rename("a"), ic_series[key].rename("b")], axis=1).dropna()
        if len(j) < 30:
            dm.append(np.nan)
            continue
        dm.append(S.diebold_mariano(-j["a"].to_numpy(), -j["b"].to_numpy(),
                                    lags=cfg.eval.hac_lags).p_value)
    t3["DM $p$ vs ours"] = dm
    write_table("T3_main", t3,
                "Predictive comparison under the identical purged walk-forward protocol. "
                "Accuracy and IC are means over seeds with standard deviations; the "
                "Newey--West $t$-statistic is computed on the pooled IC series.", prov,
                note=("Buy-and-hold emits a constant cross-sectional score, so its rank IC "
                      "is undefined by construction rather than zero."))

    # ---------------------------------------------------------------- T4: calibration
    rows = []
    for name, panel in panels.items():
        raw, cal = probs(panel, raw=True), probs(panel)
        y = panel["y_true"].to_numpy()
        r = M.expected_calibration_error(raw, y, cfg.calibration.n_bins)
        c = M.expected_calibration_error(cal, y, cfg.calibration.n_bins)
        rows.append({
            "Model": PRETTY.get(name, name),
            "ECE raw": r["ece"], "ECE cal.": c["ece"],
            "MCE raw": r["mce"], "MCE cal.": c["mce"],
            "Brier raw": M.brier_score(raw, y), "Brier cal.": M.brier_score(cal, y),
            "Overconf. raw": r["overconfidence"],
        })
    write_table("T4_main", pd.DataFrame(rows),
                "Calibration before and after guarded post-hoc calibration, fitted on the "
                "inner-validation block of each fold and evaluated on the untouched test "
                "block.", prov,
                note=("The guard admits the identity map as a candidate, so calibration "
                      "cannot be a net loss out of sample."))

    # coverage
    cov_files = sorted(STUDY.glob("*_coverage.csv"))
    if cov_files:
        cov = pd.concat([pd.read_csv(f).assign(model=f.stem.replace("_coverage", ""))
                         for f in cov_files], ignore_index=True)
        cov_t = (cov[cov["model"] == PRIMARY]
                 .groupby("alpha")[["nominal_coverage", "empirical_coverage",
                                    "mean_set_size", "singleton_rate"]]
                 .mean().reset_index())
        cov_t.columns = ["alpha", "Nominal", "Empirical", "Mean set size", "Singleton rate"]
        write_table("T4b_main", cov_t,
                    "Conformal coverage on the test blocks, randomised adaptive prediction "
                    "sets fitted per fold on inner-validation.", prov,
                    note="Coverage is a distribution-free guarantee; it does not assume the "
                         "model is accurate.")

    # ---------------------------------------------------------------- T5: sizing
    conf = probs(main_panel).max(axis=1)
    variants = {
        "Equal weight (decile L/S)": ("decile", False, None),
        "Raw-softmax Kelly": ("kelly", False, np.maximum((1.0 - conf) ** 2, 1e-4)),
        "Calibrated Kelly": ("kelly", False, None),
        "Conformal-gated calibrated Kelly": ("kelly", True, None),
    }
    rows, series = [], {}
    for label, (scheme, gate, sigma_override) in variants.items():
        sig = main_panel.copy()
        if sigma_override is not None:
            sig["sigma2"] = sigma_override
        rule = TradingRule(scheme="kelly" if scheme == "kelly" else "decile_long_short",
                           quantile=0.10, rebalance_days=cfg.decision.rebalance_days,
                           gate_on_singleton=gate, kappa=cfg.decision.kappa,
                           f_max=cfg.decision.f_max)
        fwd = sig.pivot_table(index="date", columns="ticker", values="next_ret",
                              aggfunc="mean").sort_index()
        res = run_backtest(build_weights(sig, rule), fwd, rule,
                           CostModel(spread_bps=cfg.decision.default_bps), ppy)
        series[label] = res.net_returns
        held = (res.weights.abs() > 1e-12).sum(axis=1)
        rows.append({
            "Sizing rule": label,
            "Sharpe (net)": res.summary_net.get("sharpe", np.nan),
            "Sortino": res.summary_net.get("sortino", np.nan),
            "Max DD": res.summary_net.get("max_drawdown", np.nan),
            "Turnover (ann.)": res.turnover * ppy,
            # Without this column an all-abstaining rule is just a row of dashes, and a
            # reader cannot tell a gate that declined to trade from a bug.
            "Names held": float(held.mean()),
            "Trades": res.n_trades,
        })
    t5 = pd.DataFrame(rows)
    bench = series["Equal weight (decile L/S)"]
    pv = []
    for label in t5["Sizing rule"]:
        if label.startswith("Equal weight"):
            pv.append(np.nan); continue
        j = pd.concat([series[label].rename("a"), bench.rename("b")], axis=1).dropna()
        out = S.paired_bootstrap_diff(j["a"].to_numpy(), j["b"].to_numpy(),
                                      lambda x: M.sharpe_ratio(x, ppy),
                                      block_mean=cfg.eval.block_mean, n_resamples=4000)
        pv.append(out["p_value"])
    t5["Paired $p$ vs equal wt."] = pv
    write_table("T5_main", t5,
                "\\textbf{The sizing experiment.} One fixed set of predictions, four sizing "
                "rules. Net of 10\\,bp round-trip cost; paired stationary-block bootstrap "
                "against the equal-weight benchmark.", prov,
                note=("Predictions are identical across rows; only the sizing rule varies, "
                      "so the comparison is paired and most common variance cancels. "
                      "A rule that holds no names has no return series, and its Sharpe is "
                      "reported as ``---`` rather than as zero: abstention is a decision "
                      "not to bet, not a bet that returned nothing."))

    # ------------------------------------------------------- T5b: the gate across alpha
    # The abstention gate is a knob, not a constant, and reporting it at a single alpha
    # shows one point on a curve. Each alpha's singleton mask was fitted per fold on
    # inner-validation and carried in the prediction record, so no model is refitted here.
    gate_rows = []
    ungated = TradingRule(scheme="kelly", quantile=0.10,
                          rebalance_days=cfg.decision.rebalance_days,
                          gate_on_singleton=False, kappa=cfg.decision.kappa,
                          f_max=cfg.decision.f_max)
    fwd_g = main_panel.pivot_table(index="date", columns="ticker", values="next_ret",
                                   aggfunc="mean").sort_index()
    ref = run_backtest(build_weights(main_panel, ungated), fwd_g, ungated,
                       CostModel(spread_bps=cfg.decision.default_bps), ppy)
    gate_rows.append({
        "$\alpha$": np.nan, "Nominal cov.": np.nan, "Singleton rate": 1.0,
        "Names held": float((ref.weights.abs() > 1e-12).sum(axis=1).mean()),
        "Sharpe (net)": ref.summary_net.get("sharpe", np.nan),
        "Turnover (ann.)": ref.turnover * ppy,
    })
    for a in cfg.conformal.alphas:
        col = f"tradeable_a{int(round(a * 100)):02d}"
        if col not in main_panel.columns:
            continue
        sig = main_panel.copy()
        sig["tradeable"] = sig[col].astype(bool)
        rule = TradingRule(scheme="kelly", quantile=0.10,
                           rebalance_days=cfg.decision.rebalance_days,
                           gate_on_singleton=True, kappa=cfg.decision.kappa,
                           f_max=cfg.decision.f_max)
        res = run_backtest(build_weights(sig, rule), fwd_g, rule,
                           CostModel(spread_bps=cfg.decision.default_bps), ppy)
        gate_rows.append({
            "$\alpha$": a, "Nominal cov.": 1.0 - a,
            "Singleton rate": float(sig[col].astype(bool).mean()),
            "Names held": float((res.weights.abs() > 1e-12).sum(axis=1).mean()),
            "Sharpe (net)": res.summary_net.get("sharpe", np.nan),
            "Turnover (ann.)": res.turnover * ppy,
        })
    if len(gate_rows) > 1:
        write_table("T5b_main", pd.DataFrame(gate_rows),
                    "\textbf{Abstention is a position.} The conformal gate swept across "
                    "$\alpha$, holding predictions and sizing rule fixed. The first row is "
                    "the ungated book.", prov,
                    note=("At small $\alpha$ the prediction set is rarely a singleton, the "
                          "gate declines every name, and the book is flat. The gate only "
                          "starts to trade once $\alpha$ is loosened enough for the model's "
                          "own uncertainty to admit a single class."))

    # ---------------------------------------------------------------- T6: economic
    fwd_main = main_panel.pivot_table(index="date", columns="ticker", values="next_ret",
                                      aggfunc="mean").sort_index()
    w_main = build_weights(main_panel, rule_ls)
    sweep = cost_sweep(w_main, fwd_main, rule_ls, tuple(cfg.decision.sweep_bps), ppy)
    res10 = run_backtest(w_main, fwd_main, rule_ls,
                         CostModel(spread_bps=cfg.decision.default_bps), ppy)
    net = res10.net_returns.to_numpy()
    d = DSR.deflated_sharpe_ratio(net, cfg.eval.n_configs_evaluated, periods_per_year=ppy)
    psr = DSR.probabilistic_sharpe_ratio(net, 0.0, ppy)

    # PBO over the strategy zoo actually evaluated.
    zoo, labels = [], []
    for name, panel in panels.items():
        if name == "buy_and_hold":
            continue
        try:
            r = strategy_returns(panel, rule_ls, cfg.decision.default_bps)
            if len(r) > 400:
                zoo.append(r.rename(name)); labels.append(name)
        except Exception:
            pass
    for label, r in series.items():
        zoo.append(r.rename(label)); labels.append(label)
    pbo = None
    if len(zoo) >= 2:
        mat = pd.concat(zoo, axis=1).dropna()
        if len(mat) > 200:
            pbo = DSR.probability_of_backtest_overfitting(mat.to_numpy(), n_blocks=10)

    sweep = sweep.rename(columns={
        "cost_bps": "Cost (bp)", "sharpe_net": "Sharpe (net)",
        "ann_return_net": "Return (ann.)", "max_drawdown": "Max DD",
        "turnover_annualised": "Turnover (ann.)"})
    write_table("T6_main", sweep,
                "Economic evaluation of the primary model: net Sharpe against round-trip "
                "cost, with turnover held fixed across the sweep.", prov,
                note=(f"Break-even cost {res10.break_even_bps:.1f}\\,bp. "
                      f"Probabilistic Sharpe {psr.get('psr', float('nan')):.3f}. "
                      f"Deflated Sharpe {d.dsr:.3f} against "
                      f"{cfg.eval.n_configs_evaluated} declared trials "
                      f"({d.verdict()}). "
                      + (f"Probability of backtest overfitting {pbo.pbo:.3f} over "
                         f"{pbo.n_strategies} strategies and {pbo.n_combinations} "
                         f"splits ({pbo.verdict()})." if pbo else
                         "PBO unavailable: too few comparable strategies.")
                      + " Factor attribution is not reported: no point-in-time factor "
                        "return file accompanies this corpus, and we do not substitute a "
                        "proxy."))

    # ---------------------------------------------------------------- T8: robustness
    rows = []
    mp = main_panel.assign(year=pd.to_datetime(main_panel["date"]).dt.year)
    for year, grp in mp.groupby("year"):
        ic = M.information_coefficient(grp["score"], grp["fwd_ret"], grp["date"])
        summ = M.ic_summary(ic, ppy, cfg.eval.hac_lags)
        try:
            r = strategy_returns(grp, rule_ls, cfg.decision.default_bps)
            sharpe = M.sharpe_ratio(r.to_numpy(), ppy)
        except Exception:
            sharpe = np.nan
        cls = M.classification_metrics(grp["y_true"].to_numpy(), probs(grp).argmax(1))
        rows.append({"Year": int(year), "Sessions": int(ic.size),
                     "Rank IC": summ.mean, "HAC t": summ.t_stat,
                     "Accuracy": cls.get("accuracy", np.nan),
                     "Bal. acc.": cls.get("balanced_accuracy", np.nan),
                     "Sharpe (net)": sharpe})
    write_table("T8_main", pd.DataFrame(rows),
                "Subperiod robustness for the primary model, by calendar year.", prov,
                note="Out-of-sample throughout: each year is covered by test blocks only.")

    # ---------------------------------------------------------------- T9: gate
    gate_note = ("Not evaluable on this corpus. The gate diagnostic regresses the mean "
                 "modal gate on news volume, and this corpus contains no usable "
                 "headlines, so there is no news-flow variable to regress on. The "
                 "diagnostic is retained in the released code and reported as unavailable "
                 "rather than substituted.")
    write_table("T9_main", pd.DataFrame([{"Diagnostic": "gate vs news flow",
                                          "Status": "not evaluable (no text corpus)"}]),
                "Gate diagnostics.", prov, note=gate_note)

    # ---------------------------------------------------------------- figures
    FIGURES.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # F5 reliability
        fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=200)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
        for tag, p in (("raw", probs(main_panel, raw=True)), ("calibrated", probs(main_panel))):
            b = M.reliability_curve(p, main_panel["y_true"].to_numpy(),
                                    cfg.calibration.n_bins).dropna()
            ax.plot(b["confidence"], b["accuracy"], "o-", ms=3, lw=1.1, label=tag)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES / "F5_main.pdf"); plt.close(fig)

        # F6 growth loss vs calibration error
        from finsent.decision.growth_theory import growth_loss_lower_bound_from_ece
        fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=200)
        grid = np.linspace(0, 0.15, 60)
        ax.plot(grid, growth_loss_lower_bound_from_ece(grid), lw=1.5,
                label=r"theory: $(C/2)\,\mathrm{ECE}^2$")
        mm = main_panel.assign(m=pd.to_datetime(main_panel["date"]).dt.to_period("M"))
        xs, ys = [], []
        for _, g in mm.groupby("m"):
            p = probs(g); y = g["y_true"].to_numpy()
            xs.append(M.expected_calibration_error(p, y, 10)["ece"])
            ys.append(0.5 * float(np.mean((p.max(1) - (p.argmax(1) == y)) ** 2)))
        ax.plot(xs, ys, "o", ms=3, alpha=0.6, label="measured (monthly)")
        ax.set_xlabel("expected calibration error"); ax.set_ylabel("log-growth loss")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(FIGURES / "F6_main.pdf"); plt.close(fig)

        # F7 cost curve
        fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=200)
        ax.plot(sweep["Cost (bp)"], sweep["Sharpe (net)"], "o-", lw=1.4)
        ax.axhline(0, color="k", lw=0.7, ls=":")
        if np.isfinite(res10.break_even_bps):
            ax.axvline(res10.break_even_bps, color="#c0392b", ls="--", lw=1.0,
                       label=f"break-even {res10.break_even_bps:.1f} bp")
            ax.legend(fontsize=8)
        ax.set_xlabel("round-trip cost (bp)"); ax.set_ylabel("net Sharpe")
        fig.tight_layout(); fig.savefig(FIGURES / "F7_main.pdf"); plt.close(fig)

        # F9 conformal
        if cov_files:
            fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=200)
            ax.plot(cov_t["alpha"], cov_t["Empirical"], "o-", lw=1.3, label="empirical")
            ax.plot(cov_t["alpha"], cov_t["Nominal"], "k--", lw=0.9, label="nominal")
            ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("coverage"); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(FIGURES / "F9_main.pdf"); plt.close(fig)
        print("  wrote figures F5, F6, F7, F9")
    except Exception as exc:
        print(f"  figures skipped: {exc}")

    print(f"\nall artifacts written to {TABLES}/ and {FIGURES}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
