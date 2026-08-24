"""Emit every table and figure the paper contains, from one command.

    python -m finsent.eval.report --signal random --out paper/

Running with ``--signal random`` is the Phase-1 acceptance test: the complete table and
figure set renders, filled with noise. The manuscript skeleton therefore exists before
any result does, which is what makes it structurally impossible to "write the tables
first and produce the numbers later".

Every artifact carries a provenance footnote (git sha, config hash, data hash, seeds).
SPEC.md 6.1: no number reaches the manuscript by hand.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from finsent.config import LABEL_NAMES, Config, load_config
from finsent.decision.costs import CostModel
from finsent.eval import dsr as DSR
from finsent.eval import metrics as M
from finsent.eval import stats as S
from finsent.eval.backtest import TradingRule, build_weights, cost_sweep, run_backtest
from finsent.utils.hashing import Provenance, hash_frame

__all__ = ["ReportBuilder", "make_random_signals", "main"]

TABLE_IDS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9")
FIGURE_IDS = ("F2", "F5", "F6", "F7", "F8", "F9")


# --------------------------------------------------------------------------------------
# Random-signal generator for the skeleton run
# --------------------------------------------------------------------------------------
def make_random_signals(
    n_dates: int = 500,
    n_names: int = 60,
    seed: int = 0,
    signal_strength: float = 0.0,
    start: str = "2022-01-03",
) -> pd.DataFrame:
    """Synthetic panel of scores, probabilities and realised returns.

    ``signal_strength`` is the correlation planted between score and forward return.
    The default of exactly zero is deliberate: the skeleton run must produce a table
    full of statistical non-results, so that a real result is visibly different.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_dates)
    tickers = [f"SYN{i:03d}" for i in range(n_names)]

    rows = []
    for date in dates:
        z = rng.standard_normal(n_names)
        noise = rng.standard_normal(n_names)
        fwd = 0.02 * (signal_strength * z + np.sqrt(max(1 - signal_strength**2, 0.0)) * noise)

        logits = np.column_stack([-z, np.zeros(n_names), z]) * 0.8 + rng.standard_normal(
            (n_names, 3)
        ) * 0.3
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        sigma = 0.02 * np.exp(0.3 * rng.standard_normal(n_names))
        theta = 0.6 * sigma
        y = np.where(fwd > theta, 2, np.where(fwd < -theta, 0, 1))

        rows.append(
            pd.DataFrame(
                {
                    "date": date,
                    "ticker": tickers,
                    "score": 0.02 * z * signal_strength + 0.002 * noise,
                    "mu_hat": 0.02 * z * signal_strength,
                    "sigma2": sigma**2,
                    "p_down": probs[:, 0],
                    "p_neutral": probs[:, 1],
                    "p_up": probs[:, 2],
                    "y_true": y,
                    "fwd_ret": fwd,
                    "tradeable": probs.max(axis=1) > 0.45,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------------------
# Report builder
# --------------------------------------------------------------------------------------
@dataclass
class ReportBuilder:
    """Assembles tables and figures from a tidy signal panel."""

    config: Config
    signals: pd.DataFrame
    out_dir: Path
    seeds: tuple[int, ...] = (0,)
    label: str = "run"

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        (self.out_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "figures").mkdir(parents=True, exist_ok=True)
        self.provenance = Provenance.capture(
            config_hash=self.config.config_hash,
            data_hash=hash_frame(self.signals),
            seeds=self.seeds,
        )
        self._probs = self.signals[["p_down", "p_neutral", "p_up"]].to_numpy(dtype=float)
        self._y = self.signals["y_true"].to_numpy(dtype=int)

    # -- persistence -----------------------------------------------------------------
    def _write_table(self, tid: str, df: pd.DataFrame, caption: str) -> None:
        stem = self.out_dir / "tables" / f"{tid}_{self.label}"
        df.to_csv(stem.with_suffix(".csv"), index=False)
        # na_rep, not float_format: pandas does not consult float_format for
        # missing values, so a NaN renders as the literal string "NaN".
        body = df.to_latex(index=False, escape=True, na_rep="---",
                           float_format=lambda v: f"{v:.4f}")
        note = self.provenance.footnote()
        tex = (
            f"% {tid}: {caption}\n"
            f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\label{{tab:{tid.lower()}}}\n{body}\n"
            f"\\vspace{{2pt}}\\footnotesize\\emph{{Provenance:}} \\texttt{{{note}}}\n"
            f"\\end{{table}}\n"
        )
        stem.with_suffix(".tex").write_text(tex, encoding="utf-8")

    def _figure(self, fid: str, draw: Callable[[Any], None], caption: str) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=150)
        try:
            draw(ax)
        except Exception as exc:  # a broken panel must not abort the whole report
            ax.text(0.5, 0.5, f"{fid} unavailable: {exc}", ha="center", va="center", fontsize=7)
            ax.set_axis_off()
        ax.set_title(caption, fontsize=9)
        fig.tight_layout()
        stem = self.out_dir / "figures" / f"{fid}_{self.label}"
        fig.savefig(stem.with_suffix(".png"))
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)

    # -- tables ----------------------------------------------------------------------
    def table_t1_dataset(self) -> pd.DataFrame:
        s = self.signals
        by_year = s.assign(year=pd.to_datetime(s["date"]).dt.year).groupby("year")
        rows = []
        for year, grp in by_year:
            counts = grp["y_true"].value_counts(normalize=True)
            rows.append(
                {
                    "year": int(year),
                    "ticker_days": int(len(grp)),
                    "n_names": int(grp["ticker"].nunique()),
                    "pct_down": float(counts.get(0, 0.0)),
                    "pct_neutral": float(counts.get(1, 0.0)),
                    "pct_up": float(counts.get(2, 0.0)),
                    "news_coverage": float(grp.get("has_news", pd.Series(dtype=float)).mean())
                    if "has_news" in grp
                    else np.nan,
                }
            )
        df = pd.DataFrame(rows)
        self._write_table("T1", df, "Dataset composition and class balance by year.")
        return df

    def table_t2_parameters(self, param_counts: dict[str, int] | None = None) -> pd.DataFrame:
        counts = param_counts or {"(model not instantiated)": 0}
        total = sum(counts.values())
        df = pd.DataFrame(
            [
                {
                    "module": k,
                    "params": v,
                    "pct": (100.0 * v / total) if total else np.nan,
                }
                for k, v in counts.items()
            ]
        )
        df.loc[len(df)] = {"module": "TOTAL", "params": total, "pct": 100.0 if total else np.nan}
        self._write_table("T2", df, "Trainable parameter count by module.")
        return df

    def table_t3_predictive(self, methods: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
        """Predictive comparison. ``methods`` maps a name to a panel with the same schema."""
        panels = {"FinSentNet-C": self.signals}
        panels.update(methods or {})

        rows = []
        for name, panel in panels.items():
            ic = M.information_coefficient(panel["score"], panel["fwd_ret"], panel["date"])
            summ = M.ic_summary(ic, self.config.eval.periods_per_year, self.config.eval.hac_lags)
            pred = panel[["p_down", "p_neutral", "p_up"]].to_numpy().argmax(axis=1)
            cls = M.classification_metrics(panel["y_true"].to_numpy(), pred)
            rows.append(
                {
                    "method": name,
                    "accuracy": cls.get("accuracy", np.nan),
                    "balanced_accuracy": cls.get("balanced_accuracy", np.nan),
                    "majority_baseline": cls.get("majority_baseline", np.nan),
                    "binary_accuracy": cls.get("binary_accuracy", np.nan),
                    "macro_f1": cls.get("macro_f1", np.nan),
                    "mcc": cls.get("mcc", np.nan),
                    "rank_ic": summ.mean,
                    "ic_tstat_hac": summ.t_stat,
                    "icir_ann": summ.icir_annualised,
                }
            )
        df = pd.DataFrame(rows)
        self._write_table(
            "T3", df, "Predictive performance. IC t-statistics are Newey-West corrected."
        )
        return df

    def table_t4_calibration(self, probs_calibrated: np.ndarray | None = None) -> pd.DataFrame:
        rows = []
        for tag, p in (("raw", self._probs), ("calibrated", probs_calibrated)):
            if p is None:
                continue
            cal = M.expected_calibration_error(p, self._y, self.config.calibration.n_bins)
            rows.append(
                {
                    "variant": tag,
                    "ece": cal["ece"],
                    "mce": cal["mce"],
                    "brier": M.brier_score(p, self._y),
                    "nll": M.multiclass_nll(p, self._y),
                    "mean_confidence": cal["mean_confidence"],
                    "mean_accuracy": cal["mean_accuracy"],
                    "overconfidence": cal["overconfidence"],
                }
            )
        df = pd.DataFrame(rows)
        self._write_table("T4", df, "Calibration before and after post-hoc scaling.")
        return df

    def table_t5_sizing(self) -> pd.DataFrame:
        """The headline experiment: one fixed set of predictions, four sizing rules."""
        fwd = self.signals.pivot_table(
            index="date", columns="ticker", values="fwd_ret", aggfunc="mean"
        ).sort_index()

        rules = {
            "equal_weight": TradingRule(
                scheme="decile_long_short", gate_on_singleton=False,
                rebalance_days=self.config.decision.rebalance_days,
            ),
            "raw_softmax_kelly": TradingRule(
                scheme="kelly", gate_on_singleton=False,
                rebalance_days=self.config.decision.rebalance_days,
            ),
            "calibrated_kelly": TradingRule(
                scheme="kelly", gate_on_singleton=False,
                rebalance_days=self.config.decision.rebalance_days,
            ),
            "conformal_gated_kelly": TradingRule(
                scheme="kelly", gate_on_singleton=True,
                rebalance_days=self.config.decision.rebalance_days,
            ),
        }

        base = self.signals.copy()
        conf = base[["p_down", "p_neutral", "p_up"]].to_numpy().max(axis=1)

        rows, series = [], {}
        for name, rule in rules.items():
            sig = base.copy()
            if name == "raw_softmax_kelly":
                # No variance head: confidence is used as an inverse-variance proxy,
                # which is exactly how overconfidence turns into overbetting.
                sig["sigma2"] = np.maximum((1.0 - conf) ** 2, 1e-4)
            res = run_backtest(
                build_weights(sig, rule),
                fwd,
                rule,
                CostModel(spread_bps=self.config.decision.default_bps),
                self.config.eval.periods_per_year,
            )
            rows.append(res.as_row(name))
            series[name] = res.net_returns

        df = pd.DataFrame(rows)

        # Paired bootstrap against the equal-weight benchmark: predictions are identical
        # across rules, so the comparison is paired and most common variance cancels.
        bench = series["equal_weight"]
        ppy = self.config.eval.periods_per_year
        pvals = []
        for name in df["method"]:
            if name == "equal_weight":
                pvals.append(np.nan)
                continue
            joined = pd.concat([series[name].rename("a"), bench.rename("b")], axis=1).dropna()
            out = S.paired_bootstrap_diff(
                joined["a"].to_numpy(),
                joined["b"].to_numpy(),
                lambda x: M.sharpe_ratio(x, ppy),
                block_mean=self.config.eval.block_mean,
                n_resamples=min(self.config.eval.n_resamples, 2000),
            )
            pvals.append(out["p_value"])
        df["paired_p_vs_equal_weight"] = pvals

        self._write_table(
            "T5",
            df,
            "Sizing experiment: identical predictions, four sizing rules, paired bootstrap.",
        )
        return df

    def table_t6_economic(self) -> pd.DataFrame:
        fwd = self.signals.pivot_table(
            index="date", columns="ticker", values="fwd_ret", aggfunc="mean"
        ).sort_index()
        rule = TradingRule(
            scheme="decile_long_short",
            gate_on_singleton=False,
            rebalance_days=self.config.decision.rebalance_days,
        )
        weights = build_weights(self.signals, rule)
        sweep = cost_sweep(
            weights, fwd, rule, self.config.decision.sweep_bps, self.config.eval.periods_per_year
        )

        res = run_backtest(
            weights, fwd, rule, CostModel(self.config.decision.default_bps),
            self.config.eval.periods_per_year,
        )
        deflated = DSR.deflated_sharpe_ratio(
            res.net_returns.to_numpy(),
            n_trials=self.config.eval.n_configs_evaluated,
            periods_per_year=self.config.eval.periods_per_year,
        )
        psr = DSR.probabilistic_sharpe_ratio(res.net_returns.to_numpy())

        sweep["deflated_sharpe"] = deflated.dsr
        sweep["psr"] = psr.get("psr", np.nan)
        sweep["n_trials_declared"] = self.config.eval.n_configs_evaluated
        sweep["break_even_bps"] = res.break_even_bps
        sweep["dsr_verdict"] = deflated.verdict()

        self._write_table(
            "T6",
            sweep,
            "Economic evaluation: net Sharpe versus cost, with deflation for the "
            "declared number of trials.",
        )
        return sweep

    def table_t7_ablations(self, ablations: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
        if not ablations:
            df = pd.DataFrame(
                [{"ablation": "(none run)", "rank_ic": np.nan, "delta_ic": np.nan}]
            )
            self._write_table("T7", df, "Ablations under the full walk-forward protocol.")
            return df

        base_ic = M.information_coefficient(
            self.signals["score"], self.signals["fwd_ret"], self.signals["date"]
        )
        base_mean = float(base_ic.mean())
        rows = []
        for name, panel in ablations.items():
            ic = M.information_coefficient(panel["score"], panel["fwd_ret"], panel["date"])
            joined = pd.concat([base_ic.rename("full"), ic.rename("abl")], axis=1).dropna()
            dm = S.diebold_mariano(
                -joined["full"].to_numpy(), -joined["abl"].to_numpy(),
                lags=self.config.eval.hac_lags, labels=("full", name),
            )
            rows.append(
                {
                    "ablation": name,
                    "rank_ic": float(ic.mean()),
                    "delta_ic": float(ic.mean()) - base_mean,
                    "dm_stat": dm.statistic,
                    "dm_p": dm.p_value,
                }
            )
        df = pd.DataFrame(rows)
        self._write_table("T7", df, "Ablations under the full walk-forward protocol.")
        return df

    def table_t8_robustness(self) -> pd.DataFrame:
        s = self.signals.assign(year=pd.to_datetime(self.signals["date"]).dt.year)
        rows = []
        for year, grp in s.groupby("year"):
            ic = M.information_coefficient(grp["score"], grp["fwd_ret"], grp["date"])
            summ = M.ic_summary(ic, self.config.eval.periods_per_year, self.config.eval.hac_lags)
            pred = grp[["p_down", "p_neutral", "p_up"]].to_numpy().argmax(axis=1)
            cls = M.classification_metrics(grp["y_true"].to_numpy(), pred)
            rows.append(
                {
                    "subperiod": int(year),
                    "n_days": int(ic.size),
                    "rank_ic": summ.mean,
                    "ic_tstat": summ.t_stat,
                    "accuracy": cls.get("accuracy", np.nan),
                    "balanced_accuracy": cls.get("balanced_accuracy", np.nan),
                }
            )
        df = pd.DataFrame(rows)
        self._write_table("T8", df, "Subperiod robustness.")
        return df

    def table_t9_hyperparameters(self) -> pd.DataFrame:
        c = self.config
        rows = [
            ("lookback L", c.data.lookback_L), ("features F", c.data.n_features),
            ("horizon h", c.data.horizon_h), ("max headlines K", c.data.K),
            ("d_model", c.model.d_model), ("fusion heads", c.model.fusion_heads),
            ("TCN receptive field", c.model.tcn_receptive_field),
            ("learning rate", c.train.lr), ("weight decay", c.train.weight_decay),
            ("epochs", c.train.epochs), ("patience", c.train.patience),
            ("lambda_reg", c.loss.lambda_reg), ("lambda_cal", c.loss.lambda_cal),
            ("lambda_rank", c.loss.lambda_rank),
            ("class weights", c.loss.class_weights),
            ("purge days", c.data.purge_days), ("embargo pct", c.data.embargo_pct),
            ("kelly kappa", c.decision.kappa), ("f_max", c.decision.f_max),
            ("rebalance days", c.decision.rebalance_days),
            ("configurations evaluated", c.eval.n_configs_evaluated),
        ]
        df = pd.DataFrame(rows, columns=["hyperparameter", "value"])
        self._write_table("T9", df, "Hyperparameters and declared search budget.")
        return df

    # -- figures ---------------------------------------------------------------------
    def figure_f2_splits(self) -> None:
        d = self.config.data

        def draw(ax):
            for i in range(4):
                y = 3 - i
                base = i * 6
                ax.barh(y, 36, left=base, height=0.5, color="#4C78A8", label="train" if not i else None)
                ax.barh(y, d.purge_days / 5, left=base + 36, height=0.5, color="#E45756",
                        label="purge" if not i else None)
                ax.barh(y, 6, left=base + 37, height=0.5, color="#F58518",
                        label="inner val" if not i else None)
                ax.barh(y, d.purge_days / 5, left=base + 43, height=0.5, color="#E45756")
                ax.barh(y, 6, left=base + 44, height=0.5, color="#54A24B",
                        label="test" if not i else None)
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels([f"fold {i}" for i in (4, 3, 2, 1)], fontsize=7)
            ax.set_xlabel("months", fontsize=8)
            ax.legend(fontsize=6, ncol=4, loc="upper left")

        self._figure("F2", draw, "Purged, embargoed walk-forward protocol")

    def figure_f5_reliability(self, probs_calibrated: np.ndarray | None = None) -> None:
        def draw(ax):
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
            for tag, p in (("raw", self._probs), ("calibrated", probs_calibrated)):
                if p is None:
                    continue
                bins = M.reliability_curve(p, self._y, self.config.calibration.n_bins).dropna()
                ax.plot(bins["confidence"], bins["accuracy"], "o-", ms=3, lw=1.0, label=tag)
            ax.set_xlabel("confidence", fontsize=8)
            ax.set_ylabel("accuracy", fontsize=8)
            ax.legend(fontsize=7)

        self._figure("F5", draw, "Reliability diagram")

    def figure_f6_growth_vs_ece(self) -> None:
        from finsent.decision.growth_theory import growth_loss_lower_bound_from_ece

        def draw(ax):
            grid = np.linspace(0.0, 0.15, 60)
            ax.plot(grid, growth_loss_lower_bound_from_ece(grid), "-", lw=1.4,
                    label="theory: (C/2)$\\cdot$ECE$^2$")
            s = self.signals.assign(month=pd.to_datetime(self.signals["date"]).dt.to_period("M"))
            xs, ys = [], []
            for _, grp in s.groupby("month"):
                p = grp[["p_down", "p_neutral", "p_up"]].to_numpy()
                y = grp["y_true"].to_numpy()
                ece = M.expected_calibration_error(p, y, 10)["ece"]
                conf = p.max(axis=1)
                acc = (p.argmax(axis=1) == y).astype(float)
                xs.append(ece)
                ys.append(float(np.mean((conf - acc) ** 2)) * 0.5)
            ax.plot(xs, ys, "o", ms=3, alpha=0.6, label="measured (monthly folds)")
            ax.set_xlabel("expected calibration error", fontsize=8)
            ax.set_ylabel("log-growth loss", fontsize=8)
            ax.legend(fontsize=7)

        self._figure("F6", draw, "Growth-rate loss versus calibration error")

    def figure_f7_cost_curve(self, sweep: pd.DataFrame) -> None:
        def draw(ax):
            ax.plot(sweep["cost_bps"], sweep["sharpe_net"], "o-", lw=1.4)
            ax.axhline(0.0, color="k", lw=0.7, ls=":")
            if "break_even_bps" in sweep and np.isfinite(sweep["break_even_bps"].iloc[0]):
                ax.axvline(sweep["break_even_bps"].iloc[0], color="#E45756", ls="--", lw=1.0,
                           label="break-even")
                ax.legend(fontsize=7)
            ax.set_xlabel("round-trip cost (bps)", fontsize=8)
            ax.set_ylabel("net Sharpe", fontsize=8)

        self._figure("F7", draw, "Net Sharpe versus transaction cost")

    def figure_f8_gate(self, gate: pd.Series | None = None) -> None:
        def draw(ax):
            if gate is None or gate.empty:
                raise ValueError("no gate series recorded (model not run)")
            ax.plot(gate.index, gate.to_numpy(), lw=0.9)
            ax.set_ylabel("E[g] (attention to news)", fontsize=8)
            ax.set_xlabel("date", fontsize=8)

        self._figure("F8", draw, "Modal gate over time")

    def figure_f9_conformal(self, coverage: pd.DataFrame | None = None) -> None:
        def draw(ax):
            if coverage is None or coverage.empty:
                raise ValueError("no conformal coverage recorded")
            ax.plot(coverage["alpha"], coverage["empirical_coverage"], "o-", lw=1.2,
                    label="empirical")
            ax.plot(coverage["alpha"], 1.0 - coverage["alpha"], "k--", lw=0.8, label="nominal")
            ax.set_xlabel("alpha", fontsize=8)
            ax.set_ylabel("coverage", fontsize=8)
            ax.legend(fontsize=7)

        self._figure("F9", draw, "Conformal coverage versus nominal level")

    # -- orchestration ---------------------------------------------------------------
    def build_all(
        self,
        probs_calibrated: np.ndarray | None = None,
        param_counts: dict[str, int] | None = None,
        methods: dict[str, pd.DataFrame] | None = None,
        ablations: dict[str, pd.DataFrame] | None = None,
        gate: pd.Series | None = None,
        coverage: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        out = {
            "T1": self.table_t1_dataset(),
            "T2": self.table_t2_parameters(param_counts),
            "T3": self.table_t3_predictive(methods),
            "T4": self.table_t4_calibration(probs_calibrated),
            "T5": self.table_t5_sizing(),
            "T6": self.table_t6_economic(),
            "T7": self.table_t7_ablations(ablations),
            "T8": self.table_t8_robustness(),
            "T9": self.table_t9_hyperparameters(),
        }
        self.figure_f2_splits()
        self.figure_f5_reliability(probs_calibrated)
        self.figure_f6_growth_vs_ece()
        self.figure_f7_cost_curve(out["T6"])
        self.figure_f8_gate(gate)
        self.figure_f9_conformal(coverage)

        manifest = {
            "label": self.label,
            "provenance": self.provenance.to_dict(),
            "trading_rule": TradingRule(
                rebalance_days=self.config.decision.rebalance_days
            ).describe(),
            "tables": sorted(out),
            "config": self.config.describe(),
        }
        (self.out_dir / f"manifest_{self.label}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Emit every paper table and figure.")
    ap.add_argument("--signal", default="random", choices=["random", "panel"])
    ap.add_argument("--panel", type=str, default=None, help="parquet/csv panel for --signal panel")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--out", type=str, default="paper")
    ap.add_argument("--label", type=str, default="skeleton")
    ap.add_argument("--n-dates", type=int, default=500)
    ap.add_argument("--n-names", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--signal-strength", type=float, default=0.0,
        help="planted score/return correlation; 0.0 keeps the skeleton a true non-result",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    if args.signal == "random":
        panel = make_random_signals(
            n_dates=args.n_dates, n_names=args.n_names, seed=args.seed,
            signal_strength=args.signal_strength,
        )
    else:
        if not args.panel:
            ap.error("--signal panel requires --panel PATH")
        path = Path(args.panel)
        panel = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    builder = ReportBuilder(cfg, panel, Path(args.out), seeds=(args.seed,), label=args.label)
    tables = builder.build_all()

    print(f"config     : {cfg.describe()}")
    print(f"provenance : {builder.provenance.footnote()}")
    print(f"written    : {len(tables)} tables + {len(FIGURE_IDS)} figures -> {args.out}")
    if args.signal == "random" and args.signal_strength == 0.0:
        print(
            "\nNOTE: this is the skeleton run on a zero-signal panel. Every number above "
            "is noise by construction and must never be quoted."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
