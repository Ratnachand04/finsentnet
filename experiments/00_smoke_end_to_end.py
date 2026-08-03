"""End-to-end smoke run: synthetic prices to a full paper artifact set.

    python experiments/00_smoke_end_to_end.py --out runs/smoke

Runs the entire protocol -- point-in-time universe, causal features, volatility-scaled
labels, purged walk-forward, per-fold calibration, conformal gating, sizing comparison,
cost sweep, deflated Sharpe -- on synthetic data with a **known** planted signal, using
the logistic baseline as the model. It requires no market data, no news feed and no GPU.

What this is for
----------------
It proves the pipeline executes end to end and that the reported numbers respond
correctly to a known truth: with ``--signal-strength 0`` every metric must come back at
noise, and with a planted edge the IC must recover approximately the planted value.

What this is **not** for
------------------------
Nothing printed here may appear in the manuscript. The data are simulated. The output
directory is stamped ``SYNTHETIC`` for that reason.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Run from anywhere without an editable install or a PYTHONPATH incantation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finsent.config import load_config  # noqa: E402
from finsent.data.features_causal import FEATURE_NAMES, build_feature_panel  # noqa: E402
from finsent.data.labeling import class_balance, vol_scaled_labels  # noqa: E402
from finsent.data.synthetic import make_ohlcv_panel  # noqa: E402
from finsent.data.uniqueness import (  # noqa: E402
    average_uniqueness,
    effective_sample_size,
    uniqueness_from_horizon,
)
from finsent.data.universe import (  # noqa: E402
    UniverseConfig,
    build_liquidity_universe,
    membership_stats,
)
from finsent.eval.report import ReportBuilder  # noqa: E402
from finsent.models.baselines import LogisticFive  # noqa: E402
from finsent.training.walkforward import FoldPrediction, run_walk_forward  # noqa: E402


def build_dataset(n_names: int, n_days: int, horizon: int, k_band: float,
                  signal_strength: float, seed: int) -> pd.DataFrame:
    """Synthetic OHLCV -> causal features -> volatility-scaled labels -> tidy panel."""
    panel = make_ohlcv_panel(n_names=n_names, n_days=n_days, seed=seed)

    close = pd.DataFrame({k: v["close"] for k, v in panel.items()})
    volume = pd.DataFrame({k: v["volume"] for k, v in panel.items()})
    universe = build_liquidity_universe(
        close, volume, UniverseConfig(n_names=max(n_names - 4, 4), min_adv_usd=0.0, min_price=0.0)
    )
    print(f"universe: {membership_stats(universe)}")

    features = build_feature_panel(panel, cross_sectional=True)

    label_rows = []
    rng = np.random.default_rng(seed + 1)
    for ticker, ohlcv in panel.items():
        result = vol_scaled_labels(ohlcv["open"], horizon=horizon, k_band=k_band)
        t1 = uniqueness_from_horizon(len(ohlcv), horizon)
        label_rows.append(
            pd.DataFrame(
                {
                    "date": ohlcv.index,
                    "ticker": ticker,
                    "y_dir": result.y.to_numpy(),
                    "y_ret": result.forward_return.to_numpy(),
                    "fwd_ret": result.forward_return.to_numpy(),
                    "weight": average_uniqueness(t1),
                }
            )
        )
    labels = pd.concat(label_rows, ignore_index=True)

    frame = features.merge(labels, on=["date", "ticker"], how="inner")
    frame = frame.merge(universe, on=["date", "ticker"], how="inner")
    frame = frame.dropna(subset=["y_dir", "fwd_ret", *FEATURE_NAMES])

    # A synthetic "sentiment" column, planted with a controllable amount of real signal.
    # At strength zero it is pure noise, which is what makes the null run meaningful.
    noise = rng.standard_normal(len(frame))
    frame["sentiment"] = (
        signal_strength * frame["fwd_ret"].to_numpy() / frame["fwd_ret"].std()
        + np.sqrt(max(1 - signal_strength**2, 0.0)) * noise
    )

    frame["y_dir"] = frame["y_dir"].astype(int)
    print(f"labels: {class_balance(frame['y_dir'])}")
    print(
        f"effective sample size: {effective_sample_size(frame['weight']):,.0f} "
        f"of {len(frame):,d} rows"
    )
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def make_fit_predict(feature_columns: list[str]):
    """Wrap the logistic baseline in the protocol's fit/predict contract."""

    def fit_predict(train, inner_val, test, seed, fold) -> FoldPrediction:
        model = LogisticFive(columns=tuple(feature_columns))
        model.fit(train, train["y_dir"].to_numpy())

        def logits(df):
            X = model._design(df, fit=False)  # noqa: SLF001 - deliberate, same package
            return X @ model.coef_

        residual_var = float(np.var(train["fwd_ret"].to_numpy())) or 1e-6
        test_scores = model.predict(test).score

        return FoldPrediction(
            logits_val=logits(inner_val),
            y_val=inner_val["y_dir"].to_numpy(dtype=int),
            logits_test=logits(test),
            y_test=test["y_dir"].to_numpy(dtype=int),
            mu_test=test_scores * float(np.std(train["fwd_ret"].to_numpy())),
            sigma2_test=np.full(len(test), residual_var),
            fwd_ret_test=test["fwd_ret"].to_numpy(dtype=float),
            dates_test=test["date"].to_numpy(),
            tickers_test=test["ticker"].to_numpy(),
        )

    return fit_predict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/smoke")
    ap.add_argument("--n-names", type=int, default=40)
    ap.add_argument("--n-days", type=int, default=1600)
    ap.add_argument("--signal-strength", type=float, default=0.06,
                    help="planted correlation between the sentiment feature and the label")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"config: {cfg.describe()}\n")

    frame = build_dataset(
        n_names=args.n_names,
        n_days=args.n_days,
        horizon=cfg.data.horizon_h,
        k_band=cfg.data.k_band,
        signal_strength=args.signal_strength,
        seed=0,
    )

    columns = ["sentiment", "ret_5", "rsi_14", "ewma_vol_20", "mom_12_1"]

    # The synthetic sample is far shorter than the real study period, so the protocol is
    # scaled down. The purge and embargo are NOT relaxed: those are correctness, not
    # convenience.
    import dataclasses

    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data, train_min_years=2, inner_val_months=3, test_months=3,
            refit_every_months=3,
        ),
    )

    result = run_walk_forward(
        frame, make_fit_predict(columns), cfg, seeds=args.seeds,
        progress=lambda msg: print("  " + msg),
    )
    print("\n" + result.summary())

    if result.panel.empty:
        print("\nno out-of-sample predictions produced; widen --n-days")
        return 1

    print("\ncalibration (mean over folds):")
    print(result.calibration[["ece_raw", "ece_calibrated", "brier_raw",
                              "brier_calibrated"]].mean().round(5).to_string())

    print("\nconformal coverage:")
    print(result.coverage.groupby("alpha")[
        ["empirical_coverage", "mean_set_size", "singleton_rate"]
    ].mean().round(4).to_string())

    out = Path(args.out)
    builder = ReportBuilder(cfg, result.panel, out, seeds=tuple(args.seeds), label="SYNTHETIC")
    tables = builder.build_all(coverage=result.coverage)

    print(f"\nwrote {len(tables)} tables and the figure set to {out}")
    print(f"provenance: {builder.provenance.footnote()}")
    print(
        "\nSYNTHETIC DATA. Nothing in these artifacts may be quoted in the manuscript; "
        "the run exists to prove the pipeline executes and that the metrics respond "
        "correctly to a known planted signal."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
