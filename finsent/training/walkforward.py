"""Walk-forward orchestration: fit, calibrate, conform, predict — once per fold.

This module encodes the protocol so that no experiment can accidentally deviate from it.
Every model, including every baseline, is driven through the same loop, which means a
comparison cannot quietly use a different split, a different calibration set or a
different trading rule.

Per fold, in this order:

1. fit the model on the **train** block only;
2. fit the calibration map on the **inner-validation** block;
3. fit the conformal quantile on the same inner-validation block;
4. predict on the **test** block, which has been touched by nothing above.

Step 2 and 3 sharing the inner-validation block, and step 4 seeing it for the first time,
is what removes the V2 triple-dipping defect in which loss weights, early stopping and
temperature were all tuned against the data the result was then reported on.

The fitting function is injected, so this module has no PyTorch dependency and the
baselines exercise exactly the same protocol as the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.special import softmax

from finsent.config import Config
from finsent.data.splits import Fold, PurgedWalkForward
from finsent.eval import metrics as M
from finsent.training.calibrate import select_calibrator
from finsent.training.conformal import AdaptiveConformal, SplitConformal

__all__ = ["FoldPrediction", "FitPredict", "WalkForwardResult", "run_walk_forward"]


@dataclass
class FoldPrediction:
    """Raw model output for one fold, before calibration."""

    logits_val: np.ndarray
    y_val: np.ndarray
    logits_test: np.ndarray
    y_test: np.ndarray
    mu_test: np.ndarray
    sigma2_test: np.ndarray
    fwd_ret_test: np.ndarray
    dates_test: np.ndarray
    tickers_test: np.ndarray
    gate_test: np.ndarray | None = None


class FitPredict(Protocol):
    """What a model must provide to be evaluated under this protocol."""

    def __call__(
        self,
        train: pd.DataFrame,
        inner_val: pd.DataFrame,
        test: pd.DataFrame,
        seed: int,
        fold: Fold,
    ) -> FoldPrediction: ...


@dataclass
class WalkForwardResult:
    """Stacked out-of-sample predictions across every fold and seed."""

    panel: pd.DataFrame
    fold_metrics: pd.DataFrame
    calibration: pd.DataFrame
    coverage: pd.DataFrame
    n_folds: int
    seeds: tuple[int, ...]
    raw_probs: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 3)))

    def summary(self) -> str:
        ic = M.information_coefficient(
            self.panel["score"], self.panel["fwd_ret"], self.panel["date"]
        )
        s = M.ic_summary(ic)
        return (
            f"{self.n_folds} folds x {len(self.seeds)} seeds | "
            f"{len(self.panel):,d} out-of-sample predictions\n"
            f"  Rank-IC {s.mean:+.4f} (HAC t = {s.t_stat:+.2f}), ICIR_ann {s.icir_annualised:.2f}"
        )


def run_walk_forward(
    frame: pd.DataFrame,
    fit_predict: FitPredict,
    cfg: Config,
    seeds: Sequence[int] | None = None,
    date_column: str = "date",
    progress: Callable[[str], None] | None = None,
) -> WalkForwardResult:
    """Drive one model through the full purged walk-forward protocol.

    ``frame`` is a tidy panel containing at least ``date``, ``ticker``, the features the
    model consumes, ``y_dir``, ``y_ret`` and ``fwd_ret``.
    """
    seeds = tuple(seeds if seeds is not None else cfg.seed_list)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame[date_column]).unique()))

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

    rows: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    cal_rows: list[dict] = []
    cov_rows: list[dict] = []
    raw_probs: list[np.ndarray] = []
    n_folds = 0

    for fold, train, val, test in splitter.split_frame(frame, date_column):
        n_folds += 1
        if train.empty or val.empty or test.empty:
            continue

        for seed in seeds:
            if progress:
                progress(f"fold {fold.index} seed {seed}: {fold.describe(dates)}")

            pred = fit_predict(train, val, test, seed, fold)

            # (2) calibration map, fitted AND selected inside the inner-validation block.
            # Identity competes as a real candidate: when the model is weak, a
            # near-uniform softmax is already well calibrated and fitting a map only adds
            # variance. Judging the candidates on a later slice of the validation block
            # means the winner has survived the same drift the test block will impose.
            calibrator, cal_trace = select_calibrator(
                pred.logits_val,
                pred.y_val,
                methods=("none", "temperature", cfg.calibration.method),
                max_iter=cfg.calibration.max_iter,
                n_bins=cfg.calibration.n_bins,
                seed=seed,
            )
            probs_val = calibrator.predict_proba(pred.logits_val)
            probs_test = calibrator.predict_proba(pred.logits_test)
            raw_test = softmax(pred.logits_test, axis=1)
            raw_probs.append(raw_test)

            # (3) conformal quantiles, fitted on the same inner-validation block. Every
            # alpha in the grid is fitted here and its singleton mask is carried into the
            # prediction record, so the abstention gate can be re-examined at any alpha
            # at report time without refitting a calibrator or retraining a model. The
            # primary alpha -- the one that fills the plain ``tradeable`` column, and so
            # the one the headline trading rule uses -- stays the second entry of the
            # grid, exactly as before.
            primary_alpha = cfg.conformal.alphas[min(1, len(cfg.conformal.alphas) - 1)]
            gates: dict[str, np.ndarray] = {}
            tradeable = None
            for a in cfg.conformal.alphas:
                predictor = SplitConformal(alpha=a, score=cfg.conformal.score, seed=seed)
                predictor.fit(probs_val, pred.y_val)
                mask = predictor.singleton_mask(probs_test)
                gates[f"tradeable_a{int(round(a * 100)):02d}"] = mask
                cov_rows.append({"fold": fold.index, "seed": seed, "variant": "split",
                                 **predictor.evaluate(probs_test, pred.y_test)})
                if a == primary_alpha:
                    tradeable = mask

                # The online variant of Gibbs and Candes. Section 6.2 promises both, and
                # only the adaptive one holds coverage when exchangeability fails across
                # regimes, which financial data reliably arranges. It consumes each
                # outcome only after emitting that row's set, so the ordering mirrors
                # live use; passing the label first would be a leak.
                if cfg.conformal.adaptive_enabled:
                    online = AdaptiveConformal(
                        alpha_target=a, gamma=cfg.conformal.adaptive_gamma,
                        score=cfg.conformal.score,
                    ).fit(probs_val, pred.y_val)
                    out = online.run(probs_test, pred.y_test)
                    cov_rows.append({
                        "fold": fold.index, "seed": seed, "variant": "adaptive",
                        "alpha": a, "nominal_coverage": 1.0 - a,
                        "empirical_coverage": out["empirical_coverage"],
                        "coverage_gap": out["empirical_coverage"] - (1.0 - a),
                        "mean_set_size": out["mean_set_size"],
                        "singleton_rate": out["singleton_rate"],
                        "abstention_rate": 1.0 - out["singleton_rate"],
                        "q_hat": float("nan"),   # q_hat moves every step here
                        "final_alpha": out["final_alpha"],
                        "n": int(pred.y_test.size),
                    })
            if tradeable is None:  # a grid that somehow omits its own primary entry
                tradeable = np.zeros(len(probs_test), dtype=bool)

            rows.append(
                pd.DataFrame(
                    {
                        "date": pred.dates_test,
                        "ticker": pred.tickers_test,
                        "fold": fold.index,
                        "seed": seed,
                        "score": pred.mu_test,
                        "mu_hat": pred.mu_test,
                        "sigma2": pred.sigma2_test,
                        "p_down": probs_test[:, 0],
                        "p_neutral": probs_test[:, 1],
                        "p_up": probs_test[:, 2],
                        "p_down_raw": raw_test[:, 0],
                        "p_neutral_raw": raw_test[:, 1],
                        "p_up_raw": raw_test[:, 2],
                        "y_true": pred.y_test,
                        "fwd_ret": pred.fwd_ret_test,
                        "tradeable": tradeable,
                        **gates,
                        "gate_mean": pred.gate_test
                        if pred.gate_test is not None
                        else np.nan,
                    }
                )
            )

            cls = M.classification_metrics(pred.y_test, probs_test.argmax(axis=1))
            ic = M.information_coefficient(
                pd.Series(pred.mu_test), pd.Series(pred.fwd_ret_test), pred.dates_test
            )
            fold_rows.append(
                {
                    "fold": fold.index,
                    "seed": seed,
                    "n_train": len(train),
                    "n_val": len(val),
                    "n_test": len(test),
                    "accuracy": cls.get("accuracy", np.nan),
                    "balanced_accuracy": cls.get("balanced_accuracy", np.nan),
                    "mcc": cls.get("mcc", np.nan),
                    "rank_ic": float(ic.mean()) if len(ic) else np.nan,
                }
            )

            raw_stats = M.expected_calibration_error(raw_test, pred.y_test, cfg.calibration.n_bins)
            cal_stats = M.expected_calibration_error(
                probs_test, pred.y_test, cfg.calibration.n_bins
            )
            cal_rows.append(
                {
                    "fold": fold.index,
                    "seed": seed,
                    "ece_raw": raw_stats["ece"],
                    "ece_calibrated": cal_stats["ece"],
                    "mce_raw": raw_stats["mce"],
                    "mce_calibrated": cal_stats["mce"],
                    "brier_raw": M.brier_score(raw_test, pred.y_test),
                    "brier_calibrated": M.brier_score(probs_test, pred.y_test),
                    "overconfidence_raw": raw_stats["overconfidence"],
                    "calibrator": type(calibrator).__name__,
                    "selection_trace": ";".join(
                        f"{t['method']}={t['ece']:.4f}" for t in cal_trace
                    ),
                }
            )


    panel = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["date", "ticker", "score", "fwd_ret", "y_true"])
    )
    return WalkForwardResult(
        panel=panel,
        fold_metrics=pd.DataFrame(fold_rows),
        calibration=pd.DataFrame(cal_rows),
        coverage=pd.DataFrame(cov_rows),
        n_folds=n_folds,
        seeds=seeds,
        raw_probs=np.vstack(raw_probs) if raw_probs else np.zeros((0, 3)),
    )
