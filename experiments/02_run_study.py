"""Walk-forward study on the real price panel.

    python experiments/02_run_study.py --models all --seeds 0 1 2 3 4

Drives every model through the identical purged, embargoed walk-forward
protocol of ``finsent.training.walkforward``, so a comparison cannot use a
different split, a different calibration set or a different trading rule.
Writes one out-of-sample prediction panel per model to ``runs/study/``.

Scope
-----
The dumps shipped with this repository contain no usable text: of 5,391 news
rows, 72 are non-empty and those carry 64 template titles across 9 distinct
timestamps. The text modality is therefore **not evaluated**, and the
cross-modal model is run in its forced price-only state --- the configuration
that modality dropout exists to support. Nothing is fabricated to fill the gap.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finsent.config import load_config  # noqa: E402
from finsent.data.features_causal import FEATURE_NAMES  # noqa: E402
from finsent.training.walkforward import FoldPrediction, run_walk_forward  # noqa: E402

PANEL = Path("data/cache/study_panel.parquet")
OUT = Path("runs/study")

LOGIT_FEATURES = ("ret_5", "rsi_14", "ewma_vol_20", "mom_12_1", "bb_pctb")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _logits_from_score(score: np.ndarray, sharpness: float = 1.0) -> np.ndarray:
    """Map a scalar directional score to 3-class logits (DOWN, NEUTRAL, UP)."""
    s = np.nan_to_num(np.asarray(score, dtype=float).ravel(), nan=0.0) * sharpness
    return np.column_stack([-s, np.zeros_like(s), s])


def _pack(train, val, test, logits_val, logits_test, mu_test, sigma2_test, gate=None):
    return FoldPrediction(
        logits_val=logits_val,
        y_val=val["y_dir"].to_numpy(dtype=int),
        logits_test=logits_test,
        y_test=test["y_dir"].to_numpy(dtype=int),
        mu_test=np.nan_to_num(mu_test, nan=0.0),
        sigma2_test=np.maximum(sigma2_test, 1e-10),
        fwd_ret_test=test["fwd_ret"].to_numpy(dtype=float),
        dates_test=test["date"].to_numpy(),
        tickers_test=test["ticker"].to_numpy(),
        gate_test=gate,
    )


def _zscore(train: pd.DataFrame, other: pd.DataFrame, cols):
    X = train[list(cols)].to_numpy(dtype=float)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd[sd < 1e-12] = 1.0
    Z = (other[list(cols)].to_numpy(dtype=float) - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------------------
def make_null_signal():
    def fit_predict(train, val, test, seed, fold):
        rng = np.random.default_rng(seed + 991 * fold.index)
        sv, st = rng.standard_normal(len(val)), rng.standard_normal(len(test))
        sd = float(np.std(train["fwd_ret"])) or 1e-4
        return _pack(train, val, test, _logits_from_score(sv), _logits_from_score(st),
                     st * sd * 1e-3, np.full(len(test), sd**2))
    return fit_predict


def make_buy_and_hold():
    """Always long, equally weighted: a constant positive score for every name."""
    def fit_predict(train, val, test, seed, fold):
        sd = float(np.std(train["fwd_ret"])) or 1e-4
        return _pack(train, val, test,
                     _logits_from_score(np.full(len(val), 0.5)),
                     _logits_from_score(np.full(len(test), 0.5)),
                     np.full(len(test), 0.25 * sd), np.full(len(test), sd**2))
    return fit_predict


def make_tsmom(column: str = "mom_12_1"):
    def fit_predict(train, val, test, seed, fold):
        sd = float(np.std(train["fwd_ret"])) or 1e-4
        sv = np.nan_to_num(val[column].to_numpy(dtype=float))
        st = np.nan_to_num(test[column].to_numpy(dtype=float))
        return _pack(train, val, test, _logits_from_score(sv, 3.0),
                     _logits_from_score(st, 3.0), st * sd, np.full(len(test), sd**2))
    return fit_predict


def make_logit5(columns=LOGIT_FEATURES, epochs: int = 600, lr: float = 0.5, l2: float = 1e-3):
    """Multinomial logistic regression, plain gradient descent, 18 parameters."""
    def fit_predict(train, val, test, seed, fold):
        Xtr = np.column_stack([np.ones(len(train)), _zscore(train, train, columns)])
        y = train["y_dir"].to_numpy(dtype=int)
        onehot = np.zeros((len(y), 3))
        onehot[np.arange(len(y)), y] = 1.0
        w = np.zeros((Xtr.shape[1], 3))
        sw = train["weight"].to_numpy(dtype=float)
        sw = sw / max(sw.sum(), 1e-12)
        for _ in range(epochs):
            z = Xtr @ w
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            w -= lr * (Xtr.T @ ((p - onehot) * sw[:, None]) + l2 * w)

        def logits(df):
            X = np.column_stack([np.ones(len(df)), _zscore(train, df, columns)])
            return X @ w

        lt = logits(test)
        pt = np.exp(lt - lt.max(1, keepdims=True))
        pt /= pt.sum(1, keepdims=True)
        sd = float(np.std(train["fwd_ret"])) or 1e-4
        return _pack(train, val, test, logits(val), lt,
                     (pt[:, 2] - pt[:, 0]) * sd, np.full(len(test), sd**2))
    return fit_predict


def make_gbm(columns=FEATURE_NAMES, max_iter: int = 200):
    """Gradient-boosted trees on the same twelve features -- the referee's benchmark."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    def fit_predict(train, val, test, seed, fold):
        model = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=0.06, max_depth=4,
            l2_regularization=1.0, early_stopping=False, random_state=seed,
        )
        model.fit(train[list(columns)].to_numpy(dtype=float),
                  train["y_dir"].to_numpy(dtype=int),
                  sample_weight=train["weight"].to_numpy(dtype=float))

        def logits(df):
            p = np.clip(model.predict_proba(df[list(columns)].to_numpy(dtype=float)), 1e-9, 1)
            return np.log(p)

        lt = logits(test)
        pt = np.exp(lt)
        pt /= pt.sum(1, keepdims=True)
        sd = float(np.std(train["fwd_ret"])) or 1e-4
        return _pack(train, val, test, logits(val), lt,
                     (pt[:, 2] - pt[:, 0]) * sd, np.full(len(test), sd**2))
    return fit_predict


# --------------------------------------------------------------------------------------
# FinSentNet-C, price-only configuration
# --------------------------------------------------------------------------------------
def make_finsentnet(cfg, frame: pd.DataFrame, epochs: int = 12, lr: float = 3e-4,
                    dates_per_batch: int = 24, device: str | None = None,
                    max_rows: int = 1536):
    """Train the cross-modal architecture in its forced price-only state.

    The dataset is built **once over the whole panel** and blocks are selected by
    date. Building it per block would be wrong: a window needs ``lookback``
    prior sessions, so a 126-session test block constructed in isolation would
    silently lose its first 59 sessions -- nearly half the evaluation data.
    Selecting by date instead keeps every block whole while the windows still
    only ever reach backwards, so no future information enters.
    """
    import torch

    from finsent.data.panel import PanelDataset
    from finsent.models.finsentnet_c import FinSentNetC
    from finsent.training.objectives import CompositeObjective, LossWeights

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = dev.type == "cuda" and torch.cuda.is_bf16_supported()

    dataset = PanelDataset(
        frame[["date", "ticker", *FEATURE_NAMES]],
        frame[["date", "ticker", "y_dir", "y_ret", "weight"]],
        lookback=cfg.data.lookback_L,
        max_headlines=cfg.data.K,
        embedding_dim=cfg.data.embedding_dim,
    )
    print(f"    dataset: {dataset.describe()}  device={dev}")

    def fit_predict(train, val, test, seed, fold):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = FinSentNetC.from_config(cfg).to(dev)
        model.use_text = False   # no text exists in this corpus; forced price-only

        objective = CompositeObjective(LossWeights(
            lambda_reg=cfg.loss.lambda_reg,
            lambda_cal=cfg.loss.lambda_cal,
            lambda_rank=0.0,     # O(n^2) per date; prohibitive at 139 names x 13 folds
            sigma_warmup_epochs=cfg.loss.sigma_warmup_epochs,
        )).to(dev)
        objective.set_class_weights_from(
            torch.as_tensor(train["y_dir"].to_numpy(dtype=np.int64)))

        opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=cfg.train.weight_decay)

        train_dates = pd.DatetimeIndex(sorted(pd.to_datetime(train["date"]).unique()))
        model.train()
        for epoch in range(epochs):
            for batch in dataset.iter_date_batches(dates_per_batch, shuffle=True,
                                                   seed=seed + epoch, subset=train_dates):
                for sub in batch.row_chunks(max_rows):
                    t = sub.to_torch(dev)
                    opt.zero_grad(set_to_none=True)
                    # Activations in bf16, loss arithmetic in fp32. bf16 carries fp32's
                    # exponent range, so no gradient scaler is needed and the Gaussian
                    # NLL cannot overflow through log-variance; the halved activation
                    # footprint is what makes a 60x128 recurrent stack fit next to a
                    # 4-block TCN on an 8 GB card.
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        out = model(price=t["price"])
                    loss, _ = objective(
                        out.heads.logits.float(), out.heads.mu.float(),
                        out.heads.logvar.float(),
                        t["y_dir"], t["y_ret"], t["weights"], None, epoch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    opt.step()

        @torch.no_grad()
        def infer(block: pd.DataFrame):
            model.eval()
            dates = pd.DatetimeIndex(sorted(pd.to_datetime(block["date"]).unique()))
            chunks = []
            for batch in dataset.iter_date_batches(dates_per_batch, shuffle=False,
                                                   subset=dates):
                for sub in batch.row_chunks(max_rows):
                    t = sub.to_torch(dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        out = model(price=t["price"])
                    chunks.append(pd.DataFrame({
                        "date": pd.to_datetime(sub.dates),
                        "ticker": sub.tickers.astype(str),
                        "l0": out.heads.logits[:, 0].float().cpu().numpy(),
                        "l1": out.heads.logits[:, 1].float().cpu().numpy(),
                        "l2": out.heads.logits[:, 2].float().cpu().numpy(),
                        "mu": out.heads.mu.float().cpu().numpy(),
                        "var": out.heads.sigma2.float().cpu().numpy(),
                        "gate": out.fusion.gate_mean.float().cpu().numpy(),
                    }))
            preds = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            keys = block[["date", "ticker"]].copy()
            keys["date"] = pd.to_datetime(keys["date"])
            keys["ticker"] = keys["ticker"].astype(str)
            merged = keys.merge(preds, on=["date", "ticker"], how="left")
            logits = merged[["l0", "l1", "l2"]].to_numpy(dtype=float)
            return (np.nan_to_num(logits),
                    np.nan_to_num(merged["mu"].to_numpy(dtype=float)),
                    np.nan_to_num(merged["var"].to_numpy(dtype=float), nan=1e-6),
                    np.nan_to_num(merged["gate"].to_numpy(dtype=float), nan=0.5))

        lv, _, _, _ = infer(val)
        lt, mt, vt, gt = infer(test)
        packed = _pack(train, val, test, lv, lt, mt, np.maximum(vt, 1e-10), gate=gt)

        # 13 folds x 3 seeds means 39 models are built in one process. Without an
        # explicit release the allocator holds each one's arena and fragments, which is
        # how a run that fits comfortably on fold 1 dies on fold 4.
        del model, opt, objective
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return packed

    return fit_predict


MODELS = {
    "null_signal": lambda cfg: make_null_signal(),
    "buy_and_hold": lambda cfg: make_buy_and_hold(),
    "tsmom": lambda cfg: make_tsmom(),
    "logit5": lambda cfg: make_logit5(),
    "gbm": lambda cfg: make_gbm(),
    "finsentnet_c": make_finsentnet,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--panel", default=str(PANEL))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-rows", type=int, default=1536,
                    help="cap on rows per optimiser step; lower it if the device runs out")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave models whose panel is already on disk alone. The network "
                         "takes hours and the baselines take minutes, so a failure late "
                         "in the run should not cost the part that already succeeded.")
    args = ap.parse_args()

    cfg = load_config()
    frame = pd.read_parquet(args.panel)
    print(f"panel: {len(frame):,} rows, {frame.ticker.nunique()} tickers, "
          f"{frame.date.nunique()} sessions, "
          f"{frame.date.min().date()}..{frame.date.max().date()}\n")

    names = list(MODELS) if args.models == ["all"] else args.models
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in names:
        if name not in MODELS:
            print(f"  unknown model {name!r}; skipping")
            continue
        if args.skip_existing and (out / f"{name}.parquet").exists():
            print(f"=== {name}: already on disk, skipping ===" + chr(10))
            continue
        seeds = args.seeds if name not in ("buy_and_hold", "tsmom") else args.seeds[:1]
        print(f"=== {name} (seeds {seeds}) ===")
        t0 = time.time()
        builder = MODELS[name]
        fp = (builder(cfg, frame, epochs=args.epochs, max_rows=args.max_rows)
              if name == "finsentnet_c" else builder(cfg))
        result = run_walk_forward(frame, fp, cfg, seeds=seeds)
        print(f"  {result.summary()}")
        print(f"  elapsed {time.time()-t0:.0f}s")
        result.panel.to_parquet(out / f"{name}.parquet", index=False)
        result.calibration.to_csv(out / f"{name}_calibration.csv", index=False)
        result.coverage.to_csv(out / f"{name}_coverage.csv", index=False)
        result.fold_metrics.to_csv(out / f"{name}_folds.csv", index=False)
        print()

    print(f"written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
