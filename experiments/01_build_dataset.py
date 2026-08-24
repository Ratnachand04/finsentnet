"""Build the study dataset from the real price dumps.

    python experiments/01_build_dataset.py

Produces ``data/cache/study_panel.parquet``: a tidy panel of causal,
cross-sectionally ranked features, volatility-scaled labels, uniqueness weights
and point-in-time universe membership, ready for the walk-forward study.

Every contract in SPEC.md is applied here and nowhere else, so there is a single
place to audit: features are lagged one session for open execution, labels are
open-to-open over the configured horizon, the universe is screened
point-in-time, and label overlap is quantified rather than ignored.
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
from finsent.data.features_causal import FEATURE_NAMES, build_feature_panel  # noqa: E402
from finsent.data.labeling import class_balance, vol_scaled_labels  # noqa: E402
from finsent.data.sql_dump_loader import (  # noqa: E402
    DumpConfig,
    load_price_panel,
    panel_to_ohlcv_dict,
    panel_to_wide,
)
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

CACHE = Path("data/cache/study_panel.parquet")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-names", type=int, default=150)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--out", default=str(CACHE))
    ap.add_argument("--rebuild-prices", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    h = cfg.data.horizon_h
    print(f"config: {cfg.describe()}\n")

    t0 = time.time()
    prices = load_price_panel(DumpConfig(), rebuild=args.rebuild_prices)
    print(f"price panel: {len(prices):,} rows, {prices.ticker.nunique()} tickers, "
          f"{prices.date.min().date()}..{prices.date.max().date()}  [{time.time()-t0:.0f}s]")

    # ---- point-in-time universe -------------------------------------------------
    close = panel_to_wide(prices, "close")
    volume = panel_to_wide(prices, "volume")
    ucfg = UniverseConfig(n_names=args.n_names, min_price=5.0, min_adv_usd=5.0e6,
                          lookback_days=60, rebalance="monthly")
    universe = build_liquidity_universe(close, volume, ucfg)
    stats = membership_stats(universe)
    print(f"universe: mean {stats['mean_size']:.0f} names, "
          f"{stats['n_unique_tickers']} distinct, "
          f"{stats['n_dropped_from_first_date']} dropped since first date, "
          f"survivorship check passed={stats['survivorship_check_passed']}")

    # ---- features: causal, ranked cross-sectionally, lagged one session ---------
    t0 = time.time()
    ohlcv = panel_to_ohlcv_dict(prices)
    features = build_feature_panel(ohlcv, cross_sectional=True, lag_sessions=1)
    print(f"features: {len(features):,} rows x {len(FEATURE_NAMES)}  [{time.time()-t0:.0f}s]")

    # ---- labels: open-to-open, volatility-scaled band ---------------------------
    t0 = time.time()
    rows = []
    for ticker, frame in ohlcv.items():
        res = vol_scaled_labels(frame["open"], horizon=h, k_band=cfg.data.k_band,
                                vol_span=cfg.data.vol_span)
        # Per-period return actually earned by holding from open t to open t+1.
        nxt = frame["open"].shift(-1) / frame["open"] - 1.0
        t1 = uniqueness_from_horizon(len(frame), h)
        rows.append(pd.DataFrame({
            "date": frame.index,
            "ticker": ticker,
            "y_dir": res.y.to_numpy(),
            "y_ret": res.forward_return.to_numpy(),
            "fwd_ret": res.forward_return.to_numpy(),
            "next_ret": nxt.to_numpy(),
            "weight": average_uniqueness(t1),
        }))
    labels = pd.concat(rows, ignore_index=True)
    print(f"labels: {len(labels):,} rows  [{time.time()-t0:.0f}s]")

    # ---- assemble ---------------------------------------------------------------
    frame = features.merge(labels, on=["date", "ticker"], how="inner")
    frame = frame.merge(universe, on=["date", "ticker"], how="inner")
    frame = frame[frame["date"] >= args.start]
    frame = frame.dropna(subset=["y_dir", "fwd_ret", "next_ret", *FEATURE_NAMES])
    frame["y_dir"] = frame["y_dir"].astype(int)
    frame = frame.sort_values(["date", "ticker"]).reset_index(drop=True)

    bal = class_balance(frame["y_dir"])
    ess = effective_sample_size(frame["weight"])
    print()
    print(f"study panel: {len(frame):,} rows, {frame.ticker.nunique()} tickers, "
          f"{frame.date.nunique()} sessions, {frame.date.min().date()}..{frame.date.max().date()}")
    print(f"class balance: DOWN {bal['DOWN']:.3f}  NEUTRAL {bal['NEUTRAL']:.3f}  UP {bal['UP']:.3f}")
    print(f"effective sample size: {ess:,.0f} of {len(frame):,} rows "
          f"(factor {len(frame)/max(ess,1):.1f})")
    print(f"mean names per session: {frame.groupby('date').size().mean():.0f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
