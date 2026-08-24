"""The training panel: features in RAM, windows sliced on demand.

Deliberately does **not** materialise ``(n_samples, 60, 12)`` windows to disk. For 400
names over nine years that array is roughly 2.5 GB in float32, while the underlying panel
is ``400 x 2400 x 12 = 46 MB`` and fits in memory with room to spare. Slicing on access
costs nothing measurable and removes an entire class of preprocessing bug, because there
is only one copy of the data and it is the one the features were computed from.

Date batching
-------------
``iter_date_batches`` yields all universe members present on a sampled set of dates
rather than a random selection of rows. That is required, not stylistic: the ranking loss
and every cross-sectional metric are defined within a date, and a batch scattered across
years cannot express either.

Torch is optional. The panel, the window slicing and the batching are pure NumPy, so the
data layer can be tested and inspected without a deep-learning install; ``to_torch``
converts a batch only when a model is actually being trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from finsent.data.features_causal import FEATURE_NAMES

__all__ = ["Sample", "Batch", "PanelDataset"]


@dataclass
class Batch:
    """One date-grouped batch, in NumPy. Convert with ``to_torch`` when training."""

    price: np.ndarray            # (B, L, F)
    text_emb: np.ndarray         # (B, K, E)
    text_mask: np.ndarray        # (B, K) bool
    lag_hours: np.ndarray        # (B, K)
    source_ids: np.ndarray       # (B, K) int
    y_dir: np.ndarray            # (B,) int
    y_ret: np.ndarray            # (B,)
    weights: np.ndarray          # (B,) average uniqueness
    group_ids: np.ndarray        # (B,) date group index, for the ranking loss
    tickers: np.ndarray          # (B,) str
    dates: np.ndarray            # (B,) datetime64

    def __len__(self) -> int:
        return int(self.price.shape[0])

    def slice(self, lo: int, hi: int) -> "Batch":
        """A row-slice of this batch.

        Date-batches are sized in *dates*, but the number of names in the universe
        varies across the panel, so a fixed date count yields a variable row count and
        therefore a variable activation footprint. Capping by rows is what keeps a
        training step inside a fixed memory budget; on an 8 GB card the difference is
        between a run that finishes and a run that dies with cudaErrorUnknown four
        folds in.

        Slicing rows is safe for this model because the loss carries no
        cross-sectional coupling: the ranking term is the only date-coupled component
        and it is disabled in the price-only configuration, so a summed sub-batch
        gradient equals the full-batch gradient up to the mean normalisation.
        """
        def s(a):
            return a[lo:hi] if getattr(a, "size", 0) else a

        return Batch(
            price=s(self.price),
            text_emb=s(self.text_emb),
            text_mask=s(self.text_mask),
            lag_hours=s(self.lag_hours),
            source_ids=s(self.source_ids),
            y_dir=s(self.y_dir),
            y_ret=s(self.y_ret),
            weights=s(self.weights),
            group_ids=s(self.group_ids),
            tickers=s(self.tickers),
            dates=s(self.dates),
        )

    def group_chunks(self, max_rows: int):
        """Yield slices of at most ``max_rows`` rows that never split a date.

        The row cap exists to bound activation memory, but the cross-sectional ranking
        term is computed within a date: splitting a date across two optimiser steps
        would have it rank a fraction of the universe against itself, which is a
        different and easier problem than the one the paper reports. Cutting only on
        group boundaries keeps the cap while leaving each date whole.

        A single date larger than ``max_rows`` is yielded whole rather than split,
        because a partial cross-section is worse than a large one.
        """
        n = len(self)
        if max_rows <= 0 or n <= max_rows:
            yield self
            return

        gids = self.group_ids
        # Date-batches arrive grouped, so boundaries are where group_ids changes.
        edges = [0] + [i for i in range(1, n) if gids[i] != gids[i - 1]] + [n]
        lo = 0
        for k in range(1, len(edges)):
            if edges[k] - lo > max_rows:
                cut = edges[k - 1]
                if cut > lo:                     # at least one whole date fits
                    yield self.slice(lo, cut)
                    lo = cut
                if edges[k] - lo > max_rows:     # one date exceeds the cap on its own
                    yield self.slice(lo, edges[k])
                    lo = edges[k]
        if lo < n:
            yield self.slice(lo, n)

    def row_chunks(self, max_rows: int):
        """Yield row-slices of at most ``max_rows`` rows, or self when it already fits."""
        n = len(self)
        if max_rows <= 0 or n <= max_rows:
            yield self
            return
        for lo in range(0, n, max_rows):
            yield self.slice(lo, min(lo + max_rows, n))

    def to_torch(self, device=None):
        """Materialise as torch tensors. Imported lazily so NumPy users never need it."""
        import torch

        def t(x, dtype):
            return torch.as_tensor(x, dtype=dtype, device=device)

        out = {
            "price": t(np.ascontiguousarray(self.price), torch.float32),
            "y_dir": t(np.ascontiguousarray(self.y_dir), torch.long),
            "y_ret": t(np.ascontiguousarray(self.y_ret), torch.float32),
            "weights": t(np.ascontiguousarray(self.weights), torch.float32),
            "group_ids": t(np.ascontiguousarray(self.group_ids), torch.long),
        }
        # Text tensors are only materialised when the batch actually carries text.
        # With no corpus attached, a (B, K, 768) block of zeros is 88 MB at B = 3595:
        # allocating and copying it to the device every optimiser step cost 4.9 s per
        # step in profiling, against 5 ms of real compute. Absent text, there is
        # nothing to send.
        if self.text_emb.size:
            out["text_embeddings"] = t(np.ascontiguousarray(self.text_emb), torch.float32)
            out["text_mask"] = t(np.ascontiguousarray(self.text_mask), torch.bool)
            out["lag_hours"] = t(np.ascontiguousarray(self.lag_hours), torch.float32)
            out["source_ids"] = t(np.ascontiguousarray(self.source_ids), torch.long)
        return out


@dataclass
class Sample:
    """A single row's worth of the data contract, used by tests and by inspection."""

    ticker: str
    date: pd.Timestamp
    price: np.ndarray
    y_dir: int
    y_ret: float
    weight: float


@dataclass
class PanelDataset:
    """Tidy panel plus cached news embeddings, sliced into causal windows on access.

    Parameters
    ----------
    features
        Tidy frame with ``date``, ``ticker`` and the twelve feature columns.
    labels
        Tidy frame with ``date``, ``ticker``, ``y_dir``, ``y_ret`` and ``weight``.
    embeddings
        Optional ``(n_rows, K, E)`` array aligned to ``index``, plus its mask. When
        absent every sample is treated as a no-news day, which is the correct default:
        the model has a learned representation for exactly that state.
    """

    features: pd.DataFrame
    labels: pd.DataFrame
    lookback: int = 60
    max_headlines: int = 8
    embedding_dim: int = 768
    feature_names: tuple[str, ...] = FEATURE_NAMES

    index: pd.DataFrame = field(init=False, repr=False)
    _panel: np.ndarray = field(init=False, repr=False)
    _dates: pd.DatetimeIndex = field(init=False, repr=False)
    _tickers: list[str] = field(init=False, repr=False)
    _emb: dict[tuple[str, pd.Timestamp], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = (
        field(default_factory=dict, repr=False)
    )

    def __post_init__(self) -> None:
        feats = self.features.copy()
        feats["date"] = pd.to_datetime(feats["date"])
        labs = self.labels.copy()
        labs["date"] = pd.to_datetime(labs["date"])

        self._dates = pd.DatetimeIndex(sorted(feats["date"].unique()))
        self._tickers = sorted(feats["ticker"].astype(str).unique())

        date_pos = {d: i for i, d in enumerate(self._dates)}
        ticker_pos = {t: i for i, t in enumerate(self._tickers)}

        # Dense (n_dates, n_tickers, n_features) cube. NaN marks "not in the universe",
        # which keeps point-in-time membership explicit rather than implicit.
        self._panel = np.full(
            (len(self._dates), len(self._tickers), len(self.feature_names)), np.nan, np.float32
        )
        di = feats["date"].map(date_pos).to_numpy()
        ti = feats["ticker"].astype(str).map(ticker_pos).to_numpy()
        self._panel[di, ti] = feats[list(self.feature_names)].to_numpy(dtype=np.float32)

        merged = labs.merge(
            feats[["date", "ticker"]].drop_duplicates(), on=["date", "ticker"], how="inner"
        )
        merged["date_pos"] = merged["date"].map(date_pos)
        merged["ticker_pos"] = merged["ticker"].astype(str).map(ticker_pos)
        merged = merged.dropna(subset=["date_pos", "ticker_pos", "y_dir", "y_ret"])

        # A window needs `lookback` prior observations, so earlier positions are dropped
        # rather than zero-padded: padding would teach the model that a short history is
        # a real state, and it never is at inference.
        merged = merged[merged["date_pos"] >= self.lookback - 1]
        if "weight" not in merged.columns:
            merged["weight"] = 1.0

        self.index = merged.reset_index(drop=True)

    # -- basics ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._dates

    @property
    def tickers(self) -> list[str]:
        return self._tickers

    def attach_embeddings(
        self,
        key: tuple[str, pd.Timestamp],
        embeddings: np.ndarray,
        mask: np.ndarray,
        lag_hours: np.ndarray,
        source_ids: np.ndarray,
    ) -> None:
        """Register cached FinBERT vectors for one (ticker, date)."""
        self._emb[(str(key[0]), pd.Timestamp(key[1]))] = (
            np.asarray(embeddings, dtype=np.float32),
            np.asarray(mask, dtype=bool),
            np.asarray(lag_hours, dtype=np.float32),
            np.asarray(source_ids, dtype=np.int64),
        )

    def window(self, date_pos: int, ticker_pos: int) -> np.ndarray:
        """The causal ``(L, F)`` window ending at ``date_pos`` inclusive."""
        lo = date_pos - self.lookback + 1
        w = self._panel[lo : date_pos + 1, ticker_pos, :]
        return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    def windows(self, date_pos: np.ndarray, ticker_pos: np.ndarray) -> np.ndarray:
        """Vectorised batch of causal ``(B, L, F)`` windows.

        Slicing one window per row in a Python loop is the single largest cost in the
        training loop: a date-batch here is the whole cross-section on several dates,
        so a batch of 24 dates at 139 names is 3,336 individual slices rebuilt every
        optimiser step. Gathering them with one fancy-index operation instead moves the
        work into NumPy and leaves the GPU, rather than the interpreter, as the
        bottleneck.

        The gather is backward-only by construction: row ``i`` reads offsets
        ``[-L+1, 0]`` relative to its own ``date_pos``, so no future bar can enter.
        """
        dp = np.asarray(date_pos, dtype=np.int64)
        tp = np.asarray(ticker_pos, dtype=np.int64)
        offsets = np.arange(-self.lookback + 1, 1, dtype=np.int64)
        rows = dp[:, None] + offsets[None, :]              # (B, L)
        np.clip(rows, 0, self._panel.shape[0] - 1, out=rows)
        out = self._panel[rows, tp[:, None], :]            # (B, L, F)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def sample(self, row: int) -> Sample:
        r = self.index.iloc[row]
        return Sample(
            ticker=str(r["ticker"]),
            date=pd.Timestamp(r["date"]),
            price=self.window(int(r["date_pos"]), int(r["ticker_pos"])),
            y_dir=int(r["y_dir"]),
            y_ret=float(r["y_ret"]),
            weight=float(r["weight"]),
        )

    # -- batching --------------------------------------------------------------------
    @property
    def has_text(self) -> bool:
        """True when any cached headline embeddings have been attached."""
        return bool(self._emb)

    def _empty_text(self, n: int) -> tuple[np.ndarray, ...]:
        """Placeholder text arrays.

        When no embeddings are attached the placeholders are genuinely empty rather
        than a ``(n, K, E)`` block of zeros, which at realistic date-batch sizes is
        tens of megabytes allocated and copied per step for no purpose.
        """
        k = self.max_headlines if self.has_text else 0
        e = self.embedding_dim if self.has_text else 0
        return (
            np.zeros((n, k, e), np.float32),
            np.zeros((n, k), bool),
            np.zeros((n, k), np.float32),
            np.zeros((n, k), np.int64),
        )

    def build_batch(self, rows: Sequence[int]) -> Batch:
        rows = np.asarray(rows, dtype=int)
        sub = self.index.iloc[rows]
        n = len(sub)

        price = self.windows(
            sub["date_pos"].to_numpy(dtype=np.int64),
            sub["ticker_pos"].to_numpy(dtype=np.int64),
        )

        emb, mask, lag, src = self._empty_text(n)
        if self._emb:
            for i, (ticker, date) in enumerate(zip(sub["ticker"], sub["date"])):
                cached = self._emb.get((str(ticker), pd.Timestamp(date)))
                if cached is None:
                    continue
                e, m, l, s = cached
                k = min(len(m), self.max_headlines)
                emb[i, :k] = e[:k]
                mask[i, :k] = m[:k]
                lag[i, :k] = l[:k]
                src[i, :k] = s[:k]

        codes = pd.factorize(sub["date"].to_numpy())[0]

        return Batch(
            price=price,
            text_emb=emb,
            text_mask=mask,
            lag_hours=lag,
            source_ids=src,
            y_dir=sub["y_dir"].to_numpy(dtype=np.int64),
            y_ret=sub["y_ret"].to_numpy(dtype=np.float32),
            weights=sub["weight"].to_numpy(dtype=np.float32),
            group_ids=codes.astype(np.int64),
            tickers=sub["ticker"].to_numpy(),
            dates=sub["date"].to_numpy(),
        )

    def iter_date_batches(
        self,
        dates_per_batch: int = 16,
        shuffle: bool = True,
        seed: int = 0,
        subset: pd.DatetimeIndex | None = None,
    ) -> Iterator[Batch]:
        """Yield batches grouped by date.

        ``shuffle`` reorders whole *dates*, never rows within a date, and never mixes a
        date's members across batches. Shuffling rows would silently destroy the
        cross-sectional structure the ranking loss and the IC both depend on.
        """
        available = pd.DatetimeIndex(sorted(pd.to_datetime(self.index["date"]).unique()))
        if subset is not None:
            available = available.intersection(pd.DatetimeIndex(pd.to_datetime(subset)))
        if len(available) == 0:
            return

        order = np.arange(len(available))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)

        by_date = {d: g for d, g in self.index.groupby(pd.to_datetime(self.index["date"])).groups.items()}
        for start in range(0, len(order), dates_per_batch):
            chunk = [available[i] for i in order[start : start + dates_per_batch]]
            rows = np.concatenate([np.asarray(by_date[d]) for d in chunk if d in by_date])
            if rows.size:
                yield self.build_batch(rows)

    def rows_for_dates(self, dates: pd.DatetimeIndex) -> np.ndarray:
        keep = pd.to_datetime(self.index["date"]).isin(pd.DatetimeIndex(pd.to_datetime(dates)))
        return np.nonzero(keep.to_numpy())[0]

    def describe(self) -> str:
        return (
            f"PanelDataset: {len(self):,d} samples | {len(self._tickers)} tickers | "
            f"{len(self._dates)} sessions | window {self.lookback}x{len(self.feature_names)} | "
            f"news cached for {len(self._emb):,d} (ticker, date) keys"
        )
