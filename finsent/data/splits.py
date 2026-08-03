"""Purged, embargoed walk-forward splits (Lopez de Prado, 2018, ch. 7 and 12).

This is the module a quant interviewer opens first, and it is the one the V2 codebase
did not have. Three defects are corrected:

1. **A single 60/20/20 temporal split was called "walk-forward".** It is not.
   Walk-forward means *rolling retraining*: the model is refitted on each new window and
   evaluated only on data after that window. ``PurgedWalkForward`` produces 18 such
   folds over the configured period.

2. **No purging.** With a 60-day input window and an ``h``-day forward label, adjacent
   samples share label windows, so a sample near the train/test boundary has a label
   determined partly by prices inside the test period. Purging removes every training
   sample whose label window overlaps the evaluation window.

3. **No embargo.** Serial correlation carries information a short distance past the
   boundary even after purging; an embargo of ``embargo_pct`` of the sample kills it.

Layout of one fold::

    |<---- TRAIN (expanding) ---->|purge h|<- INNER VAL ->|purge h + embargo|<- TEST ->|
                                                  ^                              ^
                            early stopping, lambda search,            touched exactly
                            vector scaling, conformal quantiles       once, at report time

The inner-validation block is what removes the V2 triple-dipping defect, in which loss
weights, early stopping and temperature were all fitted on the same held-out data.

Validation of the module itself lives in ``tests/test_no_leakage.py``: a deliberate leak
is planted in synthetic data, and the splitter must fail to learn it. A splitter that
cannot catch a planted leak cannot be trusted to catch an accidental one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "Fold",
    "PurgedWalkForward",
    "CombinatorialPurgedCV",
    "purge_indices",
    "embargo_indices",
    "TRADING_DAYS_PER_MONTH",
    "TRADING_DAYS_PER_YEAR",
]

TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold, expressed as positional slices into a sorted date index."""

    index: int
    train: tuple[int, int]
    inner_val: tuple[int, int]
    test: tuple[int, int]
    purge_days: int
    embargo_days: int

    def slice(self, name: str) -> slice:
        lo, hi = getattr(self, name)
        return slice(lo, hi)

    def dates(self, date_index: pd.DatetimeIndex, name: str) -> pd.DatetimeIndex:
        return date_index[self.slice(name)]

    def mask(self, dates: pd.Series | np.ndarray, date_index: pd.DatetimeIndex, name: str):
        """Boolean mask selecting rows of a tidy panel that fall in this block."""
        block = set(self.dates(date_index, name))
        return pd.Series(pd.to_datetime(pd.Series(np.asarray(dates)))).isin(block).to_numpy()

    def describe(self, date_index: pd.DatetimeIndex) -> str:
        def span(name: str) -> str:
            d = self.dates(date_index, name)
            return f"{d[0].date()}..{d[-1].date()}" if len(d) else "(empty)"

        return (
            f"fold {self.index:>2d} | train {span('train')} "
            f"| val {span('inner_val')} | test {span('test')} "
            f"| purge {self.purge_days}d embargo {self.embargo_days}d"
        )


@dataclass
class PurgedWalkForward:
    """Rolling refit with purge and embargo on both internal boundaries.

    Parameters
    ----------
    dates
        Sorted unique session dates covering the whole study period.
    horizon
        Label horizon ``h`` in sessions. The purge width equals this: a training sample
        at position ``p`` has a label window ``[p, p+h]``, so it must be dropped when
        ``p + h >= evaluation_start``.
    embargo_pct
        Extra buffer before each evaluation block, as a fraction of the full sample.
    """

    dates: pd.DatetimeIndex
    horizon: int = 5
    embargo_pct: float = 0.01
    train_min_years: int = 3
    inner_val_months: int = 6
    test_months: int = 6
    refit_every_months: int = 6
    train_mode: str = "expanding"

    def __post_init__(self) -> None:
        self.dates = pd.DatetimeIndex(pd.to_datetime(self.dates)).sort_values().unique()
        if self.train_mode not in {"expanding", "rolling"}:
            raise ValueError(f"train_mode must be expanding|rolling, got {self.train_mode!r}")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1; a zero purge is not a purge")

    # -- geometry --------------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.dates)

    @property
    def embargo_days(self) -> int:
        return int(np.ceil(self.embargo_pct * self.n))

    @property
    def _train_min_days(self) -> int:
        return self.train_min_years * TRADING_DAYS_PER_YEAR

    def __iter__(self) -> Iterator[Fold]:
        h = self.horizon
        emb = self.embargo_days
        val_len = self.inner_val_months * TRADING_DAYS_PER_MONTH
        test_len = self.test_months * TRADING_DAYS_PER_MONTH
        step = self.refit_every_months * TRADING_DAYS_PER_MONTH

        first_test_start = self._train_min_days + h + val_len + h + emb

        fold = 0
        test_start = first_test_start
        while test_start + test_len <= self.n:
            val_end = test_start - h - emb
            val_start = val_end - val_len
            train_end = val_start - h
            train_start = 0 if self.train_mode == "expanding" else max(
                0, train_end - self._train_min_days
            )

            if train_end - train_start < self._train_min_days // 2:
                break

            yield Fold(
                index=fold,
                train=(train_start, train_end),
                inner_val=(val_start, val_end),
                test=(test_start, test_start + test_len),
                purge_days=h,
                embargo_days=emb,
            )
            fold += 1
            test_start += step

    def folds(self) -> list[Fold]:
        return list(self)

    def n_folds(self) -> int:
        return len(self.folds())

    def summary(self) -> str:
        folds = self.folds()
        if not folds:
            return (
                "PurgedWalkForward produced 0 folds: the period is shorter than "
                f"train_min_years={self.train_min_years} plus the evaluation blocks."
            )
        head = "\n".join(f.describe(self.dates) for f in folds[:3])
        tail = folds[-1].describe(self.dates)
        return f"{len(folds)} folds over {self.n} sessions\n{head}\n  ...\n{tail}"

    # -- convenience -----------------------------------------------------------------
    def split_positions(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Positional index arrays ``(train, inner_val, test)`` per fold."""
        out = []
        for f in self:
            out.append(
                (
                    np.arange(*f.train),
                    np.arange(*f.inner_val),
                    np.arange(*f.test),
                )
            )
        return out

    def split_frame(self, frame: pd.DataFrame, date_column: str = "date"):
        """Yield ``(fold, train_df, val_df, test_df)`` for a tidy panel."""
        d = pd.to_datetime(frame[date_column])
        for f in self:
            train_dates = set(self.dates[f.slice("train")])
            val_dates = set(self.dates[f.slice("inner_val")])
            test_dates = set(self.dates[f.slice("test")])
            yield (
                f,
                frame[d.isin(train_dates)],
                frame[d.isin(val_dates)],
                frame[d.isin(test_dates)],
            )


def purge_indices(
    train_positions: np.ndarray,
    eval_positions: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Drop training positions whose label window reaches into the evaluation block.

    General form of the purge for a fixed horizon: a sample at ``p`` is contaminated
    when ``p + horizon >= min(eval_positions)`` and ``p < max(eval_positions)``.
    """
    if eval_positions.size == 0:
        return train_positions
    lo, hi = int(eval_positions.min()), int(eval_positions.max())
    contaminated = (train_positions + horizon >= lo) & (train_positions <= hi)
    return train_positions[~contaminated]


def embargo_indices(
    train_positions: np.ndarray,
    eval_positions: np.ndarray,
    embargo: int,
) -> np.ndarray:
    """Drop training positions within ``embargo`` sessions after the evaluation block."""
    if eval_positions.size == 0 or embargo <= 0:
        return train_positions
    hi = int(eval_positions.max())
    return train_positions[(train_positions <= hi) | (train_positions > hi + embargo)]


@dataclass
class CombinatorialPurgedCV:
    """Combinatorial purged cross-validation (Lopez de Prado, 2018, ch. 12).

    Splits the sample into ``n_groups`` contiguous blocks and evaluates on every
    combination of ``n_test_groups`` of them, purging and embargoing around each test
    block. Used only for the robustness section: it yields many backtest paths and
    therefore a distribution of Sharpe ratios rather than a single number, which is the
    honest way to show how much of a result is path-dependent.
    """

    dates: pd.DatetimeIndex
    n_groups: int = 6
    n_test_groups: int = 2
    horizon: int = 5
    embargo_pct: float = 0.01

    def __post_init__(self) -> None:
        self.dates = pd.DatetimeIndex(pd.to_datetime(self.dates)).sort_values().unique()
        if self.n_test_groups >= self.n_groups:
            raise ValueError("n_test_groups must be smaller than n_groups")

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        from itertools import combinations

        n = len(self.dates)
        bounds = np.linspace(0, n, self.n_groups + 1).astype(int)
        groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(self.n_groups)]
        embargo = int(np.ceil(self.embargo_pct * n))

        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test = np.concatenate([groups[i] for i in combo])
            train = np.concatenate([groups[i] for i in range(self.n_groups) if i not in combo])
            train = purge_indices(train, test, self.horizon)
            train = embargo_indices(train, test, embargo)
            if train.size and test.size:
                yield np.sort(train), np.sort(test)

    def n_paths(self) -> int:
        from math import comb

        return comb(self.n_groups, self.n_test_groups)
