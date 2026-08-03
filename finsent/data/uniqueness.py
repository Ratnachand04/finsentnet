"""Sample uniqueness for overlapping labels (Lopez de Prado, 2018, ch. 4).

The problem, stated plainly: with a 60-day input window and an ``h``-day forward label,
consecutive samples are near-duplicates. Their labels are computed from overlapping
price paths, so the effective number of independent observations is far below the row
count. Two consequences follow, and V2 addressed neither:

* Any standard error computed as ``1/sqrt(n)`` is optimistic, often by a factor of two
  or three.
* The network can reach low training loss by memorising a much smaller set of
  effectively distinct episodes than the row count suggests.

``average_uniqueness`` computes, for each sample, the average over its label window of
``1 / (number of concurrent labels)``. Those values become per-sample weights in the
loss and give the *effective* sample size that the paper's capacity argument uses.

Interface convention
--------------------
All functions take ``t1`` as a **positional** array: ``t1[i]`` is the index position at
which sample ``i``'s label resolves, so the label spans positions ``[i, t1[i]]``
inclusive. ``NaN`` marks a sample whose label never resolved (end of sample). Keeping
this positional avoids the date-index ambiguity that makes overlap code subtly wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "concurrency",
    "average_uniqueness",
    "effective_sample_size",
    "sequential_bootstrap",
    "uniqueness_from_horizon",
]

_EPS = 1e-12


def _as_positional(t1) -> np.ndarray:
    return np.asarray(pd.Series(t1).to_numpy(), dtype=float)


def uniqueness_from_horizon(n_samples: int, horizon: int) -> np.ndarray:
    """Convenience ``t1`` for the fixed-horizon case: ``t1[i] = i + horizon``."""
    t1 = np.arange(n_samples, dtype=float) + horizon
    t1[t1 > n_samples - 1] = np.nan
    return t1


def concurrency(t1, n_periods: int | None = None) -> np.ndarray:
    """Number of labels live at each period, by difference-array accumulation."""
    t1 = _as_positional(t1)
    n_samples = t1.size
    finite = np.isfinite(t1)
    if not finite.any():
        return np.zeros(n_periods or n_samples, dtype=float)

    n = int(n_periods or max(n_samples, int(np.nanmax(t1)) + 1))
    delta = np.zeros(n + 1, dtype=float)

    starts = np.nonzero(finite)[0]
    ends = np.minimum(t1[finite].astype(int), n - 1)
    keep = ends >= starts
    np.add.at(delta, starts[keep], 1.0)
    np.add.at(delta, ends[keep] + 1, -1.0)

    return np.cumsum(delta)[:n]


def average_uniqueness(t1, n_periods: int | None = None) -> np.ndarray:
    """Average uniqueness per sample: ``mean_t 1/concurrency(t)`` over its label window.

    A value of 1.0 means the label is completely non-overlapping. With ``h = 5`` on a
    daily panel the typical value is near ``1/5 = 0.2`` -- which is exactly the factor
    by which a naive standard error is wrong.
    """
    t1 = _as_positional(t1)
    n_samples = t1.size
    finite = np.isfinite(t1)
    out = np.full(n_samples, np.nan)
    if not finite.any():
        return out

    n = int(n_periods or max(n_samples, int(np.nanmax(t1)) + 1))
    conc = np.maximum(concurrency(t1, n_periods=n), 1.0)
    inv_cum = np.concatenate([[0.0], np.cumsum(1.0 / conc)])

    idx = np.nonzero(finite)[0]
    ends = np.minimum(t1[finite].astype(int), n - 1)
    keep = ends >= idx
    idx, ends = idx[keep], ends[keep]

    width = (ends - idx + 1).astype(float)
    out[idx] = (inv_cum[ends + 1] - inv_cum[idx]) / width
    return out


def effective_sample_size(uniqueness) -> float:
    """Sum of average uniqueness: the number of *independent* samples you really have.

    This is the denominator in the paper's capacity argument. Quoting a raw row count
    beside a parameter count, as V2 did, overstates the sample by roughly ``1/h``.
    """
    u = np.asarray(uniqueness, dtype=float)
    return float(np.nansum(u))


def sequential_bootstrap(t1, size: int | None = None, seed: int = 0) -> np.ndarray:
    """Sequential bootstrap: draw with probability inverse to accumulated overlap.

    A standard bootstrap of overlapping financial labels oversamples redundant
    observations. This scheme reweights after each draw so already well-covered periods
    become less likely, producing a resample whose average uniqueness is materially
    higher. Used for the robustness section, not the main training loop, because it is
    O(size x n_samples).
    """
    rng = np.random.default_rng(seed)
    t1 = _as_positional(t1)
    finite = np.nonzero(np.isfinite(t1))[0]
    if finite.size == 0:
        return np.array([], dtype=int)

    size = int(size or finite.size)
    n_periods = int(np.nanmax(t1)) + 1
    ends = np.minimum(t1[finite].astype(int), n_periods - 1)

    live = np.zeros(n_periods, dtype=float)
    chosen: list[int] = []

    for _ in range(size):
        inv_cum = np.concatenate([[0.0], np.cumsum(1.0 / (live + 1.0))])
        width = (ends - finite + 1).astype(float)
        avg_u = np.where(width > 0, (inv_cum[ends + 1] - inv_cum[finite]) / np.maximum(width, 1), 0.0)
        total = avg_u.sum()
        if total < _EPS:
            break
        pick = int(rng.choice(finite.size, p=avg_u / total))
        chosen.append(int(finite[pick]))
        live[finite[pick] : ends[pick] + 1] += 1.0

    return np.asarray(chosen, dtype=int)
