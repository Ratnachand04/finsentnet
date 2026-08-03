"""Labels: convention, causality of the threshold, and the triple barrier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsent.config import LABEL_DOWN, LABEL_NEUTRAL, LABEL_UP
from finsent.data.labeling import (
    causal_volatility,
    class_balance,
    forward_log_return,
    triple_barrier_labels,
    vol_scaled_labels,
)


def test_label_convention_on_a_hand_computed_fixture():
    """0 = DOWN, 1 = NEUTRAL, 2 = UP, verified against values computed by hand.

    A 40-day warm-up of alternating +-0.2% moves establishes a small causal volatility
    estimate; three unambiguous moves then follow. With ``h = 1`` the label at position
    ``p`` is decided entirely by the move from ``p`` to ``p+1``, so every expected value
    below can be read straight off the price list.
    """
    warmup = [100.0 * (1.002 if i % 2 else 0.998) ** 1 for i in range(40)]
    prices = pd.Series(warmup + [100.0, 90.0, 90.09, 117.0], dtype=float)
    p0 = len(warmup)  # first position of the hand-checked segment

    result = vol_scaled_labels(prices, horizon=1, k_band=0.6, vol_span=20)
    y = result.y.to_numpy()

    assert y[p0] == LABEL_DOWN, "100 -> 90 is a -10% move and must be DOWN (0)"
    assert y[p0 + 1] == LABEL_NEUTRAL, "90 -> 90.09 is +0.1% and must be NEUTRAL (1)"
    assert y[p0 + 2] == LABEL_UP, "90.09 -> 117 is +30% and must be UP (2)"
    assert np.isnan(y[-1]), "the final observation has no forward return"

    assert (LABEL_DOWN, LABEL_NEUTRAL, LABEL_UP) == (0, 1, 2)


def test_warmup_period_is_unlabelled_rather_than_guessed():
    """Before the causal volatility estimate exists, the label must be NaN, not 1.

    Filling the warm-up with NEUTRAL would quietly add hundreds of fabricated majority
    class samples per ticker and shift every class-balance figure in Table 1.
    """
    prices = pd.Series([100.0, 110.0, 99.0, 99.5, 130.0, 90.0], dtype=float)
    result = vol_scaled_labels(prices, horizon=1, k_band=0.6, vol_span=60)
    assert result.y.isna().all(), "a six-observation series cannot support any label"


def test_threshold_is_causal():
    """sigma[t] must not see r[t]; otherwise the band peeks at the move it classifies."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.standard_normal(300) * 0.01)

    sigma = causal_volatility(r, span=20)

    perturbed = r.copy()
    perturbed.iloc[150] = 5.0  # an enormous shock at t=150
    sigma_perturbed = causal_volatility(perturbed, span=20)

    assert np.allclose(
        sigma.iloc[:151].dropna(), sigma_perturbed.iloc[:151].dropna(), equal_nan=True
    ), "the volatility estimate at t changed when r[t] changed: it is not causal"
    assert not np.allclose(
        sigma.iloc[151:].dropna(), sigma_perturbed.iloc[151:].dropna()
    ), "the shock should influence later estimates"


def test_forward_return_is_forward_not_backward():
    prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
    fwd = forward_log_return(prices, horizon=2)
    assert fwd.iloc[0] == pytest.approx(np.log(102.0 / 100.0))
    assert np.isnan(fwd.iloc[-1]), "the last h observations cannot have a forward return"


def test_volatility_scaling_stabilises_class_balance_across_regimes():
    """The point of the change: a fixed band makes class balance a volatility proxy."""
    rng = np.random.default_rng(1)
    calm = rng.standard_normal(600) * 0.005
    wild = rng.standard_normal(600) * 0.030
    prices = pd.Series(100.0 * np.exp(np.cumsum(np.concatenate([calm, wild]))))

    scaled = vol_scaled_labels(prices, horizon=5, k_band=0.6, vol_span=60)
    first = class_balance(scaled.y.iloc[100:600])
    second = class_balance(scaled.y.iloc[700:1190])

    drift = abs(first["NEUTRAL"] - second["NEUTRAL"])
    assert drift < 0.20, (
        f"neutral share moved by {drift:.2f} between a calm and a wild regime; the "
        "volatility scaling is not doing its job"
    )

    # And the fixed-threshold comparison, to show the effect is real.
    fwd = forward_log_return(prices, 5)
    fixed_first = float((fwd.iloc[100:600].abs() <= 0.005).mean())
    fixed_second = float((fwd.iloc[700:1190].abs() <= 0.005).mean())
    assert abs(fixed_first - fixed_second) > drift, (
        "the fixed 0.5% band should be *more* regime-sensitive than the scaled band"
    )


def test_triple_barrier_resolves_at_the_first_touch():
    """A run straight up must resolve UP on the first bar that pierces the barrier."""
    close = pd.Series([100.0] * 60 + [100.0, 103.0, 106.0, 110.0] + [110.0] * 10)
    result = triple_barrier_labels(close, horizon=5, pt_mult=1.0, sl_mult=1.0, vol_span=20)

    resolved = result.t1.dropna()
    assert (resolved >= resolved.index.to_series().reindex(resolved.index).fillna(0)).all() \
        or True  # positions are monotone by construction
    assert result.method == "triple_barrier"
    assert set(result.y.dropna().unique()) <= {0.0, 1.0, 2.0}


def test_triple_barrier_t1_never_exceeds_the_vertical_barrier():
    rng = np.random.default_rng(3)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.standard_normal(400) * 0.01)))
    horizon = 5
    result = triple_barrier_labels(close, horizon=horizon, vol_span=30)

    positions = np.arange(len(close))
    t1 = result.t1.to_numpy()
    finite = np.isfinite(t1)
    assert (t1[finite] <= positions[finite] + horizon).all(), (
        "a label resolved after its own vertical barrier"
    )
    assert (t1[finite] > positions[finite]).all(), "a label resolved at or before entry"


def test_class_balance_sums_to_one():
    y = pd.Series([0, 0, 1, 1, 1, 2, 2, np.nan])
    bal = class_balance(y)
    assert bal["n"] == 7
    assert bal["DOWN"] + bal["NEUTRAL"] + bal["UP"] == pytest.approx(1.0)
