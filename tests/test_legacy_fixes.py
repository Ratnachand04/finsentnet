"""The same two defects, verified fixed in the legacy V2 modules.

Both bugs were found and corrected in the research path first
(``finsent.data.features_causal``, ``finsent.training.calibrate``). But the V2 modules
are still imported by ``finsent/data/pipeline.py``, ``finsent/training/trainer.py`` and
the finsentnet_pro application, so leaving them defective would mean shipping a product
with a look-ahead bug while the paper describes the corrected one — which is the exact
train/deploy divergence this rebuild exists to eliminate.

These tests are deliberately written against the *legacy* API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsent.data.features import (
    DERIVED_FEATURE_COLUMNS,
    compute_all_features,
    lag_for_open_execution,
)


def _ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.standard_normal(n) * 0.015
    close = 100.0 * np.exp(np.cumsum(ret))
    prev_close = np.concatenate([[close[0] / np.exp(ret[0])], close[:-1]])
    open_ = prev_close * np.exp(0.4 * ret + rng.standard_normal(n) * 0.005)
    wick = np.abs(rng.standard_normal(n)) * 0.01 * close
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) + wick,
            "Low": np.minimum(open_, close) - wick,
            "Close": close,
            "Volume": np.exp(13 + rng.standard_normal(n) * 0.3),
        }
    )


# --------------------------------------------------------------------------------------
# Bug 1 — open-execution lag in the legacy feature builder
# --------------------------------------------------------------------------------------
def test_legacy_features_are_lagged_by_default():
    """The safe behaviour must be the default, not something a caller remembers."""
    df = _ohlcv()

    lagged = compute_all_features(df.copy())
    unlagged = compute_all_features(df.copy(), execution_lag=0)

    a = lagged["RSI"].to_numpy()[1:]
    b = unlagged["RSI"].to_numpy()[:-1]
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() > 100
    assert np.allclose(a[both], b[both]), (
        "the default output must equal the unlagged output shifted by one session"
    )


def test_legacy_lag_leaves_raw_ohlcv_untouched():
    """Labels and execution prices read the raw columns; shifting those would break them."""
    df = _ohlcv()
    out = compute_all_features(df.copy())

    for col in ("Open", "High", "Low", "Close", "Volume"):
        assert np.allclose(out[col].to_numpy(), df[col].to_numpy()), (
            f"{col} was shifted; only derived features may be lagged"
        )


def test_legacy_lag_removes_the_spurious_forward_correlation():
    """The measurement that exposed the bug, run against the legacy builder.

    On driftless data the trailing return must carry no information about the
    open-to-open forward return. Unlagged it does, because the two share r[t].
    """
    frames = [_ohlcv(n=600, seed=s) for s in range(12)]

    def pooled_corr(execution_lag: int) -> float:
        xs, ys = [], []
        for df in frames:
            out = compute_all_features(df.copy(), execution_lag=execution_lag)
            fwd = np.log(df["Open"].shift(-5) / df["Open"])
            joined = pd.DataFrame({"f": out["Returns"], "y": fwd}).dropna()
            xs.append(joined["f"].to_numpy())
            ys.append(joined["y"].to_numpy())
        x, y = np.concatenate(xs), np.concatenate(ys)
        return float(np.corrcoef(x, y)[0, 1])

    unlagged = pooled_corr(0)
    lagged = pooled_corr(1)

    assert unlagged > 0.10, (
        f"the unlagged bug should be plainly visible; measured {unlagged:.4f}. If this "
        "fails the fixture no longer reproduces the defect it guards against."
    )
    assert abs(lagged) < 0.05, (
        f"after the one-session lag the trailing return must carry no forward "
        f"information on driftless data; measured {lagged:.4f}"
    )


def test_legacy_lag_helper_rejects_a_negative_lag():
    with pytest.raises(ValueError, match="cannot be negative"):
        lag_for_open_execution(_ohlcv(50), sessions=-1)


def test_derived_column_list_matches_what_is_produced():
    out = compute_all_features(_ohlcv(200))
    for col in DERIVED_FEATURE_COLUMNS:
        assert col in out.columns, f"{col} is declared derived but never produced"


# --------------------------------------------------------------------------------------
# Bug 2 — the legacy calibrator grading itself on its own fitting data
# --------------------------------------------------------------------------------------
torch = pytest.importorskip("torch", reason="PyTorch not available")

from finsent.training.calibration import TemperatureScaler  # noqa: E402


class _FixedLogitModel(torch.nn.Module):
    """Replays a fixed logit matrix so the calibrator can be tested without a network."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.cursor = 0

    def forward(self, price, text_ids, text_mask):  # noqa: ARG002 - legacy signature
        n = price.shape[0]
        out = self.logits[self.cursor : self.cursor + n]
        self.cursor += n
        return {"direction_logits": out}


def _loader(logits: torch.Tensor, labels: torch.Tensor, batch: int = 256):
    """Minimal stand-in for the legacy DataLoader contract."""
    n = logits.shape[0]
    for i in range(0, n, batch):
        j = min(i + batch, n)
        yield {
            "price": torch.zeros(j - i, 1, 1),
            "text_ids": torch.zeros(j - i, 1, dtype=torch.long),
            "text_mask": torch.ones(j - i, 1, dtype=torch.long),
            "label": labels[i:j],
        }


def _make(n: int, sharpness: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 3, generator=g)
    probs = torch.softmax(z, dim=-1)
    labels = torch.multinomial(probs, 1, generator=g).squeeze(-1)
    return sharpness * z, labels


def test_legacy_calibrator_still_corrects_genuine_overconfidence():
    """The guard must not disable calibration when it is actually needed."""
    logits, labels = _make(6000, sharpness=3.0, seed=0)
    scaler = TemperatureScaler()

    results = scaler.calibrate(
        _FixedLogitModel(logits), _loader(logits, labels), torch.device("cpu")
    )

    assert results["optimal_temperature"] > 1.5, (
        f"a model stating 3x its earned confidence needs T well above 1; "
        f"got {results['optimal_temperature']:.3f}"
    )
    assert not results["identity_rejected_fit"]
    assert results["ece_after"] < results["ece_before"]


def test_legacy_calibrator_falls_back_to_identity_when_fitting_would_hurt():
    """A near-uniform softmax is already calibrated; the fit must be rejectable."""
    logits, labels = _make(4000, sharpness=0.05, seed=1)
    scaler = TemperatureScaler()

    results = scaler.calibrate(
        _FixedLogitModel(logits), _loader(logits, labels), torch.device("cpu")
    )

    assert results["ece_after"] <= results["ece_before"] + 1e-12, (
        "held-out ECE got worse; the identity guard is not working"
    )
    if results["identity_rejected_fit"]:
        assert results["optimal_temperature"] == pytest.approx(1.0)


def test_legacy_calibrator_reports_metrics_on_held_out_rows():
    """The reported improvement must not be measured on the fitting rows."""
    logits, labels = _make(3000, sharpness=2.0, seed=2)
    scaler = TemperatureScaler()

    results = scaler.calibrate(
        _FixedLogitModel(logits), _loader(logits, labels), torch.device("cpu")
    )

    assert results["n_fit"] + results["n_holdout"] == 3000
    assert results["n_holdout"] > 1000, "the holdout must be a substantial share"
    assert results["n_fit"] > 0


def test_legacy_calibrator_never_divides_by_a_zero_baseline():
    """A perfectly calibrated input gave ece_before = 0 and an infinite 'reduction'."""
    n = 2000
    logits = torch.zeros(n, 3)  # exactly uniform: ECE is ~0 by construction
    labels = torch.randint(0, 3, (n,), generator=torch.Generator().manual_seed(3))

    results = TemperatureScaler().calibrate(
        _FixedLogitModel(logits), _loader(logits, labels), torch.device("cpu")
    )

    assert np.isfinite(results["ece_reduction"]), "ece_reduction must never be inf or nan"
