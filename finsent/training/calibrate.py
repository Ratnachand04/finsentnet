"""Post-hoc probability calibration, fitted once per walk-forward fold.

V2 had four calibration mechanisms operating at once and reconciled none of them:

1. a learnable temperature inside a sigmoid confidence head (a no-op: dividing a linear
   layer's output by a learned scalar before a sigmoid merely rescales that layer's own
   weights, and the optimiser can undo it for free);
2. a post-hoc temperature on the 3-class softmax, calibrating a different object;
3. an "ECE penalty" during training, using hard bins, which is not differentiable;
4. focal loss, which itself changes calibration -- Mukhoti et al. (2020) show it tends to
   *under*-confide, so it partially cancels mechanism 2 in an undocumented way.

This module is the whole of the post-hoc story now. Exactly one calibration map is fitted,
on the **inner-validation** block of each fold, by minimising negative log-likelihood.
Vector scaling generalises temperature scaling with a per-class scale and bias, which
matters here because the three classes are not symmetric: NEUTRAL dominates and is
systematically over-predicted.

Implemented in NumPy/SciPy so it is testable and reproducible without PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

__all__ = ["TemperatureScaling", "VectorScaling", "fit_calibrator", "calibration_report"]

_EPS = 1e-12


def _nll(logits: np.ndarray, labels: np.ndarray) -> float:
    log_p = log_softmax(logits, axis=1)
    return float(-np.mean(log_p[np.arange(labels.size), labels]))


@dataclass
class TemperatureScaling:
    """Single-parameter calibration, ``z / T`` (Guo et al., 2017).

    Retained as the ablation baseline against vector scaling, and because it is the
    method most readers will expect to see named.
    """

    temperature: float = 1.0
    n_fit: int = 0
    max_iter: int = 200

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaling":
        z = np.asarray(logits, dtype=float)
        y = np.asarray(labels, dtype=int).ravel()

        def objective(log_t: np.ndarray) -> float:
            return _nll(z / np.exp(log_t[0]), y)

        res = minimize(objective, x0=np.array([0.0]), method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.temperature = float(np.exp(res.x[0]))
        self.n_fit = int(y.size)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=float) / max(self.temperature, _EPS)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return softmax(self.transform(logits), axis=1)


@dataclass
class VectorScaling:
    """Per-class scale and bias, ``z_k * a_k + b_k`` (Guo et al., 2017, s. 4.2).

    Two departures from the textbook form, both needed and both discovered by measuring
    rather than by reasoning:

    **Scales are constrained positive** (parameterised as ``exp(log_scale)``). An
    unconstrained fit on this problem returned scales such as ``[-0.019, -2.667, 3.313]``.
    A negative scale reverses the model's own ordering for that class, which changes the
    argmax and therefore the accuracy -- a calibration map must not do that. It is not
    calibrating at that point, it is re-predicting.

    **Regularised toward the identity map.** With three classes the fit has six free
    parameters, and when the model is weak the logits have a standard deviation near 0.1,
    so the negative log-likelihood surface is almost flat and the optimiser drifts to
    whatever fits the validation fold's class balance. That balance then drifts: in
    testing, a fold with a validation DOWN share of 0.28 had a test share of 0.46, and
    the unregularised calibrator turned a test ECE of 0.003 into 0.055. The penalty
    ``l2 * (||log a||^2 + ||b||^2)`` keeps the map near identity unless the data really
    demand otherwise, which is the correct prior for a calibrator.

    Both effects are worth reporting in the paper: post-hoc calibration is not free, and
    it can *degrade* out-of-sample calibration when the underlying model is weak.
    """

    n_classes: int = 3
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_fit: int = 0
    max_iter: int = 200
    l2: float = 1.0
    converged: bool = False

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "VectorScaling":
        z = np.asarray(logits, dtype=float)
        y = np.asarray(labels, dtype=int).ravel()
        k = z.shape[1]
        self.n_classes = k
        n = max(y.size, 1)

        def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return np.exp(theta[:k]), theta[k:]

        def objective(theta: np.ndarray) -> float:
            a, b = unpack(theta)
            penalty = self.l2 * (float(theta[:k] @ theta[:k]) + float(theta[k:] @ theta[k:])) / n
            return _nll(z * a[None, :] + b[None, :], y) + penalty

        x0 = np.zeros(2 * k)  # log-scale 0 and bias 0 == the identity map
        res = minimize(objective, x0=x0, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})

        self.scale, self.bias = unpack(res.x)
        self.n_fit = int(y.size)
        self.converged = bool(res.success)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, dtype=float)
        return z * self.scale[None, :] + self.bias[None, :]

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return softmax(self.transform(logits), axis=1)

    def describe(self) -> str:
        return (
            f"VectorScaling(n_fit={self.n_fit}, converged={self.converged}) "
            f"scale={np.round(self.scale, 3).tolist()} bias={np.round(self.bias, 3).tolist()}"
        )


@dataclass
class IdentityCalibration:
    """The null calibrator. A real candidate, not a placeholder.

    When the underlying model is weak its softmax sits near uniform, and a near-uniform
    predictor is already almost perfectly calibrated -- in testing, confidence 0.361
    against accuracy 0.3645. Fitting anything at that point adds variance without
    removing bias, so identity must be allowed to win the selection.
    """

    n_fit: int = 0
    converged: bool = True

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "IdentityCalibration":
        self.n_fit = int(np.asarray(labels).size)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=float)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return softmax(self.transform(logits), axis=1)

    def describe(self) -> str:
        return f"IdentityCalibration(n_fit={self.n_fit})"


def _make(method: str, max_iter: int):
    if method == "vector_scaling":
        return VectorScaling(max_iter=max_iter)
    if method == "temperature":
        return TemperatureScaling(max_iter=max_iter)
    if method == "none":
        return IdentityCalibration()
    raise ValueError(f"unknown calibration method {method!r}")


def fit_calibrator(
    logits: np.ndarray,
    labels: np.ndarray,
    method: str = "vector_scaling",
    max_iter: int = 200,
):
    """Fit one calibrator. ``method='none'`` returns an identity map."""
    return _make(method, max_iter).fit(logits, labels)


def select_calibrator(
    logits: np.ndarray,
    labels: np.ndarray,
    methods: tuple[str, ...] = ("none", "temperature", "vector_scaling"),
    max_iter: int = 200,
    n_bins: int = 15,
    holdout: float = 0.5,
    seed: int = 0,
):
    """Fit candidates on one half of the validation block and pick on the other.

    Why the extra split exists
    --------------------------
    A calibrator fitted and judged on the same data always looks like an improvement. In
    this project the unguarded version reduced inner-validation ECE from 0.020 to 0.004
    while *raising* test ECE from 0.003 to 0.056: it had learned the validation fold's
    class balance, and that balance drifted (a validation DOWN share of 0.28 against a
    test share of 0.46 in one fold).

    Splitting the validation block turns calibration into ordinary model selection, with
    the identity map as a genuine competitor. Nothing here touches the test block, so
    the guarantee the paper reports is still out-of-sample.

    The split is **temporal**, not random: the second half is later than the first, so
    the selection is judged under exactly the kind of drift it needs to survive.
    """
    from finsent.eval.metrics import expected_calibration_error, multiclass_nll

    z = np.asarray(logits, dtype=float)
    y = np.asarray(labels, dtype=int).ravel()
    n = y.size
    cut = max(int(n * (1.0 - holdout)), 10)

    if n < 40:  # too small to select; keep the map closest to identity
        return _make("none", max_iter).fit(z, y), []

    z_fit, y_fit = z[:cut], y[:cut]
    z_sel, y_sel = z[cut:], y[cut:]

    trace: list[dict[str, object]] = []
    best, best_score = None, np.inf
    for method in methods:
        candidate = _make(method, max_iter).fit(z_fit, y_fit)
        probs = candidate.predict_proba(z_sel)
        ece = expected_calibration_error(probs, y_sel, n_bins)["ece"]
        nll = multiclass_nll(probs, y_sel)
        # Rank on ECE, the quantity the paper claims to improve, with NLL as a tiebreak.
        score = ece + 0.01 * nll
        trace.append({"method": method, "ece": ece, "nll": nll, "score": score})
        if score < best_score:
            best, best_score = candidate, score

    # Refit the winner on the whole validation block: selection is done, so the extra
    # data is free and reduces the calibrator's own variance.
    winner = type(best)(max_iter=max_iter) if not isinstance(best, IdentityCalibration) \
        else IdentityCalibration()
    return winner.fit(z, y), trace


def calibration_report(
    logits_cal: np.ndarray,
    labels_cal: np.ndarray,
    logits_test: np.ndarray,
    labels_test: np.ndarray,
    method: str = "vector_scaling",
    n_bins: int = 15,
) -> dict[str, object]:
    """Before/after calibration metrics on a held-out block -- the paper's Table 4.

    The calibrator is fitted on ``*_cal`` and evaluated on ``*_test``. Reporting the
    improvement on the same data used to fit it would be meaningless, and is the kind of
    thing that produces an ECE of 0.001 in a manuscript and 0.09 in deployment.
    """
    from finsent.eval.metrics import brier_score, expected_calibration_error, multiclass_nll

    calibrator = fit_calibrator(logits_cal, labels_cal, method)
    raw = softmax(np.asarray(logits_test, dtype=float), axis=1)
    cal = calibrator.predict_proba(logits_test)

    out: dict[str, object] = {"method": method, "calibrator": calibrator}
    for tag, p in (("raw", raw), ("calibrated", cal)):
        stats = expected_calibration_error(p, labels_test, n_bins)
        out[tag] = {
            "ece": stats["ece"],
            "mce": stats["mce"],
            "brier": brier_score(p, labels_test),
            "nll": multiclass_nll(p, labels_test),
            "overconfidence": stats["overconfidence"],
        }
    out["ece_reduction"] = out["raw"]["ece"] - out["calibrated"]["ece"]  # type: ignore[index]
    out["probs_calibrated"] = cal
    return out
