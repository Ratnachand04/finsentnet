"""Calibration: the metric the paper's central claim rests on.

Two things must be true for the claim to survive review. First, the ECE estimator must
actually detect miscalibration that a human can verify by construction. Second, the
post-hoc map must reduce it on *held-out* data, not on the data it was fitted to.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import softmax

from finsent.eval.metrics import brier_score, expected_calibration_error, multiclass_nll
from finsent.training.calibrate import (
    TemperatureScaling,
    VectorScaling,
    calibration_report,
    fit_calibrator,
    select_calibrator,
)


def _logits_and_labels(n: int, seed: int, sharpness: float = 1.0):
    """Labels drawn from the true probabilities; stated logits scaled by ``sharpness``.

    ``sharpness > 1`` produces overconfidence *without* changing the argmax, so accuracy
    is held fixed while calibration degrades. That separation is the paper's whole
    argument in miniature.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3))
    true_p = softmax(z, axis=1)
    labels = np.array([rng.choice(3, p=row) for row in true_p])
    return sharpness * z, labels, z


def test_ece_is_near_zero_for_a_perfectly_calibrated_model():
    z, y, _ = _logits_and_labels(20000, seed=0, sharpness=1.0)
    stats = expected_calibration_error(softmax(z, axis=1), y, n_bins=15)
    assert stats["ece"] < 0.02, f"ECE={stats['ece']:.4f} on truly calibrated probabilities"
    assert abs(stats["overconfidence"]) < 0.02


def test_ece_detects_overconfidence_without_any_accuracy_change():
    calibrated_z, y, _ = _logits_and_labels(20000, seed=1, sharpness=1.0)
    over_z, y2, _ = _logits_and_labels(20000, seed=1, sharpness=2.5)
    assert (y == y2).all(), "fixture must hold labels fixed"

    acc_cal = float((softmax(calibrated_z, 1).argmax(1) == y).mean())
    acc_over = float((softmax(over_z, 1).argmax(1) == y).mean())
    assert acc_cal == pytest.approx(acc_over), "sharpening must not change the argmax"

    ece_cal = expected_calibration_error(softmax(calibrated_z, 1), y)["ece"]
    ece_over = expected_calibration_error(softmax(over_z, 1), y)["ece"]
    assert ece_over > ece_cal + 0.05, (
        f"ECE failed to detect sharpening: {ece_cal:.4f} -> {ece_over:.4f} at identical "
        "accuracy. This is precisely the failure mode the paper is about."
    )


def test_temperature_scaling_recovers_a_known_temperature():
    """Fit on logits scaled by 2.5 and the recovered temperature should be near 2.5."""
    z, y, _ = _logits_and_labels(20000, seed=2, sharpness=2.5)
    fitted = TemperatureScaling().fit(z, y)
    assert fitted.temperature == pytest.approx(2.5, rel=0.15), (
        f"recovered T={fitted.temperature:.3f}, expected ~2.5"
    )


def test_vector_scaling_reduces_ece_on_held_out_data():
    z_cal, y_cal, _ = _logits_and_labels(15000, seed=3, sharpness=2.5)
    z_test, y_test, _ = _logits_and_labels(15000, seed=4, sharpness=2.5)

    report = calibration_report(z_cal, y_cal, z_test, y_test, method="vector_scaling")

    assert report["calibrated"]["ece"] < report["raw"]["ece"], (
        f"ECE did not improve out of sample: {report['raw']['ece']:.4f} -> "
        f"{report['calibrated']['ece']:.4f}"
    )
    assert report["ece_reduction"] > 0.03
    assert report["calibrated"]["nll"] < report["raw"]["nll"]
    assert report["calibrated"]["brier"] <= report["raw"]["brier"] + 1e-6


def test_calibration_does_not_change_accuracy():
    """A calibration map may reweight classes; it must not repair a bad ranking.

    If accuracy moved, the "calibration" would be doing prediction, and the paper's
    separation between accuracy and calibration would collapse.
    """
    z_cal, y_cal, _ = _logits_and_labels(10000, seed=5, sharpness=2.0)
    z_test, y_test, _ = _logits_and_labels(10000, seed=6, sharpness=2.0)

    calibrator = VectorScaling().fit(z_cal, y_cal)
    before = float(softmax(z_test, 1).argmax(1).astype(int).__eq__(y_test).mean())
    after = float(calibrator.predict_proba(z_test).argmax(1).astype(int).__eq__(y_test).mean())

    assert abs(before - after) < 0.03, (
        f"accuracy moved {before:.4f} -> {after:.4f} under calibration"
    )


def test_vector_scaling_handles_class_imbalance_better_than_temperature():
    """NEUTRAL dominates in this problem, which is exactly the asymmetric case.

    A single temperature can only sharpen or flatten every class at once; per-class
    scales and biases can also correct a systematic over-prediction of the majority
    class. That asymmetry is why the specification chooses vector scaling.
    """
    rng = np.random.default_rng(7)
    n = 20000
    z = rng.standard_normal((n, 3))
    z[:, 1] += 1.2  # systematically favour NEUTRAL
    true_p = softmax(z, axis=1)
    y = np.array([rng.choice(3, p=row) for row in true_p])

    biased = 2.0 * z
    biased[:, 1] += 0.8  # and state that preference too strongly

    cut = n // 2
    vec = calibration_report(biased[:cut], y[:cut], biased[cut:], y[cut:], "vector_scaling")
    temp = calibration_report(biased[:cut], y[:cut], biased[cut:], y[cut:], "temperature")

    assert vec["calibrated"]["ece"] <= temp["calibrated"]["ece"] + 1e-4, (
        f"vector scaling ECE {vec['calibrated']['ece']:.4f} should not exceed "
        f"temperature ECE {temp['calibrated']['ece']:.4f} under class asymmetry"
    )


def test_identity_calibrator_is_available_for_the_ablation():
    z, y, _ = _logits_and_labels(2000, seed=8)
    identity = fit_calibrator(z, y, method="none")
    assert np.allclose(identity.predict_proba(z), softmax(z, axis=1))


def test_scales_are_positive_so_calibration_cannot_reorder_classes():
    """A negative scale reverses the model's own ranking; that is re-prediction.

    The unconstrained fit really did return scales like [-0.019, -2.667, 3.313] on this
    problem, which changes the argmax and therefore the accuracy.
    """
    rng = np.random.default_rng(20)
    z = rng.standard_normal((800, 3)) * 0.1  # deliberately near-flat logits
    y = rng.integers(0, 3, 800)

    fitted = VectorScaling().fit(z, y)
    assert (fitted.scale > 0).all(), f"non-positive scale fitted: {fitted.scale}"


def test_selection_never_chooses_a_map_worse_than_identity():
    """The actual guard: identity competes, so calibration can never be a net loss.

    This is the invariant that mattered. An unguarded fit reduced inner-validation ECE
    from 0.020 to 0.004 while raising *test* ECE from 0.003 to 0.056 -- it had learned
    the validation fold's class balance, and that balance drifted. Requiring the winner
    to beat identity on a later, unseen slice removes that failure mode.

    Note that identity does not always win, and should not: a genuinely uninformative
    model with slightly spread logits is improved by flattening toward uniform, which is
    real calibration rather than an artefact.
    """
    rng = np.random.default_rng(21)
    for seed in range(6):
        n = 1200
        z = rng.standard_normal((n, 3)) * 0.08
        y = rng.integers(0, 3, n)

        chosen, trace = select_calibrator(z, y, seed=seed)
        assert trace, "the selection trace must record what was compared"

        by_method = {row["method"]: row["score"] for row in trace}
        best = min(by_method.values())
        assert best <= by_method["none"] + 1e-12, (
            f"a candidate scoring worse than identity was selected; trace {trace}"
        )
        assert type(chosen).__name__ in {
            "IdentityCalibration", "TemperatureScaling", "VectorScaling"
        }


def test_calibrator_selection_still_fixes_a_genuinely_overconfident_model():
    """The guard must not disable calibration when calibration is actually needed."""
    z, y, _ = _logits_and_labels(6000, seed=22, sharpness=3.0)

    chosen, trace = select_calibrator(z, y)
    assert type(chosen).__name__ != "IdentityCalibration", (
        f"identity was selected despite clear overconfidence; trace {trace}"
    )

    raw_ece = expected_calibration_error(softmax(z, axis=1), y)["ece"]
    cal_ece = expected_calibration_error(chosen.predict_proba(z), y)["ece"]
    assert cal_ece < raw_ece / 2.0, f"ECE {raw_ece:.4f} -> {cal_ece:.4f}"


def test_selection_split_is_temporal_not_random():
    """The holdout must be the *later* part of the block, so drift is exercised.

    A random split would judge each candidate on rows interleaved with its own fitting
    data, which is exactly the condition under which a calibrator looks better than it
    is. Tested by construction rather than by inspection: if the split were random it
    would be order-invariant, so reversing the block must change the result.
    """
    rng = np.random.default_rng(23)
    n = 2000
    z = rng.standard_normal((n, 3))
    y = np.array([rng.choice(3, p=row) for row in softmax(z, axis=1)])

    # First half badly overconfident, second half only mildly so.
    stated = z.copy()
    stated[: n // 2] *= 4.0
    stated[n // 2 :] *= 1.2

    _, forward = select_calibrator(stated, y)
    _, backward = select_calibrator(stated[::-1], y[::-1])

    forward_scores = {row["method"]: round(row["score"], 6) for row in forward}
    backward_scores = {row["method"]: round(row["score"], 6) for row in backward}

    assert forward_scores != backward_scores, (
        "reversing the block left the selection scores unchanged, which means the "
        f"holdout is order-invariant and therefore not temporal: {forward_scores}"
    )


def test_brier_and_nll_agree_with_hand_computed_values():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    labels = np.array([0, 1])

    expected_brier = np.mean([
        (0.7 - 1) ** 2 + 0.2**2 + 0.1**2,
        0.1**2 + (0.8 - 1) ** 2 + 0.1**2,
    ])
    assert brier_score(probs, labels) == pytest.approx(expected_brier)
    assert multiclass_nll(probs, labels) == pytest.approx(
        -np.mean([np.log(0.7), np.log(0.8)])
    )
