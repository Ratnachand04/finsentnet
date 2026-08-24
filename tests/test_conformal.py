"""Conformal coverage must be a guarantee, not a hope.

The point of these tests is that the coverage claim is checkable *without* assuming the
model is any good. Synthetic probabilities are deliberately miscalibrated, and coverage
must still land on the nominal level -- that is what distinguishes a conformal guarantee
from a calibration result.
"""

from __future__ import annotations

import numpy as np
import pytest

from finsent.training.conformal import (
    AdaptiveConformal,
    SplitConformal,
    aps_scores,
    conformal_quantile,
    coverage_report,
)


def _synthetic(n: int, seed: int, sharpness: float = 1.0, shift: float = 0.0):
    """Probabilities and labels where the label really is drawn from the probabilities.

    ``sharpness != 1`` makes the model over- or under-confident without changing which
    class it prefers, so coverage can be tested independently of calibration.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3)) + shift
    true_p = np.exp(z - z.max(axis=1, keepdims=True))
    true_p /= true_p.sum(axis=1, keepdims=True)

    labels = np.array([rng.choice(3, p=row) for row in true_p])

    stated = np.exp(sharpness * z - (sharpness * z).max(axis=1, keepdims=True))
    stated /= stated.sum(axis=1, keepdims=True)
    return stated, labels


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_split_conformal_attains_nominal_coverage(alpha):
    p_cal, y_cal = _synthetic(4000, seed=1)
    p_test, y_test = _synthetic(4000, seed=2)

    predictor = SplitConformal(alpha=alpha).fit(p_cal, y_cal)
    result = predictor.evaluate(p_test, y_test)

    tolerance = 3.0 * np.sqrt(alpha * (1 - alpha) / 4000)
    assert abs(result["coverage_gap"]) < max(tolerance, 0.02), (
        f"alpha={alpha}: empirical coverage {result['empirical_coverage']:.4f} vs "
        f"nominal {1 - alpha:.2f}"
    )


def test_coverage_holds_even_when_the_model_is_badly_miscalibrated():
    """The guarantee is distribution-free: it does not require calibrated probabilities.

    This is the property that makes conformal worth the section in the paper. A model
    twice as confident as it should be still gets its stated coverage; it simply pays
    for it with larger sets.
    """
    p_cal, y_cal = _synthetic(4000, seed=3, sharpness=2.5)
    p_test, y_test = _synthetic(4000, seed=4, sharpness=2.5)

    result = SplitConformal(alpha=0.10).fit(p_cal, y_cal).evaluate(p_test, y_test)
    assert abs(result["coverage_gap"]) < 0.03, (
        f"coverage {result['empirical_coverage']:.3f} under a badly overconfident model"
    )


def test_set_size_grows_as_alpha_shrinks():
    p_cal, y_cal = _synthetic(3000, seed=5)
    p_test, _ = _synthetic(3000, seed=6)

    sizes = []
    for alpha in (0.30, 0.20, 0.10, 0.05):
        predictor = SplitConformal(alpha=alpha).fit(p_cal, y_cal)
        sizes.append(float(predictor.predict_sets(p_test).sum(axis=1).mean()))

    assert all(a <= b + 1e-9 for a, b in zip(sizes, sizes[1:])), (
        f"mean set size must be non-decreasing as alpha falls; got {sizes}"
    )


def test_prediction_sets_are_never_empty():
    """An empty set is a degenerate abstention that says nothing about direction."""
    p_cal, y_cal = _synthetic(2000, seed=7)
    p_test, _ = _synthetic(2000, seed=8)

    sets = SplitConformal(alpha=0.30).fit(p_cal, y_cal).predict_sets(p_test)
    assert (sets.sum(axis=1) >= 1).all()


def test_singleton_mask_is_the_trading_gate():
    p_cal, y_cal = _synthetic(2000, seed=9)
    p_test, _ = _synthetic(2000, seed=10)

    predictor = SplitConformal(alpha=0.10).fit(p_cal, y_cal)
    mask = predictor.singleton_mask(p_test)
    sets = predictor.predict_sets(p_test)

    assert (mask == (sets.sum(axis=1) == 1)).all()
    assert 0.0 < mask.mean() < 1.0, (
        f"singleton rate {mask.mean():.3f}: a gate that always or never fires is not a gate"
    )


def test_finite_sample_quantile_correction_is_applied():
    """The (n+1) correction is what makes the guarantee exact rather than asymptotic."""
    scores = np.linspace(0.0, 1.0, 100)
    q = conformal_quantile(scores, alpha=0.10)
    plain = float(np.quantile(scores, 0.90))
    assert q >= plain, "the corrected quantile must be at least the empirical one"


def test_aps_scores_are_small_when_the_truth_is_ranked_first():
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    labels = np.array([0, 2])
    confident = aps_scores(probs, labels)

    wrong = aps_scores(probs, np.array([1, 1]))
    assert (confident < wrong).all(), "a correctly ranked truth must score lower"


def test_adaptive_conformal_recovers_coverage_after_a_regime_break():
    """Under shift, split conformal drifts and the adaptive update pulls it back.

    This is the 2020/2022 problem in miniature, and showing both behaviours is more
    persuasive than showing only the version that works.
    """
    p_cal, y_cal = _synthetic(3000, seed=11)
    p_shift, y_shift = _synthetic(3000, seed=12, sharpness=3.0, shift=1.5)

    split = SplitConformal(alpha=0.10).fit(p_cal, y_cal).evaluate(p_shift, y_shift)

    adaptive = AdaptiveConformal(alpha_target=0.10, gamma=0.02).fit(p_cal, y_cal)
    online = adaptive.run(p_shift, y_shift)

    target = 0.90
    assert abs(online["empirical_coverage"] - target) <= abs(
        split["empirical_coverage"] - target
    ) + 0.02, (
        f"adaptive coverage {online['empirical_coverage']:.3f} should be no worse than "
        f"split coverage {split['empirical_coverage']:.3f} under shift"
    )
    assert online["alpha_path"].size == y_shift.size


def test_coverage_report_returns_one_row_per_alpha():
    p_cal, y_cal = _synthetic(1500, seed=13)
    p_test, y_test = _synthetic(1500, seed=14)

    rows = coverage_report(p_cal, y_cal, p_test, y_test, alphas=(0.05, 0.10, 0.20))
    assert len(rows) == 3
    assert [r["alpha"] for r in rows] == [0.05, 0.10, 0.20]
    for row in rows:
        assert 0.0 <= row["empirical_coverage"] <= 1.0
        assert 1.0 <= row["mean_set_size"] <= 3.0


def test_fitting_on_test_data_is_not_silently_possible():
    """A quantile chosen on the evaluation data is a fitted parameter, not a guarantee."""
    predictor = SplitConformal(alpha=0.10)
    with pytest.raises(RuntimeError, match="fit must be called"):
        predictor.predict_sets(np.array([[0.5, 0.3, 0.2]]))


# --------------------------------------------------------------------------------------
# Adaptive conformal under distribution shift
# --------------------------------------------------------------------------------------
def _planted_block(rng, n, sharpness):
    """Probabilities whose informativeness we control directly.

    A positive sharpness puts mass on the true class; a negative one puts mass on a
    wrong class, which is what a regime break looks like to a model fitted before it.
    """
    y = rng.integers(0, 3, n)
    logits = rng.standard_normal((n, 3)) * 0.3
    logits[np.arange(n), y] += sharpness
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True), y


def test_both_variants_hold_coverage_when_exchangeability_holds():
    """With no shift the online update should cost nothing worth noticing."""
    rng = np.random.default_rng(0)
    p_cal, y_cal = _planted_block(rng, 4000, 0.35)
    p_test, y_test = _planted_block(rng, 4000, 0.35)

    for alpha in (0.05, 0.10, 0.20):
        split = SplitConformal(alpha=alpha, score="aps", seed=0).fit(p_cal, y_cal)
        online = AdaptiveConformal(alpha_target=alpha, gamma=0.005, score="aps").fit(
            p_cal, y_cal)

        s = split.evaluate(p_test, y_test)["empirical_coverage"]
        a = online.run(p_test, y_test)["empirical_coverage"]
        assert abs(s - (1 - alpha)) < 0.02, f"split off nominal at alpha={alpha}: {s:.4f}"
        assert abs(a - (1 - alpha)) < 0.02, f"adaptive off nominal at alpha={alpha}: {a:.4f}"


def test_adaptive_holds_coverage_where_split_collapses():
    """The reason the paper carries the online variant at all.

    Split conformal assumes exchangeability between calibration and test. Break it and
    the guarantee goes with it. The Gibbs--Candes update re-derives its own level from
    realised coverage, so it recovers without being told a break occurred.
    """
    rng = np.random.default_rng(0)
    p_cal, y_cal = _planted_block(rng, 4000, 0.35)
    p_shift, y_shift = _planted_block(rng, 4000, -0.20)   # the model is now wrong

    alpha = 0.10
    split = SplitConformal(alpha=alpha, score="aps", seed=0).fit(p_cal, y_cal)
    online = AdaptiveConformal(alpha_target=alpha, gamma=0.005, score="aps").fit(
        p_cal, y_cal)

    s = split.evaluate(p_shift, y_shift)["empirical_coverage"]
    out = online.run(p_shift, y_shift)
    a = out["empirical_coverage"]

    assert s < 1 - alpha - 0.20, (
        f"the planted break did not actually break split coverage: {s:.4f}"
    )
    assert abs(a - (1 - alpha)) < 0.03, (
        f"adaptive failed to recover coverage under shift: {a:.4f} against {1-alpha}"
    )
    assert a - s > 0.25, "the adaptive variant bought nothing on this break"
    assert out["final_alpha"] < alpha, (
        "alpha should tighten when coverage is short, not loosen"
    )


def test_adaptive_consumes_the_label_only_after_emitting_the_set():
    """Ordering is the whole leak surface of an online predictor.

    ``step`` must return a set built from information available before the outcome. If
    the label influenced the set it emitted, coverage would be trivially perfect.
    """
    rng = np.random.default_rng(1)
    p_cal, y_cal = _planted_block(rng, 2000, 0.4)
    online = AdaptiveConformal(alpha_target=0.10, gamma=0.005, score="aps").fit(
        p_cal, y_cal)

    row = np.array([0.5, 0.3, 0.2])
    before = online.alpha_t
    sets = [AdaptiveConformal(alpha_target=0.10, gamma=0.005, score="aps")
            .fit(p_cal, y_cal).step(row, label=k) for k in range(3)]
    assert all(np.array_equal(sets[0], s) for s in sets[1:]), (
        "the emitted set depends on the label it has not seen yet"
    )
    assert online.alpha_t == before, "fitting alone must not move alpha"


def test_adaptive_quantile_lookup_equals_the_reference_exactly():
    """The online quantile is an index into pre-sorted scores; it must not approximate.

    alpha_t moves on every observation, so the naive implementation recomputed a full
    quantile over the whole calibration set once per row. Reading it off a sorted array
    instead is only legitimate if it agrees with ``conformal_quantile`` bit for bit --
    and the obvious index, ceil(level*n)-1, does not: numpy's ``method="higher"`` scales
    by n-1, and the difference shows up in the fourth decimal.
    """
    rng = np.random.default_rng(0)
    for n in (37, 500, 4096):
        scores = rng.random(n)
        online = AdaptiveConformal(alpha_target=0.10, gamma=0.005, score="aps")
        online._calibration_scores = scores
        online._sorted_scores = np.sort(scores)
        for alpha in np.linspace(0.001, 0.5, 120):
            online.alpha_t = float(alpha)
            assert online._quantile() == conformal_quantile(scores, float(alpha)), (
                f"indexed quantile disagrees at n={n}, alpha={alpha:.4f}"
            )


def test_adaptive_run_is_not_quadratic_in_the_test_block():
    """Guards the fix: doubling the block must not quadruple the work."""
    import time

    rng = np.random.default_rng(2)

    def timed(n):
        y = rng.integers(0, 3, n)
        p = rng.random((n, 3)); p /= p.sum(axis=1, keepdims=True)
        yt = rng.integers(0, 3, n)
        pt = rng.random((n, 3)); pt /= pt.sum(axis=1, keepdims=True)
        online = AdaptiveConformal(alpha_target=0.10, gamma=0.005, score="aps").fit(p, y)
        start = time.perf_counter()
        online.run(pt, yt)
        return time.perf_counter() - start

    timed(500)                       # warm the interpreter
    small, large = timed(1000), timed(4000)
    ratio = large / max(small, 1e-6)
    assert ratio < 8.0, (
        f"4x the rows cost {ratio:.1f}x the time; the per-step quantile is scaling with "
        "the calibration set again"
    )
