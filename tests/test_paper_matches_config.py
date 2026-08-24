"""Appendix C claims that a test asserts every constant it quotes. This is that test.

The manuscript states a list of hyperparameters in prose. Prose does not recompile when
a configuration changes, so without a check the two drift apart quietly and the paper
ends up describing a model nobody ran. That is not hypothetical here: the training loop
had itself drifted from the configuration on four separate settings before this suite
grew a check for it.

Patterns are matched against the manuscript with whitespace collapsed, because LaTeX
line-wrapping is free to split ``dates per\\nbatch 16`` wherever it likes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from finsent.config import load_config

PAPER = Path("paper/main.tex")

pytestmark = pytest.mark.skipif(not PAPER.exists(), reason="manuscript not present")


@pytest.fixture(scope="module")
def appendix() -> str:
    text = PAPER.read_text(encoding="utf-8")
    start = text.index("All constants live in a single configuration")
    end = text.index("Every generated table", start)
    return " ".join(text[start:end].split())


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# (label, configured value, regex with one capturing group)
CASES = [
    ("lookback",        lambda c: c.data.lookback_L,         r"Lookback \$L=(\d+)\$"),
    ("max headlines",   lambda c: c.data.K,                  r"maximum headlines \$K=(\d+)\$"),
    ("horizon",         lambda c: c.data.horizon_h,          r"horizon \$h=(\d+)\$"),
    ("features",        lambda c: c.data.n_features,         r"features \$F=(\d+)\$"),
    ("epochs",          lambda c: c.train.epochs,            r"epochs (\d+) with patience"),
    ("patience",        lambda c: c.train.patience,          r"with patience (\d+)"),
    ("dates per batch", lambda c: c.train.dates_per_batch,   r"dates per batch (\d+)"),
    ("ema decay",       lambda c: c.train.ema_decay,         r"EMA decay \$([\d.]+)\$"),
    ("kappa",           lambda c: c.decision.kappa,          r"\\kappa=([\d.]+)"),
    ("f_max",           lambda c: c.decision.f_max,          r"f_\{\\max\}=([\d.]+)"),
    ("rebalance days",  lambda c: c.decision.rebalance_days, r"rebalance every (\d+) sessions"),
    ("purge",           lambda c: c.data.purge_days,         r"purge (\d+) sessions"),
    ("lambda_reg",      lambda c: c.loss.lambda_reg,         r"=\((\d\.\d),[\d.,]+\)"),
    ("lambda_cal",      lambda c: c.loss.lambda_cal,         r"=\(\d\.\d,(\d\.\d),\d\.\d\)"),
    ("lambda_rank",     lambda c: c.loss.lambda_rank,        r"=\(\d\.\d,\d\.\d,(\d\.\d)\)"),
]


@pytest.mark.parametrize("label,getter,pattern",
                         [c for c in CASES if c[2] is not None],
                         ids=[c[0] for c in CASES if c[2] is not None])
def test_quoted_constant_matches_the_configuration(appendix, cfg, label, getter, pattern):
    match = re.search(pattern, appendix)
    assert match, f"Appendix C no longer quotes {label}; pattern {pattern!r} found nothing"
    quoted = float(match.group(1))
    configured = float(getter(cfg))
    assert quoted == pytest.approx(configured), (
        f"{label}: the manuscript says {quoted} and the configuration says {configured}"
    )


def test_learning_rate_and_weight_decay_match(appendix, cfg):
    """These two are written in scientific notation, so they need their own parse."""
    lr = re.search(r"learning rate \$(\d+)\\times10\^\{(-?\d+)\}\$", appendix)
    assert lr, "Appendix C no longer quotes a learning rate"
    assert float(f"{lr.group(1)}e{lr.group(2)}") == pytest.approx(cfg.train.lr)

    wd = re.search(r"weight decay \$10\^\{(-?\d+)\}\$", appendix)
    assert wd, "Appendix C no longer quotes a weight decay"
    assert float(f"1e{wd.group(1)}") == pytest.approx(cfg.train.weight_decay)


def test_receptive_field_claim_is_arithmetically_true(appendix):
    """The paper asserts the TCN sees the whole window. Check the arithmetic, not the prose.

    R = 1 + 2(k-1) * sum(dilations) must exceed the lookback, or the deepest dilations
    read padding and the capacity spent on them buys nothing.
    """
    dil = re.search(r"TCN dilations \$\\\{([\d,]+)\\\}\$ with receptive field (\d+)", appendix)
    assert dil, "Appendix C no longer quotes the dilations and receptive field"
    dilations = [int(d) for d in dil.group(1).split(",")]
    claimed = int(dil.group(2))
    kernel = 3
    computed = 1 + 2 * (kernel - 1) * sum(dilations)
    assert computed == claimed, f"receptive field is {computed}, paper says {claimed}"

    lookback = int(re.search(r"Lookback \$L=(\d+)\$", appendix).group(1))
    assert claimed > lookback, (
        f"receptive field {claimed} does not cover the {lookback}-session window"
    )


def test_every_quoted_parameter_count_matches_the_configuration():
    """The manuscript states the trainable-parameter total in four separate places.

    It is a static architectural fact, not a run output, so it stays typed in the prose
    and is guarded here instead of being turned into a generated macro. Four copies of a
    number is three opportunities for one of them to go stale.
    """
    text = PAPER.read_text(encoding="utf-8")
    expected = load_config().model.expected_trainable

    quoted = re.findall(r"\\num\{(\d{1,3})\{,\}(\d{3})\}", text)
    totals = [int(a + b) for a, b in quoted]
    assert totals, "the manuscript no longer quotes a parameter total"

    assert totals.count(expected) >= 4, (
        f"expected the parameter total {expected} in at least four places, found "
        f"{totals.count(expected)}"
    )
    # A near miss is the failure mode that matters: 448,543 for 448,453 reads as correct
    # and is not. Other quoted numbers (the effective sample size, for one) are free to
    # differ, so only flag values close enough to be a corrupted copy of this one.
    near_misses = [t for t in set(totals)
                   if t != expected and abs(t - expected) < 0.02 * expected]
    assert not near_misses, (
        f"the manuscript quotes {near_misses}, which look like stale copies of {expected}"
    )


def test_the_split_protocol_matches_the_configuration():
    """The Splits paragraph is prose, and prose does not recompile when a split changes."""
    text = " ".join(PAPER.read_text(encoding="utf-8").split())
    start = text.index("Purged, embargoed walk-forward")
    para = text[start:text.index("Combinatorially purged", start)]
    d = load_config().data

    years = re.search(r"expanding training window of at least (\w+) years", para)
    assert years, "the training-window minimum is no longer stated"
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "nine": 9, "twelve": 12}
    assert words.get(years.group(1).lower()) == d.train_min_years

    months = re.search(r"(\w+)-month inner-validation and test blocks", para)
    assert months, "the block lengths are no longer stated"
    assert words.get(months.group(1).lower()) == d.inner_val_months == d.test_months

    refit = re.search(r"refitted every (\w+) months", para)
    assert refit and words.get(refit.group(1).lower()) == d.refit_every_months

    # The purge is stated as "equal to h" rather than as a number, so check the identity
    # the sentence asserts rather than a literal.
    assert "purge equal to $h$" in para, "the purge is no longer tied to the horizon"
    assert d.purge_days == d.horizon_h, (
        f"the paper says purge equals h, but the configuration has purge={d.purge_days} "
        f"and h={d.horizon_h}"
    )

    embargo = re.search(r"embargo ([\d.]+)\\% of the sample", para)
    assert embargo, "the embargo is no longer stated"
    assert float(embargo.group(1)) / 100.0 == pytest.approx(d.embargo_pct)
