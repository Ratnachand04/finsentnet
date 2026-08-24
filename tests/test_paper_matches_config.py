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
