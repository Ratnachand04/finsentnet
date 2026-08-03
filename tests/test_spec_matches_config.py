"""SPEC.md and configs/base.yaml must agree, mechanically.

The eight contradictions catalogued in SPEC.md section 1 existed because the paper, the
mathematical companion and the code each carried their own copy of the same constant.
This test is the mechanism that stops that recurring: every constant SPEC.md declares
frozen is read back out of the configuration and compared.

If a value legitimately changes, change it in ``configs/base.yaml`` and in ``SPEC.md``,
and this test will confirm they moved together.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from finsent.config import LABEL_DOWN, LABEL_NEUTRAL, LABEL_UP, ConfigError, Config, load_config

SPEC_PATH = Path(__file__).resolve().parents[1] / "SPEC.md"


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="module")
def spec_text() -> str:
    return SPEC_PATH.read_text(encoding="utf-8")


def test_label_convention_is_frozen():
    assert (LABEL_DOWN, LABEL_NEUTRAL, LABEL_UP) == (0, 1, 2), (
        "contradiction #1: the label convention is frozen at DOWN=0, NEUTRAL=1, UP=2"
    )


def test_spec_declares_the_same_shapes_as_the_config(cfg, spec_text):
    for symbol, value in (
        ("L", cfg.data.lookback_L),
        ("F", cfg.data.n_features),
        ("h", cfg.data.horizon_h),
        ("K", cfg.data.K),
        ("d_model", cfg.model.d_model),
    ):
        pattern = rf"\|\s*`{re.escape(symbol)}`\s*\|\s*{value}\s*\|"
        assert re.search(pattern, spec_text), (
            f"SPEC.md does not declare {symbol} = {value}; the manuscript and the "
            "configuration have drifted apart again"
        )


def test_feature_list_length_matches_declared_count(cfg):
    assert len(cfg.data.feature_list) == cfg.data.n_features == 12, (
        "contradiction #6: F must be 12 and the list must have exactly that many entries"
    )


def test_patience_is_a_single_value(cfg, spec_text):
    assert cfg.train.patience == 8, "contradiction #5: patience is frozen at 8"
    assert "**8**" in spec_text or "patience" in spec_text


def test_class_weights_are_inverse_frequency(cfg):
    assert cfg.loss.class_weights == "inverse_frequency", (
        "contradiction #2: a fixed weight vector that up-weights the majority class "
        "(V2 used (0.3, 0.4, 0.3) with NEUTRAL as the majority) is a bug"
    )


def test_purge_equals_horizon(cfg):
    assert cfg.data.purge_days == cfg.data.horizon_h, (
        "a purge narrower than the label horizon leaves overlapping labels in training"
    )


def test_receptive_field_covers_the_lookback(cfg):
    assert cfg.model.tcn_receptive_field >= cfg.data.lookback_L, (
        f"receptive field {cfg.model.tcn_receptive_field} < L={cfg.data.lookback_L}: "
        "the deepest dilations would read only padding"
    )


def test_early_stopping_is_not_a_financial_metric(cfg):
    assert not cfg.train.early_stop_metric.endswith("sharpe"), (
        "V2 stopped on validation Sharpe while also calibrating on the same data; "
        "that is the triple-dipping defect"
    )


def test_trial_budget_is_declared(cfg):
    assert cfg.eval.n_configs_evaluated >= 1, (
        "the number of evaluated configurations must be declared; it feeds the "
        "Deflated Sharpe Ratio"
    )


def test_news_lag_is_at_least_24_hours(cfg):
    assert cfg.data.min_lag_hours >= 24


def test_validation_rejects_a_reintroduced_contradiction():
    """The validator must actually reject bad configurations, not just document them."""
    cfg = load_config()
    bad = dict(cfg.raw)
    bad["loss"] = {**bad["loss"], "class_weights": "fixed_0.3_0.4_0.3"}
    with pytest.raises(ConfigError, match="inverse_frequency"):
        Config.from_mapping(bad)

    bad2 = dict(cfg.raw)
    bad2["train"] = {**bad2["train"], "early_stop_metric": "val_sharpe"}
    with pytest.raises(ConfigError, match="triple-dipping"):
        Config.from_mapping(bad2)

    bad3 = dict(cfg.raw)
    bad3["data"] = {
        **bad3["data"],
        "splits": {**bad3["data"]["splits"], "purge_days": 1},
    }
    with pytest.raises(ConfigError, match="purge_days"):
        Config.from_mapping(bad3)


def test_spec_forbids_the_prohibited_components(spec_text):
    """The prohibitions are part of the specification, not folklore."""
    for phrase in ("TechScore", "token-id bridge", "if/else", "shuffling"):
        assert phrase.lower() in spec_text.lower(), (
            f"SPEC.md section 8 must explicitly prohibit '{phrase}'"
        )


def test_config_hash_is_stable_and_short(cfg):
    assert len(cfg.config_hash) == 12
    assert cfg.config_hash == load_config().config_hash
