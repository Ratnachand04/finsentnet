"""Typed configuration for FinSentNet-C.

`configs/base.yaml` is the single source of truth (see ``SPEC.md``). This module loads
it into frozen dataclasses, validates the invariants that the specification requires,
and computes a stable ``config_hash`` that is stamped onto every artifact.

Design rule enforced by ``tests/test_spec_matches_config.py``: no constant that appears
in the manuscript may be hard-coded in Python. If you need a number, read it from here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "LossConfig",
    "TrainConfig",
    "CalibrationConfig",
    "ConformalConfig",
    "DecisionConfig",
    "EvalConfig",
    "load_config",
    "DEFAULT_CONFIG_PATH",
    "LABEL_DOWN",
    "LABEL_NEUTRAL",
    "LABEL_UP",
    "LABEL_NAMES",
]

# --- Frozen label convention (SPEC.md 2.1). Never re-derive these. ------------------
LABEL_DOWN: int = 0
LABEL_NEUTRAL: int = 1
LABEL_UP: int = 2
LABEL_NAMES: tuple[str, str, str] = ("DOWN", "NEUTRAL", "UP")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


class ConfigError(ValueError):
    """Raised when a configuration violates a specification invariant."""


def _get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Fetch ``a.b.c`` from a nested mapping, returning ``default`` if absent."""
    node: Any = mapping
    for key in path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


@dataclass(frozen=True)
class DataConfig:
    universe_mode: str
    n_names: int
    min_price: float
    min_adv_usd: float
    universe_lookback_days: int
    start: str
    end: str
    news_source: str
    max_headlines_per_day: int
    news_lookback_hours: int
    min_lag_hours: int
    encoder: str
    embedding_dim: int
    n_sources: int
    lookback_L: int
    n_features: int
    feature_list: tuple[str, ...]
    cross_sectional_rank: bool
    winsorize: float
    horizon_h: int
    label_method: str
    k_band: float
    vol_span: int
    pt_mult: float
    sl_mult: float
    vertical_days: int
    train_min_years: int
    train_mode: str
    inner_val_months: int
    test_months: int
    refit_every_months: int
    purge_days: int
    embargo_pct: float
    use_average_uniqueness: bool

    @property
    def K(self) -> int:  # noqa: N802 - matches the symbol used in SPEC.md
        return self.max_headlines_per_day


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    conv_kernels: tuple[int, ...]
    conv_channels: int
    tcn_blocks: int
    tcn_dilations: tuple[int, ...]
    tcn_kernel: int
    tcn_channels: int
    tcn_dropout: float
    gru_hidden: int
    gru_layers: int
    gru_bidirectional: bool
    text_emb_dim: int
    text_proj_dim: int
    recency_dim: int
    source_emb_dim: int
    attn_hidden: int
    n_sources: int
    fusion_blocks: int
    fusion_heads: int
    modality_dropout_text: float
    modality_dropout_price: float
    n_classes: int
    head_hidden: int
    head_dropout: float
    heteroscedastic: bool
    logvar_clamp: tuple[float, float]
    expected_trainable: int
    param_tolerance_pct: float

    @property
    def tcn_receptive_field(self) -> int:
        """Receptive field of the dilated causal stack, in timesteps."""
        return 1 + 2 * (self.tcn_kernel - 1) * sum(self.tcn_dilations)


@dataclass(frozen=True)
class LossConfig:
    lambda_reg: float
    lambda_cal: float
    lambda_rank: float
    class_weights: str
    focal_enabled: bool
    focal_gamma: float
    label_smoothing: float
    sigma_warmup_epochs: int
    mmce_kernel_width: float


@dataclass(frozen=True)
class TrainConfig:
    optimizer: str
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    warmup_pct: float
    epochs: int
    patience: int
    dates_per_batch: int
    grad_clip: float
    ema_decay: float
    precision: str
    early_stop_metric: str


@dataclass(frozen=True)
class CalibrationConfig:
    method: str
    fit_on: str
    n_bins: int
    max_iter: int


@dataclass(frozen=True)
class ConformalConfig:
    score: str
    alphas: tuple[float, ...]
    adaptive_enabled: bool
    adaptive_gamma: float


@dataclass(frozen=True)
class DecisionConfig:
    score: str
    gate_on_singleton: bool
    kappa: float
    f_max: float
    covariance: str
    risk_aversion: float
    turnover_penalty: float
    gross_leverage_max: float
    dollar_neutral: bool
    long_short_quantile: float
    default_bps: float
    impact_eta: float
    sweep_bps: tuple[float, ...]
    rebalance_days: int
    regime_n_states: int
    regime_max_iter: int


@dataclass(frozen=True)
class EvalConfig:
    primary: tuple[str, ...]
    periods_per_year: int
    bootstrap_type: str
    block_mean: int
    n_resamples: int
    ci_level: float
    hac_lags: int
    n_configs_evaluated: int
    risk_free_rate: float


@dataclass(frozen=True)
class Config:
    """Root configuration object."""

    name: str
    version: str
    seed_list: tuple[int, ...]
    deterministic: bool
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    train: TrainConfig
    calibration: CalibrationConfig
    conformal: ConformalConfig
    decision: DecisionConfig
    eval: EvalConfig
    paths: Mapping[str, str]
    raw: Mapping[str, Any] = field(repr=False, default_factory=dict)

    # -- construction ---------------------------------------------------------------
    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "Config":
        g = lambda p, d=None: _get(cfg, p, d)  # noqa: E731 - local shorthand

        data = DataConfig(
            universe_mode=g("data.universe.mode", "pit_liquidity"),
            n_names=int(g("data.universe.n_names", 400)),
            min_price=float(g("data.universe.min_price", 5.0)),
            min_adv_usd=float(g("data.universe.min_adv_usd", 5.0e6)),
            universe_lookback_days=int(g("data.universe.lookback_days", 60)),
            start=str(g("data.period.start")),
            end=str(g("data.period.end")),
            news_source=str(g("data.news.source", "fnspid")),
            max_headlines_per_day=int(g("data.news.max_headlines_per_day", 8)),
            news_lookback_hours=int(g("data.news.lookback_hours", 72)),
            min_lag_hours=int(g("data.news.min_lag_hours", 24)),
            encoder=str(g("data.news.encoder")),
            embedding_dim=int(g("data.news.embedding_dim", 768)),
            n_sources=int(g("data.news.n_sources", 32)),
            lookback_L=int(g("data.features.lookback_L", 60)),
            n_features=int(g("data.features.n_features", 12)),
            feature_list=tuple(g("data.features.list", []) or []),
            cross_sectional_rank=bool(g("data.features.cross_sectional_rank", True)),
            winsorize=float(g("data.features.winsorize", 0.01)),
            horizon_h=int(g("data.labels.horizon_h", 5)),
            label_method=str(g("data.labels.method", "vol_scaled_band")),
            k_band=float(g("data.labels.k_band", 0.6)),
            vol_span=int(g("data.labels.vol_span", 60)),
            pt_mult=float(g("data.labels.triple_barrier.pt_mult", 2.0)),
            sl_mult=float(g("data.labels.triple_barrier.sl_mult", 2.0)),
            vertical_days=int(g("data.labels.triple_barrier.vertical_days", 5)),
            train_min_years=int(g("data.splits.train_min_years", 3)),
            train_mode=str(g("data.splits.train_mode", "expanding")),
            inner_val_months=int(g("data.splits.inner_val_months", 6)),
            test_months=int(g("data.splits.test_months", 6)),
            refit_every_months=int(g("data.splits.refit_every_months", 6)),
            purge_days=int(g("data.splits.purge_days", 5)),
            embargo_pct=float(g("data.splits.embargo_pct", 0.01)),
            use_average_uniqueness=bool(g("data.weights.use_average_uniqueness", True)),
        )

        model = ModelConfig(
            d_model=int(g("model.d_model", 128)),
            conv_kernels=tuple(int(k) for k in g("model.price.conv_kernels", [3, 7, 15])),
            conv_channels=int(g("model.price.conv_channels", 32)),
            tcn_blocks=int(g("model.price.tcn.blocks", 4)),
            tcn_dilations=tuple(int(d) for d in g("model.price.tcn.dilations", [1, 2, 4, 8])),
            tcn_kernel=int(g("model.price.tcn.kernel", 3)),
            tcn_channels=int(g("model.price.tcn.channels", 64)),
            tcn_dropout=float(g("model.price.tcn.dropout", 0.1)),
            gru_hidden=int(g("model.price.gru.hidden", 128)),
            gru_layers=int(g("model.price.gru.layers", 1)),
            gru_bidirectional=bool(g("model.price.gru.bidirectional", False)),
            text_emb_dim=int(g("model.text.emb_dim", 768)),
            text_proj_dim=int(g("model.text.proj_dim", 128)),
            recency_dim=int(g("model.text.recency_dim", 16)),
            source_emb_dim=int(g("model.text.source_emb_dim", 8)),
            attn_hidden=int(g("model.text.attn_hidden", 64)),
            n_sources=int(g("data.news.n_sources", 32)),
            fusion_blocks=int(g("model.fusion.blocks", 1)),
            fusion_heads=int(g("model.fusion.heads", 4)),
            modality_dropout_text=float(g("model.fusion.modality_dropout.text", 0.15)),
            modality_dropout_price=float(g("model.fusion.modality_dropout.price", 0.15)),
            n_classes=int(g("model.heads.n_classes", 3)),
            head_hidden=int(g("model.heads.hidden", 128)),
            head_dropout=float(g("model.heads.dropout", 0.1)),
            heteroscedastic=bool(g("model.heads.heteroscedastic", True)),
            logvar_clamp=tuple(float(v) for v in g("model.heads.logvar_clamp", [-10.0, 2.0])),
            expected_trainable=int(g("model.param_budget.expected_trainable", 452000)),
            param_tolerance_pct=float(g("model.param_budget.tolerance_pct", 5.0)),
        )

        loss = LossConfig(
            lambda_reg=float(g("loss.lambda_reg", 1.0)),
            lambda_cal=float(g("loss.lambda_cal", 0.5)),
            lambda_rank=float(g("loss.lambda_rank", 0.2)),
            class_weights=str(g("loss.class_weights", "inverse_frequency")),
            focal_enabled=bool(g("loss.focal.enabled", False)),
            focal_gamma=float(g("loss.focal.gamma", 2.0)),
            label_smoothing=float(g("loss.label_smoothing", 0.0)),
            sigma_warmup_epochs=int(g("loss.sigma_warmup_epochs", 3)),
            mmce_kernel_width=float(g("loss.mmce_kernel_width", 0.4)),
        )

        train = TrainConfig(
            optimizer=str(g("train.optimizer", "adamw")),
            lr=float(g("train.lr", 3.0e-4)),
            weight_decay=float(g("train.weight_decay", 1.0e-2)),
            betas=tuple(float(b) for b in g("train.betas", [0.9, 0.98])),
            warmup_pct=float(g("train.warmup_pct", 0.05)),
            epochs=int(g("train.epochs", 30)),
            patience=int(g("train.patience", 8)),
            dates_per_batch=int(g("train.dates_per_batch", 16)),
            grad_clip=float(g("train.grad_clip", 1.0)),
            ema_decay=float(g("train.ema_decay", 0.999)),
            precision=str(g("train.precision", "bf16")),
            early_stop_metric=str(g("train.early_stop_metric", "inner_val_nll")),
        )

        calibration = CalibrationConfig(
            method=str(g("calibration.method", "vector_scaling")),
            fit_on=str(g("calibration.fit_on", "inner_val")),
            n_bins=int(g("calibration.n_bins", 15)),
            max_iter=int(g("calibration.max_iter", 200)),
        )

        conformal = ConformalConfig(
            score=str(g("conformal.score", "aps")),
            alphas=tuple(float(a) for a in g("conformal.alphas", [0.05, 0.10, 0.20])),
            adaptive_enabled=bool(g("conformal.adaptive.enabled", True)),
            adaptive_gamma=float(g("conformal.adaptive.gamma", 0.005)),
        )

        decision = DecisionConfig(
            score=str(g("decision.score", "mu_hat")),
            gate_on_singleton=bool(g("decision.gate_on_singleton", True)),
            kappa=float(g("decision.kelly.kappa", 0.25)),
            f_max=float(g("decision.kelly.f_max", 0.10)),
            covariance=str(g("decision.portfolio.covariance", "ledoit_wolf")),
            risk_aversion=float(g("decision.portfolio.risk_aversion", 5.0)),
            turnover_penalty=float(g("decision.portfolio.turnover_penalty", 0.001)),
            gross_leverage_max=float(g("decision.portfolio.gross_leverage_max", 2.0)),
            dollar_neutral=bool(g("decision.portfolio.dollar_neutral", True)),
            long_short_quantile=float(g("decision.portfolio.long_short_quantile", 0.1)),
            default_bps=float(g("decision.costs.default_bps", 10.0)),
            impact_eta=float(g("decision.costs.impact_eta", 0.5)),
            sweep_bps=tuple(float(b) for b in g("decision.costs.sweep_bps", [0.0, 10.0])),
            rebalance_days=int(g("decision.rebalance_days", 5)),
            regime_n_states=int(g("decision.regime.n_states", 3)),
            regime_max_iter=int(g("decision.regime.max_iter", 200)),
        )

        ev = EvalConfig(
            primary=tuple(g("eval.primary", []) or []),
            periods_per_year=int(g("eval.periods_per_year", 252)),
            bootstrap_type=str(g("eval.bootstrap.type", "stationary")),
            block_mean=int(g("eval.bootstrap.block_mean", 10)),
            n_resamples=int(g("eval.bootstrap.n_resamples", 5000)),
            ci_level=float(g("eval.bootstrap.ci_level", 0.95)),
            hac_lags=int(g("eval.dm_test.hac_lags", 10)),
            n_configs_evaluated=int(g("eval.n_configs_evaluated", 1)),
            risk_free_rate=float(g("eval.risk_free_rate", 0.0)),
        )

        obj = cls(
            name=str(g("meta.name", "finsentnet_c")),
            version=str(g("meta.version", "0.0.0")),
            seed_list=tuple(int(s) for s in g("meta.seed_list", [0])),
            deterministic=bool(g("meta.deterministic", True)),
            data=data,
            model=model,
            loss=loss,
            train=train,
            calibration=calibration,
            conformal=conformal,
            decision=decision,
            eval=ev,
            paths=dict(g("paths", {}) or {}),
            raw=dict(cfg),
        )
        obj.validate()
        return obj

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls.from_mapping(cfg)

    # -- invariants -----------------------------------------------------------------
    def validate(self) -> None:
        """Assert the invariants that SPEC.md declares non-negotiable."""
        d, m = self.data, self.model

        if len(d.feature_list) != d.n_features:
            raise ConfigError(
                f"data.features.list has {len(d.feature_list)} entries but "
                f"n_features={d.n_features}. SPEC.md 2.6 requires exactly one list."
            )

        if m.tcn_receptive_field < d.lookback_L:
            raise ConfigError(
                f"TCN receptive field {m.tcn_receptive_field} < lookback_L={d.lookback_L}: "
                "the deepest dilations would read only padding (SPEC.md 2.2)."
            )

        if d.purge_days != d.horizon_h:
            raise ConfigError(
                f"splits.purge_days={d.purge_days} must equal labels.horizon_h="
                f"{d.horizon_h}; a narrower purge leaks overlapping labels (SPEC.md 2.5)."
            )

        if d.min_lag_hours < 24:
            raise ConfigError(
                f"news.min_lag_hours={d.min_lag_hours} < 24 (SPEC.md 2.3)."
            )

        if self.loss.class_weights != "inverse_frequency":
            raise ConfigError(
                "loss.class_weights must be 'inverse_frequency'. Contradiction #2 in "
                "SPEC.md: a fixed vector that up-weights the majority class is a bug."
            )

        if self.train.early_stop_metric.endswith("sharpe"):
            raise ConfigError(
                "train.early_stop_metric must not be a financial metric: early stopping "
                "on validation Sharpe while also calibrating there is the V2 "
                "triple-dipping defect (SPEC.md 2.5)."
            )

        if m.n_classes != 3:
            raise ConfigError("model.heads.n_classes must be 3 (DOWN/NEUTRAL/UP).")

        if m.d_model % m.fusion_heads != 0:
            raise ConfigError(
                f"d_model={m.d_model} is not divisible by fusion heads={m.fusion_heads}."
            )

        if not 0.0 < self.decision.kappa <= 1.0:
            raise ConfigError("decision.kelly.kappa must lie in (0, 1].")

        if self.eval.n_configs_evaluated < 1:
            raise ConfigError(
                "eval.n_configs_evaluated must be declared (>=1); it feeds the "
                "Deflated Sharpe Ratio (SPEC.md 6.3)."
            )

        label_map = _get(self.raw, "data.labels.convention", {}) or {}
        expected = {"DOWN": LABEL_DOWN, "NEUTRAL": LABEL_NEUTRAL, "UP": LABEL_UP}
        if label_map and dict(label_map) != expected:
            raise ConfigError(
                f"label convention {dict(label_map)} != frozen {expected} (SPEC.md 2.1)."
            )

    # -- identity -------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("raw", None)
        return out

    @property
    def config_hash(self) -> str:
        """Stable 12-hex-character digest of the resolved configuration."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def describe(self) -> str:
        m = self.model
        return (
            f"{self.name} v{self.version} [{self.config_hash}]  "
            f"L={self.data.lookback_L} F={self.data.n_features} h={self.data.horizon_h} "
            f"d_model={m.d_model} rf={m.tcn_receptive_field}"
        )


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the configuration (defaults to ``configs/base.yaml``)."""
    return Config.from_yaml(path or DEFAULT_CONFIG_PATH)
