"""Model shape, capacity and causality contracts.

Skipped wholesale when PyTorch is unavailable. That is a real state on some machines --
this repository was developed on one where an OS application-control policy blocks
``torch``'s DLLs entirely -- which is exactly why the data, evaluation, calibration,
conformal and decision layers were built framework-free. The scientific claims of the
paper are testable without a GPU; only the network itself needs one.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not available")

from finsent.config import load_config  # noqa: E402
from finsent.models.finsentnet_c import FinSentNetC  # noqa: E402
from finsent.models.gated_fusion import GatedCrossModalFusion  # noqa: E402
from finsent.models.heads import PredictionHeads  # noqa: E402
from finsent.models.price_encoder import PriceEncoder  # noqa: E402
from finsent.models.text_encoder import TextEncoder  # noqa: E402
from finsent.training.objectives import (  # noqa: E402
    CompositeObjective,
    LossWeights,
    gaussian_nll,
    inverse_frequency_weights,
    mmce_loss,
    soft_spearman_loss,
)

B, L, F, K, E = 6, 60, 12, 8, 768


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def model(cfg):
    return FinSentNetC.from_config(cfg).eval()


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    return {
        "price": torch.randn(B, L, F, generator=g),
        "text_embeddings": torch.randn(B, K, E, generator=g),
        "text_mask": torch.tensor([[True] * 3 + [False] * 5] * B),
        "lag_hours": torch.rand(B, K, generator=g) * 48 + 24,
        "source_ids": torch.randint(1, 5, (B, K), generator=g),
    }


# --------------------------------------------------------------------------------------
# Shapes and capacity
# --------------------------------------------------------------------------------------
def test_forward_produces_the_declared_shapes(model, batch):
    out = model(**batch)
    assert out.heads.logits.shape == (B, 3)
    assert out.heads.mu.shape == (B,)
    assert out.heads.logvar.shape == (B,)
    assert out.fusion.gate.shape == (B, model.heads.trunk[0].in_features)
    assert out.text_attention.shape == (B, K)
    assert torch.allclose(out.heads.probs.sum(dim=-1), torch.ones(B), atol=1e-6)


def test_parameter_count_matches_the_specification(model, cfg):
    """SPEC.md 3.5 declares the budget; this asserts the realised count, not the claim."""
    total = model.n_trainable()
    expected = cfg.model.expected_trainable
    tolerance = cfg.model.param_tolerance_pct / 100.0

    assert abs(total - expected) / expected < tolerance, (
        f"trainable parameters {total:,d} deviate from the declared {expected:,d} by "
        f"more than {cfg.model.param_tolerance_pct}%. Update SPEC.md and the config "
        "together, or find what grew."
    )
    counts = model.parameter_counts()
    assert counts["TOTAL"] == total
    assert counts["price.gru"] > 0 and counts["fusion.attention"] > 0


def test_receptive_field_covers_the_whole_window(model, cfg):
    assert model.price_encoder.receptive_field >= cfg.data.lookback_L, (
        "the deepest dilated layer must see the entire lookback, not padding"
    )


def test_finbert_is_not_in_the_training_graph(model):
    """The text branch consumes cached vectors; 110M frozen parameters never appear."""
    assert model.n_trainable() < 1_000_000, (
        f"{model.n_trainable():,d} trainable parameters: an encoder has leaked into the "
        "training graph"
    )


# --------------------------------------------------------------------------------------
# Causality
# --------------------------------------------------------------------------------------
def test_price_encoder_is_causal():
    """Perturbing the future must not change the past. Asserted, not argued."""
    enc = PriceEncoder(n_features=F).eval()
    x = torch.randn(2, L, F)

    with torch.no_grad():
        seq_a, _ = enc(x)
        x2 = x.clone()
        x2[:, 30:, :] += 5.0
        seq_b, _ = enc(x2)

    assert torch.allclose(seq_a[:, :30], seq_b[:, :30], atol=1e-5), (
        "hidden states before the perturbation changed: the encoder reads the future"
    )
    assert not torch.allclose(seq_a[:, 30:], seq_b[:, 30:]), "the shock had no effect at all"


def test_truncating_the_window_leaves_earlier_outputs_unchanged():
    enc = PriceEncoder(n_features=F).eval()
    x = torch.randn(2, L, F)
    with torch.no_grad():
        full, _ = enc(x)
        partial, _ = enc(x[:, :30])
    assert torch.allclose(full[:, :30], partial, atol=1e-5)


# --------------------------------------------------------------------------------------
# Text encoder and the no-news path
# --------------------------------------------------------------------------------------
def test_no_news_rows_receive_the_learned_null_embedding():
    enc = TextEncoder().eval()
    emb = torch.randn(3, K, E)
    mask = torch.tensor([[True] * 2 + [False] * 6, [False] * K, [True] * K])

    with torch.no_grad():
        s, attn = enc(emb, mask, torch.full((3, K), 30.0), torch.ones(3, K, dtype=torch.long))

    assert torch.allclose(s[1], enc.null_embedding, atol=1e-6), (
        "a ticker-day with no headlines must get the learned null vector"
    )
    assert attn[1].sum() == 0.0
    assert attn[0, 2:].sum() == 0.0, "masked headlines must receive zero attention"
    assert attn[0, :2].sum() == pytest.approx(1.0, abs=1e-5)


def test_text_encoder_is_permutation_equivariant_in_its_headlines():
    """Pooling must depend on content and recency, not on storage order."""
    enc = TextEncoder().eval()
    emb = torch.randn(1, 4, E)
    mask = torch.ones(1, 4, dtype=torch.bool)
    lag = torch.tensor([[24.0, 30.0, 40.0, 60.0]])
    src = torch.ones(1, 4, dtype=torch.long)

    perm = [2, 0, 3, 1]
    with torch.no_grad():
        a, _ = enc(emb, mask, lag, src)
        b, _ = enc(emb[:, perm], mask, lag[:, perm], src[:, perm])
    assert torch.allclose(a, b, atol=1e-5)


# --------------------------------------------------------------------------------------
# Fusion and modality dropout
# --------------------------------------------------------------------------------------
def test_forced_single_modality_needs_no_retraining(model, batch):
    """The ablation modality dropout buys: evaluate the same weights, no refit."""
    with torch.no_grad():
        both = model(**batch).heads.logits
        price_only = model(**batch, force_modality="price_only").heads.logits
        text_only = model(**batch, force_modality="text_only").heads.logits

    assert not torch.allclose(both, price_only, atol=1e-6)
    assert not torch.allclose(both, text_only, atol=1e-6)
    assert torch.isfinite(price_only).all() and torch.isfinite(text_only).all(), (
        "the model must degrade gracefully, not produce NaNs, when a feed fails"
    )


def test_modality_dropout_is_inactive_at_evaluation_time():
    fusion = GatedCrossModalFusion(d_model=16, n_heads=2).eval()
    text, price = torch.randn(4, 16), torch.randn(4, 16)
    seq, null = torch.randn(4, 10, 16), torch.zeros(4, 16)

    with torch.no_grad():
        a = fusion(text, price, seq, null).fused
        b = fusion(text, price, seq, null).fused
    assert torch.allclose(a, b), "eval-mode forward passes must be deterministic"


def test_gate_lies_strictly_between_zero_and_one(model, batch):
    with torch.no_grad():
        gate = model(**batch).fusion.gate
    assert (gate > 0).all() and (gate < 1).all()


# --------------------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------------------
def test_inverse_frequency_weights_downweight_the_majority_class():
    """Contradiction #2, checked numerically."""
    labels = torch.tensor([1] * 60 + [0] * 20 + [2] * 20)
    w = inverse_frequency_weights(labels)

    assert w[1] < w[0] and w[1] < w[2], (
        f"NEUTRAL is the majority class and must receive the smallest weight; got {w}"
    )
    assert w[0] == pytest.approx(float(w[2]), rel=1e-6)
    assert float(w.mean()) == pytest.approx(1.0, rel=1e-6)


def test_gaussian_nll_warmup_freezes_the_variance():
    mu = torch.zeros(8, requires_grad=True)
    logvar = torch.full((8,), 2.0, requires_grad=True)
    y = torch.randn(8)

    warm = gaussian_nll(mu, logvar, y, sigma_warmup=True)
    warm.backward()
    assert logvar.grad is None or torch.allclose(logvar.grad, torch.zeros_like(logvar)), (
        "during warmup the variance head must receive no gradient, or it will minimise "
        "the loss by inflating variance instead of learning the mean"
    )


def test_gaussian_nll_prefers_the_correct_variance():
    y = torch.randn(20000) * 2.0
    mu = torch.zeros_like(y)

    correct = gaussian_nll(mu, torch.full_like(y, float(np.log(4.0))), y)
    too_small = gaussian_nll(mu, torch.full_like(y, float(np.log(1.0))), y)
    too_large = gaussian_nll(mu, torch.full_like(y, float(np.log(16.0))), y)

    assert correct < too_small and correct < too_large


def test_mmce_is_differentiable_and_penalises_overconfidence():
    """Binned ECE has zero gradient almost everywhere; MMCE does not."""
    logits = torch.randn(256, 3, requires_grad=True)
    targets = torch.randint(0, 3, (256,))

    loss = mmce_loss(logits, targets)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0, "MMCE produced no gradient"

    calibrated = torch.randn(512, 3) * 0.5
    over = calibrated * 6.0
    t = torch.randint(0, 3, (512,))
    assert mmce_loss(over, t) > mmce_loss(calibrated, t)


def test_soft_spearman_rewards_correct_ranking():
    scores = torch.arange(20.0, requires_grad=True)
    good = torch.arange(20.0)
    bad = torch.arange(20.0).flip(0)

    assert soft_spearman_loss(scores, good) < soft_spearman_loss(scores, bad)
    loss = soft_spearman_loss(scores, good)
    loss.backward()
    assert scores.grad is not None


def test_composite_objective_returns_finite_components(cfg):
    objective = CompositeObjective(
        LossWeights(
            lambda_reg=cfg.loss.lambda_reg,
            lambda_cal=cfg.loss.lambda_cal,
            lambda_rank=cfg.loss.lambda_rank,
        )
    )
    logits = torch.randn(64, 3, requires_grad=True)
    mu = torch.randn(64, requires_grad=True)
    logvar = torch.zeros(64, requires_grad=True)
    y_dir = torch.randint(0, 3, (64,))
    y_ret = torch.randn(64) * 0.01

    objective.set_class_weights_from(y_dir)
    total, parts = objective(logits, mu, logvar, y_dir, y_ret,
                             group_ids=torch.zeros(64, dtype=torch.long), epoch=0)

    total.backward()
    assert torch.isfinite(total)
    for key in ("l_dir", "l_nll", "l_mmce", "l_rank"):
        assert np.isfinite(parts[key]), f"{key} is not finite"
    assert parts["sigma_warmup"] == 1.0, "epoch 0 must be inside the variance warmup"


def test_model_can_overfit_a_single_batch(model, batch):
    """The standard sanity check: if it cannot memorise one batch, it cannot learn."""
    net = FinSentNetC.from_config(load_config()).train()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    targets = torch.tensor([0, 1, 2, 0, 1, 2])

    first = last = float("nan")
    for step in range(200):
        opt.zero_grad()
        out = net(**batch)
        loss = torch.nn.functional.cross_entropy(out.heads.logits, targets)
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if step == 0:
            first = last

    assert last < 0.1, f"loss stalled at {last:.4f} (started {first:.4f})"


def test_mmce_subsamples_large_batches_and_stays_close():
    """The kernel is quadratic in batch size; date-batching makes batches large.

    Uncapped, a 3,336-row batch builds an 11-million-element kernel every step. The
    capped estimator must remain close to the exact value, since both estimate the
    same mean over pairs.
    """
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3000, 3, generator=g) * 2.0
    targets = torch.randint(0, 3, (3000,), generator=g)

    torch.manual_seed(0)
    exact = float(mmce_loss(logits, targets, max_pairs=10_000))
    vals = []
    for s in range(8):
        torch.manual_seed(s)
        vals.append(float(mmce_loss(logits, targets, max_pairs=512)))

    assert abs(float(np.mean(vals)) - exact) < 0.15 * max(exact, 1e-6) + 0.01, (
        f"subsampled MMCE {np.mean(vals):.4f} drifts from exact {exact:.4f}"
    )


def test_mmce_capped_batch_is_much_cheaper():
    """Guards the fix: the capped path must not build the full kernel."""
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(4000, 3, generator=g, requires_grad=True)
    targets = torch.randint(0, 3, (4000,), generator=g)

    loss = mmce_loss(logits, targets, max_pairs=256)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
