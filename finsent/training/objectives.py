"""The training objective, in four differentiable parts.

    L = L_dir + lambda_reg * L_NLL + lambda_cal * L_MMCE + lambda_rank * L_rank

Each part exists to fix something specific that was wrong in V2.

``weighted_cross_entropy``
    Class weights are **inverse frequency**, computed from the training fold. V2 used a
    fixed ``(0.3, 0.4, 0.3)`` that put the *largest* weight on NEUTRAL, the majority
    class -- the exact opposite of what the accompanying prose claimed the weights were
    doing. Per-sample average-uniqueness weights are applied too, so that a cluster of
    overlapping near-duplicate labels does not count as many independent observations.

``gaussian_nll``
    Trains the heteroscedastic head that the decision layer needs. The ``sigma_warmup``
    argument is not a nicety: without it the model minimises the objective by inflating
    variance on hard samples and never learns the mean, and the loss curve looks healthy
    the entire time.

``mmce_loss``
    Maximum Mean Calibration Error (Kumar, Sarawagi & Jain, ICML 2018). V2 described
    binned ECE as a "differentiable training penalty"; binned ECE is a step function of
    the parameters and has zero gradient almost everywhere. MMCE is a kernel statistic
    with a genuine gradient, and it is the only calibration term in the objective.

``soft_spearman_loss``
    Optimises the Rank-IC that the paper actually reports. Because batches are grouped by
    date, the ranking is cross-sectional, which is what a dollar-neutral portfolio trades.

Focal loss is available but **off by default**: Mukhoti et al. (2020) show it changes
calibration on its own (it tends to under-confide), which would interact with the MMCE
term in a way nobody could disentangle in an ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "inverse_frequency_weights",
    "weighted_cross_entropy",
    "focal_loss",
    "gaussian_nll",
    "mmce_loss",
    "soft_spearman_loss",
    "LossWeights",
    "CompositeObjective",
]

_EPS = 1e-12


def inverse_frequency_weights(labels: torch.Tensor, n_classes: int = 3) -> torch.Tensor:
    """``w_c = N / (C * n_c)``, normalised to mean 1.

    The majority class receives the *smallest* weight. Absent classes get weight 1
    rather than infinity, so a fold with no DOWN days does not produce a NaN loss.
    """
    counts = torch.bincount(labels.reshape(-1).long(), minlength=n_classes).float()
    total = counts.sum().clamp(min=1.0)
    weights = torch.where(
        counts > 0, total / (n_classes * counts.clamp(min=1.0)), torch.ones_like(counts)
    )
    return weights / weights.mean().clamp(min=_EPS)


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy with both class and per-sample weights."""
    losses = F.cross_entropy(
        logits,
        targets.long(),
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    if sample_weights is None:
        return losses.mean()
    w = sample_weights.reshape(-1).to(losses.dtype)
    return (losses * w).sum() / w.sum().clamp(min=_EPS)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal loss (Lin et al., 2017). Ablation only -- see the module docstring."""
    log_p = F.log_softmax(logits, dim=-1)
    idx = targets.long().reshape(-1, 1)
    log_pt = log_p.gather(1, idx).squeeze(1)
    pt = log_pt.exp()

    losses = -((1.0 - pt) ** gamma) * log_pt
    if class_weights is not None:
        losses = losses * class_weights[targets.long()]
    if sample_weights is None:
        return losses.mean()
    w = sample_weights.reshape(-1).to(losses.dtype)
    return (losses * w).sum() / w.sum().clamp(min=_EPS)


def gaussian_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    sigma_warmup: bool = False,
) -> torch.Tensor:
    """``0.5 * (logvar + (y - mu)^2 * exp(-logvar))``.

    When ``sigma_warmup`` is set the variance is held at 1 (log-variance detached and
    zeroed), so the first epochs train the mean alone. Skipping this is the standard way
    a heteroscedastic head silently fails: variance inflation is always an easier
    descent direction than fitting a weak conditional mean.
    """
    if sigma_warmup:
        logvar = torch.zeros_like(logvar)

    inv_var = torch.exp(-logvar)
    losses = 0.5 * (logvar + (target - mu) ** 2 * inv_var)

    if sample_weights is None:
        return losses.mean()
    w = sample_weights.reshape(-1).to(losses.dtype)
    return (losses * w).sum() / w.sum().clamp(min=_EPS)


def mmce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    kernel_width: float = 0.4,
) -> torch.Tensor:
    """Maximum Mean Calibration Error with a Laplacian kernel (Kumar et al., 2018).

    ``MMCE^2 = (1/n^2) sum_ij (c_i - r_i)(c_j - r_j) k(c_i, c_j)``

    where ``c`` is the top-class confidence and ``r`` the correctness indicator. The
    statistic is zero exactly when confidence and correctness agree in distribution, and
    unlike binned ECE it has a gradient everywhere.
    """
    probs = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == targets.long()).to(conf.dtype)

    diff = conf - correct
    kernel = torch.exp(-torch.cdist(conf.unsqueeze(1), conf.unsqueeze(1), p=1) / kernel_width)
    n = conf.shape[0]
    if n < 2:
        return conf.sum() * 0.0

    value = (diff.unsqueeze(0) @ kernel @ diff.unsqueeze(1)).squeeze() / (n * n)
    return torch.sqrt(value.clamp(min=_EPS))


def soft_spearman_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    group_ids: torch.Tensor | None = None,
    tau: float = 0.1,
) -> torch.Tensor:
    """Negative differentiable Spearman correlation, computed within each group.

    Soft ranks come from pairwise sigmoid comparisons,
    ``rank_i = 1 + sum_j sigmoid((s_j - s_i) / tau)``, which is ``O(n^2)`` per group --
    fine at a few hundred names per date and exact in the ``tau -> 0`` limit.

    Optimising this directly matters because Rank-IC is the number the paper reports;
    training on cross-entropy and reporting Rank-IC leaves an unnecessary gap between
    the objective and the claim.
    """
    if group_ids is None:
        group_ids = torch.zeros_like(scores, dtype=torch.long)

    total = scores.sum() * 0.0
    n_groups = 0
    for gid in torch.unique(group_ids):
        sel = group_ids == gid
        s, y = scores[sel], targets[sel]
        if s.numel() < 5:
            continue

        soft_rank = 1.0 + torch.sigmoid((s.unsqueeze(0) - s.unsqueeze(1)) / tau).sum(dim=0)
        true_rank = y.argsort().argsort().to(s.dtype)

        a = soft_rank - soft_rank.mean()
        b = true_rank - true_rank.mean()
        denom = (a.norm() * b.norm()).clamp(min=_EPS)
        total = total + (a @ b) / denom
        n_groups += 1

    if n_groups == 0:
        return scores.sum() * 0.0
    return -total / n_groups


@dataclass
class LossWeights:
    """Mirrors the ``loss`` section of the configuration."""

    lambda_reg: float = 1.0
    lambda_cal: float = 0.5
    lambda_rank: float = 0.2
    focal_enabled: bool = False
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    mmce_kernel_width: float = 0.4
    sigma_warmup_epochs: int = 3


class CompositeObjective(nn.Module):
    """The single training objective, assembled from the four parts above."""

    def __init__(self, weights: LossWeights | None = None, n_classes: int = 3) -> None:
        super().__init__()
        self.w = weights or LossWeights()
        self.n_classes = n_classes
        self.register_buffer("class_weights", torch.ones(n_classes), persistent=False)

    def set_class_weights_from(self, labels: torch.Tensor) -> torch.Tensor:
        """Compute inverse-frequency weights from the *training fold* only.

        Deriving them from the full sample would be a mild but real leak: the class
        balance of the test period is not knowable at training time.
        """
        w = inverse_frequency_weights(labels, self.n_classes).to(self.class_weights.device)
        self.class_weights = w
        return w

    def forward(
        self,
        logits: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        y_dir: torch.Tensor,
        y_ret: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return ``(total_loss, components)``; components are detached floats for logs."""
        if self.w.focal_enabled:
            l_dir = focal_loss(
                logits, y_dir, self.w.focal_gamma, self.class_weights, sample_weights
            )
        else:
            l_dir = weighted_cross_entropy(
                logits, y_dir, self.class_weights, sample_weights, self.w.label_smoothing
            )

        warmup = epoch < self.w.sigma_warmup_epochs
        l_nll = gaussian_nll(mu, logvar, y_ret, sample_weights, sigma_warmup=warmup)
        l_cal = mmce_loss(logits, y_dir, self.w.mmce_kernel_width)
        l_rank = (
            soft_spearman_loss(mu, y_ret, group_ids)
            if self.w.lambda_rank > 0
            else logits.sum() * 0.0
        )

        total = (
            l_dir
            + self.w.lambda_reg * l_nll
            + self.w.lambda_cal * l_cal
            + self.w.lambda_rank * l_rank
        )
        return total, {
            "loss": float(total.detach()),
            "l_dir": float(l_dir.detach()),
            "l_nll": float(l_nll.detach()),
            "l_mmce": float(l_cal.detach()),
            "l_rank": float(l_rank.detach()),
            "sigma_warmup": float(warmup),
        }
