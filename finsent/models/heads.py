"""Three output heads: direction, conditional mean, conditional log-variance.

The variance head is the single most consequential change from V2, for a reason that is
entirely about the decision layer rather than about prediction accuracy.

V2 emitted a "calibrated confidence" scalar from a sigmoid with a *learnable temperature*
inside it. That construction does nothing: dividing a linear layer's output by a learned
scalar before a sigmoid is a reparametrisation of the layer's own weight scale, so the
model can undo it at no cost. It was then combined with a *second*, post-hoc temperature
on the softmax logits, calibrating a different object, and the two were never reconciled.

What position sizing actually needs is a conditional variance. ``f = mu / sigma^2``
requires ``sigma^2``, and Proposition 2 in ``finsent.decision.growth_theory`` shows that
getting it wrong by a factor of two destroys the entire growth advantage of a correct
signal. So the model predicts it directly, trained by Gaussian negative log-likelihood.

Practical warning, encoded in the training loop rather than here: for the first few
epochs the log-variance must be detached, or the model discovers that it can minimise
NLL by inflating variance on hard samples and never learns the mean at all. This is the
most common failure mode of heteroscedastic heads and it is silent -- the loss goes down.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["HeadOutput", "PredictionHeads"]


@dataclass
class HeadOutput:
    """Raw head outputs. Probabilities are derived, never stored separately."""

    logits: torch.Tensor      # (B, 3) direction logits, order DOWN, NEUTRAL, UP
    mu: torch.Tensor          # (B,)   conditional mean forward return
    logvar: torch.Tensor      # (B,)   conditional log variance

    @property
    def probs(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    @property
    def sigma2(self) -> torch.Tensor:
        return torch.exp(self.logvar)

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(0.5 * self.logvar)

    def directional_score(self) -> torch.Tensor:
        """``P(up) - P(down)``.

        Never ``P(up)`` alone: that map assigns identical actions to ``(0.35, 0.60,
        0.05)`` and ``(0.35, 0.05, 0.60)``, which carry opposite risk. V2's five-level
        signal rule had exactly that defect.
        """
        p = self.probs
        return p[:, 2] - p[:, 0]

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "logits": self.logits,
            "probs": self.probs,
            "mu": self.mu,
            "logvar": self.logvar,
            "sigma2": self.sigma2,
        }


class PredictionHeads(nn.Module):
    """Shared trunk feeding a classifier and a heteroscedastic Gaussian regressor."""

    def __init__(
        self,
        d_model: int = 128,
        hidden: int = 128,
        n_classes: int = 3,
        dropout: float = 0.1,
        logvar_clamp: tuple[float, float] = (-10.0, 2.0),
        heteroscedastic: bool = True,
    ) -> None:
        super().__init__()
        if n_classes != 3:
            raise ValueError("SPEC.md freezes the label space at 3 classes")

        self.logvar_min, self.logvar_max = logvar_clamp
        self.heteroscedastic = heteroscedastic

        self.trunk = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout))
        self.direction = nn.Linear(hidden, n_classes)
        self.mean = nn.Linear(hidden, 1)
        self.logvar = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in (self.trunk[0], self.direction, self.mean, self.logvar):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        # Start slightly under-confident about variance: a small negative bias puts the
        # initial sigma near the typical scale of a daily return rather than near 1.0,
        # which would otherwise make the first epochs' Kelly fractions vanish.
        nn.init.constant_(self.logvar.bias, -6.0)

    def forward(self, fused: torch.Tensor) -> HeadOutput:
        z = self.trunk(fused)
        logvar = self.logvar(z).squeeze(-1).clamp(self.logvar_min, self.logvar_max)
        if not self.heteroscedastic:
            logvar = torch.zeros_like(logvar)
        return HeadOutput(
            logits=self.direction(z),
            mu=self.mean(z).squeeze(-1),
            logvar=logvar,
        )
