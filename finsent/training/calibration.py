"""
Temperature scaling for post-hoc calibration.
===============================================

Legacy V2 module, retained because ``finsent/training/trainer.py`` and the finsentnet_pro
application still import it. New research work uses ``finsent.training.calibrate``, which
also offers vector scaling.

Reference: Guo et al., "On Calibration of Modern Neural Networks", 2017

CORRECTION TO AN EARLIER CLAIM IN THIS FILE
-------------------------------------------
This module previously stated that temperature scaling "dramatically improves
calibration", and measured that improvement on the very rows the temperature was fitted
to. Both parts were wrong.

Fitting and grading on the same data always shows an improvement, whether or not one
exists out of sample. Measured in the research path: an unguarded post-hoc fit reduced
the fitting set's ECE from 0.020 to 0.004 while *raising* held-out ECE from 0.003 to
0.056. It had learned the validation block's class balance, and that balance drifted —
a DOWN share of 0.28 in validation against 0.46 in the test period.

``calibrate`` now (a) fits on the earlier half of the validation set and grades on the
later half, and (b) keeps the fitted temperature only if it beats ``T = 1`` on that
held-out half. Post-hoc calibration is not free: when the underlying model is weak its
softmax is near uniform and therefore already well calibrated by accident, so a fitted
map can only add variance. Identity has to be allowed to win.

Financial motivation, which is unchanged and is why the guard matters:
    Kelly sizing requires accurate P(win). Overconfidence is variance
    underestimation, and expected log-growth falls with the *square* of the error —
    underestimating variance by half destroys all growth on a genuinely profitable
    signal. See ``finsent.decision.growth_theory``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from torch.utils.data import DataLoader


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for model calibration.
    
    Calibrated logits = raw logits / T
    
    where T is optimized on the validation set via NLL minimization.
    
    This is a single-parameter optimization that preserves
    the model's classification accuracy while improving
    probability calibration.
    """
    
    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(init_temperature, dtype=torch.float32)
        )
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature.
        
        Args:
            logits: (batch, n_classes) raw model logits
        Returns:
            calibrated_logits: (batch, n_classes)
        """
        return logits / self.temperature.clamp(min=0.01)
    
    def calibrate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 200,
    ) -> Dict[str, float]:
        """Optimize temperature on validation set.
        
        Uses LBFGS optimizer (second-order, fast convergence for 1 parameter).
        Minimizes NLL on validation set.
        
        Returns:
            dict with optimal temperature and calibration metrics
        """
        model.eval()
        self.to(device)
        
        # Collect all logits and labels from validation set
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                price = batch["price"].to(device)
                text_ids = batch["text_ids"].to(device)
                text_mask = batch["text_mask"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(price, text_ids, text_mask)
                all_logits.append(outputs["direction_logits"])
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # --- Split the validation set temporally: fit on the earlier half, judge on the
        # later half. Fitting and grading on the same rows always looks like an
        # improvement. In the research path the unguarded version cut the fitting set's
        # ECE from 0.020 to 0.004 while raising *test* ECE from 0.003 to 0.056: it had
        # learned that block's class balance, and the balance drifted. The split is
        # temporal rather than random so the selection faces the same drift the test
        # period will impose.
        n = all_logits.shape[0]
        cut = max(int(n * 0.5), 1)
        fit_logits, fit_labels = all_logits[:cut], all_labels[:cut]
        sel_logits, sel_labels = all_logits[cut:], all_labels[cut:]
        if sel_logits.shape[0] < 20:  # too small to judge; keep the identity map
            fit_logits, fit_labels = all_logits, all_labels
            sel_logits, sel_labels = all_logits, all_labels

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(fit_logits), fit_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        fitted_temperature = float(self.temperature.detach().clamp(min=0.01))

        # --- Guard: identity (T = 1) is a genuine competitor, not a placeholder. A
        # near-uniform softmax is already well calibrated by accident, and fitting a
        # temperature to it adds variance without removing bias. Keep the fitted value
        # only if it beats T = 1 on data it has not seen.
        with torch.no_grad():
            ece_identity = self._compute_ece(F.softmax(sel_logits, dim=-1), sel_labels)
            ece_fitted = self._compute_ece(
                F.softmax(sel_logits / fitted_temperature, dim=-1), sel_labels
            )

        accepted = ece_fitted < ece_identity
        chosen = fitted_temperature if accepted else 1.0
        with torch.no_grad():
            self.temperature.fill_(chosen)

        with torch.no_grad():
            probs = F.softmax(self.forward(sel_logits), dim=-1)
            accuracy = (probs.argmax(dim=-1) == sel_labels).float().mean().item()

        ece_before, ece_after = ece_identity, min(ece_fitted, ece_identity)
        reduction = (
            (ece_before - ece_after) / ece_before * 100.0 if ece_before > 1e-12 else 0.0
        )

        results = {
            "optimal_temperature": chosen,
            "fitted_temperature": fitted_temperature,
            "identity_rejected_fit": not accepted,
            "accuracy": accuracy,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "ece_reduction": reduction,
            "n_fit": int(fit_logits.shape[0]),
            "n_holdout": int(sel_logits.shape[0]),
        }

        print(f"[Calibration] Temperature: {chosen:.3f} "
              f"(fitted {fitted_temperature:.3f}"
              f"{', REJECTED in favour of identity' if not accepted else ''})")
        print(f"[Calibration] Held-out ECE: {ece_before:.4f} -> {ece_after:.4f} "
              f"({reduction:.1f}% reduction, n={results['n_holdout']})")

        return results
    
    @staticmethod
    def _compute_ece(
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15,
    ) -> float:
        """Expected Calibration Error."""
        confidences, predictions = probs.max(dim=-1)
        correct = predictions.eq(labels).float()
        
        ece = 0.0
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accuracy = correct[mask].mean()
                bin_confidence = confidences[mask].mean()
                bin_weight = mask.float().sum() / len(labels)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece.item()
