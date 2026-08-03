"""FinSentNet-C — calibration-aware cross-modal learning for equity direction.

The claim this package exists to test
-------------------------------------
Calibrated probability, not raw accuracy, determines whether a directional equity signal
survives position sizing. Cross-modal gating improves calibration more than it improves
accuracy, and calibration-aware growth-optimal sizing converts a small, statistically
fragile predictive edge into a materially better risk-adjusted outcome than the same
predictions sized by uncalibrated softmax confidence.

Layout
------
``finsent.config``    typed configuration; ``configs/base.yaml`` is the source of truth
``finsent.data``      point-in-time universe, timestamp alignment, labels, purged splits
``finsent.models``    price / text encoders, gated fusion, three-headed output
``finsent.training``  losses (weighted CE + Gaussian NLL + MMCE), calibration, conformal
``finsent.decision``  growth theory, sizing, costs, portfolio, regimes
``finsent.eval``      metrics, significance tests, deflated Sharpe, backtest, report

Everything the manuscript quotes is produced by ``finsent.eval.report``; see ``SPEC.md``
for the frozen constants and the eight contradictions this rebuild resolved.

Note: ``finsent.config``, ``finsent.data``, ``finsent.decision`` and ``finsent.eval``
depend only on numpy/pandas/scipy, so the full evaluation and decision stack runs
without PyTorch installed. Only ``finsent.models`` and ``finsent.training`` require it.
"""

__version__ = "0.3.0"
__all__ = ["config", "data", "models", "training", "decision", "eval", "utils"]
