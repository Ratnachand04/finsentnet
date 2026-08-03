"""Evaluation harness.

Built before the model on purpose (SPEC.md phase order): the harness defines which
numbers the paper contains, and therefore what the model must produce. Running
``report.py --signal random`` emits the complete table and figure set filled with noise,
so the manuscript skeleton exists before any result does.
"""

from finsent.eval import metrics, stats, dsr, attribution, backtest, report  # noqa: F401

__all__ = ["metrics", "stats", "dsr", "attribution", "backtest", "report"]
