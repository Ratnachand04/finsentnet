"""Decision layer: from calibrated probability to a position.

This package replaces the V2 signal path wholesale. Three defects are fixed here:

1. **Category error in Kelly.** V2 fed a terminal-horizon class probability into
   barrier-defined odds. The probability of hitting a target *before* a stop is a
   first-passage probability, a different object. ``sizing`` uses the continuous-outcome
   form ``f = mu / sigma^2`` instead, and ``labeling.triple_barrier_labels`` supplies
   genuine barrier outcomes whenever barrier odds are wanted.

2. **Double counting.** V2 multiplied the Kelly fraction by a separate confidence
   scalar even though the probability was already calibrated. That multiplier is gone.

3. **Discarding half the model.** V2's signal map read only ``P_up``, so
   ``(0.35, 0.60, 0.05)`` and ``(0.35, 0.05, 0.60)`` produced identical actions despite
   opposite risk. Scores now use ``mu_hat`` or ``P_up - P_down``.

``growth_theory`` carries the propositions that quantify what miscalibration costs; they
are the paper's theoretical core, not decoration.
"""

from finsent.decision import growth_theory, sizing, costs, portfolio, regime  # noqa: F401

__all__ = ["growth_theory", "sizing", "costs", "portfolio", "regime"]
