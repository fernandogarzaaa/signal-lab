"""ATTRVOL: attribution structure predicts post-event volatility.

BET 2 (SPECULATIVE) from docs/SYNTHESIS_BETS.md, pre-registered in
docs/ATTRVOL_PREREGISTRATION.md. Frozen 5-feature attribution leg
(driver_H, idio_share, ret_x_H, analogue_vol_mean, analogue_vol_std)
on trailing top-decile |abn_ret| event days; challenger har_vix_attr
vs primary baseline har_vix (both frozen ridge), secondary baseline
garch11, sanity arm naive; QLIKE primary; 5-fold expanding purged
walk-forward; GO iff mean paired QLIKE differential > 0 with 95% CI
lower bound > 0.
"""

from signal_lab.attrvol import features  # noqa: F401
