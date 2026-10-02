"""CONFMAG: conformal-abstention-gated event-magnitude prediction.

BET 3 (SPECULATIVE) from docs/SYNTHESIS_BETS.md, pre-registered in
docs/CONFMAG_PREREGISTRATION.md, the last campaign in the program.
Frozen design: |excess_3d| target (absolute 3-day abnormal return vs
SPY); LightGBM L2 point forecast on the 34 frozen HAR+VIX features;
Conformalized Quantile Regression (LightGBM tau=0.1/0.9, alpha=0.2,
temporal 20% calibration split per fold); selection = test rows with
interval width <= 25th percentile of calibration widths; primary
baseline = per-fold train mean (constant); secondary = GARCH-implied
E|excess|; MSE primary on the selected subset; conjunctive GO rule.
"""

from signal_lab.confmag import cqr, features  # noqa: F401
