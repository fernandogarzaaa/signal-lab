"""DISPVOL: HAR+VIX+dispersion vs HAR+VIX on realized_vol_5d (docs/DISPVOL_PREREGISTRATION.md).

Challenger: ridge (alpha = 1.0, standardized) on the 34 frozen
HAR+VIX features plus the 6 frozen cross-sectional dispersion
features. Primary baseline: ridge on the 34 HAR+VIX features alone
(the HARVIX challenger definition, unchanged). Secondary baseline:
the same scale-corrected GARCH(1,1) as VOL/HARVIX (code reused from
``signal_lab.vol.garch``). Sanity arm: naive persistence.
"""
