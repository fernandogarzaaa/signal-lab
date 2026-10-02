"""HARVIX: HAR+VIX rematch on realized_vol_5d (docs/HARVIX_PREREGISTRATION.md).

Challenger: ridge (alpha = 1.0, standardized) on the 31 VOL HAR-style
price features + 3 frozen VIX features. Primary baseline: the same
scale-corrected GARCH(1,1) as VOL (code reused from
``signal_lab.vol.garch``). Secondary baseline: ridge on the 31 HAR
features alone. Sanity arm: naive persistence.
"""
