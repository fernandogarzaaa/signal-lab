"""TAILQ: tail-quantile (VaR) forecasting around earnings events.

Candidate C from docs/RESEARCH_SPRINT.md section 4. Frozen design in
docs/TAILQ_PREREGISTRATION.md: the 5% conditional quantile of the 3-day
event return, challenger = LightGBM quantile regression on frozen
point-in-time features vs Engle-Manganelli (2004) SAV CAViaR and
GARCH-filtered historical-simulation VaR. Primary metric: pinball loss
at tau = 0.05. GO rule: conjunctive (pinball win with 95% CI lower
bound > 0 AND challenger passes all coverage tests while CAViaR fails
at least one).
"""
