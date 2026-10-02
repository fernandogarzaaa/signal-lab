# Experiment Registry

Every tested model, feature set, horizon, and threshold is logged here.
Results are labeled EXPLORATORY or CONFIRMATORY. The registry was created
for the gate-2 retest (Phase 5 of the original program was never built);
pre-registry experiments are reconstructed from merged PRs and committed
artifacts and marked as a lower-bound estimate.

## Pre-registry baseline (lower bound, reconstructed 2026-10-01)

| IDs | Experiment | Status |
|---|---|---|
| EXP-001..015 | Gate 1: 5 models x 3 variants (A/B/C), dev walk-forward | EXPLORATORY |
| EXP-016 | Gate 1: calibration diagnostic (Platt/isotonic) | EXPLORATORY |
| EXP-017 | Gate 1: abstention diagnostic | EXPLORATORY |
| EXP-018..020 | Walk-forward label schemes: weak / jev / jev-conf06 | EXPLORATORY |
| EXP-021..023 | FinBERT model experiments: logreg / HGB / logreg+PCA50 | EXPLORATORY |
| EXP-024..026 | FinBERT ablations (0.3.0/0.4.0): TF-IDF / FinBERT / combined | EXPLORATORY |

**Tested-experiment count at registry creation: 26.** Ad-hoc runs before
the research program that left no committed trace are not counted.

## Gate-2 run (all EXPLORATORY unless noted)

| ID | Date (UTC) | Experiment | Result pointer |
|---|---|---|---|
| EXP-027 | 2026-10-01 | Gate-2 retest: FinBERT primary arm, TF-IDF arm, widened dev data; pre-registered in docs/GATE_2_PREREGISTRATION.md | docs/GATE_2.md (pending) |
| EXP-028 | 2026-10-01 | STEP 2: leakage check on early-fold price signal (purge/embargo re-run, shuffled labels, injected future feature) | report section (pending) |
| EXP-029 | 2026-10-01 | STEP 0a: SPY-gap integrity check. Rebuilt the original gate-1 frame (909 pre-widening rows) with SPY truncated to >= 2024-10-15 (pre-backfill state) vs full backfilled SPY. Result: 0 differing cells across all 49 columns; the fallback market leg was used on 325 pre-2024-10-15 dates but none fed any label or feature. Gate-1 results unchanged. Script: scripts/gate2_step0_spy_check.py, scripts/gate2_step0_spy_fulldiff.py | this row |
| EXP-030 | 2026-10-01 | STEP 1: bulk news-source access verification (EXPLORATORY). GDELT bulk GKG verified: free, no auth, no documented rate limit, HTTPS, predictable 15-min slots, 27-col TSV, Extras carries PAGE_PRECISEPUBTIMESTAMP + PAGE_TITLE. Measured throughput 1.4-2.2 MB/s from this box -> pre-registered 3-year pull projects to 38-78h, infeasible in-session. BigQuery gdelt-bq.gdeltv2.gkg exists but needs GCP creds (unavailable). Hard stop triggered -> docs/NEGATIVE_RESULT.md fallback. | docs/NEGATIVE_RESULT.md |

**Tested-experiment count: 30** (26 baseline + 4 gate-2).

## VOL volatility campaign (CONFIRMATORY, pre-registered 2026-10-02)

Pre-registration: `docs/VOL_PREREGISTRATION.md` (frozen, binding; merged
as PR #51, commit e5569c6). Target: realized_vol_5d (sample std ddof=1
of daily log returns over (t0, t0+5]). Baseline: GARCH(1,1) MLE on
trailing 252 trading days, analytic 5-day term structure, naive fallback
on fit failure (counted). Arms: ridge (alpha=1.0, standardized) and
LightGBM (n_estimators=300, lr=0.05, leaves=31, min_child=100,
feat_frac=0.8, bag_frac=0.8, l2=1.0, seed=7) on HAR-style point-in-time
features + SPY legs. Metric: QLIKE on variance (primary); MSE-on-variance,
MAE-on-vol, Diebold-Mariano (secondary). Validation: 5-fold expanding
purged walk-forward, seed 7, min_train 50, purge j==k, 5-trading-day
embargo. GO iff mean paired QLIKE differential > 0 AND 95% t CI lower
bound > 0; NO-GO kills the campaign. Universe: S&P 100 (101, Wikipedia
components as of 2026-09-21) + SPY = 102 tickers, frozen in
`data/universe_vol.csv`. Prices: yfinance 2022-01-01..2026-07-15.

| ID | Date (UTC) | Item | Result pointer |
|---|---|---|---|
| VOL-BUILD | 2026-10-02 | Campaign build: `src/signal_lab/vol/` (universe, prices, frame, features, garch, models, metrics, run_campaign CLI); 7 synthetic test files; no network in CI | PR (pending) |
| VOL-SELECT | 2026-10-02 | Single purged train/validation split in dev (valid 2025-08-15..2026-06-30; 88,419 train / 22,253 valid rows; 595 purged): ridge validation QLIKE 0.608419 vs LightGBM 0.590065. Champion: **lightgbm**. DSR (per-row QLIKE gains over naive, freq=1; reported, not a gate): lightgbm Sharpe +0.147, DSR 1.0; ridge Sharpe +0.143, DSR 1.0; neither flagged likely_false_discovery | data/vol_selection.json (local) |
| VOL-EVAL | 2026-10-02 | 5-fold expanding purged walk-forward (seed 7, min_train 50, purge j==k, 5d embargo), champion lightgbm vs garch11 vs naive, paired per fold on QLIKE. Per-fold d = QLIKE(garch)-QLIKE(champ): +0.149621, -0.013392, -0.028122, +0.030955, -0.081545. mean(d)=+0.011503, se=0.038925, 95% t CI (df=4) [-0.096569, +0.119576]. CI lower bound < 0 -> **NO-GO**. DM stat +0.2955, p=0.7823. GARCH fallbacks 5,166/92,667 (5.57%). Deviation: GARCH forecast uses sqrt(mean) not literal sqrt(sum) of the term structure (pre-reg literal puts the baseline on 5-day cumulative scale, 2.24x the daily-scale target; the literal-scale first run's GO is VOID) | docs/VOL_NEGATIVE_RESULT.md; docs/vol_results.json |
| VOL-TEXT | 2026-10-02 | GDELT DOC probe 2026-10-02T11:05:20Z (single-day) returned 0 articles, no error; first quarterly query hit HTTP 429 on all 4 backoff attempts (RateLimitError, same hard throttle as gate-2). No evasion attempted; TF-IDF + Loughran-McDonald text arms SKIPPED, logged as deviation | this row; docs/VOL_NEGATIVE_RESULT.md |
| VOL-VERDICT | 2026-10-02 | **NO-GO. Campaign DEAD per the frozen kill criterion.** No test-period evaluation run (GO-gated). No confirmation data touched. Next targets (event magnitude, drawdown risk) need separate pre-registrations | docs/VOL_NEGATIVE_RESULT.md |

**Tested-experiment count: 33** (31 + VOL-SELECT + VOL-EVAL; VOL-TEXT skipped, VOL-VERDICT is the verdict).

## Extraction program (APPLIED product improvements, 2026-10-01)

The research program is CLOSED (see docs/PROGRAM_STATUS.md). The
items below are product improvements for the explanatory workbench,
extracted from the open-source survey. They are not gate experiments
and carry no predictive claims.

| ID | Date (UTC) | Item | Result pointer |
|---|---|---|---|
| APP-000 | 2026-10-01 | Program closure note | docs/PROGRAM_STATUS.md |
| APP-001 | 2026-10-01 | Split-conformal abstention (src/signal_lab/models/conformal.py): rolling-window calibration per fold, abstain on empty/non-singleton sets; replaces failed ad-hoc margins | tests/test_conformal.py (10 tests) |
| APP-002 | 2026-10-01 | CAR significance tests (src/signal_lab/stats/car_tests.py): BMP / adj-BMP (Kolari-Pynnonen) / GRANK wired into event_study; near-zero-variance estimation windows dropped, not zero-filled | tests/test_car_tests.py (12 tests) |
| APP-003 | 2026-10-02 | TabPFN zero-tuning benchmark arm (src/signal_lab/models/tabpfn_arm.py, tabpfn==2.0.6 last fully-local release): purged/embargoed walk-forward reusing the validation harness; per-fold PCA-50 fit on train only, dedupe-first, point-in-time conditioning asserted per fold. Dev result (4 folds, fold 1 skipped at 43 rows): PR-AUC 0.1184 +/- 0.0704 vs no-skill 0.1481, ROC-AUC 0.3517, F1 0.0; below tuned logreg (0.2079) and LightGBM (0.1925). Honest negative: zero tuning does not beat noise on this task; the arm stands as the zero-tuning floor for future models | tests/test_tabpfn_arm.py (6 tests); data/artifacts/tabpfn_arm.json |
| APP-004 | 2026-10-02 | Fail-loud baseline/event-window leakage assertion (src/signal_lab/stats/leakage_guard.py; pattern from jiamingpan/agent-quant-research, reimplemented): BaselineLeakageError (AssertionError subclass, -O-safe) raised when any baseline label is on/after the event-window start. Wired into event_ar_panel (per-event panel check), _scar_frame (re-checks any hand-built panel before the estimation/event split), and explain_move._price_block (trailing 60-day abnormal-z baseline must end before the event day) | tests/test_leakage_guard.py (12 tests: guard unit incl. boundary, hostile panel through cross_sectional_tests, clean-path integration, explainer call shapes) |
| APP-005 | 2026-10-02 | gdelt-client migration (src/signal_lab/ingest/gdelt_client.py, gdelt-client==0.2.2): hand-rolled GDELT DOC requests loop replaced by Filters/article_search; strict query-template parsing (unknown templates fail loudly), same day bounds (start-of-day..end-of-day) and max_records semantics, tenacity retries on 429/5xx with the old backoff posture (15s base, 120s cap). Code health only, not a throttle fix; checkpointing (fetch_log), politeness sleeps, and explicit gap logging unchanged. Documented deviation: no sort=datedesc (client exposes no sort param; ordering unused downstream) | tests/test_gdelt_client.py (10 tests: template-to-Filters mapping, loud rejection of unknown queries, row mapping, empty/URL-less rows, snippet fn, RateLimitError propagation, backoff config, fetch_log error bookkeeping) |
| APP-006 | 2026-10-02 | purgedcv cross-check + Deflated Sharpe Ratio (reference: eslazarev/purged-cross-validation v0.1.10, commit aee1215; reference only, not a dependency; our harness stays canonical). Cross-check findings: (1) inverted label windows (t1 < t0) now rejected fail-loud in make_splits/single_purged_split/purge_against (gap vs purgedcv validate_times); (2) purge boundary documented as conservative: t1 == test_start is purged, stricter than purgedcv's half-open reference which keeps it; (3) embargo zones running past the trading-calendar end are skipped, not crashed. Added 5 edge-case tests incl. a randomized conservativeness check (no kept row is one the half-open reference would purge). DSR (src/signal_lab/stats/deflated_sharpe.py, Bailey & Lopez de Prado 2014): sharpe_ratio / expected_sharpe_null / deflated_sharpe_ratio / dsr_report (Sharpe, DSR, likely_false_discovery flag); reported, not a gate | tests/test_validation_splits.py (+5 cross-check tests); tests/test_deflated_sharpe.py (10 tests: stdlib-verified formula, zero-variance SR_0, DSR bounds/monotonicity in N, report sorting/flags, loud errors) |

## EVENTVOL campaign (CONFIRMATORY, pre-registered 2026-10-02)

Pre-registration: `docs/EVENTVOL_PREREGISTRATION.md` (frozen, binding;
merged as PR #56, commit e11daf2). Target: realized_vol_5d (sample std
ddof=1 of daily log returns over (t0, t0+5], decimal, daily) on
earnings-event rows. Primary baseline: iv_proxy = beta(i,t0) x
VIX(t0)/100/sqrt(252) (the market's forward-looking forecast; per-stock
historical IV is not obtainable from yfinance, reformulated honestly as
a deliberate design choice). Secondary baselines: garch11
(scale-corrected 5-day term structure), har_vix (ridge alpha=1.0 on
HAR+VIX features). Challenger: LightGBM (VOL frozen hyperparams) on
HAR+VIX+event features (Ederington-Lee cycle proxies). Metric: QLIKE
primary; Diebold-Mariano secondary. Validation: 5-fold expanding purged
walk-forward, seed 7, min_train 50, 5-trading-day embargo. GO iff mean
paired QLIKE differential (iv_proxy minus challenger) > 0 AND 95% t CI
lower bound > 0 AND challenger survives the 95% MCS (Hansen-Lunde-Nason
T_R, day-block stationary bootstrap, B=5000). Universe: frozen
data/universe_vol.csv. Prices: yfinance 2022-01-01..2026-07-15; VIX:
^VIX/^VIX3M same window; earnings: yfinance earnings calendar
(actuals only). Dev t0 <= 2026-06-30; test 2026-07-01..2026-08-31
(GO-gated); confirmation never touched.

| ID | Date (UTC) | Item | Result pointer |
|---|---|---|---|
| EVENTVOL-BUILD | 2026-10-02 | Campaign build: `src/signal_lab/eventvol/` (earnings, vix, features, iv_proxy, mcs, run_campaign CLI); 5 synthetic test files (31 tests: event frame, point-in-time truncation, iv_proxy math/fallbacks, MCS elimination/determinism, splits invariants); no network in CI | PR (pending) |
| EVENTVOL-EVAL-1 | 2026-10-02 | Dev evaluation run 1 (VOID): literal pre-reg iv_proxy (beta x VIX/100/sqrt(252), 1e-6 floor). Non-positive trailing betas produced degenerate near-zero forecasts (per-row QLIKE up to ~8.7e8), crippling the primary baseline by construction. Run VOID, discarded; deviation logged | this row; docs/EVENTVOL_NEGATIVE_RESULT.md |
| EVENTVOL-EVAL-2 | 2026-10-02 | Dev evaluation run 2 (VOID): negative-beta fallback only. Low-beta stocks (utilities/staples, beta ~0.0-0.05) still underforecast ~10x (iv_proxy fold QLIKE 140-597 vs challenger 0.65-2.95): the formula prices only the systematic leg. Strawman baseline; run VOID, discarded; deviation logged | this row; docs/EVENTVOL_NEGATIVE_RESULT.md |
| EVENTVOL-EVAL-3 | 2026-10-02 | Dev evaluation run 3 (VALID): iv_proxy = sqrt((beta x VIX_d)^2 + idio_var), market-model variance decomposition, uninformative regressions fall back to raw VIX (counted). 5-fold expanding purged walk-forward on 1,714 dev event rows (160 NaN-feature rows dropped; 51 GARCH fallbacks / 1,421; 30 iv_proxy beta fallbacks). Per-fold d = QLIKE(iv)-QLIKE(challenger): -0.113382, +0.403059, -1.729026, +0.040066, +0.155992. mean(d) = -0.248658, se = 0.379581, 95% t CI (df=4) [-1.302543, +0.805226]. DM stat -0.6551, p = 0.5482. 95% MCS (T_R, day-block stationary bootstrap B=5000): survivors {challenger, iv_proxy}; har_vix, naive, garch11 eliminated. Primary conjunct fails -> **NO-GO**. iv_proxy beats garch11 on all 5 folds, consistent with Blair-Poon-Taylor (2001) | docs/EVENTVOL_NEGATIVE_RESULT.md; docs/eventvol_results.json |
| EVENTVOL-VERDICT | 2026-10-02 | **NO-GO. Campaign DEAD per the frozen kill criterion.** No test-period evaluation run (GO-gated; 0 labelable test rows in any case). No confirmation data touched | docs/EVENTVOL_NEGATIVE_RESULT.md |

**Tested-experiment count: 36** (33 + EVENTVOL-EVAL-3; EVENTVOL-EVAL-1/2 voided, EVENTVOL-VERDICT is the verdict).

## HARVIX campaign (CONFIRMATORY, pre-registered 2026-10-02)

Pre-registration: `docs/HARVIX_PREREGISTRATION.md` (frozen, binding;
merged as PR #61, commit 8d5b848). Candidate B from
docs/RESEARCH_SPRINT.md section 4: HAR+VIX rematch on the frozen
realized_vol_5d target, all ticker-days. Challenger: har_vix = ridge
(alpha=1.0, standardized; frozen, no selection) on the 31 VOL HAR
features + 3 frozen VIX features (vix_level, vix_5d_change, vix_slope;
^VIX/^VIX3M downloaded from yfinance at build, 1,136 rows each).
Primary baseline: garch11 (same scale-corrected implementation as VOL).
Secondary baseline: har = ridge on the 31 HAR features alone. Metric:
QLIKE primary with HAC-equivalent SEs, DM test, 95% MCS (reported, not
a gate); MSE secondary. Validation: 5-fold expanding purged
walk-forward, seed 7, min_train 50, purge j==k, 5-trading-day embargo.
GO iff mean paired QLIKE differential (garch minus har_vix) > 0 AND 95%
t CI lower bound > 0; NO-GO kills the campaign. Universe: frozen
data/universe_vol.csv (102 tickers). Dev t0 <= 2026-06-30; test
2026-07-01..2026-08-31 (GO-gated); confirmation never touched.

| ID | Date (UTC) | Item | Result pointer |
|---|---|---|---|
| HARVIX-BUILD | 2026-10-02 | Campaign build: `src/signal_lab/harvix/` (frozen VIX feature leg, run_campaign CLI reusing the VOL harness: frame, features, garch, metrics, splits, periods; EVENTVOL MCS); 20 synthetic tests (VIX math/point-in-time/burn-in, ridge arms, end-to-end walk-forward smoke, confirmation enforcement); no network in CI | PR #62 |
| HARVIX-EVAL | 2026-10-02 | Dev evaluation (VALID): 5-fold expanding purged walk-forward on 111,267 dev rows (2,244 NaN-feature rows dropped; 5,166 GARCH fallbacks / 92,667 test rows). GARCH panel memoized from VOL's cached panel: HARVIX's dev test-row set is exactly VOL's 92,667 (ticker, t0) rows, all joined values verified identical. Per-fold d = QLIKE(garch)-QLIKE(har_vix): +0.125595, -0.139821, -0.054134, -4.097733, -0.095399. mean(d) = -0.852298, se = 0.812610, 95% t CI (df=4) [-3.108466, +1.403869]. DM stat -1.0488, p = 0.3534. 95% MCS: survivors {har_vix, garch11, har}; naive eliminated (p=0.0002). Secondary har_vix vs har: mean(d2) = -0.840780, CI [-3.096260, +1.414701], DM p = 0.3591. DSR: har_vix likely_false_discovery = True. Fold-4 blowup (har_vix QLIKE 4.88): 2025-04-09 VIX +12.11 spike, ridge extrapolated to floored 1e-4 forecasts on defensive names; 8 rows carry ~4.15 of the 4.88 fold mean (genuine model pathology, same as EVENTVOL's har_vix note). Ex-fold-4 mean(d) = -0.040940, still negative -> **NO-GO** | docs/HARVIX_NEGATIVE_RESULT.md; docs/harvix_results.json |
| HARVIX-VERDICT | 2026-10-02 | **NO-GO. Campaign DEAD per the frozen kill criterion.** No test-period evaluation run (GO-gated). No confirmation data touched | docs/HARVIX_NEGATIVE_RESULT.md |

**Tested-experiment count: 37** (36 + HARVIX-EVAL; HARVIX-BUILD is infra, HARVIX-VERDICT is the verdict).

## TAILQ campaign (CONFIRMATORY, pre-registered 2026-10-02)

Pre-registration: `docs/TAILQ_PREREGISTRATION.md` (frozen, binding;
merged as PR #64, commit 7226bf9). Candidate C from
docs/RESEARCH_SPRINT.md section 4: 5% conditional quantile of the
3-day earnings event return. Event set: earnings announcements
(yfinance get_earnings_dates, actuals only, t0 = earnings date mapped
to next trading day; trailing top-decile |abn_ret| days rejected in
the pre-reg because selecting on realized magnitude complicates
coverage). Challenger: lgbm_quantile = LightGBM quantile regression
(alpha=0.05, n_estimators=300, lr=0.05, leaves=15, min_child=40,
feat_frac=0.8, bag_frac=0.8, l2=1.0, seed=7, deterministic) on 15
frozen point-in-time features (HAR legs, pre-event returns, VIX
level/change/slope, beta_252, surprise, prev_surprise, dow, SPY legs;
frozen, no selection). Primary baseline: caviar = Engle-Manganelli
(2004) symmetric absolute value CAViaR f_t = b1 + b2*f_{t-1} +
b3*|r_{t-1}|, fit per (ticker, t0) by check-function minimization
(differential_evolution seed 7 + Nelder-Mead polish) on trailing 252
trading days, 1-day forecast scaled by sqrt(3) (frozen horizon
adaptation). Secondary baseline: garch_hs = GARCH(1,1)-filtered
historical simulation VaR (same sqrt(3) scaling); naive = trailing
empirical 5% quantile x sqrt(3) (sanity). Metric: pinball loss at
tau=0.05 PRIMARY; Kupiec unconditional coverage, Christoffersen
independence + conditional coverage; 95% MCS (reported, not a gate);
DSR honesty metric (reported, not a gate). Validation: 5-fold
expanding purged walk-forward on t0, seed 7, min_train 50,
horizon=3/embargo=3 trading days. GO iff mean paired pinball
differential (caviar minus challenger) > 0 AND 95% t CI lower bound
> 0 AND challenger passes all three coverage tests while CAViaR fails
at least one; miss on pinball kills the campaign. Universe: frozen
data/universe_vol.csv (102 tickers). Dev t0 <= 2026-06-30; test
2026-07-01..2026-08-31 (GO-gated); confirmation never touched.

| ID | Date (UTC) | Item | Result pointer |
|---|---|---|---|
| TAILQ-BUILD | 2026-10-02 | Campaign build: `src/signal_lab/tailq/` (event frame with 3-day event-return target reusing the EVENTVOL earnings pipeline; 15 frozen features; SAV CAViaR with DE + Nelder-Mead check-function fit; GARCH-filtered HS VaR reusing vol/garch.py; LightGBM quantile challenger with frozen hyperparams; pinball/Kupiec/Christoffersen metrics; walk-forward runner reusing splits/periods/MCS/universe/prices/VIX); 40 synthetic tests (pinball math, CAViaR recovery of known SAV params on synthetic data, point-in-time safety, no-confirmation enforcement, end-to-end walk-forward smoke); no network in CI | PR #65 |
| TAILQ-EVAL | 2026-10-02 | Dev evaluation (VALID): 5-fold expanding purged walk-forward on 1,675 dev event rows (199 dropped for insufficient history; 20 CAViaR + 26 GARCH-HS naive fallbacks / 1,396 test rows). Per-fold d = pinball(caviar)-pinball(challenger): -0.000372, -0.000410, +0.001685, +0.000640, -0.000290. mean(d) = +0.000250, se = 0.000408, 95% t CI (df=4) [-0.000882, +0.001382]. DM stat +0.6138, p = 0.5726. Coverage (pooled 1,396 rows): challenger FAILS all three (Kupiec 192.76 p=0.0000; independence 7.99 p=0.0047; conditional 200.75 p=0.0000; violation rate 14.90% vs 5% expected); CAViaR FAILS all three (147.01 p=0.0000; 16.03 p=0.0001; 163.04 p=0.0000; 13.47%). All four arms over-violate 2.6-3x: earnings event tails dwarf daily-calibrated forecasts. 95% MCS: survivors {lgbm_quantile, caviar, naive}; garch_hs eliminated (p=0.0432). DSR: all three arms likely_false_discovery = True. Deterministic re-run reproduced every fold number exactly -> **NO-GO** | docs/TAILQ_NEGATIVE_RESULT.md; docs/tailq_results.json |
| TAILQ-VERDICT | 2026-10-02 | **NO-GO. Campaign DEAD per the frozen kill criterion (miss on pinball).** No test-period evaluation run (GO-gated; 1 labelable test row in any case). No confirmation data touched | docs/TAILQ_NEGATIVE_RESULT.md |

**Tested-experiment count: 38** (37 + TAILQ-EVAL; TAILQ-BUILD is infra, TAILQ-VERDICT is the verdict).
