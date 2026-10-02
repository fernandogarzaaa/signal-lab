# Research Sprint: what the evidence says about market prediction

**Status:** RESEARCH MEMO. No experiments were run for this document and
none are authorized by it. It exists to give the next campaign a
literature-backed thesis instead of a blind test. Any campaign that
follows needs its own pre-registration before a single row of data is
touched.

**Why this exists.** Two campaigns failed honestly. Direction prediction
from news sentiment died at gate 1 (PR-AUC diff +0.0035, p=0.48;
docs/NEGATIVE_RESULT.md). Volatility prediction died against GARCH(1,1)
on QLIKE (mean paired diff +0.0115, 95% CI straddling zero;
docs/VOL_NEGATIVE_RESULT.md). The user's diagnosis: we were going in
blind and testing. This memo reports what published research and
serious open-source projects say actually works, what the honest
baselines are, and which targets are worth a pre-registered shot next.

**How to read the citations.** [peer-reviewed] marks journal-published
findings. Working papers, preprints, theses, and repo claims are flagged
as such. A finding is marked REPLICATED only when an independent
reproduction exists; otherwise it is marked as reported once. Fragile
findings are flagged explicitly.

## 1. Volatility forecasting: the literature

### 1.1 GARCH(1,1) is a genuinely hard null on daily data

Hansen and Lunde (2005, Journal of Econometrics) compared 330
daily-return volatility specifications on DM/$ data and could not reject
that none beats GARCH(1,1)
[peer-reviewed](http://www.bauer.uh.edu/rsusmel/phd/hansenlunde_garch.pdf).
Our loss to GARCH on daily data is therefore not anomalous; it is the
expected outcome of a fair fight.

### 1.2 HAR's edge comes from intraday data, which we do not have

Corsi's HAR-RV (2009, Journal of Financial Econometrics) models realized
volatility as daily/weekly/monthly components and became the default
benchmark
[peer-reviewed](https://econpapers.repec.org/article/oupjfinec/v_3a7_3ay_3a2009_3ai_3a2009_3ai_3a2_3ap_3a174-196.htm).
The literature's HAR-beats-GARCH results use precise intraday realized
volatility as input (Andersen, Bollerslev, Diebold, Labys 2003,
Econometrica). Our campaign fed HAR-style features built from daily
data with a noisy 5-day sample-std target, which deprives HAR of its
main advantage; on multi-day horizons the persistence differences
between GARCH and HAR compress. Losing that shootout is within the
literature's expected outcomes.

### 1.3 The one HAR extension with verified out-of-sample gains

Bollerslev, Patton, and Quaedvlieg (2016, Journal of Econometrics),
"HARQ": interacts the daily lag with measurement-error correction from
realized quarticity. S&P 500 futures plus 27 DJIA stocks, 1997-2013,
true out-of-sample. Daily MSE ratios vs HAR mostly below 1.0 (KO 0.89,
IBM 0.92, CVX 0.88); weekly QLIKE ratios below 1.0 (S&P 500: 0.85
rolling, 0.75 expanding). Magnitude: roughly 5-15% loss reductions, not
a step change
[peer-reviewed](https://cris.maastrichtuniversity.nl/ws/files/12579554/1600566.pdf).
Caveat: HARQ needs intraday data to compute realized quarticity. With
daily data only, it is not implementable. A clean independent
replication was not found.

Patton and Sheppard (2015, Review of Economics and Statistics), "Good
Volatility, Bad Volatility": decomposing RV into positive/negative
semivariances raises Mincer-Zarnowitz R2 by 10-20% over RV-only HAR,
driven by negative semivariance
[peer-reviewed](https://ideas.Repec.org/a/tpr/restat/v97y2015i2p683-697.html).
Note these are R2 figures, not loss ratios; economic magnitude is
modest.

Warning: a horse race over eleven HAR-type models on crude oil futures
found in-sample gains from jump and semivariance terms evaporated
out-of-sample; simple HAR was hard to beat
[peer-reviewed](https://www.sciencedirect.com/science/article/abs/pii/S037722171400040X).
Jump-only HAR-J is frequently worse than plain HAR out-of-sample.

### 1.4 Implied volatility subsumes history-based models

Blair, Poon, and Taylor (2001, Journal of Econometrics): on S&P 100
volatility, VIX gives the most accurate out-of-sample forecasts at all
horizons (1-20 days) across all performance measures, and intraday
information adds nothing significant once VIX is in the model
[peer-reviewed](https://roycheng.cn/files/papers/paper_Blair%26Poon%26Taylor_2001.pdf).
Pong, Shackleton, Taylor, and Xu (2004): combining IV-based and
model-based forecasts beats either alone. Busch, Christensen, and
Nielsen (2011): IV has incremental predictive power inside HAR models
across FX, stock, and bond markets.

Implication: the right benchmark for any volatility revisit is
**HAR augmented with VIX** (HAR-X/HAR-IV), compared against GARCH(1,1)
under QLIKE. For individual S&P 100 stocks there is no stock-level IV
analog, but VIX works as a common-factor feature. This is the arm the
literature says we should have tried first.

### 1.5 ML vs HAR/GARCH: small, fragile, but real in one case

Christensen, Siggaard, and Veliyev (2023, Journal of Financial
Econometrics): lasso, trees, random forests, and neural nets vs HAR
specifications on DJIA constituents, minimal tuning by design,
horizons 1/5/22 days. ML beats the HAR lineage even on the same three
RV lags, with gains larger at longer horizons
[peer-reviewed](https://arxiv.org/pdf/2601.13014v1). Flags: single
universe, MSE-centered evaluation, no peer-reviewed independent
replication found. Documented gains are single-digit percent.

Null results: an independent practitioner replication (XGBoost vs
GARCH/HAR on index RV, QLIKE + Diebold-Mariano + Hansen SPA) finds
GARCH(1,1) and HAR-RV "hard to beat; XGBoost wins only marginally, if
at all" [repo experiment](https://github.com/fatihhekim0glu/volforecast).
Red flags in the ML-volatility blog literature: single-index studies
on MSE only (MSE is dominated by crisis episodes), hyperparameter
tuning on the test set, and uncorrected multiple pairwise DM tests.

### 1.6 QLIKE was the right metric

Patton (2011, Journal of Econometrics): of nine common loss functions,
only MSE and QLIKE preserve the correct model ranking under a noisy
volatility proxy; QLIKE gives the greatest power in Diebold-Mariano
tests because it is less dominated by extreme episodes
[peer-reviewed](https://public.econ.duke.edu/~ap172/Patton_robust_forecast_eval_11dec08.pdf).
Our non-significant paired QLIKE difference is a genuine null, not a
metric artifact. Any retest should add HAC standard errors and a Model
Confidence Set rather than a raw mean.

## 2. Returns, sentiment, events, and tail risk: the literature

### 2.1 Cross-sectional ML: real R2, fragile economics

Gu, Kelly, and Xiu (2020, Review of Financial Studies): ~30,000 US
stocks, 1957-2016, monthly. Out-of-sample monthly stock-level R2 of
0.33-0.40%, neural nets slightly ahead; OLS has negative OOS R2.
[peer-reviewed](https://doi.org/10.1093/rfs/hhaa009). A student
replication reproduced the model ranking with lower absolute numbers
[student work](https://github.com/sheksaab/econ5130-ml-asset-pricing).

The economics are fragile: Avramov, Cheng, and Metzker (2023,
Management Science) show deep-learning signals extract profitability
from hard-to-arbitrage stocks; excluding microcaps or distressed
stocks considerably attenuates profitability, and trading costs erode
it further
[peer-reviewed](https://econpapers.repec.org/article/inmormnsc/v_3a69_3ay_3a2023_3ai_3a5_3ap_3a2587-2619.htm).
Dropping stocks below the 20th NYSE size percentile cuts the neural
net's six-factor alpha by 66%; value-weighting cuts performance 48%.
Honest reading: GKX proves nonlinear interactions exist, not that a
deployable monthly return strategy exists.

### 2.2 News sentiment alphas decay; our gate-1 result is the consensus

Tetlock (2007, Journal of Finance): WSJ pessimism predicted Dow
pressure with reversal, concentrated 1992-1999, dissipating within a
week [working draft](http://www.columbia.edu/~pt2238/papers/Tetlock_Media_Sentiment_JF.pdf).
Heston and Sinha (2016/2017): daily news predicts returns for 1-2 days
only [Fed working paper](https://www.federalreserve.gov/econres/feds/news-versus-sentiment-predicting-stock-returns-from-news-stories.htm).
Ke, Kelly, and Xiu (2024): Dow Jones Newswires sentiment is fully
incorporated within ~4 trading days, tradable net of costs only with
fast turnover [working paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3389884).
Lachana and Schroder: the traditional-media sentiment/return link has
decayed through time; after 1990 there is little relation.

Decay mechanisms: McLean and Pontiff (2016, Journal of Finance) show
across 97 published predictors that returns are 26% lower
out-of-sample and 58% lower post-publication
[peer-reviewed](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12365).
Information diffusion compressed the reaction window from a week
(Tetlock 2007) to 1-2 days. Our gate-1 finding (no 3-day direction
edge from sentiment) sits exactly where the literature says edge has
decayed to zero.

### 2.3 Event magnitude: thin literature, one robust neighbor

Predicting the SIZE of abnormal returns is much thinner than direction
work. Documented pieces: post-earnings-announcement drift is a
direction effect, not magnitude; event-induced implied volatility
follows a predictable creation/resolution cycle around scheduled
announcements (Ederington and Lee 1996, JFQA). Frazzini (2006, Journal
of Finance) links post-event drift magnitude to investors' unrealized
capital gains (disposition effect), the closest published treatment of
drift magnitude [peer-reviewed]. Claims of cross-sectionally
predicting abnormal-return size are mostly preprints; treat as
ungrounded until a baseline is set. The honest baselines for any
magnitude target: a constant forecast, pre-event implied volatility,
and GARCH.

### 2.4 Tail risk: variance is predictable, drawdowns are not (credibly)

CAViaR (Engle and Manganelli 2004, JBES) is the canonical conditional
quantile model for Value at Risk [peer-reviewed]. Volatility-managed
portfolios (Moreira and Muir 2017, Journal of Finance) show variance
forecasts raising Sharpe ratios in-sample of the spanning regression,
but Cederburg et al. (2020, Journal of Financial Economics) show
reasonable real-time implementations generally fail to beat unmanaged
portfolios [peer-reviewed]. Direct drawdown prediction has no credible
peer-reviewed track record; the best public numbers (AUC 0.99 next-day
decline classifiers, Sharpe 2.5 drawdown timers) are preprints or
low-tier journals with implausible magnitudes. Flag as unreliable.

### 2.5 The skeptical baseline

Harvey, Liu, and Zhu (2016, Review of Financial Studies): with
hundreds of tested predictors, the significance hurdle should be
t > 3.0, not 2.0; 27-53% of published factors are likely false
discoveries [peer-reviewed](https://arxiv.org/pdf/2206.15365).
Hou, Xue, and Zhang (2020, Review of Financial Studies): replicating
447 anomalies with NYSE breakpoints and value weighting, 64% are
insignificant at 5%; at t > 3.0 the failure count reaches 85%
[working paper](https://papers.ssrn.com/sol3/Papers.cfm?abstract_id=2961979).
Backtest-honesty machinery: Lopez de Prado's Probability of Backtest
Overfitting and Bailey-Lopez de Prado's Deflated Sharpe Ratio (Journal
of Portfolio Management 2014) formalize the multiple-testing haircut;
practitioners undercount trials 5-10x.

Any campaign target must therefore clear: value-weighted large-cap
evaluation, t > 3.0 or DSR > 0.95 with a full trial count, net-of-cost
reporting, and a frozen confirmation period.

## 3. Open-source reality check

Inspected read-only (clones where possible). Verdicts use REAL =
tests plus CI plus honest evaluation, not "would make money".

| Project | URL | Verdict | Evidence |
|---|---|---|---|
| arch (Sheppard) | https://github.com/bashtage/arch | REAL | 38 test files, CI, codecov, PyPI/conda, Zenodo DOI; academic-grade GARCH/ARCH, bootstrap inference, Model Confidence Set machinery |
| ruptures | https://github.com/deepcharles/ruptures | REAL | 8 test files, full CI (tests, docs, PyPI publish), CHANGELOG; standard change-point detection library |
| FinRL | https://github.com/AI4Finance-Foundation/FinRL | REAL | 3,277 commits, unit tests for envs/downloaders, train-test-trade pipeline; FAQ openly admits "single stock performance is not very good" |
| freqtrade | https://github.com/freqtradeorg/freqtrade | REAL | Live-trading platform: backtesting, dry-run, hyperopt, FreqAI; years of maintenance (clone blocked, not test-counted) |
| backtrader | https://github.com/mementum/backtrader | REAL, archived | 83 test files; no CI config; classic event-driven backtester, superseded by maintained forks |
| FinGPT | https://github.com/AI4Finance-Foundation/FinGPT | REAL research, hype-adjacent product | 6 test files, reproducible sentiment benchmarks, "not financial advice" disclaimer; robo-advisor framing is demo-grade |
| vnpy | https://github.com/vnpy/vnpy | REAL, thin tests | Only 2 test files, CI mostly lint; real framework with institutional users, testing gap |
| huseinzol05/Stock-Prediction-Models | https://github.com/huseinzol05/Stock-Prediction-Models | HYPE | Notebooks only, archived, TF 1.x era, no tests; independent audit found no point-in-time safety, no purged walk-forward; stars measure tutorial popularity, not validity |

### 3.1 Financial NLP models: classifiers, not alpha sources

ProsusAI/finbert (https://huggingface.co/ProsusAI/finbert): BERT
further trained on financial text, fine-tuned on Financial PhraseBank,
three-class sentiment. The card makes no trading claims. Independent
evals: 0.890 accuracy / 0.882 macro-F1 on all 4,846 PhraseBank
sentences (50-agree split), reproducible; but on 36 social-investor
examples FinBERT hit 69.4% vs a Twitter-RoBERTa baseline at 88.9%,
which is domain mismatch, not general inferiority. Two traps: the
native label order is {0: positive, 1: negative, 2: neutral}, which has
silently misaligned labels in multiple projects; and PhraseBank
"evaluations" are partially in-distribution since FinBERT pretrained
on PhraseBank-adjacent text. NOSIBLE/financial-sentiment-v1.1-base
claims to beat FinBERT on PhraseBank as held-out eval, but no
independent reproduction was found: treat as unproven. FinGPT
sentiment v3.3 (Llama2-13B LoRA) reports weighted F1 of 0.882/0.874/
0.903 on FPB/FiQA-SA/TFNS vs FinBERT 0.880/0.596/0.733, reproducible
via their benchmark notebook: the strongest apples-to-apples
comparison found, still sentiment classification, not trading P&L.

Bottom line: FinBERT-family numbers that survive scrutiny are
classification metrics on curated text (high-80s F1 in-domain). None
document a sentiment-to-return edge. Our gate-1 failure (decent
sentiment scores, no tradable signal) is exactly what the literature
predicts: these are text classifiers, not alpha sources.

### 3.2 Time-series foundation models: marketing exceeds evidence

Chronos (Amazon, https://github.com/amazon-science/chronos-forecasting):
well-engineered repo with tests and CI; paper claims strong zero-shot
forecasting on general benchmarks; independent benchmarks are mixed
(wins on heterogeneous M4 daily, loses to tuned classical ensembles on
synthetic series). No finance benchmarks in the paper. TimesFM
(Google): general forecasting paper only; finance applications found
are honest about limits (baseline TimesFM fails at price prediction;
finetuned TimesFM beats econometrics on VaR for S&P 100 constituents
but zero-shot is "clearly worse"); an independent reimplementation
states it "fails on random-walk data, about 31% worse than repeating
the last value" on FX. Lag-Llama: paper claims strong zero-shot, its
own README hedges toward finetuning, and an independent benchmark
found zero-shot losing to TFT and DeepAR. Moirai: no documented
finance application found. TimeGPT (Nixtla): closed model, vendor
benchmark ranks it first (treat as vendor material); an independent
finance test found classical multiple linear regression beating it on
stock values.

Verdict: finetuned TSFMs are plausible for volatility/VaR-style
tasks, not for raw price direction. Zero-shot finance claims are
marketing. Finance is near-random-walk; general benchmarks do not
transfer by assertion.

### 3.3 Borrowable engineering patterns

1. arch's inference discipline: every estimator ships with standard
   errors and bootstrap-based inference; no metric without a
   confidence interval, no arm comparison without a
   multiple-testing correction. Signal Lab should encode this as
   library code with tests, the way arch does.
2. FinRL/freqtrade's train-test-trade staging: transaction costs
   built into the environment, backtest then dry-run then live as
   mandatory separately-logged stages with their own kill criteria;
   and FinRL's FAQ culture of publishing negative results as
   first-class outputs.
3. The fixed versioned benchmark arena (Nixtla arena pattern):
   fixed universe, fixed metrics (QLIKE, pinball loss, coverage,
   DSR), fixed embargo, every future model scored on the same
   harness with naive baselines always present. This turns "we do
   not stop until we have a market prediction project" into a
   falsifiable program instead of an endless one.

Operational warning: one GDELT+FinBERT repo
(https://github.com/danielryang/macro_sentiment_trading) flags that
the free GDELT API is deprecated in favor of BigQuery. Our DOC API
throttle experience is consistent with this; treat GDELT as
unreliable infrastructure for future campaigns.

## 4. Candidate targets for the next campaign

All three need only yfinance (prices, VIX, earnings calendar). None
needs GDELT. Each sketch below is a starting point for a full
pre-registration, not a promise of success.

### Candidate A: event-window realized volatility vs implied vol (highest conviction)

- **What to predict:** 5-day post-earnings realized volatility per
  stock (same realized_vol_5d definition), or equivalently the
  absolute abnormal return over the (0,+1) event window.
- **Evidence it is predictable:** event-induced implied volatility
  follows a predictable creation/resolution cycle around scheduled
  announcements (Ederington and Lee 1996, JFQA); Blair, Poon, and
  Taylor (2001) show implied volatility subsumes history-based models
  for S&P 100 volatility out-of-sample.
- **Baseline to beat:** (i) pre-event at-the-money implied
  volatility (the market's own forecast), (ii) GARCH(1,1),
  (iii) HAR+VIX. These are the right bars because IV is the
  documented state of the art and GARCH is the Hansen-Lunde null.
  The question is whether anything (sentiment, order flow, cross
  terms) adds to IV, not whether volatility is predictable.
- **Metric:** QLIKE primary, Diebold-Mariano and Model Confidence
  Set for arm comparison (per Patton 2011 and the arch pattern).
- **Data:** yfinance daily prices plus earnings dates (yfinance
  earnings calendar) for S&P 100, dev period only. Obtainable today.
- **Pre-registration sketch:** success iff the challenger arm beats
  the best baseline on paired QLIKE with the 95% CI lower bound
  above zero AND survives the Model Confidence Set at 95%; kill
  criterion = failure on either, campaign dead, negative result
  written, no target switching.

### Candidate B: HAR+VIX rematch on realized_vol_5d

- **What to predict:** the same realized_vol_5d target as the dead
  VOL campaign.
- **Evidence:** this is the arm the literature says we should have
  tried first (Blair-Poon-Taylor 2001; Pong-Shackleton-Taylor-Xu
  2004 on IV+model combinations). Our NO-GO was against GARCH
  without IV; the literature's highest-conviction volatility arm was
  never tested.
- **Baseline to beat:** GARCH(1,1) (unchanged) and plain HAR. The
  right bar because the claim under test is narrowly "does VIX add
  to history-based models," which the literature answers yes for
  index vol; the open question is per-stock 5-day vol.
- **Metric:** QLIKE primary with HAC standard errors, DM test,
  MCS; secondary MSE.
- **Data:** yfinance prices plus ^VIX daily, dev period only.
  Obtainable today.
- **Pre-registration sketch:** success iff HAR+VIX beats GARCH on
  paired QLIKE with 95% CI lower bound above zero; kill criterion =
  miss on the primary comparison kills the campaign (this is a
  one-arm rematch, not a fishing expedition). Note the honest prior:
  the literature expects IV to help at index level; per-stock 5-day
  may still lose.

### Candidate C: tail-quantile (VaR) forecasting around events, CAViaR baseline

- **What to predict:** the 5% conditional quantile of the 3-day
  event return (one-sided tail), evaluated as a VaR forecast.
- **Evidence:** CAViaR (Engle and Manganelli 2004, JBES) is the
  canonical quantile-dynamics model with documented out-of-sample
  tail-forecast performance; variance is the one robustly
  predictable quantity in finance. This converts the failed
  direction program into a risk-forecasting program with honest
  econometric baselines.
- **Baseline to beat:** Engle-Manganelli CAViaR and GARCH-filtered
  historical-simulation VaR. The right bar because quantile
  dynamics are the published standard for tail forecasting.
- **Metric:** quantile (pinball) loss primary; Kupiec
  unconditional-coverage and Christoffersen conditional-coverage
  tests; Model Confidence Set.
- **Data:** yfinance daily prices, dev period only. Obtainable
  today.
- **Pre-registration sketch:** success iff the challenger beats
  CAViaR on pinball loss with 95% CI lower bound above zero AND
  passes both coverage tests while the baseline fails at least
  one; kill criterion = miss on pinball, campaign dead. Success
  here is a publishable risk-forecasting contribution even if
  direction stays dead.

## 5. Targets to avoid (explicit)

- **News-sentiment direction prediction.** Tried and failed at
  gate 1; the literature says the 3-day sentiment edge decayed to
  zero years ago (Tetlock to Ke-Kelly-Xiu). Retrying without a new
  thesis (e.g., surprise-based PEAD framing) is blind testing.
- **Plain realized-volatility ML rematches without implied vol.**
  Tried and failed; the literature says GARCH is a hard null on
  daily data and we confirmed it. Only the IV-augmented arm
  (Candidate B) is justified.
- **Direct drawdown-event prediction.** No credible peer-reviewed
  baseline; best public numbers are implausible preprints. Avoid
  until a baseline is established.
- **Zero-shot time-series foundation models for price direction.**
  Marketing exceeds evidence; they fail on random-walk data and no
  paper documents a financial edge beyond vendor material.
- **Cross-sectional monthly return prediction a la GKX.** The R2
  is real but the economics live in microcaps where costs erase
  them; needs CRSP-scale data and cost modeling we do not have.
- **Anything requiring GDELT DOC.** Hard-throttled on this IP,
  and at least one independent repo flags the free API as
  deprecated toward BigQuery. Treat as unreliable infrastructure.

## 6. Recommended next step

Candidate A is the highest-conviction next campaign: it asks the
narrowest literature-backed question (does anything add to implied
volatility around earnings?), uses only obtainable data, and has the
harshest honest baselines pre-committed. Candidate B is the cheapest
(the data and harness already exist from the VOL campaign). Candidate
C is the most ambitious and the most publishable if it works.

Whichever is chosen, the campaign needs its own pre-registration
(target, baseline, metric, success rule, kill criterion, frozen
periods) committed before any data is touched, per the standing
program rules. The fixed benchmark arena (pattern 3.3.3) should be
built alongside: one harness, fixed universe, fixed metrics, naive
baselines always present.
