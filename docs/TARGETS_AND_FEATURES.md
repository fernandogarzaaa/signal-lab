# Targets and Features (Phase 2)

Target set version **2.0**, implemented in
`src/signal_lab/validation/targets.py` (targets) and
`src/signal_lab/models/context_features.py` (`build_price_features`).
Phase 2 adds versioned targets and trailing price/volume features; it
changes no Phase 1 methodology (t0 semantics, purge/embargo, per-fold
cutoffs are untouched).

## Timing semantics

**Label window.** For an article with t0 (the first tradable time after
publication, from `signal_lab.validation.timing`; t0 is an *input* to the
target builder, never recomputed), the horizon-H label window is the
half-open interval **(close(t0), close(t0+H)]**: the base is t0's close,
and the window covers the H trading days strictly after t0. For an
after-close publication, t0 is that day's close, so the first measured
return is the next trading day's -- the return that already happened
before the news existed is not in the window.

**Feature as-of rule.** Every price/volume feature is computed as of D,
the latest trading day *strictly before* the article's publish date, using
only bars with date <= D. This is the same rule as the Phase 1 ctx_*
features (shared `_asof_positions` helper: `searchsorted(..., side="left")
- 1`). `validation.point_in_time.check_frame` asserts D < pub_date per
row, and `FEATURE_POINT_IN_TIME` declares
observation/publication/availability for every new feature.

## Targets (`target_version = "2.0"`)

`TargetConfig(horizons=(1, 3, 5), benchmark="SPY")`, frozen dataclass with
`validated()`. Per article, per horizon H:

| Column | Formula | Units |
|---|---|---|
| `ret_Hd` | close(t0+H) / close(t0) - 1, compounded from daily simple returns | decimal |
| `mkt_Hd` | same window for the benchmark leg | decimal |
| `excess_Hd` | `ret_Hd - mkt_Hd` | decimal |
| `direction_Hd` | 1.0 iff `ret_Hd > 0`, else 0.0 | {0, 1} |
| `realized_vol_5d` | sample std (ddof=1) of daily log returns over the 5 trading days (t0, t0+5] | decimal, daily |

The market leg is the benchmark's daily simple return per date, falling
back to the universe equal-weight mean return on dates the benchmark has
no return -- the same fallback `models.labels` uses.

**Partial windows.** When fewer than H forward trading days carry both
ticker and market returns, that horizon's four columns are NaN; the row is
kept (other horizons may be complete). `direction_Hd` is NaN on a partial
window rather than 0 -- a 0 would be a fabricated label. Insufficient
forward history is never filled or forward-filled.

### `excess_3d == abn_ret` (equivalence)

`excess_3d` is formula-identical to the existing `abn_ret` column
produced by `models.labels.build_labels` (window_days=3, benchmark SPY):
same t0 anchor, same (close(t0), close(t0+3)] window, same daily simple
returns, same benchmark-else-universe-mean market leg, same partial-day
counting. The target builder mirrors `forward_cumulative_abnormal`
operation-for-operation, and `tests/test_targets_features.py` asserts the
equality on synthetic panels and on the real DuckDB. The v2 target set
therefore *subsumes* the Phase 1 label input. The per-fold top-decile
cutoff that turns `abn_ret`/`excess_3d` into the primary classification
label stays in `signal_lab.validation.labels` -- it is a train-sample
statistic and must be computed on train data only (audit 4c).

## Price/volume features (`PRICE_FEATURE_NAMES`, 14)

All as of D (latest trading day strictly before pub_date), NaN on
insufficient history (never filled):

| Feature | Formula | Units |
|---|---|---|
| `px_ret_1d` | close[D]/close[D-1] - 1 | decimal |
| `px_rsi_14` | Wilder's RSI(14); seed = mean of first 14 gains/losses, then avg_t = (avg_{t-1}*13 + x_t)/14; first value at the 15th close | 0-100 |
| `px_macd_hist` | MACD(12,26,9) histogram = (EMA12 - EMA26) - EMA9(MACD line), recursive EMA; NaN until 35 closes | price points |
| `px_ma_dist_20` | (close[D] - SMA20[D]) / SMA20[D] | decimal |
| `px_ma_dist_50` | (close[D] - SMA50[D]) / SMA50[D] | decimal |
| `px_realvol_5d` | std (ddof=1) of daily log returns, trailing 5d ending on D | decimal, daily |
| `px_realvol_20d` | std (ddof=1) of daily log returns, trailing 20d ending on D | decimal, daily |
| `px_atr_14` | Wilder's ATR(14) / close[D]; TR = max(high-low, \|high-prev_close\|, \|low-prev_close\|), seeded with the mean of the first 14 TRs | fraction of price |
| `px_vol_z_20` | (vol[D] - mean(vol[D-19..D])) / std(ddof=1); exactly 0.0 when trailing volume is constant | z-score |
| `px_relvol_20` | vol[D] / median(vol[D-19..D]) | ratio |
| `px_vs_spy_5d` | ticker 5d trailing return minus SPY 5d trailing return, same window ending on D | decimal |
| `px_vs_spy_20d` | ticker 20d trailing return minus SPY 20d trailing return, same window ending on D | decimal |
| `mkt_spy_ret_20d` | SPY 20d trailing return ending on D | decimal |
| `mkt_spy_vol_20d` | std (ddof=1) of SPY daily log returns, trailing 20d ending on D | decimal, daily |

Notes:

- `px_ret_1d` fills the 1-day gap deliberately: `ctx_mom5`/`ctx_mom20`
  already cover the 5d/20d trailing returns, so it does not duplicate them.
- Volatility features use *log* returns; momentum/return features use
  *simple* returns. Realized-volatility columns are in daily units, not
  annualized.
- The SPY-relative features align the benchmark leg by date (latest SPY
  trading day on or before D, the same searchsorted pattern as
  `ctx_sector_rel5`). If the benchmark is absent from the price panel they
  stay NaN (logged, not raised).
- **VIX: skipped.** There is no local VIX series in the DuckDB backfill,
  and adding one would put a network dependency in the research path.
  `mkt_spy_ret_20d` / `mkt_spy_vol_20d` cover the market-regime role a VIX
  leg would play.

## Wiring

`build_dataset()` (`models/build_and_train.py`) attaches the target
columns and the 14 price-feature columns to the labeled frame after the
legacy labeling step; every existing column is kept. Rows with NaN price
features (insufficient pre-publication history) are kept, not dropped --
`featurize` raises on NaN dense input, so downstream callers must drop or
impute first (the walk-forward smoke test drops them explicitly). The
price query now selects the full OHLCV panel
(`ticker, date, open, high, low, close, volume`).
