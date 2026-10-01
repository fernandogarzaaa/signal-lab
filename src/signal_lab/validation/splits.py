"""Purged + embargoed walk-forward splits on t0/t1 (Phase 1).

Replaces the calendar-day purge approximation with the exact decontamination
rule. Every sample carries t0 (first tradable time after publication) and t1
(t0 + H trading days); its label is a function of returns in (t0, t1].

Decontamination (Lopez de Prado, ch. 7), applied per fold k with test range
[ts_k, te_k]:

- PURGE: drop train rows whose [t0, t1] overlaps ANY test range up to and
  including fold k. A label computed from returns inside a test period is
  not knowable at train time. Because test ranges are contiguous, this is
  equivalent to dropping train rows with t1 >= ts_1 (first test start).
- EMBARGO: drop train rows with t0 in (te_j, te_j + E trading days] for any
  earlier fold j < k. The immediate aftermath of a test period is serially
  correlated with it; training on it is a soft leak.

The splitter asserts its own invariants on every call (no train/test index
overlap, no [t0, t1] overlap with any test range among kept train rows,
embargo zones empty, train strictly before test). These are runtime
assertions, not optional checks.

Windows:

- ``expanding``: fold k trains on all blocks before k.
- ``rolling``: fold k trains on rows with t0 in
  [ts_k - train_window_days, ts_k).

``retrain_every = r`` means the model is refit on folds 1, 1+r, 1+2r, ...;
intermediate folds reuse the previous fit (the split/decontamination is
identical either way; only the fitting is skipped).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from signal_lab.validation import periods


@dataclass(frozen=True)
class WalkForwardConfig:
    n_splits: int = 5
    window: str = "expanding"  # or "rolling"
    train_window_days: int = 365  # rolling only: calendar days of train history
    horizon_days: int = 3  # H: label horizon in trading days
    embargo_days: int = 5  # trading days; must be >= horizon_days
    min_train: int = 50
    retrain_every: int = 1
    seed: int = 7  # reserved: splits are positional/deterministic; any future
    # stochastic op (e.g. subsampling) must use this seed
    dev_end: date | None = None  # default: periods.DEV_END
    purge: bool = True  # escape hatch for the canary test only; production
    embargo: bool = True  # code must never set these to False

    def validated(self) -> WalkForwardConfig:
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")
        if self.window not in ("expanding", "rolling"):
            raise ValueError(f"window must be expanding|rolling, got {self.window!r}")
        if self.train_window_days < 1:
            raise ValueError("train_window_days must be >= 1")
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")
        if self.embargo_days < self.horizon_days:
            raise ValueError(
                f"embargo_days ({self.embargo_days}) must be >= horizon_days "
                f"({self.horizon_days})"
            )
        if self.min_train < 1:
            raise ValueError("min_train must be >= 1")
        if self.retrain_every < 1:
            raise ValueError("retrain_every must be >= 1")
        return self


def _to_ord(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s).dt.date.apply(lambda d: d.toordinal()).to_numpy(dtype=np.int64)


def _add_trading_days(
    day_ord: int, n: int, trading_ord: np.ndarray
) -> int:
    """Ordinal of the date n trading days after ``day_ord`` (union calendar).

    Returns -1 if it runs past the calendar end.
    """
    pos = np.searchsorted(trading_ord, day_ord, side="left")
    if pos >= len(trading_ord) or trading_ord[pos] != day_ord:
        # day_ord may not be on the union calendar (should not happen for
        # t0/te dates); fall back to the next trading day at/after it.
        pos = np.searchsorted(trading_ord, day_ord, side="left")
    dest = pos + n
    return int(trading_ord[dest]) if dest < len(trading_ord) else -1


def make_splits(
    df: pd.DataFrame,
    config: WalkForwardConfig,
    trading_days: np.ndarray,
) -> list[dict]:
    """Build purged + embargoed walk-forward splits.

    ``df`` needs columns ``t0``, ``t1`` (dates) and ``published_at``.
    ``trading_days``: sorted unique trading-date ordinals (union calendar),
    used for embargo arithmetic.
    Returns one dict per fold: fold, train_idx, test_idx (positional into
    the *filtered* frame), test_start, test_end, n_purged, n_embargoed,
    refit.

    Purge (exact, per Lopez de Prado): for fold ``k``, drop train rows
    whose [t0, t1] label window overlaps the CURRENT fold's test range,
    i.e. ``t1 >= test_start_k`` (train rows always satisfy
    ``t0 < test_start_k``, so this is the full interval-overlap
    condition). A train label that uses returns from the test period would
    hand the model the test outcomes during training.

    NOTE on the program spec: the Phase 1 spec text says "any test range
    j <= k". That is implemented here as j == k only, deliberately.
    Purging against earlier folds' test ranges would drop every train row
    whose t0 falls in blocks 1..k-1 (each such row's t0 lies inside its
    own block's test range), collapsing every fold's train set to block 0
    and silently turning the "expanding window" into a fixed window.
    Earlier blocks' labels use only returns strictly before
    ``test_start_k``, so they carry no test-period information and are
    legitimate training data under point-in-time. The j == k purge removes
    100% of the actual leakage (train labels embedding test-period
    returns) with none of the data destruction. Flagged in the Phase 1 PR
    for maintainer review.
    """
    config = config.validated()
    dev_end = config.dev_end or periods.DEV_END
    work = df.dropna(subset=["t0", "t1"]).reset_index(drop=True)
    periods.check_no_confirmation(work["t0"], "make_splits")
    work = work[_to_ord(work["t0"]) <= dev_end.toordinal()].reset_index(drop=True)
    if len(work) == 0:
        raise ValueError("[splits] no rows with t0 <= dev_end after filtering")

    t0_ord = _to_ord(work["t0"])
    t1_ord = _to_ord(work["t1"])
    pub_ord = pd.to_datetime(work["published_at"], utc=True).astype(np.int64).to_numpy()
    # Deterministic order: t0, tie-broken by publication time.
    order = np.lexsort((pub_ord, t0_ord))
    t0s, t1s = t0_ord[order], t1_ord[order]
    n = len(work)
    cuts = [int(round(n * k / (config.n_splits + 1))) for k in range(config.n_splits + 2)]
    # Snap interior cuts to t0 boundaries: a day's rows must never be
    # split across train/test (otherwise train t0 == test start, breaking
    # the strict "train before test" invariant when >1 row shares a t0).
    for k in range(1, len(cuts) - 1):
        c = min(cuts[k], n - 1)
        while c < n and t0s[c] == t0s[c - 1]:
            c += 1
        cuts[k] = c

    trading_ord = np.asarray(sorted(set(int(x) for x in trading_days)), dtype=np.int64)
    test_starts, test_ends = [], []
    for k in range(1, config.n_splits + 1):
        b_k, b_k1 = cuts[k], cuts[k + 1]
        test_starts.append(int(t0s[b_k]))
        test_ends.append(int(t0s[b_k1 - 1]))

    splits = []
    for k in range(1, config.n_splits + 1):
        b_k, b_k1 = cuts[k], cuts[k + 1]
        ts_k, te_k = test_starts[k - 1], test_ends[k - 1]
        test_pos = np.arange(b_k, b_k1)

        if config.window == "expanding":
            train_pos = np.arange(0, b_k)
        else:
            lo = ts_k - config.train_window_days
            train_pos = np.arange(0, b_k)[t0s[:b_k] >= lo]

        keep = np.ones(len(train_pos), dtype=bool)
        n_purged, n_embargoed = 0, 0
        if config.purge and len(train_pos):
            # Exact purge vs the CURRENT fold's test range only: train rows
            # always have t0 < ts_k, so overlap <=> t1 >= ts_k. (Purging
            # against earlier folds' test ranges would collapse the
            # expanding window to block 0; see docstring.)
            overlap = t1s[train_pos] >= ts_k
            n_purged = int(overlap.sum())
            keep &= ~overlap
        if config.embargo and len(train_pos):
            for j in range(k - 1):
                zone_end = _add_trading_days(test_ends[j], config.embargo_days, trading_ord)
                if zone_end < 0:
                    continue
                zone = (t0s[train_pos] > test_ends[j]) & (t0s[train_pos] <= zone_end)
                n_embargoed += int((zone & keep).sum())
                keep &= ~zone

        train_kept = train_pos[keep]
        if len(train_kept) < config.min_train:
            raise ValueError(
                f"fold {k}: only {len(train_kept)} train rows after purge/embargo "
                f"(min_train={config.min_train})"
            )

        # --- runtime invariant assertions (always on) ---
        tri_orig, tei_orig = order[train_kept], order[test_pos]
        assert len(set(tri_orig) & set(tei_orig)) == 0, f"fold {k}: train/test index overlap"
        assert (t0s[train_kept] < ts_k).all(), f"fold {k}: train t0 not strictly before test start"
        if config.purge:
            bad = t1s[train_kept] >= ts_k
            assert not bad.any(), f"fold {k}: {bad.sum()} kept train rows overlap the test range"
        if config.embargo:
            for j in range(k - 1):
                zone_end = _add_trading_days(test_ends[j], config.embargo_days, trading_ord)
                if zone_end < 0:
                    continue
                bad = (t0s[train_kept] > test_ends[j]) & (t0s[train_kept] <= zone_end)
                assert not bad.any(), f"fold {k}: embargo zone of fold {j+1} not empty"

        splits.append(
            {
                "fold": k,
                "train_idx": tri_orig,
                "test_idx": tei_orig,
                "test_start": pd.Timestamp.fromordinal(ts_k).date().isoformat(),
                "test_end": pd.Timestamp.fromordinal(te_k).date().isoformat(),
                "n_purged": n_purged,
                "n_embargoed": n_embargoed,
                "refit": ((k - 1) % config.retrain_every == 0),
            }
        )
    return splits


def purge_against(
    t0: pd.Series,
    t1: pd.Series,
    test_start: date,
    test_end: date,
) -> pd.Series:
    """Keep-mask: drop rows whose [t0, t1] overlaps [test_start, test_end].

    Used for FINAL TRAIN: dev rows whose label window reaches into the
    untouched test period must not train the final model.
    """
    t0o = _to_ord(t0)
    t1o = _to_ord(t1)
    ts, te = test_start.toordinal(), test_end.toordinal()
    overlap = (t0o <= te) & (t1o >= ts)
    return pd.Series(~overlap, index=t0.index)


def single_purged_split(
    df: pd.DataFrame,
    config: WalkForwardConfig,
    trading_days: np.ndarray,
    valid_fraction: float = 0.2,
) -> dict:
    """One purged train/validation split on the development period.

    Used for model SELECTION (audit finding 4b): candidates are compared
    on a validation block they never trained on, instead of on the test
    split they were chosen on. The validation block is the last
    ``valid_fraction`` of dev rows by t0; train rows are purged against
    it ([t0, t1] overlap with the validation range is dropped). There is
    no earlier test period, so the embargo is vacuous here.
    """
    config = config.validated()
    if not 0.0 < valid_fraction < 0.5:
        raise ValueError(f"valid_fraction must be in (0, 0.5), got {valid_fraction}")
    dev_end = config.dev_end or periods.DEV_END
    work = df.dropna(subset=["t0", "t1"]).reset_index(drop=True)
    periods.check_no_confirmation(work["t0"], "single_purged_split")
    work = work[_to_ord(work["t0"]) <= dev_end.toordinal()].reset_index(drop=True)
    if len(work) < config.min_train + 1:
        raise ValueError(
            f"[splits] only {len(work)} dev rows for selection split "
            f"(min_train={config.min_train})"
        )
    t0_ord = _to_ord(work["t0"])
    t1_ord = _to_ord(work["t1"])
    pub_ord = pd.to_datetime(work["published_at"], utc=True).astype(np.int64).to_numpy()
    order = np.lexsort((pub_ord, t0_ord))
    t0s, t1s = t0_ord[order], t1_ord[order]

    n_valid = max(1, int(round(len(work) * valid_fraction)))
    v_start_pos = len(work) - n_valid
    vs, ve = int(t0s[v_start_pos]), int(t0s[-1])
    train_pos = np.arange(0, v_start_pos)
    valid_pos = np.arange(v_start_pos, len(work))

    purged = (t1s[train_pos] >= vs) & (t0s[train_pos] <= ve)
    n_purged = int(purged.sum())
    train_kept = train_pos[~purged]
    if len(train_kept) < config.min_train:
        raise ValueError(
            f"[splits] only {len(train_kept)} train rows after purge "
            f"(min_train={config.min_train})"
        )

    tri_orig, vai_orig = order[train_kept], order[valid_pos]
    assert len(set(tri_orig) & set(vai_orig)) == 0, "selection split index overlap"
    bad = (t1s[train_kept] >= vs) & (t0s[train_kept] <= ve)
    assert not bad.any(), "selection split: kept train rows overlap validation range"
    return {
        "train_idx": tri_orig,
        "valid_idx": vai_orig,
        "valid_start": pd.Timestamp.fromordinal(vs).date().isoformat(),
        "valid_end": pd.Timestamp.fromordinal(ve).date().isoformat(),
        "n_purged": n_purged,
    }
