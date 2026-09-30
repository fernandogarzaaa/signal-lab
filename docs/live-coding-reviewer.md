# Live-Coding Reviewer — LSEG Data Scientist

Companion to [DRILLS.md](DRILLS.md). The drills are the exam; this is the
textbook. Work it in order:

1. **Type** every snippet below by hand. Do not copy-paste. The point is
   finger memory, not reading.
2. Say the one-line explanation under each snippet **out loud** as you type
   it. Interviewers hear your reasoning; train it now.
3. Then do DRILLS.md timed (25 min, blank editor, no docs).
4. Then mock rounds with Cookie as interviewer.

Every snippet in this file runs as written. If one doesn't run for you,
your environment is the bug, not the snippet.

---

## A. Python fluency

The warm-up layer. If these aren't automatic, everything downstream feels
slow.

```python
# Filter + transform in one pass. You will write this shape constantly.
nums = [3, 1, 4, 1, 5, 9, 2, 6]
evens_squared = [x * x for x in nums if x % 2 == 0]   # [16, 4, 36]
```

```python
# Dict comprehension.
words = ["alpha", "beta", "gamma"]
lengths = {w: len(w) for w in words}                  # {"alpha": 5, ...}
```

```python
# The counting pattern. Label distributions, value counts, quick EDA.
from collections import Counter
labels = [0, 1, 0, 0, 1, 0]
counts = Counter(labels)                              # Counter({0: 4, 1: 2})
pos_rate = counts[1] / sum(counts.values())           # 0.333...
```

```python
# Safe nested access. Never let a missing key crash a live round.
article = {"meta": {"source": {"domain": "reuters.com"}}}
domain = article.get("meta", {}).get("source", {}).get("domain")
```

```python
# zip + enumerate together: the "walk two lists" shape.
tickers = ["AAPL", "MSFT"]
closes = [100.0, 200.0]
for i, (t, c) in enumerate(zip(tickers, closes)):
    print(i, t, c)
```

**Bug-hunt pattern (D1's answer).** The original bug: `i += 1` inside a
`for` loop does nothing to the loop variable. The fix needs a `while` loop
so you control the index:

```python
def skip_sum(nums, skip):
    total = 0
    i = 0
    while i < len(nums):
        if nums[i] == skip:
            i += 2      # skip the value AND the element after it
        else:
            total += nums[i]
            i += 1
    return total

assert skip_sum([1, 5, 99, 2, 5, 3], 5) == 3
assert skip_sum([5, 1], 5) == 0          # skip at work: value + next gone
assert skip_sum([5], 5) == 0            # skip at last index: no crash
assert skip_sum([], 5) == 0
```

Say out loud: "O(n) time, O(1) space. Edge cases: skip at the last index,
consecutive skips, empty list." Stating the complexity unprompted is what
separates a pass from a strong pass.

**Flatten (D2's answer).** Recursion with a prefix accumulator:

```python
def flatten(d, prefix="", out=None):
    out = {} if out is None else out
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flatten(v, key, out)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                ikey = f"{key}.{i}"
                if isinstance(item, dict):
                    flatten(item, ikey, out)
                else:
                    out[ikey] = item
        else:
            out[key] = v
    return out

assert flatten({"a": {"b": {"c": 1}}, "d": 2}) == {"a.b.c": 1, "d": 2}
assert flatten({"a": [{"b": 1}]}) == {"a.0.b": 1}
```

The inverse. Note the honest limitation: list indices become string keys,
so `unflatten` restores dicts, not lists. Say that out loud in the round;
naming the limitation scores points.

```python
def unflatten(d):
    root = {}
    for compound, v in d.items():
        parts = compound.split(".")
        node = root
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return root
```

---

## B. pandas

The core of data-science live coding. Four verbs cover 80%: `merge`,
`groupby`, `sort_values`, `shift`.

**Event join (D3's pandas half).** For each news item, the ticker's
next-day return. The lookahead trap: `shift(-1)` must happen **after**
sorting and **inside** each ticker group, never across the whole frame.

```python
import pandas as pd

news = pd.DataFrame([
    {"ts": "2026-01-05", "ticker": "AAPL", "sentiment": 0.8},
    {"ts": "2026-01-06", "ticker": "AAPL", "sentiment": -0.2},
    {"ts": "2026-01-05", "ticker": "MSFT", "sentiment": 0.5},
])
prices = pd.DataFrame([
    {"ts": "2026-01-05", "ticker": "AAPL", "close": 100.0},
    {"ts": "2026-01-06", "ticker": "AAPL", "close": 102.0},
    {"ts": "2026-01-07", "ticker": "AAPL", "close": 101.0},
    {"ts": "2026-01-05", "ticker": "MSFT", "close": 200.0},
    {"ts": "2026-01-06", "ticker": "MSFT", "close": 205.0},
])
news["ts"] = pd.to_datetime(news["ts"])
prices["ts"] = pd.to_datetime(prices["ts"])

prices = prices.sort_values(["ticker", "ts"])
prices["next_close"] = prices.groupby("ticker")["close"].shift(-1)
prices["next_ret"] = prices["next_close"] / prices["close"] - 1

out = news.merge(prices[["ts", "ticker", "next_ret"]],
                 on=["ts", "ticker"], how="left")
# AAPL 01-05 -> 0.02 | AAPL 01-06 -> -0.0099 | MSFT 01-05 -> 0.025
```

**Groupby aggregation with missing values.** Decide the NaN policy out
loud before you code it; interviewers watch for silent NaNs.

```python
df = pd.DataFrame({"ticker": ["A", "A", "B"], "ret": [0.01, None, 0.03]})
df["ret"] = df["ret"].fillna(0)          # state the choice: fill vs drop
print(df.groupby("ticker")["ret"].mean())
```

**Pivot: long to wide.** Sentiment per ticker per day as a matrix:

```python
wide = out.pivot_table(index="ts", columns="ticker",
                       values="sentiment", aggfunc="mean")
```

---

## C. sklearn and imbalanced data

Your home turf. This is where you convert Signal Lab from a project into
interview answers.

**The standard imbalanced pipeline.** Four things to say while typing:
stratify the split, weight the classes, judge by PR-AUC never accuracy,
inspect the confusion matrix.

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=7, stratify=labels)

clf = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2),
    LogisticRegression(class_weight="balanced", max_iter=1000))
clf.fit(X_train, y_train)

scores = clf.predict_proba(X_test)[:, 1]
print("PR-AUC:", average_precision_score(y_test, scores))
print(confusion_matrix(y_test, scores > 0.5))
```

**Metrics from scratch (D4's answer).** They may ask you to derive these
without sklearn, to check you know what the library hides:

```python
def prf(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1

def pr_auc(y_true, scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    total_pos = sum(y_true)
    ap, prev_rec = 0.0, 0.0
    for i in order:
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec = tp / total_pos if total_pos else 0.0
        ap += prec * (rec - prev_rec)   # area under the step function
        prev_rec = rec
    return ap
```

Two-sentence answer for "PR-AUC vs ROC-AUC": with 10% positives, ROC-AUC
is inflated by the huge number of easy true negatives, so it flatters bad
models. PR-AUC ignores true negatives and asks only "of what you flagged,
how much was real, and how much of the real stuff did you find," which is
the business question.

**Walk-forward splitter (D5's answer).** Expanding train window, fixed
test window, and a test that proves no leakage:

```python
def walk_forward_expanding(n, initial_train, test_size, step=None):
    step = step or test_size
    train_end = initial_train
    while train_end + test_size <= n:
        yield list(range(train_end)), list(range(train_end, train_end + test_size))
        train_end += step

def test_no_lookahead():
    for tr, te in walk_forward_expanding(100, 60, 10):
        assert max(tr) < min(te)   # no test index precedes a train index
```

---

## D. Statistics, said out loud

The posting asks explicitly how you incorporate statistics into data
science. Your one-paragraph answer, built from stage 4 of Signal Lab:

> "In the event study I test whether sentiment spikes move prices. The
> null hypothesis is zero cumulative abnormal return around the event.
> I compute each event's abnormal return against a market model, average
> across events, and run a t-test. The key discipline is reporting the
> non-result: one event, CAR 3.7%, not statistically significant, so no
> trade. Statistics here isn't a tool for finding alpha, it's the thing
> that stops you from fooling yourself."

**Flash answers** (two sentences each, say them verbatim until smooth):

- *p-value:* the probability of seeing an effect this big if the null were
  true. It is not the probability the null is true.
- *95% confidence interval:* the range of effect sizes compatible with the
  data. If it includes zero, you don't have an effect.
- *Why not accuracy on imbalance:* a model that predicts "nothing ever
  happens" scores 90% accuracy and learns nothing. Accuracy rewards the
  majority class; PR-AUC rewards finding the rare class.
- *What class_weight="balanced" does:* multiplies the loss of each sample
  by a weight inversely proportional to its class frequency, so the 10%
  minority class counts as much as the 90% majority during training.
- *Walk-forward vs shuffled k-fold on time series:* shuffling puts future
  rows in the training set, so the model trains on information from after
  the test point. That's lookahead bias, and it inflates every metric.
- *What breaks a t-test on returns:* returns aren't normal; they have fat
  tails. With few events the test over-rejects. Answer: bootstrap the
  distribution or use a rank test, and say so.

---

## E. SQL

DuckDB dialect (that's what the drills check against). Three shapes cover
nearly every DS SQL round: grouped aggregation, a date spine with a cross
join, and a time-shifted join.

```sql
-- 1. Top 3 tickers by average daily sentiment, last 7 days.
SELECT ticker, AVG(sentiment) AS avg_sent
FROM news
WHERE published_at >= CURRENT_DATE - INTERVAL 7 DAY
GROUP BY ticker
ORDER BY avg_sent DESC
LIMIT 3;
```

```sql
-- 2. News count per source domain per day, zero-news days included.
-- The trick: build the date spine first, then LEFT JOIN.
WITH days AS (
  SELECT CAST(day AS DATE) AS d
  FROM range(DATE '2026-01-01', CURRENT_DATE, INTERVAL 1 DAY) t(day)
),
domains AS (SELECT DISTINCT source_domain FROM news)
SELECT d.d AS day, dm.source_domain, COUNT(n.url) AS n_news
FROM days d
CROSS JOIN domains dm
LEFT JOIN news n
  ON CAST(n.published_at AS DATE) = d.d
 AND n.source_domain = dm.source_domain
GROUP BY 1, 2
ORDER BY 1, 2;
```

```sql
-- 3. For each positive-sentiment article, the ticker's close the next day.
SELECT n.ticker, n.published_at, p.close AS next_day_close
FROM news n
JOIN prices p
  ON p.ticker = n.ticker
 AND CAST(p.date AS DATE) = CAST(n.published_at AS DATE) + INTERVAL 1 DAY
WHERE n.sentiment > 0;
```

Say out loud on #3: "Joining on date-plus-one-day keeps the label strictly
after the article timestamp. Joining on the same day would leak."

---

## F. The live-round protocol

Tape this to your monitor:

1. **Restate** the problem in your own words. Ask one clarifying question
   (input size? edge cases?).
2. **Examples first.** Write 2-3 input/output pairs as comments before any
   logic. This is where edge cases get caught.
3. **Brute force out loud**, then optimize only if asked. A working O(n^2)
   beats a half-remembered O(n) you can't finish.
4. **Code in small runs.** Write 5-10 lines, mentally execute, then
   continue. Narrate as you go: "now I handle the empty case."
5. **Test out loud.** Run your examples through the code verbally. Fix what
   breaks, and say what you fixed.
6. **State complexity** at the end, unprompted. Time and space.

When stuck: say "let me think for ten seconds," go silent, then narrate
the next concrete step. Silence with a stated purpose reads as composure;
unexplained silence reads as lost.

---

## What's next

1. Type this whole file. Out loud.
2. DRILLS.md, timed, blank editor, no docs. Score yourself by its rubric.
3. Mock rounds with Cookie: I play interviewer, you share your thinking,
   no AI assistance. We'll start with D1-D4 shapes and the statistics
   probe, then fold in Globetec's prep notes when they arrive.
