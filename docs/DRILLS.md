# Python Live-Coding Drills

Intel: LSEG first rounds include timed live Python coding on a shared screen
with no syntax reference, followed by solution interpretation plus ML and
Python concept questions. Drill accordingly: blank editor, timer on, no docs,
no autocomplete.

## Rules

- 25 minutes per drill unless stated.
- Talk through your reasoning out loud as you type, like the real round.
- After the timer: review, then rewrite the weak parts once.

## Drill set

### D1. Skip-sum with bug hunt (reported question shape)
You are given this function. Find the bugs, fix them, and state the time
complexity.

```python
def skip_sum(nums, skip):
    total = 0
    for i in range(len(nums)):
        if nums[i] == skip:
            i += 1  # intended to skip the next element too
        total += nums[i]
    return total
```

Expected behavior: sum all ints except `skip` values and the element
immediately after each `skip`. Handle edge cases: `skip` at the last index,
consecutive skips, empty list.

### D2. Nested dict to dot notation (reported question shape)
Flatten `{"a": {"b": {"c": 1}}, "d": 2}` to `{"a.b.c": 1, "d": 2}`.
Then extend: handle lists by index (`{"a": [{"b": 1}]}` -> `{"a.0.b": 1}`).
Then write the inverse `unflatten`.

### D3. Event join
Given `news = [(timestamp, ticker, sentiment)]` and
`prices = [(timestamp, ticker, close)]`, produce for each news item the
next-day return per ticker without using pandas. Then redo it in pandas and
compare readability. State where a lookahead bug could hide.

### D4. Imbalanced metrics from scratch
Implement precision, recall, F1, and PR-AUC from raw label/score lists using
only the standard library. Then explain in two sentences when PR-AUC is the
right choice over ROC-AUC.

### D5. Walk-forward splitter
Implement a generator yielding (train_idx, test_idx) pairs for walk-forward
validation over a time-ordered dataset: expanding train window, fixed test
window, no shuffling. Write a test proving no test timestamp precedes a train
timestamp.

### D6. SQL fluency (write on paper, then check in DuckDB)
1. Top 3 tickers by average daily sentiment last week.
2. News count per source domain per day, days with zero news included.
3. For each ticker, the close price on the day after each positive-sentiment
   article (join with date arithmetic).

### D7. Explain-your-solution set (say out loud, 2 minutes each)
- Why did you choose a t-test in M4, and what breaks if returns are not normal?
- What does class_weight="balanced" do to logistic regression's loss?
- Walk-forward vs shuffled k-fold for time series: why does shuffling lie?
- How would you detect label leakage in a feature pipeline?

## Scoring

Pass a drill when the code runs correctly on first execution and you can
answer the follow-up without pausing. Anything less: redo it tomorrow.
