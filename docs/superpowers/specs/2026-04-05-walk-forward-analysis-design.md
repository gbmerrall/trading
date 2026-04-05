# Walk-Forward Analysis — Design Spec

**Date:** 2026-04-05
**Status:** Approved

---

## Overview

Add Walk-Forward Analysis (WFA) to the backtesting framework. WFA validates that a strategy's
parameters are robust by repeatedly optimising on a training window and testing on a subsequent
out-of-sample window. The stitched test-window results form a composite equity curve that
represents realistic out-of-sample performance.

This work introduces two new modules:
- `backtest/metrics.py` — 10 built-in objective functions + a callable protocol
- `backtest/optimization.py` — grid/random search, WFA orchestrator, result type

The runner's existing `_calculate_metrics` is refactored to delegate to `metrics.py` so the same
metrics appear in both regular backtests and WFA results. No behaviour change to existing callers.

---

## File Layout

```
backtest/
├── metrics.py          # NEW
├── optimization.py     # NEW
├── runner.py           # MODIFIED — _calculate_metrics delegates to metrics.py
└── ... (unchanged)

tests/
├── test_metrics.py         # NEW
└── test_optimization.py    # NEW
```

---

## metrics.py

### MetricFn Protocol

```python
MetricFn = Callable[[list[dict], list[dict]], float]
# args: portfolio_history, trades
# returns: scalar score — higher is always better
```

### Built-in Functions

All functions have signature `(portfolio_history: list[dict], trades: list[dict]) -> float`.

| Name | Description |
|------|-------------|
| `total_return` | `(final_value - initial_value) / initial_value` |
| `cagr` | Annualised total return using `TRADING_DAYS_PER_YEAR = 252` |
| `sharpe_ratio` | Mean daily excess return / std of daily returns, annualised |
| `sortino_ratio` | Like Sharpe but denominator uses downside deviation only |
| `calmar_ratio` | CAGR / abs(max_drawdown) |
| `max_drawdown` | Largest peak-to-trough decline; returned as negative float |
| `ulcer_index` | RMS of percentage drawdown depth over time; returned as negative float |
| `profit_factor` | Gross winning P&L / gross losing P&L |
| `win_rate` | Count of profitable trades / total trades |
| `expectancy` | Mean P&L per trade |
| `recovery_factor` | Total return / abs(max_drawdown) |

**Sentinel value:** any function returns `float('-inf')` when `trades` is empty (covers division-
by-zero cases such as profit_factor with no losing trades, win_rate with zero trades, etc.).
The `min_trades` check is the optimizer's responsibility: if `len(trades) < min_trades`, the
optimizer substitutes `float('-inf')` without calling the metric at all.

**Sign convention:** metrics where lower is worse (drawdown, ulcer index) are negated so the
optimizer can always maximise uniformly.

### METRICS Registry

```python
METRICS: dict[str, MetricFn] = {
    "total_return":    total_return,
    "cagr":            cagr,
    "sharpe_ratio":    sharpe_ratio,
    "sortino_ratio":   sortino_ratio,
    "calmar_ratio":    calmar_ratio,
    "max_drawdown":    max_drawdown,
    "ulcer_index":     ulcer_index,
    "profit_factor":   profit_factor,
    "win_rate":        win_rate,
    "expectancy":      expectancy,
    "recovery_factor": recovery_factor,
}
```

The `objective` parameter on `WalkForwardOptimizer` accepts either a key from this dict or any
`MetricFn` callable directly. Custom functions work without registration.

---

## optimization.py

### Search Strategies

Both implement `generate(param_space: dict[str, list]) -> list[dict]`.

**`GridSearch`** — exhaustive; produces the Cartesian product of all parameter lists.

**`RandomSearch(n: int, seed: int = 42)`** — samples `n` combinations uniformly at random
without replacement (or with replacement if `n` exceeds the space size). `seed` makes runs
reproducible.

`WalkForwardOptimizer` calls `searcher.generate(param_space)` and iterates — it is agnostic to
which searcher is in use.

### WalkForwardOptimizer

```python
WalkForwardOptimizer(
    strategy_class: type[BaseStrategy],
    param_space:    dict[str, list],
    data:           pd.DataFrame,
    train_size:     int,            # bars
    test_size:      int,            # bars
    window_type:    str,            # "sliding" or "anchored"
    searcher:       GridSearch | RandomSearch,  # default: GridSearch()
    objective:      str | MetricFn,            # default: "sharpe_ratio"
    min_trades:     int,            # default: 5 — windows with fewer are skipped
)
```

**Constraints validated at construction:**
- `train_size >= 1`, `test_size >= 1`
- `train_size + test_size <= len(data)`
- `window_type` is `"sliding"` or `"anchored"`
- `strategy_class` is a subclass of `BaseStrategy`
- `objective` is a string key in `METRICS` or a callable

### Window Types

**Sliding:**
```
Window 1:  [T0 ----train(N)---- T1][T1 --test(M)-- T2]
Window 2:      [T1 ----train(N)---- T2][T2 --test(M)-- T3]
```
Train window is always `train_size` bars. Both train and test advance by `test_size` each step.

**Anchored:**
```
Window 1:  [T0 --------train-------- T1][T1 --test(M)-- T2]
Window 2:  [T0 ----------train---------- T2][T2 --test(M)-- T3]
```
Train always starts at `T0` and expands. Test advances by `test_size` each step.

### Per-Window Execution

For each window:
1. Slice train data.
2. Run `searcher.generate(param_space)` to get candidate parameter sets.
3. For each candidate: instantiate `strategy_class(**params)`. Apply warmup buffer using that
   instance's `warmup_period` — the full window slice is passed to the strategy (so indicators
   initialise correctly) but the first `warmup_period` bars are excluded from performance scoring.
   Run `BacktestRunnerImpl` with no benchmarks, score the result with the objective function.
4. Select the best-scoring parameter set.
5. Apply best params to the test window (also with warmup buffer applied to scoring).
6. Record test equity curve, trades, and all 10 metrics for this window.

**`BacktestRunnerImpl` reuse:** WFA creates a runner per window with a single strategy and an
empty benchmarks list. This keeps WFA decoupled from benchmark logic and avoids modifying the
runner.

**Leakage prevention:**
- Train and test windows share no bars.
- The 1-day signal shift is baked into every strategy — no extra handling needed at the WFA level.
- Warmup buffer on the test window prevents the first `warmup_period` bars from contributing to
  test metrics (those bars' indicator values were seeded by train data).

### WalkForwardResult

```python
@dataclass
class WalkForwardResult:
    equity_curve:        pd.Series     # stitched out-of-sample equity curve
    windows:             pd.DataFrame  # one row per window (see below)
    summary:             dict          # aggregate metrics over composite curve
    best_params_overall: dict          # params that appeared most often across windows;
                                       # ties broken by highest mean objective score
```

**`windows` DataFrame columns:**
`train_start`, `train_end`, `test_start`, `test_end`, `best_params` (Python dict stored as
object dtype — use `result.windows["best_params"].tolist()` to iterate), `objective_score`,
`n_trades`, `total_return`, `cagr`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`,
`max_drawdown`, `ulcer_index`, `profit_factor`, `win_rate`, `expectancy`, `recovery_factor`

**`summary` dict:** all 10 metrics computed over the full stitched `equity_curve`, plus
`n_windows`, `n_windows_with_trades`, and `param_stability` (fraction of windows where
`best_params_overall` was selected).

---

## runner.py Changes

`_calculate_metrics(portfolio_history, trades)` is refactored to call the functions in
`metrics.py` rather than reimplementing the logic. The method signature and return dict keys are
unchanged; existing tests pass without modification.

---

## Testing

### test_metrics.py

- Each metric function tested in isolation with synthetic `portfolio_history` / `trades` dicts.
- Sentinel value (`float('-inf')`) returned when trade count < `min_trades`.
- Sign convention verified for drawdown metrics (must be negative).
- CAGR verified with a known 1-year doubling scenario (expected: 1.0).
- Sharpe/Sortino verified with a flat-return series (expected: 0.0 or `float('-inf')`).

### test_optimization.py

- **Window splitting:** verify windows are contiguous, non-overlapping, cover the full date range,
  and that anchored windows always start at index 0.
- **Warmup buffer:** verify signals from the first `warmup_period` bars of each window are
  excluded from scoring.
- **Data leakage:** verify no test-window bars appear in the train slice.
- **GridSearch:** verify Cartesian product size equals product of all list lengths.
- **RandomSearch:** verify `n` combinations returned, results are reproducible with same seed,
  different seed gives different results.
- **End-to-end:** run WFA on synthetic data with a known-optimal parameter; verify the optimizer
  selects it in at least 50% of windows.
- **`WalkForwardResult`:** verify equity curve length equals sum of test window sizes minus
  warmup bars, `windows` DataFrame has correct column set.

---

## Usage Example

```python
from backtest.optimization import WalkForwardOptimizer, RandomSearch
from backtest.metrics import sharpe_ratio
from backtest.strategy import RSIStrategy

optimizer = WalkForwardOptimizer(
    strategy_class=RSIStrategy,
    param_space={
        "period":      [7, 14, 21],
        "lower_bound": [20, 25, 30],
        "upper_bound": [70, 75, 80],
    },
    data=df,
    train_size=252,
    test_size=63,
    window_type="sliding",
    searcher=RandomSearch(n=50, seed=42),
    objective="sharpe_ratio",
    min_trades=5,
)

result = optimizer.run()
print(result.summary)
print(result.windows[["test_start", "best_params", "sharpe_ratio"]])
result.equity_curve.plot()
```

---

## Out of Scope

- Visualisation / reporting (planned for Phase 5)
- Bayesian / evolutionary optimisation (can be added as a third searcher later)
- Multi-asset WFA
- Parallel window execution
