# Walk-Forward Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Walk-Forward Analysis to the framework via `backtest/metrics.py` (10 objective functions) and `backtest/optimization.py` (GridSearch, RandomSearch, WalkForwardOptimizer).

**Architecture:** `metrics.py` provides pure functions `(portfolio_history, trades) -> float` consumed by both the WFA optimizer and a refactored `runner._calculate_metrics`. `optimization.py` contains the search strategies, window splitter, warmup filter, and WFA orchestrator. `BacktestRunnerImpl` is called unchanged per window; the optimizer does all pre/post-processing.

**Tech Stack:** Python stdlib (`itertools`, `dataclasses`, `random`), pandas, numpy, pytest. No new dependencies.

---

## Data shapes (read before writing any code)

Throughout this plan, `portfolio_history` and `trades` have these exact shapes — they come directly from `BacktestRunnerImpl.run()`:

```python
# portfolio_history — from runner result["strategy_returns"] converted back:
portfolio_history = [
    {"date": pd.Timestamp("2020-01-02"), "value": 10000.0},
    {"date": pd.Timestamp("2020-01-03"), "value": 10050.0},
    ...
]

# trades — from runner result["trades"]:
trades = [
    {
        "entry_date": pd.Timestamp("2020-01-02"),
        "exit_date":  pd.Timestamp("2020-01-10"),
        "entry":  100.0,
        "exit":   110.0,
        "shares": 90,
        "pnl":    900.0,
    },
    ...
]
```

`strategy_returns` is a `pd.Series` with a `DatetimeIndex` and float values (portfolio value at each date). Convert to `portfolio_history` with:
```python
portfolio_history = [{"date": d, "value": v} for d, v in series.items()]
```

---

## File map

| File | Status | Responsibility |
|------|--------|----------------|
| `backtest/metrics.py` | CREATE | 10 metric functions + `MetricFn` + `METRICS` registry |
| `backtest/optimization.py` | CREATE | `GridSearch`, `RandomSearch`, `_generate_windows`, `_filter_by_warmup`, `WalkForwardOptimizer`, `WalkForwardResult` |
| `backtest/runner.py` | MODIFY | `_calculate_metrics` delegates to `metrics.py` |
| `tests/test_metrics.py` | CREATE | Unit tests for all 10 metric functions |
| `tests/test_optimization.py` | CREATE | Unit tests for search, windows, warmup, optimizer, result |

---

## Task 1: Portfolio-value metrics (`total_return`, `cagr`, `max_drawdown`, `ulcer_index`)

**Files:**
- Create: `backtest/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics.py`:

```python
import math
from datetime import timedelta

import pandas as pd
import pytest

from backtest.metrics import (
    METRICS,
    cagr,
    max_drawdown,
    total_return,
    ulcer_index,
)


def _history(values):
    """Build portfolio_history from a list of floats."""
    dates = pd.date_range("2020-01-02", periods=len(values), freq="B")
    return [{"date": d, "value": float(v)} for d, v in zip(dates, values)]


def _trades(*pnls):
    """Build a minimal trades list from P&L values."""
    base = pd.Timestamp("2020-01-02")
    return [
        {
            "entry_date": base + timedelta(days=i * 2),
            "exit_date": base + timedelta(days=i * 2 + 1),
            "entry": 100.0,
            "exit": 100.0 + pnl,
            "shares": 1,
            "pnl": float(pnl),
        }
        for i, pnl in enumerate(pnls)
    ]


class TestTotalReturn:
    def test_positive_return(self):
        h = _history([10000, 11000])
        assert total_return(h, []) == pytest.approx(0.10)

    def test_negative_return(self):
        h = _history([10000, 9000])
        assert total_return(h, []) == pytest.approx(-0.10)

    def test_flat(self):
        h = _history([10000, 10000, 10000])
        assert total_return(h, []) == pytest.approx(0.0)

    def test_empty_history_returns_neginf(self):
        assert total_return([], []) == float("-inf")


class TestCAGR:
    def test_one_year_double(self):
        # 252 trading days, value doubles → CAGR = 1.0 (100%)
        values = [10000.0 * (2 ** (i / 252)) for i in range(253)]
        h = _history(values)
        assert cagr(h, []) == pytest.approx(1.0, rel=1e-3)

    def test_flat(self):
        h = _history([10000] * 253)
        assert cagr(h, []) == pytest.approx(0.0, abs=1e-6)

    def test_empty_history_returns_neginf(self):
        assert cagr([], []) == float("-inf")

    def test_single_bar_returns_neginf(self):
        assert cagr(_history([10000]), []) == float("-inf")


class TestMaxDrawdown:
    def test_peak_then_trough(self):
        # Peak 120, trough 80 → drawdown = (80 - 120) / 120 = -1/3
        h = _history([100, 120, 80, 90])
        assert max_drawdown(h, []) == pytest.approx(-1 / 3, rel=1e-6)

    def test_no_drawdown(self):
        h = _history([100, 110, 120, 130])
        assert max_drawdown(h, []) == pytest.approx(0.0, abs=1e-9)

    def test_returns_negative(self):
        h = _history([100, 80])
        assert max_drawdown(h, []) < 0

    def test_empty_history_returns_neginf(self):
        assert max_drawdown([], []) == float("-inf")


class TestUlcerIndex:
    def test_no_drawdown_is_zero(self):
        h = _history([100, 110, 120, 130])
        # No drawdown → ulcer_index should be 0 (negated → 0)
        assert ulcer_index(h, []) == pytest.approx(0.0, abs=1e-9)

    def test_returns_non_positive(self):
        h = _history([100, 90, 80, 70])
        assert ulcer_index(h, []) <= 0

    def test_worse_drawdown_lower_value(self):
        # Deeper sustained drawdown → more negative ulcer index
        h_shallow = _history([100, 95, 90, 95, 100])
        h_deep = _history([100, 80, 60, 80, 100])
        assert ulcer_index(h_deep, []) < ulcer_index(h_shallow, [])

    def test_empty_history_returns_neginf(self):
        assert ulcer_index([], []) == float("-inf")


class TestMETRICSRegistry:
    def test_registry_contains_expected_keys(self):
        expected = {
            "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "max_drawdown", "ulcer_index", "profit_factor",
            "win_rate", "expectancy", "recovery_factor",
        }
        assert set(METRICS.keys()) == expected

    def test_registry_values_are_callable(self):
        for name, fn in METRICS.items():
            assert callable(fn), f"METRICS['{name}'] is not callable"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_metrics.py -v 2>&1 | head -20
```

Expected: `ImportError` — `backtest.metrics` does not exist yet.

- [ ] **Step 3: Create `backtest/metrics.py` with the four portfolio-value metrics**

```python
"""Performance metric functions for the trading backtest framework.

All functions share the signature:
    (portfolio_history: list[dict], trades: list[dict]) -> float

portfolio_history entries: {"date": pd.Timestamp, "value": float}
trades entries: {"entry_date", "exit_date", "entry", "exit", "shares", "pnl"}

Sign convention:
- Higher is always better.
- Metrics where lower is worse (drawdown, ulcer index) are returned as
  negative floats so the optimizer can maximise uniformly.
- float('-inf') is returned for degenerate inputs (empty history, zero
  trades where trades are required) to ensure such windows are never
  selected as best.
"""

import math
from typing import Callable

import numpy as np
import pandas as pd

from .constants import TradingConstants

# Type alias for metric functions
MetricFn = Callable[[list[dict], list[dict]], float]

TRADING_DAYS = TradingConstants.TRADING_DAYS_PER_YEAR


def total_return(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return (final_value - initial_value) / initial_value.

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        Return as a fraction (0.10 for 10%). float('-inf') if history empty.
    """
    if not portfolio_history:
        return float("-inf")
    start = portfolio_history[0]["value"]
    end = portfolio_history[-1]["value"]
    return (end - start) / start


def cagr(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return annualised growth rate using 252 trading days per year.

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        CAGR as a fraction (0.20 for 20%). float('-inf') if fewer than 2 bars.
    """
    if len(portfolio_history) < 2:
        return float("-inf")
    start = portfolio_history[0]["value"]
    end = portfolio_history[-1]["value"]
    n_bars = len(portfolio_history) - 1
    if start <= 0:
        return float("-inf")
    return (end / start) ** (TRADING_DAYS / n_bars) - 1


def max_drawdown(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return largest peak-to-trough decline as a negative fraction.

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        Negative fraction (e.g. -0.20 for 20% drawdown). 0.0 if no drawdown.
        float('-inf') if history empty.
    """
    if not portfolio_history:
        return float("-inf")
    values = pd.Series([e["value"] for e in portfolio_history], dtype=float)
    peak = values.expanding().max()
    drawdowns = (values - peak) / peak
    return float(drawdowns.min())


def ulcer_index(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return the negated Ulcer Index (RMS of percentage drawdown depths).

    The Ulcer Index is the square root of the mean squared percentage
    drawdown from the running peak. It is negated so higher is better.

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        Negated Ulcer Index (<= 0). float('-inf') if history empty.
    """
    if not portfolio_history:
        return float("-inf")
    values = pd.Series([e["value"] for e in portfolio_history], dtype=float)
    peak = values.expanding().max()
    pct_drawdown = (values - peak) / peak  # <= 0
    ui = math.sqrt(float((pct_drawdown ** 2).mean()))
    return -ui


# Partial registry — completed in Task 2
METRICS: dict[str, MetricFn] = {
    "total_return": total_return,
    "cagr":         cagr,
    "max_drawdown": max_drawdown,
    "ulcer_index":  ulcer_index,
}
```

- [ ] **Step 4: Run tests to verify the four metrics pass (registry test will fail — expected)**

```bash
uv run pytest tests/test_metrics.py -v -k "not Registry and not sharpe and not sortino and not calmar and not profit and not win and not expectancy and not recovery"
```

Expected: all TestTotalReturn, TestCAGR, TestMaxDrawdown, TestUlcerIndex tests pass.

- [ ] **Step 5: Commit**

```bash
git add backtest/metrics.py tests/test_metrics.py
git commit -m "feat: add metrics.py with total_return, cagr, max_drawdown, ulcer_index"
```

---

## Task 2: Trade-based metrics + complete `METRICS` registry

**Files:**
- Modify: `backtest/metrics.py`
- Modify: `tests/test_metrics.py`

- [ ] **Step 1: Add the failing tests to `tests/test_metrics.py`**

Append to the end of `tests/test_metrics.py` (after the existing classes):

```python
class TestSharpeRatio:
    def test_positive_returns_positive_sharpe(self):
        # Steady 0.1% daily gain → positive Sharpe
        values = [10000 * (1.001 ** i) for i in range(100)]
        h = _history(values)
        result = sharpe_ratio(h, [])
        assert result > 0

    def test_flat_returns_neginf(self):
        h = _history([10000] * 50)
        assert sharpe_ratio(h, []) == float("-inf")

    def test_empty_history_returns_neginf(self):
        assert sharpe_ratio([], []) == float("-inf")

    def test_single_bar_returns_neginf(self):
        assert sharpe_ratio(_history([10000]), []) == float("-inf")


class TestSortinoRatio:
    def test_only_upside_returns_high_value(self):
        # All positive returns → downside std = 0 → very high sortino
        values = [10000 * (1.001 ** i) for i in range(100)]
        h = _history(values)
        result = sortino_ratio(h, [])
        assert result > 0

    def test_flat_returns_neginf(self):
        h = _history([10000] * 50)
        assert sortino_ratio(h, []) == float("-inf")

    def test_empty_history_returns_neginf(self):
        assert sortino_ratio([], []) == float("-inf")

    def test_greater_than_sharpe_for_positive_returns(self):
        # When all returns are positive, sortino >= sharpe (no downside penalty)
        values = [10000 * (1.001 ** i) for i in range(100)]
        h = _history(values)
        assert sortino_ratio(h, []) >= sharpe_ratio(h, [])


class TestCalmarRatio:
    def test_positive_cagr_with_drawdown(self):
        # Rising trend with a dip
        values = [100, 110, 90, 120, 130]
        h = _history(values)
        result = calmar_ratio(h, [])
        assert result > 0

    def test_no_drawdown_returns_inf(self):
        # Monotonic rise → no drawdown → calmar = inf
        h = _history([100, 110, 120, 130])
        assert calmar_ratio(h, []) == float("inf")

    def test_empty_history_returns_neginf(self):
        assert calmar_ratio([], []) == float("-inf")


class TestProfitFactor:
    def test_all_winners(self):
        t = _trades(100, 200, 50)
        result = profit_factor([], t)
        assert result == float("inf")

    def test_mixed_trades(self):
        t = _trades(100, -50)  # gross win 100, gross loss 50
        assert profit_factor([], t) == pytest.approx(2.0)

    def test_all_losers(self):
        t = _trades(-100, -50)
        assert profit_factor([], t) == pytest.approx(0.0)

    def test_empty_trades_returns_neginf(self):
        assert profit_factor([], []) == float("-inf")


class TestWinRate:
    def test_all_winners(self):
        t = _trades(10, 20, 30)
        assert win_rate([], t) == pytest.approx(1.0)

    def test_mixed(self):
        t = _trades(10, -5, 20, -3)  # 2 wins, 2 losses
        assert win_rate([], t) == pytest.approx(0.5)

    def test_all_losers(self):
        t = _trades(-10, -20)
        assert win_rate([], t) == pytest.approx(0.0)

    def test_empty_trades_returns_neginf(self):
        assert win_rate([], []) == float("-inf")


class TestExpectancy:
    def test_positive_expectancy(self):
        t = _trades(100, 50, -20)  # mean = 130/3
        assert expectancy([], t) == pytest.approx(130 / 3)

    def test_zero_expectancy(self):
        t = _trades(50, -50)
        assert expectancy([], t) == pytest.approx(0.0)

    def test_empty_trades_returns_neginf(self):
        assert expectancy([], []) == float("-inf")


class TestRecoveryFactor:
    def test_positive_return_with_drawdown(self):
        values = [100, 120, 80, 130]
        h = _history(values)
        result = recovery_factor(h, [])
        assert result > 0

    def test_no_drawdown_returns_inf(self):
        h = _history([100, 110, 120])
        assert recovery_factor(h, []) == float("inf")

    def test_empty_history_returns_neginf(self):
        assert recovery_factor([], []) == float("-inf")
```

Also add the missing imports at the top of `test_metrics.py` (after the existing imports):

```python
from backtest.metrics import (
    METRICS,
    cagr,
    calmar_ratio,
    expectancy,
    max_drawdown,
    profit_factor,
    recovery_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    ulcer_index,
    win_rate,
)
```

Replace the existing `from backtest.metrics import (...)` block at the top of the file with the block above.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_metrics.py -v 2>&1 | tail -10
```

Expected: `ImportError` for the new metric names.

- [ ] **Step 3: Add the seven trade-based metrics and complete the registry in `backtest/metrics.py`**

Add after the `ulcer_index` function and before the `METRICS` dict:

```python
def sharpe_ratio(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return annualised Sharpe ratio (mean daily return / std daily return).

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        Annualised Sharpe ratio. float('-inf') if fewer than 2 bars or std = 0.
    """
    if len(portfolio_history) < 2:
        return float("-inf")
    values = pd.Series([e["value"] for e in portfolio_history], dtype=float)
    daily_returns = values.pct_change().dropna()
    std = float(daily_returns.std())
    if std == 0:
        return float("-inf")
    return float(daily_returns.mean() / std * math.sqrt(TRADING_DAYS))


def sortino_ratio(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return annualised Sortino ratio (mean daily return / downside deviation).

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        Annualised Sortino ratio. float('-inf') if fewer than 2 bars or
        no negative returns (downside std = 0).
    """
    if len(portfolio_history) < 2:
        return float("-inf")
    values = pd.Series([e["value"] for e in portfolio_history], dtype=float)
    daily_returns = values.pct_change().dropna()
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) == 0:
        # All returns positive — excellent; return a large finite value
        mean_return = float(daily_returns.mean())
        if mean_return <= 0:
            return float("-inf")
        return mean_return * math.sqrt(TRADING_DAYS) * 1e6
    downside_std = float(negative_returns.std())
    if downside_std == 0:
        return float("-inf")
    return float(daily_returns.mean() / downside_std * math.sqrt(TRADING_DAYS))


def calmar_ratio(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return CAGR / abs(max_drawdown).

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        Calmar ratio. float('inf') if no drawdown. float('-inf') if insufficient data.
    """
    if len(portfolio_history) < 2:
        return float("-inf")
    c = cagr(portfolio_history, trades)
    if c == float("-inf"):
        return float("-inf")
    dd = max_drawdown(portfolio_history, trades)
    if dd == 0.0:
        return float("inf")
    return c / abs(dd)


def profit_factor(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return gross winning P&L / gross losing P&L.

    Args:
        portfolio_history: Unused; present for interface consistency.
        trades: List of trade dicts with 'pnl' key.

    Returns:
        Profit factor. float('inf') if all trades win. 0.0 if all lose.
        float('-inf') if no trades.
    """
    if not trades:
        return float("-inf")
    gross_win = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = sum(abs(t["pnl"]) for t in trades if t["pnl"] < 0)
    if gross_loss == 0:
        return float("inf")
    return gross_win / gross_loss


def win_rate(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return fraction of trades that are profitable.

    Args:
        portfolio_history: Unused; present for interface consistency.
        trades: List of trade dicts with 'pnl' key.

    Returns:
        Win rate as a fraction (0.0 to 1.0). float('-inf') if no trades.
    """
    if not trades:
        return float("-inf")
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return wins / len(trades)


def expectancy(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return mean P&L per trade.

    Args:
        portfolio_history: Unused; present for interface consistency.
        trades: List of trade dicts with 'pnl' key.

    Returns:
        Mean P&L per trade in currency units. float('-inf') if no trades.
    """
    if not trades:
        return float("-inf")
    return sum(t["pnl"] for t in trades) / len(trades)


def recovery_factor(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return total_return / abs(max_drawdown).

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        Recovery factor. float('inf') if no drawdown. float('-inf') if insufficient data.
    """
    if not portfolio_history:
        return float("-inf")
    tr = total_return(portfolio_history, trades)
    if tr == float("-inf"):
        return float("-inf")
    dd = max_drawdown(portfolio_history, trades)
    if dd == 0.0:
        return float("inf")
    return tr / abs(dd)
```

Replace the partial `METRICS` dict at the bottom of the file with:

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

- [ ] **Step 4: Run all metrics tests**

```bash
uv run pytest tests/test_metrics.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run the full suite to check nothing regressed**

```bash
uv run pytest -q
```

Expected: all existing tests plus new metrics tests pass.

- [ ] **Step 6: Commit**

```bash
git add backtest/metrics.py tests/test_metrics.py
git commit -m "feat: complete metrics.py with all 10 objective functions and METRICS registry"
```

---

## Task 3: Refactor `runner._calculate_metrics` to delegate to `metrics.py`

**Files:**
- Modify: `backtest/runner.py`

The runner's method keeps returning the same four keys with the same values. The difference is internal: it now calls `metrics.py` functions for `total_return`, `win_rate`, and `max_drawdown`. The runner multiplies by `PERCENT_MULTIPLIER` (100) to preserve backward compatibility with existing tests that expect percentages.

- [ ] **Step 1: Run existing runner tests to establish baseline**

```bash
uv run pytest tests/test_runner.py -v -k "calculate_metrics"
```

Expected: all `_calculate_metrics` tests pass with the current implementation.

- [ ] **Step 2: Update the import block in `runner.py`**

At the top of `backtest/runner.py`, add the metrics import after the existing imports:

```python
from . import metrics as _metrics
```

The full import section should look like:

```python
import pandas as pd
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from .portfolio import Portfolio
from .validation import (
    validate_dataframe, validate_price_data, validate_positive_number,
    sanitize_file_path, ValidationError
)
from .config import get_backtest_config, get_portfolio_config, FileConfig
from .constants import TradingConstants, ValidationLimits
from . import metrics as _metrics
```

- [ ] **Step 3: Replace the `_calculate_metrics` method body**

Find `_calculate_metrics` (currently lines 81–135 in `runner.py`) and replace the entire method with:

```python
    def _calculate_metrics(self, portfolio_history: List[Dict], trades: List[Dict]) -> Dict[str, Any]:
        """
        Calculate performance metrics from portfolio history and trades.

        Args:
            portfolio_history: List of portfolio value records with 'date' and 'value' keys
            trades: List of trade records with 'pnl' key

        Returns:
            Dictionary with keys: total_return, win_rate, num_trades, max_drawdown.
            Values for total_return, win_rate, and max_drawdown are percentages
            (multiplied by 100) for backward compatibility.

        Raises:
            ValidationError: If portfolio_history contains an invalid start value
        """
        if not portfolio_history:
            return {
                "total_return": 0.0,
                "win_rate": 0.0,
                "num_trades": 0,
                "max_drawdown": 0.0,
            }

        start_value = portfolio_history[0]["value"]
        if start_value <= 0:
            raise ValidationError(f"Invalid start value: {start_value}")

        pct = TradingConstants.PERCENT_MULTIPLIER
        return {
            "total_return": _metrics.total_return(portfolio_history, trades) * pct,
            "win_rate":     _metrics.win_rate(portfolio_history, trades) * pct,
            "num_trades":   len(trades),
            "max_drawdown": _metrics.max_drawdown(portfolio_history, trades) * pct,
        }
```

Note: `win_rate` from `metrics.py` returns `float('-inf')` for empty trades, which multiplied by 100 is still `float('-inf')`. But the runner checks `if not portfolio_history` first and returns 0.0 win_rate for empty history. When portfolio_history is non-empty but trades is empty, `win_rate * pct = float('-inf') * 100 = float('-inf')`. The runner test `test_calculate_metrics_no_trades` tests this case — check what it asserts and patch if needed.

- [ ] **Step 4: Check what the no-trades test asserts**

```bash
grep -A 10 "no_trades\|num_trades.*0" tests/test_runner.py | head -30
```

If a test asserts `metrics['win_rate'] == 0.0` for a non-empty portfolio with no trades, patch the `_calculate_metrics` method to handle that case:

```python
        win_rate_raw = _metrics.win_rate(portfolio_history, trades)
        return {
            "total_return": _metrics.total_return(portfolio_history, trades) * pct,
            "win_rate":     0.0 if win_rate_raw == float("-inf") else win_rate_raw * pct,
            "num_trades":   len(trades),
            "max_drawdown": _metrics.max_drawdown(portfolio_history, trades) * pct,
        }
```

- [ ] **Step 5: Run all runner tests**

```bash
uv run pytest tests/test_runner.py -v
```

Expected: all pass. If any fail, read the assertion and adjust the `_calculate_metrics` body — do not change the tests.

- [ ] **Step 6: Run the full suite**

```bash
uv run pytest -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add backtest/runner.py
git commit -m "refactor: runner._calculate_metrics delegates to metrics.py

Keeps same four return keys and percentage scale. No behaviour change
for existing callers or tests."
```

---

## Task 4: `GridSearch` and `RandomSearch`

**Files:**
- Create: `backtest/optimization.py`
- Create: `tests/test_optimization.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_optimization.py`:

```python
import pytest

from backtest.optimization import GridSearch, RandomSearch


class TestGridSearch:
    def test_single_param(self):
        gs = GridSearch()
        result = gs.generate({"a": [1, 2, 3]})
        assert result == [{"a": 1}, {"a": 2}, {"a": 3}]

    def test_two_params_cartesian_product(self):
        gs = GridSearch()
        result = gs.generate({"a": [1, 2], "b": [10, 20]})
        assert len(result) == 4
        assert {"a": 1, "b": 10} in result
        assert {"a": 1, "b": 20} in result
        assert {"a": 2, "b": 10} in result
        assert {"a": 2, "b": 20} in result

    def test_size_is_product_of_lengths(self):
        gs = GridSearch()
        space = {"a": [1, 2, 3], "b": [10, 20], "c": [100, 200, 300, 400]}
        result = gs.generate(space)
        assert len(result) == 3 * 2 * 4

    def test_single_value_params_returns_one_combination(self):
        gs = GridSearch()
        result = gs.generate({"a": [5], "b": [10]})
        assert result == [{"a": 5, "b": 10}]

    def test_returns_list_of_dicts(self):
        gs = GridSearch()
        result = gs.generate({"x": [1, 2]})
        assert all(isinstance(r, dict) for r in result)


class TestRandomSearch:
    def test_returns_n_combinations(self):
        rs = RandomSearch(n=5, seed=42)
        result = rs.generate({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        assert len(result) == 5

    def test_reproducible_with_same_seed(self):
        space = {"a": list(range(20)), "b": list(range(20))}
        r1 = RandomSearch(n=10, seed=42).generate(space)
        r2 = RandomSearch(n=10, seed=42).generate(space)
        assert r1 == r2

    def test_different_seeds_give_different_results(self):
        space = {"a": list(range(20)), "b": list(range(20))}
        r1 = RandomSearch(n=10, seed=1).generate(space)
        r2 = RandomSearch(n=10, seed=2).generate(space)
        assert r1 != r2

    def test_n_larger_than_space_returns_all_with_replacement(self):
        rs = RandomSearch(n=20, seed=42)
        result = rs.generate({"a": [1, 2, 3]})
        assert len(result) == 20

    def test_returns_list_of_dicts(self):
        rs = RandomSearch(n=3, seed=0)
        result = rs.generate({"x": [1, 2, 3, 4, 5]})
        assert all(isinstance(r, dict) for r in result)

    def test_each_combination_is_valid(self):
        space = {"a": [1, 2], "b": [10, 20]}
        rs = RandomSearch(n=10, seed=99)
        for combo in rs.generate(space):
            assert combo["a"] in [1, 2]
            assert combo["b"] in [10, 20]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_optimization.py -v 2>&1 | head -10
```

Expected: `ImportError`.

- [ ] **Step 3: Create `backtest/optimization.py` with search classes**

```python
"""Walk-Forward Analysis: search strategies, window splitting, and optimizer.

Public API:
    GridSearch          — exhaustive Cartesian product search
    RandomSearch        — random sampling of parameter space
    WalkForwardOptimizer — orchestrates WFA across sliding/anchored windows
    WalkForwardResult   — dataclass holding all WFA output
"""

import itertools
import random
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from .metrics import METRICS, MetricFn
from .strategy import BaseStrategy
from .validation import ValidationError


class GridSearch:
    """Exhaustive search over all combinations of a parameter space.

    Args:
        None

    Example:
        gs = GridSearch()
        combos = gs.generate({"a": [1, 2], "b": [10, 20]})
        # → [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20}]
    """

    def generate(self, param_space: dict[str, list]) -> list[dict]:
        """Return all combinations of param_space values.

        Args:
            param_space: Mapping of parameter names to lists of candidate values.

        Returns:
            List of dicts, one per combination.
        """
        keys = list(param_space.keys())
        values = list(param_space.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


class RandomSearch:
    """Random sampling of parameter combinations.

    Samples n combinations uniformly at random. If n exceeds the total number
    of unique combinations, samples with replacement.

    Args:
        n: Number of combinations to sample.
        seed: Random seed for reproducibility.

    Example:
        rs = RandomSearch(n=50, seed=42)
        combos = rs.generate({"period": [7, 14, 21], "threshold": [0.05, 0.10]})
    """

    def __init__(self, n: int, seed: int = 42):
        """
        Args:
            n: Number of combinations to sample.
            seed: Random seed for reproducibility.
        """
        self.n = n
        self.seed = seed

    def generate(self, param_space: dict[str, list]) -> list[dict]:
        """Return n randomly sampled combinations from param_space.

        Args:
            param_space: Mapping of parameter names to lists of candidate values.

        Returns:
            List of n dicts, each a sampled combination.
        """
        all_combos = GridSearch().generate(param_space)
        rng = random.Random(self.seed)
        if self.n >= len(all_combos):
            return rng.choices(all_combos, k=self.n)
        return rng.sample(all_combos, k=self.n)
```

- [ ] **Step 4: Run search tests**

```bash
uv run pytest tests/test_optimization.py -v
```

Expected: all pass.

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add backtest/optimization.py tests/test_optimization.py
git commit -m "feat: add GridSearch and RandomSearch to optimization.py"
```

---

## Task 5: Window splitting (`_generate_windows`)

**Files:**
- Modify: `backtest/optimization.py`
- Modify: `tests/test_optimization.py`

- [ ] **Step 1: Add failing tests to `tests/test_optimization.py`**

Append after the `TestRandomSearch` class:

```python
class TestGenerateWindows:
    """Tests for the _generate_windows module-level function."""

    def _make_data(self, n):
        """Return a DataFrame with n rows and a DatetimeIndex."""
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": range(1, n + 1)}, index=dates)

    # --- sliding ---

    def test_sliding_window_count(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(100)
        windows = _generate_windows(data, train_size=60, test_size=20, window_type="sliding")
        # First window: train 0-59, test 60-79
        # Second window: train 20-79, test 80-99
        assert len(windows) == 2

    def test_sliding_train_size_is_fixed(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(200)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="sliding")
        for train, test in windows:
            assert len(train) == 100

    def test_sliding_test_windows_contiguous(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(250)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="sliding")
        for i in range(len(windows) - 1):
            _, test_i = windows[i]
            _, test_next = windows[i + 1]
            assert test_i.index[-1] < test_next.index[0]

    def test_sliding_no_overlap_between_train_and_test(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(200)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="sliding")
        for train, test in windows:
            assert len(set(train.index) & set(test.index)) == 0

    def test_sliding_covers_full_range(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(200)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="sliding")
        all_test_dates = set()
        for _, test in windows:
            all_test_dates.update(test.index.tolist())
        expected = set(data.index[100:])  # first train_size bars are never in test
        assert all_test_dates == expected

    # --- anchored ---

    def test_anchored_train_always_starts_at_index_zero(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(200)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="anchored")
        for train, _ in windows:
            assert train.index[0] == data.index[0]

    def test_anchored_train_grows(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(300)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="anchored")
        train_sizes = [len(train) for train, _ in windows]
        for i in range(len(train_sizes) - 1):
            assert train_sizes[i + 1] > train_sizes[i]

    def test_anchored_test_windows_contiguous(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(300)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="anchored")
        for i in range(len(windows) - 1):
            _, test_i = windows[i]
            _, test_next = windows[i + 1]
            assert test_i.index[-1] < test_next.index[0]

    def test_anchored_no_overlap_between_train_and_test(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(300)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="anchored")
        for train, test in windows:
            assert len(set(train.index) & set(test.index)) == 0

    def test_invalid_window_type_raises(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(200)
        with pytest.raises(ValidationError):
            _generate_windows(data, train_size=100, test_size=50, window_type="invalid")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_optimization.py::TestGenerateWindows -v 2>&1 | head -10
```

Expected: `ImportError` for `_generate_windows`.

- [ ] **Step 3: Add `_generate_windows` to `backtest/optimization.py`**

Add after the `RandomSearch` class:

```python
def _generate_windows(
    data: pd.DataFrame,
    train_size: int,
    test_size: int,
    window_type: str,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split data into (train, test) window pairs.

    Args:
        data: Full dataset with DatetimeIndex.
        train_size: Number of bars in each training window (sliding mode) or
                    in the initial training window (anchored mode).
        test_size: Number of bars in each test window.
        window_type: "sliding" — fixed train size, both windows advance by test_size.
                     "anchored" — train always starts at index 0, expands each step.

    Returns:
        List of (train_df, test_df) tuples in chronological order.

    Raises:
        ValidationError: If window_type is not "sliding" or "anchored".
    """
    if window_type not in ("sliding", "anchored"):
        raise ValidationError(
            f"window_type must be 'sliding' or 'anchored', got '{window_type}'"
        )

    n = len(data)
    windows = []
    test_start = train_size

    while test_start + test_size <= n:
        test_end = test_start + test_size
        test_df = data.iloc[test_start:test_end]

        if window_type == "sliding":
            train_df = data.iloc[test_start - train_size:test_start]
        else:  # anchored
            train_df = data.iloc[0:test_start]

        windows.append((train_df, test_df))
        test_start += test_size

    return windows
```

- [ ] **Step 4: Run window tests**

```bash
uv run pytest tests/test_optimization.py::TestGenerateWindows -v
```

Expected: all pass.

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add backtest/optimization.py tests/test_optimization.py
git commit -m "feat: add _generate_windows with sliding and anchored modes"
```

---

## Task 6: Warmup filtering (`_filter_by_warmup`)

**Files:**
- Modify: `backtest/optimization.py`
- Modify: `tests/test_optimization.py`

- [ ] **Step 1: Add failing tests to `tests/test_optimization.py`**

Append after `TestGenerateWindows`:

```python
class TestFilterByWarmup:
    """Tests for the _filter_by_warmup module-level function."""

    def _make_history(self, dates):
        return [{"date": d, "value": 10000.0 + i * 10} for i, d in enumerate(dates)]

    def _make_trades(self, exit_dates):
        return [
            {
                "entry_date": d - pd.Timedelta(days=1),
                "exit_date": d,
                "entry": 100.0,
                "exit": 110.0,
                "shares": 1,
                "pnl": 10.0,
            }
            for d in exit_dates
        ]

    def test_removes_history_before_cutoff(self):
        from backtest.optimization import _filter_by_warmup
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        history = self._make_history(dates)
        cutoff = dates[3]
        filtered_h, _ = _filter_by_warmup(history, [], cutoff)
        assert all(e["date"] >= cutoff for e in filtered_h)
        assert len(filtered_h) == 7

    def test_removes_trades_before_cutoff(self):
        from backtest.optimization import _filter_by_warmup
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        trades = self._make_trades(dates)
        cutoff = dates[5]
        _, filtered_t = _filter_by_warmup([], trades, cutoff)
        assert all(t["exit_date"] >= cutoff for t in filtered_t)
        assert len(filtered_t) == 5

    def test_empty_inputs_return_empty(self):
        from backtest.optimization import _filter_by_warmup
        cutoff = pd.Timestamp("2020-01-10")
        h, t = _filter_by_warmup([], [], cutoff)
        assert h == []
        assert t == []

    def test_cutoff_at_start_returns_all(self):
        from backtest.optimization import _filter_by_warmup
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        history = self._make_history(dates)
        trades = self._make_trades(dates)
        cutoff = dates[0]
        h, t = _filter_by_warmup(history, trades, cutoff)
        assert len(h) == 5
        assert len(t) == 5

    def test_cutoff_after_all_returns_empty(self):
        from backtest.optimization import _filter_by_warmup
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        history = self._make_history(dates)
        trades = self._make_trades(dates)
        cutoff = dates[-1] + pd.Timedelta(days=1)
        h, t = _filter_by_warmup(history, trades, cutoff)
        assert h == []
        assert t == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_optimization.py::TestFilterByWarmup -v 2>&1 | head -10
```

Expected: `ImportError`.

- [ ] **Step 3: Add `_filter_by_warmup` to `backtest/optimization.py`**

Add after `_generate_windows`:

```python
def _filter_by_warmup(
    portfolio_history: list[dict],
    trades: list[dict],
    cutoff_date: pd.Timestamp,
) -> tuple[list[dict], list[dict]]:
    """Remove portfolio history entries and trades before the cutoff date.

    This is used to exclude the warmup period from metric scoring after
    BacktestRunnerImpl has run on the full window slice.

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: List of trade dicts with 'exit_date' key.
        cutoff_date: First date to INCLUDE in the filtered output.

    Returns:
        Tuple of (filtered_portfolio_history, filtered_trades).
    """
    filtered_history = [e for e in portfolio_history if e["date"] >= cutoff_date]
    filtered_trades = [t for t in trades if t["exit_date"] >= cutoff_date]
    return filtered_history, filtered_trades
```

- [ ] **Step 4: Run warmup tests**

```bash
uv run pytest tests/test_optimization.py::TestFilterByWarmup -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backtest/optimization.py tests/test_optimization.py
git commit -m "feat: add _filter_by_warmup for post-run warmup exclusion"
```

---

## Task 7: `WalkForwardOptimizer.__init__` and validation

**Files:**
- Modify: `backtest/optimization.py`
- Modify: `tests/test_optimization.py`

- [ ] **Step 1: Add failing tests to `tests/test_optimization.py`**

Append after `TestFilterByWarmup`:

```python
class TestWalkForwardOptimizerInit:
    """Tests for WalkForwardOptimizer construction and validation."""

    def _make_data(self, n=300):
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": range(1, n + 1)}, index=dates)

    def _valid_kwargs(self, data):
        from backtest.strategy import ConsecutiveDaysStrategy
        return dict(
            strategy_class=ConsecutiveDaysStrategy,
            param_space={"consecutive_days": [1, 2, 3]},
            data=data,
            train_size=100,
            test_size=50,
            window_type="sliding",
        )

    def test_valid_construction_does_not_raise(self):
        from backtest.optimization import WalkForwardOptimizer
        data = self._make_data()
        WalkForwardOptimizer(**self._valid_kwargs(data))

    def test_invalid_train_size_raises(self):
        from backtest.optimization import WalkForwardOptimizer
        data = self._make_data()
        kwargs = self._valid_kwargs(data)
        kwargs["train_size"] = 0
        with pytest.raises(ValidationError, match="train_size"):
            WalkForwardOptimizer(**kwargs)

    def test_invalid_test_size_raises(self):
        from backtest.optimization import WalkForwardOptimizer
        data = self._make_data()
        kwargs = self._valid_kwargs(data)
        kwargs["test_size"] = 0
        with pytest.raises(ValidationError, match="test_size"):
            WalkForwardOptimizer(**kwargs)

    def test_train_plus_test_exceeds_data_raises(self):
        from backtest.optimization import WalkForwardOptimizer
        data = self._make_data(100)
        kwargs = self._valid_kwargs(data)
        kwargs["train_size"] = 80
        kwargs["test_size"] = 30  # 80 + 30 > 100
        with pytest.raises(ValidationError):
            WalkForwardOptimizer(**kwargs)

    def test_invalid_window_type_raises(self):
        from backtest.optimization import WalkForwardOptimizer
        data = self._make_data()
        kwargs = self._valid_kwargs(data)
        kwargs["window_type"] = "rolling"
        with pytest.raises(ValidationError, match="window_type"):
            WalkForwardOptimizer(**kwargs)

    def test_non_strategy_class_raises(self):
        from backtest.optimization import WalkForwardOptimizer
        data = self._make_data()
        kwargs = self._valid_kwargs(data)
        kwargs["strategy_class"] = list  # not a BaseStrategy subclass
        with pytest.raises(ValidationError, match="strategy_class"):
            WalkForwardOptimizer(**kwargs)

    def test_invalid_objective_string_raises(self):
        from backtest.optimization import WalkForwardOptimizer
        data = self._make_data()
        kwargs = self._valid_kwargs(data)
        kwargs["objective"] = "not_a_metric"
        with pytest.raises(ValidationError, match="objective"):
            WalkForwardOptimizer(**kwargs)

    def test_callable_objective_accepted(self):
        from backtest.optimization import WalkForwardOptimizer
        data = self._make_data()
        kwargs = self._valid_kwargs(data)
        kwargs["objective"] = lambda ph, t: 0.0
        WalkForwardOptimizer(**kwargs)  # should not raise

    def test_string_objective_accepted(self):
        from backtest.optimization import WalkForwardOptimizer
        data = self._make_data()
        kwargs = self._valid_kwargs(data)
        kwargs["objective"] = "calmar_ratio"
        WalkForwardOptimizer(**kwargs)  # should not raise

    def test_default_objective_is_sharpe(self):
        from backtest.optimization import WalkForwardOptimizer
        data = self._make_data()
        opt = WalkForwardOptimizer(**self._valid_kwargs(data))
        from backtest.metrics import sharpe_ratio
        assert opt._objective_fn is sharpe_ratio

    def test_default_searcher_is_grid_search(self):
        from backtest.optimization import GridSearch, WalkForwardOptimizer
        data = self._make_data()
        opt = WalkForwardOptimizer(**self._valid_kwargs(data))
        assert isinstance(opt.searcher, GridSearch)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_optimization.py::TestWalkForwardOptimizerInit -v 2>&1 | head -10
```

Expected: `ImportError`.

- [ ] **Step 3: Add `WalkForwardOptimizer.__init__` to `backtest/optimization.py`**

Add after `_filter_by_warmup`:

```python
class WalkForwardOptimizer:
    """Optimise strategy parameters using Walk-Forward Analysis.

    For each train window, exhaustively or randomly searches the parameter
    space and selects the best-scoring parameter set. Applies those
    parameters to the subsequent out-of-sample test window. Stitches
    test-window equity curves into a composite result.

    Args:
        strategy_class: A BaseStrategy subclass (not an instance).
        param_space: Dict mapping parameter names to lists of candidate values.
        data: Full historical DataFrame with DatetimeIndex and 'Close' column.
        train_size: Number of bars in each training window.
        test_size: Number of bars in each test window.
        window_type: "sliding" or "anchored".
        searcher: GridSearch() or RandomSearch(n, seed). Default: GridSearch().
        objective: Metric name string (key of METRICS) or MetricFn callable.
                   Default: "sharpe_ratio".
        min_trades: Minimum trades required to score a window. Windows with
                    fewer trades receive float('-inf') as their score.
                    Default: 5.

    Example:
        opt = WalkForwardOptimizer(
            strategy_class=RSIStrategy,
            param_space={"period": [7, 14, 21]},
            data=df,
            train_size=252,
            test_size=63,
            window_type="sliding",
            objective="sharpe_ratio",
        )
        result = opt.run()
    """

    def __init__(
        self,
        strategy_class: type,
        param_space: dict[str, list],
        data: pd.DataFrame,
        train_size: int,
        test_size: int,
        window_type: str = "sliding",
        searcher: Union[GridSearch, RandomSearch, None] = None,
        objective: Union[str, MetricFn] = "sharpe_ratio",
        min_trades: int = 5,
    ):
        """
        Args:
            strategy_class: A BaseStrategy subclass (not an instance).
            param_space: Dict mapping parameter names to lists of candidate values.
            data: Full historical DataFrame with DatetimeIndex and 'Close' column.
            train_size: Number of bars in each training window.
            test_size: Number of bars in each test window.
            window_type: "sliding" or "anchored".
            searcher: Search strategy. Default: GridSearch().
            objective: Metric to optimise. Default: "sharpe_ratio".
            min_trades: Minimum trades to score a window. Default: 5.

        Raises:
            ValidationError: If any argument is invalid.
        """
        if train_size < 1:
            raise ValidationError(f"train_size must be >= 1, got {train_size}")
        if test_size < 1:
            raise ValidationError(f"test_size must be >= 1, got {test_size}")
        if train_size + test_size > len(data):
            raise ValidationError(
                f"train_size ({train_size}) + test_size ({test_size}) = "
                f"{train_size + test_size} exceeds data length ({len(data)})"
            )
        if window_type not in ("sliding", "anchored"):
            raise ValidationError(
                f"window_type must be 'sliding' or 'anchored', got '{window_type}'"
            )
        if not (isinstance(strategy_class, type) and issubclass(strategy_class, BaseStrategy)):
            raise ValidationError(
                f"strategy_class must be a subclass of BaseStrategy, got {strategy_class}"
            )
        if isinstance(objective, str):
            if objective not in METRICS:
                raise ValidationError(
                    f"objective '{objective}' not in METRICS. "
                    f"Valid keys: {sorted(METRICS.keys())}"
                )
            objective_fn = METRICS[objective]
        elif callable(objective):
            objective_fn = objective
        else:
            raise ValidationError("objective must be a metric name string or callable")

        self.strategy_class = strategy_class
        self.param_space = param_space
        self.data = data
        self.train_size = train_size
        self.test_size = test_size
        self.window_type = window_type
        self.searcher = searcher if searcher is not None else GridSearch()
        self._objective_fn = objective_fn
        self.min_trades = min_trades
```

- [ ] **Step 4: Run init tests**

```bash
uv run pytest tests/test_optimization.py::TestWalkForwardOptimizerInit -v
```

Expected: all pass.

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add backtest/optimization.py tests/test_optimization.py
git commit -m "feat: add WalkForwardOptimizer.__init__ with full validation"
```

---

## Task 8: Per-window training (`_run_train_window`)

**Files:**
- Modify: `backtest/optimization.py`
- Modify: `tests/test_optimization.py`

- [ ] **Step 1: Add failing tests to `tests/test_optimization.py`**

Append after `TestWalkForwardOptimizerInit`:

```python
class TestRunTrainWindow:
    """Tests for WalkForwardOptimizer._run_train_window."""

    def _make_data(self, n=200):
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        prices = [100.0 + i * 0.5 for i in range(n)]
        return pd.DataFrame({"Close": prices}, index=dates)

    def _make_optimizer(self, data):
        from backtest.optimization import WalkForwardOptimizer
        from backtest.strategy import ConsecutiveDaysStrategy
        return WalkForwardOptimizer(
            strategy_class=ConsecutiveDaysStrategy,
            param_space={"consecutive_days": [1, 2, 3]},
            data=data,
            train_size=100,
            test_size=50,
            window_type="sliding",
            min_trades=0,  # don't skip windows with few trades
        )

    def test_returns_dict_with_best_params_key(self):
        data = self._make_data()
        opt = self._make_optimizer(data)
        train_data = data.iloc[:100]
        result = opt._run_train_window(train_data)
        assert "best_params" in result

    def test_best_params_keys_match_param_space(self):
        data = self._make_data()
        opt = self._make_optimizer(data)
        train_data = data.iloc[:100]
        result = opt._run_train_window(train_data)
        assert set(result["best_params"].keys()) == {"consecutive_days"}

    def test_best_params_value_is_from_param_space(self):
        data = self._make_data()
        opt = self._make_optimizer(data)
        train_data = data.iloc[:100]
        result = opt._run_train_window(train_data)
        assert result["best_params"]["consecutive_days"] in [1, 2, 3]

    def test_returns_objective_score(self):
        data = self._make_data()
        opt = self._make_optimizer(data)
        train_data = data.iloc[:100]
        result = opt._run_train_window(train_data)
        assert "objective_score" in result
        assert isinstance(result["objective_score"], float)

    def test_all_min_trades_filtered_returns_first_params(self):
        """When every candidate has too few trades, first params set is returned."""
        data = self._make_data()
        from backtest.optimization import WalkForwardOptimizer
        from backtest.strategy import ConsecutiveDaysStrategy
        opt = WalkForwardOptimizer(
            strategy_class=ConsecutiveDaysStrategy,
            param_space={"consecutive_days": [1, 2, 3]},
            data=data,
            train_size=100,
            test_size=50,
            window_type="sliding",
            min_trades=9999,  # impossible threshold
        )
        train_data = data.iloc[:100]
        result = opt._run_train_window(train_data)
        # Should not raise — falls back to first candidate
        assert "best_params" in result
        assert result["objective_score"] == float("-inf")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_optimization.py::TestRunTrainWindow -v 2>&1 | head -10
```

Expected: `AttributeError` — method does not exist.

- [ ] **Step 3: Add `_run_train_window` to `WalkForwardOptimizer`**

Add this method inside the `WalkForwardOptimizer` class (after `__init__`):

```python
    def _run_train_window(self, train_data: pd.DataFrame) -> dict:
        """Find the best-scoring parameter set for a training window.

        Runs BacktestRunnerImpl for each candidate parameter set, applies
        warmup filtering, and scores with the objective function. Returns
        the parameter set with the highest score.

        Args:
            train_data: Sliced training data with DatetimeIndex.

        Returns:
            Dict with keys "best_params" (dict) and "objective_score" (float).
        """
        from .runner import BacktestRunnerImpl

        candidates = self.searcher.generate(self.param_space)
        best_params = candidates[0]
        best_score = float("-inf")

        for params in candidates:
            strategy = self.strategy_class(**params)
            warmup_n = strategy.warmup_period
            runner = BacktestRunnerImpl(strategy=strategy, benchmarks=[])

            try:
                result = runner.run(data=train_data, start_capital=None)
            except Exception:
                continue

            portfolio_history = [
                {"date": d, "value": v}
                for d, v in result["strategy_returns"].items()
            ]
            trades = result["trades"]

            # Apply warmup buffer
            if warmup_n > 0 and len(train_data) > warmup_n:
                cutoff = train_data.index[warmup_n]
                portfolio_history, trades = _filter_by_warmup(
                    portfolio_history, trades, cutoff
                )

            # Apply min_trades guard
            if len(trades) < self.min_trades:
                score = float("-inf")
            else:
                score = self._objective_fn(portfolio_history, trades)

            if score > best_score:
                best_score = score
                best_params = params

        return {"best_params": best_params, "objective_score": best_score}
```

- [ ] **Step 4: Run train window tests**

```bash
uv run pytest tests/test_optimization.py::TestRunTrainWindow -v
```

Expected: all pass.

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add backtest/optimization.py tests/test_optimization.py
git commit -m "feat: add WalkForwardOptimizer._run_train_window with warmup filtering"
```

---

## Task 9: `WalkForwardResult` + `run()` method

**Files:**
- Modify: `backtest/optimization.py`
- Modify: `tests/test_optimization.py`

- [ ] **Step 1: Add failing tests to `tests/test_optimization.py`**

Append after `TestRunTrainWindow`:

```python
class TestWalkForwardResult:
    """Tests for WalkForwardOptimizer.run() and WalkForwardResult."""

    def _make_oscillating_data(self, n=500):
        """Create data with 3-day up/down oscillations."""
        import math
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        prices = [100.0 + 10 * math.sin(i * math.pi / 3) for i in range(n)]
        return pd.DataFrame({"Close": prices}, index=dates)

    def _make_optimizer(self, data=None, min_trades=0):
        from backtest.optimization import WalkForwardOptimizer
        from backtest.strategy import ConsecutiveDaysStrategy
        if data is None:
            data = self._make_oscillating_data()
        return WalkForwardOptimizer(
            strategy_class=ConsecutiveDaysStrategy,
            param_space={"consecutive_days": [1, 2, 3]},
            data=data,
            train_size=150,
            test_size=50,
            window_type="sliding",
            min_trades=min_trades,
        )

    def test_run_returns_walk_forward_result(self):
        from backtest.optimization import WalkForwardResult
        opt = self._make_optimizer()
        result = opt.run()
        assert isinstance(result, WalkForwardResult)

    def test_equity_curve_is_series(self):
        opt = self._make_optimizer()
        result = opt.run()
        assert isinstance(result.equity_curve, pd.Series)

    def test_equity_curve_has_datetime_index(self):
        opt = self._make_optimizer()
        result = opt.run()
        assert isinstance(result.equity_curve.index, pd.DatetimeIndex)

    def test_windows_is_dataframe(self):
        opt = self._make_optimizer()
        result = opt.run()
        assert isinstance(result.windows, pd.DataFrame)

    def test_windows_has_correct_columns(self):
        opt = self._make_optimizer()
        result = opt.run()
        expected_cols = {
            "train_start", "train_end", "test_start", "test_end",
            "best_params", "objective_score", "n_trades",
            "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "max_drawdown", "ulcer_index",
            "profit_factor", "win_rate", "expectancy", "recovery_factor",
        }
        assert expected_cols.issubset(set(result.windows.columns))

    def test_windows_row_count_matches_expected(self):
        from backtest.optimization import _generate_windows
        data = self._make_oscillating_data()
        opt = self._make_optimizer(data=data)
        result = opt.run()
        expected_windows = _generate_windows(data, 150, 50, "sliding")
        assert len(result.windows) == len(expected_windows)

    def test_summary_is_dict(self):
        opt = self._make_optimizer()
        result = opt.run()
        assert isinstance(result.summary, dict)

    def test_summary_has_all_metric_keys(self):
        opt = self._make_optimizer()
        result = opt.run()
        for key in ["total_return", "cagr", "sharpe_ratio", "sortino_ratio",
                    "calmar_ratio", "max_drawdown", "ulcer_index",
                    "profit_factor", "win_rate", "expectancy", "recovery_factor"]:
            assert key in result.summary, f"Missing summary key: {key}"

    def test_summary_has_meta_keys(self):
        opt = self._make_optimizer()
        result = opt.run()
        assert "n_windows" in result.summary
        assert "n_windows_with_trades" in result.summary
        assert "param_stability" in result.summary

    def test_best_params_overall_is_dict(self):
        opt = self._make_optimizer()
        result = opt.run()
        assert isinstance(result.best_params_overall, dict)

    def test_best_params_overall_keys_match_param_space(self):
        opt = self._make_optimizer()
        result = opt.run()
        assert set(result.best_params_overall.keys()) == {"consecutive_days"}

    def test_anchored_window_type_works(self):
        from backtest.optimization import WalkForwardOptimizer
        from backtest.strategy import ConsecutiveDaysStrategy
        data = self._make_oscillating_data()
        opt = WalkForwardOptimizer(
            strategy_class=ConsecutiveDaysStrategy,
            param_space={"consecutive_days": [1, 2]},
            data=data,
            train_size=150,
            test_size=50,
            window_type="anchored",
            min_trades=0,
        )
        result = opt.run()
        assert isinstance(result.equity_curve, pd.Series)
        assert len(result.windows) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_optimization.py::TestWalkForwardResult -v 2>&1 | head -10
```

Expected: `AttributeError` — `run()` method does not exist.

- [ ] **Step 3: Add `WalkForwardResult` dataclass and `run()` method to `backtest/optimization.py`**

Add the `WalkForwardResult` dataclass after the imports (before `GridSearch`):

```python
@dataclass
class WalkForwardResult:
    """Results from a Walk-Forward Analysis run.

    Attributes:
        equity_curve: Stitched out-of-sample equity curve as a pd.Series
            with DatetimeIndex. Values are portfolio value at each date.
        windows: pd.DataFrame with one row per test window. Columns:
            train_start, train_end, test_start, test_end, best_params (dict),
            objective_score, n_trades, total_return, cagr, sharpe_ratio,
            sortino_ratio, calmar_ratio, max_drawdown, ulcer_index,
            profit_factor, win_rate, expectancy, recovery_factor.
        summary: dict of aggregate metrics computed over the full equity_curve,
            plus n_windows, n_windows_with_trades, param_stability.
        best_params_overall: The parameter set selected most often across
            windows. Ties broken by highest mean objective score.
    """

    equity_curve: pd.Series
    windows: pd.DataFrame
    summary: dict
    best_params_overall: dict
```

Add the `run()` method inside `WalkForwardOptimizer` (after `_run_train_window`):

```python
    def run(self) -> WalkForwardResult:
        """Execute Walk-Forward Analysis and return the composite result.

        For each window pair:
        1. Optimise parameters on the training window.
        2. Evaluate the best parameters on the test window with warmup filtering.
        3. Record per-window metrics and the equity curve segment.

        Returns:
            WalkForwardResult with equity_curve, windows, summary, best_params_overall.
        """
        from .runner import BacktestRunnerImpl
        import backtest.metrics as _metrics

        windows = _generate_windows(
            self.data, self.train_size, self.test_size, self.window_type
        )

        all_equity_segments: list[pd.Series] = []
        all_trades: list[dict] = []
        window_rows: list[dict] = []

        metric_names = [
            "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "max_drawdown", "ulcer_index",
            "profit_factor", "win_rate", "expectancy", "recovery_factor",
        ]

        for train_data, test_data in windows:
            train_result = self._run_train_window(train_data)
            best_params = train_result["best_params"]
            objective_score = train_result["objective_score"]

            # --- Evaluate on test window ---
            strategy = self.strategy_class(**best_params)
            warmup_n = strategy.warmup_period
            runner = BacktestRunnerImpl(strategy=strategy, benchmarks=[])

            try:
                test_run = runner.run(data=test_data, start_capital=None)
            except Exception:
                # If the test window fails, record a zero-trade window
                row = {
                    "train_start": train_data.index[0],
                    "train_end": train_data.index[-1],
                    "test_start": test_data.index[0],
                    "test_end": test_data.index[-1],
                    "best_params": best_params,
                    "objective_score": objective_score,
                    "n_trades": 0,
                    **{m: float("-inf") for m in metric_names},
                }
                window_rows.append(row)
                continue

            test_equity = test_run["strategy_returns"]
            test_portfolio_history = [
                {"date": d, "value": v} for d, v in test_equity.items()
            ]
            test_trades = test_run["trades"]

            # Apply warmup buffer to scoring
            if warmup_n > 0 and len(test_data) > warmup_n:
                cutoff = test_data.index[warmup_n]
                scored_history, scored_trades = _filter_by_warmup(
                    test_portfolio_history, test_trades, cutoff
                )
            else:
                scored_history, scored_trades = test_portfolio_history, test_trades

            # Compute all metrics for this window
            window_metrics = {
                name: METRICS[name](scored_history, scored_trades)
                for name in metric_names
            }

            row = {
                "train_start": train_data.index[0],
                "train_end": train_data.index[-1],
                "test_start": test_data.index[0],
                "test_end": test_data.index[-1],
                "best_params": best_params,
                "objective_score": objective_score,
                "n_trades": len(scored_trades),
                **window_metrics,
            }
            window_rows.append(row)
            all_equity_segments.append(test_equity)
            all_trades.extend(scored_trades)

        # --- Stitch equity curve ---
        if all_equity_segments:
            equity_curve = pd.concat(all_equity_segments)
        else:
            equity_curve = pd.Series(dtype=float)

        # --- Compute summary ---
        windows_df = pd.DataFrame(window_rows)
        full_portfolio_history = [
            {"date": d, "value": v} for d, v in equity_curve.items()
        ]
        summary_metrics = {
            name: METRICS[name](full_portfolio_history, all_trades)
            for name in metric_names
        }

        n_windows = len(window_rows)
        n_windows_with_trades = sum(1 for r in window_rows if r["n_trades"] > 0)

        # --- best_params_overall ---
        best_params_overall = self._compute_best_params_overall(window_rows)

        param_stability = (
            sum(
                1 for r in window_rows
                if r["best_params"] == best_params_overall
            ) / n_windows
            if n_windows > 0 else 0.0
        )

        summary = {
            **summary_metrics,
            "n_windows": n_windows,
            "n_windows_with_trades": n_windows_with_trades,
            "param_stability": param_stability,
        }

        return WalkForwardResult(
            equity_curve=equity_curve,
            windows=windows_df,
            summary=summary,
            best_params_overall=best_params_overall,
        )

    def _compute_best_params_overall(self, window_rows: list[dict]) -> dict:
        """Return the param set selected most often; break ties by mean objective score.

        Args:
            window_rows: List of window result dicts, each containing 'best_params'
                and 'objective_score'.

        Returns:
            The most-selected param dict, or an empty dict if no windows.
        """
        if not window_rows:
            return {}

        from collections import Counter, defaultdict

        count: Counter = Counter()
        scores: dict = defaultdict(list)

        for row in window_rows:
            key = str(sorted(row["best_params"].items()))
            count[key] += 1
            scores[key].append(row["objective_score"])

        # Find the maximum frequency
        max_count = max(count.values())
        candidates = [k for k, c in count.items() if c == max_count]

        # Break ties by mean score
        best_key = max(
            candidates,
            key=lambda k: sum(s for s in scores[k] if s != float("-inf")) / max(1, len(scores[k])),
        )

        # Find the original dict for this key
        for row in window_rows:
            if str(sorted(row["best_params"].items())) == best_key:
                return row["best_params"]

        return window_rows[0]["best_params"]
```

- [ ] **Step 4: Run result tests**

```bash
uv run pytest tests/test_optimization.py::TestWalkForwardResult -v
```

Expected: all pass.

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

Expected: all pass.

- [ ] **Step 6: Run ruff**

```bash
uv run ruff check backtest/optimization.py backtest/metrics.py
```

Fix any issues reported.

- [ ] **Step 7: Commit**

```bash
git add backtest/optimization.py tests/test_optimization.py
git commit -m "feat: add WalkForwardResult and WalkForwardOptimizer.run()"
```

---

## Task 10: End-to-end integration test with known-optimal parameter

**Files:**
- Modify: `tests/test_optimization.py`

This test verifies that WFA discovers a known-optimal parameter in synthetic data. The data is constructed so that `consecutive_days=2` is clearly the best choice: prices oscillate in runs of exactly 2 days up then 2 days down. A 2-day consecutive strategy buys at the trough and sells at the peak on every cycle; 1-day and 3-day strategies generate worse timing.

- [ ] **Step 1: Add the end-to-end test to `tests/test_optimization.py`**

Append after `TestWalkForwardResult`:

```python
class TestEndToEnd:
    """End-to-end integration tests for WalkForwardOptimizer."""

    def _make_two_day_cycle_data(self, n=600):
        """Return data with strict 2-day up / 2-day down cycles.

        Price pattern (repeating):
          day 0: 100  (start of 2-day down)
          day 1: 98
          day 2: 100  (start of 2-day up)
          day 3: 102

        consecutive_days=2: buys after 2 down days (day 1→2), sells after
        2 up days (day 3→0). The signal is shifted 1 day so execution is
        on the correct bar. This is the optimal parameter.
        """
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        prices = []
        for i in range(n):
            phase = i % 4
            if phase == 0:
                prices.append(100.0)
            elif phase == 1:
                prices.append(98.0)
            elif phase == 2:
                prices.append(100.0)
            else:
                prices.append(102.0)
        return pd.DataFrame({"Close": prices}, index=dates)

    def test_optimizer_selects_known_optimal_param_majority_of_windows(self):
        """consecutive_days=2 should win in at least 50% of windows."""
        from backtest.optimization import WalkForwardOptimizer
        from backtest.strategy import ConsecutiveDaysStrategy

        data = self._make_two_day_cycle_data(n=600)
        opt = WalkForwardOptimizer(
            strategy_class=ConsecutiveDaysStrategy,
            param_space={"consecutive_days": [1, 2, 3]},
            data=data,
            train_size=200,
            test_size=100,
            window_type="sliding",
            objective="total_return",
            min_trades=0,
        )
        result = opt.run()

        selected = [r["best_params"]["consecutive_days"] for r in result.windows.to_dict("records")]
        frac_optimal = selected.count(2) / len(selected)
        assert frac_optimal >= 0.5, (
            f"Expected consecutive_days=2 selected in >=50% of windows, "
            f"got {frac_optimal:.0%}. Selection counts: {dict(zip(*np.unique(selected, return_counts=True)))}"
        )

    def test_random_search_end_to_end(self):
        """RandomSearch completes and returns a valid result."""
        from backtest.optimization import RandomSearch, WalkForwardOptimizer
        from backtest.strategy import ConsecutiveDaysStrategy

        data = self._make_two_day_cycle_data(n=400)
        opt = WalkForwardOptimizer(
            strategy_class=ConsecutiveDaysStrategy,
            param_space={"consecutive_days": [1, 2, 3, 4, 5]},
            data=data,
            train_size=150,
            test_size=50,
            window_type="anchored",
            searcher=RandomSearch(n=3, seed=7),
            objective="total_return",
            min_trades=0,
        )
        result = opt.run()
        assert len(result.windows) > 0
        assert isinstance(result.equity_curve, pd.Series)
        assert isinstance(result.summary, dict)

    def test_custom_objective_callable(self):
        """A custom callable objective is accepted and used."""
        from backtest.optimization import WalkForwardOptimizer
        from backtest.strategy import ConsecutiveDaysStrategy

        call_count = {"n": 0}

        def my_metric(portfolio_history, trades):
            call_count["n"] += 1
            if not portfolio_history:
                return float("-inf")
            return portfolio_history[-1]["value"] - portfolio_history[0]["value"]

        data = self._make_two_day_cycle_data(n=400)
        opt = WalkForwardOptimizer(
            strategy_class=ConsecutiveDaysStrategy,
            param_space={"consecutive_days": [1, 2]},
            data=data,
            train_size=150,
            test_size=50,
            window_type="sliding",
            objective=my_metric,
            min_trades=0,
        )
        result = opt.run()
        assert call_count["n"] > 0, "Custom objective was never called"
        assert len(result.windows) > 0
```

Also add `import numpy as np` to the imports at the top of `tests/test_optimization.py` if not already present.

- [ ] **Step 2: Run end-to-end tests**

```bash
uv run pytest tests/test_optimization.py::TestEndToEnd -v
```

Expected: all pass.

- [ ] **Step 3: Run the complete test suite with coverage**

```bash
uv run pytest --cov=backtest -q 2>&1 | tail -20
```

Expected: all tests pass, coverage ≥ 75%.

- [ ] **Step 4: Run ruff on all modified files**

```bash
uv run ruff check backtest/metrics.py backtest/optimization.py backtest/runner.py
```

Fix any issues.

- [ ] **Step 5: Commit**

```bash
git add tests/test_optimization.py
git commit -m "test: add end-to-end WFA integration tests with known-optimal parameter"
```

---

## Completion checklist

After all tasks, verify the following:

- [ ] `uv run pytest -q` — all pass
- [ ] `uv run ruff check backtest/` — no errors
- [ ] `uv run pytest --cov=backtest -q` — coverage ≥ 75%
- [ ] `from backtest.optimization import WalkForwardOptimizer, GridSearch, RandomSearch, WalkForwardResult` works
- [ ] `from backtest.metrics import METRICS, sharpe_ratio, total_return` works
- [ ] The usage example from the spec runs without error on synthetic data:

```python
import pandas as pd
from backtest.optimization import WalkForwardOptimizer, RandomSearch
from backtest.strategy import ConsecutiveDaysStrategy

dates = pd.date_range("2018-01-01", periods=500, freq="B")
df = pd.DataFrame({"Close": [100 + i * 0.1 for i in range(500)]}, index=dates)

opt = WalkForwardOptimizer(
    strategy_class=ConsecutiveDaysStrategy,
    param_space={"consecutive_days": [1, 2, 3]},
    data=df,
    train_size=200,
    test_size=50,
    window_type="sliding",
    searcher=RandomSearch(n=3, seed=42),
    objective="sharpe_ratio",
    min_trades=0,
)
result = opt.run()
print(result.summary)
print(result.windows[["test_start", "best_params", "objective_score"]])
```
