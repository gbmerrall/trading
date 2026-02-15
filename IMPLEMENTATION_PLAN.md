# Implementation Plan: Trading Strategies & Walk-Forward Analysis

## Overview

This document outlines the roadmap for enhancing the trading backtesting framework. The goals are:
1. **Expand Strategy Library**: Implement 12 trading strategies defined in `STRATEGIES.md`.
2. **Implement Walk-Forward Analysis (WFA)**: Add strategy optimization and validation using
   sliding window techniques.
3. **Refactor Core Architecture**: Update the codebase to support scalable strategy development.

---

## Part 1: Architecture Refactoring

### 1.1 Dependency Management

*   **Action**: Add the `ta` library (by bukosabino) to `Pipfile`.
*   **Reasoning**: Provides tested implementations of technical indicators (RSI, MACD, Bollinger
    Bands, etc.). Chosen over `pandas-ta` because `pandas-ta` has not been updated since 2022 and
    has known compatibility issues with pandas 2.x.
*   **Validation step**: After install, run `python -c "import ta; print(ta.__version__)"` to
    confirm compatibility with the current pandas version.
*   **Action**: Remove `pydantic` from `Pipfile` (installed but unused; the project uses
    `validation.py` with custom `ValidationError`).

### 1.2 Strategy Interface (`backtest/strategy.py`)

*   **Action**: Create an abstract base class `BaseStrategy`.
*   **Key Methods**:
    *   `generate_signals(data: pd.DataFrame) -> pd.DataFrame`: Returns DataFrame with `buy` and
        `sell` boolean columns. May include optional columns (`stop_loss`, `position_size`) that
        the runner checks via `if column in signals.columns`.
    *   `get_parameters() -> Dict[str, Any]`: Returns current strategy parameters (required for
        WFA optimization).
    *   `set_parameters(params: Dict[str, Any])`: Updates parameters dynamically. Must re-validate
        after setting.
*   **Signal format**: The primary interface remains boolean `buy`/`sell` columns. Optional columns
    are additive and do not break backward compatibility.
*   **Logging**: `BaseStrategy` should log at key boundaries using the project's logging setup:
    parameters used, data range, number of signals generated.
*   **Warmup**: Each strategy must declare a `warmup_period` property (integer) representing the
    minimum number of bars needed before signals are valid. Used by WFA to discard early signals.

### 1.3 Retrofit `ConsecutiveDaysStrategy`

*   **Action**: Make `ConsecutiveDaysStrategy` inherit from `BaseStrategy`.
*   **Action**: Add `get_parameters()`, `set_parameters()`, and `warmup_period` property.
*   **Constraint**: All existing tests must pass without modification after this change. Run full
    test suite before proceeding to Part 2.

### 1.4 Configuration System (`backtest/config.py`)

*   **Action**: Each strategy owns its parameters via constructor defaults. The `StrategyConfig`
    dataclass provides optional overrides but is NOT the canonical parameter source.
*   **Change**:
    *   Keep `consecutive_days` on `StrategyConfig` for backward compatibility.
    *   Do NOT add per-strategy fields (e.g., `rsi_period`, `macd_fast`) to `StrategyConfig`.
    *   Strategies accept parameters in `__init__` and fall back to sensible defaults if not
        provided. Configuration integration is opt-in, not required.
*   **Reasoning**: Avoids bloating `StrategyConfig` with 12+ strategy-specific fields. Strategies
    remain self-contained and independently testable.

---

## Part 2: Strategy Implementation

Each strategy follows this pattern:
1. Write tests FIRST (TDD requirement from CLAUDE.md)
2. Use synthetic data with precomputed expected values for deterministic assertions
3. Inherit from `BaseStrategy`
4. Validate inputs using `validate_dataframe()`, `validate_price_data()`
5. Shift signals by 1 day (`signals.shift(1).fillna(False)`) to prevent look-ahead bias
6. Declare `warmup_period` based on longest indicator lookback

### 2.1 Technical Indicator Strategies

Each uses the `ta` library for indicator calculations.

*   **MovingAverageCrossoverStrategy**:
    *   Params: `short_window` (default 20), `long_window` (default 50).
    *   Warmup: `long_window`.
    *   Logic: Buy when short SMA crosses above long SMA; sell on reverse crossover.

*   **RSIStrategy**:
    *   Params: `period` (default 14), `lower_bound` (default 30), `upper_bound` (default 70).
    *   Warmup: `period`.
    *   Logic: Buy when RSI < `lower_bound`; sell when RSI > `upper_bound`.

*   **BollingerBandsStrategy**:
    *   Params: `period` (default 20), `std_dev` (default 2.0).
    *   Warmup: `period`.
    *   Logic: Buy on lower band touch/break; sell on upper band touch/break (mean reversion).

*   **MACDStrategy**:
    *   Params: `fast` (default 12), `slow` (default 26), `signal` (default 9).
    *   Warmup: `slow + signal`.
    *   Logic: Buy when MACD line crosses above signal line; sell on reverse.

*   **ParabolicSARStrategy**:
    *   Params: `af` (default 0.02), `max_af` (default 0.2).
    *   Warmup: 1 (minimal).
    *   Logic: Buy when price crosses above SAR; sell when price crosses below SAR.

### 2.2 Price Action Strategies

*   **BreakoutStrategy**:
    *   Params: `lookback_period` (default 20).
    *   Warmup: `lookback_period`.
    *   Logic: Buy when close exceeds N-day high; sell when close breaks below N-day low.

*   **GapStrategy**:
    *   Params: `min_gap_pct` (default 0.02).
    *   Warmup: 1.
    *   Logic: Detect overnight gaps (open vs previous close). Buy on gap-down (fill expected);
        sell on gap-up. Requires `Open` and `Close` columns.

*   **FibonacciRetracementStrategy**:
    *   Params: `swing_lookback` (default 20), `retracement_levels` (default [0.382, 0.5, 0.618]).
    *   Warmup: `swing_lookback`.
    *   Logic: Identify swing high/low over lookback period. Buy near support levels (61.8%);
        sell near resistance levels (38.2%). Requires `High`, `Low`, `Close` columns.

### 2.3 Composite Strategies

*   **MeanReversionStrategy**:
    *   Params: `rsi_period` (default 14), `rsi_lower` (default 30), `rsi_upper` (default 70),
        `bb_period` (default 20), `bb_std` (default 2.0).
    *   Warmup: `max(rsi_period, bb_period)`.
    *   Logic: Buy when RSI < lower AND price < lower Bollinger Band. Sell when RSI > upper AND
        price > upper Bollinger Band.

*   **MomentumStrategy**:
    *   Params: `roc_period` (default 12), `buy_threshold` (default 0.0),
        `sell_threshold` (default 0.0).
    *   Warmup: `roc_period`.
    *   Logic: Single-asset Rate-of-Change. Buy when ROC > `buy_threshold`; sell when
        ROC < `sell_threshold`.

*   **VolatilityStrategy**:
    *   Params: `atr_period` (default 14), `atr_multiplier` (default 2.0).
    *   Warmup: `atr_period`.
    *   Logic: ATR-based breakout. Buy when close > previous close + ATR * multiplier; sell when
        close < previous close - ATR * multiplier. Single-asset, price data only.

*   **EnsembleStrategy**:
    *   Params: `strategies` (list of `BaseStrategy`), `threshold` (default 0.5).
    *   Warmup: `max(s.warmup_period for s in strategies)`.
    *   Logic: Runs all sub-strategies, produces buy/sell when proportion of agreeing signals
        exceeds `threshold`.

---

## Part 3: Walk-Forward Analysis (WFA) Implementation

### 3.1 Components

*   **`WalkForwardOptimizer` Class** (`backtest/optimization.py`):
    *   **Input**: `BaseStrategy` class (not instance), parameter search space (dict of param name
        to list of values), historical data, objective function (default: Sharpe ratio).
    *   **Logic**:
        1.  **Split Data**: Divide into Train (In-Sample) and Test (Out-of-Sample) windows.
        2.  **Apply warmup buffer**: Discard signals from the first `strategy.warmup_period` bars
            of each window to avoid incomplete indicator data.
        3.  **Optimize**: For each Train window, run grid search over parameter space. Instantiate
            strategy via `set_parameters()`, run backtest, score with objective function.
        4.  **Validate**: Apply best parameters to subsequent Test window.
        5.  **Stitch**: Combine Test window results into composite Walk-Forward equity curve.
    *   **Output**: Composite performance metrics, parameter stability over time, per-window
        results.

### 3.2 Data Splitting Logic

*   **Sliding Window**: Train on [T0, T1], Test on [T1, T2]. Slide: Train on [T1, T2],
    Test on [T2, T3].
*   **Anchored Window**: Train on [T0, T1], Test on [T1, T2]. Expand: Train on [T0, T2],
    Test on [T2, T3].
*   **Warmup buffer**: Each window's signals are discarded for the first N bars where
    N = `strategy.warmup_period`. This prevents signals based on incomplete indicator calculations
    and avoids signal-shift leakage at train/test boundaries.

### 3.3 Data Leakage Prevention

*   Train/test windows must not overlap.
*   The 1-day signal shift means a signal generated on the last bar of training data would execute
    on the first bar of test data. The warmup buffer on the test window handles this.
*   Tests must verify: no train-period prices appear in test-period signal calculations.

### 3.4 Integration

*   Create `backtest/optimization.py` for `WalkForwardOptimizer` and `GridSearch`.
*   The runner is NOT modified. WFA creates its own runner instances internally for each window.
*   WFA results include per-window parameter values to assess parameter stability.

---

## Part 4: Testing & Validation

### 4.1 Strategy Test Approach

For each strategy:
*   **Synthetic data tests**: Create small datasets (10-20 bars) with known price patterns
    (uptrend, downtrend, mean-reverting). Precompute expected indicator values by hand and assert
    signal output matches.
*   **Edge cases**: Empty data, single row, all-NaN prices, flat prices (zero volatility),
    extreme values.
*   **Warmup verification**: Assert that no signals are generated during the warmup period.
*   **Signal shift verification**: Assert signals are delayed by 1 day relative to the condition
    that triggered them.

### 4.2 WFA Tests

*   **Window splitting**: Verify train/test windows are contiguous, non-overlapping, and cover
    the full date range.
*   **Data leakage**: Verify no information from test windows influences train-window optimization.
*   **Warmup buffer**: Verify signals from warmup bars are excluded from performance calculation.
*   **End-to-end**: Run WFA on synthetic data with a known-optimal parameter and verify the
    optimizer discovers it.

### 4.3 Integration Tests

*   Run full backtests with each new strategy through the existing runner.
*   Verify results dict contains expected keys and valid metrics.
*   Run a full WFA pipeline on a small dataset to verify the end-to-end flow.

---

## Execution Phases

### Phase 1: Foundation (atomic steps, run tests between each)

1.  Add `ta` library to Pipfile, verify import works with current pandas version.
2.  Remove `pydantic` from Pipfile (unused).
3.  Create `BaseStrategy` ABC in `backtest/strategy.py`.
4.  Retrofit `ConsecutiveDaysStrategy` to inherit from `BaseStrategy`. Run full test suite.
5.  Add `warmup_period`, `get_parameters()`, `set_parameters()` to `ConsecutiveDaysStrategy`.
    Run full test suite.

### Phase 2: Core Strategies (5 strategies, tests first)

1.  `MovingAverageCrossoverStrategy` + tests.
2.  `RSIStrategy` + tests.
3.  `MACDStrategy` + tests.
4.  `BollingerBandsStrategy` + tests.
5.  `BreakoutStrategy` + tests.

### Phase 3: Walk-Forward Analysis

1.  Implement `GridSearch` in `backtest/optimization.py` + tests.
2.  Implement `WalkForwardOptimizer` with sliding and anchored windows + tests.
3.  End-to-end WFA integration test with a Phase 2 strategy.

### Phase 4: Remaining Strategies (tests first)

1.  `ParabolicSARStrategy` + tests.
2.  `GapStrategy` + tests.
3.  `FibonacciRetracementStrategy` + tests.
4.  `MeanReversionStrategy` + tests.
5.  `MomentumStrategy` + tests.
6.  `VolatilityStrategy` + tests.
7.  `EnsembleStrategy` + tests.

### Phase 5: Polish

1.  WFA reporting and visualization.
2.  Performance profiling for grid search.
3.  Coverage audit (target 75%+).
