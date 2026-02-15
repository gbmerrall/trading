# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **trading strategy backtesting framework** built on an event-driven architecture. The framework tests trading strategies against historical price data and compares performance against multiple benchmarks (Buy & Hold, SPY, Dollar Cost Averaging).

Core philosophy: simplicity over complexity, test-driven development, comprehensive input validation.

## Architecture

### Event-Driven Backtesting Loop
The core backtesting loop (`backtest/runner.py`) processes historical data day-by-day:
1. Generate trading signals from strategy
2. Execute trades through Portfolio
3. Track portfolio value over time
4. Compare against benchmarks

### Module Organization

```
backtest/
├── portfolio.py      # Portfolio simulation engine (multi-asset support)
├── strategy.py       # BaseStrategy ABC and strategy implementations
├── benchmarks.py     # Benchmark strategies (BuyAndHold, SPYBuyAndHold, DCA)
├── runner.py         # BacktestRunnerImpl - orchestrates the backtest
├── config.py         # Centralized configuration system
├── validation.py     # Input validation utilities
└── constants.py      # Shared constants (ValidationLimits, TradingConstants)
```

### Key Design Patterns

**Configuration System**: Centralized configuration using dataclasses (`config.py`):
- `GlobalConfig` combines `PortfolioConfig`, `StrategyConfig`, `BenchmarkConfig`, `BacktestConfig`
- Factory methods for presets: `ConfigFactory.create_default()`, `.create_conservative()`, `.create_aggressive()`
- Access via singleton: `get_config()`, `set_config(config)`
- Individual component access: `get_portfolio_config()`, `get_strategy_config()`, etc.

**Base Classes**: Use inheritance to eliminate duplication:
- `BaseStrategy` - Abstract base class for all strategies (required for new strategies)
  - Enforces `generate_signals()`, `get_parameters()`, `set_parameters()`, `warmup_period`
- `BaseBenchmark` - Abstract base class for benchmarks
  - Provides common validation and calculation methods
  - All benchmarks inherit and implement `calculate_returns()`

**Validation Pattern**: All external inputs are validated using `validation.py`:
- Raises `ValidationError` (subclass of `ValueError`) with clear messages
- Validates DataFrames, price data, positive numbers, integers, file paths
- Uses `ValidationLimits` constants for consistent bounds checking

**Portfolio Management**: Dual interface for backward compatibility:
- Legacy interface: `process_day()` for single-asset backtesting
- Enhanced interface: `buy()`, `sell()`, `update_position_price()` for multi-asset support
- Tracks transactions, positions, realized/unrealized P&L

## Common Commands

### Testing
```bash
# Run full test suite with coverage
pipenv run pytest --cov=. -v

# Run specific test file
pipenv run pytest tests/test_portfolio.py -v

# Run tests matching a pattern
pipenv run pytest -k "test_consecutive" -v
```

### Linting
```bash
# Check code formatting/style
pipenv run ruff check .

# Auto-fix linting issues
pipenv run ruff check . --fix
```

### Running Backtests
```bash
# Run example backtest
pipenv run python examples/run_backtest.py
```

### Development Setup
```bash
# Install dependencies (first time)
pipenv install --dev

# Activate virtual environment (optional)
pipenv shell

# Verify Python path (MUST be inside virtual environment)
which python
```

## Test-Driven Development Requirements

From `.cursor/rules/development.mdc`:
- **Write tests BEFORE implementing strategy logic** (mandatory)
- Run tests after ANY change to Python files
- Always check for linter errors with 'ruff' after edits
- Parametrize tests for different inputs
- Test edge cases: missing data, market gaps, boundary conditions
- For strategies: verify warmup period prevents early signals, test signal shift by 1 day

## Data Flow

1. **Data Loading**: `yfinance.download()` → pandas DataFrame with DatetimeIndex
2. **Signal Generation**: `strategy.generate_signals(data)` → DataFrame with 'buy'/'sell' boolean columns
3. **Portfolio Simulation**: Event loop processes each day, executing trades via `portfolio.process_day()`
4. **Benchmark Calculation**: Each benchmark calculates returns independently
5. **Results Assembly**: Runner collects strategy/benchmark metrics and generates plots

## Important Implementation Details

### Strategy Signal Timing
Signals are **shifted by 1 day** in `strategy.py` to trade on the **next day's open** (prevents look-ahead bias):
```python
signals = signals.shift(1).fillna(False)
```

### Portfolio Value Tracking
The `Portfolio` class maintains:
- `cash`: Current cash balance
- `_positions`: Dict of `Position` objects (symbol → Position)
- `_transactions`: List of all `Transaction` objects
- `_value_history`: Portfolio value snapshots for each processed day

### Configuration Loading
The runner uses configuration defaults if parameters are None:
```python
if start_capital is None:
    start_capital = config.portfolio.start_capital
```

### Benchmark Base Class Pattern
All benchmarks must:
1. Inherit from `BaseBenchmark`
2. Implement `calculate_returns(data: pd.DataFrame, start_capital: float) -> pd.Series`
3. Use `_validate_inputs()`, `_prepare_prices()` for common validation/preparation

## Critical Constraints

1. **Pipenv Environment**: Always run Python commands inside the pipenv virtual environment. Check with `which python` first.
2. **Test Coverage**: Aim for 75%+ coverage. Run coverage reports periodically with `pytest --cov=backtest -v`.
3. **Input Validation**: Validate all external inputs at system boundaries (user input, data loading).
4. **No Over-Engineering**: Keep solutions simple. Don't add features beyond requirements.
5. **Google Python Style Guide**: 100-char line limit, sorted imports (stdlib → 3rd party → local), f-strings only with placeholders.

## Dependencies

- **Core**: `pandas`, `numpy`, `yfinance` (market data), `plotly` + `kaleido` (visualization)
- **Technical Analysis**: `ta` library for indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
- **Testing**: `pytest`, `pytest-cov`
- **Linting**: `ruff`

## Adding New Components

### Adding a New Strategy
1. **Write tests FIRST** (TDD is mandatory)
2. Inherit from `BaseStrategy` abstract base class
3. Implement required methods:
   - `generate_signals(data: pd.DataFrame) -> pd.DataFrame` - returns DataFrame with 'buy'/'sell' boolean columns
   - `get_parameters() -> Dict[str, Any]` - returns current strategy parameters
   - `set_parameters(params: Dict[str, Any])` - updates parameters dynamically (re-validate after setting)
   - `warmup_period` property - returns minimum bars needed before signals are valid (e.g., for RSI(14), return 14)
4. Validate inputs using `validate_dataframe()`, `validate_price_data()`
   - Most strategies require: `'Close'` column
   - Some strategies require: `'Close'`, `'High'`, `'Low'` (e.g., ParabolicSAR)
   - Validate all required columns with `validate_dataframe(data, required_columns=[...])`
5. Shift signals by 1 day (`signals.shift(1).fillna(False)`) to prevent look-ahead bias
6. Use the `ta` library for technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
7. Strategy parameters should have constructor defaults; `StrategyConfig` integration is optional

**Implemented Strategies:**

*Technical Indicators:*
- `ConsecutiveDaysStrategy` - Buy after N down days, sell after N up days
- `MovingAverageCrossoverStrategy` - Golden/death cross on MA crossovers
- `RSIStrategy` - Buy oversold, sell overbought based on RSI
- `MACDStrategy` - Buy/sell on MACD line crossovers
- `BollingerBandsStrategy` - Mean reversion on Bollinger Band touches
- `ParabolicSARStrategy` - Trend following with Parabolic SAR (requires High/Low)

*Price Action:*
- `BreakoutStrategy` - Breakout on N-day high, breakdown on N-day low
- `GapStrategy` - Gap trading (requires Open column)
- `FibonacciRetracementStrategy` - Support/resistance at Fib levels (requires High/Low)

*Composite Strategies:*
- `MeanReversionStrategy` - Combines RSI + Bollinger Bands for mean reversion
- `MomentumStrategy` - Rate of Change (ROC) for momentum-based trading
- `VolatilityStrategy` - ATR-based volatility breakouts (requires High/Low)
- `EnsembleStrategy` - Aggregates multiple sub-strategies via majority voting

### Adding a New Benchmark
1. Inherit from `BaseBenchmark`
2. Implement `calculate_returns(data, start_capital) -> pd.Series`
3. Use `_validate_inputs()`, `_prepare_prices()` for common logic
4. Add to `create_standard_benchmarks()` if it's a common benchmark
5. Write tests covering various data scenarios

## Known Issues & Patterns

### pandas FutureWarning
The project filters pandas FutureWarnings in `pytest.ini`:
```ini
filterwarnings = ignore::FutureWarning
```

### Cold-Start Analysis
When reviewing multi-component systems, perform cold-start trace:
1. Identify properties/methods returning None/empty for "no data yet"
2. Find call sites guarding on None/empty
3. Trace FIRST execution - can system transition out of initial state?
4. Verify tests exercise true initial state, not pre-populated data

### Configuration Persistence
Configurations can be saved/loaded:
```python
save_config_to_file(config, "path/to/config.json")
config = load_config_from_file("path/to/config.json")
```

## File Locations

- **Output**: Equity curve plots saved to `output/` directory (auto-created)
- **Tests**: All test files in `tests/` directory (prefix: `test_*.py`)
- **Examples**: Example usage scripts in `examples/`
- **Documentation**: README.md contains project overview and installation instructions
