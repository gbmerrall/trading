# Trading Strategy Backtesting Framework

A robust, event-driven backtesting framework in Python for testing trading strategies against multiple benchmarks. Built with a focus on simplicity, reliability, and professional-grade portfolio management.

## 🚀 Recent Major Improvements

This framework has been significantly enhanced with:

- **🛡️ Comprehensive Input Validation**: All external inputs are validated with clear error messages
- **🏗️ Refactored Architecture**: Eliminated code duplication with clean base classes
- **💼 Professional Portfolio Management**: Multi-asset support with detailed analytics
- **📊 Enhanced Benchmarks**: Flexible benchmark system with configurable parameters
- **🧪 Extensive Testing**: 100+ test cases with comprehensive coverage
- **🔒 Security-Focused**: Sanitized inputs prevent common security issues

## Core Philosophy

- **Simplicity Over Complexity**: Event-driven loop with transparent, debuggable processes
- **Test-Driven Development**: Comprehensive test coverage ensures reliability
- **Robust Validation**: All inputs are validated with clear, actionable error messages
- **Professional-Grade**: Suitable for both research and production backtesting
- **Extensible Design**: Easy to add new strategies and benchmarks

## Project Structure

```
trading/
├── backtest/
│   ├── portfolio.py      # Core portfolio simulation engine
│   ├── strategy.py       # Strategy implementation(s)
│   ├── benchmarks.py     # Benchmark implementations
│   └── runner.py         # Backtest runner
├── examples/
│   └── run_backtest.py   # Example script showing usage
├── tests/                # Unit tests  
├── output/               # Directory for saved plots
```

## Getting Started

### Installation

This project uses `pipenv` for dependency management.

1.  **Install `pipenv`**:
    ```bash
    pip install pipenv
    ```

    Or install how you like. Normally brew (on Mac) or `pip install --user`

2.  **Install Dependencies**:
    Navigate to the project root and run:
    ```bash
    pipenv install --dev
    ```
    This will install all required packages (`pandas`, `yfinance`, `plotly`) and development dependencies (`pytest`, `pytest-cov`, `ruff`).

### Running a Backtest

The primary example is in `examples/run_backtest.py`. To run it:

```bash
pipenv run python examples/run_backtest.py
```

This will:
1.  Download historical price data for a stock (e.g., BHP).
2.  Run the `ConsecutiveDaysStrategy` against it.
3.  Compare its performance against `BuyAndHold`, `SPYBuyAndHold`, and `DollarCostAveraging` benchmarks.
4.  Print a formatted summary of performance metrics to the console.
5.  Save a plot of the equity curves to the `output/` directory with a timestamped filename.

## Included Components

### Strategy: `ConsecutiveDaysStrategy`
A simple momentum strategy.
- **Buy Signal**: Generated after a specified number of consecutive down days.
- **Sell Signal**: Generated after a specified number of consecutive up days.

### Benchmarks
- `BuyAndHold`: A simple buy-and-hold of the target asset.
- `SPYBuyAndHold`: A buy-and-hold of the SPY ETF, for market comparison.
- `DollarCostAveraging`: Simulates investing a fixed total amount in equal portions over a regular interval (`daily`, `weekly`, or `monthly`).

## Running Tests

The project maintains high test coverage. To run the full test suite and view the coverage report:

```bash
pipenv run pytest --cov=. -v
```

To run the linter:
```bash
pipenv run ruff check .
```

## How I've found the vibe coding so far

Well it (the LLMs+Cursor Agent) gets lost often and doesn't really follow instructions all that well. 
For example, my rules say "Write tests first, run them when you make changes" but those seem to get
ignored most of the time, leading to chasing bugs.  

Also, and this may be a Cursor thing, the agent will runs tests etc before I've accepted file 
changes which means the agent thinks the issue its trying to fix still exists. I've had to put
blocks in place to prevent it running python without asking first.

The agent also likes to add complexity then it gets stuck. I better approach would be to start with
a simple reproducible testcase and then figure out what's goign on. Instead it piles on the patches
to find a fix. This can be problematic in agent mode when it's doing everything, forcing you to
interrupt.

## Contributing

Contributions are welcome! Please feel free to fork the repository, create a feature branch, and submit a pull request.

## License

MIT License 