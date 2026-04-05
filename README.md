# Trading Strategy Backtesting Framework

Having Claude Code produce this as I've been idly interested in it on and off so this is a
chance to whip something up. I also saw [Entire](https://entire.io/) drop which seems to be $60M
wrapped in a bunch of commit hooks so thought it might be interesting to check out.

Still a work in progress as interest waxes and wanes. Don't expect to get rich but OTOH who knows?

---

## Installation

Requires Python 3.10+.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Getting Started

### 1. Run a basic backtest

```python
import yfinance as yf
from backtest.runner import BacktestRunnerImpl
from backtest.strategy import RSIStrategy
from backtest.benchmarks import BuyAndHold, DollarCostAveraging

# Download historical data
data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# Set up the strategy and benchmarks
strategy = RSIStrategy(period=14, lower_bound=30, upper_bound=70)
benchmarks = [BuyAndHold(), DollarCostAveraging()]

# Run the backtest
runner = BacktestRunnerImpl(strategy=strategy, benchmarks=benchmarks)
result = runner.run(data, start_capital=10_000.0)

# Inspect results
print(result["strategy_metrics"])
# {'total_return': 0.43, 'win_rate': 0.52, 'num_trades': 38, 'max_drawdown': -0.18}

for name, metrics in result["benchmark_metrics"].items():
    print(f"{name}: {metrics}")
```

### 2. Optimise parameters with Walk-Forward Analysis

Walk-Forward Analysis (WFA) repeatedly optimises on a training window and validates on a
held-out test window. The stitched test results form a realistic out-of-sample equity curve.

```python
import yfinance as yf
from backtest.optimization import WalkForwardOptimizer, RandomSearch
from backtest.strategy import RSIStrategy

data = yf.download("AAPL", start="2015-01-01", end="2024-01-01")

optimizer = WalkForwardOptimizer(
    strategy_class=RSIStrategy,
    param_space={
        "period":      [7, 14, 21],
        "lower_bound": [25, 30, 35],
        "upper_bound": [65, 70, 75],
    },
    data=data,
    train_size=252,   # 1 year of training data
    test_size=63,     # 1 quarter of test data
    window_type="sliding",
    searcher=RandomSearch(n=50, seed=42),
    objective="sharpe_ratio",
    min_trades=5,
)

result = optimizer.run()

print(result.summary)
print(result.best_params_overall)
# {'period': 14, 'lower_bound': 30, 'upper_bound': 70}

# Per-window breakdown
print(result.windows[["test_start", "best_params", "sharpe_ratio", "n_trades"]])

# Plot the out-of-sample equity curve
result.equity_curve.plot(title="WFA Out-of-Sample Equity Curve")
```

### 3. Available objective metrics

The following metrics can be passed as the `objective` parameter to `WalkForwardOptimizer`,
or called directly for custom analysis:

```python
from backtest.metrics import sharpe_ratio, calmar_ratio, METRICS

# Call directly
score = sharpe_ratio(portfolio_history, trades)

# Or use the registry
score = METRICS["sortino_ratio"](portfolio_history, trades)

# All built-in metrics
print(list(METRICS.keys()))
# ['total_return', 'cagr', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
#  'max_drawdown', 'ulcer_index', 'profit_factor', 'win_rate', 'expectancy',
#  'recovery_factor']
```

You can also pass any callable with the signature `(portfolio_history, trades) -> float`:

```python
optimizer = WalkForwardOptimizer(
    ...,
    objective=lambda ph, t: len(t) / max(1, abs(t[0]["pnl"])),  # custom metric
)
```

---

## Strategies

### Technical Indicator Strategies

| Class | Description |
|-------|-------------|
| `ConsecutiveDaysStrategy` | Buy after N consecutive down days, sell after N consecutive up days |
| `MovingAverageCrossoverStrategy` | Golden/death cross on short- and long-term moving average crossovers |
| `RSIStrategy` | Buy when RSI is oversold, sell when overbought |
| `MACDStrategy` | Buy/sell on MACD line crossovers with the signal line |
| `BollingerBandsStrategy` | Mean reversion on Bollinger Band touches |
| `ParabolicSARStrategy` | Trend-following with Parabolic SAR (requires High/Low columns) |

### Price Action Strategies

| Class | Description |
|-------|-------------|
| `BreakoutStrategy` | Enter on N-day high breakout or low breakdown |
| `GapStrategy` | Gap trading on overnight price discontinuities (requires Open column) |
| `FibonacciRetracementStrategy` | Support/resistance at 38.2%, 50%, 61.8% retracement levels (requires High/Low) |

### Composite Strategies

| Class | Description |
|-------|-------------|
| `MeanReversionStrategy` | Combines RSI and Bollinger Bands for mean reversion signals |
| `MomentumStrategy` | Rate-of-Change (ROC) momentum — buy/sell on threshold crossovers |
| `VolatilityStrategy` | ATR-based volatility breakout detection (requires High/Low) |
| `EnsembleStrategy` | Wraps multiple sub-strategies, aggregates signals via majority voting |

### Walk-Forward Analysis

`WalkForwardOptimizer` optimises any strategy's parameters over a rolling or expanding train
window and evaluates out-of-sample on the following test window. Results include a stitched
equity curve, per-window metrics, and the most stable parameter set across all windows.

See [Getting Started](#2-optimise-parameters-with-walk-forward-analysis) above for usage.

---

## Running Tests

```bash
# Run the full test suite
pytest

# With coverage report
pytest --cov=backtest -v
```

---

## Scope Constraints

- All strategies operate on **single-asset OHLCV data**. Cross-asset strategies (pairs trading,
  relative momentum) are out of scope until the runner supports multi-asset event loops.
- **Volatility strategies** use ATR from price data only. Options-based or VIX-based approaches
  require a different data model and are out of scope.
- **Momentum strategies** use single-asset Rate-of-Change. Cross-sectional momentum (ranking
  multiple assets) requires runner changes and is out of scope.

---

## License

MIT License
