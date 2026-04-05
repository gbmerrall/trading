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

```bash
python examples/basic_backtest.py
```

RSI(14) vs Buy & Hold, SPY, and DCA on AAPL over 5 years. Prints a metric
summary and final portfolio value.

### 2. Compare all strategies side by side

```bash
python examples/compare_strategies.py           # defaults: AAPL 2020–2024
python examples/compare_strategies.py MSFT 2018-01-01 2024-01-01
```

Runs every built-in strategy on the same data and prints a table ranked by total return.

### 3. Parameter sensitivity sweep

```bash
python examples/parameter_sensitivity.py
python examples/parameter_sensitivity.py SPY 2018-01-01 2024-01-01
```

Sweeps RSI period from 5 to 30, printing Sharpe, Sortino, Calmar, and trade count at
each value. Helps identify stable parameter regions before optimising.

### 4. Walk-Forward Analysis

```bash
python examples/walk_forward_analysis.py
python examples/walk_forward_analysis.py SPY 2016-01-01 2024-01-01
```

WFA repeatedly optimises RSI parameters on a 1-year training window and validates on the
following 3-month test window. Prints per-window results, overall out-of-sample metrics,
and saves an equity curve plot to `output/`.

### 5. Ensemble strategy

```bash
python examples/ensemble_strategy.py
python examples/ensemble_strategy.py MSFT 2019-01-01 2024-01-01
```

Combines RSI, MACD, and Bollinger Bands into an ensemble that signals only when at least
2 of 3 strategies agree. Compares the ensemble against each constituent and Buy & Hold.

### Available objective metrics

The following metrics can be passed as the `objective` parameter to `WalkForwardOptimizer`,
or called directly from `backtest.metrics`:

```python
from backtest.metrics import METRICS

print(list(METRICS.keys()))
# ['total_return', 'cagr', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
#  'max_drawdown', 'ulcer_index', 'profit_factor', 'win_rate', 'expectancy',
#  'recovery_factor']
```

You can also pass any callable with the signature `(portfolio_history, trades) -> float`
as a custom objective.

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
