# Trading Strategy Backtesting Framework

Just plying with some backtesting trading strategies with support for multiple benchmarks and performance metrics. It's been on my "one day" list to doing a little vibe coding to cross it off.

## Features

- Modular design with clear separation of concerns
- Support for custom trading strategies and benchmarks
- Comprehensive performance metrics
- Interactive equity curve visualization
- Multiple benchmark strategies included
- Test-driven development approach


## Project Structure

```
trading/
├── backtest/
│   ├── __init__.py
│   ├── interfaces.py      # Core interfaces
│   ├── strategies.py      # Strategy implementations
│   ├── benchmarks.py      # Benchmark implementations
│   └── runner.py         # Backtest runner
├── examples/
│   └── run_backtest.py   # Example usage
├── tests/
│   └── test_backtest_interfaces.py
├── Pipfile
└── README.md
```

## Usage

### Creating a Custom Strategy

Implement the `Strategy` interface:

```python
from backtest.interfaces import Strategy
import pandas as pd

class MyStrategy(Strategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Your signal generation logic here
        return pd.DataFrame({
            'buy': buy_signals,
            'sell': sell_signals
        })
```

### Creating a Custom Benchmark

Implement the `Benchmark` interface:

```python
from backtest.interfaces import Benchmark
import pandas as pd

class MyBenchmark(Benchmark):
    def calculate_returns(self, data: pd.DataFrame, start_capital: float) -> pd.Series:
        # Your benchmark return calculation logic here
        return returns_series
```

### Running a Backtest

```python
from backtest.runner import BacktestRunnerImpl
from backtest.strategies import ConsecutiveDaysStrategy
from backtest.benchmarks import BuyAndHold, SPYBuyAndHold

# Create strategy and benchmarks
strategy = ConsecutiveDaysStrategy(up_days=3, down_days=3)
benchmarks = [BuyAndHold(), SPYBuyAndHold()]

# Create and run backtest
runner = BacktestRunnerImpl(strategy, benchmarks)
results = runner.run(data, start_capital=10000.0)

# Access results
print(f"Strategy Return: {results['strategy_metrics']['total_return']:.2f}%")
```

## Included Strategies

### ConsecutiveDaysStrategy
A strategy that generates buy signals after N consecutive down days and sell signals after N consecutive up days.

### Benchmarks
- `BuyAndHold`: Simple buy and hold strategy
- `SPYBuyAndHold`: S&P 500 ETF buy and hold
- `DollarCostAveraging`: Dollar-cost averaging with configurable frequency

## Performance Metrics

The framework calculates the following metrics:
- Total Return
- Win Rate
- Maximum Drawdown

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Implement the feature
5. Submit a pull request

## License

MIT License 