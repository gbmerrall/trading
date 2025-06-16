import pytest
import pandas as pd
import os
import yfinance as yf

# Add project root to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.runner import BacktestRunnerImpl
from backtest.strategy import ConsecutiveDaysStrategy
from backtest.benchmarks import BuyAndHold

def test_runner_executes_without_error_on_real_data(monkeypatch):
    """
    Tests that the backtest runner can execute with a real-world
    dataset without raising an error. This is a key regression test.
    """
    # Given: A small, real-world dataset from yfinance
    class MockYFinance:
        def download(self, *args, **kwargs):
            dates = pd.to_datetime(['2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-09', '2023-01-10'])
            prices = [124.47, 125.79, 124.42, 128.99, 129.41, 130.15]
            return pd.DataFrame({'Close': prices}, index=dates)

    monkeypatch.setattr(yf, 'download', MockYFinance().download)
    data = yf.download('AAPL', start='2023-01-01', end='2023-01-10')

    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    benchmarks = [BuyAndHold()]
    runner = BacktestRunnerImpl(strategy, benchmarks)

    # When: The backtest is run
    try:
        results = runner.run(data, start_capital=10000.0)
        # Then: No exception should be raised
        assert results is not None
    except Exception as e:
        pytest.fail(f"Backtest runner failed with real data: {e}")

def test_strategy_produces_non_zero_return():
    """
    Tests that a simple strategy on predictable data results in a non-zero
    return, ensuring the backtesting engine is correctly processing trades.
    """
    # Given: Predictable data where a trade should occur and make a profit
    dates = pd.to_datetime(['2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-09'])
    prices = [100, 99, 98, 105, 106] # 2 down days, then up
    data = pd.DataFrame({'Close': prices}, index=dates)

    # A strategy that buys after 2 down days
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    benchmarks = [BuyAndHold()]
    runner = BacktestRunnerImpl(strategy, benchmarks)

    # When: The backtest is run
    results = runner.run(data, start_capital=10000.0)
    
    # Then: The strategy should have one profitable trade and a significant return
    metrics = results['strategy_metrics']
    assert metrics['num_trades'] == 1
    assert metrics['win_rate'] == 100.0
    assert metrics['total_return'] > 0.9 # Check for a meaningful return 