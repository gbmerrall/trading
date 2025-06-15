import pytest
import pandas as pd

# Add the project root directory to the Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.interfaces import Strategy, Benchmark
from backtest.strategies import ConsecutiveDaysStrategy
from backtest.benchmarks import BuyAndHold, SPYBuyAndHold
from backtest.runner import BacktestRunnerImpl

def create_test_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    prices = pd.Series([100, 99, 98, 97, 98, 99, 100, 99, 98, 97], index=dates)
    return pd.DataFrame({'Close': prices})

def test_strategy_interface():
    """Test that Strategy interface enforces required methods."""
    # Given: A strategy class that doesn't implement generate_signals
    class InvalidStrategy(Strategy):
        pass
    
    # When/Then: Creating an instance should raise TypeError
    with pytest.raises(TypeError):
        InvalidStrategy()

def test_benchmark_interface():
    """Test that Benchmark interface enforces required methods."""
    # Given: A benchmark class that doesn't implement calculate_returns
    class InvalidBenchmark(Benchmark):
        pass
    
    # When/Then: Creating an instance should raise TypeError
    with pytest.raises(TypeError):
        InvalidBenchmark()

def test_consecutive_days_strategy():
    """Test the ConsecutiveDaysStrategy implementation."""
    # Given: Price data and strategy parameters
    data = create_test_data()
    strategy = ConsecutiveDaysStrategy(up_days=3, down_days=3)
    
    # When: Generate signals
    signals = strategy.generate_signals(data)
    
    # Then: Verify signal properties
    assert isinstance(signals, pd.DataFrame)
    assert 'buy' in signals.columns
    assert 'sell' in signals.columns
    assert signals['buy'].dtype == bool
    assert signals['sell'].dtype == bool
    assert len(signals) == len(data)

def test_buy_and_hold_benchmark():
    """Test the BuyAndHold benchmark implementation."""
    # Given: Price data and benchmark
    data = create_test_data()
    benchmark = BuyAndHold()
    start_capital = 10000.0
    
    # When: Calculate returns
    returns = benchmark.calculate_returns(data, start_capital)
    
    # Then: Verify return properties
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(data)
    assert returns.index.equals(data.index)
    assert returns.iloc[0] == start_capital
    assert returns.iloc[-1] > 0  # Should have some return

def test_spy_buy_and_hold_benchmark():
    """Test the SPYBuyAndHold benchmark implementation."""
    # Given: Price data and benchmark
    data = create_test_data()
    benchmark = SPYBuyAndHold()
    start_capital = 10000.0
    
    # When: Calculate returns
    returns = benchmark.calculate_returns(data, start_capital)
    
    # Then: Verify return properties
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(data)
    assert returns.index.equals(data.index)
    assert returns.iloc[0] == start_capital
    # Only check last value if it is not nan
    if not pd.isna(returns.iloc[-1]):
        assert returns.iloc[-1] > 0  # Should have some return

def test_backtest_runner():
    """Test the BacktestRunner implementation."""
    # Given: Strategy, benchmarks, and data
    data = create_test_data()
    strategy = ConsecutiveDaysStrategy(up_days=3, down_days=3)
    benchmarks = [BuyAndHold(), SPYBuyAndHold()]
    runner = BacktestRunnerImpl(strategy, benchmarks)
    start_capital = 10000.0
    # When: Run backtest
    results = runner.run(data, start_capital)
    # Then: Check results structure
    assert 'strategy_metrics' in results
    assert 'benchmark_metrics' in results
    assert 'strategy_returns' in results
    assert 'benchmark_returns' in results
    assert isinstance(results['benchmark_returns'], dict)

def test_backtest_runner_with_invalid_data():
    """Test BacktestRunner handles invalid data appropriately."""
    # Given: Invalid data (empty DataFrame)
    data = pd.DataFrame()
    strategy = ConsecutiveDaysStrategy(up_days=3, down_days=3)
    benchmarks = [BuyAndHold()]
    runner = BacktestRunnerImpl(strategy, benchmarks)
    start_capital = 10000.0
    with pytest.raises(ValueError, match="Data is empty"):
        runner.run(data, start_capital)

def test_backtest_runner_with_missing_columns():
    """Test BacktestRunner handles missing required columns."""
    # Given: Data missing required columns
    data = pd.DataFrame({'Open': [100, 101, 102]})
    strategy = ConsecutiveDaysStrategy(up_days=3, down_days=3)
    benchmarks = [BuyAndHold()]
    runner = BacktestRunnerImpl(strategy, benchmarks)
    start_capital = 10000.0
    with pytest.raises(ValueError, match="Data must contain 'Close' column"):
        runner.run(data, start_capital)

def test_buy_and_hold_all_nan():
    from backtest.benchmarks import BuyAndHold
    data = pd.DataFrame({'Close': [float('nan')] * 5}, index=pd.date_range('2020-01-01', periods=5))
    bh = BuyAndHold()
    try:
        bh.calculate_returns(data, 1000)
        assert False, "Should raise ValueError for all-NaN close prices"
    except ValueError as e:
        assert "All close prices are NaN" in str(e)

def test_dca_all_nan():
    from backtest.benchmarks import DollarCostAveraging
    data = pd.DataFrame({'Close': [float('nan')] * 5}, index=pd.date_range('2020-01-01', periods=5))
    dca = DollarCostAveraging()
    try:
        dca.calculate_returns(data, 1000)
        assert False, "Should raise ValueError for all-NaN close prices"
    except ValueError as e:
        assert "All close prices are NaN" in str(e)

def test_spy_buy_and_hold_missing_data(monkeypatch):
    from backtest.benchmarks import SPYBuyAndHold
    # Patch yf.download to return empty DataFrame
    monkeypatch.setattr('yfinance.download', lambda *a, **k: pd.DataFrame())
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=pd.date_range('2020-01-01', periods=3))
    spy_bh = SPYBuyAndHold()
    try:
        spy_bh.calculate_returns(data, 1000)
        assert False, "Should raise ValueError for missing SPY data"
    except ValueError as e:
        assert "Could not download SPY data" in str(e)

def test_benchmarks_short_series():
    from backtest.benchmarks import BuyAndHold, DollarCostAveraging
    data = pd.DataFrame({'Close': [100]}, index=pd.date_range('2020-01-01', periods=1))
    bh = BuyAndHold()
    dca = DollarCostAveraging()
    # Should not raise error, just return start_capital for BuyAndHold
    result = bh.calculate_returns(data, 1000)
    assert result.iloc[0] == 1000
    # DCA should also not raise error
    result = dca.calculate_returns(data, 1000)
    assert result.iloc[0] >= 0 