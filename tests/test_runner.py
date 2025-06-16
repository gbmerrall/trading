import pytest
import pandas as pd
import os
import yfinance as yf
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add project root to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.runner import BacktestRunnerImpl
from backtest.strategy import ConsecutiveDaysStrategy
from backtest.benchmarks import BuyAndHold
from backtest.validation import ValidationError


# ==================== EXISTING TESTS ====================

def test_runner_executes_without_error_on_real_data(monkeypatch):
    """
    Test that the backtest runner can execute with a real-world
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
        assert "strategy_metrics" in results
        assert "benchmark_metrics" in results
        assert "strategy_returns" in results
        assert "benchmark_returns" in results
        assert "signals" in results
    except Exception as e:
        pytest.fail(f"Backtest runner failed with real data: {e}")


def test_runner_validation_with_invalid_strategy():
    """Test that runner properly validates strategy objects."""
    class InvalidStrategy:
        pass  # Missing generate_signals method
    
    with pytest.raises(ValidationError, match="must have 'generate_signals' method"):
        BacktestRunnerImpl(strategy=InvalidStrategy(), benchmarks=[])


def test_runner_validation_with_invalid_benchmarks():
    """Test that runner properly validates benchmark objects."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=3)
    
    class InvalidBenchmark:
        pass  # Missing calculate_returns method
    
    with pytest.raises(ValidationError, match="must have 'calculate_returns' method"):
        BacktestRunnerImpl(strategy=strategy, benchmarks=[InvalidBenchmark()])


def test_runner_handles_empty_benchmark_list():
    """Test that runner works with empty benchmark list."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    # Create minimal test data
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 95, 90]}, index=dates)
    
    results = runner.run(data, start_capital=10000.0)
    
    assert results is not None
    assert len(results["benchmark_metrics"]) == 0
    assert len(results["benchmark_returns"]) == 0

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


# ==================== NEW COMPREHENSIVE TESTS ====================

# ==================== INITIALIZATION TESTS ====================

def test_runner_initialization_with_non_list_benchmarks():
    """Test that runner rejects non-list benchmarks."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    
    with pytest.raises(ValidationError, match="Benchmarks must be a list"):
        BacktestRunnerImpl(strategy, benchmarks="not_a_list")


def test_runner_initialization_with_multiple_invalid_benchmarks():
    """Test that runner validates all benchmarks in list."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    
    class InvalidBenchmark1:
        pass
    
    class InvalidBenchmark2:
        pass
    
    with pytest.raises(ValidationError, match="Benchmark 0 must have 'calculate_returns' method"):
        BacktestRunnerImpl(strategy, benchmarks=[InvalidBenchmark1(), BuyAndHold()])


# ==================== DATA VALIDATION TESTS ====================

def test_validate_data_with_missing_close_column():
    """Test data validation with missing Close column."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02'])
    data = pd.DataFrame({'Open': [100, 101]}, index=dates)
    
    with pytest.raises(ValidationError, match="Missing required columns"):
        runner.run(data, start_capital=10000.0)


def test_validate_data_with_insufficient_rows():
    """Test data validation with insufficient data points."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01'])
    data = pd.DataFrame({'Close': [100]}, index=dates)
    
    with pytest.raises(ValidationError, match="must have at least"):
        runner.run(data, start_capital=10000.0)


def test_validate_data_with_duplicate_dates():
    """Test data validation with duplicate dates."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02'])
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=dates)
    
    with pytest.raises(ValidationError, match="duplicate dates"):
        runner.run(data, start_capital=10000.0)


def test_validate_data_with_unsorted_dates():
    """Test data validation with unsorted dates."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-03', '2023-01-01', '2023-01-02'])
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=dates)
    
    with pytest.raises(ValidationError, match="must be sorted by date"):
        runner.run(data, start_capital=10000.0)


def test_validate_data_with_negative_prices():
    """Test data validation with negative prices."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, -50, 102]}, index=dates)
    
    with pytest.raises(ValidationError, match="non-positive values"):
        runner.run(data, start_capital=10000.0)


def test_validate_data_with_all_nan_prices():
    """Test data validation with all NaN prices."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [np.nan, np.nan, np.nan]}, index=dates)
    
    with pytest.raises(ValidationError, match="All values.*are NaN"):
        runner.run(data, start_capital=10000.0)


# ==================== PARAMETER VALIDATION TESTS ====================

def test_validate_negative_start_capital():
    """Test validation of negative start capital."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=dates)
    
    with pytest.raises(ValidationError, match="start_capital must be"):
        runner.run(data, start_capital=-1000.0)


def test_validate_zero_start_capital():
    """Test validation of zero start capital."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=dates)
    
    with pytest.raises(ValidationError, match="start_capital must be"):
        runner.run(data, start_capital=0.0)


def test_validate_invalid_output_file_extension():
    """Test validation of invalid output file extension."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=dates)
    
    with pytest.raises(ValidationError, match="not allowed"):
        runner.run(data, start_capital=10000.0, output_file="test.txt")


def test_validate_output_file_with_dangerous_characters():
    """Test validation of output file with dangerous characters."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=dates)
    
    with pytest.raises(ValidationError, match="dangerous character"):
        runner.run(data, start_capital=10000.0, output_file="../test.png")


# ==================== STRATEGY SIGNAL TESTS ====================

def test_strategy_with_empty_signals():
    """Test handling of strategy that generates empty signals."""
    class EmptySignalStrategy:
        def generate_signals(self, data):
            return pd.DataFrame()  # Empty DataFrame
    
    strategy = EmptySignalStrategy()
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=dates)
    
    with pytest.raises(ValidationError, match="empty signals"):
        runner.run(data, start_capital=10000.0)


def test_strategy_with_missing_signal_columns():
    """Test handling of strategy with missing buy/sell columns."""
    class InvalidSignalStrategy:
        def generate_signals(self, data):
            return pd.DataFrame({'invalid_column': [True, False, True]})
    
    strategy = InvalidSignalStrategy()
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=dates)
    
    with pytest.raises(ValidationError, match="must contain 'buy' and 'sell' columns"):
        runner.run(data, start_capital=10000.0)


def test_strategy_with_partial_signal_columns():
    """Test handling of strategy with only buy column."""
    class PartialSignalStrategy:
        def generate_signals(self, data):
            return pd.DataFrame({'buy': [True, False, True]})  # Missing 'sell'
    
    strategy = PartialSignalStrategy()
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=dates)
    
    with pytest.raises(ValidationError, match="must contain 'buy' and 'sell' columns"):
        runner.run(data, start_capital=10000.0)


# ==================== TRADING LOGIC TESTS ====================

def test_trading_with_nan_prices():
    """Test trading logic handles NaN prices correctly."""
    class SimpleStrategy:
        def generate_signals(self, data):
            signals = pd.DataFrame(index=data.index)
            signals['buy'] = [True, False, False]
            signals['sell'] = [False, False, True]
            return signals
    
    strategy = SimpleStrategy()
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, np.nan, 102]}, index=dates)
    
    results = runner.run(data, start_capital=10000.0)
    
    # Should handle NaN prices gracefully
    assert results is not None
    assert 'strategy_metrics' in results


def test_trading_with_very_small_prices():
    """Test trading logic handles very small positive prices correctly."""
    class SimpleStrategy:
        def generate_signals(self, data):
            signals = pd.DataFrame(index=data.index)
            signals['buy'] = [True, False, False]
            signals['sell'] = [False, False, True]
            return signals
    
    strategy = SimpleStrategy()
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    # Use prices that don't trigger extreme variation validation (max/min < 1000)
    data = pd.DataFrame({'Close': [1.0, 0.5, 1.2]}, index=dates)  # Ratio is only 2.4
    
    results = runner.run(data, start_capital=10000.0)
    
    # Should handle small prices gracefully
    assert results is not None
    assert 'strategy_metrics' in results


def test_multiple_buy_signals_without_sell():
    """Test handling of multiple buy signals without intervening sell."""
    class MultipleBuyStrategy:
        def generate_signals(self, data):
            signals = pd.DataFrame(index=data.index)
            signals['buy'] = [True, True, False, False]
            signals['sell'] = [False, False, False, False]
            return signals
    
    strategy = MultipleBuyStrategy()
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
    data = pd.DataFrame({'Close': [100, 101, 102, 103]}, index=dates)
    
    results = runner.run(data, start_capital=10000.0)
    
    # Should only execute first buy signal
    assert results['strategy_metrics']['num_trades'] == 1  # Position closed at end


def test_sell_signal_without_position():
    """Test handling of sell signal when no position is held."""
    class SellWithoutBuyStrategy:
        def generate_signals(self, data):
            signals = pd.DataFrame(index=data.index)
            signals['buy'] = [False, False, False]
            signals['sell'] = [True, False, False]
            return signals
    
    strategy = SellWithoutBuyStrategy()
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=dates)
    
    results = runner.run(data, start_capital=10000.0)
    
    # Should handle gracefully with no trades
    assert results['strategy_metrics']['num_trades'] == 0


def test_insufficient_cash_for_trade():
    """Test handling when insufficient cash for trade."""
    class SimpleStrategy:
        def generate_signals(self, data):
            signals = pd.DataFrame(index=data.index)
            signals['buy'] = [True, False, False]
            signals['sell'] = [False, False, False]
            return signals
    
    strategy = SimpleStrategy()
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100000, 101, 102]}, index=dates)  # Very high first price
    
    results = runner.run(data, start_capital=1000.0)  # Low capital
    
    # Should handle gracefully with no trades
    assert results['strategy_metrics']['num_trades'] == 0


# ==================== BENCHMARK TESTS ====================

def test_benchmark_calculation_error():
    """Test handling of benchmark calculation errors."""
    class FailingBenchmark:
        def calculate_returns(self, data, start_capital):
            raise Exception("Benchmark calculation failed")
    
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    runner = BacktestRunnerImpl(strategy, benchmarks=[FailingBenchmark()])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=dates)
    
    with pytest.raises(ValidationError, match="Error calculating.*benchmark"):
        runner.run(data, start_capital=10000.0)


def test_multiple_benchmarks():
    """Test runner with multiple benchmarks."""
    class MockBenchmark1:
        def calculate_returns(self, data, start_capital):
            return pd.Series([start_capital, start_capital * 1.1], index=data.index[:2])
    
    class MockBenchmark2:
        def calculate_returns(self, data, start_capital):
            return pd.Series([start_capital, start_capital * 1.05], index=data.index[:2])
    
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    benchmarks = [MockBenchmark1(), MockBenchmark2()]
    runner = BacktestRunnerImpl(strategy, benchmarks)
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 95, 90]}, index=dates)
    
    results = runner.run(data, start_capital=10000.0)
    
    assert len(results['benchmark_metrics']) == 2
    assert len(results['benchmark_returns']) == 2


# ==================== METRICS CALCULATION TESTS ====================

def test_metrics_with_empty_portfolio_history():
    """Test metrics calculation with empty portfolio history."""
    runner = BacktestRunnerImpl(ConsecutiveDaysStrategy(2), [])
    
    metrics = runner._calculate_metrics([], [])
    
    assert metrics['total_return'] == 0.0
    assert metrics['win_rate'] == 0.0
    assert metrics['num_trades'] == 0
    assert metrics['max_drawdown'] == 0.0


def test_metrics_with_invalid_start_value():
    """Test metrics calculation with invalid start value."""
    runner = BacktestRunnerImpl(ConsecutiveDaysStrategy(2), [])
    
    portfolio_history = [
        {'date': pd.Timestamp('2023-01-01'), 'value': 0.0},  # Invalid start value
        {'date': pd.Timestamp('2023-01-02'), 'value': 1000.0}
    ]
    
    with pytest.raises(ValidationError, match="Invalid start value"):
        runner._calculate_metrics(portfolio_history, [])


def test_metrics_with_losing_trades():
    """Test metrics calculation with losing trades."""
    runner = BacktestRunnerImpl(ConsecutiveDaysStrategy(2), [])
    
    portfolio_history = [
        {'date': pd.Timestamp('2023-01-01'), 'value': 10000.0},
        {'date': pd.Timestamp('2023-01-02'), 'value': 9000.0}
    ]
    
    trades = [
        {'pnl': -500.0},
        {'pnl': -300.0},
        {'pnl': 100.0}
    ]
    
    metrics = runner._calculate_metrics(portfolio_history, trades)
    
    assert metrics['total_return'] < 0
    assert abs(metrics['win_rate'] - (100.0 / 3)) < 0.01  # 1 win out of 3 trades, allow for floating point
    assert metrics['num_trades'] == 3


def test_metrics_calculation_exception():
    """Test metrics calculation with exception."""
    runner = BacktestRunnerImpl(ConsecutiveDaysStrategy(2), [])
    
    # Invalid portfolio history that will cause pandas error
    portfolio_history = [{'invalid': 'data'}]
    
    with pytest.raises(ValidationError, match="Error calculating metrics"):
        runner._calculate_metrics(portfolio_history, [])


# ==================== PLOTTING TESTS ====================

@patch('backtest.runner.go.Figure')
def test_plot_creation_success(mock_figure):
    """Test successful plot creation."""
    mock_fig = MagicMock()
    mock_figure.return_value = mock_fig
    
    runner = BacktestRunnerImpl(ConsecutiveDaysStrategy(2), [])
    
    strategy_history = [
        {'date': pd.Timestamp('2023-01-01'), 'value': 10000.0},
        {'date': pd.Timestamp('2023-01-02'), 'value': 10500.0}
    ]
    
    benchmark_returns = {
        'BuyAndHold': pd.Series([10000.0, 10200.0], 
                               index=pd.to_datetime(['2023-01-01', '2023-01-02']))
    }
    
    # Should not raise exception
    runner._plot_equity_curves(strategy_history, benchmark_returns, "test.png")
    
    mock_fig.add_trace.assert_called()
    mock_fig.write_image.assert_called_with("test.png", engine="kaleido")


def test_plot_with_empty_strategy_history():
    """Test plot creation with empty strategy history."""
    runner = BacktestRunnerImpl(ConsecutiveDaysStrategy(2), [])
    
    with pytest.raises(ValidationError, match="Strategy history is empty"):
        runner._plot_equity_curves([], {}, "test.png")


@patch('backtest.runner.go.Figure')
def test_plot_creation_exception(mock_figure):
    """Test plot creation with exception."""
    mock_figure.side_effect = Exception("Plot creation failed")
    
    runner = BacktestRunnerImpl(ConsecutiveDaysStrategy(2), [])
    
    strategy_history = [
        {'date': pd.Timestamp('2023-01-01'), 'value': 10000.0}
    ]
    
    with pytest.raises(ValidationError, match="Error creating plot"):
        runner._plot_equity_curves(strategy_history, {}, "test.png")


# ==================== CONFIGURATION TESTS ====================

def test_run_with_default_start_capital():
    """Test runner uses config default when start_capital is None."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=2)
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 95, 90]}, index=dates)
    
    results = runner.run(data, start_capital=None)  # Use config default
    
    assert results is not None
    assert 'strategy_metrics' in results


# ==================== INTEGRATION TESTS ====================

def test_complete_trading_cycle():
    """Test complete trading cycle with buy, hold, and sell."""
    class CompleteCycleStrategy:
        def generate_signals(self, data):
            signals = pd.DataFrame(index=data.index)
            signals['buy'] = [True, False, False, False, False]
            signals['sell'] = [False, False, False, True, False]
            return signals
    
    strategy = CompleteCycleStrategy()
    runner = BacktestRunnerImpl(strategy, benchmarks=[BuyAndHold()])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    data = pd.DataFrame({'Close': [100, 101, 102, 105, 106]}, index=dates)
    
    results = runner.run(data, start_capital=10000.0)
    
    assert results['strategy_metrics']['num_trades'] == 1
    assert results['strategy_metrics']['total_return'] > 0
    assert len(results['trades']) == 1
    assert results['trades'][0]['pnl'] > 0


def test_no_trading_opportunities():
    """Test scenario with no trading opportunities."""
    class NoTradeStrategy:
        def generate_signals(self, data):
            signals = pd.DataFrame(index=data.index)
            signals['buy'] = [False] * len(data)
            signals['sell'] = [False] * len(data)
            return signals
    
    strategy = NoTradeStrategy()
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=dates)
    
    results = runner.run(data, start_capital=10000.0)
    
    assert results['strategy_metrics']['num_trades'] == 0
    assert results['strategy_metrics']['total_return'] == 0.0
    assert len(results['trades']) == 0


def test_close_position_at_end():
    """Test that open position is closed at end of backtest."""
    class BuyOnlyStrategy:
        def generate_signals(self, data):
            signals = pd.DataFrame(index=data.index)
            signals['buy'] = [True, False, False]
            signals['sell'] = [False, False, False]  # Never sell
            return signals
    
    strategy = BuyOnlyStrategy()
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 101, 110]}, index=dates)
    
    results = runner.run(data, start_capital=10000.0)
    
    # Position should be closed at end, creating one trade
    assert results['strategy_metrics']['num_trades'] == 1
    assert results['trades'][0]['exit'] == 110  # Closed at last price


# ==================== ERROR HANDLING TESTS ====================

def test_general_exception_handling():
    """Test general exception handling in run method."""
    class ExceptionStrategy:
        def generate_signals(self, data):
            raise Exception("Strategy failed")
    
    strategy = ExceptionStrategy()
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=dates)
    
    with pytest.raises(ValidationError, match="Backtest execution failed"):
        runner.run(data, start_capital=10000.0)


def test_portfolio_history_validation():
    """Test validation of empty portfolio history."""
    class NoHistoryStrategy:
        def generate_signals(self, data):
            # Return valid signals but somehow portfolio history ends up empty
            signals = pd.DataFrame(index=data.index)
            signals['buy'] = [False] * len(data)
            signals['sell'] = [False] * len(data)
            return signals
    
    strategy = NoHistoryStrategy()
    runner = BacktestRunnerImpl(strategy, benchmarks=[])
    
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=dates)
    
    # Mock portfolio to return empty history
    with patch('backtest.runner.Portfolio') as mock_portfolio:
        mock_instance = mock_portfolio.return_value
        mock_instance.get_value_history.return_value = []
        mock_instance.cash = 10000.0
        mock_instance.process_day = MagicMock()
        
        with pytest.raises(ValidationError, match="No portfolio history generated"):
            runner.run(data, start_capital=10000.0)