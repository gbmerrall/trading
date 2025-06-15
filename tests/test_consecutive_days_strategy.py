import pandas as pd
import numpy as np
from backtest.strategies import ConsecutiveDaysStrategy

def create_test_data():
    """Create a sample price series for testing."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    prices = pd.Series([100, 99, 98, 97, 98, 99, 100, 99, 98, 97], index=dates)
    return prices

def test_detect_consecutive_down_days():
    """Test detection of 3 consecutive down days."""
    prices = create_test_data()
    strategy = ConsecutiveDaysStrategy(down_days=3)
    signals = strategy._detect_consecutive_down_days(prices)
    expected_indices = [3, 9]
    true_indices = list(np.where(signals)[0])
    assert true_indices == expected_indices, f"Expected signals at {expected_indices}, got {true_indices}"
    # All other days should be False
    for i in range(len(signals)):
        if i not in expected_indices:
            assert not signals.iloc[i].item()

def test_detect_consecutive_up_days():
    """Test detection of 3 consecutive up days."""
    prices = create_test_data()
    strategy = ConsecutiveDaysStrategy(up_days=3)
    signals = strategy._detect_consecutive_up_days(prices)
    assert signals.iloc[6].item()
    # All other days should be False
    for i in range(len(signals)):
        if i != 6:
            assert not signals.iloc[i].item()

def test_detect_consecutive_down_days_5_days():
    """Test detection of 5 consecutive down days."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    prices = pd.Series([100, 98, 96, 94, 92, 90, 95, 97, 99, 98], index=dates)
    strategy = ConsecutiveDaysStrategy(down_days=5)
    signals = strategy._detect_consecutive_down_days(prices)
    assert signals.iloc[5].item()
    for i in range(len(signals)):
        if i != 5:
            assert not signals.iloc[i].item()

def test_detect_consecutive_up_days_5_days():
    """Test detection of 5 consecutive up days."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    prices = pd.Series([100, 102, 104, 106, 108, 110, 109, 107, 105, 103], index=dates)
    strategy = ConsecutiveDaysStrategy(up_days=5)
    signals = strategy._detect_consecutive_up_days(prices)
    assert signals.iloc[5].item()
    for i in range(len(signals)):
        if i != 5:
            assert not signals.iloc[i].item()

def test_insufficient_data():
    """Test handling of insufficient data."""
    dates = pd.date_range(start='2024-01-01', periods=2, freq='D')
    prices = pd.Series([100, 99], index=dates)
    strategy = ConsecutiveDaysStrategy(up_days=3, down_days=3)
    signals = strategy.generate_signals(pd.DataFrame({'Close': prices}))
    assert not signals['buy'].any()
    assert not signals['sell'].any()

def test_invalid_input():
    """Test handling of invalid input data (negative prices)."""
    strategy = ConsecutiveDaysStrategy(up_days=3, down_days=3)
    dates = pd.date_range(start='2024-01-01', periods=3, freq='D')
    prices = pd.Series([100, -101, 102], index=dates)
    signals = strategy.generate_signals(pd.DataFrame({'Close': prices}))
    assert not signals['buy'].any()
    assert not signals['sell'].any()

def test_edge_cases():
    """Test edge cases for the strategy."""
    # Insufficient days
    prices = pd.Series([100, 95], index=pd.date_range('2024-01-01', periods=2))
    strategy = ConsecutiveDaysStrategy(up_days=3, down_days=3)
    signals = strategy.generate_signals(pd.DataFrame({'Close': prices}))
    assert not signals['buy'].any()
    assert not signals['sell'].any()

def test_missing_data():
    """Test handling of missing data."""
    prices = pd.Series([100, np.nan, 90, 85, 80], index=pd.date_range('2024-01-01', periods=5))
    strategy = ConsecutiveDaysStrategy(up_days=3, down_days=3)
    signals = strategy.generate_signals(pd.DataFrame({'Close': prices}))
    assert not signals['buy'].any()
    assert not signals['sell'].any() 