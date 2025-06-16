import pytest
import pandas as pd

# Add project root to the Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.strategy import ConsecutiveDaysStrategy

@pytest.fixture
def sample_data():
    """Create a sample price series for testing."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08'])
    prices = [100, 98, 96, 95, 97, 99, 101, 103] # 3 down, then 3 up
    return pd.DataFrame({'Close': prices}, index=dates)

def test_buy_signal_after_consecutive_down_days(sample_data):
    """Test that a buy signal is generated after N consecutive down days."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=3)
    signals = strategy.generate_signals(sample_data)
    # Expect buy signal on the day AFTER the 3rd down day.
    # Day 4 is the 3rd down day, so signal is for day 5.
    assert signals['buy'].iloc[4]
    assert signals['buy'].sum() == 1

def test_sell_signal_after_consecutive_up_days(sample_data):
    """Test that a sell signal is generated after N consecutive up days."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=3)
    signals = strategy.generate_signals(sample_data)
    # Expect sell signal on the day AFTER the 3rd up day.
    # Day 8 is the 3rd up day, so signal is for day 9 (which would be index 8)
    # The signal for day 8's close should be True on index 7.
    assert signals['sell'].iloc[7]
    assert signals['sell'].sum() == 1

def test_no_signal_if_not_consecutive(sample_data):
    """Test that no signal is generated if the trend is broken."""
    # Prices: [100, 98, 96, 95, 97, 99, 101, 103]
    # To break the 3-day downtrend, we modify the 3rd day.
    sample_data.loc[sample_data.index[2], 'Close'] = 99 # Now: [100, 98, 99, 95...]
    strategy = ConsecutiveDaysStrategy(consecutive_days=3)
    signals = strategy.generate_signals(sample_data)
    assert signals['buy'].sum() == 0

def test_invalid_days_parameter_raises_error():
    """Test that initializing with zero or negative days raises a ValueError."""
    with pytest.raises(ValueError):
        ConsecutiveDaysStrategy(consecutive_days=0)
    with pytest.raises(ValueError):
        ConsecutiveDaysStrategy(consecutive_days=-1)

def test_signal_shifting(sample_data):
    """Test that signals are correctly shifted by one day."""
    strategy = ConsecutiveDaysStrategy(consecutive_days=3)
    # A signal is detected on day 4 (index 3) based on prices[1,2,3]
    # After shifting, this signal should appear on day 5 (index 4)
    signals = strategy.generate_signals(sample_data)
    assert not signals['buy'].iloc[3] # Day of detection
    assert signals['buy'].iloc[4]  # Day of trade 