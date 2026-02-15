import os
import sys

import pandas as pd
import pytest

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.strategy import MovingAverageCrossoverStrategy
from backtest.validation import ValidationError


class TestMovingAverageCrossoverStrategy:
    """Tests for MovingAverageCrossoverStrategy."""

    def test_initialization_with_defaults(self):
        """Test strategy initializes with default parameters."""
        strategy = MovingAverageCrossoverStrategy()
        assert strategy.short_window == 20
        assert strategy.long_window == 50

    def test_initialization_with_custom_parameters(self):
        """Test strategy initializes with custom parameters."""
        strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=30)
        assert strategy.short_window == 10
        assert strategy.long_window == 30

    def test_invalid_short_window(self):
        """Test validation fails for invalid short_window."""
        with pytest.raises(ValidationError, match="short_window"):
            MovingAverageCrossoverStrategy(short_window=0)

        with pytest.raises(ValidationError, match="short_window"):
            MovingAverageCrossoverStrategy(short_window=-5)

    def test_invalid_long_window(self):
        """Test validation fails for invalid long_window."""
        with pytest.raises(ValidationError, match="long_window"):
            MovingAverageCrossoverStrategy(long_window=0)

        with pytest.raises(ValidationError, match="long_window"):
            MovingAverageCrossoverStrategy(long_window=-5)

    def test_short_window_greater_than_long_window(self):
        """Test validation fails when short_window >= long_window."""
        with pytest.raises(ValidationError, match="short_window.*less than.*long_window"):
            MovingAverageCrossoverStrategy(short_window=50, long_window=20)

        with pytest.raises(ValidationError, match="short_window.*less than.*long_window"):
            MovingAverageCrossoverStrategy(short_window=30, long_window=30)

    def test_warmup_period(self):
        """Test warmup_period returns long_window."""
        strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=30)
        assert strategy.warmup_period == 30

    def test_get_parameters(self):
        """Test get_parameters returns current parameters."""
        strategy = MovingAverageCrossoverStrategy(short_window=15, long_window=40)
        params = strategy.get_parameters()
        assert params == {"short_window": 15, "long_window": 40}

    def test_set_parameters(self):
        """Test set_parameters updates parameters correctly."""
        strategy = MovingAverageCrossoverStrategy()
        strategy.set_parameters({"short_window": 12, "long_window": 26})
        assert strategy.short_window == 12
        assert strategy.long_window == 26

    def test_set_parameters_partial_update(self):
        """Test set_parameters with partial parameter dict."""
        strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=30)
        strategy.set_parameters({"short_window": 15})
        assert strategy.short_window == 15
        assert strategy.long_window == 30  # Unchanged

    def test_set_parameters_validation(self):
        """Test set_parameters validates new parameters."""
        strategy = MovingAverageCrossoverStrategy()
        with pytest.raises(ValidationError):
            strategy.set_parameters({"short_window": 0})

        with pytest.raises(ValidationError):
            strategy.set_parameters({"short_window": 60, "long_window": 30})

    def test_buy_signal_on_golden_cross(self):
        """Test buy signal when short MA crosses above long MA (golden cross)."""
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        # Create downtrend then uptrend to trigger golden cross
        prices = list(range(100, 50, -1)) + list(range(50, 60))  # 50 down + 10 up = 60
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(data)

        # Buy signal should occur after golden cross (shifted by 1 day)
        # Find where short MA crosses above long MA
        assert "buy" in signals.columns
        assert "sell" in signals.columns
        assert signals["buy"].sum() > 0  # Should have at least one buy signal

    def test_sell_signal_on_death_cross(self):
        """Test sell signal when short MA crosses below long MA (death cross)."""
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        # Create uptrend then downtrend to trigger death cross
        prices = list(range(50, 100)) + list(range(100, 90, -1))  # 50 up + 10 down = 60
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(data)

        # Sell signal should occur after death cross (shifted by 1 day)
        assert "sell" in signals.columns
        assert signals["sell"].sum() > 0  # Should have at least one sell signal

    def test_no_signals_in_warmup_period(self):
        """Test no signals generated during warmup period."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        prices = list(range(100, 130))
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=20)
        signals = strategy.generate_signals(data)

        # First warmup_period rows should have no signals (due to shift + warmup)
        warmup = strategy.warmup_period
        assert not signals["buy"].iloc[:warmup].any()
        assert not signals["sell"].iloc[:warmup].any()

    def test_signal_shifting(self):
        """Test signals are shifted by 1 day to prevent look-ahead bias."""
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        # Simple pattern: down then up
        prices = [100] * 25 + [50] * 5 + list(range(50, 80))
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(data)

        # First row after shift should always be False
        assert not signals["buy"].iloc[0]
        assert not signals["sell"].iloc[0]

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        strategy = MovingAverageCrossoverStrategy()
        data = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            strategy.generate_signals(data)

    def test_insufficient_data(self):
        """Test validation fails when data has fewer rows than long_window."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = pd.DataFrame({"Close": range(10)}, index=dates)

        strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=20)
        with pytest.raises(ValidationError, match="at least.*20"):
            strategy.generate_signals(data)

    def test_missing_close_column(self):
        """Test validation fails when Close column is missing."""
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        data = pd.DataFrame({"Open": range(1, 61)}, index=dates)

        strategy = MovingAverageCrossoverStrategy()
        with pytest.raises(ValidationError, match="Close"):
            strategy.generate_signals(data)

    def test_nan_prices(self):
        """Test validation fails with all NaN prices."""
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        data = pd.DataFrame({"Close": [float("nan")] * 60}, index=dates)

        strategy = MovingAverageCrossoverStrategy()
        with pytest.raises(ValidationError, match="NaN"):
            strategy.generate_signals(data)

    def test_flat_prices_no_crossover(self):
        """Test no signals when prices are flat (no crossover)."""
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        data = pd.DataFrame({"Close": [100] * 60}, index=dates)

        strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(data)

        # Flat prices mean MAs are equal, no crossover
        assert signals["buy"].sum() == 0
        assert signals["sell"].sum() == 0

    def test_returns_same_index_as_input(self):
        """Test output signals have same index as input data."""
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        data = pd.DataFrame({"Close": range(1, 61)}, index=dates)

        strategy = MovingAverageCrossoverStrategy()
        signals = strategy.generate_signals(data)

        assert signals.index.equals(data.index)
        assert len(signals) == len(data)
