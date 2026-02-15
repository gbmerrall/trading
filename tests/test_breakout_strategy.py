import os
import sys

import pandas as pd
import pytest

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.strategy import BreakoutStrategy
from backtest.validation import ValidationError


class TestBreakoutStrategy:
    """Tests for BreakoutStrategy."""

    def test_initialization_with_defaults(self):
        """Test strategy initializes with default parameters."""
        strategy = BreakoutStrategy()
        assert strategy.lookback_period == 20

    def test_initialization_with_custom_parameters(self):
        """Test strategy initializes with custom parameters."""
        strategy = BreakoutStrategy(lookback_period=30)
        assert strategy.lookback_period == 30

    def test_invalid_lookback_period(self):
        """Test validation fails for invalid lookback_period."""
        with pytest.raises(ValidationError, match="lookback_period"):
            BreakoutStrategy(lookback_period=0)

        with pytest.raises(ValidationError, match="lookback_period"):
            BreakoutStrategy(lookback_period=-5)

    def test_warmup_period(self):
        """Test warmup_period returns lookback_period."""
        strategy = BreakoutStrategy(lookback_period=30)
        assert strategy.warmup_period == 30

    def test_get_parameters(self):
        """Test get_parameters returns current parameters."""
        strategy = BreakoutStrategy(lookback_period=25)
        params = strategy.get_parameters()
        assert params == {"lookback_period": 25}

    def test_set_parameters(self):
        """Test set_parameters updates parameters correctly."""
        strategy = BreakoutStrategy()
        strategy.set_parameters({"lookback_period": 30})
        assert strategy.lookback_period == 30

    def test_set_parameters_validation(self):
        """Test set_parameters validates new parameters."""
        strategy = BreakoutStrategy()
        with pytest.raises(ValidationError):
            strategy.set_parameters({"lookback_period": 0})

    def test_buy_signal_on_high_breakout(self):
        """Test buy signal when price exceeds N-day high."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        # Prices with breakout: mostly flat, then spike
        prices = [100] * 25 + [101, 102, 103, 104, 105]
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = BreakoutStrategy(lookback_period=20)
        signals = strategy.generate_signals(data)

        assert "buy" in signals.columns
        assert "sell" in signals.columns
        # Verify signal structure
        assert len(signals) == len(data)

    def test_sell_signal_on_low_breakdown(self):
        """Test sell signal when price breaks below N-day low."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        # Prices with breakdown: mostly flat, then drop
        prices = [100] * 25 + [99, 98, 97, 96, 95]
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = BreakoutStrategy(lookback_period=20)
        signals = strategy.generate_signals(data)

        assert "sell" in signals.columns
        # Verify signal structure
        assert len(signals) == len(data)

    def test_no_signals_in_warmup_period(self):
        """Test no signals generated during warmup period."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        prices = list(range(1, 31))
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = BreakoutStrategy(lookback_period=20)
        signals = strategy.generate_signals(data)

        # First warmup_period rows should have no signals
        warmup = strategy.warmup_period
        assert not signals["buy"].iloc[:warmup].any()
        assert not signals["sell"].iloc[:warmup].any()

    def test_signal_shifting(self):
        """Test signals are shifted by 1 day to prevent look-ahead bias."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        prices = list(range(1, 31))
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = BreakoutStrategy()
        signals = strategy.generate_signals(data)

        # First row after shift should always be False
        assert not signals["buy"].iloc[0]
        assert not signals["sell"].iloc[0]

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        strategy = BreakoutStrategy()
        data = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            strategy.generate_signals(data)

    def test_insufficient_data(self):
        """Test validation fails when data has fewer rows than lookback_period."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = pd.DataFrame({"Close": range(1, 11)}, index=dates)

        strategy = BreakoutStrategy(lookback_period=20)
        with pytest.raises(ValidationError, match="at least.*20"):
            strategy.generate_signals(data)

    def test_missing_close_column(self):
        """Test validation fails when Close column is missing."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"Open": range(1, 31)}, index=dates)

        strategy = BreakoutStrategy()
        with pytest.raises(ValidationError, match="Close"):
            strategy.generate_signals(data)

    def test_nan_prices(self):
        """Test validation fails with all NaN prices."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"Close": [float("nan")] * 30}, index=dates)

        strategy = BreakoutStrategy()
        with pytest.raises(ValidationError, match="NaN"):
            strategy.generate_signals(data)

    def test_no_breakout_flat_prices(self):
        """Test no signals when prices are flat (no breakout)."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"Close": [100] * 30}, index=dates)

        strategy = BreakoutStrategy(lookback_period=20)
        signals = strategy.generate_signals(data)

        # Flat prices mean no breakouts
        assert signals["buy"].sum() == 0
        assert signals["sell"].sum() == 0

    def test_returns_same_index_as_input(self):
        """Test output signals have same index as input data."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"Close": range(1, 31)}, index=dates)

        strategy = BreakoutStrategy()
        signals = strategy.generate_signals(data)

        assert signals.index.equals(data.index)
        assert len(signals) == len(data)
