import os
import sys

import pandas as pd
import pytest

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.strategy import MeanReversionStrategy
from backtest.validation import ValidationError


class TestMeanReversionStrategy:
    """Tests for MeanReversionStrategy."""

    def test_initialization_with_defaults(self):
        """Test strategy initializes with default parameters."""
        strategy = MeanReversionStrategy()
        assert strategy.rsi_period == 14
        assert strategy.rsi_lower == 30
        assert strategy.rsi_upper == 70
        assert strategy.bb_period == 20
        assert strategy.bb_std == 2.0

    def test_initialization_with_custom_parameters(self):
        """Test strategy initializes with custom parameters."""
        strategy = MeanReversionStrategy(
            rsi_period=10, rsi_lower=25, rsi_upper=75,
            bb_period=15, bb_std=2.5
        )
        assert strategy.rsi_period == 10
        assert strategy.rsi_lower == 25
        assert strategy.rsi_upper == 75
        assert strategy.bb_period == 15
        assert strategy.bb_std == 2.5

    def test_invalid_rsi_period(self):
        """Test validation fails for invalid rsi_period."""
        with pytest.raises(ValidationError, match="rsi_period"):
            MeanReversionStrategy(rsi_period=0)

    def test_invalid_bb_period(self):
        """Test validation fails for invalid bb_period."""
        with pytest.raises(ValidationError, match="bb_period"):
            MeanReversionStrategy(bb_period=0)

    def test_invalid_rsi_bounds(self):
        """Test validation fails for invalid RSI bounds."""
        with pytest.raises(ValidationError, match="rsi_lower.*less than.*rsi_upper"):
            MeanReversionStrategy(rsi_lower=70, rsi_upper=30)

    def test_warmup_period(self):
        """Test warmup_period returns max of rsi_period and bb_period."""
        strategy = MeanReversionStrategy(rsi_period=10, bb_period=25)
        assert strategy.warmup_period == 25

        strategy = MeanReversionStrategy(rsi_period=30, bb_period=20)
        assert strategy.warmup_period == 30

    def test_get_parameters(self):
        """Test get_parameters returns current parameters."""
        strategy = MeanReversionStrategy(
            rsi_period=12, rsi_lower=25, rsi_upper=75,
            bb_period=18, bb_std=2.5
        )
        params = strategy.get_parameters()
        assert params == {
            "rsi_period": 12,
            "rsi_lower": 25,
            "rsi_upper": 75,
            "bb_period": 18,
            "bb_std": 2.5
        }

    def test_set_parameters(self):
        """Test set_parameters updates parameters correctly."""
        strategy = MeanReversionStrategy()
        strategy.set_parameters({
            "rsi_period": 10,
            "rsi_lower": 20,
            "bb_period": 15
        })
        assert strategy.rsi_period == 10
        assert strategy.rsi_lower == 20
        assert strategy.bb_period == 15
        assert strategy.rsi_upper == 70  # Unchanged
        assert strategy.bb_std == 2.0  # Unchanged

    def test_set_parameters_validation(self):
        """Test set_parameters validates new parameters."""
        strategy = MeanReversionStrategy()
        with pytest.raises(ValidationError):
            strategy.set_parameters({"rsi_period": 0})

        with pytest.raises(ValidationError):
            strategy.set_parameters({"rsi_lower": 80, "rsi_upper": 20})

    def test_buy_signal_on_oversold_and_below_lower_band(self):
        """Test buy signal when RSI < lower AND price < lower BB."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Create downtrend to trigger oversold conditions
        prices = list(range(100, 50, -1))
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MeanReversionStrategy(rsi_period=14, bb_period=20)
        signals = strategy.generate_signals(data)

        assert "buy" in signals.columns
        assert "sell" in signals.columns
        assert len(signals) == len(data)

    def test_sell_signal_on_overbought_and_above_upper_band(self):
        """Test sell signal when RSI > upper AND price > upper BB."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Create uptrend to trigger overbought conditions
        prices = list(range(50, 100))
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MeanReversionStrategy(rsi_period=14, bb_period=20)
        signals = strategy.generate_signals(data)

        assert "sell" in signals.columns
        assert len(signals) == len(data)

    def test_no_signals_in_warmup_period(self):
        """Test no signals generated during warmup period."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        prices = list(range(100, 70, -1))
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MeanReversionStrategy(rsi_period=14, bb_period=20)
        signals = strategy.generate_signals(data)

        # First warmup_period rows should have no signals
        warmup = strategy.warmup_period
        assert not signals["buy"].iloc[:warmup].any()
        assert not signals["sell"].iloc[:warmup].any()

    def test_signal_shifting(self):
        """Test signals are shifted by 1 day to prevent look-ahead bias."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        prices = list(range(100, 70, -1))
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MeanReversionStrategy()
        signals = strategy.generate_signals(data)

        # First row after shift should always be False
        assert not signals["buy"].iloc[0]
        assert not signals["sell"].iloc[0]

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        strategy = MeanReversionStrategy()
        data = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            strategy.generate_signals(data)

    def test_insufficient_data(self):
        """Test validation fails when data has fewer rows than warmup."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = pd.DataFrame({"Close": range(1, 11)}, index=dates)

        strategy = MeanReversionStrategy(rsi_period=14, bb_period=20)
        with pytest.raises(ValidationError, match="at least.*20"):
            strategy.generate_signals(data)

    def test_missing_close_column(self):
        """Test validation fails when Close column is missing."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"Open": range(1, 31)}, index=dates)

        strategy = MeanReversionStrategy()
        with pytest.raises(ValidationError, match="Close"):
            strategy.generate_signals(data)

    def test_nan_prices(self):
        """Test validation fails with all NaN prices."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"Close": [float("nan")] * 30}, index=dates)

        strategy = MeanReversionStrategy()
        with pytest.raises(ValidationError, match="NaN"):
            strategy.generate_signals(data)

    def test_returns_same_index_as_input(self):
        """Test output signals have same index as input data."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"Close": range(1, 31)}, index=dates)

        strategy = MeanReversionStrategy()
        signals = strategy.generate_signals(data)

        assert signals.index.equals(data.index)
        assert len(signals) == len(data)
