import pandas as pd
import pytest

from backtest.strategy import RSIStrategy
from backtest.validation import ValidationError


class TestRSIStrategy:
    """Tests for RSIStrategy."""

    def test_initialization_with_defaults(self):
        """Test strategy initializes with default parameters."""
        strategy = RSIStrategy()
        assert strategy.period == 14
        assert strategy.lower_bound == 30
        assert strategy.upper_bound == 70

    def test_initialization_with_custom_parameters(self):
        """Test strategy initializes with custom parameters."""
        strategy = RSIStrategy(period=10, lower_bound=25, upper_bound=75)
        assert strategy.period == 10
        assert strategy.lower_bound == 25
        assert strategy.upper_bound == 75

    def test_invalid_period(self):
        """Test validation fails for invalid period."""
        with pytest.raises(ValidationError, match="period"):
            RSIStrategy(period=0)

        with pytest.raises(ValidationError, match="period"):
            RSIStrategy(period=-5)

    def test_invalid_lower_bound(self):
        """Test validation fails for invalid lower_bound."""
        with pytest.raises(ValidationError, match="lower_bound"):
            RSIStrategy(lower_bound=-10)

        with pytest.raises(ValidationError, match="lower_bound"):
            RSIStrategy(lower_bound=101)

    def test_invalid_upper_bound(self):
        """Test validation fails for invalid upper_bound."""
        with pytest.raises(ValidationError, match="upper_bound"):
            RSIStrategy(upper_bound=-10)

        with pytest.raises(ValidationError, match="upper_bound"):
            RSIStrategy(upper_bound=101)

    def test_lower_bound_greater_than_upper_bound(self):
        """Test validation fails when lower_bound >= upper_bound."""
        with pytest.raises(ValidationError, match="lower_bound.*less than.*upper_bound"):
            RSIStrategy(lower_bound=70, upper_bound=30)

        with pytest.raises(ValidationError, match="lower_bound.*less than.*upper_bound"):
            RSIStrategy(lower_bound=50, upper_bound=50)

    def test_warmup_period(self):
        """Test warmup_period returns period."""
        strategy = RSIStrategy(period=20)
        assert strategy.warmup_period == 20

    def test_get_parameters(self):
        """Test get_parameters returns current parameters."""
        strategy = RSIStrategy(period=10, lower_bound=25, upper_bound=75)
        params = strategy.get_parameters()
        assert params == {"period": 10, "lower_bound": 25, "upper_bound": 75}

    def test_set_parameters(self):
        """Test set_parameters updates parameters correctly."""
        strategy = RSIStrategy()
        strategy.set_parameters({"period": 20, "lower_bound": 20, "upper_bound": 80})
        assert strategy.period == 20
        assert strategy.lower_bound == 20
        assert strategy.upper_bound == 80

    def test_set_parameters_partial_update(self):
        """Test set_parameters with partial parameter dict."""
        strategy = RSIStrategy(period=14, lower_bound=30, upper_bound=70)
        strategy.set_parameters({"period": 10})
        assert strategy.period == 10
        assert strategy.lower_bound == 30  # Unchanged
        assert strategy.upper_bound == 70  # Unchanged

    def test_set_parameters_validation(self):
        """Test set_parameters validates new parameters."""
        strategy = RSIStrategy()
        with pytest.raises(ValidationError):
            strategy.set_parameters({"period": 0})

        with pytest.raises(ValidationError):
            strategy.set_parameters({"lower_bound": 80, "upper_bound": 20})

    def test_buy_signal_when_oversold(self):
        """Test buy signal when RSI drops below lower_bound."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Create downtrend then stabilize (should create oversold condition)
        prices = list(range(100, 70, -1)) + [70] * 20
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = RSIStrategy(period=14, lower_bound=30, upper_bound=70)
        signals = strategy.generate_signals(data)

        assert "buy" in signals.columns
        assert "sell" in signals.columns
        # Should have buy signals when RSI < 30
        assert signals["buy"].sum() > 0

    def test_sell_signal_when_overbought(self):
        """Test sell signal when RSI exceeds upper_bound."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Create uptrend then stabilize (should create overbought condition)
        prices = list(range(50, 80)) + [80] * 20
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = RSIStrategy(period=14, lower_bound=30, upper_bound=70)
        signals = strategy.generate_signals(data)

        assert "sell" in signals.columns
        # Should have sell signals when RSI > 70
        assert signals["sell"].sum() > 0

    def test_no_signals_in_warmup_period(self):
        """Test no signals generated during warmup period."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        prices = list(range(100, 70, -1))
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = RSIStrategy(period=14)
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

        strategy = RSIStrategy(period=14)
        signals = strategy.generate_signals(data)

        # First row after shift should always be False
        assert not signals["buy"].iloc[0]
        assert not signals["sell"].iloc[0]

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        strategy = RSIStrategy()
        data = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            strategy.generate_signals(data)

    def test_insufficient_data(self):
        """Test validation fails when data has fewer rows than period."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = pd.DataFrame({"Close": range(1, 11)}, index=dates)

        strategy = RSIStrategy(period=14)
        with pytest.raises(ValidationError, match="at least.*14"):
            strategy.generate_signals(data)

    def test_missing_close_column(self):
        """Test validation fails when Close column is missing."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"Open": range(1, 31)}, index=dates)

        strategy = RSIStrategy()
        with pytest.raises(ValidationError, match="Close"):
            strategy.generate_signals(data)

    def test_nan_prices(self):
        """Test validation fails with all NaN prices."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"Close": [float("nan")] * 30}, index=dates)

        strategy = RSIStrategy()
        with pytest.raises(ValidationError, match="NaN"):
            strategy.generate_signals(data)

    def test_returns_same_index_as_input(self):
        """Test output signals have same index as input data."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"Close": range(1, 31)}, index=dates)

        strategy = RSIStrategy()
        signals = strategy.generate_signals(data)

        assert signals.index.equals(data.index)
        assert len(signals) == len(data)
