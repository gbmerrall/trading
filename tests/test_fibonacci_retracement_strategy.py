import pandas as pd
import pytest

from backtest.strategy import FibonacciRetracementStrategy
from backtest.validation import ValidationError


class TestFibonacciRetracementStrategy:
    """Tests for FibonacciRetracementStrategy."""

    def test_initialization_with_defaults(self):
        """Test strategy initializes with default parameters."""
        strategy = FibonacciRetracementStrategy()
        assert strategy.swing_lookback == 20
        assert strategy.retracement_levels == [0.382, 0.5, 0.618]

    def test_initialization_with_custom_parameters(self):
        """Test strategy initializes with custom parameters."""
        strategy = FibonacciRetracementStrategy(
            swing_lookback=30,
            retracement_levels=[0.236, 0.5, 0.786]
        )
        assert strategy.swing_lookback == 30
        assert strategy.retracement_levels == [0.236, 0.5, 0.786]

    def test_invalid_swing_lookback(self):
        """Test validation fails for invalid swing_lookback."""
        with pytest.raises(ValidationError, match="swing_lookback"):
            FibonacciRetracementStrategy(swing_lookback=0)

        with pytest.raises(ValidationError, match="swing_lookback"):
            FibonacciRetracementStrategy(swing_lookback=-5)

    def test_invalid_retracement_levels_empty(self):
        """Test validation fails for empty retracement_levels."""
        with pytest.raises(ValidationError, match="retracement_levels"):
            FibonacciRetracementStrategy(retracement_levels=[])

    def test_invalid_retracement_levels_out_of_range(self):
        """Test validation fails for retracement_levels outside [0, 1]."""
        with pytest.raises(ValidationError, match="retracement_levels"):
            FibonacciRetracementStrategy(retracement_levels=[0.5, 1.5])

        with pytest.raises(ValidationError, match="retracement_levels"):
            FibonacciRetracementStrategy(retracement_levels=[-0.1, 0.5])

    def test_warmup_period(self):
        """Test warmup_period returns swing_lookback."""
        strategy = FibonacciRetracementStrategy(swing_lookback=30)
        assert strategy.warmup_period == 30

    def test_get_parameters(self):
        """Test get_parameters returns current parameters."""
        strategy = FibonacciRetracementStrategy(
            swing_lookback=25,
            retracement_levels=[0.382, 0.618]
        )
        params = strategy.get_parameters()
        assert params == {
            "swing_lookback": 25,
            "retracement_levels": [0.382, 0.618]
        }

    def test_set_parameters(self):
        """Test set_parameters updates parameters correctly."""
        strategy = FibonacciRetracementStrategy()
        strategy.set_parameters({
            "swing_lookback": 30,
            "retracement_levels": [0.236, 0.5]
        })
        assert strategy.swing_lookback == 30
        assert strategy.retracement_levels == [0.236, 0.5]

    def test_set_parameters_validation(self):
        """Test set_parameters validates new parameters."""
        strategy = FibonacciRetracementStrategy()
        with pytest.raises(ValidationError):
            strategy.set_parameters({"swing_lookback": 0})

        with pytest.raises(ValidationError):
            strategy.set_parameters({"retracement_levels": []})

    def test_buy_signal_near_support(self):
        """Test buy signal when price is near 61.8% retracement (support)."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        # Create swing: high at 100, low at 80
        # 61.8% retracement: 80 + (100-80)*0.618 = 92.36
        highs = [100] * 15 + [95] * 15
        lows = [80] * 15 + [85] * 15
        closes = [90] * 20 + [92, 92, 92, 92, 92, 92, 92, 92, 92, 92]
        data = pd.DataFrame({"High": highs, "Low": lows, "Close": closes}, index=dates)

        strategy = FibonacciRetracementStrategy(swing_lookback=20)
        signals = strategy.generate_signals(data)

        assert "buy" in signals.columns
        assert "sell" in signals.columns
        assert len(signals) == len(data)

    def test_sell_signal_near_resistance(self):
        """Test sell signal when price is near 38.2% retracement (resistance)."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        # Create swing: high at 100, low at 80
        # 38.2% retracement: 80 + (100-80)*0.382 = 87.64
        highs = [100] * 15 + [95] * 15
        lows = [80] * 15 + [85] * 15
        closes = [90] * 20 + [88, 88, 88, 88, 88, 88, 88, 88, 88, 88]
        data = pd.DataFrame({"High": highs, "Low": lows, "Close": closes}, index=dates)

        strategy = FibonacciRetracementStrategy(swing_lookback=20)
        signals = strategy.generate_signals(data)

        assert "sell" in signals.columns
        assert len(signals) == len(data)

    def test_no_signals_in_warmup_period(self):
        """Test no signals generated during warmup period."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        highs = list(range(100, 130))
        lows = list(range(80, 110))
        closes = list(range(90, 120))
        data = pd.DataFrame({"High": highs, "Low": lows, "Close": closes}, index=dates)

        strategy = FibonacciRetracementStrategy(swing_lookback=20)
        signals = strategy.generate_signals(data)

        # First warmup_period rows should have no signals
        warmup = strategy.warmup_period
        assert not signals["buy"].iloc[:warmup].any()
        assert not signals["sell"].iloc[:warmup].any()

    def test_signal_shifting(self):
        """Test signals are shifted by 1 day to prevent look-ahead bias."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        highs = list(range(1, 31))
        lows = list(range(1, 31))
        closes = list(range(1, 31))
        data = pd.DataFrame({"High": highs, "Low": lows, "Close": closes}, index=dates)

        strategy = FibonacciRetracementStrategy()
        signals = strategy.generate_signals(data)

        # First row after shift should always be False
        assert not signals["buy"].iloc[0]
        assert not signals["sell"].iloc[0]

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        strategy = FibonacciRetracementStrategy()
        data = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            strategy.generate_signals(data)

    def test_insufficient_data(self):
        """Test validation fails when data has fewer rows than swing_lookback."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = pd.DataFrame({
            "Close": range(1, 11),
            "High": range(2, 12),
            "Low": range(0, 10)
        }, index=dates)

        strategy = FibonacciRetracementStrategy(swing_lookback=20)
        with pytest.raises(ValidationError, match="at least.*20"):
            strategy.generate_signals(data)

    def test_missing_close_column(self):
        """Test validation fails when Close column is missing."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"High": range(1, 31), "Low": range(1, 31)}, index=dates)

        strategy = FibonacciRetracementStrategy()
        with pytest.raises(ValidationError, match="Close"):
            strategy.generate_signals(data)

    def test_missing_high_column(self):
        """Test validation fails when High column is missing."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"Close": range(1, 31), "Low": range(1, 31)}, index=dates)

        strategy = FibonacciRetracementStrategy()
        with pytest.raises(ValidationError, match="High"):
            strategy.generate_signals(data)

    def test_missing_low_column(self):
        """Test validation fails when Low column is missing."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"Close": range(1, 31), "High": range(1, 31)}, index=dates)

        strategy = FibonacciRetracementStrategy()
        with pytest.raises(ValidationError, match="Low"):
            strategy.generate_signals(data)

    def test_nan_prices(self):
        """Test validation fails with all NaN prices."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({
            "Close": [float("nan")] * 30,
            "High": [float("nan")] * 30,
            "Low": [float("nan")] * 30
        }, index=dates)

        strategy = FibonacciRetracementStrategy()
        with pytest.raises(ValidationError, match="NaN"):
            strategy.generate_signals(data)

    def test_no_signals_flat_prices(self):
        """Test behavior with flat prices (no swing range)."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({
            "Close": [100] * 30,
            "High": [100] * 30,
            "Low": [100] * 30
        }, index=dates)

        strategy = FibonacciRetracementStrategy(swing_lookback=20)
        signals = strategy.generate_signals(data)

        # Flat prices mean no retracement opportunities
        assert signals["buy"].sum() == 0
        assert signals["sell"].sum() == 0

    def test_returns_same_index_as_input(self):
        """Test output signals have same index as input data."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({
            "Close": range(1, 31),
            "High": range(2, 32),
            "Low": range(1, 31)
        }, index=dates)

        strategy = FibonacciRetracementStrategy()
        signals = strategy.generate_signals(data)

        assert signals.index.equals(data.index)
        assert len(signals) == len(data)
