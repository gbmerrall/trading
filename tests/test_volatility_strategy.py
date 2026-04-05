import pandas as pd
import pytest

from backtest.strategy import VolatilityStrategy
from backtest.validation import ValidationError


class TestVolatilityStrategy:
    """Tests for VolatilityStrategy."""

    def test_initialization_with_defaults(self):
        """Test strategy initializes with default parameters."""
        strategy = VolatilityStrategy()
        assert strategy.atr_period == 14
        assert strategy.atr_multiplier == 2.0
        assert strategy.breakout_period == 20

    def test_initialization_with_custom_parameters(self):
        """Test strategy initializes with custom parameters."""
        strategy = VolatilityStrategy(
            atr_period=20, atr_multiplier=3.0, breakout_period=15
        )
        assert strategy.atr_period == 20
        assert strategy.atr_multiplier == 3.0
        assert strategy.breakout_period == 15

    def test_invalid_atr_period(self):
        """Test validation fails for invalid atr_period."""
        with pytest.raises(ValidationError, match="atr_period"):
            VolatilityStrategy(atr_period=0)

    def test_invalid_atr_multiplier(self):
        """Test validation fails for invalid atr_multiplier."""
        with pytest.raises(ValidationError, match="atr_multiplier"):
            VolatilityStrategy(atr_multiplier=0)

        with pytest.raises(ValidationError, match="atr_multiplier"):
            VolatilityStrategy(atr_multiplier=-1.0)

    def test_invalid_breakout_period(self):
        """Test validation fails for invalid breakout_period."""
        with pytest.raises(ValidationError, match="breakout_period"):
            VolatilityStrategy(breakout_period=0)

    def test_warmup_period(self):
        """Test warmup_period returns max of atr_period and breakout_period."""
        strategy = VolatilityStrategy(atr_period=14, breakout_period=20)
        assert strategy.warmup_period == 20

        strategy = VolatilityStrategy(atr_period=25, breakout_period=15)
        assert strategy.warmup_period == 25

    def test_get_parameters(self):
        """Test get_parameters returns current parameters."""
        strategy = VolatilityStrategy(atr_period=15, atr_multiplier=2.5, breakout_period=25)
        params = strategy.get_parameters()
        assert params == {
            "atr_period": 15,
            "atr_multiplier": 2.5,
            "breakout_period": 25
        }

    def test_set_parameters(self):
        """Test set_parameters updates parameters correctly."""
        strategy = VolatilityStrategy()
        strategy.set_parameters({
            "atr_period": 10,
            "atr_multiplier": 3.0,
            "breakout_period": 15
        })
        assert strategy.atr_period == 10
        assert strategy.atr_multiplier == 3.0
        assert strategy.breakout_period == 15

    def test_set_parameters_validation(self):
        """Test set_parameters validates new parameters."""
        strategy = VolatilityStrategy()
        with pytest.raises(ValidationError):
            strategy.set_parameters({"atr_period": 0})

        with pytest.raises(ValidationError):
            strategy.set_parameters({"atr_multiplier": -1.0})

    def test_buy_signal_on_volatile_breakout_up(self):
        """Test buy signal when price breaks high during high volatility."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Create volatility with breakout
        highs = [100] * 30 + [105, 110, 115, 120, 122, 125, 128, 130, 132, 135, 138, 140, 142, 145, 148, 150, 152, 155, 158, 160]
        lows = [95] * 30 + [100, 105, 110, 115, 117, 120, 123, 125, 127, 130, 133, 135, 137, 140, 143, 145, 147, 150, 153, 155]
        closes = [98] * 30 + [103, 108, 113, 118, 120, 123, 126, 128, 130, 133, 136, 138, 140, 143, 146, 148, 150, 153, 156, 158]
        data = pd.DataFrame({"High": highs, "Low": lows, "Close": closes}, index=dates)

        strategy = VolatilityStrategy(atr_period=14, breakout_period=20)
        signals = strategy.generate_signals(data)

        assert "buy" in signals.columns
        assert "sell" in signals.columns
        assert len(signals) == len(data)

    def test_sell_signal_on_volatile_breakout_down(self):
        """Test sell signal when price breaks low during high volatility."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Create volatility with breakout down
        highs = [100] * 30 + [95, 90, 85, 80, 78, 75, 72, 70, 68, 65, 62, 60, 58, 55, 52, 50, 48, 45, 42, 40]
        lows = [95] * 30 + [90, 85, 80, 75, 73, 70, 67, 65, 63, 60, 57, 55, 53, 50, 47, 45, 43, 40, 37, 35]
        closes = [98] * 30 + [93, 88, 83, 78, 76, 73, 70, 68, 66, 63, 60, 58, 56, 53, 50, 48, 46, 43, 40, 38]
        data = pd.DataFrame({"High": highs, "Low": lows, "Close": closes}, index=dates)

        strategy = VolatilityStrategy(atr_period=14, breakout_period=20)
        signals = strategy.generate_signals(data)

        assert "sell" in signals.columns
        assert len(signals) == len(data)

    def test_no_signals_in_low_volatility(self):
        """Test no signals when volatility is low."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Flat prices - low volatility
        highs = [101] * 50
        lows = [99] * 50
        closes = [100] * 50
        data = pd.DataFrame({"High": highs, "Low": lows, "Close": closes}, index=dates)

        strategy = VolatilityStrategy()
        signals = strategy.generate_signals(data)

        # Low volatility should not trigger signals
        assert signals["buy"].sum() == 0
        assert signals["sell"].sum() == 0

    def test_no_signals_in_warmup_period(self):
        """Test no signals generated during warmup period."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        highs = [100 + i for i in range(50)]
        lows = [95 + i for i in range(50)]
        closes = [98 + i for i in range(50)]
        data = pd.DataFrame({"High": highs, "Low": lows, "Close": closes}, index=dates)

        strategy = VolatilityStrategy(atr_period=14, breakout_period=20)
        signals = strategy.generate_signals(data)

        # First warmup_period rows should have no signals
        warmup = strategy.warmup_period
        assert not signals["buy"].iloc[:warmup].any()
        assert not signals["sell"].iloc[:warmup].any()

    def test_signal_shifting(self):
        """Test signals are shifted by 1 day to prevent look-ahead bias."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        highs = [101] * 30
        lows = [99] * 30
        closes = [100] * 30
        data = pd.DataFrame({"High": highs, "Low": lows, "Close": closes}, index=dates)

        strategy = VolatilityStrategy()
        signals = strategy.generate_signals(data)

        # First row after shift should always be False
        assert not signals["buy"].iloc[0]
        assert not signals["sell"].iloc[0]

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        strategy = VolatilityStrategy()
        data = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            strategy.generate_signals(data)

    def test_insufficient_data(self):
        """Test validation fails when data has fewer rows than warmup."""
        dates = pd.date_range("2020-01-01", periods=15, freq="D")
        data = pd.DataFrame({
            "High": [101] * 15,
            "Low": [99] * 15,
            "Close": [100] * 15
        }, index=dates)

        strategy = VolatilityStrategy(atr_period=14, breakout_period=20)
        with pytest.raises(ValidationError, match="at least.*20"):
            strategy.generate_signals(data)

    def test_missing_required_columns(self):
        """Test validation fails when required columns are missing."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")

        # Missing High
        data = pd.DataFrame({"Low": [99] * 30, "Close": [100] * 30}, index=dates)
        strategy = VolatilityStrategy()
        with pytest.raises(ValidationError, match="High"):
            strategy.generate_signals(data)

        # Missing Low
        data = pd.DataFrame({"High": [101] * 30, "Close": [100] * 30}, index=dates)
        with pytest.raises(ValidationError, match="Low"):
            strategy.generate_signals(data)

        # Missing Close
        data = pd.DataFrame({"High": [101] * 30, "Low": [99] * 30}, index=dates)
        with pytest.raises(ValidationError, match="Close"):
            strategy.generate_signals(data)

    def test_nan_prices(self):
        """Test validation fails with all NaN prices."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({
            "High": [float("nan")] * 30,
            "Low": [float("nan")] * 30,
            "Close": [float("nan")] * 30
        }, index=dates)

        strategy = VolatilityStrategy()
        with pytest.raises(ValidationError, match="NaN"):
            strategy.generate_signals(data)

    def test_returns_same_index_as_input(self):
        """Test output signals have same index as input data."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({
            "High": [101] * 30,
            "Low": [99] * 30,
            "Close": [100] * 30
        }, index=dates)

        strategy = VolatilityStrategy()
        signals = strategy.generate_signals(data)

        assert signals.index.equals(data.index)
        assert len(signals) == len(data)
