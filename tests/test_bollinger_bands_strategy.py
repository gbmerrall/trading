import pandas as pd
import pytest

from backtest.strategy import BollingerBandsStrategy
from backtest.validation import ValidationError


class TestBollingerBandsStrategy:
    """Tests for BollingerBandsStrategy."""

    def test_initialization_with_defaults(self):
        """Test strategy initializes with default parameters."""
        strategy = BollingerBandsStrategy()
        assert strategy.period == 20
        assert strategy.std_dev == 2.0

    def test_initialization_with_custom_parameters(self):
        """Test strategy initializes with custom parameters."""
        strategy = BollingerBandsStrategy(period=15, std_dev=2.5)
        assert strategy.period == 15
        assert strategy.std_dev == 2.5

    def test_invalid_period(self):
        """Test validation fails for invalid period."""
        with pytest.raises(ValidationError, match="period"):
            BollingerBandsStrategy(period=0)

        with pytest.raises(ValidationError, match="period"):
            BollingerBandsStrategy(period=-5)

    def test_invalid_std_dev(self):
        """Test validation fails for invalid std_dev."""
        with pytest.raises(ValidationError, match="std_dev"):
            BollingerBandsStrategy(std_dev=0)

        with pytest.raises(ValidationError, match="std_dev"):
            BollingerBandsStrategy(std_dev=-1.0)

    def test_warmup_period(self):
        """Test warmup_period returns period."""
        strategy = BollingerBandsStrategy(period=25)
        assert strategy.warmup_period == 25

    def test_get_parameters(self):
        """Test get_parameters returns current parameters."""
        strategy = BollingerBandsStrategy(period=15, std_dev=2.5)
        params = strategy.get_parameters()
        assert params == {"period": 15, "std_dev": 2.5}

    def test_set_parameters(self):
        """Test set_parameters updates parameters correctly."""
        strategy = BollingerBandsStrategy()
        strategy.set_parameters({"period": 30, "std_dev": 3.0})
        assert strategy.period == 30
        assert strategy.std_dev == 3.0

    def test_set_parameters_partial_update(self):
        """Test set_parameters with partial parameter dict."""
        strategy = BollingerBandsStrategy(period=20, std_dev=2.0)
        strategy.set_parameters({"period": 15})
        assert strategy.period == 15
        assert strategy.std_dev == 2.0  # Unchanged

    def test_set_parameters_validation(self):
        """Test set_parameters validates new parameters."""
        strategy = BollingerBandsStrategy()
        with pytest.raises(ValidationError):
            strategy.set_parameters({"period": 0})

        with pytest.raises(ValidationError):
            strategy.set_parameters({"std_dev": -1.0})

    def test_buy_signal_on_lower_band_touch(self):
        """Test buy signal fires when price breaks below the lower Bollinger Band.

        A linear downtrend does not trigger this: price tracks the band as it falls.
        A stable baseline followed by a sharp drop puts price well below the
        established lower band, generating buy signals.
        """
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        # 25 stable days establish tight bands, then sharp drop breaks below lower band
        prices = [100.0] * 25 + [100 - i * 2 for i in range(1, 36)]
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = BollingerBandsStrategy(period=20, std_dev=2.0)
        signals = strategy.generate_signals(data)

        assert "buy" in signals.columns
        assert "sell" in signals.columns
        assert len(signals) == len(data)
        assert signals["buy"].any(), "Expected buy signals when price drops sharply below lower band"

    def test_sell_signal_on_upper_band_touch(self):
        """Test sell signal fires when price breaks above the upper Bollinger Band."""
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        # 25 stable days establish tight bands, then sharp rise breaks above upper band
        prices = [100.0] * 25 + [100 + i * 2 for i in range(1, 36)]
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = BollingerBandsStrategy(period=20, std_dev=2.0)
        signals = strategy.generate_signals(data)

        assert "sell" in signals.columns
        assert "buy" in signals.columns
        assert len(signals) == len(data)
        assert signals["sell"].any(), "Expected sell signals when price rises sharply above upper band"

    def test_no_signals_in_warmup_period(self):
        """Test no signals generated during warmup period."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = list(range(100, 70, -1)) + [70] * 20
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = BollingerBandsStrategy(period=20)
        signals = strategy.generate_signals(data)

        # First warmup_period rows should have no signals
        warmup = strategy.warmup_period
        assert not signals["buy"].iloc[:warmup].any()
        assert not signals["sell"].iloc[:warmup].any()

    def test_signal_shifting(self):
        """Test signals are shifted by 1 day to prevent look-ahead bias."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = list(range(100, 70, -1)) + [70] * 20
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = BollingerBandsStrategy(period=20)
        signals = strategy.generate_signals(data)

        # First row after shift should always be False
        assert not signals["buy"].iloc[0]
        assert not signals["sell"].iloc[0]

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        strategy = BollingerBandsStrategy()
        data = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            strategy.generate_signals(data)

    def test_insufficient_data(self):
        """Test validation fails when data has fewer rows than period."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = pd.DataFrame({"Close": range(1, 11)}, index=dates)

        strategy = BollingerBandsStrategy(period=20)
        with pytest.raises(ValidationError, match="at least.*20"):
            strategy.generate_signals(data)

    def test_missing_close_column(self):
        """Test validation fails when Close column is missing."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = pd.DataFrame({"Open": range(1, 51)}, index=dates)

        strategy = BollingerBandsStrategy()
        with pytest.raises(ValidationError, match="Close"):
            strategy.generate_signals(data)

    def test_nan_prices(self):
        """Test validation fails with all NaN prices."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = pd.DataFrame({"Close": [float("nan")] * 50}, index=dates)

        strategy = BollingerBandsStrategy()
        with pytest.raises(ValidationError, match="NaN"):
            strategy.generate_signals(data)

    def test_returns_same_index_as_input(self):
        """Test output signals have same index as input data."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = pd.DataFrame({"Close": range(1, 51)}, index=dates)

        strategy = BollingerBandsStrategy()
        signals = strategy.generate_signals(data)

        assert signals.index.equals(data.index)
        assert len(signals) == len(data)
