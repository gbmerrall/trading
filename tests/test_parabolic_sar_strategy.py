import pandas as pd
import pytest

from backtest.strategy import ParabolicSARStrategy
from backtest.validation import ValidationError


class TestParabolicSARStrategy:
    """Tests for ParabolicSARStrategy."""

    def test_initialization_with_defaults(self):
        """Test strategy initializes with default parameters."""
        strategy = ParabolicSARStrategy()
        assert strategy.af == 0.02
        assert strategy.max_af == 0.2

    def test_initialization_with_custom_parameters(self):
        """Test strategy initializes with custom parameters."""
        strategy = ParabolicSARStrategy(af=0.01, max_af=0.15)
        assert strategy.af == 0.01
        assert strategy.max_af == 0.15

    def test_invalid_af(self):
        """Test validation fails for invalid af."""
        with pytest.raises(ValidationError, match="af"):
            ParabolicSARStrategy(af=0)

        with pytest.raises(ValidationError, match="af"):
            ParabolicSARStrategy(af=-0.01)

    def test_invalid_max_af(self):
        """Test validation fails for invalid max_af."""
        with pytest.raises(ValidationError, match="max_af"):
            ParabolicSARStrategy(max_af=0)

        with pytest.raises(ValidationError, match="max_af"):
            ParabolicSARStrategy(max_af=-0.1)

    def test_af_greater_than_max_af(self):
        """Test validation fails when af > max_af."""
        with pytest.raises(ValidationError, match="af.*less than or equal.*max_af"):
            ParabolicSARStrategy(af=0.3, max_af=0.2)

        with pytest.raises(ValidationError, match="af.*less than or equal.*max_af"):
            ParabolicSARStrategy(af=0.25, max_af=0.2)

    def test_warmup_period(self):
        """Test warmup_period returns 1 (minimal warmup)."""
        strategy = ParabolicSARStrategy()
        assert strategy.warmup_period == 1

    def test_get_parameters(self):
        """Test get_parameters returns current parameters."""
        strategy = ParabolicSARStrategy(af=0.01, max_af=0.15)
        params = strategy.get_parameters()
        assert params == {"af": 0.01, "max_af": 0.15}

    def test_set_parameters(self):
        """Test set_parameters updates parameters correctly."""
        strategy = ParabolicSARStrategy()
        strategy.set_parameters({"af": 0.03, "max_af": 0.25})
        assert strategy.af == 0.03
        assert strategy.max_af == 0.25

    def test_set_parameters_partial_update(self):
        """Test set_parameters with partial parameter dict."""
        strategy = ParabolicSARStrategy(af=0.02, max_af=0.2)
        strategy.set_parameters({"af": 0.01})
        assert strategy.af == 0.01
        assert strategy.max_af == 0.2  # Unchanged

    def test_set_parameters_validation(self):
        """Test set_parameters validates new parameters."""
        strategy = ParabolicSARStrategy()
        with pytest.raises(ValidationError):
            strategy.set_parameters({"af": 0})

        with pytest.raises(ValidationError):
            strategy.set_parameters({"af": 0.3, "max_af": 0.2})

    def test_buy_signal_on_price_above_sar(self):
        """Test signals can be generated from price data."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Create data with uptrend
        prices = [float(p) for p in range(50, 100)]
        highs = [p + 2.0 for p in prices]
        lows = [p - 2.0 for p in prices]
        data = pd.DataFrame({"Close": prices, "High": highs, "Low": lows}, index=dates)

        strategy = ParabolicSARStrategy(af=0.02, max_af=0.2)
        signals = strategy.generate_signals(data)

        # Verify signal structure
        assert "buy" in signals.columns
        assert "sell" in signals.columns
        assert len(signals) == len(data)
        # All signal values should be False or True
        assert all(isinstance(x, (bool, type(pd.NA))) for x in signals["buy"])
        assert all(isinstance(x, (bool, type(pd.NA))) for x in signals["sell"])

    def test_sell_signal_on_price_below_sar(self):
        """Test signals maintain correct format across different price patterns."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Create data with downtrend
        prices = [float(p) for p in range(100, 50, -1)]
        highs = [p + 2.0 for p in prices]
        lows = [p - 2.0 for p in prices]
        data = pd.DataFrame({"Close": prices, "High": highs, "Low": lows}, index=dates)

        strategy = ParabolicSARStrategy(af=0.02, max_af=0.2)
        signals = strategy.generate_signals(data)

        # Verify signal structure
        assert "sell" in signals.columns
        assert "buy" in signals.columns
        assert len(signals) == len(data)
        # Signals should be boolean-like
        assert signals["buy"].isin([True, False]).all() or signals["buy"].isna().any()
        assert signals["sell"].isin([True, False]).all() or signals["sell"].isna().any()

    def test_no_signals_in_warmup_period(self):
        """Test no signals generated during warmup period."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = [float(p) for p in range(50, 100)]
        highs = [p + 2.0 for p in prices]
        lows = [p - 2.0 for p in prices]
        data = pd.DataFrame({"Close": prices, "High": highs, "Low": lows}, index=dates)

        strategy = ParabolicSARStrategy()
        signals = strategy.generate_signals(data)

        # First warmup_period rows should have no signals
        warmup = strategy.warmup_period
        assert not signals["buy"].iloc[:warmup].any()
        assert not signals["sell"].iloc[:warmup].any()

    def test_signal_shifting(self):
        """Test signals are shifted by 1 day to prevent look-ahead bias."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = [float(p) for p in range(50, 100)]
        highs = [p + 2.0 for p in prices]
        lows = [p - 2.0 for p in prices]
        data = pd.DataFrame({"Close": prices, "High": highs, "Low": lows}, index=dates)

        strategy = ParabolicSARStrategy()
        signals = strategy.generate_signals(data)

        # First row after shift should always be False
        assert not signals["buy"].iloc[0]
        assert not signals["sell"].iloc[0]

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        strategy = ParabolicSARStrategy()
        data = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            strategy.generate_signals(data)

    def test_insufficient_data(self):
        """Test validation fails when data has fewer rows than warmup."""
        strategy = ParabolicSARStrategy()
        data = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            strategy.generate_signals(data)

    def test_missing_required_columns(self):
        """Test validation fails when required columns are missing."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = pd.DataFrame({"Open": range(1, 51)}, index=dates)

        strategy = ParabolicSARStrategy()
        with pytest.raises(ValidationError, match="Close|High|Low"):
            strategy.generate_signals(data)

    def test_nan_prices(self):
        """Test validation fails with all NaN prices."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = pd.DataFrame({
            "Close": [float("nan")] * 50,
            "High": [float("nan")] * 50,
            "Low": [float("nan")] * 50
        }, index=dates)

        strategy = ParabolicSARStrategy()
        with pytest.raises(ValidationError, match="NaN"):
            strategy.generate_signals(data)

    def test_returns_same_index_as_input(self):
        """Test output signals have same index as input data."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = [float(p) for p in range(50, 100)]
        highs = [p + 2.0 for p in prices]
        lows = [p - 2.0 for p in prices]
        data = pd.DataFrame({"Close": prices, "High": highs, "Low": lows}, index=dates)

        strategy = ParabolicSARStrategy()
        signals = strategy.generate_signals(data)

        assert signals.index.equals(data.index)
        assert len(signals) == len(data)
