import os
import sys

import pandas as pd
import pytest

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.strategy import MomentumStrategy
from backtest.validation import ValidationError


class TestMomentumStrategy:
    """Tests for MomentumStrategy."""

    def test_initialization_with_defaults(self):
        """Test strategy initializes with default parameters."""
        strategy = MomentumStrategy()
        assert strategy.roc_period == 12
        assert strategy.roc_threshold == 0.05

    def test_initialization_with_custom_parameters(self):
        """Test strategy initializes with custom parameters."""
        strategy = MomentumStrategy(roc_period=20, roc_threshold=0.10)
        assert strategy.roc_period == 20
        assert strategy.roc_threshold == 0.10

    def test_invalid_roc_period(self):
        """Test validation fails for invalid roc_period."""
        with pytest.raises(ValidationError, match="roc_period"):
            MomentumStrategy(roc_period=0)

    def test_invalid_roc_threshold(self):
        """Test validation fails for invalid roc_threshold."""
        with pytest.raises(ValidationError, match="roc_threshold"):
            MomentumStrategy(roc_threshold=0)

        with pytest.raises(ValidationError, match="roc_threshold"):
            MomentumStrategy(roc_threshold=-0.05)

    def test_warmup_period(self):
        """Test warmup_period returns roc_period."""
        strategy = MomentumStrategy(roc_period=15)
        assert strategy.warmup_period == 15

    def test_get_parameters(self):
        """Test get_parameters returns current parameters."""
        strategy = MomentumStrategy(roc_period=20, roc_threshold=0.08)
        params = strategy.get_parameters()
        assert params == {"roc_period": 20, "roc_threshold": 0.08}

    def test_set_parameters(self):
        """Test set_parameters updates parameters correctly."""
        strategy = MomentumStrategy()
        strategy.set_parameters({"roc_period": 15, "roc_threshold": 0.07})
        assert strategy.roc_period == 15
        assert strategy.roc_threshold == 0.07

    def test_set_parameters_validation(self):
        """Test set_parameters validates new parameters."""
        strategy = MomentumStrategy()
        with pytest.raises(ValidationError):
            strategy.set_parameters({"roc_period": 0})

        with pytest.raises(ValidationError):
            strategy.set_parameters({"roc_threshold": -0.05})

    def test_buy_signal_on_positive_momentum(self):
        """Test buy signal when ROC exceeds positive threshold."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        # Create uptrend to trigger positive momentum
        prices = [100 + i * 2 for i in range(30)]  # Strong uptrend
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MomentumStrategy(roc_period=12, roc_threshold=0.05)
        signals = strategy.generate_signals(data)

        assert "buy" in signals.columns
        assert "sell" in signals.columns
        assert len(signals) == len(data)

    def test_sell_signal_on_negative_momentum(self):
        """Test sell signal when ROC falls below negative threshold."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        # Create downtrend to trigger negative momentum
        prices = [100 - i * 2 for i in range(30)]  # Strong downtrend
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MomentumStrategy(roc_period=12, roc_threshold=0.05)
        signals = strategy.generate_signals(data)

        assert "sell" in signals.columns
        assert len(signals) == len(data)

    def test_no_signals_with_low_momentum(self):
        """Test no signals when ROC is below threshold."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        # Flat prices - no momentum
        prices = [100] * 30
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MomentumStrategy(roc_period=12, roc_threshold=0.05)
        signals = strategy.generate_signals(data)

        # Should not generate signals for flat prices
        assert signals["buy"].sum() == 0
        assert signals["sell"].sum() == 0

    def test_no_signals_in_warmup_period(self):
        """Test no signals generated during warmup period."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        prices = [100 + i for i in range(30)]
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MomentumStrategy(roc_period=12)
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

        strategy = MomentumStrategy()
        signals = strategy.generate_signals(data)

        # First row after shift should always be False
        assert not signals["buy"].iloc[0]
        assert not signals["sell"].iloc[0]

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        strategy = MomentumStrategy()
        data = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            strategy.generate_signals(data)

    def test_insufficient_data(self):
        """Test validation fails when data has fewer rows than warmup."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = pd.DataFrame({"Close": range(1, 11)}, index=dates)

        strategy = MomentumStrategy(roc_period=12)
        with pytest.raises(ValidationError, match="at least.*12"):
            strategy.generate_signals(data)

    def test_missing_close_column(self):
        """Test validation fails when Close column is missing."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"Open": range(1, 31)}, index=dates)

        strategy = MomentumStrategy()
        with pytest.raises(ValidationError, match="Close"):
            strategy.generate_signals(data)

    def test_nan_prices(self):
        """Test validation fails with all NaN prices."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"Close": [float("nan")] * 30}, index=dates)

        strategy = MomentumStrategy()
        with pytest.raises(ValidationError, match="NaN"):
            strategy.generate_signals(data)

    def test_returns_same_index_as_input(self):
        """Test output signals have same index as input data."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"Close": range(1, 31)}, index=dates)

        strategy = MomentumStrategy()
        signals = strategy.generate_signals(data)

        assert signals.index.equals(data.index)
        assert len(signals) == len(data)
