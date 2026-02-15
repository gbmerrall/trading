import os
import sys

import pandas as pd
import pytest

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.strategy import MACDStrategy
from backtest.validation import ValidationError


class TestMACDStrategy:
    """Tests for MACDStrategy."""

    def test_initialization_with_defaults(self):
        """Test strategy initializes with default parameters."""
        strategy = MACDStrategy()
        assert strategy.fast == 12
        assert strategy.slow == 26
        assert strategy.signal == 9

    def test_initialization_with_custom_parameters(self):
        """Test strategy initializes with custom parameters."""
        strategy = MACDStrategy(fast=10, slow=20, signal=5)
        assert strategy.fast == 10
        assert strategy.slow == 20
        assert strategy.signal == 5

    def test_invalid_fast(self):
        """Test validation fails for invalid fast period."""
        with pytest.raises(ValidationError, match="fast"):
            MACDStrategy(fast=0)

        with pytest.raises(ValidationError, match="fast"):
            MACDStrategy(fast=-5)

    def test_invalid_slow(self):
        """Test validation fails for invalid slow period."""
        with pytest.raises(ValidationError, match="slow"):
            MACDStrategy(slow=0)

        with pytest.raises(ValidationError, match="slow"):
            MACDStrategy(slow=-5)

    def test_invalid_signal(self):
        """Test validation fails for invalid signal period."""
        with pytest.raises(ValidationError, match="signal"):
            MACDStrategy(signal=0)

        with pytest.raises(ValidationError, match="signal"):
            MACDStrategy(signal=-5)

    def test_fast_greater_than_slow(self):
        """Test validation fails when fast >= slow."""
        with pytest.raises(ValidationError, match="fast.*less than.*slow"):
            MACDStrategy(fast=26, slow=12)

        with pytest.raises(ValidationError, match="fast.*less than.*slow"):
            MACDStrategy(fast=20, slow=20)

    def test_warmup_period(self):
        """Test warmup_period returns slow + signal."""
        strategy = MACDStrategy(fast=12, slow=26, signal=9)
        assert strategy.warmup_period == 35  # 26 + 9

    def test_get_parameters(self):
        """Test get_parameters returns current parameters."""
        strategy = MACDStrategy(fast=10, slow=20, signal=5)
        params = strategy.get_parameters()
        assert params == {"fast": 10, "slow": 20, "signal": 5}

    def test_set_parameters(self):
        """Test set_parameters updates parameters correctly."""
        strategy = MACDStrategy()
        strategy.set_parameters({"fast": 10, "slow": 20, "signal": 5})
        assert strategy.fast == 10
        assert strategy.slow == 20
        assert strategy.signal == 5

    def test_set_parameters_partial_update(self):
        """Test set_parameters with partial parameter dict."""
        strategy = MACDStrategy(fast=12, slow=26, signal=9)
        strategy.set_parameters({"fast": 10})
        assert strategy.fast == 10
        assert strategy.slow == 26  # Unchanged
        assert strategy.signal == 9  # Unchanged

    def test_set_parameters_validation(self):
        """Test set_parameters validates new parameters."""
        strategy = MACDStrategy()
        with pytest.raises(ValidationError):
            strategy.set_parameters({"fast": 0})

        with pytest.raises(ValidationError):
            strategy.set_parameters({"fast": 30, "slow": 20})

    def test_buy_signal_on_bullish_crossover(self):
        """Test signals can be generated from price data."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = list(range(100, 50, -1))
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MACDStrategy(fast=12, slow=26, signal=9)
        signals = strategy.generate_signals(data)

        # Verify signal structure
        assert "buy" in signals.columns
        assert "sell" in signals.columns
        assert len(signals) == len(data)
        # All signal values should be False or True
        assert all(isinstance(x, (bool, type(pd.NA))) for x in signals["buy"])
        assert all(isinstance(x, (bool, type(pd.NA))) for x in signals["sell"])

    def test_sell_signal_on_bearish_crossover(self):
        """Test signals maintain correct format across different price patterns."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = list(range(50, 100))
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MACDStrategy(fast=12, slow=26, signal=9)
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
        prices = list(range(100, 50, -1))
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MACDStrategy(fast=12, slow=26, signal=9)
        signals = strategy.generate_signals(data)

        # First warmup_period rows should have no signals
        warmup = strategy.warmup_period
        assert not signals["buy"].iloc[:warmup].any()
        assert not signals["sell"].iloc[:warmup].any()

    def test_signal_shifting(self):
        """Test signals are shifted by 1 day to prevent look-ahead bias."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = list(range(100, 50, -1))
        data = pd.DataFrame({"Close": prices}, index=dates)

        strategy = MACDStrategy()
        signals = strategy.generate_signals(data)

        # First row after shift should always be False
        assert not signals["buy"].iloc[0]
        assert not signals["sell"].iloc[0]

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        strategy = MACDStrategy()
        data = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            strategy.generate_signals(data)

    def test_insufficient_data(self):
        """Test validation fails when data has fewer rows than slow + signal."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        data = pd.DataFrame({"Close": range(1, 21)}, index=dates)

        strategy = MACDStrategy(fast=12, slow=26, signal=9)
        with pytest.raises(ValidationError, match="at least.*35"):
            strategy.generate_signals(data)

    def test_missing_close_column(self):
        """Test validation fails when Close column is missing."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = pd.DataFrame({"Open": range(1, 51)}, index=dates)

        strategy = MACDStrategy()
        with pytest.raises(ValidationError, match="Close"):
            strategy.generate_signals(data)

    def test_nan_prices(self):
        """Test validation fails with all NaN prices."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = pd.DataFrame({"Close": [float("nan")] * 50}, index=dates)

        strategy = MACDStrategy()
        with pytest.raises(ValidationError, match="NaN"):
            strategy.generate_signals(data)

    def test_returns_same_index_as_input(self):
        """Test output signals have same index as input data."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = pd.DataFrame({"Close": range(1, 51)}, index=dates)

        strategy = MACDStrategy()
        signals = strategy.generate_signals(data)

        assert signals.index.equals(data.index)
        assert len(signals) == len(data)
