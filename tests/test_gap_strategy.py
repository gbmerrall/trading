import pandas as pd
import pytest

from backtest.strategy import GapStrategy
from backtest.validation import ValidationError


class TestGapStrategy:
    """Tests for GapStrategy."""

    def test_initialization_with_defaults(self):
        """Test strategy initializes with default parameters."""
        strategy = GapStrategy()
        assert strategy.min_gap_pct == 0.02

    def test_initialization_with_custom_parameters(self):
        """Test strategy initializes with custom parameters."""
        strategy = GapStrategy(min_gap_pct=0.05)
        assert strategy.min_gap_pct == 0.05

    def test_invalid_min_gap_pct(self):
        """Test validation fails for invalid min_gap_pct."""
        with pytest.raises(ValidationError, match="min_gap_pct"):
            GapStrategy(min_gap_pct=0)

        with pytest.raises(ValidationError, match="min_gap_pct"):
            GapStrategy(min_gap_pct=-0.05)

    def test_warmup_period(self):
        """Test warmup_period returns 1."""
        strategy = GapStrategy()
        assert strategy.warmup_period == 1

    def test_get_parameters(self):
        """Test get_parameters returns current parameters."""
        strategy = GapStrategy(min_gap_pct=0.03)
        params = strategy.get_parameters()
        assert params == {"min_gap_pct": 0.03}

    def test_set_parameters(self):
        """Test set_parameters updates parameters correctly."""
        strategy = GapStrategy()
        strategy.set_parameters({"min_gap_pct": 0.04})
        assert strategy.min_gap_pct == 0.04

    def test_set_parameters_validation(self):
        """Test set_parameters validates new parameters."""
        strategy = GapStrategy()
        with pytest.raises(ValidationError):
            strategy.set_parameters({"min_gap_pct": 0})

    def test_buy_signal_on_gap_down(self):
        """Test buy signal when gap down occurs (fill expected)."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        # Gap down: previous close > current open
        closes = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        opens = [100, 100, 100, 95, 100, 100, 100, 100, 100, 100]  # Gap down on day 4
        data = pd.DataFrame({"Close": closes, "Open": opens}, index=dates)

        strategy = GapStrategy(min_gap_pct=0.02)
        signals = strategy.generate_signals(data)

        assert "buy" in signals.columns
        assert "sell" in signals.columns
        # Should detect gap down
        assert len(signals) == len(data)

    def test_sell_signal_on_gap_up(self):
        """Test sell signal when gap up occurs."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        # Gap up: previous close < current open
        closes = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        opens = [100, 100, 100, 105, 100, 100, 100, 100, 100, 100]  # Gap up on day 4
        data = pd.DataFrame({"Close": closes, "Open": opens}, index=dates)

        strategy = GapStrategy(min_gap_pct=0.02)
        signals = strategy.generate_signals(data)

        assert "sell" in signals.columns
        # Should detect gap up
        assert len(signals) == len(data)

    def test_no_signals_with_small_gaps(self):
        """Test no signals when gaps are below threshold."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        # Small gap (1%)
        closes = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        opens = [100, 100, 100, 99, 100, 100, 100, 100, 100, 100]  # 1% gap
        data = pd.DataFrame({"Close": closes, "Open": opens}, index=dates)

        strategy = GapStrategy(min_gap_pct=0.02)  # 2% threshold
        signals = strategy.generate_signals(data)

        # Should not generate signals for small gaps
        assert signals["buy"].sum() == 0
        assert signals["sell"].sum() == 0

    def test_signal_shifting(self):
        """Test signals are shifted by 1 day to prevent look-ahead bias."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        closes = [100] * 10
        opens = [100] * 10
        data = pd.DataFrame({"Close": closes, "Open": opens}, index=dates)

        strategy = GapStrategy()
        signals = strategy.generate_signals(data)

        # First row after shift should always be False
        assert not signals["buy"].iloc[0]
        assert not signals["sell"].iloc[0]

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        strategy = GapStrategy()
        data = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            strategy.generate_signals(data)

    def test_insufficient_data(self):
        """Test validation fails when data has no rows."""
        dates = pd.date_range("2020-01-01", periods=0, freq="D")
        data = pd.DataFrame({"Close": [], "Open": []}, index=dates)

        strategy = GapStrategy()
        with pytest.raises(ValidationError, match="empty"):
            strategy.generate_signals(data)

    def test_missing_close_column(self):
        """Test validation fails when Close column is missing."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = pd.DataFrame({"Open": range(1, 11)}, index=dates)

        strategy = GapStrategy()
        with pytest.raises(ValidationError, match="Close"):
            strategy.generate_signals(data)

    def test_missing_open_column(self):
        """Test validation fails when Open column is missing."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = pd.DataFrame({"Close": range(1, 11)}, index=dates)

        strategy = GapStrategy()
        with pytest.raises(ValidationError, match="Open"):
            strategy.generate_signals(data)

    def test_nan_prices(self):
        """Test validation fails with all NaN prices."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = pd.DataFrame({
            "Close": [float("nan")] * 10,
            "Open": [float("nan")] * 10
        }, index=dates)

        strategy = GapStrategy()
        with pytest.raises(ValidationError, match="NaN"):
            strategy.generate_signals(data)

    def test_returns_same_index_as_input(self):
        """Test output signals have same index as input data."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = pd.DataFrame({"Close": range(1, 11), "Open": range(1, 11)}, index=dates)

        strategy = GapStrategy()
        signals = strategy.generate_signals(data)

        assert signals.index.equals(data.index)
        assert len(signals) == len(data)
