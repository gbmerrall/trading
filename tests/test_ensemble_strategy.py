import pandas as pd
import pytest

from backtest.strategy import (
    EnsembleStrategy,
    RSIStrategy,
    BollingerBandsStrategy,
    MACDStrategy,
    MovingAverageCrossoverStrategy
)
from backtest.validation import ValidationError


class TestEnsembleStrategy:
    """Tests for EnsembleStrategy."""

    def test_initialization_with_defaults(self):
        """Test strategy initializes with default parameters."""
        rsi = RSIStrategy()
        bb = BollingerBandsStrategy()
        strategy = EnsembleStrategy(strategies=[rsi, bb])
        assert len(strategy.strategies) == 2
        assert strategy.min_agreement == 1

    def test_initialization_with_custom_min_agreement(self):
        """Test strategy initializes with custom min_agreement."""
        rsi = RSIStrategy()
        bb = BollingerBandsStrategy()
        macd = MACDStrategy()
        strategy = EnsembleStrategy(strategies=[rsi, bb, macd], min_agreement=2)
        assert len(strategy.strategies) == 3
        assert strategy.min_agreement == 2

    def test_invalid_empty_strategies_list(self):
        """Test validation fails for empty strategies list."""
        with pytest.raises(ValidationError, match="at least one strategy"):
            EnsembleStrategy(strategies=[])

    def test_invalid_min_agreement_too_low(self):
        """Test validation fails when min_agreement < 1."""
        rsi = RSIStrategy()
        bb = BollingerBandsStrategy()
        with pytest.raises(ValidationError, match="min_agreement"):
            EnsembleStrategy(strategies=[rsi, bb], min_agreement=0)

    def test_invalid_min_agreement_too_high(self):
        """Test validation fails when min_agreement > num strategies."""
        rsi = RSIStrategy()
        bb = BollingerBandsStrategy()
        with pytest.raises(ValidationError, match="min_agreement"):
            EnsembleStrategy(strategies=[rsi, bb], min_agreement=3)

    def test_warmup_period_returns_max_of_substrategies(self):
        """Test warmup_period returns maximum warmup period of all strategies."""
        rsi = RSIStrategy(period=14)  # warmup = 14
        bb = BollingerBandsStrategy(period=20)  # warmup = 20
        macd = MACDStrategy()  # warmup = 35
        strategy = EnsembleStrategy(strategies=[rsi, bb, macd])
        assert strategy.warmup_period == 35

    def test_get_parameters(self):
        """Test get_parameters returns strategy info."""
        rsi = RSIStrategy()
        bb = BollingerBandsStrategy()
        strategy = EnsembleStrategy(strategies=[rsi, bb], min_agreement=2)
        params = strategy.get_parameters()
        assert params["num_strategies"] == 2
        assert params["min_agreement"] == 2
        assert "strategy_names" in params

    def test_set_parameters_min_agreement(self):
        """Test set_parameters updates min_agreement."""
        rsi = RSIStrategy()
        bb = BollingerBandsStrategy()
        macd = MACDStrategy()
        strategy = EnsembleStrategy(strategies=[rsi, bb, macd], min_agreement=1)
        strategy.set_parameters({"min_agreement": 2})
        assert strategy.min_agreement == 2

    def test_set_parameters_validation(self):
        """Test set_parameters validates new parameters."""
        rsi = RSIStrategy()
        bb = BollingerBandsStrategy()
        strategy = EnsembleStrategy(strategies=[rsi, bb])
        with pytest.raises(ValidationError):
            strategy.set_parameters({"min_agreement": 0})

        with pytest.raises(ValidationError):
            strategy.set_parameters({"min_agreement": 5})

    def test_majority_vote_buy_signal(self):
        """Test buy signal when majority of strategies agree."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Downtrend to trigger RSI and BB buy signals
        prices = list(range(100, 50, -1))
        data = pd.DataFrame({"Close": prices}, index=dates)

        rsi = RSIStrategy()
        bb = BollingerBandsStrategy()
        strategy = EnsembleStrategy(strategies=[rsi, bb], min_agreement=2)
        signals = strategy.generate_signals(data)

        assert "buy" in signals.columns
        assert "sell" in signals.columns
        assert len(signals) == len(data)

    def test_majority_vote_sell_signal(self):
        """Test sell signal when majority of strategies agree."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Uptrend to trigger RSI and BB sell signals
        prices = list(range(50, 100))
        data = pd.DataFrame({"Close": prices}, index=dates)

        rsi = RSIStrategy()
        bb = BollingerBandsStrategy()
        strategy = EnsembleStrategy(strategies=[rsi, bb], min_agreement=2)
        signals = strategy.generate_signals(data)

        assert "sell" in signals.columns
        assert len(signals) == len(data)

    def test_no_signal_when_insufficient_agreement(self):
        """Test no signal when not enough strategies agree."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Flat prices - unlikely to get agreement
        prices = [100] * 50
        data = pd.DataFrame({"Close": prices}, index=dates)

        rsi = RSIStrategy()
        bb = BollingerBandsStrategy()
        macd = MACDStrategy()
        strategy = EnsembleStrategy(strategies=[rsi, bb, macd], min_agreement=3)
        signals = strategy.generate_signals(data)

        # Flat prices should not trigger all 3 strategies
        assert signals["buy"].sum() == 0
        assert signals["sell"].sum() == 0

    def test_min_agreement_one_requires_any_strategy(self):
        """Test min_agreement=1 triggers when any strategy signals."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = list(range(100, 50, -1))
        data = pd.DataFrame({"Close": prices}, index=dates)

        rsi = RSIStrategy()
        strategy = EnsembleStrategy(strategies=[rsi], min_agreement=1)
        signals = strategy.generate_signals(data)

        # Should generate signals since only 1 strategy and min_agreement=1
        assert len(signals) == len(data)

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        rsi = RSIStrategy()
        strategy = EnsembleStrategy(strategies=[rsi])
        data = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            strategy.generate_signals(data)

    def test_insufficient_data(self):
        """Test validation fails when data has fewer rows than warmup."""
        dates = pd.date_range("2020-01-01", periods=15, freq="D")
        data = pd.DataFrame({"Close": range(1, 16)}, index=dates)

        macd = MACDStrategy()  # warmup = 35
        strategy = EnsembleStrategy(strategies=[macd])
        with pytest.raises(ValidationError, match="at least.*35"):
            strategy.generate_signals(data)

    def test_missing_close_column(self):
        """Test validation fails when Close column is missing."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = pd.DataFrame({"Open": range(1, 51)}, index=dates)

        rsi = RSIStrategy()
        strategy = EnsembleStrategy(strategies=[rsi])
        with pytest.raises(ValidationError, match="Close"):
            strategy.generate_signals(data)

    def test_returns_same_index_as_input(self):
        """Test output signals have same index as input data."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = pd.DataFrame({"Close": range(1, 51)}, index=dates)

        rsi = RSIStrategy()
        bb = BollingerBandsStrategy()
        strategy = EnsembleStrategy(strategies=[rsi, bb])
        signals = strategy.generate_signals(data)

        assert signals.index.equals(data.index)
        assert len(signals) == len(data)

    def test_signal_shifting(self):
        """Test signals are shifted by 1 day to prevent look-ahead bias."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = list(range(1, 51))
        data = pd.DataFrame({"Close": prices}, index=dates)

        rsi = RSIStrategy()
        strategy = EnsembleStrategy(strategies=[rsi])
        signals = strategy.generate_signals(data)

        # First row after shift should always be False
        assert not signals["buy"].iloc[0]
        assert not signals["sell"].iloc[0]

    def test_aggregation_counts_votes_correctly(self):
        """Test that min_agreement threshold controls how many strategies must agree.

        RSI and BollingerBands both generate buy signals on a stable-then-sharp-drop
        pattern (17+ buy signals each). MA generates very few. With min_agreement=1
        any single strategy can trigger, producing the most signals. With
        min_agreement=2 (RSI+BB must both agree), fewer signals are produced.
        With min_agreement=3 (all must agree), even fewer.
        """
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        # Stable baseline then sharp drop: RSI goes oversold, price breaks below BB
        prices = [100.0] * 25 + [100 - i * 2 for i in range(1, 36)]
        data = pd.DataFrame({"Close": prices}, index=dates)

        rsi = RSIStrategy()
        bb = BollingerBandsStrategy()
        ma = MovingAverageCrossoverStrategy(short_window=5, long_window=15)

        strategy1 = EnsembleStrategy(strategies=[rsi, bb, ma], min_agreement=1)
        strategy2 = EnsembleStrategy(strategies=[rsi, bb, ma], min_agreement=2)
        strategy3 = EnsembleStrategy(strategies=[rsi, bb, ma], min_agreement=3)

        sigs1 = strategy1.generate_signals(data)
        sigs2 = strategy2.generate_signals(data)
        sigs3 = strategy3.generate_signals(data)

        # Lower threshold => at least as many signals as higher threshold
        assert sigs1["buy"].sum() >= sigs2["buy"].sum() >= sigs3["buy"].sum()
        # min_agreement=1 with RSI+BB both firing should produce signals
        assert sigs1["buy"].any(), "Expected buy signals with min_agreement=1"
        # min_agreement=2 with RSI+BB both firing should still produce signals
        assert sigs2["buy"].any(), "Expected buy signals when RSI and BB both agree"
