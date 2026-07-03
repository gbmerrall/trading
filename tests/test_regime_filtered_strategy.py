"""Tests for RegimeFilteredStrategy.

All tests use in-memory synthetic data — no network calls.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.strategy import (
    BreakoutStrategy,
    MACDStrategy,
    RegimeFilteredStrategy,
    RSIStrategy,
)
from backtest.validation import ValidationError


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_data(n: int = 300, trend: bool = True, seed: int = 0) -> pd.DataFrame:
    """Return a OHLCV DataFrame with a controllable trend direction.

    Args:
        n: Number of rows.
        trend: If True, prices drift upward; if False, drift downward.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    drift = 0.3 if trend else -0.3
    closes = 100.0 + (rng.normal(drift, 1.0, n)).cumsum()
    closes = np.maximum(closes, 1.0)
    idx = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame(
        {
            "Open": closes * 0.999,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": 1_000_000,
        },
        index=idx,
    )


@pytest.fixture()
def data() -> pd.DataFrame:
    return _make_data(300)


@pytest.fixture()
def base_strategy() -> BreakoutStrategy:
    return BreakoutStrategy(lookback_period=20)


# ---------------------------------------------------------------------------
# Initialization & validation
# ---------------------------------------------------------------------------

class TestRegimeFilteredStrategyInit:
    def test_default_adx_construction(self, base_strategy):
        s = RegimeFilteredStrategy(base_strategy)
        assert s.regime_type == "ADX"
        assert s.adx_threshold == 25.0
        assert s.base_strategy is base_strategy

    def test_adx_explicit(self, base_strategy):
        s = RegimeFilteredStrategy(base_strategy, regime_type="ADX", adx_threshold=30.0)
        assert s.adx_threshold == 30.0

    def test_sma_construction(self, base_strategy):
        s = RegimeFilteredStrategy(base_strategy, regime_type="SMA", sma_period=200)
        assert s.regime_type == "SMA"
        assert s.sma_period == 200

    def test_invalid_regime_type_raises(self, base_strategy):
        with pytest.raises(ValidationError, match="regime_type"):
            RegimeFilteredStrategy(base_strategy, regime_type="MACD")

    def test_invalid_regime_type_case_sensitive(self, base_strategy):
        with pytest.raises(ValidationError, match="regime_type"):
            RegimeFilteredStrategy(base_strategy, regime_type="adx")

    def test_invalid_adx_threshold_zero(self, base_strategy):
        with pytest.raises(ValidationError):
            RegimeFilteredStrategy(base_strategy, regime_type="ADX", adx_threshold=0.0)

    def test_invalid_adx_threshold_negative(self, base_strategy):
        with pytest.raises(ValidationError):
            RegimeFilteredStrategy(base_strategy, regime_type="ADX", adx_threshold=-5.0)

    def test_invalid_sma_period_zero(self, base_strategy):
        with pytest.raises(ValidationError):
            RegimeFilteredStrategy(base_strategy, regime_type="SMA", sma_period=0)

    def test_invalid_sma_period_negative(self, base_strategy):
        with pytest.raises(ValidationError):
            RegimeFilteredStrategy(base_strategy, regime_type="SMA", sma_period=-10)

    def test_base_strategy_must_be_base_strategy(self):
        with pytest.raises(ValidationError, match="BaseStrategy"):
            RegimeFilteredStrategy("not_a_strategy")  # type: ignore[arg-type]

    def test_base_kwargs_forwarded_to_base_strategy(self):
        """Extra kwargs are forwarded to base_strategy.set_parameters() for WFA support."""
        s = RegimeFilteredStrategy(
            BreakoutStrategy(lookback_period=20),
            regime_type="SMA",
            sma_period=150,
            lookback_period=30,  # belongs to BreakoutStrategy, not the wrapper
        )
        assert s.base_strategy.lookback_period == 30

    def test_base_kwargs_do_not_override_wrapper_params(self):
        """Forwarded kwargs must not corrupt wrapper-owned attributes."""
        s = RegimeFilteredStrategy(
            BreakoutStrategy(lookback_period=20),
            regime_type="ADX",
            adx_threshold=25.0,
            lookback_period=30,
        )
        assert s.regime_type == "ADX"
        assert s.adx_threshold == pytest.approx(25.0)

    def test_no_base_kwargs_leaves_base_strategy_unchanged(self):
        """When no extra kwargs are given the base strategy keeps its original params."""
        s = RegimeFilteredStrategy(
            BreakoutStrategy(lookback_period=20),
            regime_type="ADX",
        )
        assert s.base_strategy.lookback_period == 20


# ---------------------------------------------------------------------------
# warmup_period
# ---------------------------------------------------------------------------

class TestWarmupPeriod:
    def test_adx_warmup_is_max_of_base_and_14(self):
        # base warmup=20, ADX lookback=14 → expect 20
        s = RegimeFilteredStrategy(BreakoutStrategy(lookback_period=20), regime_type="ADX")
        assert s.warmup_period == 20

    def test_adx_warmup_when_adx_longer_than_base(self):
        # RSI(7) warmup=7, ADX lookback=14 → expect 14
        s = RegimeFilteredStrategy(RSIStrategy(period=7), regime_type="ADX")
        assert s.warmup_period == 14

    def test_sma_warmup_uses_sma_period(self):
        # base warmup=20, SMA=200 → expect 200
        s = RegimeFilteredStrategy(
            BreakoutStrategy(lookback_period=20), regime_type="SMA", sma_period=200
        )
        assert s.warmup_period == 200

    def test_sma_warmup_when_base_longer(self):
        # BreakoutStrategy(lookback_period=50), SMA=30 → expect 50
        s = RegimeFilteredStrategy(
            BreakoutStrategy(lookback_period=50), regime_type="SMA", sma_period=30
        )
        assert s.warmup_period == 50


# ---------------------------------------------------------------------------
# get_parameters / set_parameters
# ---------------------------------------------------------------------------

class TestParameters:
    def test_get_parameters_contains_regime_keys(self, base_strategy):
        s = RegimeFilteredStrategy(base_strategy, regime_type="ADX", adx_threshold=25.0)
        params = s.get_parameters()
        assert "regime_type" in params
        assert "adx_threshold" in params

    def test_get_parameters_contains_base_keys(self):
        s = RegimeFilteredStrategy(
            BreakoutStrategy(lookback_period=20), regime_type="ADX"
        )
        params = s.get_parameters()
        assert "lookback_period" in params

    def test_get_parameters_sma_has_sma_period(self, base_strategy):
        s = RegimeFilteredStrategy(base_strategy, regime_type="SMA", sma_period=150)
        params = s.get_parameters()
        assert params["sma_period"] == 150

    def test_set_parameters_updates_regime_threshold(self, base_strategy):
        s = RegimeFilteredStrategy(base_strategy, regime_type="ADX", adx_threshold=25.0)
        s.set_parameters({"adx_threshold": 35.0})
        assert s.adx_threshold == 35.0

    def test_set_parameters_updates_sma_period(self, base_strategy):
        s = RegimeFilteredStrategy(base_strategy, regime_type="SMA", sma_period=200)
        s.set_parameters({"sma_period": 100})
        assert s.sma_period == 100

    def test_set_parameters_delegates_base_params(self):
        bo = BreakoutStrategy(lookback_period=20)
        s = RegimeFilteredStrategy(bo, regime_type="ADX")
        s.set_parameters({"lookback_period": 30})
        assert bo.lookback_period == 30

    def test_set_parameters_ignores_unknown_keys(self, base_strategy):
        s = RegimeFilteredStrategy(base_strategy)
        s.set_parameters({"unknown_param": 99})  # should not raise


# ---------------------------------------------------------------------------
# generate_signals — structural invariants
# ---------------------------------------------------------------------------

class TestGenerateSignalsStructure:
    def test_returns_dataframe_with_buy_sell(self, base_strategy, data):
        s = RegimeFilteredStrategy(base_strategy, regime_type="ADX")
        signals = s.generate_signals(data)
        assert "buy" in signals.columns
        assert "sell" in signals.columns

    def test_index_matches_input(self, base_strategy, data):
        s = RegimeFilteredStrategy(base_strategy, regime_type="ADX")
        signals = s.generate_signals(data)
        pd.testing.assert_index_equal(signals.index, data.index)

    def test_columns_are_bool(self, base_strategy, data):
        s = RegimeFilteredStrategy(base_strategy, regime_type="ADX")
        signals = s.generate_signals(data)
        assert signals["buy"].dtype == bool
        assert signals["sell"].dtype == bool

    def test_requires_high_low_close_for_adx(self, base_strategy):
        data_no_high = _make_data(300).drop(columns=["High"])
        s = RegimeFilteredStrategy(base_strategy, regime_type="ADX")
        with pytest.raises((ValidationError, Exception)):
            s.generate_signals(data_no_high)

    def test_sma_works_with_close_only_base(self, data):
        # RSIStrategy only needs Close; SMA filter also only needs Close
        s = RegimeFilteredStrategy(RSIStrategy(period=14), regime_type="SMA", sma_period=50)
        signals = s.generate_signals(data)
        assert "buy" in signals.columns


# ---------------------------------------------------------------------------
# generate_signals — veto logic
# ---------------------------------------------------------------------------

class TestVetoLogic:
    def _force_all_buys(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a signals DataFrame that has buy=True everywhere."""
        signals = pd.DataFrame(index=data.index)
        signals["buy"] = True
        signals["sell"] = False
        return signals

    def test_buy_requires_regime_favorable(self, data):
        """When regime is never favorable, no buys should pass through."""

        class AlwaysBuyStrategy(BreakoutStrategy):
            def generate_signals(self, data):  # type: ignore[override]
                sig = pd.DataFrame(index=data.index)
                sig["buy"] = True
                sig["sell"] = False
                return sig

        # Set an impossibly high ADX threshold so regime is never favorable
        s = RegimeFilteredStrategy(
            AlwaysBuyStrategy(lookback_period=20),
            regime_type="ADX",
            adx_threshold=200.0,  # ADX never exceeds 100
        )
        signals = s.generate_signals(data)
        assert not signals["buy"].any(), "No buys should pass with ADX threshold=200"

    def test_sell_not_vetoed_by_regime(self, data):
        """Sell signals from the base strategy must survive even in unfavorable regime."""

        class AlwaysSellStrategy(BreakoutStrategy):
            def generate_signals(self, data):  # type: ignore[override]
                sig = pd.DataFrame(index=data.index)
                sig["buy"] = False
                sig["sell"] = True
                return sig

        s = RegimeFilteredStrategy(
            AlwaysSellStrategy(lookback_period=20),
            regime_type="ADX",
            adx_threshold=200.0,
        )
        signals = s.generate_signals(data)
        assert signals["sell"].any(), "Sell signals must not be vetoed by the regime filter"

    def test_sma_favorable_above_sma(self):
        """With a very short SMA, a rising price series should allow buys through."""

        class AlwaysBuyStrategy(BreakoutStrategy):
            def generate_signals(self, data):  # type: ignore[override]
                sig = pd.DataFrame(index=data.index)
                sig["buy"] = True
                sig["sell"] = False
                return sig

        # Strongly uptrending data so close > SMA(5) almost always
        trending_data = _make_data(n=100, trend=True, seed=7)
        s = RegimeFilteredStrategy(
            AlwaysBuyStrategy(lookback_period=5),
            regime_type="SMA",
            sma_period=5,
        )
        signals = s.generate_signals(trending_data)
        # With a strong uptrend and short SMA, at least some buys must pass
        assert signals["buy"].sum() > 0, "Some buys should pass in a rising regime"

    def test_filtered_buys_subset_of_base_buys(self, data):
        """Regime-filtered buy signals must be a strict subset of base strategy buys."""
        base = BreakoutStrategy(lookback_period=20)
        base_signals = base.generate_signals(data)

        s = RegimeFilteredStrategy(base, regime_type="ADX", adx_threshold=25.0)
        filtered_signals = s.generate_signals(data)

        # Every filtered buy must also be a base buy
        extra_buys = filtered_signals["buy"] & (base_signals["buy"] == False)
        assert not extra_buys.any(), "Filtered strategy introduced extra buys not in base"

    def test_no_lookahead_first_row_buy_always_false(self, data):
        """First row must never be a buy — regime is shifted so day 0 is always unfavorable."""

        class AlwaysBuyStrategy(BreakoutStrategy):
            def generate_signals(self, data):  # type: ignore[override]
                sig = pd.DataFrame(index=data.index)
                sig["buy"] = True
                sig["sell"] = False
                return sig

        for regime in ("ADX", "SMA"):
            s = RegimeFilteredStrategy(
                AlwaysBuyStrategy(lookback_period=1),
                regime_type=regime,
                adx_threshold=0.01,
                sma_period=1,
            )
            signals = s.generate_signals(data)
            assert not signals["buy"].iloc[0], f"First row buy must be False for {regime}"
