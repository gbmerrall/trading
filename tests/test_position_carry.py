"""Tests for position carry-in at WFA window boundaries.

Each WFA window starts a fresh portfolio in cash. For event-based strategies
(crossovers), the entry event may have fired before the window started; without
carry-in the whole window would sit in cash despite the strategy being "long".
_window_signals_with_carry forces a buy on the window's first bar when the most
recent signal event before the window was a buy.
"""

import pandas as pd

from backtest.optimization import (
    WalkForwardOptimizer,
    _carry_in_position,
    _window_signals_with_carry,
)
from backtest.strategy import BaseStrategy


def _signals(index, buy_at=(), sell_at=()):
    signals = pd.DataFrame({"buy": False, "sell": False}, index=index)
    for i in buy_at:
        signals.iloc[i, signals.columns.get_loc("buy")] = True
    for i in sell_at:
        signals.iloc[i, signals.columns.get_loc("sell")] = True
    return signals


class TestCarryInPosition:
    def setup_method(self):
        self.index = pd.bdate_range("2024-01-01", periods=20)

    def test_no_prior_events_means_flat(self):
        signals = _signals(self.index)
        assert _carry_in_position(signals, self.index[10]) is False

    def test_prior_buy_means_long(self):
        signals = _signals(self.index, buy_at=[5])
        assert _carry_in_position(signals, self.index[10]) is True

    def test_buy_then_sell_means_flat(self):
        signals = _signals(self.index, buy_at=[3], sell_at=[7])
        assert _carry_in_position(signals, self.index[10]) is False

    def test_sell_then_buy_means_long(self):
        signals = _signals(self.index, buy_at=[7], sell_at=[3])
        assert _carry_in_position(signals, self.index[10]) is True

    def test_event_at_window_start_not_counted_as_prior(self):
        # The event on the boundary bar belongs to the window itself.
        signals = _signals(self.index, buy_at=[10])
        assert _carry_in_position(signals, self.index[10]) is False


class TestWindowSignalsWithCarry:
    def setup_method(self):
        self.index = pd.bdate_range("2024-01-01", periods=20)
        self.data = pd.DataFrame({"Close": [100.0] * 20}, index=self.index)

    def test_forces_buy_on_first_bar_when_carried_long(self):
        signals = _signals(self.index, buy_at=[5])
        window = self.data.iloc[10:15]
        out = _window_signals_with_carry(signals, window)
        assert bool(out["buy"].iloc[0]) is True

    def test_no_forced_buy_when_flat(self):
        signals = _signals(self.index, buy_at=[3], sell_at=[7])
        window = self.data.iloc[10:15]
        out = _window_signals_with_carry(signals, window)
        assert bool(out["buy"].iloc[0]) is False

    def test_no_forced_buy_when_first_bar_sells(self):
        # Carried long but the window's first bar already exits: entering and
        # exiting on the same bar would be a phantom round trip.
        signals = _signals(self.index, buy_at=[5], sell_at=[10])
        window = self.data.iloc[10:15]
        out = _window_signals_with_carry(signals, window)
        assert bool(out["buy"].iloc[0]) is False
        assert bool(out["sell"].iloc[0]) is True

    def test_original_signals_not_mutated(self):
        signals = _signals(self.index, buy_at=[5])
        window = self.data.iloc[10:15]
        _window_signals_with_carry(signals, window)
        assert bool(signals["buy"].iloc[10]) is False


class SingleBuyStrategy(BaseStrategy):
    """Emits exactly one buy event early in the series and never sells.

    Without carry-in, every WFA test window after the event holds pure cash;
    with carry-in, each window re-enters at its first bar.
    """

    def __init__(self, buy_bar: int = 5):
        self.buy_bar = buy_bar

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame({"buy": False, "sell": False}, index=data.index)
        if len(signals) > self.buy_bar:
            signals.iloc[self.buy_bar, signals.columns.get_loc("buy")] = True
        return signals

    def get_parameters(self) -> dict:
        return {"buy_bar": self.buy_bar}

    def set_parameters(self, params: dict) -> None:
        self.buy_bar = params.get("buy_bar", self.buy_bar)

    @property
    def warmup_period(self) -> int:
        return 0


class TestCarryThroughWalkForward:
    def test_test_windows_participate_after_early_entry(self):
        # Rising market; single buy event at bar 5, long forever after.
        index = pd.bdate_range("2023-01-01", periods=200)
        prices = [100.0 * (1.002 ** i) for i in range(200)]
        data = pd.DataFrame({"Close": prices}, index=index)

        opt = WalkForwardOptimizer(
            strategy_class=SingleBuyStrategy,
            param_space={"buy_bar": [5]},
            data=data,
            train_size=60,
            test_size=30,
            min_trades=0,
        )
        result = opt.run()

        # Every test window starts after bar 5, so every window must carry the
        # long position in and capture the uptrend.
        assert result.summary["n_windows_with_trades"] == result.summary["n_windows"]
        assert result.summary["total_return"] > 0.10
