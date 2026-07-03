"""Tests for configurable transaction costs (percentage rate + flat per-trade fee).

Covers:
- PortfolioConfig.commission_fixed field and validation.
- Portfolio applying a flat commission on buy and sell.
- BacktestRunnerImpl reading commissions from config, sizing positions so the
  fee is affordable, and recording trade pnl net of both sides' commissions.
- The WFA parallel worker applying an explicitly passed PortfolioConfig.
"""

import pandas as pd
import pytest

from backtest.config import (
    GlobalConfig,
    PortfolioConfig,
    set_config,
)
from backtest.optimization import _run_candidate_worker
from backtest.portfolio import Portfolio
from backtest.runner import BacktestRunnerImpl
from backtest.strategy import BaseStrategy
from backtest.validation import ValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ScriptedStrategy(BaseStrategy):
    """Deterministic strategy: buy on a fixed bar index, sell on another."""

    def __init__(self, buy_bar: int = 2, sell_bar: int = 6):
        self.buy_bar = buy_bar
        self.sell_bar = sell_bar

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals["buy"] = False
        signals["sell"] = False
        signals.iloc[self.buy_bar, signals.columns.get_loc("buy")] = True
        signals.iloc[self.sell_bar, signals.columns.get_loc("sell")] = True
        return signals

    def get_parameters(self) -> dict:
        return {"buy_bar": self.buy_bar, "sell_bar": self.sell_bar}

    def set_parameters(self, params: dict) -> None:
        self.buy_bar = params.get("buy_bar", self.buy_bar)
        self.sell_bar = params.get("sell_bar", self.sell_bar)

    @property
    def warmup_period(self) -> int:
        return 0


def make_step_data(n: int = 10, low: float = 100.0, high: float = 110.0,
                   step_at: int = 5) -> pd.DataFrame:
    """Close-only price data: `low` before step_at, `high` from step_at onward."""
    dates = pd.bdate_range("2024-01-01", periods=n)
    closes = [low if i < step_at else high for i in range(n)]
    return pd.DataFrame({"Close": closes}, index=dates)


# ---------------------------------------------------------------------------
# PortfolioConfig
# ---------------------------------------------------------------------------


class TestPortfolioConfigCommissionFixed:
    def test_default_is_zero(self):
        assert PortfolioConfig().commission_fixed == 0.0

    def test_negative_rejected_by_validate(self):
        config = PortfolioConfig(commission_fixed=-1.0)
        with pytest.raises(ValueError):
            config.validate()

    def test_positive_value_accepted(self):
        config = PortfolioConfig(commission_fixed=3.0)
        config.validate()
        assert config.commission_fixed == 3.0


# ---------------------------------------------------------------------------
# Portfolio flat commission
# ---------------------------------------------------------------------------


class TestPortfolioFlatCommission:
    def test_buy_deducts_fixed_commission(self):
        p = Portfolio(start_capital=10_000.0, commission_fixed=3.0)
        assert p.buy("XYZ", 10, 100.0, pd.Timestamp("2024-01-02"))
        assert p.cash == pytest.approx(10_000.0 - 10 * 100.0 - 3.0)

    def test_sell_deducts_fixed_commission(self):
        p = Portfolio(start_capital=10_000.0, commission_fixed=3.0)
        p.buy("XYZ", 10, 100.0, pd.Timestamp("2024-01-02"))
        cash_after_buy = p.cash
        p.sell("XYZ", 10, 110.0, pd.Timestamp("2024-01-03"))
        assert p.cash == pytest.approx(cash_after_buy + 10 * 110.0 - 3.0)

    def test_rate_and_fixed_combine(self):
        p = Portfolio(start_capital=10_000.0, commission_rate=0.01, commission_fixed=3.0)
        assert p.buy("XYZ", 10, 100.0, pd.Timestamp("2024-01-02"))
        expected_commission = 10 * 100.0 * 0.01 + 3.0
        assert p.cash == pytest.approx(10_000.0 - 10 * 100.0 - expected_commission)

    def test_buy_fails_when_fixed_fee_unaffordable(self):
        p = Portfolio(start_capital=1_000.0, commission_fixed=3.0)
        # 10 shares at 100 costs exactly 1000; the fee pushes it over.
        assert p.buy("XYZ", 10, 100.0, pd.Timestamp("2024-01-02")) is False

    def test_negative_commission_fixed_rejected(self):
        with pytest.raises(ValidationError):
            Portfolio(start_capital=10_000.0, commission_fixed=-1.0)

    def test_default_commission_fixed_is_zero(self):
        p = Portfolio(start_capital=10_000.0)
        p.buy("XYZ", 10, 100.0, pd.Timestamp("2024-01-02"))
        assert p.cash == pytest.approx(10_000.0 - 1_000.0)


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


def _set_commissions(rate: float = 0.0, fixed: float = 0.0, slippage: float = 0.0) -> None:
    config = GlobalConfig()
    config.portfolio.commission_rate = rate
    config.portfolio.commission_fixed = fixed
    config.portfolio.slippage_pct = slippage
    set_config(config)


class TestRunnerAppliesCommissions:
    def test_final_value_reflects_flat_commission(self):
        data = make_step_data()
        strategy = ScriptedStrategy(buy_bar=2, sell_bar=6)

        _set_commissions(fixed=0.0)
        result_free = BacktestRunnerImpl(strategy=strategy, benchmarks=[]).run(data)
        # 100 shares at 100, sold at 110: 10_000 + 1_000
        assert result_free["strategy_returns"].iloc[-1] == pytest.approx(11_000.0)

        _set_commissions(fixed=3.0)
        result_fee = BacktestRunnerImpl(strategy=strategy, benchmarks=[]).run(data)
        # Sizing must leave room for the entry fee: (10_000 - 3) // 100 = 99 shares.
        # Final: 10_000 - 3 - 3 + 99 * 10 = 10_984
        assert result_fee["strategy_returns"].iloc[-1] == pytest.approx(10_984.0)

    def test_trade_pnl_net_of_both_commissions(self):
        data = make_step_data()
        strategy = ScriptedStrategy(buy_bar=2, sell_bar=6)
        _set_commissions(fixed=3.0)
        result = BacktestRunnerImpl(strategy=strategy, benchmarks=[]).run(data)
        trades = result["trades"]
        assert len(trades) == 1
        # 99 shares, entry 100, exit 110, minus $3 each side.
        assert trades[0]["pnl"] == pytest.approx(99 * 10.0 - 6.0)

    def test_percentage_rate_from_config_applied(self):
        data = make_step_data()
        strategy = ScriptedStrategy(buy_bar=2, sell_bar=6)
        _set_commissions(rate=0.01)
        result = BacktestRunnerImpl(strategy=strategy, benchmarks=[]).run(data)
        # Shares sized so value * 1.01 <= 10_000 → 99 shares.
        # Entry commission 99.0, exit commission 108.9.
        expected_final = 10_000.0 - 99.0 - 108.9 + 99 * 10.0
        assert result["strategy_returns"].iloc[-1] == pytest.approx(expected_final)


class TestRunnerAppliesSlippage:
    def test_config_field_default_and_validation(self):
        assert PortfolioConfig().slippage_pct == 0.0
        bad = PortfolioConfig(slippage_pct=-0.01)
        with pytest.raises(ValueError):
            bad.validate()
        bad_high = PortfolioConfig(slippage_pct=1.0)
        with pytest.raises(ValueError):
            bad_high.validate()

    def test_slippage_worsens_both_fills(self):
        data = make_step_data()
        strategy = ScriptedStrategy(buy_bar=2, sell_bar=6)
        _set_commissions(slippage=0.01)
        result = BacktestRunnerImpl(strategy=strategy, benchmarks=[]).run(data)
        trades = result["trades"]
        assert len(trades) == 1
        # Buy fills at 100 * 1.01 = 101 → (10_000) // 101 = 99 shares.
        # Sell fills at 110 * 0.99 = 108.9.
        assert trades[0]["entry"] == pytest.approx(101.0)
        assert trades[0]["exit"] == pytest.approx(108.9)
        assert trades[0]["pnl"] == pytest.approx((108.9 - 101.0) * 99)
        expected_final = 10_000.0 - 99 * 101.0 + 99 * 108.9
        assert result["strategy_returns"].iloc[-1] == pytest.approx(expected_final)

    def test_zero_slippage_fills_at_open(self):
        data = make_step_data()
        strategy = ScriptedStrategy(buy_bar=2, sell_bar=6)
        _set_commissions()
        result = BacktestRunnerImpl(strategy=strategy, benchmarks=[]).run(data)
        assert result["trades"][0]["entry"] == pytest.approx(100.0)
        assert result["trades"][0]["exit"] == pytest.approx(110.0)


# ---------------------------------------------------------------------------
# WFA parallel worker
# ---------------------------------------------------------------------------


class TestWorkerAppliesPortfolioConfig:
    def test_worker_score_reflects_commission(self):
        data = make_step_data(n=20, step_at=5)
        train = data

        score_free = _run_candidate_worker(
            ScriptedStrategy,
            {"buy_bar": 2, "sell_bar": 6},
            train,
            train,
            1,
            "expectancy",
            None,
            portfolio_config=PortfolioConfig(),
        )
        score_fee = _run_candidate_worker(
            ScriptedStrategy,
            {"buy_bar": 2, "sell_bar": 6},
            train,
            train,
            1,
            "expectancy",
            None,
            portfolio_config=PortfolioConfig(commission_fixed=50.0),
        )
        assert score_free == pytest.approx(1_000.0)
        # 99 shares (sized under the $50 fee), $50 each side.
        assert score_fee == pytest.approx(99 * 10.0 - 100.0)

    def test_worker_config_does_not_require_global_state(self):
        # A caller-supplied config must win even if the process global differs.
        _set_commissions(fixed=999.0)
        data = make_step_data(n=20, step_at=5)
        score = _run_candidate_worker(
            ScriptedStrategy,
            {"buy_bar": 2, "sell_bar": 6},
            data,
            data,
            1,
            "expectancy",
            None,
            portfolio_config=PortfolioConfig(commission_fixed=0.0),
        )
        assert score == pytest.approx(1_000.0)
