import math
from datetime import timedelta

import pandas as pd
import pytest

from backtest.metrics import (
    METRICS,
    cagr,
    calmar_ratio,
    expectancy,
    max_drawdown,
    profit_factor,
    recovery_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    ulcer_index,
    win_rate,
)


def _history(values):
    """Build portfolio_history from a list of floats."""
    dates = pd.date_range("2020-01-02", periods=len(values), freq="B")
    return [{"date": d, "value": float(v)} for d, v in zip(dates, values)]


def _trades(*pnls):
    """Build a minimal trades list from P&L values."""
    base = pd.Timestamp("2020-01-02")
    return [
        {
            "entry_date": base + timedelta(days=i * 2),
            "exit_date": base + timedelta(days=i * 2 + 1),
            "entry": 100.0,
            "exit": 100.0 + pnl,
            "shares": 1,
            "pnl": float(pnl),
        }
        for i, pnl in enumerate(pnls)
    ]


class TestTotalReturn:
    def test_positive_return(self):
        h = _history([10000, 11000])
        assert total_return(h, []) == pytest.approx(0.10)

    def test_negative_return(self):
        h = _history([10000, 9000])
        assert total_return(h, []) == pytest.approx(-0.10)

    def test_flat(self):
        h = _history([10000, 10000, 10000])
        assert total_return(h, []) == pytest.approx(0.0)

    def test_empty_history_returns_neginf(self):
        assert total_return([], []) == float("-inf")


class TestCAGR:
    def test_one_year_double(self):
        # 252 trading days, value doubles → CAGR = 1.0 (100%)
        values = [10000.0 * (2 ** (i / 252)) for i in range(253)]
        h = _history(values)
        assert cagr(h, []) == pytest.approx(1.0, rel=1e-3)

    def test_flat(self):
        h = _history([10000] * 253)
        assert cagr(h, []) == pytest.approx(0.0, abs=1e-6)

    def test_empty_history_returns_neginf(self):
        assert cagr([], []) == float("-inf")

    def test_single_bar_returns_neginf(self):
        assert cagr(_history([10000]), []) == float("-inf")


class TestMaxDrawdown:
    def test_peak_then_trough(self):
        # Peak 120, trough 80 → drawdown = (80 - 120) / 120 = -1/3
        h = _history([100, 120, 80, 90])
        assert max_drawdown(h, []) == pytest.approx(-1 / 3, rel=1e-6)

    def test_no_drawdown(self):
        h = _history([100, 110, 120, 130])
        assert max_drawdown(h, []) == pytest.approx(0.0, abs=1e-9)

    def test_returns_negative(self):
        h = _history([100, 80])
        assert max_drawdown(h, []) < 0

    def test_empty_history_returns_neginf(self):
        assert max_drawdown([], []) == float("-inf")


class TestUlcerIndex:
    def test_no_drawdown_is_zero(self):
        h = _history([100, 110, 120, 130])
        # No drawdown → ulcer_index should be 0 (negated → 0)
        assert ulcer_index(h, []) == pytest.approx(0.0, abs=1e-9)

    def test_returns_non_positive(self):
        h = _history([100, 90, 80, 70])
        assert ulcer_index(h, []) <= 0

    def test_worse_drawdown_lower_value(self):
        # Deeper sustained drawdown → more negative ulcer index
        h_shallow = _history([100, 95, 90, 95, 100])
        h_deep = _history([100, 80, 60, 80, 100])
        assert ulcer_index(h_deep, []) < ulcer_index(h_shallow, [])

    def test_empty_history_returns_neginf(self):
        assert ulcer_index([], []) == float("-inf")


class TestMETRICSRegistry:
    def test_registry_contains_expected_keys(self):
        expected = {
            "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "max_drawdown", "ulcer_index", "profit_factor",
            "win_rate", "expectancy", "recovery_factor",
        }
        assert set(METRICS.keys()) == expected

    def test_registry_values_are_callable(self):
        for name, fn in METRICS.items():
            assert callable(fn), f"METRICS['{name}'] is not callable"


class TestSharpeRatio:
    def test_positive_returns_positive_sharpe(self):
        # Steady 0.1% daily gain → positive Sharpe
        values = [10000 * (1.001 ** i) for i in range(100)]
        h = _history(values)
        result = sharpe_ratio(h, [])
        assert result > 0

    def test_flat_returns_neginf(self):
        h = _history([10000] * 50)
        assert sharpe_ratio(h, []) == float("-inf")

    def test_empty_history_returns_neginf(self):
        assert sharpe_ratio([], []) == float("-inf")

    def test_single_bar_returns_neginf(self):
        assert sharpe_ratio(_history([10000]), []) == float("-inf")


class TestSortinoRatio:
    def test_only_upside_returns_high_value(self):
        # All positive returns → downside std = 0 → very high sortino
        values = [10000 * (1.001 ** i) for i in range(100)]
        h = _history(values)
        result = sortino_ratio(h, [])
        assert result > 0

    def test_flat_returns_neginf(self):
        h = _history([10000] * 50)
        assert sortino_ratio(h, []) == float("-inf")

    def test_empty_history_returns_neginf(self):
        assert sortino_ratio([], []) == float("-inf")

    def test_greater_than_sharpe_for_positive_returns(self):
        # When all returns are positive, sortino >= sharpe (no downside penalty)
        values = [10000 * (1.001 ** i) for i in range(100)]
        h = _history(values)
        assert sortino_ratio(h, []) >= sharpe_ratio(h, [])


class TestCalmarRatio:
    def test_positive_cagr_with_drawdown(self):
        # Rising trend with a dip
        values = [100, 110, 90, 120, 130]
        h = _history(values)
        result = calmar_ratio(h, [])
        assert result > 0

    def test_no_drawdown_returns_inf(self):
        # Monotonic rise → no drawdown → calmar = inf
        h = _history([100, 110, 120, 130])
        assert calmar_ratio(h, []) == float("inf")

    def test_empty_history_returns_neginf(self):
        assert calmar_ratio([], []) == float("-inf")


class TestProfitFactor:
    def test_all_winners(self):
        t = _trades(100, 200, 50)
        result = profit_factor([], t)
        assert result == float("inf")

    def test_mixed_trades(self):
        t = _trades(100, -50)  # gross win 100, gross loss 50
        assert profit_factor([], t) == pytest.approx(2.0)

    def test_all_losers(self):
        t = _trades(-100, -50)
        assert profit_factor([], t) == pytest.approx(0.0)

    def test_empty_trades_returns_neginf(self):
        assert profit_factor([], []) == float("-inf")


class TestWinRate:
    def test_all_winners(self):
        t = _trades(10, 20, 30)
        assert win_rate([], t) == pytest.approx(1.0)

    def test_mixed(self):
        t = _trades(10, -5, 20, -3)  # 2 wins, 2 losses
        assert win_rate([], t) == pytest.approx(0.5)

    def test_all_losers(self):
        t = _trades(-10, -20)
        assert win_rate([], t) == pytest.approx(0.0)

    def test_empty_trades_returns_neginf(self):
        assert win_rate([], []) == float("-inf")


class TestExpectancy:
    def test_positive_expectancy(self):
        t = _trades(100, 50, -20)  # mean = 130/3
        assert expectancy([], t) == pytest.approx(130 / 3)

    def test_zero_expectancy(self):
        t = _trades(50, -50)
        assert expectancy([], t) == pytest.approx(0.0)

    def test_empty_trades_returns_neginf(self):
        assert expectancy([], []) == float("-inf")


class TestRecoveryFactor:
    def test_positive_return_with_drawdown(self):
        values = [100, 120, 80, 130]
        h = _history(values)
        result = recovery_factor(h, [])
        assert result > 0

    def test_no_drawdown_returns_inf(self):
        h = _history([100, 110, 120])
        assert recovery_factor(h, []) == float("inf")

    def test_empty_history_returns_neginf(self):
        assert recovery_factor([], []) == float("-inf")
