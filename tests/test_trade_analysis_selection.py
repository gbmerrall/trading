"""Tests for WFA candidate selection and commission wiring in trade_analysis.py.

Verifies:
- WFA runs on every strategy with a parameter grid (no in-sample Top-5 shortlist).
- Candidates preserve STRATEGIES order and carry (name, class, grid, base_params).
- WFA entries are ranked by out-of-sample Sharpe from the WFA summary.
- The module-level commission settings are applied to the global config.
"""

import importlib
import sys
from types import SimpleNamespace

import pytest

from backtest.strategy import RegimeFilteredStrategy


def _load():
    """Import trade_analysis, patching sys.argv so the ticker guard passes."""
    orig = sys.argv[:]
    sys.argv = ["trade_analysis.py", "SPY"]
    try:
        import trade_analysis
        importlib.reload(trade_analysis)
        return trade_analysis
    finally:
        sys.argv = orig


@pytest.fixture(scope="module")
def ta():
    return _load()


class TestSelectWfaCandidates:
    def test_all_gridded_strategies_selected(self, ta):
        candidates = ta.select_wfa_candidates()
        names = [c[0] for c in candidates]
        assert names == [n for n, _ in ta.STRATEGIES if n in ta.WFA_PARAM_GRIDS]

    def test_ensembles_excluded(self, ta):
        names = [c[0] for c in ta.select_wfa_candidates()]
        assert not any(n.startswith("Ensemble") for n in names)

    def test_standard_entry_shape(self, ta):
        candidates = {c[0]: c for c in ta.select_wfa_candidates()}
        name, cls, grid, base_params = candidates["Breakout(20d)"]
        assert grid == ta.WFA_PARAM_GRIDS["Breakout(20d)"]
        assert base_params is None

    def test_regime_wrapper_entry_shape(self, ta):
        candidates = {c[0]: c for c in ta.select_wfa_candidates()}
        name, cls, grid, base_params = candidates["Breakout(20d)+ADX(25)"]
        assert cls is RegimeFilteredStrategy
        assert grid == ta.WFA_PARAM_GRIDS["Breakout(20d)+ADX(25)"]["param_grid"]
        assert base_params == ta.WFA_PARAM_GRIDS["Breakout(20d)+ADX(25)"]["base_params"]


class TestSortWfaEntries:
    def test_sorted_by_oos_sharpe_descending(self, ta):
        entries = [
            SimpleNamespace(label="low", result=SimpleNamespace(summary={"sharpe_ratio": 0.2})),
            SimpleNamespace(label="high", result=SimpleNamespace(summary={"sharpe_ratio": 1.5})),
            SimpleNamespace(label="mid", result=SimpleNamespace(summary={"sharpe_ratio": 0.9})),
        ]
        ranked = ta.sort_wfa_entries(entries)
        assert [e.label for e in ranked] == ["high", "mid", "low"]

    def test_non_finite_sharpe_ranked_last(self, ta):
        entries = [
            SimpleNamespace(
                label="bad", result=SimpleNamespace(summary={"sharpe_ratio": float("-inf")})
            ),
            SimpleNamespace(
                label="nan", result=SimpleNamespace(summary={"sharpe_ratio": float("nan")})
            ),
            SimpleNamespace(label="ok", result=SimpleNamespace(summary={"sharpe_ratio": 0.1})),
        ]
        ranked = ta.sort_wfa_entries(entries)
        assert ranked[0].label == "ok"


class TestCommissionConfig:
    def test_commission_fixed_constant(self, ta):
        assert ta.COMMISSION_FIXED == 3.0

    def test_apply_run_config_sets_global_config(self, ta):
        from backtest.config import get_portfolio_config

        ta.apply_run_config()
        assert get_portfolio_config().commission_fixed == 3.0
        assert get_portfolio_config().commission_rate == ta.COMMISSION_RATE
        assert get_portfolio_config().start_capital == ta.START_CAPITAL

    def test_apply_run_config_sets_slippage(self, ta):
        from backtest.config import get_portfolio_config

        ta.apply_run_config()
        assert get_portfolio_config().slippage_pct == ta.SLIPPAGE_PCT

    def test_min_trades_constant(self, ta):
        assert ta.MIN_TRADES == 3


class TestFixedParamsBacktest:
    def test_returns_oos_metrics_for_card_params(self, ta):
        import pandas as pd

        from backtest.strategy import BreakoutStrategy

        ta.apply_run_config()
        dates = pd.bdate_range("2023-01-01", periods=300)
        prices = [100.0 * (1.001 ** i) for i in range(300)]
        # Mild oscillation so breakouts fire repeatedly
        prices = [p * (1.02 if i % 10 < 5 else 0.99) for i, p in enumerate(prices)]
        data = pd.DataFrame({"Close": prices}, index=dates)

        baseline = ta.run_fixed_params_backtest(
            data=data,
            strategy_class=BreakoutStrategy,
            params={"lookback_period": 10},
            base_params=None,
            oos_start=dates[100],
            oos_end=dates[-1],
        )
        assert set(baseline) == {
            "sharpe_ratio", "total_return", "max_drawdown", "n_trades",
        }
        assert isinstance(baseline["n_trades"], int)

    def test_card_includes_fixed_params_baseline(self, ta):
        from backtest.strategy_card import CardCandidate, build_card

        candidate = CardCandidate(
            label="Breakout(20d)",
            strategy_class="BreakoutStrategy",
            params={"lookback_period": 20},
            summary={"sharpe_ratio": 1.0, "max_drawdown": -0.1},
            fixed_baseline={"sharpe_ratio": 0.8, "total_return": 0.2,
                            "max_drawdown": -0.12, "n_trades": 14},
        )
        card = build_card(
            ticker="SPY",
            start_date="2020-01-01",
            end_date="2026-05-01",
            start_capital=10_000.0,
            candidates=[candidate],
        )
        assert card["candidates"][0]["fixed_params_baseline"] == {
            "sharpe_ratio": 0.8, "total_return": 0.2,
            "max_drawdown": -0.12, "n_trades": 14,
        }
