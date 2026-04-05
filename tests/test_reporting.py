"""Tests for backtest.reporting — WFA visualization utilities."""

import math

import pandas as pd
import pytest

from backtest.optimization import WalkForwardOptimizer
from backtest.strategy import ConsecutiveDaysStrategy


# ---------------------------------------------------------------------------
# Shared fixture: run a small WFA to produce a real WalkForwardResult
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wfa_result():
    """Small WFA result on synthetic oscillating data."""
    n = 400
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = [100.0 + 10 * math.sin(i * math.pi / 3) for i in range(n)]
    data = pd.DataFrame({"Close": prices}, index=dates)

    opt = WalkForwardOptimizer(
        strategy_class=ConsecutiveDaysStrategy,
        param_space={"consecutive_days": [1, 2, 3]},
        data=data,
        train_size=150,
        test_size=50,
        window_type="sliding",
        min_trades=0,
    )
    return opt.run()


# ---------------------------------------------------------------------------
# plot_equity_curve
# ---------------------------------------------------------------------------

class TestPlotEquityCurve:
    def test_returns_figure(self, wfa_result):
        from plotly.graph_objects import Figure
        from backtest.reporting import plot_equity_curve
        fig = plot_equity_curve(wfa_result)
        assert isinstance(fig, Figure)

    def test_has_one_trace(self, wfa_result):
        from backtest.reporting import plot_equity_curve
        fig = plot_equity_curve(wfa_result)
        assert len(fig.data) == 1

    def test_chained_curve_starts_at_start_capital(self, wfa_result):
        from backtest.reporting import plot_equity_curve, _chain_equity_curve
        start_capital = 10_000.0
        chained = _chain_equity_curve(
            wfa_result.equity_curve, wfa_result.windows, start_capital
        )
        assert chained.iloc[0] == pytest.approx(start_capital, rel=1e-6)

    def test_chained_curve_has_no_large_jumps(self, wfa_result):
        """No value should jump by more than 50% between consecutive bars."""
        from backtest.reporting import _chain_equity_curve
        chained = _chain_equity_curve(
            wfa_result.equity_curve, wfa_result.windows, 10_000.0
        )
        pct_changes = chained.pct_change().dropna().abs()
        assert (pct_changes > 0.5).sum() == 0, (
            f"Large jumps detected at: {chained[pct_changes > 0.5].index.tolist()}"
        )

    def test_chain_false_returns_raw_length(self, wfa_result):
        from backtest.reporting import plot_equity_curve
        fig = plot_equity_curve(wfa_result, chain=False)
        # Trace x values should match the raw equity_curve index length
        assert len(fig.data[0].x) == len(wfa_result.equity_curve)


# ---------------------------------------------------------------------------
# plot_parameter_stability
# ---------------------------------------------------------------------------

class TestPlotParameterStability:
    def test_returns_figure(self, wfa_result):
        from plotly.graph_objects import Figure
        from backtest.reporting import plot_parameter_stability
        fig = plot_parameter_stability(wfa_result)
        assert isinstance(fig, Figure)

    def test_has_one_trace_per_param(self, wfa_result):
        from backtest.reporting import plot_parameter_stability
        n_params = len(wfa_result.best_params_overall)
        fig = plot_parameter_stability(wfa_result)
        # Each param produces one trace
        assert len(fig.data) == n_params

    def test_trace_length_matches_window_count(self, wfa_result):
        from backtest.reporting import plot_parameter_stability
        fig = plot_parameter_stability(wfa_result)
        n_windows = len(wfa_result.windows)
        for trace in fig.data:
            assert len(trace.x) == n_windows

    def test_explicit_param_subset(self, wfa_result):
        from backtest.reporting import plot_parameter_stability
        fig = plot_parameter_stability(wfa_result, params=["consecutive_days"])
        assert len(fig.data) == 1


# ---------------------------------------------------------------------------
# plot_metrics_by_window
# ---------------------------------------------------------------------------

class TestPlotMetricsByWindow:
    def test_returns_figure(self, wfa_result):
        from plotly.graph_objects import Figure
        from backtest.reporting import plot_metrics_by_window
        fig = plot_metrics_by_window(wfa_result)
        assert isinstance(fig, Figure)

    def test_default_metrics_produce_three_traces(self, wfa_result):
        from backtest.reporting import plot_metrics_by_window
        fig = plot_metrics_by_window(wfa_result)
        assert len(fig.data) >= 3

    def test_custom_metric_list(self, wfa_result):
        from backtest.reporting import plot_metrics_by_window
        fig = plot_metrics_by_window(wfa_result, metrics=["total_return"])
        assert len(fig.data) == 1

    def test_bar_count_matches_window_count(self, wfa_result):
        from backtest.reporting import plot_metrics_by_window
        fig = plot_metrics_by_window(wfa_result, metrics=["sharpe_ratio"])
        n_windows = len(wfa_result.windows)
        assert len(fig.data[0].x) == n_windows

    def test_unknown_metric_silently_excluded(self, wfa_result):
        from backtest.reporting import plot_metrics_by_window
        # "not_a_metric" doesn't exist in windows columns — should be dropped
        fig = plot_metrics_by_window(
            wfa_result, metrics=["sharpe_ratio", "not_a_metric"]
        )
        assert len(fig.data) == 1
