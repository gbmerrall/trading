"""Tests for backtest/reporting_html.py."""

import os
import tempfile
from unittest.mock import MagicMock

import pandas as pd
import plotly.graph_objects as go

from backtest.reporting_html import (
    ReportData,
    WfaEntry,
    _build_comparison_table,
    _build_wfa_table,
    generate_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_returns(value: float = 10000.0, periods: int = 100) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=periods, freq="B")
    return pd.Series([value] * periods, index=idx)


def make_metrics(total_return: float = 10.0, max_drawdown: float = -5.0) -> dict:
    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate": 55.0,
        "num_trades": 20,
    }


def make_wfa_result() -> MagicMock:
    mock = MagicMock()
    mock.windows = pd.DataFrame([
        {
            "test_start": pd.Timestamp("2021-01-01"),
            "test_end": pd.Timestamp("2021-04-01"),
            "best_params": {"roc_period": 12, "roc_threshold": 0.05},
            "sharpe_ratio": 0.85,
            "total_return": 0.12,
            "n_trades": 5,
        },
        {
            "test_start": pd.Timestamp("2021-04-01"),
            "test_end": pd.Timestamp("2021-07-01"),
            "best_params": {"roc_period": 9, "roc_threshold": 0.03},
            "sharpe_ratio": -0.20,
            "total_return": -0.04,
            "n_trades": 3,
        },
    ])
    mock.summary = {
        "sharpe_ratio": 0.45,
        "total_return": 0.08,
        "max_drawdown": -0.06,
        "win_rate": 0.58,
        "n_windows": 2,
        "n_windows_with_trades": 2,
    }
    mock.best_params_overall = {"roc_period": 12, "roc_threshold": 0.05}
    return mock


def make_report_data(wfa_result=None) -> ReportData:
    returns = make_returns()
    wfa_entries = []
    if wfa_result is not None:
        wfa_entries = [WfaEntry(
            label="MomentumStrategy",
            result=wfa_result,
            fig_equity=go.Figure(),
            fig_params=go.Figure(),
        )]
    return ReportData(
        ticker="TEST",
        start_date="2020-01-01",
        end_date="2023-01-01",
        start_capital=10000.0,
        results=[("StrategyA", make_metrics(12.0), returns)],
        bh_metrics=make_metrics(8.0),
        bh_returns=returns,
        dca_metrics=make_metrics(6.0),
        dca_returns=returns,
        fig_comparison=go.Figure(),
        wfa_entries=wfa_entries,
    )


# ---------------------------------------------------------------------------
# ReportData
# ---------------------------------------------------------------------------

def test_report_data_instantiates():
    data = ReportData(
        ticker="TEST",
        start_date="2020-01-01",
        end_date="2023-01-01",
        start_capital=10000.0,
        results=[("StrategyA", make_metrics(), make_returns())],
        bh_metrics=make_metrics(8.0),
        bh_returns=make_returns(),
        dca_metrics=make_metrics(6.0),
        dca_returns=make_returns(),
        fig_comparison=go.Figure(),
        wfa_entries=[],
    )
    assert data.ticker == "TEST"
    assert data.wfa_entries == []


# ---------------------------------------------------------------------------
# _build_comparison_table
# ---------------------------------------------------------------------------

def test_comparison_table_contains_strategy_name():
    returns = make_returns()
    results = [("MyStrategy", make_metrics(total_return=15.3), returns)]
    html = _build_comparison_table(
        results,
        make_metrics(8.0),
        make_returns(),
        make_metrics(6.0),
        make_returns(),
    )
    assert "MyStrategy" in html
    assert "+15.3%" in html


def test_comparison_table_ranks_strategies():
    r1 = make_returns()
    r2 = make_returns()
    results = [
        ("First", make_metrics(total_return=20.0), r1),
        ("Second", make_metrics(total_return=10.0), r2),
    ]
    html = _build_comparison_table(results, make_metrics(), make_returns(), make_metrics(), make_returns())
    assert html.index("First") < html.index("Second")


def test_comparison_table_contains_benchmarks():
    html = _build_comparison_table(
        [],
        make_metrics(8.0),
        make_returns(),
        make_metrics(6.0),
        make_returns(),
    )
    assert "Buy" in html and "Hold" in html
    assert "DCA" in html
    assert "benchmark-row" in html


def test_comparison_table_positive_return_class():
    results = [("Up", make_metrics(total_return=5.0), make_returns())]
    html = _build_comparison_table(results, make_metrics(), make_returns(), make_metrics(), make_returns())
    assert 'class="positive"' in html


def test_comparison_table_negative_return_class():
    results = [("Down", make_metrics(total_return=-5.0), make_returns())]
    html = _build_comparison_table(results, make_metrics(), make_returns(), make_metrics(), make_returns())
    assert 'class="negative"' in html


# ---------------------------------------------------------------------------
# _build_wfa_table
# ---------------------------------------------------------------------------

def test_wfa_table_contains_params():
    html = _build_wfa_table(make_wfa_result())
    assert "roc_period=12" in html
    assert "roc_threshold=0.05" in html


def test_wfa_table_contains_returns():
    html = _build_wfa_table(make_wfa_result())
    assert "+12.0%" in html
    assert "-4.0%" in html


def test_wfa_table_colours_positive_and_negative():
    html = _build_wfa_table(make_wfa_result())
    assert 'class="positive"' in html
    assert 'class="negative"' in html


def test_wfa_table_contains_sharpe():
    html = _build_wfa_table(make_wfa_result())
    assert "0.85" in html
    assert "-0.20" in html


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

def test_generate_report_writes_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.html")
        generate_report(make_report_data(), output_path)
        assert os.path.exists(output_path)


def test_generate_report_contains_ticker():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.html")
        generate_report(make_report_data(), output_path)
        content = open(output_path).read()
        assert "TEST" in content


def test_generate_report_contains_strategy_table():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.html")
        generate_report(make_report_data(), output_path)
        content = open(output_path).read()
        assert "StrategyA" in content
        assert "Strategy Comparison" in content


def test_generate_report_no_wfa_shows_placeholder():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.html")
        generate_report(make_report_data(wfa_result=None), output_path)
        content = open(output_path).read()
        assert "No WFA configured" in content


def test_generate_report_with_wfa_shows_table():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.html")
        generate_report(make_report_data(wfa_result=make_wfa_result()), output_path)
        content = open(output_path).read()
        assert "Walk-Forward Analysis" in content
        assert "roc_period=12" in content
        assert "MomentumStrategy" in content


def test_generate_report_creates_parent_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "nested", "dir", "report.html")
        generate_report(make_report_data(), output_path)
        assert os.path.exists(output_path)
