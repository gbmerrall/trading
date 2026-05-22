# HTML Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace terminal table output and separate chart windows with a single self-contained `output/report.html` file containing styled tables and interactive plotly charts.

**Architecture:** A new `backtest/reporting_html.py` module owns all HTML generation. It defines a `ReportData` dataclass and three functions: `_build_comparison_table`, `_build_wfa_table`, and `generate_report`. `bhp_analysis.py` is simplified to assemble a `ReportData` and call `generate_report` — all `print` table functions and `fig.show()` calls are removed.

**Tech Stack:** Python stdlib (`os`, `dataclasses`, `datetime`), plotly (already installed), pytest

---

## File Map

- **Create:** `backtest/reporting_html.py` — HTML generation module
- **Create:** `tests/test_reporting_html.py` — unit tests
- **Modify:** `bhp_analysis.py` — remove old output functions, wire up ReportData

---

### Task 1: Create `reporting_html.py` skeleton with `ReportData`

**Files:**
- Create: `backtest/reporting_html.py`
- Create: `tests/test_reporting_html.py`

- [ ] **Step 1: Write the failing import test**

```python
# tests/test_reporting_html.py
import pandas as pd
import plotly.graph_objects as go

from backtest.reporting_html import ReportData, generate_report, _build_comparison_table, _build_wfa_table


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


def test_report_data_instantiates():
    data = ReportData(
        ticker="TEST",
        start_date="2020-01-01",
        end_date="2023-01-01",
        start_capital=10000.0,
        results=[("StrategyA", make_metrics(), make_returns())],
        bh_metrics=make_metrics(8.0),
        bh_returns=make_returns(10000.0),
        dca_metrics=make_metrics(6.0),
        dca_returns=make_returns(10000.0),
        fig_comparison=go.Figure(),
        wfa_result=None,
        wfa_label="",
        fig_wfa_equity=None,
        fig_wfa_params=None,
    )
    assert data.ticker == "TEST"
    assert data.wfa_result is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_reporting_html.py::test_report_data_instantiates -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'backtest.reporting_html'`

- [ ] **Step 3: Create `backtest/reporting_html.py` with the dataclass and stubs**

```python
"""HTML report generation for backtesting analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go


_CSS = """
*, *::before, *::after { box-sizing: border-box; }
body {
    font-family: system-ui, -apple-system, sans-serif;
    margin: 0 auto;
    max-width: 1300px;
    padding: 32px 24px;
    color: #1a1a1a;
    background: #fff;
}
h1 { font-size: 1.6rem; border-bottom: 2px solid #222; padding-bottom: 10px; margin-bottom: 4px; }
h2 { font-size: 1.2rem; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 48px; }
.meta { color: #666; font-size: 0.85rem; margin-bottom: 32px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 0.875rem; }
th {
    background: #f0f0f0;
    text-align: left;
    padding: 8px 14px;
    border-bottom: 2px solid #bbb;
    white-space: nowrap;
}
td { padding: 6px 14px; font-family: 'Courier New', monospace; border-bottom: 1px solid #eee; }
tr:nth-child(even) td { background: #fafafa; }
.benchmark-row td {
    background: #f4f4ee !important;
    border-left: 3px solid #999;
    color: #555;
}
.positive { color: #1a7a1a; font-weight: 600; }
.negative { color: #b00000; font-weight: 600; }
.summary-block {
    background: #f8f8f8;
    border: 1px solid #ddd;
    border-left: 4px solid #666;
    border-radius: 3px;
    padding: 16px 20px;
    margin-bottom: 28px;
    font-family: 'Courier New', monospace;
    font-size: 0.875rem;
    line-height: 1.8;
}
"""


@dataclass
class ReportData:
    """All data needed to generate the HTML report."""

    ticker: str
    start_date: str
    end_date: str
    start_capital: float
    results: list[tuple]
    bh_metrics: dict
    bh_returns: pd.Series
    dca_metrics: dict
    dca_returns: pd.Series
    fig_comparison: go.Figure
    wfa_result: Any | None
    wfa_label: str
    fig_wfa_equity: go.Figure | None
    fig_wfa_params: go.Figure | None


def _build_comparison_table(
    results: list[tuple],
    bh_metrics: dict,
    bh_returns: pd.Series,
    dca_metrics: dict,
    dca_returns: pd.Series,
) -> str:
    """Build the strategy comparison HTML table fragment."""
    raise NotImplementedError


def _build_wfa_table(wfa_result: Any) -> str:
    """Build the WFA per-window HTML table fragment."""
    raise NotImplementedError


def generate_report(data: ReportData, output_path: str = "output/report.html") -> None:
    """Assemble all analysis results into a self-contained HTML report and write to disk."""
    raise NotImplementedError
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/test_reporting_html.py::test_report_data_instantiates -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add backtest/reporting_html.py tests/test_reporting_html.py
git commit -m "feat: scaffold reporting_html module with ReportData dataclass"
```

---

### Task 2: Implement `_build_comparison_table`

**Files:**
- Modify: `backtest/reporting_html.py`
- Modify: `tests/test_reporting_html.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_reporting_html.py`:

```python
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
    assert "Buy &amp; Hold" in html or "Buy & Hold" in html
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_reporting_html.py -k "comparison_table" -v
```

Expected: `FAILED` with `NotImplementedError`

- [ ] **Step 3: Implement `_build_comparison_table`**

Replace the stub in `backtest/reporting_html.py`:

```python
def _build_comparison_table(
    results: list[tuple],
    bh_metrics: dict,
    bh_returns: pd.Series,
    dca_metrics: dict,
    dca_returns: pd.Series,
) -> str:
    """Build the strategy comparison HTML table fragment."""
    rows = []
    for rank, (name, metrics, returns) in enumerate(results, start=1):
        ret = metrics["total_return"]
        ret_class = "positive" if ret >= 0 else "negative"
        start_dt = returns.index[0].strftime("%Y-%m-%d") if len(returns) else "N/A"
        end_dt = returns.index[-1].strftime("%Y-%m-%d") if len(returns) else "N/A"
        rows.append(
            f"<tr>"
            f"<td>{rank}</td>"
            f"<td>{name}</td>"
            f"<td>{start_dt}</td>"
            f"<td>{end_dt}</td>"
            f'<td class="{ret_class}">{ret:+.1f}%</td>'
            f"<td>{metrics['max_drawdown']:.1f}%</td>"
            f"<td>{metrics['win_rate']:.1f}%</td>"
            f"<td>{metrics['num_trades']}</td>"
            f"</tr>"
        )

    for label, metrics, returns in [
        ("Buy & Hold", bh_metrics, bh_returns),
        ("DCA (monthly)", dca_metrics, dca_returns),
    ]:
        ret = metrics["total_return"]
        ret_class = "positive" if ret >= 0 else "negative"
        start_dt = returns.index[0].strftime("%Y-%m-%d") if len(returns) else "N/A"
        end_dt = returns.index[-1].strftime("%Y-%m-%d") if len(returns) else "N/A"
        rows.append(
            f'<tr class="benchmark-row">'
            f"<td>—</td>"
            f"<td>{label}</td>"
            f"<td>{start_dt}</td>"
            f"<td>{end_dt}</td>"
            f'<td class="{ret_class}">{ret:+.1f}%</td>'
            f"<td>{metrics['max_drawdown']:.1f}%</td>"
            f"<td>n/a</td>"
            f"<td>n/a</td>"
            f"</tr>"
        )

    body = "\n".join(rows)
    return (
        "<table>"
        "<thead><tr>"
        "<th>Rank</th><th>Strategy</th><th>Start</th><th>End</th>"
        "<th>Return</th><th>Max Drawdown</th><th>Win%</th><th>Trades</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_reporting_html.py -k "comparison_table" -v
```

Expected: all 5 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add backtest/reporting_html.py tests/test_reporting_html.py
git commit -m "feat: implement _build_comparison_table with tests"
```

---

### Task 3: Implement `_build_wfa_table`

**Files:**
- Modify: `backtest/reporting_html.py`
- Modify: `tests/test_reporting_html.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_reporting_html.py`:

```python
from unittest.mock import MagicMock


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_reporting_html.py -k "wfa_table" -v
```

Expected: `FAILED` with `NotImplementedError`

- [ ] **Step 3: Implement `_build_wfa_table`**

Replace the stub in `backtest/reporting_html.py`:

```python
def _build_wfa_table(wfa_result: Any) -> str:
    """Build the WFA per-window HTML table fragment."""
    rows = []
    for _, row in wfa_result.windows.iterrows():
        param_str = " ".join(f"{k}={v}" for k, v in row["best_params"].items())
        test_range = f"{row['test_start'].date()} &rarr; {row['test_end'].date()}"
        sharpe = f"{row['sharpe_ratio']:.2f}" if isinstance(row["sharpe_ratio"], float) else "n/a"
        ret = row["total_return"]
        ret_str = f"{ret * 100:+.1f}%" if isinstance(ret, float) else "n/a"
        ret_class = "positive" if isinstance(ret, float) and ret >= 0 else "negative"
        rows.append(
            f"<tr>"
            f"<td>{test_range}</td>"
            f"<td><code>{param_str}</code></td>"
            f"<td>{sharpe}</td>"
            f'<td class="{ret_class}">{ret_str}</td>'
            f"<td>{int(row['n_trades'])}</td>"
            f"</tr>"
        )

    body = "\n".join(rows)
    return (
        "<table>"
        "<thead><tr>"
        "<th>Test Period</th><th>Best Params</th><th>Sharpe</th>"
        "<th>Return</th><th>Trades</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_reporting_html.py -k "wfa_table" -v
```

Expected: all 4 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add backtest/reporting_html.py tests/test_reporting_html.py
git commit -m "feat: implement _build_wfa_table with tests"
```

---

### Task 4: Implement `generate_report`

**Files:**
- Modify: `backtest/reporting_html.py`
- Modify: `tests/test_reporting_html.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_reporting_html.py`:

```python
import os
import tempfile


def make_report_data(wfa_result=None) -> ReportData:
    returns = make_returns()
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
        wfa_result=wfa_result,
        wfa_label="MomentumStrategy",
        fig_wfa_equity=go.Figure() if wfa_result else None,
        fig_wfa_params=go.Figure() if wfa_result else None,
    )


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_reporting_html.py -k "generate_report" -v
```

Expected: `FAILED` with `NotImplementedError`

- [ ] **Step 3: Implement `generate_report`**

Replace the stub in `backtest/reporting_html.py`:

```python
def generate_report(data: ReportData, output_path: str = "output/report.html") -> None:
    """Assemble all analysis results into a self-contained HTML report and write to disk."""
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    comparison_table = _build_comparison_table(
        data.results, data.bh_metrics, data.bh_returns, data.dca_metrics, data.dca_returns
    )
    chart_comparison = data.fig_comparison.to_html(
        full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}
    )

    if data.wfa_result is not None:
        wfa_table = _build_wfa_table(data.wfa_result)
        s = data.wfa_result.summary
        ret = s["total_return"] * 100
        ret_class = "positive" if ret >= 0 else "negative"
        wfa_section = (
            f"<h2>Walk-Forward Analysis: {data.wfa_label}</h2>"
            f"{wfa_table}"
            f'<div class="summary-block">'
            f'Sharpe ratio &nbsp; {s["sharpe_ratio"]:.2f}<br>'
            f'Total return &nbsp; <span class="{ret_class}">{ret:+.1f}%</span><br>'
            f'Max drawdown &nbsp; {s["max_drawdown"] * 100:.1f}%<br>'
            f'Win rate &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {s["win_rate"] * 100:.1f}%<br>'
            f'Windows &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {s["n_windows"]} ({s["n_windows_with_trades"]} with trades)<br>'
            f'Best params &nbsp;&nbsp; {data.wfa_result.best_params_overall}'
            f"</div>"
        )
        if data.fig_wfa_equity is not None:
            wfa_section += data.fig_wfa_equity.to_html(
                full_html=False, include_plotlyjs=False, config={"displayModeBar": False}
            )
        if data.fig_wfa_params is not None:
            wfa_section += data.fig_wfa_params.to_html(
                full_html=False, include_plotlyjs=False, config={"displayModeBar": False}
            )
    else:
        wfa_section = "<h2>Walk-Forward Analysis</h2><p>No WFA configured.</p>"

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        f"<title>{data.ticker} Trading Analysis</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{data.ticker} Trading Analysis</h1>\n"
        f'<p class="meta">{data.start_date} to {data.end_date}'
        f" &nbsp;|&nbsp; Starting capital: ${data.start_capital:,.0f}"
        f" &nbsp;|&nbsp; Run: {run_time}</p>\n"
        "<h2>Strategy Comparison</h2>\n"
        f"{comparison_table}\n"
        f"{chart_comparison}\n"
        f"{wfa_section}\n"
        "</body>\n"
        "</html>"
    )

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Report saved to {output_path}")
```

- [ ] **Step 4: Run all reporting_html tests**

```bash
uv run pytest tests/test_reporting_html.py -v
```

Expected: all tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add backtest/reporting_html.py tests/test_reporting_html.py
git commit -m "feat: implement generate_report and complete reporting_html module"
```

---

### Task 5: Update `bhp_analysis.py`

**Files:**
- Modify: `bhp_analysis.py`

- [ ] **Step 1: Replace the entire file**

The new version removes `print_comparison_table`, `print_wfa_summary`, and `plot_all_vs_benchmarks`. It adds `build_comparison_figure` (returns figure only, no save/show) and a simplified `main` that assembles `ReportData` and calls `generate_report`.

```python
"""BHP trading strategy analysis.

Runs all built-in strategies against NYSE:BHP, then runs Walk-Forward Analysis
on MomentumStrategy. All results are written to output/report.html.

Usage:
    uv run python bhp_analysis.py
"""

import warnings

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

warnings.filterwarnings("ignore")

from backtest.benchmarks import BuyAndHold, DollarCostAveraging
from backtest.optimization import RandomSearch, WalkForwardOptimizer
from backtest.reporting import plot_equity_curve, plot_parameter_stability
from backtest.reporting_html import ReportData, generate_report
from backtest.runner import BacktestRunnerImpl
from backtest.strategy import (
    BollingerBandsStrategy,
    BreakoutStrategy,
    ConsecutiveDaysStrategy,
    FibonacciRetracementStrategy,
    GapStrategy,
    MACDStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
    ParabolicSARStrategy,
    RSIStrategy,
    VolatilityStrategy,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TICKER = "BHP"
START_DATE = "2015-01-01"
END_DATE = "2026-01-01"
START_CAPITAL = 10_000.0

STRATEGIES = [
    ("ConsecutiveDays(3)", ConsecutiveDaysStrategy(consecutive_days=3)),
    ("MovingAvgCrossover", MovingAverageCrossoverStrategy(short_window=20, long_window=50)),
    ("RSI(14, 30/70)", RSIStrategy(period=14, lower_bound=30, upper_bound=70)),
    ("MACD(12/26/9)", MACDStrategy()),
    ("BollingerBands(20)", BollingerBandsStrategy()),
    ("ParabolicSAR", ParabolicSARStrategy()),
    ("Breakout(20d)", BreakoutStrategy(lookback_period=20)),
    ("Gap(2%)", GapStrategy(min_gap_pct=0.02)),
    ("Fibonacci(20d)", FibonacciRetracementStrategy(swing_lookback=20)),
    ("MeanReversion", MeanReversionStrategy()),
    ("Momentum(ROC-12)", MomentumStrategy()),
    ("Volatility(ATR-14)", VolatilityStrategy()),
]

WFA_STRATEGY_CLASS = MomentumStrategy
WFA_PARAM_SPACE = {
    "roc_period":    [6, 9, 12, 15, 20, 26],
    "roc_threshold": [0.02, 0.03, 0.05, 0.07, 0.10],
}
WFA_LABEL = "MomentumStrategy"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from yfinance, flattening multi-level columns if needed."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)
    return data


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------

def run_strategy_comparison(
    data: pd.DataFrame,
) -> tuple[list[tuple], dict, pd.Series, dict, pd.Series]:
    """Run all strategies and return sorted results plus benchmark metrics/returns."""
    bh_runner = BacktestRunnerImpl(
        strategy=ConsecutiveDaysStrategy(consecutive_days=1),
        benchmarks=[BuyAndHold(), DollarCostAveraging(frequency="monthly")],
    )
    bh_result = bh_runner.run(data, start_capital=START_CAPITAL)
    bh_metrics = bh_result["benchmark_metrics"]["BuyAndHold"]
    bh_returns = bh_result["benchmark_returns"]["BuyAndHold"]
    dca_metrics = bh_result["benchmark_metrics"]["DollarCostAveraging"]
    dca_returns = bh_result["benchmark_returns"]["DollarCostAveraging"]

    results = []
    for name, strategy in STRATEGIES:
        try:
            runner = BacktestRunnerImpl(strategy=strategy, benchmarks=[])
            res = runner.run(data, start_capital=START_CAPITAL)
            results.append((name, res["strategy_metrics"], res["strategy_returns"]))
        except Exception as exc:
            print(f"  [skip] {name}: {exc}")

    results.sort(key=lambda x: x[1]["total_return"], reverse=True)
    return results, bh_metrics, bh_returns, dca_metrics, dca_returns


def build_comparison_figure(
    results: list[tuple],
    bh_returns: pd.Series,
    dca_returns: pd.Series,
) -> go.Figure:
    """Build the all-strategies vs benchmarks equity curve figure."""
    fig = go.Figure()
    for name, _, returns in results:
        fig.add_trace(go.Scatter(
            x=returns.index, y=returns,
            name=name,
            line=dict(width=1),
            opacity=0.7,
        ))
    fig.add_trace(go.Scatter(
        x=bh_returns.index, y=bh_returns,
        name="Buy & Hold",
        line=dict(color="black", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=dca_returns.index, y=dca_returns,
        name="DCA (monthly)",
        line=dict(color="dimgray", width=2, dash="dot"),
    ))
    fig.update_layout(
        title=f"All Strategies vs. Benchmarks: {TICKER}",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=600,
    )
    return fig


# ---------------------------------------------------------------------------
# Walk-Forward Analysis
# ---------------------------------------------------------------------------

def run_wfa(data: pd.DataFrame):
    """Run WFA on MomentumStrategy."""
    optimizer = WalkForwardOptimizer(
        strategy_class=WFA_STRATEGY_CLASS,
        param_space=WFA_PARAM_SPACE,
        data=data,
        train_size=252,
        test_size=63,
        window_type="sliding",
        searcher=RandomSearch(n=40, seed=42),
        objective="sharpe_ratio",
        min_trades=2,
        n_jobs=-1,
    )
    return optimizer.run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Downloading {TICKER} ({START_DATE} to {END_DATE})...")
    data = download_data(TICKER, START_DATE, END_DATE)
    print(f"  {len(data)} trading days loaded.\n")

    print("Running strategy comparison...")
    results, bh_metrics, bh_returns, dca_metrics, dca_returns = run_strategy_comparison(data)
    fig_comparison = build_comparison_figure(results, bh_returns, dca_returns)

    wfa_result = None
    fig_wfa_equity = None
    fig_wfa_params = None
    if WFA_STRATEGY_CLASS is not None:
        print(f"\nRunning Walk-Forward Analysis on {WFA_LABEL}...")
        print(f"  Data: {len(data)} bars  |  Windows: 252-bar train / 63-bar test")
        wfa_result = run_wfa(data)
        fig_wfa_equity = plot_equity_curve(
            wfa_result,
            title=f"WFA Out-of-Sample Equity: {TICKER} ({WFA_LABEL})",
            start_capital=START_CAPITAL,
        )
        fig_wfa_params = plot_parameter_stability(wfa_result)

    generate_report(
        ReportData(
            ticker=TICKER,
            start_date=START_DATE,
            end_date=END_DATE,
            start_capital=START_CAPITAL,
            results=results,
            bh_metrics=bh_metrics,
            bh_returns=bh_returns,
            dca_metrics=dca_metrics,
            dca_returns=dca_returns,
            fig_comparison=fig_comparison,
            wfa_result=wfa_result,
            wfa_label=WFA_LABEL,
            fig_wfa_equity=fig_wfa_equity,
            fig_wfa_params=fig_wfa_params,
        ),
        output_path="output/report.html",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite to verify nothing is broken**

```bash
uv run pytest --cov=backtest -v
```

Expected: all existing tests plus new `test_reporting_html.py` tests pass. Coverage should remain above 75%.

- [ ] **Step 3: Run linter**

```bash
uv run ruff check bhp_analysis.py backtest/reporting_html.py
```

Expected: no errors. Fix any flagged issues before committing.

- [ ] **Step 4: Commit**

```bash
git add bhp_analysis.py
git commit -m "refactor: wire bhp_analysis to HTML report, remove terminal output"
```

---

## Verification

After all tasks are complete, do a manual smoke-check:

```bash
uv run python bhp_analysis.py
```

Expected terminal output:
```
Downloading BHP (2015-01-01 to 2026-01-01)...
  NNNN trading days loaded.

Running strategy comparison...
  [skip] ...  (if any)

Running Walk-Forward Analysis on MomentumStrategy...
  Data: NNNN bars  |  Windows: 252-bar train / 63-bar test
Report saved to output/report.html
```

Then open `output/report.html` in a browser and verify:
- Header shows ticker, date range, run timestamp
- Strategy comparison table is ranked by return with green/red return values
- Buy & Hold and DCA rows appear in a muted style below the strategies
- All-strategies equity curve chart is interactive (hover, zoom)
- WFA per-window table shows param combinations and returns
- WFA summary block shows overall stats
- WFA equity and parameter stability charts render correctly
