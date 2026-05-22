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
