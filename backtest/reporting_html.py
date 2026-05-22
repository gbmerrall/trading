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
            f"<td>&#8212;</td>"
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
