"""Walk-Forward Analysis visualization and reporting.

All public functions accept a WalkForwardResult and return a plotly Figure
so the caller controls rendering (show, write_image, embed in a notebook, etc.).

Public API:
    plot_equity_curve        — stitched out-of-sample equity, chained across windows
    plot_parameter_stability — which parameter values were selected per window
    plot_metrics_by_window   — per-window metric bar charts
    save_wfa_report          — write all three plots to a directory as PNG files
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .optimization import WalkForwardResult


def _chain_equity_curve(
    equity_curve: pd.Series,
    windows: pd.DataFrame,
    start_capital: float,
) -> pd.Series:
    """Re-scale each window segment so the curve is continuous from start_capital.

    The raw equity_curve concatenates independent backtests that each start at
    the config's default capital, producing discontinuities at window boundaries.
    This function scales each segment so it begins where the previous one ended.

    Args:
        equity_curve: Raw stitched equity curve from WalkForwardResult.
        windows: Per-window DataFrame from WalkForwardResult.
        start_capital: Value the chained curve should start at.

    Returns:
        Continuous pd.Series with no value jumps at window boundaries.
    """
    parts = []
    running = start_capital

    for _, row in windows.iterrows():
        mask = (equity_curve.index >= row["test_start"]) & (
            equity_curve.index <= row["test_end"]
        )
        segment = equity_curve[mask]
        if segment.empty:
            continue
        scale = running / segment.iloc[0]
        scaled = segment * scale
        parts.append(scaled)
        running = scaled.iloc[-1]

    if not parts:
        return equity_curve

    chained = pd.concat(parts)
    return chained[~chained.index.duplicated(keep="first")]


def plot_equity_curve(
    result: WalkForwardResult,
    title: str = "WFA Out-of-Sample Equity Curve",
    start_capital: float = 10_000.0,
    chain: bool = True,
    mark_windows: bool = True,
) -> go.Figure:
    """Plot the stitched out-of-sample equity curve.

    Args:
        result: Output from WalkForwardOptimizer.run().
        title: Figure title.
        start_capital: Starting value for the chained curve (ignored if chain=False).
        chain: If True, re-scale each window segment so the curve is continuous.
               If False, plot the raw concatenated series (may have value jumps).
        mark_windows: If True, draw vertical lines at each test-window boundary.

    Returns:
        plotly Figure.
    """
    equity = (
        _chain_equity_curve(result.equity_curve, result.windows, start_capital)
        if chain
        else result.equity_curve
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        name="Out-of-sample equity",
        line=dict(color="royalblue", width=2),
        hovertemplate="%{x|%Y-%m-%d}: $%{y:,.0f}<extra></extra>",
    ))

    if mark_windows and not result.windows.empty:
        # Add faint vertical lines at window test_start boundaries
        for _, row in result.windows.iterrows():
            fig.add_vline(
                x=row["test_start"],
                line=dict(color="rgba(128,128,128,0.3)", width=1, dash="dot"),
            )

    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100 if len(equity) > 1 else 0.0
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{len(result.windows)} windows · "
                        f"{total_ret:+.1f}% total return</sup>"),
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        width=1200,
        height=500,
        margin=dict(t=80),
    )
    return fig


def plot_parameter_stability(
    result: WalkForwardResult,
    params: Sequence[str] | None = None,
    title: str = "Parameter Stability Across Windows",
) -> go.Figure:
    """Plot which parameter values were selected in each test window.

    Numeric parameters are shown as line traces. Categorical parameters are
    shown as scatter points. Each parameter gets its own subplot row.

    Args:
        result: Output from WalkForwardOptimizer.run().
        params: Parameter names to include. Defaults to all keys in best_params_overall.
        title: Figure title.

    Returns:
        plotly Figure.
    """
    windows = result.windows
    if windows.empty or "best_params" not in windows.columns:
        return go.Figure().update_layout(title=title)

    all_params = list(result.best_params_overall.keys()) if params is None else list(params)
    if not all_params:
        return go.Figure().update_layout(title=title)

    n_params = len(all_params)
    fig = make_subplots(
        rows=n_params,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"param: {p}" for p in all_params],
        vertical_spacing=0.08,
    )

    x_labels = [str(row["test_start"].date()) for _, row in windows.iterrows()]

    colors = [
        "royalblue", "firebrick", "seagreen", "darkorange",
        "mediumpurple", "deeppink", "goldenrod",
    ]

    for i, param in enumerate(all_params):
        row_num = i + 1
        values = [row["best_params"].get(param) for _, row in windows.iterrows()]

        # Mode line — show the most-selected value as a horizontal reference
        mode_value = result.best_params_overall.get(param)

        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=values,
                name=param,
                mode="lines+markers",
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=8),
                showlegend=True,
            ),
            row=row_num,
            col=1,
        )
        if mode_value is not None:
            fig.add_hline(
                y=mode_value,
                line=dict(color=colors[i % len(colors)], width=1, dash="dash"),
                annotation_text=f"mode={mode_value}",
                annotation_position="top right",
                row=row_num,
                col=1,
            )

    n_with_mode = int(result.summary.get("param_stability", 0) * len(windows))
    fig.update_layout(
        title=dict(
            text=(
                f"{title}<br><sup>Most stable params: {result.best_params_overall} "
                f"({n_with_mode}/{len(windows)} windows)</sup>"
            )
        ),
        height=250 * n_params + 80,
        width=1200,
        hovermode="x unified",
        margin=dict(t=100),
    )
    fig.update_xaxes(tickangle=45, row=n_params, col=1)
    return fig


def plot_metrics_by_window(
    result: WalkForwardResult,
    metrics: Sequence[str] = ("sharpe_ratio", "total_return", "max_drawdown"),
    title: str = "Per-Window Metrics",
) -> go.Figure:
    """Bar charts of selected metrics across all test windows.

    Args:
        result: Output from WalkForwardOptimizer.run().
        metrics: Metric column names to plot (must be in result.windows).
        title: Figure title.

    Returns:
        plotly Figure.
    """
    windows = result.windows
    if windows.empty:
        return go.Figure().update_layout(title=title)

    metrics = [m for m in metrics if m in windows.columns]
    if not metrics:
        return go.Figure().update_layout(title=title)

    n = len(metrics)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=list(metrics),
        vertical_spacing=0.08,
    )

    x_labels = [str(row["test_start"].date()) for _, row in windows.iterrows()]
    colors = [
        "royalblue", "seagreen", "firebrick", "darkorange",
        "mediumpurple", "deeppink",
    ]

    for i, metric in enumerate(metrics):
        row_num = i + 1
        values = windows[metric].tolist()

        # Replace -inf with NaN for display
        display_values = [
            v if (isinstance(v, float) and not math.isinf(v)) else None
            for v in values
        ]

        bar_colors = []
        for v in display_values:
            if v is None:
                bar_colors.append("lightgrey")
            elif metric in ("max_drawdown", "ulcer_index"):
                bar_colors.append("firebrick" if v < 0 else "seagreen")
            else:
                bar_colors.append("seagreen" if v >= 0 else "firebrick")

        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=display_values,
                name=metric,
                marker_color=bar_colors,
                showlegend=False,
            ),
            row=row_num,
            col=1,
        )

        # Summary value line
        summary_val = result.summary.get(metric)
        if summary_val is not None and isinstance(summary_val, float) and not math.isinf(summary_val):
            fig.add_hline(
                y=summary_val,
                line=dict(color=colors[i % len(colors)], width=1.5, dash="dash"),
                annotation_text=f"overall={summary_val:.2f}",
                annotation_position="top right",
                row=row_num,
                col=1,
            )

    fig.update_layout(
        title=title,
        height=250 * n + 80,
        width=1200,
        hovermode="x unified",
        bargap=0.3,
        margin=dict(t=80),
    )
    fig.update_xaxes(tickangle=45, row=n, col=1)
    return fig


def save_wfa_report(
    result: WalkForwardResult,
    output_dir: str | Path = "output",
    prefix: str = "wfa",
    start_capital: float = 10_000.0,
) -> list[Path]:
    """Save all three WFA plots to PNG files in output_dir.

    Args:
        result: Output from WalkForwardOptimizer.run().
        output_dir: Directory to write files into (created if absent).
        prefix: Filename prefix. Files are named {prefix}_equity.png, etc.
        start_capital: Passed through to plot_equity_curve for chain scaling.

    Returns:
        List of Path objects for the files written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = {
        "equity": plot_equity_curve(result, start_capital=start_capital),
        "params": plot_parameter_stability(result),
        "metrics": plot_metrics_by_window(result),
    }

    written = []
    for name, fig in plots.items():
        path = output_dir / f"{prefix}_{name}.png"
        fig.write_image(str(path), engine="kaleido")
        written.append(path)

    return written
