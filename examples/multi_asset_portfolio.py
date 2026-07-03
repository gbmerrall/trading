"""Multi-asset portfolio blending example.

Runs two independent pre-optimised strategies on separate assets and blends
their equity curves into a capital-weighted master portfolio.

Portfolio:
  - SPY  70%  BreakoutStrategy(lookback_period=20)
  - COIN 30%  MACDStrategy(fast=12, slow=26, signal=9)

Note: this example predates the cost model and uses fixed in-sample parameters
with the default (zero-cost) config. Treat the numbers as illustrative of the
blending mechanics, not as a validated strategy. Multi-asset support inside the
runner itself is out of scope (see README).

Usage:
    uv run python examples/multi_asset_portfolio.py
"""

import math
import sys
import warnings
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from backtest.constants import TradingConstants
from backtest.runner import BacktestRunnerImpl
from backtest.strategy import BreakoutStrategy, MACDStrategy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

START_DATE = "2020-01-01"
END_DATE = "2026-05-01"
START_CAPITAL = 10_000.0
SPY_WEIGHT = 0.70
COIN_WEIGHT = 0.30
TRADING_DAYS = TradingConstants.TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from yfinance, flattening multi-level columns if needed.

    Args:
        ticker: Ticker symbol.
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex.
    """
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)
    return data


def align_to_spy_calendar(spy_data: pd.DataFrame, coin_data: pd.DataFrame) -> pd.DataFrame:
    """Reindex coin_data to the SPY trading calendar, forward-filling weekend/holiday gaps.

    Args:
        spy_data: SPY OHLCV DataFrame defining the master calendar.
        coin_data: Asset OHLCV DataFrame to align (may have daily/weekend entries).

    Returns:
        coin_data reindexed to spy_data.index with NaN gaps filled forward.
    """
    return coin_data.reindex(spy_data.index, method="ffill")


# ---------------------------------------------------------------------------
# Return & equity curve helpers
# ---------------------------------------------------------------------------

def blend_returns(
    spy_daily: pd.Series,
    coin_daily: pd.Series,
    spy_weight: float,
    coin_weight: float,
) -> pd.Series:
    """Combine two daily-return series by capital weight.

    Args:
        spy_daily: SPY daily percentage returns.
        coin_daily: COIN daily percentage returns.
        spy_weight: Fraction allocated to SPY (e.g. 0.70).
        coin_weight: Fraction allocated to COIN (e.g. 0.30).

    Returns:
        Blended daily return Series on the same index as spy_daily.
    """
    return spy_daily * spy_weight + coin_daily * coin_weight


def build_equity_curve(daily_returns: pd.Series, start_capital: float) -> pd.Series:
    """Convert a daily-return series to a cumulative equity curve.

    Args:
        daily_returns: Daily percentage returns (0.01 = 1%).
        start_capital: Starting portfolio value in dollars.

    Returns:
        Cumulative equity Series starting at start_capital.
    """
    return start_capital * (1 + daily_returns).cumprod()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_portfolio_metrics(equity_curve: pd.Series) -> dict:
    """Compute Sharpe ratio and max drawdown from an equity curve.

    Args:
        equity_curve: Portfolio value Series with DatetimeIndex.

    Returns:
        Dict with 'sharpe_ratio' (float) and 'max_drawdown' (float, <= 0).
    """
    daily_returns = equity_curve.pct_change().dropna()

    # Sharpe ratio
    if len(daily_returns) < 2:
        sharpe = float("-inf")
    else:
        mean_r = float(daily_returns.mean())
        std_r = float(daily_returns.std())
        if std_r < 1e-12:
            sharpe = 1e6 if mean_r > 0 else float("-inf")
        else:
            sharpe = mean_r / std_r * math.sqrt(TRADING_DAYS)

    # Max drawdown
    peak = equity_curve.expanding().max()
    safe_peak = peak.where(peak > 0, other=float("nan"))
    drawdowns = (equity_curve - safe_peak) / safe_peak
    max_dd = float(drawdowns.min()) if not drawdowns.isna().all() else 0.0

    return {"sharpe_ratio": sharpe, "max_drawdown": max_dd}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def build_multi_asset_figure(
    spy_curve: pd.Series,
    coin_curve: pd.Series,
    master_curve: pd.Series,
    spy_label: str,
    coin_label: str,
) -> go.Figure:
    """Build a three-trace equity curve comparison figure.

    Args:
        spy_curve: SPY standalone equity curve normalised to start_capital.
        coin_curve: COIN standalone equity curve normalised to start_capital.
        master_curve: Blended master portfolio equity curve.
        spy_label: Legend label for the SPY trace.
        coin_label: Legend label for the COIN trace.

    Returns:
        Plotly Figure with three scatter traces.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spy_curve.index, y=spy_curve,
        name=spy_label,
        line=dict(color="#1f77b4", width=1.5),
        opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=coin_curve.index, y=coin_curve,
        name=coin_label,
        line=dict(color="#ff7f0e", width=1.5),
        opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=master_curve.index, y=master_curve,
        name="Master Portfolio (70/30)",
        line=dict(color="#2ca02c", width=2.5),
    ))
    fig.update_layout(
        title="Multi-Asset Ensemble: SPY Breakout + COIN MACD",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=600,
    )
    return fig


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

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
.summary-block {
    background: #f8f8f8;
    border: 1px solid #ddd;
    border-left: 4px solid #2ca02c;
    border-radius: 3px;
    padding: 16px 20px;
    margin-bottom: 28px;
    font-family: 'Courier New', monospace;
    font-size: 0.875rem;
    line-height: 1.8;
}
.positive { color: #1a7a1a; font-weight: 600; }
.negative { color: #b00000; font-weight: 600; }
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
"""


def generate_multi_asset_report(
    fig: go.Figure,
    metrics: dict,
    correlation: float,
    spy_metrics: dict,
    coin_metrics: dict,
    output_path: str = "output/multi_asset_ensemble.html",
) -> None:
    """Write a self-contained HTML report for the multi-asset analysis.

    Args:
        fig: Plotly figure containing the three equity curves.
        metrics: Master portfolio metrics dict (sharpe_ratio, max_drawdown).
        correlation: Pearson correlation between SPY and COIN daily returns.
        spy_metrics: SPY standalone portfolio metrics dict.
        coin_metrics: COIN standalone portfolio metrics dict.
        output_path: Path to write the HTML file.
    """
    import os

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    def _fmt_pct(v: float) -> str:
        cls = "positive" if v >= 0 else "negative"
        return f'<span class="{cls}">{v * 100:+.2f}%</span>'

    def _fmt_sharpe(v: float) -> str:
        cls = "positive" if v >= 0 else "negative"
        return f'<span class="{cls}">{v:.3f}</span>'

    rows = [
        ("<b>SPY Breakout (70%)</b>",
         _fmt_sharpe(spy_metrics["sharpe_ratio"]),
         _fmt_pct(spy_metrics["max_drawdown"])),
        ("<b>COIN MACD (30%)</b>",
         _fmt_sharpe(coin_metrics["sharpe_ratio"]),
         _fmt_pct(coin_metrics["max_drawdown"])),
        ("<b>Master Portfolio</b>",
         _fmt_sharpe(metrics["sharpe_ratio"]),
         _fmt_pct(metrics["max_drawdown"])),
    ]
    table_rows = "".join(
        f"<tr><td>{name}</td><td>{sharpe}</td><td>{dd}</td></tr>"
        for name, sharpe, dd in rows
    )
    table = (
        "<table><thead><tr>"
        "<th>Strategy</th><th>Sharpe Ratio</th><th>Max Drawdown</th>"
        "</tr></thead>"
        f"<tbody>{table_rows}</tbody></table>"
    )

    corr_class = "positive" if correlation >= 0 else "negative"
    summary = (
        '<div class="summary-block">'
        f'Pearson correlation (SPY vs COIN) &nbsp; '
        f'<span class="{corr_class}">{correlation:+.4f}</span><br>'
        f'Master Portfolio Sharpe &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; '
        f'{_fmt_sharpe(metrics["sharpe_ratio"])}<br>'
        f'Master Portfolio Max Drawdown &nbsp; {_fmt_pct(metrics["max_drawdown"])}'
        "</div>"
    )

    chart_html = fig.to_html(
        full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}
    )

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        "<title>Multi-Asset Ensemble Analysis</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>Multi-Asset Ensemble Analysis</h1>\n"
        f'<p class="meta">{START_DATE} to {END_DATE}'
        f" &nbsp;|&nbsp; Starting capital per leg: ${START_CAPITAL:,.0f}"
        f" &nbsp;|&nbsp; Weights: SPY {SPY_WEIGHT:.0%} / COIN {COIN_WEIGHT:.0%}"
        f" &nbsp;|&nbsp; Run: {run_time}</p>\n"
        "<h2>Performance Summary</h2>\n"
        f"{summary}\n"
        f"{table}\n"
        "<h2>Equity Curves</h2>\n"
        f"{chart_html}\n"
        "</body>\n"
        "</html>"
    )

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Report saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the multi-asset ensemble analysis and write the HTML report."""
    warnings.filterwarnings("ignore")

    print(f"Downloading SPY ({START_DATE} to {END_DATE})...")
    spy_raw = download_data("SPY", START_DATE, END_DATE)
    print(f"  {len(spy_raw)} trading days loaded.")

    print("Downloading COIN...")
    coin_raw = download_data("COIN", START_DATE, END_DATE)
    print(f"  {len(coin_raw)} calendar days loaded (before alignment).")

    coin_aligned = align_to_spy_calendar(spy_raw, coin_raw)
    print(f"  COIN aligned to {len(coin_aligned)} SPY trading days.\n")

    print("Running SPY backtest  (BreakoutStrategy lookback=20)...")
    spy_runner = BacktestRunnerImpl(BreakoutStrategy(lookback_period=20), benchmarks=[])
    spy_result = spy_runner.run(spy_raw, start_capital=START_CAPITAL)
    spy_returns_raw: pd.Series = spy_result["strategy_returns"]

    print("Running COIN backtest (MACDStrategy fast=12 slow=26 signal=9)...")
    coin_runner = BacktestRunnerImpl(MACDStrategy(fast=12, slow=26, signal=9), benchmarks=[])
    coin_result = coin_runner.run(coin_aligned, start_capital=START_CAPITAL)
    coin_returns_raw: pd.Series = coin_result["strategy_returns"]

    # Align both return series to a shared index (intersection of post-warmup dates)
    shared_index = spy_returns_raw.index.intersection(coin_returns_raw.index)
    spy_returns = spy_returns_raw.reindex(shared_index)
    coin_returns = coin_returns_raw.reindex(shared_index)

    spy_daily = spy_returns.pct_change().fillna(0)
    coin_daily = coin_returns.pct_change().fillna(0)

    master_daily = blend_returns(spy_daily, coin_daily, SPY_WEIGHT, COIN_WEIGHT)

    spy_curve = build_equity_curve(spy_daily, START_CAPITAL)
    coin_curve = build_equity_curve(coin_daily, START_CAPITAL)
    master_curve = build_equity_curve(master_daily, START_CAPITAL)

    correlation = float(spy_daily.corr(coin_daily))

    spy_metrics = compute_portfolio_metrics(spy_curve)
    coin_metrics = compute_portfolio_metrics(coin_curve)
    master_metrics = compute_portfolio_metrics(master_curve)

    print("\n" + "=" * 50)
    print("MULTI-ASSET ENSEMBLE RESULTS")
    print("=" * 50)
    print(f"  Pearson correlation (SPY vs COIN): {correlation:+.4f}")
    print(f"  Master Portfolio Sharpe ratio:     {master_metrics['sharpe_ratio']:.4f}")
    print(f"  Master Portfolio Max Drawdown:     {master_metrics['max_drawdown'] * 100:.2f}%")
    print(f"  SPY standalone Sharpe:             {spy_metrics['sharpe_ratio']:.4f}")
    print(f"  COIN standalone Sharpe:            {coin_metrics['sharpe_ratio']:.4f}")
    print("=" * 50 + "\n")

    fig = build_multi_asset_figure(
        spy_curve, coin_curve, master_curve,
        spy_label="SPY Breakout(20d)",
        coin_label="COIN MACD(12/26/9)",
    )

    generate_multi_asset_report(
        fig=fig,
        metrics=master_metrics,
        correlation=correlation,
        spy_metrics=spy_metrics,
        coin_metrics=coin_metrics,
        output_path="output/multi_asset_ensemble.html",
    )


if __name__ == "__main__":
    main()
