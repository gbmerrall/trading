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
    warnings.filterwarnings("ignore")
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
