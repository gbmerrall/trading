"""Stock trading strategy analysis.

Runs all built-in strategies against the given ticker, ranks them by Sharpe
ratio, then runs Walk-Forward Analysis on the Top 5. All results are written
to output/<ticker>_report.html.

Usage:
    uv run python trade_analysis.py <TICKER>
"""

import sys
import warnings

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from backtest.benchmarks import BuyAndHold, DollarCostAveraging
from backtest.llm_analysis import generate_executive_summary
from backtest.optimization import GridSearch, RandomSearch, WalkForwardOptimizer
from backtest.reporting import plot_equity_curve, plot_parameter_stability
from backtest.reporting_html import ReportData, WfaEntry, generate_report
from backtest.strategy_card import CardCandidate, build_card, write_card
from backtest.runner import compare_strategies
from backtest.strategy import (
    BollingerBandsStrategy,
    BreakoutStrategy,
    ConsecutiveDaysStrategy,
    EnsembleStrategy,
    FibonacciRetracementStrategy,
    GapStrategy,
    MACDStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
    ParabolicSARStrategy,
    RegimeFilteredStrategy,
    RSIStrategy,
    VolatilityStrategy,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: uv run python trade_analysis.py <TICKER>")
    sys.exit(1)
TICKER = sys.argv[1].upper()
START_DATE = "2020-01-01"
END_DATE = "2026-05-01"
START_CAPITAL = 10_000.0

# ---------------------------------------------------------------------------
# Tier 1: Standalone strategies
# ---------------------------------------------------------------------------
STRATEGIES = [
    ("ConsecutiveDays(3)", ConsecutiveDaysStrategy(consecutive_days=3)),
    (
        "MovingAvgCrossover",
        MovingAverageCrossoverStrategy(short_window=20, long_window=50),
    ),
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
    # ---------------------------------------------------------------------------
    # Tier 2: Regime-filtered wrappers
    # ---------------------------------------------------------------------------
    (
        "Breakout(20d)+ADX(25)",
        RegimeFilteredStrategy(
            base_strategy=BreakoutStrategy(lookback_period=20),
            regime_type="ADX",
            adx_threshold=25.0,
        ),
    ),
    (
        "Breakout(20d)+SMA(200)",
        RegimeFilteredStrategy(
            base_strategy=BreakoutStrategy(lookback_period=20),
            regime_type="SMA",
            sma_period=200,
        ),
    ),
    # ---------------------------------------------------------------------------
    # Tier 3: Ensembles
    # ---------------------------------------------------------------------------
    (
        "Ensemble_Pair(Strict)",
        EnsembleStrategy(
            strategies=[BreakoutStrategy(lookback_period=20), MeanReversionStrategy()],
            min_agreement=2,
        ),
    ),
    (
        "Ensemble_Triple(Majority)",
        EnsembleStrategy(
            strategies=[
                BreakoutStrategy(lookback_period=20),
                MeanReversionStrategy(),
                MomentumStrategy(),
            ],
            min_agreement=2,
        ),
    ),
    (
        "Ensemble_Quad(Loose)",
        EnsembleStrategy(
            strategies=[
                BreakoutStrategy(lookback_period=20),
                MeanReversionStrategy(),
                MomentumStrategy(),
                GapStrategy(min_gap_pct=0.02),
            ],
            min_agreement=2,
        ),
    ),
]

# Maps each strategy name to its WFA configuration.
#
# Standard strategies:  name → plain param dict
# Regime wrappers:      name → {"param_grid": {...}, "base_params": {...}}
#   base_params are fixed constructor kwargs (passed straight to WalkForwardOptimizer);
#   param_grid keys are the ones WFA will search over.
#
# Ensemble and other non-parameterisable strategies are intentionally absent —
# they are skipped by the WFA loop when their name is not found here.
WFA_PARAM_GRIDS: dict[str, dict] = {
    # Tier 1 — standalones
    "ConsecutiveDays(3)": {"consecutive_days": [2, 3, 4, 5]},
    "MovingAvgCrossover": {
        "short_window": [10, 15, 20],
        "long_window": [40, 50, 60, 100],
    },
    "RSI(14, 30/70)": {
        "period": [7, 10, 14, 20],
        "lower_bound": [25, 30],
        "upper_bound": [70, 75],
    },
    "MACD(12/26/9)": {"fast": [8, 10, 12], "slow": [21, 26], "signal": [7, 9]},
    "BollingerBands(20)": {"period": [10, 15, 20, 25], "std_dev": [1.5, 2.0, 2.5]},
    "ParabolicSAR": {"af": [0.01, 0.02, 0.03], "max_af": [0.1, 0.2, 0.3]},
    "Breakout(20d)": {"lookback_period": [10, 20, 30, 40, 50]},
    "Gap(2%)": {"min_gap_pct": [0.01, 0.02, 0.03, 0.05]},
    "Fibonacci(20d)": {"swing_lookback": [10, 15, 20, 30, 40]},
    "MeanReversion": {
        "rsi_period": [10, 14, 20],
        "bb_period": [15, 20, 25],
        "bb_std": [1.5, 2.0, 2.5],
    },
    "Momentum(ROC-12)": {
        "roc_period": [6, 9, 12, 15, 20],
        "roc_threshold": [0.02, 0.03, 0.05],
    },
    "Volatility(ATR-14)": {
        "atr_period": [10, 14, 20],
        "atr_multiplier": [1.5, 2.0, 2.5],
        "breakout_period": [10, 15, 20],
    },
    # Tier 2 — regime wrappers
    "Breakout(20d)+ADX(25)": {
        "param_grid": {
            "lookback_period": [10, 20, 30],
            "adx_threshold": [20.0, 25.0, 30.0],
        },
        "base_params": {"base_strategy": BreakoutStrategy(), "regime_type": "ADX"},
    },
    "Breakout(20d)+SMA(200)": {
        "param_grid": {"lookback_period": [10, 20, 30], "sma_period": [100, 150, 200]},
        "base_params": {"base_strategy": BreakoutStrategy(), "regime_type": "SMA"},
    },
}


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
# Comparison figure
# ---------------------------------------------------------------------------


def build_comparison_figure(
    results: list[tuple],
    bh_returns: pd.Series,
    dca_returns: pd.Series,
) -> go.Figure:
    """Build the Top 3 strategies vs benchmarks equity curve figure."""
    fig = go.Figure()
    for name, _, returns in results:
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns,
                name=name,
                line=dict(width=1),
                opacity=0.7,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=bh_returns.index,
            y=bh_returns,
            name="Buy & Hold",
            line=dict(color="black", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dca_returns.index,
            y=dca_returns,
            name="DCA (monthly)",
            line=dict(color="dimgray", width=2, dash="dot"),
        )
    )
    fig.update_layout(
        title=f"Top 3 Strategies vs. Benchmarks: {TICKER}",
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


def _pick_searcher(param_grid: dict) -> "GridSearch | RandomSearch":
    """Return GridSearch for grids with <= 50 combinations, otherwise RandomSearch."""
    n_combos = 1
    for values in param_grid.values():
        n_combos *= len(values)
    return GridSearch() if n_combos <= 50 else RandomSearch(n=40, seed=42)


def run_wfa(
    data: pd.DataFrame,
    strategy_class: type,
    param_grid: dict,
    base_params: dict | None = None,
):
    """Run Walk-Forward Analysis for the given strategy class and parameter grid.

    Args:
        data: Full price history.
        strategy_class: BaseStrategy subclass to optimise.
        param_grid: Parameter search space passed to the searcher.
        base_params: Fixed constructor kwargs merged with each candidate before
                     instantiation (used for RegimeFilteredStrategy wrappers).
    """
    optimizer = WalkForwardOptimizer(
        strategy_class=strategy_class,
        param_space=param_grid,
        data=data,
        train_size=252,
        test_size=63,
        window_type="sliding",
        searcher=_pick_searcher(param_grid),
        objective="sharpe_ratio",
        min_trades=2,
        # Serial is faster here: per-backtest cost (~5ms) is smaller than the
        # ProcessPoolExecutor IPC + fork/join overhead per training window.
        n_jobs=1,
        base_params=base_params,
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

    print("Running strategy comparison (all strategies, ranked by Sharpe ratio)...")
    results, benchmark_metrics, benchmark_returns = compare_strategies(
        data,
        STRATEGIES,
        benchmarks=[BuyAndHold(), DollarCostAveraging(frequency="monthly")],
        start_capital=START_CAPITAL,
        sort_metric="sharpe_ratio",
        top_n=len(STRATEGIES),
    )
    wfa_candidates = results[:5]
    bh_metrics = benchmark_metrics.get("BuyAndHold", {})
    bh_returns = benchmark_returns.get("BuyAndHold", pd.Series(dtype=float))
    dca_metrics = benchmark_metrics.get("DollarCostAveraging", {})
    dca_returns = benchmark_returns.get("DollarCostAveraging", pd.Series(dtype=float))
    fig_comparison = build_comparison_figure(results, bh_returns, dca_returns)

    print("\nRunning Walk-Forward Analysis on Top 5 shortlist...")
    print(f"  Data: {len(data)} bars  |  Windows: 252-bar train / 63-bar test")
    wfa_entries = []
    for name, _, _ in wfa_candidates:
        instance = next((inst for n, inst in STRATEGIES if n == name), None)
        if instance is None:
            continue
        cls = type(instance)
        config = WFA_PARAM_GRIDS.get(name)
        if config is None:
            print(f"  [skip] {name}: no parameter grid defined.")
            continue
        if "param_grid" in config and "base_params" in config:
            param_grid = config["param_grid"]
            base_params = config["base_params"]
        else:
            param_grid = config
            base_params = None
        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v)
        print(f"  {name} ({n_combos} combinations)...")
        wfa_result = run_wfa(data, cls, param_grid, base_params)
        wfa_entries.append(
            WfaEntry(
                label=name,
                result=wfa_result,
                fig_equity=plot_equity_curve(
                    wfa_result,
                    title=f"WFA Out-of-Sample Equity: {TICKER} ({name})",
                    start_capital=START_CAPITAL,
                ),
                fig_params=plot_parameter_stability(wfa_result),
            )
        )

    card_candidates = []
    for entry in wfa_entries:
        instance = next((inst for n, inst in STRATEGIES if n == entry.label), None)
        if instance is None:
            continue
        card_candidates.append(
            CardCandidate(
                label=entry.label,
                strategy_class=type(instance).__name__,
                params=entry.result.best_params_overall,
                summary=entry.result.summary,
            )
        )
    card = build_card(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        start_capital=START_CAPITAL,
        candidates=card_candidates,
    )
    card_path = f"output/{TICKER.lower()}_card.json"
    write_card(card, card_path)
    print(f"Strategy card written to {card_path}")

    print("\nGenerating executive summary...")
    summary = generate_executive_summary(TICKER, wfa_entries, bh_metrics)
    print("=" * 42)
    print("========== EXECUTIVE SUMMARY ==========")
    print("=" * 42)
    print(summary)
    print("=" * 42)

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
            wfa_entries=wfa_entries,
            executive_summary=summary,
        ),
        output_path=f"output/{TICKER.lower()}_report.html",
    )


if __name__ == "__main__":
    main()
