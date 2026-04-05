"""Ensemble strategy: combine multiple strategies via majority voting.

Builds an ensemble of RSI, MACD, and Bollinger Bands and compares it against
each constituent strategy individually and against Buy & Hold. The ensemble
only signals when a majority of sub-strategies agree.

Usage:
    python examples/ensemble_strategy.py
    python examples/ensemble_strategy.py MSFT 2019-01-01 2024-01-01
"""

import sys
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.benchmarks import BuyAndHold
from backtest.metrics import METRICS
from backtest.runner import BacktestRunnerImpl
from backtest.strategy import (
    BollingerBandsStrategy,
    EnsembleStrategy,
    MACDStrategy,
    RSIStrategy,
)


def download_data(ticker: str, start: str, end: str):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, type(data.columns)) and hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)
    return data


def run_backtest(strategy, data, start_capital: float) -> dict:
    runner = BacktestRunnerImpl(strategy=strategy, benchmarks=[])
    result = runner.run(data, start_capital=start_capital)

    portfolio_history = [
        {"date": d, "value": v}
        for d, v in result["strategy_returns"].items()
    ]
    trades = result["trades"]

    base = result["strategy_metrics"]
    return {
        "total_return": base["total_return"],
        "max_drawdown": base["max_drawdown"],
        "win_rate": base["win_rate"],
        "num_trades": base["num_trades"],
        "sharpe_ratio": METRICS["sharpe_ratio"](portfolio_history, trades),
        "sortino_ratio": METRICS["sortino_ratio"](portfolio_history, trades),
        "final_value": result["strategy_returns"].iloc[-1],
    }


def fmt_ratio(v: float) -> str:
    if v == float("-inf") or v == float("inf"):
        return "   n/a"
    return f"{v:>6.2f}"


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    start = sys.argv[2] if len(sys.argv) > 2 else "2019-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2024-01-01"
    start_capital = 10_000.0

    print(f"Downloading {ticker} ({start} to {end})...")
    data = download_data(ticker, start, end)
    print(f"  {len(data)} trading days\n")

    # Build constituent strategies
    rsi = RSIStrategy(period=14, lower_bound=30, upper_bound=70)
    macd = MACDStrategy(fast=12, slow=26, signal=9)
    bb = BollingerBandsStrategy(period=20, std_dev=2.0)

    # Build the ensemble (signals when at least 2 of 3 sub-strategies agree)
    ensemble = EnsembleStrategy(
        strategies=[
            RSIStrategy(period=14, lower_bound=30, upper_bound=70),
            MACDStrategy(fast=12, slow=26, signal=9),
            BollingerBandsStrategy(period=20, std_dev=2.0),
        ],
        min_agreement=2,
    )

    # Also grab Buy & Hold for reference
    bh_runner = BacktestRunnerImpl(strategy=rsi, benchmarks=[BuyAndHold()])
    bh_result = bh_runner.run(data, start_capital=start_capital)
    bh_m = bh_result["benchmark_metrics"]["BuyAndHold"]

    strategies = {
        "RSI(14)": rsi,
        "MACD(12/26/9)": macd,
        "BollingerBands(20, 2σ)": bb,
        "Ensemble (majority vote)": ensemble,
    }

    results = {}
    for name, strategy in strategies.items():
        try:
            results[name] = run_backtest(strategy, data, start_capital)
        except Exception as exc:
            print(f"  [skip] {name}: {exc}")

    # Print comparison table
    col = 26
    print(f"{'Strategy':<{col}} {'Return':>8} {'Drawdown':>10} {'Win%':>7} {'Sharpe':>8} {'Sortino':>8} {'Trades':>7}")
    print("-" * (col + 52))

    for name, m in results.items():
        marker = "  <--" if name == "Ensemble (majority vote)" else ""
        print(
            f"{name:<{col}} "
            f"{m['total_return']:>+7.1f}% "
            f"{m['max_drawdown']:>9.1f}% "
            f"{m['win_rate']:>6.1f}% "
            f"{fmt_ratio(m['sharpe_ratio'])} "
            f"{fmt_ratio(m['sortino_ratio'])} "
            f"{m['num_trades']:>7}"
            f"{marker}"
        )

    print("-" * (col + 52))
    print(
        f"{'Buy & Hold (benchmark)':<{col}} "
        f"{bh_m['total_return']:>+7.1f}% "
        f"{bh_m['max_drawdown']:>9.1f}% "
        f"{'n/a':>6}  "
        f"{'n/a':>8} "
        f"{'n/a':>8} "
        f"{'n/a':>7}"
    )

    print()
    print("Ensemble warmup period:", ensemble.warmup_period, "bars")
    print(
        "The ensemble signals when at least 2 of 3 sub-strategies agree (min_agreement=2). "
        "This reduces trade frequency but filters out weaker signals."
    )


if __name__ == "__main__":
    main()
