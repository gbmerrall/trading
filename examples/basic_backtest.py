"""Basic backtest: run a single strategy against buy-and-hold benchmarks.

Downloads 5 years of AAPL data, runs RSI(14) and compares it against
Buy & Hold and Dollar Cost Averaging.

Usage:
    python examples/basic_backtest.py
"""

import sys
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.benchmarks import BuyAndHold, DollarCostAveraging, SPYBuyAndHold
from backtest.runner import BacktestRunnerImpl
from backtest.strategy import RSIStrategy


def download_data(ticker: str, start: str, end: str):
    """Download OHLCV data from yfinance and flatten any multi-level columns."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, type(data.columns)) and hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)
    return data


def print_metrics(label: str, metrics: dict) -> None:
    print(f"  {label}:")
    print(f"    Total return : {metrics['total_return']:+.1f}%")
    print(f"    Max drawdown : {metrics['max_drawdown']:.1f}%")
    print(f"    Win rate     : {metrics['win_rate']:.1f}%")
    print(f"    Trades       : {metrics['num_trades']}")


def main():
    ticker = "AAPL"
    start = "2019-01-01"
    end = "2024-01-01"
    start_capital = 10_000.0

    print(f"Downloading {ticker} data ({start} to {end})...")
    data = download_data(ticker, start, end)
    print(f"  {len(data)} trading days loaded.\n")

    strategy = RSIStrategy(period=14, lower_bound=30, upper_bound=70)
    benchmarks = [BuyAndHold(), SPYBuyAndHold(), DollarCostAveraging()]

    runner = BacktestRunnerImpl(strategy=strategy, benchmarks=benchmarks)
    result = runner.run(data, start_capital=start_capital)

    print(f"Results — ${start_capital:,.0f} starting capital")
    print("-" * 40)
    print_metrics("RSI(14) strategy", result["strategy_metrics"])
    print()
    for name, metrics in result["benchmark_metrics"].items():
        print_metrics(name, metrics)

    final_value = result["strategy_returns"].iloc[-1]
    print(f"\n  Final portfolio value: ${final_value:,.2f}")


if __name__ == "__main__":
    main()
