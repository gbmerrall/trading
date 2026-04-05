"""Compare multiple strategies on the same ticker and time period.

Runs every built-in strategy against the same data and prints a ranked
comparison table sorted by total return.

Usage:
    python examples/compare_strategies.py
    python examples/compare_strategies.py MSFT 2020-01-01 2024-01-01
"""


import yfinance as yf


from backtest.benchmarks import BuyAndHold
from backtest.runner import BacktestRunnerImpl
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
    RSIStrategy,
    VolatilityStrategy,
)


def download_data(ticker: str, start: str, end: str):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, type(data.columns)) and hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)
    return data


def build_strategies():
    """Return a list of (name, strategy) pairs covering all built-in strategies."""
    return [
        ("ConsecutiveDays(3)",       ConsecutiveDaysStrategy(consecutive_days=3)),
        ("MovingAvgCrossover(20/50)", MovingAverageCrossoverStrategy(short_window=20, long_window=50)),
        ("RSI(14, 30/70)",           RSIStrategy(period=14, lower_bound=30, upper_bound=70)),
        ("MACD(12/26/9)",            MACDStrategy()),
        ("BollingerBands(20, 2σ)",   BollingerBandsStrategy()),
        ("ParabolicSAR",             ParabolicSARStrategy()),
        ("Breakout(20d)",            BreakoutStrategy(lookback_period=20)),
        ("Gap(2%)",                  GapStrategy(min_gap_pct=0.02)),
        ("Fibonacci(20d)",           FibonacciRetracementStrategy(swing_lookback=20)),
        ("MeanReversion",            MeanReversionStrategy()),
        ("Momentum(ROC-12)",         MomentumStrategy()),
        ("Volatility(ATR-14)",       VolatilityStrategy()),
        ("Ensemble(RSI+MACD+BB)",    EnsembleStrategy(
            strategies=[
                RSIStrategy(period=14, lower_bound=30, upper_bound=70),
                MACDStrategy(),
                BollingerBandsStrategy(),
            ]
        )),
    ]


def run_strategy(strategy, data, start_capital):
    """Run a single strategy and return its metrics, or None on failure."""
    try:
        runner = BacktestRunnerImpl(strategy=strategy, benchmarks=[])
        result = runner.run(data, start_capital=start_capital)
        return result["strategy_metrics"]
    except Exception as exc:
        return {"error": str(exc)}


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    start = sys.argv[2] if len(sys.argv) > 2 else "2020-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2024-01-01"
    start_capital = 10_000.0

    print(f"Downloading {ticker} ({start} to {end})...")
    data = download_data(ticker, start, end)
    print(f"  {len(data)} trading days\n")

    # Benchmark for reference
    bh_runner = BacktestRunnerImpl(strategy=ConsecutiveDaysStrategy(consecutive_days=999), benchmarks=[BuyAndHold()])
    bh_result = bh_runner.run(data, start_capital=start_capital)
    bh_metrics = bh_result["benchmark_metrics"]["BuyAndHold"]

    strategies = build_strategies()
    results = []

    for name, strategy in strategies:
        metrics = run_strategy(strategy, data, start_capital)
        if "error" in metrics:
            print(f"  [skip] {name}: {metrics['error']}")
        else:
            results.append((name, metrics))

    # Sort by total return descending
    results.sort(key=lambda x: x[1]["total_return"], reverse=True)

    # Print table
    col_w = 28
    print(f"{'Strategy':<{col_w}} {'Return':>8} {'Drawdown':>10} {'Win%':>7} {'Trades':>7}")
    print("-" * (col_w + 36))

    for name, m in results:
        print(
            f"{name:<{col_w}} "
            f"{m['total_return']:>+7.1f}% "
            f"{m['max_drawdown']:>9.1f}% "
            f"{m['win_rate']:>6.1f}% "
            f"{m['num_trades']:>7}"
        )

    print("-" * (col_w + 36))
    print(
        f"{'Buy & Hold (benchmark)':<{col_w}} "
        f"{bh_metrics['total_return']:>+7.1f}% "
        f"{bh_metrics['max_drawdown']:>9.1f}% "
        f"{'n/a':>6}  "
        f"{'n/a':>7}"
    )


if __name__ == "__main__":
    main()
