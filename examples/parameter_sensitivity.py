"""Parameter sensitivity analysis: sweep one parameter and observe metric changes.

Runs RSI with different period values (5 to 30) and prints how Sharpe ratio,
total return, and trade count respond. Useful for finding a stable parameter
region before committing to a value.

Usage:
    python examples/parameter_sensitivity.py
    python examples/parameter_sensitivity.py SPY 2018-01-01 2024-01-01
"""


import yfinance as yf


from backtest.metrics import METRICS
from backtest.runner import BacktestRunnerImpl
from backtest.strategy import RSIStrategy


def download_data(ticker: str, start: str, end: str):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, type(data.columns)) and hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)
    return data


def backtest_rsi(data, period: int, lower: int, upper: int, start_capital: float) -> dict:
    """Run a single RSI backtest and return both runner metrics and extended metrics."""
    strategy = RSIStrategy(period=period, lower_bound=lower, upper_bound=upper)
    runner = BacktestRunnerImpl(strategy=strategy, benchmarks=[])
    result = runner.run(data, start_capital=start_capital)

    # Build portfolio_history and trades in the format metrics.py expects
    portfolio_history = [
        {"date": d, "value": v}
        for d, v in result["strategy_returns"].items()
    ]
    trades = result["trades"]

    return {
        "total_return": result["strategy_metrics"]["total_return"],
        "max_drawdown": result["strategy_metrics"]["max_drawdown"],
        "num_trades": result["strategy_metrics"]["num_trades"],
        "sharpe_ratio": METRICS["sharpe_ratio"](portfolio_history, trades),
        "sortino_ratio": METRICS["sortino_ratio"](portfolio_history, trades),
        "calmar_ratio": METRICS["calmar_ratio"](portfolio_history, trades),
    }


def fmt(value, is_pct: bool = False) -> str:
    if value == float("-inf") or value == float("inf"):
        return f"{'n/a':>8}"
    if is_pct:
        return f"{value:>+8.1f}%"
    return f"{value:>8.2f}"


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    start = sys.argv[2] if len(sys.argv) > 2 else "2018-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2024-01-01"
    start_capital = 10_000.0

    # RSI bounds to hold fixed while sweeping period
    lower_bound = 30
    upper_bound = 70
    periods = list(range(5, 31, 1))

    print(f"Downloading {ticker} ({start} to {end})...")
    data = download_data(ticker, start, end)
    print(f"  {len(data)} trading days\n")
    print(f"RSI sensitivity sweep — lower={lower_bound}, upper={upper_bound}\n")

    header = f"{'Period':>7}  {'Return':>9} {'Drawdown':>10} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'Trades':>7}"
    print(header)
    print("-" * len(header))

    best_sharpe = float("-inf")
    best_period = None

    for period in periods:
        try:
            m = backtest_rsi(data, period, lower_bound, upper_bound, start_capital)
        except Exception as exc:
            print(f"  period={period}: error — {exc}")
            continue

        marker = " <-- best Sharpe" if m["sharpe_ratio"] > best_sharpe else ""
        if m["sharpe_ratio"] > best_sharpe:
            best_sharpe = m["sharpe_ratio"]
            best_period = period

        print(
            f"{period:>7}  "
            f"{fmt(m['total_return'], is_pct=True)} "
            f"{fmt(m['max_drawdown'], is_pct=True)} "
            f"{fmt(m['sharpe_ratio'])} "
            f"{fmt(m['sortino_ratio'])} "
            f"{fmt(m['calmar_ratio'])} "
            f"{m['num_trades']:>7}"
            f"{marker}"
        )

    print("-" * len(header))
    if best_period is not None:
        print(f"\nBest Sharpe ratio at period={best_period} ({best_sharpe:.2f})")
    print("\nNote: in-sample sweep — validate with walk_forward_analysis.py before trading.")


if __name__ == "__main__":
    main()
