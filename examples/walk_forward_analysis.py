"""Walk-Forward Analysis: optimise RSI parameters and validate out-of-sample.

Divides 8 years of data into 1-year training windows and 3-month test windows.
For each window, RandomSearch finds the best RSI period and bounds. Three plots
are saved to output/: equity curve, parameter stability, and per-window metrics.

Pass n_jobs > 1 (or -1 for all CPUs) to evaluate candidates in parallel:

Usage:
    python examples/walk_forward_analysis.py
    python examples/walk_forward_analysis.py SPY 2016-01-01 2024-01-01
    python examples/walk_forward_analysis.py AAPL 2016-01-01 2024-01-01 -1
"""


import yfinance as yf


from backtest.optimization import RandomSearch, WalkForwardOptimizer
from backtest.reporting import save_wfa_report
from backtest.strategy import RSIStrategy


def download_data(ticker: str, start: str, end: str):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, type(data.columns)) and hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)
    return data


def fmt_pct(value: float) -> str:
    if not isinstance(value, float) or value == float("-inf") or value == float("inf"):
        return "    n/a"
    return f"{value * 100:+.1f}%"


def fmt_ratio(value: float) -> str:
    if not isinstance(value, float) or value == float("-inf") or value == float("inf"):
        return "    n/a"
    return f"{value:.2f}"


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    start = sys.argv[2] if len(sys.argv) > 2 else "2016-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2024-01-01"
    n_jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    train_size = 252    # 1 year
    test_size = 63      # 1 quarter
    start_capital = 10_000.0

    print(f"Downloading {ticker} ({start} to {end})...")
    data = download_data(ticker, start, end)
    print(f"  {len(data)} trading days loaded.\n")

    param_space = {
        "period":      [7, 10, 14, 20, 28],
        "lower_bound": [20, 25, 30, 35],
        "upper_bound": [65, 70, 75, 80],
    }

    optimizer = WalkForwardOptimizer(
        strategy_class=RSIStrategy,
        param_space=param_space,
        data=data,
        train_size=train_size,
        test_size=test_size,
        window_type="sliding",
        searcher=RandomSearch(n=40, seed=42),
        objective="sharpe_ratio",
        min_trades=3,
        n_jobs=n_jobs,
    )

    jobs_label = "all CPUs" if n_jobs == -1 else f"{n_jobs} worker(s)"
    print("Running Walk-Forward Analysis...")
    print(f"  Parameter space : {sum(len(v) for v in param_space.values())} values across {len(param_space)} params")
    print("  Search          : RandomSearch(n=40, seed=42)")
    print(f"  Windows         : {train_size}-bar train / {test_size}-bar test")
    print(f"  Parallel        : {jobs_label}\n")

    result = optimizer.run()

    # --- Per-window summary ---
    print("Per-window results:")
    print(f"  {'Test period':<24} {'Best params':<32} {'Sharpe':>7} {'Return':>8} {'Trades':>7}")
    print("  " + "-" * 82)

    for _, row in result.windows.iterrows():
        params = row["best_params"]
        param_str = f"p={params.get('period')} lo={params.get('lower_bound')} hi={params.get('upper_bound')}"
        test_range = f"{row['test_start'].date()} → {row['test_end'].date()}"
        sharpe = fmt_ratio(row["sharpe_ratio"])
        ret = fmt_pct(row["total_return"])
        print(f"  {test_range:<24} {param_str:<32} {sharpe:>7} {ret:>8} {int(row['n_trades']):>7}")

    # --- Overall summary ---
    s = result.summary
    print()
    print("Overall out-of-sample performance:")
    print(f"  Sharpe ratio     : {fmt_ratio(s['sharpe_ratio'])}")
    print(f"  Total return     : {fmt_pct(s['total_return'])}")
    print(f"  Max drawdown     : {fmt_pct(s['max_drawdown'])}")
    print(f"  Calmar ratio     : {fmt_ratio(s['calmar_ratio'])}")
    print(f"  Win rate         : {fmt_pct(s['win_rate'])}")
    print(f"  Windows          : {s['n_windows']} ({s['n_windows_with_trades']} with trades)")
    print(f"  Param stability  : {s['param_stability'] * 100:.0f}% of windows used the same params")
    print(f"  Best params      : {result.best_params_overall}")

    # --- Save all three WFA plots ---
    try:
        output_dir = Path(__file__).parent.parent / "output"
        written = save_wfa_report(
            result,
            output_dir=output_dir,
            prefix=f"wfa_{ticker.lower()}",
            start_capital=start_capital,
        )
        print("\nPlots saved:")
        for path in written:
            print(f"  {path}")
    except Exception as exc:
        print(f"\nCould not save plots: {exc}")


if __name__ == "__main__":
    main()
