# Trading Strategy Backtesting Framework

Having Claude Code produce this as I've been idly interested in it on and off so this is a 
chance to whip something up. I also saw [Entire](https://entire.io/) drop which seems to be $60M
wrapped in a bunch of commit hooks so thought it might be interesting to check out. 

Still a work in progress as interest waxes and wanes. Don't expect to get rich but OTOH who knows?

### Running a Backtest

The primary example is in `examples/run_backtest.py`. To run it:

```bash
python examples/run_backtest.py
```

This will:
1.  Download historical price data for a stock (e.g., BHP).
2.  Run the `ConsecutiveDaysStrategy` against it.
3.  Compare its performance against `BuyAndHold`, `SPYBuyAndHold`, and `DollarCostAveraging` benchmarks.
4.  Print a formatted summary of performance metrics to the console.
5.  Save a plot of the equity curves to the `output/` directory with a timestamped filename.

## Strategies

See [STRATEGIES.md](STRATEGIES.md) for what's been implemented.


## License

MIT License 