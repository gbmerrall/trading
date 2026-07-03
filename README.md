# Trading Strategy Backtesting Framework

Having Claude Code produce this as I've been idly interested in it on and off so this is a
chance to whip something up. I also saw [Entire](https://entire.io/) drop which seems to be $60M
wrapped in a bunch of commit hooks so thought it might be interesting to check out.

Still a work in progress as interest waxes and wanes. Don't expect to get rich but OTOH who knows?

---

## Installation

Requires Python 3.12+.

### With uv (recommended)

```bash
uv sync
```

That installs all dependencies and the `backtest` package itself as an editable install.

### With pip

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install as editable package (includes all dependencies)
pip install -e .
```

---

## The Main Pipeline: trade_analysis.py

```bash
uv run python trade_analysis.py SPY
```

This is the primary entry point. For the given ticker it:

1. Downloads OHLCV history (2020 to present, adjusted prices).
2. Runs every built-in strategy over the full period for a descriptive comparison
   (this table is in-sample and used only for context — it does not gate anything).
3. Runs Walk-Forward Analysis on **every** strategy with a parameter grid defined in
   `WFA_PARAM_GRIDS` (14 at the time of writing). Strategies are ranked by their WFA
   **out-of-sample Sharpe ratio**, not the in-sample comparison.
4. For each candidate, also backtests its fixed `best_params_overall` over the same
   out-of-sample span, so the card records how the *deployed* parameter set performed —
   the WFA summary itself is earned by adaptive per-window parameters.
5. Writes `output/<ticker>_report.html` (full report) and `output/<ticker>_card.json`
   (the machine-readable strategy card consumed by the daily cron / reconciler).

Run settings live as constants at the top of the script:

| Constant | Default | Meaning |
|----------|---------|---------|
| `START_CAPITAL` | 10,000 | Starting capital for every simulation |
| `COMMISSION_FIXED` | 3.0 | Flat USD fee charged per fill (each buy and each sell) |
| `COMMISSION_RATE` | 0.0 | Percentage-of-value commission per fill |
| `SLIPPAGE_PCT` | 0.0005 | 5 bps per fill: buys fill above the open, sells below |
| `MIN_TRADES` | 3 | Training windows with fewer trades score `-inf` in WFA |

---

## Cost Modelling

Costs are configured on `PortfolioConfig` and applied by the runner to every simulation
(including WFA training and test windows, and parallel workers):

```python
from backtest.config import ConfigFactory, set_config

config = ConfigFactory.create_default()
config.portfolio.commission_fixed = 3.0    # flat fee per fill
config.portfolio.commission_rate = 0.0     # plus a percentage of trade value
config.portfolio.slippage_pct = 0.0005     # buys fill higher, sells fill lower
set_config(config)
```

Position sizing accounts for costs (shares are sized so cost + commission fits the
budget), and trade `pnl` is net of both sides' commissions.

---

## Getting Started

### 1. Run a basic backtest

```bash
python examples/basic_backtest.py
```

RSI(14) vs Buy & Hold, SPY, and DCA on AAPL over 5 years. Prints a metric
summary and final portfolio value.

### 2. Compare all strategies side by side

```bash
python examples/compare_strategies.py           # defaults: AAPL 2020–2024
python examples/compare_strategies.py MSFT 2018-01-01 2024-01-01
```

Runs every built-in strategy on the same data and prints a table ranked by total return.

### 3. Parameter sensitivity sweep

```bash
python examples/parameter_sensitivity.py
python examples/parameter_sensitivity.py SPY 2018-01-01 2024-01-01
```

Sweeps RSI period from 5 to 30, printing Sharpe, Sortino, Calmar, and trade count at
each value. Helps identify stable parameter regions before optimising.

### 4. Walk-Forward Analysis

```bash
python examples/walk_forward_analysis.py
python examples/walk_forward_analysis.py SPY 2016-01-01 2024-01-01
python examples/walk_forward_analysis.py AAPL 2016-01-01 2024-01-01 -1   # all CPUs
```

WFA repeatedly optimises RSI parameters on a 1-year training window and validates on the
following 3-month test window. Prints per-window results and overall out-of-sample metrics.
Saves three plots to `output/`: chained equity curve, parameter stability, and per-window
metric bars. Pass a 4th argument to control parallel candidate evaluation (`-1` = all CPUs,
`2` = 2 workers, `1` = sequential).

### 5. Ensemble strategy

```bash
python examples/ensemble_strategy.py
python examples/ensemble_strategy.py MSFT 2019-01-01 2024-01-01
```

Combines RSI, MACD, and Bollinger Bands into an ensemble that signals only when at least
2 of 3 strategies agree. Compares the ensemble against each constituent and Buy & Hold.

### 6. Multi-asset portfolio blend

```bash
uv run python examples/multi_asset_portfolio.py
```

Runs two pre-optimised strategies on separate assets (SPY breakout, COIN MACD) and blends
their equity curves into a capital-weighted portfolio. Illustrative of the blending
mechanics only — it predates the cost model and uses fixed in-sample parameters.

### Available objective metrics

The following metrics can be passed as the `objective` parameter to `WalkForwardOptimizer`,
or called directly from `backtest.metrics`:

```python
from backtest.metrics import METRICS

print(list(METRICS.keys()))
# ['total_return', 'cagr', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
#  'max_drawdown', 'ulcer_index', 'profit_factor', 'win_rate', 'expectancy',
#  'recovery_factor']
```

You can also pass any callable with the signature `(portfolio_history, trades) -> float`
as a custom objective.

Metric conventions worth knowing:

- **Sharpe and Sortino measure excess return** over the configured annual
  `risk_free_rate` (`BacktestConfig`, default 2%), applied as `rf / 252` daily.
- **Degenerate windows return `-inf`**: zero-volatility Sharpe, or Sortino with no
  negative returns, can never win the optimizer.
- Higher is always better; drawdown-style metrics are negated accordingly.

---

## Strategies

### Technical Indicator Strategies

| Class | Description |
|-------|-------------|
| `ConsecutiveDaysStrategy` | Buy after N consecutive down days, sell after N consecutive up days |
| `MovingAverageCrossoverStrategy` | Golden/death cross on short- and long-term moving average crossovers |
| `RSIStrategy` | Buy when RSI is oversold, sell when overbought |
| `MACDStrategy` | Buy/sell on MACD line crossovers with the signal line |
| `BollingerBandsStrategy` | Mean reversion on Bollinger Band touches |
| `ParabolicSARStrategy` | Trend-following with Parabolic SAR (requires High/Low columns) |

### Price Action Strategies

| Class | Description |
|-------|-------------|
| `BreakoutStrategy` | Enter on N-day high breakout or low breakdown |
| `GapStrategy` | Gap trading on overnight price discontinuities (requires Open column) |
| `FibonacciRetracementStrategy` | Support/resistance at 38.2%, 50%, 61.8% retracement levels (requires High/Low) |

### Composite and Wrapper Strategies

| Class | Description |
|-------|-------------|
| `MeanReversionStrategy` | Combines RSI and Bollinger Bands for mean reversion signals |
| `MomentumStrategy` | Rate-of-Change (ROC) momentum — buy/sell on threshold crossovers |
| `VolatilityStrategy` | ATR-based volatility breakout detection (requires High/Low) |
| `EnsembleStrategy` | Wraps multiple sub-strategies, aggregates signals via majority voting |
| `RegimeFilteredStrategy` | Wraps any strategy; vetoes buys in unfavourable regimes (ADX trend strength or SMA bull/bear filter) |

All strategies shift signals by one bar and the runner executes at the next open, so a
signal computed from Monday's close trades at Tuesday's open.

### Walk-Forward Analysis

`WalkForwardOptimizer` optimises any strategy's parameters over a rolling or expanding train
window and evaluates out-of-sample on the following test window. Results include a stitched
equity curve, per-window metrics, and the most stable parameter set across all windows.

Methodology details:

- **Full-context signals**: signals for each window are generated on all history up to the
  window's end, then sliced — lookback indicators are already warm at the window's first
  bar, so no out-of-sample bars are discarded for warmup.
- **Position carry**: if the strategy's last entry event fired before a window started,
  a buy is forced on the window's first bar. Event-based strategies (crossovers) would
  otherwise idle in cash through windows whose entry signal predates them.
- **Cost-aware**: commissions and slippage from the global config apply in training and
  test simulations alike, including parallel workers.
- `RandomSearch` samples unique combinations (returns the whole grid when `n` covers it).

Pass `n_jobs=-1` to evaluate parameter candidates in parallel using all available CPUs.
Custom callable objectives fall back to sequential automatically.

`backtest.reporting` turns a `WalkForwardResult` into three plotly figures:

```python
from backtest.reporting import plot_equity_curve, plot_parameter_stability, plot_metrics_by_window, save_wfa_report

fig = plot_equity_curve(result, chain=True, start_capital=10_000)
fig.show()

# Or save all three plots at once:
save_wfa_report(result, output_dir="output", prefix="my_run")
```

### The Strategy Card

`trade_analysis.py` writes `output/<ticker>_card.json` — the single artifact that crosses
from analysis to deployment. Per candidate it records:

- `params`: the modal best parameter set across WFA windows (`best_params_overall`) —
  what the daily cron actually trades.
- `wfa_baseline`: out-of-sample metrics earned by *adaptive* per-window parameters.
- `fixed_params_baseline`: metrics from a plain backtest of `params` over the same
  out-of-sample span — the honest estimate for the deployed configuration. If this is
  much worse than `wfa_baseline`, the strategy's edge depends on re-optimisation.

`recommended` is the candidate with the highest WFA out-of-sample Sharpe. Sanity-check it
against `param_stability` and the fixed baseline before deploying — the best of 14 noisy
out-of-sample curves still overstates its true edge (winner's curse).

---

## Running Tests

```bash
# Run the full test suite
uv run pytest

# With coverage report
uv run pytest --cov=backtest -v
```

---

## Known Caveats

- Prices come from yfinance with `auto_adjust=True`. History is back-adjusted by later
  dividends/splits, so the price (and signal) shown for a historical date can drift from
  what a live run saw on that date — expect small discrepancies when reconciling.
- Fills are modelled at the open plus/minus slippage; there is no volume or spread model.
- Single ticker at a time: picking the ticker is itself a selection decision the
  framework cannot de-bias.

## Scope Constraints

- All strategies operate on **single-asset OHLCV data**. Cross-asset strategies (pairs trading,
  relative momentum) are out of scope until the runner supports multi-asset event loops.
- **Volatility strategies** use ATR from price data only. Options-based or VIX-based approaches
  require a different data model and are out of scope.
- **Momentum strategies** use single-asset Rate-of-Change. Cross-sectional momentum (ranking
  multiple assets) requires runner changes and is out of scope.

---

## License

MIT License
