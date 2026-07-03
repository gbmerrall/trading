import pandas as pd
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from .portfolio import Portfolio
from .validation import (
    validate_dataframe, validate_price_data, validate_positive_number, 
    sanitize_file_path, ValidationError
)
from .config import get_backtest_config, get_portfolio_config, get_strategy_config, FileConfig
from .constants import TradingConstants, ValidationLimits
from . import metrics as _metrics


class BacktestRunnerImpl:
    """Implementation of the BacktestRunner with comprehensive validation and configuration."""

    def __init__(self, strategy, benchmarks: List):
        """
        Initialize the backtest runner.
        
        Args:
            strategy: Trading strategy object with generate_signals method
            benchmarks: List of benchmark objects with calculate_returns method
            
        Raises:
            ValidationError: If inputs are invalid
        """
        if not hasattr(strategy, 'generate_signals'):
            raise ValidationError("Strategy must have 'generate_signals' method")
        
        if not isinstance(benchmarks, list):
            raise ValidationError(f"Benchmarks must be a list, got {type(benchmarks).__name__}")
        
        for i, benchmark in enumerate(benchmarks):
            if not hasattr(benchmark, 'calculate_returns'):
                raise ValidationError(f"Benchmark {i} must have 'calculate_returns' method")
        
        self.strategy = strategy
        self.benchmarks = benchmarks

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data for backtesting.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValidationError: If validation fails
        """
        validate_dataframe(data, required_columns=['Close'], min_rows=2)
        validate_price_data(data, 'Close')
        
        # Additional checks for backtesting
        if data.index.duplicated().any():
            raise ValidationError("Data index contains duplicate dates")
        
        if not data.index.is_monotonic_increasing:
            raise ValidationError("Data must be sorted by date in ascending order")

    def _validate_backtest_params(self, start_capital: float, output_file: Optional[str]) -> None:
        """
        Validate backtest parameters.

        Args:
            start_capital: Starting capital amount
            output_file: Optional output file path

        Raises:
            ValidationError: If validation fails
        """
        validate_positive_number(
            start_capital,
            "start_capital",
            allow_zero=False,
            max_value=ValidationLimits.MAX_START_CAPITAL,
        )
        
        if output_file is not None:
            sanitize_file_path(output_file, allowed_extensions=FileConfig.ALLOWED_IMAGE_EXTENSIONS)

    def _calculate_metrics(self, portfolio_history: List[Dict], trades: List[Dict]) -> Dict[str, Any]:
        """
        Calculate performance metrics from portfolio history and trades.
        
        Args:
            portfolio_history: List of portfolio value records
            trades: List of trade records
            
        Returns:
            Dictionary of performance metrics
            
        Raises:
            ValidationError: If calculation fails
        """
        try:
            if not portfolio_history:
                return {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'win_rate': 0.0,
                    'num_trades': 0,
                    'max_drawdown': 0.0,
                }

            start_value = portfolio_history[0]["value"]
            if start_value <= 0:
                raise ValidationError(f"Invalid start value: {start_value}")

            pct = TradingConstants.PERCENT_MULTIPLIER

            # Calculate individual metrics from metrics module
            tr_raw = _metrics.total_return(portfolio_history, trades)
            wr_raw = _metrics.win_rate(portfolio_history, trades)
            mdd_raw = _metrics.max_drawdown(portfolio_history, trades)
            sr_raw = _metrics.sharpe_ratio(portfolio_history, trades)

            # Adjust win_rate: metrics.py returns float('-inf') if no trades
            win_rate = 0.0 if wr_raw == float("-inf") else wr_raw * pct
            sharpe = 0.0 if sr_raw == float("-inf") else float(sr_raw)

            return {
                "total_return": float(tr_raw * pct),
                "sharpe_ratio": sharpe,
                "win_rate": float(win_rate),
                "num_trades": len(trades),
                "max_drawdown": float(mdd_raw * pct),
            }
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Error calculating metrics: {str(e)}") from e

    def _plot_equity_curves(
        self,
        strategy_history: List[Dict],
        benchmark_returns: Dict[str, pd.Series],
        output_file: str
    ) -> None:
        """
        Plot equity curves using plotly with configuration.
        
        Args:
            strategy_history: Strategy performance history
            benchmark_returns: Benchmark performance data
            output_file: Path to save the plot
            
        Raises:
            ValidationError: If plotting fails
        """
        try:
            config = get_backtest_config()
            
            fig = go.Figure()
            
            if not strategy_history:
                raise ValidationError("Strategy history is empty")
            
            strategy_df = pd.DataFrame(strategy_history).set_index('date')

            # Add strategy line
            fig.add_trace(go.Scatter(
                x=strategy_df.index,
                y=strategy_df['value'],
                name='Strategy',
                line=dict(color=config.plot_colors.get('strategy', 'blue'))
            ))
            
            # Add benchmark lines with configured colors
            default_colors = ["green", "red", "orange", "purple"]
            for i, (name, returns) in enumerate(benchmark_returns.items()):
                if not isinstance(returns, pd.Series):
                    returns = returns.squeeze()
                
                color_key = f'benchmark{i+1}'
                color = config.plot_colors.get(color_key, default_colors[i % len(default_colors)])
                
                fig.add_trace(go.Scatter(
                    x=returns.index, 
                    y=returns, 
                    name=name, 
                    line=dict(color=color)
                ))

            fig.update_layout(
                title=config.plot_title,
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                hovermode="x unified",
                width=FileConfig.DEFAULT_PLOT_WIDTH * 100,
                height=FileConfig.DEFAULT_PLOT_HEIGHT * 100
            )
            
            fig.write_image(output_file, engine="kaleido")
            
        except Exception as e:
            raise ValidationError(f"Error creating plot: {str(e)}") from e

    def run(
        self,
        data: pd.DataFrame,
        start_capital: Optional[float] = None,
        output_file: Optional[str] = None,
        precomputed_signals: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Run the backtest using an event-driven approach.

        Trades are executed at the Open price (since signals are generated at the
        previous day's close). End-of-day portfolio values are marked to market at
        the Close price. Stop-losses are checked against the daily Low.

        Args:
            data: Price data DataFrame with at minimum a 'Close' column. Including
                'Open' enables realistic execution pricing; including 'Low' enables
                hard stop-loss support.
            start_capital: Starting capital amount. If None, uses config default.
            output_file: Optional path to save equity curve plot
            precomputed_signals: Optional DataFrame with 'buy'/'sell' columns already
                computed (e.g. from a larger context window). When provided, skips
                strategy.generate_signals(data). The index must align with data's index.

        Returns:
            Dictionary containing backtest results and metrics

        Raises:
            ValidationError: If validation or execution fails
        """
        portfolio_config = get_portfolio_config()
        strategy_config = get_strategy_config()

        # Use configuration default if not provided
        if start_capital is None:
            start_capital = portfolio_config.start_capital

        # Comprehensive input validation
        self._validate_data(data)
        self._validate_backtest_params(start_capital, output_file)

        try:
            # Use pre-computed signals when provided (avoids lookback window crashes
            # when the data slice is shorter than the strategy's warmup period).
            if precomputed_signals is not None:
                signals = precomputed_signals
            else:
                signals = self.strategy.generate_signals(data)

            if signals.empty:
                raise ValidationError("Strategy generated empty signals")

            if not all(col in signals.columns for col in ['buy', 'sell']):
                raise ValidationError("Strategy signals must contain 'buy' and 'sell' columns")

            has_stop_loss_col = 'stop_loss' in signals.columns

            # Close is always required; Open and Low are optional with close fallback
            close_prices = data["Close"]
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.squeeze()

            if "Open" in data.columns:
                open_prices = data["Open"]
                if isinstance(open_prices, pd.DataFrame):
                    open_prices = open_prices.squeeze()
            else:
                open_prices = close_prices

            # Low prices are only needed when strategies emit stop_loss levels
            if "Low" in data.columns and has_stop_loss_col:
                low_prices = data["Low"]
                if isinstance(low_prices, pd.DataFrame):
                    low_prices = low_prices.squeeze()
            else:
                low_prices = None

            # Convert per-bar lookups to numpy arrays indexed by integer position.
            # Profiling showed ~75% of WFA time was spent in DatetimeIndex.get_loc
            # from per-iteration .loc calls. Reindex signals to data's index so
            # positional indexing is safe; pandas-level .loc fallbacks are no longer
            # needed inside the loop.
            dates = close_prices.index
            close_arr = close_prices.to_numpy(dtype=float, copy=False)
            open_arr = open_prices.reindex(dates).to_numpy(dtype=float, copy=False)
            low_arr = (
                low_prices.reindex(dates).to_numpy(dtype=float, copy=False)
                if low_prices is not None
                else None
            )
            aligned_signals = signals.reindex(dates)
            buy_arr = aligned_signals["buy"].fillna(False).to_numpy(dtype=bool, copy=False)
            sell_arr = aligned_signals["sell"].fillna(False).to_numpy(dtype=bool, copy=False)
            stop_loss_arr = (
                aligned_signals["stop_loss"].to_numpy(dtype=float, copy=False)
                if has_stop_loss_col
                else None
            )

            commission_rate = portfolio_config.commission_rate
            commission_fixed = portfolio_config.commission_fixed
            slippage_pct = portfolio_config.slippage_pct

            def _commission(trade_value: float) -> float:
                return trade_value * commission_rate + commission_fixed

            portfolio = Portfolio(
                start_capital,
                commission_rate=commission_rate,
                commission_fixed=commission_fixed,
            )
            symbol = portfolio.default_symbol
            trades = []

            shares_held = 0
            entry_price = 0.0
            entry_date = None
            entry_commission = 0.0
            active_stop_loss: Optional[float] = None
            pos_size_pct = strategy_config.position_size_pct
            use_pct_sizing = strategy_config.position_size_method == 'percentage'

            # Event-driven backtesting loop (positional numpy iteration)
            n_bars = len(dates)
            for i in range(n_bars):
                close_price = close_arr[i]
                if close_price != close_price or close_price <= 0:  # NaN-safe, faster than pd.isna
                    continue

                date = dates[i]
                open_price = open_arr[i]
                if open_price != open_price or open_price <= 0:
                    open_price = close_price

                buy_signal = buy_arr[i]
                sell_signal = sell_arr[i]

                # --- Stop-loss check (evaluated before normal signals) ---
                stop_loss_triggered = False
                if shares_held > 0 and active_stop_loss is not None and low_arr is not None:
                    low_price = low_arr[i]
                    if low_price == low_price:  # not NaN
                        if low_price <= active_stop_loss:
                            # Gap-down open gets the open price; otherwise execute at the stop level
                            exec_price = open_price if open_price <= active_stop_loss else active_stop_loss
                            exec_price *= 1.0 - slippage_pct
                            portfolio.sell(symbol, shares_held, exec_price, date)
                            exit_commission = _commission(exec_price * shares_held)
                            trades.append({
                                "entry_date": entry_date,
                                "exit_date": date,
                                "entry": entry_price,
                                "exit": exec_price,
                                "shares": shares_held,
                                "pnl": (exec_price - entry_price) * shares_held
                                - entry_commission - exit_commission,
                                "exit_reason": "stop_loss",
                            })
                            shares_held = 0
                            entry_price = 0.0
                            entry_date = None
                            entry_commission = 0.0
                            active_stop_loss = None
                            stop_loss_triggered = True

                # --- Normal signal execution ---
                if not stop_loss_triggered:
                    if shares_held == 0 and buy_signal:
                        position_value = (
                            portfolio.cash * pos_size_pct if use_pct_sizing else portfolio.cash
                        )
                        # Buys fill above the open by the slippage fraction
                        fill_price = open_price * (1.0 + slippage_pct)
                        # Size so that share cost plus commission fits within the budget
                        affordable = position_value - commission_fixed
                        shares_to_buy = (
                            int(affordable // (fill_price * (1.0 + commission_rate)))
                            if affordable > 0
                            else 0
                        )
                        if shares_to_buy > 0:
                            success = portfolio.buy(symbol, shares_to_buy, fill_price, date)
                            if success:
                                shares_held = shares_to_buy
                                entry_price = fill_price
                                entry_date = date
                                entry_commission = _commission(fill_price * shares_to_buy)
                                if stop_loss_arr is not None:
                                    sl_val = stop_loss_arr[i]
                                    active_stop_loss = None if sl_val != sl_val else float(sl_val)

                    elif shares_held > 0 and sell_signal:
                        # Sells fill below the open by the slippage fraction
                        fill_price = open_price * (1.0 - slippage_pct)
                        portfolio.sell(symbol, shares_held, fill_price, date)
                        exit_commission = _commission(fill_price * shares_held)
                        trades.append({
                            "entry_date": entry_date,
                            "exit_date": date,
                            "entry": entry_price,
                            "exit": fill_price,
                            "shares": shares_held,
                            "pnl": (fill_price - entry_price) * shares_held
                            - entry_commission - exit_commission,
                            "exit_reason": "signal",
                        })
                        shares_held = 0
                        entry_price = 0.0
                        entry_date = None
                        entry_commission = 0.0
                        active_stop_loss = None

                # End-of-day mark-to-market at close, regardless of intraday execution
                portfolio.update_position_price(symbol, close_price)
                portfolio._record_portfolio_value(date)

            # Record open position at last close for trade accounting
            if shares_held > 0:
                last_date = close_prices.index[-1]
                last_price = close_prices.iloc[-1]
                # Position is still open: only the entry commission has been paid.
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": last_date,
                    "entry": entry_price,
                    "exit": last_price,
                    "shares": shares_held,
                    "pnl": (last_price - entry_price) * shares_held - entry_commission,
                    "exit_reason": "end_of_data",
                })

            # Calculate final results
            strategy_history = portfolio.get_value_history()
            if not strategy_history:
                raise ValidationError("No portfolio history generated")

            # Calculate raw benchmark returns on the full data window
            benchmark_returns_raw = {}
            for bench in self.benchmarks:
                bench_name = bench.__class__.__name__
                try:
                    benchmark_returns_raw[bench_name] = bench.calculate_returns(data, start_capital)
                except Exception as e:
                    raise ValidationError(
                        f"Error calculating {bench_name} benchmark: {str(e)}"
                    ) from e

            # Trim to post-warmup period so metrics and date ranges exclude dead cash
            # time. Only when the runner generated signals itself: precomputed signals
            # come from a wider context, so every bar of this window is already live.
            warmup_n = (
                0 if precomputed_signals is not None
                else getattr(self.strategy, 'warmup_period', 0)
            )
            if warmup_n > 0 and len(close_prices) > warmup_n:
                cutoff_date = close_prices.index[warmup_n]
                strategy_history = [r for r in strategy_history if r['date'] >= cutoff_date]
                trades = [t for t in trades if t['entry_date'] >= cutoff_date]
                benchmark_returns = {
                    name: returns[returns.index >= cutoff_date]
                    for name, returns in benchmark_returns_raw.items()
                }
            else:
                benchmark_returns = benchmark_returns_raw

            strategy_returns = pd.DataFrame(strategy_history).set_index('date')['value']
            strategy_metrics = self._calculate_metrics(strategy_history, trades)

            benchmark_metrics = {}
            for bench_name, bench_returns in benchmark_returns.items():
                bench_history = [
                    {'date': date, 'value': value}
                    for date, value in bench_returns.items()
                ]
                benchmark_metrics[bench_name] = self._calculate_metrics(bench_history, [])

            # Generate plot if requested
            backtest_config = get_backtest_config()
            if output_file and backtest_config.save_plots:
                self._plot_equity_curves(strategy_history, benchmark_returns, output_file)
            
            return {
                "strategy_metrics": strategy_metrics,
                "benchmark_metrics": benchmark_metrics,
                "strategy_returns": strategy_returns,
                "benchmark_returns": benchmark_returns,
                "signals": signals,
                "trades": trades,
            }
            
        except ValidationError:
            raise  # Re-raise validation errors as-is
        except Exception as e:
            raise ValidationError(f"Backtest execution failed: {str(e)}") from e


def compare_strategies(
    data: pd.DataFrame,
    strategies: List[tuple],
    benchmarks: Optional[List] = None,
    start_capital: Optional[float] = None,
    sort_metric: str = "sharpe_ratio",
    top_n: int = 3,
) -> tuple:
    """Run multiple strategies, returning top_n sorted by a risk-adjusted metric.

    Strategies are ranked in descending order by sort_metric (e.g. "sharpe_ratio",
    "total_return"). The comparison table in reports will reflect only the returned
    top_n, so callers that want a full league table should pass top_n=len(strategies).

    Args:
        data: Price data DataFrame passed to each BacktestRunnerImpl.
        strategies: List of (name, strategy_instance) tuples.
        benchmarks: Optional list of benchmark instances. When provided, benchmarks
            are evaluated once alongside the first strategy.
        start_capital: Starting capital; uses config default if None.
        sort_metric: Key in the strategy_metrics dict to rank by (default "sharpe_ratio").
        top_n: Number of top strategies to return (default 3).

    Returns:
        Tuple of:
        - results: list of top_n (name, metrics_dict, returns_series) sorted descending
        - benchmark_metrics: dict mapping benchmark name to metrics dict
        - benchmark_returns: dict mapping benchmark name to pd.Series
    """
    portfolio_config = get_portfolio_config()
    if start_capital is None:
        start_capital = portfolio_config.start_capital

    benchmarks = benchmarks or []
    benchmark_metrics: Dict[str, Any] = {}
    benchmark_returns: Dict[str, pd.Series] = {}

    # Evaluate benchmarks once, piggybacking on the first strategy run
    if benchmarks and strategies:
        _, first_strategy = strategies[0]
        bench_runner = BacktestRunnerImpl(strategy=first_strategy, benchmarks=benchmarks)
        try:
            bench_result = bench_runner.run(data, start_capital=start_capital)
            benchmark_metrics = bench_result["benchmark_metrics"]
            benchmark_returns = bench_result["benchmark_returns"]
        except Exception as exc:
            print(f"  [warn] Benchmark run failed: {exc}")

    results = []
    for name, strategy in strategies:
        try:
            runner = BacktestRunnerImpl(strategy=strategy, benchmarks=[])
            res = runner.run(data, start_capital=start_capital)
            results.append((name, res["strategy_metrics"], res["strategy_returns"]))
        except Exception as exc:
            print(f"  [skip] {name}: {exc}")

    def _sort_key(item: tuple) -> float:
        val = item[1].get(sort_metric, float("-inf"))
        # Guard against NaN or -inf produced when there are no trades
        if val != val or val == float("-inf"):
            return float("-inf")
        return float(val)

    results.sort(key=_sort_key, reverse=True)
    return results[:top_n], benchmark_metrics, benchmark_returns
