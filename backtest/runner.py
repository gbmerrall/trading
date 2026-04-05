import pandas as pd
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from .portfolio import Portfolio
from .validation import (
    validate_dataframe, validate_price_data, validate_positive_number, 
    sanitize_file_path, ValidationError
)
from .config import get_backtest_config, get_portfolio_config, FileConfig
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
                    'win_rate': 0.0, 
                    'num_trades': 0, 
                    'max_drawdown': 0.0
                }

            start_value = portfolio_history[0]["value"]
            if start_value <= 0:
                raise ValidationError(f"Invalid start value: {start_value}")

            pct = TradingConstants.PERCENT_MULTIPLIER
            
            # Calculate individual metrics from metrics module
            tr_raw = _metrics.total_return(portfolio_history, trades)
            wr_raw = _metrics.win_rate(portfolio_history, trades)
            mdd_raw = _metrics.max_drawdown(portfolio_history, trades)
            
            # Adjust win_rate: metrics.py returns float('-inf') if no trades
            win_rate = 0.0 if wr_raw == float("-inf") else wr_raw * pct
            
            return {
                "total_return": float(tr_raw * pct),
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
        start_capital: float = None, 
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the backtest using an event-driven approach.
        
        Args:
            data: Price data DataFrame with 'Close' column
            start_capital: Starting capital amount. If None, uses config default.
            output_file: Optional path to save equity curve plot
            
        Returns:
            Dictionary containing backtest results and metrics
            
        Raises:
            ValidationError: If validation or execution fails
        """
        portfolio_config = get_portfolio_config()

        # Use configuration default if not provided
        if start_capital is None:
            start_capital = portfolio_config.start_capital
            
        # Comprehensive input validation
        self._validate_data(data)
        self._validate_backtest_params(start_capital, output_file)
        
        try:
            # Generate trading signals
            signals = self.strategy.generate_signals(data)
            
            if signals.empty:
                raise ValidationError("Strategy generated empty signals")
            
            if not all(col in signals.columns for col in ['buy', 'sell']):
                raise ValidationError("Strategy signals must contain 'buy' and 'sell' columns")
            
            # Ensure close_prices is a Series
            close_prices = data["Close"]
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.squeeze()
            
            portfolio = Portfolio(start_capital)
            trades = []

            shares_held = 0
            entry_price = 0.0
            entry_date = None

            # Event-driven backtesting loop
            for date, price in close_prices.items():
                if pd.isna(price) or price <= 0:
                    continue  # Skip invalid prices

                buy_signal = signals.loc[date, "buy"] if date in signals.index else False
                sell_signal = signals.loc[date, "sell"] if date in signals.index else False

                # Trading logic with validation
                if shares_held == 0 and buy_signal:
                    # Invest all available cash
                    shares_to_buy = int(portfolio.cash // price)
                    if shares_to_buy > 0:
                        portfolio.process_day(date, price, buy_signal=True, shares=shares_to_buy)
                        shares_held = shares_to_buy
                        entry_price = price
                        entry_date = date

                elif shares_held > 0 and sell_signal:
                    # Sell entire position
                    portfolio.process_day(date, price, sell_signal=True, shares=shares_held)
                    pnl = (price - entry_price) * shares_held
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry": entry_price,
                        "exit": price,
                        "shares": shares_held,
                        "pnl": pnl,
                    })
                    shares_held = 0
                    entry_price = 0.0
                    entry_date = None
                else:
                    # Hold position
                    portfolio.process_day(date, price)

            # Close any remaining position at the last price
            # Note: We don't call process_day again since the last iteration already did
            if shares_held > 0:
                last_date = close_prices.index[-1]
                last_price = close_prices.iloc[-1]
                pnl = (last_price - entry_price) * shares_held
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": last_date,
                    "entry": entry_price,
                    "exit": last_price,
                    "shares": shares_held,
                    "pnl": pnl,
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

            # Trim to post-warmup period so metrics and date ranges exclude dead cash time
            warmup_n = getattr(self.strategy, 'warmup_period', 0)
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
