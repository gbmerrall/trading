import pandas as pd
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from .portfolio import Portfolio


class BacktestRunnerImpl:
    """Implementation of the BacktestRunner."""

    def __init__(self, strategy, benchmarks: List):
        """
        Initialize the backtest runner.
        """
        self.strategy = strategy
        self.benchmarks = benchmarks

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data.
        """
        if data.empty:
            raise ValueError("Data is empty")
        if "Close" not in data.columns:
            raise ValueError("Data must contain 'Close' column")

    def _calculate_metrics(self, portfolio_history: list, trades: list) -> Dict[str, Any]:
        """
        Calculate performance metrics from portfolio history and trades.
        """
        if not portfolio_history:
            return {'total_return': 0.0, 'win_rate': 0.0, 'num_trades': 0, 'max_drawdown': 0.0}

        returns_df = pd.DataFrame(portfolio_history).set_index('date')['value']
        
        # Total Return
        start_value = returns_df.iloc[0]
        end_value = returns_df.iloc[-1]
        total_return = (end_value / start_value - 1) * 100 if start_value != 0 else 0.0

        # Win Rate
        if not trades:
            win_rate = 0.0
        else:
            wins = sum(1 for trade in trades if trade['pnl'] > 0)
            win_rate = (wins / len(trades)) * 100

        # Max Drawdown
        rolling_max = returns_df.expanding().max()
        drawdowns = (returns_df - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "num_trades": len(trades),
            "max_drawdown": max_drawdown,
        }

    def _plot_equity_curves(self, strategy_history: list,
                           benchmark_returns: Dict[str, pd.Series],
                           output_file: str) -> None:
        """
        Plot equity curves using plotly.
        """
        fig = go.Figure()
        
        strategy_df = pd.DataFrame(strategy_history).set_index('date')

        fig.add_trace(go.Scatter(
            x=strategy_df.index,
            y=strategy_df['value'],
            name='Strategy',
            line=dict(color='blue')
        ))
        
        colors = ["green", "red", "orange", "purple"]
        for (name, returns), color in zip(benchmark_returns.items(), colors):
            if not isinstance(returns, pd.Series):
                returns = returns.squeeze()
            fig.add_trace(go.Scatter(x=returns.index, y=returns, name=name, line=dict(color=color)))

        fig.update_layout(
            title="Equity Curves",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            hovermode="x unified",
        )
        
        fig.write_image(output_file)

    def run(
        self, data: pd.DataFrame, start_capital: float, output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the backtest using an event-driven approach.
        """
        self._validate_data(data)
        
        signals = self.strategy.generate_signals(data)
        
        # Ensure close_prices is a Series
        close_prices = data["Close"]
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.squeeze()
        
        portfolio = Portfolio(start_capital)
        trades = []
        
        shares_held = 0
        entry_price = 0

        for date, price in close_prices.items():
            buy_signal = signals.loc[date, 'buy']
            sell_signal = signals.loc[date, 'sell']

            # Trading logic
            if shares_held == 0 and buy_signal:
                # Invest all available cash
                shares_to_buy = portfolio.cash // price
                if shares_to_buy > 0:
                    portfolio.process_day(date, price, buy_signal=True, shares=shares_to_buy)
                    shares_held = shares_to_buy
                    entry_price = price
            elif shares_held > 0 and sell_signal:
                # Sell entire position
                portfolio.process_day(date, price, sell_signal=True, shares=shares_held)
                trades.append({'entry': entry_price, 'exit': price, 'pnl': price - entry_price})
                shares_held = 0
            else:
                # Hold
                portfolio.process_day(date, price)

        # After loop, close any open position at the last price
        if shares_held > 0:
            last_date = close_prices.index[-1]
            last_price = close_prices.iloc[-1]
            portfolio.process_day(last_date, last_price, sell_signal=True, shares=shares_held)
            trades.append({'entry': entry_price, 'exit': last_price, 'pnl': last_price - entry_price})

        # Final calculations
        strategy_history = portfolio.get_value_history()
        strategy_returns = pd.DataFrame(strategy_history).set_index('date')['value']
        strategy_metrics = self._calculate_metrics(strategy_history, trades)
        
        benchmark_returns = {
            bench.__class__.__name__: bench.calculate_returns(data, start_capital)
            for bench in self.benchmarks
        }
        
        benchmark_metrics = {
            name: self._calculate_metrics(
                [{'date': date, 'value': v} for date, v in returns.items()],
                trades=[]
            )
            for name, returns in benchmark_returns.items()
        }
        
        if output_file:
            self._plot_equity_curves(strategy_history, benchmark_returns, output_file)
        
        return {
            "strategy_metrics": strategy_metrics,
            "benchmark_metrics": benchmark_metrics,
            "strategy_returns": strategy_returns,
            "benchmark_returns": benchmark_returns,
            "signals": signals,
        }
