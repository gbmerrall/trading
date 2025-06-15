import pandas as pd
from typing import Dict, List, Optional
import plotly.graph_objects as go
from .interfaces import BacktestRunner, Strategy, Benchmark

class BacktestRunnerImpl(BacktestRunner):
    """Implementation of the BacktestRunner interface."""
    
    def __init__(self, strategy: Strategy, benchmarks: List[Benchmark]):
        """
        Initialize the backtest runner.
        
        Args:
            strategy (Strategy): The trading strategy to backtest
            benchmarks (List[Benchmark]): List of benchmark strategies for comparison
        """
        self.strategy = strategy
        self.benchmarks = benchmarks
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data.
        
        Args:
            data (pd.DataFrame): Price data to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if data.empty:
            raise ValueError("Data is empty")
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            returns (pd.Series): Series of returns
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        if isinstance(returns, pd.DataFrame):
            returns = returns.squeeze()
        total_return = (returns.iloc[-1] / returns.iloc[0] - 1) * 100 if returns.iloc[0] != 0 else 0
        
        # Calculate win rate
        trades = returns.pct_change(fill_method=None)
        winning_trades = trades[trades > 0]
        win_rate = (len(winning_trades) / len(trades)) * 100 if len(trades) > 0 else 0
        
        # Calculate maximum drawdown
        rolling_max = returns.expanding().max()
        drawdowns = (returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown
        }
    
    def _plot_equity_curves(self, strategy_returns: pd.Series, 
                           benchmark_returns: Dict[str, pd.Series],
                           output_file: str) -> None:
        """
        Plot equity curves using plotly.
        
        Args:
            strategy_returns (pd.Series): Strategy returns
            benchmark_returns (Dict[str, pd.Series]): Dictionary of benchmark returns
            output_file (str): Path to save the plot
        """
        fig = go.Figure()
        
        # Add strategy curve
        fig.add_trace(go.Scatter(
            x=strategy_returns.index,
            y=strategy_returns,
            name='Strategy',
            line=dict(color='blue')
        ))
        
        # Add benchmark curves
        colors = ['green', 'red', 'orange', 'purple']
        for (name, returns), color in zip(benchmark_returns.items(), colors):
            fig.add_trace(go.Scatter(
                x=returns.index,
                y=returns,
                name=name,
                line=dict(color=color)
            ))
        
        fig.update_layout(
            title='Equity Curves',
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            hovermode='x unified'
        )
        
        fig.write_image(output_file)
    
    def run(self, data: pd.DataFrame, start_capital: float, output_file: Optional[str] = None) -> Dict:
        """
        Run the backtest.
        
        Args:
            data (pd.DataFrame): Price data for backtesting
            start_capital (float): Initial capital to invest
            output_file (str): Path to save the equity curves plot
            
        Returns:
            Dict: Backtest results including performance metrics and equity curves
        """
        self._validate_data(data)
        
        # Generate signals
        signals = self.strategy.generate_signals(data)
        
        # Calculate strategy returns
        position = 0
        strategy_returns = pd.Series(start_capital, index=data.index)
        
        for i in range(1, len(data)):
            if signals['buy'].iloc[i]:
                strategy_returns.iloc[i] = strategy_returns.iloc[i-1] * (1 + data['Close'].pct_change(fill_method=None).iloc[i])
            elif signals['sell'].iloc[i]:
                strategy_returns.iloc[i] = strategy_returns.iloc[i-1] * (1 - data['Close'].pct_change(fill_method=None).iloc[i])
            else:
                strategy_returns.iloc[i] = strategy_returns.iloc[i-1]
        
        # Calculate benchmark returns
        benchmark_returns = {}
        for benchmark in self.benchmarks:
            benchmark_returns[benchmark.__class__.__name__] = benchmark.calculate_returns(data, start_capital)
        
        # Calculate metrics
        strategy_metrics = self._calculate_metrics(strategy_returns)
        benchmark_metrics = {
            name: self._calculate_metrics(returns)
            for name, returns in benchmark_returns.items()
        }
        
        # Plot equity curves
        if output_file:
            self._plot_equity_curves(strategy_returns, benchmark_returns, output_file)
        
        return {
            'strategy_metrics': strategy_metrics,
            'benchmark_metrics': benchmark_metrics,
            'strategy_returns': strategy_returns,
            'benchmark_returns': benchmark_returns
        } 