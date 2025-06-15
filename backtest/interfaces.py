from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any

class Strategy(ABC):
    """Base class for all trading strategies."""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from price data.
        
        Args:
            data (pd.DataFrame): DataFrame with price data, must have 'Close' column
            
        Returns:
            pd.DataFrame: DataFrame with 'buy' and 'sell' boolean columns
        """
        pass

class Benchmark(ABC):
    """Base class for all benchmark strategies."""
    
    @abstractmethod
    def calculate_returns(self, data: pd.DataFrame, start_capital: float) -> pd.Series:
        """
        Calculate returns for the benchmark strategy.
        
        Args:
            data (pd.DataFrame): DataFrame with price data, must have 'Close' column
            start_capital (float): Initial capital to invest
            
        Returns:
            pd.Series: Series of cumulative returns
        """
        pass

class BacktestRunner:
    """Main class for running backtests."""
    
    def __init__(self, strategy: Strategy, benchmarks: List[Benchmark], data: pd.DataFrame):
        """
        Initialize the backtest runner.
        
        Args:
            strategy (Strategy): Trading strategy to test
            benchmarks (List[Benchmark]): List of benchmark strategies
            data (pd.DataFrame): Price data for backtesting
        """
        self.strategy = strategy
        self.benchmarks = benchmarks
        self.data = self._validate_data(data)
        
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the input data.
        
        Args:
            data (pd.DataFrame): Price data to validate
            
        Returns:
            pd.DataFrame: Validated data
            
        Raises:
            ValueError: If data is invalid
        """
        if data.empty:
            raise ValueError("Data cannot be empty")
            
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
            
        return data
    
    def _calculate_metrics(self, returns: pd.Series, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            returns (pd.Series): Series of returns
            signals (pd.DataFrame): DataFrame of trading signals
            
        Returns:
            Dict[str, Any]: Dictionary of performance metrics
        """
        total_return = (returns.iloc[-1] - returns.iloc[0]) / returns.iloc[0]
        
        # Calculate win rate
        trades = signals['buy'].sum()
        if trades > 0:
            winning_trades = (returns[signals['buy']] > 0).sum()
            win_rate = winning_trades / trades
        else:
            win_rate = 0.0
            
        # Calculate max drawdown
        cumulative_returns = (1 + returns.pct_change()).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': trades,
            'max_drawdown': max_drawdown
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Returns:
            Dict[str, Any]: Dictionary containing strategy returns, benchmark returns,
                           and performance metrics
        """
        # Generate signals
        signals = self.strategy.generate_signals(self.data)
        
        # Calculate strategy returns
        strategy_returns = self.data['Close'].pct_change() * signals['buy'].shift(1)
        strategy_returns = (1 + strategy_returns).cumprod()
        
        # Calculate benchmark returns
        benchmark_returns = {}
        for benchmark in self.benchmarks:
            benchmark_returns[benchmark.__class__.__name__] = benchmark.calculate_returns(
                self.data, strategy_returns.iloc[0]
            )
            
        # Calculate metrics
        metrics = self._calculate_metrics(strategy_returns, signals)
        
        return {
            'strategy_returns': strategy_returns,
            'benchmark_returns': benchmark_returns,
            'metrics': metrics
        } 