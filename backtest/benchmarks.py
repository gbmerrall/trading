"""
Benchmark strategies for backtesting comparison.

This module provides various benchmark strategies to compare against custom trading strategies.
All benchmarks inherit from a common base class and use configuration for parameters.
"""

import pandas as pd
import yfinance as yf
from abc import ABC, abstractmethod
from .validation import validate_positive_number, ValidationError
from .config import get_benchmark_config


class BaseBenchmark(ABC):
    """
    Abstract base class for all benchmark strategies.
    
    Provides common functionality for data validation, price preparation,
    and portfolio value calculations.
    """
    
    def _validate_inputs(self, data: pd.DataFrame, start_capital: float) -> None:
        """
        Validate inputs for benchmark calculations.
        
        Args:
            data: Price data DataFrame
            start_capital: Starting capital amount
            
        Raises:
            ValueError: If validation fails (for test compatibility)
        """
        try:
            validate_positive_number(start_capital, "start_capital")
            
            # Check if data is DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Expected pandas DataFrame, got {type(data).__name__}")
            
            # Check for minimum data points
            if len(data) < 1:
                raise ValueError("DataFrame must have at least 1 row")
                
        except ValidationError as e:
            # Convert ValidationError to ValueError for test compatibility
            raise ValueError(str(e))
    
    def _validate_close_column(self, data: pd.DataFrame) -> None:
        """
        Validate that Close column exists and handle specific error message.
        
        Args:
            data: Price data DataFrame
            
        Raises:
            ValueError: If Close column is missing
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
    
    def _prepare_prices(self, data: pd.DataFrame) -> pd.Series:
        """
        Prepare price data for calculations.
        
        Args:
            data: Raw price data
            
        Returns:
            Clean price series
            
        Raises:
            ValueError: If all prices are NaN
        """
        self._validate_close_column(data)
        prices = data['Close'].copy()
        
        # Check for all NaN values before any processing
        if prices.isna().all():
            raise ValueError("All close prices are NaN")
        
        # Forward fill to handle missing values
        prices = prices.ffill()
        
        return prices
    
    def _calculate_portfolio_value(self, prices: pd.Series, start_capital: float) -> pd.Series:
        """
        Calculate portfolio value over time for buy and hold strategy.
        
        Args:
            prices: Price series
            start_capital: Starting capital
            
        Returns:
            Portfolio value series
        """
        # Calculate returns
        returns = prices / prices.iloc[0]
        portfolio_values = start_capital * returns
        
        return portfolio_values
    
    @abstractmethod
    def calculate_returns(self, data: pd.DataFrame, start_capital: float) -> pd.Series:
        """
        Calculate benchmark returns.
        
        Args:
            data: Price data DataFrame
            start_capital: Starting capital amount
            
        Returns:
            Portfolio value series over time
        """
        pass


class BuyAndHold(BaseBenchmark):
    """
    Simple buy and hold benchmark strategy.
    
    Buys the asset at the first available price and holds until the end.
    """
    
    def calculate_returns(self, data: pd.DataFrame, start_capital: float) -> pd.Series:
        """
        Calculate buy and hold returns.
        
        Args:
            data: Price data DataFrame with 'Close' column
            start_capital: Starting capital amount
            
        Returns:
            Portfolio value series over time
        """
        self._validate_inputs(data, start_capital)
        prices = self._prepare_prices(data)
        return self._calculate_portfolio_value(prices, start_capital)


class SPYBuyAndHold(BaseBenchmark):
    """
    S&P 500 buy and hold benchmark using SPY ETF.
    
    Downloads SPY data for the same date range as the input data
    and calculates buy and hold returns.
    """
    
    def calculate_returns(self, data: pd.DataFrame, start_capital: float) -> pd.Series:
        """
        Calculate SPY buy and hold returns.
        
        Args:
            data: Price data DataFrame (used for date range)
            start_capital: Starting capital amount
            
        Returns:
            Portfolio value series over time
        """
        self._validate_inputs(data, start_capital)
        
        # Get date range from input data
        start_date = data.index.min()
        end_date = data.index.max()
        
        # Download SPY data
        config = get_benchmark_config()
        try:
            spy_data = yf.download(
                config.market_symbol, 
                start=start_date, 
                end=end_date + pd.Timedelta(days=1),
                progress=False
            )
        except Exception as e:
            raise ValueError(f"Failed to download SPY data: {str(e)}")
        
        if spy_data.empty:
            raise ValueError("Could not download SPY data")
        
        # Align SPY data with input data dates
        spy_prices = spy_data['Close'].reindex(data.index, method='ffill')
        
        # Check if all prices are NaN after alignment
        if spy_prices.isna().all():
            raise ValueError("All SPY close prices are NaN after alignment")
        
        # Forward fill any remaining NaN values
        spy_prices = spy_prices.ffill()
        
        return self._calculate_portfolio_value(spy_prices, start_capital)


class DollarCostAveraging(BaseBenchmark):
    """
    Dollar Cost Averaging (DCA) benchmark strategy.
    
    Invests a fixed amount at regular intervals (daily, weekly, or monthly).
    """
    
    def __init__(self, frequency: str = 'monthly'):
        """
        Initialize DCA benchmark.
        
        Args:
            frequency: Investment frequency ('daily', 'weekly', or 'monthly')
        """
        config = get_benchmark_config()
        valid_frequencies = config.valid_frequencies
        
        if frequency not in valid_frequencies:
            raise ValueError(f"Invalid frequency: {frequency}. Must be one of {valid_frequencies}")
        
        self.frequency = frequency
    
    def _get_investment_dates(self, data: pd.DataFrame) -> pd.DatetimeIndex:
        """
        Get investment dates based on frequency.
        
        Args:
            data: Price data DataFrame
            
        Returns:
            DatetimeIndex of investment dates
        """
        all_dates = data.index
        
        if self.frequency == 'daily':
            return all_dates
        elif self.frequency == 'weekly':
            # Invest on Mondays (weekday 0)
            return all_dates[all_dates.weekday == 0]
        elif self.frequency == 'monthly':
            # Invest on first day of each month
            return all_dates[all_dates.day == 1]
        else:
            raise ValueError(f"Invalid frequency: {self.frequency}")
    
    def calculate_returns(self, data: pd.DataFrame, start_capital: float) -> pd.Series:
        """
        Calculate dollar cost averaging returns.
        
        Args:
            data: Price data DataFrame with 'Close' column
            start_capital: Starting capital amount
            
        Returns:
            Portfolio value series over time
        """
        self._validate_inputs(data, start_capital)
        
        # Handle all NaN prices case
        if 'Close' in data.columns and data['Close'].isna().all():
            return pd.Series(start_capital, index=data.index)
        
        prices = self._prepare_prices(data)
        investment_dates = self._get_investment_dates(data)
        
        # If no investment dates, return start capital
        if len(investment_dates) == 0:
            return pd.Series(start_capital, index=data.index)
        
        # Calculate investment amount per period
        investment_amount = start_capital / len(investment_dates)
        
        # Track shares and cash over time
        shares = 0.0
        cash = start_capital
        portfolio_values = []
        
        for date in data.index:
            if date in investment_dates and not pd.isna(prices[date]):
                # Invest cash
                shares_to_buy = investment_amount / prices[date]
                shares += shares_to_buy
                cash -= investment_amount
            
            # Calculate portfolio value
            portfolio_value = shares * prices[date] + cash
            portfolio_values.append(portfolio_value)
        
        return pd.Series(portfolio_values, index=data.index)


# Convenience function for creating common benchmark combinations
def create_standard_benchmarks(spy_symbol: str = None, dca_frequency: str = None) -> list[BaseBenchmark]:
    """
    Create a standard set of benchmarks for comparison.
    
    Args:
        spy_symbol: Symbol for market benchmark. If None, uses config default.
        dca_frequency: Frequency for dollar cost averaging. If None, uses config default.
        
    Returns:
        List of benchmark instances
    """
    config = get_benchmark_config()
    
    if spy_symbol is None:
        spy_symbol = config.market_symbol
    if dca_frequency is None:
        dca_frequency = config.dca_frequency
    
    return [
        BuyAndHold(),
        SPYBuyAndHold(),
        DollarCostAveraging(frequency=dca_frequency)
    ]