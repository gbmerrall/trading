import pandas as pd
import yfinance as yf
import numpy as np
from .interfaces import Benchmark

class BuyAndHold(Benchmark):
    """Simple buy and hold benchmark strategy."""
    
    def calculate_returns(self, data: pd.DataFrame, start_capital: float) -> pd.Series:
        """
        Calculate buy and hold returns.
        
        Args:
            data (pd.DataFrame): DataFrame with price data, must have 'Close' column
            start_capital (float): Initial capital to invest
            
        Returns:
            pd.Series: Series of cumulative returns
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        close = data['Close']
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        if close.isnull().all():
            raise ValueError("All close prices are NaN")
        close = close.ffill().bfill()
        returns = close.pct_change(fill_method=None)
        cumulative_returns = (1 + returns).cumprod()
        cumulative_returns.iloc[0] = 1.0
        return start_capital * cumulative_returns

class SPYBuyAndHold(Benchmark):
    """SPY buy and hold benchmark strategy."""
    
    def calculate_returns(self, data: pd.DataFrame, start_capital: float) -> pd.Series:
        """
        Calculate SPY buy and hold returns.
        
        Args:
            data (pd.DataFrame): DataFrame with price data, must have 'Close' column
            start_capital (float): Initial capital to invest
            
        Returns:
            pd.Series: Series of cumulative returns
        """
        spy_data = yf.download('SPY', 
                             start=data.index[0],
                             end=data.index[-1],
                             auto_adjust=True,
                             progress=False)
        
        if spy_data.empty or 'Close' not in spy_data.columns:
            raise ValueError("Could not download SPY data or missing 'Close' column")
            
        close = spy_data['Close'].reindex(data.index).ffill().bfill()
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        if close.isnull().all():
            raise ValueError("All SPY close prices are NaN after alignment")
            
        spy_returns = close.pct_change(fill_method=None)
        cumulative_returns = (1 + spy_returns).cumprod()
        cumulative_returns.iloc[0] = 1.0
        return start_capital * cumulative_returns

class DollarCostAveraging(Benchmark):
    """Dollar cost averaging benchmark strategy."""
    
    def __init__(self, frequency: str = 'monthly', amount: float = 1000.0):
        """
        Initialize the strategy.
        
        Args:
            frequency (str): Investment frequency ('daily', 'weekly', 'monthly')
            amount (float): Amount to invest each period
        """
        self.frequency = frequency
        self.amount = amount
    
    def calculate_returns(self, data: pd.DataFrame, start_capital: float) -> pd.Series:
        """
        Calculate dollar cost averaging returns.
        
        Args:
            data (pd.DataFrame): DataFrame with price data, must have 'Close' column
            start_capital (float): Initial capital to invest
            
        Returns:
            pd.Series: Series of cumulative returns
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        close = data['Close']
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        if close.isnull().all():
            raise ValueError("All close prices are NaN")
        close = close.ffill().bfill()
        close = close.replace([np.inf, -np.inf], np.nan)
        if self.frequency == 'daily':
            schedule = pd.Series(True, index=data.index)
        elif self.frequency == 'weekly':
            schedule = pd.Series(False, index=data.index)
            schedule[data.index.weekday == 0] = True  # Invest on Mondays
        elif self.frequency == 'monthly':
            schedule = pd.Series(False, index=data.index)
            schedule[data.index.day == 1] = True  # Invest on first day of month
        else:
            raise ValueError("Invalid frequency. Must be 'daily', 'weekly', or 'monthly'")
            
        shares = pd.Series(0.0, index=data.index)
        shares.loc[schedule] = self.amount / close.loc[schedule].replace(0, np.nan)
        shares = shares.fillna(0)
        
        cumulative_shares = shares.cumsum()
        portfolio_value = cumulative_shares * close
        
        return portfolio_value 