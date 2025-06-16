import pandas as pd
import yfinance as yf

class BuyAndHold:
    """A benchmark that simulates a buy-and-hold strategy."""
    
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
        if not isinstance(close, pd.Series):
            close = close.squeeze()

        is_all_nan = close.isnull().all()
        if is_all_nan:
            raise ValueError("All close prices are NaN")
            
        close = close.ffill().bfill()
        returns = close.pct_change(fill_method=None)
        cumulative_returns = (1 + returns).cumprod()
        cumulative_returns.iloc[0] = 1.0
        return start_capital * cumulative_returns

class SPYBuyAndHold:
    """A benchmark that compares performance against a buy-and-hold of SPY."""
    
    def calculate_returns(self, data: pd.DataFrame, start_capital: float) -> pd.Series:
        """
        Calculate SPY buy and hold returns.
        
        Args:
            data (pd.DataFrame): DataFrame with price data, must have 'Close' column
            start_capital (float): Initial capital to invest
            
        Returns:
            pd.Series: Series of cumulative returns
        """
        # TODO: Consider caching this data to avoid repeated downloads
        spy_data = yf.download('SPY', 
                             start=data.index[0],
                             end=data.index[-1],
                             auto_adjust=True,
                             progress=False)
        
        if spy_data.empty or 'Close' not in spy_data.columns:
            raise ValueError("Could not download SPY data or missing 'Close' column")
            
        close = spy_data['Close'].reindex(data.index).ffill().bfill()
        if not isinstance(close, pd.Series):
            close = close.squeeze()

        is_all_nan = close.isnull().all()
        if is_all_nan:
            raise ValueError("All SPY close prices are NaN after alignment")
            
        spy_returns = close.pct_change(fill_method=None)
        cumulative_returns = (1 + spy_returns).cumprod()
        cumulative_returns.iloc[0] = 1.0
        return start_capital * cumulative_returns

class DollarCostAveraging:
    """
    A benchmark that simulates a dollar-cost averaging (DCA) strategy.
    
    This benchmark simulates investing a total of `start_capital` in equal portions
    over a series of regular intervals (`frequency`) throughout the backtest period.
    """
    
    def __init__(self, frequency: str = 'monthly'):
        """
        Initialize the strategy.
        
        Args:
            frequency (str): Investment frequency ('daily', 'weekly', 'monthly')
        """
        self.frequency = frequency
    
    def calculate_returns(self, data: pd.DataFrame, start_capital: float) -> pd.Series:
        """
        Calculate dollar cost averaging returns.
        
        Args:
            data (pd.DataFrame): DataFrame with price data, must have 'Close' column
            start_capital (float): Total capital to invest over the period.
            
        Returns:
            pd.Series: Series of cumulative returns
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        close = data['Close']
        if not isinstance(close, pd.Series):
            close = close.squeeze()
        
        is_all_nan = close.isnull().all()
        if is_all_nan:
            return pd.Series(start_capital, index=data.index)

        schedule = pd.Series(False, index=data.index)
        if self.frequency == 'daily':
            investment_days = data.index
        elif self.frequency == 'weekly':
            investment_days = data.resample('W').first().index
        elif self.frequency == 'monthly':
            investment_days = data.resample('MS').first().index
        else:
            raise ValueError("Invalid frequency. Must be 'daily', 'weekly', or 'monthly'")
        
        valid_investment_days = investment_days.intersection(data.index)
        
        if len(valid_investment_days) == 0:
            return pd.Series(start_capital, index=data.index)
            
        periodic_investment_amount = start_capital / len(valid_investment_days)
        schedule.loc[valid_investment_days] = True

        investments = pd.Series(0.0, index=data.index)
        investments.loc[schedule] = periodic_investment_amount
        
        shares_bought = (investments / close).fillna(0)
        cumulative_shares = shares_bought.cumsum()
        
        cash_spent = investments.cumsum()
        cash = start_capital - cash_spent

        portfolio_value = cumulative_shares * close + cash
        
        return portfolio_value