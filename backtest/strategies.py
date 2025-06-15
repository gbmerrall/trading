import pandas as pd
from .interfaces import Strategy

class ConsecutiveDaysStrategy(Strategy):
    """Strategy that generates signals based on consecutive up and down days."""
    
    def __init__(self, up_days: int = 3, down_days: int = 3):
        """
        Initialize the strategy.
        
        Args:
            up_days (int): Number of consecutive up days required for sell signal
            down_days (int): Number of consecutive down days required for buy signal
        """
        self.up_days = up_days
        self.down_days = down_days
    
    def _detect_consecutive_down_days(self, prices: pd.Series) -> pd.Series:
        """
        Detect consecutive down days in a price series.
        
        Args:
            prices (pd.Series): Series of closing prices
            
        Returns:
            pd.Series: Boolean series with True values indicating buy signals
        """
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()
        if len(prices) < self.down_days or bool(prices.isna().any()):
            return pd.Series(False, index=prices.index)
        
        daily_changes = prices.pct_change(fill_method=None)
        signals = pd.Series(False, index=prices.index)
        
        for i in range(self.down_days - 1, len(prices)):
            if (daily_changes.iloc[i-self.down_days+1:i+1] < 0).all():
                signals.iloc[i] = True
                
        return signals
    
    def _detect_consecutive_up_days(self, prices: pd.Series) -> pd.Series:
        """
        Detect consecutive up days in a price series.
        
        Args:
            prices (pd.Series): Series of closing prices
            
        Returns:
            pd.Series: Boolean series with True values indicating sell signals
        """
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()
        if len(prices) < self.up_days or bool(prices.isna().any()):
            return pd.Series(False, index=prices.index)
        
        daily_changes = prices.pct_change(fill_method=None)
        signals = pd.Series(False, index=prices.index)
        
        for i in range(self.up_days - 1, len(prices)):
            if (daily_changes.iloc[i-self.up_days+1:i+1] > 0).all():
                signals.iloc[i] = True
                
        return signals
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from price data.
        
        Args:
            data (pd.DataFrame): DataFrame with price data, must have 'Close' column
            
        Returns:
            pd.DataFrame: DataFrame with 'buy' and 'sell' boolean columns
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
            
        prices = data['Close']
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()
        buy_signals = self._detect_consecutive_down_days(prices)
        sell_signals = self._detect_consecutive_up_days(prices)
        
        return pd.DataFrame({
            'buy': buy_signals,
            'sell': sell_signals
        }, index=data.index) 