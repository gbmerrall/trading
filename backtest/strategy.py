import pandas as pd
from .validation import validate_dataframe, validate_price_data, validate_integer, ValidationError
from .config import get_strategy_config, ValidationLimits


class ConsecutiveDaysStrategy:
    """
    A strategy that generates a buy signal after a number of consecutive
    down days and a sell signal after a number of consecutive up days.
    """
    
    def __init__(self, consecutive_days: int = None):
        """
        Initialize the strategy.
        
        Args:
            consecutive_days: The number of consecutive days for the signal.
                             If None, uses configuration default.
        
        Raises:
            ValidationError: If consecutive_days is invalid
        """
        if consecutive_days is None:
            consecutive_days = get_strategy_config().consecutive_days
            
        validate_integer(
            consecutive_days, 
            "consecutive_days", 
            min_value=ValidationLimits.MIN_CONSECUTIVE_DAYS, 
            max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
        )
        self.consecutive_days = consecutive_days

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy and sell signals based on consecutive price movements.
        
        Args:
            data: DataFrame with OHLCV data, must contain 'Close' column
                  with DatetimeIndex and at least consecutive_days rows
                  
        Returns:
            DataFrame with 'buy' and 'sell' boolean columns, same index as input
            
        Raises:
            ValidationError: If data validation fails
        """
        config = get_strategy_config()
        
        # Comprehensive input validation
        min_rows = self.consecutive_days if config.require_minimum_data else 1
        validate_dataframe(
            data, 
            required_columns=['Close'], 
            min_rows=min_rows
        )
        validate_price_data(data, 'Close')
        
        # Check for sufficient non-null data
        close_prices = data['Close']
        non_null_count = close_prices.count()
        if config.require_minimum_data and non_null_count < self.consecutive_days:
            raise ValidationError(
                f"Need at least {self.consecutive_days} non-null price values, "
                f"got {non_null_count}"
            )
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index, dtype=bool)
        signals['buy'] = False
        signals['sell'] = False

        try:
            # Calculate price changes
            price_change = close_prices.diff()

            # Buy signal: consecutive_days of down days
            down_days = price_change < 0
            consecutive_down = down_days.rolling(
                window=self.consecutive_days, 
                min_periods=self.consecutive_days
            ).sum() == self.consecutive_days
            signals['buy'] = consecutive_down

            # Sell signal: consecutive_days of up days  
            up_days = price_change > 0
            consecutive_up = up_days.rolling(
                window=self.consecutive_days,
                min_periods=self.consecutive_days
            ).sum() == self.consecutive_days
            signals['sell'] = consecutive_up

            # Shift signals to trade on the next day's open
            signals = signals.shift(1).fillna(False)
            
        except Exception as e:
            raise ValidationError(f"Error generating signals: {str(e)}") from e
        
        return signals 