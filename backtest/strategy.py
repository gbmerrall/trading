from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd

from .validation import validate_dataframe, validate_integer, validate_price_data, ValidationError
from .config import get_strategy_config, ValidationLimits


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    All strategies must implement:
    - generate_signals(data): Generate buy/sell signals from price data
    - get_parameters(): Return current strategy parameters
    - set_parameters(params): Update strategy parameters dynamically
    - warmup_period (property): Minimum bars needed before signals are valid
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy and sell signals from price data.

        Args:
            data: DataFrame with OHLCV data and DatetimeIndex

        Returns:
            DataFrame with 'buy' and 'sell' boolean columns, same index as input.
            May include optional columns ('stop_loss', 'position_size').

        Raises:
            ValidationError: If data validation fails
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Return current strategy parameters.

        Returns:
            Dictionary mapping parameter names to their current values.
            Required for Walk-Forward Analysis optimization.
        """
        pass

    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters dynamically.

        Args:
            params: Dictionary of parameter names to new values

        Raises:
            ValidationError: If parameters are invalid
        """
        pass

    @property
    @abstractmethod
    def warmup_period(self) -> int:
        """
        Minimum number of bars needed before signals are valid.

        Returns:
            Integer representing the warmup period (e.g., for RSI(14), returns 14).
            Used by Walk-Forward Analysis to discard early signals.
        """
        pass


class ConsecutiveDaysStrategy(BaseStrategy):
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

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return current strategy parameters.

        Returns:
            Dictionary with 'consecutive_days' parameter
        """
        return {"consecutive_days": self.consecutive_days}

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters dynamically.

        Args:
            params: Dictionary containing 'consecutive_days' key

        Raises:
            ValidationError: If consecutive_days is invalid
        """
        if "consecutive_days" in params:
            consecutive_days = params["consecutive_days"]
            validate_integer(
                consecutive_days,
                "consecutive_days",
                min_value=ValidationLimits.MIN_CONSECUTIVE_DAYS,
                max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
            )
            self.consecutive_days = consecutive_days

    @property
    def warmup_period(self) -> int:
        """
        Minimum bars needed before signals are valid.

        Returns:
            The consecutive_days parameter, as that's the rolling window size
        """
        return self.consecutive_days

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