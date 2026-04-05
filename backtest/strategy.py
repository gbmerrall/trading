from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd
import ta

from .validation import (
    validate_dataframe,
    validate_integer,
    validate_positive_number,
    validate_price_data,
    ValidationError
)
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
        

        return signals


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    A strategy that buys when a short-term moving average crosses above a long-term
    moving average (golden cross) and sells on the reverse crossover (death cross).
    """

    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Initialize the Moving Average Crossover strategy.

        Args:
            short_window: Period for short-term moving average (default 20)
            long_window: Period for long-term moving average (default 50)

        Raises:
            ValidationError: If parameters are invalid
        """
        validate_integer(
            short_window,
            "short_window",
            min_value=1,
            max_value=ValidationLimits.MAX_INDICATOR_PERIOD
        )
        validate_integer(
            long_window,
            "long_window",
            min_value=1,
            max_value=ValidationLimits.MAX_INDICATOR_PERIOD
        )

        if short_window >= long_window:
            raise ValidationError(
                f"short_window ({short_window}) must be less than long_window ({long_window})"
            )

        self.short_window = short_window
        self.long_window = long_window

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return current strategy parameters.

        Returns:
            Dictionary with 'short_window' and 'long_window' parameters
        """
        return {"short_window": self.short_window, "long_window": self.long_window}

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters dynamically.

        Args:
            params: Dictionary containing 'short_window' and/or 'long_window' keys

        Raises:
            ValidationError: If parameters are invalid
        """
        if "short_window" in params:
            short_window = params["short_window"]
            validate_integer(
                short_window,
                "short_window",
                min_value=1,
                max_value=ValidationLimits.MAX_INDICATOR_PERIOD
            )
            self.short_window = short_window

        if "long_window" in params:
            long_window = params["long_window"]
            validate_integer(
                long_window,
                "long_window",
                min_value=1,
                max_value=ValidationLimits.MAX_INDICATOR_PERIOD
            )
            self.long_window = long_window

        # Re-validate relationship after updates
        if self.short_window >= self.long_window:
            raise ValidationError(
                f"short_window ({self.short_window}) must be less than "
                f"long_window ({self.long_window})"
            )

    @property
    def warmup_period(self) -> int:
        """
        Minimum bars needed before signals are valid.

        Returns:
            The long_window parameter, as that's the longest lookback period
        """
        return self.long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy and sell signals based on moving average crossovers.

        Args:
            data: DataFrame with OHLCV data, must contain 'Close' column
                  with DatetimeIndex and at least long_window rows

        Returns:
            DataFrame with 'buy' and 'sell' boolean columns, same index as input

        Raises:
            ValidationError: If data validation fails
        """
        # Comprehensive input validation
        validate_dataframe(
            data,
            required_columns=['Close'],
            min_rows=self.long_window
        )
        validate_price_data(data, 'Close')

        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index, dtype=bool)
        signals['buy'] = False
        signals['sell'] = False

        # Calculate moving averages using simple rolling mean
        close_prices = data['Close']
        short_ma = close_prices.rolling(window=self.short_window, min_periods=self.short_window).mean()
        long_ma = close_prices.rolling(window=self.long_window, min_periods=self.long_window).mean()

        # Detect crossovers
        # Golden cross: short MA crosses above long MA (buy signal)
        # Previous day: short_ma <= long_ma, Current day: short_ma > long_ma
        golden_cross = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        signals['buy'] = golden_cross

        # Death cross: short MA crosses below long MA (sell signal)
        # Previous day: short_ma >= long_ma, Current day: short_ma < long_ma
        death_cross = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        signals['sell'] = death_cross

        # Shift signals to trade on the next day's open
        signals = signals.shift(1).fillna(False)


        return signals


class RSIStrategy(BaseStrategy):
    """
    A strategy that buys when RSI drops below a lower threshold (oversold)
    and sells when RSI exceeds an upper threshold (overbought).
    """

    def __init__(self, period: int = 14, lower_bound: float = 30, upper_bound: float = 70):
        """
        Initialize the RSI strategy.

        Args:
            period: RSI calculation period (default 14)
            lower_bound: RSI lower threshold for buy signals (default 30)
            upper_bound: RSI upper threshold for sell signals (default 70)

        Raises:
            ValidationError: If parameters are invalid
        """
        validate_integer(
            period,
            "period",
            min_value=1,
            max_value=ValidationLimits.MAX_INDICATOR_PERIOD
        )
        validate_positive_number(
            lower_bound,
            "lower_bound",
            allow_zero=True,
            max_value=100
        )
        validate_positive_number(
            upper_bound,
            "upper_bound",
            allow_zero=True,
            max_value=100
        )

        if lower_bound >= upper_bound:
            raise ValidationError(
                f"lower_bound ({lower_bound}) must be less than upper_bound ({upper_bound})"
            )

        self.period = period
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return current strategy parameters.

        Returns:
            Dictionary with 'period', 'lower_bound', and 'upper_bound' parameters
        """
        return {
            "period": self.period,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters dynamically.

        Args:
            params: Dictionary containing parameter keys to update

        Raises:
            ValidationError: If parameters are invalid
        """
        if "period" in params:
            period = params["period"]
            validate_integer(
                period,
                "period",
                min_value=1,
                max_value=ValidationLimits.MAX_INDICATOR_PERIOD
            )
            self.period = period

        if "lower_bound" in params:
            lower_bound = params["lower_bound"]
            validate_positive_number(
                lower_bound,
                "lower_bound",
                allow_zero=True,
                max_value=100
            )
            self.lower_bound = lower_bound

        if "upper_bound" in params:
            upper_bound = params["upper_bound"]
            validate_positive_number(
                upper_bound,
                "upper_bound",
                allow_zero=True,
                max_value=100
            )
            self.upper_bound = upper_bound

        # Re-validate relationship after updates
        if self.lower_bound >= self.upper_bound:
            raise ValidationError(
                f"lower_bound ({self.lower_bound}) must be less than "
                f"upper_bound ({self.upper_bound})"
            )

    @property
    def warmup_period(self) -> int:
        """
        Minimum bars needed before signals are valid.

        Returns:
            The period parameter
        """
        return self.period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy and sell signals based on RSI levels.

        Args:
            data: DataFrame with OHLCV data, must contain 'Close' column
                  with DatetimeIndex and at least period rows

        Returns:
            DataFrame with 'buy' and 'sell' boolean columns, same index as input

        Raises:
            ValidationError: If data validation fails
        """
        # Comprehensive input validation
        validate_dataframe(
            data,
            required_columns=['Close'],
            min_rows=self.period
        )
        validate_price_data(data, 'Close')

        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index, dtype=bool)
        signals['buy'] = False
        signals['sell'] = False

        # Calculate RSI using ta library
        rsi = ta.momentum.RSIIndicator(close=data['Close'], window=self.period).rsi()

        # Buy signal: RSI below lower_bound (oversold)
        signals['buy'] = rsi < self.lower_bound

        # Sell signal: RSI above upper_bound (overbought)
        signals['sell'] = rsi > self.upper_bound

        # Shift signals to trade on the next day's open
        signals = signals.shift(1).fillna(False)


        return signals


class MACDStrategy(BaseStrategy):
    """
    A strategy that buys when the MACD line crosses above the signal line
    (bullish crossover) and sells when MACD crosses below signal (bearish crossover).
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Initialize the MACD strategy.

        Args:
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)

        Raises:
            ValidationError: If parameters are invalid
        """
        validate_integer(
            fast,
            "fast",
            min_value=1,
            max_value=ValidationLimits.MAX_INDICATOR_PERIOD
        )
        validate_integer(
            slow,
            "slow",
            min_value=1,
            max_value=ValidationLimits.MAX_INDICATOR_PERIOD
        )
        validate_integer(
            signal,
            "signal",
            min_value=1,
            max_value=ValidationLimits.MAX_INDICATOR_PERIOD
        )

        if fast >= slow:
            raise ValidationError(
                f"fast ({fast}) must be less than slow ({slow})"
            )

        self.fast = fast
        self.slow = slow
        self.signal = signal

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return current strategy parameters.

        Returns:
            Dictionary with 'fast', 'slow', and 'signal' parameters
        """
        return {"fast": self.fast, "slow": self.slow, "signal": self.signal}

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters dynamically.

        Args:
            params: Dictionary containing parameter keys to update

        Raises:
            ValidationError: If parameters are invalid
        """
        if "fast" in params:
            fast = params["fast"]
            validate_integer(
                fast,
                "fast",
                min_value=1,
                max_value=ValidationLimits.MAX_INDICATOR_PERIOD
            )
            self.fast = fast

        if "slow" in params:
            slow = params["slow"]
            validate_integer(
                slow,
                "slow",
                min_value=1,
                max_value=ValidationLimits.MAX_INDICATOR_PERIOD
            )
            self.slow = slow

        if "signal" in params:
            signal = params["signal"]
            validate_integer(
                signal,
                "signal",
                min_value=1,
                max_value=ValidationLimits.MAX_INDICATOR_PERIOD
            )
            self.signal = signal

        # Re-validate relationship after updates
        if self.fast >= self.slow:
            raise ValidationError(
                f"fast ({self.fast}) must be less than slow ({self.slow})"
            )

    @property
    def warmup_period(self) -> int:
        """
        Minimum bars needed before signals are valid.

        Returns:
            slow + signal (longest lookback period for MACD calculation)
        """
        return self.slow + self.signal

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy and sell signals based on MACD crossovers.

        Args:
            data: DataFrame with OHLCV data, must contain 'Close' column
                  with DatetimeIndex and at least (slow + signal) rows

        Returns:
            DataFrame with 'buy' and 'sell' boolean columns, same index as input

        Raises:
            ValidationError: If data validation fails
        """
        # Comprehensive input validation
        validate_dataframe(
            data,
            required_columns=['Close'],
            min_rows=self.warmup_period
        )
        validate_price_data(data, 'Close')

        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index, dtype=bool)
        signals['buy'] = False
        signals['sell'] = False

        # Calculate MACD using ta library
        macd_indicator = ta.trend.MACD(
            close=data['Close'],
            window_fast=self.fast,
            window_slow=self.slow,
            window_sign=self.signal
        )
        macd_line = macd_indicator.macd()
        signal_line = macd_indicator.macd_signal()

        # Buy signal: MACD crosses above signal line (bullish crossover)
        # Previous day: macd <= signal, Current day: macd > signal
        bullish_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        signals['buy'] = bullish_cross

        # Sell signal: MACD crosses below signal line (bearish crossover)
        # Previous day: macd >= signal, Current day: macd < signal
        bearish_cross = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        signals['sell'] = bearish_cross

        # Shift signals to trade on the next day's open
        signals = signals.shift(1).fillna(False)


        return signals


class BollingerBandsStrategy(BaseStrategy):
    """
    A strategy that buys when price touches/breaks the lower Bollinger Band
    and sells when price touches/breaks the upper Bollinger Band (mean reversion).
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize the Bollinger Bands strategy.

        Args:
            period: Moving average period (default 20)
            std_dev: Standard deviation multiplier for bands (default 2.0)

        Raises:
            ValidationError: If parameters are invalid
        """
        validate_integer(
            period,
            "period",
            min_value=1,
            max_value=ValidationLimits.MAX_INDICATOR_PERIOD
        )
        validate_positive_number(
            std_dev,
            "std_dev",
            allow_zero=False,
            max_value=10.0
        )

        self.period = period
        self.std_dev = std_dev

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return current strategy parameters.

        Returns:
            Dictionary with 'period' and 'std_dev' parameters
        """
        return {"period": self.period, "std_dev": self.std_dev}

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters dynamically.

        Args:
            params: Dictionary containing parameter keys to update

        Raises:
            ValidationError: If parameters are invalid
        """
        if "period" in params:
            period = params["period"]
            validate_integer(
                period,
                "period",
                min_value=1,
                max_value=ValidationLimits.MAX_INDICATOR_PERIOD
            )
            self.period = period

        if "std_dev" in params:
            std_dev = params["std_dev"]
            validate_positive_number(
                std_dev,
                "std_dev",
                allow_zero=False,
                max_value=10.0
            )
            self.std_dev = std_dev

    @property
    def warmup_period(self) -> int:
        """
        Minimum bars needed before signals are valid.

        Returns:
            The period parameter
        """
        return self.period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy and sell signals based on Bollinger Bands.

        Args:
            data: DataFrame with OHLCV data, must contain 'Close' column
                  with DatetimeIndex and at least period rows

        Returns:
            DataFrame with 'buy' and 'sell' boolean columns, same index as input

        Raises:
            ValidationError: If data validation fails
        """
        # Comprehensive input validation
        validate_dataframe(
            data,
            required_columns=['Close'],
            min_rows=self.period
        )
        validate_price_data(data, 'Close')

        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index, dtype=bool)
        signals['buy'] = False
        signals['sell'] = False

        # Calculate Bollinger Bands using ta library
        bb_indicator = ta.volatility.BollingerBands(
            close=data['Close'],
            window=self.period,
            window_dev=self.std_dev
        )
        lower_band = bb_indicator.bollinger_lband()
        upper_band = bb_indicator.bollinger_hband()
        close_prices = data['Close']

        # Buy signal: price touches or breaks below lower band (mean reversion)
        signals['buy'] = close_prices <= lower_band

        # Sell signal: price touches or breaks above upper band (mean reversion)
        signals['sell'] = close_prices >= upper_band

        # Shift signals to trade on the next day's open
        signals = signals.shift(1).fillna(False)


        return signals


class ParabolicSARStrategy(BaseStrategy):
    """
    A strategy that buys when price crosses above the Parabolic SAR
    and sells when price crosses below the Parabolic SAR (trend following).
    """

    def __init__(self, af: float = 0.02, max_af: float = 0.2):
        """
        Initialize the Parabolic SAR strategy.

        Args:
            af: Acceleration factor (default 0.02)
            max_af: Maximum acceleration factor (default 0.2)

        Raises:
            ValidationError: If parameters are invalid
        """
        validate_positive_number(
            af,
            "af",
            allow_zero=False,
            max_value=1.0
        )
        validate_positive_number(
            max_af,
            "max_af",
            allow_zero=False,
            max_value=1.0
        )

        if af > max_af:
            raise ValidationError(
                f"af ({af}) must be less than or equal to max_af ({max_af})"
            )

        self.af = af
        self.max_af = max_af

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return current strategy parameters.

        Returns:
            Dictionary with 'af' and 'max_af' parameters
        """
        return {"af": self.af, "max_af": self.max_af}

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters dynamically.

        Args:
            params: Dictionary containing parameter keys to update

        Raises:
            ValidationError: If parameters are invalid
        """
        if "af" in params:
            af = params["af"]
            validate_positive_number(
                af,
                "af",
                allow_zero=False,
                max_value=1.0
            )
            self.af = af

        if "max_af" in params:
            max_af = params["max_af"]
            validate_positive_number(
                max_af,
                "max_af",
                allow_zero=False,
                max_value=1.0
            )
            self.max_af = max_af

        # Re-validate relationship after updates
        if self.af > self.max_af:
            raise ValidationError(
                f"af ({self.af}) must be less than or equal to max_af ({self.max_af})"
            )

    @property
    def warmup_period(self) -> int:
        """
        Minimum bars needed before signals are valid.

        Returns:
            1 (minimal warmup for Parabolic SAR)
        """
        return 1

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy and sell signals based on Parabolic SAR crossovers.

        Args:
            data: DataFrame with OHLCV data, must contain 'Close', 'High', 'Low' columns
                  with DatetimeIndex and at least 1 row

        Returns:
            DataFrame with 'buy' and 'sell' boolean columns, same index as input

        Raises:
            ValidationError: If data validation fails
        """
        # Comprehensive input validation
        validate_dataframe(
            data,
            required_columns=['Close', 'High', 'Low'],
            min_rows=self.warmup_period
        )
        validate_price_data(data, 'Close')
        validate_price_data(data, 'High')
        validate_price_data(data, 'Low')

        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index, dtype=bool)
        signals['buy'] = False
        signals['sell'] = False

        # Calculate Parabolic SAR using ta library
        psar_indicator = ta.trend.PSARIndicator(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            step=self.af,
            max_step=self.max_af
        )
        psar = psar_indicator.psar()
        close_prices = data['Close']

        # Align the psar series with the data index (ta library may alter index)
        psar = psar.reindex(data.index)

        # Buy signal: price crosses above SAR (SAR switches from above to below price)
        prev_below = close_prices.shift(1) < psar.shift(1)
        curr_above = close_prices > psar
        signals['buy'] = prev_below & curr_above

        # Sell signal: price crosses below SAR (SAR switches from below to above price)
        prev_above = close_prices.shift(1) > psar.shift(1)
        curr_below = close_prices < psar
        signals['sell'] = prev_above & curr_below

        # Shift signals to trade on the next day's open
        signals = signals.shift(1).fillna(False)


        return signals


class BreakoutStrategy(BaseStrategy):
    """
    A strategy that buys when price exceeds the N-day high (breakout)
    and sells when price breaks below the N-day low (breakdown).
    """

    def __init__(self, lookback_period: int = 20):
        """
        Initialize the Breakout strategy.

        Args:
            lookback_period: Number of days to look back for high/low (default 20)

        Raises:
            ValidationError: If parameters are invalid
        """
        validate_integer(
            lookback_period,
            "lookback_period",
            min_value=1,
            max_value=ValidationLimits.MAX_INDICATOR_PERIOD
        )

        self.lookback_period = lookback_period

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return current strategy parameters.

        Returns:
            Dictionary with 'lookback_period' parameter
        """
        return {"lookback_period": self.lookback_period}

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters dynamically.

        Args:
            params: Dictionary containing 'lookback_period' key

        Raises:
            ValidationError: If parameters are invalid
        """
        if "lookback_period" in params:
            lookback_period = params["lookback_period"]
            validate_integer(
                lookback_period,
                "lookback_period",
                min_value=1,
                max_value=ValidationLimits.MAX_INDICATOR_PERIOD
            )
            self.lookback_period = lookback_period

    @property
    def warmup_period(self) -> int:
        """
        Minimum bars needed before signals are valid.

        Returns:
            The lookback_period parameter
        """
        return self.lookback_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy and sell signals based on price breakouts.

        Args:
            data: DataFrame with OHLCV data, must contain 'Close' column
                  with DatetimeIndex and at least lookback_period rows

        Returns:
            DataFrame with 'buy' and 'sell' boolean columns, same index as input

        Raises:
            ValidationError: If data validation fails
        """
        # Comprehensive input validation
        validate_dataframe(
            data,
            required_columns=['Close'],
            min_rows=self.lookback_period
        )
        validate_price_data(data, 'Close')

        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index, dtype=bool)
        signals['buy'] = False
        signals['sell'] = False

        close_prices = data['Close']

        # Calculate rolling high and low over lookback period
        rolling_high = close_prices.rolling(
            window=self.lookback_period,
            min_periods=self.lookback_period
        ).max()
        rolling_low = close_prices.rolling(
            window=self.lookback_period,
            min_periods=self.lookback_period
        ).min()

        # Buy signal: price exceeds the N-day high (breakout)
        # Compare current price to previous period's high
        signals['buy'] = close_prices > rolling_high.shift(1)

        # Sell signal: price breaks below the N-day low (breakdown)
        # Compare current price to previous period's low
        signals['sell'] = close_prices < rolling_low.shift(1)

        # Shift signals to trade on the next day's open
        signals = signals.shift(1).fillna(False)


        return signals


class GapStrategy(BaseStrategy):
    """
    A strategy that trades overnight gaps. Buys on gap-down (expecting fill)
    and sells on gap-up (expecting reversal or fade).
    """

    def __init__(self, min_gap_pct: float = 0.02):
        """
        Initialize the Gap strategy.

        Args:
            min_gap_pct: Minimum gap percentage to trigger signal (default 0.02 = 2%)

        Raises:
            ValidationError: If parameters are invalid
        """
        validate_positive_number(
            min_gap_pct,
            "min_gap_pct",
            allow_zero=False,
            max_value=1.0
        )

        self.min_gap_pct = min_gap_pct

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return current strategy parameters.

        Returns:
            Dictionary with 'min_gap_pct' parameter
        """
        return {"min_gap_pct": self.min_gap_pct}

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters dynamically.

        Args:
            params: Dictionary containing 'min_gap_pct' key

        Raises:
            ValidationError: If parameters are invalid
        """
        if "min_gap_pct" in params:
            min_gap_pct = params["min_gap_pct"]
            validate_positive_number(
                min_gap_pct,
                "min_gap_pct",
                allow_zero=False,
                max_value=1.0
            )
            self.min_gap_pct = min_gap_pct

    @property
    def warmup_period(self) -> int:
        """
        Minimum bars needed before signals are valid.

        Returns:
            1 (minimal warmup for gap detection)
        """
        return 1

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy and sell signals based on overnight gaps.

        Args:
            data: DataFrame with OHLCV data, must contain 'Close' and 'Open' columns
                  with DatetimeIndex and at least 1 row

        Returns:
            DataFrame with 'buy' and 'sell' boolean columns, same index as input

        Raises:
            ValidationError: If data validation fails
        """
        # Comprehensive input validation
        validate_dataframe(
            data,
            required_columns=['Close', 'Open'],
            min_rows=self.warmup_period
        )
        validate_price_data(data, 'Close')
        validate_price_data(data, 'Open')

        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index, dtype=bool)
        signals['buy'] = False
        signals['sell'] = False

        close_prices = data['Close']
        open_prices = data['Open']

        # Calculate gap: (open - previous close) / previous close
        prev_close = close_prices.shift(1)
        gap_pct = (open_prices - prev_close) / prev_close

        # Buy signal: gap down (negative gap) exceeding threshold
        # Expecting the gap to fill (price to rise back)
        signals['buy'] = gap_pct < -self.min_gap_pct

        # Sell signal: gap up (positive gap) exceeding threshold
        # Expecting reversal or fade
        signals['sell'] = gap_pct > self.min_gap_pct

        # Shift signals to trade on the next day's open
        signals = signals.shift(1).fillna(False)


        return signals


class FibonacciRetracementStrategy(BaseStrategy):
    """
    A strategy that identifies swing highs/lows and trades Fibonacci retracement levels.
    Buys near support (61.8% retracement) and sells near resistance (38.2% retracement).
    """

    def __init__(self, swing_lookback: int = 20, retracement_levels: list = None):
        """
        Initialize the Fibonacci Retracement strategy.

        Args:
            swing_lookback: Number of days to look back for swing high/low (default 20)
            retracement_levels: List of Fibonacci levels between 0 and 1 (default [0.382, 0.5, 0.618])

        Raises:
            ValidationError: If parameters are invalid
        """
        if retracement_levels is None:
            retracement_levels = [0.382, 0.5, 0.618]

        validate_integer(
            swing_lookback,
            "swing_lookback",
            min_value=1,
            max_value=ValidationLimits.MAX_INDICATOR_PERIOD
        )

        if not isinstance(retracement_levels, list) or len(retracement_levels) == 0:
            raise ValidationError("retracement_levels must be a non-empty list")

        for level in retracement_levels:
            if not isinstance(level, (int, float)) or level < 0 or level > 1:
                raise ValidationError(
                    f"retracement_levels must contain numbers between 0 and 1, got {level}"
                )

        self.swing_lookback = swing_lookback
        self.retracement_levels = retracement_levels

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return current strategy parameters.

        Returns:
            Dictionary with 'swing_lookback' and 'retracement_levels' parameters
        """
        return {
            "swing_lookback": self.swing_lookback,
            "retracement_levels": self.retracement_levels
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters dynamically.

        Args:
            params: Dictionary containing parameter keys to update

        Raises:
            ValidationError: If parameters are invalid
        """
        if "swing_lookback" in params:
            swing_lookback = params["swing_lookback"]
            validate_integer(
                swing_lookback,
                "swing_lookback",
                min_value=1,
                max_value=ValidationLimits.MAX_INDICATOR_PERIOD
            )
            self.swing_lookback = swing_lookback

        if "retracement_levels" in params:
            retracement_levels = params["retracement_levels"]
            if not isinstance(retracement_levels, list) or len(retracement_levels) == 0:
                raise ValidationError("retracement_levels must be a non-empty list")

            for level in retracement_levels:
                if not isinstance(level, (int, float)) or level < 0 or level > 1:
                    raise ValidationError(
                        f"retracement_levels must contain numbers between 0 and 1, got {level}"
                    )

            self.retracement_levels = retracement_levels

    @property
    def warmup_period(self) -> int:
        """
        Minimum bars needed before signals are valid.

        Returns:
            The swing_lookback parameter
        """
        return self.swing_lookback

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy and sell signals based on Fibonacci retracement levels.

        Args:
            data: DataFrame with OHLCV data, must contain 'Close', 'High', 'Low' columns
                  with DatetimeIndex and at least swing_lookback rows

        Returns:
            DataFrame with 'buy' and 'sell' boolean columns, same index as input

        Raises:
            ValidationError: If data validation fails
        """
        # Comprehensive input validation
        validate_dataframe(
            data,
            required_columns=['Close', 'High', 'Low'],
            min_rows=self.swing_lookback
        )
        validate_price_data(data, 'Close')
        validate_price_data(data, 'High')
        validate_price_data(data, 'Low')

        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index, dtype=bool)
        signals['buy'] = False
        signals['sell'] = False

        close_prices = data['Close']
        high_prices = data['High']
        low_prices = data['Low']

        # Calculate swing high and low over lookback period
        swing_high = high_prices.rolling(
            window=self.swing_lookback,
            min_periods=self.swing_lookback
        ).max()
        swing_low = low_prices.rolling(
            window=self.swing_lookback,
            min_periods=self.swing_lookback
        ).min()

        # Calculate swing range
        swing_range = swing_high - swing_low

        # Only generate signals when there is a meaningful swing range
        # If swing_range is 0 or near-zero, no retracement levels exist
        has_swing = swing_range > 0

        # Find support level (61.8% retracement from high)
        # Support = swing_low + swing_range * 0.618
        support_level = swing_low + swing_range * 0.618

        # Find resistance level (38.2% retracement from high)
        # Resistance = swing_low + swing_range * 0.382
        resistance_level = swing_low + swing_range * 0.382

        # Define tolerance for "near" (2% of swing range)
        tolerance = swing_range * 0.02

        # Buy signal: price is near support level (within tolerance)
        # Price close to 61.8% retracement indicates potential bounce
        # Only signal when there is a meaningful swing range
        near_support = (close_prices >= support_level - tolerance) & (
            close_prices <= support_level + tolerance
        ) & has_swing
        signals['buy'] = near_support

        # Sell signal: price is near resistance level (within tolerance)
        # Price close to 38.2% retracement indicates potential rejection
        # Only signal when there is a meaningful swing range
        near_resistance = (close_prices >= resistance_level - tolerance) & (
            close_prices <= resistance_level + tolerance
        ) & has_swing
        signals['sell'] = near_resistance

        # Shift signals to trade on the next day's open
        signals = signals.shift(1).fillna(False)


        return signals


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy combining RSI and Bollinger Bands.

    Generates buy signals when RSI is oversold AND price is below the lower Bollinger Band,
    anticipating a reversion to the mean. Generates sell signals when RSI is overbought AND
    price is above the upper Bollinger Band.

    This composite strategy requires both conditions to be met simultaneously, reducing
    false signals compared to using either indicator alone.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_lower: int = 30,
        rsi_upper: int = 70,
        bb_period: int = 20,
        bb_std: float = 2.0
    ):
        """
        Initialize mean reversion strategy.

        Args:
            rsi_period: Period for RSI calculation (default: 14)
            rsi_lower: Lower RSI threshold for oversold condition (default: 30)
            rsi_upper: Upper RSI threshold for overbought condition (default: 70)
            bb_period: Period for Bollinger Bands calculation (default: 20)
            bb_std: Standard deviation multiplier for Bollinger Bands (default: 2.0)

        Raises:
            ValidationError: If parameters are invalid
        """
        self.rsi_period = rsi_period
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.bb_period = bb_period
        self.bb_std = bb_std
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate strategy parameters."""
        validate_integer(self.rsi_period, "rsi_period", min_value=1)
        validate_integer(self.bb_period, "bb_period", min_value=1)
        validate_positive_number(self.bb_std, "bb_std")
        validate_integer(self.rsi_lower, "rsi_lower", min_value=0, max_value=100)
        validate_integer(self.rsi_upper, "rsi_upper", min_value=0, max_value=100)

        if self.rsi_lower >= self.rsi_upper:
            raise ValidationError("rsi_lower must be less than rsi_upper")

    @property
    def warmup_period(self) -> int:
        """Return the warmup period (max of RSI and BB periods)."""
        return max(self.rsi_period, self.bb_period)

    def get_parameters(self) -> Dict[str, Any]:
        """Return current strategy parameters."""
        return {
            "rsi_period": self.rsi_period,
            "rsi_lower": self.rsi_lower,
            "rsi_upper": self.rsi_upper,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters.

        Args:
            params: Dictionary of parameter names and values

        Raises:
            ValidationError: If new parameters are invalid
        """
        valid_keys = {"rsi_period", "rsi_lower", "rsi_upper", "bb_period", "bb_std"}
        for key in params:
            if key not in valid_keys:
                raise ValidationError(f"Unknown parameter: {key}")
        if "rsi_period" in params:
            self.rsi_period = params["rsi_period"]
        if "rsi_lower" in params:
            self.rsi_lower = params["rsi_lower"]
        if "rsi_upper" in params:
            self.rsi_upper = params["rsi_upper"]
        if "bb_period" in params:
            self.bb_period = params["bb_period"]
        if "bb_std" in params:
            self.bb_std = params["bb_std"]
        self._validate_parameters()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals combining RSI and Bollinger Bands.

        Buy when: RSI < rsi_lower AND price < lower Bollinger Band
        Sell when: RSI > rsi_upper AND price > upper Bollinger Band

        Args:
            data: DataFrame with DatetimeIndex and 'Close' column

        Returns:
            DataFrame with same index and 'buy'/'sell' boolean columns

        Raises:
            ValidationError: If data is invalid or insufficient
        """
        validate_dataframe(data, required_columns=['Close'])
        validate_price_data(data, column='Close')

        if len(data) < self.warmup_period:
            raise ValidationError(
                f"Insufficient data. Need at least {self.warmup_period} rows, got {len(data)}"
            )

        signals = pd.DataFrame(index=data.index)
        close_prices = data['Close']

        # Calculate RSI
        rsi_indicator = ta.momentum.RSIIndicator(close=close_prices, window=self.rsi_period)
        rsi = rsi_indicator.rsi()

        # Calculate Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(
            close=close_prices,
            window=self.bb_period,
            window_dev=self.bb_std
        )
        lower_band = bb_indicator.bollinger_lband()
        upper_band = bb_indicator.bollinger_hband()

        # Generate signals: both conditions must be true
        signals['buy'] = (rsi < self.rsi_lower) & (close_prices < lower_band)
        signals['sell'] = (rsi > self.rsi_upper) & (close_prices > upper_band)

        # Shift signals to trade on the next day's open
        signals = signals.shift(1).fillna(False)


        return signals


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy using Rate of Change (ROC) indicator.

    Generates buy signals when ROC exceeds a positive threshold (strong upward momentum)
    and sell signals when ROC falls below a negative threshold (strong downward momentum).

    ROC measures the percentage change in price over a specified period, making it
    effective for identifying trending markets and momentum shifts.
    """

    def __init__(self, roc_period: int = 12, roc_threshold: float = 0.05):
        """
        Initialize momentum strategy.

        Args:
            roc_period: Period for ROC calculation (default: 12)
            roc_threshold: ROC threshold for signal generation (default: 0.05 = 5%)
                          Buy when ROC > threshold, sell when ROC < -threshold

        Raises:
            ValidationError: If parameters are invalid
        """
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate strategy parameters."""
        validate_integer(self.roc_period, "roc_period", min_value=1)
        validate_positive_number(self.roc_threshold, "roc_threshold")

    @property
    def warmup_period(self) -> int:
        """Return the warmup period (ROC period)."""
        return self.roc_period

    def get_parameters(self) -> Dict[str, Any]:
        """Return current strategy parameters."""
        return {
            "roc_period": self.roc_period,
            "roc_threshold": self.roc_threshold
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters.

        Args:
            params: Dictionary of parameter names and values

        Raises:
            ValidationError: If new parameters are invalid
        """
        valid_keys = {"roc_period", "roc_threshold"}
        for key in params:
            if key not in valid_keys:
                raise ValidationError(f"Unknown parameter: {key}")
        if "roc_period" in params:
            self.roc_period = params["roc_period"]
        if "roc_threshold" in params:
            self.roc_threshold = params["roc_threshold"]
        self._validate_parameters()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Rate of Change (ROC).

        Buy when: ROC > threshold (positive momentum)
        Sell when: ROC < -threshold (negative momentum)

        Args:
            data: DataFrame with DatetimeIndex and 'Close' column

        Returns:
            DataFrame with same index and 'buy'/'sell' boolean columns

        Raises:
            ValidationError: If data is invalid or insufficient
        """
        validate_dataframe(data, required_columns=['Close'])
        validate_price_data(data, column='Close')

        if len(data) < self.warmup_period:
            raise ValidationError(
                f"Insufficient data. Need at least {self.warmup_period} rows, got {len(data)}"
            )

        signals = pd.DataFrame(index=data.index)
        close_prices = data['Close']

        # Calculate ROC - returns percentage change (0-100 scale)
        roc_indicator = ta.momentum.ROCIndicator(close=close_prices, window=self.roc_period)
        roc = roc_indicator.roc()

        # Convert threshold to percentage (0-100 scale) for comparison with ROC
        # ROC in ta library returns values like 5.0 for 5% change
        threshold_pct = self.roc_threshold * 100

        # Generate signals based on ROC thresholds
        signals['buy'] = roc > threshold_pct
        signals['sell'] = roc < -threshold_pct

        # Shift signals to trade on the next day's open
        signals = signals.shift(1).fillna(False)


        return signals


class VolatilityStrategy(BaseStrategy):
    """
    Volatility breakout strategy using Average True Range (ATR).

    Generates buy signals when price breaks above recent high during high volatility periods,
    and sell signals when price breaks below recent low during high volatility periods.

    ATR measures market volatility, helping identify periods when breakouts are more likely
    to be genuine rather than false signals in low-volatility sideways markets.
    """

    def __init__(self, atr_period: int = 14, atr_multiplier: float = 2.0, breakout_period: int = 20):
        """
        Initialize volatility strategy.

        Args:
            atr_period: Period for ATR calculation (default: 14)
            atr_multiplier: Multiplier for ATR threshold (default: 2.0)
                           ATR must exceed (mean ATR * multiplier) for high volatility
            breakout_period: Period for high/low breakout detection (default: 20)

        Raises:
            ValidationError: If parameters are invalid
        """
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.breakout_period = breakout_period
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate strategy parameters."""
        validate_integer(self.atr_period, "atr_period", min_value=1)
        validate_positive_number(self.atr_multiplier, "atr_multiplier")
        validate_integer(self.breakout_period, "breakout_period", min_value=1)

    @property
    def warmup_period(self) -> int:
        """Return the warmup period (max of ATR and breakout periods)."""
        return max(self.atr_period, self.breakout_period)

    def get_parameters(self) -> Dict[str, Any]:
        """Return current strategy parameters."""
        return {
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
            "breakout_period": self.breakout_period
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters.

        Args:
            params: Dictionary of parameter names and values

        Raises:
            ValidationError: If new parameters are invalid
        """
        valid_keys = {"atr_period", "atr_multiplier", "breakout_period"}
        for key in params:
            if key not in valid_keys:
                raise ValidationError(f"Unknown parameter: {key}")
        if "atr_period" in params:
            self.atr_period = params["atr_period"]
        if "atr_multiplier" in params:
            self.atr_multiplier = params["atr_multiplier"]
        if "breakout_period" in params:
            self.breakout_period = params["breakout_period"]
        self._validate_parameters()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on ATR volatility and price breakouts.

        Buy when: price breaks above recent high AND ATR indicates high volatility
        Sell when: price breaks below recent low AND ATR indicates high volatility

        Args:
            data: DataFrame with DatetimeIndex and 'High', 'Low', 'Close' columns

        Returns:
            DataFrame with same index and 'buy'/'sell' boolean columns

        Raises:
            ValidationError: If data is invalid or insufficient
        """
        validate_dataframe(data, required_columns=['High', 'Low', 'Close'])

        # Validate each price column
        for column in ['High', 'Low', 'Close']:
            validate_price_data(data, column=column)

        if len(data) < self.warmup_period:
            raise ValidationError(
                f"Insufficient data. Need at least {self.warmup_period} rows, got {len(data)}"
            )

        signals = pd.DataFrame(index=data.index)
        close_prices = data['Close']

        # Calculate ATR
        atr_indicator = ta.volatility.AverageTrueRange(
            high=data['High'],
            low=data['Low'],
            close=close_prices,
            window=self.atr_period
        )
        atr = atr_indicator.average_true_range()

        # Calculate ATR threshold for high volatility
        # Use rolling mean ATR to adapt to changing market conditions
        mean_atr = atr.rolling(window=self.atr_period).mean()
        high_volatility = atr > (mean_atr * self.atr_multiplier)

        # Calculate breakout levels
        highest_high = data['High'].rolling(window=self.breakout_period).max()
        lowest_low = data['Low'].rolling(window=self.breakout_period).min()

        # Generate signals: breakout + high volatility
        # Buy when close breaks above recent high during high volatility
        signals['buy'] = (close_prices > highest_high.shift(1)) & high_volatility

        # Sell when close breaks below recent low during high volatility
        signals['sell'] = (close_prices < lowest_low.shift(1)) & high_volatility

        # Shift signals to trade on the next day's open
        signals = signals.shift(1).fillna(False)


        return signals


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble strategy that aggregates signals from multiple sub-strategies.

    Combines signals from multiple strategies using majority voting. A buy or sell signal
    is generated only when at least min_agreement strategies agree.

    This approach reduces false signals by requiring consensus, while still capturing
    strong opportunities identified by multiple independent strategies.
    """

    def __init__(self, strategies: list, min_agreement: int = 1):
        """
        Initialize ensemble strategy.

        Args:
            strategies: List of BaseStrategy instances to aggregate
            min_agreement: Minimum number of strategies that must agree for a signal
                          (default: 1 = any strategy can trigger signal)

        Raises:
            ValidationError: If parameters are invalid
        """
        self.strategies = strategies
        self.min_agreement = min_agreement
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate strategy parameters."""
        if not isinstance(self.strategies, list):
            raise ValidationError("strategies must be a list")

        if len(self.strategies) == 0:
            raise ValidationError("Ensemble strategy requires at least one strategy")

        for i, strategy in enumerate(self.strategies):
            if not isinstance(strategy, BaseStrategy):
                raise ValidationError(
                    f"Strategy at index {i} must inherit from BaseStrategy"
                )

        if not isinstance(self.min_agreement, int) or self.min_agreement < 1:
            raise ValidationError("min_agreement must be an integer >= 1")

        if self.min_agreement > len(self.strategies):
            raise ValidationError(
                f"min_agreement ({self.min_agreement}) cannot exceed number of "
                f"strategies ({len(self.strategies)})"
            )

    @property
    def warmup_period(self) -> int:
        """Return the warmup period (max of all sub-strategies)."""
        return max(strategy.warmup_period for strategy in self.strategies)

    def get_parameters(self) -> Dict[str, Any]:
        """Return current strategy parameters."""
        return {
            "num_strategies": len(self.strategies),
            "min_agreement": self.min_agreement,
            "strategy_names": [type(s).__name__ for s in self.strategies]
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters.

        Note: Can only update min_agreement. Sub-strategies are immutable.

        Args:
            params: Dictionary of parameter names and values

        Raises:
            ValidationError: If new parameters are invalid
        """
        if "min_agreement" in params:
            self.min_agreement = params["min_agreement"]
            self._validate_parameters()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals by aggregating sub-strategy signals.

        Buy when: At least min_agreement strategies generate buy signals
        Sell when: At least min_agreement strategies generate sell signals

        Args:
            data: DataFrame with DatetimeIndex and required columns for sub-strategies

        Returns:
            DataFrame with same index and 'buy'/'sell' boolean columns

        Raises:
            ValidationError: If data is invalid or insufficient
        """
        validate_dataframe(data)

        if len(data) < self.warmup_period:
            raise ValidationError(
                f"Insufficient data. Need at least {self.warmup_period} rows, got {len(data)}"
            )

        # Generate signals from each sub-strategy
        all_buy_signals = []
        all_sell_signals = []

        for strategy in self.strategies:
            sub_signals = strategy.generate_signals(data)
            all_buy_signals.append(sub_signals['buy'])
            all_sell_signals.append(sub_signals['sell'])

        # Aggregate signals using majority voting
        # Count how many strategies voted for each signal
        buy_votes = pd.DataFrame(all_buy_signals).T.sum(axis=1)
        sell_votes = pd.DataFrame(all_sell_signals).T.sum(axis=1)

        # Generate final signals based on min_agreement threshold
        signals = pd.DataFrame(index=data.index)
        signals['buy'] = buy_votes >= self.min_agreement
        signals['sell'] = sell_votes >= self.min_agreement

        # Note: Individual strategies already shift their signals,
        # so no additional shifting needed here


        return signals
