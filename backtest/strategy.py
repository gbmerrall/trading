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
            max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
        )
        validate_integer(
            long_window,
            "long_window",
            min_value=1,
            max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
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
                max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
            )
            self.short_window = short_window

        if "long_window" in params:
            long_window = params["long_window"]
            validate_integer(
                long_window,
                "long_window",
                min_value=1,
                max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
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

        try:
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

        except Exception as e:
            raise ValidationError(f"Error generating signals: {str(e)}") from e

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
            max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
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
                max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
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

        try:
            # Calculate RSI using ta library
            rsi = ta.momentum.RSIIndicator(close=data['Close'], window=self.period).rsi()

            # Buy signal: RSI below lower_bound (oversold)
            signals['buy'] = rsi < self.lower_bound

            # Sell signal: RSI above upper_bound (overbought)
            signals['sell'] = rsi > self.upper_bound

            # Shift signals to trade on the next day's open
            signals = signals.shift(1).fillna(False)

        except Exception as e:
            raise ValidationError(f"Error generating signals: {str(e)}") from e

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
            max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
        )
        validate_integer(
            slow,
            "slow",
            min_value=1,
            max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
        )
        validate_integer(
            signal,
            "signal",
            min_value=1,
            max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
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
                max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
            )
            self.fast = fast

        if "slow" in params:
            slow = params["slow"]
            validate_integer(
                slow,
                "slow",
                min_value=1,
                max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
            )
            self.slow = slow

        if "signal" in params:
            signal = params["signal"]
            validate_integer(
                signal,
                "signal",
                min_value=1,
                max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
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

        try:
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

        except Exception as e:
            raise ValidationError(f"Error generating signals: {str(e)}") from e

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
            max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
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
                max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
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

        try:
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

        except Exception as e:
            raise ValidationError(f"Error generating signals: {str(e)}") from e

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

        try:
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

        except Exception as e:
            raise ValidationError(f"Error generating signals: {str(e)}") from e

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
            max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
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
                max_value=ValidationLimits.MAX_CONSECUTIVE_DAYS
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

        try:
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

        except Exception as e:
            raise ValidationError(f"Error generating signals: {str(e)}") from e

        return signals
