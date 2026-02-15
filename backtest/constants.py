"""
Shared constants for the trading backtest system.

This module contains all validation limits and trading constants used throughout
the application. Centralized here to avoid duplication and circular imports.
"""


class TradingConstants:
    """Constants related to trading and financial markets."""

    # Market timing
    TRADING_DAYS_PER_YEAR = 252
    TRADING_DAYS_PER_MONTH = 21
    TRADING_DAYS_PER_WEEK = 5

    # Common market symbols
    DEFAULT_MARKET_BENCHMARK = "SPY"
    ALTERNATIVE_BENCHMARKS = ["QQQ", "IWM", "DIA"]

    # Percentage calculations
    PERCENT_MULTIPLIER = 100.0
    BASIS_POINTS_MULTIPLIER = 10000.0

    # Default ratios
    DEFAULT_CUMULATIVE_RETURN_BASE = 1.0
    EXTREME_PRICE_VARIATION_THRESHOLD = 1000  # max/min price ratio


class ValidationLimits:
    """Limits and thresholds for input validation."""

    # Portfolio limits
    MAX_COMMISSION_RATE = 0.10  # 10% maximum commission rate
    MIN_START_CAPITAL = 1.0  # Minimum starting capital
    MAX_START_CAPITAL = 1e12  # Maximum starting capital (1 trillion)

    # Strategy limits
    MIN_CONSECUTIVE_DAYS = 1
    MAX_CONSECUTIVE_DAYS = TradingConstants.TRADING_DAYS_PER_YEAR

    # Position limits
    MIN_SHARES = 1
    MAX_SHARES = 1_000_000
    MIN_PRICE = 0.01  # Minimum price (1 cent)
    MAX_PRICE = 1_000_000.0  # Maximum price per share

    # Data quality thresholds
    MAX_PRICE_VARIATION_RATIO = 1000  # Price max/min ratio threshold
    MIN_DATA_POINTS = 2  # Minimum data points for backtesting
