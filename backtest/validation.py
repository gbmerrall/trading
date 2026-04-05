"""Input validation utilities for the trading backtest system."""

import pandas as pd
import numpy as np
from typing import Any, Optional
import numbers

from .constants import ValidationLimits, TradingConstants


class ValidationError(ValueError):
    """Custom exception for validation errors."""
    pass


def validate_dataframe(
    data: pd.DataFrame, 
    required_columns: list[str] = None,
    min_rows: int = None,
    allow_empty: bool = False
) -> None:
    """
    Validate a pandas DataFrame for trading data.
    
    Args:
        data: DataFrame to validate
        required_columns: List of column names that must be present
        min_rows: Minimum number of rows required (uses config default if None)
        allow_empty: Whether to allow empty DataFrames
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, pd.DataFrame):
        raise ValidationError(f"Expected pandas DataFrame, got {type(data).__name__}")
    
    if not allow_empty and data.empty:
        raise ValidationError("DataFrame cannot be empty")
    
    if min_rows is None:
        min_rows = ValidationLimits.MIN_DATA_POINTS
    
    if len(data) < min_rows:
        raise ValidationError(f"DataFrame must have at least {min_rows} rows, got {len(data)}")
    
    if required_columns:
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
    
    # Check for datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValidationError("DataFrame must have a DatetimeIndex")


def validate_price_data(data: pd.DataFrame, column: str = 'Close') -> None:
    """
    Validate price data for trading operations.
    
    Args:
        data: DataFrame containing price data
        column: Name of the price column to validate
        
    Raises:
        ValidationError: If validation fails
    """
    validate_dataframe(data, required_columns=[column])
    
    prices = data[column]
    
    # Check for all NaN values
    if prices.isnull().all():
        raise ValidationError(f"All values in '{column}' column are NaN")
    
    # Check for negative prices
    non_null_prices = prices.dropna()
    if (non_null_prices <= 0).any():
        raise ValidationError(f"'{column}' column contains non-positive values")
    
    # Check for extreme values that might indicate data errors
    max_variation = ValidationLimits.MAX_PRICE_VARIATION_RATIO
    
    if non_null_prices.max() / non_null_prices.min() > max_variation:
        raise ValidationError(
            f"'{column}' column has extreme price variation "
            f"(max/min > {max_variation}), "
            "possible data quality issue"
        )


def validate_positive_number(
    value: Any, 
    name: str, 
    allow_zero: bool = False,
    max_value: Optional[float] = None
) -> None:
    """
    Validate that a value is a positive number.
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
        allow_zero: Whether zero is acceptable
        max_value: Maximum allowed value
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, numbers.Number):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
    
    if np.isnan(value) or np.isinf(value):
        raise ValidationError(f"{name} cannot be NaN or infinite")
    
    if allow_zero:
        if value < 0:
            raise ValidationError(f"{name} must be >= 0, got {value}")
    else:
        if value <= 0:
            raise ValidationError(f"{name} must be > 0, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be <= {max_value}, got {value}")


def validate_integer(
    value: Any, 
    name: str, 
    min_value: int = 1,
    max_value: Optional[int] = None
) -> None:
    """
    Validate that a value is a valid integer within range.
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")
    
    if value < min_value:
        raise ValidationError(f"{name} must be >= {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be <= {max_value}, got {value}")


def validate_string_choice(
    value: Any, 
    name: str, 
    choices: list[str],
    case_sensitive: bool = True
) -> str:
    """
    Validate that a string value is one of the allowed choices.
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
        choices: List of valid choices
        case_sensitive: Whether comparison should be case sensitive
        
    Returns:
        Validated string value (normalized to lowercase if case_sensitive=False)
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string, got {type(value).__name__}")
    
    comparison_value = value if case_sensitive else value.lower()
    comparison_choices = choices if case_sensitive else [c.lower() for c in choices]
    
    if comparison_value not in comparison_choices:
        raise ValidationError(f"{name} must be one of {choices}, got '{value}'")
    
    return comparison_value


def sanitize_file_path(file_path: str, allowed_extensions: list[str] = None) -> str:
    """
    Sanitize and validate a file path for security.
    
    Args:
        file_path: File path to sanitize
        allowed_extensions: List of allowed file extensions (with dots)
        
    Returns:
        Sanitized file path
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(file_path, str):
        raise ValidationError(f"File path must be a string, got {type(file_path).__name__}")
    
    if not file_path.strip():
        raise ValidationError("File path cannot be empty")
    
    # Remove potentially dangerous characters
    dangerous_chars = ['..', '<', '>', '|', '\0']
    for char in dangerous_chars:
        if char in file_path:
            raise ValidationError(f"File path contains dangerous character: '{char}'")
    
    if allowed_extensions:
        file_ext = '.' + file_path.split('.')[-1] if '.' in file_path else ''
        if file_ext not in allowed_extensions:
            raise ValidationError(
                f"File extension '{file_ext}' not allowed. "
                f"Allowed extensions: {allowed_extensions}"
            )
    
    return file_path.strip() 