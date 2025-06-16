import pytest
import pandas as pd
import numpy as np

# Add project root to the Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.validation import (
    ValidationError, validate_dataframe, validate_price_data, 
    validate_positive_number, validate_integer, validate_string_choice,
    sanitize_file_path
)


class TestValidateDataFrame:
    """Test cases for DataFrame validation."""
    
    def test_valid_dataframe(self):
        """Test validation passes for valid DataFrame."""
        df = pd.DataFrame({'Close': [100, 101, 102]}, 
                         index=pd.date_range('2023-01-01', periods=3))
        validate_dataframe(df, required_columns=['Close'])
        # Should not raise any exception
    
    def test_non_dataframe_input(self):
        """Test validation fails for non-DataFrame input."""
        with pytest.raises(ValidationError, match="Expected pandas DataFrame"):
            validate_dataframe([1, 2, 3])
    
    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(ValidationError, match="DataFrame cannot be empty"):
            validate_dataframe(df)
    
    def test_missing_required_columns(self):
        """Test validation fails when required columns are missing."""
        df = pd.DataFrame({'Open': [100, 101, 102]})
        with pytest.raises(ValidationError, match="Missing required columns"):
            validate_dataframe(df, required_columns=['Close'])
    
    def test_insufficient_rows(self):
        """Test validation fails when DataFrame has too few rows."""
        df = pd.DataFrame({'Close': [100]}, 
                         index=pd.date_range('2023-01-01', periods=1))
        with pytest.raises(ValidationError, match="must have at least 5 rows"):
            validate_dataframe(df, min_rows=5)
    
    def test_non_datetime_index(self):
        """Test validation fails for non-datetime index."""
        df = pd.DataFrame({'Close': [100, 101, 102]})
        with pytest.raises(ValidationError, match="must have a DatetimeIndex"):
            validate_dataframe(df)


class TestValidatePriceData:
    """Test cases for price data validation."""
    
    def test_valid_price_data(self):
        """Test validation passes for valid price data."""
        df = pd.DataFrame({'Close': [100.0, 101.5, 102.3]},
                         index=pd.date_range('2023-01-01', periods=3))
        validate_price_data(df, 'Close')
        # Should not raise any exception
    
    def test_all_nan_prices(self):
        """Test validation fails when all prices are NaN."""
        df = pd.DataFrame({'Close': [np.nan, np.nan, np.nan]},
                         index=pd.date_range('2023-01-01', periods=3))
        with pytest.raises(ValidationError, match="All values in 'Close' column are NaN"):
            validate_price_data(df, 'Close')
    
    def test_negative_prices(self):
        """Test validation fails for negative prices."""
        df = pd.DataFrame({'Close': [100.0, -50.0, 102.3]},
                         index=pd.date_range('2023-01-01', periods=3))
        with pytest.raises(ValidationError, match="contains non-positive values"):
            validate_price_data(df, 'Close')
    
    def test_zero_prices(self):
        """Test validation fails for zero prices."""
        df = pd.DataFrame({'Close': [100.0, 0.0, 102.3]},
                         index=pd.date_range('2023-01-01', periods=3))
        with pytest.raises(ValidationError, match="contains non-positive values"):
            validate_price_data(df, 'Close')
    
    def test_extreme_price_variation(self):
        """Test validation fails for extreme price variations."""
        df = pd.DataFrame({'Close': [1.0, 2000.0, 1.5]},
                         index=pd.date_range('2023-01-01', periods=3))
        with pytest.raises(ValidationError, match="extreme price variation"):
            validate_price_data(df, 'Close')


class TestValidatePositiveNumber:
    """Test cases for positive number validation."""
    
    def test_valid_positive_number(self):
        """Test validation passes for valid positive numbers."""
        validate_positive_number(10.5, "test_value")
        validate_positive_number(1, "test_value")
        # Should not raise any exception
    
    def test_zero_not_allowed(self):
        """Test validation fails for zero when not allowed."""
        with pytest.raises(ValidationError, match="must be > 0"):
            validate_positive_number(0, "test_value", allow_zero=False)
    
    def test_zero_allowed(self):
        """Test validation passes for zero when allowed."""
        validate_positive_number(0, "test_value", allow_zero=True)
        # Should not raise any exception
    
    def test_negative_number(self):
        """Test validation fails for negative numbers."""
        with pytest.raises(ValidationError, match="must be"):
            validate_positive_number(-5.0, "test_value")
    
    def test_non_number_input(self):
        """Test validation fails for non-numeric input."""
        with pytest.raises(ValidationError, match="must be a number"):
            validate_positive_number("not a number", "test_value")
    
    def test_nan_input(self):
        """Test validation fails for NaN input."""
        with pytest.raises(ValidationError, match="cannot be NaN"):
            validate_positive_number(np.nan, "test_value")
    
    def test_infinite_input(self):
        """Test validation fails for infinite input."""
        with pytest.raises(ValidationError, match="cannot be NaN or infinite"):
            validate_positive_number(np.inf, "test_value")
    
    def test_max_value_constraint(self):
        """Test validation fails when exceeding max value."""
        with pytest.raises(ValidationError, match="must be <= 100"):
            validate_positive_number(150, "test_value", max_value=100)


class TestValidateInteger:
    """Test cases for integer validation."""
    
    def test_valid_integer(self):
        """Test validation passes for valid integers."""
        validate_integer(5, "test_value", min_value=1, max_value=10)
        # Should not raise any exception
    
    def test_non_integer_input(self):
        """Test validation fails for non-integer input."""
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_integer(5.5, "test_value")
    
    def test_below_minimum(self):
        """Test validation fails for values below minimum."""
        with pytest.raises(ValidationError, match="must be >= 5"):
            validate_integer(3, "test_value", min_value=5)
    
    def test_above_maximum(self):
        """Test validation fails for values above maximum."""
        with pytest.raises(ValidationError, match="must be <= 10"):
            validate_integer(15, "test_value", max_value=10)


class TestValidateStringChoice:
    """Test cases for string choice validation."""
    
    def test_valid_choice(self):
        """Test validation passes for valid choice."""
        result = validate_string_choice("daily", "frequency", ["daily", "weekly", "monthly"])
        assert result == "daily"
    
    def test_invalid_choice(self):
        """Test validation fails for invalid choice."""
        with pytest.raises(ValidationError, match="must be one of"):
            validate_string_choice("yearly", "frequency", ["daily", "weekly", "monthly"])
    
    def test_case_insensitive(self):
        """Test case insensitive validation."""
        result = validate_string_choice("DAILY", "frequency", ["daily", "weekly"], case_sensitive=False)
        assert result == "daily"
    
    def test_non_string_input(self):
        """Test validation fails for non-string input."""
        with pytest.raises(ValidationError, match="must be a string"):
            validate_string_choice(123, "frequency", ["daily", "weekly"])


class TestSanitizeFilePath:
    """Test cases for file path sanitization."""
    
    def test_valid_file_path(self):
        """Test sanitization passes for valid file path."""
        result = sanitize_file_path("output/chart.png", [".png", ".jpg"])
        assert result == "output/chart.png"
    
    def test_dangerous_characters(self):
        """Test sanitization fails for dangerous characters."""
        with pytest.raises(ValidationError, match="dangerous character"):
            sanitize_file_path("../../../etc/passwd", [".txt"])
    
    def test_invalid_extension(self):
        """Test sanitization fails for invalid extensions."""
        with pytest.raises(ValidationError, match="not allowed"):
            sanitize_file_path("malicious.exe", [".png", ".jpg"])
    
    def test_empty_path(self):
        """Test sanitization fails for empty path."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            sanitize_file_path("   ", [".png"])
    
    def test_non_string_input(self):
        """Test sanitization fails for non-string input."""
        with pytest.raises(ValidationError, match="must be a string"):
            sanitize_file_path(123, [".png"])


# Test fixtures for integration testing
@pytest.fixture
def valid_price_data():
    """Create valid price data for testing."""
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    prices = [100 + i for i in range(10)]  # Simple increasing prices
    return pd.DataFrame({'Close': prices}, index=dates)


@pytest.fixture 
def invalid_price_data():
    """Create invalid price data for testing."""
    dates = pd.date_range('2023-01-01', periods=5, freq='D') 
    prices = [100, -50, np.nan, 0, 105]  # Contains negative, NaN, and zero
    return pd.DataFrame({'Close': prices}, index=dates)


class TestIntegrationValidation:
    """Integration tests for validation across components."""
    
    def test_strategy_with_invalid_data_raises_validation_error(self, invalid_price_data):
        """Test that strategy raises ValidationError for invalid data."""
        from backtest.strategy import ConsecutiveDaysStrategy
        
        strategy = ConsecutiveDaysStrategy(consecutive_days=3)
        with pytest.raises(ValidationError):
            strategy.generate_signals(invalid_price_data)
    
    def test_portfolio_with_invalid_capital_raises_validation_error(self):
        """Test that portfolio raises ValidationError for invalid capital."""
        from backtest.portfolio import Portfolio
        
        with pytest.raises(ValidationError):
            Portfolio(start_capital=-1000)
    
    def test_runner_with_invalid_strategy_raises_validation_error(self):
        """Test that runner raises ValidationError for invalid strategy."""
        from backtest.runner import BacktestRunnerImpl
        
        class InvalidStrategy:
            pass  # Missing generate_signals method
        
        with pytest.raises(ValidationError, match="must have 'generate_signals' method"):
            BacktestRunnerImpl(strategy=InvalidStrategy(), benchmarks=[]) 