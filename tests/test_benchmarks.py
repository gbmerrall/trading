import pytest
import pandas as pd
import numpy as np
import yfinance as yf

# Add project root to the Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.benchmarks import BuyAndHold, SPYBuyAndHold, DollarCostAveraging

@pytest.fixture
def sample_price_data():
    """Fixture for sample price data."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
    prices = [100, 110, 105, 115]
    return pd.DataFrame({'Close': prices}, index=dates)

@pytest.fixture
def sample_price_data_with_nan():
    """Fixture for sample price data with NaN values."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
    prices = [100, np.nan, 105, 115]
    return pd.DataFrame({'Close': prices}, index=dates)

# --- Tests for BuyAndHold ---

def test_buy_and_hold_calculation(sample_price_data):
    """Test the basic calculation of the BuyAndHold benchmark."""
    benchmark = BuyAndHold()
    start_capital = 10000.0
    returns = benchmark.calculate_returns(sample_price_data, start_capital)

    assert isinstance(returns, pd.Series)
    assert returns.iloc[0] == start_capital
    # Final value should be 10000 * (115 / 100) = 11500
    assert np.isclose(returns.iloc[-1], 11500)

def test_buy_and_hold_missing_close_column():
    """Test that BuyAndHold raises an error if 'Close' column is missing."""
    benchmark = BuyAndHold()
    # Create with datetime index to avoid DatetimeIndex validation error
    dates = pd.to_datetime(['2023-01-01'])
    data = pd.DataFrame({'Open': [100]}, index=dates)
    with pytest.raises(ValueError, match="Data must contain 'Close' column"):
        benchmark.calculate_returns(data, 10000.0)

def test_buy_and_hold_all_nan_prices():
    """Test that BuyAndHold raises an error if all prices are NaN."""
    benchmark = BuyAndHold()
    # Create with datetime index to avoid DatetimeIndex validation error
    dates = pd.to_datetime(['2023-01-01', '2023-01-02'])
    data = pd.DataFrame({'Close': [np.nan, np.nan]}, index=dates)
    with pytest.raises(ValueError, match="All close prices are NaN"):
        benchmark.calculate_returns(data, 10000.0)

def test_buy_and_hold_with_nan_prices(sample_price_data_with_nan):
    """Test that BuyAndHold correctly handles some NaN values."""
    benchmark = BuyAndHold()
    start_capital = 10000.0
    returns = benchmark.calculate_returns(sample_price_data_with_nan, start_capital)
    # The ffill() will propagate 100, so the final value should be 10000 * (115/100)
    assert np.isclose(returns.iloc[-1], 11500)


# --- Tests for SPYBuyAndHold ---

def test_spy_buy_and_hold_calculation(monkeypatch):
    """Test the basic calculation of the SPYBuyAndHold benchmark."""
    class MockYFinance:
        def download(self, *args, **kwargs):
            dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
            prices = [400, 401, 402, 403]
            return pd.DataFrame({'Close': prices}, index=dates)

    monkeypatch.setattr(yf, 'download', MockYFinance().download)

    benchmark = SPYBuyAndHold()
    # Create dummy data with Close column (even though SPY will use its own data)
    dummy_data = pd.DataFrame(
        {'Close': [100, 100]}, 
        index=pd.to_datetime(['2023-01-01', '2023-01-04'])
    )
    returns = benchmark.calculate_returns(dummy_data, 10000.0)

    assert isinstance(returns, pd.Series)
    assert returns.iloc[0] == 10000.0
    # Final value should be 10000 * (403 / 400) = 10075
    assert np.isclose(returns.iloc[-1], 10075)

def test_spy_buy_and_hold_download_failure(monkeypatch):
    """Test that SPYBuyAndHold raises an error if the download fails."""
    class MockYFinance:
        def download(self, *args, **kwargs):
            return pd.DataFrame() 

    monkeypatch.setattr(yf, 'download', MockYFinance().download)
    benchmark = SPYBuyAndHold()
    dummy_data = pd.DataFrame(
        {'Close': [100, 100]}, 
        index=pd.to_datetime(['2023-01-01', '2023-01-04'])
    )

    with pytest.raises(ValueError, match="Could not download SPY data"):
        benchmark.calculate_returns(dummy_data, 10000.0)

def test_spy_buy_and_hold_all_nan_after_reindex(monkeypatch):
    """Test SPYBuyAndHold raises error if prices are all NaN after reindexing."""
    class MockYFinance:
        def download(self, *args, **kwargs):
            dates = pd.to_datetime(['2023-02-01', '2023-02-02'])
            prices = [400, 401]
            return pd.DataFrame({'Close': prices}, index=dates)

    monkeypatch.setattr(yf, 'download', MockYFinance().download)
    benchmark = SPYBuyAndHold()
    dummy_data = pd.DataFrame(
        {'Close': [100, 100]}, 
        index=pd.to_datetime(['2023-01-01', '2023-01-04'])
    )

    with pytest.raises(ValueError, match="All SPY close prices are NaN after alignment"):
        benchmark.calculate_returns(dummy_data, 10000.0)


# --- Tests for DollarCostAveraging ---

def test_dca_monthly(sample_price_data):
    """Test monthly dollar cost averaging."""
    benchmark = DollarCostAveraging(frequency='monthly')
    returns = benchmark.calculate_returns(sample_price_data, 10000.0)
    # Only one investment day (2023-01-01), so all 10k is invested.
    # Shares bought = 10000 / 100 = 100.
    # Final value = 100 shares * 115 (final price) = 11500.
    assert np.isclose(returns.iloc[0], 10000)
    assert np.isclose(returns.iloc[-1], 11500)

def test_dca_daily(sample_price_data):
    """Test daily dollar cost averaging."""
    benchmark = DollarCostAveraging(frequency='daily')
    returns = benchmark.calculate_returns(sample_price_data, 10000.0)
    # 4 investment days, 2500 each.
    # Day 1: 2500/100=25 sh. Total: 25 sh. Cash: 7500. Value: 25*100+7500=10000
    # Day 2: 2500/110=22.72 sh. Total: 47.72 sh. Cash: 5000. Value: 47.72*110+5000=10250
    # Day 3: 2500/105=23.80 sh. Total: 71.52 sh. Cash: 2500. Value: 71.52*105+2500=10010
    # Day 4: 2500/115=21.73 sh. Total: 93.25 sh. Cash: 0. Value: 93.25*115=10724
    assert np.isclose(returns.iloc[0], 10000)
    assert np.isclose(returns.iloc[-1], 10726.73, atol=1)

def test_dca_invalid_frequency():
    """Test that DCA raises an error for an invalid frequency."""
    with pytest.raises(ValueError, match="Invalid frequency"):
        DollarCostAveraging(frequency='yearly')

def test_dca_no_investment_days(sample_price_data):
    """Test DCA returns start capital if there are no investment days."""
    benchmark = DollarCostAveraging(frequency='monthly')
    # Create data that doesn't contain the start of the month
    data = sample_price_data[sample_price_data.index.day > 1]
    returns = benchmark.calculate_returns(data, 10000.0)
    assert (returns == 10000.0).all()

def test_dca_all_nan_prices():
    """Test that DCA handles all NaN prices gracefully."""
    benchmark = DollarCostAveraging(frequency='monthly')
    dates = pd.to_datetime(['2023-01-01', '2023-01-02'])
    data = pd.DataFrame({'Close': [np.nan, np.nan]}, index=dates)
    returns = benchmark.calculate_returns(data, 10000.0)
    assert (returns == 10000.0).all() 