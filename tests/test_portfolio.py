import pytest
import pandas as pd

# Add the project root directory to the Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.portfolio import Portfolio

@pytest.fixture
def sample_data():
    """Create a sample of price data for testing."""
    dates = pd.to_datetime(['2023-01-03', '2023-01-04', '2023-01-05'])
    prices = [100.0, 102.0, 99.0]
    return pd.DataFrame({'Close': prices}, index=dates)

def test_portfolio_initialization():
    """Tests that the portfolio is initialized with the correct starting capital and no positions."""
    # Given a starting capital
    start_capital = 10000.0
    
    # When a portfolio is created
    portfolio = Portfolio(start_capital=start_capital)
    
    # Then it should have the correct cash and empty positions
    assert portfolio.cash == start_capital
    assert portfolio.positions == {}
    assert portfolio.get_value_history() == []

def test_portfolio_buy_signal(sample_data):
    """Tests that a buy signal correctly updates cash, positions, and value history."""
    # Given a portfolio and a buy signal on a specific day
    portfolio = Portfolio(start_capital=10000.0)
    current_date = sample_data.index[0]
    current_price = sample_data['Close'].loc[current_date]
    
    # When processing a day with a buy signal (buy 10 shares)
    portfolio.process_day(current_date, current_price, buy_signal=True, shares=10)
    
    # Then cash should decrease by the cost of the shares
    expected_cash = 10000.0 - (10 * 100.0)
    assert portfolio.cash == expected_cash
    
    # And the position should be recorded
    assert portfolio.positions['asset'] == 10
    
    # And the total value should be recorded in the history
    history = portfolio.get_value_history()
    assert len(history) == 1
    assert history[0]['date'] == current_date
    assert history[0]['value'] == 10000.0 # Value is cash + assets

def test_portfolio_sell_signal(sample_data):
    """Tests that a sell signal correctly updates cash and positions."""
    # Given a portfolio that already holds a position
    portfolio = Portfolio(start_capital=10000.0)
    portfolio.positions['asset'] = 20 # Assume we own 20 shares
    
    current_date = sample_data.index[1]
    current_price = sample_data['Close'].loc[current_date]

    # When processing a day with a sell signal (sell 5 shares)
    portfolio.process_day(current_date, current_price, sell_signal=True, shares=5)

    # Then cash should increase by the value of the shares sold
    expected_cash = 10000.0 + (5 * 102.0)
    assert portfolio.cash == expected_cash
    
    # And the position should be updated
    assert portfolio.positions['asset'] == 15
    
    # And the total value should reflect the new state
    history = portfolio.get_value_history()
    expected_value = expected_cash + (15 * 102.0)
    assert history[0]['value'] == expected_value 