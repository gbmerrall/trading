import pytest
import pandas as pd

# Add the project root directory to the Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.portfolio import Portfolio, Transaction, Position
from backtest.validation import ValidationError


@pytest.fixture
def sample_data():
    """Create a sample of price data for testing."""
    dates = pd.to_datetime(['2023-01-03', '2023-01-04', '2023-01-05'])
    prices = [100.0, 102.0, 99.0]
    return pd.DataFrame({'Close': prices}, index=dates)


# ==================== BACKWARD COMPATIBILITY TESTS ====================

def test_portfolio_initialization():
    """Test that the portfolio is initialized with the correct starting capital and no positions."""
    # Given a starting capital
    start_capital = 10000.0
    
    # When a portfolio is created
    portfolio = Portfolio(start_capital=start_capital)
    
    # Then it should have the correct cash and empty positions
    assert portfolio.cash == start_capital
    assert portfolio.positions == {}
    assert portfolio.get_value_history() == []


def test_portfolio_buy_signal(sample_data):
    """Test that a buy signal correctly updates cash, positions, and value history."""
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
    assert history[0]['value'] == 10000.0  # Value is cash + assets


def test_portfolio_sell_signal(sample_data):
    """Test that a sell signal correctly updates cash and positions."""
    # Given a portfolio that already holds a position
    portfolio = Portfolio(start_capital=10000.0)
    portfolio.buy('asset', 20, 100.0)  # Buy 20 shares at $100
    
    current_date = sample_data.index[1]
    current_price = sample_data['Close'].loc[current_date]

    # When processing a day with a sell signal (sell 5 shares)
    portfolio.process_day(current_date, current_price, sell_signal=True, shares=5)

    # Then cash should increase by the value of the shares sold
    # Started with $10000, bought 20 shares at $100 = $8000 cash remaining
    # Sold 5 shares at $102 = $8000 + $510 = $8510 cash
    expected_cash = 8000.0 + (5 * 102.0)
    assert portfolio.cash == expected_cash
    
    # And the position should be updated
    assert portfolio.positions['asset'] == 15


# ==================== ENHANCED INTERFACE TESTS ====================

def test_portfolio_initialization_with_parameters():
    """Test portfolio initialization with enhanced parameters."""
    portfolio = Portfolio(
        start_capital=50000.0,
        commission_rate=0.001,  # 0.1% commission
        default_symbol='STOCK'
    )
    
    assert portfolio.start_capital == 50000.0
    assert portfolio.cash == 50000.0
    assert portfolio.commission_rate == 0.001
    assert portfolio.default_symbol == 'STOCK'


def test_buy_single_asset():
    """Test buying shares of a single asset."""
    portfolio = Portfolio(start_capital=10000.0)
    date = pd.Timestamp('2023-01-01')
    
    # Buy 50 shares of AAPL at $150
    success = portfolio.buy('AAPL', 50, 150.0, date)
    
    assert success is True
    assert portfolio.cash == 10000.0 - (50 * 150.0)  # $2500 remaining
    assert portfolio.positions['AAPL'] == 50
    
    # Check position details
    positions_df = portfolio.get_positions_summary()
    assert len(positions_df) == 1
    assert positions_df.iloc[0]['symbol'] == 'AAPL'
    assert positions_df.iloc[0]['shares'] == 50
    assert positions_df.iloc[0]['avg_cost'] == 150.0


def test_buy_multiple_assets():
    """Test buying shares of multiple assets."""
    portfolio = Portfolio(start_capital=35000.0)  # Increased capital to afford all purchases
    date = pd.Timestamp('2023-01-01')
    
    # Buy shares of multiple assets
    success1 = portfolio.buy('AAPL', 20, 150.0, date)    # $3,000
    success2 = portfolio.buy('GOOGL', 10, 2000.0, date)  # $20,000
    success3 = portfolio.buy('MSFT', 30, 250.0, date)    # $7,500
    
    assert success1 is True
    assert success2 is True
    assert success3 is True
    
    assert len(portfolio.positions) == 3
    assert portfolio.positions['AAPL'] == 20
    assert portfolio.positions['GOOGL'] == 10
    assert portfolio.positions['MSFT'] == 30
    
    # Check total allocation
    total_spent = (20 * 150) + (10 * 2000) + (30 * 250)  # $30,500
    assert portfolio.cash == 35000.0 - total_spent


def test_buy_insufficient_funds():
    """Test buying when insufficient funds available."""
    portfolio = Portfolio(start_capital=1000.0)
    date = pd.Timestamp('2023-01-01')
    
    # Try to buy more than we can afford
    success = portfolio.buy('AAPL', 100, 150.0, date)  # Would cost $15,000
    
    assert success is False
    assert portfolio.cash == 1000.0  # Cash unchanged
    assert len(portfolio.positions) == 0  # No position created


def test_sell_full_position():
    """Test selling an entire position."""
    portfolio = Portfolio(start_capital=10000.0)
    date = pd.Timestamp('2023-01-01')
    
    # Buy then sell
    portfolio.buy('AAPL', 50, 150.0, date)
    shares_sold = portfolio.sell('AAPL', 50, 160.0, date)
    
    assert shares_sold == 50
    assert len(portfolio.positions) == 0  # Position removed
    assert portfolio.cash == 10000.0 + (50 * 10.0)  # Profit of $10 per share


def test_sell_partial_position():
    """Test selling part of a position."""
    portfolio = Portfolio(start_capital=20000.0)  # Increased capital
    date = pd.Timestamp('2023-01-01')
    
    # Buy 100 shares, sell 30
    success = portfolio.buy('AAPL', 100, 150.0, date)
    assert success is True
    
    shares_sold = portfolio.sell('AAPL', 30, 160.0, date)
    
    assert shares_sold == 30
    assert portfolio.positions['AAPL'] == 70  # 70 shares remaining
    
    # Check position update
    positions_df = portfolio.get_positions_summary()
    assert positions_df.iloc[0]['shares'] == 70


def test_sell_nonexistent_position():
    """Test selling shares of an asset not owned."""
    portfolio = Portfolio(start_capital=10000.0)
    date = pd.Timestamp('2023-01-01')
    
    shares_sold = portfolio.sell('AAPL', 50, 150.0, date)
    
    assert shares_sold == 0
    assert portfolio.cash == 10000.0  # No change


def test_sell_more_than_owned():
    """Test selling more shares than owned."""
    portfolio = Portfolio(start_capital=10000.0)
    date = pd.Timestamp('2023-01-01')
    
    # Buy 30 shares, try to sell 50
    portfolio.buy('AAPL', 30, 150.0, date)
    shares_sold = portfolio.sell('AAPL', 50, 160.0, date)
    
    assert shares_sold == 30  # Only sold what was owned
    assert len(portfolio.positions) == 0  # Position fully sold


def test_commission_calculation():
    """Test commission calculation."""
    portfolio = Portfolio(start_capital=10000.0, commission_rate=0.01)  # 1% commission
    date = pd.Timestamp('2023-01-01')
    
    # Buy $1000 worth of stock with 1% commission
    portfolio.buy('AAPL', 10, 100.0, date)
    
    # Should have paid $1000 + $10 commission = $1010 total
    assert portfolio.cash == 10000.0 - 1010.0
    
    # Check transaction record
    transactions = portfolio.get_transaction_history()
    assert transactions.iloc[0]['commission'] == 10.0


def test_update_position_prices():
    """Test updating position prices."""
    portfolio = Portfolio(start_capital=25000.0)  # Increased capital
    date = pd.Timestamp('2023-01-01')
    
    portfolio.buy('AAPL', 50, 150.0, date)
    portfolio.buy('GOOGL', 5, 2000.0, date)  # Reduced shares to fit budget
    
    # Update prices
    portfolio.update_all_prices({'AAPL': 160.0, 'GOOGL': 2100.0})
    
    positions_df = portfolio.get_positions_summary()
    aapl_pos = positions_df[positions_df['symbol'] == 'AAPL'].iloc[0]
    googl_pos = positions_df[positions_df['symbol'] == 'GOOGL'].iloc[0]
    
    assert aapl_pos['last_price'] == 160.0
    assert googl_pos['last_price'] == 2100.0


# ==================== ANALYTICS TESTS ====================

def test_performance_metrics():
    """Test portfolio performance metrics calculation."""
    portfolio = Portfolio(start_capital=10000.0)
    date = pd.Timestamp('2023-01-01')
    
    # Buy stock and update price
    portfolio.buy('AAPL', 50, 150.0, date)
    portfolio.update_position_price('AAPL', 160.0)  # $10 profit per share
    
    metrics = portfolio.get_performance_metrics()
    
    assert metrics['total_value'] == 10500.0  # $2500 cash + $8000 position value
    assert abs(metrics['total_return_pct'] - 5.0) < 0.01  # ~5% return (allow for floating point)
    assert metrics['unrealized_pnl'] == 500.0  # $10 * 50 shares
    assert metrics['num_positions'] == 1
    assert metrics['num_transactions'] == 1


def test_allocation_calculation():
    """Test portfolio allocation calculation."""
    portfolio = Portfolio(start_capital=10000.0)
    date = pd.Timestamp('2023-01-01')
    
    # Buy different assets
    portfolio.buy('AAPL', 20, 150.0, date)    # $3000
    portfolio.buy('GOOGL', 2, 2000.0, date)   # $4000
    # Remaining cash: $3000
    
    allocation = portfolio.get_allocation()
    
    assert abs(allocation['CASH'] - 30.0) < 0.1      # ~30%
    assert abs(allocation['AAPL'] - 30.0) < 0.1      # ~30%
    assert abs(allocation['GOOGL'] - 40.0) < 0.1     # ~40%


def test_transaction_history():
    """Test transaction history tracking."""
    portfolio = Portfolio(start_capital=30000.0)  # Increased capital
    date1 = pd.Timestamp('2023-01-01')
    date2 = pd.Timestamp('2023-01-02')
    
    # Execute multiple transactions
    portfolio.buy('AAPL', 50, 150.0, date1)
    portfolio.buy('GOOGL', 5, 2000.0, date1)
    portfolio.sell('AAPL', 20, 160.0, date2)
    
    transactions = portfolio.get_transaction_history()
    
    assert len(transactions) == 3
    assert transactions.iloc[0]['action'] == 'BUY'
    assert transactions.iloc[0]['symbol'] == 'AAPL'
    assert transactions.iloc[2]['action'] == 'SELL'
    assert transactions.iloc[2]['shares'] == 20


def test_realized_vs_unrealized_pnl():
    """Test realized vs unrealized P&L calculation."""
    portfolio = Portfolio(start_capital=20000.0)  # Increased capital
    date = pd.Timestamp('2023-01-01')
    
    # Buy 100 shares at $100, sell 50 at $110, price now $120
    portfolio.buy('AAPL', 100, 100.0, date)
    portfolio.sell('AAPL', 50, 110.0, date)
    portfolio.update_position_price('AAPL', 120.0)
    
    metrics = portfolio.get_performance_metrics()
    
    # Realized: sold 50 shares for $10 profit each = $500
    assert metrics['realized_pnl'] == 500.0
    
    # Unrealized: 50 shares bought at $100, now worth $120 = $1000 unrealized
    assert metrics['unrealized_pnl'] == 1000.0
    
    # Total P&L should be $1500
    assert metrics['total_pnl'] == 1500.0


# ==================== ERROR HANDLING TESTS ====================

def test_invalid_initialization():
    """Test portfolio initialization with invalid parameters."""
    with pytest.raises(ValidationError):
        Portfolio(start_capital=-1000.0)
    
    with pytest.raises(ValidationError):
        Portfolio(start_capital=10000.0, commission_rate=0.2)  # 20% commission too high


def test_invalid_buy_parameters():
    """Test buy method with invalid parameters."""
    portfolio = Portfolio(start_capital=10000.0)
    date = pd.Timestamp('2023-01-01')
    
    # Invalid symbol
    with pytest.raises(ValidationError):
        portfolio.buy('', 50, 150.0, date)
    
    # Invalid shares
    with pytest.raises(ValidationError):
        portfolio.buy('AAPL', 0, 150.0, date)
    
    # Invalid price
    with pytest.raises(ValidationError):
        portfolio.buy('AAPL', 50, -150.0, date)


def test_invalid_sell_parameters():
    """Test sell method with invalid parameters."""
    portfolio = Portfolio(start_capital=10000.0)
    date = pd.Timestamp('2023-01-01')
    
    # Same validation as buy
    with pytest.raises(ValidationError):
        portfolio.sell('', 50, 150.0, date)
    
    with pytest.raises(ValidationError):
        portfolio.sell('AAPL', 0, 150.0, date)
    
    with pytest.raises(ValidationError):
        portfolio.sell('AAPL', 50, -150.0, date)


# ==================== INTEGRATION TESTS ====================

def test_portfolio_workflow():
    """Test complete portfolio workflow."""
    portfolio = Portfolio(start_capital=100000.0, commission_rate=0.001)
    
    # Day 1: Buy initial positions
    date1 = pd.Timestamp('2023-01-01')
    portfolio.buy('AAPL', 100, 150.0, date1)
    portfolio.buy('GOOGL', 20, 2000.0, date1)
    portfolio.buy('MSFT', 50, 300.0, date1)
    
    # Day 2: Prices change, rebalance
    date2 = pd.Timestamp('2023-01-02')
    portfolio.update_all_prices({'AAPL': 155.0, 'GOOGL': 2100.0, 'MSFT': 290.0})
    portfolio.sell('GOOGL', 10, 2100.0, date2)  # Take some profits
    
    # Day 3: More price changes
    date3 = pd.Timestamp('2023-01-03')
    portfolio.update_all_prices({'AAPL': 160.0, 'GOOGL': 2050.0, 'MSFT': 310.0})
    
    # Check final state
    metrics = portfolio.get_performance_metrics()
    positions = portfolio.get_positions_summary()
    transactions = portfolio.get_transaction_history()
    
    assert metrics['num_positions'] == 3
    assert metrics['num_transactions'] == 4  # 3 buys + 1 sell
    assert len(positions) == 3
    assert len(transactions) == 4
    assert metrics['total_value'] > 100000.0  # Portfolio should have grown


def test_backward_compatibility_integration(sample_data):
    """Test that legacy interface still works with enhanced portfolio."""
    portfolio = Portfolio(start_capital=10000.0)
    
    # Use legacy interface
    for i, (date, row) in enumerate(sample_data.iterrows()):
        price = row['Close']
        if i == 0:  # Buy on first day
            portfolio.process_day(date, price, buy_signal=True, shares=50)
        elif i == 2:  # Sell on last day
            portfolio.process_day(date, price, sell_signal=True, shares=25)
        else:
            portfolio.process_day(date, price)
    
    # Check that enhanced features still work
    history = portfolio.get_value_history()
    metrics = portfolio.get_performance_metrics()
    transactions = portfolio.get_transaction_history()
    
    assert len(history) == 3
    assert metrics['num_transactions'] == 2
    assert len(transactions) == 2
    assert 'asset' in portfolio.positions


# ==================== DATA STRUCTURE TESTS ====================

def test_transaction_dataclass():
    """Test Transaction dataclass validation."""
    date = pd.Timestamp('2023-01-01')
    
    # Valid transaction
    txn = Transaction(
        date=date,
        symbol='AAPL',
        action='BUY',
        shares=50,
        price=150.0,
        value=7500.0
    )
    assert txn.symbol == 'AAPL'
    
    # Invalid action
    with pytest.raises(ValidationError):
        Transaction(date=date, symbol='AAPL', action='INVALID', shares=50, price=150.0, value=7500.0)


def test_position_dataclass():
    """Test Position dataclass calculations."""
    pos = Position(symbol='AAPL', shares=100, avg_cost=150.0, total_cost=15000.0, last_price=160.0)
    
    assert pos.market_value == 16000.0  # 100 * 160
    assert pos.unrealized_pnl == 1000.0  # 16000 - 15000
    assert abs(pos.unrealized_pnl_percent - 6.67) < 0.01  # ~6.67% (allow for floating point)


def test_string_representations():
    """Test string representations of portfolio."""
    portfolio = Portfolio(start_capital=10000.0)
    date = pd.Timestamp('2023-01-01')
    
    portfolio.buy('AAPL', 50, 150.0, date)
    
    str_repr = str(portfolio)
    repr_repr = repr(portfolio)
    
    assert 'Portfolio' in str_repr
    assert 'value=' in str_repr
    assert 'Portfolio' in repr_repr
    assert 'start_capital=' in repr_repr

