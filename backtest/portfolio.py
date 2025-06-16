"""
Enhanced Portfolio Management System

This module provides a comprehensive portfolio management system that supports
multiple assets, detailed position tracking, transaction history, and advanced
portfolio analytics.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .validation import validate_positive_number, validate_integer, ValidationError

# Hardcode constants to avoid config issues
class TradingConstants:
    PERCENT_MULTIPLIER = 100.0

class ValidationLimits:
    MIN_START_CAPITAL = 1.0
    MAX_START_CAPITAL = 1e12
    MAX_COMMISSION_RATE = 0.10
    MIN_SHARES = 1
    MAX_SHARES = 1_000_000
    MIN_PRICE = 0.01
    MAX_PRICE = 1_000_000.0


@dataclass
class Transaction:
    """
    Represents a single transaction in the portfolio.
    
    Attributes:
        date: Transaction date
        symbol: Asset symbol
        action: 'BUY' or 'SELL'
        shares: Number of shares traded
        price: Price per share
        value: Total transaction value (shares * price)
        commission: Commission paid
    """
    date: pd.Timestamp
    symbol: str
    action: str  # 'BUY' or 'SELL'
    shares: int
    price: float
    value: float
    commission: float = 0.0
    
    def __post_init__(self):
        """Validate transaction data after initialization."""
        if self.action not in ['BUY', 'SELL']:
            raise ValidationError(f"Action must be 'BUY' or 'SELL', got '{self.action}'")
        if self.shares <= 0:
            raise ValidationError(f"Shares must be positive, got {self.shares}")
        if self.price <= 0:
            raise ValidationError(f"Price must be positive, got {self.price}")


@dataclass
class Position:
    """
    Represents a position in a single asset.
    
    Attributes:
        symbol: Asset symbol
        shares: Number of shares held
        avg_cost: Average cost basis per share
        total_cost: Total cost basis of position
        last_price: Last known price per share
    """
    symbol: str
    shares: int = 0
    avg_cost: float = 0.0
    total_cost: float = 0.0
    last_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.shares * self.last_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss for the position."""
        return self.market_value - self.total_cost
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """Unrealized profit/loss as percentage."""
        return (self.unrealized_pnl / self.total_cost * TradingConstants.PERCENT_MULTIPLIER) if self.total_cost > 0 else 0.0


class Portfolio:
    """
    Enhanced portfolio management system supporting multiple assets,
    detailed position tracking, and comprehensive analytics.
    
    Maintains backward compatibility with the original single-asset interface
    while providing advanced multi-asset portfolio management capabilities.
    """
    
    def __init__(
        self, 
        start_capital: float = None,
        commission_rate: float = None,
        default_symbol: str = None
    ):
        """
        Initialize portfolio with enhanced configuration support.
        
        Args:
            start_capital: Starting capital amount
            commission_rate: Commission rate (0.0 to 1.0)
            default_symbol: Default symbol for legacy interface
        """
        # Use hardcoded defaults to avoid config issues
        self.start_capital = start_capital if start_capital is not None else 10000.0
        self.commission_rate = commission_rate if commission_rate is not None else 0.0  # Explicit 0.0 default
        self.default_symbol = default_symbol if default_symbol is not None else 'asset'
        
        # Validate parameters
        validate_positive_number(self.start_capital, "start_capital", max_value=ValidationLimits.MAX_START_CAPITAL)
        
        # Additional validation for start_capital minimum
        if self.start_capital < ValidationLimits.MIN_START_CAPITAL:
            raise ValidationError(f"start_capital must be >= {ValidationLimits.MIN_START_CAPITAL}, got {self.start_capital}")
        
        if not (0.0 <= self.commission_rate <= ValidationLimits.MAX_COMMISSION_RATE):
            raise ValidationError(f"commission_rate must be between 0.0 and {ValidationLimits.MAX_COMMISSION_RATE}")
        
        # Initialize portfolio state
        self.cash = self.start_capital
        self._positions: Dict[str, Position] = {}
        self._transactions: List[Transaction] = []
        self._value_history: List[Dict[str, Any]] = []
        self._realized_pnl = 0.0
        self._total_commissions = 0.0
        
        # Current portfolio value (updated by process_day)
        self.current_value = self.start_capital

    # ==================== BACKWARD COMPATIBILITY INTERFACE ====================
    
    @property
    def positions(self) -> Dict[str, int]:
        """
        Legacy interface: Return positions as simple dict of symbol -> shares.
        
        Returns:
            Dictionary mapping symbol to number of shares
        """
        return {symbol: pos.shares for symbol, pos in self._positions.items()}
    
    def get_value_history(self) -> List[Dict[str, Any]]:
        """Legacy interface: Return value history."""
        return self._value_history.copy()
    
    def process_day(
        self, 
        date: pd.Timestamp, 
        price: float, 
        buy_signal: bool = False, 
        sell_signal: bool = False, 
        shares: int = 0
    ) -> None:
        """
        Legacy interface: Process a single day with buy/sell signals.
        
        Args:
            date: Trading date
            price: Asset price for the day
            buy_signal: Whether to buy shares
            sell_signal: Whether to sell shares
            shares: Number of shares to trade
        """
        # Validation
        if not isinstance(date, pd.Timestamp):
            raise ValidationError(f"date must be pd.Timestamp, got {type(date).__name__}")
        
        validate_positive_number(price, "price", max_value=ValidationLimits.MAX_PRICE)
        
        if not isinstance(buy_signal, bool):
            raise ValidationError(f"buy_signal must be boolean, got {type(buy_signal).__name__}")
        
        if not isinstance(sell_signal, bool):
            raise ValidationError(f"sell_signal must be boolean, got {type(sell_signal).__name__}")
        
        if not isinstance(shares, int) or shares < 0:
            raise ValidationError(f"shares must be non-negative integer, got {shares}")
        
        if buy_signal and sell_signal:
            raise ValidationError("Cannot have both buy_signal and sell_signal True")

        # Execute trades using the new interface
        if buy_signal and shares > 0:
            success = self.buy(self.default_symbol, shares, price, date)
            if not success:
                raise ValidationError(
                    f"Insufficient cash for purchase. Need ${shares * price:.2f}, have ${self.cash:.2f}"
                )
        elif sell_signal and shares > 0:
            shares_sold = self.sell(self.default_symbol, shares, price, date)
            if shares_sold < shares:
                available_shares = self._positions.get(self.default_symbol, Position(self.default_symbol)).shares
                raise ValidationError(
                    f"Cannot sell {shares} shares, only have {available_shares}"
                )
        
        # Update position prices and record portfolio value
        self.update_position_price(self.default_symbol, price)
        self._record_portfolio_value(date)

    # ==================== ENHANCED INTERFACE ====================
    
    def buy(
        self, 
        symbol: str, 
        shares: int, 
        price: float, 
        date: Optional[pd.Timestamp] = None
    ) -> bool:
        """
        Buy shares of a specified asset.
        
        Args:
            symbol: Asset symbol to buy
            shares: Number of shares to buy
            price: Price per share
            date: Transaction date (defaults to current timestamp)
            
        Returns:
            True if purchase successful, False if insufficient funds
            
        Raises:
            ValidationError: If parameters are invalid
        """
        # Validation
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValidationError("Symbol must be a non-empty string")
        
        validate_integer(shares, "shares", min_value=ValidationLimits.MIN_SHARES, max_value=ValidationLimits.MAX_SHARES)
        validate_positive_number(price, "price", max_value=ValidationLimits.MAX_PRICE)
        
        if date is None:
            date = pd.Timestamp.now()
        elif not isinstance(date, pd.Timestamp):
            raise ValidationError(f"Date must be pd.Timestamp, got {type(date).__name__}")
        
        # Calculate costs
        trade_value = shares * price
        commission = trade_value * self.commission_rate
        total_cost = trade_value + commission
        
        # Check available cash
        if total_cost > self.cash:
            return False  # Insufficient funds
        
        # Execute trade
        self.cash -= total_cost
        self._total_commissions += commission
        
        # Update position
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        
        position = self._positions[symbol]
        
        position.shares += shares
        position.total_cost += trade_value
        position.avg_cost = position.total_cost / position.shares
        position.last_price = price
        
        # Record transaction
        transaction = Transaction(
            date=date,
            symbol=symbol,
            action='BUY',
            shares=shares,
            price=price,
            value=trade_value,
            commission=commission
        )
        self._transactions.append(transaction)
        
        return True
    
    def sell(
        self, 
        symbol: str, 
        shares: int, 
        price: float, 
        date: Optional[pd.Timestamp] = None
    ) -> int:
        """
        Sell shares of a specified asset.
        
        Args:
            symbol: Asset symbol to sell
            shares: Number of shares to sell
            price: Price per share
            date: Transaction date (defaults to current timestamp)
            
        Returns:
            Number of shares actually sold
            
        Raises:
            ValidationError: If parameters are invalid
        """
        # Validation
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValidationError("Symbol must be a non-empty string")
        
        validate_integer(shares, "shares", min_value=ValidationLimits.MIN_SHARES, max_value=ValidationLimits.MAX_SHARES)
        validate_positive_number(price, "price", max_value=ValidationLimits.MAX_PRICE)
        
        if date is None:
            date = pd.Timestamp.now()
        elif not isinstance(date, pd.Timestamp):
            raise ValidationError(f"Date must be pd.Timestamp, got {type(date).__name__}")
        
        # Check position exists and has sufficient shares
        if symbol not in self._positions:
            return 0  # No position to sell
        
        position = self._positions[symbol]
        shares_to_sell = min(shares, position.shares)
        
        if shares_to_sell <= 0:
            return 0  # No shares to sell
        
        # Calculate proceeds
        trade_value = shares_to_sell * price
        commission = trade_value * self.commission_rate
        net_proceeds = trade_value - commission
        
        # Calculate realized P&L
        cost_basis = position.avg_cost * shares_to_sell
        realized_pnl = trade_value - cost_basis - commission
        self._realized_pnl += realized_pnl
        
        # Execute trade
        self.cash += net_proceeds
        self._total_commissions += commission
        
        # Update position
        position.shares -= shares_to_sell
        position.total_cost = position.avg_cost * position.shares  # Recalculate total cost
        position.last_price = price
        
        # Remove position if fully sold
        if position.shares == 0:
            del self._positions[symbol]
        
        # Record transaction
        transaction = Transaction(
            date=date,
            symbol=symbol,
            action='SELL',
            shares=shares_to_sell,
            price=price,
            value=trade_value,
            commission=commission
        )
        self._transactions.append(transaction)
        
        return shares_to_sell
    
    def update_position_price(self, symbol: str, price: float) -> None:
        """
        Update the last known price for a position.
        
        Args:
            symbol: Asset symbol
            price: Current price per share
        """
        validate_positive_number(price, "price", max_value=ValidationLimits.MAX_PRICE)
        
        if symbol in self._positions:
            self._positions[symbol].last_price = price
    
    def update_all_prices(self, prices: Dict[str, float]) -> None:
        """
        Update prices for multiple positions.
        
        Args:
            prices: Dictionary mapping symbol to current price
        """
        for symbol, price in prices.items():
            self.update_position_price(symbol, price)
    
    def get_total_value(self, prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            prices: Optional price overrides for positions
            
        Returns:
            Total portfolio value (cash + positions)
        """
        total_value = self.cash
        
        for symbol, position in self._positions.items():
            if prices and symbol in prices:
                price = prices[symbol]
            else:
                price = position.last_price
            
            total_value += position.shares * price
        
        return total_value

    # ==================== ANALYTICS AND REPORTING ====================
    
    def get_positions_summary(self) -> pd.DataFrame:
        """
        Get detailed summary of all positions.
        
        Returns:
            DataFrame with position details
        """
        if not self._positions:
            return pd.DataFrame(columns=[
                'symbol', 'shares', 'avg_cost', 'total_cost', 'last_price', 
                'market_value', 'unrealized_pnl', 'unrealized_pnl_percent'
            ])
        
        data = []
        for position in self._positions.values():
            data.append({
                'symbol': position.symbol,
                'shares': position.shares,
                'avg_cost': position.avg_cost,
                'total_cost': position.total_cost,
                'last_price': position.last_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_percent': position.unrealized_pnl_percent
            })
        
        return pd.DataFrame(data)
    
    def get_transaction_history(self) -> pd.DataFrame:
        """
        Get complete transaction history.
        
        Returns:
            DataFrame with all transactions
        """
        if not self._transactions:
            return pd.DataFrame(columns=[
                'date', 'symbol', 'action', 'shares', 'price', 'value', 'commission'
            ])
        
        data = []
        for txn in self._transactions:
            data.append({
                'date': txn.date,
                'symbol': txn.symbol,
                'action': txn.action,
                'shares': txn.shares,
                'price': txn.price,
                'value': txn.value,
                'commission': txn.commission
            })
        
        return pd.DataFrame(data)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        total_value = self.get_total_value()
        total_return = total_value - self.start_capital
        total_return_pct = (total_return / self.start_capital) * TradingConstants.PERCENT_MULTIPLIER
        
        # Calculate unrealized P&L
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self._positions.values())
        
        # Total P&L
        total_pnl = self._realized_pnl + unrealized_pnl
        
        return {
            'start_capital': self.start_capital,
            'current_cash': self.cash,
            'total_value': total_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'realized_pnl': self._realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl,
            'total_commissions': self._total_commissions,
            'num_positions': len(self._positions),
            'num_transactions': len(self._transactions)
        }
    
    def get_allocation(self) -> Dict[str, float]:
        """
        Calculate portfolio allocation percentages.
        
        Returns:
            Dictionary mapping asset/cash to allocation percentage
        """
        total_value = self.get_total_value()
        if total_value <= 0:
            return {}
        
        allocation = {'CASH': (self.cash / total_value) * TradingConstants.PERCENT_MULTIPLIER}
        
        for symbol, position in self._positions.items():
            market_value = position.market_value
            allocation[symbol] = (market_value / total_value) * TradingConstants.PERCENT_MULTIPLIER
        
        return allocation

    # ==================== INTERNAL METHODS ====================
    
    def _record_portfolio_value(self, date: pd.Timestamp) -> None:
        """
        Record portfolio value for the given date.
        
        Args:
            date: Date to record value for
        """
        total_value = self.get_total_value()
        self.current_value = total_value
        
        self._value_history.append({
            'date': date,
            'value': total_value,
            'cash': self.cash,
            'positions_value': total_value - self.cash
        })
    
    def _get_total_value(self, current_price: float) -> float:
        """
        Legacy method: Calculate total value with single asset price.
        
        Args:
            current_price: Current price of the default asset
            
        Returns:
            Total portfolio value
        """
        if self.default_symbol in self._positions:
            position = self._positions[self.default_symbol]
            return self.cash + (position.shares * current_price)
        else:
            return self.cash

    # ==================== STRING REPRESENTATIONS ====================
    
    def __str__(self) -> str:
        """String representation of portfolio."""
        total_value = self.get_total_value()
        num_positions = len(self._positions)
        return f"Portfolio(value=${total_value:.2f}, cash=${self.cash:.2f}, positions={num_positions})"
    
    def __repr__(self) -> str:
        """Detailed representation of portfolio."""
        return (f"Portfolio(start_capital={self.start_capital}, "
                f"commission_rate={self.commission_rate}, "
                f"default_symbol='{self.default_symbol}')") 