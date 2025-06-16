import pandas as pd

class Portfolio:
    """
    Manages the state of a trading portfolio, including cash, positions,
    and the history of its total value.
    """
    def __init__(self, start_capital: float = 10000.0):
        """Initializes the portfolio."""
        self.start_capital = start_capital
        self.cash = start_capital
        self.positions = {}
        self._value_history = []

    def get_value_history(self) -> list:
        """Returns the recorded history of the portfolio's value."""
        return self._value_history

    def _get_total_value(self, current_price: float) -> float:
        """Calculates the current total value of the portfolio (cash + assets)."""
        asset_value = self.positions.get('asset', 0) * current_price
        return self.cash + asset_value

    def process_day(self, date: pd.Timestamp, price: float, buy_signal: bool = False, sell_signal: bool = False, shares: int = 0):
        """
        Processes the portfolio state for a single day based on signals.
        For simplicity, this portfolio only handles one asset, named 'asset'.
        """
        if buy_signal:
            cost = shares * price
            if self.cash >= cost:
                self.cash -= cost
                self.positions['asset'] = self.positions.get('asset', 0) + shares

        elif sell_signal:
            # We assume we can only sell shares we own.
            shares_to_sell = min(shares, self.positions.get('asset', 0))
            if shares_to_sell > 0:
                revenue = shares_to_sell * price
                self.cash += revenue
                self.positions['asset'] -= shares_to_sell
                if self.positions['asset'] == 0:
                    del self.positions['asset']
        
        # Record the total portfolio value at the end of the day
        current_value = self._get_total_value(price)
        self._value_history.append({'date': date, 'value': current_value}) 