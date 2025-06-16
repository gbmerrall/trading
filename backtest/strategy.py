import pandas as pd

class ConsecutiveDaysStrategy:
    """
    A strategy that generates a buy signal after a number of consecutive
    down days and a sell signal after a number of consecutive up days.
    """
    def __init__(self, consecutive_days: int = 3):
        """
        Initialize the strategy.
        :param consecutive_days: The number of consecutive days for the signal.
        """
        if consecutive_days < 1:
            raise ValueError("consecutive_days must be at least 1")
        self.consecutive_days = consecutive_days

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy and sell signals.
        """
        signals = pd.DataFrame(index=data.index, dtype=bool)
        signals['buy'] = False
        signals['sell'] = False

        # Calculate price changes
        price_change = data['Close'].diff()

        # Buy signal: 'consecutive_days' of down days
        down_days = price_change < 0
        signals['buy'] = down_days.rolling(window=self.consecutive_days).sum() == self.consecutive_days

        # Sell signal: 'consecutive_days' of up days
        up_days = price_change > 0
        signals['sell'] = up_days.rolling(window=self.consecutive_days).sum() == self.consecutive_days

        # Shift signals to trade on the next day's open
        signals = signals.shift(1).infer_objects(copy=False).fillna(False)
        return signals 