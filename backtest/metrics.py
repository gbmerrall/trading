"""Performance metric functions for the trading backtest framework.

All functions share the signature:
    (portfolio_history: list[dict], trades: list[dict]) -> float

portfolio_history entries: {"date": pd.Timestamp, "value": float}
trades entries: {"entry_date", "exit_date", "entry", "exit", "shares", "pnl"}

Sign convention:
- Higher is always better.
- Metrics where lower is worse (drawdown, ulcer index) are returned as
  negative floats so the optimizer can maximise uniformly.
- float('-inf') is returned for degenerate inputs (empty history, zero
  trades where trades are required) to ensure such windows are never
  selected as best.
"""

import math
from typing import Callable

import numpy as np
import pandas as pd

from .constants import TradingConstants

# Type alias for metric functions
MetricFn = Callable[[list[dict], list[dict]], float]

TRADING_DAYS = TradingConstants.TRADING_DAYS_PER_YEAR


def total_return(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return (final_value - initial_value) / initial_value.

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        Return as a fraction (0.10 for 10%). float('-inf') if history empty.
    """
    if not portfolio_history:
        return float("-inf")
    start = portfolio_history[0]["value"]
    end = portfolio_history[-1]["value"]
    if start == 0:
        return float("-inf")
    return (end - start) / start


def cagr(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return annualised growth rate using 252 trading days per year.

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        CAGR as a fraction (0.20 for 20%). float('-inf') if fewer than 2 bars.
    """
    if len(portfolio_history) < 2:
        return float("-inf")
    start = portfolio_history[0]["value"]
    end = portfolio_history[-1]["value"]
    n_bars = len(portfolio_history) - 1
    if start <= 0 or end <= 0:
        return float("-inf")
    return (end / start) ** (TRADING_DAYS / n_bars) - 1


def max_drawdown(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return largest peak-to-trough decline as a negative fraction.

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        Negative fraction (e.g. -0.20 for 20% drawdown). 0.0 if no drawdown.
        float('-inf') if history empty.
    """
    if not portfolio_history:
        return float("-inf")
    values = pd.Series([e["value"] for e in portfolio_history], dtype=float)
    peak = values.expanding().max()
    # Handle case where all values are 0
    if (peak == 0).all():
        return 0.0
    drawdowns = (values - peak) / peak
    return float(drawdowns.min())


def ulcer_index(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return the negated Ulcer Index (RMS of percentage drawdown depths).

    The Ulcer Index is the square root of the mean squared percentage
    drawdown from the running peak. It is negated so higher is better.

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        Negated Ulcer Index (<= 0). float('-inf') if history empty.
    """
    if not portfolio_history:
        return float("-inf")
    values = pd.Series([e["value"] for e in portfolio_history], dtype=float)
    peak = values.expanding().max()
    # Handle case where all values are 0
    if (peak == 0).all():
        return 0.0
    pct_drawdown = (values - peak) / peak  # <= 0
    ui = math.sqrt(float((pct_drawdown ** 2).mean()))
    return -ui


def sharpe_ratio(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return annualised Sharpe ratio (mean daily return / std daily return).

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        Annualised Sharpe ratio. float('-inf') if fewer than 2 bars or std = 0.
    """
    if len(portfolio_history) < 2:
        return float("-inf")
    values = pd.Series([e["value"] for e in portfolio_history], dtype=float)
    daily_returns = values.pct_change().dropna()
    if len(daily_returns) < 1:
        return float("-inf")
    
    mean_return = float(daily_returns.mean())
    std = float(daily_returns.std())
    
    if std < 1e-12:
        return 1e6 if mean_return > 0 else float("-inf")
        
    return mean_return / std * math.sqrt(TRADING_DAYS)


def sortino_ratio(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return annualised Sortino ratio (mean daily return / downside deviation).

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        Annualised Sortino ratio. float('-inf') if fewer than 2 bars or
        no negative returns (downside std = 0).
    """
    if len(portfolio_history) < 2:
        return float("-inf")
    values = pd.Series([e["value"] for e in portfolio_history], dtype=float)
    daily_returns = values.pct_change().dropna()
    if len(daily_returns) < 1:
        return float("-inf")
    
    mean_return = float(daily_returns.mean())
    negative_returns = daily_returns[daily_returns < 0]
    
    if len(negative_returns) == 0:
        return 1e6 if mean_return > 0 else float("-inf")
        
    downside_std = float(negative_returns.std())
    
    if downside_std < 1e-12:
        return 1e6 if mean_return > 0 else float("-inf")
        
    return mean_return / downside_std * math.sqrt(TRADING_DAYS)


def calmar_ratio(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return CAGR / abs(max_drawdown).

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        Calmar ratio. float('inf') if no drawdown. float('-inf') if insufficient data.
    """
    if len(portfolio_history) < 2:
        return float("-inf")
    c = cagr(portfolio_history, trades)
    if c == float("-inf"):
        return float("-inf")
    dd = max_drawdown(portfolio_history, trades)
    if dd == 0.0:
        return float("inf")
    return c / abs(dd)


def profit_factor(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return gross winning P&L / gross losing P&L.

    Args:
        portfolio_history: Unused; present for interface consistency.
        trades: List of trade dicts with 'pnl' key.

    Returns:
        Profit factor. float('inf') if all trades win. 0.0 if all lose.
        float('-inf') if no trades.
    """
    if not trades:
        return float("-inf")
    gross_win = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = sum(abs(t["pnl"]) for t in trades if t["pnl"] < 0)
    if gross_loss == 0:
        return float("inf")
    return gross_win / gross_loss


def win_rate(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return fraction of trades that are profitable.

    Args:
        portfolio_history: Unused; present for interface consistency.
        trades: List of trade dicts with 'pnl' key.

    Returns:
        Win rate as a fraction (0.0 to 1.0). float('-inf') if no trades.
    """
    if not trades:
        return float("-inf")
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return wins / len(trades)


def expectancy(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return mean P&L per trade.

    Args:
        portfolio_history: Unused; present for interface consistency.
        trades: List of trade dicts with 'pnl' key.

    Returns:
        Mean P&L per trade in currency units. float('-inf') if no trades.
    """
    if not trades:
        return float("-inf")
    return sum(t["pnl"] for t in trades) / len(trades)


def recovery_factor(portfolio_history: list[dict], trades: list[dict]) -> float:
    """Return total_return / abs(max_drawdown).

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: Unused; present for interface consistency.

    Returns:
        Recovery factor. float('inf') if no drawdown. float('-inf') if insufficient data.
    """
    if not portfolio_history:
        return float("-inf")
    tr = total_return(portfolio_history, trades)
    if tr == float("-inf"):
        return float("-inf")
    dd = max_drawdown(portfolio_history, trades)
    if dd == 0.0:
        return float("inf")
    return tr / abs(dd)


METRICS: dict[str, MetricFn] = {
    "total_return":    total_return,
    "cagr":            cagr,
    "sharpe_ratio":    sharpe_ratio,
    "sortino_ratio":   sortino_ratio,
    "calmar_ratio":    calmar_ratio,
    "max_drawdown":    max_drawdown,
    "ulcer_index":     ulcer_index,
    "profit_factor":   profit_factor,
    "win_rate":        win_rate,
    "expectancy":      expectancy,
    "recovery_factor": recovery_factor,
}
