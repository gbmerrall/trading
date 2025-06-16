import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any
import warnings

# Ignore the specific pandas FutureWarning about downcasting
warnings.simplefilter(action='ignore', category=FutureWarning)

# Add the project root directory to the Python path
# This allows us to import from the 'backtest' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.runner import BacktestRunnerImpl
from backtest.strategy import ConsecutiveDaysStrategy
from backtest.benchmarks import BuyAndHold, SPYBuyAndHold, DollarCostAveraging

def print_metrics(title: str, metrics: Dict[str, Any]):
    """Prints a formatted summary of performance metrics."""
    print(f"\n--- {title} ---")
    print(f"  Total Return:     {metrics.get('total_return', 0.0):.2f}%")
    print(f"  Win Rate:         {metrics.get('win_rate', 0.0):.2f}%")
    print(f"  Number of Trades: {metrics.get('num_trades', 0)}")
    print(f"  Maximum Drawdown: {metrics.get('max_drawdown', 0.0):.2f}%")

def main():
    """
    Main function to run the backtest for a single symbol against multiple benchmarks.
    """
    # Define parameters
    symbol = "BHP"
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    start_capital = 10000.0

    # Load data for the primary symbol
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
    
    if data.empty:
        print(f"No data loaded for {symbol}. Exiting.")
        return

    # Define strategy and benchmarks
    strategy = ConsecutiveDaysStrategy(consecutive_days=3)
    benchmarks = [
        BuyAndHold(), 
        SPYBuyAndHold(), 
        DollarCostAveraging(frequency='monthly')
    ]

    # Initialize runner
    runner = BacktestRunnerImpl(strategy=strategy, benchmarks=benchmarks)

    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Define output file path with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%dT%H%M')
    output_file = os.path.join(output_dir, f"{symbol}_{timestamp}.png")

    # Run backtest
    print(f"--- Running backtest for {symbol} ---")
    results = runner.run(data, start_capital=start_capital, output_file=output_file)

    # Print metrics
    print_metrics("Strategy", results["strategy_metrics"])
    for name, metrics in results["benchmark_metrics"].items():
        print_metrics(name, metrics)

    print(f"\nEquity curve plot saved to {output_file}\n")


if __name__ == "__main__":
    main()