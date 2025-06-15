import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from backtest.strategies import ConsecutiveDaysStrategy
from backtest.benchmarks import BuyAndHold, SPYBuyAndHold, DollarCostAveraging
from backtest.runner import BacktestRunnerImpl

def main():
    # Download historical data
    symbol = 'BHP'
    data = yf.download(symbol, start='2020-01-01', end='2024-01-01', auto_adjust=True, progress=False)
    
    if data.empty:
        print(f"Could not download data for {symbol}")
        return
    
    # Create strategy and benchmarks
    strategy = ConsecutiveDaysStrategy(up_days=3, down_days=3)
    benchmarks = [
        BuyAndHold(),
        SPYBuyAndHold(),
        DollarCostAveraging(frequency='monthly', amount=1000.0)
    ]
    
    # Create backtest runner
    runner = BacktestRunnerImpl(strategy, benchmarks)
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Generate output filename with symbol and current date/time
    timestamp = datetime.now().strftime('%Y-%m-%dT%H%M')
    output_file = f'output/{symbol}_{timestamp}.png'
    
    # Run backtest
    results = runner.run(data, start_capital=10000.0, output_file=output_file)
    
    # Print results
    print("\nStrategy Performance:")
    metrics = results['strategy_metrics']
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
    
    print("\nBenchmark Performance:")
    for benchmark_name, benchmark_metrics in results['benchmark_metrics'].items():
        print(f"\n{benchmark_name}:")
        total_return = benchmark_metrics['total_return']
        if isinstance(total_return, pd.Series):
            total_return = total_return.iloc[-1]
        win_rate = benchmark_metrics['win_rate']
        if isinstance(win_rate, pd.Series):
            win_rate = win_rate.iloc[-1]
        max_drawdown = benchmark_metrics['max_drawdown']
        if isinstance(max_drawdown, pd.Series):
            max_drawdown = max_drawdown.iloc[-1]
        print(f"Total Return: {total_return:.2f}%")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")

if __name__ == '__main__':
    main() 