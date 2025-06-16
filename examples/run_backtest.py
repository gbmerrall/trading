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
from backtest.benchmarks import create_standard_benchmarks
from backtest.config import (
    ConfigFactory, set_config, get_config, FileConfig,
    load_config_from_environment, save_config_to_file
)

def print_metrics(title: str, metrics: Dict[str, Any]):
    """
    Print a formatted summary of performance metrics.
    
    Args:
        title: Title for the metrics section
        metrics: Dictionary containing performance metrics
    """
    print(f"\n--- {title} ---")
    print(f"  Total Return:     {metrics.get('total_return', 0.0):.2f}%")
    print(f"  Win Rate:         {metrics.get('win_rate', 0.0):.2f}%")
    print(f"  Number of Trades: {metrics.get('num_trades', 0)}")
    print(f"  Maximum Drawdown: {metrics.get('max_drawdown', 0.0):.2f}%")

def main():
    """
    Main function demonstrating configuration usage and backtest execution.
    
    Shows different configuration approaches:
    1. Default configuration
    2. Preset configurations (conservative, aggressive)
    3. Environment-based configuration
    4. Custom configuration
    """
    # Example 1: Using default configuration
    print("=== Example 1: Default Configuration ===")
    default_config = ConfigFactory.create_default()
    set_config(default_config)
    run_backtest_example("BHP", "Default")
    
    # Example 2: Using conservative preset
    print("\n=== Example 2: Conservative Configuration ===")
    conservative_config = ConfigFactory.create_conservative()
    set_config(conservative_config)
    run_backtest_example("BHP", "Conservative")
    
    # Example 3: Using aggressive preset
    print("\n=== Example 3: Aggressive Configuration ===")
    aggressive_config = ConfigFactory.create_aggressive()
    set_config(aggressive_config)
    run_backtest_example("BHP", "Aggressive")
    
    # Example 4: Custom configuration
    print("\n=== Example 4: Custom Configuration ===")
    custom_config = ConfigFactory.create_default()
    
    # Customize settings
    custom_config.portfolio.start_capital = 25000.0
    custom_config.portfolio.commission_rate = 0.005  # 0.5% commission
    custom_config.strategy.consecutive_days = 4
    custom_config.benchmark.market_symbol = 'QQQ'  # Use NASDAQ instead of SPY
    custom_config.benchmark.dca_frequency = 'weekly'
    custom_config.backtest.plot_title = "Custom Strategy vs QQQ"
    
    set_config(custom_config)
    run_backtest_example("AAPL", "Custom")
    
    # Example 5: Save and load configuration
    print("\n=== Example 5: Configuration Persistence ===")
    
    # Save current configuration
    output_dir = FileConfig.get_output_dir()
    config_file = output_dir / "example_config.json"
    save_config_to_file(get_config(), str(config_file))
    print(f"Configuration saved to: {config_file}")
    
    # You could also load from environment variables
    # env_config = load_config_from_environment()

def run_backtest_example(symbol: str, config_name: str):
    """
    Run a backtest example with the current configuration.
    
    Args:
        symbol: Stock symbol to backtest
        config_name: Name of the configuration being used
    """
    try:
        config = get_config()
        
        # Define date range
        start_date = "2020-01-01"
        end_date = "2023-01-01"
        
        # Load data for the primary symbol
        print(f"Downloading data for {symbol}...")
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        if data.empty:
            print(f"No data loaded for {symbol}. Skipping.")
            return

        # Create strategy with configuration
        strategy = ConsecutiveDaysStrategy()  # Uses config default
        
        # Create benchmarks with configuration
        benchmarks = create_standard_benchmarks()  # Uses config defaults
        
        print(f"Strategy: {config.strategy.consecutive_days} consecutive days")
        print(f"Benchmark symbol: {config.benchmark.market_symbol}")
        print(f"DCA frequency: {config.benchmark.dca_frequency}")
        print(f"Starting capital: ${config.portfolio.start_capital:,.2f}")
        print(f"Commission rate: {config.portfolio.commission_rate:.3%}")

        # Initialize runner
        runner = BacktestRunnerImpl(strategy=strategy, benchmarks=benchmarks)

        # Create output file with configuration name
        output_dir = FileConfig.get_output_dir()
        timestamp = datetime.now().strftime(FileConfig.TIMESTAMP_FORMAT)
        output_file = output_dir / f"{symbol}_{config_name}_{timestamp}{FileConfig.DEFAULT_IMAGE_EXTENSION}"

        # Run backtest
        print(f"\n--- Running {config_name} backtest for {symbol} ---")
        print(f"Period: {start_date} to {end_date}")
        
        results = runner.run(
            data, 
            start_capital=config.backtest.start_capital,
            output_file=str(output_file) if config.backtest.save_plots else None
        )

        # Print comprehensive results
        print_metrics("Strategy Performance", results["strategy_metrics"])
        
        for name, metrics in results["benchmark_metrics"].items():
            print_metrics(f"{name} Benchmark", metrics)

        if config.backtest.save_plots:
            print(f"Equity curve plot saved to: {output_file}")
        
        print(f"Total trades executed: {results['strategy_metrics']['num_trades']}")

    except Exception as e:
        print(f"Error running {config_name} backtest: {str(e)}")

if __name__ == "__main__":
    main()