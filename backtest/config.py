"""
Configuration Management System

This module provides centralized configuration for all trading backtest components,
replacing magic numbers with named constants and providing configurable defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


# ==================== TRADING CONSTANTS ====================

class TradingConstants:
    """Constants related to trading and financial markets."""
    
    # Market timing
    TRADING_DAYS_PER_YEAR = 252
    TRADING_DAYS_PER_MONTH = 21
    TRADING_DAYS_PER_WEEK = 5
    
    # Common market symbols
    DEFAULT_MARKET_BENCHMARK = 'SPY'
    ALTERNATIVE_BENCHMARKS = ['QQQ', 'IWM', 'DIA']
    
    # Percentage calculations
    PERCENT_MULTIPLIER = 100.0
    BASIS_POINTS_MULTIPLIER = 10000.0
    
    # Default ratios
    DEFAULT_CUMULATIVE_RETURN_BASE = 1.0
    EXTREME_PRICE_VARIATION_THRESHOLD = 1000  # max/min price ratio


# ==================== VALIDATION LIMITS ====================

class ValidationLimits:
    """Limits and thresholds for input validation."""
    
    # Portfolio limits
    MAX_COMMISSION_RATE = 0.10  # 10% maximum commission rate
    MIN_START_CAPITAL = 1.0     # Minimum starting capital
    MAX_START_CAPITAL = 1e12    # Maximum starting capital (1 trillion)
    
    # Strategy limits
    MIN_CONSECUTIVE_DAYS = 1
    MAX_CONSECUTIVE_DAYS = TradingConstants.TRADING_DAYS_PER_YEAR
    
    # Position limits
    MIN_SHARES = 1
    MAX_SHARES = 1_000_000
    MIN_PRICE = 0.01  # Minimum price (1 cent)
    MAX_PRICE = 1_000_000.0  # Maximum price per share
    
    # Data quality thresholds
    MAX_PRICE_VARIATION_RATIO = 1000  # Price max/min ratio threshold
    MIN_DATA_POINTS = 2  # Minimum data points for backtesting


# ==================== FILE AND PATH CONFIGURATION ====================

class FileConfig:
    """Configuration for file handling and paths."""
    
    # Output settings
    DEFAULT_OUTPUT_DIR = 'output'
    ALLOWED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']
    ALLOWED_DATA_EXTENSIONS = ['.csv', '.json', '.parquet']
    DEFAULT_IMAGE_EXTENSION = '.png'
    
    # File naming
    TIMESTAMP_FORMAT = '%Y-%m-%dT%H%M'
    DEFAULT_PLOT_DPI = 300
    DEFAULT_PLOT_WIDTH = 12
    DEFAULT_PLOT_HEIGHT = 8
    
    @staticmethod
    def get_output_dir() -> Path:
        """Get the output directory path, creating it if necessary."""
        output_path = Path(FileConfig.DEFAULT_OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        return output_path


# ==================== PORTFOLIO CONFIGURATION ====================

@dataclass
class PortfolioConfig:
    """Configuration for portfolio management."""
    
    # Default portfolio settings
    start_capital: float = 10_000.0
    commission_rate: float = 0.0
    default_symbol: str = 'ASSET'
    
    # Position management
    max_position_size: Optional[float] = None  # None = no limit
    max_positions: Optional[int] = None        # None = no limit
    
    # Risk management
    max_drawdown_limit: Optional[float] = None  # Stop trading if exceeded
    max_daily_loss: Optional[float] = None      # Daily loss limit
    
    # Reporting
    track_detailed_history: bool = True
    calculate_metrics: bool = True
    
    def validate(self) -> None:
        """Validate portfolio configuration."""
        if self.start_capital < ValidationLimits.MIN_START_CAPITAL:
            raise ValueError(f"start_capital must be >= {ValidationLimits.MIN_START_CAPITAL}")
        
        if self.start_capital > ValidationLimits.MAX_START_CAPITAL:
            raise ValueError(f"start_capital must be <= {ValidationLimits.MAX_START_CAPITAL}")
        
        if self.commission_rate > ValidationLimits.MAX_COMMISSION_RATE:
            raise ValueError(f"commission_rate must be <= {ValidationLimits.MAX_COMMISSION_RATE}")


# ==================== STRATEGY CONFIGURATION ====================

@dataclass
class StrategyConfig:
    """Configuration for trading strategies."""
    
    # Strategy parameters
    consecutive_days: int = 3
    
    # Signal generation
    require_minimum_data: bool = True
    minimum_data_points: int = ValidationLimits.MIN_CONSECUTIVE_DAYS
    
    # Position sizing
    use_full_capital: bool = True
    position_size_method: str = 'fixed_dollar'  # 'fixed_dollar', 'fixed_shares', 'percentage'
    
    def validate(self) -> None:
        """Validate strategy configuration."""
        if not (ValidationLimits.MIN_CONSECUTIVE_DAYS <= self.consecutive_days <= ValidationLimits.MAX_CONSECUTIVE_DAYS):
            raise ValueError(
                f"consecutive_days must be between {ValidationLimits.MIN_CONSECUTIVE_DAYS} "
                f"and {ValidationLimits.MAX_CONSECUTIVE_DAYS}"
            )


# ==================== BENCHMARK CONFIGURATION ====================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark strategies."""
    
    # Market benchmark settings
    market_symbol: str = TradingConstants.DEFAULT_MARKET_BENCHMARK
    download_timeout: int = 30  # seconds
    retry_attempts: int = 3
    
    # DCA settings
    dca_frequency: str = 'monthly'
    valid_frequencies: List[str] = field(default_factory=lambda: ['daily', 'weekly', 'monthly'])
    
    # Caching
    cache_benchmark_data: bool = False
    cache_duration_hours: int = 24
    
    def validate(self) -> None:
        """Validate benchmark configuration."""
        if self.dca_frequency not in self.valid_frequencies:
            raise ValueError(f"dca_frequency must be one of {self.valid_frequencies}")


# ==================== BACKTEST RUNNER CONFIGURATION ====================

@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    
    # Execution settings
    start_capital: float = 10_000.0
    verbose: bool = True
    save_plots: bool = True
    save_data: bool = False
    
    # Plot settings
    plot_title: str = "Equity Curves"
    plot_colors: Dict[str, str] = field(default_factory=lambda: {
        'strategy': 'blue',
        'benchmark1': 'green', 
        'benchmark2': 'red',
        'benchmark3': 'orange',
        'benchmark4': 'purple'
    })
    
    # Performance calculation
    calculate_sharpe_ratio: bool = False
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    
    # Output settings
    output_format: str = 'png'
    include_timestamp: bool = True
    
    def validate(self) -> None:
        """Validate backtest configuration."""
        if self.start_capital <= 0:
            raise ValueError("start_capital must be positive")
        
        if self.output_format not in ['png', 'jpg', 'svg', 'pdf']:
            raise ValueError(f"Invalid output_format: {self.output_format}")


# ==================== GLOBAL CONFIGURATION ====================

@dataclass
class GlobalConfig:
    """Global configuration combining all component configurations."""
    
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    # Environment settings
    environment: str = 'development'  # 'development', 'testing', 'production'
    debug: bool = False
    log_level: str = 'INFO'
    
    # Data source settings
    data_source: str = 'yfinance'
    data_cache_enabled: bool = True
    
    def validate(self) -> None:
        """Validate all configuration components."""
        self.portfolio.validate()
        self.strategy.validate()
        self.benchmark.validate()
        self.backtest.validate()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GlobalConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'portfolio' in config_dict:
            config.portfolio = PortfolioConfig(**config_dict['portfolio'])
        
        if 'strategy' in config_dict:
            config.strategy = StrategyConfig(**config_dict['strategy'])
        
        if 'benchmark' in config_dict:
            config.benchmark = BenchmarkConfig(**config_dict['benchmark'])
        
        if 'backtest' in config_dict:
            config.backtest = BacktestConfig(**config_dict['backtest'])
        
        # Set global settings
        for key, value in config_dict.items():
            if key not in ['portfolio', 'strategy', 'benchmark', 'backtest']:
                if hasattr(config, key):
                    setattr(config, key, value)
        
        config.validate()
        return config


# ==================== CONFIGURATION FACTORY ====================

class ConfigFactory:
    """Factory for creating different configuration presets."""
    
    @staticmethod
    def create_default() -> GlobalConfig:
        """Create default configuration."""
        return GlobalConfig()
    
    @staticmethod
    def create_conservative() -> GlobalConfig:
        """Create conservative trading configuration."""
        config = GlobalConfig()
        config.portfolio.start_capital = 50_000.0
        config.portfolio.commission_rate = 0.001  # 0.1% commission
        config.strategy.consecutive_days = 5  # More conservative
        config.backtest.calculate_sharpe_ratio = True
        return config
    
    @staticmethod
    def create_aggressive() -> GlobalConfig:
        """Create aggressive trading configuration."""
        config = GlobalConfig()
        config.portfolio.start_capital = 10_000.0
        config.portfolio.commission_rate = 0.0  # No commission
        config.strategy.consecutive_days = 2  # More aggressive
        config.benchmark.dca_frequency = 'daily'
        return config
    
    @staticmethod
    def create_testing() -> GlobalConfig:
        """Create configuration for testing."""
        config = GlobalConfig()
        config.environment = 'testing'
        config.portfolio.start_capital = 1_000.0
        config.backtest.save_plots = False
        config.backtest.verbose = False
        config.debug = True
        return config
    
    @staticmethod
    def create_production() -> GlobalConfig:
        """Create production configuration."""
        config = GlobalConfig()
        config.environment = 'production'
        config.portfolio.commission_rate = 0.005  # Realistic commission
        config.data_cache_enabled = True
        config.backtest.calculate_sharpe_ratio = True
        config.log_level = 'WARNING'
        return config


# ==================== ENVIRONMENT-BASED CONFIGURATION ====================

def load_config_from_environment() -> GlobalConfig:
    """Load configuration from environment variables."""
    config = GlobalConfig()
    
    # Portfolio settings
    if os.getenv('BACKTEST_START_CAPITAL'):
        config.portfolio.start_capital = float(os.getenv('BACKTEST_START_CAPITAL'))
    
    if os.getenv('BACKTEST_COMMISSION_RATE'):
        config.portfolio.commission_rate = float(os.getenv('BACKTEST_COMMISSION_RATE'))
    
    # Strategy settings
    if os.getenv('STRATEGY_CONSECUTIVE_DAYS'):
        config.strategy.consecutive_days = int(os.getenv('STRATEGY_CONSECUTIVE_DAYS'))
    
    # Benchmark settings
    if os.getenv('BENCHMARK_SYMBOL'):
        config.benchmark.market_symbol = os.getenv('BENCHMARK_SYMBOL')
    
    # Environment
    if os.getenv('BACKTEST_ENVIRONMENT'):
        config.environment = os.getenv('BACKTEST_ENVIRONMENT')
    
    if os.getenv('BACKTEST_DEBUG'):
        config.debug = os.getenv('BACKTEST_DEBUG').lower() == 'true'
    
    config.validate()
    return config


# ==================== SINGLETON CONFIGURATION MANAGER ====================

class ConfigManager:
    """Singleton configuration manager for global access."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def set_config(self, config: GlobalConfig) -> None:
        """Set the global configuration."""
        config.validate()
        self._config = config
    
    def get_config(self) -> GlobalConfig:
        """Get the global configuration."""
        if self._config is None:
            self._config = ConfigFactory.create_default()
        return self._config
    
    def reset(self) -> None:
        """Reset configuration to default."""
        self._config = None


# Global configuration manager instance
config_manager = ConfigManager()


# ==================== CONVENIENCE FUNCTIONS ====================

def get_config() -> GlobalConfig:
    """Get the current global configuration."""
    return config_manager.get_config()


def set_config(config: GlobalConfig) -> None:
    """Set the global configuration."""
    config_manager.set_config(config)


def get_portfolio_config() -> PortfolioConfig:
    """Get portfolio configuration."""
    return get_config().portfolio


def get_strategy_config() -> StrategyConfig:
    """Get strategy configuration."""
    return get_config().strategy


def get_benchmark_config() -> BenchmarkConfig:
    """Get benchmark configuration."""
    return get_config().benchmark


def get_backtest_config() -> BacktestConfig:
    """Get backtest configuration."""
    return get_config().backtest


# ==================== CONFIGURATION LOADING ====================

def load_config_from_file(file_path: str) -> GlobalConfig:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    import json
    from pathlib import Path
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    if path.suffix.lower() == '.json':
        with open(path, 'r') as f:
            config_dict = json.load(f)
    elif path.suffix.lower() in ['.yaml', '.yml']:
        try:
            import yaml
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required for YAML configuration files")
    else:
        raise ValueError(f"Unsupported configuration file format: {path.suffix}")
    
    return GlobalConfig.from_dict(config_dict)


def save_config_to_file(config: GlobalConfig, file_path: str) -> None:
    """
    Save configuration to JSON or YAML file.
    
    Args:
        config: Configuration to save
        file_path: Path to save configuration
    """
    import json
    from pathlib import Path
    import dataclasses
    
    def serialize_config(obj):
        """Custom serializer for dataclasses."""
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix.lower() == '.json':
        with open(path, 'w') as f:
            json.dump(config, f, default=serialize_config, indent=2)
    elif path.suffix.lower() in ['.yaml', '.yml']:
        try:
            import yaml
            with open(path, 'w') as f:
                yaml.dump(dataclasses.asdict(config), f, default_flow_style=False)
        except ImportError:
            raise ImportError("PyYAML is required for YAML configuration files")
    else:
        raise ValueError(f"Unsupported configuration file format: {path.suffix}") 