import pytest
import tempfile
from pathlib import Path

# Add project root to the Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.config import (
    TradingConstants, ValidationLimits, FileConfig,
    PortfolioConfig, StrategyConfig, BenchmarkConfig, BacktestConfig, GlobalConfig,
    ConfigFactory, ConfigManager,
    get_config, set_config, load_config_from_file, save_config_to_file
)


class TestTradingConstants:
    """Test trading constants."""
    
    def test_trading_days(self):
        """Test trading day constants."""
        assert TradingConstants.TRADING_DAYS_PER_YEAR == 252
        assert TradingConstants.TRADING_DAYS_PER_MONTH == 21
        assert TradingConstants.TRADING_DAYS_PER_WEEK == 5
    
    def test_default_values(self):
        """Test default values."""
        assert TradingConstants.DEFAULT_MARKET_BENCHMARK == 'SPY'
        assert TradingConstants.PERCENT_MULTIPLIER == 100.0
        assert TradingConstants.DEFAULT_CUMULATIVE_RETURN_BASE == 1.0


class TestValidationLimits:
    """Test validation limits."""
    
    def test_portfolio_limits(self):
        """Test portfolio validation limits."""
        assert ValidationLimits.MAX_COMMISSION_RATE == 0.10
        assert ValidationLimits.MIN_START_CAPITAL == 1.0
        assert ValidationLimits.MAX_START_CAPITAL == 1e12
    
    def test_strategy_limits(self):
        """Test strategy validation limits."""
        assert ValidationLimits.MIN_CONSECUTIVE_DAYS == 1
        assert ValidationLimits.MAX_CONSECUTIVE_DAYS == 252


class TestPortfolioConfig:
    """Test portfolio configuration."""
    
    def test_default_values(self):
        """Test default portfolio configuration values."""
        config = PortfolioConfig()
        assert config.start_capital == 10_000.0
        assert config.commission_rate == 0.0
        assert config.default_symbol == 'ASSET'
        assert config.track_detailed_history is True
    
    def test_validation_success(self):
        """Test successful validation."""
        config = PortfolioConfig(start_capital=50000.0, commission_rate=0.001)
        config.validate()  # Should not raise
    
    def test_validation_failure(self):
        """Test validation failures."""
        # Invalid start capital
        config = PortfolioConfig(start_capital=-1000.0)
        with pytest.raises(ValueError):
            config.validate()
        
        # Invalid commission rate
        config = PortfolioConfig(commission_rate=0.2)  # 20%
        with pytest.raises(ValueError):
            config.validate()


class TestStrategyConfig:
    """Test strategy configuration."""
    
    def test_default_values(self):
        """Test default strategy configuration values."""
        config = StrategyConfig()
        assert config.consecutive_days == 3
        assert config.require_minimum_data is True
        assert config.use_full_capital is True
    
    def test_validation_success(self):
        """Test successful validation."""
        config = StrategyConfig(consecutive_days=5)
        config.validate()  # Should not raise
    
    def test_validation_failure(self):
        """Test validation failures."""
        config = StrategyConfig(consecutive_days=0)
        with pytest.raises(ValueError):
            config.validate()
        
        config = StrategyConfig(consecutive_days=300)
        with pytest.raises(ValueError):
            config.validate()


class TestBenchmarkConfig:
    """Test benchmark configuration."""
    
    def test_default_values(self):
        """Test default benchmark configuration values."""
        config = BenchmarkConfig()
        assert config.market_symbol == 'SPY'
        assert config.dca_frequency == 'monthly'
        assert config.download_timeout == 30
        assert config.cache_benchmark_data is False
    
    def test_validation_success(self):
        """Test successful validation."""
        config = BenchmarkConfig(market_symbol='QQQ', dca_frequency='weekly')
        config.validate()  # Should not raise
    
    def test_validation_failure(self):
        """Test validation failures."""
        config = BenchmarkConfig(dca_frequency='yearly')
        with pytest.raises(ValueError):
            config.validate()


class TestBacktestConfig:
    """Test backtest configuration."""
    
    def test_default_values(self):
        """Test default backtest configuration values."""
        config = BacktestConfig()
        assert config.start_capital == 10_000.0
        assert config.verbose is True
        assert config.save_plots is True
        assert config.plot_title == "Equity Curves"
        assert config.output_format == 'png'
    
    def test_validation_success(self):
        """Test successful validation."""
        config = BacktestConfig(start_capital=25000.0, output_format='svg')
        config.validate()  # Should not raise
    
    def test_validation_failure(self):
        """Test validation failures."""
        config = BacktestConfig(start_capital=-5000.0)
        with pytest.raises(ValueError):
            config.validate()
        
        config = BacktestConfig(output_format='gif')
        with pytest.raises(ValueError):
            config.validate()


class TestGlobalConfig:
    """Test global configuration."""
    
    def test_default_creation(self):
        """Test creating default global configuration."""
        config = GlobalConfig()
        assert isinstance(config.portfolio, PortfolioConfig)
        assert isinstance(config.strategy, StrategyConfig)
        assert isinstance(config.benchmark, BenchmarkConfig)
        assert isinstance(config.backtest, BacktestConfig)
        assert config.environment == 'development'
    
    def test_validation(self):
        """Test global configuration validation."""
        config = GlobalConfig()
        config.validate()  # Should not raise
    
    def test_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            'portfolio': {'start_capital': 25000.0, 'commission_rate': 0.001},
            'strategy': {'consecutive_days': 5},
            'benchmark': {'market_symbol': 'QQQ'},
            'backtest': {'verbose': False},
            'environment': 'production'
        }
        
        config = GlobalConfig.from_dict(config_dict)
        assert config.portfolio.start_capital == 25000.0
        assert config.strategy.consecutive_days == 5
        assert config.benchmark.market_symbol == 'QQQ'
        assert config.backtest.verbose is False
        assert config.environment == 'production'


class TestConfigFactory:
    """Test configuration factory."""
    
    def test_create_default(self):
        """Test creating default configuration."""
        config = ConfigFactory.create_default()
        assert isinstance(config, GlobalConfig)
        assert config.portfolio.start_capital == 10_000.0
        assert config.environment == 'development'
    
    def test_create_conservative(self):
        """Test creating conservative configuration."""
        config = ConfigFactory.create_conservative()
        assert config.portfolio.start_capital == 50_000.0
        assert config.portfolio.commission_rate == 0.001
        assert config.strategy.consecutive_days == 5
        assert config.backtest.calculate_sharpe_ratio is True
    
    def test_create_aggressive(self):
        """Test creating aggressive configuration."""
        config = ConfigFactory.create_aggressive()
        assert config.portfolio.commission_rate == 0.0
        assert config.strategy.consecutive_days == 2
        assert config.benchmark.dca_frequency == 'daily'
    
    def test_create_testing(self):
        """Test creating testing configuration."""
        config = ConfigFactory.create_testing()
        assert config.environment == 'testing'
        assert config.portfolio.start_capital == 1_000.0
        assert config.backtest.save_plots is False
        assert config.debug is True
    
    def test_create_production(self):
        """Test creating production configuration."""
        config = ConfigFactory.create_production()
        assert config.environment == 'production'
        assert config.portfolio.commission_rate == 0.005
        assert config.log_level == 'WARNING'


class TestConfigManager:
    """Test configuration manager."""
    
    def test_singleton(self):
        """Test that ConfigManager is a singleton."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2
    
    def test_default_config(self):
        """Test getting default configuration."""
        manager = ConfigManager()
        manager.reset()  # Ensure clean state
        
        config = manager.get_config()
        assert isinstance(config, GlobalConfig)
    
    def test_set_and_get_config(self):
        """Test setting and getting configuration."""
        manager = ConfigManager()
        test_config = ConfigFactory.create_conservative()
        
        manager.set_config(test_config)
        retrieved_config = manager.get_config()
        
        assert retrieved_config.portfolio.start_capital == 50_000.0
        assert retrieved_config.strategy.consecutive_days == 5


class TestConfigPersistence:
    """Test configuration file operations."""
    
    def test_save_and_load_json(self):
        """Test saving and loading JSON configuration."""
        config = ConfigFactory.create_conservative()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save configuration
            save_config_to_file(config, temp_file)
            
            # Load configuration
            loaded_config = load_config_from_file(temp_file)
            
            # Verify loaded configuration matches original
            assert loaded_config.portfolio.start_capital == config.portfolio.start_capital
            assert loaded_config.strategy.consecutive_days == config.strategy.consecutive_days
            assert loaded_config.benchmark.market_symbol == config.benchmark.market_symbol
            
        finally:
            os.unlink(temp_file)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_config_from_file('nonexistent.json')
    
    def test_invalid_file_format(self):
        """Test loading from invalid file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_file = f.name
            f.write("invalid content")
        
        try:
            with pytest.raises(ValueError, match="Unsupported configuration file format"):
                load_config_from_file(temp_file)
        finally:
            os.unlink(temp_file)


class TestGlobalFunctions:
    """Test global configuration functions."""
    
    def test_get_set_config(self):
        """Test global get/set configuration functions."""
        original_config = get_config()
        
        test_config = ConfigFactory.create_aggressive()
        set_config(test_config)
        
        retrieved_config = get_config()
        assert retrieved_config.strategy.consecutive_days == 2
        
        # Reset to original
        set_config(original_config)
    
    def test_component_getters(self):
        """Test component-specific getter functions."""
        from backtest.config import (
            get_portfolio_config, get_strategy_config, 
            get_benchmark_config, get_backtest_config
        )
        
        # Set test configuration
        test_config = ConfigFactory.create_conservative()
        set_config(test_config)
        
        # Test component getters
        portfolio_config = get_portfolio_config()
        assert portfolio_config.start_capital == 50_000.0
        
        strategy_config = get_strategy_config()
        assert strategy_config.consecutive_days == 5
        
        benchmark_config = get_benchmark_config()
        assert benchmark_config.market_symbol == 'SPY'
        
        backtest_config = get_backtest_config()
        assert backtest_config.calculate_sharpe_ratio is True


class TestFileConfig:
    """Test file configuration utilities."""
    
    def test_allowed_extensions(self):
        """Test allowed file extensions."""
        assert '.png' in FileConfig.ALLOWED_IMAGE_EXTENSIONS
        assert '.jpg' in FileConfig.ALLOWED_IMAGE_EXTENSIONS
        assert '.csv' in FileConfig.ALLOWED_DATA_EXTENSIONS
    
    def test_get_output_dir(self):
        """Test getting output directory."""
        output_dir = FileConfig.get_output_dir()
        assert isinstance(output_dir, Path)
        assert output_dir.exists()  # Should be created if it doesn't exist 