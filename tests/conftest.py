"""
pytest configuration and fixtures for the trading backtest test suite.
"""

import pytest


@pytest.fixture(autouse=True)
def reset_config():
    """
    Reset the ConfigManager singleton between tests to ensure test isolation.

    This fixture runs automatically before and after every test to prevent
    configuration state from leaking between tests.
    """
    from backtest.config import config_manager

    config_manager.reset()
    yield
    config_manager.reset()
