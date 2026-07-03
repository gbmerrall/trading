"""Tests for the Three-Tier tournament refactoring in trade_analysis.py.

Verifies:
- The two RegimeFilteredStrategy entries are in STRATEGIES.
- WFA_PARAM_GRIDS is keyed by strategy name (str), not class.
- Standard entries map name → plain param dict.
- Regime wrapper entries map name → {"param_grid": ..., "base_params": ...}.
- EnsembleStrategy names are absent from WFA_PARAM_GRIDS (skip path).
- run_wfa accepts a base_params keyword argument.
- The WFA loop routing helper correctly distinguishes wrapper vs standalone configs.
"""

import importlib
import inspect
import sys

import pytest

from backtest.strategy import (
    BreakoutStrategy,
    EnsembleStrategy,
    RegimeFilteredStrategy,
)


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

def _load():
    """Import trade_analysis, patching sys.argv so the ticker guard passes."""
    orig = sys.argv[:]
    sys.argv = ["trade_analysis.py", "SPY"]
    try:
        import trade_analysis
        importlib.reload(trade_analysis)
        return trade_analysis
    finally:
        sys.argv = orig


@pytest.fixture(scope="module")
def ta():
    return _load()


# ---------------------------------------------------------------------------
# STRATEGIES list — regime wrapper presence
# ---------------------------------------------------------------------------

class TestStrategiesListRegimeEntries:
    def test_adx_wrapper_present(self, ta):
        names = [n for n, _ in ta.STRATEGIES]
        assert "Breakout(20d)+ADX(25)" in names

    def test_sma_wrapper_present(self, ta):
        names = [n for n, _ in ta.STRATEGIES]
        assert "Breakout(20d)+SMA(200)" in names

    def test_adx_entry_is_regime_filtered_strategy(self, ta):
        inst = next(i for n, i in ta.STRATEGIES if n == "Breakout(20d)+ADX(25)")
        assert isinstance(inst, RegimeFilteredStrategy)

    def test_sma_entry_is_regime_filtered_strategy(self, ta):
        inst = next(i for n, i in ta.STRATEGIES if n == "Breakout(20d)+SMA(200)")
        assert isinstance(inst, RegimeFilteredStrategy)

    def test_adx_wrapper_wraps_breakout(self, ta):
        inst = next(i for n, i in ta.STRATEGIES if n == "Breakout(20d)+ADX(25)")
        assert isinstance(inst.base_strategy, BreakoutStrategy)

    def test_sma_wrapper_wraps_breakout(self, ta):
        inst = next(i for n, i in ta.STRATEGIES if n == "Breakout(20d)+SMA(200)")
        assert isinstance(inst.base_strategy, BreakoutStrategy)

    def test_adx_threshold_is_25(self, ta):
        inst = next(i for n, i in ta.STRATEGIES if n == "Breakout(20d)+ADX(25)")
        assert inst.adx_threshold == pytest.approx(25.0)

    def test_sma_period_is_200(self, ta):
        inst = next(i for n, i in ta.STRATEGIES if n == "Breakout(20d)+SMA(200)")
        assert inst.sma_period == 200


# ---------------------------------------------------------------------------
# WFA_PARAM_GRIDS — keyed by name, not by class
# ---------------------------------------------------------------------------

class TestWfaParamGridsKeyedByName:
    def test_all_keys_are_strings(self, ta):
        for key in ta.WFA_PARAM_GRIDS:
            assert isinstance(key, str), f"Key {key!r} is not a string"

    def test_no_class_keys(self, ta):
        for key in ta.WFA_PARAM_GRIDS:
            assert not isinstance(key, type), f"Key {key!r} is still a class"

    def test_standard_strategy_name_present(self, ta):
        assert "Breakout(20d)" in ta.WFA_PARAM_GRIDS

    def test_standard_entry_is_plain_dict(self, ta):
        config = ta.WFA_PARAM_GRIDS["Breakout(20d)"]
        assert "param_grid" not in config, "Standard entry must be a plain param dict"
        assert "base_params" not in config
        assert "lookback_period" in config

    def test_adx_wrapper_entry_present(self, ta):
        assert "Breakout(20d)+ADX(25)" in ta.WFA_PARAM_GRIDS

    def test_adx_wrapper_has_param_grid_key(self, ta):
        config = ta.WFA_PARAM_GRIDS["Breakout(20d)+ADX(25)"]
        assert "param_grid" in config

    def test_adx_wrapper_has_base_params_key(self, ta):
        config = ta.WFA_PARAM_GRIDS["Breakout(20d)+ADX(25)"]
        assert "base_params" in config

    def test_adx_wrapper_param_grid_contains_lookback(self, ta):
        config = ta.WFA_PARAM_GRIDS["Breakout(20d)+ADX(25)"]
        assert "lookback_period" in config["param_grid"]

    def test_adx_wrapper_param_grid_contains_adx_threshold(self, ta):
        config = ta.WFA_PARAM_GRIDS["Breakout(20d)+ADX(25)"]
        assert "adx_threshold" in config["param_grid"]

    def test_adx_base_params_has_regime_type_adx(self, ta):
        config = ta.WFA_PARAM_GRIDS["Breakout(20d)+ADX(25)"]
        assert config["base_params"].get("regime_type") == "ADX"

    def test_sma_wrapper_entry_present(self, ta):
        assert "Breakout(20d)+SMA(200)" in ta.WFA_PARAM_GRIDS

    def test_sma_wrapper_param_grid_contains_sma_period(self, ta):
        config = ta.WFA_PARAM_GRIDS["Breakout(20d)+SMA(200)"]
        assert "sma_period" in config["param_grid"]

    def test_sma_base_params_has_regime_type_sma(self, ta):
        config = ta.WFA_PARAM_GRIDS["Breakout(20d)+SMA(200)"]
        assert config["base_params"].get("regime_type") == "SMA"

    def test_ensemble_names_absent_from_wfa_grids(self, ta):
        ensemble_names = [n for n, i in ta.STRATEGIES if isinstance(i, EnsembleStrategy)]
        for name in ensemble_names:
            assert name not in ta.WFA_PARAM_GRIDS, (
                f"Ensemble '{name}' must not be in WFA_PARAM_GRIDS"
            )


# ---------------------------------------------------------------------------
# run_wfa — accepts base_params keyword argument
# ---------------------------------------------------------------------------

class TestRunWfaSignature:
    def test_run_wfa_accepts_base_params(self, ta):
        sig = inspect.signature(ta.run_wfa)
        assert "base_params" in sig.parameters, (
            "run_wfa must accept a base_params keyword argument"
        )

    def test_base_params_defaults_to_none(self, ta):
        sig = inspect.signature(ta.run_wfa)
        param = sig.parameters["base_params"]
        assert param.default is None


# ---------------------------------------------------------------------------
# WFA routing logic — name-keyed lookup produces correct (param_grid, base_params)
# ---------------------------------------------------------------------------

class TestWfaRoutingLogic:
    """Verify the config-unpacking rules the main() loop must implement."""

    def _unpack(self, ta, name: str) -> tuple[dict, dict | None]:
        """Replicate the routing logic from main()."""
        config = ta.WFA_PARAM_GRIDS.get(name)
        if config is None:
            return None, None
        if "param_grid" in config and "base_params" in config:
            return config["param_grid"], config["base_params"]
        return config, None

    def test_unknown_name_returns_none(self, ta):
        param_grid, base_params = self._unpack(ta, "NonExistentStrategy")
        assert param_grid is None

    def test_standard_strategy_returns_grid_and_no_base_params(self, ta):
        param_grid, base_params = self._unpack(ta, "Breakout(20d)")
        assert param_grid is not None
        assert base_params is None

    def test_adx_wrapper_returns_grid_and_base_params(self, ta):
        param_grid, base_params = self._unpack(ta, "Breakout(20d)+ADX(25)")
        assert param_grid is not None
        assert base_params is not None

    def test_sma_wrapper_returns_grid_and_base_params(self, ta):
        param_grid, base_params = self._unpack(ta, "Breakout(20d)+SMA(200)")
        assert param_grid is not None
        assert base_params is not None

    def test_adx_base_params_contain_base_strategy(self, ta):
        _, base_params = self._unpack(ta, "Breakout(20d)+ADX(25)")
        assert "base_strategy" in base_params

    def test_sma_base_params_contain_base_strategy(self, ta):
        _, base_params = self._unpack(ta, "Breakout(20d)+SMA(200)")
        assert "base_strategy" in base_params
