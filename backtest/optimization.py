"""Walk-Forward Analysis: search strategies, window splitting, and optimizer.

Public API:
    GridSearch          — exhaustive Cartesian product search
    RandomSearch        — random sampling of parameter space
    WalkForwardOptimizer — orchestrates WFA across sliding/anchored windows
    WalkForwardResult   — dataclass holding all WFA output
"""

import itertools
import random
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, Callable

import numpy as np
import pandas as pd

from .metrics import METRICS, MetricFn
from .strategy import BaseStrategy
from .validation import ValidationError


class GridSearch:
    """Exhaustive search over all combinations of a parameter space.

    Args:
        None

    Example:
        gs = GridSearch()
        combos = gs.generate({"a": [1, 2], "b": [10, 20]})
        # → [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20}]
    """

    def generate(self, param_space: dict[str, list]) -> list[dict]:
        """Return all combinations of param_space values.

        Args:
            param_space: Mapping of parameter names to lists of candidate values.

        Returns:
            List of dicts, one per combination.
        """
        if not param_space:
            return []
            
        keys = list(param_space.keys())
        values = list(param_space.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


class RandomSearch:
    """Random sampling of parameter combinations.

    Samples n combinations uniformly at random. If n exceeds the total number
    of unique combinations, samples with replacement.

    Args:
        n: Number of combinations to sample.
        seed: Random seed for reproducibility.

    Example:
        rs = RandomSearch(n=50, seed=42)
        combos = rs.generate({"period": [7, 14, 21], "threshold": [0.05, 0.10]})
    """

    def __init__(self, n: int, seed: int = 42):
        """
        Args:
            n: Number of combinations to sample.
            seed: Random seed for reproducibility.
        """
        self.n = n
        self.seed = seed

    def generate(self, param_space: dict[str, list]) -> list[dict]:
        """Return n randomly sampled combinations from param_space.

        Args:
            param_space: Mapping of parameter names to lists of candidate values.

        Returns:
            List of n dicts, each a sampled combination.
        """
        all_combos = GridSearch().generate(param_space)
        if not all_combos:
            return []
            
        rng = random.Random(self.seed)
        if self.n >= len(all_combos):
            return rng.choices(all_combos, k=self.n)
        return rng.sample(all_combos, k=self.n)


def _generate_windows(
    data: pd.DataFrame,
    train_size: int,
    test_size: int,
    window_type: str,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split data into (train, test) window pairs.

    Args:
        data: Full dataset with DatetimeIndex.
        train_size: Number of bars in each training window (sliding mode) or
                    in the initial training window (anchored mode).
        test_size: Number of bars in each test window.
        window_type: "sliding" — fixed train size, both windows advance by test_size.
                     "anchored" — train always starts at index 0, expands each step.

    Returns:
        List of (train_df, test_df) tuples in chronological order.

    Raises:
        ValidationError: If window_type is not "sliding" or "anchored".
    """
    if window_type not in ("sliding", "anchored"):
        raise ValidationError(
            f"window_type must be 'sliding' or 'anchored', got '{window_type}'"
        )

    n = len(data)
    windows = []
    test_start = train_size

    while test_start + test_size <= n:
        test_end = test_start + test_size
        test_df = data.iloc[test_start:test_end]

        if window_type == "sliding":
            train_df = data.iloc[test_start - train_size:test_start]
        else:  # anchored
            train_df = data.iloc[0:test_start]

        windows.append((train_df, test_df))
        test_start += test_size

    return windows


def _filter_by_warmup(
    portfolio_history: list[dict],
    trades: list[dict],
    cutoff_date: pd.Timestamp,
) -> tuple[list[dict], list[dict]]:
    """Remove portfolio history entries and trades before the cutoff date.

    This is used to exclude the warmup period from metric scoring after
    BacktestRunnerImpl has run on the full window slice.

    Args:
        portfolio_history: List of {'date', 'value'} dicts.
        trades: List of trade dicts with 'exit_date' key.
        cutoff_date: First date to INCLUDE in the filtered output.

    Returns:
        Tuple of (filtered_portfolio_history, filtered_trades).
    """
    filtered_history = [e for e in portfolio_history if e["date"] >= cutoff_date]
    filtered_trades = [t for t in trades if t["exit_date"] >= cutoff_date]
    return filtered_history, filtered_trades


class WalkForwardOptimizer:
    """Optimise strategy parameters using Walk-Forward Analysis.

    For each train window, exhaustively or randomly searches the parameter
    space and selects the best-scoring parameter set. Applies those
    parameters to the subsequent out-of-sample test window. Stitches
    test-window equity curves into a composite result.

    Args:
        strategy_class: A BaseStrategy subclass (not an instance).
        param_space: Dict mapping parameter names to lists of candidate values.
        data: Full historical DataFrame with DatetimeIndex and 'Close' column.
        train_size: Number of bars in each training window.
        test_size: Number of bars in each test window.
        window_type: "sliding" or "anchored".
        searcher: GridSearch() or RandomSearch(n, seed). Default: GridSearch().
        objective: Metric name string (key of METRICS) or MetricFn callable.
                   Default: "sharpe_ratio".
        min_trades: Minimum trades required to score a window. Windows with
                    fewer trades receive float('-inf') as their score.
                    Default: 5.

    Example:
        opt = WalkForwardOptimizer(
            strategy_class=RSIStrategy,
            param_space={"period": [7, 14, 21]},
            data=df,
            train_size=252,
            test_size=63,
            window_type="sliding",
            objective="sharpe_ratio",
        )
        result = opt.run()
    """

    def __init__(
        self,
        strategy_class: type,
        param_space: dict[str, list],
        data: pd.DataFrame,
        train_size: int,
        test_size: int,
        window_type: str = "sliding",
        searcher: Union[GridSearch, RandomSearch, None] = None,
        objective: Union[str, MetricFn] = "sharpe_ratio",
        min_trades: int = 5,
    ):
        """
        Args:
            strategy_class: A BaseStrategy subclass (not an instance).
            param_space: Dict mapping parameter names to lists of candidate values.
            data: Full historical DataFrame with DatetimeIndex and 'Close' column.
            train_size: Number of bars in each training window.
            test_size: Number of bars in each test window.
            window_type: "sliding" or "anchored".
            searcher: Search strategy. Default: GridSearch().
            objective: Metric to optimise. Default: "sharpe_ratio".
            min_trades: Minimum trades to score a window. Default: 5.

        Raises:
            ValidationError: If any argument is invalid.
        """
        if train_size < 1:
            raise ValidationError(f"train_size must be >= 1, got {train_size}")
        if test_size < 1:
            raise ValidationError(f"test_size must be >= 1, got {test_size}")
        if train_size + test_size > len(data):
            raise ValidationError(
                f"train_size ({train_size}) + test_size ({test_size}) = "
                f"{train_size + test_size} exceeds data length ({len(data)})"
            )
        if window_type not in ("sliding", "anchored"):
            raise ValidationError(
                f"window_type must be 'sliding' or 'anchored', got '{window_type}'"
            )
        if not (isinstance(strategy_class, type) and issubclass(strategy_class, BaseStrategy)):
            raise ValidationError(
                f"strategy_class must be a subclass of BaseStrategy, got {strategy_class}"
            )
        if isinstance(objective, str):
            if objective not in METRICS:
                raise ValidationError(
                    f"objective '{objective}' not in METRICS. "
                    f"Valid keys: {sorted(METRICS.keys())}"
                )
            objective_fn = METRICS[objective]
        elif callable(objective):
            objective_fn = objective
        else:
            raise ValidationError("objective must be a metric name string or callable")

        self.strategy_class = strategy_class
        self.param_space = param_space
        self.data = data
        self.train_size = train_size
        self.test_size = test_size
        self.window_type = window_type
        self.searcher = searcher if searcher is not None else GridSearch()
        self._objective_fn = objective_fn
        self.min_trades = min_trades

    def _run_train_window(self, train_data: pd.DataFrame) -> dict:
        """Find the best-scoring parameter set for a training window.

        Runs BacktestRunnerImpl for each candidate parameter set, applies
        warmup filtering, and scores with the objective function. Returns
        the parameter set with the highest score.

        Args:
            train_data: Sliced training data with DatetimeIndex.

        Returns:
            Dict with keys "best_params" (dict) and "objective_score" (float).
        """
        from .runner import BacktestRunnerImpl

        candidates = self.searcher.generate(self.param_space)
        if not candidates:
            return {"best_params": {}, "objective_score": float("-inf")}
            
        best_params = candidates[0]
        best_score = float("-inf")

        for params in candidates:
            strategy = self.strategy_class(**params)
            warmup_n = strategy.warmup_period
            runner = BacktestRunnerImpl(strategy=strategy, benchmarks=[])

            try:
                result = runner.run(data=train_data, start_capital=None)
            except Exception:
                continue

            portfolio_history = [
                {"date": d, "value": v}
                for d, v in result["strategy_returns"].items()
            ]
            trades = result["trades"]

            # Apply warmup buffer
            if warmup_n > 0 and len(train_data) > warmup_n:
                cutoff = train_data.index[warmup_n]
                portfolio_history, trades = _filter_by_warmup(
                    portfolio_history, trades, cutoff
                )

            # Apply min_trades guard
            if len(trades) < self.min_trades:
                score = float("-inf")
            else:
                score = self._objective_fn(portfolio_history, trades)

            if score > best_score:
                best_score = score
                best_params = params

        return {"best_params": best_params, "objective_score": best_score}
