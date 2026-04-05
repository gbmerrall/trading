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
