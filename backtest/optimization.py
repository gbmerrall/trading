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
