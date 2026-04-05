"""Walk-Forward Analysis: search strategies, window splitting, and optimizer.

Public API:
    GridSearch          — exhaustive Cartesian product search
    RandomSearch        — random sampling of parameter space
    WalkForwardOptimizer — orchestrates WFA across sliding/anchored windows
    WalkForwardResult   — dataclass holding all WFA output
"""

import itertools
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd

from .metrics import METRICS, MetricFn
from .strategy import BaseStrategy
from .validation import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Results from a Walk-Forward Analysis run.

    Attributes:
        equity_curve: Stitched out-of-sample equity curve as a pd.Series
            with DatetimeIndex. Values are portfolio value at each date.
        windows: pd.DataFrame with one row per test window. Columns:
            train_start, train_end, test_start, test_end, best_params (dict),
            objective_score, n_trades, total_return, cagr, sharpe_ratio,
            sortino_ratio, calmar_ratio, max_drawdown, ulcer_index,
            profit_factor, win_rate, expectancy, recovery_factor.
        summary: dict of aggregate metrics computed over the full equity_curve,
            plus n_windows, n_windows_with_trades, param_stability.
        best_params_overall: The parameter set selected most often across
            windows. Ties broken by highest mean objective score.
    """

    equity_curve: pd.Series
    windows: pd.DataFrame
    summary: dict
    best_params_overall: dict


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


def _run_candidate_worker(
    strategy_class: type,
    params: dict,
    train_data: pd.DataFrame,
    warmup_n: int,
    min_trades: int,
    objective_key: str,
) -> float:
    """Evaluate one parameter set against a training window.

    Module-level so ProcessPoolExecutor can pickle it. Returns the objective
    score, or float('-inf') on any failure.

    Args:
        strategy_class: BaseStrategy subclass to instantiate.
        params: Constructor kwargs for strategy_class.
        train_data: Training window DataFrame.
        warmup_n: Bars to exclude from the start when scoring.
        min_trades: Minimum trades required to score (returns -inf otherwise).
        objective_key: Key into METRICS for the objective function.

    Returns:
        Objective score as a float.
    """
    from backtest.runner import BacktestRunnerImpl
    from backtest.metrics import METRICS

    try:
        strategy = strategy_class(**params)
        runner = BacktestRunnerImpl(strategy=strategy, benchmarks=[])
        result = runner.run(data=train_data, start_capital=None)

        portfolio_history = [
            {"date": d, "value": v}
            for d, v in result["strategy_returns"].items()
        ]
        trades = result["trades"]

        if warmup_n > 0 and len(train_data) > warmup_n:
            cutoff = train_data.index[warmup_n]
            portfolio_history, trades = _filter_by_warmup(portfolio_history, trades, cutoff)

        if len(trades) < min_trades:
            return float("-inf")

        return METRICS[objective_key](portfolio_history, trades)
    except Exception:
        return float("-inf")


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
        searcher: "GridSearch | RandomSearch | None" = None,
        objective: "str | MetricFn" = "sharpe_ratio",
        min_trades: int = 5,
        n_jobs: int = 1,
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

        if n_jobs == 0 or n_jobs < -1:
            raise ValidationError(
                f"n_jobs must be 1 (sequential), -1 (all CPUs), or a positive integer, got {n_jobs}"
            )

        self.strategy_class = strategy_class
        self.param_space = param_space
        self.data = data
        self.train_size = train_size
        self.test_size = test_size
        self.window_type = window_type
        self.searcher = searcher if searcher is not None else GridSearch()
        self._objective_fn = objective_fn
        # Store the original string key so the parallel worker can re-resolve it.
        # None when objective was a custom callable (parallel falls back to sequential).
        self._objective_key: str | None = objective if isinstance(objective, str) else None
        self.min_trades = min_trades
        self.n_jobs = n_jobs

    def _run_train_window(self, train_data: pd.DataFrame) -> dict:
        """Find the best-scoring parameter set for a training window.

        Evaluates all candidates from the searcher, applies warmup filtering,
        and scores with the objective function. When n_jobs != 1 and the
        objective is a named metric, candidates are evaluated in parallel using
        ProcessPoolExecutor. Falls back to sequential for custom callables.

        Args:
            train_data: Sliced training data with DatetimeIndex.

        Returns:
            Dict with keys "best_params" (dict) and "objective_score" (float).
        """
        candidates = self.searcher.generate(self.param_space)
        if not candidates:
            return {"best_params": {}, "objective_score": float("-inf")}

        # Warmup period is the same for all candidates of the same strategy class.
        warmup_n = self.strategy_class(**candidates[0]).warmup_period

        actual_jobs = os.cpu_count() or 1 if self.n_jobs == -1 else self.n_jobs
        use_parallel = actual_jobs != 1 and self._objective_key is not None

        if use_parallel and self._objective_key is None:
            logger.warning(
                "n_jobs=%s requested but objective is a custom callable; "
                "falling back to sequential evaluation.",
                self.n_jobs,
            )

        if use_parallel:
            return self._run_train_window_parallel(
                candidates, train_data, warmup_n, actual_jobs
            )
        return self._run_train_window_sequential(candidates, train_data, warmup_n)

    def _run_train_window_sequential(
        self,
        candidates: list[dict],
        train_data: pd.DataFrame,
        warmup_n: int,
    ) -> dict:
        """Sequential candidate evaluation (original behaviour)."""
        from .runner import BacktestRunnerImpl

        best_params = candidates[0]
        best_score = float("-inf")

        for params in candidates:
            strategy = self.strategy_class(**params)
            runner = BacktestRunnerImpl(strategy=strategy, benchmarks=[])

            try:
                result = runner.run(data=train_data, start_capital=None)
            except Exception as exc:
                logger.warning(
                    "Candidate skipped during training (params=%s): %s", params, exc
                )
                continue

            portfolio_history = [
                {"date": d, "value": v}
                for d, v in result["strategy_returns"].items()
            ]
            trades = result["trades"]

            if warmup_n > 0 and len(train_data) > warmup_n:
                cutoff = train_data.index[warmup_n]
                portfolio_history, trades = _filter_by_warmup(
                    portfolio_history, trades, cutoff
                )

            score = (
                float("-inf")
                if len(trades) < self.min_trades
                else self._objective_fn(portfolio_history, trades)
            )

            if score > best_score:
                best_score = score
                best_params = params

        return {"best_params": best_params, "objective_score": best_score}

    def _run_train_window_parallel(
        self,
        candidates: list[dict],
        train_data: pd.DataFrame,
        warmup_n: int,
        n_workers: int,
    ) -> dict:
        """Parallel candidate evaluation using ProcessPoolExecutor."""
        best_params = candidates[0]
        best_score = float("-inf")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_params = {
                executor.submit(
                    _run_candidate_worker,
                    self.strategy_class,
                    params,
                    train_data,
                    warmup_n,
                    self.min_trades,
                    self._objective_key,
                ): params
                for params in candidates
            }
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    score = future.result()
                except Exception as exc:
                    logger.warning(
                        "Parallel candidate failed (params=%s): %s", params, exc
                    )
                    score = float("-inf")

                if score > best_score:
                    best_score = score
                    best_params = params

        return {"best_params": best_params, "objective_score": best_score}

    def run(self) -> WalkForwardResult:
        """Execute Walk-Forward Analysis and return the composite result.

        For each window pair:
        1. Optimise parameters on the training window.
        2. Evaluate the best parameters on the test window with warmup filtering.
        3. Record per-window metrics and the equity curve segment.

        Returns:
            WalkForwardResult with equity_curve, windows, summary, best_params_overall.
        """
        from .runner import BacktestRunnerImpl
        import backtest.metrics as _metrics

        windows = _generate_windows(
            self.data, self.train_size, self.test_size, self.window_type
        )

        all_equity_segments: list[pd.Series] = []
        all_trades: list[dict] = []
        window_rows: list[dict] = []

        metric_names = [
            "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "max_drawdown", "ulcer_index",
            "profit_factor", "win_rate", "expectancy", "recovery_factor",
        ]

        for train_data, test_data in windows:
            train_result = self._run_train_window(train_data)
            best_params = train_result["best_params"]
            objective_score = train_result["objective_score"]

            # --- Evaluate on test window ---
            strategy = self.strategy_class(**best_params)
            warmup_n = strategy.warmup_period
            runner = BacktestRunnerImpl(strategy=strategy, benchmarks=[])

            try:
                test_run = runner.run(data=test_data, start_capital=None)
            except Exception as exc:
                logger.warning(
                    "Test window failed (params=%s, test_start=%s): %s",
                    best_params,
                    test_data.index[0],
                    exc,
                )
                # If the test window fails, record a zero-trade window
                row = {
                    "train_start": train_data.index[0],
                    "train_end": train_data.index[-1],
                    "test_start": test_data.index[0],
                    "test_end": test_data.index[-1],
                    "best_params": best_params,
                    "objective_score": objective_score,
                    "n_trades": 0,
                    **{m: float("-inf") for m in metric_names},
                }
                window_rows.append(row)
                continue

            test_equity = test_run["strategy_returns"]
            test_portfolio_history = [
                {"date": d, "value": v} for d, v in test_equity.items()
            ]
            test_trades = test_run["trades"]

            # Apply warmup buffer to scoring
            if warmup_n > 0 and len(test_data) > warmup_n:
                cutoff = test_data.index[warmup_n]
                scored_history, scored_trades = _filter_by_warmup(
                    test_portfolio_history, test_trades, cutoff
                )
                scored_equity = test_equity[test_equity.index >= cutoff]
            else:
                scored_history, scored_trades = test_portfolio_history, test_trades
                scored_equity = test_equity

            # Calculate all 10 metrics for this window
            row = {
                "train_start": train_data.index[0],
                "train_end": train_data.index[-1],
                "test_start": test_data.index[0],
                "test_end": test_data.index[-1],
                "best_params": best_params,
                "objective_score": objective_score,
                "n_trades": len(scored_trades),
            }
            for m_name in metric_names:
                row[m_name] = _metrics.METRICS[m_name](scored_history, scored_trades)

            window_rows.append(row)
            all_equity_segments.append(scored_equity)
            all_trades.extend(scored_trades)

        # --- Stitch results ---
        if not all_equity_segments:
            raise ValidationError("WFA produced no results across all windows")

        # Composite equity curve: just concatenate segments
        composite_equity = pd.concat(all_equity_segments)
        # Remove duplicate dates at window boundaries
        composite_equity = composite_equity[~composite_equity.index.duplicated(keep="first")]

        # Summary Metrics
        summary_history = [
            {"date": d, "value": v} for d, v in composite_equity.items()
        ]
        summary = {}
        for m_name in metric_names:
            summary[m_name] = _metrics.METRICS[m_name](summary_history, all_trades)

        windows_df = pd.DataFrame(window_rows)
        
        summary["n_windows"] = len(windows_df)
        summary["n_windows_with_trades"] = int((windows_df["n_trades"] > 0).sum())

        # Best Params Overall (Mode)
        param_series = windows_df["best_params"].apply(lambda d: tuple(sorted(d.items())))
        mode_params_tuple = param_series.mode()[0]
        best_params_overall = dict(mode_params_tuple)
        
        summary["param_stability"] = float((param_series == mode_params_tuple).mean())

        return WalkForwardResult(
            equity_curve=composite_equity,
            windows=windows_df,
            summary=summary,
            best_params_overall=best_params_overall,
        )
