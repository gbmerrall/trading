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

from .config import PortfolioConfig
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

    Samples n unique combinations uniformly at random. If n meets or exceeds
    the total number of combinations, every combination is returned exactly once.

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
        """Return up to n unique randomly sampled combinations from param_space.

        Args:
            param_space: Mapping of parameter names to lists of candidate values.

        Returns:
            List of dicts, each a unique sampled combination.
        """
        all_combos = GridSearch().generate(param_space)
        if not all_combos:
            return []

        if self.n >= len(all_combos):
            return all_combos

        rng = random.Random(self.seed)
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


def _carry_in_position(signals_full: pd.DataFrame, window_start) -> bool:
    """Return True if the strategy is long entering the window.

    Determined from the full-context signals: if the most recent buy/sell event
    strictly before window_start is a buy, the strategy would be holding a
    position at the window boundary.

    Args:
        signals_full: Signals over the full context with 'buy'/'sell' columns.
        window_start: First timestamp of the window (excluded from the scan).

    Returns:
        True if long at the boundary, False if flat or no prior events.
    """
    prior = signals_full.loc[signals_full.index < window_start]
    buys = prior.index[prior["buy"].fillna(False).astype(bool)]
    sells = prior.index[prior["sell"].fillna(False).astype(bool)]
    if len(buys) == 0:
        return False
    if len(sells) == 0:
        return True
    return buys[-1] > sells[-1]


def _window_signals_with_carry(
    signals_full: pd.DataFrame, window_data: pd.DataFrame
) -> pd.DataFrame:
    """Slice signals to the window, carrying an open position across the boundary.

    Windows are simulated with a fresh all-cash portfolio. If the strategy was
    long entering the window (its entry event fired before the window started),
    a buy is forced on the first bar so the window re-enters at its opening
    price — otherwise event-based strategies would idle in cash until the next
    entry event, understating exposure. No buy is forced if the first bar
    already carries a sell signal (the position would exit immediately anyway).

    Args:
        signals_full: Signals over the full context with 'buy'/'sell' columns.
        window_data: The window slice of price data (index defines the window).

    Returns:
        A signals DataFrame aligned to window_data.index; the input is not mutated.
    """
    window_signals = signals_full.loc[window_data.index].copy()
    if _carry_in_position(signals_full, window_data.index[0]):
        if not bool(window_signals["sell"].iloc[0]):
            window_signals.iloc[0, window_signals.columns.get_loc("buy")] = True
    return window_signals


def _run_candidate_worker(
    strategy_class: type,
    params: dict,
    train_data: pd.DataFrame,
    context_data: pd.DataFrame,
    min_trades: int,
    objective_key: str,
    base_params: dict | None = None,
    portfolio_config: PortfolioConfig | None = None,
) -> float:
    """Evaluate one parameter set against a training window.

    Module-level so ProcessPoolExecutor can pickle it. Returns the objective
    score, or float('-inf') on any failure.

    Signals are generated on context_data (full history up to train_end) then
    sliced to train_data's index so that lookback indicators have access to data
    before the training window starts. Because of that context, indicators are
    already warm at the window's first bar and no warmup filtering is applied.

    Args:
        strategy_class: BaseStrategy subclass to instantiate.
        params: Constructor kwargs for strategy_class (from param_space).
        train_data: Training window DataFrame (used for portfolio simulation).
        context_data: Full history up to and including the training window end.
            Used for signal generation so lookback indicators are not truncated.
        min_trades: Minimum trades required to score (returns -inf otherwise).
        objective_key: Key into METRICS for the objective function.
        base_params: Fixed constructor kwargs merged with params before instantiation.
        portfolio_config: PortfolioConfig to install in this process's global config
            before running. Required for correctness under ProcessPoolExecutor spawn
            mode, where the parent's config singleton is not inherited.

    Returns:
        Objective score as a float.
    """
    from backtest.runner import BacktestRunnerImpl
    from backtest.metrics import METRICS
    from backtest.config import get_config

    if portfolio_config is not None:
        get_config().portfolio = portfolio_config

    try:
        merged = {**(base_params or {}), **params}
        strategy = strategy_class(**merged)
        signals_full = strategy.generate_signals(context_data)
        train_signals = _window_signals_with_carry(signals_full, train_data)
        runner = BacktestRunnerImpl(strategy=strategy, benchmarks=[])
        result = runner.run(data=train_data, start_capital=None, precomputed_signals=train_signals)

        portfolio_history = [
            {"date": d, "value": v}
            for d, v in result["strategy_returns"].items()
        ]
        trades = result["trades"]

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
        base_params: dict | None = None,
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
            base_params: Fixed constructor kwargs always passed to strategy_class,
                         merged with (and overridden by) each param_space candidate.
                         Use for required args that should not be part of the search.

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
        self.base_params: dict = base_params or {}
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

    def _context_for(self, window_data: pd.DataFrame) -> pd.DataFrame:
        """Return all data up to and including the last date of window_data.

        Used to give signal generation the full historical lookback context even
        when window_data itself is a short slice of self.data.

        Args:
            window_data: A sliced window from self.data.

        Returns:
            self.data.iloc[:pos] where pos is one past the last index of window_data.
        """
        pos = int(self.data.index.searchsorted(window_data.index[-1], side="right"))
        return self.data.iloc[:pos]

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

        actual_jobs = os.cpu_count() or 1 if self.n_jobs == -1 else self.n_jobs
        # Parallel workers re-resolve the objective from its METRICS key, so a
        # custom callable objective always evaluates sequentially.
        use_parallel = actual_jobs != 1 and self._objective_key is not None

        if use_parallel:
            return self._run_train_window_parallel(candidates, train_data, actual_jobs)
        return self._run_train_window_sequential(candidates, train_data)

    def _run_train_window_sequential(
        self,
        candidates: list[dict],
        train_data: pd.DataFrame,
    ) -> dict:
        """Sequential candidate evaluation."""
        from .runner import BacktestRunnerImpl

        best_params = candidates[0]
        best_score = float("-inf")
        context_data = self._context_for(train_data)

        for params in candidates:
            strategy = self.strategy_class(**{**self.base_params, **params})
            runner = BacktestRunnerImpl(strategy=strategy, benchmarks=[])

            try:
                signals_full = strategy.generate_signals(context_data)
                train_signals = _window_signals_with_carry(signals_full, train_data)
                result = runner.run(data=train_data, start_capital=None, precomputed_signals=train_signals)
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
        n_workers: int,
    ) -> dict:
        """Parallel candidate evaluation using ProcessPoolExecutor."""
        best_params = candidates[0]
        best_score = float("-inf")
        context_data = self._context_for(train_data)
        # _run_train_window only calls this path when _objective_key is not None
        assert self._objective_key is not None
        objective_key: str = self._objective_key

        from .config import get_portfolio_config

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_params = {
                executor.submit(
                    _run_candidate_worker,
                    self.strategy_class,
                    params,
                    train_data,
                    context_data,
                    self.min_trades,
                    objective_key,
                    self.base_params,
                    get_portfolio_config(),
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
            # Signals are generated on full history up to test_end so that lookback
            # indicators (e.g. a 100-bar MA) are not truncated by the 63-bar test slice.
            strategy = self.strategy_class(**{**self.base_params, **best_params})
            runner = BacktestRunnerImpl(strategy=strategy, benchmarks=[])

            try:
                test_context = self._context_for(test_data)
                signals_full = strategy.generate_signals(test_context)
                test_signals = _window_signals_with_carry(signals_full, test_data)
                test_run = runner.run(data=test_data, start_capital=None, precomputed_signals=test_signals)
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
            # Signals come from full context, so every test-window bar is live:
            # no warmup bars are discarded from scoring.
            scored_history = [
                {"date": d, "value": v} for d, v in test_equity.items()
            ]
            scored_trades = test_run["trades"]
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

        # Composite equity curve: just concatenate segments (used for plotting)
        composite_equity = pd.concat(all_equity_segments)
        # Remove duplicate dates at window boundaries
        composite_equity = composite_equity[~composite_equity.index.duplicated(keep="first")]

        # Chain equity segments for summary metrics so each window picks up where
        # the previous left off, giving accurate compound return and drawdown.
        chained_segments = []
        current_capital = float(all_equity_segments[0].iloc[0])
        for seg in all_equity_segments:
            if seg.empty:
                continue
            scale = current_capital / float(seg.iloc[0])
            chained_seg = seg * scale
            chained_segments.append(chained_seg)
            current_capital = float(chained_seg.iloc[-1])
        chained_equity = pd.concat(chained_segments)
        chained_equity = chained_equity[~chained_equity.index.duplicated(keep="first")]

        # Summary Metrics computed on chained curve for accurate cross-window aggregates
        summary_history = [
            {"date": d, "value": v} for d, v in chained_equity.items()
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
