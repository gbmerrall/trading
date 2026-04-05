import pytest
import pandas as pd
import numpy as np
from backtest.optimization import GridSearch, RandomSearch


class TestGridSearch:
    def test_single_param(self):
        gs = GridSearch()
        result = gs.generate({"a": [1, 2, 3]})
        assert result == [{"a": 1}, {"a": 2}, {"a": 3}]

    def test_two_params_cartesian_product(self):
        gs = GridSearch()
        result = gs.generate({"a": [1, 2], "b": [10, 20]})
        assert len(result) == 4
        assert {"a": 1, "b": 10} in result
        assert {"a": 1, "b": 20} in result
        assert {"a": 2, "b": 10} in result
        assert {"a": 2, "b": 20} in result

    def test_size_is_product_of_lengths(self):
        gs = GridSearch()
        space = {"a": [1, 2, 3], "b": [10, 20], "c": [100, 200, 300, 400]}
        result = gs.generate(space)
        assert len(result) == 3 * 2 * 4

    def test_single_value_params_returns_one_combination(self):
        gs = GridSearch()
        result = gs.generate({"a": [5], "b": [10]})
        assert result == [{"a": 5, "b": 10}]

    def test_returns_list_of_dicts(self):
        gs = GridSearch()
        result = gs.generate({"x": [1, 2]})
        assert all(isinstance(r, dict) for r in result)


class TestRandomSearch:
    def test_returns_n_combinations(self):
        rs = RandomSearch(n=5, seed=42)
        result = rs.generate({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        assert len(result) == 5

    def test_reproducible_with_same_seed(self):
        space = {"a": list(range(20)), "b": list(range(20))}
        r1 = RandomSearch(n=10, seed=42).generate(space)
        r2 = RandomSearch(n=10, seed=42).generate(space)
        assert r1 == r2

    def test_different_seeds_give_different_results(self):
        space = {"a": list(range(20)), "b": list(range(20))}
        r1 = RandomSearch(n=10, seed=1).generate(space)
        r2 = RandomSearch(n=10, seed=2).generate(space)
        assert r1 != r2

    def test_n_larger_than_space_returns_all_with_replacement(self):
        rs = RandomSearch(n=20, seed=42)
        result = rs.generate({"a": [1, 2, 3]})
        assert len(result) == 20

    def test_returns_list_of_dicts(self):
        rs = RandomSearch(n=3, seed=0)
        result = rs.generate({"x": [1, 2, 3, 4, 5]})
        assert all(isinstance(r, dict) for r in result)

    def test_each_combination_is_valid(self):
        space = {"a": [1, 2], "b": [10, 20]}
        rs = RandomSearch(n=10, seed=99)
        for combo in rs.generate(space):
            assert combo["a"] in [1, 2]
            assert combo["b"] in [10, 20]


class TestGenerateWindows:
    """Tests for the _generate_windows module-level function."""

    def _make_data(self, n):
        """Return a DataFrame with n rows and a DatetimeIndex."""
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": range(1, n + 1)}, index=dates)

    # --- sliding ---

    def test_sliding_window_count(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(100)
        windows = _generate_windows(data, train_size=60, test_size=20, window_type="sliding")
        # First window: train 0-59, test 60-79
        # Second window: train 20-79, test 80-99
        assert len(windows) == 2

    def test_sliding_train_size_is_fixed(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(200)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="sliding")
        for train, test in windows:
            assert len(train) == 100

    def test_sliding_test_windows_contiguous(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(250)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="sliding")
        for i in range(len(windows) - 1):
            _, test_i = windows[i]
            _, test_next = windows[i + 1]
            assert test_i.index[-1] < test_next.index[0]

    def test_sliding_no_overlap_between_train_and_test(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(200)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="sliding")
        for train, test in windows:
            assert len(set(train.index) & set(test.index)) == 0

    def test_sliding_covers_full_range(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(200)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="sliding")
        all_test_dates = set()
        for _, test in windows:
            all_test_dates.update(test.index.tolist())
        expected = set(data.index[100:])  # first train_size bars are never in test
        assert all_test_dates == expected

    # --- anchored ---

    def test_anchored_train_always_starts_at_index_zero(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(200)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="anchored")
        for train, _ in windows:
            assert train.index[0] == data.index[0]

    def test_anchored_train_grows(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(300)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="anchored")
        train_sizes = [len(train) for train, _ in windows]
        for i in range(len(train_sizes) - 1):
            assert train_sizes[i + 1] > train_sizes[i]

    def test_anchored_test_windows_contiguous(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(300)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="anchored")
        for i in range(len(windows) - 1):
            _, test_i = windows[i]
            _, test_next = windows[i + 1]
            assert test_i.index[-1] < test_next.index[0]

    def test_anchored_no_overlap_between_train_and_test(self):
        from backtest.optimization import _generate_windows
        data = self._make_data(300)
        windows = _generate_windows(data, train_size=100, test_size=50, window_type="anchored")
        for train, test in windows:
            assert len(set(train.index) & set(test.index)) == 0

    def test_invalid_window_type_raises(self):
        from backtest.optimization import _generate_windows
        from backtest.validation import ValidationError
        data = self._make_data(200)
        with pytest.raises(ValidationError):
            _generate_windows(data, train_size=100, test_size=50, window_type="invalid")


class TestFilterByWarmup:
    """Tests for the _filter_by_warmup module-level function."""

    def _make_history(self, dates):
        return [{"date": d, "value": 10000.0 + i * 10} for i, d in enumerate(dates)]

    def _make_trades(self, exit_dates):
        return [
            {
                "entry_date": d - pd.Timedelta(days=1),
                "exit_date": d,
                "entry": 100.0,
                "exit": 110.0,
                "shares": 1,
                "pnl": 10.0,
            }
            for d in exit_dates
        ]

    def test_removes_history_before_cutoff(self):
        from backtest.optimization import _filter_by_warmup
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        history = self._make_history(dates)
        cutoff = dates[3]
        filtered_h, _ = _filter_by_warmup(history, [], cutoff)
        assert all(e["date"] >= cutoff for e in filtered_h)
        assert len(filtered_h) == 7

    def test_removes_trades_before_cutoff(self):
        from backtest.optimization import _filter_by_warmup
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        trades = self._make_trades(dates)
        cutoff = dates[5]
        _, filtered_t = _filter_by_warmup([], trades, cutoff)
        assert all(t["exit_date"] >= cutoff for t in filtered_t)
        assert len(filtered_t) == 5

    def test_empty_inputs_return_empty(self):
        from backtest.optimization import _filter_by_warmup
        cutoff = pd.Timestamp("2020-01-10")
        h, t = _filter_by_warmup([], [], cutoff)
        assert h == []
        assert t == []

    def test_cutoff_at_start_returns_all(self):
        from backtest.optimization import _filter_by_warmup
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        history = self._make_history(dates)
        trades = self._make_trades(dates)
        cutoff = dates[0]
        h, t = _filter_by_warmup(history, trades, cutoff)
        assert len(h) == 5
        assert len(t) == 5

    def test_cutoff_after_all_returns_empty(self):
        from backtest.optimization import _filter_by_warmup
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        history = self._make_history(dates)
        trades = self._make_trades(dates)
        cutoff = dates[-1] + pd.Timedelta(days=1)
        h, t = _filter_by_warmup(history, trades, cutoff)
        assert h == []
        assert t == []


class TestWalkForwardOptimizerInit:
    """Tests for WalkForwardOptimizer construction and validation."""

    def _make_data(self, n=300):
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": range(1, n + 1)}, index=dates)

    def _valid_kwargs(self, data):
        from backtest.strategy import ConsecutiveDaysStrategy
        return dict(
            strategy_class=ConsecutiveDaysStrategy,
            param_space={"consecutive_days": [1, 2, 3]},
            data=data,
            train_size=100,
            test_size=50,
            window_type="sliding",
        )

    def test_valid_construction_does_not_raise(self):
        from backtest.optimization import WalkForwardOptimizer
        data = self._make_data()
        WalkForwardOptimizer(**self._valid_kwargs(data))

    def test_invalid_train_size_raises(self):
        from backtest.optimization import WalkForwardOptimizer
        from backtest.validation import ValidationError
        data = self._make_data()
        kwargs = self._valid_kwargs(data)
        kwargs["train_size"] = 0
        with pytest.raises(ValidationError, match="train_size"):
            WalkForwardOptimizer(**kwargs)

    def test_invalid_test_size_raises(self):
        from backtest.optimization import WalkForwardOptimizer
        from backtest.validation import ValidationError
        data = self._make_data()
        kwargs = self._valid_kwargs(data)
        kwargs["test_size"] = 0
        with pytest.raises(ValidationError, match="test_size"):
            WalkForwardOptimizer(**kwargs)

    def test_train_plus_test_exceeds_data_raises(self):
        from backtest.optimization import WalkForwardOptimizer
        from backtest.validation import ValidationError
        data = self._make_data(100)
        kwargs = self._valid_kwargs(data)
        kwargs["train_size"] = 80
        kwargs["test_size"] = 30  # 80 + 30 > 100
        with pytest.raises(ValidationError):
            WalkForwardOptimizer(**kwargs)

    def test_invalid_window_type_raises(self):
        from backtest.optimization import WalkForwardOptimizer
        from backtest.validation import ValidationError
        data = self._make_data()
        kwargs = self._valid_kwargs(data)
        kwargs["window_type"] = "rolling"
        with pytest.raises(ValidationError, match="window_type"):
            WalkForwardOptimizer(**kwargs)

    def test_non_strategy_class_raises(self):
        from backtest.optimization import WalkForwardOptimizer
        from backtest.validation import ValidationError
        data = self._make_data()
        kwargs = self._valid_kwargs(data)
        kwargs["strategy_class"] = list  # not a BaseStrategy subclass
        with pytest.raises(ValidationError, match="strategy_class"):
            WalkForwardOptimizer(**kwargs)

    def test_invalid_objective_string_raises(self):
        from backtest.optimization import WalkForwardOptimizer
        from backtest.validation import ValidationError
        data = self._make_data()
        kwargs = self._valid_kwargs(data)
        kwargs["objective"] = "not_a_metric"
        with pytest.raises(ValidationError, match="objective"):
            WalkForwardOptimizer(**kwargs)

    def test_callable_objective_accepted(self):
        from backtest.optimization import WalkForwardOptimizer
        data = self._make_data()
        kwargs = self._valid_kwargs(data)
        kwargs["objective"] = lambda ph, t: 0.0
        WalkForwardOptimizer(**kwargs)  # should not raise

    def test_string_objective_accepted(self):
        from backtest.optimization import WalkForwardOptimizer
        data = self._make_data()
        kwargs = self._valid_kwargs(data)
        kwargs["objective"] = "calmar_ratio"
        WalkForwardOptimizer(**kwargs)  # should not raise

    def test_default_objective_is_sharpe(self):
        from backtest.optimization import WalkForwardOptimizer
        data = self._make_data()
        opt = WalkForwardOptimizer(**self._valid_kwargs(data))
        from backtest.metrics import sharpe_ratio
        assert opt._objective_fn is sharpe_ratio

    def test_default_searcher_is_grid_search(self):
        from backtest.optimization import GridSearch, WalkForwardOptimizer
        data = self._make_data()
        opt = WalkForwardOptimizer(**self._valid_kwargs(data))
        assert isinstance(opt.searcher, GridSearch)


class TestRunTrainWindow:
    """Tests for WalkForwardOptimizer._run_train_window."""

    def _make_data(self, n=200):
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        prices = [100.0 + i * 0.5 for i in range(n)]
        return pd.DataFrame({"Close": prices}, index=dates)

    def _make_optimizer(self, data):
        from backtest.optimization import WalkForwardOptimizer
        from backtest.strategy import ConsecutiveDaysStrategy
        return WalkForwardOptimizer(
            strategy_class=ConsecutiveDaysStrategy,
            param_space={"consecutive_days": [1, 2, 3]},
            data=data,
            train_size=100,
            test_size=50,
            window_type="sliding",
            min_trades=0,  # don't skip windows with few trades
        )

    def test_returns_dict_with_best_params_key(self):
        data = self._make_data()
        opt = self._make_optimizer(data)
        train_data = data.iloc[:100]
        result = opt._run_train_window(train_data)
        assert "best_params" in result

    def test_best_params_keys_match_param_space(self):
        data = self._make_data()
        opt = self._make_optimizer(data)
        train_data = data.iloc[:100]
        result = opt._run_train_window(train_data)
        assert set(result["best_params"].keys()) == {"consecutive_days"}

    def test_best_params_value_is_from_param_space(self):
        data = self._make_data()
        opt = self._make_optimizer(data)
        train_data = data.iloc[:100]
        result = opt._run_train_window(train_data)
        assert result["best_params"]["consecutive_days"] in [1, 2, 3]

    def test_returns_objective_score(self):
        data = self._make_data()
        opt = self._make_optimizer(data)
        train_data = data.iloc[:100]
        result = opt._run_train_window(train_data)
        assert "objective_score" in result
        assert isinstance(result["objective_score"], float)

    def test_all_min_trades_filtered_returns_first_params(self):
        """When every candidate has too few trades, first params set is returned."""
        data = self._make_data()
        from backtest.optimization import WalkForwardOptimizer
        from backtest.strategy import ConsecutiveDaysStrategy
        opt = WalkForwardOptimizer(
            strategy_class=ConsecutiveDaysStrategy,
            param_space={"consecutive_days": [1, 2, 3]},
            data=data,
            train_size=100,
            test_size=50,
            window_type="sliding",
            min_trades=9999,  # impossible threshold
        )
        train_data = data.iloc[:100]
        result = opt._run_train_window(train_data)
        # Should not raise — falls back to first candidate
        assert "best_params" in result
        assert result["objective_score"] == float("-inf")


class TestWalkForwardResult:
    """Tests for WalkForwardOptimizer.run() and WalkForwardResult."""

    def _make_oscillating_data(self, n=500):
        """Create data with 3-day up/down oscillations."""
        import math
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        prices = [100.0 + 10 * math.sin(i * math.pi / 3) for i in range(n)]
        return pd.DataFrame({"Close": prices}, index=dates)

    def _make_optimizer(self, data=None, min_trades=0):
        from backtest.optimization import WalkForwardOptimizer
        from backtest.strategy import ConsecutiveDaysStrategy
        if data is None:
            data = self._make_oscillating_data()
        return WalkForwardOptimizer(
            strategy_class=ConsecutiveDaysStrategy,
            param_space={"consecutive_days": [1, 2, 3]},
            data=data,
            train_size=150,
            test_size=50,
            window_type="sliding",
            min_trades=min_trades,
        )

    def test_run_returns_walk_forward_result(self):
        from backtest.optimization import WalkForwardResult
        opt = self._make_optimizer()
        result = opt.run()
        assert isinstance(result, WalkForwardResult)

    def test_equity_curve_is_series(self):
        opt = self._make_optimizer()
        result = opt.run()
        assert isinstance(result.equity_curve, pd.Series)

    def test_equity_curve_has_datetime_index(self):
        opt = self._make_optimizer()
        result = opt.run()
        assert isinstance(result.equity_curve.index, pd.DatetimeIndex)

    def test_windows_is_dataframe(self):
        opt = self._make_optimizer()
        result = opt.run()
        assert isinstance(result.windows, pd.DataFrame)

    def test_windows_has_correct_columns(self):
        opt = self._make_optimizer()
        result = opt.run()
        expected_cols = {
            "train_start", "train_end", "test_start", "test_end",
            "best_params", "objective_score", "n_trades",
            "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "max_drawdown", "ulcer_index",
            "profit_factor", "win_rate", "expectancy", "recovery_factor",
        }
        assert expected_cols.issubset(set(result.windows.columns))

    def test_windows_row_count_matches_expected(self):
        from backtest.optimization import _generate_windows
        data = self._make_oscillating_data()
        opt = self._make_optimizer(data=data)
        result = opt.run()
        expected_windows = _generate_windows(data, 150, 50, "sliding")
        assert len(result.windows) == len(expected_windows)

    def test_summary_is_dict(self):
        opt = self._make_optimizer()
        result = opt.run()
        assert isinstance(result.summary, dict)

    def test_summary_has_all_metric_keys(self):
        opt = self._make_optimizer()
        result = opt.run()
        for key in ["total_return", "cagr", "sharpe_ratio", "sortino_ratio",
                    "calmar_ratio", "max_drawdown", "ulcer_index",
                    "profit_factor", "win_rate", "expectancy", "recovery_factor"]:
            assert key in result.summary, f"Missing summary key: {key}"

    def test_summary_has_meta_keys(self):
        opt = self._make_optimizer()
        result = opt.run()
        assert "n_windows" in result.summary
        assert "n_windows_with_trades" in result.summary
        assert "param_stability" in result.summary

    def test_best_params_overall_is_dict(self):
        opt = self._make_optimizer()
        result = opt.run()
        assert isinstance(result.best_params_overall, dict)

    def test_best_params_overall_keys_match_param_space(self):
        opt = self._make_optimizer()
        result = opt.run()
        assert set(result.best_params_overall.keys()) == {"consecutive_days"}

    def test_anchored_window_type_works(self):
        from backtest.optimization import WalkForwardOptimizer
        from backtest.strategy import ConsecutiveDaysStrategy
        data = self._make_oscillating_data()
        opt = WalkForwardOptimizer(
            strategy_class=ConsecutiveDaysStrategy,
            param_space={"consecutive_days": [1, 2]},
            data=data,
            train_size=150,
            test_size=50,
            window_type="anchored",
            min_trades=0,
        )
        result = opt.run()
        assert isinstance(result.equity_curve, pd.Series)
        assert len(result.windows) > 0
