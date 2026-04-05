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
