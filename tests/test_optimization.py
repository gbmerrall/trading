import pytest

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
