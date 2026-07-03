"""Tests for ensemble strategy entries in trade_analysis.STRATEGIES.

Verifies the three committee configurations are present, correctly parameterised,
and excluded from WFA_PARAM_GRIDS so the skip path is exercised rather than
raising a KeyError.
"""

import pytest

from backtest.strategy import (
    BreakoutStrategy,
    EnsembleStrategy,
    GapStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_strategies():
    """Import STRATEGIES lazily to avoid module-level side effects (sys.argv check)."""
    import importlib
    import sys

    # Patch sys.argv so trade_analysis doesn't exit early on the ticker guard
    orig = sys.argv[:]
    sys.argv = ["trade_analysis.py", "SPY"]
    try:
        import trade_analysis
        importlib.reload(trade_analysis)
        return trade_analysis.STRATEGIES, trade_analysis.WFA_PARAM_GRIDS
    finally:
        sys.argv = orig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def strategies_and_grids():
    return _get_strategies()


@pytest.fixture(scope="module")
def strategies(strategies_and_grids):
    return strategies_and_grids[0]


@pytest.fixture(scope="module")
def wfa_grids(strategies_and_grids):
    return strategies_and_grids[1]


@pytest.fixture(scope="module")
def ensemble_entries(strategies):
    return [(label, inst) for label, inst in strategies if isinstance(inst, EnsembleStrategy)]


# ---------------------------------------------------------------------------
# Presence tests (fail until three entries are added)
# ---------------------------------------------------------------------------

class TestEnsembleEntriesPresent:
    def test_exactly_three_ensemble_entries(self, ensemble_entries):
        assert len(ensemble_entries) == 3, (
            f"Expected 3 EnsembleStrategy entries, got {len(ensemble_entries)}"
        )

    def test_pair_strict_label_present(self, ensemble_entries):
        labels = [label for label, _ in ensemble_entries]
        assert "Ensemble_Pair(Strict)" in labels

    def test_triple_majority_label_present(self, ensemble_entries):
        labels = [label for label, _ in ensemble_entries]
        assert "Ensemble_Triple(Majority)" in labels

    def test_quad_loose_label_present(self, ensemble_entries):
        labels = [label for label, _ in ensemble_entries]
        assert "Ensemble_Quad(Loose)" in labels


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

def _get_entry(ensemble_entries, label):
    for lbl, inst in ensemble_entries:
        if lbl == label:
            return inst
    pytest.fail(f"No ensemble entry with label '{label}'")


class TestPairStrictConfig:
    def test_two_sub_strategies(self, ensemble_entries):
        inst = _get_entry(ensemble_entries, "Ensemble_Pair(Strict)")
        assert len(inst.strategies) == 2

    def test_min_agreement_is_2(self, ensemble_entries):
        inst = _get_entry(ensemble_entries, "Ensemble_Pair(Strict)")
        assert inst.min_agreement == 2

    def test_contains_breakout_20(self, ensemble_entries):
        inst = _get_entry(ensemble_entries, "Ensemble_Pair(Strict)")
        types = [type(s) for s in inst.strategies]
        assert BreakoutStrategy in types

    def test_contains_mean_reversion(self, ensemble_entries):
        inst = _get_entry(ensemble_entries, "Ensemble_Pair(Strict)")
        types = [type(s) for s in inst.strategies]
        assert MeanReversionStrategy in types

    def test_breakout_lookback_is_20(self, ensemble_entries):
        inst = _get_entry(ensemble_entries, "Ensemble_Pair(Strict)")
        bo = next(s for s in inst.strategies if isinstance(s, BreakoutStrategy))
        assert bo.lookback_period == 20


class TestTripleMajorityConfig:
    def test_three_sub_strategies(self, ensemble_entries):
        inst = _get_entry(ensemble_entries, "Ensemble_Triple(Majority)")
        assert len(inst.strategies) == 3

    def test_min_agreement_is_2(self, ensemble_entries):
        inst = _get_entry(ensemble_entries, "Ensemble_Triple(Majority)")
        assert inst.min_agreement == 2

    def test_contains_breakout_mean_reversion_momentum(self, ensemble_entries):
        inst = _get_entry(ensemble_entries, "Ensemble_Triple(Majority)")
        types = {type(s) for s in inst.strategies}
        assert types == {BreakoutStrategy, MeanReversionStrategy, MomentumStrategy}


class TestQuadLooseConfig:
    def test_four_sub_strategies(self, ensemble_entries):
        inst = _get_entry(ensemble_entries, "Ensemble_Quad(Loose)")
        assert len(inst.strategies) == 4

    def test_min_agreement_is_2(self, ensemble_entries):
        inst = _get_entry(ensemble_entries, "Ensemble_Quad(Loose)")
        assert inst.min_agreement == 2

    def test_contains_all_four_types(self, ensemble_entries):
        inst = _get_entry(ensemble_entries, "Ensemble_Quad(Loose)")
        types = {type(s) for s in inst.strategies}
        assert types == {BreakoutStrategy, MeanReversionStrategy, MomentumStrategy, GapStrategy}

    def test_gap_min_gap_pct_is_0_02(self, ensemble_entries):
        inst = _get_entry(ensemble_entries, "Ensemble_Quad(Loose)")
        gap = next(s for s in inst.strategies if isinstance(s, GapStrategy))
        assert gap.min_gap_pct == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# WFA exclusion: ensemble classes must NOT be in WFA_PARAM_GRIDS
# ---------------------------------------------------------------------------

class TestEnsembleWfaExclusion:
    def test_ensemble_class_not_in_wfa_grids(self, wfa_grids):
        assert EnsembleStrategy not in wfa_grids, (
            "EnsembleStrategy must not have a WFA param grid — "
            "it cannot be parameterised via a flat grid"
        )
