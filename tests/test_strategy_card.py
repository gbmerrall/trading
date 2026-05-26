"""Tests for the strategy card builder."""

import json

from backtest.strategy_card import CardCandidate, build_card, write_card

_SUMMARY_A = {
    "max_drawdown": -0.333, "sharpe_ratio": 1.1, "total_return": 0.85,
    "calmar_ratio": 0.4, "sortino_ratio": 1.5, "n_windows": 12,
    "n_windows_with_trades": 10, "ulcer_index": -0.2,  # extra keys ignored
}
_SUMMARY_B = {
    "max_drawdown": -0.20, "sharpe_ratio": 0.7, "total_return": 0.40,
    "calmar_ratio": 0.2, "sortino_ratio": 0.9, "n_windows": 12,
    "n_windows_with_trades": 8,
}


def _candidates():
    return [
        CardCandidate("Breakout(20d)", "BreakoutStrategy", {"lookback_period": 30}, _SUMMARY_A),
        CardCandidate("RSI(14, 30/70)", "RSIStrategy", {"period": 10}, _SUMMARY_B),
    ]


def test_build_card_shape_and_recommended():
    card = build_card(
        ticker="NVDA", start_date="2020-01-01", end_date="2026-05-01",
        start_capital=10000.0, candidates=_candidates(),
    )
    assert card["ticker"] == "NVDA"
    assert card["start_capital"] == 10000.0
    assert card["backtest_window"] == {"start": "2020-01-01", "end": "2026-05-01"}
    # recommended = highest sharpe_ratio
    assert card["recommended"] == "Breakout(20d)"
    assert len(card["candidates"]) == 2
    first = card["candidates"][0]
    assert first["strategy_class"] == "BreakoutStrategy"
    assert first["params"] == {"lookback_period": 30}
    # baseline copies only the selected keys, verbatim fraction for max_drawdown
    assert first["wfa_baseline"]["max_drawdown"] == -0.333
    assert "ulcer_index" not in first["wfa_baseline"]
    assert set(first["wfa_baseline"]) == {
        "max_drawdown", "sharpe_ratio", "total_return", "calmar_ratio",
        "sortino_ratio", "n_windows", "n_windows_with_trades",
    }


def test_build_card_has_generated_at_iso():
    card = build_card("NVDA", "2020-01-01", "2026-05-01", 10000.0, _candidates())
    # parseable ISO-8601 with trailing Z
    assert card["generated_at"].endswith("Z")


def test_write_card_roundtrip(tmp_path):
    card = build_card("NVDA", "2020-01-01", "2026-05-01", 10000.0, _candidates())
    out = tmp_path / "nvda_card.json"
    write_card(card, out)
    loaded = json.loads(out.read_text())
    assert loaded == card
