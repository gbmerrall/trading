"""Tests for the executive-summary payload construction in llm_analysis.

The LLM call itself is not tested (network); these cover the pure payload
builder, candidate capping, non-finite formatting, and the no-API-key path.
"""


from backtest.deployment_gate import DEPLOY_NOTHING, GateThresholds
from backtest.llm_analysis import (
    MAX_CANDIDATES,
    SUMMARY_MODEL,
    build_summary_payload,
    build_system_instruction,
    generate_executive_summary,
)
from backtest.strategy_card import CardCandidate


def _candidate(label: str, sharpe: float = 1.0, fixed: "dict | None" = None) -> CardCandidate:
    return CardCandidate(
        label=label,
        strategy_class="BreakoutStrategy",
        params={"lookback_period": 20},
        summary={
            "total_return": 0.25,
            "max_drawdown": -0.10,
            "sharpe_ratio": sharpe,
            "param_stability": 0.6,
        },
        fixed_baseline=fixed,
    )


BH_METRICS = {"total_return": 80.0, "sharpe_ratio": 0.7, "max_drawdown": -33.0}


class TestBuildSummaryPayload:
    def test_caps_candidates_and_reports_total(self):
        candidates = [_candidate(f"S{i}") for i in range(8)]
        payload = build_summary_payload("SPY", candidates, BH_METRICS)
        assert f"TOP {MAX_CANDIDATES} STRATEGY WFA RESULTS" in payload
        assert "of 8 analysed" in payload
        for i in range(MAX_CANDIDATES):
            assert f"S{i}" in payload
        assert "S5" not in payload

    def test_includes_fixed_params_baseline(self):
        fixed = {"sharpe_ratio": 0.88, "total_return": 0.9,
                 "max_drawdown": -0.15, "n_trades": 14}
        payload = build_summary_payload("SPY", [_candidate("S0", fixed=fixed)], BH_METRICS)
        assert "Fixed-Params" in payload
        assert "0.88" in payload
        assert "14" in payload

    def test_missing_fixed_baseline_is_explicit(self):
        payload = build_summary_payload("SPY", [_candidate("S0", fixed=None)], BH_METRICS)
        assert "not available" in payload

    def test_non_finite_values_render_as_na(self):
        candidate = _candidate("S0", sharpe=float("-inf"))
        payload = build_summary_payload("SPY", [candidate], BH_METRICS)
        assert "-inf" not in payload
        assert "n/a" in payload


class TestGenerateExecutiveSummary:
    def test_skips_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        result = generate_executive_summary("SPY", [_candidate("S0")], BH_METRICS)
        assert "Skipping LLM summary" in result


class TestModelChoice:
    def test_uses_gemini_3_series(self):
        assert SUMMARY_MODEL.startswith("gemini-3")


class TestSystemInstruction:
    def test_requires_machine_readable_verdict(self):
        instruction = build_system_instruction()
        assert "VERDICT:" in instruction
        assert DEPLOY_NOTHING in instruction

    def test_threshold_numbers_come_from_config(self):
        custom = GateThresholds(stability_min_pct=55.0, min_trades_warn=30)
        instruction = build_system_instruction(custom)
        # The prose cutoffs track the shared thresholds object, not hard-coded 40/20.
        assert "below 55%" in instruction
        assert "~30" in instruction
