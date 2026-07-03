"""Tests for the deployment-gate harness that cross-checks the LLM summary.

Covers the fraction->percent unit conversion, the four gates, ranking, verdict,
warnings, missing-baseline / non-finite handling, strict VERDICT-line parsing,
and the harness-vs-LLM comparison. The BHP ground-truth test locks in the same
outcome the standalone wfa_scorer harness produces, proving the port is faithful.
"""

from backtest.deployment_gate import (
    DEFAULT_THRESHOLDS,
    DEPLOY_NOTHING,
    GateThresholds,
    compare_verdicts,
    parse_llm_verdict,
    render_gate_table,
    score_candidates,
)
from backtest.strategy_card import CardCandidate


def _candidate(
    label: str,
    *,
    adaptive_sharpe: float,
    adaptive_dd_pct: float,
    stability_pct: float,
    fixed_sharpe: float,
    fixed_dd_pct: float,
    fixed_trades: int,
    params: "dict | None" = None,
    fixed: bool = True,
) -> CardCandidate:
    """Build a CardCandidate from percent-style numbers.

    Summary/fixed metrics are stored as FRACTIONS (as the real pipeline does);
    the percent-style keyword args are divided by 100 here for readability.
    """
    fixed_baseline = None
    if fixed:
        fixed_baseline = {
            "sharpe_ratio": fixed_sharpe,
            "total_return": 1.0,
            "max_drawdown": fixed_dd_pct / 100.0,
            "n_trades": fixed_trades,
        }
    return CardCandidate(
        label=label,
        strategy_class="BreakoutStrategy",
        params=params or {"lookback_period": 20},
        summary={
            "total_return": 1.0,
            "max_drawdown": adaptive_dd_pct / 100.0,
            "sharpe_ratio": adaptive_sharpe,
            "param_stability": stability_pct / 100.0,
        },
        fixed_baseline=fixed_baseline,
    )


# Benchmark in PERCENT units, as produced by runner._calculate_metrics.
BH_METRICS = {"total_return": 142.5, "sharpe_ratio": 0.52, "max_drawdown": -43.6}


def _bhp_candidates() -> list[CardCandidate]:
    """The five BHP strategies, replicating the wfa_scorer ground-truth fixture."""
    return [
        _candidate(
            "MeanReversion", adaptive_sharpe=0.85, adaptive_dd_pct=-30.5,
            stability_pct=66.7, fixed_sharpe=0.78, fixed_dd_pct=-30.5, fixed_trades=11,
        ),
        _candidate(
            "Gap(2%)", adaptive_sharpe=0.33, adaptive_dd_pct=-36.8,
            stability_pct=61.9, fixed_sharpe=0.44, fixed_dd_pct=-31.6, fixed_trades=22,
        ),
        _candidate(
            "ConsecutiveDays(3)", adaptive_sharpe=0.27, adaptive_dd_pct=-30.1,
            stability_pct=42.9, fixed_sharpe=0.68, fixed_dd_pct=-30.3, fixed_trades=10,
        ),
        _candidate(
            "RSI(14, 30/70)", adaptive_sharpe=0.24, adaptive_dd_pct=-31.2,
            stability_pct=42.9, fixed_sharpe=0.76, fixed_dd_pct=-17.3, fixed_trades=16,
        ),
        _candidate(
            "BollingerBands(20)", adaptive_sharpe=0.21, adaptive_dd_pct=-37.4,
            stability_pct=28.6, fixed_sharpe=0.94, fixed_dd_pct=-23.6, fixed_trades=23,
        ),
    ]


class TestUnitConversion:
    def test_fraction_metrics_scaled_to_percent(self):
        report = score_candidates([_bhp_candidates()[0]], BH_METRICS)
        score = report.scored[0]
        assert score.parameter_stability_pct is not None
        assert score.fixed_max_drawdown_pct is not None
        # param_stability 0.667 -> 66.7%, fixed max_drawdown -0.305 -> -30.5%.
        assert abs(score.parameter_stability_pct - 66.7) < 1e-6
        assert abs(score.fixed_max_drawdown_pct - (-30.5)) < 1e-6

    def test_benchmark_units_preserved(self):
        report = score_candidates([_bhp_candidates()[0]], BH_METRICS)
        assert report.benchmark_sharpe == 0.52
        assert report.benchmark_max_drawdown_pct == -43.6


class TestBhpGroundTruth:
    def setup_method(self):
        self.report = score_candidates(_bhp_candidates(), BH_METRICS)

    def _by_label(self, label: str):
        return next(s for s in self.report.scored if s.label == label)

    def test_three_strategies_pass_all_gates(self):
        for label in ("RSI(14, 30/70)", "MeanReversion", "ConsecutiveDays(3)"):
            assert self._by_label(label).passed, label

    def test_bollinger_fails_stability_only(self):
        bollinger = self._by_label("BollingerBands(20)")
        assert not bollinger.passed
        assert bollinger.failed_gates == ["gate3_stability"]

    def test_gap_fails_gate1(self):
        gap = self._by_label("Gap(2%)")
        assert not gap.passed
        assert "gate1_sharpe" in gap.failed_gates

    def test_candidate_is_most_coherent_passer(self):
        # MeanReversion's fixed 0.78 tracks its adaptive 0.85 (gap 0.07); RSI's
        # fixed 0.76 sits 0.52 above its adaptive 0.24 -> less coherent.
        assert self.report.candidate is not None
        assert self.report.candidate.label == "MeanReversion"
        assert self.report.verdict_label == "MeanReversion"

    def test_ranking_orders_passers_by_coherence(self):
        passers = [s.label for s in self.report.scored if s.passed]
        assert passers == ["MeanReversion", "ConsecutiveDays(3)", "RSI(14, 30/70)"]

    def test_small_sample_warning_present(self):
        # Candidate MeanReversion rests on 11 trades.
        assert any("11 trades" in w for w in self.report.warnings)

    def test_curve_fit_warning_present(self):
        assert any("suspected curve-fit" in w for w in self.report.warnings)
        assert any("BollingerBands(20)" in w for w in self.report.warnings)


class TestGateEdges:
    def test_missing_fixed_baseline_fails_all_gates(self):
        cand = _candidate(
            "NoBaseline", adaptive_sharpe=0.9, adaptive_dd_pct=-20.0,
            stability_pct=80.0, fixed_sharpe=0.0, fixed_dd_pct=0.0,
            fixed_trades=0, fixed=False,
        )
        report = score_candidates([cand], BH_METRICS)
        score = report.scored[0]
        assert score.missing_baseline
        assert not score.passed
        # Every fixed-baseline gate fails; stability (an adaptive property) is not
        # a deploy signal on its own, so it may still read as passed.
        assert "gate1_sharpe" in score.failed_gates
        assert "gate2_drawdown" in score.failed_gates
        assert "gate4_not_material_below" in score.failed_gates
        assert report.verdict_label == DEPLOY_NOTHING

    def test_non_finite_fixed_sharpe_fails_gracefully(self):
        cand = _candidate(
            "Degenerate", adaptive_sharpe=0.5, adaptive_dd_pct=-20.0,
            stability_pct=80.0, fixed_sharpe=float("-inf"), fixed_dd_pct=-10.0,
            fixed_trades=30,
        )
        report = score_candidates([cand], BH_METRICS)
        score = report.scored[0]
        assert score.fixed_sharpe is None
        assert not score.gate1_sharpe
        assert not score.passed

    def test_gate4_fails_when_fixed_far_below_adaptive(self):
        # Fixed 0.50 vs adaptive 0.90 -> 0.40 below, exceeds 0.10 tolerance.
        cand = _candidate(
            "Collapse", adaptive_sharpe=0.90, adaptive_dd_pct=-20.0,
            stability_pct=80.0, fixed_sharpe=0.50, fixed_dd_pct=-10.0,
            fixed_trades=40,
        )
        report = score_candidates([cand], BH_METRICS)
        score = report.scored[0]
        assert not score.gate4_not_material_below
        assert not score.passed

    def test_gate4_fixed_above_adaptive_never_penalised(self):
        cand = _candidate(
            "Robust", adaptive_sharpe=0.60, adaptive_dd_pct=-20.0,
            stability_pct=80.0, fixed_sharpe=0.95, fixed_dd_pct=-10.0,
            fixed_trades=40,
        )
        report = score_candidates([cand], BH_METRICS)
        assert report.scored[0].gate4_not_material_below

    def test_all_fail_yields_deploy_nothing(self):
        cand = _candidate(
            "Weak", adaptive_sharpe=0.10, adaptive_dd_pct=-40.0,
            stability_pct=10.0, fixed_sharpe=0.10, fixed_dd_pct=-43.0,
            fixed_trades=5,
        )
        report = score_candidates([cand], BH_METRICS)
        assert report.candidate is None
        assert report.verdict_label == DEPLOY_NOTHING

    def test_empty_candidates_deploy_nothing(self):
        report = score_candidates([], BH_METRICS)
        assert report.candidate is None
        assert report.verdict_label == DEPLOY_NOTHING


class TestThresholdConfig:
    def test_custom_stability_threshold_flips_pass(self):
        # ConsecutiveDays stability 42.9%: passes at 40, fails at 45.
        cands = [_bhp_candidates()[2]]  # ConsecutiveDays
        assert score_candidates(cands, BH_METRICS).scored[0].passed
        strict = GateThresholds(stability_min_pct=45.0)
        assert not score_candidates(cands, BH_METRICS, strict).scored[0].passed


class TestParseLlmVerdict:
    LABELS = ["RSI(14, 30/70)", "MeanReversion", "BollingerBands(20)"]

    def test_extracts_exact_label(self):
        text = "Some analysis.\n\nVERDICT: RSI(14, 30/70)"
        parsed = parse_llm_verdict(text, self.LABELS)
        assert parsed.present
        assert parsed.resolved == "RSI(14, 30/70)"

    def test_deploy_nothing_token(self):
        parsed = parse_llm_verdict("Nope.\nVERDICT: DEPLOY_NOTHING", self.LABELS)
        assert parsed.resolved == DEPLOY_NOTHING

    def test_deploy_nothing_with_space_variant(self):
        parsed = parse_llm_verdict("VERDICT: deploy nothing", self.LABELS)
        assert parsed.resolved == DEPLOY_NOTHING

    def test_missing_verdict_line(self):
        parsed = parse_llm_verdict("Four nice paragraphs, no verdict.", self.LABELS)
        assert not parsed.present
        assert parsed.resolved is None

    def test_unknown_label_is_unresolved(self):
        parsed = parse_llm_verdict("VERDICT: SomeMadeUpStrategy", self.LABELS)
        assert parsed.present
        assert parsed.resolved is None
        assert parsed.raw == "SomeMadeUpStrategy"

    def test_case_insensitive_label_match(self):
        parsed = parse_llm_verdict("VERDICT: meanreversion", self.LABELS)
        assert parsed.resolved == "MeanReversion"

    def test_last_verdict_line_wins(self):
        text = "VERDICT: MeanReversion\nmore text\nVERDICT: RSI(14, 30/70)"
        parsed = parse_llm_verdict(text, self.LABELS)
        assert parsed.resolved == "RSI(14, 30/70)"


class TestCompareVerdicts:
    def test_match_passes(self):
        report = score_candidates(_bhp_candidates(), BH_METRICS)
        parsed = parse_llm_verdict("VERDICT: MeanReversion", ["MeanReversion"])
        outcome = compare_verdicts(report, parsed)
        assert outcome.ok

    def test_mismatch_halts(self):
        report = score_candidates(_bhp_candidates(), BH_METRICS)
        parsed = parse_llm_verdict("VERDICT: RSI(14, 30/70)", ["RSI(14, 30/70)"])
        outcome = compare_verdicts(report, parsed)
        assert not outcome.ok
        assert "mismatch" in outcome.reason.lower()

    def test_missing_verdict_halts(self):
        report = score_candidates(_bhp_candidates(), BH_METRICS)
        parsed = parse_llm_verdict("no verdict here", ["RSI(14, 30/70)"])
        outcome = compare_verdicts(report, parsed)
        assert not outcome.ok
        assert "no machine-readable" in outcome.reason.lower()

    def test_llm_deploy_when_harness_deploy_nothing_halts(self):
        weak = _candidate(
            "Weak", adaptive_sharpe=0.10, adaptive_dd_pct=-40.0,
            stability_pct=10.0, fixed_sharpe=0.10, fixed_dd_pct=-43.0,
            fixed_trades=5, params={"lookback_period": 20},
        )
        report = score_candidates([weak], BH_METRICS)
        parsed = parse_llm_verdict("VERDICT: Weak", ["Weak"])
        outcome = compare_verdicts(report, parsed)
        assert not outcome.ok

    def test_both_deploy_nothing_passes(self):
        weak = _candidate(
            "Weak", adaptive_sharpe=0.10, adaptive_dd_pct=-40.0,
            stability_pct=10.0, fixed_sharpe=0.10, fixed_dd_pct=-43.0,
            fixed_trades=5,
        )
        report = score_candidates([weak], BH_METRICS)
        parsed = parse_llm_verdict("VERDICT: DEPLOY_NOTHING", ["Weak"])
        outcome = compare_verdicts(report, parsed)
        assert outcome.ok
        assert outcome.harness_verdict == DEPLOY_NOTHING


class TestRenderGateTable:
    def test_renders_table_and_verdict(self):
        report = score_candidates(_bhp_candidates(), BH_METRICS)
        text = render_gate_table(report)
        assert "DEPLOY MeanReversion" in text
        assert "Benchmark (Buy & Hold)" in text
        # Top-ranked passer sits above a failer in the table body.
        assert text.index("MeanReversion") < text.index("Gap(2%)")

    def test_default_thresholds_shared(self):
        # The prose thresholds and the harness share the same object.
        assert DEFAULT_THRESHOLDS.stability_min_pct == 40.0
        assert DEFAULT_THRESHOLDS.min_trades_warn == 20
