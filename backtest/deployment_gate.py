"""Deterministic deployment-gate scoring — the objective check on the LLM summary.

This is the in-repo home of the standalone ``wfa_scorer`` harness. The LLM
executive summary is prone to reaching the wrong conclusion from correct data,
so the pipeline scores the same WFA candidates through a fixed set of gates and
compares the harness's verdict against the LLM's. A mismatch halts the pipeline
for a human rather than shipping a suspect "deploy" call.

Every deployment claim is judged on the FIXED-PARAMS baseline only (the single
parameter set actually traded live). The adaptive WFA Sharpe is used solely as
the Gate 4 cross-check and is never a ranking key.

Unit convention: all gate arithmetic is in PERCENT units (percentage points for
drawdown, percent for stability) so the thresholds read naturally (5 pp, 40 %).
``CardCandidate`` summary/fixed metrics arrive as FRACTIONS and are scaled here;
``bh_metrics`` already arrives in percent from runner._calculate_metrics.
"""

from __future__ import annotations

import math
import re

from dataclasses import dataclass, field
from typing import Any

from .strategy_card import CardCandidate

# Sentinel verdict meaning "no strategy clears the gates".
DEPLOY_NOTHING = "DEPLOY_NOTHING"

# Percent multiplier for fraction -> percent conversion (matches runner's).
_PCT = 100.0


@dataclass(frozen=True)
class GateThresholds:
    """The single source of truth for every gate cutoff.

    These constants are shared by both the harness (which enforces them) and the
    LLM system instruction (which describes them in prose). Tuning one value here
    moves the scoring and the prompt narrative together — they can never drift.

    Attributes:
        drawdown_reduction_pp: Gate 2 — the fixed-params max drawdown must be at
            least this many percentage points shallower than the benchmark's.
        stability_min_pct: Gate 3 — parameter stability (percent) must be at
            least this value.
        material_below: Gate 4 — the fixed Sharpe must not fall more than this
            far below the adaptive Sharpe. Fixed ABOVE adaptive is a positive
            robustness signal and is never penalised.
        min_trades_warn: Verdict warning — warn if the deploy candidate rests on
            fewer trades than this (small-sample risk).
    """

    drawdown_reduction_pp: float = 5.0
    stability_min_pct: float = 40.0
    material_below: float = 0.10
    min_trades_warn: int = 20


DEFAULT_THRESHOLDS = GateThresholds()


@dataclass(frozen=True)
class GateScore:
    """One strategy scored against the four deployment gates.

    All ``*_pct`` fields are in percent units. Any field may be None when the
    fixed-params baseline is absent or a metric was non-finite; such a strategy
    fails every gate and can never be a deploy candidate.
    """

    label: str
    params: dict[str, Any]
    gate1_sharpe: bool
    gate2_drawdown: bool
    gate3_stability: bool
    gate4_not_material_below: bool
    fixed_sharpe: float | None
    adaptive_sharpe: float | None
    fixed_max_drawdown_pct: float | None
    parameter_stability_pct: float | None
    fixed_trades: int | None
    drawdown_reduction_pp: float | None
    fixed_adaptive_gap: float | None  # |fixed - adaptive| Sharpe; smaller = more coherent
    missing_baseline: bool

    @property
    def passed(self) -> bool:
        """A strategy passes iff all four gates pass."""
        return (
            self.gate1_sharpe
            and self.gate2_drawdown
            and self.gate3_stability
            and self.gate4_not_material_below
        )

    @property
    def failed_gates(self) -> list[str]:
        """Names of the gates this strategy failed, in gate order."""
        failures = []
        if not self.gate1_sharpe:
            failures.append("gate1_sharpe")
        if not self.gate2_drawdown:
            failures.append("gate2_drawdown")
        if not self.gate3_stability:
            failures.append("gate3_stability")
        if not self.gate4_not_material_below:
            failures.append("gate4_not_material_below")
        return failures


@dataclass(frozen=True)
class GateReport:
    """The full deterministic scoring result for one ticker's candidates."""

    thresholds: GateThresholds
    benchmark_sharpe: float
    benchmark_max_drawdown_pct: float
    scored: list[GateScore]  # ordered by the ranking rules, passers first
    candidate: GateScore | None
    verdict_label: str  # a strategy label, or DEPLOY_NOTHING
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedVerdict:
    """Result of extracting the machine-readable VERDICT line from the summary."""

    present: bool  # a VERDICT: line was found
    raw: str | None  # the exact text after "VERDICT:"
    resolved: str | None  # a known label, DEPLOY_NOTHING, or None if unmatched


@dataclass(frozen=True)
class ValidationOutcome:
    """The comparison of the harness verdict against the LLM verdict."""

    ok: bool
    harness_verdict: str
    llm_verdict: str | None
    reason: str


def _finite(value: Any) -> float | None:
    """Coerce to a finite float, or None when missing / non-finite."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _score_one(
    candidate: CardCandidate,
    benchmark_sharpe: float,
    benchmark_max_drawdown_pct: float,
    thresholds: GateThresholds,
) -> GateScore:
    """Score a single candidate against all four gates (fixed-params baseline)."""
    summary = candidate.summary or {}
    adaptive_sharpe = _finite(summary.get("sharpe_ratio"))

    stability_frac = _finite(summary.get("param_stability"))
    stability_pct = None if stability_frac is None else stability_frac * _PCT

    fixed = candidate.fixed_baseline
    missing = not fixed

    fixed_sharpe = _finite(fixed.get("sharpe_ratio")) if fixed else None
    fixed_dd_frac = _finite(fixed.get("max_drawdown")) if fixed else None
    fixed_dd_pct = None if fixed_dd_frac is None else fixed_dd_frac * _PCT
    fixed_trades = None
    if fixed and fixed.get("n_trades") is not None:
        try:
            fixed_trades = int(fixed["n_trades"])
        except (TypeError, ValueError):
            fixed_trades = None

    bench_dd = _finite(benchmark_max_drawdown_pct)
    reduction = None
    if fixed_dd_pct is not None and bench_dd is not None:
        # Magnitudes so payload sign conventions do not matter.
        reduction = abs(bench_dd) - abs(fixed_dd_pct)

    bench_sharpe = _finite(benchmark_sharpe)

    gap = None
    if fixed_sharpe is not None and adaptive_sharpe is not None:
        gap = abs(fixed_sharpe - adaptive_sharpe)

    gate1 = fixed_sharpe is not None and bench_sharpe is not None and fixed_sharpe >= bench_sharpe
    gate2 = reduction is not None and reduction >= thresholds.drawdown_reduction_pp
    gate3 = stability_pct is not None and stability_pct >= thresholds.stability_min_pct
    # Fixed-above-adaptive passes trivially; only a material shortfall fails.
    gate4 = (
        fixed_sharpe is not None
        and adaptive_sharpe is not None
        and fixed_sharpe >= adaptive_sharpe - thresholds.material_below
    )

    return GateScore(
        label=candidate.label,
        params=candidate.params,
        gate1_sharpe=gate1,
        gate2_drawdown=gate2,
        gate3_stability=gate3,
        gate4_not_material_below=gate4,
        fixed_sharpe=fixed_sharpe,
        adaptive_sharpe=adaptive_sharpe,
        fixed_max_drawdown_pct=fixed_dd_pct,
        parameter_stability_pct=stability_pct,
        fixed_trades=fixed_trades,
        drawdown_reduction_pp=reduction,
        fixed_adaptive_gap=gap,
        missing_baseline=missing,
    )


def _ranking_key(scored: GateScore) -> tuple[float, float, float, float]:
    """Ranking among passers, applied in order (each key sorts ascending):

    1. Smaller fixed-vs-adaptive Sharpe gap (robustness / coherence). A fixed
       baseline that tracks the adaptive result is trustworthy; a fixed number
       that sits far ABOVE its adaptive walk-forward on weak stability is a
       hindsight-fit signal, even though Gate 4 (which only guards the fixed-
       BELOW-adaptive direction) lets it pass.
    2. Higher parameter stability.
    3. Larger max-drawdown reduction.
    4. Higher fixed Sharpe (final tiebreak).

    None fields sort last (+inf).
    """
    inf = float("inf")

    def asc(value: float | None) -> float:
        """A 'smaller is better' field: None sorts last."""
        return inf if value is None else value

    def desc(value: float | None) -> float:
        """A 'larger is better' field, negated for ascending sort; None last."""
        return inf if value is None else -value

    return (
        asc(scored.fixed_adaptive_gap),
        desc(scored.parameter_stability_pct),
        desc(scored.drawdown_reduction_pp),
        desc(scored.fixed_sharpe),
    )


def _build_warnings(
    candidate: GateScore, all_scored: list[GateScore], thresholds: GateThresholds
) -> list[str]:
    """Deploy-nothing override checks, surfaced as warnings (never suppressed)."""
    warnings: list[str] = []

    trades = candidate.fixed_trades
    if trades is not None and trades < thresholds.min_trades_warn:
        warnings.append(
            f"Candidate '{candidate.label}' rests on only {trades} trades "
            f"(< {thresholds.min_trades_warn}); its edge could be a handful of lucky trades."
        )

    # The single highest fixed Sharpe in the whole set that fails ONLY the
    # stability gate -> best raw edge is a suspected curve-fit.
    rated = [s for s in all_scored if s.fixed_sharpe is not None]
    if rated:
        best_raw = max(rated, key=lambda s: s.fixed_sharpe or float("-inf"))
        if best_raw.failed_gates == ["gate3_stability"]:
            stability = best_raw.parameter_stability_pct or 0.0
            warnings.append(
                f"Highest fixed Sharpe in the set ({best_raw.fixed_sharpe:.2f}, "
                f"'{best_raw.label}') fails ONLY the stability gate "
                f"({stability:.1f}% < {thresholds.stability_min_pct:.0f}%) "
                f"- best raw edge is a suspected curve-fit."
            )

    return warnings


def score_candidates(
    candidates: list[CardCandidate],
    bh_metrics: dict[str, Any],
    thresholds: GateThresholds = DEFAULT_THRESHOLDS,
) -> GateReport:
    """Score WFA candidates against the deployment gates and derive a verdict.

    Args:
        candidates: Card candidates (label, params, adaptive summary, fixed
            baseline). Summary/fixed metrics are fractions; converted to percent
            internally.
        bh_metrics: Buy & Hold metrics in percent units (sharpe_ratio raw,
            max_drawdown in percent) as produced by the runner.
        thresholds: Gate thresholds; defaults to DEFAULT_THRESHOLDS.

    Returns:
        A GateReport with scored strategies ranked passers-first, the deploy
        candidate (if any), the verdict label, and any warnings.
    """
    bench_sharpe = float(bh_metrics.get("sharpe_ratio", float("nan")))
    bench_dd = float(bh_metrics.get("max_drawdown", float("nan")))

    scored = [
        _score_one(c, bench_sharpe, bench_dd, thresholds) for c in candidates
    ]
    scored.sort(key=lambda s: (not s.passed, _ranking_key(s)))

    passers = [s for s in scored if s.passed]
    if not passers:
        return GateReport(
            thresholds=thresholds,
            benchmark_sharpe=bench_sharpe,
            benchmark_max_drawdown_pct=bench_dd,
            scored=scored,
            candidate=None,
            verdict_label=DEPLOY_NOTHING,
            warnings=[],
        )

    candidate = passers[0]
    return GateReport(
        thresholds=thresholds,
        benchmark_sharpe=bench_sharpe,
        benchmark_max_drawdown_pct=bench_dd,
        scored=scored,
        candidate=candidate,
        verdict_label=candidate.label,
        warnings=_build_warnings(candidate, scored, thresholds),
    )


# ---------------------------------------------------------------------------
# LLM verdict extraction and comparison.
# ---------------------------------------------------------------------------

# The summary must end with a line of exactly this shape. Matched case-insensitively
# on the VERDICT token; the captured payload is compared verbatim to known labels.
_VERDICT_LINE = re.compile(r"(?im)^\s*VERDICT\s*:\s*(.+?)\s*$")


def parse_llm_verdict(summary_text: str, known_labels: list[str]) -> ParsedVerdict:
    """Strictly extract the machine-readable verdict from an LLM summary.

    The last VERDICT: line wins (a well-formed summary has exactly one, at the
    end). Matching against known labels is exact first, then case-insensitive; a
    verdict that resolves to no known label is reported as unresolved (present but
    unmatched), which the comparison treats as a failure — a missing or wrong
    verdict is as bad as a mismatched one.

    Args:
        summary_text: The full LLM summary string.
        known_labels: The exact candidate labels shown to the model.

    Returns:
        A ParsedVerdict describing presence, the raw payload, and the resolved
        label (a known label, DEPLOY_NOTHING, or None).
    """
    matches = _VERDICT_LINE.findall(summary_text or "")
    if not matches:
        return ParsedVerdict(present=False, raw=None, resolved=None)

    raw = matches[-1].strip()
    normalised = raw.upper().replace(" ", "_")
    if normalised == DEPLOY_NOTHING:
        return ParsedVerdict(present=True, raw=raw, resolved=DEPLOY_NOTHING)

    for label in known_labels:
        if raw == label:
            return ParsedVerdict(present=True, raw=raw, resolved=label)
    lowered = raw.lower()
    for label in known_labels:
        if lowered == label.lower():
            return ParsedVerdict(present=True, raw=raw, resolved=label)

    return ParsedVerdict(present=True, raw=raw, resolved=None)


def compare_verdicts(report: GateReport, parsed: ParsedVerdict) -> ValidationOutcome:
    """Compare the harness verdict against the parsed LLM verdict.

    A clean pass requires the LLM's machine-readable verdict to resolve to
    exactly the harness's pick (including both agreeing on DEPLOY_NOTHING). A
    missing verdict line, an unresolvable label, or a genuine disagreement all
    fail — the pipeline must not present the summary as a recommendation.

    Args:
        report: The harness scoring result.
        parsed: The extracted LLM verdict.

    Returns:
        A ValidationOutcome; ``ok`` is True only on an exact match.
    """
    harness = report.verdict_label

    if not parsed.present:
        return ValidationOutcome(
            ok=False,
            harness_verdict=harness,
            llm_verdict=None,
            reason="LLM summary contains no machine-readable 'VERDICT:' line.",
        )

    if parsed.resolved is None:
        return ValidationOutcome(
            ok=False,
            harness_verdict=harness,
            llm_verdict=parsed.raw,
            reason=(
                f"LLM verdict '{parsed.raw}' does not match any analysed strategy "
                "label or DEPLOY_NOTHING."
            ),
        )

    if parsed.resolved == harness:
        return ValidationOutcome(
            ok=True,
            harness_verdict=harness,
            llm_verdict=parsed.resolved,
            reason="LLM verdict matches the harness verdict.",
        )

    return ValidationOutcome(
        ok=False,
        harness_verdict=harness,
        llm_verdict=parsed.resolved,
        reason=(
            f"Verdict mismatch: harness says '{harness}', LLM says "
            f"'{parsed.resolved}'."
        ),
    )


# ---------------------------------------------------------------------------
# Plain-text rendering for the human at the console.
# ---------------------------------------------------------------------------

_HEADERS = [
    "Strategy",
    "G1",
    "G2",
    "G3",
    "G4",
    "FixedShrp",
    "AdaptShrp",
    "FixedDD",
    "Stability",
    "Trades",
    "Result",
]


def _mark(value: bool) -> str:
    return "PASS" if value else "----"


def _num(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _row_cells(scored: GateScore) -> list[str]:
    return [
        scored.label,
        _mark(scored.gate1_sharpe),
        _mark(scored.gate2_drawdown),
        _mark(scored.gate3_stability),
        _mark(scored.gate4_not_material_below),
        _num(scored.fixed_sharpe, ".2f"),
        _num(scored.adaptive_sharpe, ".2f"),
        _num(scored.fixed_max_drawdown_pct, ".1f") + ("%" if scored.fixed_max_drawdown_pct is not None else ""),
        _num(scored.parameter_stability_pct, ".1f") + ("%" if scored.parameter_stability_pct is not None else ""),
        "n/a" if scored.fixed_trades is None else str(scored.fixed_trades),
        "PASS" if scored.passed else "FAIL",
    ]


def render_gate_table(report: GateReport) -> str:
    """Render the deployment-gate pass/fail table plus verdict as plain text."""
    rows = [_row_cells(s) for s in report.scored]
    widths = [len(h) for h in _HEADERS]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [
        f"Benchmark (Buy & Hold): Sharpe {report.benchmark_sharpe:.2f}, "
        f"Max DD {report.benchmark_max_drawdown_pct:.1f}%",
        f"Gates: G1 fixed Sharpe >= benchmark | G2 DD >= {report.thresholds.drawdown_reduction_pp:.0f}pp "
        f"shallower | G3 stability >= {report.thresholds.stability_min_pct:.0f}% | "
        f"G4 fixed >= adaptive - {report.thresholds.material_below:.2f}",
        "",
        fmt(_HEADERS),
        fmt(["-" * w for w in widths]),
    ]
    lines.extend(fmt(row) for row in rows)

    lines.append("")
    if report.candidate is None:
        lines.append("Harness verdict: DEPLOY NOTHING - no strategy clears all four gates.")
    else:
        lines.append(f"Harness verdict: DEPLOY {report.verdict_label} with {report.candidate.params}")
    for warning in report.warnings:
        lines.append(f"  ! {warning}")

    return "\n".join(lines)
