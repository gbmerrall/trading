"""LLM-powered executive summary generation for backtest results."""

import math
import os

from google import genai

from .deployment_gate import DEFAULT_THRESHOLDS, DEPLOY_NOTHING, GateThresholds
from .strategy_card import CardCandidate

# Gemini 3 series model for the summary. MEDIUM thinking: the verdict weighs
# multiple strategies across several criteria for a real deployment decision,
# and the call runs once per analysis so the cost delta over LOW is negligible.
SUMMARY_MODEL = "gemini-3.5-flash"
THINKING_LEVEL = "MEDIUM"

# Only the top candidates (already sorted by OOS Sharpe) go to the LLM; a
# 4-paragraph summary cannot meaningfully discuss more.
MAX_CANDIDATES = 5


def _fmt(value, spec: str = ".2f", scale: float = 1.0) -> str:
    """Format a numeric value, rendering non-finite or missing values as 'n/a'.

    Args:
        value: Numeric value (or None) to format.
        spec: format() spec applied to the scaled value.
        scale: Multiplier applied before formatting (e.g. 100 for percentages).

    Returns:
        Formatted string, or 'n/a' when the value is missing or non-finite.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    return format(v * scale, spec)


def _fmt_pct(value, spec: str = ".1f") -> str:
    """Format a fraction as a percentage string, or 'n/a' when not finite."""
    formatted = _fmt(value, spec, 100)
    return formatted if formatted == "n/a" else f"{formatted}%"


def build_summary_payload(
    ticker: str, candidates: list[CardCandidate], bh_metrics: dict
) -> str:
    """Build the data payload sent to the LLM.

    Args:
        ticker: The asset ticker symbol.
        candidates: Card candidates sorted best-first by WFA out-of-sample
            Sharpe. Only the first MAX_CANDIDATES are included.
        bh_metrics: Buy & Hold metrics dict (percent units for total_return
            and max_drawdown, raw Sharpe).

    Returns:
        The payload string.
    """
    shown = candidates[:MAX_CANDIDATES]

    payload = f"Here is the backtest data for {ticker}:\n\n"

    payload += "--- BENCHMARK (BUY & HOLD) ---\n"
    payload += (
        f"Total Return: {_fmt(bh_metrics.get('total_return'), '.1f')}%"
        " (IGNORE for deployment decisions)\n"
    )
    payload += f"Sharpe Ratio: {_fmt(bh_metrics.get('sharpe_ratio'))}\n"
    payload += f"Max Drawdown: {_fmt(bh_metrics.get('max_drawdown'), '.1f')}%\n\n"

    payload += (
        f"--- TOP {len(shown)} STRATEGY WFA RESULTS "
        f"(of {len(candidates)} analysed, ranked by out-of-sample Sharpe) ---\n"
        "All results are net of commissions and slippage; Sharpe is excess over the "
        "risk-free rate.\n\n"
    )

    for candidate in shown:
        summary = candidate.summary
        payload += f"Strategy: {candidate.label}\n"
        payload += "- Adaptive WFA (re-optimised each window):\n"
        payload += f"    Total Return: {_fmt_pct(summary.get('total_return'))}\n"
        payload += f"    Max Drawdown: {_fmt_pct(summary.get('max_drawdown'))}\n"
        payload += f"    Sharpe Ratio: {_fmt(summary.get('sharpe_ratio'))}\n"
        payload += (
            f"- Parameter Stability: {_fmt_pct(summary.get('param_stability'))}"
            " (share of windows selecting the modal parameter set)\n"
        )
        payload += f"- Deployed Params (modal across windows): {candidate.params}\n"
        if candidate.fixed_baseline:
            fixed = candidate.fixed_baseline
            payload += (
                "- Fixed-Params OOS Baseline (deployed params held constant over the same"
                " out-of-sample span):\n"
                f"    Sharpe Ratio: {_fmt(fixed.get('sharpe_ratio'))}\n"
                f"    Total Return: {_fmt_pct(fixed.get('total_return'))}\n"
                f"    Max Drawdown: {_fmt_pct(fixed.get('max_drawdown'))}\n"
                f"    Trades: {fixed.get('n_trades', 'n/a')}\n"
            )
        else:
            payload += "- Fixed-Params OOS Baseline: not available\n"
        payload += "\n"

    return payload


def build_system_instruction(thresholds: GateThresholds = DEFAULT_THRESHOLDS) -> str:
    """Build the LLM system instruction, wiring in the shared gate thresholds.

    The numeric cutoffs the analyst is told to apply (stability floor, drawdown
    reduction, small-sample trade count, material-below tolerance) come straight
    from ``thresholds`` — the same object the deterministic harness enforces — so
    the prose and the scoring can never drift.

    Args:
        thresholds: The deployment-gate thresholds shared with the harness.

    Returns:
        The full system-instruction string.
    """
    stability_min = thresholds.stability_min_pct
    min_trades = thresholds.min_trades_warn
    dd_pp = thresholds.drawdown_reduction_pp
    material = thresholds.material_below
    return (
        "You are a brutally honest Senior Quantitative Analyst advising a trader who wants to deploy a "
        "strategy starting TODAY — not someone looking back at history. Your job is to translate raw "
        "Walk-Forward Analysis data into a 4-paragraph plain-English executive summary. "
        "Apply these non-negotiable evaluation principles:\n\n"
        "PRINCIPLE 1 — NO HINDSIGHT BIAS: Buy & Hold total return is irrelevant for forward deployment. "
        "The trader cannot go back in time. Never recommend Buy & Hold on the basis of past total return. "
        "Its only role in this analysis is as a drawdown and Sharpe ratio reference point.\n\n"
        "PRINCIPLE 2 — SHARPE RATIO IS THE PRIMARY METRIC: A strategy with a higher Sharpe ratio than "
        "Buy & Hold is superior, regardless of total return. Sharpe ratio measures excess return over the "
        "risk-free rate per unit of risk actually taken, net of trading costs. A strategy that earns 30% "
        "with a Sharpe of 1.2 beats one that earns 200% with a Sharpe of 0.4 — every time, for a rational "
        "trader.\n\n"
        "PRINCIPLE 3 — DRAWDOWN PROTECTION IS CRITICAL: Surviving crashes is more important than "
        "capturing bull runs. A strategy that cuts maximum drawdown meaningfully versus Buy & Hold "
        f"(at least {dd_pp:.0f} percentage points shallower) while delivering a positive Sharpe is highly "
        "valuable — it lets the trader stay solvent and stay in the game. Emphasise this loudly if "
        "present.\n\n"
        "PRINCIPLE 4 — PARAMETER STABILITY GATES DEPLOYMENT: High out-of-sample returns mean nothing "
        "if parameters shift wildly across windows. A curve-fit strategy will fail when deployed live. "
        f"If parameter stability is below {stability_min:.0f}%, treat the strategy's returns with deep "
        f"scepticism. Stability between {stability_min:.0f}% and 60% is acceptable, not strong — describe "
        "it as such, never as proof against curve-fitting. Only above 60% may you call stability a genuine "
        "strength.\n\n"
        "PRINCIPLE 5 — THE FIXED-PARAMS BASELINE IS THE DEPLOYMENT ESTIMATE: The trader deploys ONE "
        "fixed parameter set (the deployed params shown), not the adaptive re-optimisation the WFA "
        "numbers were earned with. Where a fixed-params baseline is provided, weight it above the "
        "adaptive WFA result when judging deployability. A fixed baseline that falls more than "
        f"{material:.2f} Sharpe below the adaptive result means the edge lives in constant re-optimisation, "
        "not in the parameters — do not deploy on the adaptive number. A fixed baseline at or above the "
        "adaptive result is a good sign of robustness. Also sanity-check the trade count in both "
        "directions: with a flat fee per trade, very high trade counts erode small edges; and very low "
        f"trade counts (under ~{min_trades}) mean the baseline rests on a small sample whose Sharpe and "
        "return could be a handful of lucky trades — temper your confidence accordingly and say so "
        "explicitly.\n\n"
        "Structure your response as exactly 4 paragraphs:\n"
        "Paragraph 1 — The Benchmark Reality: Describe Buy & Hold's Sharpe ratio and max drawdown "
        "(not its total return). Explain how brutal that drawdown would feel in real-time.\n"
        "Paragraph 2 — The Contenders: Discuss only the two or three strategies genuinely worth "
        "considering; dismiss the rest in a single sentence. For each contender, lead with Sharpe ratio "
        "vs Buy & Hold's Sharpe, then the fixed-vs-adaptive comparison, then drawdown reduction, then "
        "parameter stability. Call out curve-fitting risk explicitly where stability is low or the fixed "
        "baseline collapses.\n"
        "Paragraph 3 — Risk Assessment: Which strategy (if any) provides genuine edge? A strategy passes "
        "only if it clears all four bars: fixed Sharpe >= Buy & Hold Sharpe, max drawdown at least "
        f"{dd_pp:.0f} percentage points shallower than Buy & Hold, parameter stability >= "
        f"{stability_min:.0f}%, and a fixed-params baseline that does not fall more than {material:.2f} "
        "Sharpe below the adaptive result.\n"
        "Paragraph 4 — The Verdict: One sentence naming the single strategy to deploy, or that nothing "
        "should be deployed. State the single most important reason. No hedging, no lists.\n\n"
        "After the four paragraphs, on a NEW FINAL LINE and nothing after it, emit a machine-readable "
        "verdict in EXACTLY this format:\n"
        "  VERDICT: <label>\n"
        "where <label> is copied VERBATIM from one of the 'Strategy:' names listed in the data above "
        f"(matching the strategy you recommend in Paragraph 4), or the literal token {DEPLOY_NOTHING} if "
        "no strategy clears all four bars. Do not invent a label, do not include the params, and do not "
        "write anything after this line. This line is parsed by an automated check.\n\n"
        "Avoid jargon. Write for a smart non-technical trader."
    )


def generate_executive_summary(
    ticker: str,
    candidates: list[CardCandidate],
    bh_metrics: dict,
    thresholds: GateThresholds = DEFAULT_THRESHOLDS,
) -> str:
    """Generate an LLM executive summary of the WFA results.

    The summary ends with a machine-readable ``VERDICT:`` line that the
    deployment-gate harness parses and compares against its own deterministic
    verdict; the ``thresholds`` are shared with that harness so the prose and the
    scoring stay in lockstep.

    Args:
        ticker: The asset ticker symbol (e.g. "SPY").
        candidates: Card candidates sorted best-first by WFA out-of-sample Sharpe.
        bh_metrics: Buy & Hold metrics dict with 'total_return', 'sharpe_ratio'
            and 'max_drawdown' (percent units for return/drawdown).
        thresholds: Deployment-gate thresholds shared with the harness.

    Returns:
        The LLM-generated summary string, or a graceful skip/error message.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return (
            "GOOGLE_API_KEY not found in environment variables. Skipping LLM summary."
        )

    system_instruction = build_system_instruction(thresholds)
    payload = build_summary_payload(ticker, candidates, bh_metrics)
    with open(f"output/{ticker.lower()}_payload.txt", "w") as f:
        f.write(system_instruction)
        f.write("\n\n")
        f.write(str(payload))

    try:
        client = genai.Client()
        response = client.models.generate_content(
            model=SUMMARY_MODEL,
            contents=payload,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                thinking_config=genai.types.ThinkingConfig(
                    thinking_level=THINKING_LEVEL,
                ),
            ),
        )
        return response.text or ""

    except Exception as e:
        return f"Error generating LLM summary: {str(e)}"
