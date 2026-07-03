"""Build and write the machine-readable strategy card emitted by trade_analysis.py.

The card is the single artifact that crosses laptop -> homelab. It records, per WFA
shortlist candidate, the strategy class, its best_params_overall, and the WFA baseline
metrics the reconcile tool needs (notably max_drawdown for the kill switch).
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keys copied verbatim from a WalkForwardResult.summary dict into the card baseline.
_BASELINE_KEYS = (
    "max_drawdown",
    "sharpe_ratio",
    "total_return",
    "calmar_ratio",
    "sortino_ratio",
    "n_windows",
    "n_windows_with_trades",
)


@dataclass(frozen=True)
class CardCandidate:
    """One WFA candidate destined for the strategy card.

    fixed_baseline holds metrics from a plain backtest of `params` (the exact
    parameters deployed live) over the WFA out-of-sample span — the WFA summary
    itself was earned by adaptive per-window parameters, so this is the number
    that matches what actually gets traded.
    """

    label: str
    strategy_class: str
    params: dict[str, Any]
    summary: dict[str, Any]
    fixed_baseline: dict[str, Any] | None = None


def _baseline(summary: dict[str, Any]) -> dict[str, Any]:
    return {k: summary[k] for k in _BASELINE_KEYS if k in summary}


def build_card(
    ticker: str,
    start_date: str,
    end_date: str,
    start_capital: float,
    candidates: list[CardCandidate],
) -> dict[str, Any]:
    """Build the strategy card dict from WFA shortlist candidates."""
    recommended = None
    if candidates:
        recommended = max(
            candidates, key=lambda c: c.summary.get("sharpe_ratio", float("-inf"))
        ).label

    return {
        "ticker": ticker,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backtest_window": {"start": start_date, "end": end_date},
        "start_capital": start_capital,
        "candidates": [
            {
                "label": c.label,
                "strategy_class": c.strategy_class,
                "params": c.params,
                "wfa_baseline": _baseline(c.summary),
                "fixed_params_baseline": c.fixed_baseline,
            }
            for c in candidates
        ],
        "recommended": recommended,
    }


def write_card(card: dict[str, Any], path: str | Path) -> None:
    """Write the card dict to a JSON file."""
    Path(path).write_text(json.dumps(card, indent=2))
