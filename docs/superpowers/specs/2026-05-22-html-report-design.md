# HTML Report Design

**Date:** 2026-05-22
**Scope:** Replace terminal output + separate chart windows in `bhp_analysis.py` with a single self-contained HTML report.

---

## Goal

After running `uv run python bhp_analysis.py`, the user gets `output/report.html` — one file containing all analysis results (tables + charts) that opens in any browser. No server required, fully shareable.

---

## Architecture

### New module: `backtest/reporting_html.py`

Owns all HTML generation. Three functions:

- `_build_comparison_table(results, bh_metrics, bh_returns, dca_metrics, dca_returns) -> str`
  Renders the strategy comparison as an HTML `<table>` fragment. Strategies ranked by return. Benchmark rows visually separated with a muted left-border style.

- `_build_wfa_table(wfa_result) -> str`
  Renders the WFA per-window results as an HTML `<table>` fragment.

- `generate_report(data: ReportData, output_path: str) -> None`
  Assembles the full HTML page: inlines the table fragments, embeds plotly chart divs via `fig.to_html(full_html=False, include_plotlyjs="cdn")`, writes the file. Prints confirmation to terminal.

### New dataclass: `ReportData`

Defined in `reporting_html.py`. Carries everything `generate_report` needs:

```python
@dataclass
class ReportData:
    ticker: str
    start_date: str
    end_date: str
    start_capital: float
    results: list[tuple]          # (name, metrics_dict, returns_series)
    bh_metrics: dict
    bh_returns: pd.Series
    dca_metrics: dict
    dca_returns: pd.Series
    fig_comparison: go.Figure     # all-strategies equity curve
    wfa_result: Any               # WalkForwardResult or None
    fig_wfa_equity: go.Figure | None
    fig_wfa_params: go.Figure | None
```

### Changes to `bhp_analysis.py`

- Remove `print_comparison_table()` and `print_wfa_summary()` (output moves to HTML)
- Remove `plot_best_vs_benchmark()` and `plot_all_vs_benchmarks()` (replaced by `fig_comparison` built inside `reporting_html`)
- `main()` assembles a `ReportData` and calls `generate_report(data, "output/report.html")`
- Terminal output reduced to progress lines and the final "Report saved to..." confirmation

---

## HTML Structure

Single scrollable page. Sections in order:

1. **Header** — ticker, date range, run timestamp
2. **Strategy Comparison** — styled table (ranked by total return) + all-strategies equity curve chart
3. **Walk-Forward Analysis** — per-window table, overall summary stats block, WFA equity chart, parameter stability chart
   - If `wfa_result` is None, section shows "No WFA configured"

---

## Styling

Inline `<style>` block in the HTML head (no external CSS files):

- White background, system sans-serif font
- Table: alternating row shading, monospace for numeric columns
- Return column: green text for positive, red for negative
- Benchmark rows: muted background + thin coloured left-border to distinguish from strategy rows
- Section headers use a simple `<h2>` with a bottom border

---

## Plotly Charts

- Loaded via CDN: `include_plotlyjs="cdn"` — keeps file size small
- All three charts (comparison equity, WFA equity, parameter stability) embedded as `<div>` fragments using `fig.to_html(full_html=False)`
- Charts retain full interactivity (hover, zoom, pan) in the browser

---

## Output

- Path: `output/report.html` (directory auto-created)
- `fig.show()` calls removed — report is the only output artifact
- Terminal prints: `Report saved to output/report.html`

---

## Out of Scope

- Offline-only mode (CDN plotly) — can switch to `include_plotlyjs=True` later if needed
- Multiple tickers in one report
- PDF export
