# Daily Trading Pipeline — Design Spec

**Date:** 2026-05-24
**Status:** Spec (no implementation in this repo)
**Target repo:** new — `trading-pipeline/`
**Depends on:** the existing `backtest` package in this repo (consumed as a path/git dependency)
**Deployment context:** single-user, runs on the author's home network via cron.

## 1. Purpose

A daily, automated pipeline that turns the research framework in this repo into
an operational decision support tool. After each market close (or before the
next open) it produces a single advisory recommendation for one pre-approved
(ticker, strategy) pair, persists the run, and pushes a notification to a
self-hosted ntfy.sh topic. There is no broker integration — the human executes
manually.

## 2. Scope (v1)

- **In scope:** one ticker, one pre-approved strategy, three-step pipeline
  (technical → research → arbiter), SQLite run log, ntfy notification, cron
  triggered CLI.
- **Out of scope:** multi-ticker fan-out, strategy selection at runtime, broker
  routing, position sizing, computed stop/target (strategy-defined exits only),
  historical performance attribution.

## 3. Architecture

Single agno **Workflow** invoked by cron through a CLI entrypoint. Three steps
run sequentially with typed Pydantic payloads:

```
cron ──► CLI ──► Workflow
                  ├─ Step 1: technical_engine()    [pure function]
                  ├─ Step 2: research_agent        [agno Agent + Finnhub tools + Gemini]
                  └─ Step 3: arbiter()             [pure function + Gemini summary]
                                  │
                                  ├─► SQLite append/upsert
                                  └─► ntfy publish
```

Agno Workflow (not Team): this is a deterministic sequential pipeline, not
collaborative reasoning. Per agno docs, agents chain via
`result = agent.run(input=...).content` with `output_schema=PydanticModel`.

## 4. Components

### 4.1 Technical Engine — `pipeline/technical.py`

Pure function. No LLM.

- Inputs: ticker, strategy descriptor from config.
- Downloads ~60 trading days via `yfinance.download(auto_adjust=True)` and
  flattens any multi-level columns.
- Imports the configured strategy class from the existing `backtest` package
  (added as a path or git dep in `pyproject.toml` via `[tool.uv.sources]`).
- Constructs the strategy with `strategy_params`, calls `generate_signals(data)`,
  reads the last row to determine action.
- Warmup guard: if `len(data) < strategy.warmup_period`, force `HOLD` and set
  `warmup_ok=False`.
- Returns a `TechnicalSignal` (§5).
- `invalidation_price`: the strategy's natural exit reference when available
  (e.g., recent swing low for breakout). May be `None`; arbiter does not invent one.

### 4.2 Research Agent — `pipeline/research.py`

`agno.agent.Agent` with `model=Gemini(id="gemini-3.1-flash-lite")` and
`output_schema=ResearchReport`. Two tools registered via `@tool`:

- `get_company_news(symbol, since_iso)` — wraps
  `finnhub.Client.company_news(...)`. Returns up to `research.max_headlines`.
- `get_earnings_calendar(symbol, window_days)` — wraps
  `finnhub.Client.earnings_calendar(...)`. Returns the next upcoming earnings
  date within window or `None`.

Agent instructions:
1. Fetch news for the configured `news_lookback_hours`.
2. Classify each headline `pos|neu|neg` with confidence in [0, 1].
3. Fetch earnings calendar for `±earnings_window_days`.
4. Emit a `ResearchReport` matching the schema with a one-sentence rationale.

**Regime mapping (deterministic, applied after the agent returns) — kept out
of the LLM so the rule is auditable from the SQLite log:**

```
if earnings_in_window:                regime = RED
elif sentiment_score < -0.4:          regime = RED
elif sentiment_score < -0.1:          regime = YELLOW
elif headline_count == 0:             regime = YELLOW   # insufficient context
else:                                  regime = GREEN
```

`sentiment_score = mean(sign(sentiment) * confidence)` over scored headlines,
range [-1, 1].

### 4.3 Arbiter — `pipeline/arbiter.py`

Pure function over `(TechnicalSignal, ResearchReport)`. Decision table:

| Technical | Regime | Decision           |
|-----------|--------|--------------------|
| BUY       | GREEN  | RECOMMEND_BUY      |
| BUY       | YELLOW | BUY_WITH_CAUTION   |
| BUY       | RED    | STAND_DOWN         |
| SELL      | *any*  | RECOMMEND_EXIT     |
| HOLD      | *any*  | NO_ACTION          |
| *any*     | *any*  | NO_ACTION (if `warmup_ok=False`) |

Exit signals always honored — never blocked by regime (an open position should
be closed when the strategy says so regardless of news context).

After deciding, one Gemini 3.5 Flash call generates `summary` (3–4 sentences)
from the structured recommendation.

## 5. Data Contracts

All in `pipeline/models.py` (Pydantic v2):

```python
class TechnicalSignal(BaseModel):
    ticker: str
    as_of: date
    action: Literal["BUY", "SELL", "HOLD"]
    close: float
    strategy_name: str
    strategy_params: dict[str, Any]
    warmup_ok: bool
    invalidation_price: float | None

class Headline(BaseModel):
    headline: str
    url: str
    published_at: datetime
    sentiment: Literal["pos", "neu", "neg"]
    confidence: float

class ResearchReport(BaseModel):
    ticker: str
    as_of: date
    regime: Literal["GREEN", "YELLOW", "RED"]
    sentiment_score: float
    headline_count: int
    earnings_in_window: bool
    next_earnings_date: date | None
    top_headlines: list[Headline]
    rationale: str

class TradeRecommendation(BaseModel):
    ticker: str
    as_of: date
    decision: Literal[
        "RECOMMEND_BUY", "BUY_WITH_CAUTION",
        "STAND_DOWN", "RECOMMEND_EXIT", "NO_ACTION",
    ]
    technical: TechnicalSignal
    research: ResearchReport
    summary: str
```

## 6. Persistence — `pipeline/persistence.py`

Single-file SQLite (`runs.db`), stdlib `sqlite3`, no ORM. Single table:

```sql
CREATE TABLE IF NOT EXISTS runs (
  ticker TEXT NOT NULL,
  as_of DATE NOT NULL,
  ran_at TIMESTAMP NOT NULL,
  decision TEXT NOT NULL,
  technical_action TEXT NOT NULL,
  technical_close REAL NOT NULL,
  regime TEXT NOT NULL,
  sentiment_score REAL NOT NULL,
  earnings_in_window INTEGER NOT NULL,
  summary TEXT NOT NULL,
  payload_json TEXT NOT NULL,        -- full TradeRecommendation as JSON
  PRIMARY KEY (ticker, as_of)
);
```

Re-running the same trading day overwrites the row via `INSERT OR REPLACE`.
`payload_json` is the source of truth; the flat columns exist for quick `SELECT`
inspection from the sqlite CLI.

## 7. Notification — `pipeline/notify.py`

Single `publish(rec: TradeRecommendation) -> None`. POSTs to the configured
ntfy topic via `pyreqwest`.

- **Title:** `{decision} {ticker}`
- **Body:** `summary` followed by:
  ```
  Close: {close}
  Strategy: {strategy_name} {strategy_params}
  Regime: {regime} (sentiment={sentiment_score:+.2f}, {headline_count} headlines)
  {Earnings: {next_earnings_date}  — if earnings_in_window}
  {Invalidation: {invalidation_price}  — if present}
  ```
- **Tags:** `green_circle` (BUY), `warning` (BUY_WITH_CAUTION),
  `red_circle` (EXIT), `no_entry` (STAND_DOWN), `zzz` (NO_ACTION).

Home-network ntfy → no auth header needed by default. If the topic is
protected, set `NTFY_TOKEN` and the publisher adds a bearer header.

## 8. Configuration

`config.toml` at repo root, loaded with stdlib `tomllib`:

```toml
[ticker]
symbol = "NVDA"

[strategy]
name = "Breakout"                    # resolved against backtest.strategy
params = { lookback_period = 20 }

[data]
lookback_days = 60

[research]
news_lookback_hours  = 24
earnings_window_days = 2
max_headlines        = 10

[llm]
sentiment_model = "gemini-3.1-flash-lite"
summary_model   = "gemini-3.5-flash"

[notify]
ntfy_url = "https://ntfy.home.lan/trading"

[paths]
db_path = "runs.db"
```

Secrets via environment only: `FINNHUB_API_KEY`, `GOOGLE_API_KEY`, optional
`NTFY_TOKEN`. Loaded once into a frozen `PipelineConfig` dataclass and passed
explicitly through the steps (no globals).

## 9. CLI & Scheduling

Entrypoint: `uv run python -m pipeline.daily [TICKER]`. If `TICKER` is omitted,
falls back to `config.ticker.symbol`.

Cron — runs weekdays after US close (adjust TZ on the host):

```
30 21 * * 1-5  cd /opt/trading-pipeline && /usr/local/bin/uv run python -m pipeline.daily >> logs/daily.log 2>&1
```

## 10. Errors

Fail-fast, no retries — this is a single-user advisory tool on cron; if a
component is down the simplest signal is no message and a log line.

| Failure                          | Behavior                                                     |
|----------------------------------|--------------------------------------------------------------|
| yfinance / Finnhub network error | Log and exit non-zero. No row written, no ntfy.              |
| Per-headline sentiment LLM error | Score that headline as `neu` / 0.0; pipeline continues.      |
| Summary LLM error                | Fall back to deterministic template summary.                 |
| SQLite write error               | Log and exit; do not publish ntfy (it would be a lie).       |
| ntfy publish error               | Log; SQLite write is the source of truth.                    |

Logging via `loguru` with `{extra}` in the format string per CLAUDE.md;
structured kwargs for `ticker`, `as_of`, `decision`, `step`. Cron captures
stdout/stderr into `logs/daily.log`.

## 11. Testing

`pytest`, target ≥75% coverage. No live network in any test.

- `tests/test_technical.py` — synthetic OHLC fixtures; verify action extraction
  per strategy, warmup guard, last-bar selection, `invalidation_price` when
  the strategy exposes one.
- `tests/test_research.py` — Finnhub tool callables monkeypatched; agent
  `.run()` patched to return canned JSON; assert the deterministic regime
  mapping (earnings flag, score thresholds, empty-headlines → YELLOW).
- `tests/test_arbiter.py` — table-driven over every (action × regime) combo
  and the `warmup_ok=False` short-circuit; assert decision and that exits
  bypass regime.
- `tests/test_persistence.py` — round-trip a `TradeRecommendation`; rerun
  same `(ticker, as_of)` and assert single row with updated content.
- `tests/test_notify.py` — `pyreqwest` POST mocked; assert title, body, tags.
- `tests/test_workflow_smoke.py` — end-to-end with all external calls mocked.

## 12. Repo Layout

```
trading-pipeline/
├── pyproject.toml
├── config.toml
├── runs.db                     # gitignored
├── logs/                       # gitignored
├── pipeline/
│   ├── __init__.py
│   ├── daily.py                # CLI: python -m pipeline.daily
│   ├── workflow.py             # agno Workflow wiring
│   ├── config.py               # tomllib loader + PipelineConfig dataclass
│   ├── models.py               # Pydantic contracts
│   ├── technical.py            # Technical Engine
│   ├── research.py             # Research Agent + Finnhub tools
│   ├── arbiter.py              # Arbiter + summary LLM call
│   ├── persistence.py          # SQLite upsert
│   └── notify.py               # ntfy publish
└── tests/
```

## 13. Dependencies (`pyproject.toml` — `[project]` only, no build-system)

- `agno`
- `google-genai`
- `finnhub-python`
- `pyreqwest`
- `pydantic >=2`
- `loguru`
- `yfinance`
- `backtest` — path or git source via `[tool.uv.sources]`, pinned to a commit

Dev: `pytest`, `pytest-cov`, `ruff`, `ty`.

## 14. Deferred

- Multi-strategy voting on the same ticker.
- Watchlist fan-out (v1 = one cron line per ticker if needed).
- Hindsight P&L attribution job (schema already supports it).
- Computed stops/targets/sizing.
