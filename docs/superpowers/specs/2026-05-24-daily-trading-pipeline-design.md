# Daily Trading Pipeline — Design Spec

**Date:** 2026-05-24
**Status:** Spec (no implementation in this repo)
**Target repo:** new — `trading-pipeline/`
**Depends on:** the existing `backtest` package in this repo (consumed as a path/git dependency)

## 1. Purpose

A daily, automated pipeline that turns the research framework in this repo into an
operational decision support tool. After each market close (or before the next open)
it produces a single advisory recommendation for one pre-approved
(ticker, strategy) pair, persists the run, and pushes a notification to a
self-hosted ntfy.sh topic. There is no broker integration — the human
executes manually.

## 2. Scope (v1)

- **In scope:** one ticker, one pre-approved strategy, three-step pipeline
  (technical → research → arbiter), SQLite run log, ntfy notification, cron
  triggered CLI.
- **Out of scope:** multi-ticker fan-out, strategy selection at runtime, broker
  routing, paper/live mode toggle, position sizing, stop/target computation
  (strategy-defined exits only), backtest replay of recommendations.

## 3. Architecture

Single agno **Workflow** invoked by an external cron / systemd timer through a
CLI entrypoint. Three steps run sequentially with typed Pydantic payloads:

```
cron ──► CLI ──► Workflow
                  ├─ Step 1: technical_engine()    [pure function]
                  ├─ Step 2: research_agent        [agno Agent + Finnhub tools + Gemini]
                  └─ Step 3: arbiter()             [pure function + Gemini summary]
                                  │
                                  ├─► SQLite append/upsert
                                  └─► ntfy publish
```

Rationale for agno Workflow over Team: this is a deterministic sequential
pipeline, not collaborative reasoning. Workflow gives typed step composition
without inter-agent chatter. Per agno docs, agents chain via
`result = agent.run(input=...).content` with `output_schema=PydanticModel`.

## 4. Components

### 4.1 Technical Engine — `pipeline/technical.py`

Pure function. No LLM.

- Inputs: ticker (str), strategy descriptor from config.
- Downloads ~60 trading days via `yfinance.download(auto_adjust=True)` and
  flattens any multi-level columns (same helper as `trade_analysis.py`).
- Imports the configured strategy class from the existing `backtest` package
  (added as a `tool.uv.sources` path or git dep in `pyproject.toml`).
- Constructs the strategy with `strategy_params`, calls `generate_signals(data)`,
  reads the **last row** to determine action.
- Warmup guard: if `len(data) < strategy.warmup_period`, force `HOLD` and set
  `warmup_ok=False`.
- Returns a `TechnicalSignal` (see §5).
- `invalidation_price`: the strategy's natural exit reference when available
  (e.g., recent swing low for breakout). May be `None`; arbiter does not invent one.

### 4.2 Research Agent — `pipeline/research.py`

`agno.agent.Agent` with `model=Gemini(id="gemini-3.1-flash-lite")` and
`output_schema=ResearchReport`. Tools registered via `@tool`:

- `get_company_news(symbol, since_iso)` — wraps
  `finnhub.Client.company_news(symbol, _from=..., to=...)`. Returns
  `[{headline, url, datetime, source, summary}, ...]` truncated to
  `research.max_headlines`.
- `get_earnings_calendar(symbol, window_days)` — wraps
  `finnhub.Client.earnings_calendar(_from, to, symbol=...)`. Returns the next
  upcoming earnings date within window or `None`.

Agent instructions:
1. Call `get_company_news` for the configured `news_lookback_hours`.
2. For each headline, classify `pos|neu|neg` with a confidence in [0, 1].
3. Call `get_earnings_calendar` for `±earnings_window_days`.
4. Emit a `ResearchReport` matching the schema; one-sentence rationale.

**Regime post-processing (deterministic, applied after the agent returns):**

```
if earnings_in_window:                regime = RED
elif sentiment_score < -0.4:          regime = RED
elif sentiment_score < -0.1:          regime = YELLOW
else:                                  regime = GREEN
```

`sentiment_score = mean(sign(sentiment) * confidence)` over scored headlines,
in [-1, 1]. If `headline_count == 0`, treat as `YELLOW` (insufficient context).

Keeping the regime mapping out of the LLM ensures reproducibility and makes the
rule auditable from the SQLite log.

### 4.3 Arbiter — `pipeline/arbiter.py`

Pure function over `(TechnicalSignal, ResearchReport)`. Decision table:

| Technical | Regime  | Decision            |
|-----------|---------|---------------------|
| BUY       | GREEN   | RECOMMEND_BUY       |
| BUY       | YELLOW  | BUY_WITH_CAUTION    |
| BUY       | RED     | STAND_DOWN          |
| SELL      | *any*   | RECOMMEND_EXIT      |
| HOLD      | *any*   | NO_ACTION           |
| *any*     | *any*   | NO_ACTION (if `warmup_ok=False`) |

Exit signals always honored — never blocked by regime (an open position should
be closed when the strategy says so regardless of news context).

After deciding, one Gemini 3.5 Flash call generates `summary` (3–4 sentences,
plain prose) from the structured `TradeRecommendation`. On LLM failure the
summary falls back to a deterministic template: `"{decision} {ticker} @ {close}.
Strategy: {strategy_name} {strategy_params}. Regime: {regime} ({rationale})."`.

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
    confidence: float  # 0..1

class ResearchReport(BaseModel):
    ticker: str
    as_of: date
    regime: Literal["GREEN", "YELLOW", "RED"]
    sentiment_score: float          # -1..1
    headline_count: int
    earnings_in_window: bool
    next_earnings_date: date | None
    top_headlines: list[Headline]   # up to max_headlines
    rationale: str

class TradeRecommendation(BaseModel):
    run_id: UUID
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

Single-file SQLite (`runs.db`), stdlib `sqlite3`, no ORM. Append/upsert:

```sql
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL UNIQUE,
  ticker TEXT NOT NULL,
  as_of DATE NOT NULL,
  ran_at TIMESTAMP NOT NULL,
  technical_action TEXT NOT NULL,
  technical_close REAL NOT NULL,
  strategy_name TEXT NOT NULL,
  strategy_params_json TEXT NOT NULL,
  regime TEXT NOT NULL,
  sentiment_score REAL NOT NULL,
  earnings_in_window INTEGER NOT NULL,
  decision TEXT NOT NULL,
  summary TEXT NOT NULL,
  payload_json TEXT NOT NULL          -- full TradeRecommendation as JSON
);
CREATE INDEX IF NOT EXISTS idx_runs_ticker_asof ON runs(ticker, as_of);
```

Idempotency: natural key is `(ticker, as_of)`. Re-running the same trading day
replaces the row (`INSERT ... ON CONFLICT(ticker, as_of) DO UPDATE`).
Implementation note: enforce the natural key via a `UNIQUE(ticker, as_of)`
constraint in addition to the `run_id` unique constraint.

## 7. Notification — `pipeline/notify.py`

Single `publish(rec: TradeRecommendation) -> None`. POSTs to the configured
ntfy topic via `pyreqwest`.

- **Title:** `{decision} {ticker}`
- **Body:** `summary` followed by a key-facts block:
  ```
  Close: {close}
  Strategy: {strategy_name} {strategy_params}
  Regime: {regime} (sentiment={sentiment_score:+.2f}, {headline_count} headlines)
  {Earnings: {next_earnings_date} if earnings_in_window}
  {Invalidation: {invalidation_price} if present}
  ```
- **Tags** (ntfy emoji shortcodes):
  - `RECOMMEND_BUY` → `green_circle`
  - `BUY_WITH_CAUTION` → `warning`
  - `RECOMMEND_EXIT` → `red_circle`
  - `STAND_DOWN` → `no_entry`
  - `NO_ACTION` → `zzz`
- **Priority:** default; `RECOMMEND_BUY` / `RECOMMEND_EXIT` use `high`.
- Auth: optional bearer token from `NTFY_TOKEN`.

Always fires — even on pipeline failure rows — so silent failures are visible.

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
ntfy_url = "https://ntfy.example.com/trading"

[paths]
db_path = "runs.db"
```

Secrets via environment only — never in `config.toml`:
- `FINNHUB_API_KEY`
- `GOOGLE_API_KEY`
- `NTFY_TOKEN` (optional)

Config is loaded once at CLI startup into a frozen `PipelineConfig` dataclass
and passed explicitly into each step (no globals).

## 9. CLI & Scheduling

Entrypoint: `uv run python -m pipeline.daily [TICKER]`. If `TICKER` is omitted,
falls back to `config.ticker.symbol`. Exit codes:

- `0` — pipeline ran end to end (regardless of decision)
- `1` — unrecoverable error before/while persisting (notification still attempted)

Scheduling lives outside the code (systemd timer or cron). Example crontab:

```
30 21 * * 1-5  cd /opt/trading-pipeline && /usr/local/bin/uv run python -m pipeline.daily >> logs/daily.log 2>&1
```

## 10. Errors & Resilience

| Failure                          | Behavior                                                                 |
|----------------------------------|--------------------------------------------------------------------------|
| yfinance download error          | One retry (5s backoff). On second failure: log, persist `NO_ACTION` row with failure summary, notify, exit 1. |
| Strategy raises during signals   | Same as above.                                                           |
| Finnhub error (news/calendar)    | One retry. On second failure: continue with `headline_count=0` / `next_earnings_date=None`; regime becomes YELLOW. |
| Per-headline sentiment LLM error | Score that headline as `neu` / 0.0; pipeline continues.                  |
| Summary LLM error                | Fall back to deterministic template summary.                             |
| SQLite write error               | Log; still attempt ntfy publish so the user sees the run; exit 1.        |
| ntfy publish error               | Log; do not retry; SQLite write is the source of truth.                  |

All logging via `loguru` with `{extra}` enabled per CLAUDE.md, structured kwargs
for `ticker`, `as_of`, `decision`, `step`.

## 11. Testing

`pytest`, target ≥75% coverage. No live network in any test.

- `tests/test_technical.py` — synthetic OHLC fixtures; verify action extraction
  per strategy, warmup guard, last-bar selection, `invalidation_price` when present.
- `tests/test_research.py` — Finnhub tool callables monkeypatched; agent `.run()`
  patched to return canned JSON; verify the deterministic regime mapping
  (earnings flag, score thresholds, empty-headlines → YELLOW).
- `tests/test_arbiter.py` — table-driven over every (action × regime) combo and
  the `warmup_ok=False` short-circuit; assert decision string and that exit
  signals bypass regime.
- `tests/test_persistence.py` — round-trip a `TradeRecommendation`; rerun same
  `(ticker, as_of)` and assert single row with updated content.
- `tests/test_notify.py` — `pyreqwest` POST mocked; assert title, body lines,
  tags, and priority per decision.
- `tests/test_workflow_smoke.py` — end-to-end with every external dependency
  mocked; one happy path (`BUY + GREEN`) and one failure injection (Finnhub
  down → YELLOW).

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
│   ├── persistence.py          # SQLite append/upsert
│   └── notify.py               # ntfy publish
└── tests/
    ├── conftest.py
    ├── fixtures/
    ├── test_technical.py
    ├── test_research.py
    ├── test_arbiter.py
    ├── test_persistence.py
    ├── test_notify.py
    └── test_workflow_smoke.py
```

## 13. Dependencies (`pyproject.toml` — `[project]` only, no build-system)

- `agno`
- `google-genai` (Gemini)
- `finnhub-python`
- `pyreqwest`
- `pydantic >=2`
- `loguru`
- `yfinance`
- `pandas`, `numpy` (transitive via backtest)
- `backtest` — path or git source via `[tool.uv.sources]`, pinned to a commit

Dev: `pytest`, `pytest-cov`, `ruff`, `ty`.

## 14. Open Questions / Deferred

- **Multi-strategy voting on the same ticker** — deferred to v2; current design
  picks one pre-approved strategy from config.
- **Watchlist fan-out** — deferred; v1 runs per `uv run python -m pipeline.daily TICKER`
  invocation, so cron can list multiple tickers as separate jobs.
- **Performance attribution** — the SQLite schema records enough to compute
  hindsight P&L by joining future closes to each recommendation, but the
  attribution job itself is out of scope.
- **Position sizing / stops** — deferred; current design relies on
  strategy-defined exits plus an optional `invalidation_price` for the human's
  reference.
