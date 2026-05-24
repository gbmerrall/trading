# Daily Trading Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a daily advisory pipeline that runs a pre-approved strategy against fresh price data, augments the signal with LLM-scored news + earnings context, persists the run to SQLite, and notifies via ntfy.

**Architecture:** Single agno Workflow with three sequential steps (Technical Engine → Research Agent → Arbiter), typed Pydantic payloads between steps. Cron-triggered CLI. See spec: `docs/superpowers/specs/2026-05-24-daily-trading-pipeline-design.md`.

**Tech Stack:** Python ≥3.12, `agno` (workflow + Gemini), `google-genai`, `finnhub-python`, `yfinance`, `pydantic v2`, `pyreqwest` (ntfy POST), `loguru`, stdlib `sqlite3` + `tomllib`. Existing `backtest` package consumed as a path/git dep.

**Repo:** new repo `trading-pipeline/` — none of this is implemented in the current `trading` repo. The `backtest` package from this repo is imported as a dependency.

---

## File Structure

```
trading-pipeline/
├── pyproject.toml              # uv project, no [build-system] (CLI-only)
├── config.toml                 # runtime config (committed; secrets via env)
├── .env.example                # documents required env vars
├── .gitignore                  # runs.db, logs/, .venv/, .env
├── README.md                   # short: setup, cron line, manual run
├── pipeline/
│   ├── __init__.py
│   ├── models.py               # Pydantic contracts (TechnicalSignal, ResearchReport, ...)
│   ├── config.py               # tomllib loader, PipelineConfig dataclass
│   ├── technical.py            # Technical Engine (pure)
│   ├── finnhub_tools.py        # @tool wrappers around finnhub client
│   ├── research.py             # Research Agent + deterministic regime mapping
│   ├── arbiter.py              # Decision table + summary LLM call
│   ├── persistence.py          # SQLite upsert
│   ├── notify.py               # ntfy publish
│   ├── workflow.py             # wires the three steps together
│   └── daily.py                # CLI entrypoint (python -m pipeline.daily)
└── tests/
    ├── conftest.py             # shared fixtures (synthetic OHLC, etc.)
    ├── fixtures/
    │   ├── breakout_buy.csv    # OHLC fixture that triggers BUY on Breakout(20)
    │   ├── breakout_hold.csv   # OHLC fixture that triggers HOLD
    │   ├── finnhub_news.json   # sample finnhub news payload
    │   └── finnhub_earnings.json
    ├── test_models.py
    ├── test_config.py
    ├── test_technical.py
    ├── test_finnhub_tools.py
    ├── test_research.py
    ├── test_arbiter.py
    ├── test_persistence.py
    ├── test_notify.py
    └── test_workflow_smoke.py
```

One file = one responsibility. The Technical Engine, Arbiter, and Persistence are pure functions over Pydantic models — trivial to unit test. The Research Agent isolates the LLM/tool surface in a single file so the rest of the pipeline can be tested without mocking agno.

---

## Task 1: Project bootstrap

**Files:**
- Create: `trading-pipeline/pyproject.toml`
- Create: `trading-pipeline/.gitignore`
- Create: `trading-pipeline/.env.example`
- Create: `trading-pipeline/config.toml`
- Create: `trading-pipeline/pipeline/__init__.py` (empty)

- [ ] **Step 1: Initialize uv project**

```bash
mkdir -p trading-pipeline && cd trading-pipeline
uv init --no-package --python 3.12
rm -f main.py hello.py  # remove uv init scaffolding
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "trading-pipeline"
version = "0.1.0"
description = "Daily advisory trading pipeline (technical + research + arbiter)"
requires-python = ">=3.12"
dependencies = [
    "agno>=0.3",
    "google-genai>=0.3",
    "finnhub-python>=2.4",
    "pyreqwest>=0.5",
    "pydantic>=2.5",
    "loguru>=0.7",
    "yfinance>=0.2",
    "pandas>=2.0",
    "numpy>=1.26",
    # backtest package from the trading repo
    "backtest",
]

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "ruff>=0.6",
    "ty>=0.0.1a1",
]

[tool.uv.sources]
# Adjust path to wherever the trading repo lives locally; switch to a git source for deployment.
backtest = { path = "../trading", editable = true }

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ty]
# placeholder; ty picks sane defaults
```

- [ ] **Step 3: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/
runs.db
runs.db-journal
logs/
.env
```

- [ ] **Step 4: Write `.env.example`**

```
FINNHUB_API_KEY=
GOOGLE_API_KEY=
NTFY_TOKEN=   # optional; only if your ntfy topic is auth-protected
```

- [ ] **Step 5: Write `config.toml` (default values)**

```toml
[ticker]
symbol = "NVDA"

[strategy]
name   = "BreakoutStrategy"
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

- [ ] **Step 6: Sync and verify**

```bash
uv sync
uv run python -c "import agno, finnhub, yfinance, pyreqwest, loguru, pydantic; print('ok')"
```
Expected: `ok`

- [ ] **Step 7: Commit**

```bash
git init && git add . && git commit -m "chore: bootstrap trading-pipeline project"
```

---

## Task 2: Pydantic contracts (`models.py`)

**Files:**
- Create: `trading-pipeline/pipeline/models.py`
- Create: `trading-pipeline/tests/__init__.py` (empty)
- Create: `trading-pipeline/tests/test_models.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_models.py
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from pipeline.models import (
    Headline,
    ResearchReport,
    TechnicalSignal,
    TradeRecommendation,
)


def _signal(**overrides):
    base = dict(
        ticker="NVDA",
        as_of=date(2026, 5, 22),
        action="BUY",
        close=950.0,
        strategy_name="BreakoutStrategy",
        strategy_params={"lookback_period": 20},
        warmup_ok=True,
        invalidation_price=915.0,
    )
    base.update(overrides)
    return TechnicalSignal(**base)


def _report(**overrides):
    base = dict(
        ticker="NVDA",
        as_of=date(2026, 5, 22),
        regime="GREEN",
        sentiment_score=0.3,
        headline_count=5,
        earnings_in_window=False,
        next_earnings_date=None,
        top_headlines=[],
        rationale="Stable coverage with mildly positive sentiment.",
    )
    base.update(overrides)
    return ResearchReport(**base)


def test_technical_signal_roundtrips():
    sig = _signal()
    assert TechnicalSignal.model_validate_json(sig.model_dump_json()) == sig


def test_technical_signal_rejects_bad_action():
    with pytest.raises(ValidationError):
        _signal(action="MAYBE")


def test_headline_requires_confidence_in_range():
    with pytest.raises(ValidationError):
        Headline(
            headline="x",
            url="https://x",
            published_at=datetime.now(timezone.utc),
            sentiment="pos",
            confidence=1.5,
        )


def test_research_report_rejects_bad_regime():
    with pytest.raises(ValidationError):
        _report(regime="MAGENTA")


def test_trade_recommendation_composes():
    rec = TradeRecommendation(
        ticker="NVDA",
        as_of=date(2026, 5, 22),
        decision="RECOMMEND_BUY",
        technical=_signal(),
        research=_report(),
        summary="Buy recommended.",
    )
    assert rec.decision == "RECOMMEND_BUY"
```

- [ ] **Step 2: Run test (expect ImportError)**

```bash
uv run pytest tests/test_models.py -v
```
Expected: collection error / ImportError for `pipeline.models`.

- [ ] **Step 3: Implement `pipeline/models.py`**

```python
"""Pydantic contracts passed between pipeline steps."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TechnicalSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    as_of: date
    action: Literal["BUY", "SELL", "HOLD"]
    close: float
    strategy_name: str
    strategy_params: dict[str, Any]
    warmup_ok: bool
    invalidation_price: float | None = None


class Headline(BaseModel):
    model_config = ConfigDict(frozen=True)

    headline: str
    url: str
    published_at: datetime
    sentiment: Literal["pos", "neu", "neg"]
    confidence: float = Field(ge=0.0, le=1.0)


class ResearchReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    as_of: date
    regime: Literal["GREEN", "YELLOW", "RED"]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    headline_count: int = Field(ge=0)
    earnings_in_window: bool
    next_earnings_date: date | None = None
    top_headlines: list[Headline]
    rationale: str


class TradeRecommendation(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    as_of: date
    decision: Literal[
        "RECOMMEND_BUY",
        "BUY_WITH_CAUTION",
        "STAND_DOWN",
        "RECOMMEND_EXIT",
        "NO_ACTION",
    ]
    technical: TechnicalSignal
    research: ResearchReport
    summary: str
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
uv run pytest tests/test_models.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/models.py tests/__init__.py tests/test_models.py
git commit -m "feat: add pydantic contracts for pipeline payloads"
```

---

## Task 3: Config loader (`config.py`)

**Files:**
- Create: `trading-pipeline/pipeline/config.py`
- Create: `trading-pipeline/tests/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
import textwrap

import pytest

from pipeline.config import PipelineConfig, load_config


def _write(tmp_path, body: str):
    p = tmp_path / "config.toml"
    p.write_text(textwrap.dedent(body).lstrip())
    return p


def test_load_config_reads_all_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "fk")
    monkeypatch.setenv("GOOGLE_API_KEY", "gk")
    cfg_path = _write(
        tmp_path,
        """
        [ticker]
        symbol = "NVDA"

        [strategy]
        name = "BreakoutStrategy"
        params = { lookback_period = 20 }

        [data]
        lookback_days = 60

        [research]
        news_lookback_hours = 24
        earnings_window_days = 2
        max_headlines = 10

        [llm]
        sentiment_model = "gemini-3.1-flash-lite"
        summary_model   = "gemini-3.5-flash"

        [notify]
        ntfy_url = "https://ntfy.home.lan/trading"

        [paths]
        db_path = "runs.db"
        """,
    )
    cfg = load_config(cfg_path)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.ticker == "NVDA"
    assert cfg.strategy_name == "BreakoutStrategy"
    assert cfg.strategy_params == {"lookback_period": 20}
    assert cfg.lookback_days == 60
    assert cfg.news_lookback_hours == 24
    assert cfg.earnings_window_days == 2
    assert cfg.max_headlines == 10
    assert cfg.sentiment_model == "gemini-3.1-flash-lite"
    assert cfg.summary_model == "gemini-3.5-flash"
    assert cfg.ntfy_url == "https://ntfy.home.lan/trading"
    assert cfg.db_path.name == "runs.db"
    assert cfg.finnhub_api_key == "fk"
    assert cfg.google_api_key == "gk"
    assert cfg.ntfy_token is None  # optional


def test_load_config_requires_finnhub_key(tmp_path, monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "gk")
    cfg_path = _write(
        tmp_path,
        """
        [ticker]
        symbol = "NVDA"
        [strategy]
        name = "BreakoutStrategy"
        params = {}
        [data]
        lookback_days = 60
        [research]
        news_lookback_hours = 24
        earnings_window_days = 2
        max_headlines = 10
        [llm]
        sentiment_model = "x"
        summary_model = "y"
        [notify]
        ntfy_url = "https://x"
        [paths]
        db_path = "runs.db"
        """,
    )
    with pytest.raises(RuntimeError, match="FINNHUB_API_KEY"):
        load_config(cfg_path)


def test_pipeline_config_is_frozen(tmp_path, monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "fk")
    monkeypatch.setenv("GOOGLE_API_KEY", "gk")
    cfg_path = _write(
        tmp_path,
        """
        [ticker]
        symbol = "NVDA"
        [strategy]
        name = "BreakoutStrategy"
        params = {}
        [data]
        lookback_days = 60
        [research]
        news_lookback_hours = 24
        earnings_window_days = 2
        max_headlines = 10
        [llm]
        sentiment_model = "x"
        summary_model = "y"
        [notify]
        ntfy_url = "https://x"
        [paths]
        db_path = "runs.db"
        """,
    )
    cfg = load_config(cfg_path)
    with pytest.raises(Exception):
        cfg.ticker = "AAPL"  # type: ignore[misc]
```

- [ ] **Step 2: Run test (expect ImportError)**

```bash
uv run pytest tests/test_config.py -v
```
Expected: ImportError for `pipeline.config`.

- [ ] **Step 3: Implement `pipeline/config.py`**

```python
"""Configuration loader. TOML for tunables, env vars for secrets."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    ticker: str
    strategy_name: str
    strategy_params: dict[str, Any]
    lookback_days: int
    news_lookback_hours: int
    earnings_window_days: int
    max_headlines: int
    sentiment_model: str
    summary_model: str
    ntfy_url: str
    db_path: Path
    finnhub_api_key: str
    google_api_key: str
    ntfy_token: str | None


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_config(config_path: Path | str = "config.toml") -> PipelineConfig:
    path = Path(config_path)
    raw = tomllib.loads(path.read_text())
    return PipelineConfig(
        ticker=raw["ticker"]["symbol"],
        strategy_name=raw["strategy"]["name"],
        strategy_params=dict(raw["strategy"].get("params", {})),
        lookback_days=int(raw["data"]["lookback_days"]),
        news_lookback_hours=int(raw["research"]["news_lookback_hours"]),
        earnings_window_days=int(raw["research"]["earnings_window_days"]),
        max_headlines=int(raw["research"]["max_headlines"]),
        sentiment_model=raw["llm"]["sentiment_model"],
        summary_model=raw["llm"]["summary_model"],
        ntfy_url=raw["notify"]["ntfy_url"],
        db_path=Path(raw["paths"]["db_path"]),
        finnhub_api_key=_require_env("FINNHUB_API_KEY"),
        google_api_key=_require_env("GOOGLE_API_KEY"),
        ntfy_token=os.environ.get("NTFY_TOKEN"),
    )
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
uv run pytest tests/test_config.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/config.py tests/test_config.py
git commit -m "feat: add tomllib-based config loader with env-var secrets"
```

---

## Task 4: Technical Engine (`technical.py`)

**Files:**
- Create: `trading-pipeline/tests/conftest.py`
- Create: `trading-pipeline/tests/fixtures/breakout_buy.csv`
- Create: `trading-pipeline/tests/fixtures/breakout_hold.csv`
- Create: `trading-pipeline/pipeline/technical.py`
- Create: `trading-pipeline/tests/test_technical.py`

- [ ] **Step 1: Generate the OHLC fixtures**

These are checked-in CSVs so tests never hit yfinance. Run this once to create them, then commit the resulting files.

```bash
mkdir -p tests/fixtures
uv run python - <<'PY'
import pandas as pd
import numpy as np
from pathlib import Path

# 60 bars; flat then a clean breakout on the last bar → BUY for Breakout(20)
dates = pd.bdate_range("2026-02-01", periods=60)
close = np.concatenate([np.full(59, 100.0), [120.0]])
high  = close + 1.0
low   = close - 1.0
open_ = close
vol   = np.full(60, 1_000_000)
buy = pd.DataFrame(
    {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
    index=dates,
)
buy.index.name = "Date"
buy.to_csv("tests/fixtures/breakout_buy.csv")

# 60 bars, perfectly flat → HOLD for Breakout(20)
hold = pd.DataFrame(
    {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000_000},
    index=dates,
)
hold.index.name = "Date"
hold.to_csv("tests/fixtures/breakout_hold.csv")
PY
```

- [ ] **Step 2: Write `tests/conftest.py`**

```python
"""Shared fixtures."""

from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load_ohlc(name: str) -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / name, parse_dates=["Date"], index_col="Date")
    return df


@pytest.fixture
def ohlc_breakout_buy() -> pd.DataFrame:
    return _load_ohlc("breakout_buy.csv")


@pytest.fixture
def ohlc_breakout_hold() -> pd.DataFrame:
    return _load_ohlc("breakout_hold.csv")
```

- [ ] **Step 3: Write failing test**

```python
# tests/test_technical.py
from datetime import date

import pandas as pd
import pytest

from pipeline.models import TechnicalSignal
from pipeline.technical import run_technical_engine


def test_breakout_emits_buy(monkeypatch, ohlc_breakout_buy):
    monkeypatch.setattr(
        "pipeline.technical._download",
        lambda ticker, lookback_days: ohlc_breakout_buy,
    )
    sig = run_technical_engine(
        ticker="NVDA",
        strategy_name="BreakoutStrategy",
        strategy_params={"lookback_period": 20},
        lookback_days=60,
    )
    assert isinstance(sig, TechnicalSignal)
    assert sig.action == "BUY"
    assert sig.warmup_ok is True
    assert sig.close == pytest.approx(120.0)
    assert sig.ticker == "NVDA"
    assert sig.as_of == ohlc_breakout_buy.index[-1].date()


def test_flat_data_emits_hold(monkeypatch, ohlc_breakout_hold):
    monkeypatch.setattr(
        "pipeline.technical._download",
        lambda ticker, lookback_days: ohlc_breakout_hold,
    )
    sig = run_technical_engine(
        ticker="NVDA",
        strategy_name="BreakoutStrategy",
        strategy_params={"lookback_period": 20},
        lookback_days=60,
    )
    assert sig.action == "HOLD"
    assert sig.warmup_ok is True


def test_insufficient_data_forces_hold(monkeypatch, ohlc_breakout_buy):
    short = ohlc_breakout_buy.iloc[:5]
    monkeypatch.setattr("pipeline.technical._download", lambda *a, **k: short)
    sig = run_technical_engine(
        ticker="NVDA",
        strategy_name="BreakoutStrategy",
        strategy_params={"lookback_period": 20},
        lookback_days=60,
    )
    assert sig.action == "HOLD"
    assert sig.warmup_ok is False


def test_unknown_strategy_raises(monkeypatch, ohlc_breakout_buy):
    monkeypatch.setattr("pipeline.technical._download", lambda *a, **k: ohlc_breakout_buy)
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_technical_engine(
            ticker="NVDA",
            strategy_name="DoesNotExist",
            strategy_params={},
            lookback_days=60,
        )
```

- [ ] **Step 4: Run tests (expect import error)**

```bash
uv run pytest tests/test_technical.py -v
```
Expected: ImportError for `pipeline.technical`.

- [ ] **Step 5: Implement `pipeline/technical.py`**

```python
"""Technical Engine: download data → run strategy → emit last-bar signal."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from loguru import logger

import backtest.strategy as strategies
from pipeline.models import TechnicalSignal


def _download(ticker: str, lookback_days: int) -> pd.DataFrame:
    end = date.today()
    # Buffer for weekends/holidays.
    start = end - timedelta(days=int(lookback_days * 1.6) + 5)
    data = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False
    )
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)
    return data


def _resolve_strategy(name: str) -> type:
    cls = getattr(strategies, name, None)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}")
    return cls


def _last_action(signals: pd.DataFrame) -> str:
    last = signals.iloc[-1]
    if bool(last.get("buy", False)):
        return "BUY"
    if bool(last.get("sell", False)):
        return "SELL"
    return "HOLD"


def _invalidation_price(action: str, data: pd.DataFrame, params: dict[str, Any]) -> float | None:
    """For Breakout-style strategies: invalidation = lookback-period low."""
    if action != "BUY":
        return None
    lookback = params.get("lookback_period")
    if not lookback:
        return None
    window = data["Low"].iloc[-(int(lookback) + 1) : -1]
    return float(window.min()) if not window.empty else None


def run_technical_engine(
    ticker: str,
    strategy_name: str,
    strategy_params: dict[str, Any],
    lookback_days: int,
) -> TechnicalSignal:
    data = _download(ticker, lookback_days)
    strategy_cls = _resolve_strategy(strategy_name)
    strategy = strategy_cls(**strategy_params)
    as_of = data.index[-1].date()
    close = float(data["Close"].iloc[-1])

    if len(data) < strategy.warmup_period:
        logger.warning(
            "Insufficient data for warmup; forcing HOLD",
            ticker=ticker,
            bars=len(data),
            warmup=strategy.warmup_period,
        )
        return TechnicalSignal(
            ticker=ticker,
            as_of=as_of,
            action="HOLD",
            close=close,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            warmup_ok=False,
            invalidation_price=None,
        )

    signals = strategy.generate_signals(data)
    action = _last_action(signals)
    return TechnicalSignal(
        ticker=ticker,
        as_of=as_of,
        action=action,
        close=close,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        warmup_ok=True,
        invalidation_price=_invalidation_price(action, data, strategy_params),
    )
```

- [ ] **Step 6: Run tests (expect pass)**

```bash
uv run pytest tests/test_technical.py -v
```
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add pipeline/technical.py tests/conftest.py tests/test_technical.py tests/fixtures/
git commit -m "feat: add Technical Engine with yfinance + backtest.strategy"
```

---

## Task 5: Finnhub tool wrappers (`finnhub_tools.py`)

**Files:**
- Create: `trading-pipeline/tests/fixtures/finnhub_news.json`
- Create: `trading-pipeline/tests/fixtures/finnhub_earnings.json`
- Create: `trading-pipeline/pipeline/finnhub_tools.py`
- Create: `trading-pipeline/tests/test_finnhub_tools.py`

- [ ] **Step 1: Write fixtures**

`tests/fixtures/finnhub_news.json`:
```json
[
  {
    "category": "company",
    "datetime": 1779148800,
    "headline": "NVDA crushes earnings expectations",
    "id": 1,
    "image": "",
    "related": "NVDA",
    "source": "Reuters",
    "summary": "...",
    "url": "https://example.com/1"
  },
  {
    "category": "company",
    "datetime": 1779062400,
    "headline": "Analyst downgrades NVDA over valuation concerns",
    "id": 2,
    "image": "",
    "related": "NVDA",
    "source": "Bloomberg",
    "summary": "...",
    "url": "https://example.com/2"
  }
]
```

`tests/fixtures/finnhub_earnings.json`:
```json
{
  "earningsCalendar": [
    {
      "date": "2026-05-23",
      "epsActual": null,
      "epsEstimate": 1.05,
      "hour": "amc",
      "quarter": 1,
      "revenueActual": null,
      "revenueEstimate": 26000000000,
      "symbol": "NVDA",
      "year": 2026
    }
  ]
}
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_finnhub_tools.py
import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pipeline.finnhub_tools import (
    build_finnhub_tools,
    get_company_news_impl,
    get_earnings_calendar_impl,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_client():
    client = MagicMock()
    client.company_news.return_value = json.loads(
        (FIXTURES / "finnhub_news.json").read_text()
    )
    client.earnings_calendar.return_value = json.loads(
        (FIXTURES / "finnhub_earnings.json").read_text()
    )
    return client


def test_get_company_news_truncates_and_normalizes(fake_client):
    result = get_company_news_impl(
        client=fake_client,
        symbol="NVDA",
        since=datetime(2026, 5, 22, tzinfo=timezone.utc),
        max_headlines=10,
    )
    assert len(result) == 2
    first = result[0]
    assert first["symbol"] == "NVDA"
    assert first["headline"].startswith("NVDA crushes")
    assert "url" in first
    assert isinstance(first["published_at"], str)  # ISO-formatted


def test_get_company_news_respects_max(fake_client):
    result = get_company_news_impl(
        client=fake_client,
        symbol="NVDA",
        since=datetime(2026, 5, 22, tzinfo=timezone.utc),
        max_headlines=1,
    )
    assert len(result) == 1


def test_get_earnings_calendar_returns_next(fake_client):
    result = get_earnings_calendar_impl(
        client=fake_client,
        symbol="NVDA",
        reference_day=date(2026, 5, 22),
        window_days=2,
    )
    assert result == date(2026, 5, 23)


def test_get_earnings_calendar_returns_none_when_empty(fake_client):
    fake_client.earnings_calendar.return_value = {"earningsCalendar": []}
    result = get_earnings_calendar_impl(
        client=fake_client,
        symbol="NVDA",
        reference_day=date(2026, 5, 22),
        window_days=2,
    )
    assert result is None


def test_build_finnhub_tools_returns_two_callables():
    tools = build_finnhub_tools(api_key="fake", max_headlines=10, earnings_window_days=2)
    assert len(tools) == 2
    assert all(callable(t) for t in tools)
```

- [ ] **Step 3: Run tests (expect ImportError)**

```bash
uv run pytest tests/test_finnhub_tools.py -v
```
Expected: ImportError for `pipeline.finnhub_tools`.

- [ ] **Step 4: Implement `pipeline/finnhub_tools.py`**

```python
"""Finnhub API tool wrappers for the Research Agent.

Split into two layers:
- `*_impl` functions: pure, client-injected — easy to unit test.
- `build_finnhub_tools(...)`: constructs the live finnhub client and returns
  zero-arg-bound callables suitable for registration with agno.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import finnhub
from loguru import logger


def get_company_news_impl(
    *,
    client: Any,
    symbol: str,
    since: datetime,
    max_headlines: int,
) -> list[dict[str, Any]]:
    until = datetime.now(timezone.utc)
    raw = client.company_news(
        symbol, _from=since.date().isoformat(), to=until.date().isoformat()
    )
    items: list[dict[str, Any]] = []
    for entry in raw[:max_headlines]:
        items.append(
            {
                "symbol": symbol,
                "headline": entry["headline"],
                "url": entry["url"],
                "published_at": datetime.fromtimestamp(
                    entry["datetime"], tz=timezone.utc
                ).isoformat(),
                "source": entry.get("source", ""),
                "summary": entry.get("summary", ""),
            }
        )
    return items


def get_earnings_calendar_impl(
    *,
    client: Any,
    symbol: str,
    reference_day: date,
    window_days: int,
) -> date | None:
    start = (reference_day - timedelta(days=window_days)).isoformat()
    end = (reference_day + timedelta(days=window_days)).isoformat()
    raw = client.earnings_calendar(_from=start, to=end, symbol=symbol)
    events = raw.get("earningsCalendar", []) if isinstance(raw, dict) else []
    if not events:
        return None
    # Earliest upcoming or current.
    dates = sorted(date.fromisoformat(e["date"]) for e in events)
    return dates[0]


def build_finnhub_tools(
    *,
    api_key: str,
    max_headlines: int,
    earnings_window_days: int,
) -> list[Callable[..., Any]]:
    client = finnhub.Client(api_key=api_key)

    def get_company_news(symbol: str, since_iso: str) -> list[dict[str, Any]]:
        """Return up to `max_headlines` items for `symbol` since ISO timestamp `since_iso`."""
        since = datetime.fromisoformat(since_iso)
        logger.info("Fetching news", symbol=symbol, since=since_iso)
        return get_company_news_impl(
            client=client, symbol=symbol, since=since, max_headlines=max_headlines
        )

    def get_earnings_calendar(symbol: str) -> str | None:
        """Return the next earnings date (ISO) within the configured window, or None."""
        logger.info("Fetching earnings", symbol=symbol)
        d = get_earnings_calendar_impl(
            client=client,
            symbol=symbol,
            reference_day=date.today(),
            window_days=earnings_window_days,
        )
        return d.isoformat() if d else None

    return [get_company_news, get_earnings_calendar]
```

- [ ] **Step 5: Run tests (expect pass)**

```bash
uv run pytest tests/test_finnhub_tools.py -v
```
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add pipeline/finnhub_tools.py tests/fixtures/finnhub_*.json tests/test_finnhub_tools.py
git commit -m "feat: add Finnhub tool wrappers for news and earnings"
```

---

## Task 6: Research Agent + regime mapping (`research.py`)

The agent itself is wrapped behind a single function so tests don't need agno running.

**Files:**
- Create: `trading-pipeline/pipeline/research.py`
- Create: `trading-pipeline/tests/test_research.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_research.py
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

from pipeline.models import Headline, ResearchReport
from pipeline.research import _apply_regime_rules, run_research_agent


def _hl(sentiment: str, conf: float = 0.8) -> Headline:
    return Headline(
        headline="x",
        url="https://x",
        published_at=datetime.now(timezone.utc),
        sentiment=sentiment,
        confidence=conf,
    )


def test_regime_red_on_earnings_in_window():
    score, regime = _apply_regime_rules(
        headlines=[_hl("pos")], earnings_in_window=True
    )
    assert regime == "RED"


def test_regime_red_on_very_negative_sentiment():
    headlines = [_hl("neg", 0.9), _hl("neg", 0.9), _hl("neg", 0.8)]
    score, regime = _apply_regime_rules(headlines=headlines, earnings_in_window=False)
    assert regime == "RED"
    assert score < -0.4


def test_regime_yellow_on_mildly_negative():
    headlines = [_hl("neg", 0.5), _hl("neu", 0.5), _hl("pos", 0.2)]
    score, regime = _apply_regime_rules(headlines=headlines, earnings_in_window=False)
    assert regime == "YELLOW"
    assert -0.4 <= score < -0.1


def test_regime_yellow_on_no_headlines():
    score, regime = _apply_regime_rules(headlines=[], earnings_in_window=False)
    assert regime == "YELLOW"
    assert score == 0.0


def test_regime_green_default():
    headlines = [_hl("pos", 0.8), _hl("pos", 0.8)]
    score, regime = _apply_regime_rules(headlines=headlines, earnings_in_window=False)
    assert regime == "GREEN"
    assert score > 0


def test_run_research_agent_post_processes_agent_output(monkeypatch):
    """Agent returns a ResearchReport; we re-apply the deterministic regime rules."""
    canned = ResearchReport(
        ticker="NVDA",
        as_of=date(2026, 5, 22),
        regime="GREEN",  # agent lied — we should override
        sentiment_score=0.0,
        headline_count=2,
        earnings_in_window=True,  # earnings present → must become RED
        next_earnings_date=date(2026, 5, 23),
        top_headlines=[_hl("pos"), _hl("pos")],
        rationale="x",
    )
    fake_agent = MagicMock()
    fake_agent.run.return_value = MagicMock(content=canned)

    out = run_research_agent(
        agent=fake_agent,
        ticker="NVDA",
        as_of=date(2026, 5, 22),
        news_lookback_hours=24,
    )
    assert out.regime == "RED"
    assert out.earnings_in_window is True
    assert out.next_earnings_date == date(2026, 5, 23)
```

- [ ] **Step 2: Run tests (expect ImportError)**

```bash
uv run pytest tests/test_research.py -v
```
Expected: ImportError for `pipeline.research`.

- [ ] **Step 3: Implement `pipeline/research.py`**

```python
"""Research Agent: news + earnings → ResearchReport with deterministic regime."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from agno.agent import Agent
from agno.models.google import Gemini
from loguru import logger

from pipeline.models import Headline, ResearchReport


_SENTIMENT_SIGN = {"pos": 1.0, "neu": 0.0, "neg": -1.0}


def _apply_regime_rules(
    *, headlines: list[Headline], earnings_in_window: bool
) -> tuple[float, str]:
    if not headlines:
        score = 0.0
    else:
        score = sum(_SENTIMENT_SIGN[h.sentiment] * h.confidence for h in headlines) / len(
            headlines
        )

    if earnings_in_window:
        regime = "RED"
    elif score < -0.4:
        regime = "RED"
    elif score < -0.1:
        regime = "YELLOW"
    elif not headlines:
        regime = "YELLOW"
    else:
        regime = "GREEN"
    return score, regime


def build_research_agent(
    *,
    model_id: str,
    api_key: str,
    tools: list[Callable[..., Any]],
) -> Agent:
    return Agent(
        name="Market Research Agent",
        model=Gemini(id=model_id, api_key=api_key),
        tools=tools,
        output_schema=ResearchReport,
        instructions=[
            "You are a financial market research agent.",
            "Step 1: Call get_company_news with the ticker and the ISO timestamp "
            "covering the requested lookback window.",
            "Step 2: For each headline, classify sentiment as 'pos', 'neu', or 'neg' "
            "and assign a confidence between 0 and 1.",
            "Step 3: Call get_earnings_calendar to determine the next earnings date.",
            "Step 4: Emit a ResearchReport. Set regime to GREEN tentatively; the "
            "host will recompute the regime deterministically.",
            "Keep rationale to one sentence.",
        ],
    )


def run_research_agent(
    *,
    agent: Agent,
    ticker: str,
    as_of: date,
    news_lookback_hours: int,
) -> ResearchReport:
    since = datetime.now(timezone.utc) - timedelta(hours=news_lookback_hours)
    prompt = (
        f"Research ticker {ticker}. Use {since.isoformat()} as the news lookback "
        f"start. Today is {as_of.isoformat()}."
    )
    logger.info("Invoking research agent", ticker=ticker, as_of=as_of.isoformat())
    raw = agent.run(input=prompt).content
    if not isinstance(raw, ResearchReport):
        raise RuntimeError(f"Research agent returned unexpected payload: {type(raw)}")

    score, regime = _apply_regime_rules(
        headlines=list(raw.top_headlines),
        earnings_in_window=raw.earnings_in_window,
    )
    return raw.model_copy(
        update={
            "sentiment_score": score,
            "regime": regime,
            "headline_count": len(raw.top_headlines),
            "ticker": ticker,
            "as_of": as_of,
        }
    )
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
uv run pytest tests/test_research.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/research.py tests/test_research.py
git commit -m "feat: add Research Agent with deterministic regime mapping"
```

---

## Task 7: Arbiter (`arbiter.py`)

**Files:**
- Create: `trading-pipeline/pipeline/arbiter.py`
- Create: `trading-pipeline/tests/test_arbiter.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_arbiter.py
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

from pipeline.arbiter import _decide, run_arbiter
from pipeline.models import Headline, ResearchReport, TechnicalSignal


def _sig(action="BUY", warmup_ok=True):
    return TechnicalSignal(
        ticker="NVDA",
        as_of=date(2026, 5, 22),
        action=action,
        close=950.0,
        strategy_name="BreakoutStrategy",
        strategy_params={"lookback_period": 20},
        warmup_ok=warmup_ok,
        invalidation_price=915.0 if action == "BUY" else None,
    )


def _report(regime="GREEN", earnings=False):
    return ResearchReport(
        ticker="NVDA",
        as_of=date(2026, 5, 22),
        regime=regime,
        sentiment_score=0.2,
        headline_count=3,
        earnings_in_window=earnings,
        next_earnings_date=date(2026, 5, 23) if earnings else None,
        top_headlines=[],
        rationale="x",
    )


@pytest.mark.parametrize(
    "action,regime,expected",
    [
        ("BUY", "GREEN", "RECOMMEND_BUY"),
        ("BUY", "YELLOW", "BUY_WITH_CAUTION"),
        ("BUY", "RED", "STAND_DOWN"),
        ("SELL", "GREEN", "RECOMMEND_EXIT"),
        ("SELL", "YELLOW", "RECOMMEND_EXIT"),
        ("SELL", "RED", "RECOMMEND_EXIT"),
        ("HOLD", "GREEN", "NO_ACTION"),
        ("HOLD", "RED", "NO_ACTION"),
    ],
)
def test_decision_table(action, regime, expected):
    assert _decide(_sig(action=action), _report(regime=regime)) == expected


def test_warmup_failure_short_circuits_to_no_action():
    sig = _sig(action="BUY", warmup_ok=False)
    rep = _report(regime="GREEN")
    assert _decide(sig, rep) == "NO_ACTION"


def test_run_arbiter_uses_llm_summary_when_available():
    sig = _sig("BUY")
    rep = _report("GREEN")
    fake_agent = MagicMock()
    fake_agent.run.return_value = MagicMock(
        content="Strong breakout with clean macro backdrop. Buying recommended."
    )
    rec = run_arbiter(sig, rep, summary_agent=fake_agent)
    assert rec.decision == "RECOMMEND_BUY"
    assert "Strong breakout" in rec.summary
    assert rec.technical == sig
    assert rec.research == rep


def test_run_arbiter_falls_back_to_template_on_llm_error():
    sig = _sig("BUY")
    rep = _report("YELLOW")
    fake_agent = MagicMock()
    fake_agent.run.side_effect = RuntimeError("LLM down")
    rec = run_arbiter(sig, rep, summary_agent=fake_agent)
    assert rec.decision == "BUY_WITH_CAUTION"
    assert "BUY_WITH_CAUTION NVDA" in rec.summary
    assert "YELLOW" in rec.summary
```

- [ ] **Step 2: Run tests (expect ImportError)**

```bash
uv run pytest tests/test_arbiter.py -v
```
Expected: ImportError for `pipeline.arbiter`.

- [ ] **Step 3: Implement `pipeline/arbiter.py`**

```python
"""Arbiter: combines TechnicalSignal + ResearchReport into a TradeRecommendation."""

from __future__ import annotations

from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from loguru import logger

from pipeline.models import ResearchReport, TechnicalSignal, TradeRecommendation


def _decide(signal: TechnicalSignal, report: ResearchReport) -> str:
    if not signal.warmup_ok:
        return "NO_ACTION"
    if signal.action == "SELL":
        return "RECOMMEND_EXIT"
    if signal.action == "HOLD":
        return "NO_ACTION"
    # BUY
    if report.regime == "GREEN":
        return "RECOMMEND_BUY"
    if report.regime == "YELLOW":
        return "BUY_WITH_CAUTION"
    return "STAND_DOWN"


def build_summary_agent(*, model_id: str, api_key: str) -> Agent:
    return Agent(
        name="Recommendation Summarizer",
        model=Gemini(id=model_id, api_key=api_key),
        instructions=[
            "You write concise 3-4 sentence trade recommendation summaries for the "
            "operator. State the decision, the strategy reason, and the news/risk context. "
            "Do not invent numbers. Plain prose, no bullets.",
        ],
    )


def _template_summary(
    decision: str, signal: TechnicalSignal, report: ResearchReport
) -> str:
    earnings = (
        f" Earnings on {report.next_earnings_date.isoformat()}."
        if report.earnings_in_window and report.next_earnings_date
        else ""
    )
    return (
        f"{decision} {signal.ticker} @ {signal.close:.2f}. "
        f"Strategy: {signal.strategy_name} {signal.strategy_params}. "
        f"Regime: {report.regime} (sentiment={report.sentiment_score:+.2f}, "
        f"{report.headline_count} headlines).{earnings}"
    )


def _llm_summary(agent: Agent, decision: str, signal: TechnicalSignal, report: ResearchReport) -> str:
    payload = {
        "decision": decision,
        "ticker": signal.ticker,
        "close": signal.close,
        "strategy": signal.strategy_name,
        "strategy_params": signal.strategy_params,
        "invalidation_price": signal.invalidation_price,
        "regime": report.regime,
        "sentiment_score": report.sentiment_score,
        "headline_count": report.headline_count,
        "earnings_in_window": report.earnings_in_window,
        "next_earnings_date": (
            report.next_earnings_date.isoformat() if report.next_earnings_date else None
        ),
        "rationale": report.rationale,
    }
    result = agent.run(input=f"Summarize this trade recommendation: {payload}")
    content = result.content if hasattr(result, "content") else result
    return str(content)


def run_arbiter(
    signal: TechnicalSignal,
    report: ResearchReport,
    *,
    summary_agent: Agent,
) -> TradeRecommendation:
    decision = _decide(signal, report)
    try:
        summary = _llm_summary(summary_agent, decision, signal, report)
    except Exception as exc:  # noqa: BLE001 — explicit fallback path
        logger.warning("Summary LLM failed; using template", error=str(exc))
        summary = _template_summary(decision, signal, report)
    return TradeRecommendation(
        ticker=signal.ticker,
        as_of=signal.as_of,
        decision=decision,
        technical=signal,
        research=report,
        summary=summary,
    )
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
uv run pytest tests/test_arbiter.py -v
```
Expected: 11 passed (8 parametrized + 3 others).

- [ ] **Step 5: Commit**

```bash
git add pipeline/arbiter.py tests/test_arbiter.py
git commit -m "feat: add Arbiter with decision table and LLM summary fallback"
```

---

## Task 8: Persistence (`persistence.py`)

**Files:**
- Create: `trading-pipeline/pipeline/persistence.py`
- Create: `trading-pipeline/tests/test_persistence.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_persistence.py
import json
from datetime import date, datetime, timezone

import pytest

from pipeline.models import Headline, ResearchReport, TechnicalSignal, TradeRecommendation
from pipeline.persistence import init_db, upsert_run


def _rec(decision="RECOMMEND_BUY", summary="Buy."):
    sig = TechnicalSignal(
        ticker="NVDA",
        as_of=date(2026, 5, 22),
        action="BUY",
        close=950.0,
        strategy_name="BreakoutStrategy",
        strategy_params={"lookback_period": 20},
        warmup_ok=True,
        invalidation_price=915.0,
    )
    rep = ResearchReport(
        ticker="NVDA",
        as_of=date(2026, 5, 22),
        regime="GREEN",
        sentiment_score=0.3,
        headline_count=2,
        earnings_in_window=False,
        next_earnings_date=None,
        top_headlines=[
            Headline(
                headline="x",
                url="https://x",
                published_at=datetime.now(timezone.utc),
                sentiment="pos",
                confidence=0.8,
            )
        ],
        rationale="ok",
    )
    return TradeRecommendation(
        ticker="NVDA",
        as_of=date(2026, 5, 22),
        decision=decision,
        technical=sig,
        research=rep,
        summary=summary,
    )


def test_upsert_inserts_a_row(tmp_path):
    db = tmp_path / "runs.db"
    init_db(db)
    upsert_run(db, _rec())
    import sqlite3

    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT ticker, as_of, decision, summary FROM runs").fetchall()
    assert rows == [("NVDA", "2026-05-22", "RECOMMEND_BUY", "Buy.")]


def test_upsert_replaces_existing_natural_key(tmp_path):
    db = tmp_path / "runs.db"
    init_db(db)
    upsert_run(db, _rec(decision="RECOMMEND_BUY", summary="first"))
    upsert_run(db, _rec(decision="BUY_WITH_CAUTION", summary="second"))
    import sqlite3

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT decision, summary FROM runs WHERE ticker='NVDA' AND as_of='2026-05-22'"
        ).fetchall()
    assert rows == [("BUY_WITH_CAUTION", "second")]


def test_payload_json_is_complete(tmp_path):
    db = tmp_path / "runs.db"
    init_db(db)
    rec = _rec()
    upsert_run(db, rec)
    import sqlite3

    with sqlite3.connect(db) as conn:
        (payload_json,) = conn.execute("SELECT payload_json FROM runs").fetchone()
    restored = TradeRecommendation.model_validate(json.loads(payload_json))
    assert restored == rec
```

- [ ] **Step 2: Run tests (expect ImportError)**

```bash
uv run pytest tests/test_persistence.py -v
```
Expected: ImportError for `pipeline.persistence`.

- [ ] **Step 3: Implement `pipeline/persistence.py`**

```python
"""SQLite append/upsert for daily pipeline runs."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from pipeline.models import TradeRecommendation


_SCHEMA = """
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
  payload_json TEXT NOT NULL,
  PRIMARY KEY (ticker, as_of)
);
"""


def init_db(path: Path | str) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)


def upsert_run(path: Path | str, rec: TradeRecommendation) -> None:
    payload = rec.model_dump_json()
    row = (
        rec.ticker,
        rec.as_of.isoformat(),
        datetime.now(timezone.utc).isoformat(),
        rec.decision,
        rec.technical.action,
        rec.technical.close,
        rec.research.regime,
        rec.research.sentiment_score,
        int(rec.research.earnings_in_window),
        rec.summary,
        payload,
    )
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs
              (ticker, as_of, ran_at, decision, technical_action, technical_close,
               regime, sentiment_score, earnings_in_window, summary, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
    logger.info(
        "Persisted run",
        ticker=rec.ticker,
        as_of=rec.as_of.isoformat(),
        decision=rec.decision,
    )
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
uv run pytest tests/test_persistence.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/persistence.py tests/test_persistence.py
git commit -m "feat: add SQLite upsert persistence with payload_json"
```

---

## Task 9: Notification (`notify.py`)

**Files:**
- Create: `trading-pipeline/pipeline/notify.py`
- Create: `trading-pipeline/tests/test_notify.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_notify.py
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from pipeline.models import Headline, ResearchReport, TechnicalSignal, TradeRecommendation
from pipeline.notify import publish, _build_payload


def _rec(decision, regime="GREEN", earnings=False):
    sig = TechnicalSignal(
        ticker="NVDA",
        as_of=date(2026, 5, 22),
        action="BUY",
        close=950.0,
        strategy_name="BreakoutStrategy",
        strategy_params={"lookback_period": 20},
        warmup_ok=True,
        invalidation_price=915.0,
    )
    rep = ResearchReport(
        ticker="NVDA",
        as_of=date(2026, 5, 22),
        regime=regime,
        sentiment_score=0.3,
        headline_count=2,
        earnings_in_window=earnings,
        next_earnings_date=date(2026, 5, 23) if earnings else None,
        top_headlines=[],
        rationale="x",
    )
    return TradeRecommendation(
        ticker="NVDA",
        as_of=date(2026, 5, 22),
        decision=decision,
        technical=sig,
        research=rep,
        summary="Test summary.",
    )


def test_build_payload_buy():
    body, headers = _build_payload(_rec("RECOMMEND_BUY"), token=None)
    assert headers["Title"] == "RECOMMEND_BUY NVDA"
    assert headers["Tags"] == "green_circle"
    assert "Authorization" not in headers
    assert "Test summary." in body
    assert "Close: 950.00" in body
    assert "Regime: GREEN" in body
    assert "Invalidation: 915.0" in body


def test_build_payload_caution_includes_earnings():
    body, headers = _build_payload(_rec("BUY_WITH_CAUTION", "YELLOW", earnings=True), token=None)
    assert headers["Tags"] == "warning"
    assert "Earnings: 2026-05-23" in body


def test_build_payload_includes_auth_header_when_token_set():
    _, headers = _build_payload(_rec("RECOMMEND_BUY"), token="abc123")
    assert headers["Authorization"] == "Bearer abc123"


def test_publish_posts_to_ntfy(monkeypatch):
    fake_client = MagicMock()
    fake_client.post.return_value = MagicMock(status_code=200)
    monkeypatch.setattr("pipeline.notify._client", lambda: fake_client)

    publish(
        rec=_rec("RECOMMEND_BUY"),
        ntfy_url="https://ntfy.home.lan/trading",
        token=None,
    )
    fake_client.post.assert_called_once()
    args, kwargs = fake_client.post.call_args
    assert args[0] == "https://ntfy.home.lan/trading"
    assert kwargs["headers"]["Title"] == "RECOMMEND_BUY NVDA"
    assert isinstance(kwargs["data"], (bytes, str))
```

- [ ] **Step 2: Run tests (expect ImportError)**

```bash
uv run pytest tests/test_notify.py -v
```
Expected: ImportError for `pipeline.notify`.

- [ ] **Step 3: Implement `pipeline/notify.py`**

```python
"""ntfy.sh publisher for trade recommendations."""

from __future__ import annotations

from loguru import logger
from pyreqwest import Client

from pipeline.models import TradeRecommendation


_TAG_BY_DECISION = {
    "RECOMMEND_BUY":    "green_circle",
    "BUY_WITH_CAUTION": "warning",
    "RECOMMEND_EXIT":   "red_circle",
    "STAND_DOWN":       "no_entry",
    "NO_ACTION":        "zzz",
}


def _client() -> Client:
    return Client()


def _build_payload(rec: TradeRecommendation, *, token: str | None) -> tuple[str, dict[str, str]]:
    sig = rec.technical
    rep = rec.research
    lines = [
        rec.summary,
        "",
        f"Close: {sig.close:.2f}",
        f"Strategy: {sig.strategy_name} {sig.strategy_params}",
        f"Regime: {rep.regime} (sentiment={rep.sentiment_score:+.2f}, "
        f"{rep.headline_count} headlines)",
    ]
    if rep.earnings_in_window and rep.next_earnings_date:
        lines.append(f"Earnings: {rep.next_earnings_date.isoformat()}")
    if sig.invalidation_price is not None:
        lines.append(f"Invalidation: {sig.invalidation_price}")
    body = "\n".join(lines)

    headers = {
        "Title": f"{rec.decision} {rec.ticker}",
        "Tags": _TAG_BY_DECISION[rec.decision],
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return body, headers


def publish(*, rec: TradeRecommendation, ntfy_url: str, token: str | None) -> None:
    body, headers = _build_payload(rec, token=token)
    try:
        response = _client().post(ntfy_url, headers=headers, data=body.encode("utf-8"))
        if getattr(response, "status_code", 200) >= 400:
            logger.error("ntfy returned error", status=response.status_code)
    except Exception as exc:  # noqa: BLE001 — ntfy failure must not raise
        logger.error("ntfy publish failed", error=str(exc))
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
uv run pytest tests/test_notify.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/notify.py tests/test_notify.py
git commit -m "feat: add ntfy publisher with decision-based tags"
```

---

## Task 10: Workflow wiring + CLI (`workflow.py`, `daily.py`)

**Files:**
- Create: `trading-pipeline/pipeline/workflow.py`
- Create: `trading-pipeline/pipeline/daily.py`
- Create: `trading-pipeline/tests/test_workflow_smoke.py`

- [ ] **Step 1: Write failing smoke test**

```python
# tests/test_workflow_smoke.py
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from pipeline.config import PipelineConfig
from pipeline.models import Headline, ResearchReport, TechnicalSignal


@pytest.fixture
def cfg(tmp_path):
    return PipelineConfig(
        ticker="NVDA",
        strategy_name="BreakoutStrategy",
        strategy_params={"lookback_period": 20},
        lookback_days=60,
        news_lookback_hours=24,
        earnings_window_days=2,
        max_headlines=10,
        sentiment_model="gemini-3.1-flash-lite",
        summary_model="gemini-3.5-flash",
        ntfy_url="https://ntfy.example/test",
        db_path=tmp_path / "runs.db",
        finnhub_api_key="fk",
        google_api_key="gk",
        ntfy_token=None,
    )


def _signal(action="BUY"):
    return TechnicalSignal(
        ticker="NVDA",
        as_of=date(2026, 5, 22),
        action=action,
        close=950.0,
        strategy_name="BreakoutStrategy",
        strategy_params={"lookback_period": 20},
        warmup_ok=True,
        invalidation_price=915.0,
    )


def _report(regime="GREEN"):
    return ResearchReport(
        ticker="NVDA",
        as_of=date(2026, 5, 22),
        regime=regime,
        sentiment_score=0.3,
        headline_count=2,
        earnings_in_window=False,
        next_earnings_date=None,
        top_headlines=[],
        rationale="ok",
    )


def test_pipeline_happy_path(cfg, monkeypatch):
    """BUY + GREEN → RECOMMEND_BUY, persisted, notified."""
    from pipeline import workflow

    monkeypatch.setattr(workflow, "run_technical_engine", lambda **kw: _signal("BUY"))
    monkeypatch.setattr(workflow, "build_finnhub_tools", lambda **kw: [])
    monkeypatch.setattr(workflow, "build_research_agent", lambda **kw: MagicMock())
    monkeypatch.setattr(workflow, "run_research_agent", lambda **kw: _report("GREEN"))
    monkeypatch.setattr(workflow, "build_summary_agent", lambda **kw: MagicMock())
    monkeypatch.setattr(
        workflow, "run_arbiter",
        lambda sig, rep, summary_agent: __import__("pipeline.models", fromlist=["TradeRecommendation"]).TradeRecommendation(
            ticker=sig.ticker, as_of=sig.as_of, decision="RECOMMEND_BUY",
            technical=sig, research=rep, summary="ok",
        ),
    )
    publish_mock = MagicMock()
    monkeypatch.setattr(workflow, "publish", publish_mock)

    rec = workflow.run_pipeline(cfg)
    assert rec.decision == "RECOMMEND_BUY"
    publish_mock.assert_called_once()
    # And was persisted:
    import sqlite3
    with sqlite3.connect(cfg.db_path) as conn:
        rows = conn.execute("SELECT decision FROM runs").fetchall()
    assert rows == [("RECOMMEND_BUY",)]


def test_pipeline_stand_down_on_red_regime(cfg, monkeypatch):
    from pipeline import workflow

    monkeypatch.setattr(workflow, "run_technical_engine", lambda **kw: _signal("BUY"))
    monkeypatch.setattr(workflow, "build_finnhub_tools", lambda **kw: [])
    monkeypatch.setattr(workflow, "build_research_agent", lambda **kw: MagicMock())
    monkeypatch.setattr(workflow, "run_research_agent", lambda **kw: _report("RED"))
    monkeypatch.setattr(workflow, "build_summary_agent", lambda **kw: MagicMock())
    monkeypatch.setattr(
        workflow, "run_arbiter",
        lambda sig, rep, summary_agent: __import__("pipeline.models", fromlist=["TradeRecommendation"]).TradeRecommendation(
            ticker=sig.ticker, as_of=sig.as_of, decision="STAND_DOWN",
            technical=sig, research=rep, summary="stand down",
        ),
    )
    monkeypatch.setattr(workflow, "publish", MagicMock())

    rec = workflow.run_pipeline(cfg)
    assert rec.decision == "STAND_DOWN"
```

- [ ] **Step 2: Run test (expect ImportError)**

```bash
uv run pytest tests/test_workflow_smoke.py -v
```
Expected: ImportError for `pipeline.workflow`.

- [ ] **Step 3: Implement `pipeline/workflow.py`**

```python
"""Top-level pipeline wiring."""

from __future__ import annotations

from loguru import logger

from pipeline.arbiter import build_summary_agent, run_arbiter
from pipeline.config import PipelineConfig
from pipeline.finnhub_tools import build_finnhub_tools
from pipeline.models import TradeRecommendation
from pipeline.notify import publish
from pipeline.persistence import init_db, upsert_run
from pipeline.research import build_research_agent, run_research_agent
from pipeline.technical import run_technical_engine


def run_pipeline(cfg: PipelineConfig) -> TradeRecommendation:
    logger.info("Starting pipeline", ticker=cfg.ticker)

    signal = run_technical_engine(
        ticker=cfg.ticker,
        strategy_name=cfg.strategy_name,
        strategy_params=cfg.strategy_params,
        lookback_days=cfg.lookback_days,
    )
    logger.info("Technical signal", action=signal.action, close=signal.close)

    tools = build_finnhub_tools(
        api_key=cfg.finnhub_api_key,
        max_headlines=cfg.max_headlines,
        earnings_window_days=cfg.earnings_window_days,
    )
    research_agent = build_research_agent(
        model_id=cfg.sentiment_model,
        api_key=cfg.google_api_key,
        tools=tools,
    )
    report = run_research_agent(
        agent=research_agent,
        ticker=cfg.ticker,
        as_of=signal.as_of,
        news_lookback_hours=cfg.news_lookback_hours,
    )
    logger.info("Research report", regime=report.regime, score=report.sentiment_score)

    summary_agent = build_summary_agent(
        model_id=cfg.summary_model, api_key=cfg.google_api_key
    )
    rec = run_arbiter(signal, report, summary_agent=summary_agent)
    logger.info("Decision", decision=rec.decision)

    init_db(cfg.db_path)
    upsert_run(cfg.db_path, rec)
    publish(rec=rec, ntfy_url=cfg.ntfy_url, token=cfg.ntfy_token)
    return rec
```

- [ ] **Step 4: Implement `pipeline/daily.py`**

```python
"""CLI entrypoint: `uv run python -m pipeline.daily [TICKER]`."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from loguru import logger

from pipeline.config import load_config
from pipeline.workflow import run_pipeline


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}:{line}</cyan> - "
            "<level>{message}</level> {extra}"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = list(sys.argv[1:] if argv is None else argv)
    cfg = load_config(Path("config.toml"))
    if args:
        cfg = replace(cfg, ticker=args[0].upper())
    try:
        rec = run_pipeline(cfg)
    except Exception as exc:  # noqa: BLE001 — top-level failure handler
        logger.exception("Pipeline failed", error=str(exc))
        return 1
    logger.info("Pipeline complete", decision=rec.decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run smoke tests (expect pass)**

```bash
uv run pytest tests/test_workflow_smoke.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Run full test suite + coverage check**

```bash
uv run pytest --cov=pipeline -v
```
Expected: all tests pass, coverage ≥75% across `pipeline/*`.

- [ ] **Step 7: Run ruff and ty**

```bash
uv run ruff check .
uv run ty check
```
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add pipeline/workflow.py pipeline/daily.py tests/test_workflow_smoke.py
git commit -m "feat: wire pipeline workflow and CLI entrypoint"
```

---

## Task 11: README and cron snippet

**Files:**
- Create: `trading-pipeline/README.md`

- [ ] **Step 1: Write README**

```markdown
# trading-pipeline

Daily advisory trading pipeline. Runs after the market close, produces a single
buy/sell/hold recommendation, persists it, and notifies via ntfy.

## Setup

1. Install uv.
2. `uv sync`
3. Copy `.env.example` to `.env` and fill in `FINNHUB_API_KEY` and `GOOGLE_API_KEY`.
4. Edit `config.toml` (ticker, strategy, ntfy URL).
5. Confirm the `backtest` path in `pyproject.toml` `[tool.uv.sources]` points at
   your local clone of the trading repo.

## Run

```bash
uv run python -m pipeline.daily          # uses config.ticker.symbol
uv run python -m pipeline.daily AAPL     # override
```

## Cron

```
30 21 * * 1-5  cd /opt/trading-pipeline && /usr/local/bin/uv run python -m pipeline.daily >> logs/daily.log 2>&1
```

## Tests

```bash
uv run pytest --cov=pipeline -v
```

## Inspect runs

```bash
sqlite3 runs.db "SELECT as_of, decision, regime, summary FROM runs ORDER BY as_of DESC LIMIT 10;"
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup, run, and cron instructions"
```

---

## Self-review notes (recorded for the implementing engineer)

- **Spec coverage:** every section of `2026-05-24-daily-trading-pipeline-design.md` maps to a task (§4.1→T4, §4.2→T5+T6, §4.3→T7, §5→T2, §6→T8, §7→T9, §8→T3, §9→T10+T11, §10→T10 top-level handler, §11→all test tasks, §12→T1+T10, §13→T1).
- **No placeholders:** every code step contains full code; every test has assertions.
- **Type consistency:** model field names (`top_headlines`, `next_earnings_date`, `warmup_ok`, etc.) match across `models.py`, `research.py`, `arbiter.py`, `persistence.py`, and `notify.py`.
- **TDD discipline:** every task writes the failing test first, runs it, then implements.
- **DRY/YAGNI:** no retries, no run_id UUID, no extra columns, no auth ceremony — matches the simplified spec.

Implementation will happen in the new `trading-pipeline/` repo, not in this one. This plan is the handoff artifact.
