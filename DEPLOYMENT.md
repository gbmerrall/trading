# Deployment Guide

The `backtest` package is consumed outside this repo in two ways:

1. **The daily cron** (`../daily_cron/`) — its `main.py` is a PEP 723 uv script that
   installs `quant-backtest-framework` from a wheel committed at
   `../daily_cron/wheels/`. This is the deployment that matters day to day.
2. **Ad-hoc install on another machine** (e.g. a Raspberry Pi) via `scp` + `pip`.

Either way, deployment starts by building the wheel.

---

## Step 1: Build the Wheel

From this directory:

```bash
uv build --wheel
```

This places the wheel in `dist/`:

```
dist/quant_backtest_framework-0.1.0-py3-none-any.whl
```

The version comes from `pyproject.toml`. If you have changed package behaviour, bump the
version there before building — uv caches wheels, and replacing a same-named wheel with
different contents risks the old build being served from cache (see Step 2).

---

## Step 2: Refresh the Daily Cron's Wheel

The cron declares its dependency in `daily_cron/main.py`:

```toml
# [tool.uv.sources]
# "quant-backtest-framework" = { path = "wheels/quant_backtest_framework-0.1.0-py3-none-any.whl" }
```

To deploy a new build:

```bash
cp dist/quant_backtest_framework-<version>-py3-none-any.whl ../daily_cron/wheels/
```

If you bumped the version, also update the filename in `main.py`'s `[tool.uv.sources]`
block and delete the old wheel. If you rebuilt without bumping, clear uv's cache so the
new contents are picked up:

```bash
uv cache clean quant-backtest-framework
```

The cron only imports `backtest.strategy` (signal generation), so backtest-side changes
to the runner, optimizer, or metrics do not change cron behaviour — but keep the wheel
current anyway so the homelab copy matches what was analysed.

---

## Step 3 (optional): Install on a Raspberry Pi

Copy the wheel across and install it:

```bash
scp dist/quant_backtest_framework-<version>-py3-none-any.whl pi@<raspberry-pi-ip>:~/

ssh pi@<raspberry-pi-ip>
uv pip install quant_backtest_framework-<version>-py3-none-any.whl
# or, with plain pip:
pip install quant_backtest_framework-<version>-py3-none-any.whl
```

To install `uv` on the Pi if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
python -c "import backtest; print('OK')"
```

---

## Notes

- The wheel is platform-independent (`py3-none-any`) — pure Python.
- Runtime dependencies (`pandas`, `numpy`, `yfinance`, `ta`, `plotly`, `kaleido`,
  `google-genai`, `pyyaml`, `loguru`) are pulled from PyPI automatically on install.
- If the target machine is offline, bundle dependencies with
  `pip download -r requirements.txt -d ./deps` on your dev machine and transfer them
  alongside the wheel.
- Run the test suite (`uv run pytest`) before building — the wheel snapshots whatever
  is on disk.
