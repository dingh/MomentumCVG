# C5.11A — Downstream Adjusted-Liquid Path Wiring Report

## Current repo SHA

`0d2357381e373f217e21ef2213749a5880f195a9` (C5.11A path wiring commit)

## Accepted adjusted-liquid root

`C:/MomentumCVG_env/input/adjusted_liquid`

Layout: `{root}/{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.parquet` (2,299 files, 2017–2026, accepted after C5.10D audit patch).

## Files inspected

| Area | Files |
|------|-------|
| Central paths | `src/data/paths.py` (new) |
| Provider / config | `src/data/orats_provider.py`, `src/backtest/config.py`, `configs/baseline_sp500.json` |
| Stage A scripts | `scripts/extract_spot_prices.py`, `scripts/precompute_option_surface.py`, `scripts/precompute_straddle_history.py`, `scripts/precompute_ironfly_history.py`, `scripts/build_straddle_master_universe.py`, `scripts/refresh_weekly_inputs.py` |
| Visualization | `src/visualization/chain_loader.py` |
| Split backfill (intentional legacy default) | `scripts/apply_split_adjustment.py` |
| Raw-data / C4 (unchanged) | `scripts/build_liquidity_panel.py`, `src/data/corporate_actions.py` |
| Docs / archive | `docs/tmp/*`, `docs/archive/*`, `AGENTS.md`, `docs/repo_map.md`, `docs/v1_weekly_runbook.md` |
| Tests | `tests/unit/test_adjusted_liquid_paths.py` (new), existing C5/C2/C4 tests |

## Old-root reference classification

| Group | Path / symbol | Classification | Action |
|-------|---------------|----------------|--------|
| A | `src/data/orats_provider.py` default `data_root` | Active production reader | **Updated** → `DEFAULT_ADJUSTED_LIQUID_ROOT` |
| A | `src/backtest/config.py` `DEFAULT_CONFIG` | Active default | **Updated** |
| A | `configs/baseline_sp500.json` | Active config default | **Updated** |
| A | `scripts/extract_spot_prices.py` `--data-root` | CLI default | **Updated** |
| A | `scripts/precompute_option_surface.py` `--data-root` | CLI default | **Updated** |
| A | `scripts/precompute_straddle_history.py` `--data-root` | CLI default | **Updated** |
| A | `scripts/precompute_ironfly_history.py` `--data-root` | CLI default | **Updated** |
| A | `scripts/build_straddle_master_universe.py` `--data-root` | CLI default | **Updated** |
| A | `src/visualization/chain_loader.py` default | Active default | **Updated** (None → central constant) |
| B | `scripts/refresh_weekly_inputs.py` `--orats-adj-root` | CLI default | **Updated** |
| E | `scripts/apply_split_adjustment.py` `--adj-root` (no flag) | Raw→full-mirror backfill | **Left on** `LEGACY_ORATS_ADJUSTED_ROOT` via `paths.py` |
| E | `scripts/build_liquidity_panel.py` | Raw ZIP workflow | **Unchanged** (`ORATS_Data`) |
| E | `src/data/corporate_actions.py` `get_all_unique_tickers` | Raw ZIP scan | **Unchanged** |
| C | `tests/unit/test_trading_day.py`, `test_straddle_analyzer.py` | Explicit fixture paths | **Unchanged** (parameterized / isolated) |
| C | `tests/unit/test_refresh_weekly_inputs_cli.py` | tmp_path fake root name | **Unchanged** (CLI passes explicit `--orats-adj-root`) |
| D | `docs/tmp/*`, `docs/archive/*`, `AGENTS.md`, `docs/repo_map.md`, `docs/v1_weekly_runbook.md` | Docs / prior reports | **Unchanged** (no history rewrite) |
| D | `src/features/ironfly_analyzer.py` docstring examples | Documentation only | **Unchanged** |
| D | `src/data/split_adjuster.py` docstring / example kwargs | Library docs | **Unchanged** |
| D | `src/data/trading_day.py` docstring example path | Helper docs | **Unchanged** |

## Files changed

- `src/data/paths.py` — **new** central constants
- `src/data/orats_provider.py` — default root + docstrings/error text
- `src/backtest/config.py` — `DEFAULT_CONFIG` data_root
- `src/backtest/engine.py` — example docstring only
- `src/visualization/chain_loader.py` — default root
- `configs/baseline_sp500.json` — data_root
- `scripts/extract_spot_prices.py` — `--data-root` default
- `scripts/precompute_option_surface.py` — `--data-root` default
- `scripts/precompute_straddle_history.py` — `--data-root` default
- `scripts/precompute_ironfly_history.py` — `--data-root` default
- `scripts/build_straddle_master_universe.py` — `--data-root` default
- `scripts/refresh_weekly_inputs.py` — `--orats-adj-root` default
- `scripts/apply_split_adjustment.py` — import `RAW_ORATS_ROOT` / `LEGACY_ORATS_ADJUSTED_ROOT` (full-mirror default preserved)
- `tests/unit/test_adjusted_liquid_paths.py` — **new** regression tests

## Active defaults updated

| Component | New default |
|-----------|-------------|
| `ORATSDataProvider()` | `C:/MomentumCVG_env/input/adjusted_liquid` |
| `BacktestConfig` / `DEFAULT_CONFIG` | `C:/MomentumCVG_env/input/adjusted_liquid` |
| `configs/baseline_sp500.json` | same |
| Stage A precompute / spot / universe scripts | same via `DEFAULT_ADJUSTED_LIQUID_ROOT` |
| `refresh_weekly_inputs.py --orats-adj-root` | same |
| `ChainLoader()` | same (via provider default) |

Central constant module:

```python
DEFAULT_ADJUSTED_LIQUID_ROOT = Path("C:/MomentumCVG_env/input/adjusted_liquid")
LEGACY_ORATS_ADJUSTED_ROOT = Path("C:/ORATS/data/ORATS_Adjusted")
RAW_ORATS_ROOT = Path("C:/ORATS/data/ORATS_Data")
```

## References intentionally left unchanged

- **Full-mirror split backfill:** `apply_split_adjustment.py` still defaults `--adj-root` to `LEGACY_ORATS_ADJUSTED_ROOT` when omitted (filtered mode still requires explicit `--adj-root`).
- **C4 liquidity panel:** `build_liquidity_panel.py` remains on raw `C:/ORATS/data/ORATS_Data` ZIPs.
- **Archived / prior docs:** all `docs/tmp/*` prior reports, `docs/archive/*`, `AGENTS.md`, `repo_map.md`, `v1_weekly_runbook.md`.
- **Docstring-only examples** in `ironfly_analyzer.py`, `split_adjuster.py`, `trading_day.py`.
- **Test fixtures** that pass explicit legacy paths for isolation.

## Downstream smoke command / method

Read-only inline smoke (no strategy backtest):

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -c "
from datetime import date
from src.data.orats_provider import ORATSDataProvider
from src.data.paths import DEFAULT_ADJUSTED_LIQUID_ROOT

provider = ORATSDataProvider(min_volume=0, min_open_interest=0, min_bid=0.0, max_spread_pct=1.0)
trade_date = date(2020, 1, 2)
provider._load_day_data(trade_date)
provider.get_spot_price('AAPL', trade_date)
expiries = provider.get_available_expiries('AAPL', trade_date)
provider.get_option_chain('AAPL', trade_date, expiries[0])
"
```

Ticker: `AAPL`. Date: `2020-01-02` (known C5.10B production file).

## Downstream smoke result

**PASS**

| Check | Result |
|-------|--------|
| Provider `data_root` | `C:\MomentumCVG_env\input\adjusted_liquid` |
| File path | `...\2020\ORATS_SMV_Strikes_20200102.parquet` (exists) |
| `_load_day_data` | 369,068 rows; columns include `adj_stkPx`, `adj_strike`, `adj_cBidPx` |
| `get_spot_price("AAPL")` | `74.8925` (from adjusted spot column) |
| `get_available_expiries` | 17 expiries; first `2020-01-03` |
| `get_option_chain` | 104 quotes; sample strike `50.0`, bid/ask populated |
| Legacy mirror read | **Not required** |

## Tests run

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_audit_adjusted_liquid.py tests/unit/test_adjusted_liquid_paths.py -q
# 22 passed

C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_fetch_splits_cli.py tests/unit/test_apply_split_adjustment_cli.py tests/unit/test_split_adjuster.py tests/unit/test_split_adjuster_filtered_zip.py tests/unit/test_ticker_universe.py tests/unit/test_audit_adjusted_liquid.py tests/unit/test_adjusted_liquid_paths.py -q
# 76 passed
```

New tests in `tests/unit/test_adjusted_liquid_paths.py`:

- `ORATSDataProvider()` default → adjusted-liquid root
- `DEFAULT_CONFIG` data_root → adjusted-liquid root
- Central `paths.py` constants exist and are imported by provider

## Final verdict

**ACCEPT C5.11A — READY FOR STAGE A INPUT SMOKE**

Active downstream defaults now point at the accepted C5 production adjusted-liquid root. C4 raw-data workflows and full-mirror split backfill defaults are preserved. Provider smoke on `2020-01-02` succeeds without touching `C:/ORATS/data/ORATS_Adjusted`.
