# C5.7 Scoped Split Fetch — Real-Data Run Report

**Date:** 2026-07-03  
**Status:** PASS  
**Scope:** Split-fetch CLI wiring + validation/reporting only (no adjusted backfill)

## Command

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/fetch_splits.py `
  --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv `
  --out C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet `
  --token <ORATS token via --token; not stored in repo>
```

## Inputs

| Item | Value |
|------|-------|
| Ticker universe | `C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv` |
| Input universe ticker count | 2783 |
| Mode | scoped (C5) |

## Output

| Item | Value |
|------|-------|
| Output path | `C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet` |
| Checkpoint sidecar | `C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.checkpoint.parquet` |
| Row count | 1347 |
| Unique tickers with splits | 819 |
| Columns | `ticker`, `split_date`, `divisor` |
| Required columns present | yes |
| Ticker normalization | stripped uppercase |

## Universe membership

| Check | Result |
|-------|--------|
| Input universe tickers | 2783 |
| Output split tickers | 819 |
| Intersection with input universe | 819 |
| Outside-universe tickers | **0** |

## Split date range

| | Date |
|---|------|
| Min | 2007-01-03 |
| Max | 2026-07-02 |

Note: splits before `min_split_date` (2014-01-01) in `SplitAdjuster` are dropped at adjustment time; full history is stored in the scoped split file.

## Divisor validation

| Check | Result |
|-------|--------|
| Null divisors | 0 |
| Nonpositive divisors | 0 |
| Min divisor | 0.003 |
| Max divisor | 310.0 |

## Duplicate validation

| Check | Result |
|-------|--------|
| Dedup key | `(ticker, split_date)` |
| Duplicate keys after dedup | 0 |
| Conflicting divisors | 0 (fail-fast would trigger if present) |

## Safety confirmations

| Check | Result |
|-------|--------|
| Broad cache untouched | `C:/MomentumCVG_env/cache/splits_hist.parquet` — sha256 unchanged (`bd355d31fbdc937199c1ba69ea2aee99b8ea4de9711e2a640d961e92626a4cd3`) |
| Adjusted backfill run | **No** — only split-history parquet written |
| Year directories under `adjusted_liquid/` | **None** |

## Tests (pre-fetch)

```powershell
# C5.7 unit tests
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_fetch_splits_cli.py tests/unit/test_ticker_universe.py -q
# → 32 passed

# C5 regression
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_apply_split_adjustment_cli.py tests/unit/test_split_adjuster.py tests/unit/test_split_adjuster_filtered_zip.py tests/unit/test_ticker_universe.py -q
# → 41 passed
```

## Verdict

**PASS** — Scoped split-history file is ready for downstream filtered adjusted backfill (C5.6+).
