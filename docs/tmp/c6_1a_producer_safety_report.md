# C6.1A — Option Surface Producer Safety Report

**Sprint:** 004 · **Task:** C6.1A  
**Date:** 2026-07-08  
**Mode:** Build (producer safety + path centralization + weekly entry schedule)

---

## Summary of changes

C6.1A makes `scripts/precompute_option_surface.py` safe for bounded sample runs without risking canonical cache artifacts, centralizes producer path defaults in `src/data/paths.py`, and moves weekly **entry-date** schedule generation into `src/data/trading_day.py`.

Key outcomes:

- `--dry-run` prints a full run plan without joblib or parquet writes.
- `--output-root` isolates smoke output from `DEFAULT_CACHE_ROOT`.
- Overwrite guard refuses to replace existing meta/quotes unless `--overwrite`.
- Ticker and date scope can be bounded via `--tickers`, `--tickers-file`, `--start-date`, and `--end-date`.
- Year-only bounds now resolve to **Jan 1 … Dec 31** (Feb 20 shortcut removed).
- Weekly entry schedule uses shared `weekly_trade_dates_in_range` (Friday anchor + file-existence walk-back).
- **Expiry selection is unchanged** — `OptionSurfaceBuilder._find_best_expiry` and `process_single_entry` were not modified.

---

## Files changed

| File | Change |
|------|--------|
| `src/data/paths.py` | Added `DEFAULT_CACHE_ROOT`, `DEFAULT_SPOT_PRICES_PATH`, `DEFAULT_LIQUID_TICKERS_PATH`, `DEFAULT_PRECOMPUTE_OPTION_SURFACE_LOG` |
| `src/data/trading_day.py` | Added `resolve_weekly_entry_date`, `weekly_trade_dates_in_range`, `target_weekly_expiry_from_schedule` |
| `scripts/precompute_option_surface.py` | Path defaults from `paths.py`; safety CLI; shared entry schedule; testable `main()` |
| `tests/unit/test_trading_day.py` | Weekly entry schedule + diagnostic helper tests |
| `tests/unit/test_adjusted_liquid_paths.py` | Assertions for new path constants |
| `tests/unit/test_precompute_option_surface_cli.py` | **New** — producer CLI safety tests |
| `docs/tmp/c6_1a_producer_safety_report.md` | This report |

**Not changed (by design):** `src/features/option_surface_analyzer.py`, SurfaceRunner, cache/parquet/ORATS data files.

---

## CLI flags added or modified

| Flag | Default | Notes |
|------|---------|-------|
| `--output-root` | `DEFAULT_CACHE_ROOT` | Redirects meta/quotes output directory |
| `--spot-db-path` | `DEFAULT_SPOT_PRICES_PATH` | Configurable spot parquet input |
| `--tickers` | *(none)* | Space-separated ticker subset (`nargs="+"`) |
| `--tickers-file` | `DEFAULT_LIQUID_TICKERS_PATH` when `--tickers` omitted | CSV with `Ticker` column |
| `--start-date` | Jan 1 of `--start-year` | Inclusive ISO date bound |
| `--end-date` | Dec 31 of `--end-year` | Inclusive ISO date bound (replaces Feb 20 hack) |
| `--dry-run` | off | Plan only; exit 0 |
| `--overwrite` | off | Required when output parquets already exist |
| `--log-file` | shared log path | Use `-` for stderr only |

Existing flags retained: `--data-root`, `--start-year`, `--end-year`, `--frequency`, `--workers`, delta options, `--keep-zero-bid-quotes`.

**Mutual exclusion:** `--tickers` and `--tickers-file` cannot be combined (argparse `mutually_exclusive_group`).

**Flag naming:** All flag names match the C6.1 design memo exactly; no alternate convention was needed.

---

## Dry-run behavior

`--dry-run`:

1. Resolves date bounds, tickers, output paths, and weekly entry schedule.
2. Prints a structured summary to stdout (no joblib, no parquet writes).
3. Exits `0`.

Printed fields:

- `requested_start_date` / `requested_end_date`
- `resolved_schedule_min` / `resolved_schedule_max` / `resolved_entry_date_count`
- `ticker_source`, `ticker_count`, and inline ticker list (when ≤20 tickers)
- `data_root`, `output_root`, `spot_db_path`
- `meta_output_path`, `quotes_output_path`
- `meta_exists`, `quotes_exists`
- `overwrite_required_without_flag`

---

## Output safety behavior

| Rule | Behavior |
|------|----------|
| Default overwrite | **Refused** — exit `2` if meta or quotes target exists |
| `--overwrite` | Allows full replace of both parquets |
| `--output-root` | Smoke runs can target e.g. `cache/c6_smoke/` without touching canonical files |
| Filename pattern | Unchanged: `option_surface_meta_{frequency}_{start_year}_{end_year}.parquet` |
| Narrowed date window | Filename still uses year args; dry-run/report show actual resolved schedule bounds |

Default ticker universe path moved from legacy `cache/liquid_tickers.csv` to `DEFAULT_LIQUID_TICKERS_PATH` (`input/liquidity/liquid_tickers.csv`).

---

## Weekly entry-date behavior

Implemented in `src/data/trading_day.py`:

| Function | Role |
|----------|------|
| `resolve_weekly_entry_date` | Friday anchor → last chain-file day that week (Fri→Mon, 5-day lookback) |
| `weekly_trade_dates_in_range` | Sorted weekly entry dates for `[start, end]` inclusive |
| `target_weekly_expiry_from_schedule` | Pure `schedule[i+1]` helper for C6.1B diagnostic (not wired to producer) |

**Entry date rule:** last available trading day of the calendar week (Friday when Friday file exists; otherwise walk back). Weeks with no Mon–Fri chain file are omitted.

Producer `generate_trade_dates()` now calls `weekly_trade_dates_in_range` then `sample_fridays_by_frequency` (monthly first-Friday subsampling preserved for legacy monthly mode).

---

## Expiry selection — explicit confirmation

**Not changed in C6.1A.**

- `OptionSurfaceBuilder._find_best_expiry` — unchanged (chain-scanned near target DTE).
- `process_single_entry` expiry path — unchanged.
- No new expiry failure tags (`no_target_weekly_expiry`, etc.) — deferred to C6.1C.

---

## Tests run and results

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_trading_day.py tests/unit/test_adjusted_liquid_paths.py tests/unit/test_precompute_option_surface_cli.py -q
```

**Result:** `34 passed in 0.45s`

### Coverage map

| Area | Test file |
|------|-----------|
| Weekly entry: normal Friday | `test_trading_day.py` |
| Holiday fallback (Thu) | `test_trading_day.py` |
| Missing week omitted | `test_trading_day.py` |
| Inclusive bounds | `test_trading_day.py` |
| `target_weekly_expiry_from_schedule` | `test_trading_day.py` |
| Path constants/defaults | `test_adjusted_liquid_paths.py` |
| Dry-run does not write | `test_precompute_option_surface_cli.py` |
| Overwrite guard / `--overwrite` | `test_precompute_option_surface_cli.py` |
| `--output-root` redirect | `test_precompute_option_surface_cli.py` |
| Bounded date scope | `test_precompute_option_surface_cli.py` |
| Inline `--tickers` | `test_precompute_option_surface_cli.py` |
| `--tickers` / `--tickers-file` mutual exclusion | `test_precompute_option_surface_cli.py` |
| Default full-year bounds (Dec 31) | `test_precompute_option_surface_cli.py` |

Heavy processing mocked/isolated; no real ORATS data required.

---

## Deferred items

| Item | Phase |
|------|-------|
| Weekly expiry semantics (`_find_best_expiry` → calendar-paired) | C6.1B diagnostic → C6.1C if approved |
| Soft-failure `failure_reason` tags on success-path invalid rows | C6.1D-a |
| Producer quote dedupe | C6.1D-b (after C6.4 duplicate triage) |
| `refresh_weekly_inputs` surface wiring | C8 |
| Converge `SurfaceDataPaths` / CLI cache defaults on `DEFAULT_CACHE_ROOT` | C9 / Sprint 005 |
| Migrate `get_trading_fridays` in straddle/ironfly precompute scripts | Sprint 005 |
| Regenerated surface samples / cache artifacts | C6.4 pass 2 (post-review) |
| Producer default `--frequency weekly` | C9 runbook (doc-only unless HD approves code change) |

---

## Acceptance criteria checklist

| Criterion | Status |
|-----------|--------|
| Producer can dry-run safely | ✓ |
| Output root can be redirected | ✓ |
| Existing outputs protected by default | ✓ |
| Ticker/date scope can be bounded | ✓ |
| Spot DB path configurable | ✓ |
| Producer path defaults from `paths.py` | ✓ |
| Weekly entry schedule from `trading_day.py` | ✓ |
| Year-only range no longer uses Feb 20 shortcut | ✓ |
| Expiry selection unchanged | ✓ |
| Targeted tests pass | ✓ |
| No data/cache/parquet/backtest artifacts changed | ✓ |
