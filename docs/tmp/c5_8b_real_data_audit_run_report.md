# C5.8B Real-Data Adjusted Liquid Audit Smoke Run Report

**Date/time:** 2026-07-04 (local run ~14:29–14:30 PT)  
**Commit SHA:** `05db4a49d938c6c5b264c27ba59e1c68da9cb668`

## C5.8B goal

Operational validation of `scripts/audit_adjusted_liquid.py` on real ORATS-shaped data built from:

- raw ORATS ZIP sample (10 files, 2020)
- `liquid_tickers.csv` (C4 liquid universe)
- `splits_hist_liquid.parquet` (C5.7 scoped split file)
- → scoped-split adjusted smoke output under `cache_c4_smoke`

This is not a full production backfill and does not compare against the legacy full adjusted mirror.

## Step 1 — unit / regression tests

### Audit unit tests

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_audit_adjusted_liquid.py -q
```

**Result:** 16 passed in 0.49s

### C5 regression tests

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_fetch_splits_cli.py tests/unit/test_apply_split_adjustment_cli.py tests/unit/test_split_adjuster.py tests/unit/test_split_adjuster_filtered_zip.py tests/unit/test_ticker_universe.py -q
```

**Result:** 54 passed in 0.97s

## Step 2 — raw sample construction

**Source root (read-only):** `C:/ORATS/data/ORATS_Data/2020`  
**Sample root:** `C:/MomentumCVG_env/cache_c4_smoke/c5_8b_raw_sample/2020`

Prior smoke roots cleared:

- `C:/MomentumCVG_env/cache_c4_smoke/c5_8b_raw_sample`
- `C:/MomentumCVG_env/cache_c4_smoke/adjusted_liquid_scoped_split_smoke`

**Sampling rule:** deterministic first 3 + middle 4 + last 3 from sorted 2020 ZIP list (253 source files).

**Number copied:** 10

**Copied filenames:**

1. `ORATS_SMV_Strikes_20200102.zip`
2. `ORATS_SMV_Strikes_20200103.zip`
3. `ORATS_SMV_Strikes_20200106.zip`
4. `ORATS_SMV_Strikes_20200630.zip`
5. `ORATS_SMV_Strikes_20200701.zip`
6. `ORATS_SMV_Strikes_20200702.zip`
7. `ORATS_SMV_Strikes_20200706.zip`
8. `ORATS_SMV_Strikes_20201229.zip`
9. `ORATS_SMV_Strikes_20201230.zip`
10. `ORATS_SMV_Strikes_20201231.zip`

## Step 3 — adjusted scoped-split smoke build

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/apply_split_adjustment.py `
  --raw-root C:/MomentumCVG_env/cache_c4_smoke/c5_8b_raw_sample `
  --adj-root C:/MomentumCVG_env/cache_c4_smoke/adjusted_liquid_scoped_split_smoke `
  --splits C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet `
  --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv `
  --years 2020 `
  --workers 1 `
  --overwrite
```

**Exit status:** 0 (success)

**Adjusted output root:** `C:/MomentumCVG_env/cache_c4_smoke/adjusted_liquid_scoped_split_smoke/2020`

**Adjusted parquet files written:** 10

**First output paths:**

- `ORATS_SMV_Strikes_20200102.parquet` — 369,068 rows, 1,850 tickers
- `ORATS_SMV_Strikes_20200103.parquet` — 371,018 rows, 1,850 tickers
- `ORATS_SMV_Strikes_20200106.parquet` — 347,838 rows, 1,847 tickers

**Aggregate across 10 files:**

- Total rows: 4,113,889
- Unique tickers: 2,047

**Adjustment summary:** 10 files adjusted, 0 skipped, 0 errors (~35s wall time).

## Step 4 — audit run

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/audit_adjusted_liquid.py `
  --raw-root C:/MomentumCVG_env/cache_c4_smoke/c5_8b_raw_sample `
  --adj-root C:/MomentumCVG_env/cache_c4_smoke/adjusted_liquid_scoped_split_smoke `
  --splits C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet `
  --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv `
  --years 2020 `
  --sample-files 10 `
  --sample-rows 20000 `
  --seed 57 `
  --report-path docs/tmp/c5_8b_adjusted_liquid_smoke_audit_report.md
```

**Exit status:** 0

**Audit report path:** `docs/tmp/c5_8b_adjusted_liquid_smoke_audit_report.md`

**Audit verdict:** **PASS**

### Summary metrics (from audit report)

| Check | Status / value |
|-------|------------------|
| Split file audit | PASS |
| Inventory audit | PASS (10 raw ZIP / 10 adjusted parquet, 0 missing, 0 extra) |
| Universe containment | PASS (outside-universe ticker count = 0) |
| Structural audit | PASS |
| Raw-vs-adjusted math | PASS |
| bad_split_factor_count | 0 |
| missing_required_columns | 0 |
| missing_optional_adjusted_columns | 0 |
| spot_px_mismatch_count | 0 |
| trade_date_mismatch_count | 0 |
| raw math sampled_files | 10 |
| raw math sampled_rows | 200,000 |
| raw math matched_rows | 200,000 |
| raw math unmatched_rows | 0 |
| raw math mismatch_count | 0 |

No warnings or failures were reported.

## Safety confirmations

- No real ORATS API fetch was run
- No full adjusted backfill was run
- No strategy/backtest code was run
- `C:/ORATS/data/ORATS_Adjusted` was not modified
- `C:/ORATS/data/ORATS_Data` was read-only (10 ZIP copies only)
- `C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet` was read-only
- Only safe cache smoke roots were written under `C:/MomentumCVG_env/cache_c4_smoke/`
- No legacy mirror comparison was performed

## Limitations

- Sample is 10 trading days from 2020 only (~4.1M filtered rows); not representative of full history or all edge cases.
- Smoke output lives in `cache_c4_smoke` and is not committed to git.
- Audit math spot-check sampled 20,000 rows per file (200,000 total); full row-by-row verification was not performed.
- Unique ticker count (2,047) is below full universe (2,783) because only 10 dates were processed and universe filtering applies per file.

## Code changes

None. No audit-script bugs were found during this smoke run.

## Final verdict

**C5.8B: PASS**

All acceptance criteria met: tests green, smoke build succeeded, audit PASS with zero outside-universe tickers, zero structural defects, and zero raw-vs-adjusted math mismatches on the real sample.
