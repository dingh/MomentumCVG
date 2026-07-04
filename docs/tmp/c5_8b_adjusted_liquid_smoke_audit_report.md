# Adjusted Liquid Data Layer Audit Report

**Generated:** 2026-07-04 21:30:18 UTC

## Input paths

- **raw-root:** `C:\MomentumCVG_env\cache_c4_smoke\c5_8b_raw_sample`
- **adj-root:** `C:\MomentumCVG_env\cache_c4_smoke\adjusted_liquid_scoped_split_smoke`
- **splits:** `C:\MomentumCVG_env\input\adjusted_liquid\splits_hist_liquid.parquet`
- **ticker-universe:** `C:\MomentumCVG_env\input\liquidity\liquid_tickers.csv`

## Years audited

2020

## Audit configuration

- sample-files: 10
- sample-rows: 20000
- seed: 57
- universe ticker count: 2783

## Overall verdict: **PASS**

## Split file audit

**Status:** PASS

- split_row_count: 1347
- split_unique_ticker_count: 819
- split_date_min: 2007-01-03 00:00:00
- split_date_max: 2026-07-02 00:00:00
- null_divisor_count: 0
- nonpositive_divisor_count: 0
- duplicate_key_count: 0
- conflicting_duplicate_count: 0
- outside_universe_split_ticker_count: 0

## Adjusted output inventory audit

**Status:** PASS

### Per-year inventory

#### Year 2020

- status: PASS
- raw_zip_count: 10
- adjusted_parquet_count: 10
- missing_adjusted_count: 0
- extra_adjusted_count: 0
- raw_date_min: 20200102
- raw_date_max: 20201231
- adj_date_min: 20200102
- adj_date_max: 20201231


## Universe containment audit

**Status:** PASS

- adjusted_output_unique_ticker_count: 2047
- outside_universe_ticker_count: 0
- outside_universe_examples: []

## Adjusted-column structural audit

**Status:** PASS

- files_checked: 10
- missing_required_columns: 0
- bad_split_factor_count: 0
- bad_adjusted_price_count: 0
- missing_optional_adjusted_columns: 0
- bad_optional_adjusted_price_count: 0
- spot_px_mismatch_count: 0
- trade_date_mismatch_count: 0

## Raw-vs-adjusted math spot-check

**Status:** PASS

- sampled_files: 10
- sampled_rows: 200000
- matched_rows: 200000
- unmatched_rows: 0
- math_mismatch_count: 0

## Warnings

- None

## Failures

- None

## Next recommended action

All checks passed. Safe to proceed with downstream input-layer validation on this adjusted root.
