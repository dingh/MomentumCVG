# Adjusted Liquid Data Layer Audit Report

**Generated:** 2026-07-05 01:41:15 UTC

## Input paths

- **raw-root:** `C:\ORATS\data\ORATS_Data`
- **adj-root:** `C:\MomentumCVG_env\input\adjusted_liquid`
- **splits:** `C:\MomentumCVG_env\input\adjusted_liquid\splits_hist_liquid.parquet`
- **ticker-universe:** `C:\MomentumCVG_env\input\liquidity\liquid_tickers.csv`

## Years audited

2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026

## Audit configuration

- sample-files: 25
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

#### Year 2017

- status: PASS
- raw_zip_count: 251
- adjusted_parquet_count: 251
- missing_adjusted_count: 0
- extra_adjusted_count: 0
- raw_date_min: 20170103
- raw_date_max: 20171229
- adj_date_min: 20170103
- adj_date_max: 20171229

#### Year 2018

- status: PASS
- raw_zip_count: 252
- adjusted_parquet_count: 252
- missing_adjusted_count: 0
- extra_adjusted_count: 0
- raw_date_min: 20180102
- raw_date_max: 20181231
- adj_date_min: 20180102
- adj_date_max: 20181231

#### Year 2019

- status: PASS
- raw_zip_count: 252
- adjusted_parquet_count: 252
- missing_adjusted_count: 0
- extra_adjusted_count: 0
- raw_date_min: 20190102
- raw_date_max: 20191231
- adj_date_min: 20190102
- adj_date_max: 20191231

#### Year 2020

- status: PASS
- raw_zip_count: 253
- adjusted_parquet_count: 253
- missing_adjusted_count: 0
- extra_adjusted_count: 0
- raw_date_min: 20200102
- raw_date_max: 20201231
- adj_date_min: 20200102
- adj_date_max: 20201231

#### Year 2021

- status: PASS
- raw_zip_count: 252
- adjusted_parquet_count: 252
- missing_adjusted_count: 0
- extra_adjusted_count: 0
- raw_date_min: 20210104
- raw_date_max: 20211231
- adj_date_min: 20210104
- adj_date_max: 20211231

#### Year 2022

- status: PASS
- raw_zip_count: 251
- adjusted_parquet_count: 251
- missing_adjusted_count: 0
- extra_adjusted_count: 0
- raw_date_min: 20220103
- raw_date_max: 20221230
- adj_date_min: 20220103
- adj_date_max: 20221230

#### Year 2023

- status: PASS
- raw_zip_count: 250
- adjusted_parquet_count: 250
- missing_adjusted_count: 0
- extra_adjusted_count: 0
- raw_date_min: 20230103
- raw_date_max: 20231229
- adj_date_min: 20230103
- adj_date_max: 20231229

#### Year 2024

- status: PASS
- raw_zip_count: 254
- adjusted_parquet_count: 254
- missing_adjusted_count: 0
- extra_adjusted_count: 0
- raw_date_min: 20240102
- raw_date_max: 20241231
- adj_date_min: 20240102
- adj_date_max: 20241231

#### Year 2025

- status: PASS
- raw_zip_count: 250
- adjusted_parquet_count: 250
- missing_adjusted_count: 0
- extra_adjusted_count: 0
- raw_date_min: 20250102
- raw_date_max: 20251231
- adj_date_min: 20250102
- adj_date_max: 20251231

#### Year 2026

- status: PASS
- raw_zip_count: 34
- adjusted_parquet_count: 34
- missing_adjusted_count: 0
- extra_adjusted_count: 0
- raw_date_min: 20260102
- raw_date_max: 20260220
- adj_date_min: 20260102
- adj_date_max: 20260220


## Universe containment audit

**Status:** PASS

- adjusted_output_unique_ticker_count: 2780
- outside_universe_ticker_count: 0
- outside_universe_examples: []

## Adjusted-column structural audit

**Status:** PASS

- files_checked: 2299
- missing_required_columns: 0
- bad_split_factor_count: 0
- bad_adjusted_price_count: 0
- missing_optional_adjusted_columns: 0
- bad_optional_adjusted_price_count: 0
- spot_px_mismatch_count: 0
- trade_date_mismatch_count: 0

## Raw-vs-adjusted math spot-check

**Status:** PASS

- sampled_files: 25
- sampled_rows: 500000
- matched_rows: 500000
- unmatched_rows: 0
- math_mismatch_count: 0
- join_key_columns_used: ['ticker', 'expirDate', 'strike', 'cOpra', 'pOpra']
- raw_duplicate_join_key_rows: 0
- adjusted_duplicate_join_key_rows: 0
- fallback_join_key_files: 0
- merge_expansion_rows: 0

## Warnings

- None

## Failures

- None

## Next recommended action

All checks passed. Safe to proceed with downstream input-layer validation on this adjusted root.
