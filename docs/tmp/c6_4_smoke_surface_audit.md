# C6.4 — Fresh C6 Surface Smoke Audit

**Generated:** 2026-07-11 17:39:29 UTC

## Verdict

**PASS**

## Scope and lineage

- C6.4 audit implementation commit: `c75417de79ae19ed8bcdd3fa9d0afce6045275f8`
- C6.2 baseline commit SHA: `8d776e6058114beb397894dcf0c1e0825d969c75`
- C6.3 readiness implementation commit SHA: `e2ffc2b845cf8162898e7622fada9ff8d08a7711`
- Strict weekly-expiry producer implementation: `af9d9a08772b6e8c82c32acc39cbc84b32bb4326`
- Meta path: `C:\MomentumCVG_env\cache\c6_4_surface_smoke\option_surface_meta_weekly_2024_2024.parquet`
- Quotes path: `C:\MomentumCVG_env\cache\c6_4_surface_smoke\option_surface_quotes_weekly_2024_2024.parquet`
- Adjusted-liquid data root: `C:\MomentumCVG_env\input\adjusted_liquid`
- Spot DB path: `C:\MomentumCVG_env\cache\spot_prices_adjusted.parquet`
- Ticker scope: ['AAPL', 'MSFT', 'NVDA', 'SPY', 'QQQ']
- Requested start/end: 2024-01-01 .. 2024-03-31
- Legacy cache mode: False

- Repository HEAD when producer ran: `0a386f2517deff8be116f4729abf7e2cfc09531d`

The smoke run used repository HEAD containing the accepted strict weekly producer implementation above.

## Producer command

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/precompute_option_surface.py --data-root C:/MomentumCVG_env/input/adjusted_liquid --spot-db-path C:/MomentumCVG_env/cache/spot_prices_adjusted.parquet --output-root C:/MomentumCVG_env/cache/c6_4_surface_smoke --frequency weekly --start-year 2024 --end-year 2024 --start-date 2024-01-01 --end-date 2024-03-31 --tickers AAPL MSFT NVDA SPY QQQ --overwrite --workers 8
```

## Audit command

```powershell
scripts/audit_option_surface_artifacts.py --meta-path C:/MomentumCVG_env/cache/c6_4_surface_smoke/option_surface_meta_weekly_2024_2024.parquet --quotes-path C:/MomentumCVG_env/cache/c6_4_surface_smoke/option_surface_quotes_weekly_2024_2024.parquet --frequency weekly --data-root C:/MomentumCVG_env/input/adjusted_liquid --spot-db-path C:/MomentumCVG_env/cache/spot_prices_adjusted.parquet --start-date 2024-01-01 --end-date 2024-03-31 --sample-tickers AAPL MSFT NVDA SPY QQQ --report-format c6.4 --include-assembly-readiness --audit-commit c75417de79ae19ed8bcdd3fa9d0afce6045275f8 --producer-run-head 0a386f2517deff8be116f4729abf7e2cfc09531d --weekly-expiry-producer-commit af9d9a08772b6e8c82c32acc39cbc84b32bb4326 --c6-2-commit 8d776e6058114beb397894dcf0c1e0825d969c75 --c63-commit e2ffc2b845cf8162898e7622fada9ff8d08a7711 --c61-regression-result 41 passed in 0.38s --c62-test-result 25 passed in 0.13s --c63-test-result 39 passed in 0.07s --c64-test-result 33 passed in 0.83s --producer-command C:/MomentumCVG_env/venv/Scripts/python.exe scripts/precompute_option_surface.py --data-root C:/MomentumCVG_env/input/adjusted_liquid --spot-db-path C:/MomentumCVG_env/cache/spot_prices_adjusted.parquet --output-root C:/MomentumCVG_env/cache/c6_4_surface_smoke --frequency weekly --start-year 2024 --end-year 2024 --start-date 2024-01-01 --end-date 2024-03-31 --tickers AAPL MSFT NVDA SPY QQQ --overwrite --workers 8 --output-report docs/tmp/c6_4_smoke_surface_audit.md
```

## Artifact inventory

- Meta exists: True
- Quotes exists: True
- Meta row count (scoped, raw): 65
- Quotes row count (scoped, raw): 2114
- Ticker count: 5
- Entry-date range: 2024-01-05 .. 2024-03-28
- Expiry-date range: 2024-01-12 .. 2024-04-05

### Metrics-only normalized view

- Applied: False
- Note: no identical duplicates; raw and normalized counts match
- Raw meta rows: 65
- Normalized meta rows (metrics only): 65
- Raw quote rows: 2114
- Normalized quote rows (metrics only): 2114

## Requested versus actual coverage

Coverage and readiness metrics below use the normalized in-memory view when identical duplicates are present; raw row counts remain in artifact inventory.

| Metric | Value |
|--------|-------|
| Meta row count (normalized) | 65 |
| Quote row count (normalized) | 2114 |
| Ticker count | 5 |
| Entry-date count | 13 |
| Requested date range | 2024-01-01 .. 2024-03-31 |
| Resolved schedule range | 2024-01-05 .. 2024-03-28 |
| Actual meta entry range | 2024-01-05 .. 2024-03-28 |
| Actual quote entry range | 2024-01-05 .. 2024-03-28 |
| Actual expiry range | 2024-01-12 .. 2024-04-05 |
| surface_valid count/rate | 65 / 100.0% |
| straddle_ready count/rate | 65 / 100.0% |
| ironfly_candidate_ready count/rate | 65 / 100.0% |
| ironcondor_candidate_ready count/rate | 65 / 100.0% |
| straddle_ready (among surface_valid) | 100.0% |
| ironfly_ready (among surface_valid) | 100.0% |
| ironcondor_ready (among surface_valid) | 100.0% |

## Strict weekly-expiry evidence

- Eligible rows (successor schedule week exists): 65
- Exact target matches (null failure_reason, expiry == target): 65
- Silent mismatch count (null failure_reason, non-null expiry != target): 0
- Documented target failure count: 0
- Other producer failure count: 0
- Missing expiry without failure count: 0
- no_target_weekly_expiry: 0
- target_weekly_expiry_not_listed: 0
- no_expiries_on_entry_chain: 0
- target_weekly_body_not_quotable: 0
- Weekly expiry verdict: **PASS**


## C6.2 contract results

- Raw C6.2 contract verdict: **PASS**
- Adjusted C6.4 contract verdict: **PASS** (no policy adjustment)

### schema_checks
- Status: **PASS**
- meta_columns_present: True
- quotes_columns_present: True
- missing_meta_columns: []
- missing_quotes_columns: []

### surface_valid_invariant
- Status: **PASS**
- row_count: 65
- violation_count: 0
- pass_count: 65

### failure_vocabulary
- Status: **PASS**
- known_tags: ['no_expiries_on_entry_chain', 'no_expiry_found', 'no_options_at_entry', 'no_spot_at_expiry', 'no_spot_price', 'no_strikes_in_chain', 'no_target_weekly_expiry', 'target_weekly_body_not_quotable', 'target_weekly_expiry_not_listed']
- invalid_row_count: 0
- failure_breakdown: {}
- unknown_tag_count: 0
- null_reason_on_invalid_count: 0
- reason_on_valid_count: 0

### settlement_readiness
- Status: **PASS**
- valid_row_count: 65
- null_expiry_date: 0
- null_entry_spot: 0
- null_exit_spot: 0
- null_body_strike: 0
- null_dte_actual: 0
- dte_mismatch_count: 0

### meta_grain
- Status: **PASS**
- meta_row_count: 65
- duplicate_row_count: 0
- duplicate_key_count: 0
- grain: ['ticker', 'entry_date']

### a1_a2_join_integrity
- Status: **PASS**
- meta_key_count: 65
- quote_row_count: 2114
- orphan_quote_count: 0
- valid_meta_without_quotes_count: 0
- invalid_meta_with_quotes_count: 0

### quote_grain
- Status: **PASS**
- quote_row_count: 2114
- duplicate_row_count: 0
- duplicate_key_count: 0
- grain: ['ticker', 'entry_date', 'expiry_date', 'strike', 'side']

### date_alignment
- Status: **PASS**
- schedule_entry_count: 13
- artifact_entry_count: 13
- misaligned_entry_count: 0
- policy: WARN on misaligned entry_date (legacy pre-C6.1C artifacts tolerated)

## Duplicate triage

- Duplicate triage verdict: **PASS**

### A1 grain (ticker, entry_date)
- duplicate key count: 0
- duplicate row count: 0
- affected ticker count: 0
- affected date count: 0
- IDENTICAL_DUPLICATE keys: 0
- CONFLICTING_DUPLICATE keys: 0

### A2 grain (ticker, entry_date, expiry_date, strike, side)
- duplicate key count: 0
- duplicate row count: 0
- affected ticker count: 0
- affected date count: 0
- IDENTICAL_DUPLICATE keys: 0
- CONFLICTING_DUPLICATE keys: 0


## Failure breakdown

- Known tags: ['no_expiries_on_entry_chain', 'no_expiry_found', 'no_options_at_entry', 'no_spot_at_expiry', 'no_spot_price', 'no_strikes_in_chain', 'no_target_weekly_expiry', 'target_weekly_body_not_quotable', 'target_weekly_expiry_not_listed']
- Unknown tag count: 0
- Failure breakdown: {}

## Settlement readiness

- valid_row_count: 65
- null_expiry_date: 0
- null_entry_spot: 0
- null_exit_spot: 0
- null_body_strike: 0
- null_dte_actual: 0
- dte_mismatch_count: 0

## C6.3 assembly readiness

- Readiness verdict: **PASS**
- body_pair_ready: 65 (100.0%)
- straddle_ready: 65 (100.0%)
- otm_call_wing_available: 65
- otm_put_wing_available: 65
- ironfly_candidate_ready: 65 (100.0%)
- symmetric_ironfly_pair_available (informational): 65
- ironcondor_candidate_ready: 65 (100.0%)

## Per-ticker coverage

| ticker | attempted | surface_valid | straddle | iron-fly | iron-condor | top failure |
|--------|-----------|---------------|----------|----------|-------------|-------------|
| AAPL | 13 | 13 (100.0%) | 13 (100.0%) | 13 (100.0%) | 13 (100.0%) | (none) |
| MSFT | 13 | 13 (100.0%) | 13 (100.0%) | 13 (100.0%) | 13 (100.0%) | (none) |
| NVDA | 13 | 13 (100.0%) | 13 (100.0%) | 13 (100.0%) | 13 (100.0%) | (none) |
| QQQ | 13 | 13 (100.0%) | 13 (100.0%) | 13 (100.0%) | 13 (100.0%) | (none) |
| SPY | 13 | 13 (100.0%) | 13 (100.0%) | 13 (100.0%) | 13 (100.0%) | (none) |

## Blocking failures

- (none)

## Warnings

- (none)

## Tests

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_option_surface_weekly_expiry.py tests/unit/test_precompute_option_surface_cli.py tests/unit/test_diagnose_weekly_expiry_policy.py -q
```
Result: 41 passed in 0.38s

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_option_surface_contract.py tests/unit/test_audit_option_surface_artifacts.py -q
```
Result: 25 passed in 0.13s

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_option_surface_readiness.py -q
```
Result: 39 passed in 0.07s

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_option_surface_c64_audit.py -q
```
Result: 33 passed in 0.83s

## Conclusion

C6.4 fresh smoke audit completed with verdict **PASS**. This report supplies real-artifact evidence only; it does not establish strategy backtest readiness.

