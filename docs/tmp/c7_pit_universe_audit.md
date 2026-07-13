# C7.4 production audit evidence

## Evidence header

| Field | Value |
|-------|-------|
| Task | C7.4 production PIT-universe audit |
| Repository HEAD | `77b61eca6442c869a909138deba17941e6c1da2d` |
| Audit CLI commit | `77b61eca6442c869a909138deba17941e6c1da2d` |
| Execution date (UTC) | 2026-07-13T04:30:50Z – 2026-07-13T04:39:54Z |
| Panel path | `C:/MomentumCVG_env/input/liquidity/ticker_liquidity_panel.parquet` |
| Panel SHA-256 | `67e30956cd78bea97e9f90bfbd699f5e12e302e30db0e91912abd564dcf778de` |
| Weekly path | `C:/MomentumCVG_env/input/liquidity/ticker_liquidity_weekly_observations.parquet` |
| Weekly SHA-256 | `40f507fb165add28c5fdfb1dc9ef7a2e0176874f3bc240732bc538b990673f42` |
| Liquid ticker path | `C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv` |
| Liquid ticker SHA-256 | `e3094e6f1c8138ef5934f2b3158a37b3cc92ea01250c0dcfb8703ec03eb4b68a` |
| Exact command | see primary run log |
| Exit code | `0` |
| Focused pytest result | `118 passed, 1 skipped in 5.56s` |
| Primary runtime (seconds) | `543.7` |

**Exact command:**

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe C:/MomentumCVG/scripts/audit_pit_universe.py --panel-path C:/MomentumCVG_env/input/liquidity/ticker_liquidity_panel.parquet --weekly-path C:/MomentumCVG_env/input/liquidity/ticker_liquidity_weekly_observations.parquet --liquid-tickers-path C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv --discover-samples --dvol-top-pct 0.20 --spread-bottom-pct 1.0 --max-examples 20 --output-report C:/MomentumCVG_env/log/pit_universe_audit/c7_pit_universe_audit_primary.md
```

---

## Generated audit report (verbatim)

# C7 PIT Universe Audit

## Verdict

- overall status: `PASS`
- strict mode: `False`

## Scope and parameters

- panel path: `C:\MomentumCVG_env\input\liquidity\ticker_liquidity_panel.parquet`
- weekly path: `C:\MomentumCVG_env\input\liquidity\ticker_liquidity_weekly_observations.parquet`
- liquid-tickers path: `C:\MomentumCVG_env\input\liquidity\liquid_tickers.csv`
- requested dvol_top_pct: `0.2`
- requested spread_bottom_pct: `1.0`
- sample count: `3`
- rolling-provenance result count: `3`

## Artifact inventory

- panel row count: `2434339`
- weekly row count: `2438191`
- liquid ticker count: `2783`

## Artifact checks

- `panel_nonempty`: `PASS` — panel has rows
- `panel_snapshots`: `PASS` — panel has 477 distinct snapshots
- `weekly_nonempty`: `PASS` — weekly artifact has rows
- `weekly_weeks`: `PASS` — weekly artifact has 488 distinct weeks
- `required_columns`: `PASS` — all required columns present
- `grain`: `PASS` — no duplicate (month_date, ticker) rows
- `ticker_validity`: `PASS` — all ticker values non-null and non-empty
- `build_param_homogeneity`: `PASS` — build parameters homogeneous
- `panel_metric_integrity`: `PASS` — panel metric columns numeric, finite-or-null, non-negative
- `panel_provenance_integers`: `PASS` — panel integer provenance values valid
- `panel_provenance_dates`: `PASS` — panel provenance dates valid
- `panel_provenance_boolean`: `PASS` — has_valid_atm_pair is strictly boolean
- `weekly_required_columns`: `PASS` — all required weekly columns present
- `weekly_grain`: `PASS` — no duplicate weekly grain
- `weekly_ticker_validity`: `PASS` — 0 null and 0 empty weekly ticker values
- `weekly_has_valid_quote_domain`: `PASS` — weekly_has_valid_quote is boolean
- `weekly_volume_validity`: `PASS` — weekly volume numeric, finite, non-negative
- `weekly_spread_consistency`: `PASS` — weekly spread consistent with valid flag
- `sample_discovery`: `PASS` — 3 distinct mapped discovery cases found

## Supported parameter envelope

- status: `PASS`
- supported: `True`
- reason: requested configuration within supported superset envelope
- requested dvol_top_pct: `0.2`
- requested spread_bottom_pct: `1.0`
- stamped dvol_top_pct: `0.2`
- stamped spread_bot_pct: `1.0`

## Sample discovery

- status: `PASS`
- message: 3 distinct mapped discovery cases found
- case labels=['missing_or_new_liquidity'] target=2017-01-06 trade_date=2017-01-13
- case labels=['boundary_or_gap'] target=2017-04-21 trade_date=2017-04-28
- case labels=['normal'] target=2021-09-17 trade_date=2021-09-24

## PIT sample results

### Sample `missing_or_new_liquidity`

- label or labels: `missing_or_new_liquidity`
- target_snapshot_date: `2017-01-06`
- trade_date: `2017-01-13`
- resolved_snapshot_date: `2017-01-06`
- snapshot_lag_days: `7`
- eligible_count: `3680`
- selected_count: `737`
- dvol_threshold: `0.8`
- spread_threshold: `0.0`
- window_start_date: `2016-10-21`
- window_end_date: `2017-01-06`
- window_shortfall: `0`
- membership_hash: `bee45625afeb8e63`
- production_reference_match: `True`
- status: `PASS`
- notes: (none)
- exclusion counts: `{'invalid_atm_pair': 840, 'missing_or_nonfinite_dvol': 0, 'missing_or_nonfinite_spread': 0}`
- mismatch tickers: `[]`

### Sample `boundary_or_gap`

- label or labels: `boundary_or_gap`
- target_snapshot_date: `2017-04-21`
- trade_date: `2017-04-28`
- resolved_snapshot_date: `2017-04-21`
- snapshot_lag_days: `7`
- eligible_count: `3639`
- selected_count: `728`
- dvol_threshold: `0.8`
- spread_threshold: `0.0`
- window_start_date: `2017-02-03`
- window_end_date: `2017-04-21`
- window_shortfall: `0`
- membership_hash: `df3efd03cef16529`
- production_reference_match: `True`
- status: `PASS`
- notes: (none)
- exclusion counts: `{'invalid_atm_pair': 812, 'missing_or_nonfinite_dvol': 0, 'missing_or_nonfinite_spread': 0}`
- mismatch tickers: `[]`

### Sample `normal`

- label or labels: `normal`
- target_snapshot_date: `2021-09-17`
- trade_date: `2021-09-24`
- resolved_snapshot_date: `2021-09-17`
- snapshot_lag_days: `7`
- eligible_count: `3670`
- selected_count: `735`
- dvol_threshold: `0.8`
- spread_threshold: `0.0`
- window_start_date: `2021-07-02`
- window_end_date: `2021-09-17`
- window_shortfall: `0`
- membership_hash: `3e22ad47610c6ee4`
- production_reference_match: `True`
- status: `PASS`
- notes: (none)
- exclusion counts: `{'invalid_atm_pair': 1712, 'missing_or_nonfinite_dvol': 0, 'missing_or_nonfinite_spread': 0}`
- mismatch tickers: `[]`

## Rolling provenance

### Rolling `2017-01-06`

- target snapshot: `2017-01-06`
- checked ticker count: `20`
- checked tickers, capped: `['AA', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABMD', 'ABT', 'ABX', 'ACAD', 'ACIA', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADS', 'ADSK', 'AEM', 'AEO']`
- expected week range/count: `2016-10-21 .. 2017-01-06` / `12`
- stored-panel recomputation match: `True`
- future-invariance result: `True`
- field mismatch count: `0`
- field mismatch examples (capped): `[]`
- status: `PASS`

### Rolling `2017-04-21`

- target snapshot: `2017-04-21`
- checked ticker count: `20`
- checked tickers, capped: `['AA', 'AAL', 'AAOI', 'AAP', 'AAPL', 'ABBV', 'ABT', 'ABX', 'ACAD', 'ACIA', 'ACN', 'ACOR', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADPT', 'ADSK', 'AEM', 'AEO']`
- expected week range/count: `2017-02-03 .. 2017-04-21` / `12`
- stored-panel recomputation match: `True`
- future-invariance result: `True`
- field mismatch count: `0`
- field mismatch examples (capped): `[]`
- status: `PASS`

### Rolling `2021-09-17`

- target snapshot: `2021-09-17`
- checked ticker count: `20`
- checked tickers, capped: `['A', 'AA', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACB', 'ACIC', 'ACN', 'ACWI', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEM', 'AEO', 'AEP', 'AFRM']`
- expected week range/count: `2021-07-02 .. 2021-09-17` / `12`
- stored-panel recomputation match: `True`
- future-invariance result: `True`
- field mismatch count: `0`
- field mismatch examples (capped): `[]`
- status: `PASS`


## Sample superset coverage

- sample[0]: status=`PASS` selected_count=`20` missing=`[]`
- sample[1]: status=`PASS` selected_count=`20` missing=`[]`
- sample[2]: status=`PASS` selected_count=`20` missing=`[]`

## Full-history superset coverage

- status: `PASS`
- snapshots_checked: `477`
- unique_selected_tickers: `2783`
- missing_ticker_count: `0`
- sample_missing_tickers: `[]`
- canonical_params: `(0.2, 1.0)`

## Blocking failures

_None._

## Warnings

_None._


---

## C7.4 operator verification

### Discovered sample mapping table

| Label(s) | Target snapshot | Trade date | Resolved snapshot | Lag days | Sample status | Eligible | Selected | Membership hash |
|----------|-----------------|------------|-------------------|----------|---------------|----------|----------|-----------------|
| missing_or_new_liquidity | 2017-01-06 | 2017-01-13 | 2017-01-06 | 7 | PASS | 3680 | 737 | bee45625afeb8e63 |
| boundary_or_gap | 2017-04-21 | 2017-04-28 | 2017-04-21 | 7 | PASS | 3639 | 728 | df3efd03cef16529 |
| normal | 2021-09-17 | 2021-09-24 | 2021-09-17 | 7 | PASS | 3670 | 735 | 3e22ad47610c6ee4 |

Discovery status: **PASS** — 3 distinct mapped cases (normal, boundary_or_gap, missing_or_new_liquidity).

The `missing_or_new_liquidity` case targets the earliest panel snapshot (`2017-01-06`). That snapshot contains substantive liquidity defects (`invalid_atm_pair=840` on 4520 rows), so discovery did **not** rely solely on `baseline_initial_population` fallback.

All samples satisfy: `trade_date > target_snapshot_date`, `resolved_snapshot_date == target_snapshot_date`, `resolved_snapshot_date < trade_date`.

### Independent strict-prior resolution results

| Sample | Trade date | Independent S = max(month_date < T) | Report resolved | Match |
|--------|------------|---------------------------------------|-----------------|-------|
| missing_or_new_liquidity | 2017-01-13 | 2017-01-06 | 2017-01-06 | yes |
| boundary_or_gap | 2017-04-28 | 2017-04-21 | 2017-04-21 | yes |
| normal | 2021-09-24 | 2021-09-17 | 2021-09-17 | yes |

No sample consumed `month_date == trade_date` or `month_date > trade_date` for resolution (strict prior snapshot rule holds).

### Independent membership spot-check results

Independent recomputation used `compute_reference_universe()` with canonical parameters (`dvol_top_pct=0.20`, `spread_bottom_pct=1.0`, average rank, AND filter).

| Sample | Report selected count | Independent selected count | Membership match |
|--------|----------------------:|---------------------------:|------------------|
| missing_or_new_liquidity | 737 | 737 | yes |
| boundary_or_gap | 728 | 728 | yes |
| normal | 735 | 735 | yes |

All samples: `production_reference_match=True`, audit status `PASS`, membership hashes match report.

### Primary audit exit code

`0` (overall status **PASS**, non-strict)

### Final disposition

**ACCEPT C7.4** — all mandatory correctness evidence passes on production artifacts. No blocking failures, no warnings. Proceed to C7.6 (or next scheduled C7 closeout step per sprint plan).

Operational artifacts preserved under `C:/MomentumCVG_env/log/pit_universe_audit/`.
