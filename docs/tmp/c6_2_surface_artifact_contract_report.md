# C6.2 — Surface Artifact Contract and Audit Foundation

**Generated:** 2026-07-11 04:17:10 UTC

**C6.2 commit:** `ffb6ae94d33515971905f54b7a9db5af5bec6ff9`

## Verdict

**PASS**

## Scope

### Files changed

- `src/features/option_surface_contract.py`
- `scripts/audit_option_surface_artifacts.py`
- `tests/unit/test_option_surface_contract.py`
- `docs/tmp/c6_2_surface_artifact_contract_report.md`

### Artifacts audited

- Meta: `C:\MomentumCVG_env\cache\c6_2_surface_smoke\option_surface_meta_weekly_2024_2024.parquet`
- Quotes: `C:\MomentumCVG_env\cache\c6_2_surface_smoke\option_surface_quotes_weekly_2024_2024.parquet`

### Read-only guarantee

This audit only reads parquet inputs and writes a markdown report. No parquet mutation, no backfill. No raw ORATS access; weekly date alignment reads adjusted-liquid file presence only.

## Artifact inventory

- Meta path: `C:\MomentumCVG_env\cache\c6_2_surface_smoke\option_surface_meta_weekly_2024_2024.parquet`
- Quotes path: `C:\MomentumCVG_env\cache\c6_2_surface_smoke\option_surface_quotes_weekly_2024_2024.parquet`
- Meta exists: True
- Quotes exists: True
- Meta row count: 8
- Quotes row count: 120
- Ticker count: 2
- Entry-date range: 2024-01-05 .. 2024-01-26
- Expiry-date range: 2024-01-12 .. 2024-02-02

## Schema checks

- Required A1 columns present: True
- Required A2 columns present: True

## surface_valid invariant

- Status: PASS
- Pass count: 8
- Violation count: 0

## Failure vocabulary

- Known tags: ['no_expiries_on_entry_chain', 'no_expiry_found', 'no_options_at_entry', 'no_spot_at_expiry', 'no_spot_price', 'no_strikes_in_chain', 'no_target_weekly_expiry', 'target_weekly_body_not_quotable', 'target_weekly_expiry_not_listed']
- Unknown tag count: 0
- Failure breakdown: {}

## Settlement readiness

- Valid row count: 8
- dte_actual mismatch count: 0

## A1 metadata grain

- Grain: ['ticker', 'entry_date']
- Metadata row count: 8
- Duplicate row count: 0
- Duplicate key count: 0

## A1/A2 join integrity

- Orphan quote rows: 0
- Valid metadata rows without quote rows: 0
- Invalid metadata rows with quote rows (informational): 0

## Quote grain

- Grain: ['ticker', 'entry_date', 'expiry_date', 'strike', 'side']
- Duplicate key count: 0

## Date alignment

- Status: PASS
- Policy: WARN on misaligned entry_date (legacy pre-C6.1C artifacts tolerated)
- Misaligned entry dates: 0

## Tests

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_option_surface_weekly_expiry.py tests/unit/test_precompute_option_surface_cli.py tests/unit/test_diagnose_weekly_expiry_policy.py -q
```
Result: 41 passed

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_option_surface_contract.py tests/unit/test_audit_option_surface_artifacts.py -q
```
Result: 30 passed


## Remaining limitations

- C6.3 assembly-readiness metrics (straddle_ready, ironfly_candidate_ready) deferred.
- C6.4 broader coverage / validity-rate thresholds deferred.
- C6.1D failure-taxonomy cleanup for null failure_reason on legacy invalid rows deferred.
- C7 PIT universe harness deferred.

## Check summary

- **schema_checks**: PASS
- **surface_valid_invariant**: PASS
- **failure_vocabulary**: PASS
- **settlement_readiness**: PASS
- **meta_grain**: PASS
- **a1_a2_join_integrity**: PASS
- **quote_grain**: PASS
- **date_alignment**: PASS
