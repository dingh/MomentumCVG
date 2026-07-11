# C6.3 — Surface Assembly-Readiness Audit

**Generated:** 2026-07-11 04:53:37 UTC

**Commit:** `e2ffc2b845cf8162898e7622fada9ff8d08a7711`

## Verdict

**PASS**

## Scope

- Commit baseline (C6.2 follow-up): `8d776e6058114beb397894dcf0c1e0825d969c75`
- Meta: `C:\MomentumCVG_env\cache\c6_2_surface_smoke\option_surface_meta_weekly_2024_2024.parquet`
- Quotes: `C:\MomentumCVG_env\cache\c6_2_surface_smoke\option_surface_quotes_weekly_2024_2024.parquet`
- Sample window: 2024-01-01 .. 2024-01-31
- Ticker scope: all in filtered sample
- Iron-fly symmetry tolerance: 0.0

### Read-only guarantee

This audit only reads parquet inputs and writes a markdown report. No parquet mutation, no backfill, no strategy assembly or backtest.

## Downstream assembly contract

### Straddle leg requirements
- ``build_straddle_from_surface`` requires one body call and one body put at ``body_strike`` from A2.
- v1 readiness: ``straddle_ready == body_pair_ready`` (quotable body call + quotable body put).

### Iron-fly leg requirements
- ``build_ironfly_from_surface``: long OTM put / short ATM put / short ATM call / long OTM call.
- Body legs from ``is_body`` quotes; wings from ``is_otm`` quotes on each side.
- Wing selection uses ``abs_delta`` vs ``wing_target_delta`` at assembly time (not a C6.3 gate).
- C6.3 structural rule: body pair + at least one quotable OTM call wing + at least one
  quotable OTM put wing. Symmetric wing distance is informational only.

### Iron-condor leg requirements
- ``build_ironcondor_from_surface``: long OTM put / short nearer put / short nearer call / long OTM call.
- Short-leg candidates include body (``is_body | is_otm``); long legs must be further OTM than shorts.
- Delta targets applied at assembly; C6.3 uses conservative structural rule:
  body pair + quotable puts/calls forming at least one put vertical and one call vertical spread.

### Unresolved ambiguity

- Iron-condor readiness does not enforce delta-bucket targets; S3 may reject structurally valid surfaces when ``_choose_nearest`` cannot match targets.
- Spread filters (``max_leg_spread_pct``, ``max_spread_cost_ratio``) are assembly-time only.

## C6.2 prerequisite

- Contract verdict: **PASS**
- **schema_checks**: PASS
- **surface_valid_invariant**: PASS
- **failure_vocabulary**: PASS
- **settlement_readiness**: PASS
- **meta_grain**: PASS
- **a1_a2_join_integrity**: PASS
- **quote_grain**: PASS
- **date_alignment**: PASS

## Body-pair consistency

- A1 has_body_call and has_body_put must agree exactly with quotable A2 body-leg availability in both directions.
- Surfaces with body/A1 inconsistencies: 0

## Straddle readiness

- Surface count: 8
- body_pair_ready: 8 (100.0%)
- straddle_ready: 8 (100.0%)
- Conditional (among surface_valid): 100.0%

## OTM wing availability

- otm_call_wing_available: 8 (100.0%)
- otm_put_wing_available: 8 (100.0%)
- otm_wing_pair_available: 8 (100.0%)

## Iron-fly candidate readiness

### Definition

body_pair_ready AND at least one quotable OTM call wing AND at least one quotable OTM put wing.

- ironfly_candidate_ready: 8 (100.0%)
- Conditional (among surface_valid): 100.0%

### Symmetric-wing informational metric

Symmetric wing distance is not required by the current downstream iron-fly assembler. It is reported only as an informational structural characteristic.

- symmetric_ironfly_pair_available: 8 (100.0%)
- symmetric_ironfly_pair_count: 43
- Symmetry tolerance: 0.0

## Iron-condor candidate readiness

### Definition derived from S3

body_pair_ready AND ≥1 quotable put vertical spread AND ≥1 quotable call vertical spread (long further OTM than short on each side; body may be inner short leg).

- ironcondor_candidate_ready: 8 (100.0%)
- Conditional (among surface_valid): 100.0%

### Limitations

- Does not apply delta targets or spread filters from ``BacktestRunConfig``.
- Candidate count = put_vertical_pairs × call_vertical_pairs (structural only).

## Readiness failure breakdown

- (none)

## Tests

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_option_surface_weekly_expiry.py tests/unit/test_precompute_option_surface_cli.py tests/unit/test_diagnose_weekly_expiry_policy.py -q
```
Result: 41 passed

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_option_surface_contract.py tests/unit/test_audit_option_surface_artifacts.py -q
```
Result: 32 passed

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_option_surface_readiness.py -q
```
Result: 39 passed


## Remaining limitations

- No pricing beyond quotable bid/ask/mid checks.
- No fill simulation or transaction costs.
- No strategy ranking, backtest, or Sharpe/profitability conclusion.
- C6.4 broader coverage thresholds deferred.

### Files changed (C6.3)

- `src/features/option_surface_readiness.py`
- `scripts/audit_option_surface_artifacts.py`
- `tests/unit/test_option_surface_readiness.py`
- `tests/unit/test_audit_option_surface_artifacts.py`
- `docs/tmp/c6_3_surface_assembly_readiness_report.md`
