# C5.10D — Audit Patch Report

**Date:** 2026-07-05 · **Code baseline SHA:** `31e3effb336cbac441cf4cc6dc2adc9cb69af686`  
**Triage source:** [c5_10c_audit_failure_triage_report.md](c5_10c_audit_failure_triage_report.md)

## Root cause summary

C5.10B backfill data is correct. C5.10B audit **FAIL** was a false positive: raw-vs-adjusted math merged on `(ticker, expirDate, strike)` only, causing many-to-many pairing when SPX/SPXW rows share the same 3-key with different `stkPx`.

## Files changed

| File | Change |
|------|--------|
| `scripts/audit_adjusted_liquid.py` | OPRA-aware join keys; duplicate-key guard; matched_rows fix |
| `tests/unit/test_audit_adjusted_liquid.py` | SPX/SPXW regression + fallback duplicate-key test |

## Exact audit join-key fix

**Preferred keys:** `ticker`, `expirDate`, `strike`, `cOpra`, `pOpra` (when present on both raw and adjusted).

**Fallback:** base 3-key only if OPRA columns missing — emits warning; **FAIL** if fallback keys are non-unique (no false math).

**Merge safety:** `validate="one_to_one"`; reject merge row expansion; `matched_rows` = unique sampled adjusted rows matched (≤ `sampled_rows`).

**New metrics:** `join_key_columns_used`, `raw_duplicate_join_key_rows`, `adjusted_duplicate_join_key_rows`, `fallback_join_key_files`, `merge_expansion_rows`.

## Duplicate-key handling

| Condition | Behavior |
|-----------|----------|
| OPRA keys present, unique | Merge + math check |
| OPRA missing | WARN + fallback 3-key; FAIL if duplicates |
| Duplicates on chosen keys | FAIL clearly; `math_mismatch_count` not inflated |
| Merge expansion | FAIL |

## Tests run

```powershell
pytest tests/unit/test_fetch_splits_cli.py tests/unit/test_apply_split_adjustment_cli.py tests/unit/test_split_adjuster.py tests/unit/test_split_adjuster_filtered_zip.py tests/unit/test_ticker_universe.py tests/unit/test_audit_adjusted_liquid.py -q
```

**Result:** 72 passed in 1.82s

## Production audit rerun

Same command as C5.10B; report → `docs/tmp/c5_10d_full_backfill_audit_report.md`

**Exit code:** 0 · **Verdict:** **PASS** (all categories)

## Key audit metrics (patched)

| Metric | C5.10B (broken join) | C5.10D (patched) |
|--------|----------------------|------------------|
| sampled_rows | 500,000 | 500,000 |
| matched_rows | 501,733 | **500,000** |
| unmatched_rows | 0 | 0 |
| math_mismatch_count | 8,006 | **0** |
| raw_duplicate_join_key_rows | (not tracked) | 0 |
| adjusted_duplicate_join_key_rows | (not tracked) | 0 |
| join_key_columns_used | 3-key implicit | `ticker, expirDate, strike, cOpra, pOpra` |

Inventory: 2,299 raw ZIPs = 2,299 adjusted parquets, 0 missing/extra. Outside-universe tickers: 0.

## Final recommendation

**ACCEPT C5.10B AFTER AUDIT PATCH**

Production adjusted-liquid root `C:/MomentumCVG_env/input/adjusted_liquid` (2017–2026) is structurally complete and math-consistent on the patched audit. Ready for post-backfill downstream smoke / path wiring (deferred scope).

No backfill rerun required. Do not patch `split_adjuster`.
