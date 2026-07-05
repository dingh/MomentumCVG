# C5.10C — Audit Failure Triage Report

**Date:** 2026-07-05 · **Repo SHA:** `8d9a82cedf1063edc8a9db3d15b57730f6f728f8`  
**Source audit:** [c5_10b_full_backfill_audit_report.md](c5_10b_full_backfill_audit_report.md)

Narrow diagnostic only — no backfill, no production overwrites, no code changes.

---

## Failed audit summary

C5.10B backfill **PASS** (2,299/2,299 files, exit 0). Audit **FAIL** only on raw-vs-adjusted math spot-check:

| Metric | Value |
|--------|-------|
| sampled_files | 25 |
| sampled_rows | 500,000 |
| matched_rows | 501,733 (**> sampled_rows**) |
| unmatched_rows | 0 |
| math_mismatch_count | 8,006 |

All other categories PASS (split, inventory, universe, structure).

---

## Files inspected

Required + one extra from audit failure list:

| File | stkPx fails (audit) | matched_many / sampled |
|------|---------------------|-------------------------|
| `ORATS_SMV_Strikes_20221223.parquet` | 133 | 20,133 / 20,000 |
| `ORATS_SMV_Strikes_20210921.parquet` | 151 | 20,151 / 20,000 |
| `ORATS_SMV_Strikes_20240213.parquet` | 116 | 20,116 / 20,000 |
| `ORATS_SMV_Strikes_20250514.parquet` | 122 | 20,122 / 20,000 |

---

## Audit merge-key inspection

`audit_raw_math_sample` (`scripts/audit_adjusted_liquid.py`):

- **Join keys:** `(ticker, expirDate, strike)` — `JOIN_KEYS` constant
- **Merge:** `adj_sample.merge(raw_for_merge, how="left")` — **no dedupe**
- **Math check:** `adj_* ≈ raw_* / split_factor` (`rtol=1e-8`, `atol=1e-8`)
- **matched_rows:** counts **post-merge rows** (`len(matched)`), not unique adj samples

Keys are **not guaranteed unique** in ORATS wide chains. Duplicate groups (~1.4k–1.7k/day) with **max 2 rows/key** appear in both raw and adjusted output.

---

## Duplicate/key uniqueness findings

Per inspected files (universe-filtered raw vs adjusted):

- Raw and adj share identical dup-key profile (e.g. 20221223: 475,299 rows, 473,861 unique keys, 2,876 rows in dup groups)
- **Root cause of dup keys:** same `(ticker, expirDate, strike)` with different option roots — e.g. SPX monthly `SPX230217C…` vs weekly `SPXW230217C…`, **different `stkPx`** (3831.79 vs 3831.75) on 20221223
- Many-to-many merge pairs an adj row with the **wrong** duplicate raw `stkPx` → false math FAIL
- `matched_rows > sampled_rows` explained: ~0.5–0.8% sample rows expand via duplicate raw matches (e.g. +133 on 20,000)

**Within adjusted parquet alone (no merge):** `adj_stkPx == stkPx / split_factor` → **0 failures** on all four inspected files (475k–557k rows each).

---

## Direct failed-file reproduction

Re-ran audit logic with `seed=57`, `sample_rows=20000`:

| File | math_fail (many-to-many) | math_fail (raw dedupe first) | mismatch sf==1 | sf!=1 |
|------|--------------------------|------------------------------|----------------|-------|
| 20221223 | 133 | **64** | 64 | 0 |
| 20210921 | 151 | **71** | 71 | 0 |
| 20240213 | 116 | **59** | 59 | 0 |
| 20250514 | 122 | **55** | 55 | 0 |

Deduping raw to `keep='first'` cuts false positives ~50% but **does not eliminate** them. Remaining mismatches are **100% ticker SPX**, all `split_factor == 1`.

`math_mismatch_count=8006` also **sums five price columns** per row (stkPx + 4 optional premiums) — inflates headline count vs unique rows.

---

## Example mismatch rows

20221223 — key `(SPX, 2/17/2023, 5025)` (two legitimate chain rows):

| Row | cOpra prefix | raw stkPx | adj_stkPx | split_factor |
|-----|--------------|-----------|-----------|--------------|
| A | SPX230217… | 3831.79 | 3831.79 | 1.0 |
| B | SPXW230217… | 3831.75 | 3831.75 | 1.0 |

Audit merge can pair adj row A with raw stkPx 3831.75 → reported fail (`abs_diff` 0.04). **Per-row adjustment is correct**; pairing is wrong.

Other examples (20210921, 20240213): same pattern — SPX, sf=1, small stkPx deltas (0.04–1.89), not split-scale errors.

---

## Real data bug vs audit bug assessment

| Hypothesis | Verdict |
|------------|---------|
| Real split-adjustment math error | **Rejected** — 0 within-row failures on full adjusted files |
| Audit join-key duplication | **Confirmed** — non-unique keys + many-to-many merge |
| Tolerance/rounding alone | **Partial** — tiny diffs, but root issue is wrong raw pairing |
| Raw duplicate rows | **Confirmed** — ORATS SPX/SPXW-style duplicates under same 3-key |
| `matched_rows > sampled_rows` | **Explained** — merge row multiplication |

**Not a C5.10B backfill data defect.** Production layer is internally consistent; audit spot-check methodology is unsound for ORATS duplicate keys.

---

## Recommended next action

1. **Patch audit only** (small, targeted): extend join keys with `cOpra`/`pOpra` (or dedupe both sides identically, or match on `stkPx` when keys duplicate); count matched rows as unique adj sample rows; optionally add duplicate-key WARN metric.
2. **Re-run audit** on existing production root — **no backfill rerun**.
3. Optional: document known SPX/SPXW duplicate-key behavior in audit report notes.

Do **not** rerun full backfill or patch `split_adjuster` based on this triage.

---

## Final verdict

**AUDIT BUG LIKELY — PATCH AUDIT BEFORE RERUN**
