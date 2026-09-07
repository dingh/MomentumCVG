# Sprint 008 D0 — Evidence Review

**Date:** 2026-09-07  
**Design:** [`docs/tmp/sprint008_d0_design.md`](sprint008_d0_design.md) — **accepted** + crossed-quote amendment `sprint008_d0_crossed_quote_v1`  
**Executing SHA:** `af24f5056fca806fd417fd727b7193949943e4c5`  
**Working tree at execution:** clean  
**Policy version:** `sprint008_d0_crossed_quote_v1`  
**Evidence dir:** `C:/MomentumCVG_env/runs/sprint008_d0_20260907T204449Z/` (outside repo; **fresh** rerun)  
**Prior blocked runs (superseded for review):** `…/sprint008_d0_20260907T202025Z/`, `…/sprint008_d0_20260907T193835Z/`  
**Official artifacts:** `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` (unchanged)  
**Official execution SHA:** `e205b9acc5d0400aa38169de721acb7fb8268f29`  
**Command:** `C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint008_d0_readiness.py` (with `PYTHONPATH` = repo root)  
**Status:** **D0 execution complete — evidence awaiting review**

---

## Verdict

**`READY_WITH_NARROW_ENABLING_CHANGE`**

All gates pass under the versioned crossed-quote exclusion policy. One package is intentionally held as cash:

| Field | Value |
|---|---|
| Key | `2025-04-04` / `MU` / `long` |
| Policy | `sprint008_d0_crossed_quote_v1` |
| Reason | `crossed_quote_ask_lt_bid` |
| Call | bid 5.4 / ask 4.40 |
| Put | bid 3.0 / ask 2.45 |
| \(M\) / \(H\) | 7.625 / −0.775 (quotes preserved; not repaired) |
| `in_N` | **True** (denominator unchanged) |
| `analysis_eligible` | **False** |
| Stake treatment | cash (\(q=0\)) |

No ticker-specific exception; no quote clipping. Missing/nonfinite quotes would still block (none observed).

---

## Exclusion / eligibility counts

| Metric | Value |
|---|---|
| `in_N` | 6802 |
| Analysis-eligible | 6801 |
| Crossed-quote excluded | **1** (`2025-04-04\|MU`) |
| Primary `in_N` | 5890 |
| Primary analysis-eligible | 5889 |
| Primary required-input failures | 0 |
| Primary missing \(X\) | 0 |
| Primary M3 missing | 1 (MU → `crossed_quote_excluded`) |

---

## Focused tests (pre-rerun)

```
pytest tests/unit/test_sprint008_d0_input_readiness.py -q
15 passed in 0.50s
```

---

## Stage timings (official rerun)

| Stage | Seconds |
|---|---|
| Identity | 9.50 |
| Load | 0.07 |
| Shared quotes | 0.11 |
| Reconstruct \(N\) | 0.10 |
| Package \(M/H\) | 4.94 |
| Outcomes / M1–M3 | 41.08 |
| Accounting smoke | 2.42 |
| Gates | 0.69 |
| **Total** | **58.91** |

---

## Gate results

| Gate | Result | Detail |
|---|---|---|
| G1 identity | PASS | Sprint 007 D0 gates 8/8 |
| G2 joins | PASS | bad_join_rows=0; crossed_quote_excluded=1 under policy v1 |
| G3 midpoint authority | PASS | mid and ask-debit mismatches = 0 |
| G4 required inputs | PASS | primary_n=5890; failures=0 |
| G5 outcome coverage | PASS | primary_missing_x=0; payoff_reconcile_failures=0 |
| G6 measurements | PASS | M1/M2 ok; m3_missing=1 allowed (`crossed_quote_excluded`) |
| G7 reconstruction | PASS | in_N=6802 = included_long |
| G8 accounting | PASS | equal-dollar all \(h\); dummy reject; all-reject all \(h\) |
| G9 non-goals | PASS | no association/thresholds/SurfaceRunner |

---

## Evidence files (outside repo)

- `d0_readiness_manifest.json`
- `d0_readiness_gates.json`
- `d0_readiness_coverage.json`
- `d0_long_panel_preview.parquet`
- `d0_crossed_quote_exclusions.json`
- `execution_receipt.json`

Notebook (clean, in repo): `notebooks/sprint008/d0_input_readiness.ipynb` — CLI runner used (not re-executed into evidence dir).

---

## What this does **not** claim

- No D1 association / block bootstrap
- No profitability or threshold results
- No fill attainability
- No change to Sprint 006/007 accepted economics

---

## Remaining blockers

None for D0 readiness under the amended policy. **D1–D3 remain pending** (not started).

---

## Suggested review decision

Accept the computed verdict `READY_WITH_NARROW_ENABLING_CHANGE` and the crossed-quote cash treatment for `2025-04-04|MU`, then authorize D1 design. Do not start association/profitability/threshold work until this evidence is reviewed.
