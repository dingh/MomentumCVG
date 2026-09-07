# Sprint 008 D0 — Evidence Review

**Date:** 2026-09-07  
**Design:** [`docs/tmp/sprint008_d0_design.md`](sprint008_d0_design.md) — **accepted** by implementation authorization  
**Repo HEAD at design acceptance:** `d3aa7b0`  
**Implementation:** uncommitted at execution (helper / tests / notebook / runner); record below  
**Evidence dir:** `C:/MomentumCVG_env/runs/sprint008_d0_20260907T193835Z/` (outside repo)  
**Official artifacts:** `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z`  
**Official execution SHA:** `e205b9acc5d0400aa38169de721acb7fb8268f29`  
**Command:** `C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint008_d0_readiness.py` (with `PYTHONPATH` = repo root)  
**Status:** **D0 execution complete — evidence awaiting review**

---

## Verdict

**`BLOCKED_BY_SPECIFIC_INPUT_GAP`**

One primary-window name in the reconstructed long \(N\) fails the required-input rule \(H \ge 0\):

| Field | Value |
|---|---|
| Key | `2025-04-04` / `MU` / `long` |
| Cause | Crossed body quotes (`ask < bid`) on call and put |
| Call | bid 5.4 / ask 4.40 / stored mid 4.900 |
| Put | bid 3.0 / ask 2.45 / stored mid 2.725 |
| \(M\) | 7.625 (matches `entry_cost_mid_per_share`) |
| \(H\) | −0.775 |
| Ask debit | 6.85 (= \(M+H\); ask-debit identity still holds) |

Rules were **not** relaxed. Detail: `d0_blocker_detail.json` in the evidence dir.

Allowed M3 cold starts are **not** the blocker (`m3_missing_primary=0`).

---

## Focused tests

```
pytest tests/unit/test_sprint008_d0_input_readiness.py -q
8 passed in 0.19s
```

Covers bid/ask midpoint authority (ignoring wrong stored mid), \(N\) cap, M3 window/cold-start/zero payoff, missing-outcome stake preservation, and rejected-cash accounting.

---

## Stage timings (official run)

| Stage | Seconds |
|---|---|
| Identity (Sprint 007 D0 reuse) | 9.82 |
| Load | 0.07 |
| Shared quotes | 0.11 |
| Reconstruct \(N\) | 0.10 |
| Package \(M/H\) | 3.44 |
| Outcomes / M1–M3 | 40.71 |
| Accounting smoke | 0.64 |
| Gates | 0.62 |
| **Total** | **55.51** |

---

## Gate results

| Gate | Result | Detail |
|---|---|---|
| G1 identity | PASS | Sprint 007 D0 gates 8/8 |
| G2 joins | PASS | bad_join_rows=0; shared bid/ask ok |
| G3 midpoint authority | PASS | mid and ask-debit mismatches = 0 |
| G4 required inputs | **FAIL** | primary_n=5890; failures=1 (`2025-04-04\|MU`) |
| G5 outcome coverage | PASS | primary_missing_x=0 |
| G6 measurements | PASS | M1/M2 ok; M3 missing=0 |
| G7 reconstruction | PASS | in_N=6802 = included_long; funnel match |
| G8 accounting | PASS | equal-dollar + dummy reject cash identity |
| G9 non-goals | PASS | no association/thresholds/SurfaceRunner |

---

## Coverage / reconciliation (supporting)

| Metric | Value |
|---|---|
| Long structure_ok / in_N | 6802 / 6802 |
| Primary in_N | 5890 |
| Primary required-input failures | **1** |
| Primary missing \(X\) | 0 |
| Primary M3 missing | 0 |
| \(N\) equals included longs | yes |
| Funnel constructable_long match | yes |
| Max equal-dollar budget error | ~3.6e-12 |

Historical Tier-A quantities were not used for research \(q_i(h)\).

---

## Evidence files (outside repo)

- `d0_readiness_manifest.json`
- `d0_readiness_gates.json`
- `d0_readiness_coverage.json`
- `d0_long_panel_preview.parquet`
- `d0_blocker_detail.json`
- `execution_receipt.json`

Notebook (clean, in repo): `notebooks/sprint008/d0_input_readiness.ipynb` — not re-executed into the evidence dir for this run (CLI runner used).

---

## Implementation footprint

| Path | Role |
|---|---|
| `src/backtest/sprint008_d0_input_readiness.py` | Readiness helper |
| `tests/unit/test_sprint008_d0_input_readiness.py` | Focused unit tests |
| `notebooks/sprint008/d0_input_readiness.ipynb` | Narrative entrypoint |
| `scripts/run_sprint008_d0_readiness.py` | Official-run executor |

---

## What this does **not** claim

- No D1 association / block bootstrap
- No profitability or threshold results
- No fill attainability
- No change to Sprint 006/007 accepted economics
- No relaxation of \(H \ge 0\) for crossed quotes

---

## Suggested review decision

Accept the **computed** verdict `BLOCKED_BY_SPECIFIC_INPUT_GAP` as faithful to the frozen required-input rule, **or** authorize a narrow protocol amendment (versioned) for crossed-quote treatment before D1. Do not start D1 while G4 fails under the current design.
