# Sprint 008 D0 — Evidence Review

**Date:** 2026-09-07  
**Design:** [`docs/tmp/sprint008_d0_design.md`](sprint008_d0_design.md) — **accepted** (amended for per-leg quotes, all-reject cash, `in_N` M3 pool)  
**Executing SHA:** `5023fe7566fb0acf0d2ca4665a9b93df2a45112b`  
**Working tree at execution:** clean (`git status` empty after commit `5023fe7`)  
**Evidence dir:** `C:/MomentumCVG_env/runs/sprint008_d0_20260907T202025Z/` (outside repo; **fresh** rerun)  
**Prior (superseded for review):** `C:/MomentumCVG_env/runs/sprint008_d0_20260907T193835Z/`  
**Official artifacts:** `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` (unchanged)  
**Official execution SHA:** `e205b9acc5d0400aa38169de721acb7fb8268f29`  
**Command:** `C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint008_d0_readiness.py` (with `PYTHONPATH` = repo root)  
**Status:** **D0 execution complete — evidence awaiting review**

---

## Verdict

**`BLOCKED_BY_SPECIFIC_INPUT_GAP`**

One primary-window name in the reconstructed long \(N\) fails join / required-input rules (per-leg `ask >= bid` and \(H \ge 0\)):

| Field | Value |
|---|---|
| Key | `2025-04-04` / `MU` / `long` |
| Cause | Crossed body quotes (`ask < bid`) on call and put |
| Call | bid 5.4 / ask 4.40 / stored mid 4.900 |
| Put | bid 3.0 / ask 2.45 / stored mid 2.725 |
| \(M\) | 7.625 (matches `entry_cost_mid_per_share`) |
| \(H\) | −0.775 |
| Ask debit | 6.85 (= \(M+H\); ask-debit identity still holds) |
| `per_leg_quotes_ok` | **False** |

Rules were **not** relaxed. MU was **not** auto-repaired, clipped, or excluded. Detail: `d0_blocker_detail.json` in the evidence dir.

Allowed M3 cold starts are **not** the blocker (`m3_missing_primary=0`).

---

## Focused tests (pre-rerun)

```
pytest tests/unit/test_sprint008_d0_input_readiness.py -q
12 passed in 0.31s
```

Covers bid/ask midpoint authority, \(N\) cap, M3 window/cold-start/zero payoff / capped-out history exclusion, missing-outcome stake preservation, all-reject cash across \(h\in\{0,0.25,0.50,1\}\), crossed call masked by put spread, and body/leg strike mismatch.

---

## Stage timings (official rerun)

| Stage | Seconds |
|---|---|
| Identity (Sprint 007 D0 reuse) | 9.48 |
| Load | 0.07 |
| Shared quotes | 0.11 |
| Reconstruct \(N\) | 0.10 |
| Package \(M/H\) | 4.80 |
| Outcomes / M1–M3 | 40.26 |
| Accounting smoke | 2.28 |
| Gates | 0.76 |
| **Total** | **57.86** |

---

## Gate results

| Gate | Result | Detail |
|---|---|---|
| G1 identity | PASS | Sprint 007 D0 gates 8/8 |
| G2 joins | **FAIL** | bad_join_rows=1; shared bid/ask ok; **per-leg ask≥bid required** |
| G3 midpoint authority | PASS | mid and ask-debit mismatches = 0 |
| G4 required inputs | **FAIL** | primary_n=5890; failures=1 (`2025-04-04\|MU`) |
| G5 outcome coverage | PASS | primary_missing_x=0; payoff_reconcile_failures=0 |
| G6 measurements | PASS | M1/M2 ok; M3 missing=0 |
| G7 reconstruction | PASS | in_N=6802 = included_long; funnel match |
| G8 accounting | PASS | equal-dollar all \(h\); dummy reject; **all-reject all \(h\)**; missing-outcome smoke |
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
| All-reject all \(h\) | passed (invested 0, cash \(B\)) |

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
| `docs/tmp/sprint008_d0_design.md` | Accepted design (per-leg / cash / M3 pool clarified) |

---

## What this does **not** claim

- No D1 association / block bootstrap
- No profitability or threshold results
- No fill attainability
- No change to Sprint 006/007 accepted economics
- No relaxation of \(H \ge 0\) or per-leg `ask >= bid` for crossed quotes

---

## Remaining blockers

1. **`2025-04-04|MU`** crossed quotes — fails G2 (per-leg) and G4 (\(H<0\) / required inputs) under current rules.
2. Any treatment of invalid packages (repair, clip, exclude, alternate quote source) needs an **explicit protocol decision** before D1.

---

## Suggested review decision

Accept the **computed** verdict `BLOCKED_BY_SPECIFIC_INPUT_GAP` as faithful to the frozen required-input and per-leg quote rules, **or** authorize a narrow protocol amendment (versioned) for crossed-quote / invalid-package treatment before D1. Do not start D1 while G2/G4 fail under the current design.
