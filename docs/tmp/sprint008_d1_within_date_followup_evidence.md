# Sprint 008 D1 within-date follow-up — Evidence Review

**Date:** 2026-09-08  
**Protocol (frozen before results):** [`sprint008_d1_within_date_followup_protocol.md`](sprint008_d1_within_date_followup_protocol.md)  
**Parent D1:** [`sprint008_d1_evidence_review.md`](sprint008_d1_evidence_review.md) — gate **`STOP_NO_THRESHOLDS` preserved**  
**Evidence dir:** `C:/MomentumCVG_env/runs/sprint008_d1_within_date_20260908T195615Z/`  
**Executing HEAD SHA:** `c23c364fa6f60bfaa04597eb0e136559e3576bac`  
**Working tree at execution:** **dirty** (follow-up implementation not yet committed)  
**Command:** `C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint008_d1_within_date_followup.py` (`PYTHONPATH` = repo root)  
**Official artifacts:** `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` (unchanged)  
**Environment:** Python 3.13.7; numpy 2.4.1; pandas 3.0.0; scipy 1.17.1  
**Status:** **Follow-up executed — evidence awaiting review** (exploratory; does not amend D1)

**Post-hoc disclosure:** This within-date analysis was proposed **after** viewing the original pooled-quintile D1 results.

---

## Verdict (exploratory)

Among same-date candidates, lower M1/M2 scores show **positive point** mean \(d_t = L_t - U_t\) at or above the 5pp economic reference, but **neither** measurement is Bonferroni-adjusted significant under frozen HAC. This does **not** overturn D1 `STOP_NO_THRESHOLDS` and does **not** authorize D2 or threshold search.

---

## Design vs D1 (reminder)

| | Original D1 | This follow-up |
|---|---|---|
| Groups | Pooled Q1 vs Q5 | Within-date lowest \(k\) vs highest \(k\), \(k=\lfloor n/5\rfloor\) |
| Weighting | Trade-weighted means | Equal weight per eligible date |
| Inference | Block bootstrap; family size 3 | Newey–West HAC (maxlags=3, Bartlett, small-sample); family size 2 |
| Gate | Labels + D2 gate | No gate change |

---

## Sample

| Item | Value |
|---|---|
| Development `in_N` rows | 3585 |
| Entry dates | 209 |
| Eligible dates (M1 / M2) | 209 / 209 |
| Excluded dates | 0 / 0 |
| Cutoff ties (low/high) | 0 / 0 |
| Median / mean \(k\) | 3.0 / 3.08 |
| Calendar gaps between eligible dates | min 6, median 7, max 8 days |
| HAC lag interpretation | Successive **eligible entry dates**, not calendar days |

---

## Primary results (\(h=1\), development only)

| | M1 | M2 |
|---|---:|---:|
| Mean \(L_t\) | +0.0145 | +0.0364 |
| Mean \(U_t\) | −0.0379 | −0.0391 |
| Mean \(d_t\) | **+0.0524** | **+0.0755** |
| ≥ 5pp economic reference? | yes | yes |
| Paired \(t\) (diag.) | \(t=1.03\), \(p=0.306\) | \(t=1.53\), \(p=0.127\) |
| HAC SE | 0.0445 | 0.0455 |
| HAC \(t\) / raw \(p\) | 1.18 / 0.240 | 1.66 / 0.099 |
| Bonferroni adj. \(p\) (\(2\times\)) | 0.481 | 0.197 |
| Adj. significant (\(\alpha=0.05\))? | **no** | **no** |
| Ordinary 95% HAC CI | [−0.035, +0.140] | [−0.014, +0.165] |
| Adjusted 97.5% HAC CI | [−0.048, +0.153] | [−0.027, +0.178] |

### Development halves (descriptive mean \(d_t\))

| Half | M1 | M2 |
|---|---:|---:|
| 2020–2021 (\(n=105\)) | +0.057 | +0.034 |
| 2022–2023 (\(n=104\)) | +0.048 | +0.118 |

---

## Comparison with original D1

| Measurement | D1 \(\Delta\) (Q1−Q5) | D1 label | Follow-up mean \(d_t\) |
|---|---:|---|---:|
| M1 | +0.063 | inconclusive | +0.052 |
| M2 | +0.089 | inconclusive | +0.076 |
| M3 | −0.035 | unsupported | (out of scope) |

Point gaps remain directionally favorable and similar in magnitude, but within-date equal-weighted inference still fails multiplicity-controlled significance — consistent with D1’s inconclusive statistical support.

---

## Runtime

| Stage | Seconds |
|---|---:|
| Build panel | 46.16 |
| Filter / economics | 0.05 |
| Analyze M1 | 0.91 |
| Analyze M2 | 0.91 |
| **Total** | **~48.1** |

---

## Artifacts

- `within_date_paired_observations.parquet` / `.csv`
- `within_date_excluded_dates.parquet` / `.csv`
- `within_date_followup_report.json` / `.md`
- `within_date_measurement_summaries.json`
- `execution_receipt.json`

Notebook: `notebooks/sprint008/d1_within_date_followup.ipynb`  
Runner: `scripts/run_sprint008_d1_within_date_followup.py`

---

## Limitations

- Post-hoc relative to pooled D1; not a pre-registered D1 endpoint.
- HAC lags are in eligible-date units; weekly spacing is disclosed but not modeled as calendar-time HAC.
- Exploratory only: no new D1 label, no D2 authorization, evaluation `2024+` closed.
- Hypothetical fills not claimed attainable.
