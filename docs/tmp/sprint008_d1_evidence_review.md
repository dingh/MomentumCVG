# Sprint 008 D1 — Evidence Review

**Date:** 2026-09-07  
**Design:** [`docs/tmp/sprint008_d1_design.md`](sprint008_d1_design.md) — **accepted** (clarifications: scenario-specific \(q(h)\), bootstrap multiplicity, within-date constant scores/returns)  
**Executing SHA:** `72629a0d29f56771d1ff4a4ee3fe9cb227d593e4`  
**Working tree at execution:** clean  
**Evidence dir:** `C:/MomentumCVG_env/runs/sprint008_d1_20260907T223037Z/` (outside repo)  
**Official artifacts:** `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` (unchanged)  
**Official execution SHA:** `e205b9acc5d0400aa38169de721acb7fb8268f29`  
**Command:** `C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint008_d1_validation.py` (`PYTHONPATH` = repo root)  
**Seed / bootstrap:** `20260907`; \(L=4\); 10,000 reps; Bonferroni quantiles \(0.05/6\), \(1-0.05/6\); family size 3  
**Environment:** Python 3.13.7; numpy 2.x / pandas / scipy (venv `C:/MomentumCVG_env/venv`)  
**Status:** **D1 execution complete — evidence awaiting review**

---

## Sprint-level gate

**`STOP_NO_THRESHOLDS`** — zero measurements labeled `supported`. D2 threshold work is **not** authorized under the frozen protocol.

---

## Per-measurement labels

| Measurement | Label | Decision row | Point \(\Delta\) (Q1−Q5) | Adj. lower (Bonf.) | Adj. upper |
|---|---|---:|---:|---:|---:|
| **M1** (\(H/M\)) | `inconclusive` | 8 | +0.0632 | −0.0644 | +0.1735 |
| **M2** (\(H/S_0\)) | `inconclusive` | 8 | +0.0886 | −0.0577 | +0.2204 |
| **M3** (full-cross hurdle) | `unsupported` | 2 | −0.0352 | −0.2002 | +0.1197 |

Row 8 = economic bar and positive point \(\Delta\), but **Bonferroni lower bound does not exceed 0** (`P-stat` false).  
Row 2 = `P-wrong` (point \(\Delta\le 0\)).

### Predicate checklist

| Predicate | M1 | M2 | M3 |
|---|---|---|---|
| P-cov | true | true | true |
| P-Δ-def | true | true | true |
| P-econ (\(\Delta\ge 0.05\)) | true | true | false |
| P-stat (adj. lower \(>0\)) | **false** | **false** | false |
| P-sign | true | true | false |
| P-half | true | true | false |
| P-wd | true | true | true |
| P-gross | true | true | true (N/A) |
| P-wrong | false | false | **true** |

---

## Primary results (development 2020–2023, \(h=1\))

| Metric | Value |
|---|---|
| `in_N` development | 3585 (crossed-quote excluded in window: 0) |
| Analysis \(n\) (M1/M2/M3) | 3585 / 3585 / 3585 |
| Entry dates \(T\) | 209 |
| Bootstrap validity | 10,000 / 10,000 valid for each \(\Delta\) |

**Spearman diagnostics (not gate):** M1 \(\rho\approx -0.016\) (\(p\approx 0.34\)); M2 \(\rho\approx -0.025\) (\(p\approx 0.13\)); M3 \(\rho\approx -0.003\) (\(p\approx 0.84\)).

**Development halves (\(\Delta\)):** M1 Dev-A/B \(+0.040\) / \(+0.104\); M2 \(+0.042\) / \(+0.130\); M3 \(-0.140\) / \(+0.051\).

**Within-date:** all three pass `P-wd` (median \(\rho<0\); \(\approx 52\%\)–\(55\%\) of dates with \(\rho<0\); 0 constant exclusions).

### Quintile mean net returns (\(h=1\))

| | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| M1 mean \(r\) | +0.021 | +0.016 | −0.030 | +0.030 | −0.042 |
| M2 mean \(r\) | +0.061 | +0.021 | −0.046 | −0.013 | −0.028 |
| M3 mean \(r\) | −0.026 | +0.030 | +0.013 | −0.032 | +0.009 |

M1/M2 show a favorable Q1 vs Q5 **point** gap meeting the 5pp bar, but block dependence makes the Bonferroni-adjusted interval cross zero — hence inconclusive, not supported.

### Gross-edge / winners (disclosure)

- M1/M2 `P-gross` passed (Q1 gross midpoint retention vs sample).
- Top-5 winners by dollar net (shared pool): mix of quintiles; M1 top-5 dollar share in Q1 \(\approx 51\%\); M2 top-5 includes Q5 mass — disclosed in `d1_winner_attribution.json`.

### Sensitivity (\(h\in\{0,0.25,0.50\}\); not used for labels)

Frozen quintiles; scenario-specific \(q(h)\). Point \(\Delta\) generally rises with \(h\) for M1/M2; M3 remains wrong-signed across sensitivities. See `d1_sensitivity.json`.

---

## Runtime

| Stage | Seconds |
|---|---|
| Build panel | 46.06 |
| Primary economics | 0.03 |
| M1 (incl. bootstrap) | 2.43 |
| M2 | 2.40 |
| M3 | 2.41 |
| **Total** | **53.33** |

---

## Evidence files

- `d1_manifest.json`, `d1_measurement_labels.json`, `d1_gate.json`
- `d1_delta_bootstrap.json`, `d1_spearman_diagnostic.json`
- `d1_quintile_tables.parquet`, `d1_within_date_summary.json`, `d1_stability_halves.json`
- `d1_winner_attribution.json`, `d1_sensitivity.json`
- `execution_receipt.json`

Notebook (clean): `notebooks/sprint008/d1_measurement_validation.ipynb`

---

## Limitations

- Evaluation period (`2024+`) intentionally closed; no frozen-rule evaluation performed.
- Inconclusive M1/M2 means point economics look promising but fail multiplicity-controlled uncertainty under consecutive-date blocks — protocol was not relaxed.
- M3 unsupported on development under the preregistered \(\Delta\) direction.
- Prior Sprint 006/007 inspection of overlapping history means even development results are not pristine discovery samples.
- Hypothetical fills are not claimed attainable.

---

## What this does **not** claim

- No D2 thresholds or cutoff selection
- No evaluation-period association
- No change to Sprint 006/007 accepted economics
- No signal / sizing / structure changes

---

## Suggested review decision

Accept the computed labels and **`STOP_NO_THRESHOLDS`** gate as faithful to the accepted D1 design, **or** authorize a versioned protocol amendment before any D2 work. Do not start threshold search while zero measurements are `supported`.
