# Sprint 008 D1 follow-up — within-date lowest vs highest 20%

**Status:** `PROTOCOL FROZEN — FOLLOW-UP EXECUTED; EVIDENCE AWAITING REVIEW` (historical protocol status)

**Review annotation (2026-09-12):** Reviewed and accepted as part of the D1 closeout. Does not amend original D1 labels or `STOP_NO_THRESHOLDS`. Does not authorize D2. Execution metadata below is unchanged.
**Drafted / frozen:** 2026-09-08 (before examining follow-up results)  
**Evidence:** [`sprint008_d1_within_date_followup_evidence.md`](sprint008_d1_within_date_followup_evidence.md) — `C:/MomentumCVG_env/runs/sprint008_d1_within_date_20260908T195615Z/`  
**Authorization:** User request authorizing this exploratory follow-up; does **not** amend D1  
**Parent D1 design:** [`sprint008_d1_design.md`](sprint008_d1_design.md) — `ACCEPTED`  
**Parent D1 evidence:** [`sprint008_d1_evidence_review.md`](sprint008_d1_evidence_review.md) — gate **`STOP_NO_THRESHOLDS`** (preserved)  
**Agenda:** [`docs/agenda/current_sprint.md`](../agenda/current_sprint.md)

---

## Research question

> Among candidates available on the **same entry date**, do lower **M1** or **M2** scores identify higher subsequent **net percentage** returns?

This analysis was proposed **after** viewing the original pooled-quintile D1 results. It is **exploratory additional evidence**. It does **not** replace D1 labels, reopen D1 classification, authorize D2, select thresholds, or open evaluation-period outcomes.

---

## Unchanged economics (inherited)

| Pin | Value |
|---|---|
| Measurements | **M1** = \(H/M\); **M2** = \(H/S_0\) only (M3 out of scope here) |
| Window | Development dates **2020–2023** only; evaluation `2024+` closed |
| Universe | Frozen `42:8` long ATM straddles; hold-to-expiry |
| Primary scenario | Full cross \(h=1\); fees \(=0\) |
| Budget / stake | Equal-dollar \(B/N\) with **original** \(N\) unchanged by exclusions or group selection |
| Quantities | Scenario-specific \(q_i(1)=(B/N)/(M_i+H_i+\mathrm{fees}_i)\); no reallocation when selecting L/U groups |
| Crossed quotes | Policy `sprint008_d0_crossed_quote_v1`: cash; not analysis-eligible; remain in \(N\) |
| Construction | Reuse accepted D0/D1 panel, M/H, sizing, and net-return identities |

Required D0/D1 input defects surface as **hard failures**, not silent sample shrinkage.

---

## How this differs from D1

| | Original D1 | This follow-up |
|---|---|---|
| Groups | Pooled development sample → frozen Q1…Q5 | **Within each entry date**: lowest \(k\) vs highest \(k\) by score |
| Contrast | Trade-weighted \(\Delta=\overline{r}_{Q1}-\overline{r}_{Q5}\) | Date-level \(d_t=L_t-U_t\); primary estimate = **equal-weight mean of \(d_t\)** |
| Inference | Overlapping block bootstrap + Bonferroni over **3** measurements | Paired date series + Newey–West/HAC; Bonferroni over **2** tests (M1, M2) |
| Gate | Exhaustive `supported` / `unsupported` / `inconclusive` + D2 gate | **No** D1/D2 gate change; descriptive + inferential report only |

---

## Within-date grouping (frozen before results)

For each measurement \(m\in\{\mathrm{M1},\mathrm{M2}\}\) and each development entry date \(t\):

1. Start with `in_N` candidates that are `analysis_eligible` and have a **finite entry-time** score \(m\).
2. Determine eligibility and group membership **without** using future returns / outcomes.
3. **Exclude** the date if fewer than **5** scored candidates, or if scores show **no variation** (all equal).
4. Sort ascending by \((m, \mathrm{ticker})\) (deterministic ties).
5. \(k=\lfloor n_{\mathrm{scored}}/5\rfloor\).
6. Lowest-score group \(L\): first \(k\) rows; highest-score group \(U\): last \(k\) rows.
7. Disclose excluded dates and reasons; disclose group sizes \(k\); disclose cutoff ties (score at L/U boundary shared with an adjacent non-selected name).

**Returns:** Use original \(h=1\) scenario quantities and costs. Do **not** resize selected groups.

**Missing outcome:** If any selected (\(L\) or \(U\)) trade lacks a required finite outcome / association-valid return, treat as an **input failure** (raise). Do not replace the trade or silently drop the date.

---

## Paired date-level observations

For every eligible date \(t\):

\[
L_t=\mathrm{mean}\{r_i(h=1): i\in L_t\},\quad
U_t=\mathrm{mean}\{r_i(h=1): i\in U_t\},\quad
d_t=L_t-U_t
\]

Primary point estimate: \(\overline{d}=\frac{1}{T^*}\sum_t d_t\) over eligible dates \(T^*\) (one observation per date).

---

## Statistical protocol (frozen before results)

Test \(H_0:\mathbb{E}[d_t]=0\) vs two-sided alternative.

| Item | Freeze |
|---|---|
| Diagnostic | Ordinary paired \(t\)-test of \(L_t\) vs \(U_t\) (equivalently one-sample \(t\) on \(d_t\)) |
| Primary inference | Intercept-only regression of \(d_t\) with **Newey–West / HAC** SEs |
| HAC | `maxlags=3`, **Bartlett** kernel, **small-sample correction on**, **\(t\)-based** inference |
| Lag interpretation | Lags are successive **eligible entry-date** observations (not calendar days). Report calendar gaps explicitly |
| Family | The two follow-up tests (M1, M2); size **2** |
| Adjusted \(p\) | \(\min(1,\,2\times p_{\mathrm{HAC}})\) |
| Intervals | Ordinary **95%** HAC; multiplicity-adjusted **97.5%** HAC |
| No search | Do **not** tune lags, group fractions, or methods based on results |

**Economic reference (separate from significance):** retain the **5 percentage-point** benchmark on \(\overline{d}\). A **significant negative** \(\overline{d}\) is unfavorable evidence.

Report statistical significance and economic magnitude separately. Do not invent a new D1/D2 classification label for this follow-up.

---

## Outputs

- Date-level paired observations (parquet/csv)
- Readable report: eligible/excluded counts & reasons; means of \(L_t\), \(U_t\), \(d_t\); paired \(t\) and HAC results; adjusted \(p\); CIs; descriptive 2020–2021 and 2022–2023 mean differences; comparison to original D1; execution SHA, dirty/clean tree, versions, command, input identity, runtime
- Sprint agenda update: follow-up executed, awaiting review; **preserve** original D1 `STOP_NO_THRESHOLDS`

---

## Explicit non-goals

- No threshold search; no D2; no evaluation-period outcomes
- No M3 in this follow-up
- No change to sizing, signals, or Sprint 006/007 evidence
- No amendment of original D1 labels or gate from this study alone
