# Sprint 008 D1 — Measurement validation and gate decision

**Status:** `DRAFT — AWAITING REVIEW`  
**Drafted:** 2026-09-07  
**Agenda:** [`docs/agenda/current_sprint.md`](../agenda/current_sprint.md)  
**Working plan:** [`docs/agenda/sprint8_long_filter_plan.md`](../agenda/sprint8_long_filter_plan.md)  
**Prerequisite:** D0 **accepted** — [`docs/tmp/sprint008_d0_evidence_review.md`](sprint008_d0_evidence_review.md); evidence `C:/MomentumCVG_env/runs/sprint008_d0_20260907T204449Z/` (SHA `af24f50`); policy `sprint008_d0_crossed_quote_v1`  
**D0 design:** [`docs/tmp/sprint008_d0_design.md`](sprint008_d0_design.md)  
**Official artifacts:** `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` (immutable)

---

## Question

> Which of the three frozen entry-time measurements (M1, M2, M3), if any, reliably distinguish equal-dollar long-straddle **net return per dollar** on **2020–2023 development** data in a way that supports a simple cutoff?

**Required answer (after accepted implementation):** per measurement `supported` / `unsupported` / `inconclusive`, plus one sprint-level gate for whether D2 threshold work is authorized.

---

## Working hypothesis (not a verdict)

At least the cost-intensity benchmark **M1** (\(H/M\)) will show a directionally negative association with net returns under primary \(h=1\), but mechanical cost induction may inflate that link. D1 must separate gross-payoff vs cost and apply preregistered economic/stability/coverage criteria before any measurement is labeled `supported`.

---

## Authorization

This design is a **draft for review**. D1 implementation, association output, threshold selection, and D2 remain **unauthorized** until this design is accepted.

---

## 1. Inherited pins (do not reopen)

| Pin | Value |
|---|---|
| Population | Frozen `42:8` long ATM straddles; capped constructable \(N\) from D0 |
| Budget | \(B=\$10{,}000\) per entry date |
| Sizing | \(q_i(h)=(B/N)/(M_i + h H_i + \mathrm{fees}_i)\); \(\mathrm{fees}_i=0\) |
| Denominator \(N\) | Original capped `in_N` count **unchanged** by crossed-quote exclusions or later filters |
| Crossed quotes | Policy `sprint008_d0_crossed_quote_v1`: excluded packages stay in \(N\), stake cash, \(q=0\), not analysis-eligible |
| Scenarios | Primary \(h=1\); diagnostic \(h=0\); sensitivity \(h\in\{0.25,0.50\}\) |
| Measurements | **M1** \(H/M\); **M2** \(H/S_0\); **M3** \((M+H+\mathrm{fees})/(S_0\mu_t)\) with D0 M3 history rules |
| Chronological split | Development `2020-01-01`→`2023-12-31`; evaluation `2024-01-01`→`2026-07-10` **reserved for frozen-rule evaluation (D2)** |
| Engine | No `SurfaceRunner`; reuse D0 panel / helpers |

Sprint 006/007 accepted conclusions remain unreinterpreted. Hypothetical fills are not claimed attainable.

---

## 2. Analysis population and outcomes

### 2.1 Development universe (association only)

Trade-level association uses:

- `trade_date` ∈ [`2020-01-01`, `2023-12-31`];
- `in_N == True`;
- `analysis_eligible == True` (crossed-quote exclusions **out** of measurement/profitability denominators);
- finite outcome \(X \ge 0\);
- finite measurement under test (M3 cold-starts excluded from that measurement’s association only).

Crossed-quote and other cash holds remain in portfolio accounting (cash) but **do not** receive fabricated trade returns or winner/loser labels.

### 2.2 Primary outcome

Under scenario \(h\):

\[
r_i(h) = \frac{q_i(h)\,\bigl(X_i - (M_i + h H_i + \mathrm{fees}_i)\bigr)}{B/N}
= \frac{X_i - (M_i + h H_i + \mathrm{fees}_i)}{M_i + h H_i + \mathrm{fees}_i}
\]

when \(q_i(h)>0\) and \(X\) finite (equal-stake identity). Primary association uses **\(h=1\)**. Report \(h=0\) and intermediates as sensitivity only (same ranking protocol; no scenario shopping).

### 2.3 Gross payoff vs cost decomposition (required for M1/M2)

Using the same stake \(B/N\):

| Component | Per-share | Dollar / return form |
|---|---|---|
| Gross payoff | \(X\) | \(q_i(h)\,X\) / \((B/N)\) |
| Modeled entry cost | \(M + hH + \mathrm{fees}\) | \(1\) in net-return identity when fully invested |
| Net | \(X - (M + hH + \mathrm{fees})\) | \(r_i(h)\) |

Acknowledge that cost-based scores induce some mechanical association with net. Net significance alone does **not** support a filter.

---

## 3. Frozen association protocol (before output)

### 3.1 Rank correlation

For each measurement \(m\in\{\mathrm{M1},\mathrm{M2},\mathrm{M3}\}\) on the development analysis set:

- Spearman \(\rho(m, r(h=1))\);
- expected sign: **negative** (higher score → worse net returns);
- uncertainty via consecutive-date block resampling (§3.5).

A significant Spearman alone is **insufficient** for `supported`.

### 3.2 Five predefined score groups

- Sort analysis-eligible development trades by measurement **ascending** (low cost/hurdle first).
- Form **5 equal-count quintiles** (groups Q1…Q5).
- **Ties:** deterministic order by `(measurement, trade_date, ticker)` ascending; assign ranks with average-rank Spearman; for quintile cuts use the sorted order above (stable, no profitability optimization).
- **Missing measurement:** omit from that measurement’s correlation and groups; disclose counts; do **not** impute.
- Per group report: \(n\), mean \(r(h=1)\), block-bootstrap interval (§3.5), share of development winning-trade dollar profits, and contribution of the largest winners (top 5 and top 10 winners by dollar P&L within the development analysis set, attributed by group membership).

### 3.3 Within-date ranking

On each development date with ≥2 analysis-eligible names and finite \(m\):

- Spearman of \(m\) vs \(r(h=1)\) **within the date**;
- summarize: median within-date \(\rho\), fraction of dates with \(\rho<0\).

A measurement that only separates *dates* (not names within a date) is weaker cutoff evidence.

### 3.4 Chronological stability (development only)

Split development into two calendar halves **without peeking at association results for cut choice**:

| Subperiod | Dates |
|---|---|
| Dev-A | `2020-01-01` → `2021-12-31` |
| Dev-B | `2022-01-01` → `2023-12-31` |

Require the same qualitative pattern (negative rank association and/or monotonically weaker mean \(r\) from low to high groups) in both halves for `supported`. Do **not** open evaluation (`2024+`) for measurement selection.

### 3.5 Dependence: consecutive-date block resampling

**Frozen before association output:**

| Item | Decision |
|---|---|
| Block length | **4 consecutive traded entry dates** in the analysis calendar (dates with ≥1 analysis-eligible trade) |
| Sampling unit | A block = those 4 dates’ **full** analysis-eligible cross-sections |
| Procedure | Resample blocks with replacement to build paths of equal length (in blocks) to the original development calendar; recompute Spearman and group-mean gaps on each path |
| Paths | **1,000** bootstrap paths |
| Interval | Percentile 2.5% / 97.5% of the path statistic |
| Justification | Preserves within-date dependence and short-horizon serial dependence (recurring names across nearby weeks); length 4 ≈ one month of weekly entries — fixed a priori, not tuned to results |

Do **not** resample individual trades as iid. Do **not** choose block length from measurement–profitability fit.

### 3.6 Multiplicity

Three measurements tested. Freeze:

- Report per-measurement intervals **unadjusted** (primary disclosure).
- For the sprint-level D2 gate only: require that a measurement meeting `supported` criteria still has block-bootstrap 95% interval for Spearman entirely on the expected side of 0 **and** Q5−Q1 mean-\(r\) gap on the expected side after a conservative Bonferroni-style check on the two primary interval endpoints (treat the three Spearman tests as a family; require the unadjusted interval still excludes 0 after noting family size — i.e. disclose that three tests were run; do **not** fish for an adjusted \(p\) that salvages a borderline result).
- No additional measurements beyond M1–M3.

---

## 4. Classification criteria (per measurement)

Apply **all** dimensions below on development data under \(h=1\). Labels:

| Label | Meaning |
|---|---|
| `supported` | Economically meaningful, stable, directionally consistent with a simple cutoff rationale |
| `unsupported` | No usable discriminatory relationship under these criteria |
| `inconclusive` | Mixed, underpowered, or unstable |

### 4.1 Economic magnitude

- Expected: Q1 mean \(r\) above Q5 mean \(r\) (low score better).
- **`supported` requires both:** (a) point Q1−Q5 mean-\(r\) gap \(\ge 0.05\) (5 percentage points of stake return); (b) block-bootstrap 95% interval for (Q1−Q5) excludes 0 in the expected direction.
- If the interval excludes 0 but the point gap is in \([0.02, 0.05)\), classify **`inconclusive`** on magnitude (unless the overall label is already `unsupported` on other grounds).

### 4.2 Uncertainty

Spearman point estimate negative; block-bootstrap 95% interval for Spearman excludes 0 (expected side).

### 4.3 Stability

Same qualitative pattern in Dev-A and Dev-B (§3.4): negative Spearman sign in both; Q1 mean \(r\) ≥ Q5 mean \(r\) in both.

### 4.4 Coverage

- Each quintile has \(\ge 100\) trades **or** \(\ge 5\%\) of the measurement’s development analysis \(n\) (whichever is larger as a floor: use \(\max(100,\,0.05n)\)).
- ≥20 distinct entry dates represented in Q5 (poor region) and in Q1.

### 4.5 Cutoff support

Evidence that a **simple** threshold could remove a consistently poor region (e.g. Q5 mean \(r\) below Q1–Q4 and below 0, or Q5 clearly worst monotonic step) without needing a complex model. Monotonic mean-\(r\) decline across Q1→Q5 is supportive but not required if Q5 is a clear poor extreme.

### 4.6 Gross-edge check (M1 and M2 only)

Survivors in the better region (Q1 or below a notional high-score cut) must retain **non-trivial gross payoff** contribution: mean gross-payoff return in Q1 \(\ge 50\%\) of the development analysis-set mean gross-payoff return (or Q1 mean gross \(\ge 0\) with disclosed share of total gross dollars ≥ 15%). Failure → cannot be `supported` on net alone (`inconclusive` or `unsupported` per remaining evidence).

### 4.7 Within-date evidence

Median within-date Spearman \(\le 0\), or ≥50% of eligible dates have within-date \(\rho<0\). Failure alone does not force `unsupported` if other dimensions pass, but blocks `supported` (→ `inconclusive`).

### 4.8 Label mapping

| Condition | Label |
|---|---|
| All of §4.1–4.7 pass (4.6 N/A for M3) | `supported` |
| Spearman/groups flat or wrong-signed with intervals including 0; no poor extreme | `unsupported` |
| Mixed halves, thin coverage, gross-edge fail, or within-date fail while some net association exists | `inconclusive` |

---

## 5. Sprint-level gate for D2

| Gate | Rule |
|---|---|
| **Authorize D2** | ≥1 measurement labeled `supported` under §4 |
| **Stop (no thresholds)** | All three `unsupported` or `inconclusive` — D2 is a short stop record only |
| **Evaluation period** | **Not used** in D1 decisions; reserved for a frozen rule after D2 design acceptance |

If multiple measurements are `supported`, D1 ranks them for D2 candidacy by (1) Q1−Q5 gap, (2) within-date consistency, (3) gross-edge (for cost measures), without selecting a cutoff.

---

## 6. Sensitivity and disclosure (non-decision)

Report but **do not** use for classification:

- Association under \(h\in\{0,0.25,0.50\}\);
- Full-history companion descriptive tables (`2018-10-26`→`2026-07-10`) labeled exploratory;
- Evaluation-period **descriptive** replay of development group boundaries (no re-selection; clearly marked retrospective).

---

## 7. Minimal implementation footprint (only after design acceptance)

```
notebooks/sprint008/d1_measurement_validation.ipynb
src/backtest/sprint008_d1_measurement_validation.py
tests/unit/test_sprint008_d1_measurement_validation.py
```

**Reuse:** D0 readiness helper (`sprint008_d0_input_readiness.py`) for panel construction, equal-dollar quantities, crossed-quote flags, M1–M3; Sprint 007 artifact identity as needed. No second economic engine; no threshold search code in D1.

**Proposed helper surface:**

- Build / load development analysis panel from official artifacts + D0 logic.
- Compute \(r_i(h)\), gross/cost components.
- Spearman + quintile tables + within-date summary.
- Consecutive-date block bootstrap.
- Per-measurement classification + sprint gate JSON.

**Focused tests (synthetic):** quintile tie order; missing M3 omitted; crossed-quote rows excluded from association but retained in \(N\); block bootstrap preserves date cross-section; Q1−Q5 gap and label mapping edge cases; no evaluation labels in decision path.

---

## 8. Planned evidence outputs (outside repo)

Under `C:/MomentumCVG_env/runs/sprint008_d1_<UTC>/` (after authorized execution):

| Artifact | Content |
|---|---|
| `d1_manifest.json` | SHA, command, pins, verdicts |
| `d1_measurement_labels.json` | Per-measurement label + criterion checklist |
| `d1_gate.json` | Sprint-level D2 authorize / stop |
| `d1_spearman_bootstrap.json` | Point + intervals by measurement / \(h\) |
| `d1_quintile_tables.parquet` | Group stats, winner contributions |
| `d1_within_date_summary.json` | Within-date Spearman summary |
| `d1_stability_halves.json` | Dev-A / Dev-B pattern |
| `execution_receipt.json` | SHA, clean-tree state |

Committed notebook remains clean (narrative entrypoint).

---

## 9. Non-goals / stop rule

- No threshold grid, cutoff selection, or D2 portfolio comparison.
- No evaluation-period rule selection.
- No short-side / wings / signal-window search.
- No fill-attainability claims.
- No edits to official Sprint 006/007 artifacts.

**Stop** after this design is reviewed. Implementation starts only on explicit acceptance.

---

## 10. Summary for reviewers

**Approach:** On 2020–2023 development equal-dollar long trades, validate M1/M2/M3 vs net return per stake with Spearman, five equal-count groups, gross vs cost (cost measures), large-winner attribution, within-date and half-period stability, and consecutive-date block-bootstrap uncertainty — then assign `supported` / `unsupported` / `inconclusive` and a single D2 gate.

**Preserved:** D0 \(N\) denominator and crossed-quote cash policy; evaluation window sealed for frozen-rule use.

**Provisional path:** Design acceptance → implement/execute D1 → evidence review → D2 design only if authorized.
