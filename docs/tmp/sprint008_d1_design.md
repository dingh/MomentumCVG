# Sprint 008 D1 — Measurement validation and gate decision

**Status:** `DRAFT — AWAITING REVIEW`  
**Drafted:** 2026-09-07  
**Revised:** 2026-09-07 (review findings vs commit `8ea69e1`)  
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

At least the cost-intensity benchmark **M1** (\(H/M\)) will show a directionally favorable Q1−Q5 net-return gap under primary \(h=1\), but mechanical cost induction may inflate that link. D1 must separate gross midpoint P&L vs execution drag and apply the preregistered statistical/economic gate before any measurement is labeled `supported`.

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
| Chronological split | Development `2020-01-01`→`2023-12-31` for **all** D1 association / sensitivity / classification; evaluation `2024-01-01`→`2026-07-10` **closed until a rule is frozen for evaluation** |
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

Earlier completed observations (including pre-2020) may still enter M3’s accepted historical \(\mu_t\) calculation under D0 rules; they are **not** association rows.

### 2.2 Primary outcome and cost identity

Define scenario all-in cost per share:

\[
C_i(h) = M_i + h H_i + \mathrm{fees}_i
\]

with \(\mathrm{fees}_i=0\). When \(q_i(h)>0\) and \(C_i(h)>0\) and \(X\) finite:

\[
r_i(h) = \frac{X_i - C_i(h)}{C_i(h)}
= \frac{q_i(h)\,(X_i - C_i(h))}{B/N}
\]

(equal-stake identity). Primary gate uses **\(h=1\)**. Report \(h\in\{0,0.25,0.50\}\) as development-only sensitivity (same frozen quintile memberships and quantities; no scenario shopping).

If \(C_i(h)\le 0\) or non-finite: trade is **invalid for association** under that \(h\) (disclose count; do not impute).

### 2.3 Gross midpoint P&L vs execution drag (required)

For eligible trades with \(C_i(h)>0\):

| Component | Return form | Dollar form |
|---|---|---|
| Gross midpoint P&L return | \(g_i(h)=(X_i - M_i)/C_i(h)\) | \(q_i(h)\,(X_i - M_i)\) |
| Execution drag | \(d_i(h)=(h H_i + \mathrm{fees}_i)/C_i(h)\) | \(q_i(h)\,(h H_i + \mathrm{fees}_i)\) |
| Net return | \(r_i(h)=g_i(h)-d_i(h)\) | \(q_i(h)\,(X_i - C_i(h))\) |

**Reconciliation (every eligible trade):** dollar net \(=\) dollar gross midpoint P&L \(-\) dollar drag; invested stake \(q_i(h)\,C_i(h)=B/N\).

Acknowledge that cost-based scores induce some mechanical association with net. Net significance alone does **not** support a filter.

**Do not** use \(X/C\) (expiry payoff over cost) or “nonnegative expiry payoff” as the gross-edge criterion.

### 2.4 Large-winner attribution (disclosure)

On the development analysis set for each measurement:

- Rank trades by dollar net P&L \(q_i(h)\,(X_i-C_i(h))\) descending.
- Attribute **top 5** and **top 10** winners to their frozen quintile memberships (counts and dollar share).
- No filtered-portfolio comparison; no selected “notional cutoff.”

---

## 3. Primary statistical endpoint and multiplicity

### 3.1 Primary contrast

For each measurement \(m\in\{\mathrm{M1},\mathrm{M2},\mathrm{M3}\}\) under \(h=1\), on the development analysis set with frozen quintiles (§4):

\[
\Delta_m = \overline{r}_{\mathrm{Q1}}(m) - \overline{r}_{\mathrm{Q5}}(m)
\]

Expected direction: \(\Delta_m > 0\) (low score / low hurdle better).

### 3.2 Family and Bonferroni-adjusted bootstrap intervals

| Item | Freeze |
|---|---|
| Family | The three primary contrasts \(\Delta_{\mathrm{M1}},\Delta_{\mathrm{M2}},\Delta_{\mathrm{M3}}\) under \(h=1\) |
| Family size | **Always 3**, even if a measurement is later labeled inconclusive or has missing scores |
| Family-level \(\alpha\) | \(0.05\) |
| Individual two-sided level | \(1 - 0.05/3 = 98.3333\%\) |
| Percentile quantiles | \(\alpha/(2\times 3)=0.05/6\approx 0.0083333\) and \(1-0.05/6\approx 0.9916667\) |
| **Statistical support** | Adjusted **lower** bound of \(\Delta_m\) **strictly exceeds 0** |
| Economic magnitude | Point \(\Delta_m \ge 0.05\) (retained) |
| Ordinary 95% intervals | May be reported as **descriptive** evidence only; **must not** replace the adjusted gate |

The same adjusted-\(\Delta\) rule applies in per-measurement classification (§6) and the sprint-level D2 gate (§7).

### 3.3 Spearman (supporting diagnostic only)

Pooled Spearman \(\rho(m,r(h=1))\) and ordinary/descriptive intervals may be reported. They:

- are **not** mandatory for `supported`;
- **must not** reject a measurement that meets the \(\Delta\) gate, economic bar, stability, coverage, gross-edge (if applicable), and within-date checks solely because pooled Spearman is weak or non-significant.

---

## 4. Quintile grouping (frozen on original development sample)

For each measurement \(m\):

1. Restrict to the development analysis set with finite \(m\) and valid \(r(h=1)\).
2. Sort by `(m, trade_date, ticker)` ascending (deterministic ties).
3. Assign **5 equal-count quintiles** Q1…Q5 along that order (standard equal-count split: ranks \(1..n\) → group \(\lceil 5\cdot k/n\rceil\) with the usual edge handling so sizes differ by at most 1).
4. **Freeze** these memberships on the original sample. Reuse the **same** memberships for:
   - point \(\Delta_m\) and group tables;
   - every bootstrap replication (§5);
   - Dev-A / Dev-B stability (§6.3);
   - \(h\in\{0,0.25,0.50\}\) sensitivity (same names/groups; recompute \(r(h)\) only).

**Missing \(m\):** omit from that measurement’s groups and \(\Delta_m\); disclose \(n\); do not impute.

**Empty Q1 or Q5 after filters:** \(\Delta_m\) is **undefined** → measurement cannot be `supported` (see §6).

Per group report: \(n\), mean \(r(h=1)\), mean \(g(h=1)\), mean \(d(h=1)\), winner attribution (§2.4).

---

## 5. Consecutive-date block bootstrap (reproducible)

### 5.1 Calendar and blocks

- **Entry-date calendar** \(D=(d_1,\ldots,d_T)\): distinct development `trade_date`s that contain ≥1 analysis-eligible trade for the panel, sorted ascending. (\(T\) is fixed from the original sample.)
- **Block length** \(L=4\) consecutive entry dates.
- **Overlapping moving blocks:** starts \(s=1,\ldots,T-L+1\); block \(B_s=\{d_s,\ldots,d_{s+L-1}\}\).
- Each sampled date contributes its **full** analysis-eligible cross-section (all tickers on that date that are in the measurement’s analysis set).

### 5.2 Path construction

Freeze:

| Item | Value |
|---|---|
| Replications | **10,000** |
| RNG seed | **`20260907`** (NumPy Generator default bit generator; document in manifest) |
| Sampling | Draw block **starts** independently with replacement from \(\{1,\ldots,T-L+1\}\) |
| Concatenation | Append blocks in draw order until the concatenated date multiset has length \(\ge T\) |
| Truncation | Keep the first \(T\) date occurrences (with replacement structure preserved); drop any overflow |
| Within-path trades | Union of full cross-sections for those \(T\) (possibly repeated) dates |

**Preserve during resampling:** original scenario quantities \(q_i(h)\), entry-time measurement values (including M3 as of entry), frozen quintile labels, and crossed-quote / eligibility flags. Do **not** recompute M3 or re-form quintiles inside the bootstrap.

### 5.3 Bootstrap statistics

On each valid path, recompute \(\Delta_m\) using frozen Q1/Q5 memberships restricted to trades present on the path (same labels). Optionally recompute pooled Spearman as a diagnostic.

**Invalid path statistic** (exclude from percentile endpoints; count toward validity disclosure):

- Q1 or Q5 empty on the path;
- fewer than 2 trades in either Q1 or Q5;
- non-finite \(\Delta_m\).

Report: `n_reps=10000`, `n_valid_Δ`, `n_invalid_Δ` by reason. **Do not** silently change block length, seed, or quantiles after seeing results.

### 5.4 Interval construction

From the empirical distribution of **valid** path \(\Delta_m\) values, take Bonferroni percentile endpoints (§3.2). If `n_valid_Δ < 1000`, mark the adjusted interval **undefined** for gating (classification → `inconclusive` on uncertainty).

---

## 6. Classification (exhaustive decision table)

Evaluate each measurement on development data under \(h=1\). Exactly one label.

### 6.1 Predicate definitions (deterministic)

| ID | Predicate | True when |
|---|---|---|
| **P-cov** | Coverage OK | Each of Q1…Q5 has \(n_g \ge \max(100,\,0.05\,n_{\mathrm{anal}})\); Q1 and Q5 each span ≥20 distinct entry dates |
| **P-Δ-def** | \(\Delta\) defined | Q1 and Q5 non-empty; point \(\Delta_m\) finite |
| **P-econ** | Economic bar | \(\Delta_m \ge 0.05\) |
| **P-stat** | Adjusted statistical support | Bonferroni bootstrap lower bound for \(\Delta_m\) is defined and \(> 0\) |
| **P-sign** | Direction | Point \(\Delta_m > 0\) |
| **P-half** | Development-half stability | Using **frozen** quintile labels: in Dev-A (`2020-01-01`→`2021-12-31`) and Dev-B (`2022-01-01`→`2023-12-31`), both halves have finite \(\Delta\) and \(\Delta>0\) (group means on trades whose `trade_date` falls in the half). **Do not** require significant Spearman in either half |
| **P-wd** | Within-date robustness | Among development dates with ≥2 analysis-eligible names, finite \(m\), and **non-constant** \(m\) within the date: (median within-date Spearman \(\rho(m,r)\le 0\)) **OR** (≥50% of such dates have \(\rho<0\)). Dates with constant \(m\) or <2 names are excluded from the denominator and disclosed |
| **P-gross** | Gross-edge (M1, M2 only; N/A for M3) | See §6.2 |
| **P-wrong** | Wrong-signed / negligible | \(\Delta_m \le 0\) **or** (\(\Delta_m < 0.02\) **and** adjusted interval does not have lower bound \(>0\)) |

### 6.2 Gross-edge criterion (M1 and M2 only)

**Group:** frozen **Q1** (best / lowest-score group) on the development analysis set under \(h=1\).

**Formulas** (only trades with \(C_i(1)>0\)):

\[
\overline{g}_{\mathrm{Q1}} = \mathrm{mean}_{i\in\mathrm{Q1}} g_i(1),\quad
\overline{g}_{\mathrm{all}} = \mathrm{mean}_{i\in\mathrm{anal}} g_i(1)
\]

\[
G^{+}_{\mathrm{Q1}} = \sum_{i\in\mathrm{Q1}} \mathbf{1}\{g_i(1)>0\}\,q_i(1)\,(X_i-M_i),\quad
G^{+}_{\mathrm{all}} = \sum_{i\in\mathrm{anal}} \mathbf{1}\{g_i(1)>0\}\,q_i(1)\,(X_i-M_i)
\]

**Pass (`P-gross`)** iff **both**:

1. \(\overline{g}_{\mathrm{all}}\) is finite and \(\overline{g}_{\mathrm{Q1}} \ge 0.50\times \overline{g}_{\mathrm{all}}\) when \(\overline{g}_{\mathrm{all}}>0\); **or**, when \(\overline{g}_{\mathrm{all}}\le 0\), \(\overline{g}_{\mathrm{Q1}} \ge \overline{g}_{\mathrm{all}}\) (Q1 no worse than the sample mean in a non-positive gross environment);
2. \(G^{+}_{\mathrm{all}}>0\) and \(G^{+}_{\mathrm{Q1}} / G^{+}_{\mathrm{all}} \ge 0.15\); if \(G^{+}_{\mathrm{all}}=0\), criterion (2) **fails** (no positive gross-profit mass to retain).

Zero / non-finite \(C\): excluded from means and sums (already invalid for association). Empty Q1 → `P-gross` false.

**M3:** `P-gross` is **not applicable** (treat as passed for the decision table).

### 6.3 Ordered decision table

Apply **first matching row** (top to bottom). Every measurement gets exactly one label.

| # | Condition | Label |
|---|---|---|
| 1 | Not `P-cov` **or** not `P-Δ-def` **or** adjusted interval undefined (`n_valid_Δ<1000`) | `inconclusive` |
| 2 | `P-wrong` | `unsupported` |
| 3 | `P-econ` **and** `P-stat` **and** `P-sign` **and** `P-half` **and** `P-wd` **and** `P-gross` | `supported` |
| 4 | `P-econ` **and** `P-stat` **and** `P-sign` **and** `P-half` **and** `P-gross`, but not `P-wd` | `inconclusive` |
| 5 | `P-econ` **and** `P-stat` **and** `P-sign` **and** `P-wd` **and** `P-gross`, but not `P-half` | `inconclusive` |
| 6 | `P-econ` **and** `P-stat` **and** `P-sign` **and** `P-half` **and** `P-wd`, but not `P-gross` (M1/M2) | `inconclusive` |
| 7 | `P-sign` **and** `P-stat` **and** not `P-econ` **and** \(\Delta_m \ge 0.02\) | `inconclusive` |
| 8 | `P-econ` **and** `P-sign` **and** not `P-stat` | `inconclusive` |
| 9 | Supported-style group separation (`P-econ` **and** `P-sign` **and** `P-half`) with weak/non-significant pooled Spearman | Still eligible for row 3 if `P-stat`/`P-wd`/`P-gross` hold — **Spearman does not block** |
| 10 | Otherwise (including flat/wrong-signed without meeting row 2’s `P-wrong` already) | `unsupported` if \(\Delta_m\le 0\) or adjusted upper bound \(<0\); else `inconclusive` |

Row 9 is a clarification, not a separate exit: weak Spearman never overrides rows 1–8.

**Only `supported` permits D2 candidacy for that measurement.**

---

## 7. Sprint-level gate for D2

| Gate | Rule |
|---|---|
| **Authorize D2** | ≥1 measurement labeled `supported` under §6 (implies that measurement’s Bonferroni lower bound for \(\Delta>0\) and \(\Delta\ge 0.05\)) |
| **Stop (no thresholds)** | Zero `supported` measurements — D2 is a short stop record only |
| **Evaluation period** | **Closed** in D1; no eval outcomes in outputs |

If multiple measurements are `supported`, rank for D2 candidacy by (1) point \(\Delta_m\), (2) within-date pass margin (fraction of dates with \(\rho<0\)), (3) `G^{+}_{\mathrm{Q1}}/G^{+}_{\mathrm{all}}\) for M1/M2 — **without selecting a cutoff**.

---

## 8. Sensitivity and disclosure (development only; non-decision)

Report but **do not** use for classification:

- \(\Delta_m\) and group means under \(h\in\{0,0.25,0.50\}\) with **frozen** quintile memberships and original quantities;
- Descriptive ordinary 95% bootstrap intervals for \(\Delta_m\) and Spearman;
- Validity counts for bootstrap paths.

**Forbidden in D1 outputs:**

- Any table or replay that uses `trade_date` ≥ `2024-01-01` for association, sensitivity, or classification;
- Full-history companion tables that extend into the evaluation period.

Defer evaluation-period and post-freeze full-path descriptive work until the measurement/threshold rule is frozen for evaluation (D2+).

---

## 9. Minimal implementation footprint (only after design acceptance)

```
notebooks/sprint008/d1_measurement_validation.ipynb
src/backtest/sprint008_d1_measurement_validation.py
tests/unit/test_sprint008_d1_measurement_validation.py
```

**Reuse:** D0 readiness helper for panel construction, equal-dollar quantities, crossed-quote flags, M1–M3; Sprint 007 artifact identity as needed. No second economic engine; no threshold search code in D1.

**Proposed helper surface:**

- Build development analysis panel (2020–2023 only for association rows).
- Compute \(C\), \(r\), \(g\), \(d\), dollar reconciliations.
- Freeze quintiles; compute \(\Delta_m\), winner attribution, within-date summary, half-period \(\Delta\).
- Overlapping moving-block bootstrap (seed `20260907`, 10,000 reps); Bonferroni intervals.
- Ordered classification + sprint gate JSON.

**Focused tests (synthetic):** quintile freeze/reuse; Bonferroni quantiles; gross/drag identity; `P-gross` edge cases (\(G^{+}_{\mathrm{all}}=0\), \(\overline{g}_{\mathrm{all}}\le 0\)); crossed-quote excluded from association; bootstrap preserves date cross-section and labels; decision-table exhaustiveness; no evaluation dates in decision path.

---

## 10. Planned evidence outputs (outside repo)

Under `C:/MomentumCVG_env/runs/sprint008_d1_<UTC>/` (after authorized execution):

| Artifact | Content |
|---|---|
| `d1_manifest.json` | SHA, command, pins, seed, family α, verdicts |
| `d1_measurement_labels.json` | Per-measurement label + predicate checklist |
| `d1_gate.json` | Sprint-level D2 authorize / stop |
| `d1_delta_bootstrap.json` | Point \(\Delta\), Bonferroni & descriptive 95% intervals, validity counts |
| `d1_spearman_diagnostic.json` | Supporting Spearman only |
| `d1_quintile_tables.parquet` | Group stats, \(g\)/\(d\)/\(r\), winner contributions |
| `d1_within_date_summary.json` | Within-date robustness |
| `d1_stability_halves.json` | Dev-A / Dev-B \(\Delta\) |
| `execution_receipt.json` | SHA, clean-tree state |

Committed notebook remains clean (narrative entrypoint).

---

## 11. Non-goals / stop rule

- No threshold grid, cutoff selection, or D2 portfolio comparison.
- No evaluation-period association, replay, or rule selection.
- No short-side / wings / signal-window search.
- No fill-attainability claims.
- No edits to official Sprint 006/007 artifacts.

**Stop** after this design is reviewed. Implementation starts only on explicit acceptance.

---

## 12. Summary for reviewers

**Primary gate:** Bonferroni-adjusted block-bootstrap interval for \(\Delta=\overline{r}_{\mathrm{Q1}}-\overline{r}_{\mathrm{Q5}}\) (family of three under \(h=1\)); lower bound \(>0\) plus point \(\Delta\ge 0.05\). Spearman is diagnostic only.

**Economics:** Gross midpoint P&L \((X-M)/C\) vs drag \((hH+\mathrm{fees})/C\); explicit Q1 gross-edge on means and positive gross-profit dollar share.

**Protocol:** Frozen quintiles; overlapping 4-date blocks; 10,000 reps; seed `20260907`; development-only outputs; exhaustive label table.

**Preserved:** D0 \(N\), crossed-quote cash policy, measurements, sizing, footprint.

**Provisional path:** Design acceptance → implement/execute D1 → evidence review → D2 design only if authorized.
