# Sprint 008 — closeout

**Status:** `D3 SUBMITTED — AWAITING FINAL REVIEW`  
**Submitted:** 2026-09-12  
**D2 reviewed through:** `c9b0a6a`  
**Design of record for D2:** [`docs/tmp/sprint008_d2_design.md`](../tmp/sprint008_d2_design.md) — accepted, reviewed commit `c2ba972`

Sprint 006 remains **accepted and unchanged**. Sprint 007 remains **closed** with `EXECUTION_CALIBRATION_REQUIRED`. This memo does not replace those conclusions, retune `42:8`, or authorize a live filter.

This closeout uses committed evidence only. No experiment was rerun.

---

## 1. Verdict

**Central question:** Can entry-time cost measurements improve the long book while retaining valuable winners?

**Answer:** Not as an income-generating strategy. Original D1 did not support a threshold search. An accepted later amendment evaluated two already specified exclusion rules on the later window. M1 (`H/M`, drop the within-date highest-score group) showed statistically supported **relative** improvement there and kept most winning profit, including all ten largest winning trades. That book was still unprofitable over the full later window. M2 (`H/S_0`, the same rule) remained inconclusive. Neither result promotes a production filter or an optimal cutoff.

| Conclusion | Result |
|---|---|
| **D0–D2 evidence** | Accepted. D2 accepted through `c9b0a6a` |
| **Historical D1 gate** | `STOP_NO_THRESHOLDS` unchanged (M1/M2 `inconclusive`; M3 `unsupported`) |
| **D2 amendment** | Replaced threshold search with one retrospective evaluation of the unchanged M1 and M2 exclude-U rules |
| **D2 labels** | M1 `relative_benefit`; M2 `inconclusive` |
| **Income-generating strategy** | **Not established** |
| **This memo** | Submitted for final review. Not yet reviewer-accepted |

---

## 2. What the sprint did

D0 froze the equal-dollar long-only protocol and confirmed input readiness (`READY_WITH_NARROW_ENABLING_CHANGE`, crossed-quote policy `sprint008_d0_crossed_quote_v1`).

D1 tested whether entry-time measurements distinguish equal-dollar net profitability. M1 and M2 were `inconclusive`. M3 was `unsupported`. The gate **`STOP_NO_THRESHOLDS`** remains the historical D1 decision. This closeout does not rewrite it.

Two reviewed D1 follow-ups then examined the measurements without opening a threshold search. The within-date comparison contrasted the lowest and highest score groups. The cost-diagnosis follow-up, corrected in `870d4b7`, applied exactly two fixed exclusions: drop the highest-score group on M1, and separately on M2, using \(k=\lfloor n/5\rfloor\). Those follow-ups did not complete planned D2.

The accepted D2 amendment (design `c2ba972`; evidence reviewed through `c9b0a6a`) **replaced** threshold selection with one retrospective evaluation of those same unchanged rules on `2024-01-01` through `2026-07-10`. It is not a pristine holdout. Earlier sprints had already inspected that history.

---

## 3. Comparison (periods not pooled)

Development figures are copied from the reviewed cost-diagnosis evidence (209 entry dates, `2020-01-01` through `2023-12-31`). Evaluation figures are copied from the accepted D2 evidence (132 entry dates). Development inference used family size 4 and 98.75% intervals. Evaluation inference used family size 2 and 97.5% intervals. Do not pool the series or treat the p-values as one test.

| Period | Rule | Baseline P&L | Filtered P&L | Incremental P&L | Mean weekly uplift | Adjusted inference | Winning-profit retention | Drawdown (baseline / filtered) |
|---|---|---:|---:|---:|---:|---|---:|---|
| Development | M1 | \$6,628.20 | \$20,823.08 | \$14,194.88 | 0.68 pp | inconclusive; adj. p 1.00 | 83.8% | \$-63,649.91 / \$-43,314.51 |
| Development | M2 | \$6,628.20 | \$25,088.18 | \$18,459.98 | 0.88 pp | inconclusive; adj. p 0.91 | 84.0% | \$-63,649.91 / \$-41,965.84 |
| Evaluation | M1 | \$-39,244.98 | \$-8,150.05 | \$31,094.93 | 2.36 pp | `relative_benefit`; 97.5% CI [0.98, 3.74] pp; adj. p 0.0003 | 87.0% | \$-53,311.48 / \$-37,901.86 |
| Evaluation | M2 | \$-39,244.98 | \$-28,194.16 | \$11,050.82 | 0.84 pp | `inconclusive`; 97.5% CI [−1.17, 2.85] pp; adj. p 0.693 | 83.2% | \$-53,311.48 / \$-46,184.30 |

**Dollar-profit retention is not winner-count retention.** Winning-profit retention is the share of baseline winning dollars kept. Winner-count retention is the share of winning trades kept. The reviewed development memo published dollar retention only. The accepted evaluation evidence publishes both:

| Evaluation rule | Winning-profit retention | Winner-count retention | Largest winners |
|---|---:|---:|---|
| M1 | 87.0% | 778/914 = 85.1% | all 10 of the 10 largest winning trades retained (100% of their profit) |
| M2 | 83.2% | 761/914 = 83.3% | 9 of 10 largest winning trades retained (86.7% of their profit) |

D2 anchors checked against [`sprint008_d2_evidence_review.md`](../tmp/sprint008_d2_evidence_review.md): window `2024-01-01` through `2026-07-10`; 132 dates (52 + 52 + 28 partial-2026); baseline \$-39,244.98; M1 filtered \$-8,150.05 and incremental \$31,094.93; M1 uplift 2.36 pp with adjusted interval [0.98, 3.74] pp and adjusted p 0.0003; M1 dollar retention 87.0% and all ten largest winning trades retained; M2 filtered \$-28,194.16 and `inconclusive`.

Only the 2024 M1 filtered slice was dollar-positive (\$5,969.72). The 2025 and partial-2026 filtered books still lost money. Those slices are descriptive. They are not extra tests.

---

## 4. Accounting

Each entry date has a fixed research budget \(B=\$10{,}000\). \(N\) is the original pre-filter capped count and is not reduced by exclusions. Each name is allocated \(B/N\). Quantity is \(q=(B/N)/C\) with full-cross cost \(C=M+H\) and modeled fees \(=0\). Excluded names, and the one evaluation crossed-quote name, keep that stake in cash at zero return. Retained quantities are not resized.

Reported P&L and drawdown are cumulative **fixed-budget dollar** results, one \$10,000 budget per entry date, with the running peak including the initial zero. They are not compounded account equity and not intraholding-period risk.

---

## 5. Limitations

- The evaluation window was inspected in earlier sprints. D2 is retrospective validation, not independent confirmation.
- There is no direct test of M1 against M2.
- The contrasts do not separate a selection benefit from the effect of holding more cash.
- Quote-based full-cross results do not establish attainable live fills.
- Relative improvement does not establish absolute profitability.
- Development and evaluation inference must not be pooled. They use different windows and different family sizes.

---

## 6. Practical implications

Keep M1 as a **candidate for future validation**. Do not promote it to production. Do not treat \(k=\lfloor n/5\rfloor\) as an optimal threshold. Historical `STOP_NO_THRESHOLDS` still records that D1 did not authorize threshold search.

Unresolved for later work, not studied here:

- Whether the M1 relative gain is selection of better names or simply less capital at risk.
- The remaining profitability gap: even the better filtered book lost \$8,150.05 over the later window under these assumptions.

The user intends **Sprint 009 to focus on the short side**. Scope is to be defined separately. This memo does not design or execute that sprint, and it does not revise Sprint 007’s `EXECUTION_CALIBRATION_REQUIRED` outcome.

---

## 7. Evidence chain

| Deliverable | Status | Evidence |
|---|---|---|
| D0 | Accepted | [`sprint008_d0_evidence_review.md`](../tmp/sprint008_d0_evidence_review.md); `C:/MomentumCVG_env/runs/sprint008_d0_20260907T204449Z/` |
| D1 | Closed; gate unchanged | [`sprint008_d1_evidence_review.md`](../tmp/sprint008_d1_evidence_review.md); `C:/MomentumCVG_env/runs/sprint008_d1_20260907T223037Z/` |
| Within-date follow-up | Reviewed | [`sprint008_d1_within_date_followup_evidence.md`](../tmp/sprint008_d1_within_date_followup_evidence.md) |
| Cost diagnosis | Reviewed (`870d4b7`) | [`sprint008_d1_cost_diagnosis_evidence.md`](../tmp/sprint008_d1_cost_diagnosis_evidence.md) |
| D2 | Reviewed through `c9b0a6a` | [`sprint008_d2_evidence_review.md`](../tmp/sprint008_d2_evidence_review.md); `C:/MomentumCVG_env/runs/sprint008_d2_20260912T232144Z/` |
| D3 | Submitted for final review | this memo |

Official Sprint 006 artifacts were not mutated.
