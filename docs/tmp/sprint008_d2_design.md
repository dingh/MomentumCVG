# Sprint 008 D2 — frozen-rule retrospective validation

**Status:** `DRAFT — AWAITING REVIEW`  
**Drafted:** 2026-09-12  
**Authorization:** Planning and documentation only. Not accepted. Do not implement or execute until review accepts this amendment.  
**Agenda:** [`docs/agenda/current_sprint.md`](../agenda/current_sprint.md)  
**Working plan:** [`docs/agenda/sprint8_long_filter_plan.md`](../agenda/sprint8_long_filter_plan.md)  
**D1 closeout:** [`sprint008_d1_evidence_review.md`](sprint008_d1_evidence_review.md) — reviewed in commit `48174db`  
**Parent follow-ups:** [`sprint008_d1_within_date_followup_protocol.md`](sprint008_d1_within_date_followup_protocol.md), [`sprint008_d1_cost_diagnosis_protocol.md`](sprint008_d1_cost_diagnosis_protocol.md), corrected evidence [`sprint008_d1_cost_diagnosis_evidence.md`](sprint008_d1_cost_diagnosis_evidence.md)

---

## Question

> Do the two existing exclusion rules improve the long book during the later period, and what valuable winners do they sacrifice?

**Required answer (after accepted implementation):** for each rule, a preregistered relative-inference label (`relative_benefit` / `relative_harm` / `inconclusive`), plus the economic, coverage, winner-sacrifice, and stability reports below. An inconclusive result completes D2.

---

## 1. Explicit amendment

This design is a **bounded amendment**. If accepted, it **replaces** the originally planned D2 threshold-selection study (cutoff grid on a `supported` measurement, then later-period evaluation) with **one frozen retrospective validation**.

| Preserved | Not done by this amendment |
|---|---|
| Historical D1 labels: M1/M2 `inconclusive`, M3 `unsupported` | Threshold search or new cutoffs |
| Historical gate **`STOP_NO_THRESHOLDS`** | Combining M1 and M2 |
| D1 development evidence and follow-up methods | Pooling development and evaluation for significance |
| D3 as the subsequent sprint closeout | Automatic next experiment after D2 |

The fraction excluded is the existing within-date rule \(k=\lfloor n/5\rfloor\). Numerical score boundaries may vary by date. That is not a new threshold search.

**Not a pristine holdout.** Evaluation dates `2024-01-01` through `2026-07-10` were inspected in earlier sprints. Report this as retrospective validation of already frozen rules, not independent confirmation.

This design inspected code, schemas, and **already published development evidence only**. It did not inspect or calculate evaluation-period outcomes.

---

## 2. Exactly three books

Same evaluation calendar, same quantities.

| Book | Rule |
|---|---|
| Unfiltered baseline | Every executed `in_N` candidate; crossed-quote stakes cash |
| M1 exclude-U | Drop the highest-score group on M1 \(H/M\) when a valid within-date split exists |
| M2 exclude-U | Same on M2 \(H/S_0\), independently |

Do not combine filters. Do not add cutoffs. Dates without a valid split keep the baseline allocation.

### Grouping (reuse D1 exactly)

Entry-only, without using future returns. Existing functions: `scored_candidates_for_date`, `select_within_date_groups` (`MIN_SCORED = 5`, sort by score then ticker, `GROUP_FRACTION_DENOM = 5`).

For each measurement and evaluation entry date:

1. Start from `in_N` and `analysis_eligible` candidates with a finite entry-time score.
2. Exclude the date from additional filtering if fewer than five scored names, or scores have no variation, or \(k=\lfloor n_{\mathrm{scored}}/5\rfloor < 1\). Disclose reason.
3. Sort ascending by \((score, ticker)\), mergesort.
4. Highest-score group \(U\) is the last \(k\) names. Exclude those names only.
5. Disclose cutoff ties using the existing boundary-tie flags. Actual exclusion fraction is \(k/n_{\mathrm{scored}}\), not a fixed score.

---

## 3. Population and accounting (frozen)

| Pin | Value |
|---|---|
| Window | **`2024-01-01` through `2026-07-10` inclusive** (`PRIMARY_END`) |
| Universe | Frozen `42:8` long ATM straddles; weekly hold-to-expiry |
| Budget | \(B=\$10{,}000\) per entry date |
| \(N\) | Original capped pre-filter `in_N` count; unchanged by exclusions |
| Stake / cost / quantity | \(B/N\); \(C=M+H\); \(q=(B/N)/C\); fees \(=0\) (disclose) |
| Scenario | Full cross \(h=1\) only |
| Rejected capital | Cash, zero return; no redistribution; retained quantities unchanged |
| Crossed quotes | `sprint008_d0_crossed_quote_v1`: stay in \(N\); stake cash; not analysis-eligible |
| Calendar | Complete evaluation entry-date calendar, including cash-only dates |
| Input checks | Reuse D0 `_required_input_ok` via `enforce_d0_required_inputs` on the evaluation analysis path |
| Missing outcomes | `require_all_executed_outcomes`: any executed baseline trade missing a required outcome fails explicitly, including middle-group trades |

Economics identity (existing `attach_scenario_economics`): \(r=(X-C)/C\), \(p=(B/N)\,r\), invested stake \(q\,C=B/N\) when executed.

Panel construction may load the full artifact through `build_d1_base_panel` (that helper has no date filter). **Immediately** restrict every analysis frame, print, and export to the evaluation window. Assert development dates are absent from D2 result tables. Do not aggregate or print evaluation outcomes until implementation is accepted and the official run starts. M3 is unused.

---

## 4. Primary evaluation and inference

For each rule and evaluation entry date \(t\):

\[
R^{\mathrm{base}}_t=\sum_i p_i/B,\qquad
R^{\mathrm{filt}}_t=\sum_{i\notin U_t}p_i/B,\qquad
\mathrm{uplift}_t=R^{\mathrm{filt}}_t-R^{\mathrm{base}}_t
\]

On dates with no valid split, \(U_t\) is empty and \(\mathrm{uplift}_t=0\).

**Exactly two primary contrasts** (family size **2**):

1. Mean weekly uplift, M1 exclude-U  
2. Mean weekly uplift, M2 exclude-U  

One observation per evaluation entry date on the complete calendar (not only split dates). Do not pool development and evaluation series.

Reuse `newey_west_intercept_inference`: intercept-only, `maxlags=3`, Bartlett kernel, small-sample correction on, Student-\(t\) with \(T-1\) degrees of freedom. Lags index successive **evaluation entry dates**, not calendar days. Report calendar gaps.

| Item | Freeze |
|---|---|
| Ordinary interval | 95% |
| Adjusted \(p\) | \(\min(1,\,2\times p_{\mathrm{raw}})\) |
| Adjusted individual interval | **97.5%** (`1 - 0.05/2`) |
| Other summaries | Descriptive only; no extra significance gates |

Freeze these choices before evaluation output. Do not retune lags, fractions, or family membership after seeing results.

---

## 5. Required reports

Reuse `build_portfolio_comparison`, `fixed_budget_max_drawdown`, `weekly_uplift_distribution`, and `exclusion_window_metrics` (or thin wrappers). Do not rewrite the accounting.

For each rule:

- Baseline and filtered absolute dollar P&L, incremental P&L, and mean returns on original \(B\).
- Trade coverage, actual exclusion fraction \(k/n\), invested and cash fractions.
- Losses avoided and winning profits sacrificed. Verify incremental P&L = losses avoided − winning profits sacrificed.
- Winning-profit retention: \(\sum_{\mathrm{retained}}\max(p_i,0)/\sum_{\mathrm{baseline}}\max(p_i,0)\). NA if the denominator is zero.
- Top-5 and top-10 baseline winner-profit retention, same denominator rule. These measure retention of baseline winners, not concentration of incremental P&L.
- Cumulative fixed-budget dollar P&L and peak-to-trough drawdown. Running peak is \(\max(0,\text{cumulative P\&L so far})\). Not compounded equity and not intraholding-period risk.
- Weekly uplift: mean, median, standard deviation, fractions positive / zero / negative, five largest positive and five largest negative weeks (date and dollar contribution). Concentration versus total positive contributions and total absolute negative contributions separately. Do not quote shares of a small net improvement.

Descriptive calendar splits (no new tests):

| Slice | Dates | Label |
|---|---|---|
| 2024 | `2024-01-01`–`2024-12-31` | full year |
| 2025 | `2025-01-01`–`2025-12-31` | full year |
| 2026 | `2026-01-01`–`2026-07-10` | **partial year** (ends at the pinned evaluation boundary) |

Compare with already recorded development findings by **copying published figures** from the reviewed cost-diagnosis evidence. Keep periods in separate columns. Do not recompute a pooled development-plus-evaluation contrast.

---

## 6. Interpretation and completion

Apply this table to each rule’s **Bonferroni-adjusted 97.5% interval** for mean uplift. Do not invent another label.

| Adjusted interval | Label |
|---|---|
| Entirely above 0 | `relative_benefit` |
| Entirely below 0 | `relative_harm` |
| Includes 0 | `inconclusive` |

Report separately, without cutoffs:

- Economic magnitude of the point uplift (percentage points and dollars).
- Absolute profitability of baseline and filtered books (a relative gain on a losing book is not the same as a profitable book).
- Consistency across 2024, 2025, and partial 2026.
- Winning profits sacrificed and top-5/top-10 retention.

Statistical significance does **not** automatically promote a filter. Do not set a winner-retention pass/fail threshold. An inconclusive outcome completes D2. No post-result cutoff tuning and no automatic next experiment. D3 remains the subsequent sprint closeout.

---

## 7. Implementation plan (after acceptance only)

Smallest footprint. No `SurfaceRunner`, no sizing or signal changes, no shorts.

| Path | Role |
|---|---|
| `src/backtest/sprint008_d2_fixed_exclusion_validation.py` | Evaluation-window filter, two-book exclusion, two-contrast HAC, reports |
| `scripts/run_sprint008_d2_fixed_exclusion_validation.py` | Official runner; timestamped evidence dir under `C:/MomentumCVG_env/runs/sprint008_d2_<UTC>/` |
| `notebooks/sprint008/d2_fixed_exclusion_validation.ipynb` | Entrypoint calling the runner helper only |
| `tests/unit/test_sprint008_d2_fixed_exclusion_validation.py` | Synthetic-frame tests |

**Reuse:** `build_d1_base_panel`, `attach_scenario_economics`, `enforce_d0_required_inputs`, `require_all_executed_outcomes`, `scored_candidates_for_date`, `select_within_date_groups`, `build_portfolio_comparison`, `fixed_budget_max_drawdown`, `weekly_uplift_distribution`, `exclusion_window_metrics`, `newey_west_intercept_inference`, `bonferroni_adjust_p`.

**Necessary changes:** evaluation-window filter (`2024-01-01`–`2026-07-10` inclusive); assert no development dates in D2 outputs; family size 2 and 97.5% intervals; calendar-year splits with partial-2026 label; interpretation table; progress prints (`build_panel`, `filter_eval`, `d0_checks`, `economics`, `M1`, `M2`, `inference`). Load the panel once.

**Evidence artifacts:** trade-level and date-level parquet/csv; measurement summaries; readable markdown with the tables in §5–§6; plots of cumulative P&L (leading zero) only; execution receipt (SHA, dirty/clean tree, command, versions, timings, protocol pins). Do not mutate Sprint 006/007 evidence.

**Focused tests (synthetic frames only; do not open official evaluation outcomes):**

- Period isolation: development rows absent from D2 analysis outputs; dates after `2026-07-10` excluded if present.
- Entry-only grouping unchanged when future outcomes are swapped; \(k=\lfloor n/5\rfloor\); no-split dates keep baseline.
- Original \(N\) and cash: crossed-quote and excluded-U stakes stay cash; retained \(q\) unchanged.
- Incremental P&L equals losses avoided minus winning profits sacrificed.
- Missing executed outcome raises, including a middle-group trade.
- Drawdown includes initial zero (losing path and recovery path).
- Inference family size 2; adjusted interval level 0.975; adjusted \(p=\min(1,2p)\).

After implementation is accepted, run those tests plus existing D0/D1/follow-up/cost-diagnosis tests, then one official evaluation run. Stop for evidence review. Do not start D3 in the same step.

---

## Unresolved design decisions

None that block review. The window end, grouping, family size, HAC settings, year boundaries, and interpretation table are frozen above, pending acceptance of this draft. If review rejects the amendment, D2 remains the unimplemented original threshold study and the historical stop stands.
