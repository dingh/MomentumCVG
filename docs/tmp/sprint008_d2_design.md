# Sprint 008 D2 — frozen-rule retrospective validation

**Status:** `ACCEPTED`
**Drafted:** 2026-09-12
**Revised:** 2026-09-12, after review of `269fde0`
**Accepted:** 2026-09-12, reviewed commit `c2ba972`
**Authorization:** Accepted amendment. Implementation and one official evaluation run are authorized. Do not start D3 in the same step.
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

This design is a **bounded amendment**. **Accepted 2026-09-12** (reviewed commit `c2ba972`). It **replaces** the originally planned D2 threshold-selection study with **one frozen retrospective validation** of the unchanged M1 and M2 exclude-U rules on `2024-01-01` through `2026-07-10`. It does not change historical D1 findings or `STOP_NO_THRESHOLDS`.

| Preserved | Not done by this amendment |
|---|---|
| Historical D1 labels: M1/M2 `inconclusive`, M3 `unsupported` | Threshold search or new cutoffs |
| Historical gate **`STOP_NO_THRESHOLDS`** | Combining M1 and M2 |
| D1 development evidence and follow-up methods | Pooling development and evaluation for significance |
| D3 as the subsequent sprint closeout | Automatic next experiment after D2 |

The fraction excluded is the existing within-date rule \(k=\lfloor n/5\rfloor\). Numerical score boundaries may vary by date. That is not a new threshold search.

**Not a pristine holdout.** Evaluation dates `2024-01-01` through `2026-07-10` were inspected in earlier sprints. Report this as retrospective validation of already frozen rules, not independent confirmation.

This design inspected code, artifact schemas (column names and file metadata only), and **already published development evidence**. It did not inspect or calculate evaluation-period outcomes, including date-level status counts or P&L.

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
| Calendar | Authoritative evaluation calendar in §3.1, including verified \(N=0\) dates |
| Input checks | Reuse D0 `_required_input_ok` via `enforce_d0_required_inputs` on the evaluation analysis path |
| Missing outcomes | `require_all_executed_outcomes`: any executed baseline trade missing a required outcome fails explicitly, including middle-group trades |

Economics identity (existing `attach_scenario_economics`): \(r=(X-C)/C\), \(p=(B/N)\,r\), invested stake \(q\,C=B/N\) when executed.

Panel construction may load the full artifact through `build_d1_base_panel` (that helper has no date filter). **Immediately** restrict every analysis frame, print, and export to the evaluation window. Assert development dates are absent from D2 result tables. Do not aggregate or print evaluation outcomes until implementation is accepted and the official run starts. M3 is unused.

### 3.1 Complete evaluation calendar

The long-trade panel cannot define the calendar. A legitimate zero-long date has no trade rows. An absent long row is not evidence that \(N=0\).

**Calendar authority.** Official run `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z/` (`OFFICIAL_RUN_DIR`).

| Item | Pin |
|---|---|
| File | `date_status_sprint006_baseline_v1_mid.parquet` |
| Columns | `trade_date`, `status`, `reason` (`surface_runner.DATE_STATUS_COLUMNS`) |
| Schema check (no outcomes) | File exists; those three columns; metadata row count 403. Cross twin has the same columns and metadata row count. |
| Window | Inclusive `2024-01-01` through `PRIMARY_END` (`2026-07-10`). Use the existing inclusive filter pattern in `filter_to_window`. Do not use `is_evaluation_date` as the D2 window: it is `>= 2024-01-01` with no end, and changing it would change the D1 firewall. |
| Uniqueness | `trade_date` unique, non-null, sorted. Duplicate or null date raises. |
| Status values | Only `traded`, `valid_no_trade`, `failed`. Any other value raises. |
| Cross check | Mid and cross `date_status` must match on evaluation `trade_date`, `status`, and `reason`. Disagreement raises. Design inspection did not compare those values. |
| Not used as P&L | `date_summary` engine P&L is not D2 accounting and not a substitute for \(N\). |

`date_status` alone cannot establish \(N=0\). `traded` means at least one included name of either side. `valid_no_trade` / `no_included_names` means nothing was included, not that no constructable long existed. `candidate_view` is one row per `trade_log` row, so a date with no long rows is simply absent there.

**Zero-long vs missing data.** Companion file `funnel_summary_sprint006_baseline_v1_mid.parquet`. Required columns are present in the official schema: `trade_date`, `date_status`, `date_reason`, `n_post_signal_long`, `n_constructable_long`, `n_included_long`, `n_included_short`. Join one-to-one to the calendar on `trade_date`. Reconcile to reconstructed `in_N` from `reconstruct_capped_long_n` (`MAX_NAMES = 25`). No new field.

| Condition | Action |
|---|---|
| Calendar date missing from funnel, or `date_status` / `date_reason` disagree | Raise |
| `status == failed`, reason `missing_features`, or any required long count is null | Raise. Do not emit a cash row |
| `n_constructable_long == 0` and `n_included_long == 0`, both non-null, and the reconstructed panel has no `structure_ok` and no `in_N` long on that date | Verified \(N=0\). If `status == traded`, also require `n_included_short > 0`; otherwise raise. If reason is `empty_signals`, also require `n_post_signal_long == 0` |
| `n_constructable_long > 0` | Has long candidates. Reconstructed `in_N` count must equal \(\min(n_{\mathrm{constructable\_long}}, \mathrm{MAX\_NAMES})\). No panel rows, or a count mismatch, raises. Do not treat as cash |
| `n_included_long > 0` while `n_constructable_long == 0` | Raise |
| Panel `in_N` rows whose date is not on the evaluation calendar | Raise |

Verified \(N=0\): baseline and filtered P&L are 0, uplift is 0, invested fraction is 0, cash fraction is 1, \(N=0\). Dates with candidates but no valid score split keep the baseline allocation (uplift 0); that is not \(N=0\). Missing or failed required data raises. Both M1 and M2 portfolio frames must contain exactly this calendar before HAC.

---

## 4. Primary evaluation and inference

For each rule and evaluation entry date \(t\):

\[
R^{\mathrm{base}}_t=\sum_i p_i/B,\qquad
R^{\mathrm{filt}}_t=\sum_{i\notin U_t}p_i/B,\qquad
\mathrm{uplift}_t=R^{\mathrm{filt}}_t-R^{\mathrm{base}}_t
\]

On dates with no valid split, \(U_t\) is empty and \(\mathrm{uplift}_t=0\). Verified \(N=0\) dates also have \(\mathrm{uplift}_t=0\). Both primary series use the same complete evaluation calendar, including those zero-return dates.

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

Reuse the accounting identities inside `build_portfolio_comparison`, plus `fixed_budget_max_drawdown`, `weekly_uplift_distribution`, and `exclusion_window_metrics`. Do not call `build_portfolio_comparison` unchanged: it skips dates outside 2020–2023 and hardcodes development-half reconciliation. The smallest change is specified in §7. Do not rewrite the per-date stake, cash, uplift, or retention arithmetic, and do not change `DEV_*` constants.

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
| `src/backtest/sprint008_d1_cost_diagnosis.py` | Keyword-only window, calendar, and reporting-period args on `build_portfolio_comparison`. Defaults preserve D1 |
| `src/backtest/sprint008_d2_fixed_exclusion_validation.py` | Calendar classification, eval-window call, two-book exclusion, two-contrast HAC, reports |
| `scripts/run_sprint008_d2_fixed_exclusion_validation.py` | Official runner; timestamped evidence dir under `C:/MomentumCVG_env/runs/sprint008_d2_<UTC>/` |
| `notebooks/sprint008/d2_fixed_exclusion_validation.ipynb` | Entrypoint calling the runner helper only |
| `tests/unit/test_sprint008_d2_fixed_exclusion_validation.py` | Synthetic-frame tests, including calendar cases |
| `tests/unit/test_sprint008_d1_cost_diagnosis.py` | Default-path regression only: D1 window and half-period keys unchanged |

**Reuse unchanged:** `build_d1_base_panel`, `attach_scenario_economics`, `enforce_d0_required_inputs`, `require_all_executed_outcomes`, `scored_candidates_for_date`, `select_within_date_groups`, `fixed_budget_max_drawdown`, `weekly_uplift_distribution`, `exclusion_window_metrics`, `newey_west_intercept_inference`, `bonferroni_adjust_p`, `reconstruct_capped_long_n`, `filter_to_window` (calendar restriction only). Do not change their default formulas.

**Smallest engine change.** In `src/backtest/sprint008_d1_cost_diagnosis.py`, extend `build_portfolio_comparison` with keyword-only arguments. Positional callers stay valid. Defaults must reproduce today's development loop, `half_periods` keys `2020-2021` and `2022-2023`, and `half_period_reconcile_ok` (tolerance \(0.05\)). Do not change `DEV_START` / `DEV_END` / `DEV_A_*` / `DEV_B_*`, and do not change cost-diagnosis `FAMILY_SIZE`.

```text
build_portfolio_comparison(
    trade_level, paired, measurement, *,
    window_start: date = DEV_START,
    window_end: date = DEV_END,
    entry_calendar: pd.DataFrame | None = None,
    reporting_periods: dict[str, tuple[date, date]] | None = None,
)
```

- `entry_calendar is None` and `reporting_periods is None`: current behavior. Iterate trade-panel dates inside the development window. Do not invent cash dates.
- D2 call: `window_start=2024-01-01`, `window_end=PRIMARY_END`, `entry_calendar` required, reporting periods `2024` (`2024-01-01`–`2024-12-31`), `2025` (`2025-01-01`–`2025-12-31`), `2026_partial` (`2026-01-01`–`2026-07-10`, label **partial year**).
- When `entry_calendar` is set, iterate that calendar inside the window, not `groupby` of the trade panel. Required columns: `trade_date` (unique), `n_in_N`, `long_book_class` in `{verified_zero_long, has_long_candidates}`. `verified_zero_long` iff `n_in_N == 0`. A verified-zero date with any `structure_ok` or `in_N` panel row raises. A `has_long_candidates` date with no rows, or a row count other than `n_in_N`, raises. In-window panel dates missing from the calendar raise. Out-of-window trades are not aggregated.
- When `reporting_periods` is set, write `reporting_periods` and `period_reconcile_ok` instead of development halves. Sum of period `pnl_baseline`, `pnl_filtered`, `losses_avoided`, and `winning_profits_sacrificed` must match the full-window totals (same \(0.05\) tolerance). Reuse `exclusion_window_metrics` for each period. Attach the partial-year label in the D2 writer; do not change `exclusion_window_metrics`.
- Keep the existing per-date identities (uplift \(=-\sum_U p/B\), gross−drag, cash fraction, retention, drawdown including the initial zero).

**D2-only calendar helper** in `sprint008_d2_fixed_exclusion_validation.py`, not in the D1 module: `load_d2_evaluation_calendar(run_dir) -> pd.DataFrame`. It applies §3.1 and returns the classified calendar. Portfolio construction consumes that frame. It must not read evaluation P&L to classify dates.

**Other necessary changes:** assert no development dates in D2 outputs; family size 2 and 97.5% intervals; both uplift series equal the calendar length; interpretation table; progress prints (`build_panel`, `calendar`, `filter_eval`, `d0_checks`, `economics`, `M1`, `M2`, `inference`). Load the panel once.

**Evidence artifacts:** trade-level and date-level parquet/csv; measurement summaries; readable markdown with the tables in §5–§6; plots of cumulative P&L (leading zero) only; execution receipt (SHA, dirty/clean tree, command, versions, timings, protocol pins). Do not mutate Sprint 006/007 evidence.

**Focused tests (synthetic frames only; do not open official evaluation outcomes).** Specifications only until this draft is accepted.

Portfolio window (`build_portfolio_comparison` defaults and explicit args):

- An evaluation date inside `2024-01-01`–`2026-07-10` is retained when the D2 window and calendar are passed.
- Dates before `2024-01-01` and after `2026-07-10` are excluded from the D2 portfolio frame.
- Dollar totals for 2024 + 2025 + partial 2026 reconcile to the full evaluation totals for baseline P&L, filtered P&L, losses avoided, and winning profits sacrificed.
- A default call (no new kwargs) still uses the development window and half-period reconcile. A synthetic development frame does not include evaluation dates, and still emits `half_periods` / `half_period_reconcile_ok`. Pin this in `tests/unit/test_sprint008_d1_cost_diagnosis.py` so D1 behavior stays unchanged.

Calendar (synthetic date_status, funnel counts, and a long panel):

- A calendar date absent from the long panel, with verified \(N=0\) counts, produces a portfolio row: P&L 0, uplift 0, cash fraction 1. Both measurement series contain that date.
- A verified \(N=0\) date (`n_constructable_long == 0`, `n_included_long == 0`, no `structure_ok` rows) does the same.
- These must raise, not become cash: `status == failed` or null funnel long counts; `n_constructable_long > 0` with no panel rows; panel `in_N` rows whose date is absent from the calendar; funnel / `date_status` disagreement.

Other:

- Period isolation: development rows absent from D2 analysis outputs.
- Entry-only grouping unchanged when future outcomes are swapped; \(k=\lfloor n/5\rfloor\); no-split dates keep baseline.
- Original \(N\) and cash: crossed-quote and excluded-U stakes stay cash; retained \(q\) unchanged.
- Incremental P&L equals losses avoided minus winning profits sacrificed.
- Missing executed outcome raises, including a middle-group trade.
- Drawdown includes initial zero (losing path and recovery path).
- Inference family size 2; adjusted interval level 0.975; adjusted \(p=\min(1,2p)\); both uplift series have one row per calendar date.

After implementation is accepted, run those tests plus existing D0/D1/follow-up/cost-diagnosis tests, then one official evaluation run. Stop for evidence review. Do not start D3 in the same step.

---

## Unresolved design decisions

No unresolved input-source gap. `date_status` is the calendar; funnel long counts plus reconstructed `in_N` distinguish verified \(N=0\) from missing or failed data. Disagreement raises. No field is invented.

Window end, grouping, family size, HAC settings, year boundaries, interpretation labels, and the calendar contract above are frozen, pending acceptance of this draft. If review rejects the amendment, D2 remains the unimplemented original threshold study and the historical stop stands.
