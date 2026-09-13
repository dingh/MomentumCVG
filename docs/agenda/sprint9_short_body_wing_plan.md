# Sprint 009 — Short-body economics, execution costs, protection, and conditional entry filtering

**Status:** `DRAFT — AWAITING REVIEW`  
**Updated:** 2026-09-12  
**Mode:** Audit. Planning only. Implementation has not started.  
**Agenda:** [`docs/agenda/current_sprint.md`](current_sprint.md)  
**Canonical path:** `docs/agenda/sprint9_short_body_wing_plan.md` — do not duplicate under `docs/tmp/`.  
**Prior closeouts:** [`docs/sprint_memos/008_closeout.md`](../sprint_memos/008_closeout.md) (accepted through `61cbf30`), [`docs/sprint_memos/007_closeout.md`](../sprint_memos/007_closeout.md), [`docs/sprint_memos/006_closeout.md`](../sprint_memos/006_closeout.md)  
**Frozen contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) — immutable; not edited by this sprint  

This document is the sprint-level research protocol. It is not accepted and does not authorize D0 implementation, a backtest, or a new economic run. Deliverable-specific notebooks, runners, and evidence files are created only after the relevant step is accepted. Do not add empty implementation or evidence files now.

---

## 1. Summary

| Item | Sprint 009 proposal |
|---|---|
| **Central question** | Where does the selected short book lose its economic edge, and can better trade selection improve it while accounting for the value of protection? |
| **Theme** | Short-body economics, execution costs, protection value, and conditional entry filtering |
| **Population** | Official short iron-fly candidates and strikes from the frozen `42:8` book. Conclusions are conditional on that population, including its wing-availability restrictions |
| **Reference book** | Official cross book. Quantities stay fixed. A midpoint repricing at those quantities is a diagnostic, not the separately sized official midpoint run |
| **Structures in scope** | Current iron fly (`wing_delta_target = 0.15`, `_choose_below_nearest`) and a body-only counterfactual that drops wings without adding names or resizing |
| **Windows** | Development `2020-01-01` through `2023-12-31` for D1–D3 inspection and any rule freeze. Later period `2024-01-01` through `2026-07-10` only after a freeze or an explicit stop. Retrospective evaluation, not an untouched holdout |
| **Not the goal** | Force profitability; remove wings; search new wing strikes; promote a production filter; replace Sprint 007’s execution-observation handoff |

An inconclusive measurement, a decision not to freeze a rule, or a skipped D4 is a valid completion.

---

## 2. What is authoritative

| Source | Role |
|---|---|
| [`docs/agenda/current_sprint.md`](current_sprint.md) | Active sprint status |
| This plan | Proposed Sprint 009 protocol, pending review |
| [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) and official run `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` | Frozen selection, structures, and the reference cross book |
| [`docs/sprint_memos/006_closeout.md`](../sprint_memos/006_closeout.md) | Accepted cross economics. Not revised here |
| [`docs/sprint_memos/007_closeout.md`](../sprint_memos/007_closeout.md) | `EXECUTION_CALIBRATION_REQUIRED` remains. This sprint does not implement the observer and does not cancel that handoff |
| [`docs/sprint_memos/008_closeout.md`](../sprint_memos/008_closeout.md) | Long-side filter closed through `61cbf30`. `STOP_NO_THRESHOLDS` unchanged. M1 is a candidate only, not a production rule |
| [`docs/known_bugs.md`](../known_bugs.md) KB-001 | Iron-condor `body_credit_per_share` bug. Out of scope. Do not route this sprint through the condor builder |
| Sprint 007 D0/D2 designs | Artifact keys, leg-log fields, and the statement that a body/wing price split is **not** a wingless P&L |

**Not current authority**

- [`docs/development_workflow.md`](../development_workflow.md) roadmap table for Sprints 006–008. It still says Sprint 006 was not started. That table is a stale sketch.
- [`docs/archive/ironfly_body_pnl_decomposition_framework.md`](../archive/ironfly_body_pnl_decomposition_framework.md). Historical notes only. Do not import its formulas, region labels, or “move the wings” recommendations.
- Sprint 008’s equal-dollar \(B=\$10{,}000\) long book. Different accounting. Do not reuse it as the short-book denominator.

---

## 3. Relationship to prior sprints

Sprint 006 rejected/deferred the frozen `42:8` hypothesis on the official cross book. Short iron-fly losses dominate that accepted cross result (primary-window short `pnl_total` \(-146{,}279.85\) on 3,322 included short rows). That figure is a reconciliation anchor for the full primary window. It is not a development-window result and must not be copied into D1 as if it were.

Sprint 007 found gross midpoint margin, entry-price concession as the dominant mid-to-cross gap, and unknown package-fill attainability. Expensive-package concentration and short-side fragility were secondary. Neither validates a filter or a wingless book.

Sprint 008 did not establish an income-generating long filter. Its chronological split and the distinction between dollar-profit retention and winner-count retention are reused as method, not as a short-side result.

This sprint does not reinterpret those conclusions, mutate official run directories, edit the frozen contract, or claim that quote crosses are attainable live fills.

---

## 4. Frozen comparison and accounting

### 4.1 Identity and population

- Selection, calendar, hold-to-expiry, and intrinsic settlement stay the frozen `42:8` / CVG / liquidity / `max_names_per_side=25` contract.
- Wing rule stays `wing_delta_target = 0.15` with `_choose_below_nearest`. No alternative strike or delta search.
- Start from the same accepted short candidates and strikes. Reference quantities are the official **cross** portfolio quantities. Freeze those quantities across body/wing comparisons and across midpoint versus cross diagnostics.
- Reconcile directly to the official cross short-book result for the window under study.
- A midpoint P&L computed at those frozen cross quantities is a diagnostic. It is not `trade_log_mid`, which was sized separately. Do not report the official midpoint run as the “same trades without spread.”
- Removing wings must not add names, change selection, or resize retained positions. Names that failed wing liquidity and never entered the official short book stay out. Body-only results are conditional on the iron-fly-selected population.

### 4.2 Leg roles and cash signs

Official iron-fly leg order, to be **verified** in D0 rather than assumed if a row disagrees:

| `leg_index` | Role | `unit_quantity` |
|---|---|---|
| 0 | long OTM put wing | \(+1\) |
| 1 | short ATM put body | \(-1\) |
| 2 | short ATM call body | \(-1\) |
| 3 | long OTM call wing | \(+1\) |

\(Q\) is the official cross quantity magnitude on that short trade. Portfolio quantity on a leg is \(Q \times\) `unit_quantity`.

Entry cash follows the official leg log. Buys have positive entry cash (debit). Sells have negative entry cash (credit). Expiry payoff is intrinsic times `unit_quantity`, then scaled by \(Q\).

### 4.3 Spread concession, without double counting

| Role | Concession, per share | Dollar concession |
|---|---|---|
| Sold body leg | midpoint minus bid | \(Q \times (\mathrm{mid}-\mathrm{bid})\) |
| Purchased wing | ask minus midpoint | \(Q \times (\mathrm{ask}-\mathrm{mid})\) |

That concession is the cash difference between the cross fill and the midpoint fill. Official cross P&L already embeds the cross fill. Attribution splits that embedded cost. It does not subtract spread once in the fill and again as a fee.

Fees in the official book are unmodeled. This sprint keeps **fees = 0**, stated explicitly. Do not invent a commission schedule. A later fee would be an additive term, not a rewrite of bid/ask concession.

### 4.4 Primary measure and forbidden denominators

Paired dollar P&L is the primary attribution measure.

Any ratio needs a named denominator, used the same way on every name in that comparison:

- Body midpoint credit \(C_{\mathrm{body}} = \mathrm{mid}_{\mathrm{call}} + \mathrm{mid}_{\mathrm{put}}\), in premium per share, or \(Q \times C_{\mathrm{body}}\) in dollars.
- Do **not** apply the iron fly’s finite `max_loss_per_share` or `return_on_max_loss` to the uncovered body. That cap exists only while the wings are held.
- A large wing spread percentage on a cheap wing is not, by itself, large economic damage. Report dollars beside any percentage.

Cumulative results are sums of these fixed-quantity dollars across the authoritative date calendar. They are not compounded account equity, not View B compounded returns, and not a new equal-dollar book. Drawdown is peak-to-trough of that cumulative dollar series, with the peak including the initial zero.

### 4.5 Calendar and missing data

Use the official `date_status` calendar. Keep verified no-trade dates. A missing row is not cash and is not a dropped date. If a required quote, quantity, strike, or settlement field is missing, D0 records a blocker. Do not impute.

The official primary window has `n_valid_no_trade = 0`. That is a fact to confirm, not a license to omit dates.

---

## 5. Windows and firewall

| Window | Dates | Use |
|---|---|---|
| Development | `2020-01-01` through `2023-12-31` | D1 attribution, D2 protection, D3 measurements and the only place a rule may be frozen |
| Later period | `2024-01-01` through `2026-07-10` | D4 if a rule is frozen; otherwise unused for selection. 2026 is partial through `2026-07-10` |

D1–D3 economic inspection uses development history first. Later-period attribution or protection summaries are produced only after the D4 rule is frozen or D4 is recorded as skipped. That companion is descriptive. It must not choose wings versus no wings, and it must not retune the filter.

The later period was inspected in Sprints 006–008. Call it retrospective evaluation. It is not an untouched holdout and not independent confirmation.

Do not pool development and later-period inference. They are different windows and, if D4 runs, a different contrast family.

---

## 6. Deliverable map

| ID | Question | Continues when | Stop / skip |
|---|---|---|---|
| **D0** | Can we trust the body/wing comparison? | Matched short-book dataset reconciles to the official cross iron fly | Named identity or reconciliation blocker. No D1 |
| **D1** | Where does the short book lose economic margin? | Development decomposition reconciles | Reconciliation failure. No economic story from a broken identity |
| **D2** | What protection do the wings provide? | Development with/without-wings comparison is identified and labeled as a counterfactual | Same. A large wing cost does not by itself authorize wing removal |
| **D3** | Can entry measurements identify unattractive trades? | Exactly one predeclared rule meets the freeze rule | Inconclusive, both fail, or tie-break fails → D4 skipped |
| **D4** | Does the frozen rule improve later-period economics? | D3 froze exactly one rule | Skipped with the D3 reason. No substitute rule |
| **D5** | What does the evidence justify? | Always, including after a stop | Must not assume wing removal, a new wing rule, or a production filter |

No deliverable selects a signal window, a new wing, a size, or a live fill.

---

## 7. D0 — Can we trust the body/wing comparison?

**Question.** Can accepted artifacts support a matched trade-level body/wing dataset that reproduces the official short iron fly?

**Inputs.** Official run `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` and `run_receipt.json`. Expected files already used in Sprint 007: `trade_log_cross`, `trade_log_mid`, `leg_log_cross`, `leg_log_mid`, `date_status_*`, `decision_report.json`. Read-only. Sprint 007 D0 confirmed paired leg identity, quote identity, and settlement identity between fills. D0 here confirms those properties still hold for the short iron-fly subset and that body plus wings add to the official short book.

**Bounded analysis.** Inventory the artifacts. Build one matched row per official included short iron fly, with four legs. Verify leg identity, option side, strike, expiry, quantity sign, premium sign, bid/ask/mid, fill price, and expiry settlement. Confirm cross quantities will be the reference. Confirm midpoint at those quantities is computable from quotes and is distinct from `trade_log_mid` quantities. Confirm the official date calendar, including any verified no-trade date. Do not open development-versus-later economic comparisons in D0.

**Footprint.** One small read-only helper under `src/backtest/` and focused tests. Reuse Sprint 007 artifact-validation patterns. Do not call `build_ironfly_from_surface` to reselect wings. No `SurfaceRunner` rerun unless a required field is absent. A missing field is a blocker and a plan amendment, not a silent repair.

**Evidence and checks.** Artifact inventory, key uniqueness, leg-role verification, quantity-sign checks, quote sanity (missing bid/ask, crossed quotes), settlement present, and

\[
\sum \text{leg P\&L} = \text{official cross trade P\&L} = \text{body dollars} + \text{wing dollars}
\]

within \(\max(\$0.01,\ 10^{-9}\times|\text{official}|)\). Primary-window short included count and short `pnl_total` must match the accepted closeout when the window is the full primary window.

**Done when.** A written readiness verdict is either `READY` or `BLOCKED` with a named gap. Implementation of D1 is not authorized by D0 drafting.

**Continuation.** `READY` allows a later D1 design/execution authorization. `BLOCKED` stops the sprint.

---

## 8. D1 — Where does the short book lose economic margin?

**Question.** On development history, how much of iron-fly P&L is body midpoint profit, body execution concession, the midpoint price of protection, the spread paid to buy the wings, and wing expiry payout?

**Inputs.** D0 matched dataset. Development dates only.

**Bounded analysis.** At frozen official cross quantities, for each included short iron fly:

| Term | Definition | Sign |
|---|---|---|
| Body midpoint profit | Body expiry payoff minus body midpoint entry cash, times \(Q\) | P&L |
| Body execution concession | \(Q\times(\mathrm{mid}-\mathrm{bid})\) on each sold body leg | Cost, ≥ 0 if the quote is normal |
| Wing midpoint premium | \(Q\times\mathrm{mid}\) on each purchased wing | Cost of protection, separate from spread |
| Wing execution concession | \(Q\times(\mathrm{ask}-\mathrm{mid})\) on each purchased wing | Spread paid to acquire protection |
| Wing expiry payout | Wing intrinsic payoff times \(Q\) | Gross protection, before paying for the wings |

Identity, which must be tested rather than assumed:

\[
P_{\mathrm{fly,cross}} = P_{\mathrm{body,mid}} - \mathrm{Conc}_{\mathrm{body}} + \mathrm{Pay}_{\mathrm{wing}} - \mathrm{Prem}_{\mathrm{wing,mid}} - \mathrm{Conc}_{\mathrm{wing}}
\]

Report dollar sums. Also report these ratios, each with its denominator named:

- body concession / body midpoint credit;
- wing midpoint premium / body midpoint credit (price of protection);
- wing concession / body midpoint credit (spread paid for protection, same denominator as the body concession).

Wing concession / wing midpoint premium may be shown only as a descriptive spread percentage, next to the dollar concession. It must not be the headline damage figure.

Body-only economics at midpoint and at cross use the same \(Q\) and the same body legs. Cross body-only is \(P_{\mathrm{body,mid}} - \mathrm{Conc}_{\mathrm{body}}\). Midpoint body-only is \(P_{\mathrm{body,mid}}\). Neither is the official midpoint run.

**Footprint.** One decomposition function plus tests for the identity, concession signs, and the ban on using max-loss as an uncovered-body denominator. No new structure search.

**Evidence and checks.** Development dollar table; ratio table; reconciliation to official cross short P&L on the development dates; count of undefined quotes. No later-period table in D1.

**Done when.** The identity holds or a blocker is recorded. The memo separates the price of protection from the spread paid to acquire it, and it separates body concession dollars from wing spread percentages.

**Continuation.** A successful decomposition continues to D2. It does not choose a filter or drop the wings.

---

## 9. D2 — What protection do the wings provide?

**Question.** Holding body quantities fixed, what do the wings change in development-history dollars, and what does that not measure?

**Inputs.** D1 terms on the development window. Same names, strikes, and \(Q\).

**Bounded analysis.** Compare each official iron fly with the same body and no wings.

- Wing purchase cost at the reference cross fill = wing midpoint premium + wing execution concession.
- Wing purchase cost at midpoint = wing midpoint premium only. Diagnostic, same \(Q\).
- Payouts forgone if wings are removed = wing expiry payout.
- Gross protection payout = wing expiry payout. It is not net value.
- Net wing contribution at cross = wing expiry payout − cross purchase cost.
- Absolute P&L with wings = official cross iron-fly P&L at \(Q\).
- Absolute P&L without wings = body-only cross P&L at the same \(Q\).

Also report, on development dates only:

- how often wing expiry payout is positive, and the distribution of those payouts in dollars;
- worst trades and worst entry dates with and without wings;
- loss concentration;
- fixed-quantity cumulative dollar P&L and drawdown for both books, peak including the initial zero.

**Payoff geometry, not a new path model.** Inside both wing strikes, wing expiry payoff is zero, so the fly is worse than the body by the purchase cost. Beyond a wing strike, the short body keeps losing one-for-one with spot, and the long wing’s intrinsic offsets that continuation. The fly’s further loss is bounded by wing width minus the net credit at the chosen fill. The uncovered body is not bounded that way. That statement is payoff geometry. It is not a measured intraholding path.

**Not measured unless the artifacts already contain the path.** Intraholding-period mark-to-market losses, margin calls, and liquidation risk are not in the official expiry settlement. If those paths are unavailable, say so. Do not claim they were measured. Do not invent a margin model in this sprint.

**Historical tails.** Payout frequency is the frequency of expiry spots in this sample. It is not a probability for unseen tails. A rare payout, or no payout beyond a strike, does not prove the wing is worthless.

**Footprint.** One comparison function on the D1 table, plus tests that dropping wings does not change \(Q\), strikes, or the name set.

**Evidence and checks.** Development comparison table with gross payout and net contribution in separate columns. Label the no-wing book as a counterfactual on the iron-fly-selected population.

**Done when.** Costs saved, payouts forgone, absolute P&L, frequency, concentration, and drawdown are reported with the path limitation explicit.

**Continuation.** D2 does not authorize wing removal. D3 still uses the iron fly, not the body-only book, as the filter population. Whether a later descriptive companion on 2024+ is worth filing is decided only after the D4 freeze or skip, and it still cannot select a structure.

---

## 10. D3 — Can entry measurements identify unattractive trades?

**Question.** On development history only, does one predeclared entry-only measurement support freezing a single exclusion rule?

**Inputs.** D0 quotes and official cross quantities. Profitability labels come from the official cross iron-fly P&L at those quantities. D1/D2 may explain the measurements. They must not add a third measurement or change a formula after results are seen.

These scores are entry accounting ratios. They are not expected-return forecasts. They do not use the exit spot, the expiry payoff, or a model of the future move.

### 10.1 Two measurements, frozen before analysis

Both use per-share quotes. Higher means less attractive. The planned rule drops the highest scores.

**M1 — body execution burden**

\[
C_{\mathrm{body}} = \mathrm{mid}_{\mathrm{call}} + \mathrm{mid}_{\mathrm{put}}
\]
\[
H_{\mathrm{body}} = (\mathrm{mid}_{\mathrm{call}}-\mathrm{bid}_{\mathrm{call}}) + (\mathrm{mid}_{\mathrm{put}}-\mathrm{bid}_{\mathrm{put}})
\]
\[
M1 = H_{\mathrm{body}} / C_{\mathrm{body}}
\]

Units: fraction of body midpoint credit given up by selling the body at the bid. Dimensionless.

**M2 — complete-package entry burden**

\[
D_{\mathrm{wing}} = \mathrm{mid}_{\mathrm{put\ wing}} + \mathrm{mid}_{\mathrm{call\ wing}}
\]
\[
H_{\mathrm{wing}} = (\mathrm{ask}_{\mathrm{put\ wing}}-\mathrm{mid}_{\mathrm{put\ wing}}) + (\mathrm{ask}_{\mathrm{call\ wing}}-\mathrm{mid}_{\mathrm{call\ wing}})
\]
\[
M2 = (D_{\mathrm{wing}} + H_{\mathrm{body}} + H_{\mathrm{wing}}) / C_{\mathrm{body}}
\]

Units: fraction of body midpoint credit consumed, at entry, by the midpoint price of the wings plus body and wing spread concessions. Dimensionless. The wing payout is intentionally absent. Including it would be a forecast, not an entry measurement.

**Undefined scores.** If any required quote is missing, the quote is crossed (`bid > ask`), or \(C_{\mathrm{body}} \le 0\), the score is undefined. Do not impute zero. Do not drop the name from the unfiltered book. An exclusion rule may omit only defined scores. Undefined names stay in both the filtered and unfiltered books and are counted.

### 10.2 Candidate rule family

Exactly two candidates. No combinations. No cutoff grid. No second expression invented after seeing ranks.

On each development date, for one measurement:

- Let \(n\) be the count of names with a defined score.
- If \(n < 5\), exclude nobody that date. The filtered book equals the unfiltered book. Record the date.
- Otherwise exclude the highest-score group, \(k=\lfloor n/5\rfloor\), sorting by score descending, then ticker ascending.
- Retained names keep official cross quantities.
- Excluded names contribute cash at zero return that date. They are not resized onto the survivors.

### 10.3 What “separation” means

Do not use a Pearson correlation p-value as the continuation criterion. Do not compute one as a gate.

Evaluate, on development dates only:

- economic separation: whether the excluded group has worse mean paired dollar P&L than the retained group, summarized at date level;
- uncertainty: HAC inference on the date-level dollar P&L difference, filtered minus unfiltered, including cash dates;
- a companion normalized difference, that dollar difference divided by the date’s official short `capital_at_risk_dollars`, reported beside the dollars and not used as a second family member;
- winner retention: dollar-profit retention and winner-count retention, reported separately, plus the ten largest winning trades;
- stability: sign of the mean date-level dollar difference in each development year. Year checks are not added to the Bonferroni family.

HAC, frozen: maxlags 3, Bartlett kernel, small-sample correction, Student-t with \(T-1\) degrees of freedom. Adjusted p = \(\min(1,\ 2\times\text{raw p})\). Adjusted interval is 97.5%. Family size stays 2 even if one measurement is undefined on every row.

A verified no-trade date stays in the dollar series at zero. It is omitted from the normalized companion only when the denominator is zero.

### 10.4 Freeze rule

Freeze at most one measurement and its exclude-highest-group expression. A candidate is eligible only if all of the following hold on development history:

1. Excluded-minus-retained date-level mean dollar P&L is negative (the dropped group is worse).
2. The adjusted HAC interval for mean date-level dollar uplift (filtered minus unfiltered) lies entirely above zero.
3. The normalized companion has a non-negative point estimate. Disagreement with the dollar sign blocks a freeze. It is not a second test to shop.
4. Winning-profit retention is at least 80% of development baseline winning dollars.
5. At least 8 of the 10 largest development winning trades are retained. Report the dollar share of those ten separately. The count is not the dollar percentage.
6. Mean date-level dollar uplift is positive in each of 2020, 2021, 2022, and 2023.

If both are eligible, freeze the one with the larger adjusted lower bound on the mean dollar uplift. If those bounds differ by less than \$1, freeze neither.

If none are eligible, record `STOP_NO_RULE`. That is a complete D3.

**Footprint.** Score function, exclusion function, and inference wrapper. Reuse Sprint 008 HAC and retention reporting patterns. New tests for undefined denominators, sort stability, \(k=\lfloor n/5\rfloor\), and the family-size lock. No threshold search.

**Evidence and checks.** Development score coverage, undefined counts, the two-candidate table, freeze or stop decision written before any later-period filter number is computed.

**Done when.** Exactly one rule is frozen, or `STOP_NO_RULE` is recorded with the failed predicates named.

**Continuation.** One frozen rule authorizes D4 design/execution later. `STOP_NO_RULE` skips D4. It does not authorize a new measurement.

---

## 11. D4 — Does the frozen rule improve later-period economics?

**Question.** On `2024-01-01` through `2026-07-10`, does the single frozen rule improve economics relative to the unfiltered book and relative to a same-exposure cash benchmark?

**Inputs.** The frozen D3 rule, unchanged. Official cross quantities. Later-period calendar. Not available if D3 did not freeze a rule.

**Two comparisons, both required.**

1. **Matching unfiltered expression.** Same names and original quantities. Excluded names are cash at zero. This shows incremental P&L, including the effect of holding more cash.
2. **Same-exposure benchmark.** Scale every unfiltered name’s official-cross dollar P&L by

\[
f_t = \frac{\mathrm{CAR}_{\mathrm{retained},t}}{\mathrm{CAR}_{\mathrm{unfiltered},t}}
\]

where CAR is official cross `capital_at_risk_dollars`. The filtered book is not scaled. It keeps original quantities. The benchmark shrinks the unfiltered mix so retained capital-at-risk matches the filter. It does not drop names. This CAR scaler is valid only because both sides are iron flies. Do not apply it to the uncovered body.

If unfiltered CAR is zero on a verified empty date, both dollar results are zero. Do not drop the date.

**Inference, frozen.** Date-level mean dollar difference. Same HAC settings as D3. Family size 2: contrast 1 and contrast 2. Adjusted interval 97.5%. Do not add a third contrast after seeing results. Year slices, including partial 2026, are descriptive.

**Report.** Absolute P&L, incremental P&L versus each comparison, losses avoided, winning profits sacrificed, dollar-profit retention, winner-count retention, retained exposure (CAR and name count), worst trades, worst dates, and fixed-quantity cumulative dollar drawdown. State whether either book is profitable. Relative improvement is not absolute profitability.

**Footprint.** One evaluation function. Tests that the rule text matches the D3 freeze record, that quantities are unscaled on the filtered book, and that \(f_t\) uses CAR rather than max-loss of a straddle.

**Done when.** Both comparisons are reported, or D4 is recorded skipped with the D3 reason. The rule is not revised after later-period results.

**After the decision is frozen.** One descriptive later-period attribution and protection summary may then be produced, clearly labeled retrospective and non-selecting. It is not a second D4 and not a reason to reopen D3.

---

## 12. D5 — What does the evidence justify?

**Question.** Given the accepted chain, what can be said about body economics, execution-cost attribution, protection cost and value, filtering, and remaining limitations, and what single follow-up is prioritized?

**Inputs.** Accepted D0–D4 evidence, including a skipped D4.

**Bounded analysis.** A closeout memo. It must cover:

- body economics at midpoint and cross, at frozen quantities;
- execution-cost attribution, with protection price separated from spread paid;
- protection cost, gross payout, and net contribution, plus the unmeasured path risks;
- filtering evidence, or the reason no rule was frozen;
- remaining profitability and implementation limits, including unknown live fills;
- one prioritized direction for later work.

That direction is not pre-filled. Wing removal, a new wing rule, and a production filter are allowed outcomes only if the evidence supports them. They are not the required outcome. Sprint 007’s `EXECUTION_CALIBRATION_REQUIRED` handoff remains unless the closeout explicitly says this evidence does not replace it — the default is that it does not.

**Footprint.** `docs/sprint_memos/009_closeout.md` when D5 is authorized. Not created in this planning step.

**Done when.** The memo answers the central question, including a stop, and does not treat quote results as attainable fills.

---

## 13. Dependencies and branches

```text
Plan accepted
    → D0
        blocked → stop
        ready → D1 (development attribution)
            identity fails → stop
            reconciles → D2 (development protection counterfactual)
                → D3 (development scores; freeze or STOP_NO_RULE)
                    STOP_NO_RULE → D4 skipped
                    one rule frozen → D4 (later period, rule unchanged)
                → descriptive later-period attribution/protection
                  only after freeze or skip; not a selector
                → D5
```

Choices that wait for evidence, and must not be answered in this plan:

- Whether D0 finds a missing field that blocks a no-rerun path.
- The development dollar split among the five D1 terms.
- Whether net wing contribution is positive.
- Whether either measurement meets the freeze predicates.
- Whether a frozen rule, if any, helps on the later period after the cash-matched comparison.
- What single follow-up D5 should prioritize.

---

## 14. Proposed implementation increments

These start only after plan acceptance and a separate authorization for that increment. Estimates are planning ranges, not commitments.

| Increment | Work | Rough effort | Depends on |
|---|---|---|---|
| 1 | D0 inventory, matched dataset, reconciliation tests | 1–2 days | Plan acceptance |
| 2 | D1 development decomposition and identity tests | about 1 day | D0 ready |
| 3 | D2 development protection comparison and tests | 1–2 days | D1 identity |
| 4 | D3 scores, two-rule family, freeze record, tests | about 2 days | D0 quotes; D1/D2 reviewed so formulas are not quietly edited |
| 5 | D4 later-period evaluation, only if a rule is frozen | 1–2 days | D3 freeze |
| 6 | D5 closeout | about 1 day | D0–D4 status, including skips |

If D4 is skipped, the remaining path is the descriptive companion plus the closeout. No increment reruns `SurfaceRunner` or retunes `42:8`.

Separate one-page designs are written when an increment is authorized, following the existing workflow. This plan is the review package. Do not create those designs, runners, or evidence directories now.

---

## 15. Out of scope

- Signal-window or universe expansion.
- Long-side changes, including promotion of Sprint 008 M1.
- Alternative wing strike or delta searches.
- Sizing or leverage optimization.
- Brokerage margin, live execution, paper-trading plumbing, or the Sprint 007 execution observer.
- Intraday hedging or a new exit policy.
- Iron condor, and any KB-001 fix, unless a later amendment says otherwise.
- Treating historical quote crosses as attainable fills.
- Mutating official Sprint 006/007/008 evidence directories or the frozen contract.

Data limits and narrow enabling work belong in the deliverable that finds them. This sprint does not expand to repair every operational gap.

---

## 16. Decisions required before kickoff

The reviewer should accept, reject, or amend these before any implementation:

1. Reference quantities are the official cross book, not the official midpoint book and not a new equal-dollar book.
2. Body-only is a counterfactual on the iron-fly-selected population, not a newly selected short-straddle strategy.
3. The five-term attribution and the concession definitions in §8, including fees = 0.
4. The two measurements, score direction, and undefined-denominator rule in §10.1. Rejecting a measurement requires a plan amendment before D3, not a post-result replacement.
5. The freeze predicates, including the 80% winning-profit floor and the 8-of-10 largest-winner count. These are proposed gates, not results.
6. D4’s cash-matched benchmark uses official iron-fly capital at risk, and the filtered quantities stay unscaled.
7. Later-period attribution and protection summaries wait until a freeze or a skip, and cannot select a structure.
8. Sequential authorization: accepting this plan does not start D0.
9. Sprint 007’s execution-observation handoff is not replaced by this draft.

---

## 17. Definition of done

Sprint 009 is complete only after a later acceptance, not by this draft. When executed, it is complete when:

- [ ] D0 records `READY` or a named blocker.
- [ ] D1 reconciles the five-term identity on development history, or stops on that identity.
- [ ] D2 reports protection cost, gross payout, and net contribution, and states that intraholding margin and liquidation paths were not measured if they are absent.
- [ ] D3 freezes at most one rule under the predeclared predicates, or records `STOP_NO_RULE`.
- [ ] D4 evaluates that rule against both benchmarks, or is skipped with the D3 reason. The rule is not revised on the later period.
- [ ] Dollar-profit retention and winner-count retention are not treated as the same number.
- [ ] D5 answers the central question without assuming wing removal or a production filter.
- [ ] Sprint 006/007/008 accepted results stay unreinterpreted.
- [ ] Quote fills are not claimed attainable.
- [ ] Focused tests pass for any new financial calculation.
- [ ] Official evidence directories and the frozen contract are unchanged.

---

## 18. Authorization

**Not started.** This draft does not authorize code, tests that compute new official economics, notebooks, evidence directories, or a kickoff status of accepted.
