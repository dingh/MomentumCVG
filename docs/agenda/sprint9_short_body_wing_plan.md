# Sprint 009 — Short-body economics, execution costs, protection, and conditional entry filtering

**Status:** `CLOSED — DIAGNOSTIC SCOPE COMPLETED`  
**Revised from:** closeout authorization 2026-09-14 (D2 accepted; D3/D4 superseded; amended D5 completed)  
**Updated:** 2026-09-14  
**Closeout:** [`docs/sprint_memos/009_closeout.md`](../sprint_memos/009_closeout.md)  
**D0 design:** [`docs/tmp/sprint009_d0_design.md`](../tmp/sprint009_d0_design.md) — **ACCEPTED** at `5329726`.  
**D0 evidence:** [`docs/tmp/sprint009_d0_evidence_review.md`](../tmp/sprint009_d0_evidence_review.md) — **ACCEPTED** through `82e3b46`. Implementation `004ba80`.  
**D1 design:** [`docs/tmp/sprint009_d1_design.md`](../tmp/sprint009_d1_design.md) — **ACCEPTED** at `e109a9e`. Implementation `5669773`.  
**D1 evidence:** [`docs/tmp/sprint009_d1_evidence_review.md`](../tmp/sprint009_d1_evidence_review.md) — **ACCEPTED** through `28f5ea4`. Implementation `5669773`. Output `C:/MomentumCVG_env/runs/sprint009_d1_20260914T025142Z`. Supersedes `0d63293` / `sprint009_d1_20260914T001504Z`.  
**D2 design:** [`docs/tmp/sprint009_d2_design.md`](../tmp/sprint009_d2_design.md) — **ACCEPTED** at `a34f21e`. Implementation `52be625`.  
**D2 evidence:** [`docs/tmp/sprint009_d2_evidence_review.md`](../tmp/sprint009_d2_evidence_review.md) — **REVIEWED / ACCEPTED**. Run verdict `READY`. Output `C:/MomentumCVG_env/runs/sprint009_d2_20260914T153106Z`.  
**Mode:** Audit. Sprint closed. Do not start Sprint 010 from this plan.  
**Agenda:** [`docs/agenda/current_sprint.md`](current_sprint.md)  
**Canonical path:** `docs/agenda/sprint9_short_body_wing_plan.md` — do not duplicate under `docs/tmp/`.  
**Prior closeouts:** [`docs/sprint_memos/008_closeout.md`](../sprint_memos/008_closeout.md) (accepted through `61cbf30`), [`docs/sprint_memos/007_closeout.md`](../sprint_memos/007_closeout.md), [`docs/sprint_memos/006_closeout.md`](../sprint_memos/006_closeout.md)  
**Frozen contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) — immutable; not edited by this sprint  

This document is the accepted sprint-level research protocol, closed under the **2026-09-14 scope amendment**. D0–D2 evidence is accepted. Original D3/D4 are **SUPERSEDED — NOT EXECUTED**. Amended D5 is the diagnostic closeout. Official inputs stay read-only.

---

## 0. Scope amendment (2026-09-14)

Accepted D0–D2 evidence shows that the short book’s remaining edge problem is not primarily entry filtering of already-selected names. The body can print positive expiry dollars while the existing protection architecture consumes that edge at a cost that exceeds observed payout. Body profit also concentrates in 2022. Intraperiod mark-to-market, margin, liquidation, and unseen-tail risks remain unmeasured.

**Amendment decision.** Sizing and protection architecture now take priority over entry filtering. Original §10 (D3) and §11 (D4) methodology is retained below under **SUPERSEDED — NOT EXECUTED**. No filter was tested or frozen. Do not record `STOP_NO_RULE`. Closing does not require executing D3/D4 or producing a later-period companion. Amended §12 (D5) is the diagnostic closeout and research handoff in [`009_closeout.md`](../sprint_memos/009_closeout.md).

---

## 1. Summary

| Item | Sprint 009 proposal |
|---|---|
| **Central question** | Where does the selected short book lose its economic edge, and can better trade selection improve it while accounting for the value of protection? |
| **Theme** | Short-body economics, execution costs, protection value, and conditional entry filtering |
| **Population** | Official short iron-fly candidates and strikes from the frozen `42:8` book. Conclusions are conditional on that population, including its wing-availability restrictions |
| **Reference book** | Official cross book. Quantities stay fixed. A midpoint repricing at those quantities is a diagnostic, not the separately sized official midpoint run |
| **Structures in scope** | Current iron fly (`wing_delta_target = 0.15`, `_choose_below_nearest`) and a body-only counterfactual that drops wings without adding names or resizing. Original D3/D4 filter pairs are superseded and were not executed |
| **Windows** | Development `2020-01-01` through `2023-12-31` for D1–D2. Later period not required for closeout |
| **Not the goal** | Force profitability; authorize uncovered trading; search new wing strikes; establish production readiness; resolve Sprint 007’s execution-calibration requirement |

Diagnostic closeout completed under the 2026-09-14 amendment. Original D3/D4 were not executed and are not recorded as `STOP_NO_RULE`.

---

## 2. What is authoritative

| Source | Role |
|---|---|
| [`docs/agenda/current_sprint.md`](current_sprint.md) | Active sprint status |
| This plan | Closed under the 2026-09-14 amendment. D0–D2 accepted. D3/D4 superseded. Amended D5 completed |
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

- Body midpoint credit \(C_{\mathrm{body}} = \mathrm{mid}_{\mathrm{call}} + \mathrm{mid}_{\mathrm{put}}\), in premium per share, or \(Q \times C_{\mathrm{body}}\) in dollars. This is the measurement denominator in §10.1. It is not a capital-at-risk figure.
- Iron-fly exposure for the M2 candidate only: official cross `capital_at_risk_dollars`. That quantity exists because the wings are held. It is not a return-on-capital claim by itself.
- Body-only exposure for the M1 candidate: entry-known notional \(Q \times S_0\), summed across positions, where \(S_0\) is the official entry spot. This matches underlying notional. It does not match brokerage margin, Greeks, or tail risk. Do **not** use the iron fly’s finite `max_loss_per_share` or `return_on_max_loss` as the body’s capital at risk.
- A large wing spread percentage on a cheap wing is not, by itself, large economic damage. Report dollars beside any percentage.

Normalized companions divide a dollar difference by the matching unfiltered exposure for that date. They are diagnostics. They are not return on capital and not a second primary test.

Cumulative results are sums of these fixed-quantity dollars across the authoritative date calendar. They are not compounded account equity, not View B compounded returns, and not a new equal-dollar book. Drawdown is peak-to-trough of that cumulative dollar series, with the peak including the initial zero.

### 4.5 Short-side calendar contract

Official `date_status` describes the **whole** portfolio. A date marked traded can contain zero included shorts, including a long-only date. Do not treat whole-book status as a short-book status.

D0 builds the authoritative date list from official `date_status` and reconciles it to official `funnel_summary` (`n_included_short`, `date_status`) and to included short trade and leg rows. Every authoritative date is retained. D0 may open later-period artifacts only for this readiness and reconciliation. It must not produce new comparative later-period economic results.

Each authoritative date is exactly one of:

| Class | Meaning |
|---|---|
| Verified positive short book | `n_included_short > 0`, included short trade rows and quantities match that count, and required leg and settlement fields are present |
| Verified zero-short book | Short inclusion is verified at zero. This includes official no-trade dates and traded dates that are long-only or otherwise have no included short. Not a missing file |
| Blocker | Missing, failed, or inconsistent short rows, quantities, legs, or settlement. This is not cash |

Verified zero-short dates contribute zero to applicable dollar series and to both exposure benchmarks. A missing short row must not be rewritten as cash. A zero unfiltered exposure omits that date from the normalized diagnostic only, and the omission is disclosed. The dollar series still keeps the verified zero.

---

## 5. Windows and firewall

| Window | Dates | Use |
|---|---|---|
| Development | `2020-01-01` through `2023-12-31` | D1 attribution and D2 protection. Original D3 inspection path superseded |
| Later period | `2024-01-01` through `2026-07-10` | Original D4 path superseded. Not required for closeout. 2026 is partial through `2026-07-10` |

D1–D2 economic inspection used development history. A later-period companion is not required under the 2026-09-14 amendment.

The later period was inspected in Sprints 006–008. Call it retrospective evaluation. It is not an untouched holdout and not independent confirmation.

Do not pool development and later-period inference. They are different windows and, if D4 runs, a different contrast family.

---

## 6. Deliverable map

| ID | Question | Continues when | Stop / skip |
|---|---|---|---|
| **D0** | Can we trust the body/wing comparison? | Matched short-book dataset reconciles to the official cross iron fly | Named identity or reconciliation blocker. No D1 |
| **D1** | Where does the short book lose economic margin? | Development decomposition reconciles | Reconciliation failure. No economic story from a broken identity |
| **D2** | What protection do the wings provide? | Development with/without-wings comparison is identified and labeled as a counterfactual | Same. A large wing cost does not by itself authorize wing removal |
| **D3** | Can entry measurements identify unattractive trades? | — | **SUPERSEDED — NOT EXECUTED** (2026-09-14 amendment). Methodology retained in §10 |
| **D4** | Does the frozen rule improve later-period economics? | — | **SUPERSEDED — NOT EXECUTED**. No later-period companion required for closeout |
| **D5** | What does the evidence justify? | Always | **COMPLETED** as diagnostic closeout ([`009_closeout.md`](../sprint_memos/009_closeout.md)) |

No deliverable selects a signal window, a new wing, a size, or a live fill.

---

## 7. D0 — Can we trust the body/wing comparison?

**Design:** [`docs/tmp/sprint009_d0_design.md`](../tmp/sprint009_d0_design.md) — **ACCEPTED** at `5329726`. Evidence **ACCEPTED** through `82e3b46` at implementation `004ba80`. The earlier run is superseded. The question, population, and calendar rules below are unchanged. That design specifies the column projection and checks. It does not replace D1–D5.

**Question.** Can the accepted artifacts support a matched body/wing dataset that reproduces every selected short iron fly and correctly accounts for every trading date?

**Inputs.** Official run `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` and `run_receipt.json`. Expected files already used in Sprint 007, plus the official funnel summary: `trade_log_cross`, `trade_log_mid`, `leg_log_cross`, `leg_log_mid`, `date_status_*`, `funnel_summary_*`, `decision_report.json`. Read-only. Sprint 007 D0 confirmed paired leg identity, quote identity, and settlement identity between fills. D0 here confirms those properties still hold for the short iron-fly subset, that body plus wings add to the official short book, and that the short-side calendar in §4.5 classifies every authoritative date.

**Bounded analysis.** Inventory the artifacts. Build one matched row per official included short iron fly, with four legs. Verify leg identity, option side, strike, expiry, quantity sign, premium sign, bid/ask/mid, fill price, and expiry settlement. Confirm cross quantities will be the reference. Pair each selected short trade key and its leg identity, quotes, and settlement to the midpoint artifacts by set difference, not an inner join. Confirm midpoint P&L at those cross quantities is computable from quotes and is distinct from `trade_log_mid` P&L. Reconcile, by date: whole-book `date_status`; funnel `n_included_short`; included short trade rows and their quantities; required leg rows and settlement fields. Assign the §4.5 classification. Later-period row-level readiness data may be stored. Do not compute development-versus-later economic comparisons, filter results, or protection summaries in D0. The pairing and saved-column contract are in [`sprint009_d0_design.md`](../tmp/sprint009_d0_design.md).

**Footprint.** One small read-only helper under `src/backtest/` and focused tests. Reuse Sprint 007 artifact-validation patterns. Do not call `build_ironfly_from_surface` to reselect wings. No `SurfaceRunner` rerun unless a required field is absent. A missing field is a blocker and a plan amendment, not a silent repair.

**Evidence and checks.** Artifact inventory, key uniqueness, leg-role verification, quantity-sign checks, quote sanity (missing bid/ask, crossed quotes), settlement present, the short-book classification counts, and

\[
\sum \text{leg P\&L} = \text{official cross trade P\&L} = \text{body dollars} + \text{wing dollars}
\]

within \(\max(\$0.01,\ 10^{-9}\times|\text{official}|)\). Primary-window short included count and short `pnl_total` must match the accepted closeout when the window is the full primary window.

**Done when.** A written readiness verdict is either `READY` or `BLOCKED` with a named gap. Implementation of D1 is not authorized by D0 drafting.

**Continuation.** `READY` allows a later D1 design/execution authorization. `BLOCKED` stops the sprint.

---

## 8. D1 — Where does the short book lose economic margin?

**Design:** [`docs/tmp/sprint009_d1_design.md`](../tmp/sprint009_d1_design.md) — **ACCEPTED** at `e109a9e`. Implementation `5669773`.  
**Evidence:** [`docs/tmp/sprint009_d1_evidence_review.md`](../tmp/sprint009_d1_evidence_review.md) — **ACCEPTED** through `28f5ea4`. Implementation `5669773`. Supersedes `0d63293` / `sprint009_d1_20260914T001504Z`. The question, terms, and identity below are unchanged. That design pins D0 column names, the development slice, and outputs. It does not replace D2–D5.

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

**Design:** [`docs/tmp/sprint009_d2_design.md`](../tmp/sprint009_d2_design.md) — **ACCEPTED** at `a34f21e`. Implementation `52be625`.  
**Evidence:** [`docs/tmp/sprint009_d2_evidence_review.md`](../tmp/sprint009_d2_evidence_review.md) — **REVIEWED / ACCEPTED**. Run verdict `READY`. Output `C:/MomentumCVG_env/runs/sprint009_d2_20260914T153106Z`. The question and boundaries below are unchanged. That design pins verified D1 column names, ranking, loss avoided, and outputs.

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

**Continuation.** D2 does not authorize wing removal or uncovered trading. Body-only P&L is a fixed-quantity research comparison. It does not establish brokerage-margin feasibility. Under the 2026-09-14 amendment, original D3/D4 are superseded and not executed. The diagnostic closeout is §12 / [`009_closeout.md`](../sprint_memos/009_closeout.md).

---

## 10. D3 — Can entry measurements identify unattractive trades?

**Status:** **SUPERSEDED — NOT EXECUTED** (2026-09-14 amendment). No filter was tested or frozen. Do not record `STOP_NO_RULE`. The methodology below is retained for history only.

**Question.** On development history only, does one predeclared entry-only measurement support freezing a single measurement/expression/exclusion combination?

**Inputs.** D0 quotes, classifications, and official cross quantities. Two profitability expressions, both counterfactuals on the same official iron-fly-selected population:

| Candidate | Score | Expression being filtered |
|---|---|---|
| M1 | Body execution burden | Body-only cross book at frozen \(Q\) |
| M2 | Complete-package entry burden | Cross iron-fly book at the same frozen \(Q\) |

M1 answers whether body execution burden identifies unattractive **body-only** trades. It is not scored on iron-fly P&L. M2 is scored on iron-fly P&L. Do not cross the pairs. Do not add a measurement or a cutoff grid. D1/D2 may explain the scores. They must not change a formula after results are seen.

Body-only results are fixed-quantity research comparisons. They do not establish feasibility on brokerage margin and do not authorize wing removal.

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

**Undefined scores.** A score is undefined if a quote that **its** formula uses is missing, that quote is crossed (`bid > ask`), or \(C_{\mathrm{body}} \le 0\). M1 does not become undefined because a wing quote is missing. M2 does. Do not impute zero. Do not drop the name from the unfiltered book of that expression. An exclusion rule may omit only defined scores. Undefined names stay in both the filtered and unfiltered books of that expression and are counted.

### 10.2 Candidate rule family

Exactly two candidates. No cross-combinations. No cutoff grid. No second expression invented after seeing ranks.

On each development date, for one candidate:

- Let \(n\) be the count of names with a defined score for that measurement.
- If \(n < 5\), exclude nobody that date. The filtered book equals the unfiltered book of **that** expression. Record the date.
- Otherwise exclude the highest-score group, \(k=\lfloor n/5\rfloor\), sorting by score descending, then ticker ascending.
- Retained names keep official cross quantities on that expression.
- Excluded names contribute cash at zero return that date on that expression. They are not resized onto the survivors.
- Verified zero-short dates contribute zero. They are not dropped.

Each candidate uses its matching expression for unfiltered and filtered P&L, excluded-versus-retained comparisons, winner retention, downside reporting, and the freeze predicates below.

### 10.3 What “separation” means

Do not use a Pearson correlation p-value as the continuation criterion. Do not compute one as a gate.

Evaluate, on development dates only:

- economic separation: whether the excluded group has worse mean paired dollar P&L than the retained group, on that candidate’s expression, summarized at date level;
- uncertainty: HAC inference on the date-level dollar P&L difference, filtered minus unfiltered, on that expression, including verified zero-short dates at zero;
- a diagnostic companion: that dollar difference divided by the date’s unfiltered exposure for **that** expression. M2 uses official short `capital_at_risk_dollars`. M1 uses \(\sum Q S_0\). The companion is not return on capital and not a second family member;
- winner retention and downside on that expression: dollar-profit retention and winner-count retention, reported separately, plus the ten largest winning trades and the worst trades and dates;
- stability: sign of the mean date-level dollar difference in each development year. Year checks are not added to the Bonferroni family.

HAC, frozen: maxlags 3, Bartlett kernel, small-sample correction, Student-t with \(T-1\) degrees of freedom. Adjusted p = \(\min(1,\ 2\times\text{raw p})\). Adjusted interval is 97.5%. Family size stays 2 even if one measurement is undefined on every row.

A verified zero-short date stays in the dollar series at zero. A zero unfiltered exposure omits that date from the diagnostic companion only. Those omissions are disclosed. They are not silent cash substitutions for missing rows.

### 10.4 Freeze rule

Freeze at most one measurement/expression/rule combination. A candidate is eligible only if all of the following hold on **its** development-history expression:

1. Excluded-minus-retained date-level mean dollar P&L is negative (the dropped group is worse).
2. The adjusted HAC interval for mean date-level dollar uplift (filtered minus unfiltered) lies entirely above zero.
3. The diagnostic companion has a non-negative point estimate. Disagreement with the dollar sign blocks a freeze. It is not a second test to shop, and it is not return on capital.
4. Winning-profit retention is at least 80% of development baseline winning dollars.
5. At least 8 of the 10 largest development winning trades are retained. Report the dollar share of those ten separately. The count is not the dollar percentage.
6. Mean date-level dollar uplift is positive in each of 2020, 2021, 2022, and 2023.

Each candidate is judged only against its own matching unfiltered expression. Eligibility uses the six predicates above. Family size stays 2.

- If exactly one candidate is eligible, freeze that measurement/expression/rule.
- If both are eligible, freeze the one with the larger adjusted lower confidence bound on mean date-level **dollar** uplift.
- If those dollar bounds differ by less than \$1, freeze neither and record `STOP_NO_RULE` with the tie reason.
- If neither is eligible, record `STOP_NO_RULE` with the failed predicates.

Do not rank the candidates by exposure-scaled confidence bounds. M1’s denominator is underlying notional and M2’s is iron-fly capital at risk. Those denominators cannot support a comparable ranking. Exposure normalization stays inside each candidate’s diagnostic companion and, if that pair is frozen, inside that pair’s D4 benchmark.

The common official iron-fly population, original cross quantities, and shared calendar make dollar uplift comparable for choosing which filter experiment to run next. That comparison is not a comparison of risk-adjusted returns and not a judgment of production suitability.

**Footprint.** Score function, exclusion function, and inference wrapper. Reuse Sprint 008 HAC and retention reporting patterns. New tests for undefined denominators, sort stability, \(k=\lfloor n/5\rfloor\), and the family-size lock. No threshold search.

**Evidence and checks.** Development score coverage, undefined counts, the two-candidate table, freeze or stop decision written before any later-period filter number is computed.

**Done when.** Exactly one rule is frozen, or `STOP_NO_RULE` is recorded with the failed predicates or the dollar tie-break named.

**Continuation.** One frozen rule authorizes D4 design/execution later. `STOP_NO_RULE` skips D4. It does not authorize a new measurement.

---

## 11. D4 — Does the frozen rule improve later-period economics?

**Status:** **SUPERSEDED — NOT EXECUTED** (2026-09-14 amendment). No later-period companion is required for closeout. The methodology below is retained for history only.

**Question.** On `2024-01-01` through `2026-07-10`, does the single frozen measurement/expression/rule improve economics relative to its matching unfiltered expression and relative to its matching exposure-scaled benchmark?

**Inputs.** The frozen D3 combination, unchanged. Official cross quantities. Later-period calendar from §4.5. Not available if D3 did not freeze a pair. If the frozen pair is M1, the expression remains the body-only cross book. If it is M2, the expression remains the cross iron fly. Do not evaluate the other expression as a new candidate.

**Two contrasts, both required. Family size for this evaluation is 2. Development inference is not pooled with it.**

1. **Matching unfiltered expression.** Same names and original quantities on that expression. Excluded names are cash at zero. This shows incremental P&L, including the effect of holding more cash.
2. **Matching exposure-scaled benchmark.** On each date,

\[
f_t = \frac{E_{\mathrm{retained},t}}{E_{\mathrm{unfiltered},t}}
\]

Scale every unfiltered position’s dollar P&L on that expression by \(f_t\). Keep filtered positions at their original quantities. The benchmark shrinks the unfiltered mix. It does not drop names.

| Frozen candidate | Exposure \(E\) | What the scaler matches |
|---|---|---|
| M2 iron fly | Official cross `capital_at_risk_dollars` | Iron-fly capital at risk. Not used for the body |
| M1 body-only | \(\sum Q S_0\) | Underlying notional. Not brokerage margin, Greeks, or tail risk. Not the iron fly’s maximum loss |

If unfiltered exposure is zero on a verified zero-short date, both dollar results are zero. Do not drop the date. Disclose the zero denominator if a diagnostic ratio is also shown. A missing short row is a blocker, not a zero.

**Inference, frozen.** Date-level mean dollar difference on the frozen expression. Same HAC settings as D3. Family size 2: contrast 1 and contrast 2. Adjusted interval 97.5%. Do not add a third contrast after seeing results. Year slices, including partial 2026, are descriptive.

**Report.** Absolute P&L, incremental P&L versus each comparison, losses avoided, winning profits sacrificed, dollar-profit retention, winner-count retention, retained exposure and name count, worst trades, worst dates, and fixed-quantity cumulative dollar drawdown. State whether either book is profitable. Relative improvement is not absolute profitability. An M1 result does not authorize uncovered trading.

**Footprint.** One evaluation function. Tests that the rule text matches the D3 freeze record, that filtered quantities are unscaled, that M1 uses \(\sum Q S_0\) rather than iron-fly max loss, and that M2 uses official capital at risk.

**Done when.** Both comparisons are reported, or D4 is recorded skipped with the D3 reason. The rule is not revised after later-period results.

**After the decision is frozen.** One descriptive later-period attribution and protection summary may then be produced, clearly labeled retrospective and non-selecting. It is not a second D4 and not a reason to reopen D3.

---

## 12. D5 — What does the evidence justify?

**Status:** **COMPLETED** as the diagnostic closeout under the 2026-09-14 amendment. See [`009_closeout.md`](../sprint_memos/009_closeout.md).

**Question.** Given the accepted D0–D2 chain and the superseded D3/D4 path, what historical results are established, what one investigation should come next, and what evidence is still missing for any operational decision?

**Inputs.** Accepted D0–D2 evidence. Original D3/D4 recorded as superseded, not executed.

**Bounded analysis.** A closeout memo. It may:

- establish the historical body, wing, and protection results;
- recommend one prioritized subsequent investigation;
- identify evidence still needed before any operational decision.

It must cover body economics, execution-cost attribution, protection cost and value, the reason original filtering work was superseded, and remaining profitability and implementation limits.

It must **not** establish production readiness, authorize uncovered trading, approve wing removal, promote a production filter, treat quote crosses as attainable fills, or claim to resolve Sprint 007’s `EXECUTION_CALIBRATION_REQUIRED` outcome. That handoff remains open.

**Footprint.** [`docs/sprint_memos/009_closeout.md`](../sprint_memos/009_closeout.md).

**Done when.** The memo answers the diagnostic question within those limits and records the research handoff without designing Sprint 010.

---

## 13. Dependencies and branches

```text
Plan accepted
    → D0
        blocked → stop
        ready → D1 (development attribution)
            identity fails → stop
            reconciles → D2 (development protection counterfactual)
                → 2026-09-14 amendment:
                    D3/D4 SUPERSEDED — NOT EXECUTED
                    → D5 diagnostic closeout
```

Historical (superseded) branch retained for reference only:

```text
D2 → D3 (freeze or STOP_NO_RULE) → D4 or skip → later-period companion → D5
```

Choices answered by accepted evidence:

- D0 readiness: `READY`.
- D1 five-term split: body profitable at cross; wings dominate the loss.
- Net wing contribution: negative on development history.
- Original D3/D4: superseded before execution; sizing and protection architecture take priority.
- Next investigation: candidate recovery and research sizing under explicit portfolio stress limits (handoff only; not Sprint 010 design).

---

## 14. Proposed implementation increments

These increments describe the executed path and the superseded remainder.

| Increment | Work | Status |
|---|---|---|
| 1 | D0 inventory, short-book calendar classification, reconciliation tests | **Completed** (`004ba80`, accepted) |
| 2 | D1 development decomposition and identity tests | **Completed** (`5669773`, accepted) |
| 3 | D2 development protection comparison and tests | **Completed** (`52be625`, accepted) |
| 4 | D3 two candidate-expression pairs, freeze record, tests | **SUPERSEDED — NOT EXECUTED** |
| 5 | D4 later-period evaluation of the frozen pair only | **SUPERSEDED — NOT EXECUTED** |
| 6 | D5 closeout | **Completed** ([`009_closeout.md`](../sprint_memos/009_closeout.md)) |

No increment reruns `SurfaceRunner` or retunes `42:8`. Sprint 010 is not designed here.

---

## 15. Out of scope

- Signal-window or universe expansion.
- Long-side changes, including promotion of Sprint 008 M1.
- Alternative wing strike or delta searches.
- Sizing or leverage optimization inside this sprint (handed off as next research theme only).
- Brokerage margin, live execution, paper-trading plumbing, or the Sprint 007 execution observer.
- Intraday hedging or a new exit policy.
- Iron condor, and any KB-001 fix, unless a later amendment says otherwise.
- Treating historical quote crosses as attainable fills.
- Mutating official Sprint 006/007/008 evidence directories or the frozen contract.
- Sprint 010 planning or implementation.

Data limits and narrow enabling work belong in the deliverable that finds them. This sprint does not expand to repair every operational gap.

---

## 16. Decisions required before kickoff

Historical kickoff decisions for the executed D0–D2 path remain as accepted. Items 4–7 about D3/D4 filtering are **superseded** by the 2026-09-14 amendment and were not executed.

1. Reference quantities are the official cross book, not the official midpoint book and not a new equal-dollar book.
2. Body-only is a counterfactual on the iron-fly-selected population, not a newly selected short-straddle strategy.
3. The five-term attribution and the concession definitions in §8, including fees = 0.
4. ~~The two measurement/expression pairs in §10~~ — **SUPERSEDED — NOT EXECUTED**.
5. ~~The freeze predicates~~ — **SUPERSEDED — NOT EXECUTED**.
6. ~~Exposure benchmarks for D4~~ — **SUPERSEDED — NOT EXECUTED**.
7. ~~Later-period attribution after freeze/skip~~ — **SUPERSEDED — NOT EXECUTED**; not required for closeout.
8. Sequential authorization for D0–D2 remains as executed.
9. D5 cannot close Sprint 007’s execution-calibration requirement or declare production readiness.

---

## 17. Definition of done

Sprint 009 is **complete** under the 2026-09-14 amendment when:

- [x] D0 records `READY` or a named blocker, including the short-book calendar classification.
- [x] D1 reconciles the five-term identity on development history, or stops on that identity.
- [x] D2 reports protection cost, gross payout, and net contribution, and states that intraholding margin and liquidation paths were not measured if they are absent.
- [x] Original D3/D4 are recorded **SUPERSEDED — NOT EXECUTED** (no requirement to freeze a filter or run a later-period companion).
- [x] Amended D5 states historical results and one next investigation only. It does not authorize uncovered trading, production use, or a Sprint 007 resolution.
- [x] Sprint 006/007/008 accepted results stay unreinterpreted.
- [x] Quote fills are not claimed attainable.
- [x] Focused tests passed for D0–D2 financial calculation.
- [x] Official evidence directories and the frozen contract are unchanged.

---

## 18. Authorization

**Closed.** Diagnostic scope completed. See [`009_closeout.md`](../sprint_memos/009_closeout.md). Do not start Sprint 010 from this plan.
