# Sprint 009 D2 — Protection provided by the existing wings

**Status:** `DESIGN DRAFT — AWAITING REVIEW; IMPLEMENTATION NOT STARTED`  
**Updated:** 2026-09-14  
**Agenda:** [`docs/agenda/current_sprint.md`](../agenda/current_sprint.md)  
**Working plan:** [`docs/agenda/sprint9_short_body_wing_plan.md`](../agenda/sprint9_short_body_wing_plan.md) §9 — formulas unchanged; this file pins D1 columns and outputs  
**D1 design:** [`docs/tmp/sprint009_d1_design.md`](sprint009_d1_design.md) — accepted at `e109a9e`  
**D1 evidence:** [`docs/tmp/sprint009_d1_evidence_review.md`](sprint009_d1_evidence_review.md) — **ACCEPTED** through `28f5ea4`. Implementation `5669773`.

This draft does not authorize a helper, runner, test file, chart, or evidence directory. D3–D5 are unchanged.

---

## Question

> What protection did the wings provide, and what losses would the same positions have suffered without them?

D2 is complete when it explains, on development history, the historical dollar P&L gained by removing the existing wings and the observed protection surrendered. A profitable body-only book is not required. A broken identity stops the story.

It does not establish margin-call or liquidation risk, prove that unseen tails are safe, or authorize uncovered trading. These series are fixed-quantity expiry outcomes. They are not compounded account returns and not intraperiod equity paths.

## Working hypothesis (not a verdict)

Accepted D1 already stores body cross P&L, iron-fly cross P&L, wing premium, wing concession, and wing expiry payout at official cross quantities. D2 pairs those columns. It does not rerun selection or rebuild structures.

## Authorization

Awaiting review. Acceptance of this design would still not start implementation. Do not create implementation or evidence files until that is separately authorized.

---

## Scope boundary

Development only: `2020-01-01` through `2023-12-31`. Required coverage, or `BLOCKED`: **2,087 trades** and **209 dates** (208 `verified_positive_short`, one `verified_zero_short` on `2020-03-13`). Preserve every development trade and calendar date, including that zero-short date at zero dollars. Do not drop rows to force the counts.

Same names, strikes, existing wings, and official cross quantities. Removing wings does not resize, does not add candidates, and does not change \(Q\). \(Q\) is already share-equivalent. Do not multiply by 100.

Cross is primary. Midpoint uses those same quantities and is a diagnostic. It is not the official midpoint run. Fees stay 0. Settlement stays hold-to-expiry. Do not deduct concession twice. Body execution concession already sits inside `p_body_cross`. Do not subtract `h_body` again when forming the no-wing book.

Do not rerun the baseline or D0. Do not compute later-period economics, search wings, filter, optimize size, or add a brokerage-margin model. Do not change D3–D5.

The full-primary-window −$146,279.85 is not a D2 residual or benchmark.

---

## Inputs

Read only the accepted D1 directory. Do not write into it. Do not read the superseded D1 run `C:/MomentumCVG_env/runs/sprint009_d1_20260914T001504Z/` or the superseded D0 run.

| Item | Value |
|---|---|
| D1 output | `C:/MomentumCVG_env/runs/sprint009_d1_20260914T025142Z/` |
| Trades | `trade_decomposition.parquet` |
| Dates | `date_decomposition.parquet` |
| Annual | `annual_decomposition.parquet` |
| Dollars | `aggregate_dollars.json` |
| Receipt | `execution_receipt.json` |
| Implementation | `5669773f356f6c33cef86bd0da30ce4051709a6b` |
| Receipt verdict | `READY` |
| Evidence acceptance | `28f5ea4` |

Column names below were verified on that directory. They are the calculation inputs. Do not invent aliases.

**Trade columns used:** `trade_date`, `ticker`, `direction`, `window_label`, `Q`, `quantity_cross_signed`, `input_ok`, `pnl_cross_official`, `pnl_body_cross`, `pnl_wing_cross`, `pnl_legs_sum`, `pnl_mid_at_cross_q`, `b_mid`, `h_body`, `w_mid`, `h_wing`, `w_pay`, `p_body_cross`, `p_fly_cross`.

**Date columns used:** `trade_date`, `short_book_class`, `n_trades`, `b_mid`, `h_body`, `w_mid`, `h_wing`, `w_pay`, `p_body_cross`, `p_fly_cross`.

**Annual columns used:** `year`, `n_dates`, `n_zero_short_dates`, `n_trades`, and the same dollar columns as the date file.

D1 does not store strikes or expiry spot. Every dollar in this design is already on the D1 columns. Do not read D0 for economics. A strike or spot label is not a required output. Adding one would be a new authorization, not a silent join.

Hash the three parquet files and the receipt. Record the hashes. A hash, directory, code SHA, or receipt verdict mismatch is a named blocker. These SHA-256 values were read from the accepted directory at planning and must match at execution:

| File | SHA-256 |
|---|---|
| `execution_receipt.json` | `6acdc0c54732766352732bab1a91c47d53b0d84e4b641614ac660f53f457283d` |
| `trade_decomposition.parquet` | `72da77c10d3e8ba403012ffad785e5338d69be1f2e180135e3414e03f3a7213b` |
| `date_decomposition.parquet` | `49db7b233ee628551ec6fffd05ac76fc07e2b115e9d03019cb55ad054c6cf611` |
| `annual_decomposition.parquet` | `62be4856076fa5485d51a945dc15b83b358fc37cf513df71fad0abd1eabec607` |

A `development` label outside the window, or a window date whose label is not `development`, is a named blocker. Do not reclassify.

`input_ok` must be an actual true boolean. A missing value or a truthy string fails. Required dollar fields must be finite. Duplicate `(trade_date, ticker, direction)` or duplicate calendar dates fail. Do not impute, drop, or substitute cash.

---

## Shared accounting

All amounts are dollars at official \(Q\). Copy \(Q\). Do not rescale.

Sign and missing-value epsilon: an amount is positive if it is \(> 10^{-6}\), zero if its absolute value is \(\le 10^{-6}\), and negative if it is \(< -10^{-6}\). The \(10^{-6}\) bound stops a \(10^{-11}\) residual from flipping a count. It is not a \$0.01 rounding of the dollars.

Undefined ratios stay null with a reason. The underlying trades remain in every dollar total.

| Quantity | Formula | Role |
|---|---|---|
| Body-only cross | `p_body_cross` | No-wing counterfactual. Must match `pnl_body_cross` |
| Iron-fly cross | `p_fly_cross` | With wings. Must match `pnl_cross_official` and `pnl_legs_sum` |
| Cross purchase cost | `w_mid + h_wing` | Premium plus concession. Not payout |
| Gross payout | `w_pay` | Expiry payout. Not net value and not loss avoided |
| Net wing contribution, cross | `w_pay - w_mid - h_wing` | Must match `pnl_wing_cross` |
| Body-only midpoint | `b_mid` | Diagnostic |
| Midpoint purchase cost | `w_mid` | Diagnostic. Do not add `h_wing` |
| Net wing contribution, midpoint | `w_pay - w_mid` | Diagnostic |
| Iron-fly midpoint | `b_mid + w_pay - w_mid` | Must match `pnl_mid_at_cross_q` |

Required identities, trade, date, annual, and development total:

\[
P_{\mathrm{fly,cross}} = P_{\mathrm{body,cross}} + (W_{\mathrm{pay}} - W_{\mathrm{mid}} - H_{\mathrm{wing}})
\]

\[
P_{\mathrm{fly,mid}} = B_{\mathrm{mid}} + (W_{\mathrm{pay}} - W_{\mathrm{mid}})
\]

Historical P&L gained by removing wings, in dollars, is \(-(W_{\mathrm{pay}} - W_{\mathrm{mid}} - H_{\mathrm{wing}})\), which equals body-only cross minus iron-fly cross. A positive number means the no-wing book made more at expiry. It is not a decision to drop wings.

Tolerance against the matching D1 field, same level: \(\max(\$0.01,\ 10^{-9}\times|\mathrm{reference}|)\). The development reference is the D1 development sum, not −$146,279.85.

Date and annual dollars are sums of the development trades on those dates. They must also match the saved D1 date and annual columns. The zero-short date has no trades: every dollar field is 0. That is the calendar rule, not an imputation.

---

## 1. D2-A — When did wings pay, and when did they improve net trade P&L?

Headline frequencies are trade counts over 2,087. A date-level companion uses the date sum over 209 dates, including the zero-short date as not positive. Do not present a date frequency as if it were the trade frequency.

| Event | Cross rule |
|---|---|
| Payout positive | `w_pay` \(> 10^{-6}\) |
| Payout exceeded purchase cost | `w_pay - (w_mid + h_wing)` \(> 10^{-6}\) |
| Net contribution positive | same comparison as exceeded purchase cost |

Report those three counts and the matching rates. Equal to cost is not “exceeded.”

Distributions, separately, in dollars:

- All 2,087 trades: gross payout, cross purchase cost, and net contribution. Include zeros and negatives.
- Conditional on positive payout only: the same three fields. State the conditional count. If that count is 0, conditional summaries are null with reason `no positive wing payout`. Do not drop those trades from the all-trades distribution or the dollar totals.

Report mean and the 10th, 50th, and 90th percentiles. Do not report the conditional distribution as the book.

Midpoint support, same quantities, not a second headline: midpoint purchase cost `w_mid`, midpoint net `w_pay - w_mid`, and the count of trades where midpoint net is positive. Do not rank or filter on midpoint.

---

## 2. D2-B — How much did wings reduce the worst observed losses?

Ranking uses cross P&L only. More negative is worse. Midpoint does not rank.

| List | Rank by | Units | Tie-break |
|---|---|---|---|
| 10 worst body-only trades | `p_body_cross` ascending | trades | `trade_date`, then `ticker`, then `direction`, all ascending |
| 10 worst body-only dates | date sum of `p_body_cross` ascending | calendar dates | `trade_date` ascending |
| 10 worst iron-fly trades | `p_fly_cross` ascending | trades | same trade tie-break |
| 10 worst iron-fly dates | date sum of `p_fly_cross` ascending | calendar dates | `trade_date` ascending |

Each list is length 10. Fewer eligible rows is a blocker. The ranking universe is all 2,087 trades or all 209 dates. The zero-short date stays in the date universe. It is not expected in the worst 10 unless fewer than 10 dates are negative, which must still be shown rather than backfilled.

Each body-only row shows the paired iron-fly outcome on that same trade or date. Each iron-fly row shows the paired body-only outcome. Columns: keys, `n_trades` on date rows, body cross, fly cross, gross payout, cross purchase cost, net contribution, and loss avoided. Do not re-sort the paired column.

**Loss avoided**, trade or date, using that row’s signed P&L:

\[
\max(-P_{\mathrm{body}}, 0) - \max(-P_{\mathrm{fly}}, 0)
\]

Positive means the wings reduced the realized loss. Negative means the wings increased the realized loss. Keep the negative. Zero means neither book lost, or both lost the same dollar amount. Do not equate this with `w_pay`. Do not floor it at zero.

Report the development sum of trade-level loss avoided, and separately the development sum of date-level loss avoided. Those two sums use different units. Do not add them. Date-level uses the date’s signed sum, then the formula. It is not the sum of trade-level losses inside the date.

**Loss concentration** is a share of gross losing dollars, not of net P&L.

Gross losing dollars of a book, trade unit: \(\sum \max(-P, 0)\) over all 2,087 trades. Date unit: \(\sum \max(-P_{\mathrm{date}}, 0)\) over all 209 date sums. A date that nets positive contributes 0 even if some names lost. Do not mix those denominators.

For each of the four worst-10 lists, the share is that list’s gross losing dollars in the ranked book, divided by that book’s matching denominator. Also report the single worst row’s share, same denominator and tie-break. If the denominator is 0, the share is null with reason `no gross losing dollars`. The rows stay in the dollar tables.

---

## 3. D2-C — How did wings change cumulative profit, drawdown, and annual outcomes?

Build two cross series and two midpoint diagnostic series from the 209 calendar dates sorted by `trade_date`. Include `2020-03-13` as a zero step. Do not compound. Do not insert intraperiod marks.

Let \(p_t\) be that date’s book P&L. \(C_0 = 0\) and \(\mathrm{peak}_0 = 0\) before any date.

\[
C_t = C_{t-1} + p_t, \qquad \mathrm{peak}_t = \max(\mathrm{peak}_{t-1}, C_t), \qquad dd_t = C_t - \mathrm{peak}_t
\]

Maximum dollar drawdown is \(\min_t dd_t\). It is \(\le 0\). The initial peak of zero means a book that is always negative draws down from zero, and the max drawdown equals the minimum cumulative. Do not report that minimum as a positive number without its sign.

Cross books: date `p_body_cross` and date `p_fly_cross`. Midpoint diagnostics: date `b_mid` and date `b_mid + w_pay - w_mid`. The date file has no saved midpoint fly column. Reconcile that diagnostic to the sum of trade `pnl_mid_at_cross_q` on that date.

Annual table, one row per year 2020–2023, reconciled to `annual_decomposition.parquet`: `n_dates`, `n_zero_short_dates`, `n_trades`, body cross, fly cross, net wing contribution, gross payout, cross purchase cost, count of trades with positive payout, count of trades whose net contribution is positive, and that year’s ending cumulative for both cross books. A missing year is a blocker. Do not add 2019 or 2024.

The memo states whether the body-only dollar advantage and the protection events are concentrated in particular years or in the worst-event lists. Concentration is a description of this sample. It is not a probability for unseen tails.

---

## Outputs

Write only under a new directory created by a later authorized run:

`C:/MomentumCVG_env/runs/sprint009_d2_<UTC timestamp>/`

Do not create that directory in this planning step. Do not write into D0, D1, or Sprint 006 directories.

| File | Contents |
|---|---|
| `input_inventory.json` | D1 path, code SHA, receipt verdict, hashes, coverage |
| `trade_comparison.parquet` | One row per development trade. Not later-period rows |
| `date_comparison.parquet` | One row per development date, including `2020-03-13` |
| `annual_comparison.parquet` | 2020–2023 |
| `worst_events.parquet` | Four labeled lists of 10, plus the paired columns |
| `frequency.json` | All-trade and conditional counts, rates, and distribution summaries |
| `concentration.json` | Four shares, top-1 shares, denominators, null reasons |
| `drawdown.json` | Ending cumulative and max drawdown for both cross books and both midpoint diagnostics |
| `cumulative_cross.png` | Both cross cumulatives, dollars, only if every gate passes |
| `annual_cross.png` | Grouped annual body vs fly dollars, only if every gate passes |
| `d2_report.md` / `d2_report.json` | `READY` or `BLOCKED`, the three answers, residuals |

Charts use matplotlib with the Agg backend. No new dependency. No later-period series. No return axis. If any gate fails, do not write a chart.

Forbidden report keys: `later_period_pnl`, `filter_result`, `margin_call`, `uncovered_authorization`, `primary_window_anchor_as_development`.

`d2_report.md` answers D2-A, D2-B, and D2-C only if every gate passed. Required caveats: fees = 0; concession is not deducted twice; midpoint fills are not attainable; this is not a margin or path result; gross payout is not loss avoided; the no-wing book is a counterfactual on the iron-fly-selected population; D2 does not authorize uncovered trading. If a gate fails, the report says `BLOCKED` and lists the named gaps. It does not interpret a partial book.

---

## Gates

`READY` only if all pass. Any named gap is `BLOCKED`. There is no `READY_WITH_NARROW`.

| Gate | Pass rule |
|---|---|
| Provenance | Accepted D1 directory, receipt SHA, code SHA, verdict `READY`, and recorded parquet hashes |
| Coverage | 2,087 trades, 209 dates, the zero-short date, unique keys, `input_ok` true |
| Identity | Both identities above, at trade, date, annual, and development total |
| Reconciliation | Body, wing, midpoint, and official cross fields match saved D1 at those levels |
| Ranking | Each worst list has 10 rows, the stated order and tie-break, and paired columns |
| Calendar | Zero-short date is present at zero dollars and is a step in both cumulative series |
| Scope | No later-period field and no forbidden key |

---

## Footprint (do not create yet)

| Piece | Path | Role |
|---|---|---|
| Comparison | `src/backtest/sprint009_d2_protection_comparison.py` | D1 frames in, paired terms and rankings out. No I/O beyond what the runner asks |
| Runner | `scripts/run_sprint009_d2_protection.py` | Thin CLI. Reads the accepted D1 directory. Draws charts only after the gates pass |
| Tests | `tests/unit/test_sprint009_d2_protection_comparison.py` | Hand-calculated fixtures. No official parquet |

Reuse D1’s dollar tolerance. Do not import `SurfaceRunner`, the D0 runner, or the baseline runner.

### Invocation (not run now)

```text
$env:PYTHONPATH = "C:\MomentumCVG"
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint009_d2_protection.py
```

### Tests required before an official run

A valid control must pass the same verdict function the runner uses. Each invalid case must fail the named gate and must not be `READY`.

- Dropping wings does not change \(Q\), the trade keys, or the name set. Multiplying \(Q\) by 100 fails reconciliation to saved D1.
- Gross payout, cross purchase cost, and net contribution are separate. A fixture where payout is positive and net contribution is negative is counted in payout frequency and not in net-improvement frequency.
- Ranking uses the stated column and tie-break. A paired column does not reorder the list. The worst body-only event can show a worse iron-fly loss; loss avoided stays negative.
- Loss-concentration denominators are gross losing dollars. A profitable book with a few losers does not use net P&L as the denominator. A zero denominator stays null and does not drop the trades.
- The zero-short date remains, with zeros, in the date table and in the cumulative series. It does not become a missing day.
- Drawdown peak starts at zero. A first-date loss is a drawdown from zero. An all-positive series has max drawdown 0.
- A missing or non-finite required field, or `input_ok` that is not an actual true boolean, is `BLOCKED`. A genuine D1 residual outside tolerance is `BLOCKED`. A stored identity that matches still passes.

---

## Missing data and undefined ratios

Record the trade key or date, the field, and the reason. Do not impute a payout, a zero loss, or a cash P&L.

| Condition | Result |
|---|---|
| Required field missing or non-finite | `BLOCKED` |
| `input_ok` not an actual true boolean | `BLOCKED` |
| Conditional payout distribution with no positive payout | null, reason `no positive wing payout` |
| Concentration share with no gross losing dollars | null, reason `no gross losing dollars` |
| Rate with a zero trade count | null, reason `no development trades` — also a coverage blocker on the official run |

Null ratios do not remove the observation from dollar totals, worst-event eligibility, or the cumulative series.

---

## Definition of done

D2 is done when the gates pass or a named blocker is recorded, and the memo, if `READY`, states:

- how often wings paid, and how often that payout exceeded purchase cost, with all-trade and conditional-on-payout distributions kept separate;
- the historical dollars gained by removing wings, equal to minus net wing contribution;
- paired outcomes on the worst body-only and worst iron-fly events, with loss avoided signed and not replaced by gross payout;
- loss concentration as shares of gross losing dollars;
- cumulative dollar P&L and maximum dollar drawdown for both books, peak including the initial zero, plus the 2020–2023 comparison;
- whether that advantage and that protection are concentrated in years or events in this sample.

It is not done if it claims a margin result, an unseen-tail probability, an attainable midpoint fill, or authorization to trade uncovered. D3–D5 stay as written in the sprint plan.

---

## Acceptance of this design

Review can accept, amend, or reject the cross-versus-midpoint split, the ranking tie-break, the loss-avoided sign, and the ban on a D0 economic join.

Acceptance still does not start implementation. An implementation run, if later authorized, returns `READY` or `BLOCKED`. This file must not be edited to say `READY` before that evidence exists and is reviewed.

No accounting choice is left open for the implementer. Cross is primary. Purchase cost is `w_mid + h_wing`. Net contribution is `w_pay - w_mid - h_wing`. Loss avoided can be negative. The population and quantities stay the accepted D1 development book.
