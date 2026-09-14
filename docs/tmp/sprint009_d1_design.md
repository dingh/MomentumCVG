# Sprint 009 D1 — Development margin decomposition

**Status:** **ACCEPTED** at `e109a9e`  
**Corrected from:** `5d33055` (stored ORATS mid is not the fill-model midpoint; disagreement is a diagnostic, not a gate)  
**Updated:** 2026-09-13  
**Implementation:** `5669773`. Run verdict `READY`. Evidence [`sprint009_d1_evidence_review.md`](sprint009_d1_evidence_review.md) — **ACCEPTED** through `28f5ea4`. Output `C:/MomentumCVG_env/runs/sprint009_d1_20260914T025142Z`. Supersedes `0d63293` / `sprint009_d1_20260914T001504Z`.  
**Agenda:** [`docs/agenda/current_sprint.md`](../agenda/current_sprint.md)  
**Working plan:** [`docs/agenda/sprint9_short_body_wing_plan.md`](../agenda/sprint9_short_body_wing_plan.md) §8 — formulas unchanged; this file pins saved columns and outputs  
**D0 design:** [`docs/tmp/sprint009_d0_design.md`](sprint009_d0_design.md) — accepted at `5329726`  
**D0 evidence:** [`docs/tmp/sprint009_d0_evidence_review.md`](sprint009_d0_evidence_review.md) — **ACCEPTED** through `82e3b46`. Implementation `004ba80`. Do not use the superseded run.

---

## Question

> Where does the selected short iron-fly book lose its economic margin?

On development history only. A reconciled decomposition is the answer. A profitable book is not a completion requirement. A broken identity stops the economic story.

## Working hypothesis (not a verdict)

The accepted D0 panel already stores the quotes, signed settlement, quantity, and official cross P&L this attribution needs. D1 still has to compute the five terms from those columns, cross-check them against D0’s saved body, wing, midpoint, and official results, and keep the verified zero-short date. It does not rerun selection.

## Authorization

This design is **ACCEPTED** at `e109a9e`. D1 evidence is **ACCEPTED** through `28f5ea4`. That acceptance does not authorize D2 implementation, D3–D5, or later-period economics. D2 planning is a separate draft.

---

## Scope boundary

Read the accepted D0 outputs. Do not reconstruct selection, call `SurfaceRunner`, call `build_ironfly_from_surface`, or rerun the Sprint 006 baseline.

Freeze the selected population, strikes, official cross quantities, hold-to-expiry settlement, and fees = 0. \(Q\) is already share-equivalent. Do not multiply by 100 again.

Economic calculations use trades and dates from `2020-01-01` through `2023-12-31` only, and only before any reported aggregate. Expected development coverage, from the accepted D0 evidence: **2,087 trades** and **209 dates** (208 `verified_positive_short`, one `verified_zero_short`). The zero-short date in that evidence is `2020-03-13`. Preserve it in every date-level summary with dollar contribution zero. It is not missing data and it is not dropped.

The full-primary-window short result, 3,322 trades and \(-\$146{,}279.85\), is a D0 anchor. It is **not** the development-period benchmark. Do not copy it into a D1 table or residual.

Out of this design, unchanged in the sprint plan:

- D2: protection frequency, worst losses, concentration, drawdown, with-versus-without wings.
- D3: measurements, correlations, thresholds, freeze rules.
- Later-period economic summaries, new wings, sizing, brokerage margin, attainable-fill claims.

Net wing contribution may appear only as the algebraic identity term \(W_{\mathrm{pay}} - W_{\mathrm{mid}} - H_{\mathrm{wing}}\), labeled as a reconciliation to D0 wing cross P&L. It is not a protection conclusion.

---

## Inputs

Read-only. Do not write into this directory.

| Item | Value |
|---|---|
| D0 output | `C:/MomentumCVG_env/runs/sprint009_d0_20260913T215246Z/` |
| Matched trades | `matched_short_iron_flies.parquet` |
| Short calendar | `short_calendar.parquet` |
| Implementation | `004ba80052f6586f6a230ad207e8151e695e156e` |
| Evidence acceptance | `82e3b46` |
| Superseded D0 run | `C:/MomentumCVG_env/runs/sprint009_d0_20260913T212939Z/` — do not read |

Hash both parquet files and record those hashes. A row with `window_label = development` whose `trade_date` is outside `2020-01-01` through `2023-12-31`, or a row inside those dates whose `window_label` is not `development`, is a named blocker. Do not reclassify it.

Required development coverage, or `BLOCKED`:

| Check | Required |
|---|---|
| Development trades | 2,087 |
| Development calendar dates | 209 |
| `verified_positive_short` | 208 |
| `verified_zero_short` | 1, date `2020-03-13` |
| Development `pairing_ok` false | 0 |

Do not drop rows to force these counts.

---

## 1. D1-A — Does the body generate positive margin before and after execution cost?

At the official cross \(Q\), for each development short iron fly:

| Symbol | Meaning | Sign |
|---|---|---|
| \(B_{\mathrm{mid}}\) | Body midpoint P&L | P&L, before execution concession |
| \(H_{\mathrm{body}}\) | Body execution concession | Cost. Embedded in the cross fill. Not a second fee |
| \(P_{\mathrm{body,cross}}\) | Body cross P&L | \(B_{\mathrm{mid}} - H_{\mathrm{body}}\) |

The D1-A answer uses development **dollar** sums of \(B_{\mathrm{mid}}\) and \(P_{\mathrm{body,cross}}\) first, then the concession ratio below. Positive after concession is not required for D1 to finish. If the identity fails, do not interpret the sign.

---

## 2. D1-B — How much do the wings cost, and how much is spread?

| Symbol | Meaning | What it is not |
|---|---|---|
| \(W_{\mathrm{mid}}\) | Wing midpoint premium, dollars | Not spread. Not expiry payout |
| \(H_{\mathrm{wing}}\) | Wing execution concession, dollars | Not the midpoint price of protection |
| \(W_{\mathrm{pay}}\) | Gross wing expiry payout, dollars | Not wing P&L. Not net protection value |

Dollars first. Then each of \(W_{\mathrm{mid}}\) and \(H_{\mathrm{wing}}\) over body midpoint credit. \(H_{\mathrm{wing}} / W_{\mathrm{mid}}\) may be shown only as a descriptive spread percentage beside the dollar concession. It is not the headline damage figure.

---

## 3. D1-C — Can the development result be explained with reconciled evidence?

The development official cross P&L is the sum of `pnl_cross_official` on the 2,087 development trades, including the zero-short date as zero. It is explained only if the identity below holds at trade, date, annual, and development-total levels.

Also write a descriptive annual table for 2020–2023. It is not a significance test and not a window search.

The interpretation memo answers D1-A, D1-B, and D1-C from those reconciled dollars. It does not recommend dropping wings or freezing a filter.

---

## 4. Accounting

Two midpoints appear on a D0 row. They are not the same object, and a difference between them is not by itself an error.

| Name | Source | Role in D1 |
|---|---|---|
| Stored ORATS mid | Saved `{prefix}_mid` | Provenance. Copied unchanged. Not an input to any dollar term |
| Arithmetic fill midpoint | `expected_mid_fill_price(bid, ask, unit_quantity)` | The only midpoint used in \(B_{\mathrm{mid}}\), \(H_{\mathrm{body}}\), \(W_{\mathrm{mid}}\), \(H_{\mathrm{wing}}\), and \(C_{\mathrm{body}}\) |

`option_surface.py` documents why they can differ. `_mid_entry_cost` computes entry cost with `FillAssumption.mid()` rather than the stored `option.mid` field, because that stored mid comes from ORATS and may not be exactly \((\mathrm{bid}+\mathrm{ask})/2\) after ORATS rounding or smoothing. `expected_mid_fill_price` is that same fill model: bid plus half the spread, alpha 0.5 on both sides, the convention D0 used for `pnl_mid_at_cross_q`. Call it with all three arguments, including `unit_quantity`. Do not call it with bid and ask only. Do not read `trade_log_mid.pnl_total`. Do not invent a third midpoint. Do not replace the stored mid with the arithmetic mid.

Body prefixes: `body_put`, `body_call` (leg indexes 1 and 2). Wing prefixes: `put_wing`, `call_wing` (leg indexes 0 and 3).

Per-unit fields (`bid`, `ask`, `mid`, `fill_price_cross`, `entry_cash_per_unit`, `expiry_payoff_per_unit`) are premium per share. The saved `mid` is the stored ORATS midpoint. It is not the arithmetic fill midpoint in §4.1. Saved `pnl_total_leg`, `pnl_body_cross`, `pnl_wing_cross`, `pnl_legs_sum`, `pnl_mid_at_cross_q`, and `pnl_cross_official` are already dollars at \(Q\). Do not scale a dollar field by \(Q\) again. Do not treat `expiry_payoff_per_unit` as unsigned: D0 stored unsigned intrinsic times `unit_quantity`.

### 4.1 Arithmetic midpoint used in the formulas

For each required leg, with that leg’s saved `unit_quantity`:

\[
\mathrm{mid\_fill} = \texttt{expected\_mid\_fill\_price}(\texttt{bid}, \texttt{ask}, \texttt{unit\_quantity})
\]

`ask < bid`, a missing bid or ask, or a non-finite bid or ask is a blocker. Do not repair it and do not clip a negative spread to zero. A missing or non-finite stored `{prefix}_mid` is also an invalid input. A finite stored mid that differs from `mid_fill` is not.

After the arithmetic midpoint is computed, compare it with the saved `{prefix}_mid` without changing either value. Report, on the development book:

- the count of legs whose absolute difference exceeds \(10^{-6}\) premium points per share
- the maximum absolute difference across compared legs

Those two figures are diagnostics. They do not fail a gate, do not drop a trade, and do not rewrite `mid`. Financial reconciliation still uses saved D0 dollar columns, not the stored mid.

### 4.2 Components

\(Q\) is the saved `Q`. It must be finite and positive. `quantity_cross_signed` must be negative. Body `unit_quantity` must be \(-1\). Wing `unit_quantity` must be \(+1\). Anything else is a named blocker. Do not generalize the formulas to other unit sizes.

Let \(u\) be `unit_quantity`. Reuse the D0 cash sign: a sold body has entry cash \(-\mathrm{fill} \times |u|\); a bought wing has \(+\mathrm{fill} \times |u|\).

**\(B_{\mathrm{mid}}\)** — body midpoint P&L, dollars. Sum over `body_put` and `body_call`:

\[
Q \times \big(\texttt{expiry\_payoff\_per\_unit} - \text{signed mid entry cash}\big)
\]

Signed mid entry cash uses `mid_fill` from `expected_mid_fill_price(bid, ask, unit_quantity)` and the leg’s `unit_quantity`. It does not use stored `{prefix}_mid`. For a short body that cash is negative. Do not build \(B_{\mathrm{mid}}\) from `pnl_total_leg` or `pnl_body_cross`. Those are cross results.

**\(H_{\mathrm{body}}\)** — body execution concession, dollars. Sum over the two sold body legs:

\[
Q \times (\mathrm{mid\_fill} - \texttt{bid})
\]

On an uncrossed quote this is nonnegative because `mid_fill` is halfway from bid to ask. Do not clip it. Do not also subtract a fee.

**\(W_{\mathrm{mid}}\)** — wing midpoint premium, dollars. Sum over the two purchased wings:

\[
Q \times \mathrm{mid\_fill}
\]

This is premium paid at the midpoint, not wing P&L.

**\(H_{\mathrm{wing}}\)** — wing execution concession, dollars. Sum over the two purchased wings:

\[
Q \times (\texttt{ask} - \mathrm{mid\_fill})
\]

**\(W_{\mathrm{pay}}\)** — gross wing expiry payout, dollars. Sum over the two wings:

\[
Q \times \texttt{expiry\_payoff\_per\_unit}
\]

Do not multiply by `unit_quantity` again. Do not use `pnl_total_leg` or `pnl_wing_cross`. A wing `expiry_payoff_per_unit` below \(-10^{-6}\) disagrees with unit \(+1\) and unsigned intrinsic. Name it and stop. Do not flip the sign.

**Derived body cross**

\[
P_{\mathrm{body,cross}} = B_{\mathrm{mid}} - H_{\mathrm{body}}
\]

**Required identity**

\[
P_{\mathrm{fly,cross}} = B_{\mathrm{mid}} - H_{\mathrm{body}} - W_{\mathrm{mid}} - H_{\mathrm{wing}} + W_{\mathrm{pay}}
\]

That is the sprint-plan identity with implementation names: \(B_{\mathrm{mid}}\) is body midpoint profit, \(H_{\mathrm{body}}\) is body concession, \(W_{\mathrm{mid}}\) is wing midpoint premium, \(H_{\mathrm{wing}}\) is wing concession, and \(W_{\mathrm{pay}}\) is wing expiry payout.

Execution concession is already inside the official cross fill. Subtracting \(H_{\mathrm{body}}\) or \(H_{\mathrm{wing}}\) from a cross P&L that already used bid or ask would deduct the spread twice. The identity subtracts each concession once, from the midpoint terms, to reach the cross result.

### 4.3 Body midpoint credit, the ratio denominator

\[
C_{\mathrm{body}} = Q \times (\mathrm{mid\_fill}_{\mathrm{body\_put}} + \mathrm{mid\_fill}_{\mathrm{body\_call}})
\]

Dollars. Premium credit. Not capital at risk, not \(Q \times S_0\), not `capital_at_risk_dollars`, and not max loss.

### 4.4 Independent cross-checks

Compare derived values to **saved D0 columns**. Do not check a sum of the five calculated terms against a second writing of the same five terms.

| Derived | Saved D0 column | Why this is not a self-sum |
|---|---|---|
| \(P_{\mathrm{body,cross}}\) | `pnl_body_cross` | Quote concession and signed settlement versus D0’s saved body cross dollars |
| \(W_{\mathrm{pay}} - W_{\mathrm{mid}} - H_{\mathrm{wing}}\) | `pnl_wing_cross` | Premium, spread, and payout versus D0’s saved wing cross dollars |
| \(P_{\mathrm{fly,cross}}\) | `pnl_cross_official` | Identity versus official cross P&L |
| \(P_{\mathrm{fly,cross}}\) | `pnl_legs_sum` | Identity versus D0’s saved four-leg sum |
| \(B_{\mathrm{mid}} + W_{\mathrm{pay}} - W_{\mathrm{mid}}\) | `pnl_mid_at_cross_q` | Midpoint body P&L plus midpoint wing P&L versus D0’s saved midpoint repricing |

`pnl_body_cross + pnl_wing_cross` versus `pnl_cross_official` was D0’s job. D1 may record it as provenance. It is not a substitute for the five-term identity.

### 4.5 Tolerance

Inclusive, at trade, date, annual, and development-total levels:

\[
\max(\$0.01,\ 10^{-9} \times |\text{reference P\&L}|)
\]

The reference is the saved official cross P&L at that same level: the trade’s `pnl_cross_official`, the date sum, the year sum, or the development sum. It is not \(-\$146{,}279.85\).

A failed residual names the trade key or date, the check, the derived value, the reference, and the residual. It does not impute a repair.

### 4.6 Ratios

Report dollars first. Aggregate ratios are ratios of **summed** dollars, not averages of trade ratios.

| Ratio | Numerator | Denominator |
|---|---|---|
| Body execution concession / body midpoint credit | \(\sum H_{\mathrm{body}}\) | \(\sum C_{\mathrm{body}}\) |
| Wing midpoint premium / body midpoint credit | \(\sum W_{\mathrm{mid}}\) | \(\sum C_{\mathrm{body}}\) |
| Wing execution concession / body midpoint credit | \(\sum H_{\mathrm{wing}}\) | \(\sum C_{\mathrm{body}}\) |

Each displayed ratio names both sides. These are cost-burden ratios, not returns on capital.

A trade-level ratio is null when that trade’s \(C_{\mathrm{body}}\) is not finite or is not strictly positive. Reason: `non-positive body midpoint credit` or `non-finite body midpoint credit`. The trade stays in every dollar sum.

The date-level and development-total ratio is null when the summed denominator is not finite or is not strictly positive. The zero-short date has \(C_{\mathrm{body}} = 0\), so its ratios are null with reason `zero body midpoint credit`. Its dollar row remains, with zeros.

If any development trade used in a sum has a non-finite component, the run is `BLOCKED`. Do not drop that trade to make a ratio defined. Disclose the count of trades and dates whose own ratios are null, and the reasons, even when the aggregate ratio is defined.

\(H_{\mathrm{wing}} / W_{\mathrm{mid}}\), if shown, uses summed dollars, sits next to the dollar concession, and is labeled descriptive spread percentage. Null if summed \(W_{\mathrm{mid}}\) is not strictly positive. Not a headline.

---

## 5. Invalid inputs

Record the trade key or date, the field, and the reason. Do not impute a quote, a zero concession, or a cash P&L for a missing field.

A development trade is invalid if any of these fail: `pairing_ok` is not true; \(Q\) or `quantity_cross_signed` fails the sign and magnitude rules above; a required bid, ask, stored `mid`, or `expiry_payoff_per_unit` is missing or non-finite; unit quantities are not the accepted \(+1/-1\) pattern; a quote is crossed; a required D0 dollar field listed in §4.4 is missing or non-finite.

A finite stored `mid` that differs from `expected_mid_fill_price(bid, ask, unit_quantity)` is not an invalid input. Keep the stored value. Count the difference as in §4.1.

Invalid development input is `BLOCKED`. Do not publish an interpretation from a partial book. The exception list is evidence. It is not a reason to rerun the baseline.

A verified zero-short date has no trade fields to validate. Zeros there are the calendar rule, not an imputation.

---

## 6. Outputs

Write only under a new directory created by a later authorized run:

`C:/MomentumCVG_env/runs/sprint009_d1_<UTC timestamp>/`

Do not create that directory in this planning step. Do not write into the D0 directory or the official Sprint 006 directory.

| File | Contents |
|---|---|
| `input_inventory.json` | D0 path, D0 code SHA, parquet hashes, development counts, this run’s code SHA |
| `trade_decomposition.parquet` | One row per development short iron fly. Not later-period rows |
| `date_decomposition.parquet` | One row per development calendar date, including `2020-03-13` |
| `aggregate_dollars.json` | Development dollar sums and residuals |
| `aggregate_ratios.json` | Ratios of summed dollars, named denominators, null counts and reasons |
| `annual_decomposition.parquet` | One row per calendar year 2020–2023 |
| `waterfall_development.png` | One aggregate waterfall, dollars, after the identity passes |
| `d1_report.md` | `READY` or `BLOCKED`, the three answers, named residuals |
| `d1_report.json` | Gate results, the same totals, and the stored-versus-arithmetic mid diagnostic (count and maximum absolute difference). No later-period economic field |

Forbidden report keys: `development_minus_later_pnl`, `later_period_pnl`, `filter_result`, `protection_summary`, `primary_window_anchor_as_development`.

### 6.1 Trade table

Keys: `trade_date`, `ticker`, `direction`. `direction` is `short`. `window_label` is `development`.

Copy, do not recompute or overwrite, for provenance: `Q`, `quantity_cross_signed`, `entry_spot`, each leg’s stored `mid`, `pnl_cross_official`, `pnl_body_cross`, `pnl_wing_cross`, `pnl_legs_sum`, `pnl_mid_at_cross_q`.

Derived columns: `b_mid`, `h_body`, `w_mid`, `h_wing`, `w_pay`, `p_body_cross`, `p_fly_cross`, `c_body`, the five residuals in §4.4, and the three trade-level ratios plus `ratio_reason`.

### 6.2 Date table

One row for every development date on `short_calendar.parquet`. Build it by joining the calendar to trade sums. Do not use a group-by that drops empty dates.

Columns: `trade_date`, `short_book_class`, `n_trades`, the five dollar components, `p_body_cross`, `p_fly_cross`, `c_body`, date-level residuals against the summed saved D0 fields (zero against zero on the zero-short date), ratios, `ratio_reason`.

The `2020-03-13` row is `verified_zero_short`, `n_trades = 0`, every dollar field 0, ratios null.

### 6.3 Aggregate and annual tables

Aggregate dollars: development sums of the five components, \(P_{\mathrm{body,cross}}\), \(P_{\mathrm{fly,cross}}\), \(C_{\mathrm{body}}\), and the §4.4 residuals of those sums. Also the count of null trade-level ratios by reason.

Annual rows: 2020, 2021, 2022, 2023. Each has `n_dates`, `n_zero_short_dates`, `n_trades`, the same dollar sums, residuals against that year’s official sum, and ratios of that year’s summed dollars. 2020 includes the zero-short date. A missing year among those four is a blocker if the calendar has dates in that year and the table omitted them. Do not invent a year with no calendar dates. Do not add 2019 or 2024.

### 6.4 Waterfall

One chart, matplotlib with the Agg backend, written as `waterfall_development.png`. No new dependency. Draw only the development aggregate, in identity order:

1. \(B_{\mathrm{mid}}\)
2. \(-H_{\mathrm{body}}\)
3. \(-W_{\mathrm{mid}}\)
4. \(-H_{\mathrm{wing}}\)
5. \(+W_{\mathrm{pay}}\)
6. Total \(P_{\mathrm{fly,cross}}\)

Axis in dollars. No return axis. No later-period series. If the identity fails, do not write a chart that presents an economic story. Write the blocker in the report instead.

### 6.5 Memo

`d1_report.md` answers, in this order, only if every reconciliation gate passed:

1. **D1-A.** Development dollars of \(B_{\mathrm{mid}}\) and \(P_{\mathrm{body,cross}}\), then body concession dollars and body concession / body midpoint credit. State whether body midpoint margin remains positive after concession. Do not call that a trading decision.
2. **D1-B.** Development dollars of \(W_{\mathrm{mid}}\) and \(H_{\mathrm{wing}}\) separately, then each over body midpoint credit. If the descriptive wing spread percentage is shown, keep it beside the dollar concession.
3. **D1-C.** The identity residual at development total, the annual dollar pattern as description, and one paragraph on where the development cross result comes from in these five terms.

Required caveats in that memo: fees = 0; concession is not deducted twice; midpoint fills are not claimed attainable; the primary-window \(-\$146{,}279.85\) is not this result; wing payout frequency and filter rules are not answered here.

If any gate fails, the report says `BLOCKED` and lists the named gaps. It does not add an interpretation paragraph.

---

## 7. Gates

`READY` only if all of these pass. Any named gap is `BLOCKED`. There is no `READY_WITH_NARROW`. Stored-versus-arithmetic mid disagreement is not a gate. The report still includes its count and maximum absolute difference.

| Gate | Pass rule |
|---|---|
| Provenance | D0 directory, hashes, and development coverage match §Inputs |
| Inputs | No invalid development trade under §5 |
| Body cross | Every trade, date, year, and the development total: \(P_{\mathrm{body,cross}}\) matches `pnl_body_cross` |
| Wing cross | \(W_{\mathrm{pay}} - W_{\mathrm{mid}} - H_{\mathrm{wing}}\) matches `pnl_wing_cross` at those same levels |
| Identity | \(P_{\mathrm{fly,cross}}\) matches `pnl_cross_official` and `pnl_legs_sum` at those levels |
| Midpoint | \(B_{\mathrm{mid}} + W_{\mathrm{pay}} - W_{\mathrm{mid}}\), built from `expected_mid_fill_price(bid, ask, unit_quantity)`, matches `pnl_mid_at_cross_q` at those levels |
| Calendar | 209 date rows, one zero-short row on `2020-03-13` with zeros, no dropped date |
| Scope | No later-period economic field and no forbidden key |

---

## 8. Proposed footprint (do not create yet)

| Piece | Path | Role |
|---|---|---|
| Decomposition | `src/backtest/sprint009_d1_body_wing_decomposition.py` | One function: development rows and calendar in, five terms and residuals out. No I/O beyond what the runner asks |
| Runner | `scripts/run_sprint009_d1_decomposition.py` | Thin CLI. Reads the accepted D0 directory, writes the output directory, draws the chart only after the identity passes |
| Tests | `tests/unit/test_sprint009_d1_body_wing_decomposition.py` | Hand-calculated fixtures. No official parquet |

Reuse `expected_mid_fill_price(bid, ask, unit_quantity)` and the D0 cash-sign helper. Do not import `run_d0_validation`, `load_fill_primary_tables`, or `SurfaceRunner`.

### Invocation (not run now)

```text
$env:PYTHONPATH = "C:\MomentumCVG"
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint009_d1_decomposition.py
```

The script does not call the D0 runner and does not read the superseded D0 directory.

### Tests required before an official run

Hand-calculated, not official extracts. A valid control must pass the same verdict aggregator the runner will use. Each invalid case must fail the named gate and must not be `READY`.

- Sold body mid entry cash is negative; bought wing mid entry cash is positive. Both use `expected_mid_fill_price(bid, ask, unit_quantity)`. \(H_{\mathrm{body}}\) and \(H_{\mathrm{wing}}\) match \(Q\times(\mathrm{mid\_fill}-\mathrm{bid})\) and \(Q\times(\mathrm{ask}-\mathrm{mid\_fill})\).
- With the quotes in the appendix, \(P_{\mathrm{body,cross}} = B_{\mathrm{mid}} - H_{\mathrm{body}}\) and the five-term identity equals the fixture’s official cross P&L. Subtracting concession from an already-cross body P&L fails that check.
- A fixture whose saved P&L uses share-equivalent \(Q\) fails if the decomposition multiplies by 100.
- \(W_{\mathrm{pay}}\) uses signed `expiry_payoff_per_unit` once. A positive wing intrinsic increases \(W_{\mathrm{pay}}\) and does not get multiplied by `unit_quantity` again.
- A trade with \(C_{\mathrm{body}} = 0\) stays in the dollar table with a null ratio and a reason. It is not dropped.
- A two-date calendar with one valid fly and one verified zero-short date keeps both date rows. The zero-short dollars are zero, not missing. A later-period trade in the same fixture is absent from the development sums.
- A fixture with the appendix quotes and saved D0 P&L, but a stored `{prefix}_mid` that differs from `expected_mid_fill_price(bid, ask, unit_quantity)`, still passes. The stored mid is unchanged. The report counts the discrepancy and the maximum absolute difference. A genuine mismatch against saved D0 P&L, with those same quotes, still fails the identity or midpoint gate.

---

## 9. Acceptance of this design

Review can accept, amend, or reject the column formulas, the null-ratio rule, and the ban on a later-period economic table.

This design is accepted at `e109a9e`. The corrected implementation run is complete at `5669773` and returned `READY`. That evidence is **ACCEPTED** through `28f5ea4`. `0d63293` is superseded. Acceptance does not authorize D2 implementation.

D2–D5 stay as written in the sprint plan. This design does not change their formulas, windows, or freeze rule.

No accounting choice is left open for the implementer. The midpoint used in every dollar term is `expected_mid_fill_price(bid, ask, unit_quantity)`, not the stored ORATS mid. The denominator is body midpoint credit in dollars. The development benchmark is the development sum of `pnl_cross_official`, not the primary-window anchor.

---

## Appendix — hand numbers for the later unit fixture

Not a result. \(Q = 2\). Spot at expiry 100. Body strikes 100, so body `expiry_payoff_per_unit` is 0. Wings finish out of the money, so wing payoff is 0.

| Leg | bid | ask | arithmetic `mid_fill` | unit |
|---|---:|---:|---:|---:|
| `body_put` | 2.00 | 2.20 | 2.10 = `expected_mid_fill_price(2.00, 2.20, -1)` | −1 |
| `body_call` | 2.10 | 2.30 | 2.20 = `expected_mid_fill_price(2.10, 2.30, -1)` | −1 |
| `put_wing` | 0.40 | 0.60 | 0.50 = `expected_mid_fill_price(0.40, 0.60, +1)` | +1 |
| `call_wing` | 0.30 | 0.50 | 0.40 = `expected_mid_fill_price(0.30, 0.50, +1)` | +1 |

The regression case that must pass keeps these bids, asks, unit quantities, and saved D0 P&L, and sets one stored mid away from its arithmetic value, for example `body_put` stored mid \(2.11\). Economics still use \(2.10\). The stored \(2.11\) is not overwritten.

\[
\begin{align*}
B_{\mathrm{mid}} &= 8.60 \\
H_{\mathrm{body}} &= 0.40 \\
W_{\mathrm{mid}} &= 1.80 \\
H_{\mathrm{wing}} &= 0.40 \\
W_{\mathrm{pay}} &= 0 \\
P_{\mathrm{body,cross}} &= 8.20 \\
P_{\mathrm{fly,cross}} &= 6.00 \\
C_{\mathrm{body}} &= 8.60
\end{align*}
\]

Body concession ratio \(0.40 / 8.60\). Wing premium ratio \(1.80 / 8.60\). Wing concession ratio \(0.40 / 8.60\). Official cross P&L in that fixture must be \(6.00\). Multiplying \(Q\) by 100 must fail against that official figure.
