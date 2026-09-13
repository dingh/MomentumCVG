# Sprint 009 D0 — Matched body/wing readiness

**Status:** `DRAFT — AWAITING REVIEW`  
**Updated:** 2026-09-13  
**Implementation:** **NOT STARTED.** This document does not mark D0 `READY` or complete.  
**Agenda:** [`docs/agenda/current_sprint.md`](../agenda/current_sprint.md)  
**Working plan:** [`docs/agenda/sprint9_short_body_wing_plan.md`](../agenda/sprint9_short_body_wing_plan.md) — scope accepted for D0 planning; D1–D5 unchanged  
**Official run (read-only):** `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z`  
**Accepted short-book anchor:** [`docs/sprint_memos/006_closeout.md`](../sprint_memos/006_closeout.md) §8 — primary window only  

---

## Question

> Can the accepted artifacts support a matched body/wing dataset that reproduces every selected short iron fly and correctly accounts for every trading date?

## Working hypothesis (not a verdict)

The official cross trade log, leg log, date status, and funnel summary appear to carry the fields this design requires. Sprint 007 already paired mid and cross leg identity. That is not this verdict. D0 still has to project the columns below, rebuild each included short iron fly, and classify every authoritative date. A missing field, a broken identity, or a calendar mismatch is `BLOCKED` with a named gap. It is not a cue to rerun the baseline.

## Authorization

This design is awaiting review. It does not authorize implementation, an official readiness run, or D1. Do not create helper, runner, test, or evidence files until that authorization is given separately.

---

## Scope boundary

Keep the frozen `42:8` population, official cross quantities, current wing selection, hold-to-expiry intrinsic settlement, and fees = 0.

D0 does not test profitability, thresholds, protection effectiveness, alternative wings, sizing, or brokerage margin. It does not open a development-versus-later economic comparison. Later-period rows may be read only to prove readiness and reconciliation. Dates before `2020-01-01` stay in the calendar and stay labeled `pre_study`. They are not part of the 3,322-trade anchor.

---

## 1. D0-A — Do we have the required inputs?

### 1.1 Provenance

| Item | Required value |
|---|---|
| Run directory | `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` |
| Receipt | `run_receipt.json` |
| Execution SHA | `e205b9acc5d0400aa38169de721acb7fb8268f29` |
| Contract SHA-256 | `4012b4a472448004e1a1b14e8814f506911ea0e263e35157b4e13e27ed51a54c` |
| Contract file | [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) — not edited |
| Sizing | `sizing_mode = conceptual` (Tier A). Logged `quantity` is already share-equivalent. `contract_multiplier = 100` is **not** applied again |

Reuse `sha256_file` and the official path constants in `src/backtest/sprint007_artifact_validation.py`. Re-verify receipt hashes before any panel build. Do not write into the official directory.

Do **not** call `run_d0_validation`, `load_fill_primary_tables`, or any Sprint 006/007/008 analysis runner. Those entry points either omit required columns or compute economics this step must not start.

### 1.2 Required artifacts

File names follow `expected_run_output_path`. Cross is the reference book. Mid is loaded only to prove the diagnostic midpoint book is separately sized.

| Role | File | Why |
|---|---|---|
| Receipt | `run_receipt.json` | Identity and hashes |
| Decision report | `decision_report.json` | Accepted primary short count and short `pnl_total` |
| Date status | `date_status_sprint006_baseline_v1_cross.parquet` | Whole-portfolio calendar |
| Funnel | `funnel_summary_sprint006_baseline_v1_cross.parquet` | `n_included_short` |
| Trade log | `trade_log_sprint006_baseline_v1_cross.parquet` | Selected shorts, \(Q\), spot, P&L, capital at risk |
| Leg log | `leg_log_sprint006_baseline_v1_cross.parquet` | Four legs, quotes, fills, cash, settlement |
| Mid trade log | `trade_log_sprint006_baseline_v1_mid.parquet` | Quantity contrast only. Not the diagnostic P&L |

### 1.3 Required columns

A column listing on 2026-09-13 confirmed these names exist on the official cross files. The older validators do **not** project all of them. Implementation must use this list. A missing name is `BLOCKED`, not an assumed default.

`sprint007_artifact_validation.TRADE_LOG_COLUMNS` has no `entry_spot`. Its `FUNNEL_SUMMARY_COLUMNS` has no `n_included_short`. `sprint007_d2_shortfall_bridge.TRADE_COLUMNS` adds `pnl_per_share` and still omits `entry_spot`. `sprint008_d0_input_readiness.TRADE_LOAD_COLUMNS` loads `entry_spot` but not `pnl_total` or `capital_at_risk_dollars`. None of those tuples is a sufficient schema check for this D0.

**Trade log, cross**

`trade_date`, `ticker`, `direction`, `included_in_portfolio`, `instrument_type`, `expiry_date`, `entry_spot`, `exit_spot`, `body_strike`, `quantity`, `capital_at_risk_dollars`, `pnl_total`, `fill_label`, `long_put_strike`, `long_call_strike`

`long_put_strike` and `long_call_strike` are diagnostics. Leg strikes are authoritative. A disagreement is a mismatch, not a reason to reselect wings.

**Leg log, cross**

`trade_date`, `ticker`, `direction`, `expiry_date`, `option_type`, `strike`, `leg_index`, `unit_quantity`, `portfolio_quantity`, `bid`, `ask`, `mid`, `fill_price`, `entry_cash_per_unit`, `expiry_payoff_per_unit`, `pnl_per_unit`, `pnl_total_leg`, `exit_spot`, `included_in_portfolio`, `fill_label`

**Funnel, cross**

`trade_date`, `n_included`, `n_included_long`, `n_included_short`, `date_status`, `date_reason`

**Date status, cross**

`trade_date`, `status`, `reason`

**Mid trade log, contrast only**

`trade_date`, `ticker`, `direction`, `included_in_portfolio`, `quantity`

### 1.4 Join keys

| Join | Key |
|---|---|
| Trade | `(trade_date, ticker, direction)` |
| Leg | trade key plus `(expiry_date, option_type, strike, leg_index)` |
| Calendar | `trade_date` |

Duplicate keys are a blocker. Do not keep the first row and drop the rest.

---

## 2. D0-B — Can every selected trade be reconstructed?

### 2.1 Population

One matched record per official **included** short iron fly:

- `included_in_portfolio` is true
- `direction == "short"`
- `instrument_type == "iron_fly"`
- `fill_label == "cross"`

Every such trade must have exactly four uniquely identified legs. Extra legs, missing legs, or a second row with the same leg key are blockers. Do not drop the trade.

### 2.2 Structure checks

Expected unit quantities, verified rather than assumed if a row disagrees:

| `leg_index` | Role | `option_type` | `unit_quantity` | Strike |
|---|---|---|---|---|
| 0 | long put wing | `put` | \(+1\) | strictly below `body_strike` |
| 1 | short put body | `put` | \(-1\) | equal to `body_strike` |
| 2 | short call body | `call` | \(-1\) | equal to `body_strike` |
| 3 | long call wing | `call` | \(+1\) | strictly above `body_strike` |

Both body legs share one strike. The put wing is lower. The call wing is upper. Expiry is the same on the trade and all four legs. Wing widths may differ. Do not require `long_call_strike - body_strike = body_strike - long_put_strike`. Do not call `build_ironfly_from_surface`.

### 2.3 Quantity

\[
Q = \lvert \text{official cross quantity} \rvert
\]

`quantity` on a short row is negative. \(Q\) is the magnitude and is frozen for every later comparison.

Logged quantity is already share-equivalent (`pipeline.py`: `pnl_total = abs(quantity) × pnl_per_share`). Do not multiply by `contract_multiplier`. A check that `portfolio_quantity = Q × unit_quantity` on each leg must pass. Applying 100 again would fail that check and is forbidden.

### 2.4 Cash, fills, and settlement

Fees stay 0. There is no fee column to add.

| Leg | Cross fill | Entry cash per unit |
|---|---|---|
| Sold body (`unit_quantity < 0`) | bid | \(-Q\)-scale later; per unit, \(-\mathrm{fill\_price} \times \lvert unit\_quantity \rvert\) |
| Bought wing (`unit_quantity > 0`) | ask | \(+\mathrm{fill\_price} \times \lvert unit\_quantity \rvert\) |

Reuse `expected_cross_fill_price` from `sprint007_artifact_validation.py`. Logged `fill_price` must match. Do not repair a crossed quote (`ask < bid`), a missing quote, or a non-finite quote. Those are blockers on an included trade.

`OptionLeg.calculate_intrinsic_value` returns an **unsigned** per-share intrinsic. The runner stores

\[
\text{expiry\_payoff\_per\_unit} = \text{unsigned intrinsic} \times \text{unit\_quantity}
\]

D0 must use that signed product. Do not compare the unsigned intrinsic to the logged payoff.

\[
\text{pnl\_per\_unit} = \text{expiry\_payoff\_per\_unit} - \text{entry\_cash\_per\_unit}
\]
\[
\text{pnl\_total\_leg} = Q \times \text{pnl\_per\_unit}
\]

### 2.5 Midpoint at frozen \(Q\)

A midpoint repricing uses the cross leg quotes and `expected_mid_fill_price` (bid + half spread), then the same \(Q\). It must be computed in the helper. It must not be read from `trade_log_mid.pnl_total`.

Also load mid-book `quantity` on the same trade keys. Record whether any absolute quantity differs. That difference is the proof the official midpoint book was sized separately. If every quantity happens to match, still do not use mid-book P&L as the diagnostic. The output column `pnl_mid_at_cross_q` is the diagnostic. `pnl_official_mid_book` is not an output of D0.

D0 checks that the diagnostic is computable. It does not interpret whether midpoint P&L is better.

### 2.6 Reconciliation

For each included short iron fly, body legs are indices 1 and 2. Wing legs are indices 0 and 3.

\[
P_{\mathrm{body,cross}} + P_{\mathrm{wing,cross}} = \sum_{4} \text{pnl\_total\_leg} = \text{official cross pnl\_total}
\]

Tolerance, inclusive:

\[
\max(\$0.01,\ 10^{-9} \times \lvert \text{official pnl\_total} \rvert)
\]

Apply it at trade level and to the primary-window aggregate.

**Primary-window anchor, not a development result.** Window `2020-01-01` through `2026-07-10` (`PRIMARY_START`, `PRIMARY_END`):

| Anchor | Accepted value | Source |
|---|---|---|
| Included short trades | 3,322 | [`006_closeout.md`](../sprint_memos/006_closeout.md) §8 |
| Short `pnl_total` | \(-146{,}279.85\) | same table |

The reconstructed primary-window count must equal 3,322. The reconstructed primary-window short `pnl_total` must match \(-146{,}279.85\) within the tolerance above. Also read `decision_report.json` → `by_fill.cross.primary.long_short.short`. If that block disagrees with the closeout anchor beyond the same tolerance, `BLOCKED`. Do not present either figure as the `2020–2023` development result.

---

## 3. D0-C — Is the short-side calendar complete?

`date_status.status` is the whole portfolio. `traded` can mean longs only. Never infer zero shorts from that status alone. Never turn a missing funnel row, a null `n_included_short`, or a missing trade into cash.

Authoritative dates are every `trade_date` on official cross `date_status`. A date that appears only on the funnel or only on the trade log is a blocker, not an extra date to invent and not a date to drop.

For each authoritative date, reconcile:

1. Whole-book `date_status.status` and `reason`.
2. Funnel `n_included_short`, and `date_status` / `date_reason` on that funnel row.
3. Count of included short iron-fly trade rows, and that each has a finite \(Q\).
4. Four required legs and finite settlement fields for each of those trades.

| Class | Rule |
|---|---|
| `verified_positive_short` | Funnel `n_included_short > 0`, that count equals the included short iron-fly rows, each row has finite \(Q\), and each has four valid legs |
| `verified_zero_short` | Funnel row present, `n_included_short == 0`, no included short trade rows, and whole-book status is `traded` or `valid_no_trade`. Includes valid long-only dates |
| `blocked` | `failed` status, missing funnel row, null count, count/row mismatch, missing legs, missing settlement, or any other inconsistency |

`pre_study` is a **window label**, not a fourth short-book class. Every date also receives one window label:

| Label | Dates |
|---|---|
| `pre_study` | official dates before `2020-01-01` |
| `development` | `2020-01-01` through `2023-12-31` |
| `later_period` | `2024-01-01` through `2026-07-10` |
| `unexpected` | any authoritative date after `2026-07-10` |

An `unexpected` date is `blocked`. Do not drop it.

Later-period access writes those classification and reconciliation fields only. It does not write a development-minus-later P&L, a filter result, or a protection summary.

Verified zero-short dates stay on the calendar with dollar contribution zero for later consumers. D0 itself does not build a P&L series. A blocked date is not rewritten as zero.

---

## 4. Outputs

Write only under a new directory:

`C:/MomentumCVG_env/runs/sprint009_d0_<UTC timestamp>/`

Do not create that directory in this planning step.

| File | Contents |
|---|---|
| `input_inventory.json` | Paths, receipt SHA, column lists, hash of each input file |
| `matched_short_iron_flies.parquet` | One row per included short iron fly, plus the four leg identities and the reconciliation residual |
| `short_calendar.parquet` | One row per authoritative date: window label, short-book class, funnel count, trade count, blocker reason |
| `d0_report.json` | Gate results, counts by window and class, primary-window residual versus the accepted anchor |
| `d0_report.md` | `READY` or `BLOCKED`, with named reasons |

`READY` requires every gate passed, including the primary-window anchor. Any named gap is `BLOCKED`. There is no `READY_WITH_NARROW_ENABLING_CHANGE` in this design. A missing column is a gap and a proposed resolution (project the column, or stop). It is not a baseline rerun.

### Invocation (not run now)

```text
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint009_d0_readiness.py
```

`PYTHONPATH` is the repo root. The script calls the helper and writes the directory above. It does not call `SurfaceRunner`.

### Proposed footprint (do not create yet)

| Piece | Path | Role |
|---|---|---|
| Helper | `src/backtest/sprint009_d0_body_wing_readiness.py` | Column projection, reconstruction, calendar class, report |
| Runner | `scripts/run_sprint009_d0_readiness.py` | Thin CLI. No economics beyond the identity checks |
| Tests | `tests/unit/test_sprint009_d0_body_wing_readiness.py` | Synthetic fixtures only |

Reuse, as functions, not by invoking analysis runners:

- `sha256_file`, `expected_run_output_path`, `expected_cross_fill_price`, `expected_mid_fill_price`, official path and SHA constants from `sprint007_artifact_validation.py`
- `PRIMARY_START`, `PRIMARY_END` from `surface_decision_report.py`

Do not import `run_d0_validation` or `load_fill_primary_tables` into the helper.

### Tests required before an official run

Synthetic cases, not official extracts:

- Sold body entry cash is negative; bought wing entry cash is positive.
- \(Q = \lvert quantity \rvert\). Multiplying by 100 fails the `portfolio_quantity` check.
- A swapped wing, a shared strike that is not the body, or a duplicate leg key fails. No row is dropped.
- Body plus wing plus four-leg sum failing the tolerance is a failed gate.
- A `traded` date with funnel `n_included_short == 0` and no short rows is `verified_zero_short`.
- A `traded` date with a missing funnel row, or `n_included_short` null, is `blocked`, not cash.
- A later-period date is labeled `later_period` and does not produce a comparative P&L field.

---

## 5. Acceptance of this design

Review can accept, amend, or reject the projection list, the asymmetric-wing check, the primary-window anchor, and the calendar classes.

Acceptance of this design still does not start implementation. An implementation run, if later authorized, returns `READY` or `BLOCKED`. This file must not be edited to say `READY` before that evidence exists.

D1–D5 stay as written in the sprint plan. This design does not change their formulas, windows, or freeze rule.
