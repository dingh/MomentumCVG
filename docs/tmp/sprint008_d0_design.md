# Sprint 008 D0 — Protocol freeze and input readiness

**Status:** `PROPOSED — AWAITING REVIEW`  
**Updated:** 2026-09-06  
**Agenda:** [`docs/agenda/current_sprint.md`](../agenda/current_sprint.md)  
**Working plan:** [`docs/agenda/sprint8_long_filter_plan.md`](../agenda/sprint8_long_filter_plan.md)  
**Frozen contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json)  
**Prior closeouts:** [`docs/sprint_memos/007_closeout.md`](../sprint_memos/007_closeout.md), [`docs/sprint_memos/006_closeout.md`](../sprint_memos/006_closeout.md)

---

## Question

> Can the existing Sprint 006 artifacts support the accepted long-only, equal-dollar measurement study, and what minimal work is needed to make the inputs ready?

## Working hypothesis (not a verdict)

**`READY_WITH_NARROW_ENABLING_CHANGE`** — official artifacts appear to contain the long candidate population, shared quotes, spots/strikes/exits, and settlement fields needed for \(M\), \(H\), \(S_0\), \(K\), \(X\), M1, M2, and a simple past-only M3 scale. A small read-only post-pass helper + readiness notebook is still required to freeze reconstruction, equal-dollar accounting checks, and coverage gates. No full `SurfaceRunner` rerun is indicated from inspection.

## Authorization

This document is **D0 design only**. Implementation, readiness execution, association analysis, profitability, and thresholds remain unauthorized until this design is accepted and D0 implementation is separately authorized.

---

## 1. Input identity and field mapping

### 1.1 Artifact identity

| Item | Value |
|---|---|
| Official run | `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` |
| Receipt | `run_receipt.json` (SHA-256 manifest authority) |
| Execution SHA | `e205b9acc5d0400aa38169de721acb7fb8268f29` |
| Contract | `configs/sprint006_baseline_v1.json` (immutable) |
| Primary helpers to reuse | `src/backtest/sprint007_artifact_validation.py` (identity/pairing); `src/backtest/sprint007_d2b_package_tradability.py` (`package_half_spread`, `midpoint_package_cashflow`, `package_width_to_cashflow`) |

D0 implementation must re-verify receipt identity/hashes before any panel build (reuse Sprint 007 D0 checks; do not mutate official files).

### 1.2 Inventory (confirmed present)

Paired mid/cross: `trade_log_*`, `leg_log_*`, `candidate_view_*`, `funnel_summary_*`, `date_status_*`, `date_summary_*`, plus `decision_report.json` / `run_summary_*`.

Inspection counts (schema/coverage only; no new economics):

- `trade_log` mid: 13,753 rows (= all signal candidates kept under diagnostics).
- Long rows: 6,958; `structure_ok` long: 6,802; long `no_tradeable_structure`: 156.
- In this official run, **every** `structure_ok` long is `included_in_portfolio` (0 `max_names_cap`, 0 `invalid_max_loss` on longs).
- `leg_log` covers exactly the `structure_ok` trade keys (not structure failures).
- Mid/cross included long keys: 6,802 / 6,802, symmetric difference 0.
- Primary window long constructable: 5,890 trades / 341 dates; `exit_spot` / `entry_spot` / `body_strike` NA count 0 on included longs.

### 1.3 Core quantity map

Join key: `(trade_date, ticker, direction)` with `direction == "long"`.  
Leg key: Sprint 007 `LEG_KEY` = `(trade_date, ticker, direction, expiry_date, option_type, strike, leg_index)`.

| Symbol | Definition (accepted) | Primary source | Units | Timing |
|---|---|---|---|---|
| Call/put quotes | `bid`, `ask`, `mid` on body legs | `leg_log_*` (shared mid≡cross quotes; D2B-verified pattern) | \$/share per leg | **Entry-known** |
| \(M\) | Midpoint debit of complete long straddle | Prefer recompute from unit legs: \(\sum q_u \cdot \mathrm{mid}\); cross-check `trade_log.entry_cost_mid_per_share` | \$/share | **Entry-known** |
| \(H\) | Mid→full-cross package concession | `package_half_spread` = \(0.5\sum \|q_u\|(\mathrm{ask}-\mathrm{bid})\) on the same unit legs; **not** from historical `quantity` or P&L deltas | \$/share | **Entry-known** |
| \(S_0\) | Entry spot | `trade_log.entry_spot` | \$/share | **Entry-known** |
| \(K\) | Common ATM strike | `trade_log.body_strike` (legs’ strikes must match) | \$/share | **Entry-known** |
| \(X\) | \(\lvert S_T - K \rvert\) | `abs(exit_spot - body_strike)`; cross-check \(\sum\) `expiry_payoff_per_unit` on unit long legs | \$/share | **Outcome (post-expiry)** |
| Fees | Explicit research fees | Protocol pin \(\mathrm{fees}_i=0\) | \$/share | Entry-known (constant) |

**Do not** use historical `trade_log.quantity` / short-financed Tier-A sizes for the research baseline.  
**Do not** derive \(H\) from mid vs cross `pnl_total` or resized Path-R artifacts.

### 1.4 Entry features vs outcome labels

| Entry-known (measurements / sizing inputs) | Later outcomes (labels only) |
|---|---|
| Quotes, \(M\), \(H\), \(S_0\), \(K\), \(\mathrm{fees}\), M1, M2, M3 scale inputs available at \(t\) | \(S_T\), \(X\), scenario net P&L, net return per dollar |

Future outcome availability must **not** redefine \(N\). Missing labels affect association coverage only.

### 1.5 Confirmed vs gaps

| Item | Status |
|---|---|
| Official identity + paired long quotes | **Confirmed** pattern (re-verify in D0 exec) |
| \(M\), \(H\) from unit legs; `entry_cost_mid_per_share` cross-check | **Confirmed** fields |
| \(S_0\), \(K\), \(X\) fields | **Confirmed** on constructable longs in inspection |
| Rich long panel beyond `candidate_view` | **`trade_log` is required** — `candidate_view` has only stage/reason codes |
| Legs for structure failures | **Absent** (expected; those names are outside constructable \(N\)) |
| Legs for hypothetical `max_names_cap` longs | **Unobserved in this run** (0 such rows). If any appear, quote coverage must be checked; absence would be a named gap |
| Historical quantities usable for equal-dollar baseline | **Must not** — intentional non-use, not a gap |
| Package fill attainability | **Out of scope / unanswerable** (Sprint 007 forbid-list) |

---

## 2. Candidate reconstruction

### 2.1 Definition of \(N\) (pre-experimental-filter)

For each entry date, rebuild the accepted population:

1. Rows with `direction == "long"`.
2. PIT/signal/CVG selection already embedded in official `trade_log` candidate rows (frozen `42:8` run).
3. Constructable: `structure_ok == True` (ATM long straddle built under `max_leg_spread_pct`).
4. Earnings: none (`had_earnings_nearby` unused; contract `earnings_exclusion_days = 0`).
5. Name cap: among constructable longs that day, sort by `signal_rank_pct` descending, `ticker` ascending; keep at most `max_names_per_side = 25`.

That capped set is \(N\). Equal-dollar stakes use this \(N\) **before** any measurement filter.

### 2.2 Why included-only is insufficient as a definition

Pipeline can mark constructable names `included_in_portfolio = False` for `max_names_cap` or `invalid_max_loss` after the cap. Those rules are **not** the Sprint 008 experimental filter.

Therefore D0 must **reconstruct** \(N\) from `structure_ok` + cap sort, then **report** agreement with:

- `included_in_portfolio` longs;
- `funnel_summary.n_constructable_long` / `n_included_long`.

### 2.3 Inspection result for the official run

In the official mid (and paired cross) artifacts, long `structure_ok` ≡ long included (6,802 keys; funnel constructable_long = included_long on all 403 dates). So **for this artifact set**, \(N\) coincides with included constructable longs, and `leg_log` covers \(N\).

D0 acceptance still requires the reconstruction check to pass explicitly, so a future artifact difference cannot silently change the protocol.

### 2.4 Baseline quantities

New research quantities only:

\[
q_i(h)=\frac{B/N}{M_i + h H_i + \mathrm{fees}_i}
\]

Historical short-financed `quantity` is ignored for sizing (may be loaded only for exclusion from reuse checks).

---

## 3. Protocol details (frozen for D0)

| Pin | Accepted value |
|---|---|
| Budget \(B\) | \$10,000 per entry date |
| Fees | \(\mathrm{fees}_i = 0\) (documented limitation) |
| Fractional quantities | Allowed |
| Scenarios | \(h \in \{0, 0.25, 0.50, 1\}\); **primary** \(h=1\); \(h=0\) diagnostic |
| Development | `2020-01-01` → `2023-12-31` |
| Evaluation | `2024-01-01` → `2026-07-10` (retrospective validation label) |
| Full-history companion | `2018-10-26` → `2026-07-10` (descriptive only) |

### 3.1 Units and denominators

- \(M\), \(H\), fees, \(X\): dollars per share.
- Stake per name: \(B/N\) dollars.
- Trade-level net return per dollar: scenario net \$ P&L / \((B/N)\).
- Portfolio return: date (or path) net \$ P&L / original \(B\), **including** unused cash.
- Within one \(h\): freeze \(q_i(h)\) across unfiltered vs threshold comparisons; rejected stakes stay cash; no redistribution.
- Across \(h\): quantities may differ.

### 3.2 Missingness and calendar policies

| Case | Policy |
|---|---|
| \(N=0\) | Full cash; trading P&L 0; keep date in calendar views |
| Missing quotes / non-finite \(M\) or \(H\) or \(M\le 0\) or \(H<0\) | Name **not** sizeable; document; do not impute. Prefer fail D0 readiness if any constructable long in primary window is affected |
| Missing measurement (e.g. \(S_0\le 0\) for M2; M3 cold-start) | Measurement NA; exclude from that measurement’s association only; **do not** remove from \(N\) |
| Missing outcome (\(S_T\) / \(X\)) | Exclude from trade-level association/labels; stake still counts in \(N\) and cash accounting unless D0 finds systematic absence (blocker) |
| Outcome-driven dropping of names from \(N\) | **Forbidden** |

Net P&L at scenario \(h\) (per share, long straddle research units):

\[
X - (M + h H + \mathrm{fees})
\]

Dollar P&L: \(q_i(h)\) times that per-share value (fees already in entry; no second fee layer).

---

## 4. Measurement readiness

| ID | Formula | Entry-known? | Input readiness | D0 recommendation |
|---|---|---|---|---|
| **M1** | \(H/M\) | Yes | Legs + \(M>0\) | **Include** (benchmark; D2B equivalent) |
| **M2** | \(H/S_0\) | Yes | \(H\), \(S_0>0\) | **Include** |
| **M3** | Past-only hurdle scale (below) | Yes (scale uses completed history only) | Needs prior completed \(X/S_0\) | **Include** — feasible without engine work |

### 4.1 Optional M3 — explicit simple scale (no profitability tuning)

At entry date \(t\), over long constructable trades with `expiry_date < t` and finite \(X_j, S_{0,j}>0\):

\[
\mu_t = \mathrm{mean}_j (X_j / S_{0,j})
\]

using **all** such completed observations available in the official run history before \(t\) (no rolling-window search). Cold-start: if fewer than **20** completed observations, M3 is missing.

\[
\mathrm{M3}_i = \frac{M_i / S_{0,i}}{\mu_t}
\]

Interpretation: mid debit as a fraction of spot, relative to the historical mean payoff/spot scale. Higher → richer entry price vs past realized payoff scale → expected worse net returns. Uses no same-trade \(X\).

Association criteria, consecutive-date block length, and multiplicity remain **D1**.

---

## 5. Minimal implementation and acceptance evidence

### 5.1 Proposed footprint (only after design acceptance + implementation authorization)

```
notebooks/sprint008/d0_input_readiness.ipynb
src/backtest/sprint008_d0_input_readiness.py   # narrow helper
tests/unit/test_sprint008_d0_input_readiness.py
```

Reuse: Sprint 007 artifact identity/pairing; D2B package half-spread / midpoint cashflow functions (import or thin wrap). No second economic engine; no `SurfaceRunner`; no threshold/association code.

### 5.2 D0 execution scope (when authorized)

1. Verify official run identity/hashes.
2. Build long candidate panel; reconstruct \(N\); reconcile to included/funnel.
3. Join unit legs; compute \(M\), \(H\); cross-check mid debit field.
4. Attach \(S_0\), \(K\), \(X\); compute M1, M2, M3 (coverage only).
5. Smoke equal-dollar accounting: \(\sum_i q_i(h)(M_i+hH_i)=B\) on dates with all sizeable names; cash identity under a dummy retain/reject mask; **no** profitability reporting.
6. Emit readiness tables outside repo; notebook narrative only.

### 5.3 Acceptance gates → verdict

| Verdict | When |
|---|---|
| `READY` | All identity/join/coverage/accounting checks pass with **zero** new production helpers beyond notebook-only scripts (unlikely given reuse needs) |
| `READY_WITH_NARROW_ENABLING_CHANGE` | Checks pass using the small helper/tests above; no input gap remains |
| `BLOCKED_BY_SPECIFIC_INPUT_GAP` | Named missing field/coverage (e.g. constructable longs without quotes; systematic missing \(S_T\); inability to reconstruct \(N\)) |

### 5.4 Required checks (pass/fail)

1. **Identity** — receipt SHA, execution SHA, artifact presence.
2. **Joins** — every \(N\) key has exactly two unit legs (call+put), shared mid/cross quotes, matching strikes/\(K\).
3. **Coverage** — primary-window constructable longs: finite \(M,H,S_0,K\); \(X\) finite for label coverage report; M1/M2 non-null rates; M3 cold-start rate.
4. **Reconstruction** — capped `structure_ok` long set equals declared \(N\); disclose equality/difference vs included.
5. **Accounting** — equal-stake consumption \(B/N\); within-\(h\) quantity freeze smoke; rejected cash not redistributed; historical `quantity` unused.
6. **Non-goals held** — no Spearman/groups/thresholds/P&L leaderboards in D0 outputs.

---

## 6. Non-goals / stop rule

- No D1 association, block bootstrap, or gate classification.
- No D2 thresholds or portfolio performance claims.
- No fill-attainability language.
- No edits to `configs/sprint006_baseline_v1.json` or official run files.

**Stop** after readiness evidence or a named blocker. D1 design is not authorized by D0 implementation alone.

---

## 7. Summary for reviewers

**Approach:** Artifact-first long panel from official `trade_log` + `leg_log`, reconstruct capped constructable \(N\), compute quote-based \(M/H\), build equal-dollar \(q_i(h)\), confirm M1/M2 and a simple past-only M3, prove accounting identities — via a narrow helper + notebook.

**Concrete gaps:** None identified that block M1/M2 or equal-dollar sizing on this official run, provided reconstruction and quote joins re-verify cleanly. Residual risks: (a) `candidate_view` alone is insufficient (mitigated by `trade_log`); (b) `max_names_cap` quote coverage is untested because count=0; (c) fees left at zero by protocol.

**Provisional verdict path:** `READY_WITH_NARROW_ENABLING_CHANGE`.
