# Sprint 008 D0 — Protocol freeze and input readiness

**Status:** `ACCEPTED`  
**Accepted:** 2026-09-07 (implementation authorization)  
**Updated:** 2026-09-07  
**Agenda:** [`docs/agenda/current_sprint.md`](../agenda/current_sprint.md)  
**Working plan:** [`docs/agenda/sprint8_long_filter_plan.md`](../agenda/sprint8_long_filter_plan.md)  
**Evidence review:** [`docs/tmp/sprint008_d0_evidence_review.md`](sprint008_d0_evidence_review.md) — `BLOCKED_BY_SPECIFIC_INPUT_GAP`; awaiting review  
**Official evidence:** `C:/MomentumCVG_env/runs/sprint008_d0_20260907T193835Z/`  
**Frozen contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json)  
**Prior closeouts:** [`docs/sprint_memos/007_closeout.md`](../sprint_memos/007_closeout.md), [`docs/sprint_memos/006_closeout.md`](../sprint_memos/006_closeout.md)

---

## Question

> Can the existing Sprint 006 artifacts support the accepted long-only, equal-dollar measurement study, and what minimal work is needed to make the inputs ready?

## Working hypothesis (not a verdict)

**`READY_WITH_NARROW_ENABLING_CHANGE`** — official artifacts appear to contain the long candidate population, shared quotes, spots/strikes/exits, and settlement fields needed for \(M\), \(H\), \(S_0\), \(K\), \(X\), M1, M2, and a simple past-only M3 scale. A small read-only post-pass helper + readiness notebook is still required to freeze reconstruction, equal-dollar accounting checks, and coverage gates. No full `SurfaceRunner` rerun is indicated from inspection.

## Authorization

This design is **accepted**. D0 implementation and the official readiness run are complete; evidence awaits review. Association, profitability, thresholds, and D1 remain unauthorized.

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
| Call/put quotes | `bid`, `ask` on body legs; stored ORATS `mid` is **metadata only** | `leg_log_*` (shared bid/ask mid≡cross; D2B-verified pattern) | \$/share per leg | **Entry-known** |
| \(M\) | Midpoint debit of complete long straddle: \(M=\sum q_u\,(\mathrm{bid}+0.5(\mathrm{ask}-\mathrm{bid}))\) via D2B `midpoint_package_cashflow` | Unit legs’ **bid/ask** only — **not** stored `mid` | \$/share | **Entry-known** |
| \(H\) | Mid→full-cross package concession | D2B `package_half_spread` = \(0.5\sum \|q_u\|(\mathrm{ask}-\mathrm{bid})\) on the same unit legs; **not** from historical `quantity` or P&L deltas | \$/share | **Entry-known** |
| \(S_0\) | Entry spot | `trade_log.entry_spot` | \$/share | **Entry-known** |
| \(K\) | Common ATM strike | `trade_log.body_strike` (legs’ strikes must match) | \$/share | **Entry-known** |
| \(X\) | \(\lvert S_T - K \rvert\); when outcomes are available must reconcile to \(\sum\) unit-leg `expiry_payoff_per_unit` | `abs(exit_spot - body_strike)` vs leg payoff sum | \$/share | **Outcome (post-expiry)** |
| Fees | Explicit research fees | Protocol pin \(\mathrm{fees}_i=0\) | \$/share | Entry-known (constant) |

**Authoritative midpoint:** compute \(M\) with `midpoint_package_cashflow(unit_quantity, bid, ask)`. Stored ORATS `mid` must not define fills or \(M\).

**Required reconciliations (every \(N\) key):**

1. \(M\) vs `trade_log.entry_cost_mid_per_share` within a predeclared absolute tolerance (freeze in implementation; default \(10^{-8}\) dollars/share unless artifact scale requires wider).
2. \(M + H\) equals the **total ask debit** of the two unit long legs: \(\sum q_u\,\mathrm{ask}\) (with \(q_u=+1\) on call and put), same tolerance.
3. \(H \ge 0\) and \(M > 0\) for sizeable names.

**Do not** use historical `trade_log.quantity` / short-financed Tier-A sizes for the research baseline.  
**Do not** derive \(H\) from mid vs cross `pnl_total` or resized Path-R artifacts.  
**Do not** substitute stored `mid` for \(\mathrm{bid}+0.5(\mathrm{ask}-\mathrm{bid})\).

### 1.4 Entry features vs outcome labels

| Entry-known (measurements / sizing inputs) | Later outcomes (labels only) |
|---|---|
| Quotes (bid/ask), \(M\), \(H\), \(S_0\), \(K\), \(\mathrm{fees}\), M1, M2, M3 (when history sufficient) | \(S_T\), \(X\), scenario net P&L, net return per dollar |

Future outcome availability must **not** redefine \(N\) or invested stakes. Missing outcomes stay **unknown** (see §3.2).

### 1.5 Confirmed vs gaps

| Item | Status |
|---|---|
| Official identity + paired long quotes | **Confirmed** pattern (re-verify in D0 exec) |
| \(M\), \(H\) from unit bid/ask via D2B helpers; `entry_cost_mid_per_share` and \(M+H=\)ask-debit checks | **Confirmed** fields; re-verify identities in D0 exec |
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

### 3.2 Missingness and calendar policies (deterministic)

| Case | Classification | Pass/fail / handling |
|---|---|---|
| \(N=0\) | Allowed calendar state | Full cash; trading P&L 0; keep date in calendar views — **not** a readiness failure |
| Required entry inputs for a constructable long in the primary window: two unit legs with finite bid/ask; finite \(M>0\); finite \(H\ge 0\); \(M+H\) reconciles to ask debit; finite \(S_0>0\); finite \(K\) | **Required-input failure** if any name in \(N\) fails | D0 readiness **FAIL** → `BLOCKED_BY_SPECIFIC_INPUT_GAP` (name the keys). Do not impute. Do not silently drop from \(N\) to “pass” |
| Same required-input failures outside primary window | Report coverage | Do not block solely on pre-primary holes unless they prevent M3 history construction for primary entries |
| M2 undefined only if \(S_0\le 0\) | Required-input failure when in \(N\) (primary) | Same as required \(S_0>0\) above |
| M3 cold-start / insufficient history / non-finite or non-positive \(\mu_t\) | **Allowed missing measurement** | M3 = NA; **do not** change \(N\) or allocations; report cold-start rate — **not** a readiness failure by itself |
| Missing outcome (\(S_T\) or \(X\) non-finite) on a name in \(N\) | **Allowed unknown outcome** | Keep name in \(N\); keep invested stake \(B/N\) and \(q_i(h)\); mark trade-level return **unknown**; mark any portfolio aggregate that would include that trade **incomplete**. **Never** treat missing outcome as cash, zero payoff, or zero P&L. D0 readiness **FAIL** if primary-window outcome-missing rate \(> 0\) among sizeable \(N\) names (inspection expected 0; any positive rate is an input gap) |
| Outcome-driven dropping of names from \(N\) | Forbidden | Always **FAIL** if observed |

Net P&L at scenario \(h\) is defined **only** when \(X\) is finite:

\[
X - (M + h H + \mathrm{fees})
\]

Dollar P&L: \(q_i(h)\) times that per-share value (fees already in entry; no second fee layer). If \(X\) is missing, P&L is unknown — not zero.

---

## 4. Measurement readiness

| ID | Formula | Entry-known? | Input readiness | D0 recommendation |
|---|---|---|---|---|
| **M1** | \(H/M\) | Yes | Legs + \(M>0\) | **Include** (benchmark; D2B equivalent) |
| **M2** | \(H/S_0\) | Yes | \(H\), \(S_0>0\) | **Include** |
| **M3** | Full-cross payoff hurdle / past payoff scale (below) | Yes (scale uses completed history only) | Rolling completed \(X/S_0\) history | **Include** — feasible without engine work |

### 4.1 M3 — full-cross hurdle vs rolling past-only scale (no profitability tuning)

**Score (fixed across all execution-cost sensitivities \(h\)):**

\[
\mathrm{M3}_i = \frac{M_i + H_i + \mathrm{fees}_i}{S_{0,i}\,\mu_t}
\]

Numerator is the **full-cross** all-in entry hurdle (\(h=1\)), including fees. Do **not** replace \(H_i\) with \(h H_i\) when reporting M3 under intermediate scenarios — the measurement definition stays the full-cross hurdle so M3 is comparable across \(h\).

**Historical pool and window**

- Pool candidates: restrict first to **`in_N == True`** (capped long \(N\)), then apply the eligibility rules below. Capped-out constructable longs do **not** count toward the minimum history or enter \(\mu_t\).
- Observation \(j\) is eligible for \(\mu_t\) only if:
  - `expiry_date` \(< t\) (strict completed-before-entry cutoff; holding period finished);
  - `entry_date` \(\in [t - L,\ t)\) (rolling lookback; left-closed, right-open);
  - finite \(X_j \ge 0\) (include valid **zero** payoffs);
  - finite \(S_{0,j} > 0\).
- **Fixed lookback** \(L = 364\) calendar days (52 weeks). Not tuned to profitability or coverage after looking at results.
- \[
  \mu_t = \mathrm{mean}_j (X_j / S_{0,j})
  \]
  over eligible \(j\).

**Missing M3** when any of: fewer than **20** eligible historical observations; \(\mu_t\) non-finite; or \(\mu_t \le 0\). Missing M3 does **not** change \(N\) or allocations.

Interpretation: full-cross entry cost as a fraction of spot, relative to the recent completed mean payoff/spot scale. Higher → higher hurdle vs past realized payoff scale → expected worse net returns. Uses no same-trade \(X\).

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
3. Join unit legs; compute \(M\) via `midpoint_package_cashflow(unit_quantity, bid, ask)` and \(H\) via `package_half_spread`; reconcile to `entry_cost_mid_per_share` and to ask debit \(M+H\).
4. Attach \(S_0\), \(K\), \(X\); compute M1, M2, M3 (coverage / missingness only — no association).
5. Smoke equal-dollar accounting: \(\sum_i q_i(h)(M_i+hH_i+\mathrm{fees}_i)=B\) on dates with all sizeable names; cash identity under a dummy retain/reject mask; **no** profitability reporting.
6. Emit readiness tables outside repo; notebook narrative only.

### 5.3 Acceptance gates → verdict

| Verdict | When |
|---|---|
| `READY` | All identity/join/coverage/accounting checks pass with **zero** new production helpers beyond notebook-only scripts (unlikely given reuse needs) |
| `READY_WITH_NARROW_ENABLING_CHANGE` | Checks pass using the small helper/tests above; no required-input or outcome-coverage gap remains |
| `BLOCKED_BY_SPECIFIC_INPUT_GAP` | Any §3.2 required-input failure or primary-window outcome-missing rate \(> 0\); inability to reconstruct \(N\); \(M\)/ask-debit reconciliation failure |

Allowed M3 cold starts alone do **not** force `BLOCKED_*`.

### 5.4 Required checks (pass/fail)

1. **Identity** — receipt SHA, execution SHA, artifact presence.
2. **Joins / per-leg quotes** — every \(N\) key has exactly two `+1` unit legs (one call, one put), matching leg expiry, leg strikes matching the trade body strike/expiry, shared mid/cross **bid/ask**. **Per-leg quote check (explicit):** each leg must have finite bid/ask and **`ask >= bid`**. Package-level \(H\ge 0\) alone is insufficient (a crossed call can be masked by a wide put).
3. **Midpoint authority** — \(M\) from D2B midpoint helper on bid/ask; stored `mid` not used as \(M\); \(M\) reconciles to `entry_cost_mid_per_share`; \(M+H\) reconciles to unit ask debit.
4. **Required-input coverage** — primary-window \(N\): 100% finite \(M>0\), \(H\ge 0\), per-leg `ask>=bid`, \(S_0>0\), \(K\), body/leg strike+expiry match; else FAIL.
5. **Outcome coverage** — primary-window \(N\): 100% finite \(X\ge 0\); when outcomes are available, \(X\) must reconcile to the sum of recorded unit-leg `expiry_payoff_per_unit`; else FAIL. Missing outcomes never coerced to 0 P&L/cash in any smoke path.
6. **Measurements** — M1/M2 defined wherever required inputs pass; M3 missingness equals cold-start / bad \(\mu_t\) only; missing M3 leaves \(N\) and \(q_i(h)\) unchanged.
7. **Reconstruction** — capped `structure_ok` long set equals declared \(N\); disclose equality/difference vs included.
8. **Accounting** — equal-stake consumption \(B/N\) across \(h\in\{0,0.25,0.50,1\}\); within-\(h\) quantity freeze smoke; rejected cash not redistributed; all-rejected dates leave invested \(=0\) and cash \(=B\) with no exception; historical `quantity` unused.
9. **Non-goals held** — no Spearman/groups/thresholds/P&L leaderboards in D0 outputs.

### 5.5 Focused unit-test cases (design freeze)

| Test | Expect |
|---|---|
| Synthetic two-leg bid/ask → \(M=\) `midpoint_package_cashflow`, \(H=\) half-spread, \(M+H=\) ask debit | Pass within tolerance |
| Stored `mid` deliberately ≠ bid/ask midpoint | \(M\) still follows bid/ask helper (ignores stored mid) |
| `entry_cost_mid_per_share` mismatch beyond tolerance | Readiness check fails |
| \(N\) reconstruction with >25 constructable longs | Cap keeps 25 by rank/ticker; overflow excluded from \(N\) |
| M3 with 19 eligible history rows | M3 missing; allocations unchanged |
| M3 with 20 rows including \(X=0\) | Zero payoff included in \(\mu_t\); M3 finite if \(\mu_t>0\) |
| M3 window respects `expiry < t` and entry in \([t-364,t)\) | Future/`expiry\ge t` / outside lookback excluded |
| Missing \(X\) on one invested name | Stake remains; P&L unknown; portfolio aggregate marked incomplete; not zero-filled |
| Dummy measurement reject under fixed \(q_i(h)\) | Rejected \(B/N\) stays cash; no redistribution |
| All candidates rejected under each \(h\in\{0,0.25,0.50,1\}\) | Invested \(=0\); cash \(=B\); no exception |
| Crossed call masked by wide put (\(H>0\) package-level) | Per-leg `ask>=bid` fails; readiness blocked |
| Body strike ≠ matching unit-leg strikes | Join / required-input readiness fails |
| Capped-out (`in_N=False`) history row with large \(X/S_0\) | Does not satisfy min-20 history; does not change \(\mu_t\) |

---

## 6. Non-goals / stop rule

- No D1 association, block bootstrap, or gate classification.
- No D2 thresholds or portfolio performance claims.
- No fill-attainability language.
- No edits to `configs/sprint006_baseline_v1.json` or official run files.

**Stop** after readiness evidence or a named blocker. D1 design is not authorized by D0 implementation alone.

---

## 7. Summary for reviewers

**Approach:** Artifact-first long panel from official `trade_log` + `leg_log`, reconstruct capped constructable \(N\), compute bid/ask \(M/H\) via D2B helpers (with ask-debit reconciliation), build equal-dollar \(q_i(h)\), confirm M1/M2 and fixed full-cross M3 with 364-day rolling history, prove accounting identities — via a narrow helper + notebook.

**Concrete gaps:** None identified that block M1/M2/equal-dollar sizing on this official run, provided reconstruction, bid/ask midpoint authority, and 100% primary outcome coverage re-verify cleanly. Residual risks: (a) `candidate_view` alone is insufficient (mitigated by `trade_log`); (b) `max_names_cap` quote coverage is untested because count=0; (c) fees left at zero by protocol; (d) early-window M3 cold starts are allowed and must not alter \(N\).

**Provisional verdict path:** `READY_WITH_NARROW_ENABLING_CHANGE`.
