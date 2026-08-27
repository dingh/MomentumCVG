# Sprint 007 D0 — Investigation readiness and evidence contract

**Status:** `PROPOSED — AWAITING ACCEPTANCE`  
**Updated:** 2026-08-26  
**Agenda:** [`docs/agenda/current_sprint.md`](../agenda/current_sprint.md)  
**Working plan:** [`docs/agenda/sprint7_shortfall_plan.md`](../agenda/sprint7_shortfall_plan.md)  
**Prior evidence:** [`docs/sprint_memos/006_closeout.md`](../sprint_memos/006_closeout.md)

---

## Summary

| Item | D0 design decision |
|---|---|
| **Question** | Can accepted Sprint 006 artifacts support a trusted, fast shortfall investigation without a full economic rerun? |
| **Working hypothesis** | `READY_WITH_NARROW_ENABLING_CHANGE` — subject to D0 gates below; not a verdict until acceptance evidence is complete |
| **Method** | One readiness notebook for explanation/visualization; one small artifact-validation helper in an existing source package; one unit test |
| **Primary inputs** | Official run `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` + `run_receipt.json` |
| **Runtime target** | Identity/schema/pairing checks in minutes; no multi-hour `SurfaceRunner` rerun |
| **Authorization** | This design only. D0 implementation, granular economics, D1–D4 remain unauthorized until accepted. |

Schema inspection (columns and row counts only; no new granular economics opened) suggests paired mid/cross artifacts carry the fields required for D1–D3 under the sprint unit hierarchy. D0 must **confirm** that through gates — not assume it. Shortfall-bridge decomposition, synthetic bridge math, and aggregate P&L reconciliation belong to **D2**, not D0.

---

## Frozen decisions (D0)

- Official run directory and `run_receipt.json` SHA-256 manifest are the artifact identity authority.
- Primary bridge unit = aggregate included `pnl_total` (trade level) / `cycle_pnl_total` (date level); CAR is companion only. D0 confirms field presence only; bridge math is deferred to D2.
- Trade join key: `(trade_date, ticker, direction)` on `included_in_portfolio == True`.
- Leg join key for paired identity: `(trade_date, ticker, direction, expiry_date, option_type, strike, leg_index)` plus quote/settlement fields (see pairing checks).
- Output labels: **accepted calculation**, **exploratory description**, **future hypothesis** — applied in notebook prose and any saved tables.
- D0 stops after readiness evidence or a named blocker; it does not interpret economics beyond already accepted Sprint 006 facts.

---

## Minimum artifact checks

| Check | Source | Pass criterion |
|---|---|---|
| Run identity | `run_receipt.json` | `repo_sha=e205b9a…`, `result_complete=true`, `has_unresolved_failures=false`, contract SHA matches closeout |
| Artifact presence | Receipt `runs[].outputs` + `decision_report` | All 17 expected files exist on disk |
| Artifact integrity | Receipt per-file `sha256` | On-disk hash matches receipt for every listed artifact |
| Calendar completeness | `date_status_*` | 403 rows; `n_failed=0`, `n_valid_no_trade=0` per closeout |
| Trade unique-key integrity | `trade_log_*` | No duplicate `(trade_date, ticker, direction)` rows per fill |
| Included-key parity | `trade_log_*` (`included_in_portfolio`) | Primary window: zero mid-only / cross-only included keys (reconcile to accepted 9,212) |
| Paired leg identity | `leg_log_*` | For each included trade key, mid and cross leg sets match on `(expiry_date, option_type, strike, leg_index)` |
| Paired quote identity | `leg_log_*` | For matched legs: `bid`, `ask`, `mid` identical mid vs cross (quotes are pre-fill) |
| Paired settlement identity | `leg_log_*`, `trade_log_*` | For matched legs/trades: `exit_spot`, `expiry_payoff_per_unit`, and trade-level `exit_spot` identical mid vs cross |
| Fill distinction | `leg_log_*` | `fill_price` present on leg rows; differs mid vs cross where expected; `fill_label` distinct per file |
| Schema sufficiency | Parquet column lists | Required fields present (matrix below) |
| Decision report structure | `decision_report.json` | `by_fill`, `windows`, `fill_assumption_sensitivity`, `limitations` present — structure check only in D0 |

**Not a D0 pairing criterion:** equal total row counts mid vs cross. Row counts may differ across fills; pairing is by unique keys and matched leg/quote/settlement identity.

---

## Question-to-field sufficiency matrix

| Sprint question | Primary artifacts | Required fields | D0–D3 support |
|---|---|---|---|
| **D1 gross midpoint margin** | `trade_log_mid`, `date_summary_mid`, `decision_report.json` | `pnl_total`, `capital_at_risk_dollars`, `return_on_max_loss`, `direction`, `trade_date`, `included_in_portfolio`; date-level `cycle_pnl_total`, `cycle_return_on_capital_at_risk`, side splits | **Direct** — aggregates and side/location splits |
| **D1 breadth / concentration** | `trade_log_mid`, `decision_report.json` | `pnl_total`, `ticker`, `trade_date`; existing `concentration_primary_cross_top5` pattern reusable for mid | **Direct** with D1-frozen formulas |
| **D1 stability (year)** | `date_summary_mid`, `date_status_mid` | `trade_date`, `cycle_pnl_total`, `cycle_return_on_capital_at_risk`, `status` | **Direct** after window filter |
| **D2 dollar bridge** | Paired `trade_log_*`, `date_summary_*`, `leg_log_*` | Trade: `pnl_total`, `quantity`, `capital_at_risk_dollars`, `entry_cost_per_share`, `net_credit_per_share`, `max_loss_per_share`, `spread_cost_ratio`, `leg_spread_to_credit_ratio`; leg: `fill_price`, `bid`, `ask`, `mid` | **Direct** on matched keys — bridge math in D2 |
| **D2 fixed-position reference** | Paired `trade_log_*`, `leg_log_*` | `quantity`, leg `fill_price`, `pnl_per_share`, `pnl_total_leg`, leg `portfolio_quantity` | **Feasible** — reprice under frozen quantity convention in D2 design |
| **D2 Tier-A sizing chain** (conditional) | Paired `trade_log_*` + `pipeline._apply_tier_a_sizing` logic | Short: `max_loss_per_share`, `quantity`, `net_credit_per_share`; long: `quantity`, `entry_cost_per_share`; date-level short/long P&L splits | **Diagnostic trace** in D2 when sizing/capital component material |
| **D2 side / leg role** | `trade_log_*`, `leg_log_*` | `direction`, `instrument_type`, `leg_index`, `option_type`, wing/body strikes | **Direct** |
| **D3 execution requirement** | Paired `trade_log_*`, `leg_log_*`, `decision_report.json` | Leg quote fields (`bid`, `ask`, `mid`, `fill_price`); trade package cost ratios | **Requirement only** — break-even effective fill between mid and cross (D3) |
| **D3 attainability** | — | — | **Not in artifacts** — shadow experiment only |
| **Package fill probability** | — | — | **Blocked / unanswerable** from EOD ORATS snapshots |
| **Alternative payoff counterfactual** | — | — | **Blocked / unanswerable** without new structure run |
| **Pure Momentum/CVG IC** | `candidate_view_*`, `funnel_summary_*` | Selection funnel counts only | **Out of scope** per Sprint 006 limitations |

---

## Accounting units (confirmed from schema)

| Role | Field(s) | Level | D0 scope |
|---|---|---|---|
| Primary bridge | `pnl_total` (trade); `cycle_pnl_total` (date) | Included rows, primary window | Confirm fields exist |
| Sizing companion | `quantity`, `capital_at_risk_dollars` | Trade | Confirm fields exist |
| Secondary CAR | `return_on_max_loss`, `cycle_return_on_capital_at_risk` | Trade / date | Confirm fields exist |
| Leg fill / quote | `fill_price`, `bid`, `ask`, `mid` | Leg (`leg_log_*`) | Confirm fields exist; verify quote/settlement pairing |
| Fixed-position reference | Reprice using per-share economics × frozen quantity | Trade/leg | **D2** — not D0 |
| Bridge reconciliation tolerance | — | — | **D2** — frozen in D2 design |

Unexplained bridge residual outside tolerance is a **blocker** in D2 (§6.6); D0 does not perform bridge reconciliation.

---

## Architecture: notebook-first + minimal helper

```
notebooks/sprint007/d0_readiness.ipynb          ← committed, clean; narrative and charts only
src/backtest/sprint007_artifact_validation.py   ← identity, schema, unique-key, pairing checks
tests/unit/test_sprint007_artifact_validation.py ← synthetic receipt-style fixtures
```

No `shortfall_bridge` module, bridge decomposition, or bridge unit test in D0. Those are scoped to D2.

**Reuse (read-only inspection / import):**

| Module | Use |
|---|---|
| `surface_decision_report.py` | Included-key helper pattern, window filter conventions, report structure |
| `surface_metrics.py` | Documented CAR/P&L field names only |
| `pipeline.py` | Tier-A chain documented for D2; D0 does not trace sizing |
| `sprint006_baseline.py` | Artifact naming and run layout |

Do **not** reimplement sizing, financing, or CAR math in D0. Do **not** add a new `src/analysis/` package.

---

## D0 implementation steps (when authorized)

1. **Artifact-validation helper** — receipt SHA-256 verify, schema assert, unique-key checks, included-key parity, paired leg/quote/settlement identity.
2. **Unit test** — synthetic mid/cross frames proving helper pass/fail on key parity and leg identity (no bridge math).
3. **Readiness notebook** — call helper only; document matrix, unanswerable list, gate results, and provisional D0 verdict.
4. **Fresh-kernel execution** — run notebook on clean kernel; save executed outputs outside repo with hashes (see reproducibility).

---

## Reproducibility controls

- Pin `OFFICIAL_RUN_DIR` and receipt path in helper constants; no mutable producer cache inputs.
- Read Parquet with explicit column lists; no `SELECT *`.
- Deterministic sort on join keys before any output.
- **Committed notebook:** `notebooks/sprint007/d0_readiness.ipynb` checked in clean (no stale executed outputs in repo).
- **Executed evidence (outside repo):** `C:/MomentumCVG_env/runs/sprint007_d0_<timestamp>/` containing:
  - `d0_artifact_manifest.json` (paths, hashes, columns, gate pass/fail)
  - executed notebook (`.ipynb` with outputs)
  - rendered HTML export
  - `execution_receipt.json` with SHA-256 of executed `.ipynb` and `.html`, kernel timestamp, and repo SHA
- Notebook kernels call the validation helper only; no P&L or bridge logic in notebook cells.

---

## D0 gates (working hypothesis → verdict)

The working hypothesis `READY_WITH_NARROW_ENABLING_CHANGE` holds only if **all** gates pass:

| Gate | Pass | Fail → |
|---|---|---|
| G1 Receipt integrity | All hashes match | `BLOCKED_BY_SPECIFIC_EVIDENCE_GAP` |
| G2 Schema sufficiency | Matrix fields present | `BLOCKED` or narrow adapter (no economics change) |
| G3 Unique trade keys | No duplicates per fill | `BLOCKED` |
| G4 Included-key parity | Zero mid-only / cross-only primary keys | `BLOCKED` |
| G5 Leg / quote / settlement pairing | Matched included trades have identical leg keys, quotes, settlement | `BLOCKED` |
| G6 Fresh-kernel notebook | Executed notebook + HTML hashed outside repo | `BLOCKED` |
| G7 Scope | No bridge math, no granular economics opened | rescope |

If all gates pass with only the minimal helper added: **`READY_WITH_NARROW_ENABLING_CHANGE`**. If gates pass with zero production changes: **`READY_ARTIFACT_FIRST`**.

---

## Acceptance evidence (D0 complete)

- [ ] `d0_artifact_manifest.json` with all receipt hashes matched
- [ ] Question-to-field matrix recorded with no D1–D3 gap unless explicitly blocked
- [ ] Primary unit fields confirmed on real schema (no bridge reconciliation)
- [ ] Unique-key, included-key parity, and leg/quote/settlement pairing gates passed
- [ ] `tests/unit/test_sprint007_artifact_validation.py` passes
- [ ] Clean committed notebook + fresh-kernel executed `.ipynb`/`.html` outside repo with hashes in `execution_receipt.json`
- [ ] Explicit unanswerable list documented
- [ ] Final D0 verdict: `READY_ARTIFACT_FIRST` | `READY_WITH_NARROW_ENABLING_CHANGE` | `BLOCKED_BY_SPECIFIC_EVIDENCE_GAP`

---

## Unanswerable from artifacts (explicit)

1. Complex/package order fill probability, time-to-fill, skip rate, post-fill adverse selection.
2. Counterfactual payoff P&L for alternative structures or wing rules.
3. Full-universe Momentum IC or CVG incremental value (post-signal artifacts only).
4. Whether break-even execution quality is attainable in live markets (D3 requirement vs attainability split).

---

## Stop conditions

- Receipt hash mismatch or missing artifact → `BLOCKED`; stop before D1 design.
- Required column absent → `BLOCKED` or narrow schema adapter (must not change economics).
- Unique-key violation or included-key parity fails → stop.
- Leg/quote/settlement pairing fails → stop.
- Proposal expands into bridge decomposition, aggregate P&L reconciliation, D1 metrics, or `SurfaceRunner` rerun → rescope to D2+.

---

## Non-goals

- No shortfall-bridge module, synthetic bridge decomposition, or aggregate P&L reconciliation (D2).
- No granular midpoint/cross economics opened in D0.
- No D1–D4 designs or outputs.
- No code/notebook implementation in this design revision.
- No Sprint 006 contract or artifact mutation.

---

## Expected footprint (implementation)

| Path | Purpose |
|---|---|
| `src/backtest/sprint007_artifact_validation.py` | ~80–150 LOC |
| `tests/unit/test_sprint007_artifact_validation.py` | ~60–100 LOC |
| `notebooks/sprint007/d0_readiness.ipynb` | Readiness narrative (committed clean) |

---

## Inference boundary

D0 confirms **whether** artifact-first analysis is trustworthy and fast. It does not judge gross margin, shortfall mechanism, execution requirement, or next action. Bridge reconciliation and economic interpretation belong to D2–D4 respectively.
