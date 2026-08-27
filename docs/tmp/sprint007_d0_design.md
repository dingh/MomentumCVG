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
| **Proposed answer** | `READY_WITH_NARROW_ENABLING_CHANGE` |
| **Method** | Notebook-first for explanation/visualization; deterministic financial logic in tested source functions over read-only artifacts |
| **Primary inputs** | Official run `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` + `run_receipt.json` |
| **Runtime target** | Identity/schema/dry-join checks in minutes; no multi-hour `SurfaceRunner` rerun |
| **Authorization** | This design only. D0 implementation, granular economics, D1–D4 remain unauthorized until accepted. |

Schema inspection (columns and row counts only; no new granular economics opened) shows paired mid/cross artifacts with the fields required for D1–D3 under the sprint unit hierarchy. A narrow enabling layer is still required for reproducible identity checks, paired joins, and dry reconciliation smoke — not a second economic engine.

---

## Frozen decisions (D0)

- Official run directory and `run_receipt.json` SHA-256 manifest are the artifact identity authority.
- Primary bridge unit = aggregate included `pnl_total` (trade level) / `cycle_pnl_total` (date level); CAR is companion only.
- Join key for paired mid/cross analysis: `(trade_date, ticker, direction)` on `included_in_portfolio == True`.
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
| Fill pairing | `trade_log_*`, `leg_log_*` | Equal row counts mid vs cross; `fill_label` distinct per file |
| Included-key parity | `trade_log_*` (`included_in_portfolio`) | Primary window: zero mid-only / cross-only included keys (reconcile to accepted 9,212) |
| Schema sufficiency | Parquet column lists | Required fields present (matrix below) |
| Decision report anchor | `decision_report.json` | `by_fill`, `windows`, `fill_assumption_sensitivity`, `limitations` present; used for aggregate reconciliation targets only in dry smoke |

---

## Question-to-field sufficiency matrix

| Sprint question | Primary artifacts | Required fields | D0–D3 support |
|---|---|---|---|
| **D1 gross midpoint margin** | `trade_log_mid`, `date_summary_mid`, `decision_report.json` | `pnl_total`, `capital_at_risk_dollars`, `return_on_max_loss`, `direction`, `trade_date`, `included_in_portfolio`; date-level `cycle_pnl_total`, `cycle_return_on_capital_at_risk`, side splits | **Direct** — aggregates and side/location splits |
| **D1 breadth / concentration** | `trade_log_mid`, `decision_report.json` | `pnl_total`, `ticker`, `trade_date`; existing `concentration_primary_cross_top5` pattern reusable for mid | **Direct** with D1-frozen formulas |
| **D1 stability (year)** | `date_summary_mid`, `date_status_mid` | `trade_date`, `cycle_pnl_total`, `cycle_return_on_capital_at_risk`, `status` | **Direct** after window filter |
| **D2 dollar bridge** | Paired `trade_log_*`, `date_summary_*` | `pnl_total`, `quantity`, `capital_at_risk_dollars`, `fill_price`, `entry_cost_per_share`, `net_credit_per_share`, `max_loss_per_share`, `spread_cost_ratio`, `leg_spread_to_credit_ratio` | **Direct** on matched keys |
| **D2 fixed-position reference** | Paired `trade_log_*`, `leg_log_*` | `quantity`, `fill_price`, `pnl_per_share`, `pnl_total_leg`, leg `portfolio_quantity` | **Feasible** — reprice under frozen quantity convention in D2 design |
| **D2 Tier-A sizing chain** (conditional) | Paired `trade_log_*` + `pipeline._apply_tier_a_sizing` logic | Short: `max_loss_per_share`, `quantity`, `net_credit_per_share`; long: `quantity`, `entry_cost_per_share`; date-level short/long P&L splits | **Diagnostic trace** when sizing/capital component material; reuse production sizing rules, do not reimplement |
| **D2 side / leg role** | `trade_log_*`, `leg_log_*` | `direction`, `instrument_type`, `leg_index`, `option_type`, wing/body strikes | **Direct** |
| **D3 execution requirement** | Paired `trade_log_*`, `leg_log_*`, `decision_report.json` | Quote fields (`bid`, `ask`, `mid`, `fill_price`), package cost ratios above | **Requirement only** — break-even effective fill between mid and cross |
| **D3 attainability** | — | — | **Not in artifacts** — shadow experiment only |
| **Package fill probability** | — | — | **Blocked / unanswerable** from EOD ORATS snapshots |
| **Alternative payoff counterfactual** | — | — | **Blocked / unanswerable** without new structure run |
| **Pure Momentum/CVG IC** | `candidate_view_*`, `funnel_summary_*` | Selection funnel counts only | **Out of scope** per Sprint 006 limitations |

---

## Accounting units (confirmed from schema)

| Role | Field(s) | Level |
|---|---|---|
| Primary bridge | `pnl_total` (trade); `cycle_pnl_total` (date) | Included rows, primary window |
| Sizing companion | `quantity`, `capital_at_risk_dollars` | Trade |
| Secondary CAR | `return_on_max_loss`, `cycle_return_on_capital_at_risk`; View A via `surface_decision_report` | Trade / date / portfolio |
| Fixed-position reference | Reprice using stored per-share economics × frozen quantity | Trade (convention frozen in D2) |
| Reconciliation tolerance | `$0.01` absolute on primary-window aggregate `pnl_total` bridge; `$1e-9` relative for synthetic tests | Frozen in implementation |

Unexplained residual outside tolerance is a **blocker** (§6.6); not an attribution target.

---

## Architecture: notebook-first + tested source

```
notebooks/sprint007/d0_readiness.ipynb     ← narrative, charts, human review (no hidden logic)
src/analysis/sprint007_artifacts.py        ← identity, paired load, key parity, schema asserts
src/analysis/sprint007_shortfall_bridge.py ← bridge helpers (D0: dry smoke only; D2 expands)
tests/test_sprint007_artifacts.py          ← synthetic fixtures + receipt-style parity tests
tests/test_sprint007_shortfall_bridge.py   ← bridge math on tiny paired frames
```

**Reuse (mandatory):**

| Module | Use |
|---|---|
| `surface_decision_report.py` | Window filters, View A CAR, fill sensitivity, included-key helpers, report preconditions |
| `surface_metrics.py` | Cycle P&L/CAR aggregation conventions (`Σ pnl / Σ capital_at_risk`) |
| `pipeline.py` | Tier-A dependency semantics (`equal_max_loss` short → collected credit → long budget) for diagnostic trace |
| `sprint006_baseline.py` | Artifact naming and run layout only |

Do **not** reimplement sizing, financing, or CAR math independently. Extract only if notebook/import boundaries require it, with regression tests pinned to existing behavior.

---

## D0 implementation steps (when authorized)

1. **Identity manifest** — verify receipt SHA-256 for all artifacts; write `d0_artifact_manifest.json` (paths, hashes, row counts, columns).
2. **Schema assert** — fail fast if required columns missing.
3. **Dry paired join** — load column-pruned `trade_log` mid/cross; assert included-key set equality for primary window; report counts only.
4. **Dry bridge smoke** — on synthetic 2-trade paired frame, prove bridge decomposition returns expected components; on real artifacts, reconcile **one** accepted aggregate only (`decision_report.json` primary mid/cross `pnl_total` or documented equivalent) without opening new granular tables.
5. **Notebook** — document checks, matrix, unanswerable list, and proposed D0 verdict for human review.

---

## Reproducibility controls

- Pin `OFFICIAL_RUN_DIR` and receipt path in module constants; no mutable producer cache inputs.
- Read Parquet with explicit column lists; no `SELECT *`.
- Deterministic sort on join keys before any output.
- Save D0 evidence under `C:/MomentumCVG_env/runs/sprint007_d0_<timestamp>/` (outside repo); commit only manifest summary if needed.
- Notebook kernels must call source functions; no P&L logic duplicated in notebook cells.

---

## Acceptance evidence (D0 complete)

- [ ] `d0_artifact_manifest.json` with all receipt hashes matched
- [ ] Question-to-field matrix recorded (above) with no D1–D3 gap unless explicitly blocked
- [ ] Primary unit = dollar P&L confirmed on real schema
- [ ] Dry paired-key parity: primary window zero mid-only / cross-only included keys
- [ ] Dry aggregate reconciliation to **one** accepted Sprint 006 headline (from `decision_report.json` only)
- [ ] Focused tests pass for artifact loader and synthetic bridge
- [ ] Explicit unanswerable list documented (package fills, counterfactual structures, pure signal IC)
- [ ] D0 verdict recorded: `READY_ARTIFACT_FIRST` | `READY_WITH_NARROW_ENABLING_CHANGE` | `BLOCKED_BY_SPECIFIC_EVIDENCE_GAP`

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
- Included-key parity fails → stop; investigate before any bridge interpretation.
- Dry aggregate reconciliation fails tolerance → stop; no D1 design.
- Proposal expands into D1 metrics, filter search, or `SurfaceRunner` rerun → rescope.

---

## Non-goals

- No granular midpoint/cross economics beyond one accepted aggregate reconciliation smoke.
- No D1–D4 designs, notebooks, or outputs.
- No code/notebook implementation in this design commit.
- No Sprint 006 contract or artifact mutation.

---

## Expected footprint (implementation)

| Path | Purpose |
|---|---|
| `src/analysis/sprint007_artifacts.py` | ~150–250 LOC |
| `src/analysis/sprint007_shortfall_bridge.py` | ~100–200 LOC (D0 smoke scope) |
| `tests/test_sprint007_artifacts.py` | ~80–120 LOC |
| `tests/test_sprint007_shortfall_bridge.py` | ~80–120 LOC |
| `notebooks/sprint007/d0_readiness.ipynb` | Readiness narrative |

---

## Inference boundary

D0 confirms **whether** artifact-first analysis is trustworthy and fast. It does not judge gross margin, mechanism, execution requirement, or next action. Those require accepted D1–D4 designs respectively.
