# Current sprint — 002

**Updated:** 2026-05-28  
**Status:** Planning — scope to finalize with HD  
**Mode:** TBD (expected: **Build**, scoped to approved P0 items)

---

## Context from Sprint 001 (closed)

Sprint 001 delivered audits, data-flow map, and one synthetic runner verification test. **335** unit tests green.

HD exit: structure is in place; assembly layer is well tested; **backtest results are not yet trustworthy** for go/no-go because portfolio/risk/metrics and several contracts are partial or missing. Expect **3–4 sprints** to close P0/P1 gaps one theme at a time.

**Traceability:**

| Sprint 001 artifact | Use in 002 |
|---------------------|------------|
| [repo_audit.md](../repo_audit.md) | P0/P1 backlog and Sprint 002–004 draft sequencing |
| [correctness_audit.md](../correctness_audit.md) | What to test as each gap is fixed |
| [surface_runner_data_flow.md](../surface_runner_data_flow.md) | Where to plug sizing, caps, metrics |
| [sprint_memos/001_repo_audit_verification.md](../sprint_memos/001_repo_audit_verification.md) | Closed decisions and test proof boundary |

---

## Goal (draft — discuss before locking)

Make the surface path **closer to decision-quality** by implementing the highest P0 blocker(s) from `repo_audit.md`, with tests for each behavior change.

**Not in scope until agreed:** Tier B full-sample backtest, fly vs condor matrix, broker/shadow.

---

## Candidate focus areas (pick with HD)

From [repo_audit.md § Recommended Sprint 002](../repo_audit.md#recommended-sprint-002004-backlog):

| Priority | Item | Why |
|----------|------|-----|
| P0 | Portfolio/risk/dollar-PnL (`contracts`, `pnl_dollars`, `return_on_max_loss`) | Core trust gap |
| P0 | Fix `run_surface_search.py` CLI / `SurfaceDataPaths` wiring | Unblocks real smoke |
| P0 | 50 total position cap (not per-side only) | v1 spec |
| P0 | Trade-date schedule vs surface/feature alignment | Silent skip risk |
| P0 | Surface coverage report | Know what Tier B would actually measure |

---

## Success criteria

_To be filled when Sprint 002 scope is approved._

---

## Agent instructions

1. Read this file and [v1_spec_pins.md](../v1_spec_pins.md) at session start.
2. Do not start implementation until HD approves Sprint 002 scope and mode.
3. One P0 theme per sprint preferred over a large multi-refactor.

---

## Previous sprint

Sprint 001 closed 2026-05-28 — see [sprint_memos/001_repo_audit_verification.md](../sprint_memos/001_repo_audit_verification.md).
