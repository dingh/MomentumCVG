# Current sprint — 002

**Updated:** 2026-05-28  
**Status:** Active — design + contract tests  
**Mode:** **Audit / Design** (contracts and tests define truth; minimal prod changes only to align with signed-off contracts)

---

## Goal

Nail down the **surface backtest data model** and **per-component I/O contracts** so development and verification proceed **one component at a time**, not only via end-to-end surface judgment.

Sprint 002 produces the **source-of-truth documentation** (contracts, full data flow with diagram, evaluation plan) plus **synchronous contract tests** for every Stage B component. Stage A (precompute) is **given for now**; gaps are recorded for later change.

**Capital / dollars:** Treat sizing budget as an **abstract risk unit** until the portfolio layer is implemented — do not pin $1M deployable this sprint.

---

## HD decisions (locked for Sprint 002)

| Topic | Decision |
|-------|----------|
| Precompute | Given input; note gaps in contract doc, do not block Stage B design |
| Focus order | Stage B / SurfaceRunner pipeline first |
| Architecture | Prefer decoupled `pipeline.py` steps; **open to change** if contracts are cleaner |
| Tests | Contract tests for **all** components where contracts are defined |
| Success bar | Data contract doc + full data flow doc (diagram) + evaluation plan doc + contract test suite |
| Implementation | Defer P0 implementation (sizing, 50-cap, metrics overhaul) unless a narrow fix is required for a contract test harness |

---

## Deliverables

| # | Artifact | Path | Purpose |
|---|----------|------|---------|
| 1 | **Data contract** | [surface_engine_data_contract.md](../surface_engine_data_contract.md) | Per-component input/output schemas, invariants, status (built / partial / spec-only) |
| 2 | **Data flow + diagram** | [surface_engine_data_flow.md](../surface_engine_data_flow.md) | Step-by-step flow; each box = component, criteria, paths, implementation status |
| 3 | **Evaluation plan** | [surface_engine_evaluation_plan.md](../surface_engine_evaluation_plan.md) | How to verify each component; link tests to contracts; run-level success vs decision-quality |
| 4 | **Contract tests** | `tests/contract/` | One module per component; assert schema + invariants (may xfail until implementation catches up) |
| 5 | **Sprint memo** | `docs/sprint_memos/002_data_contracts.md` | Closeout, open design questions, precompute gap log |

**Supersedes for Stage B detail:** `surface_runner_data_flow.md` remains Sprint 001 history; new flow doc is canonical after Sprint 002 sign-off.

---

## Component scope (Stage B)

Treat precompute outputs as **fixed inputs** (see contract doc § Stage A inputs).

| ID | Component | Owner (target) | Contract test |
|----|-----------|----------------|---------------|
| R0 | Run envelope | `BacktestRunConfig`, manifest (TBD), date schedule | `test_run_envelope_contract.py` |
| S1 | Universe | `pipeline.step1_get_universe` | `test_step1_universe_contract.py` |
| S2 | Signals | `pipeline.step2_score_signals` | `test_step2_signals_contract.py` |
| S3 | Structures | `pipeline.step3_get_eligible_structures` | `test_step3_structures_contract.py` |
| S4 | Exclusions | `pipeline.step4_apply_exclusions` | `test_step4_exclusions_contract.py` |
| S5 | Select + size + simulate + return | `pipeline.step5_select_and_size` | `test_step5_select_and_size_contract.py` |
| S6 | _(collapsed into S5)_ | `step6_apply_cost` deprecated stub | — |
| S7 | Settle | `StrategyAssemblyResult.settle` + hold-to-expiry | `test_settle_contract.py` (or fold into S3/S5) |
| S8 | Date / run metrics | `surface_metrics` | `test_run_metrics_contract.py` |
| ORCH | Orchestration | `SurfaceRunner` thin wrapper | Extend `test_surface_runner_data_flow.py` or `test_orchestration_contract.py` |
| IN | Stage A inputs | meta + quotes parquet schema | `test_precompute_input_contract.py` (read-only / given) |

---

## Success criteria

- [ ] `surface_engine_data_contract.md` complete and HD-reviewed
- [ ] `surface_engine_data_flow.md` includes step diagram; every box has status + criteria + decision paths
- [ ] `surface_engine_evaluation_plan.md` maps each component to verification method and test file
- [x] `tests/contract/` for implemented steps IN, R0, S1–S4, S7 (43 tests green)
- [x] S5/S8/ORCH outcomes in portfolio/metrics design doc; S6 collapsed into S5 (contracts in Sprint 003)
- [ ] Precompute gap log section populated if Stage A inputs cannot support v1 backtest contract
- [ ] No Tier B backtest run; no large engine refactor without contract sign-off

---

## Session plan (suggested)

| Session | Work |
|---------|------|
| A | Draft contracts § R0, S1–S2, Stage A inputs; tests for S1–S2 |
| B | Draft contracts § S3–S7; tests for S3, settle; diagram in data flow doc |
| C | **Done** — S5/S8/ORCH design; S6→S5 collapse; [surface_engine_portfolio_metrics_design.md](../surface_engine_portfolio_metrics_design.md); contracts deferred to Sprint 003 |
| D | HD review (contracts through S4+S7 + design doc); evaluation plan; close memo |

---

## Out of scope (Sprint 002)

- Pinning deployable capital ($) or broker thresholds
- Full portfolio layer implementation (unless required for test harness only)
- Tier B 2020+ backtest
- Iron fly vs condor research matrix
- Precompute pipeline code changes (gaps documented only)

---

## Agent instructions

1. Read this file and [v1_spec_pins.md](../v1_spec_pins.md) at session start.
2. **Contracts before code** — update docs and contract tests together.
3. When runner inline logic disagrees with pipeline contract, document as **implementation drift**; do not silently change contract to match bugs.
4. Contract tests may use synthetic fixtures shared under `tests/contract/fixtures/` or reuse patterns from `test_surface_runner_data_flow.py`.

---

## Previous sprint

Sprint 001 closed 2026-05-28 — [sprint_memos/001_repo_audit_verification.md](../sprint_memos/001_repo_audit_verification.md).
