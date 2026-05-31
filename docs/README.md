# Documentation index

This folder holds **active** project documentation for the v1 live-trading path. Historical docs live in `docs/archive/`.

## How to know what is current

| Location | Status | Rule |
|----------|--------|------|
| `docs/*.md` (except `archive/`) | **Active** | Updated during sprints; trust for current work |
| `docs/agenda/current_sprint.md` | **Active** | Single source of truth for this week |
| `docs/sprint_memos/` | **Active history** | What was done and verified each sprint |
| `docs/decisions/` | **Active ADRs** | Locked decisions with date and rationale |
| `docs/archive/` | **Stale reference** | Do not update; may contradict active docs |

When a doc goes stale, move it to `docs/archive/` and add a row to `docs/archive/README.md`.

---

## Active documents

| Document | Purpose | Last updated |
|----------|---------|--------------|
| [v1_spec_pins.md](v1_spec_pins.md) | Frozen v1 parameters and deferred decisions | 2026-05-23 |
| [v1_universe_protocol.md](v1_universe_protocol.md) | Point-in-time liquidity universe rule | 2026-05-23 |
| [v1_ops_model.md](v1_ops_model.md) | Trade volume and broker/manual threshold | 2026-05-23 |
| [backtest_evaluation_protocol.md](backtest_evaluation_protocol.md) | Go/no-go windows, Sharpe, evaluation tiers | 2026-05-23 |
| [repo_map.md](repo_map.md) | Repository layout and data flow | 2026-05-23 |
| [repo_audit.md](repo_audit.md) | Sprint 001 surface pipeline gap audit | 2026-05-24 |
| [correctness_audit.md](correctness_audit.md) | Sprint 001 test inventory and correctness gaps | 2026-05-24 |
| [surface_runner_data_flow.md](surface_runner_data_flow.md) | Sprint 001 Session A.1 SurfaceRunner data-flow map | 2026-05-25 |
| [surface_runner_reading_guide.md](surface_runner_reading_guide.md) | Guided code read-through (Sprint 001) | 2026-05-27 |
| [surface_engine_data_contract.md](surface_engine_data_contract.md) | Sprint 002 per-component I/O contracts (canonical) | 2026-05-28 |
| [surface_engine_data_flow.md](surface_engine_data_flow.md) | Sprint 002 flow diagram + box status | 2026-05-28 |
| [surface_engine_evaluation_plan.md](surface_engine_evaluation_plan.md) | Sprint 002 component verification plan | 2026-05-28 |
| [agenda/session_b_plan.md](agenda/session_b_plan.md) | Sprint 001 Session B verification plan (historical) | 2026-05-27 |
| [development_workflow.md](development_workflow.md) | Human + agent sprint workflow | 2026-05-23 |
| [baseline_status.md](baseline_status.md) | Test and smoke-command baseline | 2026-05-23 |
| [agenda/current_sprint.md](agenda/current_sprint.md) | Current sprint (Sprint 002 — contracts + diagram) | 2026-05-28 |

## Decisions

| ID | Document |
|----|----------|
| 001 | [Canonical backtest path](decisions/001_canonical_backtest_path.md) |
| 002 | [Max concurrent position cap semantics](decisions/002_position_cap_semantics.md) |

## Sprint memos

| Sprint | Document |
|--------|----------|
| 000 | [Week 0 kickoff](sprint_memos/000_week0_kickoff.md) |
| 001 | [Repo audit and verification](sprint_memos/001_repo_audit_verification.md) _(closed 2026-05-28)_ |
| — | [Week 0 review notes](sprint_memos/week0_review_notes.md) _(archived from agenda)_ |

## Planned (not yet created)

| Document | When |
|----------|------|
| `v1_weekly_runbook.md` | After liquidity panel review (Sprint 002) |

## Related (repo root)

| Document | Purpose |
|----------|---------|
| [AGENTS.md](../AGENTS.md) | Agent operating rules for this repo |

## Archived reference

See [archive/README.md](archive/README.md) for pre–Week 0 planning docs (production checklist, strategy definition outline, iron fly research plans, etc.). Pull ideas from archive into active docs when needed; do not maintain archive in place.
