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
| [v1_universe_protocol.md](v1_universe_protocol.md) | Point-in-time liquidity universe rule (12-week rolling panel) | 2026-06-29 |
| [v1_ops_model.md](v1_ops_model.md) | Trade volume and broker/manual threshold | 2026-05-23 |
| [backtest_evaluation_protocol.md](backtest_evaluation_protocol.md) | Go/no-go windows, Sharpe, evaluation tiers | 2026-05-23 |
| [repo_map.md](repo_map.md) | Repository layout and data flow | 2026-05-23 |
| [repo_audit.md](repo_audit.md) | Sprint 001 surface pipeline gap audit | 2026-05-24 |
| [correctness_audit.md](correctness_audit.md) | Sprint 001 test inventory and correctness gaps | 2026-05-24 |
| [surface_runner_data_flow.md](surface_runner_data_flow.md) | Sprint 001 Session A.1 SurfaceRunner data-flow map | 2026-05-25 |
| [surface_runner_reading_guide.md](surface_runner_reading_guide.md) | Guided code read-through (Sprint 001) | 2026-05-27 |
| [surface_engine_data_contract.md](surface_engine_data_contract.md) | Sprint 002 per-component I/O contracts (canonical) | 2026-06-20 |
| [surface_engine_data_flow.md](surface_engine_data_flow.md) | Sprint 002 flow diagram + box status | 2026-06-20 |
| [surface_engine_evaluation_plan.md](surface_engine_evaluation_plan.md) | Sprint 002 component verification plan | 2026-06-20 |
| [surface_engine_portfolio_metrics_design.md](surface_engine_portfolio_metrics_design.md) | S5/S8 portfolio & metrics design (Accepted; built Sprint 003) | 2026-06-20 |
| [development_workflow.md](development_workflow.md) | Human + agent sprint workflow; roadmap 004–008 | 2026-07-04 |
| [baseline_status.md](baseline_status.md) | Test and smoke-command baseline | 2026-07-04 |
| [agenda/current_sprint.md](agenda/current_sprint.md) | Sprint 004 — C4–C7 + C8.4 closed; C8.5 + C9 remaining | 2026-07-22 |
| [v1_weekly_runbook.md](v1_weekly_runbook.md) | Weekly Stage A refresh procedure | 2026-07-04 |
| [repo_map.md](repo_map.md) | Repository layout and data flow | 2026-07-04 |
| [known_bugs.md](known_bugs.md) | Open bugs and spec drift (fix deferred) | 2026-06-14 |

## Decisions

| ID | Document |
|----|----------|
| 001 | [Canonical backtest path](decisions/001_canonical_backtest_path.md) |
| 002 | [Max concurrent position cap semantics](decisions/002_position_cap_semantics.md) _(superseded)_ |
| 003 | [Per-side position cap](decisions/003_position_cap_per_side.md) |

## Sprint memos

| Sprint | Document |
|--------|----------|
| 000 | [Week 0 kickoff](sprint_memos/000_week0_kickoff.md) |
| 001 | [Repo audit and verification](sprint_memos/001_repo_audit_verification.md) _(closed 2026-05-28)_ |
| 002 | [Data contracts and S5/S8 design](sprint_memos/002_data_contracts.md) _(closed 2026-06-10)_ |
| 003 | [S5/S8 build and ORCH](sprint_memos/003_s5_s8_build.md) _(closed 2026-06-20)_ |
| 004 (C4) | [Liquidity panel closeout](sprint_memos/004_c4_liquidity_panel.md) _(C4 closed 2026-06-29)_ |
| 004 (C5) | [Adjusted-liquid split layer closeout](sprint_memos/004_c5_adjusted_liquid.md) _(C5 closed 2026-07-04)_ |
| 004 (C6) | [Option-surface layer closeout](sprint_memos/004_c6_option_surface.md) _(C6 closed 2026-07-11)_ |
| 004 (C7) | [PIT universe closeout](sprint_memos/004_c7_pit_universe.md) _(C7 closed 2026-07-12)_ |
| 004 (C8.4) | [Bounded snapshot evidence closeout](sprint_memos/004_c8_4_bounded_evidence.md) _(C8.4 closed 2026-07-22)_ |
| — | [Week 0 review notes](sprint_memos/week0_review_notes.md) _(archived from agenda)_ |

## Related (repo root)

| Document | Purpose |
|----------|---------|
| [AGENTS.md](../AGENTS.md) | Agent operating rules for this repo |

## Archived reference

See [archive/README.md](archive/README.md) for pre–Week 0 planning docs (production checklist, strategy definition outline, iron fly research plans, etc.). Pull ideas from archive into active docs when needed; do not maintain archive in place.
