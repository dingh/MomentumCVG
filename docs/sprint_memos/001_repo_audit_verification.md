# Sprint memo 001 — Repo audit and verification

**Status:** Closed  
**Sprint:** 001  
**Mode:** Audit → Verification  
**Started:** 2026-05-24  
**Closed:** 2026-05-28

---

## Sprint outcome

**Outcome A (with C emphasis):** Backlog is documented and prioritized; canonical path confirmed; one runner-level verification test passes. **P0 engine gaps dominate** — backtest results are not decision-quality until Sprints 002–004 close portfolio/risk/metrics and coverage gaps.

HD exit (2026-05-28): Architecture and surface assembly layer look sound; systematic unit tests on builders give partial confidence. **Low trust in backtest results today** because the runner has many options, partial wiring, and missing portfolio/dollar-PnL layer. Plan: address trust gaps **one P0 item per sprint** over the next 3–4 sprints (see [repo_audit.md](../repo_audit.md) Sprint 002–004 backlog).

---

## Delivered

| Deliverable | Location |
|-------------|----------|
| Surface pipeline gap audit | [repo_audit.md](../repo_audit.md) |
| Test inventory and correctness gaps | [correctness_audit.md](../correctness_audit.md) |
| SurfaceRunner data-flow map | [surface_runner_data_flow.md](../surface_runner_data_flow.md) |
| Reading guide (code walkthrough) | [surface_runner_reading_guide.md](../surface_runner_reading_guide.md) |
| Session B verification test | `tests/unit/test_surface_runner_data_flow.py` (9 tests) |
| Position cap semantics ADR | [decisions/002_position_cap_semantics.md](../decisions/002_position_cap_semantics.md) |
| Session B plan (historical) | `agenda/session_b_plan.md` — removed 2026-06-10; superseded by this memo |

---

## Session A — Audit

| Item | Status |
|------|--------|
| `docs/repo_audit.md` | Done — HD reviewed 2026-05-27 |
| `docs/correctness_audit.md` | Done — HD reviewed 2026-05-27 |
| `docs/surface_runner_data_flow.md` | Done — HD reviewed 2026-05-25 |
| Production code changes | None |

**Main conclusion:** `SurfaceRunner` remains the canonical path. Strongest layer: `option_surface.py` assembly/settlement. Weakest: engine portfolio/risk/metrics and Stage A→B contracts (dates, coverage).

---

## Session B — Verification

| Item | Status |
|------|--------|
| Test: synthetic `SurfaceRunner.run_single_config()` data-flow | Done |
| Production fix | None required |

**What the test proves:** Synthetic end-to-end wiring; PIT universe + signal routing; settlement `pnl_per_share` matches hand calc; invalid surface excluded; **`contracts` / `pnl_dollars` / `return_on_max_loss` absent** (documented).

**What it does not prove:** Real cache, Tier B, sizing, 50 total cap, go/no-go metrics.

---

## Decisions locked in Sprint 001

| # | Decision |
|---|----------|
| 0 | Session A.1 data-flow map before Session B test |
| 1 | Session B target = `run_single_config()` synthetic data-flow test |
| 2 | 50 max concurrent positions = **total** long + short |
| 3 | Engine logic → separable `pipeline.py` steps (runner orchestrates) |

---

## Tests run

```text
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_surface_runner_data_flow.py -q
9 passed in ~0.8s

C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/ -q
335 passed in ~4.0s
```

Baseline recorded in [baseline_status.md](../baseline_status.md).

---

## Files touched (traceability)

**Docs:** `repo_audit.md`, `correctness_audit.md`, `surface_runner_data_flow.md`, `surface_runner_reading_guide.md`, `README.md`, `v1_spec_pins.md`, `decisions/002_position_cap_semantics.md`, `baseline_status.md`, `agenda/session_b_plan.md`, `agenda/current_sprint.md` (through closeout)

**Tests:** `tests/unit/test_surface_runner_data_flow.py`

**Production code:** None

**Moved:** `docs/agenda/week0_review_notes.md` → [week0_review_notes.md](week0_review_notes.md)

---

## HD review checklist (closeout)

- [x] `docs/repo_audit.md`
- [x] `docs/correctness_audit.md`
- [x] `docs/surface_runner_data_flow.md`
- [x] P0/P1/P2 priorities accepted for follow-on sprints
- [x] SurfaceRunner remains canonical path
- [x] Session B test implemented and suite green

---

## Forward pointer (Sprint 002+)

Do **not** run Tier B go/no-go until P0 items in [repo_audit.md](../repo_audit.md) are closed incrementally.

Suggested sequencing (HD: fix trust **one gap per sprint**, 3–4 sprints):

1. **Sprint 002** — CLI smoke path + portfolio/risk/dollar-PnL layer (largest P0)
2. **Sprint 003** — Metrics on max-loss capital units + diagnostics/manifest
3. **Sprint 004** — Short-window cache smoke; controlled fly vs condor comparison

Agenda for Sprint 002: [agenda/current_sprint.md](../agenda/current_sprint.md) (to be finalized with HD).

---

## HD feedback log (archive)

| Date | Doc / section | Feedback | Action |
|------|---------------|----------|--------|
| 2026-05-25 | Audits + Session B | Map data flow before tests | Session A.1 + `surface_runner_data_flow.md` |
| 2026-05-25 | A.1 map | Approve synthetic runner test; 50 total cap; pipeline modularization | Spec + decisions updated |
| 2026-05-27 | `repo_audit.md` | Pipeline steps 1–6; portfolio layer P0 | Doc updated |
| 2026-05-27 | `correctness_audit.md` | Reviewed | — |
| 2026-05-28 | Sprint close | Code read-through; low trust in backtest results until P0s fixed | Close 001; plan 002–004 incrementally |
