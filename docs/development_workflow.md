# Development workflow

**Status:** Active  
**Last updated:** 2026-07-04 (C5 adjusted-liquid closed)

---

## Principles

Every meaningful task follows:

```text
Inspect → Plan → Test → Implement → Run → Review → Memo
```

Do not jump from a vague goal straight to code.

---

## Weekly rhythm (~20 hours)

| Slot | Hours | You | Agent |
|------|-------|-----|-------|
| Planning | 3–4 | Set goal in `docs/agenda/current_sprint.md` | Inspect + plan (Audit mode) |
| Build | 10–12 | Review diffs | Verification or Build mode |
| Verify | 3–4 | Interpret financial results | Run pytest / smoke backtest |
| Close | 2 | Approve sprint memo | Draft memo; pick next task |

---

## Sprint modes

Set **one mode** in `docs/agenda/current_sprint.md`:

| Mode | Agent may |
|------|-----------|
| **Audit** | Read code; write/update docs only |
| **Verification** | Add or change tests; fix production code only if test proves bug |
| **Build** | Implement approved plan only |

Switch modes explicitly between sessions.

---

## Document hygiene

| Location | Rule |
|----------|------|
| `docs/*.md` (not `archive/`) | Active; update when specs change |
| `docs/agenda/current_sprint.md` | One sprint at a time |
| `docs/sprint_memos/` | Immutable history after sprint closes |
| `docs/decisions/` | Locked ADRs; supersede with new numbered decision |
| `docs/archive/` | Read-only reference |

When a doc goes stale, move it to `docs/archive/` and update `docs/archive/README.md`.

### Sprint close checklist

At the end of each sprint:

1. Write or finalize `docs/sprint_memos/NNN_*.md`.
2. Update `docs/agenda/current_sprint.md` for the **next** sprint.
3. If you added a new active doc, decision ADR, or sprint memo, update the tables in **`docs/README.md`** (append row, refresh "Last updated").
4. No automation required — a short manual index update is enough.

---

## Git and commits

- Agent does not commit unless you ask.
- Review diff before accepting agent edits.

---

## Commands

```powershell
# Activate venv
& C:/MomentumCVG_env/venv/Scripts/Activate.ps1

# Tests
python -m pytest tests/ -q

# Focused test file
python -m pytest tests/unit/test_builders.py -v

# Surface backtest (requires cache; see baseline_status.md)
python scripts/run_surface_search.py --mode full_sample --start-date 2020-01-01 ...
```

---

## Sprint roadmap (004–008)

Aligned with [agenda/current_sprint.md](agenda/current_sprint.md). Sprints 000–003 closed; see [sprint_memos/](sprint_memos/).

| Sprint | Status | Theme | Delivers | Explicitly not |
|--------|--------|-------|----------|----------------|
| **004** | **Active** | Input snapshot + split/PIT + **liquidity panel (C4 ✓)** + **adjusted-liquid (C5 ✓)** + surface precompute audit | CLI, rolling panel, scoped adjust + audit, **A1/A2** tests/audit, PIT harness, runbook | Mom/CVG, straddle history, backtest smoke |
| **005** | Planned | **All feature pipeline** (straddle history, features, mom/CVG, A4, paths, trade-date calendar) | May absorb earnings/pipeline gaps from 004 | L4 backtest smoke |
| **006** | Planned | Real-data **backtest** smoke + `run_surface_search` wiring | L4 S1→S8 | Start only after **004 input + 005 features** trustworthy |
| **007** | Planned | Tier B conservative baseline | 2020→latest canonical run; conservative fills; metrics table in [backtest_evaluation_protocol.md](backtest_evaluation_protocol.md); SurfaceRunner `run_manifest` | Sensitivity matrix, paper trading |
| **008** | Planned | Decision sprint (conditional on 007) | Go/no-go memo; pass/fail thresholds or documented waiver; triage KB-001 and remaining precompute gaps | Live execution |

**Real-data split:** Sprint 004 validates **input/precompute** on real cache (splits, liquidity, option surface A1/A2). Sprint 006 runs **backtest** smoke (SurfaceRunner). Stale wording in older docs → update when that sprint starts.

**006 gate:** Do not start Sprint 006 until Sprint 004 input snapshot and Sprint 005 feature pipeline are trustworthy.

**004 vs 005:** 004 = splits, spot, rolling liquidity, **option surface precompute audit**. 005 = **entire feature branch** (straddle history, features, mom/CVG).

**Evaluation levels** ([surface_engine_evaluation_plan.md](surface_engine_evaluation_plan.md)):

| Level | Sprint |
|-------|--------|
| L1–L3 Synthetic | ✅ 002–003 |
| L4 Real-cache smoke | 006 |
| L5 Tier B go/no-go | 007–008 |

---

## 12-week outline (high level)

| Weeks | Focus |
|-------|--------|
| 0–1 | Workflow, repo audit, first verification test ✅ |
| 2–4 | Data contracts; S5/S8 build; portfolio caps ✅ |
| 5–6 | **Sprint 004–005:** build input pipeline + feature/mom/CVG audit |
| 7 | **Sprint 006:** real-data structural smoke (L4) |
| 8–9 | **Sprint 007:** conservative baseline; iron fly vs condor comparison deferred to post-baseline sensitivity |
| 9–10 | Shadow runner; ops model from measured trade counts |
| 11–12 | Paper trading; capital ramp memo |

Details in [v1_spec_pins.md](v1_spec_pins.md) and [backtest_evaluation_protocol.md](backtest_evaluation_protocol.md).
