# Development workflow

**Status:** Active  
**Last updated:** 2026-05-23 (Week 0 review revision)

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

## 12-week outline (high level)

| Weeks | Focus |
|-------|--------|
| 0–1 | Workflow, repo audit, first verification test |
| 2–4 | Correctness audits; pin portfolio caps |
| 5–8 | Weekly backtest; iron fly vs iron condor comparison; conservative fills; Tier B evaluation |
| 9–10 | Shadow runner; ops model from measured trade counts |
| 11–12 | Paper trading; capital ramp memo |

Details in [v1_spec_pins.md](v1_spec_pins.md) and [backtest_evaluation_protocol.md](backtest_evaluation_protocol.md).
