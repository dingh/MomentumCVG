# Sprint memo 000 — Week 0 kickoff

**Dates:** 2026-05-23  
**Mode:** Audit + docs  
**Hours (planned):** 8–10

---

## Goal

Set up executable agent workflow and v1 planning docs without code changes.

---

## Delivered

| Artifact | Status |
|----------|--------|
| `docs/archive/` + moved legacy docs | Done |
| `docs/README.md` | Done |
| `docs/v1_spec_pins.md` | Done (50 max positions, 7 DTE, 2020+ go/no-go) |
| `docs/v1_universe_protocol.md` | Done |
| `docs/v1_ops_model.md` | Done |
| `docs/backtest_evaluation_protocol.md` | Done |
| `docs/repo_map.md` | Done |
| `docs/development_workflow.md` | Done |
| `docs/baseline_status.md` | Done |
| `docs/decisions/001_canonical_backtest_path.md` | Done |
| `AGENTS.md` | Done |
| `.cursor/rules/` | Done |

---

## Tests run

```text
326 passed in ~3.3s
Command: C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/ -q
```

---

## Verified

- Project venv runs full unit suite green
- Active vs archived doc split exists
- v1 pins documented in one place

---

## Not verified (by design — Week 0)

- Surface backtest smoke (needs cache)
- PIT universe wiring end-to-end
- Payoff / max-loss verification tests
- Portfolio max-loss budget numbers

---

## Problems found

- Many legacy docs duplicated active vs archive during partial migration — resolved by moving all pre–Week 0 docs to `docs/archive/`
- Three backtest paths still coexist in code; decision 001 picks SurfaceRunner pending audit

---

## Week 0 exit gate

- [x] v1 spec pins signed off in docs
- [x] Canonical path documented (provisional)
- [x] pytest baseline recorded
- [x] Sprint 001 agenda ready

---

## Next: Sprint 001

See [agenda/current_sprint.md](../agenda/current_sprint.md).
