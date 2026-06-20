# Sprint memo 003 — S5/S8 build and ORCH

**Status:** Closed  
**Sprint:** 003  
**Mode:** Build  
**Started:** 2026-06-13  
**Closed:** 2026-06-20

---

## Sprint outcome

Sprint 003 delivered the **S5 trade-construction layer** (select + size + simulate + returns), **S8 cycle-return metrics**, and a **thin S1→S8 orchestration** in `SurfaceRunner`, each with contract tests plus synthetic end-to-end verification on fixtures.

No production strategy/financial logic was changed outside the approved S5/S8 scope. No real ORATS/cache backtest was run. No live/paper readiness claim.

**Exit posture:** Stage-B synthetic S1→S8 path is structurally verified (488 tests green). Decision-quality go/no-go requires real-data validation (Sprint 004+).

---

## Delivered

| Deliverable | Location |
|-------------|----------|
| Sizing config + validation | `src/backtest/run_config.py` |
| S5 SELECT + SIZE + SIMULATE | `src/backtest/pipeline.py::step5_select_and_size` |
| S8 cycle metrics | `src/backtest/surface_metrics.py` |
| ORCH thin loop | `src/backtest/surface_runner.py` |
| Contract tests (154) | `tests/contract/` |
| Synthetic S1→S8 smoke | `tests/unit/test_surface_runner_data_flow.py`, `tests/contract/test_orchestration_contract.py` |
| Data contract / flow polish | `surface_engine_data_contract.md`, `surface_engine_data_flow.md` |
| S6 stub removed | `pipeline.py` (collapsed into S5) |

---

## Drift resolved

| Item | Resolution |
|------|------------|
| S5 inline in runner | `SurfaceRunner` delegates to `step5_select_and_size` |
| S8 body-credit Sharpe | Sharpe/drawdown/`robust_score` on `cycle_return_on_capital_at_risk` |
| ORCH duplication | No inline `_select_size_and_settle`; `_assembly` stripped from trade log |
| S6 `step6_apply_cost` | Removed; economics in S5 + S3 fill |

---

## Verification (Phase 7 + 8)

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/contract/ -q
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_surface_runner_data_flow.py -q
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/ -q
```

Phase 8 closeout (2026-06-20): **154** contract + **8** data-flow + **488** full suite passed. No cleanup regressions.

---

## Carry-forward (not blocking closeout)

| Item | Notes |
|------|-------|
| **KB-001** | Iron condor M3 denominator — [known_bugs.md](../known_bugs.md) |
| **Financial checks** | Partial coverage in contract/unit tests; no dedicated end-to-end leg/premium/max-loss audit suite |
| **`run_surface_search.py`** | Missing `sizing_mode` / Tier B CLI — fails fast at config construction |
| **Real-data backtest** | Sprint 004 — split-adjusted universe / data pipeline |
| **Precompute gaps** | Earnings column; 3-month liquidity rolling avg |

---

## Next sprint

**Sprint 004** — data / split-adjusted universe pipeline and real-data structural validation (not strategy sign-off).
