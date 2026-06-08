# Surface engine — component evaluation plan

**Status:** Draft — Sprint 002  
**Last updated:** 2026-06-07  
**Companion:** [surface_engine_data_contract.md](surface_engine_data_contract.md), [surface_engine_data_flow.md](surface_engine_data_flow.md)

---

## Purpose

Define **how** each component is verified, linked to **contract tests** and (later) implementation sprints. End-to-end tests alone are insufficient for trust.

---

## Verification levels

| Level | What it proves | When to use |
|-------|----------------|-------------|
| **L1 Contract** | Output schema, columns, dtypes, grain, invariants | Every component — Sprint 002 |
| **L2 Golden** | Hand-calculated numeric values on synthetic fixture | S3, S7, S5 when built |
| **L3 Integration** | Multi-step chain on synthetic parquets | ORCH — exists (`test_surface_runner_data_flow.py`) |
| **L4 Smoke** | Real cache, short date window | After S5 built (incl. returns) — not Sprint 002 |
| **L5 Tier B** | 2020+ go/no-go | After structural success — not Sprint 002 |

Sprint 002 delivers **L1 for all components**; L2 where docstrings already pin math (S3, S7).

---

## Component evaluation matrix

| ID | Component | Contract test | L1 | L2 | Notes |
|----|-----------|---------------|----|----|-------|
| IN | Precompute inputs | `test_precompute_input_contract.py` | ✅ | — | Read-only schema; given (A1–A4) |
| R0 | Run envelope | `test_run_envelope_contract.py` | ✅ | — | Date list, config validation |
| S1 | Universe | `test_step1_universe_contract.py` | ✅ | ☐ | PIT snapshot selection |
| S2 | Signals | `test_step2_signals_contract.py` | ✅ | ☐ | Pool disjointness |
| S3 | Structures | `test_step3_structures_contract.py` | ✅ | ✅ | Synthetic surface; iron fly credit golden |
| S4 | Exclusions | `test_step4_exclusions_contract.py` | ✅ | — | Earnings window flags |
| S5 | Select + size + simulate + return | `test_step5_select_and_size_contract.py` | — | — | **Deferred** — design doc; Sprint 003 build (incl. former S6) |
| S6 | Cost+return | — | — | — | **Superseded** — collapsed into S5; no contract test |
| S7 | Settle | `test_settle_contract.py` | ✅ | ✅ | Iron fly + long straddle PnL golden |
| S8 | Run metrics | `test_run_metrics_contract.py` | — | — | **Deferred** — design doc; Sprint 003 build |
| ORCH | Orchestration | `test_orchestration_contract.py` | — | — | **Deferred** — Sprint 003 build |

Check boxes during Sprint 002 sessions.

---

## Contract test conventions

- Location: `tests/contract/`  
- Shared fixtures: `tests/contract/conftest.py` or `tests/contract/fixtures/`  
- **xfail** with reason when contract is ahead of implementation (`spec-only` steps)  
- **pass** when code matches contract (S1, S2, S3 today)  
- Do not weaken contracts to match `drift` — file drift in contract doc instead  

Run subset:

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/contract/ -v
```

---

## Per-component evaluation checklist (template)

Copy into contract doc as each section is completed.

### S_n — _name_

1. **Inputs documented** in data contract  
2. **Outputs documented** with grain  
3. **Invariants listed** (≥3 where applicable)  
4. **Contract test** asserts schema on synthetic minimal input  
5. **Golden test** (if L2) matches hand calculation  
6. **Diagram box** status updated in data flow doc  
7. **Drift** logged if runner ≠ pipeline  

---

## Sprint 002 exit gate

Sprint 002 is done when:

- Components **IN, R0, S1–S4, S7** have L1 contract tests (green)  
- **S5, S8, ORCH** have outcomes documented in [surface_engine_portfolio_metrics_design.md](surface_engine_portfolio_metrics_design.md) (S6 collapsed into S5; contracts deferred to Sprint 003)  
- Core three docs + design doc HD-reviewed  
- Precompute gap log reviewed (empty or accepted)  
- No requirement for L4/L5  

---

## Forward: mapping to implementation sprints

| After contract sign-off | Implementation sprint theme |
|-------------------------|----------------------------|
| S4 | Earnings exclusion |
| S5 + drift | Portfolio cap, sizing, returns, runner→pipeline |
| S8 | Metrics on max-loss return series |
| IN gaps | Precompute schema / coverage sprint |

---

## Change log

| Date | Change |
|------|--------|
| 2026-05-28 | Sprint 002 scaffold |
| 2026-05-31 | Session A: L1 checked for IN, R0, S1, S2 (27 contract tests green) |
| 2026-05-31 | Session B: L1+L2 for S3, S4, S7 (43 contract tests green) |
| 2026-05-31 | Session C: S5/S8/ORCH deferred; portfolio/metrics design doc |
| 2026-06-07 | S6 collapsed into S5 — fill at S3; no step6 contract |
