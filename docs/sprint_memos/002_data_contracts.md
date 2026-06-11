# Sprint memo 002 — Data contracts and S5/S8 design

**Status:** Closed  
**Sprint:** 002  
**Mode:** Audit / Design (contracts and tests define truth; minimal prod changes)  
**Started:** 2026-05-28  
**Closed:** 2026-06-10

---

## Sprint outcome

Sprint 002 delivered the **source-of-truth documentation** for the surface backtest data model — per-component I/O contracts, full data flow with diagram, and an evaluation plan — plus **synchronous contract tests** for every implemented Stage B component (IN, R0, S1–S4, S7). The pipeline tail that was *not* implementation-ready (S5 sizing/returns, S8 cycle metrics, ORCH) was instead pinned in a dedicated **design doc** with every open question resolved, so Sprint 003 can build and contract-test together.

No production strategy/financial logic was changed. Stage A (precompute) was treated as a fixed input; gaps are logged, not fixed.

**Exit posture:** design is implementation-ready for Sprint 003. Do **not** treat any backtest as decision-quality until S5/S8 are implemented and contract-tested.

---

## Delivered

| Deliverable | Location |
|-------------|----------|
| Data contract (source of truth) | [surface_engine_data_contract.md](../surface_engine_data_contract.md) |
| Data flow + diagram | [surface_engine_data_flow.md](../surface_engine_data_flow.md) |
| Component evaluation plan | [surface_engine_evaluation_plan.md](../surface_engine_evaluation_plan.md) |
| Contract test suite (43 tests) | `tests/contract/` |
| S5/S8/ORCH portfolio & metrics design | [surface_engine_portfolio_metrics_design.md](../surface_engine_portfolio_metrics_design.md) |
| Per-side position cap ADR | [decisions/003_position_cap_per_side.md](../decisions/003_position_cap_per_side.md) |
| This close memo | `docs/sprint_memos/002_data_contracts.md` |

---

## Component status (Stage B)

| ID | Component | Contract | Contract test | Status |
|----|-----------|----------|---------------|--------|
| IN | Stage A inputs | documented | `test_precompute_input_contract.py` (6) | Given / read-only |
| R0 | Run envelope | documented | `test_run_envelope_contract.py` (9) | Implemented |
| S1 | Universe | documented | `test_step1_universe_contract.py` (6) | Implemented |
| S2 | Signals | documented | `test_step2_signals_contract.py` (6) | Implemented |
| S3 | Structures | documented | `test_step3_structures_contract.py` (7) | Implemented |
| S4 | Exclusions | documented | `test_step4_exclusions_contract.py` (6) | Implemented |
| S5 | Select + size + simulate + return | **designed** | deferred → Sprint 003 | Design-only |
| S6 | _(cost)_ | **collapsed into S5** | — | Superseded |
| S7 | Settle | documented | `test_settle_contract.py` (3) | Implemented |
| S8 | Date / run metrics | **designed** | deferred → Sprint 003 | Design-only (interim body-credit metric in code) |
| ORCH | Orchestration | designed | deferred → Sprint 003 | Runner partial; S5 inline |

---

## Key decisions locked in Sprint 002

| # | Decision |
|---|----------|
| 1 | **S6 collapsed into S5** — entry fill fixed at S3 (`FillAssumption`); no separate post-trade cost step in v1. |
| 2 | **Per-side position cap** (`max_names_per_side`) replaces "50 total" — supersedes Decision 002 ([ADR 003](../decisions/003_position_cap_per_side.md)). |
| 3 | **Two sizing tiers in S5**, selected via required `sizing_mode`: Tier A (`conceptual`, fractional, no multiplier) and Tier B (`integer_lots`, ×100, capital-bound). |
| 4 | **Tier A control = per-side total budget** split equally by name count (`tier_a_mode` ∈ {`equal_premium`, `equal_max_loss`}; `tier_a_short_budget` / `tier_a_long_budget`). In `equal_max_loss`, the long side is financed by realized short premium. |
| 5 | **Three per-trade return metrics** M1 `return_on_premium`, M2 `return_on_max_loss`, M3 `return_on_atm_straddle` — always persisted; `NaN` where undefined (M2 `NaN` on straddles). |
| 6 | **Portfolio metric** = `cycle_return_on_capital_at_risk` = `Σ pnl_total / Σ capital_at_risk_dollars` per rebalance cycle, with short/long side splits. Sharpe/drawdown/go-no-go use this series, not body-credit. |
| 7 | **`contract_multiplier` pinned = 100**; **no naked short straddle in v1** (short side defined-risk only); **trade-log grain** = one row per `(trade_date, ticker, direction)`. |
| 8 | **Capital is a hard constraint** (Tier B, when `deployable_capital` set): drop names by rank or skip the date; never overrun, never proportionally rescale. `deployable_capital=None` (v1) ⇒ only per-name budget binds. |

All design questions (Q1–Q11) resolved; see design-doc change log.

---

## Tests run

```text
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/contract -q
43 passed in 0.42s
```

---

## Precompute gap log (Stage A)

Recorded in [surface_engine_data_contract.md](../surface_engine_data_contract.md) § Precompute gap log:

1. `earnings_date` not present in any Stage A artifact — blocks S4 earnings exclusion; file when S4 is implemented.
2. Liquidity panel uses monthly snapshot grain — HD decision: replace with 3-month rolling average (same column names); fix before end-to-end correctness check, not blocking Sprint 002 contracts.

---

## Implementation drift noted (for Sprint 003 to resolve)

- S5 logic lives **inline** in `SurfaceRunner._select_size_and_settle` — not yet a `pipeline.step5_*`.
- **No sizing phase** in code: `quantity`, `pnl_total`, `capital_at_risk_dollars`, M1–M3 absent from the trade log.
- S8 computes interim **body-credit** Sharpe only — target `cycle_return_on_capital_at_risk` not wired.
- `pipeline.step6_apply_cost` is a deprecated `pass` stub (cleanup in Sprint 003).
- New config fields required: `sizing_mode`, `tier_a_mode`, `tier_a_short_budget`, `tier_a_long_budget`, `contract_multiplier`, `deployable_capital`.
- `BacktestRunConfig` still accepts `short_structure='straddle'` — reject or mark unsupported for v1 (Q5).

---

## Files touched (traceability)

**Docs:** `surface_engine_data_contract.md`, `surface_engine_data_flow.md`, `surface_engine_evaluation_plan.md`, `surface_engine_portfolio_metrics_design.md`, `decisions/002_position_cap_semantics.md` (superseded), `decisions/003_position_cap_per_side.md` (new), `v1_spec_pins.md`, `README.md`, `agenda/current_sprint.md`, this memo.

**Tests:** `tests/contract/` (conftest + IN, R0, S1–S4, S7).

**Production code:** `pipeline.py` (S3/S4 extraction, `step6` deprecation note), `run_config.py` (`long_top_pct + short_bottom_pct ≤ 1` validation), `surface_runner.py` (S3/S4 delegation). No strategy/financial logic changes.

---

## HD review checklist (closeout)

- [x] `surface_engine_data_contract.md` reviewed
- [x] `surface_engine_data_flow.md` reviewed (diagram + per-box status)
- [x] `surface_engine_evaluation_plan.md` reviewed
- [x] `surface_engine_portfolio_metrics_design.md` reviewed — all open questions resolved
- [x] Contract suite green (43 passed)
- [x] Precompute gap log accepted
- [x] No Tier B backtest run; no large engine refactor without sign-off

---

## Forward pointer (Sprint 003 — Build)

1. Implement S5 in `pipeline.py` — both tiers; select + size + simulate; M1–M3 + `pnl_total` + `capital_at_risk_dollars`; runner delegates.
2. Implement S8 cycle returns — `cycle_return_on_capital_at_risk` + side splits; Sharpe on cycle series.
3. ORCH — thin S1→S8 loop; remove inline S5; orchestration contract test.
4. Contract tests for S5/S8/ORCH (written with implementation).
5. Add the new config fields; resolve the `short_structure='straddle'` tension.

Build scope detail in [surface_engine_portfolio_metrics_design.md](../surface_engine_portfolio_metrics_design.md) § Sprint 003 build scope.
