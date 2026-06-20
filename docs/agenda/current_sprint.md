# Current sprint — 003

**Updated:** 2026-06-20  
**Status:** Closed — Completed  
**Mode:** Build (closed)

---

## Goal

Turn the accepted Sprint 002 design into **trusted code**: implement the **S5 trade-construction layer** (select + size + simulate + return), **S8 cycle-return metrics**, and a **thin S1→S8 orchestration**, each with **contract tests** plus one **synthetic end-to-end smoke**.

Source of truth (all Accepted, do not re-litigate without an ADR):
- [surface_engine_portfolio_metrics_design.md](../surface_engine_portfolio_metrics_design.md) — S5/S8 economics (Tier A; Tier B sizing per [ADR 004](../decisions/004_tier_b_credit_financed_long.md))
- [surface_engine_data_contract.md](../surface_engine_data_contract.md) — component contracts
- [decisions/003_position_cap_per_side.md](../decisions/003_position_cap_per_side.md) — per-side cap
- [decisions/004_tier_b_credit_financed_long.md](../decisions/004_tier_b_credit_financed_long.md) — Tier B sizing

> **Precedence (S5/S8/ORCH):** data contract § S5 / S8 / ORCH are **built** and aligned with code (Sprint 003 closed 2026-06-20). **Authoritative for economics:** `cycle_return_on_capital_at_risk` = `Σ pnl_total / Σ capital_at_risk_dollars`, M1–M3 per design doc; Tier B sizing per ADR 004.

**Capital / dollars:** sizing budgets remain abstract risk units; do not pin $1M this sprint. **`deployable_capital`** stays on config (optional, validated when set) but is **not used in S5 Tier B sizing** per ADR 004.

---

## HD decisions (locked for Sprint 003)

| Topic | Decision |
|-------|----------|
| Tiers | Build **both** Tier A (`conceptual`) and Tier B (`integer_lots`) in S5 |
| `sizing_mode` | **Required** config field — no default; run fails fast if unset |
| Tier A control | **Per-side total budget** ÷ name count (`tier_a_mode` ∈ {`equal_premium`, `equal_max_loss`}) |
| Tier B control | **[ADR 004](../decisions/004_tier_b_credit_financed_long.md):** shorts from **`tier_b_short_max_loss_budget`** → iterative worst-first fair share → integer lots; longs from **collected short credit** only |
| `contract_multiplier` | Pinned = 100 (equity options); embedded in Tier B `quantity` |
| Short side | **Defined-risk only** — reject `short_structure='straddle'` |
| Trade-log grain | One row per `(trade_date, ticker, direction)` |
| Primary metric | `cycle_return_on_capital_at_risk` (+ short/long side splits) |
| Scope | Implementation + contract tests + **synthetic smoke**; **no real-data backtest run** |

---

## Deliverables

| # | Artifact | Path | Status |
|---|----------|------|--------|
| 1 | New config fields + validation | `src/backtest/run_config.py` | ✅ |
| 2 | S5 implementation | `src/backtest/pipeline.py::step5_select_and_size` | ✅ |
| 3 | S8 cycle metrics | `src/backtest/surface_metrics.py` | ✅ |
| 4 | ORCH thin loop | `src/backtest/surface_runner.py` | ✅ |
| 5 | Contract tests | `tests/contract/` | ✅ |
| 6 | Synthetic smoke | `tests/` | ✅ |
| 7 | Data contract fill | [surface_engine_data_contract.md](../surface_engine_data_contract.md) | ✅ |
| 8 | Code cleanup | `src/backtest/pipeline.py` — removed `step6_apply_cost` | ✅ |
| 9 | Sprint memo | [sprint_memos/003_s5_s8_build.md](../sprint_memos/003_s5_s8_build.md) | ✅ |

---

## Work breakdown (build order)

| Phase | Work | Status |
|-------|------|--------|
| 1 | Config — sizing fields + validation | ✅ |
| 2 | S5 Select — per-side cap/rank in `step5` | ✅ |
| 3 | S5 Size — Tier A + Tier B (ADR 004) | ✅ |
| 4 | S5 Simulate — settle + M1–M3 + dollar fields | ✅ |
| 5 | S8 — cycle return + Sharpe on cycle series | ✅ |
| 6 | ORCH — runner delegates S5 | ✅ |
| 7 | Synthetic smoke + full-suite verification | ✅ |
| 8 | Cleanup + docs + sprint closeout | ✅ |

---

## Progress log

| Date | Deliverable / phase | Status | Notes |
|------|---------------------|--------|-------|
| 2026-06-13 | **D1 / Phase 1 — Config fields + validation** | ✅ Done | 15 tests in `test_run_envelope_contract.py`. |
| 2026-06-14 | **D2 / Phase 2 — S5 Select extraction** | ✅ Done | Per-side cap/rank in `step5_select_and_size` (SELECT only). |
| 2026-06-14 | **D2 / Phase 3 — S5 Size (Tier A + Tier B)** | ✅ Done | ADR 004 credit-financed longs; 33+ sizing/select tests. |
| 2026-06-16 | **D2 / Phase 4 — S5 Simulate** | ✅ Done | S7 settle on included rows; M1–M3, `pnl_total`, `capital_at_risk_dollars`, `fill_label`. |
| 2026-06-17 | **D3 / Phase 5 — S8 cycle metrics** | ✅ Done | `cycle_return_on_capital_at_risk` + side splits; Sharpe/drawdown on cycle series. |
| 2026-06-18 | **D4 / Phase 6 — ORCH thin loop** | ✅ Done | `SurfaceRunner` delegates S5; removed `_select_size_and_settle`. |
| 2026-06-20 | **D6 / Phase 7 — Synthetic smoke + full-suite verification** | ✅ Done | See **Phase 7 verification** below. |
| 2026-06-20 | **D7–D9 / Phase 8 — Cleanup + closeout** | ✅ Done | Stale doc language cleaned; `step6_apply_cost` removed; sprint memo written. |

---

## Sprint 003 closeout (2026-06-20)

### What was verified

- **Stage-B synthetic S1→S8 path** — fixtures in `tests/unit/test_surface_runner_data_flow.py`; ORCH contracts in `tests/contract/test_orchestration_contract.py`.
- **Full test suite green** — see commands below.
- **S5/S8/ORCH docs match code** — data contract drift register updated; S6 collapsed into S5; no inline runner S5 logic.

### What was not done (explicit)

- **No real-data validation** — no ORATS/cache backtest run this sprint.
- **No strategy sign-off** — synthetic structural tests only.
- **No live/paper readiness claim**.
- **KB-001** remains open — iron condor M3 denominator ([known_bugs.md](../known_bugs.md)).

### Next sprint

**Sprint 004** — data / split-adjusted universe pipeline and real-data structural validation. Update this file when Sprint 004 starts.

**Carry-over:** `scripts/run_surface_search.py` still constructs configs without `sizing_mode` / Tier B budgets — fails fast at construction (intended).

Full closeout detail: [sprint_memos/003_s5_s8_build.md](../sprint_memos/003_s5_s8_build.md).

---

## Phase 7 verification (2026-06-20)

**Status: passed** — synthetic S1→S8 path verified on fixtures; no real ORATS/cache data; no strategy or live/paper claims.

### Synthetic smoke tests (existing; no new `tests/smoke/` file)

| Acceptance area | Primary tests |
|-----------------|---------------|
| Non-empty `trade_log` / `date_summary` / `run_summary` | `TestSurfaceRunnerDataFlow.test_produces_trade_log_and_summaries` |
| Included long + short + excluded diagnostic row | `test_long_and_short_routing`, `test_invalid_surface_row_excluded_with_reason`; ORCH `TestDiagnosticsBehavior` |
| S5 economics columns on trade log | `TestSurfaceRunnerS5Economics`, ORCH `TestS5ColumnsSurviveIntoTradeLog` |
| Valid S5 economics on included rows | above + positive `capital_at_risk_dollars` assertion |
| `_assembly` not in trade log | ORCH `test_assembly_not_leaked_into_trade_log` |
| S8 cycle metrics vs included rows | `test_cycle_metrics_from_s5_economics`; ORCH `test_excluded_rows_do_not_affect_cycle_return` |
| Diagnostics on/off | ORCH `TestDiagnosticsBehavior` |
| Empty / no-signal path | ORCH `TestEmptyInputs` |

Fixtures: `tests/unit/test_surface_runner_data_flow.py`; reused by `tests/contract/test_orchestration_contract.py`.

### Closeout test commands and results (Phase 8 — 2026-06-20)

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/contract/ -q
# 154 passed in 1.78s

& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_surface_runner_data_flow.py -q
# 8 passed in 0.82s

& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/ -q
# 488 passed in 5.67s
```

**Failures:** none. No Sprint 003 cleanup regressions observed.

---

## Known bugs (deferred fixes)

See **[known_bugs.md](../known_bugs.md)** for the full registry.

| ID | Summary | Where |
|----|---------|-------|
| **KB-001** | Iron condor `body_credit_per_share` = condor short legs, **not** ATM straddle → M3 not comparable vs iron fly | `option_surface.py` → `build_ironcondor_from_surface` |

Do **not** treat M3 cross-structure stats as decision-quality until KB-001 is closed.

---

## Success criteria

- [x] Sizing config fields added with validation; `short_structure='straddle'` rejected; `tier_b_short_max_loss_budget` required for `integer_lots`
- [x] Per-side cap honored ([decision 003](../decisions/003_position_cap_per_side.md)); exclusion strings match code vocabulary
- [x] S5 SELECT + SIZE: both tiers implemented; Tier B per [ADR 004](../decisions/004_tier_b_credit_financed_long.md)
- [x] S5 SIMULATE: settle + M1–M3, `pnl_total`, `capital_at_risk_dollars`, `fill_label` on included rows
- [x] S8 produces `cycle_return_on_capital_at_risk` + side splits; Sharpe on cycle series
- [x] `SurfaceRunner` orchestrates pipeline steps with no duplicated business logic
- [x] Contract tests for S5, S8, and ORCH green — 154 tests in `tests/contract/`
- [x] Synthetic end-to-end smoke green (Phase 7)
- [x] Full suite green (`tests/`) — 488 passed (Phase 7 + Phase 8 closeout re-run)
- [ ] **Financial checks (carry-forward):** leg type, strike, expiry, quantity sign, premium sign, payoff, max loss — **partial** coverage in `test_step5_select_and_size_contract.py`, `test_settle_contract.py`, `test_step3_structures_contract.py`, `tests/unit/test_builders.py`, `tests/unit/test_models.py`; no dedicated end-to-end financial audit suite. Track in Sprint 004 or post-closeout cleanup.
- [x] Data contract § S5 / S8 / ORCH filled and aligned with implementation (Phase 8)
- [x] No real-data backtest treated as go/no-go evidence — sprint scope and closeout explicitly synthetic-only; no real-data run performed

---

## Out of scope (Sprint 003)

- Real-data Tier B backtest / go-no-go run (Sprint 004+)
- Pinning deployable capital ($) or broker thresholds
- Portfolio optimizer, signal/risk-parity weighting, sector or correlation caps
- Iron fly vs condor research matrix
- Precompute (Stage A) code changes — gaps remain logged only
- KB-001 fix

---

## Previous sprint

Sprint 002 closed 2026-06-10 — [sprint_memos/002_data_contracts.md](../sprint_memos/002_data_contracts.md).
