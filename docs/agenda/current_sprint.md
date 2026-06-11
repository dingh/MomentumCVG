# Current sprint — 003

**Updated:** 2026-06-10  
**Status:** Active — Build  
**Mode:** **Build** (implementation scoped to the accepted Sprint 002 design; tests written with code)

---

## Goal

Turn the accepted Sprint 002 design into **trusted code**: implement the **S5 trade-construction layer** (select + size + simulate + return), **S8 cycle-return metrics**, and a **thin S1→S8 orchestration**, each with **contract tests** plus one **synthetic end-to-end smoke**.

Source of truth (all Accepted, do not re-litigate without an ADR):
- [surface_engine_portfolio_metrics_design.md](../surface_engine_portfolio_metrics_design.md) — S5/S8 design
- [surface_engine_data_contract.md](../surface_engine_data_contract.md) — component contracts
- [decisions/003_position_cap_per_side.md](../decisions/003_position_cap_per_side.md) — per-side cap

**Capital / dollars:** keep `deployable_capital` optional (`None` in v1); sizing budgets remain abstract risk units. Do not pin $1M this sprint.

---

## HD decisions (locked for Sprint 003)

| Topic | Decision |
|-------|----------|
| Tiers | Build **both** Tier A (`conceptual`) and Tier B (`integer_lots`) in S5 |
| `sizing_mode` | **Required** config field — no default; run fails fast if unset |
| Tier A control | **Per-side total budget** ÷ name count (`tier_a_mode` ∈ {`equal_premium`, `equal_max_loss`}) |
| `contract_multiplier` | Pinned = 100 (equity options) |
| Short side | **Defined-risk only** — reject `short_structure='straddle'` (no naked short straddle in v1) |
| Trade-log grain | One row per `(trade_date, ticker, direction)` |
| Capital | Hard constraint (when set): drop by rank / skip date; never overrun, never rescale |
| Primary metric | `cycle_return_on_capital_at_risk` (+ short/long side splits) |
| Scope | Implementation + contract tests + **synthetic smoke**; **no real-data backtest run** |

---

## Deliverables

| # | Artifact | Path | Purpose |
|---|----------|------|---------|
| 1 | New config fields + validation | `src/backtest/run_config.py` | `sizing_mode`, `tier_a_mode`, `tier_a_short_budget`, `tier_a_long_budget`, `contract_multiplier`, `deployable_capital`; reject `short_structure='straddle'` |
| 2 | S5 implementation | `src/backtest/pipeline.py::step5_select_and_size` | Select (per-side cap) + size (both tiers) + simulate (settle + returns) |
| 3 | S8 cycle metrics | `src/backtest/` metrics | `cycle_return_on_capital_at_risk` + side splits; Sharpe/drawdown on cycle series |
| 4 | ORCH thin loop | `src/backtest/surface_runner.py` | Runner delegates to pipeline steps; no inline S5 business logic |
| 5 | Contract tests | `tests/contract/` | `test_step5_select_and_size_contract.py`, `test_run_metrics_contract.py`, `test_orchestration_contract.py` |
| 6 | Synthetic smoke | `tests/` | End-to-end S1→S8 on synthetic fixtures (no real cache) |
| 7 | Data contract fill | [surface_engine_data_contract.md](../surface_engine_data_contract.md) | Complete § S5 / S8 / ORCH from the design doc |
| 8 | Code cleanup | `src/backtest/pipeline.py` | Remove deprecated `step6_apply_cost` stub |
| 9 | Sprint memo | `docs/sprint_memos/003_s5_s8_build.md` | Closeout, drift resolved, follow-ons |

---

## Work breakdown (build order)

| Phase | Work | Notes |
|-------|------|-------|
| 1 | **Config** — add the six fields + `__post_init__` validation; reject short straddle | Tests first for validation |
| 2 | **S5 Select** — extract per-side cap/rank from runner into `step5`; reuse `signal_rank_pct`, exclusion strings (`max_names_cap`, `invalid_max_loss`) | Pure function on S4 output |
| 3 | **S5 Size** — Tier A (per-side budget ÷ count, fractional, no ×100) and Tier B (integer lots, ×100, capital binding) | Verify quantity sign, premium sign, max-loss geometry |
| 4 | **S5 Simulate** — settle (S7) on included rows; compute M1–M3, `pnl_total`, `capital_at_risk_dollars`, `fill_label` | Denominators derived in S5 from S3 fields |
| 5 | **S8** — `cycle_return_on_capital_at_risk` per `trade_date` + side splits; Sharpe/drawdown on cycle series; keep `availability_rate`/`hit_rate` | Migrate off body-credit Sharpe |
| 6 | **ORCH** — runner = thin S1→S8 loop; settle inside S5 | Remove inline S5 duplication |
| 7 | **Tests** — S5/S8/ORCH contract tests + synthetic end-to-end smoke | Written with implementation |
| 8 | **Cleanup + docs** — remove `step6` stub; fill data-contract S5/S8/ORCH; write memo | — |

---

## Success criteria

- [ ] Six config fields added with validation; `short_structure='straddle'` rejected; tests cover each
- [ ] `step5_select_and_size` implements select + both sizing tiers + simulate; emits the full trade-log schema (M1–M3, `pnl_total`, `capital_at_risk_dollars`, `quantity`, `fill_label`)
- [ ] Per-side cap honored ([decision 003](../decisions/003_position_cap_per_side.md)); exclusion strings match code vocabulary
- [ ] S8 produces `cycle_return_on_capital_at_risk` + `short_cycle_return` / `long_cycle_return`; Sharpe on cycle series
- [ ] `SurfaceRunner` orchestrates pipeline steps with no duplicated business logic
- [ ] Contract tests for S5/S8/ORCH green; synthetic end-to-end smoke green
- [ ] Full suite green (`tests/` ) — no regressions on the 43 Sprint 002 contract tests
- [ ] Financial checks verified in tests: leg type, strike, expiry, **quantity sign, premium sign**, payoff, **max loss** (per AGENTS.md)
- [ ] Data contract § S5 / S8 / ORCH filled; design doc drift items resolved
- [ ] No real-data backtest treated as go/no-go evidence

---

## Session plan (suggested)

| Session | Work |
|---------|------|
| A | Config fields + validation (Phase 1); S5 Select extraction (Phase 2) + tests |
| B | S5 Size both tiers (Phase 3) + S5 Simulate / returns (Phase 4) + tests |
| C | S8 cycle metrics (Phase 5) + ORCH thin loop (Phase 6) + tests |
| D | Synthetic smoke (Phase 7); cleanup, data-contract fill, close memo (Phase 8) |

---

## Out of scope (Sprint 003)

- Real-data Tier B backtest / go-no-go run (Sprint 004+)
- Pinning deployable capital ($) or broker thresholds
- Portfolio optimizer, signal/risk-parity weighting, sector or correlation caps
- Multi-date book state / replacement policy across rebalances
- Iron fly vs condor research matrix
- Precompute (Stage A) code changes — gaps remain logged only

---

## Agent instructions

1. Read this file and [v1_spec_pins.md](../v1_spec_pins.md) at session start.
2. **Build mode:** implement only the scoped design; write tests **with** the code, not after.
3. Do not change strategy/financial logic outside the approved S5/S8 scope; verify option leg type, strike, expiry, quantity sign, premium sign, payoff, and max loss when touching sizing/settle.
4. When runner inline logic disagrees with the design, fix toward the **Accepted** design; record any drift in the memo.
5. Run the focused contract subset after each phase; run the full suite before close. Use venv `C:/MomentumCVG_env/venv/Scripts/python.exe`.
6. Canonical path is **SurfaceRunner**; no broker/live execution code.

---

## Previous sprint

Sprint 002 closed 2026-06-10 — [sprint_memos/002_data_contracts.md](../sprint_memos/002_data_contracts.md).
