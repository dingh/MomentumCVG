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

> **Precedence (S5/S8/ORCH):** the data contract's S5/S8/ORCH sections still carry pre-design vocabulary (`return_on_allocated_budget`, `return_on_max_loss`-series go/no-go) and lag the design doc — they are flagged as Sprint 003 work in that doc's own drift table. **Until Deliverable 7 fills them at closeout, the S5/S8 design doc (`cycle_return_on_capital_at_risk` = `Σ pnl_total / Σ capital_at_risk_dollars`, M1–M3) is authoritative** for those sections. Build to the design doc, not the stale contract text.

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
| 3 | S8 cycle metrics | `src/backtest/surface_metrics.py` | **Extend existing `build_date_summary` / `summarize_trade_log`** (do not add a new module): add `cycle_return_on_capital_at_risk` + short/long splits, migrate Sharpe/drawdown off body-credit onto the cycle series |
| 4 | ORCH thin loop | `src/backtest/surface_runner.py` | Runner delegates to pipeline steps; no inline S5 business logic |
| 5 | Contract tests | `tests/contract/` | `test_step5_select_and_size_contract.py`, `test_run_metrics_contract.py`, `test_orchestration_contract.py` |
| 6 | Synthetic smoke | `tests/` | End-to-end S1→S8 on synthetic fixtures (no real cache) |
| 7 | Data contract fill | [surface_engine_data_contract.md](../surface_engine_data_contract.md) | Complete § S5 / S8 / ORCH from the design doc; **replace** stale `return_on_allocated_budget` / `return_on_max_loss`-series language with `cycle_return_on_capital_at_risk` + M1–M3 (design doc is authoritative until then — see Precedence note above) |
| 8 | Code cleanup | `src/backtest/pipeline.py` | Remove deprecated `step6_apply_cost` stub |
| 9 | Sprint memo | `docs/sprint_memos/003_s5_s8_build.md` | Closeout, drift resolved, follow-ons |

---

## Work breakdown (build order)

Tests are written **with** the code in each phase (not deferred). Each phase below lists its test expectation.

| Phase | Work | Test expectation (write with the code) |
|-------|------|-------|
| 1 | **Config** — add the six fields + `__post_init__` validation; reject short straddle | **Tests first.** Validation rejects unset `sizing_mode` and `short_structure='straddle'`; accepts each valid `sizing_mode` / `tier_a_mode` combo |
| 2 | **S5 Select** — extract per-side cap/rank from runner into `step5`; reuse `signal_rank_pct`, exclusion strings (`max_names_cap`, `invalid_max_loss`) | Per-side cap honored ([decision 003](../decisions/003_position_cap_per_side.md)); exclusion strings emitted; pure function on S4 output |
| 3 | **S5 Size** — Tier A (per-side budget ÷ count, fractional, no ×100) and Tier B (integer lots, ×100, capital binding) | Both tiers: **quantity sign, premium sign, max-loss geometry**; Tier A budget÷count; Tier B integer lots + capital binding |
| 4 | **S5 Simulate** — settle (S7) on included rows; compute M1–M3, `pnl_total`, `capital_at_risk_dollars`, `fill_label` | M1–M3 / `pnl_total` / `capital_at_risk_dollars` match hand calc; denominators derived in S5 from S3 fields; `fill_label` set |
| 5 | **S8** — `cycle_return_on_capital_at_risk` per `trade_date` + side splits; Sharpe/drawdown on cycle series; keep `availability_rate`/`hit_rate` | Cycle return + short/long splits; empty-side / zero-denominator → `NaN`; Sharpe on cycle series (off body-credit) |
| 6 | **ORCH** — runner = thin S1→S8 loop; settle inside S5 | Contract test: runner delegates to pipeline steps; no duplicated S5 business logic |
| 7 | **Synthetic smoke + full-suite verification** — end-to-end S1→S8 on synthetic fixtures (no real cache); run full `tests/` | Smoke green; full suite green — **no regressions on the 43 Sprint 002 contract tests** (per-phase contract tests already landed in Phases 1–6) |
| 8 | **Cleanup + docs** — remove `step6` stub; fill data-contract S5/S8/ORCH; write memo | — |

---

## Progress log

| Date | Deliverable / phase | Status | Notes |
|------|---------------------|--------|-------|
| 2026-06-13 | **D1 / Phase 1 — Config fields + validation** | ✅ Done | Added 6 fields to `BacktestRunConfig` (`sizing_mode`, `tier_a_mode`, `tier_a_short_budget`, `tier_a_long_budget`, `contract_multiplier=100`, `deployable_capital=None`); `__post_init__` rejects unset/invalid `sizing_mode`, rejects `short_structure='straddle'`, enforces Tier A budget rules, `contract_multiplier>0`, `deployable_capital>0` when set. 15 new tests in `test_run_envelope_contract.py`. Contract subset 58 ✅; full suite 393 ✅. |

**Next start here → Deliverable 2 / Phase 2 (S5 Select).** Extract per-side cap/rank from `SurfaceRunner._select_size_and_settle` into `pipeline.step5_select_and_size` (currently a `pass` stub); reuse exclusion strings `no_tradeable_structure`, `earnings_exclusion`, `max_names_cap`, `invalid_max_loss`; honor [decision 003](../decisions/003_position_cap_per_side.md) per-side cap. Tests: `tests/contract/test_step5_select_and_size_contract.py`.

> **Carry-over note:** `scripts/run_surface_search.py` still constructs configs without `sizing_mode`, so it now fails fast at construction (intended per Q8b). Wire a `sizing_mode` (and Tier A budgets) arg when the runner delegates to `step5` in Deliverable 4 (ORCH) — out of scope for D1.

---

## Success criteria

- [x] Six config fields added with validation; `short_structure='straddle'` rejected; tests cover each (D1 — 2026-06-13)
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
