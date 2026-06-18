# Current sprint — 003

**Updated:** 2026-06-16  
**Status:** Active — Build  
**Mode:** **Build** (implementation scoped to the accepted Sprint 002 design; tests written with code)

---

## Goal

Turn the accepted Sprint 002 design into **trusted code**: implement the **S5 trade-construction layer** (select + size + simulate + return), **S8 cycle-return metrics**, and a **thin S1→S8 orchestration**, each with **contract tests** plus one **synthetic end-to-end smoke**.

Source of truth (all Accepted, do not re-litigate without an ADR):
- [surface_engine_portfolio_metrics_design.md](../surface_engine_portfolio_metrics_design.md) — S5/S8 design (Tier A; Tier B sizing superseded in part by ADR 004 — see below)
- [surface_engine_data_contract.md](../surface_engine_data_contract.md) — component contracts
- [decisions/003_position_cap_per_side.md](../decisions/003_position_cap_per_side.md) — per-side cap
- [decisions/004_tier_b_credit_financed_long.md](../decisions/004_tier_b_credit_financed_long.md) — **Tier B sizing** (integer lots, credit-financed longs, fair-share passes)

> **Precedence (S5/S8/ORCH):** the data contract's S5/S8/ORCH sections still carry pre-design vocabulary (`return_on_allocated_budget`, `return_on_max_loss`-series go/no-go) and lag the design doc — they are flagged as Sprint 003 work in that doc's own drift table. **Until Deliverable 7 fills them at closeout, the S5/S8 design doc (`cycle_return_on_capital_at_risk` = `Σ pnl_total / Σ capital_at_risk_dollars`, M1–M3) is authoritative** for those sections, **except Tier B sizing mechanics** which follow [ADR 004](../decisions/004_tier_b_credit_financed_long.md) (supersedes design doc § Tier B per-name `max_loss_budget_per_trade` and `deployable_capital` CAR binding).

**Capital / dollars:** sizing budgets remain abstract risk units; do not pin $1M this sprint. **`deployable_capital`** stays on config (optional, validated when set) but is **not used in S5 Tier B sizing** per ADR 004 — long budget = collected short credit only; short budget = `tier_b_short_max_loss_budget`.

---

## HD decisions (locked for Sprint 003)

| Topic | Decision |
|-------|----------|
| Tiers | Build **both** Tier A (`conceptual`) and Tier B (`integer_lots`) in S5 |
| `sizing_mode` | **Required** config field — no default; run fails fast if unset |
| Tier A control | **Per-side total budget** ÷ name count (`tier_a_mode` ∈ {`equal_premium`, `equal_max_loss`}) |
| Tier B control | **[ADR 004](../decisions/004_tier_b_credit_financed_long.md):** shorts from **`tier_b_short_max_loss_budget`** (total max-loss $) → iterative worst-first fair share → integer lots; longs from **collected short credit** only → same fair-share on premium; `quantity` in share-equivalent units (`contracts × 100`) |
| `contract_multiplier` | Pinned = 100 (equity options); embedded in Tier B `quantity`, not a separate simulate multiplier |
| Short side | **Defined-risk only** — reject `short_structure='straddle'` (no naked short straddle in v1) |
| Trade-log grain | One row per `(trade_date, ticker, direction)` |
| Capital (Tier B) | **Short:** deployed max-loss ≤ `tier_b_short_max_loss_budget`. **Long:** premium spend ≤ collected short credit (self-financing). **`deployable_capital` not enforced in S5 Tier B** (reserved for future). Never rescale quantities — drop by fair-share exclusion. |
| Primary metric | `cycle_return_on_capital_at_risk` (+ short/long side splits) |
| Scope | Implementation + contract tests + **synthetic smoke**; **no real-data backtest run** |

---

## Deliverables

| # | Artifact | Path | Purpose |
|---|----------|------|---------|
| 1 | New config fields + validation | `src/backtest/run_config.py` | `sizing_mode`, `tier_a_mode`, `tier_a_short_budget`, `tier_a_long_budget`, `tier_b_short_max_loss_budget`, `contract_multiplier`, `deployable_capital`; reject `short_structure='straddle'` |
| 2 | S5 implementation | `src/backtest/pipeline.py::step5_select_and_size` | Select (per-side cap) + size (both tiers) + simulate (settle + returns) |
| 3 | S8 cycle metrics | `src/backtest/surface_metrics.py` | **Extend existing `build_date_summary` / `summarize_trade_log`** (do not add a new module): add `cycle_return_on_capital_at_risk` + short/long splits, migrate Sharpe/drawdown off body-credit onto the cycle series |
| 4 | ORCH thin loop | `src/backtest/surface_runner.py` | Runner delegates to pipeline steps; no inline S5 business logic |
| 5 | Contract tests | `tests/contract/` | `test_step5_select_and_size_contract.py`, `test_run_metrics_contract.py`, `test_orchestration_contract.py` |
| 6 | Synthetic smoke | `tests/` | End-to-end S1→S8 on synthetic fixtures (no real cache) |
| 7 | Data contract fill | [surface_engine_data_contract.md](../surface_engine_data_contract.md) | Complete § S5 / S8 / ORCH from design doc + **ADR 004**; replace stale Tier B / `deployable_capital` sizing language |
| 8 | Code cleanup | `src/backtest/pipeline.py` | Remove deprecated `step6_apply_cost` stub |
| 9 | Sprint memo | `docs/sprint_memos/003_s5_s8_build.md` | Closeout, drift resolved, follow-ons |

---

## Work breakdown (build order)

Tests are written **with** the code in each phase (not deferred). Each phase below lists its test expectation.

| Phase | Work | Test expectation (write with the code) |
|-------|------|-------|
| 1 | **Config** — sizing fields + `__post_init__` validation; reject short straddle | Validation rejects unset `sizing_mode` and `short_structure='straddle'`; Tier A budget rules; **`tier_b_short_max_loss_budget` required for `integer_lots`** |
| 2 | **S5 Select** — extract per-side cap/rank from runner into `step5` | Per-side cap honored ([decision 003](../decisions/003_position_cap_per_side.md)); exclusion strings emitted; pure function on S4 output |
| 3 | **S5 Size** — Tier A (per-side budget ÷ count, fractional) and Tier B ([ADR 004](../decisions/004_tier_b_credit_financed_long.md): total short max-loss fair share + credit-financed long fair share; `quantity` = contracts × 100) | Tier A budget÷count; Tier B fair-share drops (`max_loss_exceeds_fair_share`, `premium_exceeds_fair_share`, `no_short_credit`); quantity sign; long spend ≤ short credit; `deployable_capital` ignored in Tier B |
| 4 | **S5 Simulate** — settle (S7) on included rows; compute M1–M3, `pnl_total`, `capital_at_risk_dollars`, `fill_label` | M1–M3 / `pnl_total` / `capital_at_risk_dollars` match hand calc (`pnl_total = abs(quantity) × pnl_per_share` both tiers; `quantity` sign = long/short only); `fill_label` set |
| 5 | **S8** — `cycle_return_on_capital_at_risk` per `trade_date` + side splits; Sharpe/drawdown on cycle series | Cycle return + short/long splits; empty-side / zero-denominator → `NaN`; Sharpe on cycle series (off body-credit) |
| 6 | **ORCH** — runner = thin S1→S8 loop; settle inside S5 | Contract test: runner delegates to pipeline steps; no duplicated S5 business logic |
| 7 | **Synthetic smoke + full-suite verification** | Smoke green; full suite green — no regressions on Sprint 002 contract tests |
| 8 | **Cleanup + docs** — remove `step6` stub; fill data-contract S5/S8/ORCH; write memo | — |

---

## Progress log

| Date | Deliverable / phase | Status | Notes |
|------|---------------------|--------|-------|
| 2026-06-13 | **D1 / Phase 1 — Config fields + validation** | ✅ Done | Initial 6 sizing fields + validation. 15 tests in `test_run_envelope_contract.py`. Contract subset 58 ✅. |
| 2026-06-14 | **D2 / Phase 2 — S5 Select extraction** | ✅ Done | Per-side cap/rank in `step5_select_and_size` (SELECT only). 19 tests → grew with Phase 3. Runner still inline until D4. |
| 2026-06-14 | **D2 / Phase 3 — S5 Size (Tier A + Tier B)** | ✅ Done | Tier A `conceptual` + Tier B `integer_lots` per [ADR 004](../decisions/004_tier_b_credit_financed_long.md): credit-financed longs, symmetric worst-first fair share, `tier_b_short_max_loss_budget`, share-equivalent `quantity`. Added `tier_b_short_max_loss_budget` config field. `test_step5_select_and_size_contract.py` (33 sizing/select tests). Contract subset 94 ✅. |
| 2026-06-16 | **D2 / Phase 4 — S5 Simulate** | ✅ Done | S7 settle on included rows; M1–M3, `pnl_total`, `capital_at_risk_dollars`, `fill_label`. Dollar fields use `abs(quantity)` (sign = direction only). 57 tests in `test_step5_select_and_size_contract.py`. Contract subset 117 ✅. Runner still inline until D4 (ORCH). |

**Next start here → Deliverable 3 / Phase 5 (S8 cycle metrics).** `cycle_return_on_capital_at_risk` + short/long splits; Sharpe on cycle series. Runner delegation remains Deliverable 4 (ORCH).

> **Carry-over note:** `scripts/run_surface_search.py` still constructs configs without `sizing_mode` / Tier B budgets — fails fast at construction (intended). Wire sizing args when runner delegates to `step5` in Deliverable 4 (ORCH).

---

## Known bugs (deferred fixes)

See **[known_bugs.md](../known_bugs.md)** for the full registry.

| ID | Summary | Where |
|----|---------|-------|
| **KB-001** | Iron condor `body_credit_per_share` = condor short legs, **not** ATM straddle → M3 not comparable vs iron fly | `option_surface.py` → `build_ironcondor_from_surface` |

Fix KB-001 when reviewing `option_surface.py` (post–Sprint 003 or iron fly vs condor research). Do **not** treat M3 cross-structure stats as decision-quality until closed.

---

## Success criteria

- [x] Sizing config fields added with validation; `short_structure='straddle'` rejected; `tier_b_short_max_loss_budget` required for `integer_lots` (D1 + ADR 004 — 2026-06-14)
- [x] Per-side cap honored ([decision 003](../decisions/003_position_cap_per_side.md)); exclusion strings match code vocabulary (D2 SELECT — 2026-06-14)
- [x] S5 SELECT + SIZE: both tiers implemented; Tier B per [ADR 004](../decisions/004_tier_b_credit_financed_long.md) (Phase 3 — 2026-06-14)
- [x] S5 SIMULATE: settle + M1–M3, `pnl_total`, `capital_at_risk_dollars`, `fill_label` on included rows (Phase 4 — 2026-06-16; `pnl_total = abs(quantity) × pnl_per_share`)
- [ ] S8 produces `cycle_return_on_capital_at_risk` + `short_cycle_return` / `long_cycle_return`; Sharpe on cycle series
- [ ] `SurfaceRunner` orchestrates pipeline steps with no duplicated business logic
- [ ] Contract tests for S5/S8/ORCH green; synthetic end-to-end smoke green
- [ ] Full suite green (`tests/`) — no regressions on Sprint 002 contract tests
- [ ] Financial checks verified in tests: leg type, strike, expiry, **quantity sign, premium sign**, payoff, **max loss** (per AGENTS.md)
- [ ] Data contract § S5 / S8 / ORCH filled; design doc Tier B drift resolved via ADR 004 in Deliverable 7
- [ ] No real-data backtest treated as go/no-go evidence

---

## Session plan (suggested)

| Session | Work |
|---------|------|
| A | Config fields + validation (Phase 1); S5 Select extraction (Phase 2) + tests ✅ |
| B | S5 Size both tiers (Phase 3) ✅ + S5 Simulate / returns (Phase 4) ✅ + tests |
| C | S8 cycle metrics (Phase 5) + ORCH thin loop (Phase 6) + tests |
| D | Synthetic smoke (Phase 7); cleanup, data-contract fill, close memo (Phase 8) |

---

## Out of scope (Sprint 003)

- Real-data Tier B backtest / go-no-go run (Sprint 004+)
- Pinning deployable capital ($) or broker thresholds; **`deployable_capital` enforcement in S5 Tier B** (deferred — ADR 004)
- Portfolio optimizer, signal/risk-parity weighting, sector or correlation caps
- Multi-date book state / replacement policy across rebalances
- Iron fly vs condor research matrix
- Precompute (Stage A) code changes — gaps remain logged only

---

## Agent instructions

1. Read this file and [v1_spec_pins.md](../v1_spec_pins.md) at session start.
2. **Build mode:** implement only the scoped design; write tests **with** the code, not after.
3. **Tier B sizing:** follow [ADR 004](../decisions/004_tier_b_credit_financed_long.md), not stale design-doc § Tier B per-name `max_loss_budget_per_trade` / `deployable_capital` CAR binding.
4. Do not change strategy/financial logic outside the approved S5/S8 scope; verify option leg type, strike, expiry, quantity sign, premium sign, payoff, and max loss when touching sizing/settle.
5. When runner inline logic disagrees with the accepted design, fix toward design + ADRs; record drift in the memo.
6. Run the focused contract subset after each phase; run the full suite before close. Use venv `C:/MomentumCVG_env/venv/Scripts/python.exe`.
7. Canonical path is **SurfaceRunner**; no broker/live execution code.

---

## Previous sprint

Sprint 002 closed 2026-06-10 — [sprint_memos/002_data_contracts.md](../sprint_memos/002_data_contracts.md).
