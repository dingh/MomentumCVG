# Correctness audit — Sprint 001

**Status:** Reviewed by HD  
**Date:** 2026-05-24  
**Mode:** Audit  
**Scope:** Tests and financial correctness risks for the surface-first v1 path.

---

## Executive summary

The repo has strong unit tests for the **option math layer**: core model signs, straddle builder behavior, iron butterfly builder behavior, and surface-based straddle/iron-fly/iron-condor assembly. In particular, the surface tests already include hand-calculated entry cost, net credit, max loss, fill assumptions, and expiry settlement for iron fly and iron condor.

The main correctness gap is the **engine layer**: there are no tests for `SurfaceRunner`, precompute output contracts, CLI wiring, pipeline integration, sizing/contracts, or run-level metrics. That means the option math is relatively well protected, but a full backtest can still be wrong due to date selection, missing surfaces, sizing, capital denominator, or summary calculations. Human review also flagged that confidence in the proposed Session B test should remain provisional until the SurfaceRunner functionality and data flow are explicitly mapped.

---

## Test inventory

| File | Approx. test entries (rg) | Main coverage |
|------|---------------------------|---------------|
| `tests/unit/test_builders.py` | 79 tests | `StraddleBuilder`, `IronButterflyBuilder`, strike selection, candidate enumeration, spread/yield filters, error handling |
| `tests/unit/test_option_surface_ironfly.py` | 46 tests | Surface iron fly entry economics, mid/cross fills, wing selection, max loss, settlement, filters, missing data |
| `tests/unit/test_option_surface_ironcondor.py` | 45 tests | Surface iron condor entry economics, mid/cross fills, wing selection, max loss, settlement, filters, missing data |
| `tests/unit/test_option_surface_straddle.py` | 39 tests | Surface long/short straddle entry, fills, max loss for long, settlement, spread filters |
| `tests/unit/test_models.py` | 50 tests | `OptionQuote`, `OptionLeg`, `OptionStrategy`, `Position`, signs, payoff, PnL |
| `tests/unit/test_momentum_calculator.py` | 49 tests | Momentum feature behavior |
| `tests/unit/test_cvg_calculator.py` | 45 tests | CVG feature behavior |
| `tests/unit/test_spot_price_db.py` | 15 tests | Spot DB behavior |
| `tests/unit/test_straddle_analyzer.py` | 15 tests | Straddle analyzer behavior |

Baseline from Week 0: **326 passed** via `C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/ -q`.

---

## What is well verified

### Core sign and payoff model

| Behavior | Status | Evidence |
|----------|--------|----------|
| Long vs short leg sign | Covered | `test_models.py`, `test_option_surface_*` |
| Option intrinsic value | Covered | `test_models.py` |
| Strategy payoff from signed quantities | Covered | `test_models.py` |
| Position PnL = exit value - entry cost | Covered | `test_models.py` |

### Surface assembly math

| Behavior | Status | Evidence |
|----------|--------|----------|
| Long and short straddle entry economics | Covered | `test_option_surface_straddle.py` |
| Iron fly leg signs and body strike | Covered | `test_option_surface_ironfly.py` |
| Iron fly mid/cross fill arithmetic | Covered | `test_option_surface_ironfly.py` |
| Iron fly max loss and expiry settlement | Covered | `test_option_surface_ironfly.py` |
| Iron condor leg signs and wing selection | Covered | `test_option_surface_ironcondor.py` |
| Iron condor max loss and expiry settlement | Covered | `test_option_surface_ironcondor.py` |
| Spread-cost filters | Covered at assembly level | `test_option_surface_*` |

### Builder/reference layer

| Behavior | Status | Evidence |
|----------|--------|----------|
| ATM straddle construction from raw chain | Covered | `test_builders.py` |
| Symmetric iron butterfly builder | Covered heavily | `test_builders.py` |
| Builder error handling for missing/invalid legs | Covered | `test_builders.py` |

---

## Highest-risk untested areas

### P0 — blocks decision-quality engine work

| Risk | Why dangerous | Suggested test / verification | Effort |
|------|---------------|-------------------------------|--------|
| SurfaceRunner CLI wiring untested | `run_surface_search.py` appears to pass `contract_multiplier` to `SurfaceDataPaths`, which has no such field; CLI may fail immediately. | Smoke test invoking config construction or CLI parser path without large data | S |
| Runner sizing not implemented as advertised | Current runner settles per-share and does not appear to compute integer contracts or dollar PnL from `max_loss_budget_per_trade`. | Unit test `_select_size_and_settle()` on synthetic structures expects contracts, max-loss dollars, PnL dollars | M |
| Per-side cap semantics untested in contract suite | Runner uses `max_names_per_side` per direction ([decision 003](../decisions/003_position_cap_per_side.md)); S5 contract test not yet written. | `test_step5_select_and_size_contract.py`: long and short pools capped independently | S/M |
| Trade date schedule untested | Feature dates may not match weekly precomputed surface dates; missing surfaces can dominate results. | Unit test `_get_trade_dates()` or smoke test with synthetic features/surface date mismatch | S/M |
| Metrics capital denominator wrong for v1 | Current `surface_metrics.py` summarizes return on body credit; v1 needs return on max-loss budget and Sharpe. | Unit tests for metrics on synthetic dollar PnL/date returns | M |

### P1 — needed before Tier B

| Risk | Why dangerous | Suggested test / verification | Effort |
|------|---------------|-------------------------------|--------|
| Precompute schema contract untested | Runner assumes columns exist and semantics are stable. | Test `OptionSurfaceDB` loads minimal expected schema and rejects invalid surfaces clearly | S |
| Precompute coverage unknown | Full backtest may mostly measure data availability, not strategy edge. | Coverage report plus assertions on valid rate by year/date/bucket | M |
| Universe PIT only unit-tested indirectly | `step1_get_universe()` code looks PIT, but no test inventory shows direct coverage. | Direct unit tests for month snapshot lookup and rank thresholds | S |
| Feature PIT assumption not audited | Signal features may be correct, but the runner relies on precomputed features. | Feature generation audit / tests by lag window | M |
| Long/short side allocation untested | Long straddle and short structures have different risk denominators. | Runner tests for side-specific budgets and side enable/disable behavior | M |

### P2 — before shadow/paper

| Risk | Why dangerous | Suggested test / verification | Effort |
|------|---------------|-------------------------------|--------|
| Intended order output untested | Paper/manual execution requires exact strikes, sides, quantities, prices. | Golden-file test for a shadow order sheet from synthetic trade log | M |
| Early close unsupported | Live will likely not hold to expiry forever. | Future mark-to-market / exit-surface tests | L |
| Sector/per-name caps missing | Concentration risk before real capital. | Portfolio cap tests once module exists | M |

---

## Sprint 001 verification test recommendation

The original Sprint 001 plan asked for one iron-fly payoff/max-loss test. After audit, the low-level assembly file already has strong tests for this:

- `TestIronFlyMidFill`
- `TestIronFlyCrossFill`
- `TestIronFlyAsymmetricWings`
- `TestIronFlySettle`

However, the **missing boundary** is how `SurfaceRunner` consumes saved artifacts and turns them into selected, settled trade rows. Therefore, the best Sprint 001 verification test is now adjusted:

> Add one synthetic runner-level data-flow test that verifies `SurfaceRunner.run_single_config()` can consume minimal liquidity/features/surface parquet fixtures, produce a coherent trade log, and preserve hand-calculated payoff/max-loss semantics for an included short structure.

This recommendation changed after the Session A.1 mapping in `docs/surface_runner_data_flow.md`. HD approved the preferred Session B target as a small synthetic `SurfaceRunner.run_single_config()` data-flow test because it exercises the real canonical path from input artifacts through trade log and summaries.

### Session A.1 test-selection gate

Before Session B starts, answer:

| Question | Why it matters |
|----------|----------------|
| What is the minimum complete SurfaceRunner data path from inputs to trade log? | Prevents testing an internal helper while the surrounding contract remains ambiguous. |
| Which functions currently implement each required backtest responsibility? | Separates missing functionality from untested functionality. |
| Which required function is most likely to produce wrong backtest results today? | Keeps the single Sprint 001 test focused on real financial risk. |
| Does `_select_size_and_settle()` remain the best boundary after mapping? | Confirms or replaces the current recommendation. |

Session A.1 answer: `_select_size_and_settle()` is not the best first boundary if only one test is allowed. HD approved a small full-run synthetic fixture test through `SurfaceRunner.run_single_config()`, which still covers selection/settlement but also validates date, schema, universe, signal, and surface assembly contracts.

### Proposed test target

New file:

```text
tests/unit/test_surface_runner_data_flow.py
```

### Proposed scenarios

| Scenario | Expected result |
|----------|-----------------|
| One synthetic full run with temporary liquidity/features/surface parquets | `SurfaceRunner.run_single_config()` produces trade log, date summary, and run summary |
| PIT universe snapshot contains multiple names | expected names enter the candidate set from the most recent `month_date <= trade_date` |
| Long and short signal candidates are present | long candidate routes to long straddle; short candidate routes to configured short structure |
| Included short iron fly settles at hand-calculated expiry spot | expected `pnl_per_share` is preserved through the real runner path |
| Missing/invalid surface candidate is present with diagnostics enabled | excluded row records `no_tradeable_structure` or metadata/assembly failure |
| Desired v1 sizing fields are checked | test may intentionally fail or be marked expected-fail for missing `contracts`, `pnl_dollars`, and realized return-on-max-loss |

### Why this test reduces real-money risk

It checks the boundary where correct option math becomes a backtest row through the actual canonical runner path. That is the layer most likely to hide future bugs in date alignment, schema assumptions, sizing, inclusion, and reporting. It also makes the missing `contracts` / dollar PnL behavior explicit, which should drive Sprint 002. HD is comfortable with the test intentionally failing on missing desired v1 behavior rather than weakening the assertion to match the incomplete engine.

### Alternative

If you want to keep Sprint 001 strictly at assembly level, add another explicit truth-table row to `test_option_surface_ironfly.py`. But that would have lower marginal value because the existing file already covers expiry payoff and max loss well.

---

## Recommended test backlog

### Sprint 001

| Test | Priority | Effort |
|------|----------|--------|
| Synthetic `SurfaceRunner.run_single_config()` data-flow test | P0 | M |
| Follow-up review of implemented surface DB / assembly / settlement tests | P1 | M |

### Sprint 002

| Test | Priority | Effort |
|------|----------|--------|
| CLI/config smoke test for `run_surface_search.py` construction | P0 | S |
| Runner dollar sizing / contract multiplier test | P0 | M |
| `step1_get_universe()` PIT lookup and rank threshold test | P0 | S |
| `surface_metrics.py` return-on-max-loss + Sharpe tests | P0 | M |

### Sprint 003+

| Test | Priority | Effort |
|------|----------|--------|
| Precompute schema / manifest test | P1 | M |
| Surface coverage report regression test on small fixture | P1 | M |
| Short fly vs condor config grid smoke | P1 | M |
| Shadow order sheet golden-file test | P2 | M |

---

## Areas safe to use now

- Surface assembly arithmetic for straddle, iron fly, and iron condor on synthetic surfaces.
- `FillAssumption.mid()` and `FillAssumption.cross()` at the leg/structure level.
- Core model payoff/PnL signs.
- Builder-level iron butterfly construction from raw chain fixtures.

## Areas not yet safe to trust for capital decisions

- Any `SurfaceRunner` run summary as a capital-allocation metric.
- Sharpe or drawdown from `surface_metrics.py` as a v1 go/no-go metric without max-loss/dollar PnL fixes.
- 2020+ results until surface coverage diagnostics and smoke run pass.
- Manual/paper order output, because no shadow order contract exists yet.

---

## Open questions for review

1. Do you approve moving Session B to a synthetic `SurfaceRunner.run_single_config()` data-flow test?
2. Should the test assert current behavior only, or intentionally fail on missing desired v1 fields such as `contracts` and `pnl_dollars`?
3. Should we create small synthetic runner fixtures now, or wait until Sprint 002 when sizing is implemented?
4. Do you want P0 test work to prioritize **engine mechanics** over additional option-math truth tables?

