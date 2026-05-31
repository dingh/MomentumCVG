# Surface engine — data contract (source of truth)

**Status:** Draft — Sprint 002  
**Last updated:** 2026-05-28  
**Audience:** HD + agents; supersedes informal runner behavior as spec authority

---

## How to use this document

Each component section defines:

1. **Purpose** — what decision this step supports  
2. **Inputs** — artifact or DataFrame schema (grain, required columns, PIT rules)  
3. **Outputs** — schema and grain  
4. **Invariants** — must always hold when implementation is correct  
5. **Status** — `built` | `partial` | `spec-only` | `drift` (code differs from contract)  
6. **Contract test** — `tests/contract/test_*.py`  
7. **Precompute / design notes** — gaps if inputs are insufficient  

Update **Status** as implementation and tests converge.

---

## Run-level success (structural vs decision-quality)

### Structural success (Sprint 002 target)

A backtest run is **structurally valid** when:

- [ ] Run envelope is complete (config + traceable inputs + date list)  
- [ ] Every rebalance date uses PIT universe and same-day features  
- [ ] Every candidate row is traceable through S1→S8 with explicit `exclusion_reason` when not traded  
- [ ] Traded rows include risk-unit fields (abstract budget, not necessarily USD yet)  
- [ ] Metrics use agreed return denominator (max-loss budget units, not body-credit only)  

### Decision-quality success (later sprints)

Tier B go/no-go per [backtest_evaluation_protocol.md](backtest_evaluation_protocol.md) — **out of scope** for Sprint 002.

---

## Capital model (abstract — Sprint 002)

| Concept | Definition |
|---------|------------|
| `max_loss_budget_per_trade` | Abstract risk unit allocated per position (config field); not pinned to USD |
| `max_loss_per_share` | Strategy-unit bounded loss from structure geometry + credit |
| `contracts` / `quantity` | Integer count such that `quantity × max_loss_per_share ≈ budget` (when sizing built) |
| `return_on_max_loss` | Realized PnL in budget units / budget |

Pin deployable capital ($) when S5 is implemented and validated.

---

## Stage A — inputs (given for Sprint 002)

> Precompute is **not** redesigned this sprint. If contracts cannot be met, log in § Precompute gap log.

### A1 — `option_surface_meta` (grain: `ticker`, `entry_date`)

| Column | Required | Notes |
|--------|----------|-------|
| _TBD_ | | Populate from `OptionSurfaceDB` + precompute script |

**Status:** `partial`  
**Contract test:** `tests/contract/test_precompute_input_contract.py`

### A2 — `option_surface_quotes` (grain: `ticker`, `entry_date`, `strike`, `side`)

| Column | Required | Notes |
|--------|----------|-------|
| _TBD_ | | |

**Status:** `partial`

### A3 — `ticker_liquidity_panel` (grain: `ticker`, `month_date`)

_TBD_

### A4 — `features_*` (grain: `ticker`, `date`)

_TBD_

### Precompute gap log

| Gap | Blocks component | Action |
|-----|------------------|--------|
| _none filed yet_ | | |

---

## R0 — Run envelope

**Owner:** `BacktestRunConfig`, `SurfaceDataPaths`, run manifest (TBD)

**Inputs:** _TBD_

**Outputs:** `trade_dates[]`, resolved paths, `run_id`

**Status:** `partial`

**Contract test:** `tests/contract/test_run_envelope_contract.py`

---

## S1 — Universe (`step1_get_universe`)

**Inputs:** _TBD_

**Outputs:** `ticker`, `dvol_rank_pct`, `spread_rank_pct`

**Invariants:** _TBD_ (PIT: `month_date <= trade_date`)

**Status:** `built` (logic exists; contract test pending)

**Contract test:** `tests/contract/test_step1_universe_contract.py`

---

## S2 — Signals (`step2_score_signals`)

**Inputs / outputs / invariants:** _TBD_

**Status:** `built`

**Contract test:** `tests/contract/test_step2_signals_contract.py`

---

## S3 — Structures (`step3_get_eligible_structures`)

**Inputs / outputs / invariants:** _TBD_ (see `pipeline.py` docstring)

**Status:** `built` in pipeline; `drift` — runner still inlines duplicate path

**Contract test:** `tests/contract/test_step3_structures_contract.py`

---

## S4 — Exclusions (`step4_apply_exclusions`)

_TBD_

**Status:** `spec-only` (`pass` in code)

**Contract test:** `tests/contract/test_step4_exclusions_contract.py`

---

## S5 — Select and size (`step5_select_and_size`)

_TBD_ — 50 **total** positions; abstract budget units

**Status:** `spec-only`

**Contract test:** `tests/contract/test_step5_select_and_size_contract.py`

---

## S6 — Cost and return (`step6_apply_cost`)

_TBD_

**Status:** `spec-only`

**Contract test:** `tests/contract/test_step6_apply_cost_contract.py`

---

## S7 — Settlement (hold to expiry)

_TBD_ — `exit_spot` from meta; `pnl_per_share` / dollar PnL linkage in S5/S6

**Status:** `built` (assembly layer); `partial` (runner integration)

**Contract test:** `tests/contract/test_settle_contract.py`

---

## S8 — Run metrics (`build_date_summary`, `summarize_trade_log`)

_TBD_

**Status:** `partial` (body-credit metrics only today)

**Contract test:** `tests/contract/test_run_metrics_contract.py`

---

## ORCH — Orchestration (`SurfaceRunner`)

Thin loop calling S1–S8; no business logic duplication.

**Status:** `drift` (inline S3–S5 in runner)

**Contract test:** `tests/contract/test_orchestration_contract.py` + existing `test_surface_runner_data_flow.py`

---

## Implementation drift register

| Component | Contract says | Code today | Resolution sprint |
|-----------|---------------|------------|-------------------|
| S3–S5 | `pipeline.py` | `surface_runner.py` inline | TBD |
| S5 sizing | contracts + budget units | `pnl_per_share` only | TBD |
| Cap | 50 total | `max_names_per_side` | TBD |

---

## Change log

| Date | Change |
|------|--------|
| 2026-05-28 | Sprint 002 scaffold |
