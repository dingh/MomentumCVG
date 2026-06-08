# Surface engine — data contract (source of truth)

**Status:** Draft — Sprint 002  
**Last updated:** 2026-06-07  
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

Producer code: `src/features/option_surface_analyzer.py` (`_metadata_success_row`,
`_metadata_failure_row`, `_build_quote_rows`) → written by `scripts/precompute_option_surface.py`.
The columns below are the **consumer-required** subset the runner/pipeline depends on;
the producer emits a superset (diagnostic columns kept for audits).

### A1 — `option_surface_meta` (grain: `ticker`, `entry_date`)

| Column | Required | Notes |
|--------|----------|-------|
| `ticker` | yes | Underlying symbol |
| `entry_date` | yes | Trade/entry date (PIT decision date) |
| `expiry_date` | yes | Resolved option expiry (`None` on failure rows) |
| `dte_actual` | yes | `expiry_date − entry_date` in days |
| `entry_spot` | yes | Spot at entry; settlement/moneyness reference |
| `exit_spot` | yes | Spot at expiry; drives S7 hold-to-expiry settlement |
| `body_strike` | yes | ATM body strike used by all structures |
| `surface_valid` | yes | **Primary downstream filter**: `True` only if both body legs present + ≥1 quote |
| `failure_reason` | yes | snake_case tag (`no_spot_price`, `no_expiry_found`, …) or `None` |
| `frequency`, `dte_target`, `spot_move_pct`, `realized_volatility`, `has_body_call`, `has_body_put`, `n_surface_quotes`, `processing_time` | diagnostic | Coexist on success + failure rows (identical key set) |

**Invariants:**
- Failure and success rows share an identical column set (so they coexist in one table).
- `surface_valid == (has_body_call AND has_body_put AND n_surface_quotes > 0)`.
- Rows with `surface_valid == False` may carry partial data and must be excluded before assembly.

**Status:** `built` (producer schema pinned by contract test)  
**Contract test:** `tests/contract/test_precompute_input_contract.py`

### A2 — `option_surface_quotes` (grain: `ticker`, `entry_date`, `strike`, `side`)

| Column | Required | Notes |
|--------|----------|-------|
| `ticker`, `entry_date`, `expiry_date`, `entry_spot`, `body_strike` | yes | Join keys / context (match the meta row) |
| `side` | yes | `'call'` / `'put'` |
| `is_body`, `is_otm` | yes | Body vs OTM-wing classification (`is_otm == not is_body`) |
| `strike` | yes | Strike price |
| `bid`, `ask`, `mid` | yes | Quote prices; `> 0` unless built with `keep_zero_bid_quotes` |
| `spread_pct` | yes | `(ask − bid) / mid` |
| `iv`, `delta`, `abs_delta`, `gamma`, `vega`, `theta` | yes | Greeks; wing selection uses `abs_delta` |
| `volume`, `open_interest` | yes | Liquidity |
| `strike_distance_from_body`, `abs_strike_distance_from_body`, `moneyness`, `spread`, `nearest_delta_bucket`, `delta_bucket_distance` | diagnostic | Derived convenience fields |

**Invariants:**
- ITM options are excluded by the producer (only body + OTM wings retained).
- OTM wings satisfy `min_abs_delta <= abs_delta <= max_abs_delta`; body quotes are exempt.

**Status:** `built` (consumer subset pinned by contract test)

### A3 — `ticker_liquidity_panel` (grain: `ticker`, `month_date`)

Consumed by **S1** as the point-in-time liquidity snapshot.

| Column | Required | Notes |
|--------|----------|-------|
| `month_date` | yes | Snapshot month; S1 picks `max(month_date <= trade_date)` |
| `ticker` | yes | Symbol |
| `atm_straddle_dollar_vol` | yes | Dollar-volume metric; ranked for `dvol_top_pct` |
| `atm_spread_pct` | yes | Effective spread; ranked for `spread_bottom_pct` |
| `has_valid_atm_pair` | yes | Rows `False` are dropped before ranking |

**Status:** `built` (consumer-required columns pinned by contract test)

### A4 — `features_*` (grain: `ticker`, `date`)

Consumed by **S2** for cross-sectional ranking.

| Column | Required | Notes |
|--------|----------|-------|
| `date` | yes | Must equal `trade_date` for a row to be scored |
| `ticker` | yes | Symbol |
| `<momentum_col>` (e.g. `mom_42_8_mean`) | yes | Primary ranking signal (config-named) |
| `<cvg_col>` (e.g. `cvg_42_8`) | yes | CVG conditioning filter (config-named) |
| `<count_col>` (e.g. `mom_42_8_count`) | yes | Data-quality guard; window derived from `mom_{max}_{min}_mean` |

**Status:** `built` (consumer-required columns pinned by contract test)

### Precompute gap log

| Gap | Blocks component | Action |
|-----|------------------|--------|
| `earnings_date` not present in any Stage A artifact | S4 earnings exclusion | File when S4 is implemented |
| Liquidity panel uses monthly snapshot grain | S1 universe quality | **HD decision:** replace with 3-month rolling-window average of `atm_straddle_dollar_vol` and `atm_spread_pct` (same column names; change panel build script). Fix before end-to-end correctness check — not blocking Sprint 002 contracts. |

---

## R0 — Run envelope

**Owner:** `BacktestRunConfig` (`src/backtest/run_config.py`); trade-date generation in
`scripts/precompute_option_surface.py::generate_trade_dates`.

**Inputs:** user intent — signal cols, fractions, structure choice, sizing budget, date range, fill/cost models.

**Outputs:** a validated, in-memory config object exposing `run_id`, signal/universe/structure/portfolio fields,
`start_date`/`end_date`. `trade_dates[]` are resolved separately (PIT-eligible Fridays with data files).

**Invariants (validated in `__post_init__`):**
- `start_date < end_date` (strict).
- `long_top_pct`, `short_bottom_pct` ∈ (0, 1) and `long_top_pct + short_bottom_pct ≤ 1.0` (disjoint long/short pools).
- `cvg_filter_pct`, `min_count_pct`, `dvol_top_pct`, `spread_bottom_pct` ∈ (0, 1].
- `short_structure` ∈ {ironfly, straddle, ironcondor}; `cost_model` ∈ {mid, half_spread_per_leg, full_spread_per_leg};
  `wing_selection_rule` ∈ {closest_delta, max_credit_to_width, widest}.
- `max_names_per_side ≥ 1`, `max_loss_budget_per_trade > 0`, `earnings_exclusion_days ≥ 0`.
- ironfly+closest_delta ⇒ `wing_delta_target` ∈ (0, 0.5); ironcondor ⇒ `condor_long_delta_target < condor_short_delta_target`.

**Status:** `built`

**Contract test:** `tests/contract/test_run_envelope_contract.py`

---

## S1 — Universe (`step1_get_universe`)

**Inputs:** `trade_date`, `liquidity_panel` (A3), `config`.

**Outputs:** `[ticker, dvol_rank_pct, spread_rank_pct]`, one row per eligible ticker (empty frame with these columns when none qualify).

**Invariants:**
- **I1 (PIT):** snapshot used = `max(month_date <= trade_date)`. No future panel leaks.
- **I2:** rows with `has_valid_atm_pair == False` or NaN `atm_straddle_dollar_vol` / `atm_spread_pct` are dropped *before* ranking.
- **I3:** ranks computed on the full surviving snapshot; both filters applied with **AND** logic
  (`dvol_rank_pct >= 1 − dvol_top_pct` AND `spread_rank_pct >= 1 − spread_bottom_pct`).
- **I4:** output schema is exactly `[ticker, dvol_rank_pct, spread_rank_pct]`.
- **I5:** `trade_date` earlier than every snapshot ⇒ empty frame (correct columns).
- rank-pct values ∈ [0, 1].

**Status:** `built` (L1 contract test green)

**Contract test:** `tests/contract/test_step1_universe_contract.py`

---

## S2 — Signals (`step2_score_signals`)

**Inputs:** `trade_date`, `features` (A4), `universe` (S1 output; only `ticker` used), `config`.

**Outputs:** `[ticker, direction, signal_score, signal_rank_pct, cvg_score, cvg_rank_pct]`,
one row per ticker passing momentum + CVG filters. `direction` ∈ {`long`, `short`}.

**Invariants:**
- **I1:** only rows with `date == trade_date` AND `ticker ∈ universe` are scored.
- **I2:** rows with NaN `momentum_col` or `cvg_col` are dropped; no NaN in `signal_score` / `cvg_score` output.
- **I3:** data-quality guard — `count_col >= min_count_pct × window_size`, where `window_size = max_lag − min_lag + 1` parsed from `mom_{max}_{min}_mean`.
- **I4:** long pool = top `long_top_pct` by `signal_rank_pct` (`>= 1 − long_top_pct`); short pool = bottom `short_bottom_pct` (`<= short_bottom_pct`).
- **I5 (disjoint):** long and short pools share no ticker (R0 rejects `long_top_pct + short_bottom_pct > 1`; S2 also asserts no overlap).
- **I6:** output schema is exactly the six columns above.

**Status:** `built` (contract test green)

**Contract test:** `tests/contract/test_step2_signals_contract.py`

---

## S3 — Structures (`step3_get_eligible_structures`)

**Inputs:** `trade_date`, `signals` (full S2 output), `surface_db`, `config`.

**Outputs:** one row per signal with S2 columns preserved plus structure fields:
`trade_date`, `structure_ok`, `failure_reason`, `entry_spot`, `exit_spot`, `body_strike`,
`expiry_date`, `dte_actual`, `instrument_type`, `net_credit_per_share`, `max_loss_per_share`,
`spread_cost_ratio`, `leg_spread_to_credit_ratio`, greeks/diagnostics, `theoretical_return_on_max_loss`,
`_assembly` (when `structure_ok`).

**Invariants:**
- **I1:** one row per input signal row; signal columns preserved.
- **I2:** meta loaded before assembly; `metadata_error:*` on invalid surface rows.
- **I3:** routing: long → long straddle; short → per `config.short_structure`.
- **I4:** `structure_ok=True` ⇒ `_assembly` present; assembly fields overwrite meta spot/strike on success.
- **I5:** earnings **not** set here — step4 adds `had_earnings_nearby`.

**Status:** `built` (L1+L2 contract tests green)

**Contract test:** `tests/contract/test_step3_structures_contract.py`

---

## S4 — Exclusions (`step4_apply_exclusions`)

**Inputs:** `structures` (S3 output), `earnings` (optional parquet; `None` when absent), `config`.

**Outputs:** S3 rows + `had_earnings_nearby` (bool). No rows dropped.

**Invariants:**
- **I1:** window = `[expiry_date − earnings_exclusion_days, expiry_date]` per ticker.
- **I2:** `earnings is None`, empty earnings, or `earnings_exclusion_days <= 0` ⇒ all `False`.
- **I3:** requires `ticker`, `expiry_date` on structure rows.

**Status:** `built` (L1 contract test green)

**Contract test:** `tests/contract/test_step4_exclusions_contract.py`

---

## S5 — Select, size, and simulate (`step5_select_and_size`)

> **Deferred — Sprint 002 Session C.** Outcomes and open questions in [surface_engine_portfolio_metrics_design.md](surface_engine_portfolio_metrics_design.md). Contracts + tests in **Sprint 003 build** (not spec-only xfails).

**Target role:** Turn post-S4 **candidates** into **simulated trades**: (1) **select** — per-side cap (`max_names_per_side`, [decision 003](decisions/003_position_cap_per_side.md)); (2) **size** — constraint-driven policy via `sizing_mode`: **both** Tier A (equal fractional weight / conceptual return) and Tier B (integer lots × 100 + capital limits) in Sprint 003; (3) **simulate** — S7 settle + PnL at chosen size; (4) **return** — `return_on_allocated_budget` (+ structure-native diagnostics); `fill_label` (former S6 scope, collapsed here). See [surface_engine_portfolio_metrics_design.md](surface_engine_portfolio_metrics_design.md) § Return normalization. Entry fill/cost is fixed at S3 (`config.fill`); no separate cost pass. Config supplies **constraints**, not literal quantities. See [surface_engine_portfolio_metrics_design.md](surface_engine_portfolio_metrics_design.md) § S5.

**Code today:** `SurfaceRunner._select_size_and_settle` (partial); `pipeline.step5` is `pass`.

**Status:** `deferred` — design draft

**Contract test:** _planned Sprint 003_ — `tests/contract/test_step5_select_and_size_contract.py`

---

## S6 — Cost and return (`step6_apply_cost`)

> **Collapsed into S5 (Sprint 002 Session C).** No separate v1 pipeline step or contract test.

**Was:** Post-settle spread penalty and `return_on_max_loss` labeling. **Now:** `return_on_max_loss` and `fill_label` are S5 outputs; conservative entry is `FillAssumption` at S3 assembly. `config.cost_model` is legacy — use `fill` only. `pipeline.step6_apply_cost` remains a deprecated `pass` stub until Sprint 003 cleanup.

**Status:** `superseded` — see S5

---

## S7 — Settlement (hold to expiry)

**Owner:** `StrategyAssemblyResult.settle` (`src/backtest/option_surface.py`); runner calls it from `_select_size_and_settle` with `exit_spot` from structure row (meta).

**Inputs:** `exit_spot` (from A1 meta / structure row), optional `exit_date` (defaults to `expiry_date`).

**Outputs:** `Position` with `pnl` in per-share terms (same units as `entry_cost` / `net_credit` on the assembly).

**Invariants:**
- **I1:** `pnl = exit_value − entry_cost` where `exit_value` is strategy payoff at `exit_spot` on `expiry_date`.
- **I2:** For iron fly at body strike with positive net credit, max profit at expiry ≈ `net_credit` when spot at body.
- **I3:** Long straddle loses premium when spot at body; gains when move exceeds premium (sign verified on synthetic surface).
- Dollar PnL and `return_on_max_loss` are S5 — not S7.

**Status:** `built` (L2 golden on synthetic surface in contract tests)

**Contract test:** `tests/contract/test_settle_contract.py`

---

## S8 — Run metrics (`build_date_summary`, `summarize_trade_log`)

> **Deferred — Sprint 002 Session C.** See [surface_engine_portfolio_metrics_design.md](surface_engine_portfolio_metrics_design.md) § S8.

**Target role:** Run/date summaries on **return_on_max_loss** series for go/no-go (Sharpe, drawdown, availability).

**Code today:** `partial` — body-credit returns only (`surface_metrics.py`).

**Status:** `deferred` — design draft (interim metrics usable for config search only)

**Contract test:** _planned Sprint 003_ — `tests/contract/test_run_metrics_contract.py`

---

## ORCH — Orchestration (`SurfaceRunner`)

> **Deferred — Sprint 002 Session C.** See [surface_engine_portfolio_metrics_design.md](surface_engine_portfolio_metrics_design.md) § ORCH.

**Target:** Thin loop S1→S8 via `pipeline.py`; no duplicate business logic.

**Code today:** S1–S4 delegate; S5+settle inline; S6 superseded (not called); S8 after loop.

**Status:** `deferred` — design draft

**Contract test:** _planned Sprint 003_ — `test_orchestration_contract.py` + `test_surface_runner_data_flow.py`

---

## Implementation drift register

| Component | Contract says | Code today | Resolution sprint |
|-----------|---------------|------------|-------------------|
| S5 (incl. returns) | `pipeline.py` step5 | `surface_runner._select_size_and_settle` | Sprint 003 |
| Trade log economics | `return_on_allocated_budget`, quantity | `pnl_per_share` only | Sprint 003 |
| Cap | Per-side `max_names_per_side` ([decision 003](decisions/003_position_cap_per_side.md)) | Runner aligned; pipeline `step5` still `pass` | Sprint 003 extract |
| S8 denominator | max-loss budget returns | body-credit returns | Sprint 003 |

---

## Change log

| Date | Change |
|------|--------|
| 2026-05-28 | Sprint 002 scaffold |
| 2026-05-31 | Session A: filled Stage A (A1–A4), R0, S1, S2 contracts; 27 contract tests green |
| 2026-05-31 | R0: removed long/short sum ≤ 1 as a documented gap (S2 owns disjointness) |
| 2026-05-31 | HD review: R0 enforces long+short ≤ 1; liquidity panel → 3mo rolling avg (precompute gap); Stage A stays consumer-required subset |
| 2026-05-31 | Pre-B refactor: S3/S4 moved to pipeline; runner delegates (S5 settle still in runner) |
| 2026-05-31 | Session B: S3/S4/S7 contract tests; § S7 filled |
| 2026-05-31 | Session C: S5/S8/ORCH deferred; portfolio/metrics design doc |
| 2026-06-07 | S6 collapsed into S5 — fill at S3; step6 superseded |
| 2026-06-07 | Cap semantics: per-side `max_names_per_side` (decision 003 supersedes 002) |
| 2026-06-07 | S5 return: `return_on_allocated_budget` + diagnostics (portfolio design § Return normalization) |
