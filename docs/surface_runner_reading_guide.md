# SurfaceRunner reading guide

**Status:** Active  
**Created:** Sprint 001 closeout (2026-05-27)  
**Audience:** HD — read before closing Sprint 001 or starting Sprint 002 build  
**Estimated time:** 2–3 hours (focused); can split across two sessions

---

## What this guide is for

Sprint 001 produced audits and one synthetic verification test. Those docs explain **gaps and risks**; this guide helps you **own the code path** so you know:

- What runs offline vs what runs on each backtest day
- Which module owns each responsibility today
- Where v1 behavior is **implemented**, **partial**, or **missing**
- What Session B proved vs what still requires Sprint 002 work

**Companion docs (read first if you only have 30 minutes):**

| Doc | Role |
|-----|------|
| [surface_runner_data_flow.md](surface_runner_data_flow.md) | Implemented / partial / missing map |
| [repo_audit.md](repo_audit.md) | P0/P1/P2 backlog |
| [agenda/session_b_plan.md](agenda/session_b_plan.md) | What the verification test was meant to do |

---

## Big picture (keep this diagram in mind)

```text
STAGE A — Offline precompute (ORATS → parquet, run rarely)
  scripts/precompute_option_surface.py
    → src/features/option_surface_analyzer.py (OptionSurfaceBuilder)
    → option_surface_meta_*.parquet   (one row per ticker, entry_date)
    → option_surface_quotes_*.parquet   (strikes/sides/deltas/bid/ask/…)

STAGE B — Backtest day loop (SurfaceRunner, run per config/search)
  scripts/run_surface_search.py  (optional CLI grid search)
    → BacktestRunConfig + SurfaceDataPaths
    → SurfaceRunner.run_single_config()
         load artifacts
         for each trade_date:
           step1 universe  (pipeline.py)
           step2 signals   (pipeline.py)
           build structures from OptionSurfaceDB (surface_runner + option_surface.py)
           select + settle (_select_size_and_settle)
         trade_log + date_summary + run_summary (surface_metrics.py)
```

**Canonical path:** Decision 001 — SurfaceRunner on precomputed surface, not legacy `BacktestEngine`.

---

## Suggested reading plan (tonight)

| Block | Time | Activity |
|-------|------|----------|
| A | 25 min | Skim this guide + [surface_runner_data_flow.md](surface_runner_data_flow.md) executive conclusion |
| B | 20 min | Read `tests/unit/test_surface_runner_data_flow.py` — your mental “one run” |
| C | 45 min | `surface_runner.py` top to bottom with this guide’s section notes |
| D | 30 min | `pipeline.py` step1–2; skim step3–6 stubs |
| E | 35 min | `option_surface.py`: `OptionSurfaceDB`, one builder, `settle` |
| F | 20 min | `run_config.py`, `surface_run_config.py`, `run_surface_search.py` main |
| G | 15 min | Self-check questions (end of this doc) |

Run the test once while reading Block B:

```powershell
cd C:\MomentumCVG
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_surface_runner_data_flow.py -v
```

---

## Block B — Anchor: Session B test

**File:** `tests/unit/test_surface_runner_data_flow.py`

The test builds **temporary parquets** (not your real cache) and calls `SurfaceRunner.run_single_config()` once.

| Fixture | Role in the story |
|---------|-------------------|
| `liquidity.parquet` | PIT universe: `month_date` snapshot, `atm_straddle_dollar_vol`, `atm_spread_pct` |
| `features_42_8.parquet` | One `trade_date`, momentum/CVG columns for 4 tickers |
| `surface_meta.parquet` | Valid rows for LONG1/SHORT1; invalid row for BAD1 |
| `surface_quotes.parquet` | Enough quotes to build long straddle + short iron fly |

**While reading the test, note:**

- `BacktestRunConfig.start_date` must be **strictly before** `end_date` (validation), even if only one feature date exists.
- `long_top_pct=0.25` and `short_bottom_pct=0.5` split four tickers into disjoint long/short pools (BAD1 stays short-only).
- Passing tests under `TestSurfaceRunnerV1Gaps` mean **`contracts` / `pnl_dollars` / `return_on_max_loss` are still absent** — not that sizing works.

---

## Block C — `src/backtest/surface_runner.py` (~439 lines)

This is the **orchestrator**. Read in this order.

### 1. `SurfaceRunner.__init__` (lines 62–84)

**Loads once per runner instance:**

| Attribute | Source |
|-----------|--------|
| `surface_db` | `OptionSurfaceDB.load(meta_path, quotes_path)` |
| `liquidity_panel` | parquet; normalizes `month_date` |
| `earnings` | optional parquet |
| `_features_cache` | lazy per feature file path |

**Watch for:** No schema validation on load — bad columns fail later at runtime.

### 2. `run_single_config` (lines 90–134) — **main loop**

```text
features = _load_features_for_config(config)
trade_dates = _get_trade_dates(features, config)   # every feature date in range

for trade_date in trade_dates:
    universe = _step1_universe(...)
    signals = _step2_signals(...)
    if signals.empty: continue
    structures = _build_structures_for_date(...)
    trade_rows += _select_size_and_settle(...)

trade_log → build_date_summary → summarize_trade_log → SurfaceRunResult
```

**Watch for:**

- **Trade dates come from the features file**, not from surface meta or a weekly calendar artifact.
- If feature dates and surface `entry_date` rows diverge, you get `metadata_error` exclusions — not a hard stop.
- `cost_model` on `BacktestRunConfig` is **not used here**; fills come from `config.fill` (`FillAssumption`).

### 3. `_get_trade_dates` (lines 149–156)

Filters `features["date"]` to `[start_date, end_date]`.

**v1 pin:** Weekly rebalance — **not enforced in code**; density of feature rows drives how often you “rebalance.”

### 4. `_build_structures_for_date` (lines 190–244)

Per signal row:

1. Try `surface_db.get_metadata(ticker, trade_date)` — invalid/missing → `failure_reason`, `structure_ok=False`.
2. Try `_assemble_structure` → `StrategyAssemblyResult` stored temporarily as `_assembly`.
3. `_assembly_to_row` flattens entry economics into the row.
4. `_has_earnings_nearby` sets flag (exclusion happens later).

**Watch for:** `trade_date` must equal surface `entry_date` for that ticker.

### 5. `_assemble_structure` (lines 246–293)

| `direction` | `short_structure` | Builder |
|-------------|-------------------|---------|
| `long` | (ignored) | `build_straddle_from_surface(..., direction="long")` |
| `short` | `ironfly` | `build_ironfly_from_surface` |
| `short` | `ironcondor` | `build_ironcondor_from_surface` |
| `short` | `straddle` | `build_straddle_from_surface(..., direction="short")` |

This is the **strongest layer** (well unit-tested in `tests/unit/test_option_surface_*.py`).

### 6. `_select_size_and_settle` (lines 349–418) — **largest v1 gap**

**Docstring says:** integer contracts from max-loss budget.  
**Code does:**

1. Exclude `structure_ok=False` → `no_tradeable_structure`
2. Exclude earnings → `earnings_exclusion`
3. Per direction: take top `max_names_per_side` by `signal_rank_pct` (long: highest; short: lowest)
4. Cap overflow → `max_names_cap`
5. For included rows only: `assembly.settle(exit_spot)` → **`pnl_per_share` only**

**Does not use:** `max_loss_budget_per_trade`, `min_contracts`, contract multiplier.

**Watch for:**

- `max_names_per_side` caps **per side**, not **50 total** (v1 pin is 50 long+short combined — Sprint 002).
- `_assembly` is popped before the row is appended — internal-only bridge to settlement.
- `include_diagnostics=True` keeps excluded rows in the trade log (needed for attribution).

### 7. `_resolve_max_loss_per_share` (lines 420–438)

Used only to reject `invalid_max_loss` before settle — **not** for sizing. Short straddle gets a **proxy** max loss from `SurfaceRunnerSettings.short_straddle_risk_multiplier`.

---

## Block D — `src/backtest/pipeline.py`

### Implemented (used by SurfaceRunner today)

| Function | Lines (approx) | Purpose |
|----------|----------------|---------|
| `step1_get_universe` | 57–127 | PIT liquidity: latest `month_date <= trade_date`, rank dvol + spread, AND filters |
| `step2_score_signals` | 134–247 | Join features ∩ universe; momentum ranks; CVG filter; tag `long` / `short` |

**step1 detail worth internalizing:**

- Requires **both** top `dvol_top_pct` volume **and** top `spread_bottom_pct` tightness (AND logic).
- Slightly stricter than a plain “top 20% liquidity” story — confirm intent vs [v1_universe_protocol.md](v1_universe_protocol.md).

**step2 detail:**

- Asserts long and short pools do not overlap (needs `long_top_pct + short_bottom_pct <= 1` in practice).
- `count_col` quality filter parses window from `momentum_col` name pattern `mom_{max}_{min}_mean`.

### Stubs (HD decision: implement in Sprint 002 for unit testing)

| Function | Status |
|----------|--------|
| `step3_get_eligible_structures` | Implemented in file but **not called** by SurfaceRunner (runner duplicates logic inline) |
| `step4_apply_exclusions` | `pass` |
| `step5_select_and_size` | `pass` |
| `step6_apply_cost` | `pass` |

**Design intent:** SurfaceRunner orchestrates; pipeline holds **pure, testable** steps. Runner currently inlines steps 3–5.

---

## Block E — `src/backtest/option_surface.py`

Read selectively — file is large (~800 lines).

### Core types (lines 26–167)

| Type | Role |
|------|------|
| `FillAssumption` | `mid()` vs `cross()` — drives leg prices in builders |
| `StrategyAssemblyResult` | Entry economics + `settle(exit_spot)` → `Position` with `pnl` per share |
| `OptionSurfaceDB` | Load meta/quotes; `get_metadata` / `get_quotes` |

**Sign convention:** Negative `entry_cost` = credit received (short structures).

### Builders (pick one deep read)

| Function | Read if… |
|----------|----------|
| `build_straddle_from_surface` | You care about long side / debit strategies |
| `build_ironfly_from_surface` | Primary v1 short structure |
| `build_ironcondor_from_surface` | Condor search grids |

Each builder: select legs from quote rows by delta rules → build `OptionStrategy` → entry costs → `max_loss_per_share` → diagnostics including `body_credit_per_share`.

### Settlement

`StrategyAssemblyResult.settle(exit_spot)` uses `OptionStrategy.calculate_payoff` at expiry.  
Runner passes `exit_spot` from **surface metadata** (hold-to-expiry baseline).

**Session B proved:** Runner’s `pnl_per_share` matches direct `assembly.settle()` for the synthetic fixture.

---

## Block F — Config and CLI

### `src/backtest/run_config.py` — `BacktestRunConfig`

One dataclass = **one complete run specification**. Groups:

- Signal: `momentum_col`, `cvg_col`, `count_col`, top/bottom %, CVG filter
- Universe: `dvol_top_pct`, `spread_bottom_pct`
- Structure: `short_structure`, wing/condor deltas, `fill`, spread filters
- Portfolio: `max_names_per_side`, `max_loss_budget_per_trade` (**field exists; runner sizing not wired**)
- Dates: `start_date`, `end_date`

`__post_init__` validates literals and ranges.

### `src/backtest/surface_run_config.py`

| Type | Role |
|------|------|
| `SurfaceDataPaths` | Resolves cache paths; `features_path_for_config()` infers `features_{max}_{min}.parquet` from column names |
| `SurfaceRunnerSettings` | `short_straddle_risk_multiplier`, `min_contracts` (mostly unused for sizing today) |
| `SurfaceSearchSpec` | Bundles config grid + walk-forward protocol |

### `scripts/run_surface_search.py`

- `build_configs_from_args()` — Cartesian grid of momentum × fill × structure × wing deltas
- `main()` — constructs `SurfaceRunner`, runs `SurfaceSearch`, writes parquets under `surface_search_results/`

**Known issue (documented in repo audit):** line ~243 passes `contract_multiplier` into `SurfaceDataPaths`, but that dataclass has **no** such field — will error if you run the CLI without a fix.

### `src/backtest/surface_metrics.py`

- `build_date_summary` — aggregates per `trade_date`; uses **return on body credit**, not max-loss budget
- `summarize_trade_log` — hit rate, Sharpe on body-credit returns, `robust_score = sharpe × availability`

**Not v1 go/no-go metrics** — documented P0 for Sprint 002.

---

## Block G — Stage A (optional tonight, 20 min skim)

If you want the full two-stage picture:

| File | Skim for |
|------|----------|
| `scripts/precompute_option_surface.py` | What parquets are written; ticker list source (`liquid_tickers.csv`) |
| `src/features/option_surface_analyzer.py` | `OptionSurfaceBuilder.process_single_entry` — how `exit_spot`, `surface_valid`, quotes are produced |

**Runner does not call precompute** — it only consumes the parquets.

---

## Config ↔ code quick reference

| v1 pin (spec) | Config field | Enforced in runner today? |
|---------------|--------------|---------------------------|
| Momentum + CVG signal | `momentum_col`, `cvg_col`, filters | Yes (step2) |
| PIT top-20% liquidity | `dvol_top_pct`, `spread_bottom_pct` | Yes (step1; AND rule) |
| Long straddle / short fly or condor | `short_structure`, direction | Yes (`_assemble_structure`) |
| Conservative fills | `fill=cross` | Yes (in builders) |
| Weekly rebalance | (implicit) | **Partial** — feature dates only |
| 50 max positions total | — | **No** — `max_names_per_side` per side only |
| Equal max-loss sizing | `max_loss_budget_per_trade` | **No** — no `contracts` / dollar PnL |
| Hold to expiry | `exit_spot` in meta | Yes (`settle`) |
| Go/no-go metrics | — | **No** — body-credit Sharpe / `robust_score` |

---

## Self-check before closing Sprint 001

Answer without peeking, then verify in code:

1. **What three parquet families does `SurfaceRunner.__init__` load?**  
   → Surface meta, surface quotes, liquidity panel (+ optional earnings, features per config).

2. **What determines which dates the backtest loops over?**  
   → Unique `date` values in the features file within `[start_date, end_date]`.

3. **Where does a ticker get `direction='long'` vs `'short'`?**  
   → `step2_score_signals` from momentum rank percentiles.

4. **What happens if `surface_valid=False` for a ticker on a trade date?**  
   → `get_metadata` raises; row gets `metadata_error` / `no_tradeable_structure`.

5. **Where is `pnl_per_share` computed?**  
   → `_select_size_and_settle` calls `assembly.settle(exit_spot)` — not in metrics module.

6. **Name three fields the trade log does not yet emit that v1 needs.**  
   → e.g. `contracts`, `pnl_dollars`, realized `return_on_max_loss`.

7. **What does Session B prove vs not prove?**  
   → Proves synthetic end-to-end wiring + settlement consistency; does not prove production cache, sizing, or Tier B decision quality.

8. **Where should Sprint 002 implement 50-cap and integer sizing?**  
   → Prefer `pipeline.py` step5 (HD decision) with SurfaceRunner calling it; today logic lives in `_select_size_and_settle`.

---

## After the read-through

| Action | When |
|--------|------|
| Close Sprint 001 memo + move `week0_review_notes.md` | When self-check feels solid |
| Start Sprint 002 build | Tomorrow — pick P0 from [repo_audit.md](repo_audit.md) |
| Optional code review sprint | Re-read `test_option_surface_ironfly.py` settlement section — same math Session B relied on |

---

## File index (reading order)

| # | Path |
|---|------|
| 1 | `tests/unit/test_surface_runner_data_flow.py` |
| 2 | `src/backtest/surface_runner.py` |
| 3 | `src/backtest/pipeline.py` |
| 4 | `src/backtest/option_surface.py` |
| 5 | `src/backtest/run_config.py` |
| 6 | `src/backtest/surface_run_config.py` |
| 7 | `src/backtest/surface_metrics.py` |
| 8 | `scripts/run_surface_search.py` |
| 9 | `src/backtest/surface_search.py` (if doing CLI path — search orchestration) |
| 10 | `scripts/precompute_option_surface.py` + `src/features/option_surface_analyzer.py` (Stage A) |

---

## Change log

| Date | Change |
|------|--------|
| 2026-05-27 | Initial guide for Sprint 001 closeout reading |
