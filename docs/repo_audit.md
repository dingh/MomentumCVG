# Repo audit — Sprint 001

**Status:** Reviewed by HD  
**Date:** 2026-05-24  
**Mode:** Audit  
**Scope:** Surface-first v1 backtest path, especially precompute surface store + `SurfaceRunner` scaffold.

---

## Executive summary

`SurfaceRunner` remains the right direction for v1, but it should be treated as a **scaffold**, not a finished engine. Significant engine functionality is still undeveloped, and the current audit does not yet prove that every data-flow contract needed for a comprehensive backtest is complete. The strongest completed layer is the **surface assembly layer** in `src/backtest/option_surface.py`: it can load precomputed surface rows, assemble long straddles, short iron flies, and short iron condors, apply mid/cross fills, compute max loss, and settle payoff at expiry.

The biggest gap is not the low-level option math. It is the **engine contract around it**:

1. The precompute store must be validated as rich enough for 2020+ weekly strategy research.
2. `SurfaceRunner` needs a real portfolio/risk/dollar-PnL layer.
3. The runner/search CLI currently has at least one wiring issue and lacks smoke tests.
4. Output metrics are useful for early screening but not yet decision-quality for capital allocation.

**Recommendation:** before Session B writes a verification test, add a short Session A extension that maps the SurfaceRunner functionality and data flow end-to-end. The verification test should then target the highest-risk boundary identified by that map. Sprint 002 should be a Build sprint focused on making one small surface smoke run reliable before adding more research matrix complexity.

---

## Current target flow

```text
Stage A — Precompute surface store

ORATS adjusted parquet
  → scripts/precompute_option_surface.py
  → src/features/option_surface_analyzer.py::OptionSurfaceBuilder
  → option_surface_meta_<frequency>_<start>_<end>.parquet
  → option_surface_quotes_<frequency>_<start>_<end>.parquet

Stage B — Surface backtest engine

scripts/run_surface_search.py
  → BacktestRunConfig grid
  → SurfaceRunner.run_single_config()
  → liquidity panel + features
  → pipeline.step1_get_universe()
  → pipeline.step2_score_signals()
  → pipeline.step3_get_eligible_structures()      # intended modular home for surface assembly routing
  → pipeline.step4_apply_exclusions()             # earnings + other exclusions
  → pipeline.step5_select_and_size()              # per-side cap (max_names_per_side) + sizing
  → pipeline.step6_apply_cost()                   # cost model and return metric normalization
  → option_surface.OptionSurfaceDB
  → build_straddle_from_surface() / build_ironfly_from_surface() / build_ironcondor_from_surface()
  → SurfaceRunner orchestration (thin wrapper over pipeline steps)
  → trade_log + date_summary + run_summary
```

Design intent (HD): keep `SurfaceRunner` as the orchestration layer, but modularize all core behavior into `pipeline.py` step functions so the engine is easier to unit test and extend.

---

## Stage A current state — precompute surface store

### What exists

| Area | Current implementation |
|------|------------------------|
| CLI | `scripts/precompute_option_surface.py` |
| Builder | `src/features/option_surface_analyzer.py::OptionSurfaceBuilder` |
| Trade dates | Friday-based; `weekly` = all resolved trading Fridays, `monthly` = first Friday of month |
| Ticker source | `C:/MomentumCVG_env/cache/liquid_tickers.csv` |
| Expiry selection | `OptionSurfaceBuilder._find_best_expiry()` using 7-DTE weekly or 30-DTE monthly logic |
| Metadata output | One row per `(ticker, entry_date)` with `expiry_date`, `entry_spot`, `exit_spot`, `body_strike`, validity, failure reason |
| Quote output | Body + OTM quote rows with bid/ask/mid, greeks, spread, moneyness, delta bucket |
| Diagnostics | Logs valid/failure counts and failure breakdown after build |

### What it supports well

- Flexible downstream assembly: the stored quote surface supports long straddle, short iron fly, and short iron condor without re-reading ORATS.
- Wing and fill research: delta buckets plus full bid/ask/mid allow wing-target and fill-assumption sweeps.
- Failure traceability: metadata rows preserve failures (`no_spot_price`, `no_expiry_found`, `no_options_at_entry`, `no_spot_at_expiry`, etc.) instead of silently dropping all failed ticker/date pairs.
- Coverage audit mode: `--keep-zero-bid-quotes` can retain otherwise untradeable quotes to diagnose why structures are missing.

### Main Stage A gap

The store is probably rich enough for **hold-to-expiry, entry-date-only** research, but this is not yet proven for the v1 scope. It needs explicit coverage diagnostics before trusting full-sample strategy results.

---

## Stage B current state — SurfaceRunner scaffold

### What exists

| Area | Current implementation |
|------|------------------------|
| Entry point | `scripts/run_surface_search.py` |
| Data paths | `src/backtest/surface_run_config.py::SurfaceDataPaths` |
| Single-run config | `src/backtest/run_config.py::BacktestRunConfig` |
| Search | `src/backtest/surface_search.py::SurfaceSearch` |
| Runner loop | `src/backtest/surface_runner.py::SurfaceRunner.run_single_config()` |
| Assembly | `src/backtest/option_surface.py` |
| Metrics | `src/backtest/surface_metrics.py` |

### What is already a good scaffold

- The runner has a clean sequence: load features → trade dates → universe → signals → surface assembly → selection/settlement → summaries.
- The option assembly boundary is clean: `SurfaceRunner` delegates structure math to `option_surface.py`.
- Config search already supports `ironfly`, `ironcondor`, `mid`, `cross`, wing deltas, and walk-forward/full-sample protocols.
- Diagnostics are included in trade logs when `include_diagnostics=True`, which is useful for understanding skipped structures and selection effects.

### Known implementation concern found during audit

`scripts/run_surface_search.py` constructs:

```python
SurfaceDataPaths(cache_dir=cache_dir, contract_multiplier=args.contract_multiplier)
```

but `SurfaceDataPaths` currently has no `contract_multiplier` field. That means the CLI path appears likely to fail before any run starts. This should be confirmed with a smoke command in Sprint 002 and then fixed before full backtests.

---

## Design completeness and data-flow mapping gap

Human review feedback on 2026-05-25 added an important gate: before relying on either the current audit or the proposed Session B test, the team should explicitly map SurfaceRunner functionality and data flow to confirm that the current design is complete enough to support the intended backtest.

This mapping should answer:

| Area | Mapping question |
|------|------------------|
| Inputs | Which artifacts does `SurfaceRunner` require: liquidity panel, features, surface meta, surface quotes, config, and dates? |
| Stage A contract | Which meta and quote columns are required by each strategy builder, and which missing/failure states are valid? |
| Stage B flow | How does data move from universe selection to signal ranking, candidate construction, selection, settlement, trade log, and metrics? |
| Ownership | Which responsibilities belong in `SurfaceRunner`, `option_surface.py`, `pipeline.py`, config objects, metrics, or future portfolio/risk modules? |
| Missing functionality | Which required backtest functions are absent today vs partially implemented vs already covered? |
| Test boundary | Given that map, which single Session B test provides the most confidence without prematurely locking in the wrong boundary? |

This was completed as a **Session A.1 audit extension** in `docs/surface_runner_data_flow.md`. It remains draft for HD review and should be approved before Session B starts.

---

## Missing for effective backtest

### P0 — blocks decision-quality backtesting

| Gap | Why it matters | Files | Effort |
|-----|----------------|-------|--------|
| ~~SurfaceRunner functionality/data-flow map~~ | **Done Sprint 001** — see `docs/surface_runner_data_flow.md` + `tests/unit/test_surface_runner_data_flow.py`. | — | — |
| Smoke-run CLI wiring | If `run_surface_search.py` cannot instantiate `SurfaceDataPaths`, no surface backtest can run from CLI. | `scripts/run_surface_search.py`, `surface_run_config.py` | S |
| Portfolio/risk/dollar-PnL layer (real engine) | This is the core blocker for realistic backtesting. Without a coherent portfolio/risk layer (integer contracts, dollar PnL, returns normalized on max-loss budget, and total-cap enforcement), results cannot be interpreted as deployable capital performance. | `pipeline.py` (steps 4–6), `surface_runner.py`, `run_config.py`, `surface_metrics.py` | L |
| Per-side cap config discipline | v1 uses `max_names_per_side` per direction ([decision 003](decisions/003_position_cap_per_side.md)); e.g. 25 per side for ~50-book — avoid setting 50 per side (100 total). | `run_config.py`, search scripts | S |
| Trade-date schedule contract | `_get_trade_dates()` uses every feature date in range. Need explicit weekly schedule aligned to precomputed surface dates and v1 rebalance. | `surface_runner.py`, `precompute_option_surface.py`, feature generation | M |
| Surface coverage report | Before full 2020+ backtest, need a report: valid surfaces by year/date/ticker, missing failure reasons, quote coverage by delta bucket, fly/condor assembly availability. | `precompute_option_surface.py`, new audit script or notebook, maybe `OptionSurfaceDB` | M |
| Return metric alignment | `surface_metrics.py` currently works on return on body credit, not return on max-loss budget / dollar capital. Sharpe on body-credit returns is not the v1 capital metric. | `surface_metrics.py`, `surface_runner.py` | M |

### P1 — needed before serious config search / Tier B

| Gap | Why it matters | Files | Effort |
|-----|----------------|-------|--------|
| Precompute manifest | Need reproducibility: CLI args, data root, ticker file, date count, valid rate, output paths, schema version. Logs are not enough. | `precompute_option_surface.py` | S/M |
| Candidate availability diagnostics | Need to distinguish “signal picked bad name” vs “surface missing quote” vs “spread filter rejected” vs “no wing.” | `surface_runner.py`, `option_surface.py`, metrics | M |
| Config/result persistence | Ranked summaries and trade logs are written, but full config dumps / run metadata should be stored for every run. | `run_surface_search.py`, `surface_search.py` | S |
| Separate engine stages | `pipeline.py` has pure step functions, but steps 4–6 are `pass`; `SurfaceRunner` contains a partial inline implementation. Decide whether to finish pipeline steps or keep runner-local implementation. | `pipeline.py`, `surface_runner.py` | M/L |
| Config/doc drift | `BacktestRunConfig` comments still reference straddle/ironfly history files even though the surface path assembles from `OptionSurfaceDB`. This can confuse future implementation. | `run_config.py`, docs | S |
| Long/short portfolio policy | Current config trades both long straddle and short structure simultaneously. Need explicit controls for enabling/disabling sides and allocating risk by side. | `run_config.py`, `surface_runner.py` | M |
| Tier B output fields | Need fields for Sharpe, return on max-loss, drawdown, concentration, turnover, ops count, not just body-credit returns. | `surface_runner.py`, `surface_metrics.py` | M |

### P2 — important before shadow / paper

| Gap | Why it matters | Files | Effort |
|-----|----------------|-------|--------|
| Intended order sheet | Shadow/paper needs strike, side, quantity, limit/mid/cross prices, reason codes. | `surface_runner.py`, new runbook/output module | M |
| Early-exit model | Live likely closes before expiry; surface store only supports entry + expiry spot today. Need future artifact or mark-to-market surface for exits. | `option_surface_analyzer.py`, `precompute_option_surface.py`, `option_surface.py` | L |
| Sector / cluster caps | Needed before scaling capital. | New portfolio/risk module, config | M |
| Data snapshot governance | Need versioned cache artifacts before relying on paper/live decisions. | scripts + docs | M |

---

## Decision 001 checks

| Check | Status | Notes |
|-------|--------|-------|
| Surface path can enforce PIT universe | **Partial** | `step1_get_universe()` is point-in-time by `month_date <= trade_date`, and runner calls it. Need integration test and weekly universe protocol confirmation. |
| Momentum/CVG can run without lookahead | **Partial** | `step2_score_signals()` slices features by exact date. Need feature-generation audit to prove those features are PIT. |
| Iron fly assembly is correct vs unit tests | **Met for synthetic surfaces** | Strong tests exist for iron fly entry, max loss, fills, and expiry settlement. |
| Iron condor assembly works for comparison | **Met for synthetic surfaces** | Strong tests exist for condor entry, wing selection, max loss, fills, and settlement. |
| 2020+ cache coverage reproducible | **Unknown** | Need precompute manifest/coverage report; not proven by code inspection. |
| Equal max-loss + 50-name cap can be implemented cleanly | **Partial** | Config has `max_loss_budget_per_trade`; current runner does not appear to convert it into contracts/dollar PnL. |

---

## What not to refactor yet

- Do not rewrite `BacktestEngineV2` now. It is a skeleton and not required if the surface path is made complete.
- Do not merge legacy straddle/ironfly history paths into SurfaceRunner yet. Keep the surface path clean.
- Do not add broker APIs or live execution.
- Do not implement naked long call until short-side structure and runner mechanics are stable.
- Do not run a large fly-vs-condor matrix before the P0 smoke path, sizing, and metric gaps are closed.

---

## Recommended Sprint 002–004 backlog

### Sprint 002 — make one surface smoke run real

| Item | Effort |
|------|--------|
| Use the approved Session A.1 data-flow map to drive the first build/test boundary | S |
| Fix/verify `run_surface_search.py` CLI wiring (`contract_multiplier` issue) | S |
| Add tiny smoke test or smoke script for one config / short date range | M |
| Implement the portfolio/risk/dollar-PnL layer (pipeline steps 4–6): `contracts`, `max_loss_dollars`, `pnl_dollars`, `return_on_max_loss`, and total-cap enforcement | L |
| Document per-side cap example in search defaults (`max_names_per_side=25` for ~50-book) | S |
| Run surface coverage report for the intended weekly precompute | M |

### Sprint 003 — engine metrics and portfolio layer

| Item | Effort |
|------|--------|
| Upgrade `surface_metrics.py` to use return on max-loss budget and Sharpe | M |
| Add turnover / concentration / availability diagnostics | M |
| Add config/result manifest written with every run | S |
| Add tests for runner selection/sizing on synthetic DataFrames | M |

### Sprint 004 — controlled research baseline

| Item | Effort |
|------|--------|
| Run short-window backtest smoke from cache | S/M |
| Compare iron fly vs iron condor on small fixed window | M |
| Prepare Tier B 2020+ run only after smoke + coverage + sizing pass | M/L |

---

## Open questions for review

1. Which implemented area should be reviewed first in the next sprint: surface DB loading, structure assembly tests, or settlement?
2. Should v1 research initially run both long and short sides, or isolate the short side until the runner mechanics are trusted?
3. How should `pipeline.py` functions be completed so `SurfaceRunner` can remain orchestration while core behavior is unit-testable?
4. Is hold-to-expiry settlement enough for the first Tier B baseline, or should precompute begin storing exit-surface data before any full evaluation?

