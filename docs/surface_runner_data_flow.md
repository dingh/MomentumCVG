# SurfaceRunner data-flow map — Sprint 001 Session A.1

**Status:** Reviewed by HD; Session B test target approved  
**Date:** 2026-05-25  
**Mode:** Audit / mapping only  
**Scope:** SurfaceRunner data flow, function ownership, and missing functionality needed to turn the surface path into a comprehensive v1 backtesting engine.

---

## Executive conclusion

`SurfaceRunner` is still the best current direction for the v1 backtest, but the mapping confirms HD's concern: the design is not yet complete enough to treat as a comprehensive backtesting engine. The surface assembly layer is relatively mature; the engine layer around it is partial.

The biggest finding is that the current runner can assemble and settle one strategy unit per selected row, but it does **not** yet implement the full v1 portfolio contract: weekly schedule alignment, total 50-position cap semantics, integer contracts, dollar PnL, return on max-loss budget, run manifests, and decision-quality metrics.

This changes the Session B recommendation. A private `_select_size_and_settle()` unit test is still useful, but HD approved the highest-value Sprint 001 verification test as a small synthetic `SurfaceRunner.run_single_config()` data-flow test. That test should exercise the whole current path from feature date -> universe -> signal -> surface lookup -> assembly -> selection -> settlement -> trade log, while intentionally exposing missing v1 fields where appropriate.

HD agrees with the implemented / partial / missing classification below. However, even items marked **Implemented** should receive implementation and test-case review in a following sprint before they are treated as trusted for capital decisions.

---

## Minimum complete v1 backtest flow

The intended v1 flow is:

```text
Offline artifacts
  ORATS adjusted parquet
    -> liquidity panel
    -> momentum/CVG features
    -> option surface meta + quotes

Backtest run
  BacktestRunConfig + SurfaceDataPaths
    -> SurfaceRunner.__init__()
    -> load surface DB, liquidity panel, optional earnings, feature file
    -> choose rebalance dates
    -> step1_get_universe()
    -> step2_score_signals()
    -> build structures from OptionSurfaceDB
    -> apply exclusions
    -> select positions under portfolio caps
    -> size integer contracts from max-loss budget
    -> settle hold-to-expiry PnL
    -> emit trade log, date summary, run summary, config/result manifest
```

Current implementation reaches only part of that flow. It loads the artifacts, builds signals, assembles structures, selects up to `max_names_per_side`, settles per-share PnL, and summarizes body-credit returns. It does not yet complete the portfolio/risk/metrics contract.

---

## Artifact map

| Artifact | Produced by | Consumed by | Current role | Completeness |
|----------|-------------|-------------|--------------|--------------|
| `ticker_liquidity_panel.parquet` | `scripts/build_liquidity_panel.py` | `SurfaceRunner._step1_universe()` via `step1_get_universe()` | Point-in-time dynamic universe source. | Partial: logic exists, but no direct runner integration test. |
| `liquid_tickers.csv` | `scripts/build_liquidity_panel.py` | `scripts/precompute_option_surface.py` | Static precompute ticker superset. | Partial: may be a broad enough superset, but coverage vs dynamic universe must be reported. |
| `features_<max>_<min>.parquet` | `scripts/build_features.py` | `SurfaceRunner._load_features_for_config()` | Momentum/CVG signal inputs. | Partial: calculators are tested, but runner assumes feature dates are rebalance dates. |
| `option_surface_meta_weekly_*.parquet` | `scripts/precompute_option_surface.py` | `OptionSurfaceDB.get_metadata()` | One row per `(ticker, entry_date)` with expiry, body strike, entry/exit spot, validity. | Partial: schema is suitable for hold-to-expiry, but not manifest/versioned or coverage-proven. |
| `option_surface_quotes_weekly_*.parquet` | `scripts/precompute_option_surface.py` | `OptionSurfaceDB.get_quotes()` and builders | Body and OTM quote rows for straddle/fly/condor assembly. | Partial: supports current builders; missing coverage report by date/ticker/delta/structure. |
| Optional earnings parquet | External / future script | `SurfaceRunner._has_earnings_nearby()` | Exclude candidates near earnings. | Partial: per-row logic exists; artifact contract is not pinned. |
| Search results parquets | `scripts/run_surface_search.py` | Human review / later tooling | Ranked summaries, trade logs, date summaries. | Partial: no config JSON/manifest; CLI has a wiring issue. |

---

## Stage A contract: precompute surface

### What Stage A currently does

`scripts/precompute_option_surface.py` generates weekly or monthly trade dates, reads a static ticker list from `liquid_tickers.csv`, and uses `OptionSurfaceBuilder.process_single_entry()` to produce:

| Output | Grain | Required downstream columns |
|--------|-------|-----------------------------|
| Surface metadata | `(ticker, entry_date)` | `ticker`, `entry_date`, `frequency`, `dte_target`, `dte_actual`, `expiry_date`, `entry_spot`, `exit_spot`, `body_strike`, `surface_valid`, `failure_reason`, `spot_move_pct`, `realized_volatility` |
| Surface quotes | `(ticker, entry_date, expiry_date, strike, side)` | `ticker`, `entry_date`, `expiry_date`, `entry_spot`, `body_strike`, `side`, `is_body`, `is_otm`, `strike`, `bid`, `ask`, `mid`, `spread_pct`, `iv`, `delta`, `abs_delta`, `gamma`, `vega`, `theta`, `volume`, `open_interest`, `nearest_delta_bucket` |

`OptionSurfaceBuilder` failure rows preserve `failure_reason` and set `surface_valid=False`. This is good for diagnostics, but the runner currently only sees these failures as exceptions from `OptionSurfaceDB.get_metadata()`.

### Stage A strengths

- The quote surface stores primitives, not prebuilt strategy candidates, so the same surface can support straddle, iron fly, and iron condor assembly.
- The metadata has entry spot, expiry, exit spot, body strike, and surface validity, which is enough for the current hold-to-expiry baseline.
- Quote rows carry bid/ask/mid, spread, greeks, delta, and bucket fields, which support fill assumptions and wing/delta research.
- Failure rows are preserved in metadata rather than dropped entirely.

### Stage A gaps

| Gap | Why it matters | Severity |
|-----|----------------|----------|
| No coverage report by year/date/ticker/structure | A Tier B result may mostly measure missing surface coverage rather than strategy edge. | P0 |
| No manifest/schema version | Results cannot be reliably reproduced or compared across surface rebuilds. | P1 |
| Precompute ticker source is static | Dynamic v1 universe may include ticker/date pairs not covered by the static precompute list unless coverage is proven. | P0 |
| Trade-date schedule not shared as an artifact | Runner derives dates from features, while surface precompute derives dates from ORATS trading Fridays. Mismatch can silently create no-surface rows. | P0 |
| Exit data is expiry spot only | This supports hold-to-expiry, but not early close or mark-to-market research. | P2 for v1 backtest, P0 before live-like exits |

---

## Stage B contract: SurfaceRunner

### Current call graph

```text
scripts/run_surface_search.py
  -> build_configs_from_args()
  -> SurfaceDataPaths(...)
  -> SurfaceRunner(...)
  -> SurfaceSearch.run(...)
      -> SurfaceRunner.run_single_config(config)
          -> _load_features_for_config(config)
          -> _get_trade_dates(features, config)
          -> _step1_universe(trade_date, config)
          -> _step2_signals(trade_date, features, universe, config)
          -> _build_structures_for_date(trade_date, signals, config)
              -> OptionSurfaceDB.get_metadata()
              -> _assemble_structure()
                  -> build_straddle_from_surface()
                  -> build_ironfly_from_surface()
                  -> build_ironcondor_from_surface()
              -> _has_earnings_nearby()
          -> _select_size_and_settle(...)
              -> assembly.settle(exit_spot)
          -> build_date_summary()
          -> summarize_trade_log()
```

### Function ownership map

| Responsibility | Current owner | Status | Notes |
|----------------|---------------|--------|-------|
| Parse/run config grid | `scripts/run_surface_search.py` | Partial | Builds configs, but does not persist config manifests and passes unsupported `contract_multiplier` into `SurfaceDataPaths`. |
| Resolve data paths | `SurfaceDataPaths` | Partial | Defaults exist; no contract multiplier; no file/schema validation. |
| Load surface artifacts | `OptionSurfaceDB.load()` in `SurfaceRunner.__init__()` | Implemented | Reads parquet and normalizes dates. No required-column validation. |
| Load liquidity panel | `SurfaceRunner.__init__()` | Implemented | Converts `month_date` when present. No schema validation. |
| Load feature file | `SurfaceRunner._load_features_for_config()` | Implemented | Infers feature file from column names. No schema validation. |
| Choose trade dates | `SurfaceRunner._get_trade_dates()` | Partial | Uses every feature date in config range. Does not enforce weekly rebalance or intersection with surface dates. |
| PIT universe | `pipeline.step1_get_universe()` | Partial | Uses most recent `month_date <= trade_date`; direct tests are missing. Current rule applies top volume and tight-spread filters together, which may be stricter than the v1 doc's simple top-20 description. |
| Momentum/CVG signal ranking | `pipeline.step2_score_signals()` | Partial | Slices exact feature date and ranks within universe. Depends on feature PIT correctness and date alignment. |
| Candidate structure assembly | `SurfaceRunner._build_structures_for_date()` + `option_surface.py` builders | Mostly implemented | Surface builders are well-tested; runner-level candidate availability is untested. |
| Earnings exclusion | `SurfaceRunner._has_earnings_nearby()` | Partial | Logic exists, but earnings artifact schema/default path is not pinned. |
| Portfolio selection cap | `SurfaceRunner._select_size_and_settle()` | Partial | Selects up to `max_names_per_side` per direction. Does not implement v1 50 total cap unless that policy is defined as per-side. |
| Equal max-loss sizing | `SurfaceRunner._select_size_and_settle()` | Missing | Comment says integer contracts, but code does not compute `contracts`, `max_loss_dollars`, or use `max_loss_budget_per_trade`. |
| Hold-to-expiry settlement | `StrategyAssemblyResult.settle()` called by runner | Implemented per strategy unit | Produces `pnl_per_share`; no dollar scaling. |
| Cost/fill model | `FillAssumption` inside builders | Partial | Mid/cross entry fills are modeled. `BacktestRunConfig.cost_model` is legacy and not used by runner. |
| Trade log output | `SurfaceRunner.run_single_config()` | Partial | Emits useful rows, but lacks contracts, dollar PnL, realized return on max-loss, config ID fields, and order-sheet-ready leg quantities. |
| Date summary | `surface_metrics.build_date_summary()` | Partial | Summarizes body-credit returns, not capital/max-loss returns. |
| Run summary / ranking | `surface_metrics.summarize_trade_log()` and `rank_run_summaries()` | Partial | `robust_score` uses body-credit Sharpe * availability, not v1 go/no-go metrics. |
| Result persistence | `scripts/run_surface_search.py` | Partial | Writes parquets; no config JSON, run manifest, data artifact hash/version, or human-readable summary. |

---

## Data-flow stages in detail

### 1. CLI and config construction

Current path:

```text
run_surface_search.py args
  -> build_configs_from_args()
  -> BacktestRunConfig list
  -> SurfaceSearchSpec
```

What works:

- Supports momentum column grids, fill assumptions, short structures, iron-fly wing deltas, and iron-condor short/long delta grids.
- `BacktestRunConfig.__post_init__()` validates many config fields.

Gaps:

- `SurfaceDataPaths(cache_dir=cache_dir, contract_multiplier=args.contract_multiplier)` is currently incompatible with the `SurfaceDataPaths` dataclass.
- `contract_multiplier` has no actual home in `SurfaceDataPaths`, `SurfaceRunnerSettings`, or sizing.
- `cost_model` remains in config but is not the active fill mechanism for the surface path.
- Output artifacts do not include the full config payload used for each run.

### 2. Data loading

Current path:

```text
SurfaceRunner.__init__()
  -> OptionSurfaceDB.load(meta, quotes)
  -> read liquidity_panel parquet
  -> read optional earnings parquet

SurfaceRunner._load_features_for_config()
  -> features_path_for_config(config)
  -> read features parquet
```

What works:

- Surface, liquidity, optional earnings, and features are separated cleanly.
- Feature files are inferred from the feature window in column names.

Gaps:

- Missing schema checks for all loaded artifacts.
- No up-front validation that feature dates, surface entry dates, and liquidity panel coverage overlap.
- No coverage summary before a run starts.

### 3. Trade-date selection

Current path:

```text
_get_trade_dates(features, config)
  -> all feature dates between config.start_date and config.end_date
```

What works:

- Simple and deterministic.

Gaps:

- v1 says weekly rebalance. The runner does not explicitly use the precomputed weekly Friday/trading-day schedule.
- If feature dates are monthly or stale, the runner follows them.
- If feature dates are weekly but surface dates differ because of holiday resolution or precompute gaps, the runner attempts those dates and records structure failures.
- Date density feeds Sharpe annualization assumptions.

### 4. Universe selection

Current path:

```text
step1_get_universe(trade_date, liquidity_panel, config)
  -> latest month_date <= trade_date
  -> require valid ATM pair
  -> rank dollar volume and spread
  -> keep rows passing both thresholds
```

What works:

- The panel lookup is point-in-time by month.
- Ranking is cross-sectional on the chosen month snapshot.

Gaps:

- No direct unit test for the PIT lookup and threshold behavior.
- The v1 protocol text emphasizes top 20% liquidity, while implementation requires both top dollar volume and tightest spread thresholds. That may be intended but should be pinned.
- No coverage report compares the selected dynamic universe to available surface rows.

### 5. Signal scoring

Current path:

```text
step2_score_signals(trade_date, features, universe, config)
  -> exact date feature slice
  -> inner join with universe
  -> drop missing momentum/CVG
  -> count guard
  -> long top momentum, short bottom momentum
  -> CVG filter within each side
```

What works:

- Signal ranking is pure and easy to test.
- Existing feature calculators use lagged row windows in bulk mode.

Gaps:

- No runner-level test proves exact feature date alignment.
- Feature generation still depends on straddle history, not the surface artifacts. That can be fine, but the data lineage should be documented because the surface runner is not fully surface-native at the signal layer.
- PIT confidence requires validating both feature calculators and the upstream straddle history used to build features.

### 6. Structure assembly

Current path:

```text
_build_structures_for_date()
  -> OptionSurfaceDB.get_metadata()
  -> _assemble_structure()
      long direction -> long straddle
      short + ironfly -> iron fly
      short + ironcondor -> iron condor
      short + straddle -> short straddle
  -> _assembly_to_row()
```

What works:

- Assembly math is the strongest layer.
- `OptionSurfaceDB` blocks invalid metadata rows.
- Surface builders are covered by unit tests for entry economics, fills, max loss, and settlement.

Gaps:

- Runner-level handling of missing metadata, invalid surface rows, no quotes, failed wing selection, and spread filters is untested.
- Candidate availability diagnostics are not yet aggregated into a coverage/skip report.
- `pipeline.step3_get_eligible_structures()` duplicates some of this logic but is not used by `SurfaceRunner`.

### 7. Exclusions

Current path:

```text
_has_earnings_nearby(ticker, expiry_date, exclusion_days)
  -> per-row earnings window check
```

What works:

- The runner has a simple earnings exclusion hook.

Gaps:

- Optional earnings artifact path/schema is not pinned in active docs.
- `pipeline.step4_apply_exclusions()` is still `pass`.
- No test covers exclusion interaction with diagnostics, selection, or cap counts.

### 8. Selection, sizing, and settlement

Current path:

```text
_select_size_and_settle()
  -> mark structure failures
  -> mark earnings exclusions
  -> group eligible rows by direction
  -> select top/bottom signal rows up to max_names_per_side
  -> call assembly.settle(exit_spot)
  -> store pnl_per_share
```

What works:

- Selection and exclusion reasons are understandable.
- Included rows settle through the same assembly object built from the surface.
- Diagnostic rows can be retained.

Gaps:

- Despite comments/docstrings, there is no integer contract sizing.
- `max_loss_budget_per_trade`, `SurfaceRunnerSettings.min_contracts`, and `contract_multiplier` are not used.
- No `contracts`, `max_loss_dollars`, `pnl_dollars`, or realized `return_on_max_loss`.
- `max_names_per_side` does not directly encode the v1 50 total concurrent cap.
- The private `_assembly` object in the DataFrame is useful internally but makes the boundary awkward to test unless a synthetic full-run fixture is used.

### 9. Metrics and scoring

Current path:

```text
build_date_summary(trade_log)
  -> return on body credit by date/side

summarize_trade_log(trade_log)
  -> availability, hit rate, body-credit returns, annualized Sharpe, drawdown, robust_score
```

What works:

- Provides early diagnostics and a simple ranking heuristic.
- Availability and side counts are useful.

Gaps:

- Primary v1 metric should be return on max-loss budget / capital units, not body-credit returns.
- Sharpe is calculated on body-credit returns and assumes weekly frequency.
- No dollar PnL, max-loss denominator, turnover, concentration, top-name attribution, or ops count.
- `robust_score` is not a go/no-go metric.

---

## Implemented / partial / missing checklist

| Required v1 responsibility | Status | Evidence / note |
|----------------------------|--------|-----------------|
| Canonical surface path selected | Implemented | Decision 001 and active specs point to SurfaceRunner. |
| Load surface meta/quotes | Implemented | `OptionSurfaceDB.load()` and runner init. |
| Assemble long straddle | Implemented | `build_straddle_from_surface()` with tests. |
| Assemble short iron fly | Implemented | `build_ironfly_from_surface()` with tests. |
| Assemble short iron condor | Implemented | `build_ironcondor_from_surface()` with tests. |
| Hold-to-expiry strategy-unit settlement | Implemented | `StrategyAssemblyResult.settle()` and surface tests. |
| PIT liquidity universe | Partial | `step1_get_universe()` logic exists, but runner integration and threshold semantics need verification. |
| Momentum/CVG signal ranking | Partial | `step2_score_signals()` exists; feature PIT and date alignment need verification. |
| Weekly rebalance schedule | Partial | Feature-date driven, not explicitly weekly/surface-date aligned. |
| Dynamic universe vs surface coverage | Missing | No report proves selected universe has surface coverage. |
| Candidate skip diagnostics | Partial | Row-level reasons exist; aggregate diagnostics missing. |
| Earnings exclusion | Partial | Hook exists; artifact contract/test missing. |
| 50 max concurrent positions | Partial | HD pins current v1 semantics as 50 total across long+short. Current code uses `max_names_per_side`, so implementation does not yet match. |
| Equal max-loss sizing | Missing | Config field exists; sizing is not implemented. |
| Integer contracts | Missing | Runner comments mention it; no code. |
| Dollar PnL | Missing | Only `pnl_per_share` emitted. |
| Return on max-loss budget | Missing | Theoretical assembly ROC exists; realized run metric missing. |
| Conservative fills | Partial | `FillAssumption.cross()` exists; CLI supports it; default includes mid and cross, not pinned harsh-only for go/no-go. |
| Decision-quality summaries | Missing | Metrics are body-credit diagnostics, not v1 capital metrics. |
| Config/result manifest | Missing | Search writes parquet outputs only. |
| Shadow/paper order sheet | Missing | Trade log not order-sheet-ready. |

---

## Main design boundary question

There are currently two overlapping designs:

1. `pipeline.py` describes a six-step pure-function pipeline, but steps 4-6 are still `pass`.
2. `SurfaceRunner` implements a runner-local version of steps 1-3 and partial selection/settlement.

HD preference: keep `SurfaceRunner` as the orchestration layer for now, but separate the implementation into testable pipeline functions. In the next build sprint, the missing `pipeline.py` steps should be completed or replaced with equivalent pure functions so universe, signal, structure, exclusion, selection, sizing, and cost behavior can be unit-tested independently.

Session B should still target behavior through `SurfaceRunner.run_single_config()` rather than only private helpers. That protects the actual canonical path and creates a higher-level fixture around which later unit tests can be decomposed into `pipeline.py`.

---

## Recommended Session B test boundary

### Previous candidate

The previous leading candidate was:

```text
tests/unit/test_surface_runner_selection.py
  -> direct test of SurfaceRunner._select_size_and_settle()
```

That still has value, especially for cap/exclusion/settlement behavior. However, this A.1 mapping shows that HD's highest concern is broader: whether the current design supports the backtesting data flow at all.

### HD-approved recommendation

Use a small synthetic full-run test:

```text
tests/unit/test_surface_runner_data_flow.py
```

Test shape:

1. Build tiny temporary parquet fixtures:
   - liquidity panel with one PIT month snapshot
   - features file for one feature window and one trade date
   - surface metadata for one long candidate, one short candidate, and one missing/invalid candidate
   - surface quotes sufficient to assemble a long straddle and one short iron fly or condor
2. Instantiate `SurfaceRunner` with `SurfaceDataPaths` pointing at the temporary files.
3. Run one `BacktestRunConfig` through `run_single_config()`.
4. Assert:
   - universe uses the PIT liquidity snapshot
   - signal rows produce expected long/short directions
   - trade log includes selected rows and diagnostic exclusions
   - short structure settles to hand-calculated `pnl_per_share`
   - date/run summaries are produced
   - desired v1 fields are present, or the test intentionally fails / is marked expected-fail to show current missing behavior (`contracts`, `pnl_dollars`, realized return on max-loss)

Why this is better:

- It verifies the real canonical path rather than a private helper.
- It tests Stage A/Stage B column compatibility using saved parquet fixtures.
- It catches date alignment and schema issues that `_select_size_and_settle()` would miss.
- It still exercises the selection/settlement boundary.

What it should **not** do in Sprint 001:

- It should not implement dollar sizing.
- It should not require full ORATS data.
- It should not run a Tier B sample.
- It should not refactor `SurfaceRunner`.

HD accepts that the Session B test may intentionally fail on desired v1 behavior. If the test expects `contracts`, `pnl_dollars`, and realized `return_on_max_loss`, that failure should be treated as a useful verification result: it highlights required engine work before decision-quality backtesting. Do not weaken the test merely to make the current implementation appear complete.

---

## Build sequence implied by this map

### Sprint 001 remainder

1. Proceed to Session B with the approved synthetic `SurfaceRunner.run_single_config()` data-flow test.
2. Allow the test to expose missing v1 fields, including through an intentional failure or expected-fail marker.
3. Do not implement broader engine features in Sprint 001 unless HD explicitly approves a narrow fix after the test exposes it.

### Sprint 002 P0 build

1. Fix CLI/data-path wiring.
2. Implement 50 total long+short cap semantics.
3. Implement integer contracts and dollar PnL.
4. Add realized return on max-loss budget.
5. Add coverage report for dynamic universe vs surface availability.
6. Add config/result manifest.

### Sprint 003 P0/P1 build

1. Upgrade metrics to capital/max-loss units.
2. Add turnover, concentration, and availability diagnostics.
3. Move separable engine logic into `pipeline.py` functions or equivalent pure functions for unit testing.
4. Add a short-window smoke run from cache.

---

## Open questions for HD review

1. Is the stricter universe rule in `step1_get_universe()` correct: top dollar-volume **and** tightest-spread thresholds?
2. Should the surface precompute date schedule become an explicit artifact that the runner consumes?
3. In the following sprint, which implemented area should be reviewed first: surface DB loading, straddle/fly/condor assembly tests, or settlement?

