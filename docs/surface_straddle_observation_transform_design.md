# Surface → straddle observation transform — implementation design (Sprint 005 D2)

**Status:** `REVIEWED — implementation-ready; Sprint 005 accepted in Build mode on 2026-08-01, so implementation is authorized`
**Author role:** design for a later implementation task
**Sprint:** 005, deliverable D2
**Repository commit reviewed:** `236c7991912d45d3125bd32428ec8ace8dd78535`
**Input reviewed:** accepted snapshot `e2c1f8fd44d72176` / build `20260724T045049097520Z_40b16886`
**Downstream consumers:** `MomentumCalculator`, `CVGCalculator` (Sprint 005 D1/D3/D4), `SurfaceRunner` (D5, indirectly)

---

> **Revision note (rev 3).** This revision closes the remaining implementation ambiguities after the design review. It preserves the strategy owner's frozen decision to populate volatility fields on spread-ineligible rows, while pinning status precedence, unusable-IV/RV behavior, invalid-row passthroughs, output postconditions, and immutable republishing. It also removes stale contradictions left in the rev-2 flow and makes the repository's current Audit-mode gate explicit. See the change log for details.

## 1. Purpose, scope, and non-goals

Sprint 005 needs one trustworthy weekly straddle history so that Momentum and CVG features can be computed from the accepted Sprint 004 surface rather than from the legacy chain-based path. That history is the *only* thing standing between the accepted A1/A2 artifacts and the feature calculators, so its contract has to be settled before anything else in the sprint can proceed. This section states what D2 is responsible for and, just as importantly, what it must deliberately leave to other deliverables — Sprint 004's lesson was that over-reaching transforms create duplicate audit logic and global blockers from minor irregularities.

**Purpose.** Convert the accepted snapshot's A1 (`option_surface_meta`) and A2 (`option_surface_quotes`) artifacts into exactly one weekly *straddle observation* per accepted A1 `(ticker, entry_date)` key, with economics populated where the stored body straddle supports them and explicit, reasoned nulls everywhere else. The result is loaded directly as the `straddle_history` data source of a `FeatureDataContext`.

**In scope.**

| Responsibility | Why it belongs to D2 |
|----------------|----------------------|
| Resolve A1/A2 only from the published accepted manifest and bind their paths/digests into lineage | Lineage must attach at the point of read, not later |
| Preserve the complete A1 key grid 1:1 | Calculator lookbacks are row-positional; the grid *is* part of the contract |
| Construct the long ATM straddle from the stored body legs only | Requirement 4; prevents silent substitution |
| Apply the existing economic conventions (long call + long put, midpoint entry, `0.99` per-leg spread) | Requirement 5; these are already pinned in the repository |
| Emit `return_pct`, `entry_iv`, `realized_volatility`, `vol_gap` in calculator-native units | Requirement 8; avoids a downstream adapter |
| Classify and record missingness with a stable vocabulary | Requirements 3, 6, 7 |
| Fail loudly on A1↔A2 structural inconsistency | Requirement 6; silent nulls would look like coverage loss |
| Write a deterministic, snapshot-lineaged artifact | Requirement 11 |

**Non-goals.** D2 does **not** decide or implement momentum window grids, minimum histories, CVG cross-sectional membership, sparse-history participation, zero-gap classification, the 281-window emit, feature eligibility, ranking, economic backtesting, PIT universe or liquidity filtering, trade-date selection, incremental refresh, scheduling, resumability, dashboards, a new orchestration framework, a generalized feature store, or any change to Sprint 004 producers or the snapshot itself. It also does not attempt to reproduce paper definitions where they differ from the approved weekly project contract.

One consequence is worth stating up front because it is easy to get wrong: **D2 emits rows for ticker-weeks that were never tradeable.** That is intentional. Momentum and CVG lookbacks are row-based, so removing an unavailable week would silently shift every later window for that ticker. The grid is the contract; the economics are the payload.

---

## 2. Repository review and confirmed existing contracts

Everything proposed later rests on what the code and the accepted artifacts actually do today, so this section records the review before any design is introduced. Findings are split into three kinds: behavior confirmed by reading implementations, call sites, and tests; measurements taken directly against the accepted snapshot; and requirements established by the reviewed Sprint 005 scope. Where two components disagree, the disagreement is recorded rather than resolved silently.

### 2.1 What was inspected

| Area | Paths and symbols |
|------|-------------------|
| Snapshot resolution | `src/data/input_snapshot.py` (`read_manifest`, `manifest_from_dict`, `ARTIFACT_OPTION_SURFACE_META/_QUOTES`, `compute_snapshot_id`); `src/data/snapshot_foundation.py` (`resolve_under_root`, `sha256_file`, `digest_json`, `ticker_date_keys_digest`); `src/data/snapshot_orchestrator.py` (`_validate_surface_evidence`, manifest `params` assembly); `src/data/snapshot_stage_adapters.py` (surface marker evidence) |
| A1/A2 producer semantics | `src/features/option_surface_analyzer.py` (`_metadata_success_row`, `_metadata_failure_row`, `_quote_rows`, `process_single_entry`, `DOCUMENTED_SURFACE_FAILURE_TAGS`) |
| A1/A2 validation contract | `src/features/option_surface_contract.py` (`SURFACE_META_REQUIRED_COLUMNS`, `SURFACE_QUOTES_REQUIRED_COLUMNS`, `META_GRAIN_COLUMNS`, `QUOTE_GRAIN_COLUMNS`, `check_surface_valid_invariant`, `check_a1_a2_join`, `check_expected_meta_keys`); `docs/surface_engine_data_contract.md` § A1/A2 |
| Surface → straddle construction | `src/backtest/option_surface.py` (`OptionSurfaceDB`, `FillAssumption`, `_row_to_option_quote`, `_build_strategy_entry_cost`, `_mid_entry_cost`, `build_straddle_from_surface`, `StrategyAssemblyResult.settle`) |
| Settlement / payoff / return | `src/core/models.py` (`OptionStrategy.calculate_payoff`, `OptionLeg.calculate_intrinsic_value`, `Position.pnl`, `Position.pnl_pct`) |
| IV / realized-volatility semantics | `src/data/spot_price_db.py` (`calculate_realized_volatility`, `calculate_spot_move_pct`); `src/features/straddle_analyzer.py` (`StraddleHistoryBuilder.process_single_straddle`) |
| Feature data context and calculators | `src/features/base.py` (`FeatureDataContext`, `IFeatureCalculator`); `src/features/momentum_calculator.py`; `src/features/cvg_calculator.py` (`_resolve_vol_gap_col`) |
| Existing feature build scripts and call sites | `scripts/build_features.py` (`load_straddle_history`, `build_features`, `save_features`, CLI defaults); `scripts/precompute_straddle_history.py` (legacy chain path, `MAX_SPREAD_PCT = 0.99`); `src/data/orats_provider.py` (`_apply_filters`) |
| Downstream interfaces | `src/backtest/surface_run_config.py` (`SurfaceDataPaths`, `features_path_for_config`, `infer_feature_window`); `src/backtest/surface_runner.py`; `docs/surface_engine_data_contract.md` § A4 |
| Tests | `tests/unit/test_option_surface_straddle.py`; `tests/unit/test_momentum_calculator.py`; `tests/unit/test_cvg_calculator.py`; `tests/unit/test_option_surface_contract.py`; `tests/contract/test_precompute_input_contract.py`; `tests/contract/test_settle_contract.py`; fixtures `tests/fixtures/sample_straddle_history.csv`, `sample_vol_gap_history.csv`, `sample_vol_gap_history_rv_iv.csv` |
| Conventions and prior decisions | `AGENTS.md`; `docs/README.md`; `docs/repo_map.md`; `docs/v1_spec_pins.md`; `docs/known_bugs.md` (KB-001); `docs/sprint_memos/004_closeout.md`; `docs/agenda/current_sprint.md` |
| Accepted artifacts (read-only) | `…/manifests/input_snapshot_e2c1f8fd44d72176.json`; `…/cache/surface/option_surface_meta_weekly_2018_2026.parquet`; `…/cache/surface/option_surface_quotes_weekly_2018_2026.parquet` |

### 2.2 Behavior confirmed by existing code or tests

**Snapshot resolution is manifest-driven and path-safe.** `read_manifest` validates `schema_version == "1"` and requires the full key set including `artifacts`, `params`, `overall_status`, and `blocking_failures`. Artifact values are snapshot-root-relative strings with separators normalized to `/`. `resolve_under_root` rejects absolute, drive-qualified, and `..`-escaping values and asserts containment. The accepted manifest exposes `option_surface_meta` and `option_surface_quotes` under `cache/surface/`, carries `production_accepted: true`, `overall_status: "WARN"` with `blocking_failures: []`, and publishes `params.surface_actual_a1_key_digest`.

**The A1 key digest is independently reproducible.** `snapshot_stage_adapters` computed the published digest as `ticker_date_keys_digest((entry_date, ticker) …)` with `entry_date` normalized via `pd.Timestamp(d).date()` and `ticker` via `str(t).strip().upper()`; `snapshot_orchestrator._validate_surface_evidence` re-derives it the same way. Recomputing it against the accepted A1 file reproduces `faa7e943e71b8aeaf4ea354713ab5558f44a03c9c211c6a68f53236acaa2cced` exactly, matching the manifest.

**`surface_valid` is the sole validity gate and it is a hard invariant.** The producer sets `surface_valid = has_body_call and has_body_put and n_surface_quotes > 0`, and `check_surface_valid_invariant` FAILs on any violation. `OptionSurfaceDB.get_metadata` raises `ValueError` on `surface_valid=False`. `docs/surface_engine_data_contract.md` § A1 calls it the "primary downstream filter", and `sprint_memos/004_closeout.md` § 5.5 instructs Sprint 005 to filter on it. Invalid rows may still carry partial quote rows; `check_a1_a2_join` treats that as an informational WARN, not an error.

**`build_straddle_from_surface` defines the surface-to-straddle semantics.** For `direction="long"` it selects `quotes[is_body]`, optionally drops legs with `spread_pct > max_leg_spread_pct`, requires both a body call and a body put (else `ValueError("Missing tradeable body call/put …")`), and builds a two-leg `OptionStrategy` with `quantity=+1` on each leg. Entry cost comes from `_build_strategy_entry_cost` with the supplied `FillAssumption`; `FillAssumption.mid()` prices each buy at `bid + 0.5 × (ask − bid)`. `_mid_entry_cost`'s docstring is explicit that it deliberately does *not* use the stored `mid` column, because "the stored mid comes from ORATS and may not be exactly `(bid + ask) / 2`". Settlement is `StrategyAssemblyResult.settle(exit_spot)`, which evaluates `OptionStrategy.calculate_payoff` at `expiry_date` and returns a `Position`. `tests/unit/test_option_surface_straddle.py` pins all of this with hand-checkable values, including `Missing tradeable body` when the spread filter removes a leg.

**Return convention.** `Position.pnl = exit_value − entry_cost`; `Position.pnl_pct = pnl / |entry_cost|`. The legacy `StraddleHistoryBuilder` stores `return_pct = pnl_pct × 100`, i.e. **percentage points**. For a long straddle `entry_cost > 0`, so the return has a hard floor of `−100`.

**IV and realized-volatility conventions.** `StraddleHistoryBuilder` sets `entry_iv = (call.iv + put.iv) / 2` — the plain average of the two body legs' IVs, an annualized decimal. `SpotPriceDB.calculate_realized_volatility` returns `sqrt(252 × mean(r²))` over daily log returns — also an annualized decimal — and returns `None` with fewer than three observations. `spot_move_pct`, by contrast, is stored **multiplied by 100** by both `_metadata_success_row` and `StraddleHistoryBuilder`; that unit asymmetry is real and must be carried through knowingly.

**Vol-gap sign.** `CVGCalculator._resolve_vol_gap_col` computes `vol_gap = realized_volatility − entry_iv` when the column is absent, and the module docstring states the same. Gan and Nguyen's *Continuous Volatility Gaps and Option Momentum* defines the primitive volatility gap in the same direction: realized volatility over the option's life minus entry implied volatility. The legacy `StraddleHistoryBuilder` separately stores `iv_rv_spread = entry_iv − realized_volatility`, which is the **negation**. Sprint 005 D2 therefore pins `vol_gap = realized_volatility − entry_iv`, matching both the paper and the calculator.

**The calculator contract is a plain DataFrame in a `FeatureDataContext`.** Both calculators declare `required_data_sources == ['straddle_history']`. `MomentumCalculator` reads `ticker`, `entry_date`, `return_pct`. `CVGCalculator` reads `ticker`, `entry_date` and either `vol_gap` or both `realized_volatility` and `entry_iv`. `scripts/build_features.py::load_straddle_history` enforces exactly `['ticker', 'entry_date', 'return_pct']` plus the vol-gap pair, and coerces `entry_date` with `pd.to_datetime`.

**Row position, not calendar date, drives the lookbacks.** Both calculators sort by `['ticker', 'entry_date']` and take `iloc[target_position − max_lag : target_position − min_lag + 1]` per ticker (bulk mode uses the equivalent `groupby().rolling().groupby(level=0).shift()`). Counts are NaN-aware: a row with NaN `return_pct` or NaN `vol_gap` occupies a window slot but is excluded from `*_count` and from the statistics. `MomentumCalculator.calculate` additionally uses `ticker_data.index.get_loc(...)`, which requires a unique index.

**The existing calculator test fixtures already have exactly D2's proposed shape.** `tests/unit/test_momentum_calculator.py` documents its fixture as "entry_date: Trade entry dates (weekly Fridays, **ALL dates present**)" and "return_pct: Realized return percentage (**NaN for weeks without trades**)", and `tests/fixtures/sample_straddle_history.csv` contains empty `return_pct` cells for UBER pre-IPO weeks and scattered ADP weeks while keeping every row. `tests/fixtures/sample_vol_gap_history_rv_iv.csv` carries `ticker, entry_date, return_pct, realized_volatility, entry_iv` with blanks, and the CVG test module states the constraint `realized_volatility − entry_iv == vol_gap`. The dense-grid-with-nulls design is therefore not a new idea; it is the shape the calculators were written and tested against.

**Downstream interfaces do not constrain D2 directly.** `SurfaceRunner` consumes A1/A2 and per-window A4 feature files resolved by `SurfaceDataPaths.features_path_for_config` as `features_{max}_{min}.parquet`; it never reads a straddle history. D2's only hard downstream constraint therefore runs through the calculators, and `SurfaceDataPaths` accepts an explicit `features_dir`, so a snapshot-scoped output root needs no code change in D5.

### 2.3 Measurements taken against the accepted snapshot

All figures below were produced read-only from the accepted artifacts at the commit under review.

| Measurement | Value |
|-------------|-------|
| A1 rows / unique `(ticker, entry_date)` keys | 1,063,995 / 1,063,995 (no duplicates) |
| A1 grid shape | 2,391 tickers × 445 entry dates (exact cross product) |
| A1 `entry_date` range | 2018-01-05 → 2026-07-10 (429 Fridays, 16 Thursdays) |
| A1 `frequency` | `weekly` on every row |
| `surface_valid` true / false | 314,385 / 749,610 |
| Invalid `failure_reason` breakdown | `target_weekly_expiry_not_listed` 422,340; `no_spot_price` 293,130; `target_weekly_body_not_quotable` 33,654; `no_spot_at_expiry` 486 |
| `failure_reason` on valid rows | none (all null) |
| Valid rows with null `expiry_date` / `entry_spot` / `exit_spot` / `body_strike` / `dte_actual` | 0 for each |
| Valid rows with null `realized_volatility` (and `spot_move_pct`) | 439 |
| `dte_actual` on valid rows | 7 (295,319), 6 (10,944), 8 (8,122) |
| A2 rows | 4,058,377 (27 columns, 261 MB) |
| Valid keys missing a body call or body put in A2 | 0 |
| Valid keys with more than one body call or body put | 0 |
| Body quote `strike` ≠ A1 `body_strike`, or body quote `expiry_date` ≠ A1 `expiry_date` | 0 |
| Body legs on valid keys with `bid ≤ 0`, `ask ≤ 0`, `mid ≤ 0`, or non-finite `mid` | 0 |
| Body legs on valid keys with `ask < bid` (crossed, negative `spread_pct`) | 62 legs across 51 keys |
| Body legs with `spread_pct > 0.99` | 25,450, affecting 20,240 valid keys (6.44 %) |
| Body legs with `spread_pct == 0.99` exactly | 0 |
| Body call IV vs body put IV | identical on all 628,770 legs (max abs diff 0.0) |
| Stored A2 `mid` vs `(bid + ask) / 2` on body legs | 20 legs differ, max abs diff 0.0078 |
| `0.99` decision under stored-mid vs computed-mid denominator | 0 keys disagree |
| Body legs with `bid < 0.01` | 24 |
| Invalid A1 rows that still carry quote rows | 29,166 |
| Resulting fully usable observations | 294,145 (`return_pct` non-null); 293,707 with a non-null `vol_gap` |
| Rows before manifest `feature_ready_start` (2018-01-12) | 2,391 (the single 2018-01-05 entry date) |

Two measurements deserve emphasis because they drive design decisions rather than merely describe the data. First, **the accepted snapshot contains zero A1↔A2 structural inconsistencies** — every one of the 314,385 valid keys has exactly one body call and one body put, at the A1 strike and A1 expiry, with positive finite quotes. Second, the **per-key access pattern is not viable at production scale**:

| Approach | Measured |
|----------|----------|
| `OptionSurfaceDB.load` on the full A1/A2 | 2.2 s, 2.2 GB RSS (constructor copies both frames) |
| `build_straddle_from_surface` per key on the full DB | 329 ms median per key → **≈ 28.8 hours** for 314,385 valid keys |
| Columnar read of the needed A1/A2 columns | 0.4 s, 1.1 GB RSS |
| Vectorized body-leg join plus economics over the full grid | 0.5 s, 1.4 GB peak RSS |
| Vectorized total | **≈ 0.8 s** |

The per-key cost is dominated by two full boolean scans over the 4.06 M-row quote frame inside `OptionSurfaceDB.get_metadata`/`get_quotes` for every single lookup; it is a point-query API being asked to do a full-history sweep.

A third measurement de-risks the main design choice: on 57 randomly sampled valid keys, a vectorized float implementation of the same arithmetic reproduced the production `Decimal` builder plus `settle` to within `4.4e-16` on `entry_cost`, `4.3e-14` on `pnl`, `5.0e-13` on `return_pct`, and exactly on `entry_iv`.

### 2.4 Requirements established by the reviewed Sprint 005 scope

The reviewed Sprint 005 scope requires D2 to preserve the complete accepted A1 key grid with no duplicates and no dropped keys; populate economics from valid A1/A2 rows; retain unavailable and invalid keys as rows with null derived economics and a missingness reason; never substitute another week or strike; use a simple surface-to-straddle builder with no momentum or CVG rules; preserve independently usable volatility and return information; define `vol_gap = realized_volatility − entry_iv`; be reproducible and snapshot-lineaged; and remain a thin transform rather than a `refresh_weekly_inputs.py` stage. It also establishes that the legacy chain-based `precompute_straddle_history.py` is **not** the Sprint 005 source of truth, and that inconsistent `build_features.py` helper-versus-CLI defaults must not be silently inherited.

At repository commit `236c799`, `docs/agenda/current_sprint.md` is still the earlier `SCOPE UNDER REVIEW` / Audit-mode stub and does not yet contain these D1–D5 requirements. That is an operational gate, not a design ambiguity: this document can be implementation-ready, but production code must not be changed until the reviewed Sprint 005 agenda is committed as accepted Build mode, per `AGENTS.md`.

### 2.5 Disagreements between existing components

These are recorded, not resolved, except where the reviewed Sprint 005 scope already settles them.

| # | Disagreement | Evidence | Consequence | Disposition |
|---|--------------|----------|-------------|-------------|
| C-1 | Vol-gap sign | `CVGCalculator` uses `realized_volatility − entry_iv`; `StraddleHistoryBuilder.iv_rv_spread` uses `entry_iv − realized_volatility` | Sign flip would invert every CVG `cgap`, `%pos/%neg`, and `DVG` branch | **Settled by the reviewed Sprint 005 scope and the paper definition**: D2 emits `vol_gap = realized_volatility − entry_iv`. D2 does not emit `iv_rv_spread` at all, to avoid two similarly named opposite-sign columns |
| C-2 | Spread-filter denominator | Surface path filters on A2 `spread_pct = (ask − bid) / stored_mid`; legacy `ORATSDataProvider._apply_filters` computes `(ask − bid) / ((bid + ask) / 2)` | Could in principle change which legs pass `0.99` | **Measured immaterial**: 0 of 628,770 body legs and 0 of 314,385 keys change eligibility. D2 uses the A2 `spread_pct` column, matching `build_straddle_from_surface` |
| C-3 | Stored `mid` vs midpoint | A2 stores a `mid` column, but `_mid_entry_cost` deliberately recomputes `bid + 0.5 × (ask − bid)` | 20 body legs differ, up to 0.0078 per leg in entry cost | D2 **must** price from `bid`/`ask`, not the `mid` column, to match `build_straddle_from_surface` |
| C-4 | Minimum bid | Legacy path inherits `ORATSDataProvider(min_bid=0.01)` because `StraddleHistoryBuilder` never overrides it; the surface producer only requires `bid > 0` | 24 body legs have `bid < 0.01` (min observed 0.0025) | D2 adds **no** minimum-bid rule (requirement 5: no new filters). Recorded as a legacy-versus-surface difference |
| C-5 | Volume / open interest | Legacy `precompute_straddle_history.py` sets `MIN_VOLUME = 0`, `MIN_OI = 0`; the surface producer stores volume and OI but never filters on them | 64,496 body legs have `volume == 0`; 28,094 have `open_interest == 0` | D2 adds **no** volume or OI requirement, consistent with both paths |
| C-6 | Crossed quotes | 62 body legs across 51 valid keys have `ask < bid`, producing negative `spread_pct` that trivially satisfies `≤ 0.99`; the midpoint stays positive | 51 of 314,385 keys (0.016 %) | Existing behavior includes them; requirement 5 forbids adding a new filter. Recorded as assumption **R-12**, not an open question |
| C-7 | `build_features.py` defaults | Helper defaults (`MomentumCalculator(min_periods=1)`, `CVGCalculator(min_periods=1)`) differ from CLI defaults (`--min-periods-momentum 3`, `--min-periods-cvg 5`) | Would change feature values | Out of D2 scope; belongs to D1/D3's versioned config. Noted so the later implementer does not import a default by accident |
| C-8 | Feature-ready window | Manifest `params.feature_ready_start = 2018-01-12`, but A1 begins 2018-01-05 | 2,391 rows sit before the feature-ready start | D2 preserves the **full A1 grid** (requirement 2 outranks trimming). Windowing is D3/D4's decision |

---

## 3. End-to-end data flow

The flow is deliberately short: read two artifacts, join them on one grain, compute a handful of scalars, and write one table with a receipt. Keeping it short is the point — every extra stage would be a place where the key grid could drift or where an implicit filter could creep in. The design also front-loads all validation, so the transform either produces a complete, verified artifact or produces nothing at all.

```text
accepted snapshot root  (C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886)
  └─ manifests/input_snapshot_e2c1f8fd44d72176.json
        │  read_manifest  →  assert schema_version=1, production_accepted, blocking_failures==[]
        │  resolve_under_root(artifacts["option_surface_meta"|"option_surface_quotes"])
        ▼
  A1 option_surface_meta (1,063,995 rows)      A2 option_surface_quotes (4,058,377 rows)
        │  columnar read (needed columns only)       │  columnar read, is_body rows only
        ▼                                            ▼
  ── INPUT GUARD ────────────────────────────────────────────────────────────────
  recompute ticker_date_keys_digest(A1 keys) == params.surface_actual_a1_key_digest
        ▼
  ── BODY-LEG JOIN ──────────────────────────────────────────────────────────────
  one row per A1 key, left-joined to its body call and body put
        ▼
  ── STRUCTURAL VALIDATION (fail-closed; no partial artifact) ───────────────────
  missing/duplicate body leg on a valid row │ strike or expiry disagreement
        ▼
  ── OBSERVATION CONSTRUCTION (vectorized, whole grid at once) ──────────────────
  deterministic status precedence → entry economics → payoff → pnl → return_pct
  usable entry_iv → usable realized_volatility → vol_gap
        ▼
  ── OUTPUT POST-CONDITIONS (fail-closed; D2 contract only) ─────────────────────
  exact unique A1 key grid │ status/reason consistency │ dependent-null contract
        ▼
  ── EMIT ───────────────────────────────────────────────────────────────────────
  straddle_observations_weekly.parquet        (1,063,995 rows, sorted, index=False)
  straddle_observations_weekly.lineage.json   (snapshot ids, digests, config, counts)
        ▼
  FeatureDataContext(straddle_history=pd.read_parquet(...))
        ├─ MomentumCalculator   (ticker, entry_date, return_pct)
        └─ CVGCalculator        (ticker, entry_date, vol_gap)
```

Nothing in this path touches `C:/MomentumCVG_env/cache/`, `C:/MomentumCVG_env/input/`, `C:/ORATS/`, or the legacy chain history, and nothing writes inside the snapshot root.

---

## 4. Input contract

D2 should be strict about what it will accept, because the whole value of the deliverable is that its output can be trusted to come from one specific, accepted input. The guards below are cheap — a manifest parse and one digest over 1.06 M keys — and they convert the most dangerous silent failure mode (pointing at a mutable cache that happens to have the same filenames) into an immediate, explicit error. Note that `C:/MomentumCVG_env/cache/` really does contain files named `option_surface_meta_weekly_2018_2026.parquet` and `option_surface_quotes_weekly_2018_2026.parquet` at different sizes, so this is not a hypothetical risk.

**Resolution.** The only input parameter is the snapshot root. A1 and A2 are resolved as `resolve_under_root(root, manifest.artifacts["option_surface_meta"])` and the equivalent for `option_surface_quotes`; the manifest is located via `default_manifest_path(root, snapshot_id)` or by the single file under `root/manifests/`. Direct path overrides for A1/A2 are deliberately **not** offered in the production CLI.

**Accept conditions.** The run proceeds only when all of the following hold.

| Guard | Rule | Failure mode |
|-------|------|--------------|
| Manifest parse | `read_manifest` succeeds (`schema_version == "1"`) | `ValueError` from existing code |
| Acceptance | `production_accepted is True` | Abort with explicit message |
| No blocking failures | `blocking_failures == []` | Abort |
| Status tolerance | `overall_status in {"PASS", "WARN"}` | Abort on `FAIL`. The accepted snapshot is `WARN`, so `WARN` must be allowed |
| Artifacts present | Both resolved paths exist and are files | Abort |
| A1 key-set identity | `ticker_date_keys_digest(A1 keys) == params["surface_actual_a1_key_digest"]` | Abort — protects against resolving an A1 with the wrong key grid |

The A1 key digest proves the accepted **key set**, not every A1 value or byte. D2 relies on the snapshot's published immutability for content trust and records both source-file SHA-256 digests in its lineage receipt; it does not overstate a key digest as a full-file integrity proof.

**Columns consumed.** Reading only what is needed is what makes the vectorized path fit comfortably in memory.

| Artifact | Columns read | Use |
|----------|--------------|-----|
| A1 | `ticker`, `entry_date`, `expiry_date`, `dte_actual`, `entry_spot`, `exit_spot`, `body_strike`, `spot_move_pct`, `realized_volatility`, `surface_valid`, `failure_reason` | Key grid, settlement, missingness, RV |
| A2 | `ticker`, `entry_date`, `expiry_date`, `strike`, `side`, `is_body`, `bid`, `ask`, `spread_pct`, `iv` | Body-leg pricing and eligibility |

A2's `mid`, `volume`, `open_interest`, greeks, and OTM wing rows are intentionally **not** read: `mid` must not be used for pricing (C-3), volume/OI must not gate anything (C-5), and non-body rows can never contribute to a body straddle (requirement 4).

**Assumed input types.** A1/A2 store `entry_date` and `expiry_date` as Parquet `date32[day]`, which pandas 3.0.0 materializes as `object` columns of `datetime.date`. This matters downstream and is handled in § 5.

---

## 5. Output contract and calculator compatibility

The output has one job beyond being correct: it must drop straight into `FeatureDataContext(straddle_history=...)` and work, with no renaming, unit conversion, or wrapper. That rules out several superficially reasonable choices — storing dates as `date32`, using a MultiIndex, expressing returns as fractions — each of which would force a downstream adapter. This section derives the contract from what the calculators actually read and from a reproduced failure.

### 5.1 The hard compatibility rules

| Rule | Derived from | What breaks if violated |
|------|--------------|-------------------------|
| Column names exactly `ticker`, `entry_date`, `return_pct`, `realized_volatility`, `entry_iv`, `vol_gap` | `MomentumCalculator.calculate*`, `CVGCalculator._resolve_vol_gap_col`, `build_features.load_straddle_history` | `KeyError` / `ValueError` at load |
| `entry_date` stored as **`datetime64[ns]`**, not `date32`/object-of-`date` | `MomentumCalculator.calculate` compares `history['entry_date'] <= date` with **no dtype coercion** | Reproduced against the accepted A1 under pandas 3.0.0: `TypeError: Cannot compare Timestamp with datetime.date`. `calculate_bulk` coerces and would survive; `calculate` would not |
| Unique, default `RangeIndex`; written with `index=False` | `MomentumCalculator.calculate` uses `ticker_data.index.get_loc(...)`; bulk mode assigns rolling results back by index; `save_features` writes `index=False` | Non-unique index makes `get_loc` return a slice/array and misaligns bulk assignment |
| `ticker` upper-case strings | Both calculators upper-case the requested tickers and then match with `==` / `.isin` | Silent all-NaN features for mismatched case |
| `return_pct` in **percentage points** | `StraddleHistoryBuilder`: `pnl_pct × 100`; fixture values range from `−99` to `+289` | Momentum means off by 100× |
| `realized_volatility` and `entry_iv` as **annualized decimals** | `SpotPriceDB.calculate_realized_volatility` returns `sqrt(252·mean(r²))`; `entry_iv` is raw ORATS IV | `vol_gap` becomes a mixed-unit difference |
| `vol_gap = realized_volatility − entry_iv`, emitted explicitly | `CVGCalculator._resolve_vol_gap_col` prefers an existing `vol_gap` column over deriving it | Emitting it removes any ambiguity about sign (C-1) |
| One row per `(ticker, entry_date)`, complete grid, no duplicates | Row-positional lookbacks in both calculators | Every window after a dropped week silently shifts |
| Float columns are `float64` with `NaN` (not pandas NA) for missing | `np.mean`, `np.std(ddof=1)`, `np.median` on `.values` in both calculators | Nullable dtypes can propagate `pd.NA` into numpy reductions |

### 5.2 Proposed output schema

`straddle_observations_weekly.parquet` — grain `(ticker, entry_date)`, 1,063,995 rows for the accepted snapshot, sorted ascending by `ticker` then `entry_date`.

| Column | Type | Null when | Meaning |
|--------|------|-----------|---------|
| `ticker` | `str` (upper) | never | A1 ticker |
| `entry_date` | `datetime64[ns]` | never | A1 entry date, midnight-normalized |
| `observation_status` | `str` | never | `ok` \| `body_spread_ineligible` \| `body_quote_unusable` \| `surface_invalid` |
| `missing_reason` | `str` | iff `observation_status == "ok"` | Stable tag; see § 7 |
| `surface_valid` | `bool` | never | A1 passthrough, for traceability |
| `expiry_date` | `datetime64[ns]` | when the A1 source value is null | A1 passthrough |
| `dte_actual` | `float64` | when the A1 source value is null | A1 passthrough |
| `entry_spot` | `float64` | when the A1 source value is null | A1 passthrough |
| `exit_spot` | `float64` | when the A1 source value is null | A1 passthrough; drives settlement |
| `body_strike` | `float64` | when the A1 source value is null | A1 passthrough; the straddle strike |
| `spot_move_pct` | `float64` | when the A1 source value is null | A1 passthrough — **percent units**, unlike the volatilities |
| `call_bid`, `call_ask`, `put_bid`, `put_ask` | `float64` | no body leg present | Stored body-leg quotes; make entry pricing re-derivable |
| `call_iv`, `put_iv` | `float64` | no body leg present, or the stored A2 value is null | Raw stored body-leg IVs; make `entry_iv` re-derivable and expose unusable source values |
| `call_spread_pct`, `put_spread_pct` | `float64` | no body leg present | A2 passthrough; the `0.99` test operand |
| `entry_cost` | `float64` | not `ok` | Long-straddle debit per share at midpoint (always `> 0`) |
| `exit_value` | `float64` | not `ok` | Straddle intrinsic value at `exit_spot` |
| `pnl` | `float64` | not `ok` | `exit_value − entry_cost` |
| `return_pct` | `float64` | not `ok` | `pnl / entry_cost × 100`, percentage points, floor `−100` |
| `entry_iv` | `float64` | `surface_invalid`, `body_quote_unusable`, or either leg IV is not positive and finite | Mean of body call and put IV, annualized decimal |
| `realized_volatility` | `float64` | the A1 value is null, negative, or non-finite | A1 value when usable, annualized decimal; preserved even on `surface_invalid` rows |
| `vol_gap` | `float64` | either calculator-facing component is null | `realized_volatility − entry_iv` |

The six raw leg columns (`call_bid`/`call_ask`/`put_bid`/`put_ask`, `call_iv`/`put_iv`) and the two `*_spread_pct` columns are the only additions beyond what the calculators strictly need. Together they make **every** derived economic field recomputable from the artifact alone during a D1 audit, which is far cheaper than re-reading a 261 MB A2 file to check one row. The two IV columns matter specifically because `entry_iv` is a derived mean: without them, the artifact would assert a value no reader could verify, and the frozen rule in § 7.2 would discard information irrecoverably. `iv_rv_spread` is deliberately absent (C-1). No momentum, CVG, ranking, eligibility, universe, or window column appears anywhere in the schema.

### 5.3 How compatibility is proven, not assumed

Compatibility is asserted by tests that construct a `FeatureDataContext` directly from the emitted artifact and run the real production calculators (§ 11), rather than by re-implementing their expectations. That mirrors the D1 instruction to audit through production code with no second implementation.

---

## 6. Observation-construction semantics

This section pins the arithmetic. The guiding rule is that D2 must not invent economics: every formula below already exists in the repository, and D2's contribution is to apply it to the whole grid at once and to state precisely which stored row each number came from. Where the repository offers two ways to compute the same quantity, § 2.5 records which one was chosen and why.

**Leg selection is a lookup, never a search.** For each A1 key with `surface_valid == True`, the two legs are the A2 rows with `is_body == True` and `side` of `call` and `put` for that exact `(ticker, entry_date)`. There is no nearest-strike, nearest-delta, nearest-expiry, or nearest-week fallback anywhere in the transform. `strike` and `expiry_date` on the selected legs must equal the A1 `body_strike` and `expiry_date`; a mismatch is a structural error (§ 7), not a repair opportunity. This is the direct expression of requirement 4, and it is why D2 reads only `is_body` rows: the OTM wings are not merely unused, they are unreachable.

**Structure and direction.** Long call plus long put at `body_strike`, quantity `+1` each, matching `build_straddle_from_surface(direction="long")` and the `docs/v1_spec_pins.md` pin "Long structure … Buy ATM call + put".

**Entry pricing.**

```text
entry_cost = (call_bid + call_ask) / 2 + (put_bid + put_ask) / 2
```

This is `FillAssumption.mid()` applied to two long legs — `bid + 0.5 × (ask − bid)` per leg — and it uses the raw `bid`/`ask`, not the stored `mid` column, exactly as `_mid_entry_cost` documents (C-3). On an `ok` row both legs must have positive finite bids/asks, so `entry_cost > 0`; the accepted-snapshot measurement confirms zero non-positive or non-finite entry costs across all 314,385 valid keys.

**Quote eligibility.** A key is entry-eligible when both body legs have positive finite bids/asks and finite `spread_pct <= 0.99`. On the accepted snapshot this reproduces `build_straddle_from_surface(max_leg_spread_pct=0.99)`: that function filters the body frame *before* leg selection and then raises `Missing tradeable body call/put` if either side is gone. The threshold value `0.99` is the existing repository constant (`precompute_straddle_history.MAX_SPREAD_PCT`, `IronflyHistoryBuilder(max_spread_pct=0.99)`). The comparison is `<=`, so a leg at exactly `0.99` passes; no leg in the accepted snapshot sits exactly on the boundary, so the test must be synthetic. No volume, open-interest, or minimum-bid condition is added (C-4, C-5).

**Settlement and payoff.**

```text
exit_value = max(exit_spot − body_strike, 0) + max(body_strike − exit_spot, 0)
pnl        = exit_value − entry_cost
return_pct = pnl / entry_cost × 100
```

This is `OptionStrategy.calculate_payoff` for two long legs at a single expiry, `Position.pnl`, and `Position.pnl_pct × 100`. `abs(entry_cost)` and `entry_cost` coincide here because the long straddle is always a debit; the implementation should still divide by `abs(entry_cost)` so the expression stays literally identical to `Position.pnl_pct`. Hold-to-expiry with `exit_spot` from A1 matches the § S7 contract and `v1_spec_pins.md` ("Backtest exit: Hold to expiry").

**Volatility fields.**

```text
entry_iv = (call_iv + put_iv) / 2
vol_gap  = realized_volatility − entry_iv
```

`entry_iv` follows `StraddleHistoryBuilder`. In the accepted snapshot the two body IVs are bit-identical on all 628,770 legs, so the averaging convention is currently inconsequential — but it must still be pinned, because a future producer change to per-side IV would silently alter every CVG input. The convention is recorded as assumption **R-1**; it is a pinned decision, not an open question. Both raw inputs are retained as `call_iv` and `put_iv` so the mean stays auditable. `realized_volatility` comes only from A1: finite non-negative values pass through, unusable values become null, and D2 never recomputes it or opens the spot database.

**Precision.** The transform works in `float64` rather than `Decimal`. The measured agreement with the production `Decimal` path is `≤ 5e-13` on `return_pct` across a random sample of 57 keys, which is far below any economically meaningful threshold and is pinned by an equivalence test with an explicit tolerance (§ 11). `Decimal` across 1.06 M rows would forfeit the entire vectorization benefit for no decision-relevant accuracy.

**Reuse posture.** `build_straddle_from_surface` is reused where it is both correct and affordable — as the **reference oracle in tests** — and is not called in the production path, because at 329 ms per key it would take about 28.8 hours (§ 2.3). The vectorized implementation is therefore not a competing definition of the economics; it is the same arithmetic, held to the original by an equivalence test. This keeps a single source of truth for the semantics while respecting requirement 10.

---

## 7. Missingness and structural-error behavior

The central distinction here is between *the market did not offer a usable straddle that week* and *the input artifacts contradict each other*. The first is normal, expected, and must be preserved as data — it is 72 % of the grid. The second never occurs in the accepted snapshot and, if it did, would mean the transform is reading something other than what it thinks it is reading. Treating the two the same way would let a corrupted input masquerade as poor coverage, which is precisely the kind of silent degradation Sprint 005 is meant to eliminate.

### 7.1 Ordinary unavailability — recorded as rows

Every A1 key produces exactly one output row regardless of availability. Nothing is dropped, deduplicated away, forward-filled, or replaced by an older observation.

| `observation_status` | `missing_reason` | Source of the classification | Rows in the accepted snapshot |
|----------------------|------------------|------------------------------|-------------------------------|
| `ok` | *(null)* | `surface_valid` and both body legs pass `spread_pct <= 0.99` | 294,145 |
| `body_spread_ineligible` | `body_spread_above_threshold` | `surface_valid` but at least one body leg has `spread_pct > 0.99` | 20,240 |
| `body_spread_ineligible` | `body_spread_unavailable` | `surface_valid` but a body leg has a null or non-finite `spread_pct`, so eligibility cannot be confirmed | 0 |
| `body_quote_unusable` | `body_quote_not_positive_finite` | `surface_valid` but a body leg has a non-positive or non-finite `bid`/`ask` | 0 |
| `surface_invalid` | `target_weekly_expiry_not_listed` | A1 `failure_reason` passthrough | 422,340 |
| `surface_invalid` | `no_spot_price` | A1 `failure_reason` passthrough | 293,130 |
| `surface_invalid` | `target_weekly_body_not_quotable` | A1 `failure_reason` passthrough | 33,654 |
| `surface_invalid` | `no_spot_at_expiry` | A1 `failure_reason` passthrough | 486 |
| `surface_invalid` | `surface_invalid_reason_missing` | A1 `surface_valid == False` with a null `failure_reason` | 0 |
| **Total** | | | **1,063,995** |

The `surface_invalid` reasons are A1's own vocabulary, validated by `check_failure_vocabulary` against `DOCUMENTED_SURFACE_FAILURE_TAGS`; D2 passes them through unchanged rather than inventing a parallel taxonomy. Three tags are D2's own — `body_spread_above_threshold`, `body_spread_unavailable`, and `body_quote_not_positive_finite` — because those determinations are made by D2, not by the producer. The `surface_invalid_reason_missing` placeholder exists because `check_failure_vocabulary` only WARNs on a null reason, so the column must remain non-null by construction; it is unused today.

Three of these categories have zero rows in the accepted snapshot. They exist so that a future or degraded snapshot degrades **row by row** rather than failing the whole run, which § 7.3 explains. They are cheap to carry and they cost nothing today.

**Classification precedence is frozen.** After structural leg matching succeeds, each row is classified in this order: (1) `surface_invalid`; (2) `body_quote_unusable` for any non-positive or non-finite body bid/ask; (3) `body_spread_ineligible` for a null/non-finite spread or a finite spread above `0.99`; (4) `ok`. This prevents overlapping bad conditions from being resolved by incidental `np.select` or branch order. A finite negative spread from a crossed quote remains eligible under R-12.

**Field-level nulls do not change the status.** A leg IV that is null, non-positive, or non-finite, or an A1 realized volatility that is null, negative, or non-finite, nulls only the calculator-facing fields that depend on it. An otherwise priceable row keeps `observation_status = "ok"` and its `return_pct`. Status answers "was there a usable straddle observation this week", which is a different question from "is every derived field available". Conflating the two is what would erase a valid Momentum return over an unrelated missing input, and requirement 7 forbids that.

The 29,166 invalid A1 rows that still carry partial quote rows are **not** used. `surface_valid` is the contract's primary filter, `OptionSurfaceDB.get_metadata` refuses them, and `sprint_memos/004_closeout.md` § 5.4–5.5 explicitly keeps them informational. Harvesting them would be exactly the "substitute another strike or contract" behavior requirement 4 forbids.

### 7.2 Preserving independently usable information

Requirement 7 asks that one missing input not erase unrelated valid information. Three situations arise, and they are handled differently.

**Missing realized volatility with a valid straddle (438 rows).** These keys are `ok`: `entry_cost`, `pnl`, and `return_pct` are populated from the body legs, `entry_iv` is populated, and only `realized_volatility` and therefore `vol_gap` are null. Momentum sees a usable return; CVG skips the week via its NaN-aware count. No coupling is introduced.

**Entry-ineligible spread with observable volatilities (20,240 rows).** These keys have both body legs present with positive, finite quotes — the straddle is simply too wide to claim as an entry. The trade economics (`entry_cost`, `exit_value`, `pnl`, `return_pct`) are null. Whether `entry_iv`, `realized_volatility`, and `vol_gap` should *also* be nulled changes CVG values on 6.4 % of usable ticker-weeks, so it is a strategy-owner decision rather than an implementation detail.

**Frozen rule (decision D-1).** Spread ineligibility does **not itself null** the volatility fields: `observation_status = "body_spread_ineligible"`, trade economics are null, and `entry_iv`, `realized_volatility`, and `vol_gap` are populated wherever their own components are usable. The rationale is requirement 7 read literally — both body legs are observable, their IVs are stored, and a volatility gap is an observation about the market rather than a claim that the straddle was enterable. Spread width governs whether the position could be *taken*, not whether the vol gap was *seen*.

This is **one canonical rule, not a configurable policy.** The implementation must not expose a CLI flag or config branch selecting the alternative, and the builder must not carry both code paths. A canonical artifact needs exactly one economic meaning; a switch would let the same filename at the same path hold either of two artifacts, which is the identity hazard § 9 guards against. The rule is recorded in the lineage receipt as a frozen named constant so the artifact remains self-describing, in the same way `entry_iv_rule` and `vol_gap_rule` are recorded.

**Consequence D1 must know about.** `CVGCalculator` adjusts `vol_gap` by a **per-date cross-sectional median** (`history.groupby('entry_date')[vol_gap_col].transform(lambda x: x - x.median())`, `cvg_calculator.py:272` and `:482`), and `.median()` skips NaN. Populating these 20,240 rows therefore admits them into the median on their dates, which shifts `vol_gap_adjusted`, `%pos`, `%neg`, and `volgap_mean` **for every other ticker on those dates** — not only for the ineligible rows themselves. The blast radius of D-1 is wider than its row count suggests, and D1's audit should treat it that way.

The decision is nonetheless reversible in both directions, which is why it can be frozen without foreclosing D1's analysis. To reproduce the nulled variant, D1 masks `vol_gap` where `observation_status != "ok"` before calling the calculator. To reproduce this variant from a nulled artifact, D1 would recompute `entry_iv = (call_iv + put_iv) / 2` and `vol_gap = realized_volatility − entry_iv`. Both directions are available only because `call_iv` and `put_iv` are retained in the schema (§ 5.2); without them the choice would be one-way.

The counter-evidence is recorded honestly at **R-11**: the legacy `StraddleHistoryBuilder` nulls `entry_iv` and `iv_rv_spread` together with the return when its leg filter rejects a strike, so D-1 departs from legacy precedent.

**Unusable body quotes do not inherit D-1.** If a bid or ask is non-positive or non-finite, the row is `body_quote_unusable`; trade economics, calculator-facing `entry_iv`, and `vol_gap` are null. Raw `call_iv`/`put_iv` and usable A1 `realized_volatility` remain in the artifact for auditability. This is intentionally narrower than the spread-ineligible rule: a wide but valid quote still supports an observed IV, while an unusable quote does not support a calculator-facing market observation. The accepted snapshot has zero such rows, so this rule changes no current D2 value but prevents a future implementer from guessing.

### 7.3 Structural errors — fail the run

The hard question here is where to draw the line between a condition that aborts the run and a condition that produces a null-economics row. Draw it too tight and a single bad quote in 2028 blocks the entire pipeline; draw it too loose and a corrupted input quietly reports itself as poor market coverage. The line this design uses is not a case list but a principle, so that conditions nobody has anticipated still classify correctly.

**The governing distinction.** A **structural error** is a contradiction *between A1 and A2*: the two artifacts disagree about what exists. A **row-level unavailability** is a leg's own values not supporting a calculation. The first means D2 is reading something other than what it believes it is reading, and no observation for that key can be trusted. The second is ordinary data quality, which the grid was designed to absorb.

This maps directly onto requirement 6, which asks D2 to distinguish "ordinary unavailable observations from structural inconsistencies **between A1 and A2**." A body leg with a bad quote is not an A1↔A2 inconsistency at all — A1 and A2 agree completely about which contract exists; the quote is simply unusable. It belongs in § 7.1, not here.

**D2 trusts the accepted snapshot's input-layer certification.** It does not recertify A1 or A2. Sprint 004 already certified the A1/A2 schemas, A1 uniqueness and coverage, settlement fields, weekly-date alignment, quote grain, join integrity, and `surface_valid` consistency, and the manifest publishes that acceptance. Re-running those checks would be a second input audit inside a transform that is supposed to be thin — and `check_a1_a2_join` over 4.06 M quote rows is not free. D2 fails only on its input key-set guard, when its exact body-leg join cannot satisfy the construction contract, or when its own emitted artifact violates the key/status/null contract.

**Run-fatal categories.** These are input-identity failures, A1↔A2 contradictions, or violations of D2's own output contract.

| Category | Detection | Accepted-snapshot count |
|----------|-----------|-------------------------|
| Valid A1 row with no body call or no body put | Null after the body-leg join | 0 |
| Valid A1 row with more than one body call or put | Body-leg group size `> 1` at `(ticker, entry_date, side)` | 0 |
| Body leg strike ≠ A1 `body_strike` | Direct comparison | 0 |
| Body leg expiry ≠ A1 `expiry_date` | Direct comparison | 0 |
| A1 key digest mismatch | Recomputed digest ≠ manifest `surface_actual_a1_key_digest` | matches |
| Output key set ≠ input A1 key set, or a duplicate output key | Set comparison and uniqueness check after construction | n/a (post-condition) |
| Output status/reason or dependent-null contract is violated | Vectorized assertions after construction | n/a (post-condition) |

Two of these deserve a note on why they stay despite the scope reduction.

The **A1 key digest check** is key-set identity protection, not recertification and not a full-file content proof. It answers "does this A1 carry the accepted key grid?", not "are all A1 values unchanged?" It is a useful defense against accidentally resolving the mutable cache named in R-7, reuses `ticker_date_keys_digest`, and costs one pass over the key columns. Content trust still comes from the accepted snapshot's immutability; the receipt records source-file SHA-256 values for later lineage comparison.

The **output post-conditions** prove what D2 itself promises: exactly one unique row per A1 key; `missing_reason` null exactly for `ok` rows; finite non-null trade economics for `ok` and null trade economics otherwise; `entry_iv` and `vol_gap` populated only under the frozen availability rules; and no infinite calculator-facing values. A duplicated input key cannot satisfy the key contract and surfaces here. These assertions are better targeted than calling `check_meta_grain` or re-running a full input audit.

**Explicitly not run-fatal.** Non-positive or non-finite body quotes, null/non-finite `spread_pct`, unusable leg IV, and unusable `realized_volatility` are all row-level conditions handled per § 7.1 and § 7.2. In particular, an unusable leg IV does not prevent midpoint entry, payoff, or `return_pct`, so it nulls `entry_iv` and `vol_gap` while leaving the Momentum return intact.

**Explicitly not performed.** D2 does not re-run A1 failure-vocabulary checks, settlement-field certification, weekly-date alignment checks, global A1/A2 join audits, A2 quote-grain audits, or orphan-A2 reporting. Orphan A2 keys cannot affect a left join from A1, so they are neither reported nor counted.

**Handling.** Each validation boundary is vectorized. Body-leg contradictions are aggregated into one report — counts plus up to five example keys per category — before raising; output post-conditions are evaluated together after construction. Any failure raises `StraddleObservationStructuralError` and writes **no** artifact. This gives the operator the full shape of a failure at the relevant boundary without creating a second audit framework.

This is deliberately *not* a general audit framework. It is a short fixed list of assertions inside the transform, with no report renderer, no PASS/WARN/FAIL grading, no CLI surface of its own, and no persisted audit artifact.

---

## 8. Performance and resource considerations

Requirement 10 asks that the approach be chosen from measured behavior on the real data rather than from what looks natural on a fixture, and in this case the measurement is decisive rather than marginal. The obvious implementation — loop over valid keys calling the existing builder — is roughly five orders of magnitude too slow, and the reason is structural, not incidental.

**Why the natural approach fails.** `OptionSurfaceDB` builds an in-memory frame with no index on `(ticker, entry_date)`; `get_metadata` and `get_quotes` each evaluate a full boolean mask, so every single-key lookup scans all 1.06 M metadata rows and all 4.06 M quote rows. That is correct and perfectly appropriate for `SurfaceRunner`, which performs a few hundred lookups per trade date on a filtered universe. It is the wrong shape for a full-history sweep. Measured: 329 ms median per key, giving **≈ 28.8 hours** for 314,385 valid keys, plus 2.2 GB resident from the constructor's two `.copy()` calls.

**The chosen approach.** Read only the required columns, restrict A2 to `is_body` rows before joining, pivot the two body legs onto the A1 key, and compute all economics as whole-column vector operations.

| Stage | Measured time | Measured peak RSS |
|-------|---------------|-------------------|
| Columnar read of A1 (11 cols) + A2 (10 cols) | 0.4 s | 1.1 GB |
| Body-leg join + full-grid economics | 0.5 s | 1.4 GB |
| **Total transform** | **≈ 0.8 s** | **≈ 1.4 GB** |

Restricting A2 to `is_body` collapses 4,058,377 quote rows to 657,477 before the join, which is what keeps the join cheap. Writing the ~1.06 M-row output adds well under a second. Against 64 GB of installed memory, a ~1.4 GB working set needs no chunking, no partitioning, no out-of-core engine, and no parallelism — and adding any of those would be exactly the infrastructure over-build the reviewed sprint scope warns against.

**A note for D3/D4.** Because D2 emits the full 1,063,995-row grid, each per-window A4 feature file will carry roughly 1.06 M rows rather than only the ~294 k usable ones, across 281 windows. That is a consequence of requirement 9, not a defect, but D3 should budget for it.

D3 has exactly one safe trimming option, and the distinction matters enough to state precisely. It may trim **emitted dates after feature calculation**, provided it retains enough prior history to fill the windows. It must **not** filter tickers to a PIT universe before CVG runs: ticker membership is an input to the per-date cross-sectional median (`cvg_calculator.py:272`, `:482`), so removing tickers changes CVG values for every ticker that remains. CVG cross-sectional membership is a D1 decision and is not yet authorized, so "trim to the PIT universe" is not available to D3 as written. D2 does no trimming of either kind.

---

## 9. Determinism, artifact placement, and lineage

An artifact that cannot be tied to its input, or that differs between two runs of the same code on the same data, cannot support a go/no-go decision later. The repository already has a clear idiom for this — manifests and stage markers carrying ids, digests, and counts, written atomically — so D2 should follow it at a smaller scale rather than invent a new one.

**Determinism.** The transform reads immutable inputs, performs no sampling, no hashing-order-dependent iteration, no parallel reduction, and no wall-clock-dependent computation. Output rows are sorted ascending by `(ticker, entry_date)`, which is total because the key is unique. Column order is fixed by a module-level constant. All floats are `float64`; missing values are `NaN`. The frame is written with `index=False` and snappy compression, matching `save_features`.

Byte-level Parquet equality is *not* used as the determinism criterion, because writer version strings and compression internals can vary across environments. Instead the receipt records a **content digest** over the value content in canonical row order (a SHA-256 over `pd.util.hash_pandas_object(df, index=False)`), and the acceptance check is that two runs produce the same content digest. The file's `sha256_file` digest is also recorded, but as descriptive metadata rather than a gate.

**Placement.** The artifact goes to a new snapshot-scoped derived root, keyed by `snapshot_id` so that a different input can never overwrite it:

```text
C:/MomentumCVG_env/derived/e2c1f8fd44d72176/
  ├── straddle_observations_weekly.parquet
  └── straddle_observations_weekly.lineage.json
```

Three constraints shaped this. It cannot live inside the snapshot root, because Sprint 004 is immutable (requirement 1). It should not live in `C:/MomentumCVG_env/cache/`, because `repo_map.md` designates that as the mutable producer cache and the reviewed Sprint 005 definition of done rules out mutable cache stand-ins. And it must not live in the Git repository, because the repository holds no data artifacts. A snapshot-keyed derived root satisfies all three and gives D3/D4 a natural home for `derived/<snapshot_id>/features/features_{max}_{min}.parquet`, which `SurfaceDataPaths(features_dir=...)` can already consume without any code change — so D5's smoke test needs no new plumbing.

**Publication is atomic, and the receipt is the completion marker.** Both files are written to temporary paths in the destination directory and published with `os.replace`, following the pattern in `input_snapshot.write_manifest`. The Parquet file is published **first** and the receipt **last**, so a receipt at the canonical path always implies a complete artifact beside it. A crashed or killed run can leave a stray temporary file, but it can never leave a truncated Parquet at the canonical path, and it can never leave a receipt describing an artifact that was not fully written. Consumers that need to know whether an artifact is usable check for the receipt, not the Parquet.

**The canonical path must not acquire new semantics.** A directory keyed only by `snapshot_id` is stable across runs by design, which is exactly what makes it dangerous: a code change or a change to a frozen rule could leave different bytes at a path that D3, D4, and D5 treat as fixed. D2 therefore **refuses to publish** when a receipt already exists at the destination and records a different `transform_config_version` or content digest. There is no override flag. Before treating an identical rerun as a no-op, it also requires the canonical Parquet to exist and its SHA-256 to match the existing receipt; a missing or mutated artifact is reported as corruption, not silently repaired. A genuinely new transform version must use a different explicit output root, leaving the accepted artifact intact.

The alternative — automatically encoding transform identity into the path — was considered and rejected. It would make the location unpredictable for downstream consumers, who would then need discovery logic to find the current artifact. A deliberately supplied output root is sufficient for a later version; the canonical default stays predictable and immutable.

**Lineage receipt.** A small JSON sidecar, modeled on the manifest and stage-marker conventions.

| Group | Fields |
|-------|--------|
| Identity | `schema_version`, `artifact`, `created_at_utc`, `repo_sha` |
| Input | `snapshot_id`, `build_id`, `snapshot_root`, `manifest_path`, `manifest_overall_status`, `production_accepted` |
| Sources | For A1 and A2: manifest-relative path, absolute path, `sha256_file` digest, row count |
| Input key-set proof | `a1_key_count`, `a1_key_digest` (recomputed), `manifest_a1_key_digest`, and their agreement flag |
| Transform config (all frozen constants, not runtime options) | `direction="long"`, `fill="mid"`, `max_leg_spread_pct=0.99`, `entry_iv_rule="mean_body_call_put_iv"`, `vol_gap_rule="realized_volatility_minus_entry_iv"`, `spread_ineligible_volatility_rule="preserve"` (decision D-1), `return_pct_units="percentage_points"`, `volatility_units="annualized_decimal"`, `transform_config_version` |
| Output | `row_count`, `key_count`, `output_key_digest`, `content_digest`, `file_sha256`, `column_order` |
| Coverage | Row counts per `observation_status` and per `missing_reason`; non-null counts for `return_pct`, `entry_iv`, `realized_volatility`, `vol_gap` |

Recording the digest agreement explicitly, rather than only asserting it at runtime, means a later reader of the artifact can verify the lineage claim without re-running anything.

---

## 10. Proposed repository changes for the later implementation

The change set is intentionally small and additive. Nothing that Sprint 004 or Sprint 003 depends on is edited, so the existing 1,321-test baseline should be unaffected except by the new tests. Placement follows the existing split between reusable logic in `src/` and thin operator entry points in `scripts/`.

| Path | Status | Contents | Rationale |
|------|--------|----------|-----------|
| `src/features/straddle_observations.py` | **new** | Frozen transform constants, `SNAPSHOT_STRADDLE_SCHEMA_VERSION`, `OBSERVATION_COLUMNS`, `OBSERVATION_STATUSES`, `MISSING_REASONS`, `StraddleObservationStructuralError`, `resolve_surface_inputs(snapshot_root)`, `load_surface_frames(inputs)`, `join_body_legs(meta, quotes)`, `validate_structural_integrity(joined)`, `build_observations(joined)`, `validate_output_contract(observations, meta)`, `observation_coverage(df)` | Sits beside `straddle_analyzer.py` and the calculators; this module *produces* the `straddle_history` those consume. Pure functions over DataFrames keep it unit-testable without touching disk, matching `option_surface_contract.py`'s style |
| `scripts/build_straddle_observations.py` | **new** | CLI: `--snapshot-root` (required), `--output-root`, `--dry-run`. Resolves, loads, validates, builds, publishes Parquet then lineage JSON, prints coverage. **No flag selects economic behavior or overwrites a divergent canonical artifact** | Matches the existing script convention (`precompute_option_surface.py`, `build_features.py`) and keeps I/O and argument parsing out of the library. Explicitly **not** a `refresh_weekly_inputs.py` stage, per the reviewed scope |
| `tests/unit/test_straddle_observations.py` | **new** | Synthetic-fixture unit tests for construction, missingness, boundaries, and structural errors (§ 11) | Follows `test_option_surface_straddle.py`'s hand-checkable synthetic-surface pattern |
| `tests/unit/test_straddle_observations_compat.py` | **new** | Equivalence against `build_straddle_from_surface` and end-to-end runs through both production calculators | Keeps the oracle and compatibility tests separate from the unit semantics |
| `docs/surface_straddle_observation_transform_design.md` | this document | Design of record | — |
| `docs/README.md` | *follow-up* | One index row under "Active documents" | Repository convention; left to the implementation task so this design task changes no other documentation |
| `src/backtest/option_surface.py` | **unchanged** | — | Reused as the test oracle only; KB-001 is in the condor path and is out of scope |
| `src/features/momentum_calculator.py`, `cvg_calculator.py`, `base.py` | **unchanged** | — | Requirement 8 is satisfied by matching their existing contract, not by editing them |
| `scripts/build_features.py`, `scripts/precompute_straddle_history.py`, `scripts/refresh_weekly_inputs.py` | **unchanged** | — | Legacy or out-of-scope per the reviewed Sprint 005 boundary |
| `src/data/input_snapshot.py`, `snapshot_foundation.py`, `snapshot_orchestrator.py` | **unchanged** | — | Consumed read-only; Sprint 004 stays closed |

**Consequences of the frozen rule for the module surface.** There is no runtime economic-config object. Module constants define the one approved transform and are serialized into the receipt; the builder has no alternative economic branch. `OBSERVATION_COLUMNS` includes `call_iv` and `put_iv`. `validate_structural_integrity` covers only exact body-leg contradictions, while `validate_output_contract` enforces the key grid, statuses, and dependent-null rules after construction.

**Reuse without duplication.** `read_manifest` and `default_manifest_path` for manifest handling; `resolve_under_root` for path safety; `ticker_date_keys_digest` and `sha256_file` for lineage and the input identity guard; `FillAssumption`/`build_straddle_from_surface`/`settle` as the test oracle; the `write_manifest` temp-file-plus-`os.replace` pattern for publishing both files. Nothing in that list is reimplemented.

`check_meta_grain` and `check_a1_a2_join` are deliberately **not** called. An earlier revision reused them, but they recertify an already accepted snapshot (§ 7.3), and the one condition D2 genuinely needs from them — duplicate keys — is now caught by the output post-condition at a fraction of the cost.

---

## 11. Testing and objective acceptance plan

The tests are split into two layers for a specific reason: semantics are cheapest to pin on tiny synthetic surfaces where every expected number can be worked out by hand, while compatibility and lineage claims are only meaningful against the real artifact. The synthetic layer runs in the normal pytest suite; the artifact layer is a small set of one-off acceptance checks recorded as sprint evidence, so the everyday suite does not depend on a 261 MB external file.

### 11.1 Unit tests on synthetic fixtures

Fixtures follow `test_option_surface_straddle.py`: `body_strike = 100`, call `bid 2.00 / ask 2.40`, put `bid 1.80 / ask 2.20`, `exit_spot = 102`, giving `entry_cost = 4.20`, `exit_value = 2.00`, `pnl = −2.20`, `return_pct = −52.380952…`.

| # | Requirement | Test | Objective assertion |
|---|-------------|------|---------------------|
| T1 | Exact key preservation and uniqueness | `test_output_keys_match_a1_exactly` | Output key set equals the input A1 key set; `len(df) == len(meta)`; no duplicate `(ticker, entry_date)`; row order sorted |
| T2 | Literal valid straddle calculation | `test_literal_long_straddle_values` | `entry_cost == 4.20`, `exit_value == 2.00`, `pnl == −2.20`, `return_pct ≈ −52.380952`, `entry_iv == 0.20`, `vol_gap == 0.18 − 0.20 == −0.02`, all to `1e-12` |
| T3 | Return floor and sign | `test_return_pct_floor_and_units` | `exit_spot == body_strike` ⇒ `return_pct == −100.0`; a large move gives a positive value on the same scale as the fixture CSVs |
| T4 | Consistency with existing semantics | `test_matches_build_straddle_from_surface` | On every **synthetic** key, `abs(vec − oracle) < 1e-9` for `entry_cost`, `pnl`, and `return_pct`, where the oracle is `build_straddle_from_surface(direction="long", fill=FillAssumption.mid(), max_leg_spread_pct=0.99)` plus `.settle(exit_spot)`. Self-contained: no accepted-snapshot access, so the pytest suite stays independent of the external artifact. Real-key sampling lives in A6 |
| T5 | Invalid A1 rows | `test_surface_invalid_rows_are_rows_not_gaps` | One row per invalid key; `observation_status == "surface_invalid"`; `missing_reason` equals the A1 tag; body-leg and derived straddle fields are null; A1 passthrough fields retain their source values; the key is still present |
| T6 | Missing scheduled weeks | `test_scheduled_week_never_dropped_or_backfilled` | A ticker whose middle week is invalid keeps that week's row; the neighbouring weeks' values are unchanged; no value is copied forward from an older week |
| T7 | Spread boundary | `test_spread_threshold_boundary` | `spread_pct == 0.99` on both legs ⇒ `ok`; `0.990001` on either leg ⇒ `body_spread_ineligible` with `missing_reason == "body_spread_above_threshold"`; verified against the oracle raising `Missing tradeable body` |
| T8 | Quote eligibility and precedence | `test_unusable_quote_is_row_level_not_fatal` | A body leg with `bid <= 0`, `ask <= 0`, `NaN`, or `inf` produces `body_quote_unusable` / `body_quote_not_positive_finite`, with trade economics, `entry_iv`, and `vol_gap` null — **no exception**, and the key remains. This status wins even if spread is also unavailable. A null/non-finite `spread_pct` with usable quotes produces `body_spread_ineligible` / `body_spread_unavailable` |
| T8b | Unusable leg IV is not fatal | `test_unusable_leg_iv_preserves_return` | A body leg with null, zero, negative, `NaN`, or infinite `iv` on an otherwise eligible key ⇒ `observation_status == "ok"`, `return_pct` and `entry_cost` non-null, raw leg IVs retained, `entry_iv` and `vol_gap` null. Proves an unrelated unusable input cannot erase a Momentum return |
| T9 | Independent return information | `test_unusable_rv_preserves_return` | `realized_volatility` null, negative, `NaN`, or infinite with valid legs ⇒ `observation_status == "ok"`, `return_pct` non-null, `entry_iv` non-null, calculator-facing `realized_volatility` and `vol_gap` null |
| T10 | Independent volatility information | `test_spread_ineligible_preserves_volatility` | The frozen D-1 rule only: trade economics null; `entry_iv`, `realized_volatility`, and `vol_gap` non-null. **No companion test for the rejected alternative**, since no code path produces it |
| T11 | Missing body leg | `test_missing_body_leg_is_structural` | Valid A1 row with no body call (or no body put) in A2 raises, naming the key |
| T12 | Duplicate body leg | `test_duplicate_body_leg_is_structural` | Two body calls for one key raises; the transform never silently takes the first |
| T13 | Wrong strike | `test_body_leg_strike_mismatch_is_structural` | Body leg strike ≠ A1 `body_strike` raises |
| T14 | Wrong expiry | `test_body_leg_expiry_mismatch_is_structural` | Body leg expiry ≠ A1 `expiry_date` raises |
| T15 | Duplicate A1 key | `test_duplicate_a1_key_violates_key_postcondition` | Duplicated `(ticker, entry_date)` in A1 raises. The mechanism is the **output** key post-condition (one unique row per A1 key), not an input-side audit of A1 |
| T16 | Aggregated error report | `test_structural_errors_reported_together` | Several distinct violations in one input produce one error listing every category with counts and examples |
| T16b | Output contract | `test_output_status_and_dependent_null_contract` | Status/reason combinations, trade-economics availability, volatility-field availability, and finite-value rules match §§ 5 and 7; any violation aborts before publication |
| T17 | Momentum compatibility | `test_momentum_calculator_consumes_output` | `MomentumCalculator(windows=[(8,2)], min_periods=3)` runs on `FeatureDataContext(straddle_history=out)`; `calculate` and `calculate_bulk` agree; `mom_8_2_count` excludes null-return rows but the row positions still advance across them |
| T18 | CVG compatibility | `test_cvg_calculator_consumes_output` | `CVGCalculator(windows=[(8,2)], min_periods=3)` uses the emitted `vol_gap` column directly (`_resolve_vol_gap_col` returns it without deriving), and produces identical values when `vol_gap` is dropped and derived from the two components |
| T19 | Dtype contract | `test_entry_date_is_datetime64_and_index_unique` | Round-tripped through Parquet, `entry_date` is `datetime64[ns]`, `ticker` is upper-case string, the index is a unique `RangeIndex`, and `MomentumCalculator.calculate` (which does no dtype coercion) runs without raising |
| T20 | Determinism | `test_transform_is_deterministic` | Two runs on the same input produce identical content digests, row order, and column order |
| T21 | Lineage | `test_lineage_receipt_contents` | Receipt records `snapshot_id`, `build_id`, both source digests, the recomputed and manifest A1 key digests plus their agreement, the full transform config, and coverage counts that equal the emitted frame's |
| T22 | Input guards | `test_rejects_non_accepted_snapshot` | Missing manifest, `production_accepted` not true, non-empty `blocking_failures`, `overall_status == "FAIL"`, or an A1 key digest mismatch each abort before any write |
| T23 | Scope guard | `test_output_contains_no_feature_columns` | No column matches `mom_*`, `cvg_*`, `dvg_*`, `cgap_*`, or any ranking/eligibility name; no windowing parameter appears in the config |
| T24 | Frozen rules | `test_frozen_rules_are_recorded_and_not_overridable` | The receipt records the module's frozen economic constants; the CLI exposes no economic-policy flag and the builder accepts no economic-policy argument |
| T25 | Republish guard | `test_refuses_to_overwrite_divergent_artifact` | A different `transform_config_version` or content digest always raises and writes nothing; a missing/mutated Parquet beside an existing receipt also raises; only a receipt/file-consistent identical rerun is a no-op |
| T26 | Atomic publication | `test_publication_is_atomic_and_receipt_last` | A failure injected between the two writes leaves no receipt at the canonical path; no partial Parquet is ever published there |

Note that T4 and T17–T19 run entirely on synthetic fixtures, so the whole unit layer is independent of the 261 MB external artifact. Every real-data claim lives in § 11.2.

### 11.2 Acceptance checks against the accepted snapshot

Run once on `e2c1f8fd44d72176` and recorded as sprint evidence (commands, ids, results).

| # | Check | Objective pass condition |
|---|-------|--------------------------|
| A1 | Key grid | Output row count `== 1,063,995`; key set identical to A1; zero duplicates |
| A2 | Key digest | `ticker_date_keys_digest(output keys) == faa7e943e71b8aeaf4ea354713ab5558f44a03c9c211c6a68f53236acaa2cced` |
| A3 | Status coverage | Counts match § 7.1 exactly: 294,145 `ok`; 20,240 `body_spread_ineligible` (all `body_spread_above_threshold`); 0 `body_quote_unusable`; 749,610 `surface_invalid` split 422,340 / 293,130 / 33,654 / 486 |
| A4 | Economic coverage | `return_pct` non-null on 294,145 rows; `entry_iv` non-null on 314,385; `vol_gap` non-null on **313,946** (`293,707` from `ok` rows plus `20,239` from spread-ineligible rows, under the frozen D-1 rule) |
| A5 | Independent information | Exactly 438 rows are `ok` with a null `realized_volatility` and a non-null `return_pct`; exactly 1 spread-ineligible row has a null `realized_volatility` |
| A6 | Oracle equivalence | On a **fixed, deterministic** sample of ≥ 200 `ok` keys — an explicit key list, or a fixed seed recorded in the evidence — `abs(output − build_straddle_from_surface + settle) < 1e-9` for `entry_cost`, `pnl`, and `return_pct`. A newly random sample per run would make the evidence unreproducible |
| A7 | Units sanity | `return_pct` minimum `== −100.0`; `entry_iv > 0` where populated; `realized_volatility >= 0` where populated (**not** strictly positive — 132 valid rows sit at exactly `0.0`, per R-5); `vol_gap == realized_volatility − entry_iv` wherever both are present |
| A8 | Determinism | Two full runs produce the same content digest |
| A9 | Calculator smoke | A single window through both production calculators on the real artifact completes and yields non-null features for a ticker with dense history |
| A10 | No mutation | The snapshot root's file listing and the two surface artifacts' `sha256_file` digests are unchanged after the run |

Note that A3–A5 use exact counts. They are reproducible arithmetic over an immutable input, so an inexact threshold would be strictly weaker for no benefit — and any drift in those numbers means either the input or the semantics changed, which is exactly what the check should catch. The A4 and A5 figures were verified directly against the accepted snapshot at design time: 439 valid keys carry a null `realized_volatility`, of which 438 are `ok` and exactly 1 is spread-ineligible, which is why the populated `vol_gap` count is `313,946` rather than `293,707 + 20,240`.

---

## 12. Frozen decision, assumptions, and risks

Two categories are separated below because they carry different authority. The frozen decision was made by the strategy owner and changes economic values, so it is recorded separately and cannot be revisited by an implementer. Assumptions and risks are positions the design has taken from repository evidence, recorded so a later reader can challenge them without re-deriving the evidence. No unresolved D2 semantic decision remains.

### 12.1 Frozen decision

| # | Decision | Authority | Scale of impact | Rationale and counter-evidence |
|---|----------|-----------|-----------------|-------------------------------|
| **D-1** | On `body_spread_ineligible` keys, spread status does **not itself null** `entry_iv`, `realized_volatility`, or `vol_gap`; they remain populated wherever their components are usable, while trade economics are null | Strategy owner, 2026-08-01 | **20,240 of 314,385 usable ticker-weeks (6.44 %)** — and, because CVG's median is cross-sectional per date, it shifts CVG for *every* ticker on those dates, not only these rows | Requirement 7 read literally: both body legs are observable and their IVs are stored, so a vol gap is an observation rather than a claim that the straddle was enterable. **Counter-evidence:** the legacy `StraddleHistoryBuilder` nulls `entry_iv` and `iv_rv_spread` alongside the return when its leg filter rejects a strike, so D-1 departs from legacy precedent. Reversible in both directions because `call_iv` and `put_iv` are retained (§ 7.2) |

D-1 is frozen as a single canonical rule. The implementation must not provide a switch, and an implementer must not change it without a new owner decision and a new `transform_config_version`.

### 12.2 Assumptions and risks the design has taken a position on

| # | Item | Position and evidence |
|---|------|-----------------------|
| R-1 | `entry_iv` as the simple mean of the two body IVs | Follows `StraddleHistoryBuilder`. Currently inconsequential — the two body IVs are bit-identical on all 628,770 legs in this snapshot — but pinned explicitly and recorded in the receipt so a future per-side IV producer change cannot silently alter CVG |
| R-2 | `float64` instead of `Decimal` | Measured agreement with the production `Decimal` path is `≤ 5e-13` on `return_pct`; pinned by T4 (synthetic) and A6 (real keys) with a `1e-9` tolerance |
| R-3 | Only input-identity failures, A1↔A2 contradictions, and output-contract violations abort the run | The accepted snapshot has zero body-leg contradictions, and the output checks validate D2's own promises rather than recertifying A1/A2. Data-quality conditions within a leg (bad quote, unusable IV, missing spread, unusable RV) are row-level instead, so a single degraded quote in a future snapshot cannot block the pipeline (§ 7.3) |
| R-4 | `spot_move_pct` carries A1's percent units while the volatilities are decimals | An inherited asymmetry (`_metadata_success_row` multiplies by 100). D2 preserves A1's units rather than silently rescaling; documented in the schema. Nothing downstream currently reads it |
| R-5 | Extreme `realized_volatility` values in A1 (min 0.0, max 27.67) | Passed through unchanged. Winsorizing would be a new economic decision and belongs to D1 if anywhere |
| R-6 | The full-grid emit enlarges every A4 feature file to ~1.06 M rows across 281 windows | Direct consequence of requirement 9. Flagged for D3's resource planning; D2 must not pre-trim |
| R-7 | Mutable-cache confusion | `C:/MomentumCVG_env/cache/` holds same-named surface files at different sizes. Mitigated by manifest-only resolution plus the A1 key-set digest guard. The digest is not a full-content proof; source SHA-256 values are recorded for lineage and content trust rests on snapshot immutability |
| R-8 | New `derived/<snapshot_id>/` root | No prior convention exists for derived artifacts. Chosen because the snapshot root is immutable and `cache/` is designated mutable. `SurfaceDataPaths(features_dir=...)` already accepts an arbitrary directory, so D5 needs no code change |
| R-9 | Environment pinning | Measurements were taken under pandas 3.0.0 / numpy 2.4.1 / pyarrow 23.0.0. The `entry_date` dtype requirement (§ 5.1) is a pandas-3 behavior; the design is stricter than older pandas would require, which is safe in both directions |
| R-10 | KB-001 | Affects `build_ironcondor_from_surface` only. D2 uses the straddle path and touches no condor code |
| R-11 | D-1 departs from legacy `StraddleHistoryBuilder`, which nulls `entry_iv` and `iv_rv_spread` when its leg filter rejects a strike | Recorded so the divergence is visible to anyone comparing D2 output against a legacy straddle history. The reviewed Sprint 005 scope makes D2's artifact the source of truth; the legacy path is not |
| R-12 | Crossed body quotes are included | 62 body legs on 51 valid keys (0.016 %) have `ask < bid`, giving a negative `spread_pct` that trivially passes `<= 0.99` while the midpoint stays positive. `build_straddle_from_surface` includes them and requirement 5 forbids adding new filters, so D2 preserves existing behavior. Not an open question; flagged only because the data is genuinely odd. Exclusion would be a one-line predicate plus a new `missing_reason` if an owner later wants it |
| R-13 | No trimming to the manifest's feature-ready window (`2018-01-12` → `2026-07-10`) | Requirement 2 mandates the full A1 grid, which settles it: the 2,391 rows on the single 2018-01-05 entry date stay. Recorded so nobody later "fixes" the boundary. D3/D4 may trim emitted dates at their stage (§ 8) |
| R-14 | A1 passthroughs remain passthroughs on `surface_invalid` rows | Preserve usable A1 values, including `realized_volatility`, regardless of `surface_valid`; ignore partial A2 body rows and keep `entry_iv`/`vol_gap` null. This resolves the former O-1 inconsistency without changing Momentum or CVG, because invalid rows still have no calculator-facing gap |
| R-15 | Unusable body quotes null calculator-facing IV/gap | D-1 applies to wide-but-usable quotes, not non-positive/non-finite quotes. Raw leg IVs and usable A1 RV remain auditable, but `entry_iv`/`vol_gap` are null. The accepted snapshot has zero affected rows, so this freezes future behavior without changing current economics |

---

## 13. Dependency-aware implementation sequence

The order below front-loads the two things that would invalidate the most work if they turned out to be wrong — the input guard and the semantic oracle — and defers the full-history run until after the semantics are pinned by tests. Each step is independently verifiable, so a later task can stop at any boundary and still leave the repository green.

| Step | Work | Depends on | Done when |
|------|------|------------|-----------|
| **1** | `resolve_surface_inputs` + `load_surface_frames`: manifest resolution, acceptance guards, A1 key-set digest check | — (D-1 is already frozen) | T22 passes; running against the accepted snapshot resolves both artifacts and reproduces the A1 key digest |
| **2** | `join_body_legs`: valid-key, `is_body`-restricted A2 pivoted onto the A1 key grid | Step 1 | Join output has exactly one row per A1 key; partial A2 rows for invalid A1 keys are ignored |
| **3** | `validate_structural_integrity`: missing/duplicate body legs and strike/expiry disagreements, aggregated into one error | Step 2 | T11–T14 and T16 pass; the accepted snapshot reports zero violations in every category |
| **4** | `build_observations` + `validate_output_contract`: precedence, row-level availability, economics, volatility fields under D-1, and all key/status/null post-conditions | Steps 2–3 | T1–T3, T5–T10, T15–T16b, T23–T24 pass |
| **5** | Oracle equivalence against `build_straddle_from_surface` on synthetic fixtures | Step 4 | T4 passes within `1e-9`. Real-key sampling is deferred to step 9 |
| **6** | Calculator compatibility tests through both production calculators | Step 4 | T17–T19 pass; no adapter, rename, or unit conversion anywhere in the test |
| **7** | Publisher + lineage receipt (atomic Parquet then receipt, content digest, republish guard, coverage counts) | Steps 4–6 | T20–T21, T25–T26 pass |
| **8** | `scripts/build_straddle_observations.py` CLI | Step 7 | `--dry-run` prints resolved paths, digests, and projected coverage without writing; no flag alters an economic value |
| **9** | Full-history run on `e2c1f8fd44d72176`; record acceptance evidence including the fixed A6 key sample | Step 8 | A1–A10 pass; the artifact and receipt exist under `derived/e2c1f8fd44d72176/` |
| **10** | Add the `docs/README.md` index row; brief sprint memo entry | Step 9 | Doc index lists this design; evidence recorded per `AGENTS.md` § Definition of done |

Steps 1–8 run entirely on synthetic fixtures and need no access to the external snapshot, so the normal pytest suite stays self-contained. Only step 1 (a smoke resolution) and step 9 read the real artifacts.

The former design step 0 — obtaining the volatility-rule decision — is complete: D-1 is frozen in § 12.1. Repository work may begin at step 1 only after `docs/agenda/current_sprint.md` is committed as accepted Build mode.

---

## 14. Evidence-to-design mapping

Every substantive choice above traces to something read or measured, and this table is the index. It is meant to be used adversarially: if a decision looks wrong later, the row shows exactly what would have to change for the decision to change.

| # | Design decision | Supporting evidence | Kind |
|---|-----------------|---------------------|------|
| 1 | Resolve A1/A2 only through the published manifest | `input_snapshot.read_manifest`, `resolve_under_root`; identically named files exist in the mutable `cache/` at different sizes | Confirmed code + measurement |
| 2 | Guard on `production_accepted`, `blocking_failures == []`, `overall_status ∈ {PASS, WARN}` | Accepted manifest is `WARN` with `blocking_failures: []` and `production_accepted: true`; a `PASS`-only guard would reject the accepted input | Measurement |
| 3 | Verify the recomputed A1 key digest against `params.surface_actual_a1_key_digest` | `snapshot_stage_adapters` / `_validate_surface_evidence` construct it as `ticker_date_keys_digest((date, ticker))`; recomputation reproduces `faa7e94…` exactly | Confirmed code + measurement |
| 4 | Emit one row per A1 key, full 1,063,995-row grid | A1 is an exact 2,391 × 445 cross product with zero duplicates; row-positional lookbacks in both calculators; `check_expected_meta_keys` treats the key set as the acceptance unit | Measurement + confirmed code |
| 5 | Keep unavailable weeks as null-economics rows rather than dropping them | `tests/fixtures/sample_straddle_history.csv` and `sample_vol_gap_history_rv_iv.csv` already use exactly this shape; the momentum test module documents "ALL dates present … NaN for weeks without trades" | Confirmed test fixture |
| 6 | Use only `surface_valid == True` rows and only their `is_body` legs | Producer invariant plus `check_surface_valid_invariant`; `OptionSurfaceDB.get_metadata` raises on invalid; closeout § 5.4–5.5; 29,166 invalid rows carry partial quotes that must not be harvested | Confirmed code + measurement |
| 7 | No nearest-strike / nearest-expiry / nearest-week fallback | Requirement 4; measured 0 strike and 0 expiry disagreements, so a fallback would only ever fire on corrupt input | Requirement + measurement |
| 8 | Long call + long put at `body_strike`, quantity `+1` each | `build_straddle_from_surface(direction="long")`; `v1_spec_pins.md` long-structure pin | Confirmed code + pin |
| 9 | Price from `bid`/`ask` midpoint, never the stored `mid` column | `_mid_entry_cost` docstring; 20 body legs differ from `(bid+ask)/2` by up to 0.0078 | Confirmed code + measurement |
| 10 | Per-leg `spread_pct <= 0.99` on both body legs | `precompute_straddle_history.MAX_SPREAD_PCT = 0.99`; `build_straddle_from_surface` filters before selection and raises `Missing tradeable body`; stored-mid and computed-mid denominators agree on 0 of 314,385 keys | Confirmed code + measurement |
| 11 | No volume, OI, or minimum-bid requirement | Legacy sets `MIN_VOLUME = MIN_OI = 0`; surface producer filters neither; 64,496 body legs have zero volume; adding a rule would drop tradeable weeks | Confirmed code + measurement |
| 12 | `return_pct = pnl / abs(entry_cost) × 100` | `Position.pnl_pct`; `StraddleHistoryBuilder` multiplies by 100; fixture magnitudes confirm percentage points | Confirmed code + fixture |
| 13 | `vol_gap = realized_volatility − entry_iv`, emitted explicitly | `CVGCalculator._resolve_vol_gap_col`; Gan and Nguyen define the paper's primitive gap in the same direction; legacy `iv_rv_spread` has the opposite sign (C-1) | Confirmed code + paper + reviewed scope |
| 14 | `entry_iv` = mean of the two body IVs | `StraddleHistoryBuilder`; the two IVs are identical on all 628,770 legs, so the rule is safe today but must be pinned | Confirmed code + measurement |
| 15 | `entry_date` stored as `datetime64[ns]` | A1's `date32` loads as object-of-`datetime.date`; `MomentumCalculator.calculate` performs no coercion and raises `TypeError: Cannot compare Timestamp with datetime.date` | Reproduced failure |
| 16 | Unique `RangeIndex`, written with `index=False` | `ticker_data.index.get_loc(...)` in `calculate`; index-aligned assignment in `calculate_bulk`; `save_features(index=False)` | Confirmed code |
| 17 | Vectorized whole-grid transform, not per-key builder calls | 329 ms/key → ≈ 28.8 h versus ≈ 0.8 s vectorized, at ≈ 1.4 GB peak | Measurement |
| 18 | Reuse `build_straddle_from_surface` as the test oracle only | Same measurement; sampled agreement `≤ 5e-13` makes the oracle a sound reference without being the engine | Measurement |
| 19 | Only input-identity failures, A1↔A2 contradictions, and D2 output-contract violations abort; leg-level data quality is row-level | Requirement 6 scopes structural errors to inconsistencies *between A1 and A2*; measured 0 bad quotes, 0 null leg IVs, and 0 null `spread_pct` today, so row-level handling changes nothing now and governs future snapshots | Requirement + measurement |
| 20 | Do not call `check_meta_grain` / `check_a1_a2_join`; catch duplicates via the output key post-condition | Sprint 004 already certified A1/A2 and the manifest publishes that acceptance; `check_a1_a2_join` would rescan 4.06 M quote rows to reconfirm it | Confirmed code + requirement |
| 21 | Preserve `return_pct` when `realized_volatility` is missing | 439 valid keys (438 `ok`, 1 spread-ineligible) have null RV with fully valid body legs | Measurement + requirement 7 |
| 22 | Populate volatility fields on spread-ineligible rows (**D-1**) | 20,240 keys retain positive finite body quotes with stored IVs; requirement 7 preserves independently usable information. Frozen by the strategy owner, not chosen by this design. Retaining `call_iv`/`put_iv` makes the decision reversible in both directions, so reversibility is *not* the argument for it | Owner decision + measurement + requirement 7 |
| 23 | Snapshot-keyed `derived/<snapshot_id>/` output root | Snapshot root immutable (requirement 1); `repo_map.md` marks `cache/` mutable; Sprint 005 DoD rules out mutable stand-ins; `SurfaceDataPaths(features_dir=...)` accepts any directory | Requirement + confirmed code |
| 24 | Content digest as the determinism gate, file digest as metadata | Parquet writer metadata (`created_by parquet-cpp-arrow version 23.0.0`) is environment-dependent | Measurement |
| 25 | Lineage receipt modeled on the manifest / marker idiom, written atomically | `input_snapshot.write_manifest` temp-file plus `os.replace`; snapshot markers carry evidence dicts with counts and digests | Confirmed code |
| 26 | No windows, minimum histories, CVG membership, ranking, or PIT filtering in D2 | Reviewed D1/D2 split; `SurfaceRunner` reads A4, never a straddle history | Requirement + confirmed code |
| 27 | Retain `call_iv` and `put_iv` in the output schema | `entry_iv` is a derived mean; without both inputs no reader could verify it, and D-1's populated values would be unauditable | Confirmed code + requirement |
| 28 | Publish Parquet atomically with the receipt last, and refuse divergent republish with no override | A stable snapshot-keyed path is predictable for D3–D5 but can otherwise acquire new semantics after a code change; `input_snapshot.write_manifest` establishes the temp-plus-`os.replace` idiom | Confirmed code + requirement 11 |
| 29 | D3 may trim emitted dates but not tickers before CVG | `cvg_calculator.py:272` and `:482` subtract a per-date cross-sectional median, so ticker membership changes CVG for the tickers that remain | Confirmed code |
| 30 | Preserve usable A1 passthroughs on invalid rows | A1 is the source contract; partial A2 rows remain ignored, and null `entry_iv` keeps `vol_gap` unavailable, so preservation improves auditability without changing calculators | Requirement 7 + confirmed code |
| 31 | Freeze deterministic status precedence and unusable-component rules | Several row-level conditions can overlap in future inputs; explicit precedence and finite-value rules prevent branch-order-dependent output. Accepted-snapshot counts are zero for these conditions | Requirement + measurement |

---

## Change log

| Date | Change |
|------|--------|
| 2026-07-26 | Initial design written for Sprint 005 D2 at commit `236c7991912d45d3125bd32428ec8ace8dd78535`, reviewed against accepted snapshot `e2c1f8fd44d72176` |
| 2026-08-01 | **Rev 2 — design review applied.** Q-1 frozen by the strategy owner as decision **D-1** (populate volatility fields on spread-ineligible rows) and the runtime policy switch removed, so the builder carries one canonical economic behavior (§ 7.2, § 10, T24). Added `call_iv`/`put_iv` to the output schema, making every derived field auditable and D-1 reversible in both directions (§ 5.2). Reclassified non-positive/non-finite quotes, null `spread_pct`, and null leg IV from run-fatal to row-level, with two new status values and three new missingness reasons (§ 7.1, § 7.3, T8, T8b). Reduced input validation to trust the Sprint 004 certification: dropped `check_meta_grain`, `check_a1_a2_join`, failure-vocabulary re-checks, settlement-field certification, weekly-date checks, and orphan-A2 reporting; retained the A1 key-digest identity guard and moved duplicate-key detection to an output post-condition (§ 7.3, T15). Made Parquet publication atomic with the receipt as completion marker, and added a divergent-republish guard in place of encoding config in the path (§ 9, T25, T26). Corrected A4 to 313,946 and relabeled it as the populated variant; made the A6 real-key sample deterministic; relaxed A7 to `realized_volatility >= 0` after confirming 132 valid rows at exactly `0.0` (§ 11.2). Removed PIT-universe trimming from the D3 note, since ticker membership changes CVG's cross-sectional median (§ 8). Fixed a broken reference to a nonexistent Q-5 (§ 6). Retired Q-2 and Q-3 into recorded assumptions R-12 and R-13. Added open item **O-1** on inconsistent passthrough treatment of `realized_volatility` on invalid rows (33,651 rows affected, no Sprint 005 impact). All counts re-verified read-only against the accepted snapshot |
| 2026-08-01 | **Rev 3 — implementation-readiness pass.** Preserved owner decision D-1, but resolved the remaining hidden choices: deterministic status precedence; unusable leg-IV/RV rules; calculator-facing nulls for unusable quotes; and preservation of usable A1 passthroughs on invalid rows, closing O-1 (§§ 5.2, 7, 12). Corrected the stale flow diagram so bad quotes and settlement recertification are no longer shown as structural checks; added explicit output key/status/null post-conditions. Removed the divergent-republish override and the unnecessary runtime config object, leaving one frozen transform and an immutable canonical artifact (§§ 9–11). Clarified that the A1 digest proves the accepted key grid rather than full-file content, tied the `vol_gap` sign to both `CVGCalculator` and Gan–Nguyen, reordered the evidence map, and made the repository's still-uncommitted Sprint 005 Build-mode gate explicit. No accepted-snapshot counts or D-1 economics changed. |
