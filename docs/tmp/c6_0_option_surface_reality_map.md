# C6.0 — Option Surface Reality Map and Delivery Definition

**Sprint:** 004 · **Task:** C6.0 (read-only reconnaissance)  
**Date:** 2026-07-04  
**Mode:** Audit / delivery definition only — no code, tests, cache, or data changes

---

## Scope and non-goals

### In scope (this document)

- Map producer (`precompute_option_surface.py`, `option_surface_analyzer.py`), consumer (`OptionSurfaceDB`, S3 assembly, `SurfaceDataPaths`), tests, and on-disk cache artifacts.
- Answer: **what C6 must deliver** so A1/A2 artifacts are trustworthy enough for later `SurfaceRunner` / step3 strategy assembly.
- Separate **schema validity** from **assembly-readiness**.
- Assess whether a **producer-safety patch (C6.1A)** is required before sample regeneration.

### Explicit non-goals (honored)

- No C6 implementation, design plan, code edits, test edits, production data edits, cache overwrites, full historical precompute, strategy backtests, or `SurfaceRunner` runs on real data.
- Did not modify `refresh_weekly_inputs.py`.

### What C6 must *not* be mistaken for

Per [current_sprint.md](../agenda/current_sprint.md) and [surface_engine_evaluation_plan.md](../surface_engine_evaluation_plan.md):

| Topic | Sprint 004 C6 | Deferred |
|-------|---------------|----------|
| A1/A2 schema + join + settlement-field audit on real cache | **In scope** | — |
| `surface-audit` PASS/WARN/FAIL report | **In scope** | — |
| T1–T7 pytest + ≥1 substantive sample audit run | **In scope** | — |
| Strategy profitability / Sharpe | **Out** | Sprint 006+ |
| L4/L5 real-data backtest smoke | **Out** | Sprint 006+ |
| A4 / features trust | **Out** | Sprint 005 |
| `refresh --dry-run` surface wiring | **Out** | C8 |
| `validate` umbrella inventory | **Out** | C3 (after C4–C8) |
| Full-universe surface rebuild via CLI | **Stretch / not required** | — |
| Incremental surface append / watermarks | **Out** | Sprint 005 |

**Rule:** Real-data validation in 004 means **input artifacts and precompute correctness**, not strategy evidence.

---

## Source-of-truth docs read

| Document | C6-relevant extracts |
|----------|---------------------|
| [current_sprint.md](../agenda/current_sprint.md) | Closeout blocker #9: surface precompute audit + tests. T1–T7 spec, `surface-audit` report sections, Blocks-005 B4 (settlement/join FAIL). C6 commit scope: extend contract/unit + audit module. C5 closed; C6–C9 remain. |
| [surface_engine_data_contract.md](../surface_engine_data_contract.md) | A1/A2 schemas, invariants (`surface_valid`, join keys, OTM delta window). S3 consumes meta+quotes; invalid meta excluded. Status `built` with contract test on producer row helpers. |
| [surface_engine_data_flow.md](../surface_engine_data_flow.md) | A1/A2 → S3; structural success ≠ decision-quality. Stage A marked 📦 given but 004 re-audits on real cache. |
| [surface_engine_evaluation_plan.md](../surface_engine_evaluation_plan.md) | L1 contract for IN (precompute); L4/L5 deferred. Component matrix pins `test_precompute_input_contract.py`. |
| [repo_map.md](../repo_map.md) | Data flow: `adjusted_liquid` → spot + surface cache. External paths. |
| [004_c5_adjusted_liquid.md](../sprint_memos/004_c5_adjusted_liquid.md) | Production root `C:/MomentumCVG_env/input/adjusted_liquid`. **Spot/surface full re-precompute not part of C5 closeout** — cache may predate C5 wiring. |

### A1 meta contract (required + diagnostic)

Required: `ticker`, `entry_date`, `expiry_date`, `dte_actual`, `entry_spot`, `exit_spot`, `body_strike`, `surface_valid`, `failure_reason`.  
Diagnostic (same key set on success/failure): `frequency`, `dte_target`, `spot_move_pct`, `realized_volatility`, `has_body_call`, `has_body_put`, `n_surface_quotes`, `processing_time`.

**Invariant (contract + sprint T1):**  
`surface_valid == (has_body_call AND has_body_put AND n_surface_quotes > 0)`.

### A2 quotes contract (required + diagnostic)

Required join/context: `ticker`, `entry_date`, `expiry_date`, `entry_spot`, `body_strike`, `side`, `is_body`, `is_otm`, `strike`, `bid`, `ask`, `mid`, `spread_pct`, greeks, `volume`, `open_interest`.  
Diagnostic: distance/moneyness/spread/bucket fields.

### T1–T7 checklist (from sprint spec)

| ID | Requirement |
|----|-------------|
| T1 | A1 invariant on producer rows |
| T2 | Quote ↔ meta join integrity |
| T3 | No duplicate quote grain |
| T4 | Trade dates align with `--as-of` / Friday resolver |
| T5 | Valid rows: settlement fields non-null; `dte_actual` correct |
| T6 | `failure_reason` ∈ documented vocabulary when `surface_valid == False` |
| T7 | Real-cache sample passes schema + consistency checks |

### `surface-audit` report sections (sprint spec)

Artifact inventory · Schema · Validity rate · Failure breakdown · Join integrity · Settlement readiness · Date alignment · v1 weekly DTE (WARN if far from 7).

### Blocks Sprint 005 (B4 surface)

FAIL if: settlement fields null on `surface_valid` rows, or broken meta/quotes join in sample. Low validity rate → WARN OK.

---

## Producer code map

### `scripts/precompute_option_surface.py`

#### CLI arguments

| Argument | Default | Notes |
|----------|---------|-------|
| `--data-root` | `DEFAULT_ADJUSTED_LIQUID_ROOT` (`C:/MomentumCVG_env/input/adjusted_liquid`) | **Reads C5 root correctly when run with defaults** |
| `--start-year` | `2018` | Inclusive |
| `--end-year` | `2026` | Inclusive; end calendar fixed to **Feb 20** of end year |
| `--frequency` | `monthly` | `weekly` \| `monthly` |
| `--workers` | `26` | joblib `Parallel`, backend `loky` |
| `--min-abs-delta` | `0.03` | OTM wing filter |
| `--max-abs-delta` | `0.45` | OTM wing filter |
| `--delta-buckets` | comma-separated grid | Parsed to sorted floats |
| `--keep-zero-bid-quotes` | off | Flag |

**Missing flags (C6.1A gap):** `--output-root`, `--tickers`, `--start-date`/`--end-date`, `--dry-run`, overwrite protection.

#### Hardcoded paths

| Purpose | Path |
|---------|------|
| Ticker universe | `C:/MomentumCVG_env/cache/liquid_tickers.csv` (column `Ticker`) |
| Spot DB | `C:/MomentumCVG_env/cache/spot_prices_adjusted.parquet` |
| Output root | `C:/MomentumCVG_env/cache` (always) |
| Log file | `C:/MomentumCVG_env/log/precompute_option_surface.log` (append) |

#### Output filenames

```text
option_surface_meta_{frequency}_{start_year}_{end_year}.parquet
option_surface_quotes_{frequency}_{start_year}_{end_year}.parquet
```

Example (matches consumer defaults): `option_surface_meta_weekly_2018_2026.parquet`.

#### Frequency / DTE behavior

- `weekly` → `dte_target = 7`; all resolved trading Fridays kept.
- `monthly` → `dte_target = 30`; first Friday per calendar month.
- **Default CLI frequency is `monthly`** but v1 consumer defaults expect **weekly** filenames — operator must pass `--frequency weekly`.

#### Trade-date generation

1. `get_trading_fridays(start, end, data_root)`: calendar Fridays in range; for each week walk Fri→Mon until `ORATS_SMV_Strikes_{YYYYMMDD}.parquet` exists under `data_root`.
2. `sample_fridays_by_frequency`: weekly = all; monthly = first Friday per month.
3. End date hardcoded: `datetime(end_year, 2, 20)` (not Dec 31).

**Alignment note:** Logic mirrors file-existence holiday fallback but lives **only in this script**. `src/data/trading_day.py` uses a different API (`resolve_as_of_trading_day` walk-back from `--as-of`) and explicitly does **not** import `get_trading_fridays` — T4 cross-check requires a shared spec or dual tests (C6.2).

#### Parallelization

One joblib job **per trade date**; each worker loads `SpotPriceDB`, constructs `OptionSurfaceBuilder`, runs **all tickers** for that date. Good ORATS LRU cache locality; bad for small ticker samples without a patch.

#### Safe small sample without overwriting cache?

| Approach | Overwrites `weekly_2018_2026`? | Practical? |
|----------|-------------------------------|------------|
| Re-run same year range + frequency | **Yes — full replace** | **Unsafe** |
| `--start-year 2026 --end-year 2026` | No (different filename) | Still all ~2473 tickers; long run; shared log |
| `--output-root` to `cache_c6_smoke/` | N/A | **Not supported without patch** |

**Verdict:** Not operationally safe for C6 smoke without **C6.1A**.

#### C5 adjusted-liquid root

Default `--data-root` points at `adjusted_liquid` via `paths.py`. **Existing cache artifacts (mtime 2026-04-23) likely predate C5 production backfill (2026-07-04)** — lineage must be treated as unknown/stale until regenerated or manifest-tagged.

---

### `src/features/option_surface_analyzer.py`

#### `_metadata_failure_row`

| Aspect | Detail |
|--------|--------|
| Inputs | `ticker`, `entry_date`, `dte_target`, `frequency`, `failure_reason`, `processing_time` |
| Output | Full A1 column set; numerics `None`/`False`/`0`; `surface_valid=False` |
| Failure reasons used | `no_spot_price`, `no_expiry_found`, `no_options_at_entry`, `no_strikes_in_chain`, `no_spot_at_expiry` |
| Invariants enforced | Same keys as success row |
| Gaps | Does not encode *why* body legs failed when partial chain exists |

#### `_metadata_success_row`

| Aspect | Detail |
|--------|--------|
| Inputs | Success path fields incl. body flags, `n_surface_quotes` |
| Output | `surface_valid = has_body_call AND has_body_put AND n_surface_quotes > 0`; **`failure_reason = None` always** |
| Invariants enforced | `dte_actual = (expiry - entry).days`; `spot_move_pct` stored as percent (×100) |
| **Gap (T6)** | Rows with `surface_valid=False` still have `failure_reason=None` (~25.7k rows in real cache) |
| Implied not enforced | No tag like `missing_body_legs` / `insufficient_quotes` |

#### `_nearest_bucket`

- Empty buckets → `None`; else min absolute distance (sorted buckets → smaller wins ties).
- Body quotes get `nearest_delta_bucket=None`.

#### `OptionSurfaceBuilder.__init__`

- Validates `frequency ∈ {monthly, weekly}`, delta bounds ∈ [0,1].
- Lazy `ORATSDataProvider` with **permissive** filters (`min_volume=0`, `max_spread_pct=999`).
- `dte_target` drives expiry path (≥28 → monthly logic), not `frequency` alone — must stay in sync with script.

#### `_find_best_expiry`

| Path | Behavior |
|------|----------|
| Monthly (`target_dte ≥ 28`) | Next calendar month Thu/Fri expiry; fallback closest within 4 days |
| Weekly (`target_dte < 28`) | Forward expiries within ±4 DTE; prefer Fri, then Thu, then closest |
| Failure modes | Returns `None` → `no_expiry_found`; **any exception swallowed** → logged error, `None` |
| Test before trust | Holiday weeks, symbols with only Wed expiries, `target_dte`/frequency mismatch |

#### `_quote_rows`

| Rule | Detail |
|------|--------|
| Include | Body at `body_strike`; OTM calls above / puts below body |
| Exclude | ITM; OTM outside `[min_abs_delta, max_abs_delta]`; zero bid/ask/mid unless `keep_zero_bid_quotes` |
| Output | Full A2 superset incl. `spread_pct = (ask-bid)/mid`, `is_otm = not is_body` |
| Implied | `is_body XOR is_otm` (enforced in output) |
| Exceptions | None swallowed |

#### `process_single_entry`

Linear pipeline: entry spot → expiry → chain → body strike (min distance, tie lower strike) → exit spot → quote rows → metadata.

| Stage | `failure_reason` |
|-------|------------------|
| No entry spot | `no_spot_price` |
| No expiry | `no_expiry_found` |
| Empty chain | `no_options_at_entry` |
| No strikes | `no_strikes_in_chain` |
| No exit spot | `no_spot_at_expiry` |
| Partial success | **`None`** (if body/quotes fail validity gate) |

**Body leg flags:** require `bid>0`, `ask>0`, `mid>0` on raw chain quotes — stricter than `_quote_rows` inclusion (quotes can exist in table while flags false).

**Behavior to test before trusting real artifacts:** expiry selection edge cases; success-path invalid rows with null `failure_reason`; body-strike tie logic; exit-spot requirement failing entire row even when quotes usable for entry-only research.

---

## Consumer / assembly code map

### `OptionSurfaceDB` (`src/backtest/option_surface.py`)

| Behavior | Detail |
|----------|--------|
| Load | `pd.read_parquet` both files; copies DataFrames |
| Date normalization | `entry_date` / `expiry_date` → datetime64; adds `entry_date_key` as Python `date` |
| Lookup keys | `(ticker, entry_date)` via `entry_date_key` |
| Duplicate meta | **Not detected** — `get_metadata` uses `.iloc[0]` on mask |
| Duplicate quotes | **Not detected** — returns all matching rows |
| Quote/meta coupling | Quotes **not** required to have matching meta for load; `get_quotes`/`get_metadata` raise independently |
| Invalid meta | `get_metadata` raises `ValueError` if `surface_valid=False` |
| Expiry match | **Not enforced** between meta and quote rows |

### Strategy assembly functions

#### `build_straddle_from_surface`

- Requires valid meta; body call + put in `quotes[is_body]` filtered by `side`.
- Uses first matching body row per side (`.iloc[0]`) — **duplicate body rows silently pick first**.
- Assumes `spread_pct` present for optional liquidity filter.
- Fill via `FillAssumption`; mid cost recomputed from bid/ask (not stored mid).

#### `build_ironfly_from_surface`

- Body call/put required; OTM wings via `_choose_below_nearest` on `is_otm` rows (`abs_delta ≤ wing_target`).
- **Requires at least one OTM call and one OTM put** in quotes — fails at assembly if missing (even when meta `surface_valid=True`).
- Wing width = max(call wing, put wing) distance from body.

#### `build_ironcondor_from_surface`

- Short legs from OTM **or body** (`is_otm | is_body`); long legs must be further OTM than shorts.
- `_choose_nearest` / `_choose_below_nearest` on `abs_delta`.
- KB-001: `body_credit_per_share` uses short legs not ATM body (documented drift).

### `step3_get_eligible_structures` (`pipeline.py`)

| Phase | Error surfacing |
|-------|-----------------|
| Metadata | `failure_reason = f"metadata_error:{exc}"` (KeyError or invalid surface) |
| Assembly | `failure_reason = str(exc)` (e.g. missing wings, spread filters) |
| Success | `structure_ok=True`, `_assembly` present; overwrites meta spot/strike with assembly fields |

**Fields needed for S5/S8 on success:** `structure_ok`, `max_loss_per_share`, `net_credit_per_share`, `pnl` path via `_assembly.settle(exit_spot)`, `included_in_portfolio` downstream, cycle metrics from `pnl_total` / `capital_at_risk_dollars`.

**C6 audit now:** meta validity, join, settlement fields, quote grain, wing availability for configured deltas.  
**Remains later:** earnings (S4), portfolio selection (S5), go/no-go (S6+).

### `SurfaceDataPaths` vs producer output

| Consumer default | Producer writes (when run weekly 2018–2026) |
|------------------|---------------------------------------------|
| `option_surface_meta_weekly_2018_2026.parquet` | **Same pattern** ✓ |
| `option_surface_quotes_weekly_2018_2026.parquet` | **Same pattern** ✓ |
| `cache_dir = C:\MomentumCVG_env\cache` | Hardcoded same root ✓ |

**Mismatch risk:** producer **default frequency is monthly** → would write `option_surface_meta_monthly_2018_2026.parquet`, breaking consumer defaults.

---

## Existing tests and gaps

| Requirement | Existing coverage | Test file | Gap | C6 action |
|-------------|-------------------|-----------|-----|-----------|
| A1 required columns on producer rows | Producer row helpers | `tests/contract/test_precompute_input_contract.py` | No `_quote_rows` / full builder integration | Extend in C6.2 |
| A1 `surface_valid` ⇔ body legs + quotes (T1) | Partial — only `has_body_put=False` case on success helper | same | Not testing `n_surface_quotes=0`; not on real builder output | C6.2 synthetic |
| A1/A2 join (T2) | Synthetic `OptionSurfaceDB` in conftest | `tests/contract/conftest.py`, step3 contract | No parquet join test; no orphan detection | C6.2 + C6.3 audit |
| Quote grain uniqueness (T3) | None | — | **Missing** | C6.2 fixture + C6.3 FAIL rule |
| Trade date / `--as-of` alignment (T4) | `trading_day` unit tests (C2) | `tests/unit/test_refresh_weekly_inputs_cli.py` (partial) | **No test for `get_trading_fridays` / `generate_trade_dates`** | C6.2 |
| Settlement fields + dte (T5) | Contract doc only | — | **Missing** on producer + cache | C6.3 audit |
| Failure vocabulary (T6) | Failure row tags exist | contract test | **Success-path invalid rows with `failure_reason=None` untested** | C6.2 + producer fix optional |
| Real-cache sample (T7) | None | — | **Missing** | C6.4 report |
| `OptionSurfaceDB` load/lookup | Synthetic only | unit + contract fixtures | No duplicate detection; no real parquet | C6.3 |
| `build_straddle/ironfly/ironcondor` | Synthetic L2 goldens | `tests/unit/test_option_surface_*.py` | Handmade fixtures; not producer-shaped | Keep; add assembly-readiness audit |
| `step3_get_eligible_structures` | Contract L1+L2 | `tests/contract/test_step3_structures_contract.py` | Synthetic DB | C6.3 cross-check rules |
| `SurfaceRunner` ORCH | Synthetic parquets | `tests/unit/test_surface_runner_data_flow.py`, orchestration contract | **Not real cache** | Out of scope (Sprint 006) |
| `precompute_option_surface.py` CLI | None | — | **Missing** | C6.1A + optional smoke test |
| `surface-audit` CLI | Stub | `refresh_weekly_inputs.py` | **Not implemented** | C6.3/C6.5 |

**Coverage types:**

- **Synthetic runner tests** — prove S1→S8 plumbing with handmade parquets; do not prove producer artifacts.
- **Producer-row unit tests** — pin `_metadata_*_row` keys and one invariant.
- **Real-cache artifact tests** — **absent** (T7 gap).
- **Missing** — join/grain/expiry/body-uniqueness on real data; producer CLI safety; trade-date parity.

---

## Local artifact inventory

**Root scanned (read-only):** `C:/MomentumCVG_env/cache`

### Meta / quotes pairs found

| Meta | Quotes | Size (meta / quotes) | Rows (meta / quotes) |
|------|--------|----------------------|----------------------|
| `option_surface_meta_weekly_2018_2026.parquet` | `option_surface_quotes_weekly_2018_2026.parquet` | 16.3 MB / 290.4 MB | 1,051,025 / 4,671,879 |

**No other `option_surface_meta_*.parquet` / `option_surface_quotes_*.parquet` pairs** in cache.

**Last modified:** 2026-04-23 (before C5 closeout 2026-07-04) — treat as **stale lineage** relative to `adjusted_liquid` production root unless proven otherwise.

### Other related artifacts

| File | Present | Notes |
|------|---------|-------|
| `spot_prices_adjusted.parquet` | Yes (71 MB) | Producer hard dependency |
| `ticker_liquidity_panel.parquet` | Yes (9.8 MB; 548k rows, 9259 tickers) | S1 consumer |
| `liquid_tickers.csv` | Yes (19 KB) | Producer universe (2473 tickers in surface meta) |
| `features/*.parquet` | Yes (281 files) | A4 — Sprint 005 scope |

### Weekly pair — summary statistics

*(Full quotes scan used column projection only; meta fully loaded. No files written.)*

| Metric | Value |
|--------|-------|
| Meta date range | 2018-01-05 → 2026-02-20 |
| Quotes entry_date range | 2018-01-05 → 2026-02-13 |
| Unique tickers (meta / quotes) | 2473 / 2452 |
| `frequency` | 100% `weekly` |
| `dte_target` | 100% `7` |
| `surface_valid` | 348,639 (**33.17%**) |
| Duplicate meta `(ticker, entry_date)` | **0** |
| Duplicate quote grain `(ticker, entry_date, expiry_date, strike, side)` | **9,682** rows involved |
| Orphan quote keys | **0** |
| Valid meta with no quotes | **0** |
| Valid meta null `exit_spot` / `expiry_date` / `body_strike` / `entry_spot` | **0** |
| Valid `dte_actual` mismatch | **0** |
| Valid rows ≠ exactly 1 body call + 1 body put | **57** |
| Valid rows missing OTM call **or** OTM put | **35,460** (~10.2% of valid) |
| Quote `side` not call/put | **0** |
| `is_body` / `is_otm` inconsistent | **0** |
| OTM outside [0.03, 0.45] | **0** |
| bid/ask/mid ≤ 0 | **0** |
| Null/nonfinite greeks | **0** |
| Valid expiry meta vs quote mismatch | **0** |
| Valid body_strike vs body quote strike mismatch | **0** |
| bid > mid or mid > ask | **810** rows |
| spread_pct vs `(ask-bid)/mid` | Max abs diff ≈ 4.4e-16 (float noise) |

### `failure_reason` breakdown (meta)

| Reason | Count |
|--------|------:|
| `no_expiry_found` | 424,954 |
| *(null)* | 374,372 |
| `no_spot_price` | 250,581 |
| `no_spot_at_expiry` | 1,118 |

**T6 finding:** 25,733 rows have `surface_valid=False` **and** `failure_reason` null — produced via `_metadata_success_row` when body legs / quote count fail validity gate. Valid rows correctly have null failure_reason (348,639).

### Shortcuts / limitations

- Did not exhaustively iterate all quote rows for mid-vs-(bid+ask)/2 semantics (ORATS mid may differ from naive mid — consumer already uses `FillAssumption`).
- Iron fly/condor wing availability counted structurally (OTM presence), not at specific configured delta targets.
- Artifact provenance (which `--data-root` built cache) not recorded on disk — **lineage gap**.

---

## Actual artifact schema vs A1/A2 contract

### A1 meta

| Field | In artifact | Notes |
|-------|-------------|-------|
| All listed required + diagnostic fields | **Present** | 17 columns match producer |
| Missing required | **None** | |
| Extra columns | **None** | |
| dtype surprises | `entry_date`/`expiry_date` stored as date/timestamp convertible | Consumer normalizes |
| Null rates on **`surface_valid=True`** | Settlement fields **0% null** | Passes B4 settlement check |
| Null on invalid rows | Expected for early-failure paths | |

**Downstream-required but weakly tested on real artifacts:** `exit_spot` (S7), `body_strike`, wing quote availability (S3 — not encoded in meta).

### A2 quotes

| Field | In artifact | Notes |
|-------|-------------|-------|
| All contract required + diagnostic | **Present** | 27 columns |
| Missing | **None** | |
| Extra | **None** | |
| `spread_pct` | Consistent with formula on sample | |
| Assembly gaps | Duplicate grain; 810 bid/mid/ask ordering violations | Audit FAIL/WARN |

---

## T1–T7 checklist

| ID | Classification | Evidence |
|----|----------------|----------|
| **T1** | **Enforced by code but not tested** on full builder output | `_metadata_success_row` logic; contract tests partial; real cache: 57 valid rows violate exactly-one-body-leg expectation |
| **T2** | **Not enforced but should be audited** | 0 orphan quotes; valid-without-quotes 0; producer emits quotes only on success path — C6.3 join rules |
| **T3** | **Not enforced but should be audited** | 9,682 duplicate grain rows in cache → **FAIL** |
| **T4** | **Unclear / needs design decision** | Producer `get_trading_fridays` vs `trading_day.resolve_as_of_trading_day`; meta 97% Friday / 3% Thursday holiday fallback — align with HD-004-2 in C6.1 |
| **T5** | **Enforced by code but not tested**; **passes on real valid rows** | Null settlement 0; dte mismatch 0 |
| **T6** | **Not enforced** | 25,733 invalid rows with null `failure_reason`; vocabulary not closed for soft failures |
| **T7** | **Not enforced but should be audited** | Real cache loaded; schema pass; consistency **partial FAIL** (duplicates, T6, wing gaps) |

---

## Assembly-readiness checklist

| ID | Check | Classification | Real-cache note |
|----|-------|----------------|-----------------|
| **A** | Exactly one body call + one body put | **Not enforced** | 57 valid rows fail |
| **B** | ≥1 OTM call and ≥1 OTM put for fly/condor | **Not enforced in meta** | 35,460 valid rows fail (~10%) → S3 assembly failures |
| **C** | Quote `expiry_date` matches meta | **Implied by producer** | 0 mismatches on valid rows |
| **D** | Meta `body_strike` matches body quotes | **Implied** | 0 mismatches |
| **E** | No duplicate meta keys | **Not enforced in consumer** | 0 duplicates |
| **F** | bid ≤ mid ≤ ask (or documented mid semantics) | **Not enforced** | 810 violations → WARN/FAIL per C6.1 policy |
| **G** | `spread_pct` consistent | **Enforced at production** | Pass (float noise only) |
| **H** | Delta sign sanity | **Not enforced** | call δ<0: 0%; put δ>0: 0% → **WARN** rule only |

**Schema validity ≠ assembly-readiness:** 33% `surface_valid` can PASS/WARN for coverage, but short iron fly/condor also needs OTM wings — effective assembly rate lower.

---

## Producer operational safety assessment

| Question | Answer |
|----------|--------|
| `--output-root`? | **No** |
| Ticker subset? | **No** |
| Exact start/end dates? | **No** (year bounds only) |
| Dry-run? | **No** |
| Overwrite protection? | **No** — same `{freq}_{start}_{end}` path is overwritten |
| Hardcoded shared log? | **Yes** — `precompute_option_surface.log` |
| Always canonical cache filenames? | **Yes**, derived from args |
| Sample to `cache_c6_smoke/` without patch? | **No** |

**Can we safely run a small C6 sample precompute without risking existing cache artifacts?**

**Not reliably.** A non-colliding filename (e.g. `weekly_2026_2026`) avoids overwriting `weekly_2018_2026` but still: (1) processes full ticker CSV, (2) appends shared log, (3) no dry-run, (4) no isolated output root, (5) long runtime.

**Recommendation:** **C6.1A producer-safety patch** before any regeneration sample:

- `--output-root` (default remains cache)
- `--tickers` / `--tickers-file`
- `--start-date` / `--end-date`
- `--dry-run` (plan only)
- `--overwrite` guard (default refuse if outputs exist)

---

## Hardcoded paths and cache risks

| Risk | Impact | Mitigation in C6 |
|------|--------|------------------|
| Output always under `C:/MomentumCVG_env/cache` | Colliding runs destroy artifacts | C6.1A `--output-root` + overwrite guard |
| Universe from `cache/liquid_tickers.csv` not `input/liquidity/liquid_tickers.csv` | Divergence from C4/C5 superset | Document in C6.1; optional align flag |
| Spot DB path fixed | Stale spot if not re-extracted post-C5 | C6.4 note lineage; repair via extract_spot |
| Shared log append | Concurrent runs interleave | C6.1A per-run log or `--log-file` |
| Producer default `monthly` vs consumer `weekly` | Wrong filename if operator omits flag | C6.1 design + runbook |
| Cache built 2026-04-23, C5 closed 2026-07-04 | Surface may not reflect `adjusted_liquid` | C6.4 audit existing; C6.1A sample rebuild on C5 root |
| `surface-audit` stub | No PASS/WARN/FAIL gate | C6.3 + C8 wiring (C6.5 optional defer) |

---

## Key unknowns before design

1. **Artifact lineage:** Was `weekly_2018_2026` built with pre-C5 `--data-root` / spot DB? No manifest field records it.
2. **T4 resolver unification:** Should `get_trading_fridays` move to shared module matching `trading_day.py` + HD-004-2?
3. **T6 soft failures:** Should `_metadata_success_row` emit `missing_body_legs` / `insufficient_surface_quotes` when `surface_valid=False`?
4. **Duplicate quote grain:** Upstream chain duplicates vs producer bug — needs sampled root-cause in C6.4.
5. **Wing coverage SLA:** Is ~10% valid-but-no-OTM-wings acceptable (WARN) or repair scope?
6. **Validity rate 33%:** Expected for universe breadth vs WARN threshold — document by year/ticker in audit.
7. **Re-precompute scope post-C5:** Full weekly rebuild deferred — what sample window satisfies closeout T7?

---

## Recommended C6 task breakdown

| Task | Purpose | Expected files | Tests / report | Accept criteria | Closeout? |
|------|---------|----------------|----------------|-----------------|-----------|
| **C6.1** | Option surface **design plan** (audit rules, T6 vocabulary, T4 decision, PASS/WARN/FAIL thresholds) | `docs/tmp/c6_option_surface_design_plan.md` | — | HD-approved scope | **Required** |
| **C6.1A** | **Producer safety patch** — output root, ticker/date subset, dry-run, overwrite guard | `scripts/precompute_option_surface.py`, tests | CLI unit tests | Sample run writes only under chosen root; cannot clobber default artifacts accidentally | **Required before sample regen** |
| **C6.2** | Synthetic invariant tests T1–T6 | extend `tests/contract/`, new `tests/unit/test_option_surface_analyzer.py` | pytest green | T1–T6 enforced on fixtures | **Required** |
| **C6.3** | Surface artifact audit module + standalone CLI (and later `surface-audit` wrapper) | e.g. `scripts/audit_option_surface.py` or `src/data/surface_audit.py` | module tests | Implements sprint report sections; duplicate/join/T5/T6 rules | **Required** |
| **C6.4** | Real-cache sample audit report on existing `weekly_2018_2026` | `docs/tmp/c6_4_real_cache_surface_audit.md` | C6.3 tool output | ≥1 documented window; PASS/WARN/FAIL with evidence | **Required** |
| **C6.5** | Wire `refresh_weekly_inputs surface-audit` | `refresh_weekly_inputs.py` | CLI test | Delegates to C6.3 | **Defer to C8** (same pattern as split-audit) |
| **C6.6** | Closeout memo | `docs/sprint_memos/004_c6_option_surface.md` | — | T1–T7 + audit archived | **Required at C6 close** |

### Adjustments vs sprint template

- **C6.1A elevated to closeout prerequisite** for any regenerated sample (not optional).
- **C6.4 can run read-only on existing cache immediately after C6.3** — does not require regen for first pass.
- Post-C5 **regenerated sample** (small ticker×date window on `adjusted_liquid`) is a **second audit** after C6.1A + spot refresh — document in C6.1.

---

## Final recommendation

### **NOT READY — NEED C6.1A PRODUCER SAFETY PATCH FIRST**

**Rationale (ordered):**

1. **Operational safety:** C6 closeout requires ≥1 substantive sample run; current producer cannot isolate output or ticker/date scope and will overwrite canonical filenames if year range matches.
2. **Stale cache lineage:** Existing `weekly_2018_2026` predates C5 closeout — C6.4 must audit it read-only, but **trustworthy regeneration** for `adjusted_liquid` requires a safe sample path (C6.1A) before design implementation proceeds to execution.
3. **Real-cache consistency gaps:** Duplicate quote grain (T3 FAIL), T6 vocabulary holes, and assembly-readiness gaps (A, B) mean C6 deliverables must include audit module + tests — not schema review alone.

**C6.1 design may proceed in parallel** using this map; **do not run sample precompute or wire destructive refresh** until C6.1A lands.

### What C6 must deliver (answer to core question)

C6 must prove producer-created A1/A2 artifacts are:

| Dimension | Deliverable |
|-----------|-------------|
| **Present** | C6.3 inventory + C6.4 report on cache paths |
| **Schema-compatible** | C6.2 contract tests + C6.3 schema section |
| **Internally consistent** | C6.2 T1/T5/T6 + C6.3 rules (duplicates, dte, validity) |
| **Joinable** | C6.3 join integrity (T2) |
| **Assembly-ready** | C6.3 checks A–H; wing coverage WARN thresholds in C6.1 |
| **Date-aligned** | C6.1 T4 decision + C6.3 date alignment section |
| **Safe to audit on real cache** | C6.3 read-only CLI (no writes) |
| **Safe to regenerate samples** | **C6.1A** then bounded sample + C6.4 second report |

Until those ship: **do not treat A1/A2 as trustworthy inputs for SurfaceRunner real-data work (Sprint 006)** — synthetic runner tests remain the only validated assembly path.

---

## Appendix — producer vs consumer filename alignment

```text
SurfaceDataPaths defaults:
  option_surface_meta_weekly_2018_2026.parquet
  option_surface_quotes_weekly_2018_2026.parquet

precompute_option_surface.py (weekly --start-year 2018 --end-year 2026):
  option_surface_meta_weekly_2018_2026.parquet   ✓
  option_surface_quotes_weekly_2018_2026.parquet ✓

precompute_option_surface.py (default monthly):
  option_surface_meta_monthly_2018_2026.parquet   ✗ consumer mismatch
```
