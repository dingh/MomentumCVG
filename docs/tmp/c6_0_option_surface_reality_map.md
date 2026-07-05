# C6.0 — Option Surface Reality Map and Delivery Definition

**Sprint:** 004 · **Task:** C6.0 (read-only reconnaissance)  
**Date:** 2026-07-04 · **Review updated:** 2026-07-05  
**Mode:** Audit / delivery definition only — no code, tests, cache, or data changes  
**Status:** **C6.0 reality map accepted** — subtask plan mostly correct; sequence refined below

---

## Scope and non-goals

### In scope (this document)

- Map producer (`precompute_option_surface.py`, `option_surface_analyzer.py`), consumer (`OptionSurfaceDB`, S3 assembly, `SurfaceDataPaths`), tests, and on-disk cache artifacts.
- Answer: **what C6 must deliver** so A1/A2 artifacts are trustworthy enough for later `SurfaceRunner` / step3 strategy assembly.
- Separate **schema validity** from **assembly-readiness**.
- Assess whether a **producer-safety patch (C6.1A)** is required before **regenerated surface samples** (not before design).

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

## C6 design direction (from review)

C6 should produce an **audit gate**, not just more tests.

| Mechanism | Protects |
|-----------|----------|
| **Tests (C6.2)** | Producer **code behavior** on synthetic fixtures — invariants enforced at build time |
| **Audit gate (C6.3 / C6.4)** | The **actual A1/A2 parquet artifacts** that later backtests will consume — integrity and readiness on disk |

Pytest alone cannot certify that `option_surface_meta_weekly_2018_2026.parquet` is safe for `SurfaceRunner`; the audit CLI/report is the closeout evidence layer.

### Three-layer framing (C6.1 design should adopt)

```text
Layer 1 — Producer semantics     how OptionSurfaceBuilder + precompute script build rows
Layer 2 — Artifact integrity     schema, join, grain, settlement fields, failure vocabulary on parquets
Layer 3 — Assembly-readiness     whether valid surfaces can build straddle / ironfly / ironcondor at S3
```

Layers are **sequential dependencies**: Layer 2 assumes Layer 1 is correct; Layer 3 assumes Layer 2 passes. Do not collapse them into a single `surface_valid` flag.

### Recommended semantics (`surface_valid` vs audit readiness)

**Keep `surface_valid` as the general surface validity flag** (contract + T1):

```text
surface_valid == has_body_call AND has_body_put AND n_surface_quotes > 0
```

**Do not** redefine `surface_valid` to require iron fly / iron condor wings. Wing requirements are **assembly-time** (S3), not producer meta validity.

**Add audit-level readiness concepts** (computed in C6.3 audit, not necessarily new A1 columns unless C6.1B approves):

| Concept | Meaning |
|---------|---------|
| `body_pair_ready` | Exactly one body call and one body put in quotes |
| `otm_wing_pair_available` | ≥1 OTM call and ≥1 OTM put in quotes |
| `straddle_ready` | `surface_valid` AND `body_pair_ready` |
| `ironfly_candidate_ready` | `straddle_ready` AND `otm_wing_pair_available` (wing delta fit checked at S3) |
| `ironcondor_candidate_ready` | `straddle_ready` AND sufficient OTM quotes for short+long wing selection |

### PASS / WARN / FAIL policy direction (C6.1 to lock)

| Severity | Examples |
|----------|----------|
| **FAIL** | Missing required columns; duplicate meta keys; orphan quotes; valid meta without quotes; missing settlement fields on valid rows; `dte_actual` mismatch; quote/meta `expiry_date` mismatch; `body_strike` mismatch; invalid rows with null `failure_reason` **after** producer semantics fix (C6.1B if approved) |
| **WARN** | Low overall `surface_valid` rate; missing OTM wings on otherwise valid rows; bid/mid/ask ordering violations when ORATS mid is vendor-derived; weekly DTE calendar deviations; stale artifact lineage (pre-C5 root / spot path undocumented) |
| **INFO** | Validity rate by year/ticker; failure_reason breakdown; wing coverage (`straddle_ready` vs `ironfly_candidate_ready` rates) |

Duplicate quote grain (T3): **FAIL only after C6.3/C6.4 triage** determines whether duplicates are a true integrity violation or an under-specified grain — see § Duplicate quote grain.

### Refined subtask sequence (accepted plan)

```text
C6.1  — design memo first (layers, semantics, PASS/WARN/FAIL, T4/T6 decisions)
C6.1A — producer safety patch after design (output root, ticker/date scope, dry-run, overwrite guard)
C6.1B — producer semantics patch only if design approves (e.g. T6 failure_reason on soft failures)
C6.2  — synthetic invariant tests (code behavior)
C6.3  — audit CLI / artifact gate (parquet integrity + readiness metrics)
C6.4  — real-cache audit + regenerated-smoke audit evidence (two passes if needed)
C6.6  — closeout memo
(C6.5 surface-audit wiring → defer to C8)
```

**Ordering rule:** C6.1 **before** C6.1A/C6.1B. C6.1A **before any regenerated surface sample** — not before design. C6.3 before C6.4. Spot full refresh is **not** part of C6.0/C6 closeout itself — see § Upstream spot DB.

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
| Failure vocabulary (T6) | Failure row tags exist | contract test | Success-path invalid rows with `failure_reason=None` untested | C6.2 + **C6.1B if design approves** |
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
| Duplicate quote grain `(ticker, entry_date, expiry_date, strike, side)` | **9,682** rows involved — **triage required** (see § Duplicate quote grain) |
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
| Assembly gaps | Duplicate grain (pending triage); 810 bid/mid/ask ordering violations | Audit WARN/FAIL per C6.1 policy |

---

## Duplicate quote grain (T3 — triage, not assumed producer bug)

Real cache shows **9,682** quote rows involved in duplicate groups on grain `(ticker, entry_date, expiry_date, strike, side)`.

**Do not immediately classify this as a producer bug.** C6.3/C6.4 should:

1. **Sample duplicate groups** — pull full rows for a bounded set of keys (ticker, date, strike, side).
2. **Compare differing fields** — bid/ask/mid, volume, OI, greeks, `nearest_delta_bucket`, timestamps if present.
3. **Determine root cause:**
   - **Under-specified A2 grain** — e.g. multiple OPRA roots, expiry types, or line identifiers collapsed into one row key; contract may need extra columns or a documented composite grain.
   - **Upstream chain duplicates** — raw adjusted parquets carry duplicate strikes; producer should dedupe with a deterministic rule (document in C6.1B if approved).
   - **Producer double-emit** — builder appends same quote twice (code fix in C6.1B).

4. **Emit audit outcome:** FAIL if duplicates imply ambiguous assembly leg selection; WARN + documented dedupe rule if benign and deterministic; INFO if rate is negligible after dedupe policy.

Until triage completes, T3 status in checklist below is **“needs audit / design decision”**, not **“confirmed FAIL”**.

---

## T1–T7 checklist

| ID | Classification | Evidence |
|----|----------------|----------|
| **T1** | **Enforced by code but not tested** on full builder output | `_metadata_success_row` logic; contract tests partial; real cache: 57 valid rows violate exactly-one-body-leg expectation |
| **T2** | **Not enforced but should be audited** | 0 orphan quotes; valid-without-quotes 0; producer emits quotes only on success path — C6.3 join rules |
| **T3** | **Not enforced; needs audit / design decision** | 9,682 duplicate grain rows — sample before FAIL vs grain-spec change vs dedupe rule |
| **T4** | **Unclear / needs design decision** | Producer `get_trading_fridays` vs `trading_day.resolve_as_of_trading_day`; meta 97% Friday / 3% Thursday holiday fallback — align with HD-004-2 in C6.1 |
| **T5** | **Enforced by code but not tested**; **passes on real valid rows** | Null settlement 0; dte mismatch 0 |
| **T6** | **Not enforced** | 25,733 invalid rows with null `failure_reason`; vocabulary not closed for soft failures |
| **T7** | **Not enforced but should be audited** | Real cache loaded; schema pass; Layer 2/3 checks partial (T6, duplicate triage, readiness rates) |

---

## Assembly-readiness checklist (Layer 3 — audit metrics)

These map to **audit-level readiness concepts** above. They are **not** part of `surface_valid`.

| ID | Audit metric | Maps to | Real-cache note |
|----|--------------|---------|-----------------|
| **A** | `body_pair_ready` | Exactly one body call + one body put | 57 valid rows fail |
| **B** | `otm_wing_pair_available` | ≥1 OTM call and ≥1 OTM put | 35,460 valid rows fail (~10%) → `ironfly_candidate_ready` false |
| **C** | Quote `expiry_date` matches meta | Layer 2 integrity | 0 mismatches on valid rows |
| **D** | Meta `body_strike` matches body quotes | Layer 2 integrity | 0 mismatches |
| **E** | No duplicate meta keys | Layer 2 integrity | 0 duplicates |
| **F** | bid ≤ mid ≤ ask (or documented ORATS mid semantics) | Layer 2 / WARN | 810 violations → **WARN** if vendor mid |
| **G** | `spread_pct` consistent | Layer 2 | Pass (float noise only) |
| **H** | Delta sign sanity | Layer 3 INFO/WARN | call δ<0: 0%; put δ>0: 0% |

**Schema validity ≠ assembly-readiness:** 33% `surface_valid` can WARN for coverage; `straddle_ready` and `ironfly_candidate_ready` rates describe effective S3 yield separately.

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

**Recommendation:** **C6.1A producer-safety patch** after C6.1 design, **before any regenerated surface sample**:

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
| Spot DB path fixed | Stale spot if not re-extracted post-C5 | Document lineage in C6.4; full refresh before C7/C8 — **not part of C6.0** (see § Upstream spot DB) |
| Shared log append | Concurrent runs interleave | C6.1A per-run log or `--log-file` |
| Producer default `monthly` vs consumer `weekly` | Wrong filename if operator omits flag | C6.1 design + runbook |
| Cache built 2026-04-23, C5 closed 2026-07-04 | Surface may not reflect `adjusted_liquid` | C6.4 audit existing; C6.1A sample rebuild on C5 root |
| `surface-audit` stub | No PASS/WARN/FAIL gate | C6.3 + C8 wiring (C6.5 optional defer) |

---

## Upstream spot DB (documented dependency — not C6.0 scope)

C6.0 did **not** deep-audit `extract_spot_prices.py` or inventory `spot_prices_adjusted.parquet`. Spot is a **pipeline dependency** for surface precompute and a **closeout input** for C3 `validate`, but **full spot refresh is not part of C6.0 or C6 closeout itself**.

| Role | Detail |
|------|--------|
| **Producer** | `scripts/extract_spot_prices.py` — scans `adjusted_liquid` daily parquets, writes spot DB |
| **Default input** | `--data-root` → `C:/MomentumCVG_env/input/adjusted_liquid` (C5.11A wired) |
| **Default output** | `C:/MomentumCVG_env/cache/spot_prices_adjusted.parquet` (C1 manifest key `spot_prices`) |
| **Surface usage** | `SpotPriceDB` → meta `spot_move_pct`, `realized_volatility`; entry/exit spot from `ORATSDataProvider.get_spot_price` on chains |
| **Tests** | `tests/unit/test_spot_price_db.py` covers **reader** only |
| **Lineage gap** | On-disk spot DB (71 MB) likely **pre-C5** (same era as surface cache, 2026-04-23) |

### What C6 must document (not execute in C6.0)

- Record **spot path + mtime/hash** in C6.4 audit reports for any artifact under review.
- For **regenerated C5-aligned surface smoke** (post-C6.1A): require **explicit spot lineage** in the audit report — which `spot_prices_adjusted.parquet` path and `--data-root` were used. If spot is stale relative to `adjusted_liquid`, flag **WARN** on smoke evidence.

### Operator action before C7 + C8 (outside C6 closeout)

Full spot re-extract on C5 backfill should run **before C7 (PIT) and C8 (`refresh` wiring)** and **before regenerated C5-aligned surface evidence** — not as a C6 deliverable.

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/extract_spot_prices.py `
  --data-root C:/MomentumCVG_env/input/adjusted_liquid `
  --output C:/MomentumCVG_env/cache/spot_prices_adjusted.parquet `
  --start-year 2017 `
  --end-year 2026
```

**Safety:** `extract_spot_prices.py` supports `--output` but still full-file overwrite — back up existing parquet if rollback matters.

**Defer:** full spot audit module → C3 `validate` or optional future spot recon memo (not C6.0B unless HD requests).

---

## Key unknowns before design

1. **Artifact lineage:** Was `weekly_2018_2026` built with pre-C5 `--data-root` / spot DB? No manifest field records it.
2. **T4 resolver unification:** Should `get_trading_fridays` move to shared module matching `trading_day.py` + HD-004-2?
3. **T6 soft failures:** Should `_metadata_success_row` emit `missing_body_legs` / `insufficient_surface_quotes` when `surface_valid=False`?
4. **Duplicate quote grain:** Under-specified A2 grain vs upstream duplicates vs producer double-emit — **sample in C6.3/C6.4 before FAIL** (see § Duplicate quote grain).
5. **Wing coverage SLA:** Is ~10% valid-but-no-OTM-wings acceptable (WARN) or repair scope?
6. **Validity rate 33%:** Expected for universe breadth vs WARN threshold — document by year/ticker in audit.
7. **Re-precompute scope post-C5:** Full weekly rebuild deferred — what sample window satisfies closeout T7?
8. **Spot DB lineage:** Document path/hash in C6.4; full re-extract before C7/C8 and before C5-aligned surface smoke — **not a C6 deliverable**.

---

## Recommended C6 task breakdown

**Sequence:** C6.1 → C6.1A → (C6.1B if approved) → C6.2 → C6.3 → C6.4 → C6.6. C6.1A gates **regenerated surface samples only**.

| Task | Purpose | Expected files | Tests / report | Accept criteria | Closeout? |
|------|---------|----------------|----------------|-----------------|-----------|
| **C6.1** | **Design memo first** — three layers, semantics, PASS/WARN/FAIL, T4/T6/T3 duplicate triage policy | `docs/tmp/c6_option_surface_design_plan.md` | — | HD-approved scope | **Required** |
| **C6.1A** | **Producer safety patch** (after design) — output root, ticker/date subset, dry-run, overwrite guard | `scripts/precompute_option_surface.py`, tests | CLI unit tests | Isolated sample run cannot clobber canonical cache | **Required before regenerated surface sample** |
| **C6.1B** | **Producer semantics patch** (only if design approves) — e.g. T6 soft-failure tags, dedupe rule | `option_surface_analyzer.py`, tests | pytest | Design-approved behavior only | **Conditional** |
| **C6.2** | Synthetic invariant tests — **code behavior** (T1–T6) | extend `tests/contract/`, `tests/unit/test_option_surface_analyzer.py` | pytest green | T1–T6 on fixtures | **Required** |
| **C6.3** | **Audit CLI / artifact gate** — Layer 2 + Layer 3 metrics on parquets | e.g. `scripts/audit_option_surface.py` | module tests | PASS/WARN/FAIL report sections; readiness metrics | **Required** |
| **C6.4** | **Real-cache + regenerated-smoke audit evidence** | `docs/tmp/c6_4_real_cache_surface_audit.md` (+ smoke report if regen) | C6.3 output | Pass 1: existing cache read-only; Pass 2: post-C6.1A smoke with documented spot/adj lineage | **Required** |
| **C6.5** | Wire `refresh_weekly_inputs surface-audit` | `refresh_weekly_inputs.py` | CLI test | Delegates to C6.3 | **Defer to C8** |
| **C6.6** | Closeout memo | `docs/sprint_memos/004_c6_option_surface.md` | — | T1–T7 + audit archived | **Required** |

### Adjustments vs sprint template

- **C6.1 before C6.1A/C6.1B** — design locks semantics and audit policy first.
- **C6.1A before regenerated surface sample** — not before design or read-only C6.4 pass on existing cache.
- **C6.4 pass 1** can run read-only on existing cache immediately after C6.3.
- **C6.4 pass 2** (regenerated smoke on `adjusted_liquid`) after C6.1A + documented spot lineage; optional C6.1B if design requires producer fixes first.
- **Spot full re-extract** — pipeline prerequisite before C7/C8 and before C5-aligned surface smoke; **document in C6.4**, not a C6 task row.

---

## Final recommendation

### **READY FOR C6.1 DESIGN**

### **NOT READY TO REGENERATE SURFACE SAMPLES UNTIL C6.1A PRODUCER SAFETY PATCH**

**C6.0 reality map:** **Accepted.** Findings and subtask plan are sufficient to start C6.1 design memo.

**Rationale:**

1. **Design can proceed now** — three-layer framing, semantics, and PASS/WARN/FAIL policy are scoped; no code changes required to begin C6.1.
2. **Regenerated surface samples blocked** — producer lacks isolated output root, ticker/date scope, dry-run, and overwrite guard (C6.1A after design).
3. **Audit gate is the closeout centerpiece** — C6.3/C6.4 protect on-disk A1/A2; C6.2 tests protect code only.
4. **Existing cache** — read-only C6.4 pass 1 can proceed after C6.3 without regen; stale lineage and duplicate grain require triage, not assumed producer bug.
5. **Spot DB** — document lineage in C6.4; full refresh before C7/C8 and before C5-aligned smoke — outside C6.0/C6 closeout scope.

**Do not run regenerated surface precompute until C6.1A lands.** Read-only audit of existing parquets is fine after C6.3.

### What C6 must deliver (answer to core question)

C6 must prove producer-created A1/A2 artifacts are:

| Dimension | Deliverable |
|-----------|-------------|
| **Present** | C6.3 inventory + C6.4 report on cache paths |
| **Schema-compatible** | C6.2 contract tests + C6.3 schema section (Layer 2) |
| **Internally consistent** | C6.2 T1/T5/T6 + C6.3 integrity rules (Layer 2) |
| **Joinable** | C6.3 join integrity (T2) |
| **Assembly-ready** | C6.3 readiness metrics (`straddle_ready`, `ironfly_candidate_ready`, …) — Layer 3 |
| **Date-aligned** | C6.1 T4 decision + C6.3 date alignment section |
| **Safe to audit on real cache** | C6.3 read-only CLI (no writes) |
| **Safe to regenerate samples** | C6.1 → C6.1A → (C6.1B) → smoke with documented spot/adj lineage → C6.4 pass 2 |

Until C6 closeout: **do not treat A1/A2 as trustworthy inputs for SurfaceRunner real-data work (Sprint 006)** — synthetic runner tests remain the only validated assembly path.

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
