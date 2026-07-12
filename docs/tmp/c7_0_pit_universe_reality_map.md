# C7.0 — Point-in-time universe reality map

**Sprint:** 004 · **Task:** C7 (investigation only)  
**Repository HEAD:** `86013504a5a0ec0a8a5d648649e7ab632cd38d01` (reality map at C7.0 commit); C7.2 target branch  
**Created:** 2026-07-11  
**Updated:** 2026-07-11 (C7.0/C7.1 follow-up — rolling provenance, artifact envelope, sample discovery)  
**Scope:** Document what exists at pre-C7.2 HEAD. Design policy in `c7_1_pit_universe_design_memo.md`.

---

## HD amendment (2026-07-11) — supersedes §7 policy recommendation

**Operator decision:** At trade date `t`, use the **prior completed weekly snapshot**, not same-day liquidity.

**Operational reason:** At the moment of the trade decision, that day's dollar volume does not exist yet — the market is open or the day has not finished. The panel's `atm_straddle_dollar_vol` is a **12-week rolling mean of completed weekly observations**, built from EOD ORATS after each week closes. Universe membership must therefore come from the **last fully observed week**, not from a snapshot dated on `t`.

| | HEAD code (§2) | C7 implementation target |
|---|----------------|---------------------------|
| Rule | `max(month_date <= trade_date)` | `max(month_date < trade_date)` |
| Friday trade on week-end | Same-day snapshot allowed | Prior week's snapshot |
| Rationale | — | No same-day liquidity at decision time; matches C4 Saturday-build / completed-week panel |

The reality map §2–§5 describe **pre-C7.2 HEAD** repository behavior. C7.2 changes `step1_get_universe`, `surface_engine_data_contract.md`, and sprint PIT acceptance wording per [c7_1_pit_universe_design_memo.md](c7_1_pit_universe_design_memo.md). `v1_universe_protocol.md` remains temporarily stale until C9.

---

## Inspection scope

### Required files (read)

| Path | Role |
|------|------|
| `docs/agenda/current_sprint.md` | Sprint 004 mode, blocker #6 (PIT universe), C7 commit slot |
| `docs/v1_universe_protocol.md` | Documented PIT universe rule |
| `docs/v1_weekly_runbook.md` | Operator paths, two-universe-layer warning |
| `docs/surface_engine_data_contract.md` | A3 schema, S1 invariants |
| `docs/surface_engine_data_flow.md` | Stage A → S1 data flow |
| `docs/tmp/c4_liquidity_panel_design_plan.md` | C4 producer semantics, operator timing |
| `docs/sprint_memos/004_c4_liquidity_panel.md` | C4 closeout evidence |
| `docs/sprint_memos/004_c5_adjusted_liquid.md` | Superset vs trading universe |
| `docs/sprint_memos/004_c6_option_surface.md` | C7 deferred note |
| `scripts/build_liquidity_panel.py` | A3 producer |
| `scripts/refresh_weekly_inputs.py` | CLI skeleton; no PIT audit wired |
| `src/backtest/pipeline.py` | `step1_get_universe` consumer |
| `src/backtest/run_config.py` | `dvol_top_pct`, `spread_bottom_pct` config |
| `src/data/paths.py` | `DEFAULT_LIQUID_TICKERS_PATH` |
| `src/data/trading_day.py` | `--as-of` resolution (adjusted chains; not liquidity panel) |
| `tests/contract/test_step1_universe_contract.py` | S1 contract tests |
| `tests/contract/conftest.py` | Synthetic liquidity fixture |
| `tests/unit/test_build_liquidity_panel.py` | C4 producer unit tests |
| `tests/unit/test_refresh_weekly_inputs_cli.py` | CLI skeleton tests |

### Additional files discovered by search

| Path | Relevance |
|------|-----------|
| `src/backtest/surface_runner.py` | Loads panel, calls `step1_get_universe` per trade date |
| `tests/unit/test_surface_runner_data_flow.py` | Synthetic E2E; weak PIT assertion |
| `tests/contract/test_precompute_input_contract.py` | A3 required-column check on fixture only |
| `tests/contract/test_orchestration_contract.py` | Builds synthetic liquidity parquet |
| `scripts/precompute_option_surface.py` | Consumes `liquid_tickers.csv` as precompute superset |
| `scripts/run_surface_search.py` | CLI defaults `dvol_top_pct=0.20`, `spread_bottom_pct=0.20` |
| `docs/surface_runner_data_flow.md` | Notes partial PIT verification |
| `docs/correctness_audit.md` | Lists missing direct PIT unit tests |
| `docs/v1_spec_pins.md` | Top-20% rolling liquidity (tentative window wording) |

### Search terms — where they appear

| Term | Primary locations |
|------|-------------------|
| `step1_get_universe` | `src/backtest/pipeline.py`, `surface_runner.py`, contract tests, docs |
| `ticker_liquidity_panel` | `build_liquidity_panel.py`, manifest, runbook, C4 memo |
| `month_date` | Panel grain (legacy name); week-end snapshot date in C4 |
| `snapshot_date` | Panel column (duplicate of `month_date` in C4 output) |
| `has_valid_atm_pair` | Panel + S1 eligibility gate |
| `atm_straddle_dollar_vol` | Panel rank key |
| `atm_spread_pct` | Panel spread metric |
| `dvol_top_pct` | `run_config`, panel stamp, `build_liquid_tickers` |
| `spread_bottom_pct` | `run_config`, S1 filter (config name) |
| `spread_bot_pct` | Panel builder stamp, `build_liquid_tickers` (producer name) |
| `liquid_tickers.csv` | C4 output; C5/C6 precompute superset |
| `PIT universe` / `point-in-time universe` | Sprint agenda, universe protocol, audit docs |

**Not found:** `pit_universe_audit.py`, `audit_pit_universe.py`, or any dedicated PIT harness module (C7 not implemented).

---

## 1. Current producer

### How `ticker_liquidity_panel.parquet` is produced

**Script:** `scripts/build_liquidity_panel.py`  
**Pipeline:** three stages written atomically via staging dir:

```text
ORATS raw ZIPs (ORATS_Data)
  → ticker_liquidity_daily_observations.parquet
  → ticker_liquidity_weekly_observations.parquet
  → ticker_liquidity_panel.parquet
  + liquid_tickers.csv
```

**Modes:**

| Mode | Behavior |
|------|----------|
| `backfill` | Rebuild daily → weekly → panel for `--start-date` … `--end-date` |
| `incremental` | Append exactly one new completed ORATS week; param mismatch → `LiquidityPanelError` |

### Input sources and default paths

| Input | Default path |
|-------|--------------|
| Raw ORATS daily ZIPs | `C:/ORATS/data/ORATS_Data/{YYYY}/ORATS_SMV_Strikes_{YYYYMMDD}.zip` |
| Output cache (production) | `C:/MomentumCVG_env/input/liquidity/` |
| Output cache (script default) | `C:/MomentumCVG_env/cache/` |

Liquidity uses **raw** bid/ask/volume only (`LIQUIDITY_SOURCE = raw_option_bid_x_volume_sum_dte_5_60`). Split-adjusted chains are **not** inputs to C4.

### Daily, weekly, and rolling-panel stages

1. **Daily:** per `(trade_date, ticker)` — sum ATM straddle bid-dollar vol across expiries with DTE ∈ [5, 60]; bid-vol-weighted spread; `daily_has_valid_quote` gate.
2. **Weekly:** per `(week_end_date, ticker)` — mean daily vol over ORATS days in ISO week; mean spread over valid-quote days; `weekly_has_valid_quote`.
3. **Rolling panel:** per `(snapshot_date, ticker)` — mean of last `lookback_weeks` (default 12) weekly vols (missing weeks → 0 in denominator); mean spread over weeks with valid quotes; `has_valid_atm_pair = valid_quote_weeks >= min_valid_quote_weeks` (default 3).

### Artifact grain

| Artifact | Grain | Row count (production) |
|----------|-------|------------------------|
| Daily | `trade_date`, `ticker` | (not re-counted this session) |
| Weekly | `week_end_date`, `ticker` | 2,438,191 rows / 488 weeks |
| Panel | `month_date`, `ticker` | 2,434,339 rows / 477 snapshots / 9,454 tickers |
| `liquid_tickers.csv` | `Ticker` (one row per historical qualifier) | 2,783 tickers |

Production panel has **zero** duplicate `(month_date, ticker)` keys (verified 2026-07-11).

### Meaning of `month_date`

- **Legacy column name** retained for S1 compatibility (`PANEL_STEP1_COLS`).
- **Actual semantics (C4):** week-end snapshot date `T` = last ORATS trading day in the completed ISO week (`week_end_date`).
- Both `month_date` and `snapshot_date` are set to `pd.Timestamp(snap)` in `aggregate_rolling_weekly_panel()` — identical values in production.

### Meaning of `snapshot_date`

- Present on panel rows as a duplicate of `month_date`.
- **Not consumed by `step1_get_universe`** — S1 reads only `month_date`.

### Rolling-window fields and build-parameter columns

**Provenance / window (per panel row):**

| Column | Meaning |
|--------|---------|
| `window_start_date` | First `week_end_date` in the 12-week window |
| `window_end_date` | Same as snapshot (`snap`) |
| `window_shortfall` | `max(0, lookback_weeks - len(window))` |
| `valid_quote_weeks` | Weeks in window with `weekly_has_valid_quote` |
| `zero_volume_weeks` | Weeks in window with zero/missing vol |
| `lookback_weeks`, `min_valid_quote_weeks`, `dte_min`, `dte_max` | Build params |
| `dvol_top_pct`, `spread_bot_pct` | Universe-rule params stamped at build (for incremental guard + `liquid_tickers`) |
| `liquidity_source` | Constant source tag |

Production stamped params (all rows): `lookback_weeks=12`, `min_valid_quote_weeks=3`, `dte_min=5`, `dte_max=60`, `dvol_top_pct=0.20`, `spread_bot_pct=1.0`.

### PIT / no-lookahead verifiability from panel alone

The panel carries **partial** evidence for independent verification:

| Check | Panel support |
|-------|---------------|
| Snapshot date strictly before trade date (C7 target) | Yes — via `month_date` + strict `<` rule |
| Rolling window bounds | Yes — metadata columns; **values** need independent weekly recomputation |
| Weekly source rows used | **No** — must join weekly artifact and recompute C4 formulas |
| Daily source rows used | **No** — requires daily artifact or ORATS re-read |
| Duplicate grain | Detectable from panel |
| Build param consistency | Detectable from stamped columns |
| Supported S1 parameter envelope | Readable from panel stamp vs requested config |

C4 design: snapshot `S` rolling mean uses weekly rows with `week_end_date` in the last `lookback_weeks` global weeks where `week_end_date <= S`. **C7 must independently recompute** those means from weekly observations — not filter weekly rows and assert `week_end_date <= S` (that assertion is tautological).

### Accepted C4 warnings and limitations

From `docs/sprint_memos/004_c4_liquidity_panel.md`:

| Limitation | Notes |
|------------|-------|
| Partial ORATS weeks | `week_end_date` = last available ORATS day in week |
| `spread_bot_pct = 1.0` default | Spread filter off for **superset** (`liquid_tickers.csv`) only; S1 spread threshold is separate via `BacktestRunConfig` |
| Raw ORATS only | Ranking raw; economics use adjusted chains (C5) |
| `liquid_tickers.csv` | Historical superset, not trading universe |
| `has_valid_atm_pair` regime shift | ~73% → ~50% valid rate late 2022→2023; investigated, not pipeline bug |
| Runtime | Full backfill ~15–20 h; incremental ~2 min/week |

---

## 2. Current consumer

### `src/backtest/pipeline.py::step1_get_universe`

Exact code semantics (lines 61–131 at HEAD):

#### Required input columns

`month_date`, `ticker`, `atm_straddle_dollar_vol`, `atm_spread_pct`, `has_valid_atm_pair`

(No schema validation at load time; missing columns → `KeyError`.)

#### Snapshot resolution rule

```python
trade_ts = pd.Timestamp(trade_date)
valid_months = liquidity_panel.loc[liquidity_panel["month_date"] <= trade_ts, "month_date"]
if valid_months.empty:
    return empty frame
snapshot_date = valid_months.max()
```

**Global:** one `snapshot_date` for the entire cross-section — **not** per ticker.

#### Snapshot selection scope

**Global** — all tickers come from rows where `month_date == snapshot_date`.

Tickers absent from that snapshot are excluded even if they appear on an older snapshot ≤ `trade_date`.

#### Eligibility filters (applied after snapshot slice)

All must hold (AND):

1. `month_date == snapshot_date`
2. `has_valid_atm_pair == True` (strict equality)
3. `atm_straddle_dollar_vol.notna()`
4. `atm_spread_pct.notna()`

No explicit exclusion-reason column; failing rows are silently dropped.

#### Ranking method

On the eligible cross-section **before** threshold filtering:

```python
snap["dvol_rank_pct"] = snap["atm_straddle_dollar_vol"].rank(ascending=True, method="average", pct=True)
snap["spread_rank_pct"] = snap["atm_spread_pct"].rank(ascending=False, method="average", pct=True)
```

#### Rank direction

| Metric | `ascending` | Interpretation |
|--------|-------------|----------------|
| `atm_straddle_dollar_vol` | `True` | Higher volume → higher rank pct |
| `atm_spread_pct` | `False` | Tighter spread → higher rank pct |

#### Threshold formulas

From `config.dvol_top_pct` and `config.spread_bottom_pct`:

```python
dvol_threshold   = 1.0 - config.dvol_top_pct
spread_threshold = 1.0 - config.spread_bottom_pct
keep if dvol_rank_pct >= dvol_threshold AND spread_rank_pct >= spread_threshold
```

Example: `dvol_top_pct=0.20` → keep `dvol_rank_pct >= 0.80` (top 20% by volume rank).

#### AND versus OR behavior

**AND** — both conditions required.

#### Output schema

Exactly: `[ticker, dvol_rank_pct, spread_rank_pct]`

#### Empty-result behavior

Returns empty `DataFrame` with correct columns when:

- No `month_date <= trade_date`
- Snapshot slice empty after eligibility filters
- No rows pass AND thresholds

Does **not** return `None` (runner checks for `None` defensively).

#### Tie behavior

`method="average"` with `pct=True` — tied values share averaged percentile ranks. Boundary ties can include more than exactly `ceil(n * top_pct)` names.

#### Row ordering determinism

**Membership:** deterministic w.r.t. metric values (verified: shuffled panel rows → same ticker set and ranks).

**Row order:** **not** canonical — output follows filtered DataFrame order (`reset_index(drop=True)` only). Shuffled input → same tickers/ranks but **different row order**.

#### Duplicate panel rows

**Not explicitly rejected.** Duplicate `(month_date, ticker)` would duplicate tickers in output. Production panel has zero duplicates; no runtime guard.

#### Missing liquidity behavior

| Case | Behavior |
|------|----------|
| `has_valid_atm_pair == False` | Excluded silently |
| NaN dvol or spread | Excluded silently |
| Ticker missing from snapshot | Excluded (not carried forward) |
| Empty universe | Empty frame; no reason code |

---

## 3. Existing test coverage

| Test file | Invariant tested | Synthetic / real | What it proves | What it does not prove |
|-----------|------------------|------------------|----------------|------------------------|
| `tests/contract/test_step1_universe_contract.py` | Output schema | Synthetic fixture (`liquidity_panel_two_snapshots`) | Columns `[ticker, dvol_rank_pct, spread_rank_pct]` | Real panel behavior |
| same | PIT snapshot = latest `month_date <= trade_date` | Synthetic | Jan vs Feb membership change | Global vs per-ticker protocol |
| same | `has_valid_atm_pair=False` excluded | Synthetic | D dropped | Explicit exclusion reporting |
| same | AND filter | Synthetic | Top 50% dvol with spread open | Exact boundary tie counts |
| same | No snapshot before first → empty | Synthetic | Pre-2024 empty universe | Production date range |
| same | Rank pct ∈ [0, 1] | Synthetic | Interval bounds | Rank formula vs quantile |
| `tests/contract/conftest.py` | Fixture shape | Synthetic | Two snapshots, 4+2 tickers | Rolling provenance |
| `tests/contract/test_precompute_input_contract.py` | A3 required columns on fixture | Synthetic | Column names present | Production schema extras |
| `tests/unit/test_build_liquidity_panel.py` (30 tests) | Daily/weekly/panel aggregation | Synthetic + ZIP I/O | Producer formulas, incremental, params | S1 membership on real data |
| same | `build_liquid_tickers` | Synthetic | CSV columns | Quantile vs rank parity with S1 |
| `tests/unit/test_surface_runner_data_flow.py` | E2E runner | Synthetic | Traded tickers ⊆ {LONG1, SHORT1} | Not a real PIT audit |
| `tests/contract/test_orchestration_contract.py` | ORCH loop | Synthetic tmp panel | Pipeline runs | Universe correctness |
| `tests/unit/test_refresh_weekly_inputs_cli.py` | CLI plan mentions liquidity | N/A | Scaffolding only | PIT validation |

### Explicit gap matrix (does current tests prove …?)

| Property | Proven? | Evidence |
|----------|---------|----------|
| No future / same-day snapshot | **Partial** | Synthetic only; HEAD still uses `<=` |
| Correct global snapshot selection | **Partial** | Two-snapshot fixture; not boundary/holiday cases |
| Correct ranking | **Partial** | One AND-filter case; not tie boundaries |
| Duplicate-grain rejection | **No** | Not tested in S1 or producer read path |
| Determinism | **Partial** | Rank values stable; row order not tested |
| Real production-panel behavior | **No** | No test reads `input/liquidity/` panel |
| Rolling-window provenance | **No** | Must independent-recompute from weekly grid (design memo §7) |
| Supported envelope vs superset | **No** | Not enforced |
| Full-history superset coverage | **No** | Sample-only ad hoc check |
| Future weekly-row invariance | **No** | Not tested |
| Independent reference ≠ double S1 | **No** | Harness absent |

---

## 4. Existing real-data evidence

### What C4 already proved

**Production artifact:** `C:/MomentumCVG_env/input/liquidity/`

| Metric | Value | Source |
|--------|-------|--------|
| Date range | 2017-01-06 → 2026-02-20 | Production panel read 2026-07-11 |
| Snapshot count | 477 | Same |
| Universe size (S1, `dvol_top_pct=0.20`, `spread_bottom_pct=1.0`) | 506–959 (median 631) | Full snapshot loop vs C4 memo band |
| Distinct panel tickers | 9,454 | Same |
| `liquid_tickers.csv` count | 2,783 | Same |
| Duplicate `(month_date, ticker)` | 0 | Same |
| Incremental ≡ backfill (same snapshot) | PASS | C4 memo |
| `window_shortfall > 0` snapshots | 0 / 477 | Production (weekly history from 2016-10-21 fills 12-week window) |

**C4 does NOT prove C7:**

- S1 membership matches an independent recomputation on sample dates
- No future weekly observations in any snapshot's rolling window (join check not run)
- Per-trade-date PIT resolution on calendar boundaries beyond spot checks
- Reproducible canonical membership hash across repeated audits
- Selected trading-universe tickers always ⊆ `liquid_tickers.csv` (spot check on last snapshot: 0 missing — not systematic)
- Same-day vs prior-week operator timing policy
- Global vs per-ticker protocol alignment

### Known historical regime shifts (C4 accepted)

- `has_valid_atm_pair` rate drop ~2022→2023 (vol + eligibility, not bug)
- Partial ORATS weeks → non-Friday `week_end_date` possible

### Unverified assumptions C7 must test

1. Global snapshot semantics with strict `<` (vs per-ticker carry-forward in protocol doc).
2. Rank-percentile S1 mechanics match `BacktestRunConfig` for **supported** parameter envelope only.
3. Independent rolling recomputation matches stored panel rows on bounded samples.
4. Historical snapshot `S` invariant to weekly rows with `week_end_date > S`.
5. Full-history superset coverage for canonical supported configuration.
6. Missing liquidity never yields silent PASS in an audit report.

---

## 5. Documentation and implementation conflicts

### A. Global snapshot versus per-ticker lookup

**Intended v1 model (confirmed):** At trade date `t`, pick **one** prior completed weekly snapshot for the **whole market**, then rank dollar volume on that cross-section and select the top `dvol_top_pct` fraction (and apply spread filter per config). The snapshot is **global** — not chosen separately per ticker.

**What C7 implements:**

```text
1. global_snapshot_date = max(month_date where month_date < trade_date)   # one date for all tickers
2. eligible pool       = tickers on that snapshot with has_valid_atm_pair == True
3. rank                = atm_straddle_dollar_vol on that cross-section
4. select              = top dvol_top_pct (spread filter when spread_bottom_pct < 1)
```

**Why this section exists:** `docs/v1_universe_protocol.md` step 2 says *"use the most recent `month_date <= t` **for each ticker**"*, which reads like per-ticker carry-forward (ticker A on snapshot S₁, ticker B on snapshot S₂). That is **not** what `step1_get_universe` does and **not** the intended weekly workflow.

| Model | Behavior |
|-------|----------|
| **Global (intended + code)** | One snapshot date; all tickers ranked together on the same cross-section |
| **Per-ticker (protocol wording only)** | Each ticker could use a different snapshot ≤ `t`; stale names could linger on old rows |

**Classification:** **documentation drift** — fix `v1_universe_protocol.md` at Sprint 004 closeout (C9) to say *one global snapshot before `t`*, not *for each ticker*. No S1 logic change needed for this item (only the `<` vs `<=` timing change from the HD amendment).

### B. Same-day versus prior-completed-week timing

**Correct operational model:** When trading on date `t`, you cannot know `t`'s dollar volume — only completed prior weeks are observable. The universe is therefore decided by **ranking on the prior completed weekly snapshot**, not on a snapshot whose `month_date` equals `t`.

Example (Friday entry):

```text
Trade:           Fri 2024-02-09  (decision at open — week ending 2024-02-09 not yet complete)
Panel snapshot:  Fri 2024-02-02  (last completed week; built Saturday 2024-02-03 from EOD data)
Ranking:         top dvol_top_pct of atm_straddle_dollar_vol on the 2024-02-02 cross-section (params from BacktestRunConfig)
```

| Source | Timing |
|--------|--------|
| Economic reality | Same-day liquidity unavailable at trade decision |
| C4 design / operator model | Snapshot built **Saturday** from **completed week**; used for **next** trade week |
| `step1_get_universe` at HEAD | `month_date <= trade_date` — **bug / PIT leak** for backtest (can pick same-day snapshot) |
| **HD amendment (C7 target)** | `month_date < trade_date` — prior completed week only |

**Classification:** HEAD `<=` is **confirmed defect for backtest PIT** (not merely doc ambiguity). C7.2 implements `<` per HD decision.

### C. Universe-rule wording — two layers (do not conflate)

There are **two different uses** of dvol/spread percentages. Only the second is tunable per backtest run.

#### Layer 1 — Superset contraction (C4 panel build)

| Param | Where | Production value | Purpose |
|-------|-------|------------------|---------|
| `dvol_top_pct` | `build_liquidity_panel.py`, panel stamp | 0.20 | Which names ever qualify into `liquid_tickers.csv` |
| `spread_bot_pct` | panel builder, `build_liquid_tickers()` | 1.0 | Spread leg **off** for superset — volume-only contraction |

**Why 20% vol + spread=1.0 here:** shrink the historical **precompute superset** (~2,783 tickers in `liquid_tickers.csv`), not define the only trading universe for backtests. Surface precompute and C5 scoped adjustment consume this CSV; it should be broad enough to cover any S1 selection from the full panel.

`build_liquid_tickers()` uses **quantile** thresholds per snapshot (not identical to S1 rank pct at tie boundaries).

#### Layer 2 — Trading universe (S1 / backtest)

| Param | Where | Configurable? | Supported by current artifacts? |
|-------|-------|---------------|--------------------------------|
| `dvol_top_pct` | `BacktestRunConfig` | Yes — any value in `(0, 1]` | **Only if** `requested <= superset build dvol_top_pct` stamped on panel |
| `spread_bottom_pct` | `BacktestRunConfig` | Yes — any value in `(0, 1]` | **Only if** `requested <= 1.0` (superset spread filter fully open) |

**S1 filter logic (always):** AND of rank-percentile thresholds on the full panel cross-section at the prior-week snapshot:

```text
dvol_rank_pct   >= 1 - dvol_top_pct
spread_rank_pct >= 1 - spread_bottom_pct
```

`BacktestRunConfig` permits a wider mathematical range. The **current precomputed data layer** supports only the envelope covered by the accepted superset:

```text
Production baseline (panel stamp):
    superset build dvol_top_pct   = 0.20
    superset build spread_bot_pct = 1.0

Structurally supported S1 requests against current liquid_tickers.csv:
    requested dvol_top_pct        <= 0.20
    requested spread_bottom_pct   <= 1.0
```

Spread rule nuance: superset build has spread filter **fully open** (`spread_bot_pct = 1.0`). Any S1 `spread_bottom_pct` in `(0, 1]` only **narrows** names. A **wider** dollar-volume fraction than the superset build (e.g. `dvol_top_pct = 0.50`) can select tickers never precomputed → **blocking configuration/artifact-envelope FAIL** in C7, not a silent PASS.

Examples:

| Config | Supported on current artifacts? |
|--------|--------------------------------|
| `dvol=0.20`, `spread=1.0` | **Yes** — canonical audit baseline |
| `dvol=0.20`, `spread=0.20` | **Yes** — narrower spread only |
| `dvol=0.10`, `spread=1.0` | **Yes** — narrower dvol only |
| `dvol=0.50`, `spread=1.0` | **No** — broader than superset build |

#### C7 stance

- Validate S1 **mechanics** (strict prior snapshot, rank direction, AND logic) for caller-supplied params.
- **Enforce supported artifact envelope** before superset-coverage or full-history checks.
- Default audit CLI to canonical supported params (`0.20` / `1.0`); wider configs require a broader accepted superset artifact.

**Classification:** **resolved** — two-layer model with explicit supported envelope (reality map §5D, design memo §1.5).

### D. Precompute superset versus trading universe

**Purpose of the superset:** Narrow the amount of data that must be **precomputed** for backtesting and research — e.g. option surface (A1/A2), split-adjusted chains (C5), and (in Sprint 005) signal/feature history. Without it, Stage A would need to run on the full ORATS ticker set (~9k+ names in the panel) instead of a bounded engineering pool.

**This is not the trading universe.** The trading universe is S1 output at each rebalance date `t` (prior-week snapshot + configurable rank filters on the **full panel**).

**Required invariant (supported configurations only):**

```text
trading_universe(t) ⊆ liquid_tickers.csv   when requested S1 params are within supported envelope
```

If `requested dvol_top_pct > superset build dvol_top_pct`, the audit returns a **configuration/artifact-envelope FAIL** before coverage checks — the current superset cannot certify wider universes.

If a ticker is selected within the supported envelope but absent from `liquid_tickers.csv`, surface/features may be missing → **FAIL**.

| Layer | Artifact | What it is | Used for |
|-------|----------|------------|----------|
| **Precompute superset** | `liquid_tickers.csv` (~2,783 tickers) | Historical union under superset build params (`dvol_top_pct=0.20`, `spread_bot_pct=1.0`) | Surface precompute, C5 scoped adjust, future feature precompute — **reduces compute/storage** |
| **Full liquidity panel** | `ticker_liquidity_panel.parquet` | All tickers × weekly snapshots with rolling metrics | S1 ranking cross-section |
| **Trading universe** | S1 output at `trade_date` | Top/filtered slice at prior-week snapshot | Backtest/live rebalance |

```text
ORATS universe  ⊃  panel tickers  ⊇  trading_universe(t)  ⊆  liquid_tickers.csv
                                                      (supported envelope only)
```

**C7 checks:**

1. **Per-sample** superset coverage for discovered/manual trade dates.
2. **Full-history supported-envelope coverage** (artifact-level): vectorize across all production snapshots; for canonical params (`0.20` / `1.0`), independently compute S1 membership per snapshot→trade-date mapping; assert every selected ticker ∈ `liquid_tickers.csv`. Does **not** certify unsupported wider configs.

---

## 6. Trust gaps for C7

| ID | Risk | Current evidence | Missing evidence | Severity | Proposed C7 check |
|----|------|------------------|------------------|----------|-------------------|
| G1 | Future / same-day snapshot | HEAD uses `<=` | Strict `resolved_snapshot < trade_date`; same-day **FAIL** | **High** | Per-sample PIT resolution |
| G2 | Rolling panel not independently verified | C4 design only | Recompute C4 formulas from weekly grid; compare to panel row | **High** | Independent rolling recomputation |
| G3 | Duplicate `(month_date, ticker)` | Production count = 0 | Automated FAIL on any duplicate | **Med** | Artifact grain check |
| G4 | Missing required columns | Contract test on fixture | Production panel schema audit | **Med** | Artifact schema check |
| G5 | Mixed build parameters | Production uniform stamp | Detect varying stamped params within file | **Med** | Build-param homogeneity |
| G6 | Nondeterministic membership | Shuffled-row rank stability (ad hoc) | Canonical sort + membership hash | **Med** | Determinism contract |
| G7 | S1 vs independent reference | None | Recompute ranks/thresholds without double S1 | **High** | Independent reference module |
| G8 | Global vs per-ticker doc drift | Protocol wording | Informational note; global cross-section is policy | **Low** | Report only (C9 fixes doc) |
| G9 | Same-day snapshot resolution | HEAD allows lag=0 | **FAIL** if `resolved_snapshot >= trade_date` | **High** | Strict timing gate |
| G10 | Unsupported S1 vs superset envelope | Panel stamp 0.20/1.0 | FAIL when `requested dvol > stamp dvol` | **High** | Artifact envelope check |
| G11 | Superset coverage gap | Last snapshot ad hoc | Full-history check for canonical config | **High** | Full-history coverage |
| G12 | Missing/new ticker silent exclusion | Eligibility in code | Explicit exclusion counts | **Med** | Exclusion summary |
| G13 | Static CSV used as trading universe | Docs warn | Audit note only | **Low** | Doc note |
| G14 | Quantile (superset build) vs rank (S1) | Different algorithms | S1 reference uses rank pct | **Low** | Reference semantics |
| G15 | Early `window_shortfall` | 0 in production | WARN when shortfall > 0 | **Low** | WARN policy |
| G16 | Historical snapshot changes with later weekly rows | Not tested | Future-invariance recomputation test | **High** | Future-invariance check |
| G17 | Tautological weekly filter-only check | Prior design flaw | Full independent recompute per § design memo §7 | **High** | Replace filter/assert design |
| G18 | Sample discovery uses `trade_date = S` | Resolves to snapshot before S | Map target snapshot S → trade date T with `resolve(T)=S` | **Med** | Discovery algorithm fix |
| G19 | Unparseable mixed date types | Ambiguous test plan | Explicit parse/normalize contract | **Med** | Date parsing gate |
| G20 | Source-of-truth contract stale (`<=`) | `surface_engine_data_contract.md` | Update in same commit as S1 change | **High** | C7.2 contract sync |

---

## 7. Recommended C7 scope

### C7 should implement (next commits, not this task)

1. **Pure audit module** — artifact validation, independent reference, comparison to `step1_get_universe`.
2. **Standalone CLI** — `scripts/audit_pit_universe.py` with sample/discover modes; markdown report.
3. **Unit + CLI tests** — synthetic cases in gap matrix §3.
4. **One substantive production-panel audit report** — normal date, boundary date, missing/new case if present.

### C7 should not implement

- Changes to S2–S8, signal logic, backtest runs, Sharpe claims
- Full panel rebuild (C4 scope)
- Wiring into `refresh_weekly_inputs.py validate` (C3 after C8)
- A4 feature validation

### Approved single S1 production change (C7.2 only)

C7 may make **exactly one** approved production behavior change:

```text
S1 snapshot resolution:  month_date <= trade_date  →  month_date < trade_date
```

**No other** S1 eligibility, ranking, threshold, schema, or output behavior may change without separate evidence and HD approval.

C7.2 must update in the **same behavior-changing commit**:

```text
src/backtest/pipeline.py
tests/contract/test_step1_universe_contract.py
docs/surface_engine_data_contract.md
docs/agenda/current_sprint.md
src/data/pit_universe_audit.py
tests/unit/test_pit_universe_audit.py
```

`docs/v1_universe_protocol.md` broader rewrite remains **C9**; until then it is temporarily stale and must not remain the accepted implementation contract.

### Canonical policy recommendation

**Superseded by HD amendment (2026-07-11):** implement **prior-week** global snapshot:

```text
global_snapshot_date = max(month_date < trade_date)
```

See [c7_1_pit_universe_design_memo.md](c7_1_pit_universe_design_memo.md) §1. C7.2 updates `step1_get_universe` accordingly.

**Unchanged:** global (not per-ticker) cross-section; per-ticker protocol wording is documentation drift only (C9).

---

## References

- C4 closeout: `docs/sprint_memos/004_c4_liquidity_panel.md`
- S1 contract: `docs/surface_engine_data_contract.md` § S1 (**stale `<=` at HEAD**; C7.2 updates to `<`)
- Sprint blocker #6: `docs/agenda/current_sprint.md` § Closeout blockers
