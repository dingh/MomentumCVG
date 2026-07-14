# Sprint 004 — C8.0 production refresh capability reality map

**Task:** C8.0 — production refresh capability reality map
**Mode:** Audit / reconnaissance only (read-only code + doc audit; no code, test, or external-artifact changes)
**Starting HEAD:** `5eb1c1ec6bf17adcf6f3c529d1917e2470a0d6a9`
**Status of siblings:** C1, C2, C4, C5, C6, C7 accepted; C3 not implemented; C8 not implemented.

> This document is a **reality map**, not a design. C8.1 owns design decisions. Every capability claim below is traced to current code, tests, or accepted evidence. Where documentation and code disagree, the code governs and the disagreement is recorded.

---

## 1. Scope and target question

**Target question:**

> Given either a newly available ORATS week or a retroactive change such as a newly discovered split or a newly added liquid ticker, what parts of the current codebase can already update every affected input artifact correctly, and what capabilities are still missing before the system can publish one complete, internally consistent production input snapshot?

This audit inspected repository code, tests, accepted closeout memos (C4–C7), design memos (C1/C2/C5/C6/C7), and current production-path contracts. It did **not** read or mutate any external production artifact, did not call ORATS, and did not run any producer/audit/backtest against production data.

**One-line answer:** The repository contains a set of **standalone, mostly-overwrite producers plus strong read-only, internally-consistent audits**, but has **no orchestrator, no manifest-writing refresh, no split-history delta detection, no adjusted/spot/surface merge engine, and no freshness/staleness gate**. It can build bounded isolated artifacts and prove those artifacts are internally consistent; it cannot yet detect what changed, repair only the affected slice, merge into canonical state, or certify that one complete current snapshot is internally consistent across layers.

---

## 2. Executive findings

### 2.1 Already sufficient building blocks (reusable in C8)

- **C1 receipt module** (`src/data/input_snapshot.py`): deterministic `snapshot_id`, `build_id`, JSON round-trip, `write_manifest` / `read_manifest`. `IMPLEMENTED_AND_TESTED` (`tests/unit/test_input_snapshot.py`). It is a *library* only — nothing calls `write_manifest`.
- **`--as-of` trading-day resolver** (`src/data/trading_day.py::resolve_as_of_trading_day`): last file-present day ≤ as-of, walk-back. `IMPLEMENTED_AND_TESTED` (`tests/unit/test_trading_day.py`). Caveat: resolves against the **adjusted** root (see §10).
- **Liquidity incremental append + atomic 4-file publication** (`build_liquidity_panel.py`): true one-new-week append with watermark/param guards and staged `os.replace` publication. `IMPLEMENTED_AND_TESTED` (`tests/unit/test_build_liquidity_panel.py`). This is the only producer with a genuine incremental engine and staged publication.
- **Split fetch checkpoint/resume + scoped fail-closed validation** (`fetch_splits.py`): sidecar checkpoint, conflicting-divisor / invalid-divisor / outside-universe fail-closed in scoped mode. `IMPLEMENTED_BUT_NOT_DIRECTLY_TESTED` for checkpoint/resume; validation is `IMPLEMENTED_AND_TESTED`.
- **Split math** (`split_adjuster.py`): correct cumulative future-split factor. `IMPLEMENTED_AND_TESTED` (`tests/unit/test_split_adjuster.py`).
- **Three strong read-only audits** (`audit_adjusted_liquid.py`, `audit_option_surface_artifacts.py`, `audit_pit_universe.py`): schema, math (internal), grain, join, PIT provenance, coverage. Accepted production evidence for C5/C6/C7.
- **Producer safety scoping** (surface `--dry-run`, `--output-root`, `--overwrite` guard; split filtered-mode explicit `--adj-root`; scoped fetch explicit `--out`): bounded isolated runs are safe by construction.

### 2.2 Partial building blocks (exist but not a safe production-refresh contract)

- **Bounded surface production** exists (`--start-date/--end-date/--tickers/--output-root`), but a bounded run over a **canonical filename** with `--overwrite` **replaces the whole file with only the bounded rows** — it is not a merge/append. `PARTIAL / UNSAFE_OR_AMBIGUOUS`.
- **Split `--tickers` "repair"** (`SplitAdjuster.readjust_tickers`): the `--tickers` list is used **only for logging**; it re-processes and overwrites **every** daily parquet in the selected years for the whole `ticker_universe`, non-atomically per file. It is a full overwrite, not a scoped repair. `UNSAFE_OR_AMBIGUOUS`.
- **`liquid_tickers.csv` growth**: recomputed from the full merged panel each build, so it can gain names; it structurally cannot lose names (see §11 Scenario D). No old-vs-new membership comparison exists anywhere. `PARTIAL`.
- **Split fetch is a full re-fetch + full overwrite**: no diff of new/changed/deleted split records vs the accepted file. `PARTIAL`.
- **Audits prove internal consistency, not freshness/lineage across layers**: the adjusted-math audit checks `adj == raw / stored_split_factor` using the *stored* factor; it never recomputes the factor from current split history, so a stale factor after a new split passes. `UNSAFE_OR_AMBIGUOUS` for staleness detection.

### 2.3 Missing blocking capabilities (must exist before a truthful single-current-snapshot claim)

1. **No orchestrator / no executing refresh.** `refresh` (non-dry) returns exit 2; `validate`, `split-audit`, `surface-audit` are exit-2 stubs (`refresh_weekly_inputs.py`). No subprocess wiring.
2. **No manifest is ever written by any run.** `write_manifest` is defined but never called by any producer or CLI. No snapshot receipt is produced by real work.
3. **No split-history delta detection.** Nothing compares old vs new split history or emits changed/new/deleted split records.
4. **No adjusted/spot/surface merge or scoped repair.** Spot and surface are whole-file overwrites; adjusted `--tickers` is a whole-store overwrite. No selected-ticker replacement that preserves other rows in a *merged* canonical artifact.
5. **No cross-layer freshness/staleness gate.** No audit can prove spot/surface were rebuilt after a newly discovered split, nor detect a canonical artifact missing the newest week.
6. **No cross-producer transaction / rollback / lock.** Each producer publishes independently; a mid-run failure leaves a mixed canonical state with no rollback and no concurrency guard.
7. **No spot producer append/merge/exit-contract.** `extract_spot_prices.py` rebuilds only the requested year range and overwrites the whole output; it has no exit-code contract and swallows per-date errors. It is also the only listed producer with **no test at all**.

### 2.4 Non-blocking operational maturity items (may remain outside C8)

Cloud scheduling, notifications, dashboards, distributed execution, remote backups, watermark-scoped audits (explicitly Sprint 005 per agenda), full-universe surface backfill, content fingerprints (Sprint 007 `run_manifest`). These are convenience/platform items, not production-data correctness blockers.

---

## 3. Accepted production roots and artifacts

Roots (immutable during C8.0; inspected only via repository code/docs, never read on disk here):

| Role | Path | Referenced in code |
|------|------|--------------------|
| Raw ORATS ZIPs | `C:/ORATS/data/ORATS_Data` | `paths.RAW_ORATS_ROOT`; `build_liquidity_panel.DEFAULT_DATA_ROOT` |
| Accepted liquidity root | `C:/MomentumCVG_env/input/liquidity` | runbook; `audit_pit_universe` defaults |
| Accepted adjusted-liquid root | `C:/MomentumCVG_env/input/adjusted_liquid` | `paths.DEFAULT_ADJUSTED_LIQUID_ROOT` (downstream default) |
| Stage A cache | `C:/MomentumCVG_env/cache` | `paths.DEFAULT_CACHE_ROOT`; spot/surface defaults |
| Legacy adjusted mirror (maintenance only) | `C:/ORATS/data/ORATS_Adjusted` | `paths.LEGACY_ORATS_ADJUSTED_ROOT`; split default `--adj-root` |

**Path-contract mismatch (recorded, not fixed):** The C1 receipt (`current_sprint.md` and `input_snapshot.py`) models artifacts as **cache-relative filenames under one `cache_dir`** with keys `splits`, `spot_prices`, `liquidity_panel`, `option_surface_meta`, `option_surface_quotes`. But production artifacts live under **three different roots**: liquidity panel + `liquid_tickers.csv` under `input/liquidity`; splits + adjusted chains under `input/adjusted_liquid`; spot + surface under `cache`. The current CLI preview (`DEFAULT_ARTIFACT_REL_PATHS`) also uses `splits_hist.parquet`, but the accepted C5 scoped split file is `splits_hist_liquid.parquet` under `input/adjusted_liquid`. The single-`cache_dir` receptacle cannot represent multi-root artifacts without a schema/convention change. **Deferred to C8.1** (do not migrate schema in C8.0).

---

## 4. Current dependency graph

Legend for each edge: `[code]` directly enforced by code · `[cli]` CLI convention only · `[default]` default path only · `[manual]` manually operated · `[none]` not connected.

### 4.1 Liquidity chain (raw-price based)

```
raw ORATS daily ZIPs (ORATS_Data)
   │ [code] build_liquidity_panel: load_raw_day_from_zip / discover_orats_trading_dates
   ▼
liquidity daily observations (ticker_liquidity_daily_observations.parquet)
   │ [code] aggregate_weekly_liquidity_observations
   ▼
liquidity weekly observations (ticker_liquidity_weekly_observations.parquet)
   │ [code] aggregate_rolling_weekly_panel (12-week rolling)
   ▼
rolling liquidity panel (ticker_liquidity_panel.parquet)
   │ [code] build_liquid_tickers (top-20% dvol qualifier per snapshot)
   ▼
liquid_tickers historical superset (liquid_tickers.csv)
```

All four liquidity outputs are produced **in one process** (`build_panel` → `write_artifacts`), so within a single run the edges are code-enforced. The liquidity chain reads **raw** prices only (`validate_raw_columns` rejects `adj_*`).

### 4.2 Adjusted / spot / surface chain (adjusted-price based)

```
liquid_tickers.csv ──[cli/manual]──┐
                                   ▼
                          scoped split history (splits_hist_liquid.parquet)
                             fetch_splits.py --ticker-universe --out   [cli/manual; requires ORATS token]
                                   │ [cli] apply_split_adjustment.py --splits --ticker-universe --adj-root
                                   ▼
              split-adjusted daily option chains (input/adjusted_liquid/{YYYY}/*.parquet)
                                   │ [default] extract_spot_prices.py --data-root (default adjusted_liquid)
                                   ▼
                          spot-price parquet (cache/spot_prices_adjusted.parquet)
                                   │ [default] precompute_option_surface.py (data-root + spot-db-path)
                                   ▼
                 option-surface A1 (meta) + A2 (quotes)  cache/option_surface_*_weekly_2018_2026.parquet
```

- `liquid_tickers.csv → split history`: **[cli/manual]** — the operator must pass `--ticker-universe` and `--out`; nothing wires it.
- `split history → adjusted chains`: **[cli]** — `apply_split_adjustment` requires `--splits` and (filtered) `--adj-root`; not orchestrated.
- `adjusted chains → spot`: **[default]** — spot default `--data-root` is `DEFAULT_ADJUSTED_LIQUID_ROOT`; but must be run manually.
- `spot + adjusted chains → surface`: **[default]** — surface reads spot DB (`--spot-db-path`) for RV and the adjusted chains via `ORATSDataProvider` for entry/exit spot and chain quotes.

### 4.3 Universe chain

```
liquidity panel (A3) ──[code]──► S1 PIT universe  (src/backtest/pipeline.step1_get_universe, strict prior snapshot <)
```

S1 consumes the panel directly (code-enforced within the runner). `liquid_tickers.csv` is a *precompute superset*, never the trading universe.

### 4.4 Explicit determination — does a split change affect the liquidity panel?

**No.** `build_liquidity_panel` reads **raw** ORATS ZIP columns (`stkPx`, `cBidPx`, …) and explicitly rejects adjusted input (`validate_raw_columns`). Liquidity ranking, the panel, `liquid_tickers.csv`, and therefore S1 are **unaffected** by a split-history change. A newly discovered split affects only: scoped split history → adjusted chains → spot → surface. This isolates split repair to the adjusted/spot/surface chain.

---

## 5. Artifact inventory and grains

| Artifact | Canonical / accepted path | Producer | Consumer(s) | Grain / partition | Primary/dedup key | Date/watermark field | Write behavior | Atomicity | Audit coverage |
|----------|---------------------------|----------|-------------|-------------------|-------------------|----------------------|----------------|-----------|----------------|
| Raw ORATS daily ZIP | `ORATS_Data/{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.zip` | External (ORATS) | liquidity, split-adjust, adj-audit | one file/day | filename date | filename date | external | n/a | inventory side of adjusted audit only |
| Liquidity daily obs | `input/liquidity/ticker_liquidity_daily_observations.parquet` | `build_liquidity_panel` | weekly stage; incremental gap check | (trade_date, ticker) | not enforced (append via `concat`) | `trade_date` (`daily_watermark`) | staged replace | per-file atomic (staged) | not audited directly |
| Liquidity weekly obs | `input/liquidity/ticker_liquidity_weekly_observations.parquet` | `build_liquidity_panel` | panel; PIT audit rolling recompute | (week_end_date, ticker) | not enforced | `week_end_date` (`weekly_watermark`) | staged replace | per-file atomic (staged) | PIT audit (`check_weekly_artifact`) |
| Rolling liquidity panel | `input/liquidity/ticker_liquidity_panel.parquet` | `build_liquidity_panel` | S1; PIT audit | (month_date, ticker) | not enforced | `month_date` (`panel_watermark`) | staged replace | per-file atomic (staged) | PIT audit (grain, metric, provenance) |
| `liquid_tickers.csv` | `input/liquidity/liquid_tickers.csv` | `build_liquidity_panel::build_liquid_tickers` | scoped fetch/adjust scope; surface tickers; PIT superset | one row/Ticker | `Ticker` (unique after `groupby`) | `snapshots_qualified` (count, not a date) | staged replace | per-file atomic (staged) | PIT superset coverage |
| Scoped split history | `input/adjusted_liquid/splits_hist_liquid.parquet` | `fetch_splits` (+ `.checkpoint.parquet` sidecar) | `apply_split_adjustment`; split audit | (ticker, split_date, divisor) | `(ticker, split_date)` dedup; conflicts fail-closed | `split_date` (`max_split_date` candidate) | full overwrite | direct `to_parquet` (non-atomic); sidecar checkpoint | split audit (`audit_split_file`) |
| Adjusted daily option chains | `input/adjusted_liquid/{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.parquet` | `apply_split_adjustment` / `SplitAdjuster` | `ORATSDataProvider`, spot, surface | one file/day; rows per (ticker,expiry,strike) | not enforced in file | `trade_date` = file date | direct per-file `to_parquet`, skip-existing default | per-file direct write (non-atomic) | `audit_adjusted_liquid` (inventory, structure, sampled math) |
| Spot-price parquet | `cache/spot_prices_adjusted.parquet` | `extract_spot_prices` | `SpotPriceDB`, surface RV | (date, ticker) | not enforced (grouped per day) | `date` | full overwrite of requested year range | direct `to_parquet` (non-atomic) | none (indirectly via surface settlement) |
| Option-surface A1 (meta) | `cache/option_surface_meta_weekly_2018_2026.parquet` | `precompute_option_surface` | S3 assembly; surface audit | (ticker, entry_date) | `(ticker, entry_date)` (audited, not producer-enforced) | `entry_date` | full overwrite (guarded) | direct `to_parquet` (non-atomic) | `audit_option_surface_artifacts` (grain, invariant, settlement) |
| Option-surface A2 (quotes) | `cache/option_surface_quotes_weekly_2018_2026.parquet` | `precompute_option_surface` | S3 assembly; surface audit | (ticker, entry_date, strike, side) | `(ticker, entry_date, expiry_date, strike, side)` (audited) | `entry_date` | full overwrite (guarded) | direct `to_parquet` (non-atomic) | surface audit (grain, join) |
| PIT-universe audit report | `docs/tmp/*` (operator-chosen `--output-report`) | `audit_pit_universe` | human | markdown | n/a | n/a | write report | direct write | self |
| Split audit report | operator-chosen `--report-path` | `audit_adjusted_liquid` | human | markdown | n/a | n/a | write report | direct write | self |
| Surface audit report | operator-chosen `--output-report` | `audit_option_surface_artifacts` | human | markdown | n/a | n/a | write report | direct write | self |
| Input snapshot manifest | `cache/manifests/input_snapshot_{snapshot_id}.json` (spec) | **none (never written)** | none | JSON receipt | `snapshot_id` (deterministic) | `as_of_resolved_trading_day` | `write_manifest` exists, uncalled | atomic write not used in practice | none |

**Primary-key note:** Except for scoped split history (`(ticker, split_date)` dedup + conflict fail) and the *audited* A1/A2 grains, no producer **enforces** a primary key at write time; keys are audited after the fact. Liquidity/spot outputs have no producer-side dedup. Marked ambiguous accordingly.

---

## 6. Producer capability matrix

Cells: **Yes / Partial / No / Ambiguous**. "Canonical" means updating the accepted production artifact in place, correctly preserving unaffected content.

| Producer | Full backfill | New-date append | Selected-ticker historical replacement | Existing-row merge | Deduplication | Atomic publication | Idempotent rerun | Resume/checkpoint | Exit-code contract |
|----------|--------------:|----------------:|---------------------------------------:|-------------------:|--------------:|-------------------:|-----------------:|------------------:|-------------------:|
| `build_liquidity_panel.py` | Yes | Yes (one week) | No | Partial | No | Partial | Partial | No | Yes (0/1) |
| `fetch_splits.py` | Yes | No | No | No | Yes (exact) | No | Partial | Yes | Partial |
| `apply_split_adjustment.py` | Yes | Partial | Ambiguous | No | No | No | Yes (skip-existing) | Partial | No |
| `extract_spot_prices.py` | Yes | No | No | No | No | No | Yes (overwrite) | No | No |
| `precompute_option_surface.py` | Yes | No | No | No | No | No | Yes (overwrite) | No | Yes (0/2) |

### 6.1 Explanations of Partial / Ambiguous cells

**`build_liquidity_panel.py`**
- *New-date append = Yes*: `run_incremental` appends exactly one completed week (`>panel_watermark`); fails on 0 or >1 new weeks; gap guard on daily watermark.
- *Existing-row merge = Partial*: it `concat`s new daily/weekly/panel rows onto prior frames; it does not recompute or merge prior snapshots. It cannot repair an older week (backfill overwrite required).
- *Deduplication = No*: relies on strict watermark ordering, not a dedup key. `test_run_incremental_appends_week_with_date_trade_dates` covers the append; no duplicate-injection test.
- *Atomic publication = Partial*: `write_artifacts` stages all four files then `os.replace` each — atomic per file but **not one cross-file transaction**; a crash between replaces leaves a mixed old/new set. Covered by `test_write_artifacts_commits_all_outputs` (happy path only).
- *Idempotent rerun = Partial*: rerunning incremental with the same `--end-date` correctly fails (`Nothing to append`, exit 1, artifacts unchanged — accepted C4 evidence); backfill rerun overwrites deterministically.

**`fetch_splits.py`**
- *New-date append = No*: fetches full history per ticker via `fetch_all_splits`, not a date-bounded delta.
- *Deduplication = Yes*: exact-duplicate drop; `(ticker, split_date)` conflicts fail-closed in scoped mode (`test_duplicate_conflicting_split_rows_fail`, `test_duplicate_identical_split_rows_are_deduped`).
- *Idempotent rerun = Partial*: final write always **overwrites** `--out` with the freshly fetched+validated set; no diff of new/changed/deleted vs the accepted file. Resume changes work done, not the final content.
- *Resume/checkpoint = Yes* (code): sidecar `.checkpoint.parquet`, saves after each ticker, resumes on re-run — but **not directly unit-tested** (`test_fetch_splits_cli` mocks the fetcher).
- *Exit-code contract = Partial*: uses `sys.exit(message)` on failure (non-zero via string) and returns `None` on success; not a clean 0/1/2 integer contract.

**`apply_split_adjustment.py` / `SplitAdjuster`**
- *New-date append = Partial*: default `run` skips existing output files (`if not overwrite and out_path.exists()`), so a genuinely new raw date without an adjusted file will be produced; but it does not *know* a date is new — it globs all ZIPs in the year and skips those already present. Not watermark-driven.
- *Selected-ticker historical replacement = Ambiguous / UNSAFE*: `readjust_tickers(tickers=…)` uses the `tickers` argument **only for a log line**. It then re-processes **every** ZIP in the selected years with `overwrite=True`, filtered to `self._ticker_universe` (the whole C4 universe), rewriting the entire daily parquet each time. So `--tickers NVDA` does **not** scope the write to NVDA; it overwrites the whole store (for the selected years). Unaffected tickers are preserved only because each daily file is fully rewritten with current factors. There is **no** narrow per-ticker repair.
- *Existing-row merge = No*: each daily parquet is fully rewritten; there is no merge of a ticker slice into an existing file.
- *Atomic publication = No*: `_process_single_zip` writes each parquet with a direct `df.to_parquet(out_path)` — no temp+rename. An interrupt mid-write can corrupt one file; an interrupted repair leaves a mixed factor state across days.
- *Idempotent rerun = Yes*: default skip-existing makes re-runs no-ops; `--overwrite` reproduces identical output.
- *Repair scope*: `--years` restricts the file set; `--tickers` does **not** restrict the write set (only logs). So years and tickers do **not** both restrict scope.

**`extract_spot_prices.py`**
- *Canonical row grain*: `(date, ticker)`, one row per ticker per day (`groupby('ticker').first()`).
- *Explicit dates / selected tickers = No*: only `--year` / `--start-year` / `--end-year`; no date-list or ticker filter.
- *New-date append / merge / dedup = No*: it scans the requested year range, builds a fresh DataFrame, and `to_parquet(output_path)` **replacing the whole file**. Running `--year 2026` replaces the entire multi-year spot DB with only 2026 — a silent footgun.
- *Atomic publication = No*: direct `to_parquet`.
- *Exception → success status*: `extract_spot_prices_for_date` catches all exceptions per date and returns `[]`; `main` never returns a non-zero code (returns `None`), even on "No spot prices extracted" (it just `return`s). So **an exception can still yield exit 0**, and **partial failures silently produce an incomplete output**.
- *Tests*: **none** — no `test_extract_spot_prices*.py` exists; only `SpotPriceDB` (the reader) is tested (`test_spot_price_db.py`).

**`precompute_option_surface.py`** (A1 and A2)
- *Canonical grains*: A1 `(ticker, entry_date)`; A2 `(ticker, entry_date, strike, side)`.
- *Output naming*: `option_surface_{meta,quotes}_{frequency}_{start_year}_{end_year}.parquet`. Filename is keyed by frequency + declared year bounds, **not** by the actual `--start-date/--end-date` window.
- *Date/ticker scoping*: `--start-date/--end-date`, `--tickers`/`--tickers-file`; weekly schedule with a 21-day successor tail for strict expiry.
- *Existing-output guard*: `check_overwrite_guard` returns exit 2 if either output exists without `--overwrite`.
- *Overwrite / append / selected-ticker repair / merge / dedup = No*: with `--overwrite`, it writes the whole `meta_df`/`quotes_df` to the target path, **replacing** it. A bounded run (few tickers / short window) pointed at the canonical `..._2018_2026.parquet` filename with `--overwrite` would **replace the whole canonical file with only the bounded rows** — dropping every other ticker/date. There is no merge/append/dedup.
- *Paired A1/A2 atomicity = No*: `meta_df.to_parquet(meta_path)` then `quotes_df.to_parquet(quotes_path)` are two sequential direct writes; a failure between them leaves A1 without matching A2 (or a stale mismatch). Neither write is atomic.
- *Idempotent rerun = Yes (with `--overwrite`)*: deterministic ordering (joblib preserves trade-date order) yields identical output; without `--overwrite`, the guard blocks (exit 2).
- *Bounded ≠ merge*: `--start-date/--end-date/--tickers` support means the producer can create **bounded isolated** outputs (safely, to a separate `--output-root`, as C6.4 did). It does **not** mean it can safely update the existing canonical artifact. Bounded production is not incremental merge.

---

## 7. Detailed producer findings

### 7.1 Liquidity (`build_liquidity_panel.py`)

- **Incremental reads:** the three existing artifacts (`validate_incremental_artifacts`) → watermarks (max `trade_date`, `week_end_date`, `month_date`) and stored build params (`lookback_weeks`, `min_valid_quote_weeks`, `dte_min/max`, `dvol_top_pct`, `spread_bot_pct`). Param mismatch → FAIL (`_assert_build_params_match`).
- **History recomputed:** incremental recomputes only the **one new snapshot's** rolling window (from the merged weekly obs); it appends, it does not recompute prior snapshots.
- **New-week identification:** completed ISO week whose `week_end_date > panel_watermark`.
- **No new week:** raises `LiquidityPanelError("Nothing to append…")` → exit 1, artifacts unchanged (accepted C4 evidence).
- **>1 missing week:** refuses (`Incremental expects one new week; found N`) → must backfill. So it does **not** support multiple missing weeks in one incremental run.
- **Repair of older week:** not supported by incremental; requires a `backfill` window that **overwrites** the affected span.
- **Atomic publication of all four:** staged dir + per-file `os.replace` (Partial — see §6.1). `liquid_tickers.csv` is regenerated from the whole merged panel each run.
- **liquid_tickers gain/lose names:** can **gain** (recomputed from full panel); cannot **lose** under append (panel is append-only and qualification is cumulative). No membership diff exists.
- **Old vs new universe comparison:** none anywhere.

### 7.2 Split fetch (`fetch_splits.py`)

- **Full vs bounded:** full per-ticker history for the whole fetch universe; not date-bounded.
- **Checkpoint/resume:** sidecar `<name>.checkpoint.parquet` (scoped) or output-as-checkpoint (legacy); saves after each ticker; resumes by counting `nunique()` tickers already present. Untested directly.
- **Compare old vs new history:** none. No changed/new/deleted split record surfacing.
- **Conflicts fail closed:** yes, in scoped mode (conflicting divisor, invalid divisor, outside-universe → `_fail`). Legacy mode WARNs and writes.
- **Can a failed fetch replace the accepted file?** In scoped mode: **no** — validation runs before the single final `to_parquet`, and interruptions leave only the sidecar. But a *completed* fetch that returns different data **fully overwrites** the accepted file (no diff/guard); there is no protection against ORATS returning fewer/altered rows.
- **Token:** requires `--token`/`ORATS_API_TOKEN`; not runnable in C8.0 and out of scope to invoke.

### 7.3 Split adjustment (`apply_split_adjustment.py` / `SplitAdjuster`)

- **Skip-existing:** default `run` skips files that already exist; not a repair engine — a stale-but-present adjusted file is skipped, never re-derived.
- **Selected-ticker mode rewrites full available history:** `readjust_tickers` overwrites **all** daily parquets (selected years) for the whole ticker universe; `--tickers` is logging only.
- **`--tickers` implies overwrite:** yes (the CLI docstring and code set `overwrite=True`), but the overwrite is store-wide, not ticker-scoped.
- **Ticker-scoped repair preserves unaffected tickers per daily parquet:** yes in the sense that each rewritten daily file re-includes all universe tickers with current factors; but this is because the *whole file/store is rewritten*, not because a scoped slice is merged.
- **Atomic output replacement:** no (direct `to_parquet` per file).
- **Interrupted repair → mixed state:** yes — some daily files carry new factors, others old; no transaction/rollback. Additionally a single interrupted file write can corrupt one parquet.
- **Years and tickers both restrict scope:** no — years restrict; tickers do not.

### 7.4 Spot (`extract_spot_prices.py`)

See §6.1. Grain `(date, ticker)`; year-range only; whole-file overwrite of the requested range (replaces entire output); no append/merge/dedup; non-atomic; exceptions swallowed per date; no non-zero exit on empty/partial; **no tests**. This is the weakest producer for a production-refresh contract.

### 7.5 Option surface (`precompute_option_surface.py`)

See §6.1. Separately for A1 and A2: grains `(ticker, entry_date)` / `(ticker, entry_date, strike, side)`; filename keyed by frequency+year bounds; scoping via start/end date + tickers; overwrite guard prevents accidental clobber but `--overwrite` replaces the whole file; no append/merge/dedup/selected-ticker repair; A1/A2 written as two non-atomic sequential writes (failure between leaves unpaired files); idempotent with `--overwrite`. Bounded production is safe only to a **separate `--output-root`** (the pattern accepted in C6.4). Producer-level dedup was intentionally not implemented (C6.4 found no duplicates); audits catch duplicates after the fact.

---

## 8. Audit capability matrix

| Aspect | `audit_adjusted_liquid.py` | `audit_option_surface_artifacts.py` | `audit_pit_universe.py` |
|--------|----------------------------|-------------------------------------|-------------------------|
| Inputs | raw-root, adj-root, splits, ticker-universe, years, report-path | meta-path, quotes-path, frequency, data-root, window, tickers | panel, weekly, liquid_tickers, sample dates / discover |
| Supported scope | by **year(s)** | by date window + ticker subset | by explicit sample date and/or discovered samples |
| Full-history vs sampled | inventory + structural = **all files in year**; math = **sampled** (default 10 files × 20k rows, seed 57) | contract checks on the **filtered window**; c6.4 loads bounded artifacts | artifact checks full-panel; rolling recompute **bounded** (≤20 tickers × 3 samples); full-history = superset coverage across 477 snapshots (not per-row recompute) |
| PASS/WARN/FAIL | PASS / PASS WITH WARNINGS / FAIL | PASS / WARN / FAIL | PASS / WARN / FAIL |
| Exit codes | FAIL → `SystemExit(1)`; else implicit 0 (WARN not distinguished) | 0 pass, 1 FAIL (or WARN with `--fail-on-warn`), 2 usage | 0 pass, 1 FAIL/strict-WARN, 2 usage |
| Report output | markdown (`--report-path`) | markdown (`--output-report`) | markdown (`--output-report`) |
| Mutates input? | no | no (explicit read-only guarantee) | no (output-path aliasing rejected) |
| Audit only changed dates? | no (year granularity) | window-scoped, but no "changed-only" | sample-scoped, not changed-only |
| Audit only changed tickers? | no (universe-wide) | yes via `--sample-tickers` | via checked-ticker selection |
| Establish completeness? | Partial — inventory detects raw dates lacking an adjusted parquet **for audited years**; cannot know weeks whose raw ZIP is absent | No — checks entries lie on schedule (misaligned = WARN); no missing-week detection | No — checks samples internally; no "latest week present" reference |
| Detect stale-but-valid downstream rows? | **No** — math uses **stored** `split_factor`; never recomputes from current split history | **No** — validates internal schema/invariant/join; never recomputes surface from spot/chains | **No** — internal PIT consistency only |
| Runtime/scaling evidence | C5.10D full production audit PASS (sampled math) | C6.4 bounded 5 tickers × 13 weeks | C7.4R full-history coverage, 545.6 s runtime |

### 8.1 Can existing audits prove spot/surface rows were rebuilt after a newly discovered historical split?

**No.** Three independent reasons:
1. `audit_adjusted_liquid`'s math check verifies `adj_price == raw_price / split_factor` using the **stored** `split_factor` column. It does **not** recompute the cumulative factor from the current split history and compare, so a daily parquet still carrying the pre-split factor is internally consistent and **passes**.
2. `extract_spot_prices` has **no audit** at all; nothing verifies spot values were re-derived from re-adjusted chains.
3. `audit_option_surface_artifacts` validates A1/A2 **internal** structure and settlement completeness but never recomputes surfaces from spot/chains, so a stale surface with valid structure **passes**.

Adjusted-chain math and surface structure passing separately does **not** imply the downstream layers were rebuilt after the split.

### 8.2 Can existing audits detect a canonical artifact that is internally valid but missing the newest week?

Answered per layer:
- **Adjusted chains:** Partial. Inventory compares raw ZIP dates vs adjusted parquet dates **within audited years**, so a raw date lacking an adjusted file is flagged FAIL — but only if the raw ZIP exists and that year is audited. It cannot flag a week whose raw ZIP has not arrived.
- **Spot:** No. No audit exists; nothing compares spot dates against an expected calendar.
- **Surface:** No. Date-alignment checks entries lie on the schedule (misaligned → WARN); there is no check that the artifact **contains** every scheduled week, so a surface missing the newest week passes.
- **PIT universe:** No. It validates internal PIT consistency of whatever panel is supplied; it has no external reference of "latest expected week," so a panel missing the newest week passes.

---

## 9. C1 manifest reality

- **C1 module capability** (`input_snapshot.py`): deterministic `snapshot_id` (sha256[:16] of `schema_version`, `as_of_resolved_trading_day`, `data_source`, `artifacts`, `params`), `build_id`, strict JSON round-trip, `write_manifest`/`read_manifest`. Tested (`test_input_snapshot.py`). Artifact values are **cache-relative path strings**; backslashes normalized.
- **Current C2/C8 caller behavior:** `refresh_weekly_inputs.py` computes `snapshot_id` **only for a display-only preview** in `render_plan` (`_snapshot_id_preview`). It never constructs an `InputSnapshotManifest`, never calls `write_manifest`, never checks artifact existence or content.
- **Path strings previewed:** `DEFAULT_ARTIFACT_REL_PATHS` = `splits_hist.parquet`, `spot_prices_adjusted.parquet`, `ticker_liquidity_panel.parquet`, `option_surface_meta_weekly_2018_2026.parquet`, `option_surface_quotes_weekly_2018_2026.parquet`.
- **Do previewed paths match accepted production roots?** Partially:
  - `spot_prices_adjusted.parquet`, `option_surface_*` → live under `cache` ✓ (as bare filenames).
  - `ticker_liquidity_panel.parquet` → accepted under `input/liquidity`, not `cache`.
  - `splits_hist.parquet` → **mismatch**: accepted C5 file is `splits_hist_liquid.parquet` under `input/adjusted_liquid`.
  - The receipt assumes a single `cache_dir`; production spans three roots (§3). The schema cannot represent multi-root artifacts without a convention/schema change.
- **Artifact existence checked?** No. **Content fingerprinted?** No (by design — `snapshot_id` is a logical identity, not a byte hash).
- **Reports / validation status populated?** No — `reports`, `overall_status`, `blocking_failures` are C3+ fields; nothing sets them.
- **Does any executed refresh write a manifest today?** No — no code path calls `write_manifest`.
- **Can failed/partial executions produce a manifest today?** No — no execution writes any manifest at all.

**Recorded mismatch (decision deferred to C8.1):** the single-`cache_dir` receipt vs three-root production reality, and the `splits_hist.parquet` vs `splits_hist_liquid.parquet` path drift. Do not migrate the schema in C8.0.

---

## 10. C2 refresh CLI reality

`scripts/refresh_weekly_inputs.py` + `tests/unit/test_refresh_weekly_inputs_cli.py`.

| Aspect | Reality |
|--------|---------|
| Implemented commands | `plan` (exit 0), `refresh --dry-run` (exit 0, prints plan + dry-run banner) |
| Stub commands | `validate` → exit 2; `split-audit` → exit 2 (points to `audit_adjusted_liquid.py`); `surface-audit` → exit 2; `refresh` non-dry → exit 2 ("not implemented until C8") |
| Exit-code behavior | 0 = plan/dry-run; 2 = usage error, unresolved as-of, or stub; 1 reserved/unused |
| Dry-run behavior | reuses `render_plan`; **no** subprocess, **no** manifest, **no** `generate_build_id` persistence (`test_...C14` asserts no subprocess) |
| Date resolution source | `resolve_as_of_trading_day(as_of, orats_adj_root)` — walks back over the **adjusted** root's daily parquet presence |
| Path defaults | `--cache-dir` = `cache`; `--orats-adj-root` = `DEFAULT_ADJUSTED_LIQUID_ROOT` |
| Mode argument behavior | `--mode incremental\|backfill\|repair` parsed, echoed in plan, WARN lines for missing date-window / sample-tickers — **display-only**; no enforcement |
| Skip-flag behavior | `--skip-surface` / `--skip-splits` parsed; only add a plan note; no effect on execution (there is none) |
| Subprocess capability | none |
| Manifest-write capability | none |
| Temporary C2/C9 copy | `render_plan` contains TEMP scaffolding ("(no execution)", "Provisional", bracket notes, "deferred to C3–C8") flagged for C9 blocker #13 removal |

**Do `--mode` values change execution?** No. All three modes only change displayed text; there is no execution to change. `incremental`/`backfill`/`repair` are display-only.

**`--as-of` resolves against the adjusted root — can it discover a newly available raw date not yet adjusted?** **No.** `build_cli_context` calls `resolve_as_of_trading_day(args.as_of, args.orats_adj_root)`, and `orats_adj_root` defaults to `DEFAULT_ADJUSTED_LIQUID_ROOT`. The resolver returns the latest day for which an **adjusted** parquet exists. A newly available **raw** ORATS week that has not yet been split-adjusted has **no** adjusted parquet, so the resolver cannot see it — the CLI would resolve to the last already-adjusted day and never trigger work for the new raw week. (Not fixed in C8.0; recorded for C8.1 as the "raw vs adjusted date discovery" decision.)

---

## 11. Scenario analysis (what happens today with existing scripts, manually operated)

### Scenario A — ordinary new ORATS week (new raw ZIPs; no membership change; no split change)

| Step | Today | Class |
|------|-------|-------|
| Extend liquidity | `build_liquidity_panel --mode incremental --end-date <week>` appends one week atomically(ish) | Supported |
| Adjust only new raw dates | `apply_split_adjustment` (filtered) skips existing, writes the new day's parquet(s); but must be run manually with the right flags | Partial (works via skip-existing; not watermark-aware, not orchestrated) |
| Append spot | `extract_spot_prices` cannot append — running a year replaces the whole year-range output; to add one week you must re-extract a full range and overwrite | Partial/Unsafe |
| Append A1/A2 surface rows | `precompute_option_surface` cannot append — bounded run to the canonical filename overwrites the whole file; safe append is not possible | Unsupported |
| Audit final state | adjusted + surface + PIT audits run read-only, but cannot prove freshness/completeness of the newest week | Partial |
| Write one accurate manifest | no code writes a manifest | Unsupported |

**Classification: partially supported.** **First blocking capability:** no orchestrated executing refresh; and even manually, **spot and surface have no append/merge** — the first hard technical wall is spot/surface incremental update (whole-file overwrite is the only option). A close second is that `--as-of` resolves against the adjusted root and cannot even discover the new raw week (§10).

### Scenario B — newly discovered split for an existing liquid ticker

| Need | Today |
|------|-------|
| Detect the split-history delta | **Not implemented** — `fetch_splits` full-overwrites; no diff |
| Identify the affected ticker | Not implemented (operator must know) |
| Re-adjust its historical chains | `apply_split_adjustment --tickers T --overwrite` — but this overwrites the **whole store** (years), not just T; non-atomic |
| Repair its complete spot history | `extract_spot_prices` must re-extract a full year range and overwrite; cannot target one ticker |
| Repair its complete A1 history | surface `--tickers T --overwrite` to canonical filename would **drop all other tickers**; safe path is a separate output root, which is not canonical |
| Repair its complete A2 history | same as A1 |
| Preserve unaffected rows | adjusted: yes only because the whole store is rewritten; spot/surface: **no** if scoped, because overwrite drops others |
| Audit stale downstream rows | **No** — audits cannot detect stale factors/spot/surface (§8.1) |
| Publish only after complete repair | no transaction/gate exists |

**Classification: unsupported for a safe, scoped, verified repair.** The pieces to *recompute everything* exist (fetch → full re-adjust → full re-extract → full re-precompute), but there is no scoped repair, no merge, no delta detection, and no audit that proves the repair actually propagated.

### Scenario C — newly added ticker in `liquid_tickers.csv`

| Need | Today |
|------|-------|
| Detect the ticker is newly added | Not implemented (no membership diff) |
| Fetch its split history | `fetch_splits` re-fetches the whole universe (includes it) and overwrites |
| Build its adjusted history | `apply_split_adjustment` with the updated universe rewrites the store (the new ticker's rows appear only if the whole store is re-run, since existing daily files are skipped and were written without it) |
| Build its spot history | only via full re-extract + overwrite |
| Build its surface history | only via full re-precompute + overwrite |
| Merge into canonical artifacts | **No merge** — each layer is whole-file/whole-store overwrite |
| Preserve unaffected rows | adjusted: existing daily files are **skipped by default**, so a new ticker will **not** appear unless `--overwrite`/full re-run; spot/surface overwrite the whole output |

**Classification: unsupported for incremental add.** Adding one ticker effectively forces a full rebuild of the adjusted/spot/surface chain; there is no additive merge.

### Scenario D — removed ticker from the historical superset

- **Can removal occur under current C4 construction?** Effectively **no**. `build_liquid_tickers` scans the whole append-only panel and counts cumulative `snapshots_qualified`; once a ticker qualified in any snapshot it remains in `liquid_tickers.csv`. Under incremental append the panel is never pruned, so a name cannot drop out. Only a full backfill over a *narrower window* could omit it — an intentional rebuild, not routine removal.
- **Deletion/retention policy for adjusted / spot / surface / manifest identity?** **None exists.** No code deletes or prunes any artifact for a removed ticker. Do not assume a cleanup policy — there is none.

**Classification: not applicable under current construction; no deletion policy defined.**

### Scenario E — interrupted refresh (failure after some producers completed)

- **Which artifacts may already be canonical:** whichever producers finished. Because there is no orchestrator, each producer publishes to its own canonical path independently and immediately.
- **Which writes use staging:** only `build_liquidity_panel::write_artifacts` (staging dir + per-file `os.replace`).
- **Which writes are direct:** split history, every adjusted daily parquet, spot parquet, A1, A2 — all direct `to_parquet` (non-atomic).
- **Which steps can resume:** `fetch_splits` (checkpoint); `apply_split_adjustment` default run (skip-existing) resumes by not redoing finished files. Others do not resume.
- **Which steps are idempotent:** liquidity backfill, split-adjust skip-existing, spot overwrite, surface overwrite (with `--overwrite`).
- **Rollback:** none anywhere.
- **Could a successful manifest be written incorrectly?** Not today (no manifest is written at all). If C8 wires it naively, a manifest could be written over a partially-published set — this is a C8.1 risk to design against.
- **Local lock:** none. No concurrency protection on any artifact.

**Classification: an interrupted multi-step refresh leaves a mixed canonical state with no rollback and no lock.**

### Scenario F — rerun identical inputs (per component)

| Component | Second identical run |
|-----------|----------------------|
| `build_liquidity_panel` incremental | fails "Nothing to append" (exit 1), artifacts unchanged |
| `build_liquidity_panel` backfill | recomputes and overwrites; deterministic identical output |
| `fetch_splits` | re-fetches (or resumes) and overwrites with the same validated content |
| `apply_split_adjustment` (default) | skips existing files (no-op) |
| `apply_split_adjustment --overwrite` | recomputes and overwrites identically |
| `extract_spot_prices` | recomputes the range and overwrites identically |
| `precompute_option_surface` (no `--overwrite`) | overwrite guard → exit 2 (does nothing) |
| `precompute_option_surface --overwrite` | recomputes and overwrites identically |
| `snapshot_id` | identical if the identity fields (paths + resolved day + params) are unchanged |
| `build_id` | **new** each execution (UTC timestamp + hash) |

No component **duplicates rows** on rerun (they overwrite or refuse). There is no single global idempotency guarantee because there is no single refresh.

---

## 12. Atomicity, interruption, and idempotency (summary)

- **Atomic publication:** only liquidity (per-file staged replace; not cross-file transactional). Split history, adjusted chains, spot, A1, A2 all use **direct non-atomic** `to_parquet`.
- **Cross-artifact transaction:** none. A1/A2 are two separate writes with no pairing guarantee.
- **Interruption:** can corrupt a single in-progress parquet (direct write) and can leave mixed old/new canonical state (adjusted repair; A1 without A2; some liquidity files replaced and others not on crash mid-loop).
- **Idempotency:** per-component only (see §11 F); no global snapshot idempotency.
- **Concurrency:** no lock file, no PID guard, nothing prevents two overlapping runs from racing on the same canonical path.
- **Rollback:** none.

---

## 13. Test and evidence coverage

| Capability | Implementation exists? | Direct unit/contract test? | Production evidence? | Remaining proof gap |
|------------|-----------------------|----------------------------|----------------------|---------------------|
| Liquidity incremental append | Yes (`run_incremental`) | Yes (`test_build_liquidity_panel::test_run_incremental_appends_week_with_date_trade_dates`; several `validate_incremental_*`) | C4 memo (one-week append PASS) | Multi-missing-week / older-week repair not supported (by design) |
| Liquidity incremental/backfill equivalence | Yes | Not a dedicated unit test | C4 memo ("Incremental ≡ full backfill" PASS) | Equivalence proven by accepted operator run, not by an automated test |
| Atomic liquidity four-file publication | Partial (staged, per-file replace) | Yes (`test_write_artifacts_commits_all_outputs`, happy path) | C4 | No crash/partial-replace test; not cross-file transactional |
| Split-fetch checkpoint/resume | Yes (sidecar) | **No** (fetcher mocked in `test_fetch_splits_cli`) | C5.7 report | Resume path unverified by tests |
| Split-history delta detection | **No** | No | No | Entire capability missing |
| Selected-ticker split repair | Ambiguous (`readjust_tickers` overwrites whole store) | **No** (`test_split_adjuster` covers math only) | C5 repair command documented | No scoped-repair test; `--tickers` scoping is illusory |
| Preservation of unaffected tickers during repair | Partial (whole-file rewrite) | No | No | No test asserts other tickers preserved |
| Spot append | **No** | **No** (no `test_extract_spot_prices*`) | No | Producer untested; only overwrite exists |
| Spot selected-ticker replacement | **No** | No | No | Not supported |
| Spot deduplication | **No** | No | No | No dedup key |
| Spot atomic write | **No** | No | No | Direct overwrite |
| Surface bounded production | Yes | Yes (`test_precompute_option_surface_cli`: dry-run, overwrite guard, output-root, scoping) | C6.4 bounded smoke PASS | Bounded ≠ canonical merge |
| Surface append | **No** | No | No | Not supported |
| Surface selected-ticker replacement | **No** | No | No | Overwrite drops other tickers |
| Surface A1/A2 paired atomicity | **No** | No | No | Two non-atomic writes |
| Surface duplicate detection | Audit-side yes (`check_quote_grain`/`check_meta_grain`) | Yes (`test_option_surface_contract`) | C6.4 (0 duplicates) | Producer does not dedup; audit only detects |
| PIT strict prior snapshot | Yes (`step1_get_universe` `<`) | Yes (`test_step1_universe_contract`) | C7.4R PASS | — |
| PIT full-history superset coverage | Yes (`check_full_history_superset_coverage`) | Yes (`test_pit_universe_audit`) | C7.4R (477 snapshots, 0 missing) | Coverage, not per-row rolling recompute (accepted) |
| Manifest deterministic identity | Yes (`compute_snapshot_id`) | Yes (`test_input_snapshot`) | n/a | — |
| Manifest creation after real refresh | **No** (never called) | No | No | No refresh writes a manifest |
| Refresh failure propagation | **No** (no execution) | CLI stubs tested (exit 2) | No | No real steps to propagate from |
| Refresh idempotency | **No** (no refresh) | No | No | Per-component only |
| Refresh concurrency protection | **No** | No | No | No lock |

Cross-cutting: the C6/C7 full pytest suite was not re-run for their closeouts (blocker #11 remains open); accepted C5/C6 evidence is **sampled/bounded** (C5 math seed 57; C6.4 five tickers × 13 weeks). Do not read bounded PASS as full-universe/full-history certification.

---

## 14. Blocking capability gaps (before a correct production refresh is possible)

1. **Executing orchestrator + manifest write.** A `refresh` that actually runs the wired steps per mode and writes a snapshot manifest (only on a complete, consistent publication).
2. **Raw-vs-adjusted date discovery.** `--as-of` must be able to see a newly available raw week (currently blind to it — §10).
3. **Split-history delta detection + safe update.** Diff old vs new split history; surface changed/new/deleted records; fail closed on conflict; never overwrite the accepted file with an unverified fetch.
4. **Scoped, atomic, merge-capable spot and surface updates.** Append/merge a date or ticker slice into the canonical artifact without dropping the rest; atomic (temp+rename) publication; A1/A2 as a paired transaction.
5. **Scoped adjusted repair.** Re-adjust only the affected tickers/dates and merge into daily parquets atomically, instead of a whole-store non-atomic overwrite.
6. **Cross-layer freshness/staleness audit.** Prove the adjusted `split_factor` matches current split history, that spot/surface were rebuilt after a split, and that each canonical artifact contains the newest expected week.
7. **Spot producer hardening.** Append/merge, a real exit-code contract, and failure that does not silently yield exit 0 or incomplete output; plus its first tests.
8. **Publication transaction + local lock + rollback** across producers so an interrupted refresh cannot leave a mixed canonical state or race a concurrent run.

---

## 15. Can current code support …?

- **An ordinary weekly update:** **Partially.** Liquidity has a real incremental engine; adjusted skip-existing can add the new day; but spot and surface can only whole-file overwrite, `--as-of` can't discover the raw week, and no manifest is written. Not end-to-end.
- **A retroactive split repair:** **No (not safely).** No delta detection, no scoped repair, no merge, and no audit that proves propagation; the only path is a full whole-store/whole-file rebuild of the adjusted/spot/surface chain, non-atomic and unverified for freshness.
- **A newly added liquid ticker:** **No (not incrementally).** Adding one ticker forces a full rebuild of the adjusted/spot/surface chain; there is no additive merge, and default skip-existing means the new ticker won't appear without a full re-run.
- **An interrupted-run recovery:** **No.** No transaction, no rollback, no lock; recovery is manual and can leave mixed canonical state.
- **An idempotent rerun:** **Per component only.** Individual producers overwrite/refuse deterministically, but there is no single idempotent refresh and `build_id` changes each run by design.

---

## 16. Disagreements between documentation and implementation

1. **Runbook path vs code default (`split-adjuster`):** runbook §"Routine repair" shows `apply_split_adjustment --tickers NVDA TSLA … --overwrite` implying a ticker-scoped repair. In code, `--tickers` is **logging only**; the run overwrites the whole store. Documentation overstates scoping.
2. **C1 receipt path vs C5 reality:** the receipt/CLI preview uses `splits_hist.parquet` under a single `cache_dir`, but the accepted split artifact is `splits_hist_liquid.parquet` under `input/adjusted_liquid`, and production artifacts span three roots. The receipt schema cannot represent this without change.
3. **"Weekly operator model" modes vs CLI:** `current_sprint.md` describes `incremental`/`backfill`/`repair` behaviors; the CLI treats `--mode` as **display-only** (correctly per C2 scope, but the agenda reads as if behavior exists).
4. **`--as-of` intent vs resolution root:** HD-004-2 frames `--as-of` as "last trading day ≤ date" to trigger a weekly refresh, but the resolver checks the **adjusted** root, so it cannot detect a new raw week awaiting adjustment.
5. **Agenda "atomic four-file publication" tone vs reality:** liquidity publication is per-file `os.replace`, not a cross-file transaction (a nuance the docs do not flag).
6. **Runbook "extract_spot_prices → append new dates" (incremental table)** vs code: the spot producer cannot append; it overwrites the requested year range wholesale.

---

## Open design decisions for C8.1

Each item lists **why a decision is required**, **facts discovered in C8.0**, and **risk of choosing incorrectly**. C8.0 does not choose any of these.

1. **Definition of a successful production refresh.**
   - *Why:* there is no executing refresh and no completion criterion today.
   - *Facts:* producers publish independently; no manifest; no cross-layer freshness check.
   - *Risk:* declaring "success" on a partially-published or stale-cross-layer state would certify an untrustworthy snapshot.

2. **Raw-vs-adjusted date discovery.**
   - *Why:* `--as-of` resolves against the adjusted root and is blind to new raw weeks (§10).
   - *Facts:* `resolve_as_of_trading_day(orats_adj_root=DEFAULT_ADJUSTED_LIQUID_ROOT)`; liquidity uses the raw root separately.
   - *Risk:* choosing the adjusted root as the discovery source means weekly refresh silently never triggers on new data.

3. **Split-history diff policy.**
   - *Why:* `fetch_splits` full-overwrites with no diff; repairs can't be scoped without knowing what changed.
   - *Facts:* conflicting divisors fail closed (scoped); no changed/new/deleted surfacing; full re-fetch requires an ORATS token.
   - *Risk:* overwriting the accepted split file with an unverified fetch, or missing that a factor changed, corrupts the entire adjusted/spot/surface chain.

4. **New/removed liquid-ticker policy.**
   - *Why:* additions force a full rebuild; removals cannot occur under append; no deletion/retention policy exists.
   - *Facts:* `liquid_tickers.csv` is cumulative and append-grows; adjusted default skip-existing hides new tickers; no membership diff anywhere.
   - *Risk:* an ad-hoc add/remove policy could silently drop history or leave orphan rows across layers.

5. **Historical repair range.**
   - *Why:* a split affects all pre-split dates for a ticker; the correct repair span must be defined.
   - *Facts:* `readjust_tickers` overwrites all years for the whole universe; `--years` scopes files, `--tickers` does not scope writes.
   - *Risk:* under-scoping leaves stale factors; over-scoping is a costly non-atomic whole-store rewrite.

6. **Spot merge contract.**
   - *Why:* spot can only whole-range overwrite; a running `--year` replaces the entire DB.
   - *Facts:* grain `(date, ticker)`; no dedup; exceptions swallowed; no tests; no exit contract.
   - *Risk:* silent data loss (whole-DB replacement) or silent incompleteness (swallowed per-date errors with exit 0).

7. **Surface merge contract.**
   - *Why:* surface can only whole-file overwrite; bounded runs to the canonical filename destroy other rows.
   - *Facts:* filename keyed by frequency+year bounds; `--overwrite` replaces; C6.4 safely used a separate `--output-root`.
   - *Risk:* a scoped repair against the canonical file wipes the rest of the universe/history.

8. **A1/A2 transaction model.**
   - *Why:* A1 and A2 are two independent non-atomic writes.
   - *Facts:* meta then quotes; a failure between them leaves unpaired/mismatched artifacts; audit catches it only after the fact.
   - *Risk:* downstream S3 assembly on an A1 without matching A2.

9. **Staging and publication model.**
   - *Why:* only liquidity stages; all other layers write directly.
   - *Facts:* per-file `os.replace` for liquidity; direct `to_parquet` elsewhere; no cross-file transaction.
   - *Risk:* interrupted writes corrupt files or publish partial state.

10. **Manifest path representation (multi-root).**
    - *Why:* the receipt assumes one `cache_dir`; production spans three roots (§3, §9).
    - *Facts:* keys/paths in `DEFAULT_ARTIFACT_REL_PATHS` don't match accepted roots/filenames (`splits_hist_liquid.parquet`, `input/liquidity/…`).
    - *Risk:* a manifest that mislabels or cannot locate the real artifacts is not a trustworthy receipt.

11. **Audit gates by refresh mode.**
    - *Why:* audits prove internal consistency, not freshness; no gate ties an audit to a refresh outcome.
    - *Facts:* adjusted math uses stored factor (can't detect stale); no missing-week detection; no cross-layer freshness audit.
    - *Risk:* passing internal audits could green-light a stale or incomplete snapshot.

12. **Idempotency semantics.**
    - *Why:* only per-component idempotency exists; `build_id` changes each run.
    - *Facts:* liquidity refuses no-op; surface guard blocks; spot/adjusted overwrite.
    - *Risk:* an orchestrator without defined idempotency could duplicate work or re-publish unnecessarily.

13. **Concurrency lock.**
    - *Why:* no lock exists; concurrent runs can race on canonical paths.
    - *Facts:* no PID/lock file anywhere.
    - *Risk:* interleaved writes corrupt canonical artifacts.

14. **Implementation commit sequence.**
    - *Why:* the gaps span discovery, delta detection, merge engines, transactions, and audits — too large for one commit.
    - *Facts:* the reusable blocks (C1 manifest, liquidity incremental, audits, resolver) are strong; the missing pieces are merge/transaction/freshness.
    - *Risk:* wiring an executing refresh before merge/freshness gates exist would let C8 "succeed" while publishing unsafe snapshots.

---

## Inspected files

**Docs:** `AGENTS.md`, `docs/agenda/current_sprint.md`, `docs/agenda/sprint004_execution_guardrails.md`, `docs/repo_map.md`, `docs/v1_weekly_runbook.md`, `docs/v1_universe_protocol.md`, `docs/surface_engine_data_contract.md`, `docs/sprint_memos/004_c4_liquidity_panel.md`, `004_c5_adjusted_liquid.md`, `004_c6_option_surface.md`, `004_c7_pit_universe.md`, `docs/tmp/c1_manifest_design_plan.md`, `docs/tmp/c2_cli_design_plan.md`.

**Implementation:** `scripts/refresh_weekly_inputs.py`, `src/data/input_snapshot.py`, `src/data/trading_day.py`, `src/data/paths.py`, `scripts/build_liquidity_panel.py`, `scripts/fetch_splits.py`, `scripts/apply_split_adjustment.py`, `src/data/split_adjuster.py`, `src/data/ticker_universe.py`, `scripts/extract_spot_prices.py`, `src/data/spot_price_db.py`, `src/data/orats_provider.py` (day-load + `get_spot_price`), `scripts/precompute_option_surface.py`, `src/features/option_surface_analyzer.py` (`process_single_entry`), `src/features/option_surface_contract.py`, `scripts/audit_option_surface_artifacts.py`, `scripts/audit_adjusted_liquid.py`, `scripts/audit_pit_universe.py`, `src/backtest/pipeline.py` (S1 header).

**Tests surveyed:** `tests/unit/test_input_snapshot.py`, `test_trading_day.py`, `test_refresh_weekly_inputs_cli.py`, `test_build_liquidity_panel.py`, `test_fetch_splits_cli.py`, `test_apply_split_adjustment_cli.py`, `test_split_adjuster.py`, `test_split_adjuster_filtered_zip.py`, `test_ticker_universe.py`, `test_audit_adjusted_liquid.py`, `test_adjusted_liquid_paths.py`, `test_precompute_option_surface_cli.py`, `test_option_surface_contract.py`, `test_option_surface_readiness.py`, `test_option_surface_c64_audit.py`, `test_audit_option_surface_artifacts.py`, `test_pit_universe_audit.py`, `test_audit_pit_universe_cli.py`, `test_spot_price_db.py`, `tests/contract/test_step1_universe_contract.py`, plus a full `tests/**` inventory (confirmed **no** `test_extract_spot_prices*`).

**Capability searches performed:** `append`, `incremental`, `atomic`, `merge`, `dedup(licate)`, `overwrite`, `watermark`, `readjust`, `preserve`, `write_manifest`/`default_manifest_path`/`compute_snapshot_id`/`generate_build_id`, `FileNotFoundError`, `process_single_entry` — across `src/`, `scripts/`, `tests/`.
