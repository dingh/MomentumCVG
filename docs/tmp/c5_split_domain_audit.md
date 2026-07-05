# C5.2 Split Domain Code Audit

> **Superseded for operator use** by C5 closeout ([sprint_memos/004_c5_adjusted_liquid.md](../sprint_memos/004_c5_adjusted_liquid.md), 2026-07-04). Retained as C5.2 audit evidence.

## Status

**Closed** — audit accepted; implementation completed through C5.11A.

## Executive Summary

The split domain has **sound core library code** (`OratsCorporateActionsFetcher`, `SplitAdjuster` factor math) but **no unit tests**, **no filtered adjusted output**, and **stale defaults** that still assume full-ORATS `all_tickers.parquet` and `ORATS_Adjusted/`. C5 can build on the existing engine after adding a ticker loader, row filter, scoped paths, and golden tests.

**Canonical keepers:** `src/data/corporate_actions.py` (library), `src/data/split_adjuster.py` (adjustment engine), `scripts/fetch_splits.py` and `scripts/apply_split_adjustment.py` (CLIs — modify, do not replace).

**Archive/deprecate for C5 path:** `scripts/build_ticker_universe.py` / `all_tickers.parquet` as the *precompute* ticker source (keep for optional full-ORATS maintenance only). No legacy `PriceAdjuster` class exists in the repo.

**Approved C5 pipeline after C4:** C4 `liquid_tickers.csv` → scoped split fetch → filtered adjustment → `adjusted_liquid/` → output audit. Liquidity panel stays upstream on raw ORATS; split adjust is not a C4 prerequisite.

**Blockers before C5.3:** None for ticker loader + golden tests. HD decisions needed before filtered backfill (C5.6) and scoped split fetch defaults (C5.7).

```text
Overall audit status: PASS WITH WARNINGS
Recommendation: proceed to C5.3 after HD approves this audit
```

---

## C5 Context

- C4 produced `liquid_tickers.csv` as a **historical precompute superset** (~2,783 tickers at production backfill). Column `Ticker` (capital T); optional `snapshots_qualified`.
- This is **not** the latest PIT trading universe. S1 uses the liquidity panel at rebalance date `t`.
- C5 uses `liquid_tickers.csv` only to **shrink** split fetching and adjusted parquet generation.
- C5 target output: filtered adjusted parquet pool under `C:/MomentumCVG_env/input/adjusted_liquid/`.
- C5 must **not** overwrite the full adjusted ORATS mirror at `C:/ORATS/data/ORATS_Adjusted/` unless explicitly requested.

---

## Files Audited

| File | Role today | Keep / modify / archive / out of C5 | Reason | Risks |
| ---- | ---------- | ----------------------------------- | ------ | ----- |
| `src/data/corporate_actions.py` | ORATS API client; `fetch_all_splits` / `fetch_all_earnings` with checkpoint; `get_all_unique_tickers()` full ZIP scan | **Keep** (library) | Single HTTP + checkpoint implementation; correct dedup on `(ticker, split_date)` | Stale docstring cites missing `fetch_orats.py`. `ORATS_API_TOKEN` env fallback **commented out** (line 67) but error message still claims env works. `divisor` NaN not dropped on fetch. |
| `scripts/fetch_splits.py` | Thin CLI → `fetch_all_splits` | **Keep + modify** | Right entrypoint for C5 scoped fetch | Default input `all_tickers.parquet` (full universe). `--tickers` flag is a **file path**, not symbol list — naming collision with `apply_split_adjustment --tickers`. Only reads parquet, not CSV/`Ticker` column. |
| `scripts/fetch_earnings.py` | Thin CLI → `fetch_all_earnings` from SP500.csv | **Out of C5** | Shares library only; earnings not in C5 split path | Different universe source (SP500) reinforces ticker-source fragmentation. |
| `scripts/build_ticker_universe.py` | Scan raw ZIPs → `all_tickers.parquet` | **Archive for C5 path** (keep script for maintenance) | C5 should use C4 `liquid_tickers.csv`, not re-scan full ORATS | `--years` branch **duplicates** ZIP-scan logic instead of calling `get_all_unique_tickers`. Hours-long scan unnecessary when liquid superset exists. |
| `src/data/split_adjuster.py` | Raw ZIP → adjusted parquet; `_load_cum_factors`, `_apply_adjustments` | **Keep + modify** | Canonical factor math and column contract | No row-level universe filter. `readjust_tickers()` scans **all** ZIPs and rewrites **full** daily files. **Zero unit tests** in repo. |
| `scripts/apply_split_adjustment.py` | Thin CLI → `SplitAdjuster.run` / `readjust_tickers` | **Keep + modify** | Right entrypoint for C5 filtered backfill | Default `--adj-root` is full `ORATS_Adjusted/`. `--tickers` is symbol list (opposite meaning from `fetch_splits`). No `--ticker-universe`. |
| `scripts/extract_spot_prices.py` | Reads adjusted parquets via `ORATSDataProvider`; writes `spot_prices_adjusted.parquet` | **Out of C5** (modify later) | Downstream consumer of adjustment output | Defaults to full `ORATS_Adjusted/`; no liquid-universe filter. C5 does not rewire spot extraction. |
| `scripts/refresh_weekly_inputs.py` | C2 CLI skeleton; provisional `plan` step order | **Modify later** (C8/C9) | Documents operator pipeline | Plan order: splits → adjust → **then** liquidity panel — **wrong relative to C4** (liquidity reads raw ORATS). `split-audit` stub points to C5. Receipt artifacts still reference cache-relative `splits_hist.parquet` only. |

---

## Search Findings

### `PriceAdjuster`

| Where | C5 relevance | Duplication / stale |
|-------|--------------|---------------------|
| `src/data/split_adjuster.py` module docstring only | Historical note about fixed bug | **No class in repo.** Safe to treat as documentation only; no archive action. |

### `split_factor` / `cum_factor`

| Where | C5 relevance | Duplication / stale |
|-------|--------------|---------------------|
| `src/data/split_adjuster.py` only (production code) | **Canonical** — all C5 tests should target this | No duplicate implementations found. |

### `split_date`

| Where | C5 relevance | Duplication / stale |
|-------|--------------|---------------------|
| `corporate_actions.py` — API → `split_date` as `date` | Fetch contract | Compared as `pd.Timestamp` in adjuster — OK |
| `split_adjuster.py` — strict `> trade_date` filter | Adjustment contract | Consistent in `_apply_adjustments` and `_load_cum_factors` docs |

### `splits_hist`

| Where | C5 relevance | Duplication / stale |
|-------|--------------|---------------------|
| `scripts/fetch_splits.py` default output `cache/splits_hist.parquet` | Global split store | Full-universe scope today |
| `input_snapshot.py`, `current_sprint.md`, runbook, C1 design | Receipt / docs | Assumes single global splits artifact; C5 may add `splits_hist_liquid.parquet` alongside |
| `split_adjuster.py` docstring | Operator workflow | References global file only |

### `all_tickers`

| Where | C5 relevance | Duplication / stale |
|-------|--------------|---------------------|
| `scripts/fetch_splits.py` default `--tickers` path | **Conflicts with C5** | Full ORATS scan product |
| `scripts/build_ticker_universe.py` | Builds `all_tickers.parquet` | Not needed as C5 step 1 |
| `tests/unit/test_cvg_calculator.py` | Test variable names only | Irrelevant to C5 |

### `liquid_tickers`

| Where | C5 relevance | Duplication / stale |
|-------|--------------|---------------------|
| `scripts/build_liquidity_panel.py` — writes `liquid_tickers.csv` | **C5 ticker source** | Production: `input/liquidity/` |
| `docs/v1_universe_protocol.md`, runbook, C4 memo | Documented precompute superset | Correct semantics |
| `scripts/precompute_option_surface.py`, straddle scripts | Consumer (Sprint 005 / surface) | Default path `cache/liquid_tickers.csv` — **stale vs C4 production path** |
| `refresh_weekly_inputs.py` plan step 0 | Future scope resolution | Not implemented |

### `ORATS_Adjusted`

| Where | C5 relevance | Duplication / stale |
|-------|--------------|---------------------|
| `apply_split_adjustment.py`, `split_adjuster.py`, `extract_spot_prices.py` defaults | Full adjusted mirror | C5 must avoid writing here by default |
| `orats_provider.py` | Backtest/surface consumer | Expects `adj_*` columns from adjusted parquets |
| `build_liquidity_panel.py` docstring | States downstream uses adjusted; **C4 input is raw** | Doc is directionally right for surface, not for liquidity |

### `adjusted_liquid`

| Where | C5 relevance | Duplication / stale |
|-------|--------------|---------------------|
| `docs/tmp/c5_split_adjustment_design_plan.md` only | **Planned C5 output root** | Not implemented; no code references yet |

### `apply_split_adjustment`

| Where | C5 relevance | Duplication / stale |
|-------|--------------|---------------------|
| `scripts/apply_split_adjustment.py` | C5 CLI base | Full-universe only today |
| `refresh_weekly_inputs.py`, sprint docs, runbook | Pipeline step 2 | Order vs C4 needs update at C8/C9 |

### `fetch_splits`

| Where | C5 relevance | Duplication / stale |
|-------|--------------|---------------------|
| `scripts/fetch_splits.py` | C5 CLI base | Wrong default ticker source for C5 |
| Pipeline docs | Step 1 in runbook | Should follow C4 for C5 scoped path |

---

## Canonical Keep Set

| Module / script | C5.3+ action | Notes |
|-----------------|--------------|-------|
| `src/data/corporate_actions.py` | **Keep as-is** (minor doc fix later) | `OratsCorporateActionsFetcher`, checkpoint, dedup. `get_all_unique_tickers` kept for maintenance, not C5 precompute path. |
| `src/data/split_adjuster.py` | **Keep + modify** | Add optional `ticker_universe: set[str] \| None` to filter rows before write; expose via `process_zip` / `run`. Do **not** fork factor math. |
| `scripts/fetch_splits.py` | **Keep + modify** | Add `--ticker-universe` (CSV/parquet via `load_ticker_universe`); optional separate `--output` for `splits_hist_liquid.parquet`. |
| `scripts/apply_split_adjustment.py` | **Keep + modify** | Add `--ticker-universe`, `--adj-root` for liquid pool; require explicit `--adj-root` when universe filter active (guardrail). |
| `load_ticker_universe()` (new, C5.3) | **Wrap / new helper** | Single ticker-load path for fetch + adjust + audit. |
| `scripts/fetch_earnings.py` | **Not used in C5** | No change in C5. |
| `scripts/extract_spot_prices.py` | **Not used in C5** | Rewire to `adjusted_liquid/` in a later sprint when spot DB is scoped. |
| `scripts/refresh_weekly_inputs.py` | **Not used in C5** | C8 wiring; fix plan order at C9. |

---

## Archive / Deprecate Candidates

**Do not move files in C5.2.** Recommendations only:

| Candidate | Recommendation | Migration note |
|-----------|----------------|----------------|
| `scripts/build_ticker_universe.py` for C5 precompute | **Deprecate for C5 path** | Document: use `liquid_tickers.csv` from C4. Keep script for one-off full-ORATS inventory if needed. |
| `all_tickers.parquet` as default split-fetch input | **Deprecate for C5 path** | `fetch_splits` default should not require full scan when liquid CSV exists. |
| `cache/liquid_tickers.csv` path in surface/straddle scripts | **Stale default** (not C5 scope) | Point to `input/liquidity/liquid_tickers.csv` in Sprint 005+ surface work. |
| Legacy `PriceAdjuster` | **N/A — already absent** | Docstring note only. |
| `scripts/fetch_orats.py` | **Stale reference** | Cited in `corporate_actions.py` line 10; file does not exist. Remove reference in a later doc/code hygiene commit. |

---

## Duplication and Stale Reference Findings

1. **Three ticker-universe sources:** `all_tickers.parquet` (full ZIP scan), `liquid_tickers.csv` (C4), `SP500.csv` (`fetch_earnings`). C5 must standardize on **`load_ticker_universe(liquid_tickers.csv)`** for split fetch and adjustment.

2. **Ticker loading duplicated:** `fetch_splits.main()` inline parquet read; `fetch_earnings.load_sp500_tickers()`; `build_ticker_universe` + `get_all_unique_tickers`; future C5.3 loader. Consolidate on one helper.

3. **`--tickers` flag collision:** `fetch_splits --tickers` = path to parquet file. `apply_split_adjustment --tickers` = list of symbols. Operator footgun; C5 should add `--ticker-universe` and avoid overloading `--tickers` further.

4. **ZIP scan duplication:** `build_ticker_universe.py` lines 105–144 reimplement scanning when `--years` is set instead of extending `get_all_unique_tickers`.

5. **Pipeline order stale:** `v1_weekly_runbook.md` and `refresh_weekly_inputs.py` plan: split adjust **before** liquidity panel. C4 **shipped** reading raw ORATS first; liquidity is upstream of scoped adjust for C5.

6. **`readjust_tickers()` misnamed:** Doc says "re-process only ZIP files that contain the given tickers" but implementation processes **every** ZIP in scope with `overwrite=True` and writes **all** tickers per file. Does **not** filter output rows. Unsuitable for C5 filtered pool.

7. **Missing script reference:** `corporate_actions.py` → `scripts/fetch_orats.py` (not in repo).

8. **Path drift:** C4 production `input/liquidity/` vs cache paths in runbook, precompute scripts, and C1 receipt defaults.

9. **No tests:** Zero pytest coverage for `split_adjuster`, `fetch_splits`, or `apply_split_adjustment`.

---

## Split Adjustment Contract Review

Reviewed `src/data/split_adjuster.py` (`_load_cum_factors`, `_apply_adjustments`, `_process_single_zip`).

| Contract item | Status |
|---------------|--------|
| `split_factor` = product(divisor where `split_date > trade_date`) | **Matches** — via oldest future split's `cum_factor` on descending-sorted per-ticker table |
| `split_date == trade_date` excluded | **Matches** — strict `>` in line 160 |
| Price cols adjusted: `stkPx`, `strike`, `cBidPx`, `cAskPx`, `pBidPx`, `pAskPx` | **Matches** — `PRICE_COLS` |
| Raw columns preserved | **Matches** — `df.copy()` + additive columns |
| Added: `trade_date`, `split_factor`, `adj_*`, `spot_px` | **Matches** |
| `spot_px == adj_stkPx` | **Matches** — when `adj_stkPx` present |
| `min_split_date` drops pre-cutoff splits at load | **Matches** — default 2014-01-01 |
| Divisor semantics (ORATS 2-for-1 → 2.0) | **Documented** in module docstring; **not unit-tested** |
| Row filter to C4 universe | **Missing** — writes all rows |
| Golden tests | **Missing** |

**Warnings:** No tests prove multi-split product or on-split-date edge case. NaN `divisor` from API would propagate if present in parquet.

```text
Split adjustment contract review: PASS WITH WARNINGS
```

---

## Fetch Splits Contract Review

Reviewed `corporate_actions.py` (`OratsCorporateActionsFetcher`, `fetch_all_splits`) and `fetch_splits.py`.

| Contract item | Status |
|---------------|--------|
| Token handling | **WARN** — must pass `--token`; env var path disabled in code but error text misleading |
| Checkpoint/resume per ticker | **PASS** — saves after each ticker; skips `processed` set |
| Dedupe | **PASS** — `drop_duplicates(subset=["ticker", "split_date"])` |
| Output columns `ticker`, `split_date`, `divisor` | **PASS** |
| Default ticker source | **FAIL for C5** — `all_tickers.parquet`, not `liquid_tickers.csv` |
| Can use C4 CSV directly today | **No** — CLI only accepts parquet with lowercase `ticker` column |
| Separate `splits_hist_liquid.parquet` | **Not implemented** — trivial via `--output` once universe loader exists |

```text
Fetch splits contract review: PASS WITH WARNINGS
(C5-ready after loader + CLI flag changes; not FAIL because library supports any ticker list)
```

---

## Filtered Adjusted Output Gap

**Current behavior:** `SplitAdjuster` reads each raw ZIP, adjusts **all** rows, writes full daily parquet to `adj_root`. `readjust_tickers()` still writes full daily files.

**C5 need:** Parquets under `adjusted_liquid/` containing **only** tickers in `liquid_tickers.csv`. Smaller files, no pollution from non-liquid names.

**`--tickers` / `readjust_tickers()` does not solve this:** It re-runs adjustment on all ZIPs and emits every ticker in each daily file. It is a repair hook for factor refresh, not a universe filter.

### Recommended implementation path: **Option A**

**Modify `SplitAdjuster` + `apply_split_adjustment.py`** (do not add `apply_split_adjustment_liquid.py`).

| Reason | Detail |
|--------|--------|
| Single factor engine | Row filter belongs next to `_apply_adjustments` / `_process_single_zip` |
| Avoid CLI duplication | One script, explicit flags: `--ticker-universe`, `--adj-root` |
| Guardrail | When `--ticker-universe` is set, default `--adj-root` to `input/adjusted_liquid/` and **refuse** `ORATS_Adjusted` unless `--force-full-adj-root` (or similar) |
| Option B rejected | New wrapper script duplicates argparse, logging, and drifts from `apply_split_adjustment` repair/backfill flags |

---

## Approved C5 Pipeline

C4 is complete; `liquid_tickers.csv` already on disk at `C:/MomentumCVG_env/input/liquidity/`.

```text
[C4 complete — upstream, raw ORATS]
  build_liquidity_panel.py  →  liquid_tickers.csv  (+ panel artifacts)

[C5 — downstream of C4]
  liquid_tickers.csv
    → load_ticker_universe()
    → fetch_splits.py (scoped)  →  splits_hist_liquid.parquet
    → apply_split_adjustment.py --ticker-universe … --adj-root adjusted_liquid/
    → adjusted_liquid/{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.parquet
    → audit_split_adjustment.py (C5.8)
    → C5 closeout memo (C5.10)
```

**Clarifications:**

- Liquidity panel reads **raw** `ORATS_Data` ZIPs. Split adjustment is **not** a prerequisite for C4.
- `extract_spot_prices`, surface precompute, signals, and backtest are **later**; C5 does not wire them.
- Full `ORATS_Adjusted/` mirror remains optional maintenance path, not C5 default.

**Runbook / CLI plan order** should be updated at C9 to: liquidity (C4) → scoped splits → filtered adjust → spot → surface.

---

## Blockers Before C5.3

### True blockers

None. C5.3 (ticker loader) and C5.4 (golden tests) do not require filtered backfill or split re-fetch.

### Warnings

| Item | Action |
|------|--------|
| Zero split-domain unit tests | Address in C5.4 |
| `readjust_tickers` naming / behavior | Document; do not use for C5 filtered output |
| `fetch_splits` / `apply_split_adjustment` `--tickers` meaning clash | Document in C5.7; prefer `--ticker-universe` |
| Global `splits_hist.parquet` may omit liquid tickers if never fetched | Validate coverage in C5.7 or fetch scoped file |
| `corporate_actions` env token comment | Fix when touching file |
| Runbook pipeline order vs C4 | Fix at C9 |

### HD decisions needed

| # | Decision | Audit recommendation |
|---|----------|-------------------|
| 1 | Output root `C:/MomentumCVG_env/input/adjusted_liquid/` | **Approve** |
| 2 | Separate `splits_hist_liquid.parquet` vs reuse global `splits_hist.parquet` | **Prefer separate** under `adjusted_liquid/` for clear scope; reuse OK only after coverage proof |
| 3 | Option A vs B for filtered adjust | **Option A** — extend existing script + adjuster |
| 4 | Sorted ticker list vs file order | **Sorted** (matches C4 `build_liquid_tickers`) |
| 5 | Keep `build_ticker_universe.py` in repo? | **Yes, maintenance only** — not C5 step 1 |

---

## Recommended Implementation Slices After Audit

1. **C5.3** — `load_ticker_universe()` + unit tests
2. **C5.4** — Pure split golden tests (`test_split_adjuster.py`)
3. **C5.5** — Synthetic filtered ZIP-to-parquet test
4. **C5.6** — Filtered adjusted backfill (`--ticker-universe`, row filter, `adjusted_liquid/` defaults)
5. **C5.7** — Scoped split fetch (`fetch_splits` + `splits_hist_liquid.parquet`)
6. **C5.8** — Split output audit script
7. **C5.9** — Real-data sample audit (AAPL, TSLA, NVDA)
8. **C5.10** — Closeout memo `docs/sprint_memos/004_c5_split_adjustment.md`

---

## Final Recommendation

```text
Recommendation: proceed to C5.3 after HD approves this audit.
```

Core split math and fetch library are adequate foundations. C5 implementation work is: shared ticker loader, row-filtered write path, scoped artifacts, tests, and output audit — not a new adjustment algorithm.
