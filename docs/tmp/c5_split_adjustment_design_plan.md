# C5 Design Plan — Split Adjustment Trust Layer for the C4 Liquidity Universe

## Status

**Closed** — C5 accepted 2026-07-04. See [sprint_memos/004_c5_adjusted_liquid.md](../sprint_memos/004_c5_adjusted_liquid.md).

**Audit memo:** [c5_split_domain_audit.md](c5_split_domain_audit.md)

## Why this exists

- **C4 succeeded but became broader than expected.** The rolling liquidity panel backfill (2017→2026) produced a historical precompute superset of **2,783 tickers** in `liquid_tickers.csv`, plus daily/weekly/panel artifacts. C4 scope was correct; the output is intentionally a superset, not a trading universe.
- **C5 should be intentionally smaller and more reviewable.** Rather than validating split adjustment across the full ORATS universe (~10k+ tickers), C5 scopes fetch, adjustment, and audit to the C4 liquidity-filtered precompute set.
- **Existing split-fetch and split-adjustment code already exists** but was written before C4 and has **not been formally audited** in Sprint 004. Split adjustment correctness depends entirely on fetch + factor math + write path; we should not build C5 on top of unreviewed scripts.
- **C5.2 split-domain code audit is complete** ([c5_split_domain_audit.md](c5_split_domain_audit.md)); keep/archive decisions are locked in § C5.2 audit decisions below.
- **C5 implementation puts a correctness cage around the chosen code** and produces a smaller adjusted parquet pool.
- **The C4 liquidity universe is used only as an engineering/precompute filter, not as a PIT trading universe.** Per [v1_universe_protocol.md](../v1_universe_protocol.md) and the C4 closeout memo: S1 trading universe comes from the PIT panel at rebalance date `t`, not from `liquid_tickers.csv` alone.

## C5 objective

By the end of C5, we should have **audited** the split/corporate-actions codebase, decided what to keep vs archive, and then used the C4 `liquid_tickers.csv` historical precompute universe to fetch relevant split history, apply split adjustment only to relevant raw ORATS rows, save a smaller adjusted parquet pool, and audit that the adjusted outputs obey expected split-factor and price-adjustment invariants on synthetic and real samples.

## Core design decision

C5 uses the C4 `liquid_tickers.csv` historical precompute superset as the ticker universe for split fetching and adjusted parquet generation. This is an engineering/precompute filter, not the point-in-time trading universe. The output is a smaller adjusted parquet pool containing only rows for those tickers.

## C5.2 audit decisions

Locked by [c5_split_domain_audit.md](c5_split_domain_audit.md) (2026-06-29):

- **C5.2 audit status:** PASS WITH WARNINGS.
- **Proceed to C5.3** after HD approval of this updated design doc.
- **Keep** `src/data/corporate_actions.py` as the split-fetch library.
- **Keep** `src/data/split_adjuster.py` as the canonical split-adjustment engine (extend for row filter).
- **Keep + modify** `scripts/fetch_splits.py` and `scripts/apply_split_adjustment.py`.
- **Do not create** `scripts/apply_split_adjustment_liquid.py` unless Option A becomes clearly unworkable during implementation.
- **Add** shared `load_ticker_universe()` helper (`src/data/ticker_universe.py`, C5.3).
- **Filtered adjusted output root:** `C:/MomentumCVG_env/input/adjusted_liquid/`.
- **C5-scoped split history:** `C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet` (preferred over global cache file).
- **`build_ticker_universe.py` / `all_tickers.parquet`:** full-ORATS maintenance only — not the C5 precompute path.
- **Out of C5 implementation:** `fetch_earnings.py`; `refresh_weekly_inputs.py` CLI wiring (C8). **`extract_spot_prices.py` defaults wired in C5.11A**; full spot re-extract on production root not run in C5.

## In scope

1. ~~**Audit** split-fetch, corporate-actions, and adjustment scripts~~ — **done (C5.2)**; keep/archive decisions in audit memo.
2. Document split-adjustment contract.
3. Define how to read the C4 liquidity-filtered precompute universe.
4. Fetch or validate split history only for tickers in that universe.
5. Apply split adjustment only to raw ORATS rows whose ticker is in that universe.
6. Write a smaller adjusted parquet pool.
7. Add golden tests for split factor and adjusted price logic.
8. Add one synthetic ZIP-to-parquet test, including ticker filtering.
9. Add a small split-adjustment **output** audit script.
10. Run one substantive real-data audit sample.
11. ~~Write a C5 closeout memo after tests/audit pass.~~ **Done:** [sprint_memos/004_c5_adjusted_liquid.md](../sprint_memos/004_c5_adjusted_liquid.md)

## Out of scope

- Latest PIT universe selection
- Using one latest liquidity snapshot as the universe
- Strategy logic
- S5/S8/ORCH
- Backtesting
- SurfaceRunner
- Momentum/CVG
- Feature pipeline
- Full refresh orchestration
- Perfect full-cache audit
- Full 10-year exhaustive split scan
- Incremental update handling for new tickers
- Detecting exactly which old files need to be rewritten after new tickers enter the universe
- Efficient targeted repair optimization
- Major redesign of split storage (C5.2 audit approved targeted extensions only — Option A)
- New split-adjustment algorithm unless the current one is proven wrong by audit or golden tests

## Split domain inventory (confirmed in C5.2)

C5.2 audit confirmed overlapping entrypoints; verdicts below match [c5_split_domain_audit.md](c5_split_domain_audit.md).

| Artifact | Role today | Overlap / concern |
|----------|------------|-------------------|
| `src/data/corporate_actions.py` | ORATS API fetch (splits + earnings); `get_all_unique_tickers()` full ZIP scan | **Canonical library candidate** for API + checkpoint. Docstring references missing `scripts/fetch_orats.py`. |
| `scripts/fetch_splits.py` | Thin CLI → `fetch_all_splits()` | Default ticker source `all_tickers.parquet`, not C4 `liquid_tickers.csv`. Inline parquet load duplicates future `load_ticker_universe`. |
| `scripts/fetch_earnings.py` | Thin CLI → `fetch_all_earnings()` | Parallel pattern to splits; uses **SP500.csv** universe — different source than splits. Out of C5 split path but shares `corporate_actions.py`. |
| `scripts/build_ticker_universe.py` | Scan raw ZIPs → `all_tickers.parquet` | Calls `get_all_unique_tickers()` **or** duplicates ZIP-scan logic when `--years` is set (inline copy, not shared). May be **obsolete for C5** if `liquid_tickers.csv` is the precompute universe. |
| `src/data/split_adjuster.py` | ZIP → adjusted parquet; factor math | **Canonical adjustment engine candidate.** References removed legacy `PriceAdjuster` bug; no `PriceAdjuster` in repo. `readjust_tickers()` name implies scope but scans all ZIPs. |
| `scripts/apply_split_adjustment.py` | Thin CLI → `SplitAdjuster` | Full-universe write to `ORATS_Adjusted/` when `--adj-root` omitted; filtered mode requires explicit `--adj-root` | **Keep + modify (done)** |
| `scripts/extract_spot_prices.py` | Reads adjusted parquets via `ORATSDataProvider` | Downstream consumer | **Defaults → `input/adjusted_liquid` (C5.11A)** |
| C4 `liquid_tickers.csv` | Historical precompute superset (~2,783 tickers) | **Intended C5 ticker source**; column `Ticker` vs lowercase `ticker` elsewhere. |

**Approved C5 pipeline (C5.2):** C4 liquidity panel reads raw ORATS first; split work is downstream.

```text
[C4 complete]
  build_liquidity_panel → liquid_tickers.csv

[C5]
  liquid_tickers.csv
    → load_ticker_universe()
    → fetch_splits --ticker-universe … → splits_hist_liquid.parquet
    → apply_split_adjustment --ticker-universe … → adjusted_liquid/{YYYY}/…
    → audit_split_adjustment (C5.8) → closeout memo (C5.10)
```

Runbook/CLI plan order fixed at C9. `build_ticker_universe.py` / `all_tickers.parquet` = maintenance only.

## Existing system summary

### `scripts/fetch_splits.py`

| Aspect | Detail |
|--------|--------|
| **Inputs** | ORATS API token (`--token`, required); ticker list from parquet (default `C:/MomentumCVG_env/cache/all_tickers.parquet`, column `ticker`) |
| **Outputs** | `splits_hist.parquet` (default `C:/MomentumCVG_env/cache/splits_hist.parquet`) with columns `ticker`, `split_date`, `divisor` |
| **Core algorithm** | Loads ticker list → `OratsCorporateActionsFetcher.fetch_all_splits()` → per-ticker ORATS `/hist/splits` API call with checkpoint/resume after each ticker |
| **CLI options** | `--token`, `--tickers`, `--output`, `--rate-limit` (default 0.7 s), `--max-retries` (default 5) |
| **Universe scope** | Full ticker universe from `all_tickers.parquet` (built by scanning all raw ORATS ZIPs via `get_all_unique_tickers()`). **No C4 liquidity filter today.** |

### `src/data/corporate_actions.py`

| Aspect | Detail |
|--------|--------|
| **Inputs** | ORATS API token; list of tickers |
| **Outputs** | DataFrames with `ticker`, `split_date`, `divisor` (splits) or earnings columns |
| **Core algorithm** | HTTP GET to `https://api.orats.io/datav2/hist/splits` per ticker; exponential backoff on 429/5xx; checkpoint parquet deduped on `(ticker, split_date)` |
| **Helper** | `get_all_unique_tickers(data_root)` scans all raw ZIPs and extracts unique tickers (one-time full-universe scan) |
| **Universe scope** | Caller-provided ticker list; no built-in liquidity filter |

### `scripts/apply_split_adjustment.py`

| Aspect | Detail |
|--------|--------|
| **Inputs** | Raw ORATS ZIPs (`--raw-root`, default `C:/ORATS/data/ORATS_Data`); splits parquet (`--splits`, default `C:/MomentumCVG_env/cache/splits_hist.parquet`); optional `--years`, `--tickers`, `--min-split-date` |
| **Outputs** | Adjusted parquets under `--adj-root` (default `C:/ORATS/data/ORATS_Adjusted`), layout `{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.parquet` |
| **Core algorithm** | Delegates to `SplitAdjuster.run()` or `readjust_tickers()` |
| **CLI options** | `--raw-root`, `--adj-root`, `--splits`, `--years`, `--overwrite`, `--workers`, `--tickers`, `--min-split-date` (default 2014-01-01) |
| **Filtered output** | **No.** Processes all rows in each ZIP; `--tickers` triggers re-adjustment but still reads and rewrites entire daily files (all tickers in file). No row-level universe filter on write. |

### `src/data/split_adjuster.py`

| Aspect | Detail |
|--------|--------|
| **Inputs** | Raw ZIP CSV inside `ORATS_SMV_Strikes_YYYYMMDD.zip`; cumulative factor table from splits parquet |
| **Outputs** | Parquet to `adj_root/{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.parquet` |
| **Core algorithm** | Parse trade date from filename → load CSV → for each ticker compute `split_factor` = product of divisors where `split_date > trade_date` (via precomputed `cum_factors` table) → divide price columns → write parquet |
| **Price columns adjusted** | `stkPx`, `strike`, `cBidPx`, `cAskPx`, `pBidPx`, `pAskPx` → `adj_stkPx`, `adj_strike`, `adj_cBidPx`, `adj_cAskPx`, `adj_pBidPx`, `adj_pAskPx` |
| **Columns added** | `trade_date`, `split_factor`, `adj_*` (for present price cols), `spot_px` (= `adj_stkPx`) |
| **Columns preserved** | All original CSV columns remain; adjustments are additive |
| **Split-date rule** | Strict: only splits with `split_date > trade_date` count as future. On split date itself, factor excludes that split. |
| **min_split_date** | Splits before cutoff (default 2014-01-01) dropped at load time; never appear as future splits |
| **Public test hooks** | `adjust_dataframe()`, `process_zip()` for unit/integration tests |
| **readjust_tickers** | Re-processes all ZIPs in scope with `overwrite=True`; does **not** filter to only rows containing target tickers (still writes full daily file). Doc comment acknowledges it scans all ZIPs. |
| **Filtered output** | **No.** Full raw row set written to adjusted parquet today. |

## C4 liquidity universe input

**Default expected path:**

```
C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv
```

(C4 production backfill wrote here; runbook also references cache-relative paths. C5 loader should accept any path via CLI.)

**File shape (from C4 `build_liquid_tickers`):**

| Column | Notes |
|--------|-------|
| `Ticker` | Primary ticker column (capital T) |
| `snapshots_qualified` | Count of week-end snapshots where ticker was in top-20% dvol bucket |
| `months_qualified` | Alias of `snapshots_qualified` (legacy name) |

**Semantics:** Historical precompute superset — tickers that were liquid enough at **some** point in the C4 panel window. **Not** a PIT trading universe. ~2,783 tickers after full C4 backfill.

**Loader requirements:**

- Accept CSV or parquet
- Accept `Ticker` or `ticker` column
- Dedupe tickers
- Drop null/blank tickers
- Normalize ticker strings (strip whitespace; consistent uppercase if project convention requires)

**Proposed helper:**

```python
load_ticker_universe(path) -> list[str]
```

Small, pure, unit-testable. Suggested location: `src/data/ticker_universe.py` (or adjacent to split adjuster if kept minimal).

**Deterministic ordering:** **Sorted alphabetically** (C5.2 audit decision; matches C4 `build_liquid_tickers`).

## Filtered adjusted parquet pool

**Approved output root (C5.2):**

```
C:/MomentumCVG_env/input/adjusted_liquid/
```

**Layout (mirrors ORATS by year):**

```
C:/MomentumCVG_env/input/adjusted_liquid/{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.parquet
```

Each output parquet contains **only rows** where `ticker` is in the C4 liquidity-filtered precompute universe.

**Separate from full adjusted mirror:**

```
C:/ORATS/data/ORATS_Adjusted/
```

C5 must **not** overwrite the full adjusted cache unless explicitly requested (separate `--adj-root` default for filtered mode).

## Split-adjustment contract

### Inputs

| Input | Default path |
|-------|--------------|
| Raw ORATS ZIPs | `C:/ORATS/data/ORATS_Data` |
| C4 liquidity ticker universe | `C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv` |
| Split history | `C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet` (preferred; global `splits_hist.parquet` fallback only if coverage proven) |

Minimum split columns: `ticker`, `split_date`, `divisor`

### Outputs

| Output | Path |
|--------|------|
| Filtered adjusted parquet | `C:/MomentumCVG_env/input/adjusted_liquid/{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.parquet` |

### Required output columns

- `trade_date`
- `split_factor`
- `adj_stkPx`
- `adj_strike`
- `adj_cBidPx`
- `adj_cAskPx`
- `adj_pBidPx`
- `adj_pAskPx`
- `spot_px`

(Plus preserved raw columns from source CSV.)

### Core invariant

- `split_factor` = product(`divisor` for that ticker where `split_date > trade_date`)
- `adj_price` = `raw_price / split_factor`
- `spot_px` = `adj_stkPx`
- All output tickers must be in the C4 liquidity-filtered precompute universe

### Strict split-date rule

- If `split_date == trade_date`, the split is **not** counted as future by the current contract.
- Only `split_date > trade_date` is included.
- Matches `_apply_adjustments()` in `split_adjuster.py` (`cum_factors["split_date"] > trade_date`).

### ORATS divisor semantics (unchanged)

- 2-for-1 split → `divisor = 2.0`
- 3-for-2 split → `divisor = 1.5`
- Historical raw prices divided by cumulative future divisors to express in post-all-splits terms

## Proposed implementation slices

### C5.1 — Planning doc only

**Deliverable:**

- `docs/tmp/c5_split_adjustment_design_plan.md`

No code changes.

### C5.2 — Split domain code audit ✓ (complete)

**Status:** Complete — accepted. **PASS WITH WARNINGS.** No code changed.

**Deliverable:** [docs/tmp/c5_split_domain_audit.md](c5_split_domain_audit.md)

**Outcome:**

- No true blockers before C5.3 (ticker loader + golden tests).
- Warnings accepted and documented (zero split-domain tests, stale defaults, `readjust_tickers` misnaming, `--tickers` flag collision, runbook order).
- Keep/archive decisions locked in § C5.2 audit decisions above.
- Archive file moves deferred to post-C5 if HD wants; not required for C5 closeout.

### C5.3 — Ticker universe loader

Add a small helper to load the C4 liquidity ticker universe.

**Required behavior:**

- Accepts CSV or parquet
- Accepts `Ticker` or `ticker`
- Dedupes tickers
- Drops null/blank values
- Returns sorted ticker list (or stable deterministic list)
- Has small unit tests

**Suggested files:** `src/data/ticker_universe.py`, `tests/unit/test_ticker_universe.py`

**Prerequisite:** C5.2 complete; this design doc reviewed by HD.

### C5.4 — Pure golden tests

Add tests for pure split-adjustment logic using tiny DataFrames only.

**Suggested file:** `tests/unit/test_split_adjuster.py`

**Required test cases:**

1. No split → `split_factor = 1`
2. One future split → factor applied before split date
3. On split date → split is not applied because rule is `split_date > trade_date`
4. After split date → factor = 1
5. Multiple future splits → product of all future divisors
6. Multiple tickers in same dataframe → each ticker gets its own factor
7. Missing optional price column → does not crash
8. Split before `min_split_date` is ignored

Use `SplitAdjuster.adjust_dataframe()` with synthetic splits parquet and tiny DataFrames — no ZIP I/O.

**Prerequisite:** C5.2 confirms `SplitAdjuster` is the canonical factor implementation (not a duplicate).

### C5.5 — Synthetic filtered ZIP-to-parquet test

Create a tiny raw ZIP fixture under `tmp_path` during the test and verify filtered adjusted parquet output.

**Fixture should include:**

- At least one ticker inside the C4 universe
- At least one ticker outside the C4 universe
- At least one ticker with a split
- At least one ticker without a split

**Assert:**

- Output parquet exists
- Required columns exist
- Row count reduced to only universe tickers
- Output contains no ticker outside the C4 universe
- `split_factor` correct
- `adj_*` values correct
- `spot_px == adj_stkPx`
- `overwrite=False` skips existing output
- `overwrite=True` rewrites output

### C5.6 — Filtered adjusted backfill mode

**Implementation: Option A (C5.2 approved).** Extend existing code — do not add `apply_split_adjustment_liquid.py` unless Option A fails during implementation.

**Files to modify:**

- `src/data/split_adjuster.py` — optional `ticker_universe` set; filter rows before write in `_process_single_zip`
- `scripts/apply_split_adjustment.py` — add `--ticker-universe`; wire to `load_ticker_universe()`

**Backfill only for C5** — not wired into `refresh_weekly_inputs.py` (C8 scope).

**Required behavior:**

- `--ticker-universe` → `C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv` (or override path)
- Filter raw rows to C4 universe tickers **before** write
- Apply split adjustment via existing `_apply_adjustments` / `SplitAdjuster`
- When `--ticker-universe` is set, default or strongly recommend `--adj-root C:/MomentumCVG_env/input/adjusted_liquid/`
- **Guardrail:** refuse or warn if `--ticker-universe` is set with `--adj-root` pointing at `C:/ORATS/data/ORATS_Adjusted/` (no accidental full-cache filtered writes)
- Skips existing outputs by default; supports `--overwrite`, `--years`, optional date bounds
- Preserves row-level columns plus adjusted columns

Do not implement incremental update handling in C5. Do not use `readjust_tickers()` for universe filtering.

### C5.7 — Split fetch scoped to liquidity universe

**C5.2 approved defaults** — extend `scripts/fetch_splits.py`:

| Flag / path | Default when scoped |
|-------------|---------------------|
| `--ticker-universe` | `C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv` (via `load_ticker_universe`) |
| `--output` | `C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet` |

- Fetch splits only for C4 liquid universe (~2,783 tickers).
- Do **not** overwrite global `C:/MomentumCVG_env/cache/splits_hist.parquet` by default.
- **Fallback only:** reuse global `splits_hist.parquet` if coverage for all liquid tickers is proven (document in closeout memo). Prefer scoped fetch for C5 closeout.

### C5.8 — Split output audit script

Add `scripts/audit_adjusted_liquid.py` (standalone; wired to `refresh_weekly_inputs.py split-audit` in **C8**).

**Minimum CLI options:**

- `--raw-root`
- `--adj-root`
- `--splits`
- `--ticker-universe`
- `--sample-tickers`
- `--start-date`
- `--end-date`
- `--max-files`
- `--report-path`
- `--strict`

**Minimum checks:**

1. `raw_root` exists
2. `adj_root` exists
3. `splits` file exists
4. `ticker universe` file exists
5. Required adjusted columns exist
6. Adjusted output contains only universe tickers
7. Raw row count after universe filtering equals adjusted row count
8. Expected `split_factor` matches adjusted `split_factor`
9. `adj_stkPx ≈ stkPx / split_factor`
10. `adj_strike ≈ strike / split_factor`
11. Adj bid/ask ≈ raw bid/ask / split_factor
12. `spot_px == adj_stkPx`
13. Report has PASS / WARN / FAIL

**Note:** C5.8 audits **on-disk artifacts** (raw/adjusted/splits). It does not replace C5.2 **code** audit.

### C5.9 — Real-data audit sample

Run one substantive sample audit, for example:

- **Tickers:** AAPL, TSLA, NVDA
- **Date range:** 2019-01-01 to 2024-12-31
- **Bounded file count:** `--max-files 20` or `--max-files 50`

**Smoke report path (first):**

```
C:/MomentumCVG_env/cache_c5_smoke/
```

**Formal report path (after acceptance):**

```
C:/MomentumCVG_env/cache/manifests/reports/
```

### C5.10 — Closeout memo ✓

Delivered as [sprint_memos/004_c5_adjusted_liquid.md](../sprint_memos/004_c5_adjusted_liquid.md) (2026-07-04).

## Acceptance criteria

C5 is accepted only if:

1. Planning doc exists and reflects C5.2 audit decisions.
2. **C5.2 audit memo exists and is accepted** ([c5_split_domain_audit.md](c5_split_domain_audit.md)).
3. **C5.3 starts only after this design doc is reviewed** post-audit.
4. Implementation follows **Option A** (extend `SplitAdjuster` + `apply_split_adjustment.py`) unless HD explicitly changes direction.
5. **No writes to `C:/ORATS/data/ORATS_Adjusted/`** during C5 filtered mode (guardrail enforced).
6. C4 liquidity ticker universe loader is tested.
7. Pure golden split tests pass.
8. Synthetic filtered ZIP-to-parquet test passes.
9. Filtered adjusted parquet backfill can run on a bounded sample under `input/adjusted_liquid/`.
10. Output adjusted parquets contain only C4 universe tickers.
11. Split fetch uses scoped `splits_hist_liquid.parquet`, **or** closeout memo documents proven global fallback.
12. Split output audit script exists.
13. One real-data audit report exists (PASS or PASS WITH WARNINGS).
14. C5 closeout memo exists. ✓
15. Full relevant pytest subset is green.
16. No strategy/S5/S8/ORCH files changed.

## Known risks and things to inspect

1. **`liquid_tickers.csv` is a precompute superset, not a PIT universe.** Downstream surface precompute may use it; S1 backtest universe must still use PIT panel lookup. Document clearly in audit reports and closeout memo.
2. **C5.2 audit complete** — core math trusted; defaults and filtered output still to be fixed in C5.3–C5.7.
3. **Duplication:** consolidate on `load_ticker_universe()` in C5.3; do not use `all_tickers.parquet` on C5 path.
4. **Existing `readjust_tickers` may sound targeted but still scans/processes many ZIPs.** Do not optimize in C5 unless audit blocks correctness. C5 filtered mode should filter rows at write time, not rely on `readjust_tickers` for universe scoping.
5. **Verify `split_date > trade_date` rule** is consistently documented and tested (C5.4 golden tests).
6. **Verify adjusted parquet preserves intended filtered raw rows** and adds adjusted columns without dropping in-universe data.
7. **Verify divisor semantics** from scoped split history match what `SplitAdjuster` assumes (ORATS 2-for-1 = divisor 2.0).
8. **Keep audit bounded and readable**; do not turn C5.8/C5.9 into full-cache validation.
9. **Avoid accidental writes** to full ORATS adjusted cache (`C:/ORATS/data/ORATS_Adjusted/`). Use separate `--adj-root` default for C5 filtered pool.
10. **Column name mismatch** — C4 outputs `Ticker` (capital T); older scripts expect `ticker` lowercase. Loader must handle both.
11. **Path inconsistency** — Some scripts reference `C:/MomentumCVG_env/cache/liquid_tickers.csv`; C4 production uses `C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv`. C5 should default to production C4 path; loader accepts override.
12. **Stale references** — `corporate_actions.py` cites `fetch_orats.py` (missing); runbook pipeline order may not match C4 raw-liquidity reality.

## Resolved by C5.2 audit

| Question | Decision |
|----------|----------|
| Filtered adjusted output root | `C:/MomentumCVG_env/input/adjusted_liquid/` |
| Scoped split history | Prefer `splits_hist_liquid.parquet` under same root |
| Filtered adjust implementation | **Option A** — extend `SplitAdjuster` + `apply_split_adjustment.py` |
| Ticker ordering | Sorted alphabetically |
| `build_ticker_universe.py` / `all_tickers.parquet` | Maintenance only; not C5 precompute path |
| Pipeline after C4 | `liquid_tickers.csv` → scoped fetch → filtered adjust → audit |

## Open questions for HD

1. **First real audit sample tickers** — default proposal: AAPL, TSLA, NVDA (C5.9).
2. **First bounded backfill window** — e.g. 2024 Q1 vs larger sample for C5.6 closeout evidence.
3. **`--strict` audit behavior** — fail on any mismatch vs WARN for minor tolerance (C5.8).

## Recommended next step

Review this updated design doc (post-C5.2). If approved, start **C5.3** (`load_ticker_universe` + unit tests), then **C5.4** pure golden split tests.
