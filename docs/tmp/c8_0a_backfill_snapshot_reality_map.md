# Sprint 004 — C8.0A immutable full-backfill snapshot reality map

**Task:** C8.0A — immutable full-backfill snapshot reality map
**Mode:** Audit / reconnaissance only (documentation-only repository audit; no code/test/artifact changes, no external data access)
**Starting HEAD:** `3805e46c88fcf5b58d76d4ae7a828abd8b7421c2`
**Builds on:** C8.0 (`docs/tmp/c8_0_refresh_pipeline_reality_map.md`, accepted at this HEAD).

> This maps how close the current repository is to building **one complete, immutable, isolated historical input snapshot** that is safe for feature generation and backtesting — deferring weekly incremental/repair. It is not a design. Every claim cites code, tests, or accepted evidence, and is classified with an evidence tag.

**Evidence tags:** `IMPLEMENTED_AND_TESTED` · `IMPLEMENTED_BUT_NOT_TESTED` · `SUPPORTED_ONLY_BY_MANUAL_OPERATION` · `PARTIALLY_IMPLEMENTED` · `DOCUMENTED_ONLY` · `NOT_IMPLEMENTED` · `UNSAFE`.

---

## 1. Scope and target question

**Target question:**

> Using the current full-backfill producers, path overrides, audits, manifest utilities, and downstream readers, how close is the repository to being able to create one complete, immutable, isolated historical input snapshot that is safe to use for feature generation and backtesting?

This audit deliberately ignores weekly-append/repair concerns (covered and deferred from C8.0) and focuses on the **build-from-scratch → audit → publish → consume** path.

---

## 2. Executive answer

**Feasibility: `FEASIBLE_WITH_MATERIAL_HARDENING`.**

The good news is structural: **all five producers already accept explicit input *and* output path arguments**, so each can, in principle, read from and write into an isolated snapshot root without touching accepted production artifacts. Full-overwrite semantics — a liability for incremental refresh — are actually the *correct* shape for a from-scratch backfill. C4 (liquidity) and C5 (adjusted chains) have accepted full-scale production evidence.

The blocking news is concentrated in four areas, none of which needs a major redesign but all of which need real work:

1. **No cross-layer completeness / same-generation audit.** Existing audits prove per-artifact internal consistency and (for adjusted/PIT) coverage, but nothing certifies that spot, surface, chains, and panel form one internally compatible generation, or that spot/surface derive from the *newly built* chains.
2. **Weak links in two producers.** `extract_spot_prices.py` silently swallows per-date failures and returns success, has no dedup/completeness check, and **no test at all**; `precompute_option_surface.py` writes A1 then A2 as two non-atomic writes with no pairing/completeness gate, and **has no accepted full-universe/full-history build evidence**.
3. **Downstream feature generation is not snapshot-isolatable.** The canonical backtest reader (`SurfaceRunner`/`SurfaceDataPaths`) is fully path-configurable at the Python API level, but the feature-source producer `precompute_straddle_history.py` **hard-codes** the spot-DB path, the universe path, and the `cache/` output; and the `run_surface_search.py` CLI only exposes `--cache-dir` (and passes a `contract_multiplier` kwarg the current dataclass does not accept).
4. **Manifest cannot represent the snapshot.** C1 models a single `cache_dir` with relative filenames; it omits adjusted chains and `liquid_tickers.csv` entirely, is never written by any run, writes non-atomically, and has no "current/latest" pointer.

So: a determined operator can already produce most of the raw artifacts into an isolated root by hand, but the repository cannot yet **fail closed on incompleteness**, **certify the snapshot as one generation**, **record its lineage in a manifest**, or **let a backtest consume it end-to-end (including features) without code changes**.

---

## 3. Backfill-only C8 success definition under evaluation

The narrower C8 target being evaluated (not approved here):

> Build a complete historical input snapshot from scratch in an isolated versioned root, audit it, publish it only after success, and use it for feature development and backtesting. Weekly incremental/repair deferred until shadow trading needs it.

For this to be *true*, the pipeline must be able to: (a) build every artifact from raw + current split history; (b) fail closed on any incomplete stage; (c) audit completeness and same-generation lineage, not just schema; (d) publish immutably with a lineage receipt; (e) be consumed by feature + backtest code from the isolated root. §13 lists which of these are blockers today.

---

## 4. Proposed snapshot-root concept under evaluation

Concrete inspection target (structure not endorsed; C8.1 decides):

```
C:/MomentumCVG_env/snapshots/<build_id>/
    input/liquidity/          # ticker_liquidity_{daily,weekly}_observations, panel, liquid_tickers.csv
    input/adjusted_liquid/    # splits_hist_liquid.parquet + {YYYY}/ORATS_SMV_Strikes_*.parquet
    cache/                    # spot_prices_adjusted.parquet, option_surface_{meta,quotes}_*, features/
    reports/                  # audit reports
    manifests/                # input_snapshot_<snapshot_id>.json
```

**Key structural tension found:** this layout separates `input/liquidity`, `input/adjusted_liquid`, and `cache`. But (i) the C1 manifest and the `run_surface_search.py` CLI both assume a **single `cache_dir`**; and (ii) `SurfaceDataPaths` derives the liquidity panel from `cache_dir/ticker_liquidity_panel.parquet` by default, not from `input/liquidity`. Consuming the three-root layout therefore requires per-artifact path overrides (available in the Python API, **not** in the CLI). See §10–§11.

---

## 5. Full-backfill dependency graph

Edge classes: `CURRENTLY_EXECUTABLE` (runnable now, sensible defaults) · `EXECUTABLE_WITH_MANUAL_ARGUMENTS` (runnable but requires explicit path args) · `PARTIALLY_SUPPORTED` (runs but with correctness/completeness gaps) · `NOT_SUPPORTED`.

| Edge | Class | Establishing CLI/code path |
|------|-------|----------------------------|
| raw ORATS → full liquidity backfill | EXECUTABLE_WITH_MANUAL_ARGUMENTS | `build_liquidity_panel.py --mode backfill --data-root <raw> --cache-dir <snap>/input/liquidity` (default `--cache-dir` is `cache`, so isolation needs explicit arg) |
| liquidity → `liquid_tickers` superset | CURRENTLY_EXECUTABLE | same run; `build_liquid_tickers` writes `liquid_tickers.csv` into `--cache-dir` |
| `liquid_tickers` → full scoped split fetch | PARTIALLY_SUPPORTED | `fetch_splits.py --ticker-universe <snap>/…/liquid_tickers.csv --out <snap>/…/splits_hist_liquid.parquet` — **requires ORATS token / network**; cannot run in this audit; full overwrite; no completeness validation of ticker set |
| split history → full adjusted chains | EXECUTABLE_WITH_MANUAL_ARGUMENTS | `apply_split_adjustment.py --raw-root <raw> --adj-root <snap>/input/adjusted_liquid --splits <snap>/…/splits_hist_liquid.parquet --ticker-universe <…> --overwrite` (filtered mode **forces** explicit `--adj-root`) |
| adjusted chains → full spot extraction | PARTIALLY_SUPPORTED | `extract_spot_prices.py --data-root <snap>/input/adjusted_liquid --output <snap>/cache/spot_prices_adjusted.parquet` — runs, but swallows per-date failures, no exit contract, no completeness/dedup |
| spot + chains → full weekly surface A1/A2 | PARTIALLY_SUPPORTED | `precompute_option_surface.py --data-root <snap>/input/adjusted_liquid --spot-db-path <snap>/cache/spot_prices_adjusted.parquet --output-root <snap>/cache --frequency weekly --overwrite` — bounded/full both write whole files; A1/A2 not atomic; no completeness audit |
| chains → C5 adjusted-layer audit | EXECUTABLE_WITH_MANUAL_ARGUMENTS | `audit_adjusted_liquid.py --raw-root --adj-root --splits --ticker-universe --years` (inventory full; math sampled) |
| surface → C6 surface audit | EXECUTABLE_WITH_MANUAL_ARGUMENTS | `audit_option_surface_artifacts.py --meta-path --quotes-path --data-root` (internal consistency; not completeness) |
| panel → C7 PIT-universe audit | EXECUTABLE_WITH_MANUAL_ARGUMENTS | `audit_pit_universe.py --panel-path --weekly-path --liquid-tickers-path` |
| all artifacts → input snapshot manifest | NOT_SUPPORTED | `write_manifest` exists but is **never called**; cannot represent multi-root artifacts (§10) |
| snapshot → feature generation | PARTIALLY_SUPPORTED | `precompute_straddle_history.py` **hard-codes** spot/universe/output paths (§11); `build_features.py --input --output` is configurable |
| snapshot → backtest | PARTIALLY_SUPPORTED | `SurfaceRunner(SurfaceDataPaths(...))` fully configurable in Python; `run_surface_search.py` CLI only `--cache-dir` and is out of sync with the dataclass (§11) |

**Overall:** every *producer* edge is at least `EXECUTABLE_WITH_MANUAL_ARGUMENTS`; the true gaps are the **manifest edge (`NOT_SUPPORTED`)**, the **completeness/correctness of spot & surface (`PARTIALLY_SUPPORTED`)**, and the **feature/backtest consumption edges (`PARTIALLY_SUPPORTED`)**.

---

## 6. Producer isolation and capability matrix

| Producer | Isolated inputs | Isolated outputs | Hidden global dependency | Full-history capability | Reliable failure status | Atomic publication | Safe for C8 backfill as-is |
|----------|:---------------:|:----------------:|:------------------------:|:-----------------------:|:-----------------------:|:------------------:|:--------------------------:|
| `build_liquidity_panel.py` | Yes | Yes | No | Yes | Yes | Partial | Partial |
| `fetch_splits.py` | Yes | Yes | No | Yes | Partial | No | Partial |
| `apply_split_adjustment.py` | Yes | Yes | No | Yes | Partial | No | Partial |
| `extract_spot_prices.py` | Yes | Yes | No | Yes | No | No | No |
| `precompute_option_surface.py` | Yes | Yes | No | Partial | Partial | No | Partial |

### Per-producer answers (10 questions each)

**`build_liquidity_panel.py`** (`IMPLEMENTED_AND_TESTED` for backfill; `tests/unit/test_build_liquidity_panel.py`)
1. Isolated inputs: yes — `--data-root` (raw). 2. Isolated outputs: yes — `--cache-dir` receives all four artifacts + report (`cache_dir/manifests/reports/`). **Default `--cache-dir` is `C:/MomentumCVG_env/cache`, not `input/liquidity`**, so isolation *requires* the explicit arg. 3. Hidden global read: no. 4. Sidecars outside root: no (staging dir + report under `--cache-dir`). 5. Accidental prod overwrite: only if operator points `--cache-dir` at production. 6. Creates parents: yes (`write_artifacts` staging; report `mkdir`). 7. Fails if destination exists: no — backfill overwrites. 8. Atomic: **Partial** — staging dir then per-file `os.replace` (atomic per file, not one cross-file transaction). 9. Failed run leaves apparent-complete output: unlikely (staged commit at end). 10. Nonzero on incomplete: yes (`LiquidityPanelError`, empty-source fail — `test_empty_source_fails`).

**`fetch_splits.py`** (`IMPLEMENTED_BUT_NOT_TESTED` for resume; validation `IMPLEMENTED_AND_TESTED`, `tests/unit/test_fetch_splits_cli.py`)
1. Isolated inputs: yes — `--ticker-universe`/`--tickers`. 2. Isolated outputs: yes — `--output/--out`. 3. Hidden global: no, **but requires ORATS network + `--token`** (cannot run in this audit). 4. Sidecar: yes — `<out>.checkpoint.parquet` written **beside output** (inside root). 5. Accidental prod overwrite: scoped mode requires explicit `--out`. 6. Parents: yes. 7. Fails if exists: no (overwrites/resumes). 8. Atomic: **No** — single direct `to_parquet` after validation. 9. Apparent-complete on failure: **checkpoint can be mistaken for output if operator points readers at it**; interrupted run leaves only the sidecar, not a partial main file. 10. Nonzero on incomplete: **Partial** — `sys.exit(message)` on validation failure (conflicting/invalid divisor, outside-universe fail closed), but success returns `None` and there is no completeness check that *every* universe ticker was queried/answered.

**`apply_split_adjustment.py` / `SplitAdjuster`** (`IMPLEMENTED_AND_TESTED` for math/CLI wiring; `tests/unit/test_split_adjuster.py`, `test_apply_split_adjustment_cli.py`, `test_split_adjuster_filtered_zip.py`)
1. Isolated inputs: yes — `--raw-root`, `--splits`, `--ticker-universe`. 2. Isolated outputs: yes — `--adj-root` (**filtered mode requires it explicitly** so the legacy mirror is never overwritten — a genuine isolation guard). 3. Hidden global: no in filtered mode; **default `--adj-root=None` targets the legacy mirror** if omitted. 4. Sidecar: no. 5. Accidental prod overwrite: prevented in filtered mode by the required-`--adj-root` guard. 6. Parents: yes. 7. Fails if exists: no — default **skip-existing**; `--overwrite` forces. 8. Atomic: **No** — each daily parquet is a direct `to_parquet` (`src/data/split_adjuster.py::_process_single_zip`). 9. Apparent-complete on failure: yes — an interrupted backfill leaves a partial set of daily files that *looks* like a directory of valid parquets; skip-existing on rerun then completes without knowing which were missing. 10. Nonzero on incomplete: **Partial** — per-file exceptions are logged; the run can complete having skipped/failed some days.

**`extract_spot_prices.py`** (`UNSAFE` for a completeness contract; **no test exists** — confirmed no `test_extract_spot_prices*` in `tests/**`)
1. Isolated inputs: yes — `--data-root`. 2. Isolated outputs: yes — `--output`. 3. Hidden global: no. 4. Sidecar: no. 5. Accidental prod overwrite: only if `--output` points at production. 6. Parents: yes (`Path(output).parent`… via pandas/`to_parquet`; directory must exist — minor). 7. Fails if exists: no — **whole-file overwrite of the requested year range** (running `--year 2026` replaces the entire multi-year file). 8. Atomic: **No** — single direct `to_parquet`. 9. Apparent-complete on failure: **Yes / UNSAFE** — `extract_spot_prices_for_date` catches every exception per date and returns `[]`; `main` never returns nonzero and does not check for empty/short output, so a partial spot DB is written and reported as success. 10. Reliable nonzero on incomplete: **No**.

**`precompute_option_surface.py`** (`IMPLEMENTED_AND_TESTED` for CLI safety/scoping, `tests/unit/test_precompute_option_surface_cli.py`; full build **not** evidenced)
1. Isolated inputs: yes — `--data-root`, `--spot-db-path`. 2. Isolated outputs: yes — `--output-root`. 3. Hidden global: no. 4. Sidecar: `--log-file` optional (operator-chosen). 5. Accidental prod overwrite: guarded — `check_overwrite_guard` exits 2 if outputs exist without `--overwrite`; but a bounded run to the **canonical filename** with `--overwrite` replaces the whole file. 6. Parents: yes (`--output-root`). 7. Fails if exists: yes without `--overwrite` (exit 2). 8. Atomic: **No** — A1 then A2 written as two sequential direct `to_parquet` calls; failure between them leaves unpaired artifacts. 9. Apparent-complete on failure: possible — per-(ticker,entry) failures become `surface_valid=False` rows, not crashes (`option_surface_analyzer.process_single_entry`), so a run "completes" even if most rows failed. 10. Nonzero on incomplete: **Partial** — exit 0 on a run full of failure rows; exit 2 only for the overwrite guard/usage.

**Full-history capability = Partial for surface** because the output filename encodes declared `start_year/end_year` (`option_surface_meta_{frequency}_{start}_{end}.parquet`), not the actual scope, so a bounded run can silently produce a canonically-named file whose content is not full-history.

---

## 7. Full-backfill completeness by artifact

No thresholds are invented; each item marks where **C8.1 must define the completeness denominator**.

### Liquidity
- Expected raw date range: driven by `--start-year/--end-year` (defaults 2017–2026) and raw ZIP presence (`discover_orats_trading_dates`). Accepted C7 evidence: **477 panel snapshots**, ~2.4M panel rows.
- Daily/weekly coverage, first valid rolling snapshot, last snapshot: derived from watermarks; **no explicit "all expected weeks present" gate** beyond the incremental watermark (irrelevant to backfill).
- `liquid_tickers` regeneration: recomputed from the full merged panel each run (`build_liquid_tickers`).
- Same-run correspondence: **Yes** — all four artifacts are produced in one `write_artifacts` commit, so they are one generation by construction.
- Narrow requested range changes superset meaning: **Yes** — `liquid_tickers.csv` reflects only the built window; a short backfill yields a different (smaller) historical superset. C8.1 must fix the canonical window.

### Split history
- Every liquid ticker queried: `fetch_splits` iterates the universe; **no post-fetch assertion that every ticker received a completed status** (tickers with no split are simply absent from the output — there is no explicit "queried, no split" record). `PARTIALLY_IMPLEMENTED`.
- Checkpoint vs final output: distinct files (`.checkpoint.parquet` sidecar) — but nothing prevents a reader from mistaking one for the other.
- Silent omission: a completed fetch can omit tickers ORATS returns nothing for, and this is **indistinguishable from "no split."**
- Output completeness validated: only **content** validation (dedup, conflict fail-closed, divisor sanity), not **coverage** of the universe. Accepted C5 evidence: 1,347 rows / 819 tickers over a 2,783-ticker universe (i.e., most tickers legitimately have no split, which is exactly why absence is ambiguous).

### Adjusted chains
- Every raw date → one adjusted parquet: **not guaranteed** — skip-existing + per-file failures can leave gaps; `audit_adjusted_liquid::audit_inventory` detects raw-vs-adjusted date gaps **only for audited years**. Accepted C5.10B: 2,299 files, exit 0.
- All expected liquid tickers present on a date: **not asserted** — filtered mode writes only universe rows, but there is no per-date ticker-count floor.
- Empty/failed daily processing skipped silently: possible (per-file try/except).
- File-count equivalence sufficient: **No** — equal counts do not prove per-row completeness.
- Split factors validated against current split history: **No** — the audit checks `adj == raw / stored_split_factor` using the *stored* factor, never recomputing from the split file (C8.0 finding, unchanged).

### Spot
- Expected date range / relation to chains: should equal the adjusted-chain trading days; **not enforced or audited**.
- Ticker coverage: all tickers present per day; **not audited**.
- Per-date failures swallowed: **Yes** (see §6).
- Partial DB still written: **Yes**.
- `(date, ticker)` uniqueness enforced: **No** — `groupby('ticker').first()` per day yields one row but there is no post-write uniqueness assertion.
- Null/nonfinite rejection: **No**.
- **There is no spot audit at all.** C8.1 must define the spot completeness gate.

### Surface A1/A2
- Weekly entry schedule: `weekly_trade_dates_in_range` (+21-day tail for strict expiry).
- Every eligible (ticker, entry) has an A1 row incl. failures: **Yes** for processed pairs (`process_single_entry` always emits a metadata row), but the **denominator (which pairs are "eligible")** is not audited against the universe/schedule.
- Expected A1 count / coverage denominator: **undefined** — no completeness audit; C6 evidence is bounded (65 A1 rows, 5×13).
- A2 ↔ valid A1: `audit_option_surface_artifacts` checks the join and grain on whatever is present (0 orphans in C6.4 bounded).
- Full history vs bounded overwrite distinguishable: **No** — filename encodes declared years, not actual content.
- Filenames describe content accurately: **No** (see §6).
- A1/A2 completeness audited: **No** — only internal integrity.

---

## 8. Runtime and scale evidence

| Stage | Accepted scale | Recorded runtime | Workers | Output size | Full or bounded | Lineage matches current code? |
|-------|----------------|------------------|---------|-------------|-----------------|-------------------------------|
| C4 full liquidity backfill | 2017→2026; 477 panel snapshots; ~2.4M panel rows | not recorded in memo | n/a | 4 artifacts | **Full** (accepted C4) | Yes (C4 accepted) |
| C5 full adjusted-liquid build | 2,783-ticker universe; **2,299 files**, 2017→2026; splits 1,347 rows / 819 tickers | run log `c5_10b_full_backfill_run_log.txt` (exit 0); wall-time not summarized in memo | parallel (`--workers`) | 2,299 daily parquets | **Full** (C5.10B/C5.10D PASS) | Yes (C5 accepted) |
| Spot full extraction | — | **none** | — | — | **No accepted evidence** — C5 memo: "full Stage A re-extract … not part of C5 closeout" | producer untested |
| C6 full surface build | — | **none** | — | — | **No accepted full-universe/full-history evidence** — C6.4 is bounded 5×13 (65 A1, 2,114 A2); existing `weekly_2018_2026` cache has **unknown lineage (WARN)** | C6.1C strict expiry accepted; full build never run under it |
| C5 adjusted audit | full inventory + **sampled** math (10 files × 20k rows, seed 57) | `c5_10d` PASS | n/a | report | inventory full / math sampled | Yes |
| C6 surface audit | bounded 5×13 | C6.4 PASS | n/a | report | **Bounded** | Yes |
| C7 PIT audit | 477 snapshots superset coverage; 3 bounded rolling samples | **545.6 s** | n/a | report | coverage full / rolling bounded | Yes (C7.4R) |

**Explicit gaps (verify-flagged in the task):**
- **A full production spot rebuild has NO accepted evidence.** `NOT_IMPLEMENTED` evidence-wise; producer is untested.
- **A fresh full-universe/full-history C6 surface build has NO accepted evidence.** Only bounded 5×13; the canonical historical file's lineage is explicitly unknown.
- No runtime is estimated where the repository records none; those cells are marked "none/not recorded."

---

## 9. Audit gates for a complete batch snapshot

| Gate | Existing audit | Fully supported? | Missing proof |
|------|----------------|:----------------:|---------------|
| All raw dates have adjusted output | `audit_adjusted_liquid::audit_inventory` | Partial | Only for audited years, and only where raw ZIP exists; no global calendar reference |
| Adjusted math uses **current** split history | `audit_adjusted_liquid::audit_raw_math_sample` | No | Uses stored `split_factor`; never recomputes from current split file |
| Spot covers all required dates/tickers | — | No | **No spot audit exists** |
| Spot values derive from current adjusted chains | — | No | **No spot audit / no lineage check** |
| A1 covers expected ticker-entry pairs | `audit_option_surface_artifacts` | No | Checks present rows; no eligibility denominator / completeness |
| A2 joins correctly to A1 | `audit_option_surface_artifacts` (`check_a1_a2_join`, grain checks) | Yes (internal) | Only for present rows; not completeness |
| Surface uses the newly built spot and chains | — | No | No lineage/freshness linkage between surface and its spot/chain inputs |
| PIT universe has strict prior-snapshot semantics | `audit_pit_universe` + `pipeline.step1_get_universe` (`<`) | Yes | — (`test_step1_universe_contract`, C7.4R) |
| Every S1 ticker is available downstream | — | No | No audit cross-checks S1 universe membership against surface/chain availability |
| All artifacts belong to the same generation | — | No | **No same-generation lineage concept anywhere** |

Distinguishing the five audit dimensions:
- **Internal consistency:** strong (schema, grain, join, invariant, provenance) across C5/C6/C7.
- **Source-to-output correctness:** partial — adjusted math checked against a *stored* factor (sampled); spot/surface not checked against sources.
- **Coverage completeness:** partial — adjusted inventory (by year) and PIT superset coverage (477 snapshots) exist; spot and surface have **none**.
- **Cross-layer freshness:** **absent** — no audit proves spot/surface were rebuilt from the current chains/splits.
- **Same-generation lineage:** **absent** — no shared build id / manifest binds the artifacts.

**Central conclusion:** each artifact can pass its own checks while the *batch as a whole* remains uncertified for research use. Completeness gates for spot and surface, plus a same-generation lineage check, are the core missing audit capabilities.

---

## 10. Manifest and publication reality

Source: `src/data/input_snapshot.py` (`IMPLEMENTED_AND_TESTED` as a library, `tests/unit/test_input_snapshot.py`; **never called by any run** — C8.0 finding, unchanged).

- **Snapshot-root-relative paths without changing C1?** **No, not cleanly.** Artifact values are strings under a single `cache_dir`; `manifest_to_dict`/`manifest_from_dict` require the `cache_dir` field and treat artifacts as relative filenames. A three-root snapshot (`input/liquidity`, `input/adjusted_liquid`, `cache`) cannot be represented without either flattening to one root or extending the schema.
- **Does `cache_dir` semantically allow a snapshot root?** Technically a caller may pass any path (e.g., `<snap>/cache` or the snapshot root), and `default_manifest_path(cache_dir, snapshot_id)` = `{cache_dir}/manifests/input_snapshot_{snapshot_id}.json`. But the *intent* is the Stage A cache; using it as a multi-root anchor is a semantic stretch.
- **Adjusted daily chains represented?** **No.** Artifact keys are only `splits`, `spot_prices`, `liquidity_panel`, `option_surface_meta`, `option_surface_quotes` (`ARTIFACT_*` constants). The 2,299-file adjusted chain directory is not a manifest artifact.
- **`liquid_tickers.csv` represented?** **No.**
- **Reports sufficient to reference omitted artifacts?** Report fields exist in the dataclass but are **unpopulated** (C3+); they cannot currently substitute for artifact identity.
- **One manifest identifying the complete source generation?** **No** — no build-id binding across producers, no fingerprints, only a logical `snapshot_id` over the five listed relative paths + params.
- **`write_manifest` atomic?** **No** — `target.open("w")` + `json.dump` directly (creates parents; no temp+rename).
- **Current/latest snapshot pointer today?** **No** — no `latest`/`current` symlink or pointer file mechanism.
- **Readers consume the manifest directly?** **No** — no downstream reader loads a manifest; readers take explicit paths (§11).

Separation of concerns:
- *What C1 technically permits:* arbitrary relative filename strings + params under one `cache_dir`; deterministic identity.
- *What its accepted schema intends:* a minimal receipt for five cache-resident Stage A artifacts.
- *What the current CLI assumes:* a single `--cache-dir`, preview-only, never written.
- *What a complete backfill snapshot needs:* multi-root artifact addressing (incl. adjusted chains + `liquid_tickers`), a shared generation id, existence/lineage evidence, atomic write, and a current-pointer. **These are C8.1 decisions; no schema change proposed here.**

---

## 11. Downstream consumer path reality

Path-config classes: **explicitly configurable** (constructor/CLI arg) · **configurable through BacktestConfig/JSON** · **hard-coded default but overridable** · **hard-coded and not overridable**.

| Consumer | Artifact used | How path is supplied | Point to snapshot root today? | Required code change |
|----------|---------------|----------------------|:-----------------------------:|----------------------|
| `ORATSDataProvider` (adjusted-chain reader) | adjusted daily parquets | `data_root=` ctor arg (`src/data/orats_provider.py`) | Yes | none |
| `SpotPriceDB` | spot parquet | `SpotPriceDB.load(path)` explicit | Yes | none |
| liquidity-panel / S1 reader (`SurfaceRunner`) | `ticker_liquidity_panel.parquet` | `SurfaceDataPaths.liquidity_panel_path` (else `cache_dir/…`) | Yes (via override) | none for API; CLI cannot override |
| A1/A2 surface reader (`OptionSurfaceDB.load`) | meta + quotes | explicit `(meta, quotes)` paths, wired via `SurfaceDataPaths.resolved_surface_{meta,quotes}_path` | Yes (via override) | none for API; CLI cannot override |
| `SurfaceRunner` (canonical backtest) | panel + A1/A2 + features + earnings | `SurfaceDataPaths(cache_dir, features_dir, liquidity_panel_path, surface_meta_path, surface_quotes_path, earnings_path)` | Yes (Python API) | none in API |
| `run_surface_search.py` (canonical CLI) | same | **only `--cache-dir`**; derives all others from it | Partial | CLI needs per-artifact args; **also passes `contract_multiplier=` which `SurfaceDataPaths` does not accept → out of sync / likely `TypeError`** |
| feature source `precompute_straddle_history.py` | adjusted chains + spot + universe | `--data-root` only; **`SPOT_DB_PATH`, `TRADE_UNIVERSE_FILE`, output `cache/` all hard-coded** | **No** | must add `--spot-db-path`, `--universe`, `--output`; also returns exit 0 on missing deps |
| feature builder `build_features.py` | straddle_history | `--input` / `--output` explicit | Yes | none |
| legacy engine `run_backtest.py` + `BacktestConfig` | adjusted chains (+strategy) | `--config` JSON → `data_provider.params.data_root` (`DEFAULT_CONFIG`) | Yes (JSON) | none (non-canonical path) |
| baseline `DEFAULT_CONFIG` (`src/backtest/config.py`) | data_root default | JSON override | hard-coded default but overridable | none |

**Blocking downstream question — can a snapshot built today be used directly, without replacing global files?**

- **Backtest structure/signals via the Python API: yes.** `SurfaceRunner(SurfaceDataPaths(cache_dir=<snap>/cache, liquidity_panel_path=<snap>/input/liquidity/ticker_liquidity_panel.parquet, surface_meta_path=<snap>/cache/<meta>, surface_quotes_path=<snap>/cache/<quotes>, features_dir=<snap>/cache/features))` consumes the snapshot with **no code change** (`OptionSurfaceDB.load`, `SpotPriceDB.load`, `ORATSDataProvider` all take explicit paths). This is the strongest positive finding.
- **Via the canonical CLI (`run_surface_search.py`): no.** It exposes only `--cache-dir`, so it (i) cannot point the panel at `input/liquidity`, (ii) cannot select a non-`weekly_2018_2026` surface filename, and (iii) currently passes an unsupported `contract_multiplier` kwarg — meaning the canonical CLI is not even a verified consumer at this HEAD.
- **Feature generation: no.** `precompute_straddle_history.py` hard-codes the spot DB, universe, and output paths, so features for a snapshot cannot be generated without code change (or writing files into the global `cache/`).

**Net:** downstream *readers* are snapshot-ready; downstream *entrypoints* (feature-source CLI and search CLI) are not. This is a material blocker for the "use it for feature development and backtesting" clause.

---

## 12. Full-backfill scenario analysis

### Scenario A — first complete snapshot build (empty destination)
- **Completes with current code (manual args):** liquidity backfill; adjusted-chain build; audits (adjusted/surface/PIT).
- **Requires ORATS token/network (cannot run in audit):** split fetch.
- **Completes but with correctness/completeness gaps:** spot extraction (silent partials), surface build (no completeness/atomicity), feature generation (hard-coded paths).
- **Needs code hardening before trustable:** spot exit contract + completeness; surface A1/A2 pairing + completeness audit; manifest write; feature-source path args.
- Empty destination *helps*: skip-existing in `apply_split_adjustment` is harmless when nothing exists, so a clean root avoids stale-skip risk.

### Scenario B — failed spot extraction (some dates fail, producer continues)
- Artifact written: a spot parquet **missing the failed dates**. Exit code: **0**. Existing audits: **do not detect** it (no spot audit; surface build would later emit `no_spot_price`/`no_spot_at_expiry` failure rows, which look like ordinary data gaps). `UNSAFE`.

### Scenario C — failure between surface A1 and A2 writes
- On disk: **A1 present, A2 absent** (or a stale prior A2). Rerun safety: without `--overwrite`, the overwrite guard blocks (exit 2) because A1 exists → operator must clear/`--overwrite`. Publication risk: nothing binds A1 and A2, so a naive publisher could treat the A1-only state as complete. `UNSAFE` absent a pairing gate.

### Scenario D — rebuild with the same requested historical range
- Current producers would: liquidity/adjusted (`--overwrite`) recompute-and-overwrite in place; surface refuse without `--overwrite` (exit 2) then overwrite; spot overwrite; **none assign a new build root or refuse to protect the prior snapshot.** So a same-destination rebuild **mutates the existing snapshot in place** (violating immutability). C8.1 must choose refuse / new-build-root / delete-and-rebuild. (No policy invented here.)

### Scenario E — new split discovered before a future research snapshot
- A **full rebuild from raw + current split history** naturally propagates the split: `fetch_splits` returns the new divisor → `apply_split_adjustment` recomputes all daily chains → `extract_spot_prices` re-derives spot → `precompute_option_surface` rebuilds A1/A2. **No incremental repair engine is needed** for a from-scratch snapshot — this is the central advantage of the backfill-only framing.
- Residual audit gap: because the adjusted-math audit uses the *stored* factor and there is no spot/surface lineage audit, a rebuild that **partially** ran (e.g., chains rebuilt but spot/surface skipped by skip-existing on a **non-empty** root) could still pass. Mitigated only by building into an **empty** root every time.

### Scenario F — newly added historical-superset ticker
- A full rebuild into an empty root naturally includes the ticker across split fetch, adjusted history, spot, A1, A2, and PIT coverage (all are recomputed from the current `liquid_tickers.csv`).
- **Hidden risk that disappears with an empty root:** `apply_split_adjustment` skip-existing and `precompute_option_surface` overwrite-guard would otherwise skip already-present files and silently exclude the new ticker; on an empty destination these skips cannot fire. **Risk that remains:** if any producer is pointed at a partially-populated root, the new ticker can be silently omitted.

### Scenario G — use snapshot in a backtest
Exact steps today (Python API, no code change):
```python
from src.backtest.surface_run_config import SurfaceDataPaths
from src.backtest.surface_runner import SurfaceRunner
paths = SurfaceDataPaths(
    cache_dir=Path(r"<snap>/cache"),
    liquidity_panel_path=Path(r"<snap>/input/liquidity/ticker_liquidity_panel.parquet"),
    surface_meta_path=Path(r"<snap>/cache/option_surface_meta_weekly_<S>_<E>.parquet"),
    surface_quotes_path=Path(r"<snap>/cache/option_surface_quotes_weekly_<S>_<E>.parquet"),
    features_dir=Path(r"<snap>/cache/features"),
)
runner = SurfaceRunner(data_paths=paths)
```
Blocked via CLI (`run_surface_search.py` only takes `--cache-dir`); blocked for feature generation (hard-coded `precompute_straddle_history.py`). So end-to-end CLI consumption needs code change; programmatic backtest consumption does not.

---

## 13. Blocking gaps for batch C8

Capabilities required to *build from scratch, fail closed, audit completeness, publish immutably, record lineage, and consume from backtesting* — each verified above, not assumed:

1. **Spot producer hardening** (`extract_spot_prices.py`): real exit-code contract, fail-closed on per-date failures, `(date,ticker)` uniqueness + null/nonfinite rejection, and first tests. *Evidence:* §6, §7, §8 (no evidence, no tests); Scenario B. `UNSAFE` today.
2. **Surface completeness + paired publication** (`precompute_option_surface.py`): A1/A2 as one transaction; a completeness audit against the eligible (ticker,entry) denominator; filenames that reflect actual scope. *Evidence:* §6, §7, §9; Scenario C. No full build evidence (§8).
3. **Cross-layer completeness & same-generation audit gate**: prove all raw dates have chains, spot covers chain dates/tickers, surface derives from the built spot/chains, and all artifacts share one generation. *Evidence:* §9 (five gates unsupported).
4. **Snapshot manifest that represents the real snapshot**: multi-root artifacts incl. adjusted chains + `liquid_tickers.csv`, a shared generation id, existence evidence, atomic write, and a current/latest pointer — and it must actually be **written by the build**. *Evidence:* §10.
5. **Downstream path wiring for consumption**: feature-source producer (`precompute_straddle_history.py`) must accept spot/universe/output paths; the search CLI must accept per-artifact overrides (and be reconciled with `SurfaceDataPaths`). *Evidence:* §11.
6. **Immutability/rebuild policy + empty-root guarantee**: since skip-existing/overwrite-guard only behave safely on an empty destination, the build must guarantee a fresh root (or fail). *Evidence:* Scenarios D, E, F.
7. **Adjusted-math freshness (or empty-root reliance)**: the adjusted audit cannot detect a stale factor; a from-scratch build into an empty root sidesteps this, but C8.1 must decide whether to add a factor-recomputation check. *Evidence:* §9.

---

## 14. Incremental capabilities safe to defer

Each is unnecessary for a from-scratch research snapshot because a **complete rebuild** reproduces the correct state directly:

- **New-date append** (liquidity/spot/surface): a rebuild covers all dates; no append needed.
- **Split-history delta detection**: a rebuild re-fetches full history and re-adjusts everything (Scenario E); no diff needed.
- **Selected-ticker split repair / preservation of unaffected tickers**: a rebuild readjusts all tickers correctly; no scoped repair needed.
- **Spot row-level merge / surface row-level merge**: whole-file writes are exactly right for a from-scratch build.
- **Watermark-based refresh / changed-only audits**: full audits over the whole snapshot suffice for a one-shot build.
- **Continuous scheduling / notifications / dashboards / distributed execution / remote backups**: operational maturity, not correctness — defer to shadow/weekly operation.

These map to the C8.0 "missing blocking capabilities" list; they are **blockers for weekly/repair operation** but **not** for a build-from-scratch research snapshot.

---

## 15. Feasibility classification

**`FEASIBLE_WITH_MATERIAL_HARDENING`.**

Basis (not merely the existence of full-overwrite code):
- **Producer isolation:** strong — all five accept explicit input+output paths; `apply_split_adjustment` even enforces explicit `--adj-root` in filtered mode. *(favorable)*
- **Failure correctness:** weak — spot returns success on partial output; surface counts failure rows as a "complete" run; several writes non-atomic. *(needs work)*
- **Audit completeness:** weak — no spot audit, no surface completeness, no same-generation/freshness gate. *(needs work)*
- **Manifest suitability:** weak — cannot represent the snapshot, never written, non-atomic, no pointer. *(needs work)*
- **Downstream consumption:** mixed — readers/API ready; feature-source and search CLIs not. *(needs work)*
- **Production-scale evidence:** partial — C4/C5 full-scale accepted; **spot and full surface have none.** *(gap)*

The gaps are numerous but each is a bounded hardening/wiring task on top of producers whose *shape* (from-scratch, full-overwrite, path-configurable) already fits a backfill snapshot. Nothing requires re-architecting the producers or the strategy path. Hence **material**, not **small**, and not **major redesign**.

---

## 16. Open design decisions for C8.1

Each: *why a decision is required · facts discovered · risk of choosing wrong.*

1. **Snapshot-root layout & single-vs-multi-root addressing.** *Why:* C1 manifest, `run_surface_search.py`, and `SurfaceDataPaths` defaults all assume one `cache_dir`, but the proposed layout uses three roots. *Facts:* §4, §10, §11. *Risk:* a three-root layout that the manifest/CLI can't address forces per-run manual wiring or file copying.
2. **Snapshot success definition & completeness denominators.** *Why:* no artifact defines "complete." *Facts:* §7, §9. *Risk:* certifying an incomplete snapshot as research-ready.
3. **Spot producer contract.** *Why:* silent partials + exit 0 + no tests. *Facts:* §6, §7, Scenario B. *Risk:* undetected missing spot rows silently corrupt surfaces and signals.
4. **Surface A1/A2 transaction & completeness audit.** *Why:* two non-atomic writes, no completeness gate, filename ≠ content. *Facts:* §6, §7, §9, Scenario C. *Risk:* unpaired/partial surface consumed downstream.
5. **Same-generation lineage / manifest content.** *Why:* nothing binds artifacts to one build. *Facts:* §9, §10. *Risk:* mixing artifacts from different builds without detection.
6. **Manifest schema & artifact coverage (incl. adjusted chains, liquid_tickers), atomic write, current pointer.** *Why:* current schema omits them and is never written. *Facts:* §10. *Risk:* an unusable or misleading receipt.
7. **Immutability & same-range rebuild policy.** *Why:* producers mutate in place. *Facts:* Scenario D. *Risk:* silently overwriting an accepted snapshot; loss of reproducibility.
8. **Empty-root guarantee & skip-existing behavior.** *Why:* correctness of split-adjust/surface depends on a clean destination. *Facts:* Scenarios E, F. *Risk:* new/changed tickers silently omitted on a dirty root.
9. **Downstream path wiring (feature-source CLI + search CLI) and `run_surface_search`/`SurfaceDataPaths` reconciliation.** *Why:* feature generation and the canonical CLI can't target a snapshot; the CLI passes an unsupported kwarg. *Facts:* §11. *Risk:* a built snapshot that cannot actually be used without ad-hoc code.
10. **Whether to add adjusted-math freshness (factor recomputation) or rely solely on empty-root builds.** *Why:* the audit can't detect a stale factor. *Facts:* §9, Scenario E. *Risk:* a partial rebuild passing audits with stale factors.
11. **Full-scale spot & surface production evidence requirement before trusting the snapshot.** *Why:* neither has accepted full-scale evidence. *Facts:* §8. *Risk:* first full surface/spot build surfacing scale/runtime/correctness issues late.

---

## 17. Inspected files

**Docs:** `AGENTS.md`, `docs/agenda/current_sprint.md`, `docs/agenda/sprint004_execution_guardrails.md`, `docs/tmp/c8_0_refresh_pipeline_reality_map.md`, `docs/repo_map.md`, `docs/v1_weekly_runbook.md`, `docs/v1_universe_protocol.md`, `docs/surface_engine_data_contract.md`, `docs/sprint_memos/004_c4_liquidity_panel.md`, `004_c5_adjusted_liquid.md`, `004_c6_option_surface.md`, `004_c7_pit_universe.md`.

**Producers & CLIs:** `scripts/build_liquidity_panel.py`, `fetch_splits.py`, `apply_split_adjustment.py`, `extract_spot_prices.py`, `precompute_option_surface.py`, `precompute_straddle_history.py`, `build_features.py`, `run_surface_search.py`, `run_backtest.py`, `refresh_weekly_inputs.py`, `audit_adjusted_liquid.py`, `audit_option_surface_artifacts.py`, `audit_pit_universe.py`.

**Modules:** `src/data/input_snapshot.py`, `paths.py`, `trading_day.py`, `split_adjuster.py`, `spot_price_db.py`, `orats_provider.py`, `ticker_universe.py`, `pit_universe_audit.py`; `src/features/option_surface_analyzer.py`, `option_surface_contract.py`; `src/backtest/config.py`, `pipeline.py`, `surface_run_config.py`, `surface_runner.py`, `run_config.py`, `engine.py`/`engine_v2.py` (path references).

**Tests surveyed for claims:** `tests/unit/test_build_liquidity_panel.py`, `test_fetch_splits_cli.py`, `test_apply_split_adjustment_cli.py`, `test_split_adjuster.py`, `test_split_adjuster_filtered_zip.py`, `test_precompute_option_surface_cli.py`, `test_option_surface_contract.py`, `test_audit_option_surface_artifacts.py`, `test_audit_adjusted_liquid.py`, `test_pit_universe_audit.py`, `test_input_snapshot.py`, `test_spot_price_db.py`, `tests/contract/test_step1_universe_contract.py`, plus a full `tests/**` inventory confirming **no** `test_extract_spot_prices*` and **no** dedicated `precompute_straddle_history` test.

**Searches performed:** `data_root`, `cache_dir`, `output_root`, `spot_db_path`, `option_surface_meta/quotes`, `liquid_tickers`, `SurfaceDataPaths(`, `SurfaceRunner(`, `contract_multiplier`, `write_manifest`, `to_parquet`, `os.replace`, `add_argument` (per producer), across `src/`, `scripts/`, `tests/`.
