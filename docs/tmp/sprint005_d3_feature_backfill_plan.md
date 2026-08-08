# Sprint 005 D3 — Standalone feature backfill plan

**Status:** `APPROVED FOR IMPLEMENTATION`
**Sprint mode:** Build (D1/D2 accepted; D3 next)
**Repository commit reviewed:** `0dec69ef964d0182a8c570d812ec976e1f61a4ba`
**Working tree at review:** only this plan file untracked/modified (authorized)
**`scripts/backfill_features.py`:** absent (confirmed)
**Canonical D2 input (read-only):**
`C:/MomentumCVG_env/derived/e2c1f8fd44d72176/straddle_observations_weekly.{parquet,lineage.json}`

---

## Owner-facing summary

Plain-language reading aid for the project owner. Technical contract remains in §§1–13.

### Terms

* **D2** — accepted weekly straddle observation table (~1.06M ticker–date rows) from earlier in Sprint 005.
* **D3** — this deliverable: turn D2 into the full Momentum/CVG feature file set.
* **CVG** — continuous volatility-gap signal (second ranking input beside Momentum).
* **D4** — next deliverable: inspect and document those files; does not rebuild them.
* **Lineage** — recorded provenance (which input, snapshot, code version).
* **Receipt** — small completion certificate written after a successful run.
* **Staging directory** — temporary holding folder (`features.building/`) that the backtest loader must not use.
* **Atomic publication** — final `features/` appears via one rename after all checks pass, not file-by-file into the live location.
* **Resume** — “skip windows already written and continue.” This plan does **not** support resume.

### Overview

1. **What goes into D3?** The accepted D2 table and its lineage certificate, plus the frozen feature recipe `configs/feature_backfill_v1.json` (281 lookback windows, minimum counts, required output columns). D3 does not reopen raw option data or rebuild D2.
2. **What will D3 produce?** Exactly 281 files named like `features_42_8.parquet`, each with six columns (ticker, date, momentum mean/count, CVG, CVG count), under `…/derived/e2c1f8fd44d72176/features/`, plus a completion receipt beside that folder.
3. **How will calculation run?** Load D2 once; for each window one at a time, run production Momentum and CVG on the **entire** ticker panel; require both outputs to cover every D2 key; combine them carefully; write only into staging; after all 281 pass checks, rename staging to final `features/`; then write the receipt.
4. **Time / memory / disk?** About **80–90 minutes**, peak memory roughly **0.4–0.5 GB**, finished files about **1.4–1.7 GB** (plus temporary staging space during the run).
5. **What will D4 do afterward?** Inspect that same published set and receipt (coverage, missingness, PIT). It should not recompute the 281 windows merely for proof.

### Focus questions

| Question | Plan answer |
|----------|-------------|
| Can an interrupted run leave partial files where SurfaceRunner might consume them? | **Not in final `features/`**, if implemented as written. Partials go only to `features.building/`. After a successful rename, files in `features/` are fully validated; if only the receipt then fails, the folder is complete but **not** D3-accepted until a receipt exists. |
| Is resume-from-filename/schema/row-count safe? | **No — do not resume at all.** Shallow checks can treat stale/wrong files as valid. |
| Staging then expose `features/` only after all 281 validate? | **Yes.** Core publication design. |
| Could an outer merge hide key problems? | **Yes, in principle** — so the plan forbids it and requires exact key equality then an inner one-to-one combine. |
| Snapshot/build identity still a choice? | **No.** Required CLI expected ids; production command supplies the accepted D1/D2 values. |
| Allow supplied `--repo-sha`? | **No.** Record actual clean Git HEAD; dirty tree fails before writing. |
| Necessary vs overbuilt for a one-time ~90 min run? | **Necessary:** staging/rename, no resume, full-panel CVG, exact keys, digest/row/key checks, refuse existing outputs, Git clean HEAD. **Substitute (path gap):** sibling-folder rule because D2 lineage lacks an observation path. **Soft spot:** “explicitly controlled recovery” after receipt failure is policy, not step-by-step steps. |

### Owner yes/no checklist

1. Approve building **only** the backfill script + focused tests (no feature-store / resume platform)?
2. Approve **no resume** — failed runs restart clean after manually clearing leftover staging?
3. Approve **`features.building/` → validate all 281 → rename to `features/` → receipt last**?
4. Approve **refusing** to run if `features/`, staging, or receipt already exist (no auto-delete)?
5. Approve **strict key equality + inner combine** (fail loudly on mismatches)?
6. Approve **CLI expected snapshot/build ids** + actual **clean Git HEAD** (no typed-in repo SHA)?
7. Accept the **sibling-folder + digest/row/key** substitute because D2 lineage has **no observation path field**?
8. Accept that if rename succeeds but receipt fails, **keep** the complete `features/` folder and **do not** start D4 until a receipt is produced by a controlled recovery?
9. Confirm **one** production backfill, reused as D4’s audit input (D4 does not recompute for proof)?

### Top risks

1. **Consumer safety** — if staging/rename is implemented wrong, SurfaceRunner can load incomplete `features/` because it never checks the receipt.
2. **Identity** — wrong or mutable observation input (mitigated by digest/keys; path-in-lineage gap patched by sibling+digest).
3. **Receipt-after-rename gap** — complete files without a receipt; D3/D4 must not treat that as finished; recovery steps are not spelled out.

### Complexity note

Nothing large remains that should be cut before approval. Resume was already removed. Do **not** add multiprocess, schedulers, or resume managers. Optional clarity later: a short “if receipt fails after rename, do X / don’t do Y” note—not new software.

### Review recommendation

**APPROVED** — the owner approved the recommended D3 design. No owner decision remains before implementation. The soft spot is operational detail for the rare “files published, receipt missing” case; that is a policy gap, not an unresolved design fork, if D4 requires both folder and receipt and validated `features/` is never auto-deleted.

---

## 1. Owner decision brief

**Verdict:** APPROVED FOR IMPLEMENTATION. The owner approved the recommended D3 design. No owner decision remains before implementation.

**D3 purpose.** Ship the smallest reliable standalone path that turns the accepted D2 weekly panel + `configs/feature_backfill_v1.json` into 281 SurfaceRunner-consumable feature files with a minimal completion receipt. Feature economics stay frozen by D1; D3 does not reopen them.

**Frozen shape.**

| Topic | Decision |
|-------|----------|
| Implementation surface | `scripts/backfill_features.py` + `tests/unit/test_backfill_features.py` only |
| Config authority | Explicit fields from `feature_backfill_v1.json`; inline window expansion; no `build_features.py` defaults |
| Execution unit | **One window at a time** after a single D2 load; no resume |
| Staging → final | Write only under `features.building/`; atomically rename to `features/` after full validation |
| Completion marker | Receipt written **after** successful rename: `…/features_backfill_v1.lineage.json` |
| Identity | Required CLI `--expected-snapshot-id` / `--expected-build-id`; Git HEAD + clean tree for `repo_sha` |
| Key digest | Reuse existing `a1_key_digest()` from `src/features/straddle_observations.py`; do not invent a second algorithm |
| Receipt windows | Complete ordered list of all 281 `(max_lag, min_lag)` pairs (config remains the generator authority) |
| Key merge | Exact D2 key equality checks, then **inner** 1:1 merge |
| Production run | Once, at the clean post-implementation SHA; same output is D4’s audit input |

**Owner action required.** None. Proceed to implementation under this approved plan.

---

## 2. Current repository findings

### 2.1 Preflight

| Check | Result |
|-------|--------|
| Expected HEAD | `0dec69ef964d0182a8c570d812ec976e1f61a4ba` |
| Actual HEAD | match |
| Working tree | only this plan file (authorized) |
| `scripts/backfill_features.py` | does not exist |
| Agenda | Sprint 005 ACCEPTED; D1/D2 done; D3 next |
| Spec | `configs/feature_backfill_v1.json` present (`feature_backfill_v1` / `sprint005_d1`) |

### 2.2 D2 artifact (read-only probe)

| Field | Value |
|-------|-------|
| Rows / tickers / dates | 1,063,995 / 2,391 / 445 |
| Parquet size | ~30 MB |
| Lineage `snapshot_id` | `e2c1f8fd44d72176` |
| Lineage `build_id` | `20260724T045049097520Z_40b16886` |
| Lineage `output.file_sha256` | `f0c1461ea4643154d6b26393159d2b9fc78ce2f9cd5dbdde1a0d1e3d700845c9` |
| Features dir under derived | does not exist yet |

### 2.3 Calculator / consumer contracts (inspect-only)

* `MomentumCalculator.calculate_bulk` / `CVGCalculator.calculate_bulk`: sort by `(ticker, entry_date)`; emit `ticker` + `date` (calculators rename `entry_date` → `date` internally) plus diagnostics; `tickers=None` = full panel.
* CVG stage-1/2 medians use the in-memory panel; prefiltering tickers before CVG changes medians (D1 G14). D3 must pass `tickers=None`.
* Prefer bulk path (D1 residual: `calculate()` vs bulk count `0` vs `NaN`).
* Spec publish set is six columns; calculators may emit extras upstream — D3 drops them before write.
* `SurfaceRunner` / `SurfaceDataPaths` load `features_{max}_{min}.parquet` from a configurable `features_dir` and do **not** check a D3 receipt. Incomplete files must therefore never appear under the final `features/` name.
* `scripts/build_features.py` remains reference-only: helper default grid → 272 windows; CLI Momentum/CVG `min_periods` defaults 3/5. Do not call it and do not clean it up in D3.

### 2.4 Planning benchmark (read-only, 1–3 windows, temp file deleted)

Full D2 load of required columns ≈ 0.5 s / ~46 MB.

| Windows | Momentum | CVG | Peak traced (CVG) | Publish size |
|---------|----------|-----|-------------------|--------------|
| `(6,2)` | 1.3 s | 17.0 s | ~412 MB | ~4.7 MB |
| `(6,2),(12,2)` | 2.5 s | 34.3 s | ~650 MB | — |
| `(42,8)` | 1.3 s | 19.2 s | ~412 MB | ~5.9 MB |

All three produced **1,063,995** output rows (complete D2 grid). Extrapolation for 281 serial windows: **~80–90 minutes**, peak memory **~0.4–0.5 GB**, disk **~1.4–1.7 GB**. Batching windows raises peak memory roughly linearly and does not reduce total CVG work. Resume is **not** used: an 80–90 minute one-shot run does not justify stale-output / lineage risk.

---

## 3. Recommended data flow

```text
CLI runtime paths + expected snapshot/build ids
  → require clean Git HEAD; record actual repo_sha
  → verify D2 lineage against exact fields (§4): snapshot_id, build_id,
    output.file_sha256, output.row_count, output.key_count, output.output_key_digest
  → load feature_backfill_v1.json + config sha256
  → expand exactly 281 ordered windows from explicit bounds
  → refuse if features/, features.building/, or receipt already exist
  → create empty features.building/
  → load D2 parquet once (required columns only; no ticker filter)
  → establish canonical key set: unique (ticker, entry_date)
  → for each (max,min) in order:
        MomentumCalculator(windows=[(max,min)], min_periods=spec)
        CVGCalculator(windows=[(max,min)], min_periods=spec)
        calculate_bulk(..., tickers=None) on full panel
        assert mom keys == cvg keys == canonical D2 keys (unique)
        inner 1:1 merge on (ticker, date)
        keep exactly six publish columns; sort deterministically
        write features.building/features_<max>_<min>.parquet
        release intermediates
  → validate entire staging directory (281 files, schemas, rows, keys, digests)
  → atomically rename features.building/ → features/ (same filesystem)
  → atomically write features_backfill_v1.lineage.json via temp file
```

No A1/A2 reopen, no ORATS, no D2 rebuild, no `refresh_weekly_inputs.py`, no mutable `cache/` stand-in as the observation source, no resume.

---

## 4. CLI and runtime-input contract

**Script:** `scripts/backfill_features.py`

```text
python scripts/backfill_features.py \
  --observations <path/to/straddle_observations_weekly.parquet> \
  --d2-lineage <path/to/straddle_observations_weekly.lineage.json> \
  --config configs/feature_backfill_v1.json \
  --output-root C:/MomentumCVG_env/derived/e2c1f8fd44d72176 \
  --expected-snapshot-id e2c1f8fd44d72176 \
  --expected-build-id 20260724T045049097520Z_40b16886
```

| Arg | Role |
|-----|------|
| `--observations` | Required. Path to D2 parquet. |
| `--d2-lineage` | Required. Path to D2 lineage JSON. |
| `--config` | Required. Path to versioned semantics JSON. |
| `--output-root` | Required. Staging/final features and receipt are rooted here. |
| `--expected-snapshot-id` | Required. Must equal D2 lineage `snapshot_id`. |
| `--expected-build-id` | Required. Must equal D2 lineage `build_id`. |

**No `--repo-sha`.** The script obtains the real repository SHA from Git (`git rev-parse HEAD`), requires a **clean working tree** for any run that writes output, and records that SHA in the receipt. If Git identity cannot be established or the tree is dirty, fail **before** creating `features.building/` or writing any output.

### Exact accepted D2 lineage fields (frozen)

Inspected file: `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/straddle_observations_weekly.lineage.json`.

| Check | Exact lineage field |
|-------|---------------------|
| Artifact type | `artifact` (must equal `"straddle_observations_weekly"`) |
| Snapshot ID | `snapshot_id` |
| Build ID | `build_id` |
| Observation file digest | `output.file_sha256` |
| Row count | `output.row_count` |
| Key count | `output.key_count` |
| Output key digest | `output.output_key_digest` |

**Limitation — observation output path is not in the lineage.** The accepted D2 lineage does **not** record an absolute or relative path for `straddle_observations_weekly.parquet` (no `output.path`, `output.absolute_path`, or similar). Therefore D3 cannot compare `--observations` to a lineage-recorded observation path.

**Smallest deterministic path / identity validation instead (fail closed before creating `features.building/`):**

1. Resolve `--observations`, `--d2-lineage`, and `--config` to absolute paths with Windows-appropriate normalization (`Path.resolve()`); refuse if any is missing.
2. Load D2 lineage JSON; require the exact fields in the table above to be present.
3. Assert `snapshot_id == --expected-snapshot-id` and `build_id == --expected-build-id`.
4. Require `--observations` and `--d2-lineage` to share the same resolved parent directory (published D2 layout: parquet and lineage are siblings). Fail if parents differ.
5. **Mutable-cache guard:** refuse if the resolved `--observations` path is under `C:/MomentumCVG_env/cache` (or any path with a `cache` segment that would allow a mutable stand-in).
6. Recompute `sha256_file(--observations)` and require equality with `output.file_sha256`.
7. After loading the parquet: require every `input_schema.required_columns` name; assert `len(df) == output.row_count`; assert unique `(ticker, entry_date)` count equals `output.key_count`; recompute the key digest by calling the existing `a1_key_digest()` in `src/features/straddle_observations.py` (which wraps `ticker_date_keys_digest`) and require equality with lineage `output.output_key_digest`. Do **not** create or copy a second key-digest algorithm.
8. Compute and record `sha256_file(--config)` for the file actually loaded.

No alternate field-name lists. Do not invent a lineage observation-path field. The D3 receipt still records the resolved `--observations` path that was validated and used.

Runtime paths and expected ids stay out of `feature_backfill_v1.json` (semantics-only). Production uses the accepted identity from the D1 memo (`e2c1f8fd44d72176` / `20260724T045049097520Z_40b16886`).

---

## 5. Config / grid handling

Inside the script (and mirrored in tests):

1. `json.load` the feature config.
2. Assert `spec_version == "feature_backfill_v1"`.
3. Read **every** window bound explicitly: `min_lag_start/end`, `max_lag_start/end`, `step`, `require_max_gt_min`, `order`, `expected_count`.
4. Expand with an **inline** generator (do not import `generate_momentum_windows` and do not rely on its defaults):

```python
windows = [
    (max_lag, min_lag)
    for max_lag in range(max_lag_start, max_lag_end + 1, step)
    for min_lag in range(min_lag_start, min_lag_end + 1, step)
    if max_lag > min_lag
]
```

5. Assert `len(windows) == expected_count == 281`, uniqueness, first `(6,2)`, last `(60,24)`, baseline `(42,8)` present and equal to `baseline_window`, and `order == "max_lag_outer_min_lag_inner"`.
6. Read `momentum.min_periods` and `cvg.min_periods` explicitly (both must be `1`); pass them into the calculator constructors. Never omit them.
7. Render publish columns from `output_columns_per_window` templates.

`build_features.py` is reference only — no cleanup, no shared config framework.

---

## 6. Computation and memory strategy

**Unit of work: one `(max_lag, min_lag)` window.** No resume, no skip of existing files, no reuse of staging contents.

**Why not all 281 in one frame.** One window already adds several float columns over 1.06M rows; CVG also builds temporary rolling columns. Holding diagnostics for 281 windows would be multi-GB and is unnecessary because SurfaceRunner loads one file per window.

**Why not multi-window batches.** Benchmark: two windows ≈ 2× CVG time and ~650 MB peak. Wall clock for 281 is essentially the same whether windows are batched or serial; serial keeps peak ~400 MB. No multiprocessing / scheduler.

**Canonical keys and merge (exact contract):**

1. Keep one in-memory D2 frame for the whole run: columns from `input_schema.required_columns` only; **no ticker universe filter**.
2. Build the unique canonical key set `K` from D2 `(ticker, entry_date)` (normalized timestamps). Fail on duplicate D2 keys.
3. `FeatureDataContext(straddle_history=d2_df)`.
4. `start_date = d2.entry_date.min()`, `end_date = d2.entry_date.max()`.
5. `MomentumCalculator(windows=[(max,min)], min_periods=mom_min).calculate_bulk(..., tickers=None)`.
6. `CVGCalculator(windows=[(max,min)], min_periods=cvg_min).calculate_bulk(..., tickers=None)`.
7. Production calculators already emit the date column as `date`. Treat calculator `(ticker, date)` as the feature-date key corresponding to D2 `entry_date`. Assert Momentum keys are unique and **exactly equal** to `K`. Assert CVG keys are unique and **exactly equal** to `K`. Fail immediately on any missing, duplicate, or unexpected key — do not patch with an outer merge.
8. Select only the four signal/count columns from each side; merge with `how="inner"` and `validate="one_to_one"` on `(ticker, date)`.
9. Assert the merged frame’s keys still equal `K` and `len(out) == len(K)`.
10. Keep exactly the six rendered publish columns in spec order (`ticker`, `date`, mom mean/count, cvg, cvg_count). Published column name is `date` (spec); no alternate date name is emitted.
11. Sort deterministically by `ticker`, `date` ascending.
12. Write under `features.building/` only (snappy, `index=False`).
13. `del` momentum/cvg/merged frames; continue.

---

## 7. Output and receipt contract

### 7.1 Paths

```text
C:/MomentumCVG_env/derived/e2c1f8fd44d72176/
  features.building/                 # staging only; never SurfaceRunner input
    features_<max>_<min>.parquet
  features/                          # final; created only by rename after validation
    features_<max>_<min>.parquet
  features_backfill_v1.lineage.json  # written after successful rename
```

Filename rule: `features_{max_lag}_{min_lag}.parquet` with decimal ints (e.g. `features_42_8.parquet`). Compression: snappy. No multi-window files.

### 7.2 Row set

Each file preserves the **complete D2 `(ticker, entry_date)` grid** as published `(ticker, date)` — 1,063,995 rows for the accepted artifact — including rows whose signals are null. No eligibility/liquidity filter. Feature-ready interval filtering is a D4/Sprint 006 reporting concern, not a D3 drop.

### 7.3 Safe publication (no resume)

**Preflight refuse (before any computation write):** if any of these already exists under `--output-root`, fail and write nothing:

* `features/`
* `features.building/`
* `features_backfill_v1.lineage.json`

Do **not** automatically delete or overwrite them. An incomplete prior `features.building/` is not reusable; the operator must remove it explicitly before a clean rerun.

**Publication steps:**

1. Create empty `features.building/`.
2. Compute and write all 281 window files **only** into `features.building/`.
3. Validate the entire staging directory before any rename:
   * Exactly the 281 expected filenames; no missing or unexpected files.
   * Exact six-column schema (names and order) for every file.
   * Expected row count (`len(K)`) for every file.
   * Unique `(ticker, date)` keys equal to canonical `K`.
   * Deterministic row ordering (`ticker`, `date` ascending).
   * Complete ordered window grid identity.
   * Per-file digests / metadata needed for the receipt.
4. Only after validation passes: atomically rename `features.building/` → `features/` on the **same filesystem** (`Path.replace` / `os.rename` of the directory).
5. Write the receipt atomically through a temporary sibling file then `os.replace` onto `features_backfill_v1.lineage.json`.

**Failure-state contract (precise):**

* If failure occurs **before** the validated staging directory is atomically renamed: final `features/` and the final receipt remain absent. `features.building/` may remain and must not be reused automatically. No partial or unvalidated window files may ever appear inside final `features/`.
* If all 281 files have been validated and `features.building/` has already been atomically renamed to `features/`, but atomic receipt creation then fails: the complete final `features/` directory **may remain** while the receipt is absent. Do **not** roll back the rename and do **not** automatically delete the complete directory.
* In that exceptional state, D3 is **not** accepted and D4 must **not** begin until a valid receipt is produced through an **explicitly controlled recovery** (operator-driven; no automatic resume tooling in D3). D4 requires both the completed `features/` directory and the receipt.
* Do not leave SurfaceRunner pointed at `features.building/` (that name must not exist after a successful rename).

### 7.4 Receipt (concise)

`features_backfill_v1.lineage.json` — only fields needed for Sprint 006 provenance:

```text
schema_version                  "1"
artifact                        "features_backfill_v1"
created_at_utc
repo_sha                        actual clean Git HEAD (required)
spec_version / spec_id          from feature config
feature_config_path
feature_config_sha256
snapshot_id / build_id          verified against CLI expected ids (= lineage snapshot_id / build_id)
observations_path               resolved --observations path actually used
d2_lineage_path                 resolved --d2-lineage path
observations_file_sha256        == lineage output.file_sha256
observations_row_count          == lineage output.row_count
observations_key_count          == lineage output.key_count
observations_output_key_digest  == lineage output.output_key_digest
window_count                    281
windows                         complete ordered list of all 281 [max_lag, min_lag] pairs
                                (generated from feature_backfill_v1.json; not a digest substitute)
baseline_window                 {max_lag, min_lag}
momentum_min_periods / cvg_min_periods
output_root / features_dir
files[]                         {filename, max_lag, min_lag, row_count, file_sha256}
status                          "complete"
```

Do not duplicate Sprint 004 audit receipts or D2 coverage tables. Per-file sha256 is justified: D4/D5 can verify bytes without recomputing features.

---

## 8. Exact file-by-file implementation changes

| File | Change | Why necessary |
|------|--------|----------------|
| `scripts/backfill_features.py` | **Add** standalone CLI + helpers (Git/clean-tree check, load/verify D2, expand grid, per-window compute, staging write, staging validation, atomic rename, receipt) | D3 deliverable; keeps I/O out of library modules |
| `tests/unit/test_backfill_features.py` | **Add** focused synthetic/unit coverage from §9 | CI-safe proof without Windows D2 artifact |

**Reuse (import only, no edits):** `MomentumCalculator`, `CVGCalculator`, `FeatureDataContext`, `sha256_file` from `src.data.snapshot_foundation`, and `a1_key_digest` from `src.features.straddle_observations` (no second key-digest implementation).

**Do not add** a publication framework, resume manager, orchestration layer, multiprocessing, config system, or new `src/` production module. Helpers stay as functions in the script (tests import via `importlib`).

**Explicitly unchanged:** `configs/feature_backfill_v1.json`, `scripts/build_features.py`, calculators, `straddle_observations.py`, SurfaceRunner / `SurfaceDataPaths`, D2 artifacts, Sprint 004, `refresh_weekly_inputs.py`.

---

## 9. Literal test matrix

Module: `tests/unit/test_backfill_features.py` (synthetic / temp dirs only).

| ID | Assert |
|----|--------|
| T1 | Config expands to exactly 281 unique ordered windows including `(42,8)`; matches oracle loop |
| T2 | Expansion uses only explicit spec bounds; script always passes both spec `min_periods` into calculators |
| T3 | Script path never imports `build_features.py` / never inherits its defaults |
| T4 | Small complete weekly panel through per-window path yields full canonical-grid rows |
| T5 | CVG/Momentum called with `tickers=None` (full cross-section) |
| T6 | Staging filenames `features_{max}_{min}.parquet` and exact six-column schema/order |
| T7 | Independent Momentum vs CVG counts when returns and vol gaps miss differently |
| T8 | Unique `(ticker, date)`; deterministic sort; keys equal canonical D2 grid |
| T9 | Exact Momentum key set == CVG key set == D2 `(ticker, entry_date)` set |
| T10 | Missing, duplicate, or unexpected calculator key → fail before write; no outer-merge concealment |
| T11 | Inner `one_to_one` merge used after key equality checks |
| T12 | Wrong `--expected-snapshot-id` / `--expected-build-id`, mismatched `output.file_sha256`, mismatched `output.row_count` / `output.key_count` / `output.output_key_digest`, non-sibling observations/lineage paths, or cache-path observations → fail before creating `features.building/` |
| T13 | Existing `features/`, `features.building/`, or receipt → refuse; no reuse of staging |
| T14 | Failure before rename leaves `features/` and receipt absent; partial output only under `features.building/` (never under final `features/`) |
| T15 | Final `features/` appears only after all staged files validate and rename succeeds; receipt is written only after that rename |
| T15b | If rename succeeded and receipt write fails: `features/` may remain complete while receipt is absent; D3 unaccepted; no automatic rollback/delete |
| T16 | Receipt records actual clean Git HEAD; dirty tree / missing Git identity → fail before write |
| T17 | No resume behavior (no skip flags, no “continue from existing file” path) |
| T18 | `SurfaceDataPaths(features_dir=...).features_path_for_config` resolves the emitted baseline filename |

Keep helpers testable without always invoking full CLI; include at least one end-to-end CLI path on `tmp_path` covering staging → rename → receipt.

### Commands each implementation commit must run

```powershell
& C:/MomentumCVG_env/venv/Scripts/Activate.ps1
python -m pytest tests/unit/test_backfill_features.py -q
python -m pytest tests/unit/test_feature_backfill_v1_contract.py -q
python -m pytest -q
```

Focused pair first; full suite before each commit is considered green.

---

## 10. Dependency-aware commit sequence

Every commit leaves focused + full suite green. No production backfill inside commits. No resume code in any commit.

1. **Script skeleton + grid/identity/Git helpers + T1–T3, T12, T16**
   Load config, expand 281 windows, verify exact D2 lineage fields (`snapshot_id`, `build_id`, `output.file_sha256`, `output.row_count`, `output.key_count`, `output.output_key_digest`), sibling-path + cache refuse, require clean Git HEAD; no calculator loop yet.

2. **Per-window compute + exact key checks + inner merge + staging writes + T4–T11, T18**
   One-window Momentum/CVG on synthetic panels; key equality failures; six-column staging files; SurfaceRunner naming.

3. **Staging validation + atomic rename + receipt-last + refuse-existing + T13–T15b, T17 + CLI `main`**
   Full staging validation; rename to `features/`; atomic receipt; refuse pre-existing `features/` / `features.building/` / receipt; prove no resume path; cover pre-rename vs post-rename receipt-failure states.

---

## 11. Production execution and D4 handoff

**Where D3 implementation commits stop.** After commit 3 is on the agreed branch with a clean tree and green suite. That clean HEAD is what the production run records.

**Production run (once):**

```powershell
& C:/MomentumCVG_env/venv/Scripts/Activate.ps1
# working tree must be clean
python scripts/backfill_features.py `
  --observations C:/MomentumCVG_env/derived/e2c1f8fd44d72176/straddle_observations_weekly.parquet `
  --d2-lineage C:/MomentumCVG_env/derived/e2c1f8fd44d72176/straddle_observations_weekly.lineage.json `
  --config configs/feature_backfill_v1.json `
  --output-root C:/MomentumCVG_env/derived/e2c1f8fd44d72176 `
  --expected-snapshot-id e2c1f8fd44d72176 `
  --expected-build-id 20260724T045049097520Z_40b16886
```

Expected wall time ~80–90 minutes; do not run during this planning task.

**On failure:**

* Before rename: remove leftover `features.building/` manually before retry; `features/` and receipt remain absent.
* After rename but before/without valid receipt: do **not** auto-delete `features/`; D3 stays unaccepted until an explicitly controlled recovery produces a valid receipt. D4 does not begin without both.

**D3 vs D4 boundary (explicit):**

| | D3 | D4 |
|---|----|----|
| Code | Script + unit tests | No recompute script required |
| Production 281 emit | **Runs once** at clean SHA | Consumes that emit |
| Acceptance bar | `features/` present via validated rename; receipt `status=complete` with recorded Git SHA | Coverage / missingness / PIT evidence on **that** output; requires **both** `features/` and receipt |
| Forbidden | Re-running 281 solely to “prove” D4 | Recomputing features merely for audit proof |

D5 (SurfaceRunner consumer smoke) remains separate and is out of D3.

---

## 12. Objective D3 acceptance criteria

1. `scripts/backfill_features.py` exists and is the only production entrypoint for the 281 emit.
2. Runtime inputs (paths, expected ids, digests, Git SHA) are CLI/receipt concerns; semantics come only from `feature_backfill_v1.json`.
3. Grid expands to exactly 281 ordered windows; baseline `42:8`; both `min_periods` read from spec.
4. CVG/Momentum use production `calculate_bulk` on the complete D2 panel (`tickers=None`).
5. Momentum and CVG keys each equal the canonical D2 key set; merge is inner `one_to_one`; each final file has exactly six publish columns and the complete D2 row count.
6. No partial or unvalidated window files ever appear under final `features/`; staging is `features.building/` only; rename happens only after full validation.
7. Receipt written after successful rename, recording clean Git HEAD, expected snapshot/build ids, exact D2 fields validated (`output.file_sha256`, `output.row_count`, `output.key_count`, `output.output_key_digest`), config digest, window identity, per-file digests, `status=complete`.
8. Failure-state contract holds:
   * Failure before rename → `features/` and receipt absent; leftover `features.building/` is not auto-reused.
   * Failure after validated rename but during receipt write → complete `features/` may remain without a receipt; D3 unaccepted; no automatic rollback or deletion; D4 waits for an explicitly controlled receipt recovery.
9. Pre-existing `features/`, `features.building/`, or receipt paths are refused (no auto-delete, no resume).
10. Unit tests in §9 pass without the Windows production artifact.
11. Focused + full pytest suite green after implementation commits.
12. One successful production run at the clean SHA; that output is frozen for D4 — not recomputed for proof.
13. No changes to excluded files/artifacts listed in the task.

---

## 13. Risks, non-goals, and genuine owner decisions

### Non-goals

Feature-definition changes; eligibility/liquidity filters; ranking/backtests; incremental weekly refresh; resume/checkpointing; feature stores/orchestration; raw ORATS; new coverage thresholds; D1 re-audit; D4 coverage memo; D5 smoke; `build_features.py` cleanup; calculator refactors; multiprocessing.

### Residual risks (non-blocking)

* CVG `rolling.apply` dominates runtime (~17–20 s/window); total ~1.5 h is acceptable for a one-shot emit with no resume.
* Failed runs before rename require manual removal of `features.building/` before retry (intentional: avoids silent reuse of stale staging).
* If receipt write fails after rename, complete `features/` may remain without a receipt; D3 stays unaccepted and D4 waits for explicitly controlled receipt recovery — no automatic rollback or deletion of the validated directory.
* Accepted D2 lineage does not record the observation parquet path; D3 substitutes sibling-directory + `output.file_sha256` / row / key / `output.output_key_digest` checks (§4).
* `calculate()` vs bulk count inconsistency remains; D3 uses bulk only.
* Machine must have ~2 GB free disk under `MomentumCVG_env/derived` (staging peak before rename).

### Genuine owner decisions

None remaining. The owner approved the recommended D3 design, including publication, no-resume, exact-key/inner-merge, CLI-identity/Git-SHA, reuse of `a1_key_digest()`, and the full ordered 281-window receipt list. Proceed to implementation; do not reopen menus.

---

**End of plan.** Status is `APPROVED FOR IMPLEMENTATION`.
