# C2 Design Plan — CLI skeleton, `--as-of` resolution, exit-code contract

**Commit:** C2 (implemented 2026-06-21)  
**Status:** Implemented v3 — HD decisions locked (2026-06-21)  
**Prerequisite:** C1 accepted (`src/data/input_snapshot.py`, `tests/unit/test_input_snapshot.py`)  
**Principle:** C1 receipt model stays locked — manifest is a weekly input receipt, not pipeline telemetry. C2 does not touch C1.

**Post-C5 (2026-07-04):** Default `--orats-adj-root` → `input/adjusted_liquid`. `split-audit` stub now references `audit_adjusted_liquid.py` until **C8** CLI wiring (C5 audit script exists).

> Can we define a terminal CLI skeleton with stable subcommands, `--as-of` resolution, dry-run plan behavior, and exit-code contract?

**Does NOT answer:**

- What is the final Stage A storage architecture?
- Should adjusted ORATS store be full universe or master-universe-only?
- How exactly should split adjustment, spot extraction, liquidity rebuild, or surface precompute be scoped?
- Can the weekly refresh run end-to-end?
- Can `validate` / `split-audit` / `surface-audit` pass real data?
- Is the strategy profitable?

**Sprint references:** [current_sprint.md](../agenda/current_sprint.md) · [c1_manifest_design_plan.md](c1_manifest_design_plan.md)

---

## C2 design summary

C2 is a **boring CLI skeleton**: argparse entrypoint, `--as-of` resolver in a small shared module, a short provisional `plan`, `refresh --dry-run` reusing plan output, and **blocked stubs** that return exit **2** (not 0). No subprocesses, no manifest writes, no real-cache reads in tests, no changes to C1 or Stage A scripts.

| In C2 | Out of C2 |
|-------|-----------|
| `scripts/refresh_weekly_inputs.py` | Stage A storage architecture |
| `src/data/trading_day.py` (2 functions) | Universe-scoped adjust/spot |
| `plan` + `refresh --dry-run` | Liquidity rebuild semantics |
| `--as-of` resolution | Subprocess wiring (C8) |
| Exit-code contract | Real validate/audit (C3/C5/C6) |
| Synthetic unit tests | `DEFAULT_ARTIFACT_REL_PATHS` in C1 |

---

## What was narrowed vs previous draft (v2)

| Previous draft (v2) | This revision (v3) |
|---------------------|-------------------|
| Target Stage A operator model as implementation contract | Moved to **non-binding** future notes (C3–C8) |
| Master-universe-only adjusted store (HD-C2-11) | **Removed** from C2 decisions |
| Detailed step annotations (incremental/repair semantics) | **Short provisional** step list only |
| `validate` / audit stubs return **0** | Stubs return **2** — not mistaken for pass |
| Recommend `DEFAULT_ARTIFACT_REL_PATHS` in C1 | **No C1 changes**; artifact defaults local to CLI script |
| `--mode backfill` exit 2 without date window | **Warn only** in plan; no deep mode enforcement |
| `build_ticker_universe` in plan step 0 | **Removed** from C2 plan output |
| Many HD open decisions | **Five** C2-relevant decisions only |

---

## Proposed files

| File | Purpose |
|------|---------|
| `src/data/trading_day.py` | `orats_daily_parquet_path`, `resolve_as_of_trading_day` only |
| `scripts/refresh_weekly_inputs.py` | argparse CLI, handlers, provisional plan renderer |
| `tests/unit/test_trading_day.py` | Resolver tests with synthetic `exists_fn` |
| `tests/unit/test_refresh_weekly_inputs_cli.py` | `main(argv)` exit codes, plan output |

**Not in C2:** changes to `src/data/input_snapshot.py`, Stage A scripts, sprint memo, runbook.

**Tiny sprint doc wording (propose only):** Line 5 of `current_sprint.md` says `C2+ in progress`; suggest `C1 implemented; C2 next (CLI skeleton)`.

---

## Entrypoint structure

| Choice | Decision |
|--------|----------|
| Parser | stdlib `argparse` with `subparsers` |
| Test hook | `main(argv: list[str] \| None = None) -> int` |
| Handlers | Separate functions: `cmd_plan`, `cmd_validate`, `cmd_split_audit`, `cmd_surface_audit`, `cmd_refresh` |
| Shared setup | `build_cli_context(args) -> CliContext` — resolved day, paths, mode, command string |
| Output | plan → stdout; errors → stderr |
| Logging | Minimal — no log files in C2 |

```text
scripts/refresh_weekly_inputs.py
  parse_args(argv) -> Namespace
  resolve_context(args) -> CliContext
  main(argv) -> int
if __name__ == "__main__":
    raise SystemExit(main())
```

Follows existing `scripts/` pattern (`sys.path.insert` for project root) but adds testable return codes.

---

## CLI command contract

### Global arguments (all subcommands)

| Flag | Required | Default | Notes |
|------|----------|---------|-------|
| `--as-of` | **Yes** | — | `YYYY-MM-DD` |
| `--cache-dir` | No | `C:/MomentumCVG_env/cache` | Match `build_liquidity_panel.py` |
| `--orats-adj-root` | No | `C:/MomentumCVG_env/input/adjusted_liquid` (C5.11A; was `ORATS_Adjusted` at C2) | Used for `--as-of` resolution |

No env-var overrides in C2. Tests use `tmp_path` and flags.

### Optional flags (parse only; minimal enforcement in C2)

| Flag | Subcommands | C2 behavior |
|------|-------------|-------------|
| `--mode incremental\|backfill\|repair` | `plan`, `refresh` | Display in plan; default `incremental` |
| `--dry-run` | `refresh` | Required for non-blocked refresh path |
| `--start-date`, `--end-date` | `plan`, `refresh` | Parsed; **warn** in plan if `backfill` and missing — no exit 2 |
| `--sample-tickers` | `plan`, `refresh` | Parsed; **warn** in plan if `repair` and missing |
| `--skip-surface`, `--skip-splits` | `plan`, `refresh` | Parsed; mention in plan if set |
| `--strict` | `validate` | Parsed; stub ignores |

**Mode semantics are display-only in C2.** Backfill/repair enforcement deferred to C8 when real refresh exists.

### Subcommand behavior table

| Subcommand | C2 behavior | Exit |
|------------|-------------|------|
| **`plan`** | Resolve `--as-of`; print short provisional plan | **0** |
| **`refresh --dry-run`** | Same as `plan` + `DRY-RUN: no subprocesses executed` | **0** |
| **`validate`** | stderr: `validate not implemented until C3` | **2** |
| **`split-audit`** | stderr: points to `audit_adjusted_liquid.py`; CLI wiring **C8** | **2** |
| **`surface-audit`** | stderr: `surface-audit not implemented until C6` | **2** |
| **`refresh`** (no `--dry-run`) | stderr: `refresh execution not implemented until C8` | **2** |
| Missing / invalid `--as-of` | Clear message on stderr | **2** |
| Bad paths / unresolvable trading day | Clear message on stderr | **2** |
| Future real **`validate` FAIL** | — | **1** (C3+) |

**Why stubs return 2:** exit 0 on `validate` / audits would be misread as a successful validation run.

### Exit-code contract

| Code | Meaning |
|------|---------|
| **0** | Success — `plan` or `refresh --dry-run` only |
| **1** | Blocking validation failure (C3+; unused in C2) |
| **2** | Usage/config error **or** not-implemented stub **or** blocked refresh |

---

## `plan` subcommand output

Short, clearly **provisional**. No fully-approved architecture tone.

```text
=== Weekly input refresh plan (no execution) ===
as_of_requested:             2026-06-28
as_of_resolved_trading_day:  2026-06-26
mode:                        incremental
cache_dir:                   ...
orats-adj-root:              ...
data_source:                 orats_adjusted_cache

Provisional high-level Stage A plan:
  0. resolve_candidate_universe_scope
       [read existing liquidity panel / liquid_tickers / master list; no rebuild in C2]
  1. fetch_splits
  2. apply_split_adjustment
  3. build_liquidity_panel
       [rebuild/refine rolling panel; implementation deferred]
  4. extract_spot_prices
  5. precompute_option_surface

Note: candidate universe scope should be established early so later steps
      can become universe-aware. Exact storage scope, master-universe filtering,
      rolling liquidity rebuild behavior, and subprocess wiring are deferred to C3–C8.

Feature branch (straddle history, build_features, A4): deferred to Sprint 005

Logical receipt artifacts (cache-relative):
  splits:                 splits_hist.parquet
  spot_prices:            spot_prices_adjusted.parquet
  liquidity_panel:        ticker_liquidity_panel.parquet
  option_surface_meta:    option_surface_meta_weekly_2018_2026.parquet
  option_surface_quotes:  option_surface_quotes_weekly_2018_2026.parquet

snapshot_id (preview, display-only):  a1b2c3d4e5f67890

execution: none
```

- **No files written.**
- **Step 0 in C2 is display-only** — plan lists `resolve_candidate_universe_scope` but does not read `liquid_tickers.csv`, liquidity panel, or cache artifacts (no real-cache dependency; implementation deferred to C3–C8).
- Artifact path defaults live in **`scripts/refresh_weekly_inputs.py`** (import C1 `ARTIFACT_*` key constants if useful; do not add fields to C1 module).
- **`snapshot_id` preview:** display-only via `compute_snapshot_id()` (HD-C2-4 locked: yes).
- **`--mode backfill`** without dates: print mode + optional one-line WARN in plan; exit **0**.
- **`--mode repair`** without `--sample-tickers`: print mode + optional WARN; exit **0**.

### `refresh --dry-run`

Reuse `cmd_plan()` internally. Add dry-run banner. No subprocess, no manifest, no `generate_build_id` persistence.

---

## `--as-of` resolver design

**Module:** `src/data/trading_day.py` — **two functions only:**

```python
def orats_daily_parquet_path(orats_adj_root: Path, day: date) -> Path:
    # {root}/{YYYY}/ORATS_SMV_Strikes_{YYYYMMDD}.parquet

def resolve_as_of_trading_day(
    as_of: date | str,
    orats_adj_root: Path | str,
    *,
    max_lookback_days: int = 10,
    exists_fn: Callable[[Path], bool] | None = None,
) -> date:
    ...
```

**Semantics (HD-004-2):** last calendar date `t ≤ as_of` where adjusted daily parquet exists under `--orats-adj-root`.

| Case | Behavior |
|------|----------|
| Trading day with file | `t = D` |
| Weekend / holiday | Walk back day-by-day |
| Invalid ISO date | Raise → CLI exit **2** |
| No file within lookback | Exit **2** |
| Bad/missing `orats-adj-root` | Exit **2** |

- **`exists_fn`:** defaults to `Path.is_file`; tests inject synthetic map.
- **`max_lookback_days`:** default **10**.
- **Do not** import from `scripts/precompute_option_surface.py`.
- **Do not** refactor `get_trading_fridays`, add exchange calendar, or add caching in C2.
- Holidays handled via file existence (same as precompute scripts).

Distinction from `get_trading_fridays` (informational only — not C2 scope): weekly `entry_date` schedule vs operator `--as-of` walk-back. C6 may cross-check later.

---

## C1 manifest integration (read-only)

| Action | C2 |
|--------|-----|
| Import `ARTIFACT_*`, `DEFAULT_DATA_SOURCE`, `compute_snapshot_id` | Optional, from existing C1 API |
| Artifact default **paths** as strings | Local constants in CLI script |
| `write_manifest()` | **No** |
| Change `input_snapshot.py` | **No** |

---

## Future planning notes for C3–C8, not C2 implementation contract

The following are **discussion context only**. C2 does not implement or lock any of this.

C2 may display a provisional high-level Stage A step order, but it must not enforce master-universe storage, adjusted-root v2, universe-scoped split adjustment, spot extraction scope, liquidity rebuild semantics, or surface refresh behavior. Those decisions are validated and implemented in C3–C8.

Topics deferred to later commits (non-exhaustive):

- `resolve_candidate_universe_scope` — read panel / `liquid_tickers` / master list; make downstream steps universe-aware (C3–C8)
- Master-universe vs full-universe adjusted store
- Universe-scoped `SplitAdjuster` / `extract_spot_prices`
- Rolling 3-month liquidity panel rebuild (C4)
- `validate` inventory and PASS/WARN/FAIL (C3)
- Split golden tests + `split-audit` (C5)
- Surface tests + `surface-audit` (C6)
- PIT universe harness (C7)
- Subprocess wiring + manifest write on `refresh` (C8)
- `--mode` enforcement for backfill/repair
- Refactoring `get_trading_fridays` into shared code

---

## What is explicitly provisional (C2)

- Stage A step list including step 0 (`resolve_candidate_universe_scope`) — names and bracket notes only
- Step order in plan output (universe scope before splits/adjust)
- Artifact path strings in CLI (until C3+ aligns with real cache layout)
- `--mode` display and optional WARNs
- `snapshot_id` preview in plan (display-only; HD-C2-4)

---

## What is explicitly deferred

| Item | Commit |
|------|--------|
| `resolve_candidate_universe_scope` (read panel / master list) | C3–C8 |
| `validate` implementation | C3 |
| Rolling liquidity panel script work | C4 |
| `split-audit` | C5 |
| `surface-audit` | C6 |
| PIT universe in validate | C7 |
| `refresh` execution + manifest | C8 |
| Stage A architecture decisions | C3–C8 |
| Feature branch / A4 | Sprint 005 |

---

## Test plan

**Files:** `tests/unit/test_trading_day.py`, `tests/unit/test_refresh_weekly_inputs_cli.py`

**Constraints:** synthetic only; no real ORATS; no subprocess; inject `exists_fn`; use `tmp_path` for paths.

### Resolver (`test_trading_day.py`)

| # | Test |
|---|------|
| R1 | Friday with file → same date |
| R2 | Saturday → previous Friday (mock) |
| R3 | Holiday → walk back to prior trading day |
| R4 | Invalid date → raises |
| R5 | No file in lookback → raises |
| R6 | `orats_daily_parquet_path` pattern |

### CLI (`test_refresh_weekly_inputs_cli.py`)

| # | Test |
|---|------|
| C1 | `plan` → **0** |
| C2 | `refresh --dry-run` → **0** |
| C3 | `validate` → **2**, message mentions C3 |
| C4 | `split-audit` → **2**, message mentions C5 |
| C5 | `surface-audit` → **2**, message mentions C6 |
| C6 | `refresh` without `--dry-run` → **2**, message mentions C8 |
| C7 | Plan includes `as_of_resolved_trading_day` |
| C8 | Plan labels Stage A order as **provisional** |
| C9 | Plan includes step 0 `resolve_candidate_universe_scope` and candidate-universe note |
| C10 | Plan says feature branch / A4 deferred to Sprint 005 |
| C11 | Plan includes `snapshot_id` preview line (display-only) |
| C12 | Invalid `--as-of` → **2** |
| C13 | Missing `--as-of` → **2** |
| C14 | `refresh --dry-run` does not invoke subprocess |
| C15 | `--help` parses without error |

Run after implementation:

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_trading_day.py tests/unit/test_refresh_weekly_inputs_cli.py -q
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/ -q
```

---

## HD decisions (locked — 2026-06-21)

| ID | Topic | Decision |
|----|-------|----------|
| **HD-C2-1** | Resolver location | **`src/data/trading_day.py`** (not inline in CLI script) |
| **HD-C2-2** | `max_lookback_days` default | **10** calendar days |
| **HD-C2-3** | Stub subcommands exit code | **2** (`validate`, `split-audit`, `surface-audit`, non-dry-run `refresh`) |
| **HD-C2-4** | `snapshot_id` preview in `plan` | **Yes** — display-only; no manifest write |
| **HD-C2-5** | `--mode` on `plan` | **Yes** — display-only; no deep enforcement in C2 |

---

## Risks (C2 scope only)

| Risk | Mitigation |
|------|------------|
| Provisional plan mistaken for approved architecture | Explicit "provisional" label + deferred note in output |
| `plan` fails on machines without ORATS | Exit 2 with clear message; tests fully mocked |
| Mid-week `--as-of` vs weekly `entry_date` | Document in resolver docstring; C6 cross-check later |

---

## Implementation plan (after HD approval)

1. Add `src/data/trading_day.py` (two functions).
2. Add `scripts/refresh_weekly_inputs.py`.
3. Add `tests/unit/test_trading_day.py`.
4. Add `tests/unit/test_refresh_weekly_inputs_cli.py`.
5. Run targeted pytest, then full suite.
6. Manual smoke: `plan --as-of 2026-06-26` (machine with ORATS optional).

**Estimated size:** ~200–300 lines production, ~150–200 lines tests.

---

## C2 explicit out of scope

- C1 module changes
- Stage A script changes
- Master-universe / storage architecture
- Real validation or audits
- Manifest write
- Subprocess execution
- Real-cache unit tests
- Strategy / S5 / S8 / ORCH / A4
- Runbook / sprint memo

---

## HD audit notes

| ID | Decision | Notes |
|----|----------|-------|
| HD-C2-1 | **Accepted** | Shared module for resolver tests |
| HD-C2-2 | **Accepted** | 10-day lookback for `--as-of` walk-back |
| HD-C2-3 | **Accepted** | Stubs must not return 0 |
| HD-C2-4 | **Accepted** | Show computed `snapshot_id` in plan output only |
| HD-C2-5 | **Accepted** | Parse and print `--mode`; WARN-only for backfill/repair gaps |

**Approved for implementation:** [x] Yes  [ ] No — revise and re-submit

**Approver / date:** HD / 2026-06-21
