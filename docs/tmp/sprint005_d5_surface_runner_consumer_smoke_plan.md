# Sprint 005 D5 — SurfaceRunner consumer smoke plan

**Status:** PLAN — awaiting implementation under this document  
**Mode:** Build (planning only in this commit)  

| Provenance | SHA |
|------------|-----|
| Planning baseline / Git parent | `3c59f05ed971b0d56afd39937113a4f55e0880a1` |
| D4 implementation | `22a8375d2d6c3b2dbd661697d9524548ea6def9a` |
| D4 evidence | `3c59f05ed971b0d56afd39937113a4f55e0880a1` |

---

## 1. Objective and acceptance question

**Objective.** Prove that the canonical `SurfaceRunner` can load the accepted Sprint 004 surfaces + PIT liquidity together with the accepted D3 `(42,8)` feature file and score at least one real trade date.

**Acceptance question.**

> Can the existing `SurfaceRunner` consume the accepted snapshot surfaces and PIT liquidity together with the accepted D3 `(42,8)` feature file, and successfully score at least one real trade date?

This is an interface / consumability smoke — not an economic backtest.

---

## 2. Accepted upstream identities

| Identity | Value |
|----------|-------|
| Planning baseline / Git parent | `3c59f05ed971b0d56afd39937113a4f55e0880a1` |
| D4 evidence | `3c59f05ed971b0d56afd39937113a4f55e0880a1` |
| D4 implementation | `22a8375d2d6c3b2dbd661697d9524548ea6def9a` |
| D3 evidence commit | `816e28f7b63cb9668de94f9cee037d76758fff71` |
| D3 producer SHA (receipt `repo_sha`) | `131d0ac05e1e57749d3095923927a394fdcbc25b` |
| Snapshot ID | `e2c1f8fd44d72176` |
| Build ID | `20260724T045049097520Z_40b16886` |
| Snapshot root | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886` |
| Manifest | `…/manifests/input_snapshot_e2c1f8fd44d72176.json` |
| D3 feature root | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features/` |
| D3 receipt | `…/features_backfill_v1.lineage.json` (`status=complete`) |
| Baseline feature file | `…/features/features_42_8.parquet` |
| D4 audit JSON | `…/features_quality_audit_v1.json` |
| D4 ready interval | `2018-10-26` → `2026-07-10` |
| D4 joint coverage / PIT | ~68.38% / 0 violations |

Manifest-resolved Stage A paths (relative to snapshot root; confirmed present):

| Artifact key | Relative path |
|--------------|---------------|
| `option_surface_meta` (A1) | `cache/surface/option_surface_meta_weekly_2018_2026.parquet` |
| `option_surface_quotes` (A2) | `cache/surface/option_surface_quotes_weekly_2018_2026.parquet` |
| `liquidity_panel` (A3) | `input/liquidity/ticker_liquidity_panel.parquet` |

---

## 3. Current consumer contract and proven gap

**Contract already in place (no production-code change required):**

* `SurfaceDataPaths` accepts explicit `features_dir`, `liquidity_panel_path`, `surface_meta_path`, `surface_quotes_path` and must not rely on mutable `C:/MomentumCVG_env/cache` defaults for D5.
* `features_path_for_config()` infers `features_{max}_{min}.parquet` from `momentum_col` / `cvg_col` / `count_col`.
* With smoke columns `mom_42_8_mean`, `cvg_42_8`, `mom_42_8_count`, resolution is exactly  
  `…/features/features_42_8.parquet`.
* Accepted `(42,8)` schema is the D3 six-column contract:  
  `ticker`, `date`, `mom_42_8_mean`, `mom_42_8_count`, `cvg_42_8`, `cvg_count_42_8`.
* `SurfaceRunner.__init__` loads A1/A2/liquidity; `run_single_config()` loads features, loops dates, calls step1 → step2 → structures → sizing.

**Proven gap (documentation / driver only, not a consumer defect):**

* `run_single_config()` does not return intermediate universe/signal row counts. The smoke driver must call `step1_get_universe` / `step2_score_signals` (or the runner’s thin `_step1_universe` / `_step2_signals` wrappers) on the selected date to record the scoring checkpoint, then call `run_single_config()` for completion.
* `scripts/run_surface_search.py` is **not** the D5 entrypoint (known unsupported `contract_multiplier=` into `SurfaceDataPaths`; grid-search CLI is out of scope). Do not fix it in D5.

**Compatibility defect authorization:** none. Existing `SurfaceRunner` / `SurfaceDataPaths` interfaces already consume the accepted artifacts when paths are passed explicitly.

---

## 4. Exact single-date smoke design

### Frozen smoke scope

| Item | Value |
|------|-------|
| Feature file | `features_42_8.parquet` only |
| Columns | `mom_42_8_mean`, `cvg_42_8`, `mom_42_8_count` |
| Date count | exactly one real trade date |
| Earnings | `earnings_path=None` (optional; unused) |

### Deterministic date selection rule (frozen)

1. Load unique `date` values from `features_42_8.parquet`.
2. Keep dates in the closed D4 ready interval `[2018-10-26, 2026-07-10]`.
3. Keep a date only if ≥1 ticker has both `mom_42_8_mean` and `cvg_42_8` non-null.
4. Sort remaining dates ascending; select index `len // 2` (median).

**Planning application of this rule:** selected trade date = **`2022-09-02`**.  
Bounded read-only checks: 403 ready dates; median date has 1899 both-non-null feature rows; PIT liquidity has prior snapshots (`max month_date < date` = `2022-08-26`).

Do **not** re-pick the date using PnL, structure fill rates, or favorable performance. If the frozen date fails a smoke gate, the smoke **FAIL**s.

### Interval pin for exactly one date

`BacktestRunConfig` requires `start_date < end_date`. Set:

* `start_date = 2022-09-02`
* `end_date = 2022-09-03`

Feature dates are weekly, so `_get_trade_dates` yields exactly `{2022-09-02}`.

### Smoke-only `BacktestRunConfig` (connectivity controls, not strategy pins)

Label: `run_id="sprint005_d5_surface_runner_smoke"`.

| Field | Smoke value | Why |
|-------|-------------|-----|
| `momentum_col` / `cvg_col` / `count_col` | `mom_42_8_mean` / `cvg_42_8` / `mom_42_8_count` | Frozen `(42,8)` |
| `min_count_pct` | `0.01` | Permissive quality gate |
| `long_top_pct` / `short_bottom_pct` | `0.25` / `0.25` | Nonempty pools without overlap |
| `cvg_filter_pct` | `1.0` | No CVG cull |
| `dvol_top_pct` / `spread_bottom_pct` | `1.0` / `1.0` | Full PIT panel membership |
| `short_structure` | `ironfly` | Valid v1 defined-risk short |
| `wing_selection_rule` / `wing_delta_target` | `closest_delta` / `0.25` | Minimal valid iron-fly pair |
| `max_names_per_side` | `10` | Valid bound |
| `max_loss_budget_per_trade` | `500.0` | Satisfies `> 0` |
| `earnings_exclusion_days` | `0` | No earnings artifact |
| `cost_model` / `fill` | `mid` / `FillAssumption.mid()` | Simplest valid fill |
| `include_diagnostics` | `True` | Keep structure diagnostics available |
| `sizing_mode` | `conceptual` | Smallest valid Tier A path |
| `tier_a_mode` | `equal_premium` | Matches Tier A requirement |
| `tier_a_short_budget` / `tier_a_long_budget` | `10000.0` / `10000.0` | Required `> 0` |
| `start_date` / `end_date` | `2022-09-02` / `2022-09-03` | One feature date |

Thresholds are **smoke connectivity controls only** — not accepted Sprint 006 strategy parameters.

### Path wiring (mandatory)

```text
SurfaceDataPaths(
  cache_dir=<unused dummy Path; must not be C:/MomentumCVG_env/cache as the source of truth>,
  features_dir=C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features,
  liquidity_panel_path=<snapshot_root>/<manifest.artifacts.liquidity_panel>,
  surface_meta_path=<snapshot_root>/<manifest.artifacts.option_surface_meta>,
  surface_quotes_path=<snapshot_root>/<manifest.artifacts.option_surface_quotes>,
  earnings_path=None,
)
```

Resolve A1/A2/liquidity from the accepted manifest under the snapshot root (reuse `resolve_surface_inputs` for A1/A2 and the same manifest artifact map for `liquidity_panel`). Do not copy artifacts into `cache/`.

### Startup identity gate (before `SurfaceRunner`)

Before constructing `SurfaceRunner` or reading production data through it, the smoke must perform these scalar preflight checks (actual PASS/FAIL conditions, not memo-only provenance):

1. Read the accepted Sprint 004 manifest and require:
   * `snapshot_id = e2c1f8fd44d72176`
   * `build_id = 20260724T045049097520Z_40b16886`
2. Read the accepted D3 receipt (`features_backfill_v1.lineage.json`) and require:
   * `snapshot_id = e2c1f8fd44d72176`
   * `build_id = 20260724T045049097520Z_40b16886`
   * `repo_sha = 131d0ac05e1e57749d3095923927a394fdcbc25b`
3. Resolve the receipt’s recorded `features_dir` and compare it to
   `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features/`
   using normalized/resolved paths appropriate for Windows (`Path.resolve()` / equivalent). Mismatch → FAIL.

Any identity or path mismatch **stops the smoke before `SurfaceRunner` construction**.

Keep the gate lightweight: do not re-hash feature files, repeat D3 publication validation, rerun D4, or add a validation framework / new helper abstraction.

### Required checkpoints

1. Startup identity gate passes (manifest + D3 receipt scalars + resolved `features_dir`).
2. Runner initializes against manifest-resolved surfaces + liquidity.
3. `features_path_for_config(config)` equals the accepted `features_42_8.parquet` path.
4. Feature load succeeds; expected columns present.
5. Feature rows for `2022-09-02` > 0.
6. PIT universe rows for that date > 0.
7. `step2_score_signals` rows for that date > 0 (non-null scored signals).
8. `SurfaceRunner.run_single_config(config)` returns without exception.
9. Record structure/trade row counts as diagnostics only (may be zero).

---

## 5. Exact files and commands

### Files D5 may add or modify

| Path | Action |
|------|--------|
| `docs/tmp/sprint005_d5_surface_runner_smoke_evidence.md` | **Add** (only permanent D5 repo artifact) |
| Production Python under `src/` / `scripts/` | **None** |
| Tests | **None** (reuse existing; no new test file) |
| Agenda / closeout / Sprint 006 | **None** |

### Ephemeral smoke driver (outside the repository)

Write a one-shot driver under e.g.  
`C:/MomentumCVG_env/ops_logs/d5_surface_runner_smoke_<timestamp>.py`  
(or equivalent ops_logs location). It is **not** committed. It must:

1. Assert clean Git worktree and record `D5 code SHA = HEAD`.
2. Run the startup identity gate (manifest snapshot/build; D3 receipt snapshot/build/`repo_sha`; normalized `features_dir` match). Fail closed before any `SurfaceRunner` construction.
3. Resolve snapshot manifest paths; build `SurfaceDataPaths` as above.
4. Construct the smoke-only `BacktestRunConfig`.
5. Assert `features_path_for_config` → accepted `(42,8)` file.
6. Construct `SurfaceRunner`.
7. Load features; count rows/columns for `2022-09-02`.
8. Run step1 + step2 for that date; record universe/signal counts.
9. Call `run_single_config`; record completion + diagnostic trade/structure counts.
10. Print a compact PASS/FAIL summary to stdout; exit `0` on PASS, nonzero on FAIL.
11. Write **no** trade logs, search results, or temp artifacts into the Git repo; do not modify accepted inputs.

### Production command (exact shape)

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe C:/MomentumCVG_env/ops_logs/d5_surface_runner_smoke_<timestamp>.py
```

Working directory: `C:/MomentumCVG`. Capture exit code for the evidence memo.

### Focused regression command (no new tests)

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest `
  tests/unit/test_surface_runner_data_flow.py `
  tests/contract/test_orchestration_contract.py `
  tests/contract/test_step1_universe_contract.py `
  tests/contract/test_step2_signals_contract.py `
  -q
```

Run before the production smoke. Reuses the existing synthetic `SurfaceRunner` / S1 / S2 coverage; does not require a new synthetic file because no production code changes.

---

## 6. Focused test strategy

| Item | Decision |
|------|----------|
| New synthetic regression test | **Not required** (no production-code change) |
| Reused suites | `test_surface_runner_data_flow.py`, `test_orchestration_contract.py`, `test_step1_universe_contract.py`, `test_step2_signals_contract.py` |
| Production smoke | Real accepted artifacts; one date; evidence memo |
| Compatibility fix + regression | **Not authorized** unless a real consumability defect appears at smoke time — then stop, do not silently expand scope |

---

## 7. Production execution and evidence procedure

1. Confirm HEAD is the D5 implementation baseline (this planning SHA until execution begins; execution uses clean HEAD with only the planned D5 evidence change after the smoke).
2. Confirm working tree clean; accepted snapshot / D3 / D4 inputs untouched.
3. Run focused pytest suite above; record pass counts.
4. Materialize the ephemeral ops_logs smoke driver; run the production command.
5. Write `docs/tmp/sprint005_d5_surface_runner_smoke_evidence.md` with only:

   * PASS/FAIL verdict  
   * D5 code SHA  
   * Snapshot / build IDs  
   * D3 producer SHA + receipt identity  
   * D4 evidence identity  
   * Manifest-resolved A1 / A2 / liquidity absolute paths  
   * Accepted feature root + exact `features_42_8.parquet`  
   * Selected trade date `2022-09-02` + selection rule  
   * Smoke-only columns + config fields  
   * Complete command + exit code  
   * Feature rows, PIT-universe rows, scored-signal rows for that date  
   * `run_single_config()` completed (yes/no)  
   * Structure/trade row counts (diagnostics only)  
   * Tests run + results  
   * Confirmation that no accepted input was modified  

6. Commit only the evidence memo (separate future commit). Do not push.

No JSON audit artifact, dashboard, plots, performance table, or committed trade log.

---

## 8. One-block execution sequence

**Shape: one block.**

Because inspection shows no consumer compatibility defect and no committed entrypoint is required:

1. Run focused existing tests → run ephemeral production smoke → write evidence memo.

Do **not** split D5 into implementation vs smoke phases. Do **not** start Sprint 005 closeout or Sprint 006.

---

## 9. Acceptance criteria

**PASS** only if all hold:

1. Focused reused pytest suites pass.
2. Startup identity gate passes before `SurfaceRunner` construction (manifest + D3 receipt scalars + resolved `features_dir`).
3. Manifest-resolved A1/A2/liquidity paths load; runner initializes.
4. `features_path_for_config` resolves exactly to accepted `features_42_8.parquet`.
5. Feature file loads with expected `(42,8)` signal columns.
6. Selected date is `2022-09-02` (median rule); feature rows on that date > 0.
7. PIT universe rows > 0 on that date.
8. Scored-signal rows > 0 after `step2_score_signals` on that date.
9. `run_single_config()` completes without exception for the one-date interval.
10. Evidence memo written with required fields; accepted inputs unmodified; no repo trade-log / search outputs.

**FAIL** if any checkpoint fails. Do not hunt alternate dates for economics. Structure/trade counts are diagnostic and must not gate PASS/FAIL by profitability, Sharpe, or nonzero trades.

---

## 10. Explicit non-goals and stop conditions

**Non-goals:** more than one window or trade date; full-history / multi-date backtests; parameter search / window ranking; returns / Sharpe / drawdown / profitability; trading-eligibility or count-threshold policy; Momentum/CVG semantic changes; recomputing D2/D3; re-auditing D4 or all 281 files; rebuilding surfaces/liquidity; modifying the accepted snapshot; manifest/data-store/orchestration frameworks; fixing `run_surface_search.py`; incremental refresh / scheduling / dashboards / live execution; Sprint 005 closeout; any Sprint 006 work.

**Stop conditions:**

* HEAD / tree / identities differ from the accepted checkpoint → stop and report.
* Smoke reveals a genuine consumability defect in `SurfaceRunner` / `SurfaceDataPaths` → stop; do not silently redesign; any fix requires a new narrow authorization naming defect, blocked Sprint 006 result, smallest file, and focused regression.
* Pressure to expand into economics, multi-date runs, or grid search → refuse; D5 is consumability only.
