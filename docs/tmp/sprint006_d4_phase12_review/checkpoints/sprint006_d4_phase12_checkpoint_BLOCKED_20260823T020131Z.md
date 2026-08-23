# Sprint 006 D4 — Phase 1 / Phase 2 checkpoint

**Document path:** `C:/MomentumCVG_env/runs/sprint006_d4_phase12_20260823T020131Z/sprint006_d4_phase12_checkpoint.md`
**Written (UTC):** 2026-08-23T02:01:31Z
**Location:** outside the Git repository, under `C:/MomentumCVG_env/runs/`
**Checkpoint verdict:** `BLOCKED`
**Next action:** `AWAITING HUMAN REVIEW — PHASE 3 NOT AUTHORIZED`

---

## 1. Scope and blind-inspection attestation

### Scope executed

Sprint 006 D4 **Phase 1 (pre-execution gate)** and **Phase 2 (small real-data smoke)** only, following
`docs/tmp/sprint006_d4_execution_acceptance_plan.md` at the execution commit. Phase 3 (official full
baseline), Phase 4 (blind technical verification), and Phase 5 (economic review and closeout) were **not**
started and are not authorized by this document.

Phase 1 completed and passed. Phase 2 **failed at first execution** and was stopped immediately per the
plan's §2.4 stop conditions and §9 stop-and-escalate rule. See §12 for the blocker.

### Blind-inspection attestation

The operator attests, for the entire Phase 1 + Phase 2 window recorded here:

- No `decision_report.json` or `decision_report.md` was opened, parsed, printed, or summarized. None was
  produced by this checkpoint.
- No `run_summary_*.json` economic field was opened, printed, or summarized. None was produced.
- No aggregate return, compounded return, annualized return, Sharpe, drawdown, hit rate, win rate,
  profit factor, yearly table, long/short attribution, top-five ticker concentration, turnover, or
  cross-minus-mid economic figure was computed, read, or inferred.
- No trade-level or date-level profit and loss was inspected. The smoke run aborted in preflight and
  produced **zero** economic artifacts of any kind, so there was nothing economic available to inspect.
- Repository state was read-only throughout. No repository file was created, modified, or deleted; no
  commit and no push was made.
- The only real-data content read was **identity and calendar** metadata: file digests, sizes,
  timestamps, and the `entry_date` column of the A1 surface-meta parquet used for the median-date
  derivation required by the plan (§2.1). No feature values, quote values, or price values were read.

---

## 2. Execution identity

| Item | Value |
|------|-------|
| `EXECUTION_COMMIT` | `5c31e4903a345f496eaca90d81981f3bc6c468e7` |
| Commit subject | `docs(sprint006): correct D4 fill-delta check` |
| Branch | `main` |
| `git status --porcelain` (before execution) | empty (no output) |
| `git status --porcelain` (after execution) | empty (no output) |
| Ancestry `git merge-base --is-ancestor 10133f6 HEAD` | exit `0` — `10133f6` **is** an ancestor |
| `git log --oneline -3` | `5c31e49 docs(sprint006): correct D4 fill-delta check` · `2b246d3 docs(sprint006): finalize D4 execution plan` · `e941bd5 docs(sprint006): define D4 execution and acceptance plan` |
| Plan correction — V-11 | Present. Plan line 165 cites the receipt `repo_sha` check as **V-11**; plan line 596 is the V-11 row, "Code identity — receipt `repo_sha == EXECUTION_COMMIT`". |
| Plan correction — S-9 half-spread | Present. Plan line 436 (S-9) and line 661 (manual audit) both state `cross_fill_price − mid_fill_price = ±0.5 × (ask − bid)` with the `ask > bid` / zero-spread clarification. |
| Interpreter | `C:/MomentumCVG_env/venv/Scripts/python.exe` (Python 3.13.7) |
| Working directory | `C:\MomentumCVG` |
| Shell | PowerShell |

All preconditions required by the acceptance request were satisfied before any execution step ran.

---

## 3. Phase 1 results table

| Step | Check | Expected result | Observed result | Command / source | Verdict |
|------|-------|-----------------|-----------------|------------------|---------|
| 1.1 | Clean tree | `git status --porcelain` prints nothing | printed nothing | `git status --porcelain` | `PASS` |
| 1.1 | Branch | `main` | `main` | `git branch --show-current` | `PASS` |
| 1.1 | Execution commit | 40-hex HEAD, recorded | `5c31e4903a345f496eaca90d81981f3bc6c468e7` | `git rev-parse HEAD` | `PASS` |
| 1.1 | D3 ancestry | `10133f6` is an ancestor (exit 0) | exit `0` | `git merge-base --is-ancestor 10133f6 HEAD` | `PASS` |
| 1.1 | HEAD is accepted-plan commit | HEAD carries the corrected D4 plan | HEAD = `5c31e49`, the corrected plan commit | `git log --oneline -3`; plan text at HEAD | `PASS` |
| 1.2 | Full regression suite | exit 0, zero failures, zero errors, no new skip | `1597 passed, 1 skipped in 58.06s`, exit `0` | `python -m pytest -q` | `PASS` |
| 1.2 | Skip identity | only the historical pre-existing skip | `tests/unit/test_audit_pit_universe_cli.py:865 — symlink creation not supported on this platform` | `python -m pytest -q -rs tests/unit/test_audit_pit_universe_cli.py` | `PASS` |
| 1.2 | Focused Sprint 006 subset | exit 0 | `332 passed in 10.64s`, exit `0` | focused `pytest -q` (15 files, §4 above) | `PASS` |
| 1.3 | Frozen-contract dry run | exit 0; required stdout lines; no `ERROR:` | exit `0`; all required lines present; no `ERROR:` | `run_sprint006_baseline.py --dry-run` | `PASS` |
| 1.3 | Dry run wrote nothing | placeholder directory absent afterwards | `Test-Path … = False` | `Test-Path C:/MomentumCVG_env/runs/sprint006_d4_dryrun_placeholder` | `PASS` |
| 1.4 | Contract Git blob | `805faa5cdb94618538c60d5afdd715fec84ac608` | `805faa5cdb94618538c60d5afdd715fec84ac608` | `git rev-parse HEAD:configs/sprint006_baseline_v1.json` | `PASS` |
| 1.4 | Contract blob size | `11920` | `11920` | `git cat-file -s HEAD:configs/sprint006_baseline_v1.json` | `PASS` |
| 1.4 | Contract unmodified vs HEAD | exit 0 | exit `0` | `git diff --quiet HEAD -- configs/sprint006_baseline_v1.json` | `PASS` |
| 1.4 | On-disk SHA-256 (CRLF) | `4012b4a4…a54c` | `4012b4a472448004e1a1b14e8814f506911ea0e263e35157b4e13e27ed51a54c` | `Get-FileHash -Algorithm SHA256` | `PASS` |
| 1.4 | Committed-LF SHA-256 | `3cd57f4d…f715` | `3cd57f4dc8cdf8a62af266e529459d88b4f729f369a5fb455fe84621aceef715` | LF-normalized SHA-256 (plan §1.4 snippet) | `PASS` |
| 1.4 | LF byte length | `11920` | `11920` | LF-normalized byte count | `PASS` |
| 1.4 | Frozen-parameter checklist | every §2.3 row matches verbatim | every row matched — see §6 | read of `configs/sprint006_baseline_v1.json` | `PASS` |
| 1.5 | Manifest identity | `snapshot_id=e2c1f8fd44d72176`, `build_id=20260724T045049097520Z_40b16886` | exact match | `ConvertFrom-Json` on manifest | `PASS` |
| 1.5 | Lineage identity | `status=complete`, same snapshot/build, `repo_sha=131d0ac0…` | exact match | `ConvertFrom-Json` on lineage receipt | `PASS` |
| 1.6 | Lineage receipt digest | `c585bce1…66a5` | `c585bce169d897d8a393e9cbf7c62a4e42d28e9139e4dce51eabdacc8f4866a5` | `Get-FileHash` | `PASS` |
| 1.6 | Feature config digest | `764056ce…32dd` (line-ending aware) | on-disk `764056ce7153751d93c1764b1b4cae13a521bf5c3baee729db30bb69543132dd` — exact match | `Get-FileHash` + LF-normalized hash | `PASS` |
| 1.6 | All accepted inputs present | 7 files exist, none under `C:/MomentumCVG_env/cache` | all 7 present; none under the mutable cache | digest/size/mtime enumeration | `PASS` |
| 1.6 | Pre-run digest baseline | recorded for Phase 4 re-verification | recorded — see §8 | digest/size/mtime enumeration | `PASS` |
| 1.7 | `RUN_DIR` does not exist | `Test-Path` False | `False` | `Test-Path $RUN_DIR` | `PASS` |
| 1.7 | `RUN_DIR` outside forbidden roots | outside repo and outside `…_env/cache` | `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T015847Z` — outside both | path inspection | `PASS` |
| 1.7 | Parent exists and is writable | parent exists, writable | **Deviation D-1:** `C:/MomentumCVG_env/runs` did not exist. `C:/MomentumCVG_env` exists and a write probe succeeded. `create_run_dir` uses `mkdir(parents=True)`. | `Test-Path`; write probe; `src/backtest/sprint006_baseline.py:434` | `PASS (with deviation D-1)` |
| 1.8 | Phase 1 gate | all of 1.1–1.7 pass | all pass; one benign deviation recorded | — | `PASS` |

**Phase 1 verdict: `PASS`.**

---

## 4. Test suite results

### Full regression suite

```
Command: C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest -q
platform win32 -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: C:\MomentumCVG   configfile: pytest.ini
plugins: anyio-4.12.1, cov-7.0.0, mock-3.15.1
collected 1598 items
====================== 1597 passed, 1 skipped in 58.06s =======================
```

| Field | Value |
|-------|-------|
| Collected | 1598 |
| Passed | 1597 |
| Failed | 0 |
| Errors | 0 |
| Skipped | 1 |
| pytest-reported duration | 58.06 s |
| Wall-clock duration | 60.3 s |
| Exit code | 0 |

**Skip identity (required by plan §1.2):**

```
Command: C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest -q -rs tests/unit/test_audit_pit_universe_cli.py
SKIPPED [1] tests\unit\test_audit_pit_universe_cli.py:865: symlink creation not supported on this platform
======================== 61 passed, 1 skipped in 5.95s ========================
```

This is a single platform-capability skip, consistent with the historical single pre-existing skip recorded
in `docs/baseline_status.md` (Sprint 005: 1494 passed / 1 skipped; D1: 1528 passed / 1 skipped). **No new
skip was introduced.**

### Focused Sprint 006 subset

```
Command: C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest -q \
  tests/unit/test_sprint006_baseline_adapter.py tests/unit/test_sprint006_leg_log.py \
  tests/unit/test_surface_decision_report.py tests/unit/test_surface_runner_data_flow.py \
  tests/unit/test_option_surface_straddle.py tests/unit/test_option_surface_ironfly.py \
  tests/contract/test_orchestration_contract.py tests/contract/test_step1_universe_contract.py \
  tests/contract/test_step2_signals_contract.py tests/contract/test_step3_structures_contract.py \
  tests/contract/test_step4_exclusions_contract.py tests/contract/test_step5_select_and_size_contract.py \
  tests/contract/test_settle_contract.py tests/contract/test_run_metrics_contract.py \
  tests/contract/test_run_envelope_contract.py
collected 332 items
============================ 332 passed in 10.64s =============================
```

| Field | Value |
|-------|-------|
| Collected | 332 |
| Passed | 332 |
| Failed | 0 |
| Errors | 0 |
| Skipped | 0 |
| pytest-reported duration | 10.64 s |
| Wall-clock duration | 11.8 s |
| Exit code | 0 |

All 15 focused files were collected; none was missing.

---

## 5. Dry run

```
Command: C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint006_baseline.py \
  --contract configs/sprint006_baseline_v1.json \
  --output-dir C:/MomentumCVG_env/runs/sprint006_d4_dryrun_placeholder \
  --dry-run

contract: C:\MomentumCVG\configs\sprint006_baseline_v1.json
contract identity: sprint006_baseline_v1 v1 (accepted)
contract sha256: 4012b4a472448004e1a1b14e8814f506911ea0e263e35157b4e13e27ed51a54c
run: sprint006_baseline_v1_mid fill=mid dates=2018-10-26..2026-07-10
run: sprint006_baseline_v1_cross fill=cross dates=2018-10-26..2026-07-10
features dir: C:\MomentumCVG_env\derived\e2c1f8fd44d72176\features
surface meta: C:\MomentumCVG_env\snapshots\20260724T045049097520Z_40b16886\cache\surface\option_surface_meta_weekly_2018_2026.parquet
surface quotes: C:\MomentumCVG_env\snapshots\20260724T045049097520Z_40b16886\cache\surface\option_surface_quotes_weekly_2018_2026.parquet
liquidity panel: C:\MomentumCVG_env\snapshots\20260724T045049097520Z_40b16886\input\liquidity\ticker_liquidity_panel.parquet
dry run: no economic execution performed

Exit code: 0
```

| Required stdout element (plan §1.3) | Observed | Verdict |
|-------------------------------------|----------|---------|
| `contract identity: sprint006_baseline_v1 v1 (accepted)` | present, exact | `PASS` |
| `contract sha256:` = `4012b4a4…` (CRLF checkout) | present, exact | `PASS` |
| `run: sprint006_baseline_v1_mid fill=mid dates=2018-10-26..2026-07-10` | present, exact | `PASS` |
| `run: sprint006_baseline_v1_cross fill=cross dates=2018-10-26..2026-07-10` | present, exact | `PASS` |
| Four resolved accepted paths equal to plan §2.2 | all four match character-for-character (Windows separators as emitted) | `PASS` |
| `dry run: no economic execution performed` | present | `PASS` |
| No `ERROR:` line | none | `PASS` |
| Both runs printed (not one) | both printed | `PASS` |

**Proof that the dry run wrote nothing:**

```
Test-Path C:/MomentumCVG_env/runs/sprint006_d4_dryrun_placeholder
False
```

The placeholder directory did not exist after the dry run. Independently corroborated by the `runs`
root listing in §10, which contains no `…dryrun_placeholder` entry.

---

## 6. Frozen configuration and parameter checklist

| Item | Value |
|------|-------|
| Git blob | `805faa5cdb94618538c60d5afdd715fec84ac608` |
| Git blob byte size | `11920` |
| On-disk byte size (CRLF working copy) | `12160` |
| On-disk SHA-256 | `4012b4a472448004e1a1b14e8814f506911ea0e263e35157b4e13e27ed51a54c` |
| Committed-LF SHA-256 | `3cd57f4dc8cdf8a62af266e529459d88b4f729f369a5fb455fe84621aceef715` |
| LF-normalized byte size | `11920` (equals the Git blob size) |
| `git diff --quiet HEAD -- configs/…` | exit `0` (unmodified) |

Both digest values in plan §2.1 are accounted for: the on-disk CRLF digest matched, and the
LF-normalized digest reproduced the committed value exactly.

### Frozen-parameter checklist (plan §2.3)

| Item | Frozen expectation | Observed in contract | Verdict |
|------|--------------------|----------------------|---------|
| Contract identity | `sprint006_baseline_v1`, version 1, `accepted` | `contract_id=sprint006_baseline_v1`, `contract_version=1`, `status=accepted` | `PASS` |
| Window | `max_lag=42`, `min_lag=8`, `window_size=35`, `search=false` | identical | `PASS` |
| Feature columns | `mom_42_8_mean` / `cvg_42_8` / `mom_42_8_count` / `cvg_count_42_8` | identical | `PASS` |
| Count eligibility | `min_count_pct=0.8`, `derived_required_count=28`, joint on both count columns | identical; `joint_columns=["mom_42_8_count","cvg_count_42_8"]` | `PASS` |
| Selection | `long_top_pct=0.1`, `short_bottom_pct=0.1`, `cvg_filter_pct=0.5`, `max_names_per_side=25`, tie-break ticker ascending | identical; tie-break in `ranking_and_selection.cap_tie_break` | `PASS` |
| Universe | `dvol_top_pct=0.2`, `spread_bottom_pct=1.0`, `earnings_exclusion_days=0` | identical | `PASS` |
| Structures | long ATM straddle; `short_structure=ironfly`; `wing_delta_target=0.15`; `max_leg_spread_pct=0.5`; `max_spread_cost_ratio=null` | identical | `PASS` |
| Sizing | `sizing_mode=conceptual`, `tier_a_mode=equal_max_loss`, short budget `10000.0`, long budget `10000.0` (fallback only), `contract_multiplier=100.0`, `tier_b_short_max_loss_budget=null`, `deployable_capital=null` | identical | `PASS` |
| Cost assumptions | fill is sole pricing: mid `(0.5,0.5)`, cross `(1.0,1.0)`; `cost_model="mid"` inactive legacy on both runs | identical; `execution_pricing.authoritative_field="fill"`, `no_stacked_spread_deduction=true` | `PASS` |
| Legacy pins | `max_loss_budget_per_trade=500.0`, `include_diagnostics=true`, `wing_selection_rule="closest_delta"` | identical | `PASS` |
| Run window | `2018-10-26` … `2026-07-10` inclusive | `run_start_date=2018-10-26`, `run_end_date=2026-07-10`, `run_end_inclusive=true` | `PASS` |
| Primary reporting window | `2020-01-01` … `2026-07-10` | identical | `PASS` |
| `runs[]` | exactly two; alphas `(0.5,0.5)` and `(1.0,1.0)`; exactly one `primary_decision_view: true` (cross) | exactly two; alphas correct; only `sprint006_baseline_v1_cross` has `primary_decision_view=true` | `PASS` |
| `accepted_inputs.earnings_path` | `null` | `null` | `PASS` |
| `accepted_inputs.mutable_cache_forbidden` | `true` | `true` | `PASS` |
| `shared_run_config` date fields | equal to `periods` run window | `start_date=2018-10-26`, `end_date=2026-07-10` | `PASS` |

---

## 7. Snapshot, build, and lineage identities

**Manifest** — `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/manifests/input_snapshot_e2c1f8fd44d72176.json`

```
snapshot_id : e2c1f8fd44d72176
build_id    : 20260724T045049097520Z_40b16886
```

**Feature lineage receipt** — `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features_backfill_v1.lineage.json`

```
status      : complete
snapshot_id : e2c1f8fd44d72176
build_id    : 20260724T045049097520Z_40b16886
repo_sha    : 131d0ac05e1e57749d3095923927a394fdcbc25b
```

| Identity | Expected (plan §2.2) | Observed | Verdict |
|----------|----------------------|----------|---------|
| Manifest `snapshot_id` | `e2c1f8fd44d72176` | `e2c1f8fd44d72176` | `PASS` |
| Manifest `build_id` | `20260724T045049097520Z_40b16886` | identical | `PASS` |
| Lineage `status` | `complete` | `complete` | `PASS` |
| Lineage `snapshot_id` / `build_id` | same as manifest | same | `PASS` |
| Lineage producer `repo_sha` | `131d0ac05e1e57749d3095923927a394fdcbc25b` | identical | `PASS` |

---

## 8. Pre-run input digest baseline

Recorded at Phase 1 as the immutability baseline (plan PG-2 resolution). Phase 4 must re-run the same
command and obtain byte-identical digests.

| SHA-256 | Bytes | Last write (UTC) | Path |
|---------|-------|------------------|------|
| `f34fb2556da03e9113f4a56a23e4e7dff2296810d5c848e24ff251678991b7bc` | 5,865,941 | 2026-08-09T00:11:52.1850565Z | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features/features_42_8.parquet` |
| `c585bce169d897d8a393e9cbf7c62a4e42d28e9139e4dce51eabdacc8f4866a5` | 70,409 | 2026-08-09T00:31:19.0125532Z | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features_backfill_v1.lineage.json` |
| `6737ab2073be4aab874454faf849139031bf66031e80ffc81b712ac2edff2f2c` | 808,256 | 2026-08-09T07:06:22.7646137Z | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features_quality_audit_v1.json` |
| `304753a2d5ce9900bdf462442f4f11407c8ec821ec5708ef9190027b4b3b7c4a` | 16,159,021 | 2026-07-26T07:18:45.0391122Z | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/cache/surface/option_surface_meta_weekly_2018_2026.parquet` |
| `e8b2b49094362fde3432b2851c47c72004a539db6c37f9a4fbda6f2e6d907ca4` | 261,380,171 | 2026-07-26T07:19:01.7451122Z | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/cache/surface/option_surface_quotes_weekly_2018_2026.parquet` |
| `756d78160047554b3c158e99aa24e337be933de9b47f273f21dce35b85d07d42` | 25,357,630 | 2026-07-25T23:41:43.8240857Z | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/input/liquidity/ticker_liquidity_panel.parquet` |
| `e312fd1932ca2a95b104f1c5b52bb6054270695f23c2670cdf125c10f379e1ab` | 6,955 | 2026-07-26T07:26:19.9239487Z | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/manifests/input_snapshot_e2c1f8fd44d72176.json` |

Additional pinned configuration digest:

| File | On-disk SHA-256 | On-disk bytes | LF-normalized SHA-256 | LF bytes | Pinned value | Verdict |
|------|-----------------|---------------|-----------------------|----------|--------------|---------|
| `configs/feature_backfill_v1.json` | `764056ce7153751d93c1764b1b4cae13a521bf5c3baee729db30bb69543132dd` | 2,050 | `578bf76ddc2ef73fa3570f075a39cff35bb818b7472fc63ee7de7816a72e41b4` | 1,986 | `764056ce…32dd` | `PASS` — the pin matches the on-disk (CRLF) form exactly |

Notes:

- The lineage receipt digest matched its pinned value exactly.
- No listed input resolves under `C:/MomentumCVG_env/cache` (the forbidden mutable producer cache).
- All seven inputs exist and were readable.

---

## 9. Median-date derivation (plan §2.1)

The date was **derived, not hardcoded**. Source of authority is the A1 surface-meta `entry_date`
column over the closed frozen run interval, including rows where `surface_valid=false`, per the
contract's `expected_dates.authority`.

```
Source : C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/
         cache/surface/option_surface_meta_weekly_2018_2026.parquet  (columns=['entry_date'])
Filter : 2018-10-26 <= entry_date <= 2026-07-10   (sorted unique)

n_expected_dates    403
parity              odd
lower_median_index  201  2022-09-02
upper_median_index  201  2022-09-02
first               2018-10-26
last                2026-07-10
MEDIAN_DATE         2022-09-02
```

| Item | Expected | Observed | Verdict |
|------|----------|----------|---------|
| Calendar count `n` | (derived) | 403 | recorded |
| Parity | (derived) | odd — both median indices coincide, so the lower-median convention is not load-bearing here | recorded |
| Lower median index / date | index `(n−1)//2 = 201` | `201` → `2022-09-02` | `PASS` |
| Upper median index / date | index `n//2 = 201` | `201` → `2022-09-02` | `PASS` |
| First / last calendar date | within the frozen window | `2018-10-26` / `2026-07-10` | `PASS` |
| Selected `MEDIAN_DATE` | `2022-09-02` | `2022-09-02` | `PASS` |

The derived date matched the required value, so no escalation was triggered by §2.1. No date was
substituted at any point.

---

## 10. Smoke setup, contract diff, and digests

| Item | Value |
|------|-------|
| `SMOKE_DIR` | `C:/MomentumCVG_env/runs/sprint006_d4_smoke_20260823T020000Z` |
| `SMOKE_OUT` (adapter output dir) | `C:/MomentumCVG_env/runs/sprint006_d4_smoke_20260823T020000Z/run` — **never created** |
| `SMOKE_CONTRACT` | `C:/MomentumCVG_env/runs/sprint006_d4_smoke_20260823T020000Z/sprint006_smoke_contract.json` |
| `SMOKE_DIR` existed before | `False` |
| `SMOKE_OUT` existed before run | `False` |
| Smoke contract SHA-256 | `28e068fea5fce3f66b9351febd04b1e1a05dc57ce45b6d9a416ef00e62a781cb` |
| Smoke contract bytes | 12,160 |
| Frozen contract SHA-256 after the copy | `4012b4a472448004e1a1b14e8814f506911ea0e263e35157b4e13e27ed51a54c` — unchanged |
| `git status --porcelain configs/sprint006_baseline_v1.json` | empty — the frozen contract was never modified |

### Four-line diff (`Compare-Object … -SyncWindow 0`)

```
InputObject                         SideIndicator
-----------                         -------------
    "run_start_date": "2022-09-02", =>
    "run_start_date": "2018-10-26", <=
    "run_end_date": "2022-09-02",   =>
    "run_end_date": "2026-07-10",   <=
    "start_date": "2022-09-02",     =>
    "start_date": "2018-10-26",     <=
    "end_date": "2022-09-02",       =>
    "end_date": "2026-07-10",       <=

DELTA_COUNT = 8   (4 changed lines x 2 sides)
```

Exactly the four intended date fields differ — `periods.run_start_date`, `periods.run_end_date`,
`shared_run_config.start_date`, `shared_run_config.end_date`. No other line differs. In particular,
`primary_reporting_start_date` and `primary_reporting_end_date` were correctly **not** matched.

### `runs` root listing after the checkpoint

```
C:\MomentumCVG_env\runs\sprint006_d4_smoke_20260823T020000Z
C:\MomentumCVG_env\runs\sprint006_d4_phase12_20260823T020131Z
```

No official baseline run directory exists. No dry-run placeholder directory exists.

---

## 11. Smoke checks S-1 … S-10

**Smoke invocation**

```
Command: C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint006_baseline.py \
  --contract C:/MomentumCVG_env/runs/sprint006_d4_smoke_20260823T020000Z/sprint006_smoke_contract.json \
  --output-dir C:/MomentumCVG_env/runs/sprint006_d4_smoke_20260823T020000Z/run

SMOKE_START_UTC = 2026-08-23T02:00:26.3259903Z
SMOKE_END_UTC   = 2026-08-23T02:00:26.9040827Z
Wall clock      = 0.6 s
Exit code       = 2

stderr:
ERROR: contract runs[0] (sprint006_baseline_v1_mid) is not constructible: BacktestRunConfig validation failed:
  - start_date must be before end_date, got 2022-09-02 >= 2022-09-02
```

| # | Check | Expected | Observed evidence | Verdict |
|---|-------|----------|-------------------|---------|
| S-1 | Execution | exit 0; no `ERROR:` on stderr; stdout lists 7 per-run files for both runs plus report and receipt paths | exit `2`; single `ERROR:` line (above); **no** stdout file listing; adapter aborted in preflight while constructing `runs[0]` | `FAIL` |
| S-2 | Artifact set (17 adapter files in `SMOKE_OUT`, counted before the marker) | 17 files | `SMOKE_OUT` was never created; 0 adapter files produced | `NOT RUN` |
| S-3 | Schemas (`date_status` 3 cols; `funnel_summary` 18; `leg_log` 21; `candidate_view` 9) | exact column sets | no artifacts to inspect | `NOT RUN` |
| S-4 | Date status (one row per fill at `MEDIAN_DATE`) | one row per fill | no artifacts to inspect | `NOT RUN` |
| S-5 | Funnel semantics (null vs zero; monotone stage counts) | as specified | no artifacts to inspect | `NOT RUN` |
| S-6 | Leg serialization (2-leg straddle / 4-leg iron fly; sign pattern `+ − − +`) | as specified | no artifacts to inspect | `NOT RUN` |
| S-7 | Included-trade reconciliation (four identities, tol `max(1e-6 abs, 1e-8 rel)`) | identities hold | no artifacts to inspect | `NOT RUN` |
| S-8 | Structure failures (frozen `reason_code`, retained `reason_raw`, no leg rows) | as specified | no artifacts to inspect | `NOT RUN` |
| S-9 | Fill differentiation (**conditional**): where a `(ticker, direction)` is constructable under both fills, buy legs satisfy `cross_fill_price − mid_fill_price = +0.5 × (ask − bid)` and sell legs `−0.5 × (ask − bid)` within `max(1e-6 abs, 1e-8 rel)`; strict inequality only where `ask > bid`; equality correct at zero spread; otherwise record `N/A — no overlapping constructable name` | neither leg log was produced, so overlap existence could not be determined. This is **not** the `N/A — no overlapping constructable name` outcome, which requires two real leg logs to evaluate. | `NOT RUN` |
| S-10 | No aggregate inspection | operator attests nothing economic was opened | attested in §1; no economic artifact was produced | `PASS` |

**Phase 2 verdict: `FAIL` (blocked at S-1).** No date was substituted, no directory was reused, no
configuration was relaxed, and no retry was attempted.

---

## 12. Warnings, deviations, blockers, and retained artifacts

### Deviation D-1 (benign, Phase 1.7)

`C:/MomentumCVG_env/runs` did not exist when Phase 1.7 ran, so the plan's "parent exists and is
writable" criterion was not literally satisfiable. `C:/MomentumCVG_env` exists and a create/delete write
probe succeeded, and `create_run_dir` calls `run_dir.mkdir(parents=True)`
(`src/backtest/sprint006_baseline.py:434`), so a missing `runs` root is created automatically. This is a
first-run condition, not a defect. None of the plan's 1.7 stop conditions (directory already exists, or is
inside a forbidden root) was triggered. Recorded rather than treated as a gate failure.

### Blocker B-1 (Phase 2, checkpoint-blocking)

**Summary.** The accepted plan's PG-1b Option A — a date-narrowed smoke contract with all four date
fields set to the median date — **cannot be executed by the accepted runner**. A single-date window is
structurally invalid.

| Field | Detail |
|-------|--------|
| Where | `BacktestRunConfig.validate()`, `src/backtest/run_config.py:362-366` |
| Rule | `if self.start_date >= self.end_date: errors.append("start_date must be before end_date, got …")` |
| Surfaced by | `build_run_configs` / adapter preflight, before `create_run_dir` |
| Observed error | `ERROR: contract runs[0] (sprint006_baseline_v1_mid) is not constructible: BacktestRunConfig validation failed: - start_date must be before end_date, got 2022-09-02 >= 2022-09-02` |
| Exit code | 2 |
| Artifacts produced | none — no run directory, no parquet, no JSON, no receipt |
| Same rule elsewhere | `src/backtest/engine.py:105` and `src/backtest/config.py:200` enforce the same strict inequality |

**Classification: planning gap, not a code or data defect.** The strict `start_date < end_date` rule is
long-standing intentional validation, unchanged by D0–D3, and is covered by the passing regression suite.
The accepted D4 plan assumed a single-date contract was constructible and did not verify that assumption
against `run_config.py` before acceptance. Under §9 of the plan, a planning gap must be escalated to a
human and must **not** be fixed during D4.

**No remediation was attempted.** Specifically, per the acceptance instruction: no production code was
changed; no data was changed; no retry into the same directory occurred; no alternative date was
substituted; no window was widened; the frozen contract was not touched; and Phase 3 was not started.

**Smallest separately reviewable options for human decision (proposed, not implemented):**

1. **Amend the plan's PG-1b Option A to a minimal two-date window.** Set `run_start_date` /
   `start_date` to `2022-09-02` and `run_end_date` / `end_date` to the **next** A1 expected date after
   the median, keeping the diff to the same four fields. This is a documentation change only and needs
   no code change, but it changes the meaning of "single-date smoke" and would make S-4 expect up to two
   `date_status` rows per fill, so it requires explicit re-acceptance.
2. **Reconsider Option B** (a full frozen run into a disposable directory), previously rejected in the
   plan on cost and confusion grounds.
3. **Add supported single-date capability** (for example, allowing `start_date == end_date` with an
   inclusive end, or a dedicated CLI date filter). This is a production change to shared validation used
   well beyond Sprint 006, would need its own tests, and is explicitly out of scope for D4 per plan
   §10.1.

Option 1 appears the smallest and least invasive, but the choice belongs to the reviewer.

### Retained artifacts

| Path | Content | Note |
|------|---------|------|
| `C:/MomentumCVG_env/runs/sprint006_d4_smoke_20260823T020000Z/sprint006_smoke_contract.json` | the date-narrowed smoke contract, SHA-256 `28e068fe…81cb` | preserved as failure evidence; **not** the frozen contract |
| `C:/MomentumCVG_env/runs/sprint006_d4_smoke_20260823T020000Z/NOT_THE_OFFICIAL_BASELINE.txt` | mandatory smoke marker, annotated with the failure | written per plan §2.5 |
| `C:/MomentumCVG_env/runs/sprint006_d4_smoke_20260823T020000Z/run` | — | never created |
| `C:/MomentumCVG_env/runs/sprint006_d4_phase12_20260823T020131Z/sprint006_d4_phase12_checkpoint.md` | this document | outside the repository |

Nothing was deleted or overwritten. The smoke directory is retained exactly as the failure left it, plus
the mandatory marker required by plan §2.5.

### Other warnings

None. No unexpected stderr output, no deprecation failure, no partial write, and no orphaned directory
were observed.

---

## 13. Checkpoint verdict

**`BLOCKED`**

- Phase 1 (pre-execution gate): `PASS` — every check in 1.1 through 1.8 passed, with one benign recorded
  deviation (D-1).
- Phase 2 (small real-data smoke): `FAIL` at S-1 — the accepted single-date smoke contract is not
  constructible under `BacktestRunConfig` validation (blocker B-1). S-2 through S-9 could not run.

The blocker is a **planning gap in the accepted D4 plan**, not evidence of a defect in the D0–D3
implementation. Phase 1 evidence stands on its own and can be reused if the plan's Phase 2 mechanism is
amended and re-accepted; the Phase 1 input-digest baseline in §8 remains the reference for a future
Phase 4 immutability check, provided the inputs remain untouched.

---

## 14. Next action

**`AWAITING HUMAN REVIEW — PHASE 3 NOT AUTHORIZED`**

Required before any further D4 work:

1. A human decision on blocker B-1 (see the three options in §12).
2. Re-acceptance of the amended Phase 2 mechanism, if the plan is changed.
3. Explicit authorization to run Phase 3.

No official full baseline run has been started, no official run directory exists, and no economic result
of any kind has been produced or inspected.
