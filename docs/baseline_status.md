# Baseline status

**Status:** Active
**Last recorded:** 2026-08-24 (Sprint 006 closeout documentation sync)
**Sprint 006 status:** `CLOSED — EVIDENCE ACCEPTED; FROZEN 42:8 ECONOMICS WEAK/NEGATIVE; HYPOTHESIS REJECTED/DEFERRED`

---

## Python environment

| Item | Value |
|------|-------|
| Venv | `C:/MomentumCVG_env/venv/` |
| Activate | `& C:/MomentumCVG_env/venv/Scripts/Activate.ps1` |
| Python | 3.13.7 (as of last pytest run) |

---

## Unit tests (current accepted baseline)

| Item | Value |
|------|-------|
| Command | `& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest -q` |
| Full suite | **1597 passed**, 1 skipped |
| Focused Sprint 006 subset | **332 passed** |
| Tested baseline / execution commit | `e205b9acc5d0400aa38169de721acb7fb8268f29` |
| Verification date | 2026-08-23 (Sprint 006 D4 Phase 1 gate) |
| Source | [sprint_memos/006_closeout.md](sprint_memos/006_closeout.md) §14; [tmp/sprint006_d4_phase12_checkpoint.md](tmp/sprint006_d4_phase12_checkpoint.md) |

The suite was executed at execution commit `e205b9a` **before** the official baseline run and Sprint 006 closeout documentation commits. This file syncs that accepted Phase 1 result; it does **not** claim a new test run at closeout.

### Historical (Sprint 005 closeout gate)

| Item | Value |
|------|-------|
| Command | `& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest -q` |
| Result | **1494 passed**, 1 skipped |
| Duration | `44.14s` |
| Tested code baseline | `38920791de89a65b05a20985461b0eb1f37317d9` |
| Date | 2026-08-09 |
| Source | [sprint_memos/005_closeout.md](sprint_memos/005_closeout.md) |

### Historical (Sprint 004 closeout gate)

| Item | Value |
|------|-------|
| Command | `& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest` |
| Result | **1321 passed**, 1 skipped |
| Duration | ~31.7s |
| Date | 2026-07-26 |
| Source | [sprint_memos/004_closeout.md](sprint_memos/004_closeout.md) |

### C5 adjusted-liquid regression (no ORATS cache required)

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_fetch_splits_cli.py tests/unit/test_apply_split_adjustment_cli.py tests/unit/test_split_adjuster.py tests/unit/test_split_adjuster_filtered_zip.py tests/unit/test_ticker_universe.py tests/unit/test_audit_adjusted_liquid.py tests/unit/test_adjusted_liquid_paths.py -q
```

Accepted production snapshot (C8.5): `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886` — see [sprint_memos/004_closeout.md](sprint_memos/004_closeout.md). Mutable producer root `C:/MomentumCVG_env/input/adjusted_liquid` is for rebuild/repair only ([sprint_memos/004_c5_adjusted_liquid.md](sprint_memos/004_c5_adjusted_liquid.md)).

Accepted Sprint 005 features: `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features/` — see [sprint_memos/005_closeout.md](sprint_memos/005_closeout.md).

Sprint 006 official economic run: `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` — see [sprint_memos/006_closeout.md](sprint_memos/006_closeout.md).

No integration or end-to-end economic backtest smoke test in CI yet.

---

## Smoke commands

### Always available (no ORATS cache required)

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest -q
```

### Surface backtest (requires precomputed inputs)

Trusted Stage A inputs come from the accepted snapshot manifest. Trusted Sprint 005 features come from the derived root above. Mutable `C:/MomentumCVG_env/cache/` is not the accepted handoff.

Example (from `scripts/run_surface_search.py` docstring; grid-search CLI readiness is separate from the D5 consumer smoke):

```powershell
python scripts/run_surface_search.py `
  --mode full_sample `
  --start-date 2020-01-01 `
  --end-date 2026-12-31 `
  --momentum-cols mom_42_8_mean `
  --fills cross `
  --short-structures ironfly `
  --wing-deltas 0.15
```

**Status:** Not an accepted Sprint 005 economic gate. D5 proved one-date consumability only ([sprint005_d5_surface_runner_smoke_evidence.md](sprint_memos/sprint005_d5_surface_runner_smoke_evidence.md)).

### Sprint 006 frozen baseline (accepted path)

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint006_baseline.py `
  --contract configs/sprint006_baseline_v1.json `
  --output-dir C:/MomentumCVG_env/runs/<new_run_dir>
```

**Status:** Accepted execution path for the frozen contract. Sprint 006 closeout recorded one official run; do not retune the contract after P&L ([006_closeout.md](sprint_memos/006_closeout.md)).

### Legacy backtest

```powershell
python scripts/run_backtest.py configs/baseline_sp500.json
```

**Status:** Not v1 canonical path; optional comparison only.

---

## Known gaps at baseline

- `BacktestEngineV2.run()` not implemented
- No automated economic backtest smoke in test suite
- v1 portfolio caps (max-loss budget, sector cap) not fully pinned in code
- Sprint 006 frozen `42:8` hypothesis rejected/deferred under cross fills — see [006_closeout.md](sprint_memos/006_closeout.md)

---

## Update log

| Date | Change |
|------|--------|
| 2026-05-23 | Week 0: 326 tests green via project venv |
| 2026-05-27 | Sprint 001 Session B: +9 surface runner data-flow tests; 335 total |
| 2026-07-04 | C5 closeout: adjusted-liquid path constants + audit regression subset documented |
| 2026-07-26 | Sprint 004 closeout gate: 1321 passed, 1 skipped |
| 2026-08-09 | Sync accepted Sprint 005 closeout baseline: 1494 passed, 1 skipped in 44.14s at `3892079`; closeout docs commit `c6929d3` (no new suite run claimed) |
| 2026-08-24 | Sync Sprint 006 closeout: Phase 1 gate at `e205b9a` — 1597 passed / 1 skipped full suite, 332 passed focused subset (no new suite run claimed at closeout) |
