# Baseline status

**Status:** Active
**Last recorded:** 2026-08-09 (Sprint 005 closeout documentation sync)
**Sprint 005 status:** `CLOSED — ACCEPTED WITH DOCUMENTED LIMITATIONS`

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
| Result | **1494 passed**, 1 skipped |
| Duration | `44.14s` |
| Exit code | `0` |
| Tested code baseline | `38920791de89a65b05a20985461b0eb1f37317d9` (`docs: record D5 SurfaceRunner consumer smoke`) |
| Closeout commit | `c6929d308ea072459ed9e9e8ffcdc92e6c1dd1ae` (`docs: close Sprint 005`; documentation-only) |
| Verification date | 2026-08-09 |
| Source | [sprint_memos/005_closeout.md](sprint_memos/005_closeout.md) |

The suite was executed against baseline `3892079` **before** the documentation-only closeout commit `c6929d3`. This file syncs that accepted result; it does **not** claim a new test run against `c6929d3`.

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

### Legacy backtest

```powershell
python scripts/run_backtest.py configs/baseline_sp500.json
```

**Status:** Not v1 canonical path; optional comparison only.

---

## Known gaps at baseline

- `BacktestEngineV2.run()` not implemented
- No automated economic backtest smoke in test suite
- Sprint 006 economic backtesting requires separate authorization
- v1 portfolio caps (max-loss budget, sector cap) not fully pinned in code

---

## Update log

| Date | Change |
|------|--------|
| 2026-05-23 | Week 0: 326 tests green via project venv |
| 2026-05-27 | Sprint 001 Session B: +9 surface runner data-flow tests; 335 total |
| 2026-07-04 | C5 closeout: adjusted-liquid path constants + audit regression subset documented |
| 2026-07-26 | Sprint 004 closeout gate: 1321 passed, 1 skipped |
| 2026-08-09 | Sync accepted Sprint 005 closeout baseline: 1494 passed, 1 skipped in 44.14s at `3892079`; closeout docs commit `c6929d3` (no new suite run claimed) |
