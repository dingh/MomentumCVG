# Baseline status

**Status:** Active  
**Last recorded:** 2026-07-04 (C5 adjusted-liquid closeout)

---

## Python environment

| Item | Value |
|------|-------|
| Venv | `C:/MomentumCVG_env/venv/` |
| Activate | `& C:/MomentumCVG_env/venv/Scripts/Activate.ps1` |
| Python | 3.13.7 (as of last pytest run) |

---

## Unit tests

| Item | Value |
|------|-------|
| Command | `python -m pytest tests/ -q` |
| Result | **335 passed** |
| Duration | ~4.0s |
| Date | 2026-07-04 (C5 subset; full suite count varies) |

### C5 adjusted-liquid regression (no ORATS cache required)

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_fetch_splits_cli.py tests/unit/test_apply_split_adjustment_cli.py tests/unit/test_split_adjuster.py tests/unit/test_split_adjuster_filtered_zip.py tests/unit/test_ticker_universe.py tests/unit/test_audit_adjusted_liquid.py tests/unit/test_adjusted_liquid_paths.py -q
```

Production adjusted chains: `C:/MomentumCVG_env/input/adjusted_liquid` (see [sprint_memos/004_c5_adjusted_liquid.md](sprint_memos/004_c5_adjusted_liquid.md)).

No integration or end-to-end backtest smoke test in CI yet.

---

## Smoke commands

### Always available (no ORATS cache required)

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest c:\MomentumCVG\tests\ -q
```

### Surface backtest (requires precomputed cache)

Depends on artifacts under `C:/MomentumCVG_env/cache/`:

- `ticker_liquidity_panel.parquet`
- Feature parquet(s) for momentum/CVG
- Option surface meta + quotes

Example (from `scripts/run_surface_search.py` docstring):

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

**Status:** Not run during Week 0 (docs-only). Record result here after first successful smoke in Sprint 002+.

### Legacy backtest

```powershell
python scripts/run_backtest.py configs/baseline_sp500.json
```

**Status:** Not v1 canonical path; optional comparison only.

---

## Known gaps at baseline

- `BacktestEngineV2.run()` not implemented
- No automated backtest smoke in test suite
- v1 portfolio caps (max-loss budget, sector cap) not fully pinned in code

---

## Update log

| Date | Change |
|------|--------|
| 2026-05-23 | Week 0: 326 tests green via project venv |
| 2026-05-27 | Sprint 001 Session B: +9 surface runner data-flow tests; 335 total |
| 2026-07-04 | C5 closeout: adjusted-liquid path constants + audit regression subset documented |
