# Sprint 005 D5 — SurfaceRunner consumer smoke evidence

**Production verdict:** `PASS / ACCEPT`

## Identities

| Field | Value |
|-------|-------|
| D5 execution SHA | `b19e9c8869664bf1ebc9e0b796f8045dd900a196` |
| Snapshot ID | `e2c1f8fd44d72176` |
| Build ID | `20260724T045049097520Z_40b16886` |
| Snapshot root | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886` |
| D3 producer SHA (receipt `repo_sha`) | `131d0ac05e1e57749d3095923927a394fdcbc25b` |
| D3 evidence commit | `816e28f7b63cb9668de94f9cee037d76758fff71` |
| D3 receipt | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features_backfill_v1.lineage.json` (`status=complete`; SHA-256 `c585bce169d897d8a393e9cbf7c62a4e42d28e9139e4dce51eabdacc8f4866a5`) |
| D4 implementation | `22a8375d2d6c3b2dbd661697d9524548ea6def9a` |
| D4 evidence commit | `3c59f05ed971b0d56afd39937113a4f55e0880a1` |
| Approved plan commit | `b19e9c8869664bf1ebc9e0b796f8045dd900a196` (plan retired at Sprint 005 closeout) |

## Startup identity gate

| Check | Result |
|-------|--------|
| Manifest `snapshot_id` / `build_id` | PASS (`e2c1f8fd44d72176` / `20260724T045049097520Z_40b16886`) |
| D3 receipt `snapshot_id` / `build_id` / `repo_sha` | PASS (same snapshot/build; `131d0ac…`) |
| Resolved receipt `features_dir` vs accepted feature root | PASS (`C:\MomentumCVG_env\derived\e2c1f8fd44d72176\features`) |
| `SurfaceRunner` constructed only after gate | Yes |

## Resolved paths

| Artifact | Absolute path |
|----------|---------------|
| Manifest | `C:\MomentumCVG_env\snapshots\20260724T045049097520Z_40b16886\manifests\input_snapshot_e2c1f8fd44d72176.json` |
| A1 surface meta | `C:\MomentumCVG_env\snapshots\20260724T045049097520Z_40b16886\cache\surface\option_surface_meta_weekly_2018_2026.parquet` |
| A2 surface quotes | `C:\MomentumCVG_env\snapshots\20260724T045049097520Z_40b16886\cache\surface\option_surface_quotes_weekly_2018_2026.parquet` |
| Liquidity panel | `C:\MomentumCVG_env\snapshots\20260724T045049097520Z_40b16886\input\liquidity\ticker_liquidity_panel.parquet` |
| Accepted feature root | `C:\MomentumCVG_env\derived\e2c1f8fd44d72176\features` |
| `(42,8)` feature file (`features_path_for_config`) | `C:\MomentumCVG_env\derived\e2c1f8fd44d72176\features\features_42_8.parquet` |

Mutable `C:/MomentumCVG_env/cache` defaults were not used; artifacts were not copied into cache.

## Smoke scope and config

| Item | Value |
|------|-------|
| Trade date | `2022-09-02` (plan median rule; interval `2022-09-02`–`2022-09-03`) |
| Window / file | `(42,8)` / `features_42_8.parquet` |
| Columns | `mom_42_8_mean`, `cvg_42_8`, `mom_42_8_count` |
| `run_id` | `sprint005_d5_surface_runner_smoke` |
| Signal / universe fractions | `min_count_pct=0.01`; `long_top_pct=short_bottom_pct=0.25`; `cvg_filter_pct=1.0`; `dvol_top_pct=spread_bottom_pct=1.0` |
| Structure / sizing | `ironfly` + `closest_delta` / `0.25`; `sizing_mode=conceptual`; `tier_a_mode=equal_premium`; budgets `10000.0` |
| Other | `max_names_per_side=10`; `max_loss_budget_per_trade=500.0`; `earnings_exclusion_days=0`; `cost_model=mid`; `FillAssumption.mid()`; `include_diagnostics=True` |

Thresholds are smoke connectivity controls only — not accepted Sprint 006 strategy parameters.

## Command and exit

```powershell
$env:PYTHONPATH = "C:\MomentumCVG"
& C:/MomentumCVG_env/venv/Scripts/python.exe C:/MomentumCVG_env/ops_logs/d5_surface_runner_smoke_20260809T155544.py
```

| Field | Value |
|-------|-------|
| Working directory | `C:/MomentumCVG` |
| Exit code | `0` |
| Ephemeral driver | outside repo under `C:/MomentumCVG_env/ops_logs/` (not committed) |

## Checkpoint counts (`2022-09-02`)

| Checkpoint | Count / result |
|------------|----------------|
| Feature rows | `2391` |
| PIT universe rows (`_step1_universe`) | `3869` |
| Scored-signal rows (`_step2_signals`) | `911` |
| `run_single_config()` completed | **Yes** |
| Diagnostic trade-log rows | `911` (not a PASS gate) |
| Diagnostic `included_in_portfolio` rows | `20` (not a PASS gate) |

## Tests

| Suite | Result |
|-------|--------|
| Focused: `test_surface_runner_data_flow.py`, `test_orchestration_contract.py`, `test_step1_universe_contract.py`, `test_step2_signals_contract.py` | **33 passed** |
| Full: `python -m pytest -q` | **1494 passed**, 1 skipped |

## Confirmations

* Accepted snapshot, D2 observations, D3 features/receipt, and D4 audit JSON were not modified.
* No production code, tests, configs, or sprint-status files were changed for D5.
* No trade logs, search results, or temporary artifacts were written into the Git repository.
* Sprint 005 closeout and Sprint 006 were not started.

## Residual limitation

D5 proves **consumability only**: the accepted `(42,8)` feature artifact loads through the canonical `SurfaceRunner` with snapshot surfaces/liquidity and scores at least one real trade date. It does **not** establish economic validity, strategy performance, window ranking, or go/no-go readiness.
