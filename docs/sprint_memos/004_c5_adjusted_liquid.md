# Sprint 004 — C5 adjusted-liquid split layer (closeout)

**Status:** Closed — **C5 accepted**  
**Closed:** 2026-07-04  
**Design:** [docs/tmp/c5_split_adjustment_design_plan.md](../tmp/c5_split_adjustment_design_plan.md)  
**Repo commit (path wiring):** `0d2357381e373f217e21ef2213749a5880f195a9`

---

## Deliverable

Scoped **split-adjusted option chains** for the C4 liquidity precompute universe (2,783 tickers), written to a dedicated production root and audited before downstream use.

| Artifact | Path | Notes |
|----------|------|-------|
| Adjusted daily chains | `C:/MomentumCVG_env/input/adjusted_liquid/{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.parquet` | 2,299 files, 2017→2026 |
| Scoped split history | `C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet` | 1,347 rows, 819 tickers |
| Split checkpoint | `.../splits_hist_liquid.checkpoint.parquet` | Sidecar from scoped fetch |
| Central path constants | `src/data/paths.py` | `DEFAULT_ADJUSTED_LIQUID_ROOT`, legacy/raw aliases |
| Output audit CLI | `scripts/audit_adjusted_liquid.py` | Full inventory + sampled math checks |
| Downstream wiring report | `docs/tmp/c5_11a_downstream_path_wiring_report.md` | C5.11A |

**Not modified:** `C:/ORATS/data/ORATS_Adjusted` (legacy full-universe mirror), raw `ORATS_Data` ZIPs.

---

## Pipeline (C5 scope)

```text
liquid_tickers.csv (C4 superset, 2783 names)
  → fetch_splits.py (scoped) → splits_hist_liquid.parquet
  → apply_split_adjustment.py --ticker-universe … --adj-root adjusted_liquid
  → audit_adjusted_liquid.py (PASS required)
  → ORATSDataProvider / Stage A scripts (defaults → adjusted_liquid)
```

**Liquidity panel (C4)** still reads **raw** `ORATS_Data` only — unchanged.

---

## Verification summary

| Phase | Result | Evidence |
|-------|--------|----------|
| C5.2 domain audit | PASS WITH WARNINGS | [c5_split_domain_audit.md](../tmp/c5_split_domain_audit.md) |
| C5.3 `load_ticker_universe` | PASS | `tests/unit/test_ticker_universe.py` |
| C5.4 golden split math | PASS | `tests/unit/test_split_adjuster.py` |
| C5.5 filtered ZIP→parquet | PASS | `tests/unit/test_split_adjuster_filtered_zip.py` |
| C5.6B real-data smoke (2020) | PASS | [c5_6b_smoke_report.md](../tmp/c5_6b_smoke_report.md) |
| C5.7 scoped split fetch | PASS | [c5_7_scoped_split_fetch_report.md](../tmp/c5_7_scoped_split_fetch_report.md) |
| C5.8B audit on real sample | PASS | [c5_8b_real_data_audit_run_report.md](../tmp/c5_8b_real_data_audit_run_report.md) |
| C5.9 downstream input contract | PASS | [c5_9_downstream_input_contract_smoke_report.md](../tmp/c5_9_downstream_input_contract_smoke_report.md) |
| C5.10B full backfill | PASS (exit 0, 2299 files) | [c5_10b_full_backfill_run_log.txt](../tmp/c5_10b_full_backfill_run_log.txt) |
| C5.10D post-patch audit | **PASS** | [c5_10d_full_backfill_audit_report.md](../tmp/c5_10d_full_backfill_audit_report.md) |
| C5.11A downstream defaults | PASS | [c5_11a_downstream_path_wiring_report.md](../tmp/c5_11a_downstream_path_wiring_report.md) |

**C5.10C triage:** initial audit FAIL was an **audit join bug** (SPX/SPXW OPRA keys), not a backfill defect — fixed in C5.10D.

**pytest (C5 subset, 2026-07-04):** 76 passed — `test_fetch_splits_cli`, `test_apply_split_adjustment_cli`, `test_split_adjuster`, `test_split_adjuster_filtered_zip`, `test_ticker_universe`, `test_audit_adjusted_liquid`, `test_adjusted_liquid_paths`.

---

## Production operator commands

### Full filtered backfill (already done — do not rerun casually)

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/apply_split_adjustment.py `
  --raw-root C:/ORATS/data/ORATS_Data `
  --adj-root C:/MomentumCVG_env/input/adjusted_liquid `
  --splits C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet `
  --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv `
  --years 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 `
  --workers 10
```

### Post-backfill audit (required after any adj-root rewrite)

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/audit_adjusted_liquid.py `
  --raw-root C:/ORATS/data/ORATS_Data `
  --adj-root C:/MomentumCVG_env/input/adjusted_liquid `
  --splits C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet `
  --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv
```

### Repair scope (new split for known tickers)

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/apply_split_adjustment.py `
  --tickers NVDA TSLA `
  --adj-root C:/MomentumCVG_env/input/adjusted_liquid `
  --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv `
  --overwrite
```

---

## Downstream defaults (C5.11A)

Active readers default to `DEFAULT_ADJUSTED_LIQUID_ROOT` (`src/data/paths.py`):

- `ORATSDataProvider`, `BacktestConfig` / `DEFAULT_CONFIG`, `configs/baseline_sp500.json`
- Stage A scripts: `extract_spot_prices`, `precompute_option_surface`, `precompute_straddle_history`, `precompute_ironfly_history`, `build_straddle_master_universe`
- `refresh_weekly_inputs.py --orats-adj-root`, `ChainLoader`

**Full-mirror backfill** (`apply_split_adjustment.py` with no `--adj-root`) still targets legacy `ORATS_Adjusted` intentionally.

---

## Known limitations (accepted)

| Item | Notes |
|------|-------|
| Precompute superset ≠ PIT universe | `liquid_tickers.csv` filters adjustment only; S1 still uses PIT panel |
| Legacy mirror retained | `ORATS_Adjusted` not deleted; no longer the active downstream default |
| `refresh_weekly_inputs split-audit` | Still a C2 stub — **deferred to C8**; use standalone `audit_adjusted_liquid.py` until wired |
| Spot / surface on production root | Defaults wired; full Stage A re-extract/re-precompute not part of C5 closeout |
| Incremental adj append | New liquid names / gap repair documented; no watermark engine in C5 |
| Audit sample size | Full inventory pass + 500k-row sampled math (seed 57); not exhaustive row scan |

---

## Remaining before Sprint 004 full closeout

C5 only. Sprint 004 still open for **C6–C9** (surface audit, PIT harness, `refresh` wiring, `validate` umbrella, runbook/CLI cleanup).

---

## References

| Report | Topic |
|--------|-------|
| [c5_split_domain_audit.md](../tmp/c5_split_domain_audit.md) | C5.2 code audit |
| [c5_10c_audit_failure_triage_report.md](../tmp/c5_10c_audit_failure_triage_report.md) | SPX/SPXW join triage |
| [c5_10d_audit_patch_report.md](../tmp/c5_10d_audit_patch_report.md) | Audit script patch |
| [004_c4_liquidity_panel.md](004_c4_liquidity_panel.md) | Upstream C4 panel |

---

## Active documentation map (post-closeout)

| Doc | C5-relevant content |
|-----|---------------------|
| [AGENTS.md](../../AGENTS.md) | Production root `input/adjusted_liquid`; legacy mirror note |
| [repo_map.md](../repo_map.md) | External paths + data flow |
| [v1_weekly_runbook.md](../v1_weekly_runbook.md) | Pipeline order, audit/repair commands |
| [v1_universe_protocol.md](../v1_universe_protocol.md) | Raw liquidity vs adjusted chains |
| [current_sprint.md](../agenda/current_sprint.md) | C5 ✓; C6–C9 remaining |
| [c5_split_adjustment_design_plan.md](../tmp/c5_split_adjustment_design_plan.md) | Design (status: closed) |
| [c5_11a_downstream_path_wiring_report.md](../tmp/c5_11a_downstream_path_wiring_report.md) | Default path wiring evidence |

**Historical `docs/tmp/c5_*` run reports** are retained as evidence; superseded items (e.g. C5.9 path-default gap) are bannered, not rewritten.
