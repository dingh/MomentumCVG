# Sprint 004 — C5 adjusted-liquid split layer (closeout)

**Status:** Closed — **C5 accepted**
**Closed:** 2026-07-04
**Design:** temporary C5 design draft removed at Sprint 004 closeout (see git history); canonical closeout [004_closeout.md](004_closeout.md)
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
| Downstream wiring report | C5.11A (temporary report removed at Sprint 004 closeout; see git history) | C5.11A |

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
| C5.2 domain audit | PASS WITH WARNINGS | Temporary draft removed at Sprint 004 closeout (git history) |
| C5.3 `load_ticker_universe` | PASS | `tests/unit/test_ticker_universe.py` |
| C5.4 golden split math | PASS | `tests/unit/test_split_adjuster.py` |
| C5.5 filtered ZIP→parquet | PASS | `tests/unit/test_split_adjuster_filtered_zip.py` |
| C5.6B real-data smoke (2020) | PASS | Temporary draft removed at Sprint 004 closeout (git history) |
| C5.7 scoped split fetch | PASS | Temporary draft removed at Sprint 004 closeout (git history) |
| C5.8B audit on real sample | PASS | Temporary draft removed at Sprint 004 closeout (git history) |
| C5.9 downstream input contract | PASS | Temporary draft removed at Sprint 004 closeout (git history) |
| C5.10B full backfill | PASS (exit 0, 2299 files) | Temporary run log removed at Sprint 004 closeout (git history) |
| C5.10D post-patch audit | **PASS** | Temporary draft removed at Sprint 004 closeout (git history) |
| C5.11A downstream defaults | PASS | Temporary draft removed at Sprint 004 closeout (git history) |

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

C5 only. Later Sprint 004 work (C6–C8.5) is closed in [004_closeout.md](004_closeout.md).

---

## References

| Report | Topic |
|--------|-------|
| Temporary C5.2 / C5.10C / C5.10D drafts | Removed at Sprint 004 closeout (git history) |
| [004_c4_liquidity_panel.md](004_c4_liquidity_panel.md) | Upstream C4 panel |
| [004_closeout.md](004_closeout.md) | Sprint 004 final closeout |

---

## Active documentation map (post-closeout)

| Doc | C5-relevant content |
|-----|---------------------|
| [AGENTS.md](../../AGENTS.md) | Production root `input/adjusted_liquid`; legacy mirror note |
| [repo_map.md](../repo_map.md) | External paths + data flow |
| [v1_weekly_runbook.md](../v1_weekly_runbook.md) | Producer / repair notes (historical sections marked) |
| [v1_universe_protocol.md](../v1_universe_protocol.md) | Raw liquidity vs adjusted chains |
| [current_sprint.md](../agenda/current_sprint.md) | Sprint 005 scope under review |
| [004_closeout.md](004_closeout.md) | Accepted production snapshot |

Temporary C5 design/run reports were removed at Sprint 004 closeout (available in git history).
