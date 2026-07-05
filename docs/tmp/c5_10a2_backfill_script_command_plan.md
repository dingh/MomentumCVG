# C5.10A2 — Backfill Script Command Plan

**Date:** 2026-07-04 · **Repo SHA:** `63fd91e4069d188c5c72c0d0df069b1ec8d48f1c`  
**Source:** [c5_10a1_backfill_artifact_inventory.md](c5_10a1_backfill_artifact_inventory.md)  
**Scripts inspected:** `apply_split_adjustment.py`, `audit_adjusted_liquid.py`, `split_adjuster.py`, `ticker_universe.py`

Planning only — no backfill, audit, or production data touched.

---

## Backfill readiness verdict

All flags required for C5.10B are supported and forwarded to `SplitAdjuster.run()`: `--raw-root`, `--adj-root`, `--splits`, `--ticker-universe`, `--years`, `--workers`. Filtered mode **requires** explicit `--adj-root` (fail-fast guard).

---

## Output layout

```text
C:/MomentumCVG_env/input/adjusted_liquid/{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.parquet
```

From `split_adjuster._process_single_zip`. Split files at adj-root top level coexist with `{YYYY}/` subdirs — no path conflict.

---

## Skip/resume behavior

Default `overwrite=False`. If output parquet exists → skipped. Per-file writes; docstring marks Ctrl-C safe. Re-run **same command** after interrupt: completed dates skipped, remainder processed. No `--overwrite` on clean root.

Parallelism: **by year** only; `--workers 10` with all 10 years runs **one worker per year** concurrently (HD locked; 16-core host). C5.10A1 validated 8-worker output identity on 2020; 10 workers same code path.

---

## Path safety

With planned explicit args: **no writes** to `ORATS_Adjusted`, `ORATS_Data`, or `cache/splits_hist.parquet` (raw/splits read-only; adj-root explicit).

**If args omitted:** `--adj-root` → `ORATS_Adjusted`; `--splits` → cache file. Always use full explicit command below.

---

## Exact C5.10B backfill command

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe C:/MomentumCVG/scripts/apply_split_adjustment.py `
  --raw-root C:/ORATS/data/ORATS_Data `
  --adj-root C:/MomentumCVG_env/input/adjusted_liquid `
  --splits C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet `
  --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv `
  --years 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 `
  --workers 10
```

~2,299 ZIPs; est. runtime **~1.5–3 h** (all 10 years parallel; disk-bound; C5.6B ≈15 min/year single-worker).

---

## Logging command

Same as above with transcript capture:

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe C:/MomentumCVG/scripts/apply_split_adjustment.py `
  --raw-root C:/ORATS/data/ORATS_Data `
  --adj-root C:/MomentumCVG_env/input/adjusted_liquid `
  --splits C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet `
  --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv `
  --years 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 `
  --workers 10 `
  2>&1 | Tee-Object -FilePath C:/MomentumCVG/docs/tmp/c5_10b_full_backfill_run_log.txt
```

Record `$LASTEXITCODE` in the log after completion.

---

## Exact post-backfill audit command

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe C:/MomentumCVG/scripts/audit_adjusted_liquid.py `
  --raw-root C:/ORATS/data/ORATS_Data `
  --adj-root C:/MomentumCVG_env/input/adjusted_liquid `
  --splits C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet `
  --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv `
  --years 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 `
  --sample-files 25 `
  --sample-rows 20000 `
  --report-path C:/MomentumCVG/docs/tmp/c5_10b_full_backfill_audit_report.md
```

**Full inventory:** `audit_inventory` compares every raw ZIP date vs adjusted parquet date per year. **Full scans:** universe containment + column structure on **all** parquets. **Sampled only:** raw-vs-adjusted math (`--sample-files 25`, `--sample-rows 20000`, seed 57 default) — broader than C5.8B smoke (10 files).

---

## Expected artifacts

| Artifact | Commit? |
|----------|---------|
| `input/adjusted_liquid/{YYYY}/ORATS_SMV_Strikes_*.parquet` | **No** (production data) |
| `docs/tmp/c5_10b_full_backfill_run_log.txt` | **Yes** (after run) |
| `docs/tmp/c5_10b_full_backfill_audit_report.md` | **Yes** (after audit) |
| Raw ZIPs, cache, adjusted mirror | **No** |

---

## Decisions still needed from HD

1. **Disk space** on `C:` for ~2,299 filtered daily parquets (tens–100+ GB est.).
2. **Runtime window** for multi-hour unattended run (resume-safe if interrupted).
3. **Math sample depth** — raise `--sample-files` to 50 if HD wants stronger spot-check.
4. **Downstream path wiring** — deferred until after audit (C5.10A1).

---

## Final status

**READY TO RUN C5.10B**
