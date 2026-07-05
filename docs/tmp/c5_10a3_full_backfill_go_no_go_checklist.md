# C5.10A3 — Full Backfill Go/No-Go Run Checklist

**Date:** 2026-07-04 · **SHA:** `0c460df` · **Sources:** [A1](c5_10a1_backfill_artifact_inventory.md) · [A2](c5_10a2_backfill_script_command_plan.md)

Pre-run only — no backfill, audit, or production data touched.

## 1. Inputs confirmed

| Path | A1/A2 |
|------|-------|
| `C:/ORATS/data/ORATS_Data` | yes (~2,299 ZIPs, 2017–2026) |
| `C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv` | yes (2,783 tickers) |
| `C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet` | yes (1,347 rows) |
| `C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.checkpoint.parquet` | yes (sidecar) |

Checkpoint sidecar **preserved, not used** by adjustment (`--splits` → `.parquet` only).

## 2. Output target

`C:/MomentumCVG_env/input/adjusted_liquid/{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.parquet`

C5.10A1: split artifacts at root only — **no year folders / chain parquets yet**.

## 3. Exact run sequence

Step 1 — confirm disk space and runtime window · Step 2 — run full backfill with Tee-Object · Step 3 — record `$LASTEXITCODE` · Step 4 — inspect log · Step 5 — run audit · Step 6 — commit log/report only · Step 7 — review audit before path wiring

## 4. Exact backfill command

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

No `--overwrite`. Append: `"exit_code=$LASTEXITCODE" | Add-Content C:/MomentumCVG/docs/tmp/c5_10b_full_backfill_run_log.txt`

## 5. Exact audit command

**Default:** `--sample-files 25 --sample-rows 20000 --seed 57` (all-file inventory/structure; math sampled). Optional: `--sample-files 50`.

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe C:/MomentumCVG/scripts/audit_adjusted_liquid.py `
  --raw-root C:/ORATS/data/ORATS_Data --adj-root C:/MomentumCVG_env/input/adjusted_liquid `
  --splits C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet `
  --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv `
  --years 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 `
  --sample-files 25 --sample-rows 20000 --seed 57 `
  --report-path C:/MomentumCVG/docs/tmp/c5_10b_full_backfill_audit_report.md
```

## 6. Stop conditions

Backfill nonzero exit · log errors · writes outside adj-root · audit nonzero · missing files · outside-universe tickers · structural failures · math mismatches

## 7. What to commit after C5.10B

**Yes:** `docs/tmp/c5_10b_full_backfill_run_log.txt`, `docs/tmp/c5_10b_full_backfill_audit_report.md`  
**No:** `input/adjusted_liquid/{YYYY}/…`, raw ZIPs, cache, legacy `ORATS_Adjusted`

## 8. HD final confirmation needed

HD confirmation before running C5.10B:

- [ ] C: drive has enough free space
- [ ] Multi-hour run window is available
- [ ] Use default audit depth: sample-files=25, sample-rows=20000
- [ ] Output root confirmed: `C:/MomentumCVG_env/input/adjusted_liquid`
- [ ] Proceed with no `--overwrite`

## Final status

**READY FOR HD GO/NO-GO**
