# C5.10A3 — Full Backfill Go/No-Go Run Checklist

**Date:** 2026-07-04 · **Repo SHA:** `0c460dfc52d5aa19810241a24cc6b2c04a2660fa`  
**Sources:** [c5_10a1](c5_10a1_backfill_artifact_inventory.md) · [c5_10a2](c5_10a2_backfill_script_command_plan.md)

Pre-run checklist only — no backfill, audit, or production data touched.

---

## 1. Inputs confirmed

| Artifact | A1/A2 status |
|----------|--------------|
| `C:/ORATS/data/ORATS_Data` | yes — ~2,299 ZIPs for 2017–2026 |
| `C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv` | yes — 2,783 tickers |
| `C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet` | yes — 1,347 rows, 819 split tickers |
| `C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.checkpoint.parquet` | yes — C5.7 fetch sidecar |

**Checkpoint sidecar:** preserved at adj-root; **not read** by `apply_split_adjustment.py` (uses `splits_hist_liquid.parquet` only).

---

## 2. Output target

```text
C:/MomentumCVG_env/input/adjusted_liquid/{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.parquet
```

Per C5.10A1: root has split artifacts only — **no `{YYYY}/` folders, no chain parquets yet**. Clean first run; skip-existing applies.

---

## 3. Exact run sequence

1. **Confirm disk space and runtime window** (~1.5–3 h; resume-safe if interrupted)
2. **Run full backfill** with `Tee-Object` log capture
3. **Record `$LASTEXITCODE`** after backfill
4. **Inspect log** for errors / unexpected skips
5. **Run full production audit**
6. **Commit only** log + audit markdown (not parquets)
7. **Review audit report** before downstream path wiring

---

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

No `--overwrite`. After run: `"exit_code=$LASTEXITCODE" | Add-Content C:/MomentumCVG/docs/tmp/c5_10b_full_backfill_run_log.txt`

---

## 5. Exact audit command

**Default (recommended):**

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe C:/MomentumCVG/scripts/audit_adjusted_liquid.py `
  --raw-root C:/ORATS/data/ORATS_Data `
  --adj-root C:/MomentumCVG_env/input/adjusted_liquid `
  --splits C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet `
  --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv `
  --years 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 `
  --sample-files 25 `
  --sample-rows 20000 `
  --seed 57 `
  --report-path C:/MomentumCVG/docs/tmp/c5_10b_full_backfill_audit_report.md
```

Full inventory + structure scans cover **all** parquets; math is sampled only. Use **25 files** unless HD wants stronger sampling.

**Optional stronger variant:** same command with `--sample-files 50 --sample-rows 20000 --seed 57`.

---

## 6. Stop conditions

Stop and investigate before downstream use if any occur:

- Backfill exits nonzero
- Log reports file-level errors
- Unexpected writes outside `C:/MomentumCVG_env/input/adjusted_liquid`
- Audit exits nonzero
- Audit reports missing adjusted files
- Audit reports outside-universe tickers
- Audit reports structural failures
- Audit reports raw-vs-adjusted math mismatches

---

## 7. What to commit after C5.10B

**Commit:** `docs/tmp/c5_10b_full_backfill_run_log.txt`, `docs/tmp/c5_10b_full_backfill_audit_report.md`

**Do not commit:** `input/adjusted_liquid/{YYYY}/…`, raw ORATS ZIPs, cache files, legacy `ORATS_Adjusted` mirror

---

## 8. HD final confirmation needed

HD confirmation before running C5.10B:

- [ ] C: drive has enough free space
- [ ] Multi-hour run window is available
- [ ] Use default audit depth: sample-files=25, sample-rows=20000
- [ ] Output root confirmed: `C:/MomentumCVG_env/input/adjusted_liquid`
- [ ] Proceed with no `--overwrite`

---

## Final status

**READY FOR HD GO/NO-GO**
