# C5.10A1 — Backfill Artifact Inventory and Flow Map

**Date:** 2026-07-04 (updated with HD decisions + worker check)  
**Task:** Sprint 004 C5.10A1  
**Repo SHA:** `e1725e540f0028531ed552463faa942aadcace8d`

Initial inventory was read-only. Follow-up worker smoke (2020, `--workers 8`) ran under `cache_c4_smoke/` only; smoke output dirs since removed.

---

## Raw ORATS inventory summary

**Root:** `C:/ORATS/data/ORATS_Data` · **Layout:** `{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.zip`

| Year | ZIP count | First date | Last date | Notes |
|------|-----------|------------|-----------|-------|
| 2014–2016 | 252 each | — | — | out of backfill scope |
| 2017 | 251 | 20170103 | 20171229 | **backfill start** |
| 2018–2019 | 252 each | — | — | ok |
| 2020 | 253 | 20200102 | 20201231 | ok |
| 2021 | 252 | 20210104 | 20211231 | ok |
| 2022 | 251 | 20220103 | 20221230 | ok |
| 2023 | 250 | 20230103 | 20231229 | ok |
| 2024 | 254 | 20240102 | 20241231 | ok |
| 2025 | 250 | 20250102 | 20251231 | ok |
| 2026 | 34 | 20260102 | 20260220 | partial YTD |

**Backfill scope (HD locked):** 2017 → latest 2026 = **~2,299 ZIPs**. Filename anomalies: none. Legacy mirror `ORATS_Adjusted` (3,046 full-universe parquets) is not the C5 target root.

---

## Existing adjusted-liquid root summary

**Root:** `C:/MomentumCVG_env/input/adjusted_liquid`

| Check | Result |
|-------|--------|
| `splits_hist_liquid.parquet` | exists |
| `splits_hist_liquid.checkpoint.parquet` | exists (fetch sidecar; harmless; leave in place) |
| `{YYYY}/` output folders | **none** |
| Adjusted chain parquets | **none** |

**Production-root readiness:** **clean** — first backfill uses default skip-existing (no `--overwrite`).

---

## Required input artifacts

| Path | Exists | Basic stats |
|------|--------|-------------|
| `.../liquidity/liquid_tickers.csv` | yes | 2,783 tickers |
| `.../adjusted_liquid/splits_hist_liquid.parquet` | yes | 1,347 rows; 819 tickers with splits; 2007-01-03 → 2026-07-02 |

---

## Worker parallelism check (C5.10A1 follow-up)

**Test:** 2020 only, same inputs as C5.6B, `--workers 8` vs baseline `--workers 1` (`cache_c4_smoke/adjusted_liquid_smoke/2020`).

| Check | Result |
|-------|--------|
| Code works | **PASS** — 253/253 files, 0 errors |
| Identical output | **PASS** — 253/253 byte-for-byte / `DataFrame.equals` match |
| Speed on single year | **No gain** — ~15 min both runs |

**Why:** `SplitAdjuster.run()` parallelizes **by year**, not by ZIP within a year. One year → one worker regardless of `--workers 8`.

**Backfill speed:** run **all years in one command** with `--workers 10` (one worker per year on 16-core host). Do not run year-by-year if speed matters.

---

## HD decisions (locked)

| Topic | Decision |
|-------|----------|
| Year window | **2017 → latest 2026** |
| Workers | **`--workers 10`**, all target years in one invocation (16-core host) |
| Overwrite | **No** — skip-existing on clean production root |
| Post-backfill audit | **Full inventory pass** via `audit_adjusted_liquid.py` |
| Downstream path wiring | **Done (C5.11A)** — see [c5_11a_downstream_path_wiring_report.md](c5_11a_downstream_path_wiring_report.md) |
| Checkpoint sidecar | **Leave in place** — not read by adjustment |

---

## Proposed artifact flow

```text
Raw ORATS ZIPs 2017–2026
  + liquid_tickers.csv + splits_hist_liquid.parquet
    -> apply_split_adjustment.py  (--workers 10; no --overwrite)
      -> input/adjusted_liquid/{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.parquet
        -> audit_adjusted_liquid.py  (full inventory pass)
          -> ORATSDataProvider smoke (explicit --data-root)
            -> Stage A spot/surface (path wiring when ready)
```

`build_liquidity_panel.py` stays on raw `ORATS_Data`.

---

## Final status

**READY FOR COMMAND PLANNING**

All prerequisites on disk; worker check passed; HD scope locked. Next: C5.10A2 command plan (exact PowerShell, runtime estimate, post-backfill audit command).
