# C5.10A1 — Backfill Artifact Inventory and Flow Map

**Date:** 2026-07-04  
**Task:** Sprint 004 C5.10A1 (read-only inventory)  
**Repo SHA:** `03abf500e924d7455f875e56a1d79b07331ca7a9`

No backfill, audit run, strategy code, or production data changes were performed for this report.

---

## Raw ORATS inventory summary

**Root:** `C:/ORATS/data/ORATS_Data`  
**Layout:** `{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.zip`  
**Total ZIPs:** 3,055 across 13 year folders (2014–2026)

| Year | ZIP count | First date | Last date | Notes |
|------|-----------|------------|-----------|-------|
| 2014 | 252 | 20140102 | 20141231 | ok |
| 2015 | 252 | 20150102 | 20151231 | ok |
| 2016 | 252 | 20160104 | 20161230 | ok |
| 2017 | 251 | 20170103 | 20171229 | ok |
| 2018 | 252 | 20180102 | 20181231 | ok |
| 2019 | 252 | 20190102 | 20191231 | ok |
| 2020 | 253 | 20200102 | 20201231 | ok |
| 2021 | 252 | 20210104 | 20211231 | ok |
| 2022 | 251 | 20220103 | 20221230 | ok |
| 2023 | 250 | 20230103 | 20231229 | ok |
| 2024 | 254 | 20240102 | 20241231 | ok |
| 2025 | 250 | 20250102 | 20251231 | ok |
| 2026 | 34 | 20260102 | 20260220 | partial year (YTD) |

**Filename anomalies:** none (all match `ORATS_SMV_Strikes_YYYYMMDD.zip`). Legacy mirror `C:/ORATS/data/ORATS_Adjusted`: 13 year folders, 3,046 full-universe parquets — not the C5 target root.

---

## Existing adjusted-liquid root summary

**Root:** `C:/MomentumCVG_env/input/adjusted_liquid`

| Check | Result |
|-------|--------|
| `splits_hist_liquid.parquet` | **exists** (15,474 bytes) |
| `splits_hist_liquid.checkpoint.parquet` | exists (C5.7 fetch sidecar; same size) |
| `{YYYY}/` output folders | **none** |
| `ORATS_SMV_Strikes_*.parquet` files | **none** |

**Production-root readiness:** **clean.** Only C5.7 scoped split inputs are present; no prior adjusted chain output to skip or collide with. Safe to use as the production adjusted-liquid output root for a first full backfill (skip-existing will write all dates).

---

## Required input artifacts

| Path | Exists | Basic stats |
|------|--------|-------------|
| `C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv` | yes | 2,783 rows / 2,783 unique tickers; columns `Ticker`, `snapshots_qualified`, `months_qualified` |
| `C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet` | yes | 1,347 rows; 819 tickers with splits; columns `ticker`, `split_date`, `divisor`; dates 2007-01-03 → 2026-07-02 |

Both C5 prerequisites are on disk. Split coverage (819 / 2,783 tickers) is expected; unmatched liquid tickers receive `split_factor = 1.0`.

---

## Proposed artifact flow

```text
Raw ORATS ZIPs  (C:/ORATS/data/ORATS_Data/{YYYY}/)
  + liquid_tickers.csv  (C4 storage universe filter)
  + splits_hist_liquid.parquet  (C5.7 scoped splits)
    -> scripts/apply_split_adjustment.py
         --raw-root C:/ORATS/data/ORATS_Data
         --adj-root C:/MomentumCVG_env/input/adjusted_liquid
         --splits   C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet
         --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv
         [--years YYYY ...]  [--workers N]  [--overwrite optional]
      -> adjusted_liquid/{YYYY}/ORATS_SMV_Strikes_YYYYMMDD.parquet
        -> scripts/audit_adjusted_liquid.py
             -> audit markdown report (PASS/WARN/FAIL)
               -> ORATSDataProvider smoke (C5.9 pattern; explicit --data-root)
                 -> downstream Stage A (spot / surface) — path wiring deferred to C5.10 ops
```

**Notes:** `build_liquidity_panel.py` stays on raw ZIPs. Prior smokes under `cache_c4_smoke/` (2020 + C5.8B) — not production root. Legacy consumer default remains `ORATS_Adjusted` until repointed (C5.9 WARN).

---

## Open questions for HD

1. **Backfill year window:** full raw span 2014–2026 (3,055 ZIPs) vs start at 2017 (C4 liquidity panel era)?
2. **Worker count / runtime:** C5.6B used `--workers 1` for 253 ZIPs (~15 min); confirm parallelism and disk budget for ~3k files.
3. **First-run flags:** root is empty of chain parquets — default skip-existing is fine; confirm no `--overwrite` unless repair.
4. **Post-backfill audit scope:** sample years (e.g. 2020 + 2024) vs full inventory pass via `audit_adjusted_liquid.py`?
5. **Downstream path wiring** and **checkpoint sidecar** handling before/at backfill start?

---

## Final status

**READY FOR COMMAND PLANNING**

Inputs exist, raw ORATS is complete through 2025 plus partial 2026, and the production root has no chain output. Next: bounded command plan (year batching, workers, audit window).
