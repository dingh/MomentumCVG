# V1 weekly input runbook

**Status:** Active (producer / repair notes) — **not** the accepted production handoff
**Last updated:** 2026-08-09 (aligned to Sprint 005 closeout)
**Owner:** Operator + agent

---

## Accepted production handoff (read this first)

The trusted downstream input is the **immutable published snapshot** identified by its manifest, not a mutable global cache root.

| Field | Value |
|-------|-------|
| Snapshot ID | `e2c1f8fd44d72176` |
| Final root | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886` |
| Manifest | `…/manifests/input_snapshot_e2c1f8fd44d72176.json` |
| Closeout | [sprint_memos/004_closeout.md](sprint_memos/004_closeout.md) |

Resolve Stage A artifacts from that manifest (`cache_dir` + relative `artifacts` paths). Mutable roots under `C:/MomentumCVG_env/input/` and `C:/MomentumCVG_env/cache/` may still be used as **producer / legacy working locations**, but they are **not** interchangeable with an accepted production snapshot.

Filter surfaces with `surface_valid=True` before feature or backtest work.

---

## Purpose of this runbook

Historical and operational notes for Stage A **producer scripts** (liquidity, adjusted-liquid repair, standalone audits). It does **not**:

- define the future incremental weekly refresh workflow (deferred)
- claim that `refresh_weekly_inputs.py refresh --as-of …` against mutable cache is the current production procedure
- run backtests, size trades, or evaluate strategy performance

**Companion docs:**

- [sprint_memos/004_closeout.md](sprint_memos/004_closeout.md) — accepted snapshot
- [sprint_memos/005_closeout.md](sprint_memos/005_closeout.md) — accepted features + consumer smoke
- [v1_universe_protocol.md](v1_universe_protocol.md) — PIT trading universe rule
- [agenda/current_sprint.md](agenda/current_sprint.md) — Sprint 005 closed; next sprint not authorized
- [surface_engine_data_contract.md](surface_engine_data_contract.md) — A1–A4 schemas

---

## Prerequisites (producer / repair)

| Requirement | Default path |
|-------------|--------------|
| Python venv | `C:/MomentumCVG_env/venv/` |
| Raw ORATS ZIPs | `C:/ORATS/data/ORATS_Data/` |
| Adjusted chains (producer default) | `C:/MomentumCVG_env/input/adjusted_liquid/` |
| Legacy full-universe mirror | `C:/ORATS/data/ORATS_Adjusted/` (maintenance only) |
| Liquidity panel (producer default) | `C:/MomentumCVG_env/input/liquidity/` |
| Accepted snapshot root | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/` |

```powershell
& C:/MomentumCVG_env/venv/Scripts/Activate.ps1
```

---

## Historical CLI notes (deferred / incomplete)

The following examples describe **Sprint 004-era scaffolding**. Several subcommands remain stubs or incomplete for production weekly operation. Do **not** treat them as the accepted production procedure.

```powershell
# Plan / dry-run shapes exist; full incremental weekly ops are deferred
python scripts/refresh_weekly_inputs.py plan --mode backfill --snapshots-root C:/MomentumCVG_env/snapshots ...

# Standalone adjusted-liquid audit (still useful as a producer check)
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/audit_adjusted_liquid.py `
  --raw-root C:/ORATS/data/ORATS_Data `
  --adj-root C:/MomentumCVG_env/input/adjusted_liquid `
  --splits C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet `
  --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv
```

Cold full-history rebuilds use the snapshot orchestrator (`refresh --mode backfill` / `--resume`) documented in [004_closeout.md](sprint_memos/004_closeout.md). Routine incremental weekly refresh design is **deferred**.

---

## Two universe layers (do not conflate)

| Layer | Artifact | Used for |
|-------|----------|----------|
| **Precompute superset** | `liquid_tickers.csv` | Engineering superset for surface precompute |
| **Trading universe** | S1 at rebalance `t` | Top 20% liquid names from PIT liquidity panel |

Backtests and live decisions use the **trading universe**, not the static CSV alone.

---

## Liquidity panel (producer)

**Script:** `scripts/build_liquidity_panel.py`  
**Input:** ORATS raw ZIPs — `C:/ORATS/data/ORATS_Data`  
**Closeout memo:** [sprint_memos/004_c4_liquidity_panel.md](sprint_memos/004_c4_liquidity_panel.md)

Producer output typically lands under `C:/MomentumCVG_env/input/liquidity/`. For trusted downstream consumption, prefer the published snapshot paths from the manifest.

Defaults: `lookback_weeks=12`, `min_valid_quote_weeks=3`, `spread_bot_pct=1.0`.

---

## Adjusted-liquid repair (producer)

**Closeout memo:** [sprint_memos/004_c5_adjusted_liquid.md](sprint_memos/004_c5_adjusted_liquid.md)

Producer root: `C:/MomentumCVG_env/input/adjusted_liquid`.

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/apply_split_adjustment.py `
  --tickers NVDA TSLA `
  --adj-root C:/MomentumCVG_env/input/adjusted_liquid `
  --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv `
  --splits C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet `
  --overwrite
```

Re-run `audit_adjusted_liquid.py` after any adj-root rewrite. A repair against the mutable producer root does **not** automatically update a published snapshot.

---

## Validation checklist (before trusting inputs)

Prefer the published snapshot + manifest over a mutable cache:

- [ ] Manifest `snapshot_id` matches the intended handoff (`e2c1f8fd44d72176` for current production)
- [ ] `production_accepted=true` and `params.scope=full` for production use
- [ ] Surface consumers filter `surface_valid=True`
- [ ] PIT universe uses one global snapshot with `month_date < t` (same snapshot for all tickers; never `<= t`)
- [ ] No A4/feature validation expected until Sprint 005 scope is accepted

---

## Failure modes

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Split adjustment errors | Missing splits for ticker | Re-fetch / repair scoped tickers; re-audit |
| Empty liquidity panel rows | ORATS gap on scan date | Check ORATS coverage |
| Straddle history / features | Sprint 005 not accepted yet | Hold until scope review completes |
| Treating mutable cache as production | Wrong handoff | Use published snapshot manifest |

---

## Change log

| Date | Change |
|------|--------|
| 2026-06-20 | Draft scaffold for Sprint 004; CLI TBD |
| 2026-06-21 | Features removed (→ 005); surface-audit added; HD-004-2 as-of rule |
| 2026-06-29 | C4 liquidity panel commands, paths, failure modes |
| 2026-07-04 | C5 adjusted-liquid paths, pipeline order, audit commands |
| 2026-07-26 | Snapshot-first handoff; obsolete weekly CLI claims marked historical/deferred |
