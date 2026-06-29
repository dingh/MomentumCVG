# V1 weekly input runbook

**Status:** Active (liquidity panel section updated for C4)  
**Last updated:** 2026-06-29  
**Owner:** Operator + agent (CLI maintenance)

---

## Purpose

Step-by-step procedure to refresh **Stage A inputs** for Sprint 004 scope: split-adjusted chains, spot DB, liquidity panel, option surface (A1/A2). **Feature branch (straddle history, momentum/CVG) is Sprint 005** — not in this runbook until then.

This runbook covers **data refresh and validation only**. It does not run backtests, size trades, or evaluate strategy performance.

**Companion docs:**

- [v1_universe_protocol.md](v1_universe_protocol.md) — PIT trading universe rule
- [agenda/current_sprint.md](agenda/current_sprint.md) — Sprint 004 build scope
- [surface_engine_data_contract.md](surface_engine_data_contract.md) — A1–A4 schemas

---

## Sprint 004 vs 005

| Concern | Sprint 004 (this runbook) | Sprint 005 |
|---------|---------------------------|------------|
| Splits, spot, liquidity, option surface | Yes | — |
| Straddle history, features, mom/CVG | **No** | All feature pipeline |
| `surface-audit` / `split-audit` | Yes | — |

---

## Prerequisites

| Requirement | Default path |
|-------------|--------------|
| Python venv | `C:/MomentumCVG_env/venv/` |
| Raw ORATS ZIPs | `C:/ORATS/data/ORATS_Data/` |
| Adjusted chains | `C:/ORATS/data/ORATS_Adjusted/` |
| Cache / artifacts | `C:/MomentumCVG_env/cache/` (default) or `C:/MomentumCVG_env/input/liquidity/` (production panel) |
| Liquidity panel | `ticker_liquidity_panel.parquet` under cache dir above |
| Splits history | `C:/MomentumCVG_env/cache/splits_hist.parquet` |
| Trade universe (precompute) | `liquid_tickers.csv` under same cache dir |

Activate venv:

```powershell
& C:/MomentumCVG_env/Scripts/Activate.ps1
```

---

## Quick reference (Sprint 004 target CLI)

```powershell
# Show planned steps without executing
python scripts/refresh_weekly_inputs.py plan --dry-run

# Validate existing cache (inventory + PIT + split summary)
python scripts/refresh_weekly_inputs.py split-audit --as-of 2026-06-26
python scripts/refresh_weekly_inputs.py surface-audit --as-of 2026-06-26

# Full refresh (core + surface; no feature branch)
python scripts/refresh_weekly_inputs.py refresh --as-of 2026-06-26

# Incremental
python scripts/refresh_weekly_inputs.py refresh --skip-surface
```

> **Note:** CLI paths and flags are defined during Sprint 004 build. Update this section when implementation lands.

---

## Pipeline order (end-to-end)

```text
1. fetch_splits.py
2. apply_split_adjustment.py
3. extract_spot_prices.py
4. build_liquidity_panel.py         → rolling 12-week panel (backfill or incremental)
5. precompute_option_surface.py     → optional on refresh; required for surface-audit if cache stale
6. validate / split-audit / surface-audit
```

**Not in Sprint 004 pipeline:** `precompute_straddle_history.py`, `build_features.py` → Sprint 005.

`--as-of` resolves to **last trading day ≤ date** (HD-004-2).

---

## Two universe layers (do not conflate)

| Layer | Artifact | Used for |
|-------|----------|----------|
| **Precompute superset** | `liquid_tickers.csv` | Engineering superset for surface precompute (Sprint 004) |
| **Trading universe** | S1 at rebalance `t` | Top 20% liquid names from PIT liquidity panel |

Backtests and live decisions use the **trading universe**, not the static CSV alone.

---

## Liquidity panel (`build_liquidity_panel.py`)

**Script:** `scripts/build_liquidity_panel.py`  
**Input:** ORATS raw ZIPs — `C:/ORATS/data/ORATS_Data`  
**Closeout memo:** [sprint_memos/004_c4_liquidity_panel.md](sprint_memos/004_c4_liquidity_panel.md)

### Modes

| Mode | When | Behavior |
|------|------|----------|
| `backfill` | First build, gap, param change | Rebuild daily → weekly → panel for `--start-date` … `--end-date` |
| `incremental` | Routine weekly | Append **one** new ORATS week; fails if 0 or >1 new weeks |

### Commands

```powershell
# Full window rebuild (long — hours for multi-year)
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/build_liquidity_panel.py `
  --data-root C:/ORATS/data/ORATS_Data `
  --cache-dir C:/MomentumCVG_env/input/liquidity `
  --mode backfill `
  --start-date 2017-01-01 `
  --end-date 2026-02-20 `
  --build-id liquidity_backfill_<tag>

# Weekly incremental (--end-date = last ORATS day of new week)
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/build_liquidity_panel.py `
  --data-root C:/ORATS/data/ORATS_Data `
  --cache-dir C:/MomentumCVG_env/input/liquidity `
  --mode incremental `
  --start-date 2017-01-01 `
  --end-date <week_end_date> `
  --build-id liquidity_incremental_<week_end_date>
```

Defaults: `lookback_weeks=12`, `min_valid_quote_weeks=3`, `spread_bot_pct=1.0` (do not override without backfill).

Progress bar on daily ZIP processing; `--no-progress` to disable.

### Failure modes

| Symptom | Action |
|---------|--------|
| `Nothing to append` | No ORATS week after panel watermark — wait for data or check ZIPs |
| `Incremental expects one new week; found N` | Run incremental once per week, or backfill the gap |
| `Build params mismatch` | CLI flags differ from panel stamp — backfill with matching params |

---

## Validation checklist (operator)

After refresh or before trusting cache for Sprint 006 smoke (after Sprint 005 feature audit):

- [ ] Manifest exists with `snapshot_id` and all executed step exit codes = 0
- [ ] Validation report: no missing required artifacts
- [ ] Liquidity panel date range covers intended backtest window
- [ ] Split scan: no critical anomalies (see report severity levels)
- [ ] PIT spot-check: same `trade_date` reproduces same S1 universe (validation harness)
- [ ] Split + surface audit reports archived (≥1 substantive run each)
- [ ] No A4/feature validation expected in 004

---

## Failure modes

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Split adjustment errors | Missing splits for ticker | Run `fetch_splits.py`; re-run adjustment for ticker |
| Empty liquidity panel rows | ORATS gap on scan date | Check ORATS coverage; narrow date range |
| Straddle history / features | Sprint 005 scope | See Sprint 005 runbook when available |
| Surface audit FAIL on valid rows missing exit_spot | Precompute gap | Re-run surface sample; check spot DB |

---

## Manifest location

_Default (Sprint 004):_

```text
C:/MomentumCVG_env/cache/manifests/input_snapshot_<snapshot_id>.json
C:/MomentumCVG_env/cache/manifests/reports/validate_<build_id>.md
C:/MomentumCVG_env/cache/manifests/reports/split_audit_<build_id>.md
C:/MomentumCVG_env/cache/manifests/reports/surface_audit_<build_id>.md
```

See [agenda/current_sprint.md](agenda/current_sprint.md) § Default report and manifest paths for full spec.

---

## Change log

| Date | Change |
|------|--------|
| 2026-06-20 | Draft scaffold for Sprint 004; CLI TBD |
| 2026-06-21 | Features removed (→ 005); surface-audit added; HD-004-2 as-of rule |
| 2026-06-29 | C4 liquidity panel commands, paths, failure modes |
