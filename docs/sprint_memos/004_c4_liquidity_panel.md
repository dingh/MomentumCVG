# Sprint 004 — C4 liquidity panel (closeout)

**Status:** Closed — **C4 accepted**  
**Closed:** 2026-06-29  
**Design:** [docs/tmp/c4_liquidity_panel_design_plan.md](../tmp/c4_liquidity_panel_design_plan.md)

---

## Deliverable

Rolling **12-week** point-in-time liquidity panel from **ORATS raw ZIPs** (`ORATS_Data`), written by `scripts/build_liquidity_panel.py`:

| Artifact | Description |
|----------|-------------|
| `ticker_liquidity_daily_observations.parquet` | Slim daily liquidity rows |
| `ticker_liquidity_weekly_observations.parquet` | Weekly roll-ups |
| `ticker_liquidity_panel.parquet` | Rolling panel (`month_date` = week-end snapshot) |
| `liquid_tickers.csv` | Historical superset (`snapshots_qualified` = weeks in top-20% dvol bucket) |
| `manifests/reports/liquidity_panel_{build_id}.md` | PASS/WARN build report |

**Modes:** `backfill` (window rebuild) and `incremental` (one new completed ORATS week per run).

**Production cache (operator):** `C:/MomentumCVG_env/input/liquidity/` — full backfill 2017-01-01 → 2026-02-20 completed 2026-06-28.

---

## Verification summary

| Check | Result |
|-------|--------|
| Unit tests (`test_build_liquidity_panel.py`) | **30 passed** |
| Q1 2024 smoke backfill | **PASS** (~44 min, 114 ZIPs) |
| Output sanity (liquid names, window_shortfall) | **PASS** |
| Incremental one-week append | **PASS** (after date-type fix) |
| Incremental ≡ full backfill (same final snapshot) | **PASS** |
| Incremental no new data (same `--end-date`) | **PASS** — exit 1, artifacts unchanged |
| Incremental no ORATS week after watermark | **PASS** — exit 1, artifacts unchanged |
| Full-history backfill 2017→2026-02-20 | **PASS** — 477 snapshots, 2,783 liquid names |
| Weekly universe size (top 20%) | **506–959** / week (median ~631) — in expected band |

**Smoke artifacts:** under `C:/MomentumCVG_env/cache_c4_smoke/` (not in repo).

**Phase 3 consolidated smoke doc:** not completed (stopped by operator); core failure modes covered ad hoc.

---

## Code changes at closeout

1. **Incremental date guard** — compare `datetime.date` consistently in `run_incremental` (fixes `TypeError` on watermark check).
2. **Progress bar** — `tqdm` on daily ZIP loop; `--no-progress` to disable.
3. **Tests** — regression test for incremental append with `date`-typed `trade_date`.

---

## Known limitations (accepted)

| Item | Notes |
|------|-------|
| Partial ORATS weeks | `week_end_date` = last available ORATS day in ISO week; incremental may append before calendar Friday |
| `spread_bot_pct = 1.0` default | Spread filter effectively off; universe = top 20% dvol among valid-quote names |
| Raw ORATS only | Liquidity ranking uses raw bid/ask/volume; surface/backtest still uses adjusted chains |
| `liquid_tickers.csv` | Historical superset, not PIT universe; use panel + snapshot date for S1 |
| `has_valid_atm_pair` regime shift | ~73% → ~50% valid rate late 2022→2023 (vol crush + `min_valid_quote_weeks=3`); investigated — not a pipeline bug |
| Runtime | ~15–20 h full backfill 2017→2026 on single thread; use incremental weekly (~2 min) thereafter |

---

## Operator commands

```powershell
# First-time or gap repair (long)
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/build_liquidity_panel.py `
  --data-root C:/ORATS/data/ORATS_Data `
  --cache-dir C:/MomentumCVG_env/input/liquidity `
  --mode backfill `
  --start-date 2017-01-01 `
  --end-date 2026-02-20 `
  --build-id liquidity_backfill_2017_20260220

# Weekly after new ORATS week (one week only; --end-date = that week's last ORATS day)
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/build_liquidity_panel.py `
  --data-root C:/ORATS/data/ORATS_Data `
  --cache-dir C:/MomentumCVG_env/input/liquidity `
  --mode incremental `
  --start-date 2017-01-01 `
  --end-date <week_end_date> `
  --build-id liquidity_incremental_<week_end_date>
```

Do **not** pass `--spread-bot-pct` unless intentionally changing universe params (requires backfill).

---

## Remaining before Sprint 004 full closeout

C4 only. Sprint 004 still open for C5–C9 (split audit, surface audit, PIT harness, `validate`, runbook finalization, CLI plan cleanup).

---

## References

- Investigation (valid ATM drop 2022→2023): `C:/MomentumCVG_env/cache_c4_smoke/investigation_valid_atm_drop/`
- Panel ticker-count chart: `C:/MomentumCVG_env/cache_c4_smoke/panel_weekly_ticker_counts.html`
