# V1 universe protocol

**Status:** Active  
**Last updated:** 2026-07-04 (C5 adjusted-liquid root referenced)

---

## Goal

Build a **point-in-time tradable universe** each rebalance week from a broad ORATS superset, without lookahead. The universe should favor names that were **recently liquid**, not a static S&P 500 list.

---

## Rule (v1)

At each **weekly rebalance date** `t`:

1. **Load liquidity panel** (`ticker_liquidity_panel.parquet`) built by `scripts/build_liquidity_panel.py`.
2. **Point-in-time lookup:** use the most recent `month_date <= t` for each ticker (`month_date` is the **week-end snapshot date**; legacy column name retained for step1 compatibility).
3. **Eligible pool:** tickers with `has_valid_atm_pair == True` on that snapshot (≥3 of last 12 weeks with at least one valid daily ATM quote week).
4. **Rank** eligible tickers on `atm_straddle_dollar_vol` (12-week rolling average straddle bid×volume).
5. **Select top 20%** of eligible tickers → **tradable universe for week t**.
6. **Signals and ranking** for momentum/CVG run **only within** this tradable universe.

---

## Liquidity scoring (v1)

Panel fields (built by C4 rolling window):

| Field | Role |
|-------|------|
| `atm_straddle_dollar_vol` | Primary rank key — mean weekly straddle $ vol over **12-week lookback** |
| `atm_spread_pct` | Mean spread over weeks with valid quotes (tie-break / filter when `spread_bot_pct < 1`) |
| `has_valid_atm_pair` | Eligibility gate — `valid_quote_weeks >= 3` in lookback |

**Panel build (Sprint 004 C4):** reads **ORATS raw** ZIPs (`ORATS_Data`); no split adjustment in liquidity stage. Default `lookback_weeks=12`, `min_valid_quote_weeks=3`, `dte_min=5`, `dte_max=60`, `dvol_top_pct=0.20`, `spread_bot_pct=1.0`.

**Adjusted chains (Sprint 004 C5):** surface/backtest economics read split-adjusted parquets from `C:/MomentumCVG_env/input/adjusted_liquid` (C4 liquid-universe filter). See [sprint_memos/004_c5_adjusted_liquid.md](sprint_memos/004_c5_adjusted_liquid.md).

**Precompute superset vs trading universe:**

| Layer | Artifact | Purpose |
|-------|----------|---------|
| Precompute superset | `liquid_tickers.csv` | Tickers that ever qualified; `snapshots_qualified` = count of weeks in top-20% bucket |
| Trading universe | Panel row at PIT snapshot | Top 20% of eligible names at rebalance `t` |

Never use `liquid_tickers.csv` alone as the trading universe.

---

## Rebalance linkage

- Rebalance: **weekly** (see [v1_spec_pins.md](v1_spec_pins.md)).
- Universe rebuilt **every rebalance** before signal ranking.
- New entries must pass surface tradability checks (iron fly or iron condor assembly) in addition to universe membership.

---

## Weekly trade workflow (not in this doc)

This document covers **who is eligible** each rebalance, not the full weekly sequence (universe → signal → structure → size → log).

After the **liquidity panel review** (Sprint 002), add [v1_weekly_runbook.md](v1_weekly_runbook.md) with the step-by-step weekly flow. C4 rolling panel builder shipped Sprint 004 — see runbook § Liquidity panel.

---

## Data dependencies

| Artifact | Path (default) | Script |
|----------|----------------|--------|
| Liquidity panel | `C:/MomentumCVG_env/cache/ticker_liquidity_panel.parquet` (or `input/liquidity/`) | `scripts/build_liquidity_panel.py` |
| ORATS raw chains | `C:/ORATS/data/ORATS_Data` | `build_liquidity_panel.py` input |
| Features (momentum/CVG) | `C:/MomentumCVG_env/cache/features_*.parquet` | `scripts/build_features.py` |
| Option surface | `C:/MomentumCVG_env/cache/option_surface/` | `scripts/precompute_option_surface.py` |

---

## Verification requirements (before trusting backtest)

- [ ] Universe at date `t` uses only panel rows with `month_date <= t`
- [ ] Top 20% is computed on cross-section at `t`, not global future data
- [ ] Signal features at `t` use only data available at `t`
- [ ] Integration smoke: same `t` reproduces same universe from saved inputs

---

## References

- `scripts/build_liquidity_panel.py` — panel schema and defaults (`--dvol-top-pct 0.20`)
- `src/backtest/pipeline.py` → `step1_get_universe()`
- Archived: `docs/archive/production_ready_checklist_options_strategy.md` (Section B)
