# V1 universe protocol

**Status:** Active
**Last updated:** 2026-07-26 (PIT snapshot semantics aligned to Sprint 004 C7)

Accepted production input snapshot: **`e2c1f8fd44d72176`**
(`C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886`; see [sprint_memos/004_closeout.md](sprint_memos/004_closeout.md)).

Canonical PIT closeout: [sprint_memos/004_c7_pit_universe.md](sprint_memos/004_c7_pit_universe.md).

---

## Goal

Build a **point-in-time tradable universe** each rebalance week from a broad ORATS superset, without lookahead. The universe should favor names that were **recently liquid**, not a static S&P 500 list.

---

## Rule (v1)

At each **weekly rebalance / trade date** `t`:

1. **Load liquidity panel** (`ticker_liquidity_panel.parquet`) built by `scripts/build_liquidity_panel.py`.
2. **Select one global snapshot strictly before `t`:**

   ```text
   snapshot_date = max(month_date where month_date < t)
   ```

   (`month_date` is the week-end snapshot date; legacy column name retained for step1 compatibility.)
   Same-day and future snapshots are prohibited. If no prior snapshot exists, the universe is empty.
3. **Read ticker membership from that snapshot only** — every ticker evaluated on trade date `t` uses the same `snapshot_date`. Do **not** resolve a different snapshot independently per ticker.
4. **Eligible pool:** tickers with `has_valid_atm_pair == True` on that snapshot (≥3 of last 12 weeks with at least one valid daily ATM quote week).
5. **Rank** eligible tickers on `atm_straddle_dollar_vol` (12-week rolling average straddle bid×volume) on that cross-section.
6. **Select top 20%** of eligible tickers → **tradable universe for week t**.
7. **Signals and ranking** for momentum/CVG run **only within** this tradable universe.

---

## Liquidity scoring (v1)

Panel fields (built by C4 rolling window):

| Field | Role |
|-------|------|
| `atm_straddle_dollar_vol` | Primary rank key — mean weekly straddle $ vol over **12-week lookback** |
| `atm_spread_pct` | Mean spread over weeks with valid quotes (tie-break / filter when `spread_bot_pct < 1`) |
| `has_valid_atm_pair` | Eligibility gate — `valid_quote_weeks >= 3` in lookback |

**Panel build (Sprint 004 C4):** reads **ORATS raw** ZIPs (`ORATS_Data`); no split adjustment in liquidity stage. Default `lookback_weeks=12`, `min_valid_quote_weeks=3`, `dte_min=5`, `dte_max=60`, `dvol_top_pct=0.20`, `spread_bot_pct=1.0`.

**Adjusted chains (Sprint 004 C5):** surface/backtest economics read split-adjusted parquets from the **published snapshot** (manifest-resolved paths under `…/snapshots/20260724T045049097520Z_40b16886`). The mutable producer root `C:/MomentumCVG_env/input/adjusted_liquid` is for rebuild/repair only. See [sprint_memos/004_closeout.md](sprint_memos/004_closeout.md) and [sprint_memos/004_c5_adjusted_liquid.md](sprint_memos/004_c5_adjusted_liquid.md).

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

Producer and repair notes: [v1_weekly_runbook.md](v1_weekly_runbook.md). Accepted production handoff remains the published snapshot in [sprint_memos/004_closeout.md](sprint_memos/004_closeout.md). Incremental weekly refresh design is deferred.

---

## Data dependencies

| Artifact | Trusted path | Notes |
|----------|--------------|-------|
| Accepted snapshot | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886` | Resolve liquidity, adjusted chains, spot, surface via manifest `e2c1f8fd44d72176` |
| ORATS raw chains | `C:/ORATS/data/ORATS_Data` | Producer input for liquidity panel rebuilds |
| Producer liquidity / adjusted / cache | `C:/MomentumCVG_env/input/…`, `C:/MomentumCVG_env/cache/…` | Mutable working locations — not interchangeable with a published snapshot |
| Features (momentum/CVG) | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features/` | Sprint 005 closed; see [005_closeout.md](sprint_memos/005_closeout.md) |

---

## Verification requirements (before trusting backtest)

- [ ] Universe at date `t` uses one global snapshot with `month_date < t` (never `<= t`, never per-ticker)
- [ ] Top 20% is computed on that prior snapshot's cross-section, not future panel rows
- [ ] Signal features at `t` use only data available at `t`
- [ ] Integration smoke: same `t` reproduces same universe from saved inputs

---

## References

- `scripts/build_liquidity_panel.py` — panel schema and defaults (`--dvol-top-pct 0.20`)
- `src/backtest/pipeline.py` → `step1_get_universe()`
- Archived: `docs/archive/production_ready_checklist_options_strategy.md` (Section B)
