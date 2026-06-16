# V1 universe protocol

**Status:** Active (spec only — not yet fully wired in all backtest paths)  
**Last updated:** 2026-06-11 (rolling-window liquidity tentative direction)

---

## Goal

Build a **point-in-time tradable universe** each rebalance week from a broad ORATS superset, without lookahead. The universe should favor names that were **recently liquid**, not a static S&P 500 list.

---

## Rule (v1)

At each **weekly rebalance date** `t`:

1. **Load liquidity panel** (`ticker_liquidity_panel.parquet`) built by `scripts/build_liquidity_panel.py`.
2. **Point-in-time lookup:** use the most recent `month_date <= t` for each ticker (same contract as `step1_get_universe()` in `src/backtest/pipeline.py`).
3. **Eligible pool:** all tickers with valid liquidity fields on that lookup date.
4. **Rank** tickers cross-sectionally on liquidity (see scoring below).
5. **Select top 20%** of eligible tickers → **tradable universe for week t**.
6. **Signals and ranking** for momentum/CVG run **only within** this tradable universe.

---

## Liquidity scoring (v1)

Use fields already consumed by pipeline step1:

| Field | Role |
|-------|------|
| `atm_straddle_dollar_vol` | Primary rank key (higher = more liquid) |
| `atm_spread_pct` | Tie-break / filter (lower = tighter) |

**v1 composite (to align with existing step1):**

- Rank primarily by `atm_straddle_dollar_vol` descending.
- Among ties or borderline names, prefer lower `atm_spread_pct`.
- Exact tie-break rules should match `step1_get_universe()` implementation during Sprint 002 wiring.

The panel is built from **monthly** snapshots; step1 already performs PIT `month_date` lookup. The intent is that eligibility reflects **recent** liquidity history, not a single stale month.

**Tentative direction (not yet wired):** smooth liquidity over a **rolling ~3-month window** — i.e. rank on a rolling-window average of `atm_straddle_dollar_vol` and `atm_spread_pct` (same column names; change is in the panel build script). Window length (`X` months) is configurable and not finalized. This supersedes the earlier trailing-4-week framing and aligns with the HD decision in [surface_engine_data_contract.md](surface_engine_data_contract.md) (fix before the end-to-end correctness check; not blocking earlier contracts). See also [v1_spec_pins.md](v1_spec_pins.md).

Implementation options considered:

- **Option A (minimal, current code):** use step1 as-is with `--dvol-top-pct 0.20` at each rebalance (monthly panel, PIT lookup). Still the baseline until the rolling window lands.
- **Option B (rolling window, tentative target):** rank on a rolling `X`-month average (≈3 months) of the liquidity metrics ≤ t, then take top 20%. Smooths month-boundary churn.

Until the rolling-window build ships, runs use **Option A**; treat Option B as the intended direction, not a locked pin.

---

## Precompute vs trading universe

| Layer | Purpose |
|-------|---------|
| **Precompute superset** | All tickers in ORATS cache used for feature/surface precompute (engineering) |
| **Trading universe** | Top 20% liquid names at rebalance `t` only |

Never treat the precompute superset as the trading universe in backtest or live.

---

## Rebalance linkage

- Rebalance: **weekly** (see [v1_spec_pins.md](v1_spec_pins.md)).
- Universe rebuilt **every rebalance** before signal ranking.
- New entries must pass surface tradability checks (iron fly or iron condor assembly) in addition to universe membership.

---

## Weekly trade workflow (not in this doc)

This document covers **who is eligible** each rebalance, not the full weekly sequence (universe → signal → structure → size → log).

After the **liquidity panel review** (Sprint 002), add [v1_weekly_runbook.md](v1_weekly_runbook.md) with the step-by-step weekly flow. Until then, see [development_workflow.md](development_workflow.md) and [repo_map.md](repo_map.md).

---

## Data dependencies

| Artifact | Path (default) | Script |
|----------|----------------|--------|
| Liquidity panel | `C:/MomentumCVG_env/cache/ticker_liquidity_panel.parquet` | `scripts/build_liquidity_panel.py` |
| ORATS adjusted chains | `C:/ORATS/data/ORATS_Adjusted` | — |
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
