# V1 specification pins

**Status:** Active  
**Last updated:** 2026-06-11 (Sprint 003 config-validation alignment)  
**Purpose:** Single source of truth for v1 backtest and live-prep scope. Change only via explicit decision memo.

---

## Strategy core

| Parameter | Pin | Notes |
|-----------|-----|-------|
| Signal | **Momentum + CVG** (`MomentumCVGStrategy`) | No new alpha for v1 |
| Short structure (low momentum) | **Iron fly or iron condor** | One per backtest run; wing width/delta is a search dimension. Compare via separate `BacktestRunConfig` runs (see `run_surface_search.py`). |
| Long structure (high momentum) | **Long straddle** (today) | Buy ATM call + put. **Naked long call** is deferred research. |
| Defined risk | **Short side only** | Short vol structures are defined-risk; long straddle has premium at risk, not the same max-loss-budget framing. |
| DTE bucket | **7 DTE** | One tenor for v1 |
| Rebalance | **Weekly minimum** | Bi-weekly / monthly are sensitivity tests later |
| Backtest exit | **Hold to expiry** | Simplifies v1 backtest |
| Live exit | **Deferred** | Document placeholder before paper; iron flies will not be held to expiry live |
| Delta hedge | **Off for v1** | Revisit only if risk audit requires |

---

## Portfolio

| Parameter | Pin | Notes |
|-----------|-----|-------|
| Max concurrent positions | **Per-side cap via `max_names_per_side`** | Long and short pools capped independently; e.g. `25` per side → up to 50 total when both fill. See [decisions/003_position_cap_per_side.md](decisions/003_position_cap_per_side.md). |
| Sizing method | **Tier A (`conceptual`) or Tier B (`integer_lots`)** via required `sizing_mode` | Tier A = per-side budget ÷ name count (`tier_a_mode` ∈ {`equal_premium`, `equal_max_loss`}); Tier B = integer lots × `contract_multiplier`, capital-binding. See Sprint 003 design + [decisions/003_position_cap_per_side.md](decisions/003_position_cap_per_side.md). |
| Global max-loss budget | **Deferred — not required for v1 backtest** | Sizing budgets stay abstract risk units (`deployable_capital` optional / `None`). Revisit at paper-trading stage. |
| Per-name max-loss cap | **Deferred — not required for v1 backtest** | Configurable if/when needed; revisit at paper-trading stage. |

---

## Universe

| Parameter | Pin | Notes |
|-----------|-----|-------|
| Pool | Broader than S&P 500 | Precompute superset; trade subset only |
| Selection rule | **Rolling 3-month liquidity rank, top 20%** *(tentative; window configurable)* | Point-in-time. Supersedes earlier trailing-4-week intent per [surface_engine_data_contract.md](surface_engine_data_contract.md) HD decision (rolling-window avg of `atm_straddle_dollar_vol` / `atm_spread_pct`). See [v1_universe_protocol.md](v1_universe_protocol.md) |
| Liquidity score source | **`ticker_liquidity_panel.parquet`** via pipeline step1 | Uses `atm_straddle_dollar_vol`, `atm_spread_pct` |

---

## Backtest and evaluation

| Parameter | Pin | Notes |
|-----------|-----|-------|
| Canonical engine | **SurfaceRunner path** (provisional) | See [decisions/001_canonical_backtest_path.md](decisions/001_canonical_backtest_path.md) |
| Primary go/no-go window | **2020-01-01 → latest available data** | Specific metrics TBD in evaluation protocol |
| Full sanity window | Full sample in cache | Max drawdown and regime checks |
| Primary fill assumption for go/no-go | **Harsh / conservative** | Mid fill is optimistic bound only |

---

## Execution and broker (deferred to Month 2–3)

| Parameter | Pin | Notes |
|-----------|-----|-------|
| Broker | **Undecided** | IBKR if shadow shows high weekly structure count; manual if low |
| Accounts available | IBKR, Robinhood | Robinhood not on critical path |
| Capital | ~$1M deployable | Size by max-loss budget, not notional |
| v1 live capital | **Tiny fraction of budget** | Ramp only after paper + go/no-go memo |

---

## Explicitly deferred (do not implement until gated)

- Bi-weekly / monthly rebalance variants (after weekly + harsh-fill baseline)
- Live early-close exit rule (before paper trading)
- Broker API integration (after shadow mode)
- `BacktestEngineV2` full rewrite (unless audit falsifies SurfaceRunner)
- Delta hedging
- Signal-weighted portfolio optimizer
- Clean-room `ORATS_v2` repo migration (deprecated)
- Naked long call on long side (after short-structure backtest gate)

---

## Known backtest vs live gaps

1. **Hold to expiry (backtest) vs early close (live)** — largest PnL/ops gap after fill assumptions.
2. **Fill model** — backtest must use conservative fills before go/no-go.
3. **Manual vs automated execution** — resolved by shadow trade counts in `v1_ops_model.md`.

---

## Change log

| Date | Change |
|------|--------|
| 2026-05-23 | Initial pins: 50 max positions, 7 DTE, 2020+ go/no-go window, weekly rebalance |
| 2026-05-23 | Week 0 review: short = iron fly or iron condor per run; long straddle; naked long call deferred |
| 2026-05-25 | Clarified max concurrent positions as 50 total across long+short via Decision 002 |
| 2026-06-07 | Per-side cap via `max_names_per_side` (Decision 003 supersedes 002); e.g. 25+25 ≈ 50-book |
| 2026-06-11 | Sizing method → Tier A / Tier B (`sizing_mode`) per Sprint 003 design |
| 2026-06-11 | Global max-loss budget + per-name cap → deferred (not required for v1 backtest; revisit at paper trading) |
| 2026-06-11 | Removed sector / cluster cap from v1 scope (not considered for v1) |
| 2026-06-11 | Universe selection → rolling 3-month liquidity (tentative, configurable window); supersedes trailing-4-week per data-contract HD decision |
