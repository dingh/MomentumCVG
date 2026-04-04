# Backtest Engine Redesign: Pre-Computed Data Pipeline

## Core Design Shift

The current engine is wrong at its root: it re-runs `IronButterflyBuilder.build_strategy()` + `get_option_chain()` per ticker per date *during* the backtest. Everything about instrument structure and P&L is already fully pre-computed in `ironfly_history_weekly_2018_2026.parquet`. The engine should never touch ORATS parquet files at backtest time.

**Old model:** Engine → ORATS files → build strategy → P&L  
**New model:** Engine → in-memory pre-computed tables → select / size / cost-adjust → P&L

The backtest loop becomes pure pandas operations. The output — one flat DataFrame — is the TradeRecord automatically.

---

## Layer 1: Data Inputs (Loaded Once at Engine Init)

Everything indexed for fast lookup. No I/O during the per-date loop.

| Table | File | Key columns | Maps to strategy_def |
|---|---|---|---|
| `ironfly_history` | `ironfly_history_weekly_2018_2026.parquet` | `ticker × entry_date × wing_width` | §4.2 eligibility, §4.4 wing candidates, §4.5 economics — **used for short side when `short_structure='ironfly'`** |
| `straddle_history` | `straddle_history_weekly_2018_2026.parquet` | `ticker × entry_date` | §4.5 economics — **used for long side always; used for short side when `short_structure='straddle'`** |
| `features` | `straddle_features_weekly_2018_2026.parquet` | `ticker × date` | §3.1 signal inputs |
| `liquidity_panel` | `ticker_liquidity_panel.parquet` | `ticker × month_date` | §4.1 universe construction |
| `earnings` | `earnings_hist.parquet` | `ticker × earnings_date` | §5.4 event exclusions |

### What `ironfly_history` already provides per candidate row

- Identity: `ticker, entry_date, expiry_date, dte_actual, entry_spot, body_strike`
- Structure: `wing_width, call_wing_strike, put_wing_strike, avg_wing_delta`
- Economics: `net_credit, credit_to_width, total_spread, spread_cost_ratio`
- Greeks: `net_delta, net_gamma, net_vega, net_theta`
- P&L: `exit_spot, pnl, return_pct_on_width, spot_move_pct`
- Status: `is_tradeable, failure_reason`

### Columns to compute or add before implementation

1. **`max_loss`** — compute as `wing_width − abs(net_credit)`. This is the risk unit for §5.1 and §6.3. Can be added to `ironfly_history` at load time or pre-computed into the parquet.
2. **`finished_in_wings`** bool — `abs(exit_spot − body_strike) < wing_width`. Needed for tail-move attribution (attribution framework §2).
3. **`sector`** label per ticker — needed for §5.4 sector/cluster concentration caps. Join from external table.

---

## Layer 2: Engine Config (Each Field Traces to strategy_def)

```python
@dataclass
class BacktestRunConfig:
    run_id: str                        # traceability / output folder name

    # Signal model (strategy_def §3.3, §6.1)
    momentum_col: str                  # e.g. 'mom_60_8_mean'
    cvg_col: str                       # e.g. 'cvg_60_8'
    count_col: str                     # data quality guard
    min_count_pct: float               # e.g. 0.80
    long_top_pct: float                # top N% momentum → long signal
    short_bottom_pct: float            # bottom N% momentum → short signal
    cvg_filter_pct: float              # keep top N% CVG within each side

    # Universe (strategy_def §4.1, §3.2)
    dvol_top_pct: float                # keep top N% by dollar volume
    spread_bottom_pct: float           # keep bottom N% by effective spread

    # Structure selection (strategy_def §4.3, §4.4)
    short_structure: str               # 'ironfly' | 'straddle' — instrument sold on the short side
    wing_selection_rule: str           # 'closest_delta' | 'max_credit_to_width' | 'widest' — applies to short_structure='ironfly' only
    wing_delta_target: float           # used when rule = 'closest_delta', e.g. 0.15

    # Portfolio (strategy_def §5.4, §6.2, §6.3)
    max_names_per_side: int            # hard cap per long / short side
    max_loss_budget_per_trade: float   # dollars, equal across all selected trades
    earnings_exclusion_days: int       # exclude if earnings within N days of expiry

    # Cost model (strategy_def §5.3)
    cost_model: str                    # 'mid' | 'half_spread_per_leg' | 'full_spread_per_leg'

    # Date range
    start_date: date
    end_date: date
```

---

## Layer 3: Per-Date Pipeline (6 Steps)

Each step is a pure function: `(DataFrame slice, config) → filtered / annotated DataFrame`.
No mutable state. Steps can be unit-tested independently.

```
trade_date
     │
     ▼
Step 1  GET_UNIVERSE                                     (strategy_def §4.1, §3.2)
     │  liquidity_panel as-of trade_date
     │  → eligible ticker set (dvol top-N%, spread bottom-N%)
     │
     ▼
Step 2  SCORE_SIGNALS                                    (strategy_def §3.3, §6.1)
     │  features[date == trade_date] ∩ universe
     │  → drop NaN, apply min_count_pct
     │  → cross-sectional rank by momentum_col
     │  → long candidates (top long_top_pct%)
     │  → short candidates (bottom short_bottom_pct%)
     │  → within each side, apply CVG filter (top cvg_filter_pct%)
     │  Output: (ticker, direction, signal_score, signal_rank_pct, cvg_score, cvg_rank_pct)
     │
     ▼
Step 3  GET_ELIGIBLE_STRUCTURES                          (strategy_def §4.2, §4.4)
     │  Direction-split lookup:
     │    long  tickers → straddle_history[entry_date == trade_date] ∩ long_tickers
     │                    (buy vol: long call + long put)
     │    short tickers → if short_structure == 'ironfly':
     │                        ironfly_history[entry_date == trade_date] ∩ short_tickers
     │                        → keep is_tradeable == True rows
     │                        → apply wing selection rule to pick one row per ticker:
     │                             'closest_delta'       → argmin |avg_wing_delta − wing_delta_target|
     │                             'max_credit_to_width' → argmax credit_to_width
     │                             'widest'              → argmax wing_width
     │                    if short_structure == 'straddle':
     │                        straddle_history[entry_date == trade_date] ∩ short_tickers
     │  Output: one candidate row per ticker, tagged with instrument_type column
     │
     ▼
Step 4  APPLY_EXCLUSIONS                                 (strategy_def §5.4)
     │  Flag tickers with earnings within earnings_exclusion_days of expiry_date
     │  → add had_earnings_nearby bool column
     │  → drop excluded tickers (or keep with included_in_portfolio=False)
     │
     ▼
Step 5  SELECT_AND_SIZE                                  (strategy_def §6.2, §6.3)
     │  Join signals (Step 2) → eligible structures (Steps 3+4)
     │  Tickers must have both a signal AND a tradeable structure
     │  Apply max_names_per_side cap (take top-N by signal_rank_pct)
     │  Compute max_loss = wing_width − abs(net_credit)            (§5.1 risk unit)
     │  Compute quantity = max_loss_budget_per_trade / max_loss    (§6.3 equal max-loss)
     │  Mark remaining rows included_in_portfolio=False with exclusion_reason
     │
     ▼
Step 6  APPLY_COST                                       (strategy_def §5.3)
     │  'mid'                  → spread_cost_applied = 0
     │  'half_spread_per_leg'  → spread_cost_applied = total_spread * 0.5 * quantity
     │  'full_spread_per_leg'  → spread_cost_applied = total_spread * quantity
     │  adjusted_pnl = pnl * quantity − spread_cost_applied
     │  return_on_max_loss = adjusted_pnl / (quantity × max_loss)
     │
     ▼
     trade_rows for this date → append to trade_log
```

---

## Output: `trade_log` IS the TradeRecord

One flat DataFrame. One row per trade (or per candidate if `included_in_portfolio=False` rows are kept for attribution). Every column from every step is automatically preserved.

### Column groups

**Trade identity**
```
run_id, trade_date, expiry_date, ticker, direction, instrument_type, dte_actual
```
`instrument_type`: `'long_straddle'` | `'short_straddle'` | `'short_ironfly'`

**Universe context** (§4.1 — for attribution: was the name even in the universe?)
```
dvol_rank_pct, spread_rank_pct, in_liquidity_filter
```

**Signal context** (§3.3 — for attribution §8.2 selection effect)
```
signal_score, signal_rank_pct, cvg_score, cvg_rank_pct
```

**Structure economics** (§4.5)
```
body_strike, wing_width, call_wing_strike, put_wing_strike
net_credit, credit_to_width, max_loss, avg_wing_delta
total_spread, spread_cost_ratio
```

**Sizing** (§6.3)
```
max_loss_budget_per_trade, quantity, notional_at_risk
```

**Cost adjustment** (§5.3)
```
cost_model, spread_cost_applied, effective_credit
```

**Exit / P&L** (§8.1)
```
exit_spot, spot_move_pct, finished_in_wings
pnl, adjusted_pnl, return_on_max_loss
```

**Event flags** (§5.4)
```
had_earnings_nearby
```

**Greeks at entry** (for attribution §8.2 vega/gamma decomposition)
```
net_delta, net_vega, net_gamma, net_theta
```

**Inclusion tracking** (diagnostic rows for selection effect)
```
included_in_portfolio, exclusion_reason
```

| `included_in_portfolio` | `exclusion_reason` | Meaning |
|---|---|---|
| `True` | — | Selected and traded |
| `False` | `"signal_not_in_top_pct"` | Had structure, missed signal cut |
| `False` | `"no_tradeable_structure"` | Had signal, no valid structure in history for the required instrument type |
| `False` | `"earnings_exclusion"` | Dropped by earnings filter |
| `False` | `"max_names_cap"` | Had signal + structure, cut by max_names_per_side |
| `False` | `"not_in_universe"` | Failed liquidity filter |

---

## Performance Metrics Layer

All metrics computed as aggregations on `trade_log`. No special engine state.

```python
portfolio = trade_log[trade_log.included_in_portfolio]

# strategy_def §8.1
weekly_returns     = portfolio.groupby('trade_date').adjusted_pnl.sum() / max_loss_budget_total
sharpe             = weekly_returns.mean() / weekly_returns.std() * np.sqrt(52)
max_drawdown       = compute_drawdown(weekly_returns.cumsum())
return_on_max_loss = portfolio.return_on_max_loss.mean()

# strategy_def §8.2 — Selection effect
selection_corr = portfolio.groupby('trade_date').apply(
    lambda g: g['signal_rank_pct'].corr(g['return_on_max_loss'])
)

# strategy_def §8.2 — Tail-move / payoff cap (attribution framework §2)
move_buckets = pd.qcut(portfolio.spot_move_pct, q=4, labels=['small', 'medium', 'large', 'extreme'])
portfolio.groupby(move_buckets).return_on_max_loss.mean()

# strategy_def §8.2 — Cost effect
cost_drag = portfolio.spread_cost_applied.sum() / portfolio.notional_at_risk.sum()
```

---

## What Gets Eliminated from the Current Engine

| Current component | Why it goes away |
|---|---|
| `_build_strategies()` | Replaced by Step 3 (pandas join to pre-computed history) |
| `_close_positions()` | P&L is pre-computed in `ironfly_history`; no live settlement |
| `_find_expiry_date()` | `expiry_date` already in history table |
| `ORATSDataProvider` at backtest time | Only needed during pre-compute scripts |
| `IronButterflyBuilder` at backtest time | Only needed during pre-compute scripts |
| `BacktestExecutor.execute_exit()` | P&L is pre-computed |
| `EqualWeightOptimizer` (notional-based) | Replaced by Step 5 (equal max-loss sizing) |
| `Position` / `OptionStrategy` objects | Replaced by DataFrame rows in `trade_log` |

### What survives in new form

| Old concept | New form |
|---|---|
| `IStrategy.generate_signals()` | Steps 1+2 (universe filter + signal scoring) |
| `BacktestExecutionConfig` | `BacktestRunConfig` dataclass (new schema above) |

---

## Open Decisions to Pin Before Implementation

These change the code path, not just a parameter value.

1. ~~**Signal direction for iron fly**~~ — **RESOLVED.** Long side (high momentum) = long straddle (buy vol). Short side (low momentum) = short iron fly OR short straddle (sell vol), controlled by `short_structure` config field. Both sides are traded simultaneously.

2. **Wing selection rule for first research run** — `closest_delta`, `max_credit_to_width`, or `widest`?

3. **Cost model for first research run** — `mid`, `half_spread_per_leg`, or `full_spread_per_leg`?

4. **`max_loss` source** — compute inline as `wing_width − abs(net_credit)` at load time, or add to the pre-compute parquet? (Inline is fine; pre-compute avoids repeating the formula.) For long straddle, `max_loss = net_debit` (premium paid).

5. **Keep `included_in_portfolio=False` rows in `trade_log`?** — Costs memory; enables selection effect attribution cleanly. Recommended: yes, with a separate `diagnostics` DataFrame if memory is a concern.

6. **Max-loss budget expression** — fixed dollar amount per trade (e.g. $500), or a fraction of total capital? The fraction approach changes P&L and Sharpe with compounding.

7. ~~**Short side of the iron fly**~~ — **RESOLVED.** See #1 above.
