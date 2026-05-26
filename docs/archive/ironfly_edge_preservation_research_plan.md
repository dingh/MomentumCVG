# Research Plan: Why Iron Fly Preserves Edge Unevenly

## Core observation from the feature-window sweep heatmaps

The uneven Sharpe drop from body (straddle) to iron fly does **not** look like a purely ticker-level effect.

There are **two layers**:

1. **Signal-window compatibility**
   - Some lag configurations produce a body edge that survives the iron-fly transformation.
   - Other lag configurations do not survive the transformation well.

2. **Ticker-level compatibility**
   - Within the lag configurations that do survive, some tickers likely preserve more of the edge than others.

This means the analysis should **not** start by searching only for ticker properties.

It should start by separating:
- which **signal windows** are fly-compatible
- and then, inside those windows, which **ticker properties** help preserve the edge

### Key puzzle — the (52,8) vs (50,6) pair

These two windows are nearly identical in structure (similar max_lag, similar min_lag, similar body Sharpe) yet the fly outcome is dramatically different: (52,8) preserves the edge while (50,6) turns negative. Understanding WHY is the central question.

---

## Main hypotheses

### Hypothesis A: ticker selection divergence
Even with nearby lag windows, the momentum ranking can differ enough to select **different tickers** on many weeks. If (52,8) selects tickers whose payoff rarely breaches the wings, but (50,6) selects tickers with more tail events, the fly kills the edge for (50,6).

### Hypothesis B: payoff truncation asymmetry
The fly caps both gains and losses at the wing width. If the body edge depends on occasional large positive payoffs (convexity benefit of the short straddle when vol collapses), the fly truncates those. Windows where the edge comes from small, consistent carry are more fly-compatible.

### Hypothesis C: wing cost and spread drag vary with ticker mix
Different ticker selections imply different wing costs. If (50,6) systematically selects names with expensive wings or wide spreads, the credit drag alone may explain the gap.

### Hypothesis D: temporal clustering
The two windows may agree most of the time but diverge on a handful of critical weeks. A few bad fly weeks (where the body had a large positive payoff that got truncated) could swing the annualized Sharpe.

---

## Completed work

### ✅ Step 1: identify fly-compatible signal windows (DONE)
`feature_window_sweep_results.parquet` contains body_sharpe, fly_sharpe, body_sharpe_fly_matched, and sharpe_gap_matched for all ~281 windows. The 4-panel heatmap visualizes this.

### ✅ Step 2: compare lag-window families (DONE)
The heatmap already shows the preservation pattern. The (52,8) vs (50,6) pair was identified as the most informative contrast — structurally similar windows with opposite fly outcomes.

---

## Analysis plan — starting from Step 3

### Step 3: ticker overlap analysis (NEW — do this first)

Before building matched-trade tables, answer: **do (52,8) and (50,6) even select the same tickers?**

Using the sweep_trade_log (already saved), for each trade date:
1. Re-run signal generation for both windows to get the Q1+CVG selected tickers
2. Compute Jaccard similarity: `|intersection| / |union|` per date
3. Compute overlap rate: `|intersection| / |selected|` per date

#### Output
- Time series of overlap %
- Average overlap across all dates
- Identify "divergence weeks" where the two windows select completely different names

#### Interpretation
- **High overlap (>70%):** The ticker selection is similar → the difference must come from **which weeks** the fly hurts (temporal effect) or from subtle **return differences** on the same tickers
- **Low overlap (<40%):** The windows select different names → the fly gap is driven by **ticker composition** → proceed with ticker-level analysis

---

### Step 3b: build matched-trade decomposition table

For the focus pair (52,8) and (50,6), build a per-trade table using the ironfly history parquet.

For each `(ticker, entry_date)` where both body and fly (delta_target=0.10) exist:

#### Available columns from ironfly_history_weekly_2018_2026_liquidity.parquet

**Identity:** `ticker`, `entry_date`, `expiry_date`, `body_strike`

**Body row (row_type='body'):**
- `net_credit`, `total_spread`, `spread_cost_ratio`
- `exit_value`, `pnl`, `return_pct_on_credit`
- `sc_bid`, `sc_ask`, `sc_iv`, `sc_delta` (short call)
- `sp_bid`, `sp_ask`, `sp_iv`, `sp_delta` (short put)
- `net_delta`, `net_gamma`, `net_vega`, `net_theta`
- `spot_move_pct`, `exit_spot`, `entry_spot`

**Fly row (row_type='ironfly_candidate'):**
- All body columns PLUS:
- `wing_width`, `call_wing_strike`, `put_wing_strike`, `avg_wing_delta`
- `credit_to_width`
- `lc_bid`, `lc_ask`, `lc_iv`, `lc_delta` (long call wing)
- `lp_bid`, `lp_ask`, `lp_iv`, `lp_delta` (long put wing)
- `return_pct_on_width`

#### Derived fields
- `delta_credit = fly_net_credit - body_net_credit` (always negative — buying wings costs premium)
- `delta_spread = fly_total_spread - body_total_spread` (extra spread from 2 more legs)
- `delta_exit = fly_exit_value - body_exit_value` (protection value — positive when body loss is capped)
- `delta_pnl = fly_pnl - body_pnl`
- `wing_cost = body_net_credit - fly_net_credit` (premium paid for wings)
- `wing_cost_ratio = wing_cost / body_net_credit`
- `truncated = 1 if |body_exit_value| > wing_width else 0` (did the wings actually matter?)

#### Key: tag each row with which window selected it
- `selected_52_8`: boolean — was this ticker in Q1+CVG for window (52,8) on this date?
- `selected_50_6`: boolean — same for (50,6)

This lets us compare the same trades when selected by different signals.

---

### Step 4: P&L decomposition — credit drag vs truncation

For each matched trade, decompose `delta_pnl` into two components:

```
delta_pnl = delta_credit + delta_exit
          = (fly_credit - body_credit) + (fly_exit_value - body_exit_value)
```

Where:
- **`delta_credit`** is always ≤ 0 (you pay for wings) — this is the **drag**
- **`delta_exit`** can be positive (wings cap a large body loss) or negative (wings cap a large body gain) — this is the **truncation effect**

Split trades into buckets by `spot_move_pct`:
- Small moves (|spot_move| < 3%): fly ≈ body minus wing cost
- Medium moves (3-8%): fly starts to diverge
- Large moves (>8%): fly truncates heavily

#### Compare across the two windows:
For trades selected by (52,8) vs (50,6):
- Mean `delta_credit` (should be similar if ticker mix is similar)
- Mean `delta_exit` (will differ if one window selects more tail-event tickers)
- Distribution of `truncated` flag
- **Key metric:** % of weeks where `delta_exit > 0` (fly saved money) vs `delta_exit < 0` (fly gave up upside)

#### Goal
Determine whether the gap comes from:
1. **Wing cost alone** (delta_credit differs) — ticker composition effect
2. **Truncation asymmetry** (delta_exit differs) — the signal selects names that move more/less
3. **Temporal concentration** — a few extreme weeks explain the whole difference

---

### Step 5: time-series comparison

Using sweep_trade_log, plot for both windows side by side:

1. **Cumulative P&L curves** — body and fly for each window, all on one chart (4 lines)
2. **Rolling 26-week Sharpe** — body and fly for each window
3. **Weekly delta_pnl** — bar chart of `(fly_pnl - body_pnl)` per week for each window
4. **Highlight divergence weeks** — weeks where (52,8) fly was positive but (50,6) fly was negative

#### Goal
Find whether the gap is:
- Persistent (fly always underperforms more for (50,6))
- Concentrated (a few weeks cause the entire difference)

---

### Step 6: ticker-level properties (only after Steps 3-5)

Inside the windows that survive, test which **entry-time ticker properties** predict better edge preservation. Prioritize by what's directly available in the data:

#### Tier 1 — directly available in ironfly history
1. `wing_cost_ratio` = wing_cost / body_net_credit
2. `credit_to_width` (fly column)
3. `spread_cost_ratio` (body and fly separately)
4. `avg_wing_delta`
5. `sc_iv - lc_iv` (ATM-wing IV skew, call side)
6. `sp_iv - lp_iv` (ATM-wing IV skew, put side)

#### Tier 2 — derivable from existing data
7. `spot_move_pct` distribution per ticker (historical jumpiness)
8. Earnings proximity (from `earnings_hist.parquet` in cache)

#### Tier 3 — requires additional data
9. IV-HV gap (need realized vol computation)
10. Term structure slope

#### Method
For each property in Tier 1:
- Quintile sort the matched trades
- Compare mean `delta_pnl` across quintiles
- If monotonic → property has explanatory power

---

### Step 7: synthesis — can we build a fly-compatibility filter?

If Steps 3-6 identify clear drivers, the final output is a **scoring function** that says, for a given (ticker, date):

> "This trade is fly-compatible" or "Keep this as a naked straddle"

This becomes a pre-trade filter for live deployment.

---

## Practical principles

1. Do **not** try to explain the whole heatmap with ticker properties alone. First identify the signal windows, then the ticker properties.
2. Start with the simplest analysis (overlap %, cumulative plots) before any regression or model.
3. The (52,8) vs (50,6) pair is the sharpest knife — if we can explain THIS pair, the general pattern likely follows.
4. Use bucket sorts and visual inspection before any p-values or regressions.

---

## Available data assets (already computed)

| File | Location | Key columns |
|------|----------|-------------|
| `feature_window_sweep_results.parquet` | cache/ | body_sharpe, fly_sharpe, ic_body_mean per window |
| `feature_window_sweep_trade_log.parquet` | cache/ | body_pnl_pct, fly_pnl_pct_body, ic_body per (window, date) |
| `ironfly_history_weekly_2018_2026_liquidity.parquet` | cache/ | Full trade-level data: 48 cols per (ticker, date, row_type) |
| `straddle_features_weekly_2018_2026_liquidity.parquet` | cache/ | mom_*, cvg_*, count_* for all windows |
| `earnings_hist.parquet` | cache/ | Earnings dates per ticker |
| `ticker_liquidity_panel.parquet` | cache/ | Monthly liquidity metrics |

---

## Immediate next steps (checklist)

- [x] Sweep all feature windows — body & fly Sharpe (feature_window_sweep.ipynb)
- [x] Heatmap visualization — 4-panel with gap
- [ ] **Step 3: Ticker overlap analysis** — Jaccard similarity between (52,8) and (50,6) per week
- [ ] **Step 3b: Matched-trade table** — merge body + fly rows, tag with selected_52_8 / selected_50_6
- [ ] **Step 4: P&L decomposition** — delta_credit vs delta_exit analysis, bucket by spot_move_pct
- [ ] **Step 5: Time-series comparison** — 4-line cumulative chart, rolling Sharpe, divergence weeks
- [ ] **Step 6: Tier 1 ticker properties** — quintile sorts on wing_cost_ratio, credit_to_width, spread_cost_ratio, IV skew
- [ ] **Step 7: Synthesis** — fly-compatibility scoring function

---

## Notebook plan

Create `ironfly_edge_preservation.ipynb` in `c:\MomentumCVG_env\notebook\` with the following sections:

1. **Config & load** — load ironfly history, features, sweep trade log
2. **Re-derive selected tickers** — for (52,8) and (50,6), re-run the Q1+CVG selection per date to get the actual ticker sets
3. **Overlap analysis** — Jaccard per date, time series, summary stats
4. **Matched-trade table** — build the (ticker, date) table with body/fly columns + derived fields
5. **P&L decomposition** — delta_credit, delta_exit, bucket by spot_move_pct
6. **Time-series plots** — cumulative P&L, rolling Sharpe, divergence weeks
7. **Ticker property sorts** — quintile analysis for Tier 1 properties
8. **Summary & next steps**
