"""
Per-date pipeline: six pure step functions.

Each function takes a DataFrame slice (for the current trade_date) plus
the BacktestRunConfig and returns an annotated DataFrame.  They have no
side effects, no mutable state, and no I/O — they can be unit-tested
independently with synthetic DataFrames.

Design reference: docs/backtest_engine_redesign.md §Layer 3

Column contracts
----------------
Each function documents which columns it REQUIRES on its input and which
columns it ADDS to its output.  The engine assembles the full trade_log
by passing each step's output to the next.

Direction model (resolved — Open Decisions #1, #7)
---------------------------------------------------
- direction = 'long'  (high momentum) → long straddle (buy vol).
  Structure lookup: straddle_history.  max_loss = net_debit (premium paid).
- direction = 'short' (low momentum)  → short vol structure per config.short_structure:
    'ironfly'  → short iron fly; wing candidate from ironfly_history.
    'straddle' → short straddle; position from straddle_history.
  Both sides are traded simultaneously. step3 routes per-ticker based on direction.

Remaining open questions
------------------------
- pnl units (Open Decision #4): ironfly_history stores pnl in per-share terms
  (same units as wing_width and net_credit).  Dollar P&L = return_pct_on_width
  × max_loss_budget_per_trade, which avoids any per-share / per-contract
  conversion.  Verify this assumption against the pre-compute script before use.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.backtest.run_config import BacktestRunConfig


# ---------------------------------------------------------------------------
# Step 1 — GET_UNIVERSE  (strategy_def §4.1, §3.2)
# ---------------------------------------------------------------------------

def step1_get_universe(
    trade_date: date,
    liquidity_panel: pd.DataFrame,
    config: "BacktestRunConfig",
) -> pd.DataFrame:
    """
    Build the point-in-time eligible ticker universe for trade_date.

    Uses the liquidity panel snapshot as-of trade_date to enforce that only
    information known at decision time is used (strategy_def §3.2).

    Required columns in liquidity_panel:
        month_date, ticker, atm_straddle_dollar_vol, atm_spread_pct, has_valid_atm_pair

    Returns columns:
        ticker, dvol_rank_pct, spread_rank_pct
        (one row per eligible ticker that passes BOTH dvol and spread filters)
    """
    # --- 1. Point-in-time snapshot lookup ---
    # Normalise trade_date to pd.Timestamp for comparison with month_date column.
    trade_ts = pd.Timestamp(trade_date)

    valid_months = liquidity_panel.loc[
        liquidity_panel["month_date"] <= trade_ts, "month_date"
    ]
    if valid_months.empty:
        return pd.DataFrame(columns=["ticker", "dvol_rank_pct", "spread_rank_pct"])

    snapshot_date = valid_months.max()

    # --- 2. Slice to the snapshot month and drop rows with no valid ATM pair ---
    snap = liquidity_panel[
        (liquidity_panel["month_date"] == snapshot_date)
        & (liquidity_panel["has_valid_atm_pair"] == True)  # noqa: E712
        & liquidity_panel["atm_straddle_dollar_vol"].notna()
        & liquidity_panel["atm_spread_pct"].notna()
    ].copy()

    if snap.empty:
        return pd.DataFrame(columns=["ticker", "dvol_rank_pct", "spread_rank_pct"])

    # --- 3. Rank BOTH metrics independently across the full snapshot ---
    # dvol_rank_pct: highest dollar vol → rank ~1  (ascending=True)
    # spread_rank_pct: tightest spread → rank ~1   (ascending=False)
    # Both ranks are computed on the full cross-section before any filtering.
    snap["dvol_rank_pct"] = (
        snap["atm_straddle_dollar_vol"]
        .rank(ascending=True, method="average", pct=True)
    )
    snap["spread_rank_pct"] = (
        snap["atm_spread_pct"]
        .rank(ascending=False, method="average", pct=True)
    )

    # --- 4. Apply BOTH filters simultaneously (AND logic) ---
    # A ticker must be in the top dvol_top_pct by volume AND
    # in the top spread_bottom_pct by tightness to qualify.
    # e.g. dvol_top_pct=0.20, spread_bottom_pct=0.20:
    #   keep tickers where dvol_rank_pct >= 0.80 AND spread_rank_pct >= 0.80.
    dvol_threshold   = 1.0 - config.dvol_top_pct
    spread_threshold = 1.0 - config.spread_bottom_pct
    universe = snap[
        (snap["dvol_rank_pct"]   >= dvol_threshold)
        & (snap["spread_rank_pct"] >= spread_threshold)
    ].copy()

    if universe.empty:
        return pd.DataFrame(columns=["ticker", "dvol_rank_pct", "spread_rank_pct"])

    # --- 5. Return [ticker, dvol_rank_pct, spread_rank_pct] ---
    return universe[["ticker", "dvol_rank_pct", "spread_rank_pct"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2 — SCORE_SIGNALS  (strategy_def §3.3, §6.1)
# ---------------------------------------------------------------------------

def step2_score_signals(
    trade_date: date,
    features: pd.DataFrame,
    universe: pd.DataFrame,
    config: "BacktestRunConfig",
) -> pd.DataFrame:
    """
    Rank the universe cross-sectionally by momentum and apply the CVG filter.

    Required columns in features:
        date, ticker, <momentum_col>, <cvg_col>, <count_col>

    Required columns in universe (output of step1):
        ticker

    Returns columns:
        ticker, direction, signal_score, signal_rank_pct, cvg_score, cvg_rank_pct
        (one row per ticker that passes momentum + CVG filters)

    Direction model (resolved):
        direction = 'long'  → will trade long straddle in step 3.
        direction = 'short' → will trade short iron fly or short straddle in step 3,
                              depending on config.short_structure.
        Both sides are always produced and traded simultaneously.
    """

    # 1. Slice features to rows where date == trade_date.

    # 2. Inner join with universe on ticker.
    #    Tickers not in the universe (failed liquidity filter) are dropped here.
    #    They are NOT recorded in trade_log (they never enter the pipeline).
    #    OPEN: should "not in universe" tickers appear as diagnostic rows?
    #    Recommendation: record them in step5 as exclusion_reason = 'not_in_universe'
    #    only if they also had a tradeable structure. Otherwise the diagnostic table
    #    becomes enormous.

    # 3. Drop rows where config.momentum_col is NaN or config.cvg_col is NaN.

    # 4. Apply data quality filter:
    #    Derive window_size from column name convention, e.g.:
    #       'mom_60_8_mean' → max_lag=60, min_lag=8 → window_size = 60 - 8 + 1 = 53
    #    Drop rows where count_col < config.min_count_pct * window_size.

    # 5. Cross-sectional momentum ranking (percentile rank across all survivors):
    #    signal_rank_pct = percentile rank of momentum_col (0 = lowest, 1 = highest).
    #    signal_score    = raw value of momentum_col.

    # 6. Select LONG candidates:
    #    signal_rank_pct >= (1.0 - config.long_top_pct)
    #    e.g. long_top_pct=0.10 → keep tickers with rank >= 0.90 (top 10%).

    # 7. Select SHORT candidates:
    #    signal_rank_pct <= config.short_bottom_pct
    #    e.g. short_bottom_pct=0.10 → keep tickers with rank <= 0.10 (bottom 10%).

    # 8. CVG filter — LONG candidates:
    #    Compute cvg_rank_pct = percentile rank of cvg_col WITHIN the long pool.
    #    Keep tickers where cvg_rank_pct >= (1.0 - config.cvg_filter_pct).
    #    e.g. cvg_filter_pct=0.50 → keep top 50% CVG within long pool.

    # 9. CVG filter — SHORT candidates:
    #    Same as step 8 but within the short pool.

    # 10. Tag rows with direction:
    #     Long survivors  → direction = 'long'
    #     Short survivors → direction = 'short'
    #
    #     NOTE: A ticker cannot appear on both sides simultaneously
    #     (momentum extremes don't overlap). Assert no duplicates.

    # 11. Combine long and short survivors into one DataFrame.

    # 12. Return DataFrame with columns:
    #     [ticker, direction, signal_score, signal_rank_pct, cvg_score, cvg_rank_pct]
    #     Sorted by direction then signal_rank_pct descending (for deterministic output).

    pass


# ---------------------------------------------------------------------------
# Step 3 — GET_ELIGIBLE_STRUCTURES  (strategy_def §4.2, §4.4)
# ---------------------------------------------------------------------------

def step3_get_eligible_structures(
    trade_date: date,
    signals: pd.DataFrame,
    ironfly_history: pd.DataFrame,
    straddle_history: pd.DataFrame,
    config: "BacktestRunConfig",
) -> pd.DataFrame:
    """
    For each signal ticker, look up the appropriate pre-computed structure
    based on direction and config.short_structure.

    Routing logic:
        direction = 'long'  → always look up straddle_history (long straddle, buy vol)
        direction = 'short' → look up ironfly_history  if config.short_structure == 'ironfly'
                              look up straddle_history if config.short_structure == 'straddle'

    Required columns in signals (output of step2):
        ticker, direction

    Required columns in ironfly_history:
        entry_date, ticker, is_tradeable, avg_wing_delta, credit_to_width,
        wing_width, net_credit, total_spread, spread_cost_ratio,
        body_strike, call_wing_strike, put_wing_strike,
        expiry_date, dte_actual, entry_spot,
        net_delta, net_gamma, net_vega, net_theta,
        pnl, return_pct_on_width, spot_move_pct, exit_spot, failure_reason

    Required columns in straddle_history:
        entry_date, ticker, net_debit, expiry_date, dte_actual, entry_spot,
        net_delta, net_vega, net_gamma, net_theta,
        pnl, return_pct_on_debit, spot_move_pct, exit_spot, is_tradeable, failure_reason
        (Note: no wing_width / call_wing_strike / put_wing_strike for straddles)

    Returns columns:
        One row per ticker with structure economics + added column:
            instrument_type: 'long_straddle' | 'short_straddle' | 'short_ironfly'
        Columns not applicable to the instrument type are NaN
        (e.g. wing_width is NaN for straddle rows).

    Note: tickers present in signals but absent from the relevant history table
    (or present with is_tradeable=False) will be absent from this output.
    Step 5 treats their absence as exclusion_reason = 'no_tradeable_structure'.
    """

    # 1. Split signals into long_tickers and short_tickers by direction column.

    # 2. LONG SIDE — long straddle lookup:
    #    straddle_slice = straddle_history[entry_date == trade_date]
    #    long_candidates = straddle_slice[ticker ∈ long_tickers & is_tradeable == True]
    #    Add instrument_type = 'long_straddle' to each row.
    #    Straddles have one row per ticker (no wing selection needed).

    # 3. SHORT SIDE — route based on config.short_structure:
    #
    #    If config.short_structure == 'ironfly':
    #        ironfly_slice = ironfly_history[entry_date == trade_date]
    #        short_candidates_raw = ironfly_slice[ticker ∈ short_tickers & is_tradeable == True]
    #        Add instrument_type = 'short_ironfly'.
    #
    #    If config.short_structure == 'straddle':
    #        straddle_slice = straddle_history[entry_date == trade_date]  (already loaded above)
    #        short_candidates_raw = straddle_slice[ticker ∈ short_tickers & is_tradeable == True]
    #        Add instrument_type = 'short_straddle'.

    # 4. Apply wing selection rule to short iron fly candidates only:
    #    (Skipped when short_structure == 'straddle'; straddles have one row per entry_date.)
    #
    #    If config.wing_selection_rule == 'closest_delta':
    #        For each ticker, select the row that minimises:
    #            abs(avg_wing_delta - config.wing_delta_target)
    #        Ties broken by credit_to_width descending.
    #
    #    If config.wing_selection_rule == 'max_credit_to_width':
    #        For each ticker, select the row that maximises credit_to_width.
    #        Ties broken by wing_width descending.
    #
    #    If config.wing_selection_rule == 'widest':
    #        For each ticker, select the row with the largest wing_width.
    #        Ties broken by credit_to_width descending.
    #
    #    Implementation: use groupby(ticker) + idxmin/idxmax, or sort + drop_duplicates.

    # 5. Combine long_candidates and short_candidates into a single DataFrame.
    #    Use pd.concat with consistent column schema — fill missing columns with NaN
    #    (e.g. long straddle rows will have NaN for wing_width, credit_to_width, etc.).

    # 6. Assert result has at most one row per ticker.
    #    Raise if duplicates found (indicates a bug in step 4 or routing logic).

    # 7. Return the combined DataFrame.
    #    instrument_type column distinguishes rows in downstream steps.
    #    Downstream steps join on ticker.

    pass


# ---------------------------------------------------------------------------
# Step 4 — APPLY_EXCLUSIONS  (strategy_def §5.4)
# ---------------------------------------------------------------------------

def step4_apply_exclusions(
    candidates: pd.DataFrame,
    earnings: pd.DataFrame,
    config: "BacktestRunConfig",
) -> pd.DataFrame:
    """
    Flag candidates whose expiry window contains an earnings announcement.

    Does NOT drop rows — exclusion_reason is assigned in step 5.
    This keeps the flag available for attribution even if the ticker is excluded.

    Required columns in candidates:
        ticker, expiry_date

    Required columns in earnings:
        ticker, earnings_date  (one row per announcement)

    Adds column:
        had_earnings_nearby  (bool)
    """

    # 1. For each row in candidates, define the exclusion window:
    #    window_start = expiry_date - timedelta(days=config.earnings_exclusion_days)
    #    window_end   = expiry_date
    #
    #    A candidate is flagged if any earnings row for the same ticker has:
    #        window_start <= earnings_date <= window_end

    # 2. Efficient implementation options:
    #    Option A: merge candidates with earnings on ticker, then filter by date range.
    #    Option B: for each candidate row, query earnings table with boolean mask.
    #    Prefer Option A (vectorised) for performance at scale.

    # 3. Add had_earnings_nearby column:
    #    True  → at least one earnings date falls within the window.
    #    False → no earnings in the window (safe to trade).

    # 4. Return candidates with had_earnings_nearby column added.
    #    All other columns preserved unchanged.

    pass


# ---------------------------------------------------------------------------
# Step 5 — SELECT_AND_SIZE  (strategy_def §6.2, §6.3)
# ---------------------------------------------------------------------------

def step5_select_and_size(
    signals: pd.DataFrame,
    structures: pd.DataFrame,
    config: "BacktestRunConfig",
) -> pd.DataFrame:
    """
    Join signals to structures, apply the portfolio cap, compute sizing,
    and assign included_in_portfolio + exclusion_reason to every row.

    Required columns in signals (output of step2):
        ticker, direction, signal_score, signal_rank_pct, cvg_score, cvg_rank_pct

    Required columns in structures (output of steps 3+4):
        ticker, instrument_type, had_earnings_nearby,
        wing_width (NaN for straddle rows), net_credit (or net_debit for long straddles),
        + all other history columns

    Returns columns:
        All signal columns + all structure columns + sizing columns:
            max_loss, quantity, notional_at_risk, max_loss_budget_per_trade
        + inclusion columns:
            included_in_portfolio (bool), exclusion_reason (str or None)

    Rows with included_in_portfolio=False are kept only if
    config.include_diagnostics == True.
    """

    # 1. Left join signals → structures on ticker.
    #    Left join preserves signal rows even when no structure exists for that ticker.

    # --- Assign exclusion reasons in priority order ---

    # 2. Rows where structure is missing (join produced NaN wing_width, etc.):
    #    exclusion_reason = 'no_tradeable_structure'
    #    included_in_portfolio = False

    # 3. Rows where had_earnings_nearby == True:
    #    exclusion_reason = 'earnings_exclusion'
    #    included_in_portfolio = False

    # 4. Remaining rows are candidates for the portfolio.
    #    Separate by direction to apply max_names_per_side cap independently.
    #
    #    For each direction side ('long', 'short'):
    #        Sort by signal_rank_pct (descending for 'long', ascending for 'short'
    #        because 'short' candidates have the LOWEST momentum rank).
    #        Keep the first max_names_per_side rows → included_in_portfolio = True.
    #        Remaining rows → exclusion_reason = 'max_names_cap', included = False.
    #
    #    NOTE on 'short' direction sort:
    #        Short candidates already sorted by signal_rank_pct ascending (lowest momentum
    #        first). Use signal_rank_pct ascending so the best short signal (rank nearest
    #        to 0) is kept first.

    # 5. For included rows: compute risk unit and sizing  (strategy_def §5.1, §6.3)
    #    max_loss depends on instrument_type:
    #
    #    instrument_type == 'short_ironfly':
    #        max_loss = wing_width - abs(net_credit)
    #        This is the bounded per-share loss (wings cap the downside).
    #
    #    instrument_type == 'long_straddle':
    #        max_loss = net_debit  (premium paid; entire debit is at risk if no move)
    #
    #    instrument_type == 'short_straddle':
    #        max_loss = net_credit * config.short_straddle_max_loss_multiplier
    #        (theoretical max loss is unlimited; use a risk proxy, e.g. 2× premium received)
    #        OPEN: pin the short straddle max_loss convention before implementation.
    #
    #    quantity = config.max_loss_budget_per_trade / max_loss
    #        Equal max-loss dollars across all instrument types.
    #
    #    notional_at_risk = quantity * max_loss
    #        Should equal max_loss_budget_per_trade (confirm as a sanity check).
    #
    #    NOTE on dollar P&L conversion (Open Decision #4):
    #        return_pct_on_width = pnl / wing_width  (already in ironfly_history)
    #        adjusted_pnl_dollars ≈ return_pct_on_width * max_loss_budget_per_trade
    #        This avoids per-share → per-contract × 100 conversion ambiguity.
    #        Verify this against the pre-compute script's pnl definition.

    # 6. Add max_loss_budget_per_trade column (constant from config, for traceability).

    # 7. If config.include_diagnostics == False:
    #    Drop rows where included_in_portfolio == False before returning.

    # 8. Return full trade_rows DataFrame for this date.

    pass


# ---------------------------------------------------------------------------
# Step 6 — APPLY_COST  (strategy_def §5.3)
# ---------------------------------------------------------------------------

def step6_apply_cost(
    trade_rows: pd.DataFrame,
    config: "BacktestRunConfig",
) -> pd.DataFrame:
    """
    Compute cost-adjusted P&L and the primary return metric for each traded row.

    Only rows where included_in_portfolio == True receive cost calculations.
    Diagnostic rows (included == False) get NaN for all new cost columns.

    Required columns in trade_rows:
        included_in_portfolio, quantity, max_loss, total_spread,
        pnl (per-share, from ironfly_history), return_pct_on_width,
        net_credit, max_loss_budget_per_trade

    Adds columns:
        spread_cost_applied, adjusted_pnl, return_on_max_loss,
        effective_credit, cost_model
    """

    # Only compute for included rows. For excluded rows, set all new columns to NaN.
    # Use a boolean mask: mask = trade_rows.included_in_portfolio == True

    # --- spread_cost_applied  (strategy_def §5.3) ---

    # If config.cost_model == 'mid':
    #     spread_cost_applied = 0  (best case: we fill at mid, no slippage)

    # If config.cost_model == 'half_spread_per_leg':
    #     spread_cost_applied = total_spread * 0.5 * quantity
    #     total_spread is the sum of (ask - bid) across all 4 legs, in per-share terms.
    #     Multiplied by quantity and 0.5 (half-spread on entry, none at expiry).
    #     OPEN: should there be an exit cost too? At expiry, intrinsic settlement
    #     means no bid-ask crossing. Answer: no exit cost in v1.

    # If config.cost_model == 'full_spread_per_leg':
    #     spread_cost_applied = total_spread * quantity  (worst case)

    # --- adjusted_pnl ---

    # The cleanest approach avoids per-share / per-contract unit ambiguity:
    #     adjusted_pnl = return_pct_on_width * max_loss_budget_per_trade - spread_cost_applied
    #
    # Where:
    #     return_pct_on_width  = pnl / wing_width  (already in ironfly_history, dimensionless)
    #     max_loss_budget_per_trade = max dollar risk (from config, dollars)
    #     spread_cost_applied       = dollar cost of entry slippage
    #
    # This gives adjusted_pnl in dollars directly.
    # Verify that return_pct_on_width * max_loss_budget_per_trade = pnl * quantity
    # holds before the first run.

    # --- return_on_max_loss  (primary metric, strategy_def §8.1) ---

    #     return_on_max_loss = adjusted_pnl / max_loss_budget_per_trade
    #
    # For the zero-cost (mid) case this equals return_pct_on_width exactly.
    # For non-zero cost it is slightly lower.
    # This is the per-trade metric that feeds Sharpe, selection attribution, etc.

    # --- effective_credit ---

    #     effective_credit = net_credit - (spread_cost_applied / quantity)
    #
    # The per-share credit actually received after entry cost.
    # Useful for sanity checks: effective_credit should be > 0 for a valid short fly.
    # If effective_credit <= 0 after cost, the trade should arguably not be taken.
    # OPEN: add a filter here, or flag it and let the analyst decide?

    # --- cost_model column ---

    #     cost_model = config.cost_model  (string, for reproducibility in the output table)

    # Return trade_rows with all six new columns added.

    pass
