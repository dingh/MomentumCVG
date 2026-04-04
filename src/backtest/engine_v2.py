"""
BacktestEngineV2: orchestrates the six-step per-date pipeline.

Design reference: docs/backtest_engine_redesign.md

Architecture contract
---------------------
The engine accepts pre-computed parquet data as DataFrames at construction
time.  At run time it performs ONLY pandas operations — no ORATS file I/O,
no ORATSDataProvider, no IronButterflyBuilder, no Position objects.

All strategy decisions are encoded in BacktestRunConfig.  The engine is
stateless across runs: calling run() twice with different configs on the
same engine instance is safe and independent.

Data dependency map
--------------------
ironfly_history  →  step 3 short side when short_structure='ironfly' (structure economics + pnl)
straddle_history →  step 3 long side always; step 3 short side when short_structure='straddle'
features         →  step 2 (signal scoring)
liquidity_panel  →  step 1 (universe filter)
earnings         →  step 4 (exclusion flag)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.backtest.pipeline import (
    step1_get_universe,
    step2_score_signals,
    step3_get_eligible_structures,
    step4_apply_exclusions,
    step5_select_and_size,
    step6_apply_cost,
)

if TYPE_CHECKING:
    from src.backtest.run_config import BacktestRunConfig


class BacktestEngineV2:
    """
    Orchestrates the per-date pipeline over a full backtest date range.

    Usage
    -----
    engine = BacktestEngineV2(
        ironfly_history  = pd.read_parquet('cache/ironfly_history_weekly_2018_2026.parquet'),
        straddle_history = pd.read_parquet('cache/straddle_history_weekly_2018_2026.parquet'),
        features         = pd.read_parquet('cache/straddle_features_weekly_2018_2026.parquet'),
        liquidity_panel  = pd.read_parquet('cache/ticker_liquidity_panel.parquet'),
        earnings         = pd.read_parquet('cache/earnings_hist.parquet'),
    )
    results = engine.run(config)

    results keys
    ------------
    'trade_log'  : pd.DataFrame  — flat record of every trade / diagnostic row
    'metrics'    : dict          — Sharpe, drawdown, win_rate, cost_drag, etc.
    'run_config' : BacktestRunConfig — the config that produced this result
    """

    def __init__(
        self,
        ironfly_history: pd.DataFrame,
        straddle_history: pd.DataFrame,
        features: pd.DataFrame,
        liquidity_panel: pd.DataFrame,
        earnings: pd.DataFrame,
    ) -> None:
        """
        Store and pre-index the five input tables.

        No computation happens here beyond indexing — construction is cheap.
        """

        # 1. Store each table as an instance attribute.

        # 2. Normalise date column dtypes to pd.Timestamp throughout:
        #    ironfly_history:  entry_date, expiry_date
        #    straddle_history: entry_date, expiry_date
        #    features:         date
        #    liquidity_panel:  month_date
        #    earnings:         earnings_date
        #    Consistent types prevent silent merge failures from date vs Timestamp mismatches.

        # 3. Pre-index ironfly_history by entry_date for fast per-date slice:
        #    self._ironfly_by_date = dict keyed by normalized entry_date,
        #    values = sub-DataFrame for that date.
        #    At ~750 dates × ~500 tickers × ~5 wing widths = ~1.9M rows,
        #    a dict-of-slices lookup is faster than repeated boolean indexing.

        # 4. Pre-index straddle_history by entry_date similarly:
        #    self._straddle_by_date = dict keyed by normalized entry_date.
        #    Used for both long side (always) and short side when short_structure='straddle'.

        # 5. Pre-index features by date similarly:
        #    self._features_by_date = dict keyed by normalized date.

        # 6. Store liquidity_panel directly — it is sliced month-by-month in step 1,
        #    and step 1 needs the full panel to do the point-in-time lookup.
        #    No per-date pre-indexing needed.

        # 7. Store earnings directly — size is small, per-ticker lookup in step 4.

        pass

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def run(self, config: "BacktestRunConfig") -> dict:
        """
        Execute the six-step pipeline for every trade date in the config date range.

        Returns a results dict with keys: 'trade_log', 'metrics', 'run_config'.
        """

        # 1. Determine trade dates:
        #    All unique dates in the features table that fall within
        #    [config.start_date, config.end_date].
        #    Features table is the authoritative source of trade dates (data-driven).
        #    Sort ascending for deterministic output and clean cumulative metric calc.

        # 2. Initialise accumulator:
        #    trade_log_parts = []  # will hold one DataFrame per trade_date

        # 3. Per-date loop:
        for trade_date in []:  # placeholder; replace with actual date list

            # a. Step 1 — Get universe
            #    universe = step1_get_universe(trade_date, self._liquidity_panel, config)
            #    If universe is empty (no eligible tickers): log a warning and continue.

            # b. Step 2 — Score signals
            #    signals = step2_score_signals(
            #        trade_date, self._features_by_date.get(trade_date), universe, config)
            #    If signals is empty (no tickers pass momentum + CVG filter):
            #        Record an empty-date sentinel row? Or just skip?
            #        Decision: skip — sparse coverage is normal for tight cvg_filter_pct.

            # c. Step 3 — Get eligible structures (direction-routed)
            #    structures = step3_get_eligible_structures(
            #        trade_date, signals,
            #        self._ironfly_by_date.get(trade_date, empty_df),
            #        self._straddle_by_date.get(trade_date, empty_df),
            #        config)
            #    Long tickers → long straddle lookup.
            #    Short tickers → ironfly or straddle lookup per config.short_structure.

            # d. Step 4 — Apply exclusions
            #    structures_flagged = step4_apply_exclusions(
            #        structures, self._earnings, config)

            # e. Step 5 — Select and size
            #    trade_rows = step5_select_and_size(signals, structures_flagged, config)

            # f. Step 6 — Apply cost
            #    trade_rows = step6_apply_cost(trade_rows, config)

            # g. Add trade_date column to trade_rows for traceability in the log.
            #    trade_log_parts.append(trade_rows)

            pass

        # 4. Concatenate all per-date DataFrames into a single trade_log.
        #    trade_log = pd.concat(trade_log_parts, ignore_index=True)
        #    Handle edge case: no trade dates produced any rows → return empty trade_log.

        # 5. Attach run metadata columns to trade_log:
        #    run_id     = config.run_id
        #    run_config fields are already recorded per-row in sizing / cost columns.

        # 6. Compute performance metrics.
        #    metrics = self._compute_metrics(trade_log, config)

        # 7. Return results dict.
        #    return {
        #        'trade_log'  : trade_log,
        #        'metrics'    : metrics,
        #        'run_config' : config,
        #    }

        pass

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    def _compute_metrics(
        self,
        trade_log: pd.DataFrame,
        config: "BacktestRunConfig",
    ) -> dict:
        """
        Aggregate the trade_log into summary performance metrics.

        All metrics computed on included rows only (included_in_portfolio == True).
        Diagnostic rows are present in trade_log but must NOT enter metric calculations.
        """

        # 1. Filter to included trades:
        #    included = trade_log[trade_log.included_in_portfolio == True]

        # 2. Weekly P&L series:
        #    Group by trade_date, sum adjusted_pnl → weekly_pnl Series indexed by date.
        #    normalize by total max_loss_budget deployed that week to get weekly return:
        #        weekly_budget = group.max_loss_budget_per_trade.sum()
        #        weekly_return = weekly_pnl / weekly_budget
        #    This gives a dimensionless weekly return series for Sharpe calculation.

        # 3. Sharpe ratio (annualised, weekly series):
        #    sharpe = (weekly_return.mean() / weekly_return.std()) * sqrt(52)
        #    Handle edge case: no variance (all returns identical) → Sharpe = NaN.

        # 4. Cumulative return series:
        #    cum_return = (1 + weekly_return).cumprod()
        #    Used for drawdown calculation.

        # 5. Maximum drawdown:
        #    rolling_max    = cum_return.cummax()
        #    drawdown_series = (cum_return - rolling_max) / rolling_max
        #    max_drawdown   = drawdown_series.min()   (most negative value)

        # 6. Per-trade return_on_max_loss statistics:
        #    avg_return_on_max_loss = included.return_on_max_loss.mean()
        #    median_return_on_max_loss = included.return_on_max_loss.median()

        # 7. Win rate:
        #    win_rate = (included.adjusted_pnl > 0).mean()
        #    Average win P&L and average loss P&L (for payoff ratio).

        # 8. Cost drag:
        #    total_cost = included.spread_cost_applied.sum()
        #    total_risk  = included.notional_at_risk.sum()
        #    cost_drag_bps = (total_cost / total_risk) * 10_000

        # 9. Coverage diagnostics (requires include_diagnostics == True):
        #    Counts per exclusion_reason in trade_log (for all rows including False).
        #    e.g. {'earnings_exclusion': 42, 'no_tradeable_structure': 17, 'max_names_cap': 88}

        # 10. Selection correlation (signal quality, requires diagnostics == True):
        #     Within each trade_date, correlate signal_rank_pct with return_on_max_loss
        #     for all rows that had a tradeable structure (regardless of inclusion).
        #     avg_selection_ic = mean information coefficient across dates.

        # 11. Return metrics dict:
        #     {
        #         'sharpe'                   : float,
        #         'max_drawdown'             : float,
        #         'avg_return_on_max_loss'   : float,
        #         'median_return_on_max_loss': float,
        #         'win_rate'                 : float,
        #         'total_trades'             : int,
        #         'total_dates'              : int,
        #         'cost_drag_bps'            : float,
        #         'avg_names_per_date'       : float,
        #         'exclusion_counts'         : dict,
        #         'avg_selection_ic'         : float,
        #     }

        pass
