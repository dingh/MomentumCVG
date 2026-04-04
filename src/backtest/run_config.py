"""
BacktestRunConfig: single configuration object for the pre-computed pipeline engine.

Every field maps directly to a section of docs/strategy_definition.md.
The config is the complete specification of one backtest run — no hidden
defaults, no global state, no external dependencies at construction time.

Changing any single field produces a reproducibly different run.
See docs/backtest_engine_redesign.md for the full design rationale.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal


# ---------------------------------------------------------------------------
# Valid literal values (guard against typos at construction time)
# ---------------------------------------------------------------------------

VALID_WING_SELECTION_RULES = ('closest_delta', 'max_credit_to_width', 'widest')
VALID_COST_MODELS           = ('mid', 'half_spread_per_leg', 'full_spread_per_leg')
VALID_SHORT_STRUCTURES      = ('ironfly', 'straddle')


@dataclass
class BacktestRunConfig:
    """
    Complete, self-contained configuration for one backtest run.

    Fields are grouped by the strategy_def section they implement.
    All decisions that affect results live here — nothing is implicit.

    Direction model (resolved):
      - Long side  (high momentum) = long straddle (buy vol: long call + long put).
        Looks up straddle_history for P&L.
      - Short side (low momentum) = short vol structure controlled by `short_structure`:
          'ironfly'  → short iron fly (sell body, buy wings); looks up ironfly_history.
          'straddle' → short straddle (sell ATM call + put); looks up straddle_history.
      Both sides are traded simultaneously each rebalance date.

    Remaining open decisions (see docs/backtest_engine_redesign.md §Open Decisions):
      - max_loss_budget_per_trade (#6): fixed dollar amount here; fraction-of-capital
        variant is a later enhancement.
    """

    # -----------------------------------------------------------------------
    # Identity
    # -----------------------------------------------------------------------

    run_id: str
    # Identifies this run in logs and output files.
    # Convention: descriptive slug, e.g. 'baseline_mom60_8_closest_delta_v1'

    # -----------------------------------------------------------------------
    # Signal model  (strategy_def §3.3, §6.1)
    # -----------------------------------------------------------------------

    momentum_col: str
    # Feature column name for the primary cross-sectional ranking signal.
    # Must exist in the features parquet.
    # e.g. 'mom_60_8_mean'

    cvg_col: str
    # Feature column for the CVG conditioning filter applied within each side.
    # Must exist in the features parquet.
    # e.g. 'cvg_60_8'

    count_col: str
    # Feature column for data quality guard (number of observed windows).
    # e.g. 'mom_60_8_count'

    min_count_pct: float
    # Minimum fraction of the observation window that must have valid data.
    # Tickers below this threshold are dropped before signal ranking.
    # e.g. 0.80 = require at least 80% of window weeks to have data.

    long_top_pct: float
    # Fraction of the momentum cross-section selected as long-side candidates.
    # e.g. 0.10 = top 10% by momentum score.

    short_bottom_pct: float
    # Fraction of the momentum cross-section selected as short-side candidates.
    # e.g. 0.10 = bottom 10% by momentum score.

    cvg_filter_pct: float
    # Within each side (long / short), keep top N% by CVG score.
    # Filters for tickers with the most continuous trends.
    # e.g. 0.50 = keep top 50% CVG within each side's candidate pool.

    # -----------------------------------------------------------------------
    # Universe  (strategy_def §4.1, §3.2)
    # -----------------------------------------------------------------------

    dvol_top_pct: float
    # Keep top N% of tickers by dollar volume from the liquidity panel snapshot.
    # Applied point-in-time: uses the most recent panel month_date <= trade_date.
    # e.g. 0.20 = top 20% dollar volume.

    spread_bottom_pct: float
    # Keep bottom N% of tickers by effective spread (tightest spreads = most liquid).
    # Applied in conjunction with dvol_top_pct: both filters must pass.
    # e.g. 0.20 = bottom 20% spread (i.e. 20% tightest).

    # -----------------------------------------------------------------------
    # Structure selection  (strategy_def §4.3, §4.4)
    # -----------------------------------------------------------------------

    short_structure: str
    # Instrument to use on the short side (low momentum = sell vol).
    # 'ironfly'  → short iron fly: sell ATM body, buy OTM wings for protection.
    #              Looks up ironfly_history. Max loss is bounded (wing_width − net_credit).
    # 'straddle' → short straddle: sell ATM call and put without wing protection.
    #              Looks up straddle_history. Sizing uses a notional risk proxy since
    #              max loss is theoretically unlimited.
    # Must be one of VALID_SHORT_STRUCTURES.
    # NOTE: Long side always uses long straddle regardless of this field.

    wing_selection_rule: str
    # How to pick which wing-width row to use per ticker from the pre-computed candidates.
    # Only applies when short_structure == 'ironfly'.
    # 'closest_delta'        → wing whose avg_wing_delta is nearest to wing_delta_target
    # 'max_credit_to_width'  → wing with the highest net_credit / wing_width ratio
    # 'widest'               → widest wing that exists in the history table
    # Must be one of VALID_WING_SELECTION_RULES.

    wing_delta_target: float
    # Target absolute delta for the long OTM wings.
    # Only used when short_structure == 'ironfly' and wing_selection_rule == 'closest_delta'.
    # e.g. 0.15 = target ~15-delta wings.

    # -----------------------------------------------------------------------
    # Portfolio construction  (strategy_def §5.4, §6.2, §6.3)
    # -----------------------------------------------------------------------

    max_names_per_side: int
    # Hard cap on the number of names per signal side after all filters.
    # Long side and short side are capped independently.
    # Surplus names are recorded in trade_log with exclusion_reason = 'max_names_cap'.
    # e.g. 10 = at most 10 long trades and 10 short trades per rebalance date.

    max_loss_budget_per_trade: float
    # Dollar amount of max-loss risk allocated to each individual trade.
    # Equal across all selected trades (strategy_def §6.3 v1 baseline).
    # max_loss = wing_width − abs(net_credit) per trade; quantity is derived from this.
    # e.g. 500.0 = risk at most $500 of max-loss capital per iron fly.

    earnings_exclusion_days: int
    # Exclude a ticker if it has an earnings announcement within this many
    # calendar days before the option expiry date.
    # e.g. 5 = skip any name whose earnings date falls in [expiry - 5, expiry].

    # -----------------------------------------------------------------------
    # Cost model  (strategy_def §5.3)
    # -----------------------------------------------------------------------

    cost_model: str
    # Execution cost assumption applied to each traded position.
    # 'mid'                  → no slippage cost (mid-price fills; best case)
    # 'half_spread_per_leg'  → deduct half the bid-ask spread per leg at entry
    # 'full_spread_per_leg'  → deduct full bid-ask spread per leg at entry (worst case)
    # Must be one of VALID_COST_MODELS.
    # NOTE: All cost is applied at entry only. Exit is at expiry (intrinsic value),
    # so no exit cost is modelled in v1.

    # -----------------------------------------------------------------------
    # Date range
    # -----------------------------------------------------------------------

    start_date: date
    end_date: date

    # -----------------------------------------------------------------------
    # Optional settings (rarely changed between runs)
    # -----------------------------------------------------------------------

    include_diagnostics: bool = True
    # If True, trade_log includes rows where included_in_portfolio == False.
    # These rows are essential for selection-effect attribution (strategy_def §8.2):
    # they represent candidates that were in the universe but not selected.
    # Set to False to reduce memory footprint if attribution is not needed.

    def __post_init__(self):
        # --- short_structure must be one of the two accepted strings ---
        # raise ValueError if not in VALID_SHORT_STRUCTURES

        # --- wing_selection_rule must be one of the three accepted strings ---
        # raise ValueError if not in VALID_WING_SELECTION_RULES

        # --- cost_model must be one of the three accepted strings ---
        # raise ValueError if not in VALID_COST_MODELS

        # --- Signal model fractions: must be strictly between 0 and 1 ---
        # raise ValueError if long_top_pct not in (0, 1)
        # raise ValueError if short_bottom_pct not in (0, 1)
        # raise ValueError if cvg_filter_pct not in (0, 1]
        # raise ValueError if min_count_pct not in (0, 1]

        # --- Universe fractions: must be strictly between 0 and 1 ---
        # raise ValueError if dvol_top_pct not in (0, 1]
        # raise ValueError if spread_bottom_pct not in (0, 1]

        # --- Portfolio bounds ---
        # raise ValueError if max_names_per_side < 1
        # raise ValueError if max_loss_budget_per_trade <= 0
        # raise ValueError if earnings_exclusion_days < 0

        # --- Date range ---
        # raise ValueError if start_date >= end_date

        # --- wing_delta_target only meaningful for iron fly + closest_delta rule ---
        # if short_structure == 'ironfly' and wing_selection_rule == 'closest_delta':
        #     raise ValueError if wing_delta_target not in (0, 0.5)

        pass
