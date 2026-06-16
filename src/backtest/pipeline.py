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
    (v1 rejects naked short straddle; short side must be defined-risk.)
  Both sides are traded simultaneously. step3 routes per-ticker based on direction.

Remaining open questions
------------------------
- pnl units (Open Decision #4): Sprint 003 sizing stores ``quantity`` in share-equivalent
  units for both tiers (Tier A fractional; Tier B integer lots: contracts × contract_multiplier).
  S5 Simulate will compute ``pnl_total = quantity × pnl_per_share`` with no separate
  contract-multiplier application. ``capital_at_risk_dollars`` will be computed from
  quantity and the appropriate per-share at-risk denominator (Tier A vs Tier B).
"""

from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

from src.backtest.option_surface import (
    OptionSurfaceDB,
    StrategyAssemblyResult,
    build_straddle_from_surface,
    build_ironfly_from_surface,
    build_ironcondor_from_surface,
)

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

    _EMPTY = pd.DataFrame(
        columns=["ticker", "direction", "signal_score", "signal_rank_pct", "cvg_score", "cvg_rank_pct"]
    )

    # 1. Slice features to rows where date == trade_date.
    trade_ts = pd.Timestamp(trade_date)
    feat_slice = features[features["date"] == trade_ts].copy()
    if feat_slice.empty:
        return _EMPTY

    # 2. Inner join with universe on ticker.
    feat_slice = feat_slice.merge(universe[["ticker"]], on="ticker", how="inner")
    if feat_slice.empty:
        return _EMPTY

    # 3. Drop rows where momentum_col or cvg_col is NaN.
    feat_slice = feat_slice.dropna(subset=[config.momentum_col, config.cvg_col])
    if feat_slice.empty:
        return _EMPTY

    # 4. Apply data quality filter via count_col.
    #    Derive window_size from column name: 'mom_{max_lag}_{min_lag}_mean'
    #    window_size = max_lag - min_lag + 1
    if config.count_col in feat_slice.columns:
        import re as _re
        _m = _re.match(r"^mom_(\d+)_(\d+)_mean$", config.momentum_col)
        if _m:
            _max_lag, _min_lag = int(_m.group(1)), int(_m.group(2))
            window_size = _max_lag - _min_lag + 1
            count_threshold = config.min_count_pct * window_size
            feat_slice = feat_slice[feat_slice[config.count_col] >= count_threshold]
        if feat_slice.empty:
            return _EMPTY

    # 5. Cross-sectional momentum ranking.
    feat_slice["signal_rank_pct"] = feat_slice[config.momentum_col].rank(
        ascending=True, method="average", pct=True
    )
    feat_slice["signal_score"] = feat_slice[config.momentum_col]

    # 6. Select LONG candidates: top long_top_pct by rank.
    long_threshold = 1.0 - config.long_top_pct
    long_pool = feat_slice[feat_slice["signal_rank_pct"] >= long_threshold].copy()

    # 7. Select SHORT candidates: bottom short_bottom_pct by rank.
    short_pool = feat_slice[feat_slice["signal_rank_pct"] <= config.short_bottom_pct].copy()

    # 8. CVG filter — LONG candidates.
    if not long_pool.empty:
        long_pool["cvg_rank_pct"] = long_pool[config.cvg_col].rank(
            ascending=True, method="average", pct=True
        )
        cvg_long_threshold = 1.0 - config.cvg_filter_pct
        long_pool = long_pool[long_pool["cvg_rank_pct"] >= cvg_long_threshold].copy()

    # 9. CVG filter — SHORT candidates.
    if not short_pool.empty:
        short_pool["cvg_rank_pct"] = short_pool[config.cvg_col].rank(
            ascending=True, method="average", pct=True
        )
        cvg_short_threshold = 1.0 - config.cvg_filter_pct
        short_pool = short_pool[short_pool["cvg_rank_pct"] >= cvg_short_threshold].copy()

    if long_pool.empty and short_pool.empty:
        return _EMPTY

    # 10. Tag rows with direction and assert no overlap.
    long_pool["direction"] = "long"
    short_pool["direction"] = "short"

    overlap = set(long_pool["ticker"]) & set(short_pool["ticker"])
    assert not overlap, (
        f"step2_score_signals: tickers appear on both sides: {overlap}. "
        "This indicates long_top_pct + short_bottom_pct > 1.0."
    )

    # 11. Combine.
    combined = pd.concat([long_pool, short_pool], ignore_index=True)

    # 12. Add cvg_score, select output columns, sort.
    combined["cvg_score"] = combined[config.cvg_col]

    out_cols = ["ticker", "direction", "signal_score", "signal_rank_pct", "cvg_score", "cvg_rank_pct"]
    return (
        combined[out_cols]
        .sort_values(["direction", "signal_rank_pct"], ascending=[True, False])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Step 3 — GET_ELIGIBLE_STRUCTURES  (strategy_def §4.2, §4.4)
# ---------------------------------------------------------------------------

def _assemble_from_surface(
    surface_db: OptionSurfaceDB,
    ticker: str,
    trade_date: date,
    direction: str,
    config: "BacktestRunConfig",
) -> StrategyAssemblyResult:
    if direction == "long":
        return build_straddle_from_surface(
            surface_db=surface_db,
            ticker=ticker,
            entry_date=trade_date,
            direction="long",
            fill=config.fill,
            max_leg_spread_pct=config.max_leg_spread_pct,
        )

    if config.short_structure == "ironfly":
        return build_ironfly_from_surface(
            surface_db=surface_db,
            ticker=ticker,
            entry_date=trade_date,
            wing_target_delta=config.wing_delta_target,
            fill=config.fill,
            max_leg_spread_pct=config.max_leg_spread_pct,
            max_spread_cost_ratio=config.max_spread_cost_ratio,
        )
    if config.short_structure == "ironcondor":
        return build_ironcondor_from_surface(
            surface_db=surface_db,
            ticker=ticker,
            entry_date=trade_date,
            short_delta_target=config.condor_short_delta_target,
            long_delta_target=config.condor_long_delta_target,
            fill=config.fill,
            max_leg_spread_pct=config.max_leg_spread_pct,
            max_spread_cost_ratio=config.max_spread_cost_ratio,
        )
    if config.short_structure == "straddle":
        return build_straddle_from_surface(
            surface_db=surface_db,
            ticker=ticker,
            entry_date=trade_date,
            direction="short",
            fill=config.fill,
            max_leg_spread_pct=config.max_leg_spread_pct,
        )
    raise ValueError(f"Unsupported short_structure: {config.short_structure!r}")


def _assembly_to_row(assembly: StrategyAssemblyResult) -> Dict[str, object]:
    d = {
        "instrument_type": assembly.strategy_name,
        "entry_cost_per_share": float(assembly.entry_cost),
        "entry_cost_mid_per_share": float(assembly.entry_cost_mid),
        "net_credit_per_share": float(assembly.net_credit),
        "max_loss_per_share": (
            float(assembly.max_loss_per_share)
            if assembly.max_loss_per_share is not None
            else None
        ),
        "spread_cost_per_share": float(assembly.spread_cost),
        "spread_cost_ratio": assembly.spread_cost_ratio,
        "total_leg_spread_per_share": float(assembly.total_leg_spread),
        "leg_spread_to_credit_ratio": assembly.leg_spread_to_credit_ratio,
        "strategy_net_delta": float(assembly.strategy.net_delta),
        "strategy_net_vega": float(assembly.strategy.net_vega),
        "strategy_net_gamma": float(assembly.strategy.net_gamma),
        "strategy_net_theta": float(assembly.strategy.net_theta),
        "theoretical_return_on_max_loss": assembly.return_on_max_loss,
        "expiry_date": assembly.expiry_date,
        "entry_spot": float(assembly.entry_spot),
        "body_strike": float(assembly.body_strike),
    }
    d.update(assembly.diagnostics)
    return d


def step3_get_eligible_structures(
    trade_date: date,
    signals: pd.DataFrame,
    surface_db: OptionSurfaceDB,
    config: "BacktestRunConfig",
) -> pd.DataFrame:
    """
    Build one option structure per signal row using the precomputed quote surface.

    Required columns in signals (output of step2):
        ticker, direction, signal_score, signal_rank_pct, cvg_score, cvg_rank_pct

    Returns one row per signal with all signal columns preserved, plus:
        trade_date, structure_ok, failure_reason,
        entry_spot, exit_spot, body_strike, expiry_date, dte_actual,
        instrument_type, entry_cost_per_share, net_credit_per_share,
        max_loss_per_share, spread_cost_ratio, leg_spread_to_credit_ratio,
        strategy greeks, diagnostics, theoretical_return_on_max_loss,
        _assembly (StrategyAssemblyResult when structure_ok; used by step 5 settle)

    Earnings flagging is applied in step4_apply_exclusions.

    Routing:
        direction='long'                               → build_straddle(direction='long')
        direction='short', short_structure='straddle'  → build_straddle(direction='short')
        direction='short', short_structure='ironfly'  → build_ironfly(...)
        direction='short', short_structure='ironcondor'→ build_ironcondor(...)
    """
    if signals.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for rec in signals.to_dict(orient="records"):
        ticker = rec["ticker"]
        direction = rec["direction"]
        row = dict(rec)
        row.update(
            {
                "trade_date": trade_date,
                "instrument_type": None,
                "structure_ok": False,
                "failure_reason": None,
            }
        )

        try:
            meta = surface_db.get_metadata(ticker, trade_date)
            row["entry_spot"] = float(meta["entry_spot"])
            row["exit_spot"] = float(meta["exit_spot"])
            row["body_strike"] = float(meta["body_strike"])
            row["expiry_date"] = pd.Timestamp(meta["expiry_date"]).date()
            row["dte_actual"] = int(meta["dte_actual"])
        except Exception as exc:
            row["failure_reason"] = f"metadata_error:{exc}"
            rows.append(row)
            continue

        try:
            assembly = _assemble_from_surface(
                surface_db, ticker, trade_date, direction, config
            )
            row.update(_assembly_to_row(assembly))
            row["structure_ok"] = True
            row["_assembly"] = assembly
        except Exception as exc:
            row["failure_reason"] = str(exc)

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4 — APPLY_EXCLUSIONS  (strategy_def §5.4)
# ---------------------------------------------------------------------------

def _has_earnings_nearby(
    ticker: str,
    expiry_date: Optional[date],
    earnings: pd.DataFrame,
    exclusion_days: int,
) -> bool:
    if expiry_date is None or exclusion_days <= 0:
        return False
    expiry_ts = pd.Timestamp(expiry_date)
    start_ts = expiry_ts - pd.Timedelta(days=exclusion_days)
    mask = (
        (earnings["ticker"] == ticker)
        & (earnings["earnings_date"] >= start_ts)
        & (earnings["earnings_date"] <= expiry_ts)
    )
    return bool(mask.any())


def step4_apply_exclusions(
    structures: pd.DataFrame,
    earnings: Optional[pd.DataFrame],
    config: "BacktestRunConfig",
) -> pd.DataFrame:
    """
    Flag structures whose expiry window contains an earnings announcement.

    Does NOT drop rows — exclusion_reason is assigned in step 5 / runner settle.

    Required columns in structures (output of step3):
        ticker, expiry_date

    Required columns in earnings (when provided):
        ticker, earnings_date  (one row per announcement)

    Adds column:
        had_earnings_nearby  (bool)
    """
    out = structures.copy()
    if (
        earnings is None
        or earnings.empty
        or config.earnings_exclusion_days <= 0
    ):
        out["had_earnings_nearby"] = False
        return out

    out["had_earnings_nearby"] = out.apply(
        lambda row: _has_earnings_nearby(
            ticker=row["ticker"],
            expiry_date=row.get("expiry_date"),
            earnings=earnings,
            exclusion_days=config.earnings_exclusion_days,
        ),
        axis=1,
    )
    return out


# ---------------------------------------------------------------------------
# Step 5 — SELECT_AND_SIZE  (strategy_def §6.2, §6.3)
# ---------------------------------------------------------------------------

EXCLUSION_NO_STRUCTURE = "no_tradeable_structure"
EXCLUSION_EARNINGS = "earnings_exclusion"
EXCLUSION_MAX_NAMES_CAP = "max_names_cap"
EXCLUSION_INVALID_MAX_LOSS = "invalid_max_loss"
EXCLUSION_PREMIUM_EXCEEDS_FAIR_SHARE = "premium_exceeds_fair_share"
EXCLUSION_MAX_LOSS_EXCEEDS_FAIR_SHARE = "max_loss_exceeds_fair_share"
EXCLUSION_NO_SHORT_CREDIT = "no_short_credit"


def _signed_quantity(abs_qty: float, direction: str) -> float:
    """Position sign: long = +abs_qty, short = -abs_qty."""
    if direction == "long":
        return abs_qty
    return -abs_qty


def _tier_b_record_quantity(
    contracts: int,
    direction: str,
    contract_multiplier: float,
) -> float:
    """Tier B ``quantity`` in share-equivalent units (contracts × multiplier).

    Aligns with Tier A so simulate can use ``pnl_total = quantity × pnl_per_share``
    with no extra multiplier. ``quantity / contract_multiplier`` is always integer.
    """
    return _signed_quantity(float(contracts) * contract_multiplier, direction)


def _structure_premium_per_share(row: pd.Series) -> Optional[float]:
    """M1 / Tier-A ``equal_premium`` sizing denominator (positive magnitude)."""
    direction = row.get("direction")
    if direction == "long":
        raw = row.get("entry_cost_per_share")
    else:
        raw = row.get("net_credit_per_share")
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if direction == "long":
        return abs(val)
    return max(val, 0.0)


def _at_risk_per_share(row: pd.Series) -> Optional[float]:
    """Per-share capital-at-risk for Tier B sizing and ``capital_at_risk_dollars``."""
    direction = row.get("direction")
    if direction == "long":
        premium = _structure_premium_per_share(row)
        if premium is not None and premium > 0:
            return premium
    raw = row.get("max_loss_per_share")
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def _one_contract_premium_dollars(
    row: pd.Series,
    contract_multiplier: float,
) -> Optional[float]:
    """Premium for one option contract (per-share premium × multiplier)."""
    premium = _structure_premium_per_share(row)
    if premium is None or premium <= 0:
        return None
    return premium * contract_multiplier


def _one_contract_max_loss_dollars(
    row: pd.Series,
    contract_multiplier: float,
) -> Optional[float]:
    """Max-loss dollars for one short option contract."""
    if row.get("direction") != "short":
        return None
    at_risk = _at_risk_per_share(row)
    if at_risk is None or at_risk <= 0:
        return None
    return at_risk * contract_multiplier


def _iterative_fair_share_survivors(
    work: pd.DataFrame,
    idx_list: List[object],
    budget: float,
    one_contract_cost_fn,
) -> List[object]:
    """Return indices that survive iterative fair-share affordability filtering.

  Each round: compute ``fair_share = budget / n``. If any active name cannot
  afford one contract (cost > fair_share), drop the single **worst** offender
  (highest 1-contract cost) and recalculate. Repeat until all survivors fit or
  the pool is empty. Dropping one expensive name raises fair_share for the rest.
    """
    active = list(idx_list)
    while active:
        n = len(active)
        fair_share = budget / n
        unaffordable: List[object] = []
        for idx in active:
            cost = one_contract_cost_fn(work.loc[idx])
            if cost is None or cost <= 0 or cost > fair_share:
                unaffordable.append(idx)
        if not unaffordable:
            return active
        worst_idx = max(
            unaffordable,
            key=lambda i: one_contract_cost_fn(work.loc[i]) or float("inf"),
        )
        active.remove(worst_idx)
    return []


def _collected_short_credit_dollars(
    work: pd.DataFrame,
    short_idx: List[object],
    contract_multiplier: float,
) -> float:
    total = 0.0
    for idx in short_idx:
        if not bool(work.at[idx, "included_in_portfolio"]):
            continue
        credit = _structure_premium_per_share(work.loc[idx])
        qty = work.at[idx, "quantity"]
        if credit is None or credit <= 0 or qty is None or pd.isna(qty):
            continue
        # quantity is share-equivalent units (contracts × multiplier).
        total += abs(float(qty)) * credit
    return total


def _filter_longs_by_iterative_fair_share(
    work: pd.DataFrame,
    long_idx: List[object],
    long_budget: float,
    contract_multiplier: float,
) -> List[object]:
    """Return long indices that survive the iterative fair-share affordability filter."""
    return _iterative_fair_share_survivors(
        work,
        long_idx,
        long_budget,
        lambda row: _one_contract_premium_dollars(row, contract_multiplier),
    )


def _exclude_sized_row(
    work: pd.DataFrame,
    idx: object,
    exclusion_reason: str,
) -> None:
    work.at[idx, "included_in_portfolio"] = False
    work.at[idx, "exclusion_reason"] = exclusion_reason
    work.at[idx, "quantity"] = float("nan")


def _apply_tier_a_sizing(work: pd.DataFrame, config: "BacktestRunConfig") -> None:
    """Tier A — fractional units, no contract multiplier."""
    included = work.index[work["included_in_portfolio"] == True].tolist()  # noqa: E712
    if not included:
        return

    short_idx = [i for i in included if work.at[i, "direction"] == "short"]
    long_idx = [i for i in included if work.at[i, "direction"] == "long"]
    n_short = len(short_idx)
    n_long = len(long_idx)

    short_per_name_budget = (
        config.tier_a_short_budget / n_short if n_short else None
    )

    # Short side first (equal_max_loss long financing depends on short quantities).
    for idx in short_idx:
        if config.tier_a_mode == "equal_premium":
            denom = _structure_premium_per_share(work.loc[idx])
        else:
            denom = _at_risk_per_share(work.loc[idx])
        if denom is None or denom <= 0 or short_per_name_budget is None:
            work.at[idx, "included_in_portfolio"] = False
            work.at[idx, "exclusion_reason"] = EXCLUSION_INVALID_MAX_LOSS
            continue
        abs_qty = short_per_name_budget / denom
        work.at[idx, "quantity"] = _signed_quantity(abs_qty, "short")

    if config.tier_a_mode == "equal_premium":
        long_budget = config.tier_a_long_budget
    else:
        collected = 0.0
        for idx in short_idx:
            if not bool(work.at[idx, "included_in_portfolio"]):
                continue
            credit = _structure_premium_per_share(work.loc[idx])
            qty = work.at[idx, "quantity"]
            if credit is not None and credit > 0 and qty is not None and not pd.isna(qty):
                collected += abs(float(qty)) * credit
        if n_short == 0 or collected <= 0:
            long_budget = config.tier_a_long_budget
        else:
            long_budget = collected

    long_per_name_budget = long_budget / n_long if n_long and long_budget else None
    for idx in long_idx:
        denom = _structure_premium_per_share(work.loc[idx])
        if (
            denom is None
            or denom <= 0
            or long_per_name_budget is None
            or long_per_name_budget <= 0
        ):
            work.at[idx, "included_in_portfolio"] = False
            work.at[idx, "exclusion_reason"] = EXCLUSION_INVALID_MAX_LOSS
            continue
        abs_qty = long_per_name_budget / denom
        work.at[idx, "quantity"] = _signed_quantity(abs_qty, "long")


def _apply_tier_b_sizing(work: pd.DataFrame, config: "BacktestRunConfig") -> None:
    """Tier B — integer shorts (total max-loss fair share) + credit-financed longs.

    See docs/decisions/004_tier_b_credit_financed_long.md.
    """
    included = work.index[work["included_in_portfolio"] == True].tolist()  # noqa: E712
    if not included:
        return

    short_budget = config.tier_b_short_max_loss_budget
    multiplier = config.contract_multiplier
    short_idx = [i for i in included if work.at[i, "direction"] == "short"]
    long_idx = [i for i in included if work.at[i, "direction"] == "long"]

    # --- Pass 1: shorts from total max-loss budget (iterative fair share + integer lots) ---
    if short_idx and short_budget is not None and short_budget > 0:
        short_survivors = _iterative_fair_share_survivors(
            work,
            short_idx,
            short_budget,
            lambda row: _one_contract_max_loss_dollars(row, multiplier),
        )
        short_survivor_set = set(short_survivors)

        for idx in short_idx:
            if idx in short_survivor_set:
                continue
            cost = _one_contract_max_loss_dollars(work.loc[idx], multiplier)
            reason = (
                EXCLUSION_INVALID_MAX_LOSS
                if cost is None or cost <= 0
                else EXCLUSION_MAX_LOSS_EXCEEDS_FAIR_SHARE
            )
            _exclude_sized_row(work, idx, reason)

        if short_survivors:
            fair_share = short_budget / len(short_survivors)
            for idx in short_survivors:
                cost = _one_contract_max_loss_dollars(work.loc[idx], multiplier)
                if cost is None or cost <= 0:
                    _exclude_sized_row(work, idx, EXCLUSION_INVALID_MAX_LOSS)
                    continue
                contracts = math.floor(fair_share / cost)
                if contracts < 1:
                    _exclude_sized_row(work, idx, EXCLUSION_INVALID_MAX_LOSS)
                    continue
                work.at[idx, "quantity"] = _tier_b_record_quantity(
                    contracts, "short", multiplier
                )

    collected_credit = _collected_short_credit_dollars(work, short_idx, multiplier)
    long_budget = collected_credit

    if not long_idx:
        return

    # --- Pass 2: longs financed solely from collected short credit ---
    if long_budget <= 0:
        for idx in long_idx:
            _exclude_sized_row(work, idx, EXCLUSION_NO_SHORT_CREDIT)
        return

    survivors = _filter_longs_by_iterative_fair_share(
        work, long_idx, long_budget, multiplier
    )
    survivor_set = set(survivors)

    for idx in long_idx:
        if idx in survivor_set:
            continue
        cost = _one_contract_premium_dollars(work.loc[idx], multiplier)
        reason = (
            EXCLUSION_INVALID_MAX_LOSS
            if cost is None or cost <= 0
            else EXCLUSION_PREMIUM_EXCEEDS_FAIR_SHARE
        )
        _exclude_sized_row(work, idx, reason)

    if not survivors:
        return

    fair_share = long_budget / len(survivors)
    for idx in survivors:
        cost = _one_contract_premium_dollars(work.loc[idx], multiplier)
        if cost is None or cost <= 0:
            _exclude_sized_row(work, idx, EXCLUSION_INVALID_MAX_LOSS)
            continue
        contracts = math.floor(fair_share / cost)
        if contracts < 1:
            _exclude_sized_row(work, idx, EXCLUSION_INVALID_MAX_LOSS)
            continue
        work.at[idx, "quantity"] = _tier_b_record_quantity(contracts, "long", multiplier)


def _apply_sizing(work: pd.DataFrame, config: "BacktestRunConfig") -> None:
    work["quantity"] = float("nan")
    work["sizing_mode"] = config.sizing_mode
    work["max_loss_budget_per_trade"] = float("nan")

    if config.sizing_mode == "conceptual":
        _apply_tier_a_sizing(work, config)
    elif config.sizing_mode == "integer_lots":
        _apply_tier_b_sizing(work, config)


def _is_invalid_max_loss(value: object) -> bool:
    """A row fails the geometric max-loss check when its per-share max loss is
    missing or non-positive (defined-risk structures must risk > 0 per share).

    Long straddles carry ``max_loss_per_share = premium_paid`` (> 0), so they
    are not rejected here. Iron fly / condor rows whose ``wing_width - net_credit``
    collapses to ``0`` (set by the assembler) are rejected.
    """
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return float(value) <= 0.0
    except (TypeError, ValueError):
        return True


def step5_select_and_size(
    signals: pd.DataFrame,
    structures: pd.DataFrame,
    config: "BacktestRunConfig",
) -> pd.DataFrame:
    """
    S5 Phases 1–2 — SELECT (per-side cap + rank) + SIZE (Tier A / Tier B).

    Turns post-S4 candidates into a selection- and sizing-annotated trade log.
    Every candidate row gets ``included_in_portfolio`` (bool) and
    ``exclusion_reason`` (str | None). Included rows receive ``quantity`` and
    ``sizing_mode``. Simulate (S7 settle, M1–M3, ``pnl_total``,
    ``capital_at_risk_dollars``) is a later Sprint 003 phase — not performed here.

    Signal columns (`direction`, `signal_rank_pct`) are preserved onto the
    structure rows by S3, so no separate `signals` join is performed — the
    `signals` parameter is accepted for orchestration symmetry but is unused here.

    Required columns in structures (output of steps 3+4):
        ticker, direction, signal_rank_pct, structure_ok, had_earnings_nearby,
        max_loss_per_share  (+ all other S3/S4 columns, preserved verbatim)

    Selection rules (design § S5 Phase 1; exclusion vocabulary matches the runner):
        1. structure_ok != True            → 'no_tradeable_structure' (priority)
        2. structure_ok & earnings nearby  → 'earnings_exclusion'
        3. per side (independent pools, decision 003): rank by signal_rank_pct
           (long descending, short ascending); keep top max_names_per_side →
           included_in_portfolio = True; overflow → 'max_names_cap'
        4. a selected row with missing / non-positive max_loss_per_share →
           'invalid_max_loss', included_in_portfolio = False

    Rows with included_in_portfolio == False are dropped unless
    config.include_diagnostics is True.
    """
    if structures is None or structures.empty:
        empty = structures.copy() if structures is not None else pd.DataFrame()
        empty["included_in_portfolio"] = pd.Series(dtype=bool)
        empty["exclusion_reason"] = pd.Series(dtype=object)
        empty["quantity"] = pd.Series(dtype=float)
        empty["sizing_mode"] = pd.Series(dtype=object)
        empty["max_loss_budget_per_trade"] = pd.Series(dtype=float)
        return empty

    work = structures.copy()
    work["included_in_portfolio"] = False
    work["exclusion_reason"] = None

    # --- Eligibility (read S3/S4 flags; do not re-run upstream filters) ---
    structure_ok = work["structure_ok"] == True  # noqa: E712
    work.loc[~structure_ok, "exclusion_reason"] = EXCLUSION_NO_STRUCTURE

    earnings_mask = structure_ok & (work["had_earnings_nearby"] == True)  # noqa: E712
    work.loc[earnings_mask, "exclusion_reason"] = EXCLUSION_EARNINGS

    eligible_mask = structure_ok & (work["had_earnings_nearby"] == False)  # noqa: E712
    eligible = work[eligible_mask]

    # --- Per-side cap + rank (independent long / short pools — decision 003) ---
    selected_idx: List[object] = []
    for direction, side in eligible.groupby("direction", sort=False):
        # long: best signal is the HIGHEST rank (descending); short: LOWEST (ascending).
        ascending = direction != "long"
        side_sorted = side.sort_values("signal_rank_pct", ascending=ascending)
        selected_idx.extend(side_sorted.head(config.max_names_per_side).index.tolist())
        overflow = side_sorted.iloc[config.max_names_per_side:]
        if not overflow.empty:
            work.loc[overflow.index, "exclusion_reason"] = EXCLUSION_MAX_NAMES_CAP

    if selected_idx:
        work.loc[selected_idx, "included_in_portfolio"] = True

    # --- Sizing-eligibility reject (selection boundary, not a sizing computation) ---
    if "max_loss_per_share" in work.columns:
        for idx in selected_idx:
            if _is_invalid_max_loss(work.at[idx, "max_loss_per_share"]):
                work.at[idx, "included_in_portfolio"] = False
                work.at[idx, "exclusion_reason"] = EXCLUSION_INVALID_MAX_LOSS

    # --- Phase 2 — SIZE (Tier A conceptual or Tier B integer_lots) ---
    _apply_sizing(work, config)

    if not config.include_diagnostics:
        work = work[work["included_in_portfolio"] == True].copy()  # noqa: E712

    return work.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 6 — APPLY_COST  (deprecated — collapsed into step5 for v1)
# ---------------------------------------------------------------------------

def step6_apply_cost(
    trade_rows: pd.DataFrame,
    config: "BacktestRunConfig",
) -> pd.DataFrame:
    """
    Deprecated: return labeling and fill economics live in step5 + S3 FillAssumption.

    See docs/surface_engine_portfolio_metrics_design.md (S6 collapsed into S5).

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
