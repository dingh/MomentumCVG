"""Helpers for assembling iron flies and iron condors from a precomputed quote surface.

This module is intentionally independent from the current BacktestEngine. It gives you
a flexible surface-first workflow:

1. load the precomputed metadata + quote parquets
2. select legs under arbitrary delta rules
3. apply arbitrary fill assumptions
4. compute entry economics, max loss, ROC, and expiry P&L

The same surface can therefore support many backtest variants without regenerating the
precompute files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Dict, Optional, Tuple, Literal
import pandas as pd

from src.core.models import OptionQuote, OptionLeg, OptionStrategy, Position, StrategyType


@dataclass(frozen=True)
class FillAssumption:
    """Interpolated fill assumption inside the bid/ask interval.

    Models the price at which a leg is filled as a linear interpolation
    between bid and ask.  Alpha = 0 is the passive (limit) extreme;
    Alpha = 1 is the aggressive (market) extreme.

    buy_alpha:
        Fill price for a buy order: ``bid + alpha * (ask - bid)``.
        0.0 = filled at bid (best case), 0.5 = mid, 1.0 = filled at ask.
    sell_alpha:
        Fill price for a sell order: ``ask - alpha * (ask - bid)``.
        0.0 = filled at ask (best case), 0.5 = mid, 1.0 = filled at bid.

    The two factory methods cover the most common research scenarios:
    - ``FillAssumption.mid()``   — all legs filled at mid (optimistic baseline)
    - ``FillAssumption.cross()`` — buys at ask, sells at bid (realistic / conservative)
    """
    buy_alpha: float = 0.5
    sell_alpha: float = 0.5
    label: str = "mid"

    def __post_init__(self):
        if not (0.0 <= self.buy_alpha <= 1.0):
            raise ValueError(f"buy_alpha must be in [0, 1], got {self.buy_alpha}")
        if not (0.0 <= self.sell_alpha <= 1.0):
            raise ValueError(f"sell_alpha must be in [0, 1], got {self.sell_alpha}")

    @classmethod
    def mid(cls) -> "FillAssumption":
        """All legs filled at mid — optimistic baseline, zero spread cost."""
        return cls(buy_alpha=0.5, sell_alpha=0.5, label="mid")

    @classmethod
    def cross(cls) -> "FillAssumption":
        """Buys filled at ask, sells filled at bid — conservative / market-order model."""
        return cls(buy_alpha=1.0, sell_alpha=1.0, label="cross")

    def buy_price(self, quote: OptionQuote) -> Decimal:
        """Return the fill price for a long (buy) leg."""
        spread = quote.ask - quote.bid
        return quote.bid + Decimal(str(self.buy_alpha)) * spread

    def sell_price(self, quote: OptionQuote) -> Decimal:
        """Return the fill price for a short (sell) leg."""
        spread = quote.ask - quote.bid
        return quote.ask - Decimal(str(self.sell_alpha)) * spread

@dataclass
class StrategyAssemblyResult:
    """The assembled strategy and its entry economics for one (ticker, entry_date).

    Sign convention
    ---------------
    ``entry_cost`` follows the direction of cash flow *from* the trader:
    - Negative  → net credit received (typical for iron fly / condor).
    - Positive  → net debit paid (typical for long straddle).

    ``net_credit = -entry_cost``, so a positive net_credit means you received premium.

    ``spread_cost`` is the additional cost incurred by trading away from mid:
    always non-negative (positive = you paid the spread, zero = mid fill).

    ``spread_cost_ratio = spread_cost / net_credit`` — measures friction as a
    fraction of the premium collected.  ``None`` when net_credit <= 0.

    ``return_on_max_loss = net_credit / max_loss_per_share`` — the maximum possible
    return expressed as a fraction of the capital at risk.  ``None`` when
    max_loss_per_share <= 0 (should not happen for a well-formed iron fly/condor).
    """
    strategy_name: str
    ticker: str
    entry_date: date
    expiry_date: date
    entry_spot: Decimal
    body_strike: Decimal
    strategy: OptionStrategy
    entry_cost: Decimal           # negative = net credit received
    entry_cost_mid: Decimal       # entry_cost evaluated at mid fills
    net_credit: Decimal           # = -entry_cost (positive when credit strategy)
    max_loss_per_share: Optional[Decimal]   # wing_width - net_credit
    return_on_max_loss: Optional[float]     # net_credit / max_loss_per_share
    spread_cost: Decimal                    # actual_fill_cost - mid_fill_cost (>= 0)
    spread_cost_ratio: Optional[float]      # spread_cost / net_credit
    diagnostics: Dict[str, object] = field(default_factory=dict)

    @property
    def total_leg_spread(self) -> Decimal:
        """Sum of bid-ask spreads across all legs, weighted by absolute quantity.

        This is a fill-agnostic measure of market width: it does not depend on
        ``FillAssumption`` and is identical whether you fill at mid or at the
        crossing price.  It answers: "How much total market friction exists in
        this position if you were to pay the full spread on every leg?"

            total_leg_spread = sum(|qty| * (ask - bid)  for each leg)
        """
        return sum(
            abs(leg.quantity) * (leg.option.ask - leg.option.bid)
            for leg in self.strategy.legs
        )

    @property
    def leg_spread_to_credit_ratio(self) -> Optional[float]:
        """Total market spread across all legs divided by net credit received.

        Answers: "For every $1 of credit collected, how much total bid-ask
        spread exists across the legs?"  A ratio of 1.0 means total market
        width equals the credit collected; lower is better.

        Unlike ``spread_cost_ratio`` (which is zero for mid fill), this metric
        is fill-agnostic and reflects the underlying market liquidity regardless
        of fill assumption.

        Returns ``None`` when ``net_credit <= 0``.
        """
        if self.net_credit <= 0:
            return None
        return float(self.total_leg_spread / self.net_credit)

    def settle(self, exit_spot: Decimal, exit_date: Optional[date] = None) -> Position:
        """Compute the P&L at expiry and return a ``Position`` record.

        Parameters
        ----------
        exit_spot:
            Underlying price at expiry used for payoff calculation.
        exit_date:
            Date the position is closed.  Defaults to ``self.expiry_date``.
        """
        exit_date = exit_date or self.expiry_date
        exit_value = self.strategy.calculate_payoff({self.expiry_date: exit_spot})
        return Position(
            ticker=self.ticker,
            entry_date=self.entry_date,
            strategy=self.strategy,
            quantity=1.0,
            entry_cost=self.entry_cost,
            exit_date=exit_date,
            exit_value=exit_value,
            metadata=self.diagnostics.copy(),
        )

class OptionSurfaceDB:
    """In-memory index over the two precomputed option surface parquet files.

    Wraps the metadata table (one row per ticker/date) and the quote table
    (many rows per ticker/date) and exposes fast point lookups by
    ``(ticker, entry_date)``.

    Load from parquet files:
        ``db = OptionSurfaceDB.load(meta_path, quotes_path)``

    Then query:
        ``meta, quotes = db.get_surface("AAPL", date(2024, 11, 29))``
    """

    def __init__(self, meta_df: pd.DataFrame, quotes_df: pd.DataFrame):
        self.meta_df   = meta_df.copy()
        self.quotes_df = quotes_df.copy()

        # Normalise date columns to datetime64 regardless of how they were stored
        for df in (self.meta_df, self.quotes_df):
            if "entry_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["entry_date"]):
                df["entry_date"] = pd.to_datetime(df["entry_date"])
            if "expiry_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["expiry_date"]):
                df["expiry_date"] = pd.to_datetime(df["expiry_date"])

        # Materialise a plain Python date column for O(n) equality lookups.
        # Comparing datetime64 to a Python date object requires this step.
        self.meta_df["entry_date_key"]   = self.meta_df["entry_date"].dt.date
        self.quotes_df["entry_date_key"] = self.quotes_df["entry_date"].dt.date

    @classmethod
    def load(cls, meta_path: str, quotes_path: str) -> "OptionSurfaceDB":
        """Construct an ``OptionSurfaceDB`` by reading both parquet files from disk."""
        return cls(pd.read_parquet(meta_path), pd.read_parquet(quotes_path))

    def get_metadata(self, ticker: str, entry_date: date) -> pd.Series:
        """Return the metadata row for *(ticker, entry_date)*.

        Raises
        ------
        KeyError  : no row found in the metadata table.
        ValueError: row found but ``surface_valid=False``.
        """
        mask = (self.meta_df["ticker"] == ticker) & (self.meta_df["entry_date_key"] == entry_date)
        if not mask.any():
            raise KeyError(f"No metadata row for {ticker} on {entry_date}")
        row = self.meta_df.loc[mask].iloc[0]
        if not bool(row.get("surface_valid", False)):
            raise ValueError(
                f"Surface for {ticker} on {entry_date} is invalid: "
                f"{row.get('failure_reason')}"
            )
        return row

    def get_quotes(self, ticker: str, entry_date: date) -> pd.DataFrame:
        """Return all quote-surface rows for *(ticker, entry_date)*.

        Raises KeyError when no rows are found.
        """
        mask = (self.quotes_df["ticker"] == ticker) & (self.quotes_df["entry_date_key"] == entry_date)
        out  = self.quotes_df.loc[mask].copy()
        if out.empty:
            raise KeyError(f"No quote surface rows for {ticker} on {entry_date}")
        return out

    def get_surface(self, ticker: str, entry_date: date) -> Tuple[pd.Series, pd.DataFrame]:
        """Return ``(metadata_row, quote_df)`` for *(ticker, entry_date)*."""
        return self.get_metadata(ticker, entry_date), self.get_quotes(ticker, entry_date)

def _row_to_option_quote(row: pd.Series, ticker: str, entry_date: date, expiry_date: date) -> OptionQuote:
    """Convert one quote-surface DataFrame row back into an ``OptionQuote`` domain object.

    The surface parquet stores floats/ints; this function re-wraps strikes and
    premiums as ``Decimal`` to match the ``OptionQuote`` contract exactly.
    """
    return OptionQuote(
        ticker=ticker,
        trade_date=entry_date,
        expiry_date=expiry_date,
        strike=Decimal(str(row["strike"])),
        option_type=str(row["side"]),
        bid=Decimal(str(row["bid"])),
        ask=Decimal(str(row["ask"])),
        mid=Decimal(str(row["mid"])),
        iv=float(row["iv"]),
        delta=float(row["delta"]),
        gamma=float(row["gamma"]),
        vega=float(row["vega"]),
        theta=float(row["theta"]),
        volume=int(row["volume"]),
        open_interest=int(row["open_interest"]),
    )

def _choose_nearest(df: pd.DataFrame, target_abs_delta: float) -> pd.Series:
    """Return the row in *df* whose ``abs_delta`` is closest to *target_abs_delta*.

    Tie-breaking: ``idxmin()`` returns the first minimum found, so the result
    depends on the DataFrame's row order (typically ascending strike order from
    the quote surface).  Callers should ensure *df* is sorted by strike before
    calling when deterministic tie-breaking matters.
    """
    if df.empty:
        raise ValueError("No eligible quotes available for selection")
    distances = (df["abs_delta"] - target_abs_delta).abs()
    return df.loc[distances.idxmin()]


def _choose_below_nearest(df: pd.DataFrame, target_abs_delta: float) -> pd.Series:
    """Return the row in *df* with the highest ``abs_delta`` that is still
    <= *target_abs_delta* (i.e. at least as far OTM as the target).

    Treats *target_abs_delta* as a **maximum delta threshold** rather than a
    target centre: only options with ``abs_delta <= target`` are eligible, and
    among those the one closest to the target from below is selected.  This
    guarantees the chosen wing is never more expensive (closer to the money)
    than intended.

    Raises ``ValueError`` when no eligible quotes exist after applying the
    threshold filter.
    """
    eligible = df[df["abs_delta"] <= target_abs_delta]
    if eligible.empty:
        raise ValueError(
            f"No quotes with abs_delta <= {target_abs_delta} available for selection"
        )
    return eligible.loc[eligible["abs_delta"].idxmax()]

def _build_strategy_entry_cost(strategy: OptionStrategy, fill: FillAssumption) -> Decimal:
    """Sum the signed cash flows for all legs under *fill* assumptions.

    Long legs (quantity > 0) are a cash outflow (+); short legs (quantity < 0)
    are a cash inflow (−).  For a net-credit strategy the result is negative
    (money flows *in* to the trader).
    """
    total = Decimal("0")
    for leg in strategy.legs:
        if leg.quantity > 0:    # long leg — we pay
            px = fill.buy_price(leg.option)
            total += px * abs(leg.quantity)
        else:                   # short leg — we receive
            px = fill.sell_price(leg.option)
            total -= px * abs(leg.quantity)
    return total

def _mid_entry_cost(strategy: OptionStrategy) -> Decimal:
    """Return the entry cost evaluated at mid fills.

    Explicitly computes the entry cost using ``FillAssumption.mid()`` rather
    than relying on the stored ``option.mid`` field.  The stored mid comes from
    ORATS and may not be exactly ``(bid + ask) / 2`` due to their own rounding
    or smoothing.  Using ``FillAssumption.mid()`` guarantees consistency with
    the rest of the fill model and makes the calculation self-contained.
    """
    return _build_strategy_entry_cost(strategy, FillAssumption.mid())

def _spread_cost(actual_entry_cost: Decimal, mid_entry_cost: Decimal) -> Decimal:
    """Compute the friction cost incurred by trading away from mid.

    Result is positive when the actual fill is worse than mid (which is the
    typical case — you pay the spread).  Positive for both credit and debit
    strategies.
    """
    return actual_entry_cost - mid_entry_cost

def _spread_cost_ratio(spread_cost: Decimal, net_credit: Decimal) -> Optional[float]:
    """Express spread friction as a fraction of the premium collected.

    Returns ``None`` for debit strategies (net_credit <= 0) where the ratio
    is not meaningful.  A value of 0.05 means 5 % of the premium was lost
    to bid/ask spread.
    """
    if net_credit <= 0:
        return None
    return float(abs(spread_cost) / net_credit)

def _base_diagnostics(meta: pd.Series) -> Dict[str, object]:
    """Extract the surface-level fields that every strategy diagnostic block shares."""
    return {
        "entry_spot": float(meta["entry_spot"]),
        "body_strike": float(meta["body_strike"]),
        "surface_valid": bool(meta["surface_valid"]),
        "spot_move_pct": meta.get("spot_move_pct"),
        "realized_volatility": meta.get("realized_volatility"),
    }

def build_straddle_from_surface(
    surface_db: OptionSurfaceDB,
    ticker: str,
    entry_date: date,
    direction: Literal["long", "short"],
    fill: FillAssumption = FillAssumption.mid(),
    max_leg_spread_pct: Optional[float] = None,
) -> StrategyAssemblyResult:
    """Assemble an ATM straddle from the precomputed quote surface.

    Structure: long (or short) ATM call + long (or short) ATM put at the body strike.
    Uses the body call/put quotes already selected by the surface builder.

    Parameters
    ----------
    surface_db:
        Loaded ``OptionSurfaceDB`` instance.
    ticker, entry_date:
        Identify the surface row.  Raises if ``surface_valid=False``.
    direction:
        ``"long"``  — buy call + buy put (debit, limited risk).
        ``"short"`` — sell call + sell put (credit, unlimited risk).
    fill:
        Fill assumption model.  Default is mid.
    max_leg_spread_pct:
        If set, raises when either body leg has ``spread_pct > max_leg_spread_pct``
        (liquidity filter applied before leg selection).

    Returns
    -------
    StrategyAssemblyResult with entry economics populated.

    Notes
    -----
    Risk metrics by direction:

    - Long straddle: ``max_loss_per_share = entry_cost`` (premium paid is the full
      downside). ``return_on_max_loss`` and ``leg_spread_to_credit_ratio`` are
      ``None`` — neither is meaningful for a debit structure.
    - Short straddle: ``max_loss_per_share = None`` (theoretically unlimited).
      ``leg_spread_to_credit_ratio`` is available via the property since
      ``net_credit > 0``.
    """
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    meta, quotes = surface_db.get_surface(ticker, entry_date)
    expiry_date = pd.Timestamp(meta["expiry_date"]).date()
    body_strike = Decimal(str(meta["body_strike"]))

    # ── Body legs (ATM call + put) ────────────────────────────────────────────
    body_df = quotes[quotes["is_body"]].copy()   # bool column — no == True needed
    if max_leg_spread_pct is not None:
        body_df = body_df[body_df["spread_pct"] <= max_leg_spread_pct]

    body_call_row = body_df[body_df["side"] == "call"]
    body_put_row  = body_df[body_df["side"] == "put"]

    if body_call_row.empty or body_put_row.empty:
        raise ValueError(
            f"Missing tradeable body call/put for {ticker} on {entry_date}"
        )

    body_call = _row_to_option_quote(body_call_row.iloc[0], ticker, entry_date, expiry_date)
    body_put  = _row_to_option_quote(body_put_row.iloc[0],  ticker, entry_date, expiry_date)

    # ── Strategy object ───────────────────────────────────────────────────────
    # qty=+1 for long (buy), qty=-1 for short (sell); same sign on both legs.
    qty = 1 if direction == "long" else -1

    strategy = OptionStrategy(
        ticker=ticker,
        strategy_type=StrategyType.STRADDLE,
        legs=(
            OptionLeg(option=body_call, quantity=qty),
            OptionLeg(option=body_put,  quantity=qty),
        ),
        trade_date=entry_date,
    )

    # ── Entry economics ───────────────────────────────────────────────────────
    entry_cost     = _build_strategy_entry_cost(strategy, fill)
    entry_cost_mid = _mid_entry_cost(strategy)
    # net_credit is positive for short straddle (credit received),
    # negative for long straddle (debit paid).
    net_credit  = -entry_cost
    spread_cost = _spread_cost(entry_cost, entry_cost_mid)

    if direction == "long":
        # Long straddle: max loss is the premium paid (entry_cost is always > 0 here).
        # abs() is used defensively to guarantee a non-negative Decimal regardless of
        # any unexpected sign edge-cases in the fill model.
        max_loss_per_share = abs(entry_cost)
        return_on_max_loss = None   # unlimited upside — ROC is not a meaningful bound
        # Normalise spread friction against the mid debit (no net_credit available).
        spread_cost_ratio = (
            float(abs(spread_cost) / abs(entry_cost_mid))
            if entry_cost_mid != 0
            else None
        )
    else:
        # Short straddle: unlimited downside — max loss cannot be stated per share.
        max_loss_per_share = None
        return_on_max_loss = None
        # Normalise spread friction against the credit received.
        spread_cost_ratio = (
            float(abs(spread_cost) / net_credit)
            if net_credit > 0
            else None
        )

    # ── Diagnostics ──────────────────────────────────────────────────────────
    diagnostics = _base_diagnostics(meta)
    diagnostics.update({
        "direction":      direction,
        "fill_model":     fill.label,
        "call_strike":    float(body_call.strike),
        "put_strike":     float(body_put.strike),
        "call_abs_delta": abs(body_call.delta),
        "put_abs_delta":  abs(body_put.delta),
        "call_spread_pct": float((body_call.ask - body_call.bid) / body_call.mid) if body_call.mid > 0 else None,
        "put_spread_pct":  float((body_put.ask  - body_put.bid)  / body_put.mid)  if body_put.mid  > 0 else None,
    })

    return StrategyAssemblyResult(
        strategy_name=f"{direction}_straddle",
        ticker=ticker,
        entry_date=entry_date,
        expiry_date=expiry_date,
        entry_spot=Decimal(str(meta["entry_spot"])),
        body_strike=body_strike,
        strategy=strategy,
        entry_cost=entry_cost,
        entry_cost_mid=entry_cost_mid,
        net_credit=net_credit,
        max_loss_per_share=max_loss_per_share,
        return_on_max_loss=return_on_max_loss,
        spread_cost=spread_cost,
        spread_cost_ratio=spread_cost_ratio,
        diagnostics=diagnostics,
    )

def build_ironfly_from_surface(
    surface_db: OptionSurfaceDB,
    ticker: str,
    entry_date: date,
    wing_target_delta: float,
    fill: FillAssumption = FillAssumption.mid(),
    max_leg_spread_pct: Optional[float] = None,
    max_spread_cost_ratio: Optional[float] = None,
) -> StrategyAssemblyResult:
    """Assemble a short iron butterfly from the precomputed quote surface.

    Structure: long OTM put / short ATM put / short ATM call / long OTM call.
    The ATM body strike is taken from the surface metadata (pre-selected as the
    strike nearest to entry spot).  The OTM wings are chosen by finding the
    call and put whose ``abs_delta`` is closest to *wing_target_delta*.

    Parameters
    ----------
    surface_db:
        Loaded ``OptionSurfaceDB`` instance.
    ticker, entry_date:
        Identify the surface row.  Raises if ``surface_valid=False``.
    wing_target_delta:
        Maximum absolute delta allowed for both OTM long legs (e.g. ``0.15`` selects
        the wing with the highest delta that is still ≤ 0.15, i.e. at least as far
        OTM as a 15-delta option).
    fill:
        Fill assumption model.  Default is mid; use ``FillAssumption.cross()``
        for a conservative market-order model.
    max_leg_spread_pct:
        If set, OTM quotes whose ``spread_pct > max_leg_spread_pct`` are excluded
        before wing selection (liquidity filter).
    max_spread_cost_ratio:
        If set, raises ``ValueError`` when the assembled strategy's
        ``spread_cost_ratio`` exceeds this threshold.

    Returns
    -------
    StrategyAssemblyResult with all entry economics populated.
    """
    meta, quotes = surface_db.get_surface(ticker, entry_date)
    expiry_date  = pd.Timestamp(meta["expiry_date"]).date()
    body_strike  = Decimal(str(meta["body_strike"]))

    # ── Body legs (ATM short straddle core) ───────────────────────────────────
    body_df       = quotes[quotes["is_body"]]          # bool column — no == True needed
    body_call_row = body_df[body_df["side"] == "call"]
    body_put_row  = body_df[body_df["side"] == "put"]
    if body_call_row.empty or body_put_row.empty:
        raise ValueError(f"Missing body call/put for {ticker} on {entry_date}")

    body_call = _row_to_option_quote(body_call_row.iloc[0], ticker, entry_date, expiry_date)
    body_put  = _row_to_option_quote(body_put_row.iloc[0],  ticker, entry_date, expiry_date)

    # ── OTM wing candidates ───────────────────────────────────────────────────
    otm_calls = quotes[(quotes["side"] == "call") & quotes["is_otm"]]
    otm_puts  = quotes[(quotes["side"] == "put")  & quotes["is_otm"]]

    # Optional per-leg liquidity filter before wing selection
    if max_leg_spread_pct is not None:
        otm_calls = otm_calls[otm_calls["spread_pct"] <= max_leg_spread_pct]
        otm_puts  = otm_puts[otm_puts["spread_pct"]  <= max_leg_spread_pct]

    long_call_row = _choose_below_nearest(otm_calls, wing_target_delta)
    long_put_row  = _choose_below_nearest(otm_puts,  wing_target_delta)

    long_call = _row_to_option_quote(long_call_row, ticker, entry_date, expiry_date)
    long_put  = _row_to_option_quote(long_put_row,  ticker, entry_date, expiry_date)

    # ── Strategy object (legs ordered put-side → call-side by convention) ─────
    strategy = OptionStrategy(
        ticker=ticker,
        strategy_type=StrategyType.IRON_BUTTERFLY,
        legs=(
            OptionLeg(option=long_put,  quantity=1),    # long OTM put  (wing)
            OptionLeg(option=body_put,  quantity=-1),   # short ATM put  (body)
            OptionLeg(option=body_call, quantity=-1),   # short ATM call (body)
            OptionLeg(option=long_call, quantity=1),    # long OTM call (wing)
        ),
        trade_date=entry_date,
    )

    # ── Entry economics ───────────────────────────────────────────────────────
    entry_cost     = _build_strategy_entry_cost(strategy, fill)  # negative = credit
    entry_cost_mid = _mid_entry_cost(strategy)
    net_credit     = -entry_cost   # positive = credit received

    # Wing width: use the wider side — the max loss is bounded by whichever wing
    # is further from the body, since the spot can move through that full distance.
    wing_width        = max(long_call.strike - body_strike, body_strike - long_put.strike)
    max_loss_per_share = wing_width - net_credit
    if max_loss_per_share <= 0:
        max_loss_per_share = Decimal("0")
        roc = None
    else:
        roc = float(net_credit / max_loss_per_share)

    spread_cost       = _spread_cost(entry_cost, entry_cost_mid)
    spread_cost_ratio = _spread_cost_ratio(spread_cost, net_credit)
    if max_spread_cost_ratio is not None and spread_cost_ratio is not None and spread_cost_ratio > max_spread_cost_ratio:
        raise ValueError(
            f"Iron fly spread_cost_ratio={spread_cost_ratio:.4f} exceeds "
            f"threshold {max_spread_cost_ratio:.4f}"
        )

    diagnostics = _base_diagnostics(meta)
    diagnostics.update({
        "wing_target_delta":      wing_target_delta,
        "actual_call_abs_delta":  abs(long_call.delta),
        "actual_put_abs_delta":   abs(long_put.delta),
        "long_call_strike":       float(long_call.strike),
        "long_put_strike":        float(long_put.strike),
        "wing_width":             float(wing_width),
        "fill_model":             fill.label,
    })

    return StrategyAssemblyResult(
        strategy_name="iron_fly",
        ticker=ticker,
        entry_date=entry_date,
        expiry_date=expiry_date,
        entry_spot=Decimal(str(meta["entry_spot"])),
        body_strike=body_strike,
        strategy=strategy,
        entry_cost=entry_cost,
        entry_cost_mid=entry_cost_mid,
        net_credit=net_credit,
        max_loss_per_share=max_loss_per_share,
        return_on_max_loss=roc,
        spread_cost=spread_cost,
        spread_cost_ratio=spread_cost_ratio,
        diagnostics=diagnostics,
    )

def build_ironcondor_from_surface(
    surface_db: OptionSurfaceDB,
    ticker: str,
    entry_date: date,
    short_delta_target: float,
    long_delta_target: float,
    fill: FillAssumption = FillAssumption.mid(),
    short_call_delta_target: Optional[float] = None,
    short_put_delta_target: Optional[float] = None,
    long_call_delta_target: Optional[float] = None,
    long_put_delta_target: Optional[float] = None,
    max_leg_spread_pct: Optional[float] = None,
    max_spread_cost_ratio: Optional[float] = None,
) -> StrategyAssemblyResult:
    """Assemble a short iron condor from the precomputed quote surface.

    Structure: long OTM put / short nearer-OTM put / short nearer-OTM call / long OTM call.
    Unlike an iron fly, the short legs are *not* at ATM — they are positioned at
    a higher-delta (closer-to-money) OTM strike.

    Delta targeting
    ---------------
    ``short_delta_target`` and ``long_delta_target`` set the symmetric defaults
    for both sides.  Per-leg overrides (``short_call_delta_target``, etc.) allow
    asymmetric positioning when the vol surface is skewed.

    The short legs include the ATM body quote as a candidate so the condor can
    express views right at the money if desired.

    Wing constraint
    ---------------
    Long legs are constrained to strikes *further OTM* than the selected short
    leg (``long_call.strike > short_call.strike`` and vice versa) before the
    nearest-delta selection runs.  This prevents the long from inadvertently
    being chosen inside the spread.

    Parameters
    ----------
    short_delta_target:
        Abs-delta target for both short legs (default for call and put sides).
    long_delta_target:
        Abs-delta target for both long wing legs (should be < short_delta_target).
    short_call/put_delta_target, long_call/put_delta_target:
        Per-leg overrides for asymmetric condors.  Fall back to the symmetric
        targets when ``None``.
    max_leg_spread_pct:
        Pre-filter: exclude quotes with bid-ask spread > this threshold.
    max_spread_cost_ratio:
        Raises ``ValueError`` if the assembled condor's spread_cost_ratio
        exceeds this threshold.
    """
    meta, quotes = surface_db.get_surface(ticker, entry_date)
    expiry_date  = pd.Timestamp(meta["expiry_date"]).date()
    body_strike  = Decimal(str(meta["body_strike"]))

    # Apply per-leg delta overrides, falling back to symmetric targets
    short_call_delta_target = short_call_delta_target or short_delta_target
    short_put_delta_target  = short_put_delta_target  or short_delta_target
    long_call_delta_target  = long_call_delta_target  or long_delta_target
    long_put_delta_target   = long_put_delta_target   or long_delta_target

    # Include body (ATM) quotes as short-leg candidates so the condor can be
    # positioned at-the-money if the delta target points there.
    otm_calls = quotes[(quotes["side"] == "call") & (quotes["is_otm"] | quotes["is_body"])]
    otm_puts  = quotes[(quotes["side"] == "put")  & (quotes["is_otm"] | quotes["is_body"])]

    # Optional per-leg liquidity filter
    if max_leg_spread_pct is not None:
        otm_calls = otm_calls[otm_calls["spread_pct"] <= max_leg_spread_pct]
        otm_puts  = otm_puts[otm_puts["spread_pct"]  <= max_leg_spread_pct]

    # Select short legs first
    short_call_row = _choose_nearest(otm_calls, short_call_delta_target)
    short_put_row  = _choose_nearest(otm_puts,  short_put_delta_target)

    # Long legs must be further OTM than the selected short legs
    long_call_candidates = otm_calls[otm_calls["strike"] > float(short_call_row["strike"])]
    long_put_candidates  = otm_puts[otm_puts["strike"]   < float(short_put_row["strike"])]
    if long_call_candidates.empty:
        raise ValueError("No further-OTM call wing available for condor")
    if long_put_candidates.empty:
        raise ValueError("No further-OTM put wing available for condor")

    long_call_row = _choose_below_nearest(long_call_candidates, long_call_delta_target)
    long_put_row  = _choose_below_nearest(long_put_candidates,  long_put_delta_target)

    short_call = _row_to_option_quote(short_call_row, ticker, entry_date, expiry_date)
    short_put  = _row_to_option_quote(short_put_row,  ticker, entry_date, expiry_date)
    long_call  = _row_to_option_quote(long_call_row,  ticker, entry_date, expiry_date)
    long_put   = _row_to_option_quote(long_put_row,   ticker, entry_date, expiry_date)

    # ── Strategy object (legs ordered put-side → call-side by convention) ─────
    strategy = OptionStrategy(
        ticker=ticker,
        strategy_type=StrategyType.IRON_CONDOR,
        legs=(
            OptionLeg(option=long_put,   quantity=1),   # long OTM put    (outer wing)
            OptionLeg(option=short_put,  quantity=-1),  # short nearer put (inner short)
            OptionLeg(option=short_call, quantity=-1),  # short nearer call (inner short)
            OptionLeg(option=long_call,  quantity=1),   # long OTM call   (outer wing)
        ),
        trade_date=entry_date,
    )

    # ── Entry economics ───────────────────────────────────────────────────────
    entry_cost     = _build_strategy_entry_cost(strategy, fill)  # negative = credit
    entry_cost_mid = _mid_entry_cost(strategy)
    net_credit     = -entry_cost

    # Max loss = wider of the two vertical spreads minus the net credit received.
    # Using max (not min) because the wider spread determines the worst-case payout.
    call_spread_width  = long_call.strike - short_call.strike
    put_spread_width   = short_put.strike - long_put.strike
    max_width          = max(call_spread_width, put_spread_width)
    max_loss_per_share = max_width - net_credit
    if max_loss_per_share <= 0:
        max_loss_per_share = Decimal("0")
        roc = None
    else:
        roc = float(net_credit / max_loss_per_share)

    spread_cost       = _spread_cost(entry_cost, entry_cost_mid)
    spread_cost_ratio = _spread_cost_ratio(spread_cost, net_credit)
    if max_spread_cost_ratio is not None and spread_cost_ratio is not None and spread_cost_ratio > max_spread_cost_ratio:
        raise ValueError(
            f"Iron condor spread_cost_ratio={spread_cost_ratio:.4f} exceeds "
            f"threshold {max_spread_cost_ratio:.4f}"
        )

    diagnostics = _base_diagnostics(meta)
    diagnostics.update({
        "short_call_delta_target":      short_call_delta_target,
        "short_put_delta_target":       short_put_delta_target,
        "long_call_delta_target":       long_call_delta_target,
        "long_put_delta_target":        long_put_delta_target,
        "actual_short_call_abs_delta":  abs(short_call.delta),
        "actual_short_put_abs_delta":   abs(short_put.delta),
        "actual_long_call_abs_delta":   abs(long_call.delta),
        "actual_long_put_abs_delta":    abs(long_put.delta),
        "short_call_strike":            float(short_call.strike),
        "short_put_strike":             float(short_put.strike),
        "long_call_strike":             float(long_call.strike),
        "long_put_strike":              float(long_put.strike),
        "call_spread_width":            float(call_spread_width),
        "put_spread_width":             float(put_spread_width),
        "width_asymmetry":              float(abs(call_spread_width - put_spread_width)),
        "fill_model":                   fill.label,
    })

    return StrategyAssemblyResult(
        strategy_name="iron_condor",
        ticker=ticker,
        entry_date=entry_date,
        expiry_date=expiry_date,
        entry_spot=Decimal(str(meta["entry_spot"])),
        body_strike=body_strike,
        strategy=strategy,
        entry_cost=entry_cost,
        entry_cost_mid=entry_cost_mid,
        net_credit=net_credit,
        max_loss_per_share=max_loss_per_share,
        return_on_max_loss=roc,
        spread_cost=spread_cost,
        spread_cost_ratio=spread_cost_ratio,
        diagnostics=diagnostics,
    )
