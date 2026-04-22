"""Option surface precompute utilities.

This module builds a reusable expiry-level quote surface for one (ticker, entry_date)
pair. The output is intentionally more primitive than the iron fly / condor candidate
history files:
- one metadata row per (ticker, entry_date)
- many quote-surface rows per (ticker, entry_date, expiry_date, strike, side)

The goal is to support flexible downstream assembly of:
- ATM short straddles
- iron butterflies
- iron condors

without re-reading the raw ORATS parquet files.
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

from ..data.orats_provider import ORATSDataProvider
from ..data.spot_price_db import SpotPriceDB
from ..core.models import OptionQuote

logger = logging.getLogger(__name__)


def _nearest_bucket(value: float, buckets: Sequence[float]) -> Optional[float]:
    """Return the element of *buckets* whose distance to *value* is smallest.

    Assigns each option quote to its nearest pre-defined delta level (e.g.
    0.10, 0.15, 0.20 …) so that cross-sectional comparisons use a consistent
    delta grid.  On a tie the smaller bucket wins because ``min()`` returns
    the first minimum found and callers are expected to pass *buckets* sorted
    ascending.

    Returns ``None`` when *buckets* is empty.
    """
    if not buckets:
        return None
    return min(buckets, key=lambda b: abs(value - b))


def _metadata_failure_row(
    ticker: str,
    entry_date: date,
    dte_target: int,
    frequency: str,
    failure_reason: str,
    processing_time: float,
) -> Dict:
    """Build a schema-compatible metadata row recording a surface build failure.

    All numeric/date fields are set to ``None`` / ``False`` / ``0`` so the row
    can coexist with success rows in the same output table.  The
    ``failure_reason`` field carries a short snake_case tag (e.g.
    ``'no_spot_price'``, ``'no_expiry_found'``) for easy downstream filtering.
    """
    return {
        "ticker": ticker,
        "entry_date": entry_date,
        "frequency": frequency,
        "dte_target": dte_target,
        "dte_actual": None,
        "expiry_date": None,
        "entry_spot": None,
        "exit_spot": None,
        "body_strike": None,
        "spot_move_pct": None,
        "realized_volatility": None,
        "has_body_call": False,
        "has_body_put": False,
        "n_surface_quotes": 0,
        "surface_valid": False,
        "failure_reason": failure_reason,
        "processing_time": processing_time,
    }


def _metadata_success_row(
    ticker: str,
    entry_date: date,
    expiry_date: date,
    dte_target: int,
    frequency: str,
    entry_spot: Decimal,
    exit_spot: Decimal,
    body_strike: Decimal,
    spot_move_pct: Optional[float],
    realized_volatility: Optional[float],
    has_body_call: bool,
    has_body_put: bool,
    n_surface_quotes: int,
    processing_time: float,
) -> Dict:
    """Build a metadata row for a successfully built option surface.

    ``surface_valid`` is ``True`` only when both body legs are present with
    positive premiums and at least one quote row was produced.  This flag is
    the primary filter for downstream strategy assembly — rows where
    ``surface_valid=False`` may have partial data and should be excluded.

    ``spot_move_pct`` is stored as a percentage (multiplied by 100) so values
    are on a human-readable scale in downstream analysis.
    """
    return {
        "ticker": ticker,
        "entry_date": entry_date,
        "frequency": frequency,
        "dte_target": dte_target,
        "dte_actual": (expiry_date - entry_date).days,
        "expiry_date": expiry_date,
        "entry_spot": float(entry_spot),
        "exit_spot": float(exit_spot),
        "body_strike": float(body_strike),
        "spot_move_pct": spot_move_pct * 100 if spot_move_pct is not None else None,
        "realized_volatility": realized_volatility,
        "has_body_call": bool(has_body_call),
        "has_body_put": bool(has_body_put),
        "n_surface_quotes": int(n_surface_quotes),
        "surface_valid": bool(has_body_call and has_body_put and n_surface_quotes > 0),
        "failure_reason": None,
        "processing_time": processing_time,
    }


class OptionSurfaceBuilder:
    """Build a reusable quote surface for one (ticker, entry_date) pair."""

    def __init__(
        self,
        data_root: str,
        spot_db: SpotPriceDB,
        dte_target: int = 7,
        frequency: str = "weekly",
        min_abs_delta: float = 0.03,
        max_abs_delta: float = 0.45,
        delta_buckets: Optional[Sequence[float]] = None,
        keep_zero_bid_quotes: bool = False,
    ):
        """Initialise the surface builder.

        Parameters
        ----------
        data_root:
            Root directory of the ORATS adjusted parquet files.
        spot_db:
            Pre-loaded spot-price database used for entry/exit spot lookup and
            realized-volatility calculations.
        dte_target:
            Target days-to-expiry.  Values ≥ 28 trigger the monthly expiry
            selection path; smaller values use the weekly path.  Keep this
            in sync with *frequency*.
        frequency:
            ``'monthly'`` or ``'weekly'``.  Stored for output metadata
            labelling; actual expiry selection logic is driven by *dte_target*,
            so the two parameters must be kept consistent.
        min_abs_delta / max_abs_delta:
            Absolute-delta range filter for OTM wing quotes.  Body quotes at
            *body_strike* are always included regardless of delta.
        delta_buckets:
            Ordered reference delta levels used to assign each OTM quote to its
            nearest bucket.  Defaults to
            ``[0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40]``.
        keep_zero_bid_quotes:
            When ``True``, retains quotes with zero bid/ask/mid.  Useful for
            coverage audits on illiquid names, but these quotes must be excluded
            before live strategy assembly.
        """
        if frequency not in ("monthly", "weekly"):
            raise ValueError(f"frequency must be 'monthly' or 'weekly', got {frequency!r}")
        if not (0.0 <= min_abs_delta <= max_abs_delta <= 1.0):
            raise ValueError(
                "Require 0 <= min_abs_delta <= max_abs_delta <= 1.0, "
                f"got {min_abs_delta}, {max_abs_delta}"
            )

        self.data_root = data_root
        self.spot_db = spot_db
        self.dte_target = dte_target
        self.frequency = frequency
        self.min_abs_delta = min_abs_delta
        self.max_abs_delta = max_abs_delta
        self.delta_buckets = list(delta_buckets) if delta_buckets is not None else [
            0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40
        ]
        self.keep_zero_bid_quotes = keep_zero_bid_quotes

        self.provider: Optional[ORATSDataProvider] = None

    def _init_worker_components(self) -> None:
        """Lazily create the ``ORATSDataProvider`` on first use.

        The provider is not created in ``__init__`` because this class is
        designed to be pickled and dispatched to worker processes (e.g. via
        ``joblib.Parallel``).  Creating I/O-heavy objects before serialisation
        causes errors on some platforms.  Each worker calls this method once
        before its first ``process_single_entry`` invocation.
        """
        if self.provider is None:
            self.provider = ORATSDataProvider(
                data_root=self.data_root,
                min_volume=0,
                min_open_interest=0,
                min_bid=0.0,
                max_spread_pct=999.0,
            )
            logger.info("Worker initialised OptionSurfaceBuilder with permissive provider")

    def _find_best_expiry(
        self,
        ticker: str,
        trade_date: date,
        target_dte: int,
        tolerance_days: int = 4,
    ) -> Optional[date]:
        """Locate the most appropriate expiry for *ticker* near *target_dte*.

        Two selection strategies are used based on *target_dte*:

        **Monthly (target_dte ≥ 28):**
            Looks for a Thursday or Friday expiry in the calendar month
            immediately following *trade_date* (standard monthly options expiry
            week).  Falls back to the closest forward expiry within
            *tolerance_days* of *target_dte* if no standard monthly expiry
            exists.

        **Weekly (target_dte < 28):**
            Collects all expiries within *tolerance_days* of *target_dte* and
            prefers Fridays, then Thursdays, then the closest DTE match.

        .. note::
            The branching uses ``target_dte >= 28``, **not** ``self.frequency``.
            A ``frequency='weekly'`` builder with ``dte_target=30`` will follow
            the monthly path.  Keep *dte_target* and *frequency* in sync.

        Returns ``None`` on any exception or when no suitable expiry is found.
        """
        try:
            expiries = self.provider.get_available_expiries(ticker, trade_date)
            if not expiries:
                return None

            if target_dte >= 28:
                # ── Monthly expiry path ────────────────────────────────────────
                # Identify the calendar month immediately after trade_date so we
                # can target the standard monthly options expiry cycle.
                next_month = trade_date.month + 1 if trade_date.month < 12 else 1
                next_year = trade_date.year if trade_date.month < 12 else trade_date.year + 1

                # Standard monthly options expire on Thursday or Friday of the
                # options-expiration week (weekday 3 = Thursday, 4 = Friday).
                target_month_expiries = [
                    exp for exp in expiries
                    if exp.year == next_year
                    and exp.month == next_month
                    and exp.weekday() in (3, 4)
                    and exp >= trade_date
                ]

                if not target_month_expiries:
                    # Fallback: no Thu/Fri found in the next calendar month.
                    # Accept any forward expiry within tolerance_days of target_dte.
                    expiry_diffs = [
                        (exp, abs((exp - trade_date).days - target_dte))
                        for exp in expiries
                        if exp >= trade_date
                    ]
                    if expiry_diffs:
                        best_expiry, diff = min(expiry_diffs, key=lambda x: x[1])
                        if diff <= tolerance_days:
                            return best_expiry
                    return None

                # Return the earliest standard monthly expiry in the target month.
                target_month_expiries.sort()
                return target_month_expiries[0]

            # ── Weekly expiry path ─────────────────────────────────────────────
            # Collect (expiry, dte) pairs that are strictly forward-dated and
            # fall within tolerance_days of the target DTE.
            expiry_dtes = [(exp, (exp - trade_date).days) for exp in expiries]
            valid = [
                (exp, dte) for exp, dte in expiry_dtes
                if dte > 0 and abs(dte - target_dte) <= tolerance_days
            ]
            if not valid:
                return None

            # Prefer Friday expirations (standard weekly expiry day), then
            # Thursday (used by some underlyings), then the closest DTE.
            friday_expiries = [(exp, dte) for exp, dte in valid if exp.weekday() == 4]
            if friday_expiries:
                return min(friday_expiries, key=lambda x: abs(x[1] - target_dte))[0]

            thursday_expiries = [(exp, dte) for exp, dte in valid if exp.weekday() == 3]
            if thursday_expiries:
                return min(thursday_expiries, key=lambda x: abs(x[1] - target_dte))[0]

            return min(valid, key=lambda x: abs(x[1] - target_dte))[0]
        except Exception as exc:
            logger.error(f"Error finding expiry for {ticker} on {trade_date}: {exc}")
            return None

    def _quote_rows(
        self,
        ticker: str,
        entry_date: date,
        expiry_date: date,
        entry_spot: Decimal,
        body_strike: Decimal,
        chain: Sequence[OptionQuote],
    ) -> List[Dict]:
        """Flatten raw option quotes into surface rows ready for storage.

        Inclusion rules
        ---------------
        * **Body (ATM)** — call and put at *body_strike* are always included
          (subject to the zero-bid filter below).
        * **OTM wings** — calls above *body_strike* and puts below are included
          when ``abs_delta`` falls within ``[min_abs_delta, max_abs_delta]``.
        * **ITM options** are excluded entirely; they are not needed for any
          of the target strategies (straddle, iron fly, iron condor).

        Each row carries all raw quote fields plus derived convenience columns
        (``spread_pct``, ``moneyness``, ``nearest_delta_bucket``, etc.) so
        that downstream code can work from this surface file alone without
        re-reading the raw ORATS parquet files.

        Returns an empty list when no quotes survive the inclusion filters.
        """
        rows: List[Dict] = []

        for quote in chain:
            # ── Classify the quote relative to the body strike ─────────────────
            is_body = quote.strike == body_strike
            is_otm_call = quote.option_type == "call" and quote.strike > body_strike
            is_otm_put = quote.option_type == "put" and quote.strike < body_strike

            # Exclude ITM options — they are not required by any target strategy.
            if not (is_body or is_otm_call or is_otm_put):
                continue

            # ── Liquidity filter ───────────────────────────────────────────────
            # Drop quotes with a non-positive bid/ask/mid unless the caller has
            # explicitly opted into retaining them for coverage audits.
            if not self.keep_zero_bid_quotes and (quote.bid <= 0 or quote.ask <= 0 or quote.mid <= 0):
                continue

            # ── Delta-range filter (OTM wings only) ───────────────────────────
            # Body quotes are exempt; wing quotes must fall within the configured
            # absolute-delta window to keep the surface tightly focused.
            abs_delta = abs(float(quote.delta))
            if not is_body and not (self.min_abs_delta <= abs_delta <= self.max_abs_delta):
                continue

            # ── Derived convenience fields ─────────────────────────────────────
            # Assign to the nearest pre-defined delta bucket so cross-sectional
            # comparisons use a consistent delta grid (body quotes get None).
            nearest_bucket = None if is_body else _nearest_bucket(abs_delta, self.delta_buckets)
            strike_distance = float(quote.strike - body_strike)
            spread = float(quote.ask - quote.bid)
            # spread_pct: bid/ask spread as a fraction of mid-price.  The
            # mid > 0 guard is redundant when keep_zero_bid_quotes=False (zero-mid
            # quotes are already filtered above), but is retained for correctness
            # when the flag is True.
            spread_pct = float((quote.ask - quote.bid) / quote.mid) if quote.mid > 0 else None
            moneyness = float(quote.strike / entry_spot) if entry_spot > 0 else None

            rows.append({
                "ticker": ticker,
                "entry_date": entry_date,
                "expiry_date": expiry_date,
                "entry_spot": float(entry_spot),
                "body_strike": float(body_strike),
                "side": quote.option_type,
                "is_body": bool(is_body),
                "is_otm": bool(not is_body),
                "strike": float(quote.strike),
                "strike_distance_from_body": strike_distance,
                "abs_strike_distance_from_body": abs(strike_distance),
                "moneyness": moneyness,
                "bid": float(quote.bid),
                "ask": float(quote.ask),
                "mid": float(quote.mid),
                "spread": spread,
                "spread_pct": spread_pct,
                "iv": float(quote.iv),
                "delta": float(quote.delta),
                "abs_delta": abs_delta,
                "gamma": float(quote.gamma),
                "vega": float(quote.vega),
                "theta": float(quote.theta),
                "volume": int(quote.volume),
                "open_interest": int(quote.open_interest),
                "nearest_delta_bucket": nearest_bucket,
                "delta_bucket_distance": None if nearest_bucket is None else abs(abs_delta - nearest_bucket),
            })
        return rows

    def process_single_entry(
        self,
        ticker: str,
        entry_date: date,
    ) -> Tuple[Dict, List[Dict]]:
        """Build the option surface for one (ticker, entry_date) observation.

        Runs a linear pipeline of seven stages.  Any early-exit failure
        produces a schema-compatible metadata row with a populated
        ``failure_reason`` field and an empty quote list.

        Pipeline stages
        ---------------
        1. Resolve entry spot price.
        2. Find the best-matching expiry for the configured DTE target.
        3. Load the full option chain for (ticker, entry_date, expiry_date).
        4. Identify the ATM body strike (nearest strike to entry spot;
           ties broken by choosing the lower strike).
        5. Resolve exit spot price at expiry (required for P&L metadata).
        6. Flatten the chain into surface quote rows via ``_quote_rows``.
        7. Assemble the metadata summary row.

        Returns
        -------
        (metadata_dict, quote_rows_list)
            *metadata_dict* has ``surface_valid=True`` only when all stages
            succeed and at least one quote row is produced.
        """
        started = datetime.now()
        self._init_worker_components()

        def elapsed() -> float:
            return (datetime.now() - started).total_seconds()

        # ── Stage 1: entry spot ────────────────────────────────────────────────
        entry_spot_raw = self.provider.get_spot_price(ticker, entry_date)
        if entry_spot_raw is None:
            return _metadata_failure_row(
                ticker=ticker,
                entry_date=entry_date,
                dte_target=self.dte_target,
                frequency=self.frequency,
                failure_reason="no_spot_price",
                processing_time=elapsed(),
            ), []

        # ── Stage 2: expiry selection ──────────────────────────────────────────
        expiry_date = self._find_best_expiry(ticker, entry_date, self.dte_target)
        if expiry_date is None:
            return _metadata_failure_row(
                ticker=ticker,
                entry_date=entry_date,
                dte_target=self.dte_target,
                frequency=self.frequency,
                failure_reason="no_expiry_found",
                processing_time=elapsed(),
            ), []

        # ── Stage 3: option chain ──────────────────────────────────────────────
        chain = self.provider.get_option_chain(
            ticker=ticker,
            trade_date=entry_date,
            expiry_date=expiry_date,
        )
        if not chain:
            return _metadata_failure_row(
                ticker=ticker,
                entry_date=entry_date,
                dte_target=self.dte_target,
                frequency=self.frequency,
                failure_reason="no_options_at_entry",
                processing_time=elapsed(),
            ), []

        # ── Stage 4: body strike ───────────────────────────────────────────────
        # Select the strike closest to the entry spot.  Ties are broken by
        # choosing the lower strike (standard round-down convention).
        entry_spot = Decimal(str(entry_spot_raw))
        strikes = sorted({q.strike for q in chain})
        if not strikes:
            return _metadata_failure_row(
                ticker=ticker,
                entry_date=entry_date,
                dte_target=self.dte_target,
                frequency=self.frequency,
                failure_reason="no_strikes_in_chain",
                processing_time=elapsed(),
            ), []

        body_strike = min(strikes, key=lambda s: (abs(s - entry_spot), s))
        body_call = next((q for q in chain if q.strike == body_strike and q.option_type == "call"), None)
        body_put = next((q for q in chain if q.strike == body_strike and q.option_type == "put"), None)

        # ── Stage 5: exit spot ─────────────────────────────────────────────────
        # Used to compute the realized spot move and volatility in the metadata
        # row.  The surface quotes are still analytically useful without an exit
        # spot, but the current schema requires it for row completeness.
        exit_spot_raw = self.provider.get_spot_price(ticker, expiry_date)
        if exit_spot_raw is None:
            return _metadata_failure_row(
                ticker=ticker,
                entry_date=entry_date,
                dte_target=self.dte_target,
                frequency=self.frequency,
                failure_reason="no_spot_at_expiry",
                processing_time=elapsed(),
            ), []
        exit_spot = Decimal(str(exit_spot_raw))

        # ── Stage 6: flatten quote rows ────────────────────────────────────────
        quote_rows = self._quote_rows(
            ticker=ticker,
            entry_date=entry_date,
            expiry_date=expiry_date,
            entry_spot=entry_spot,
            body_strike=body_strike,
            chain=chain,
        )

        # ── Stage 7: metadata row ──────────────────────────────────────────────
        metadata = _metadata_success_row(
            ticker=ticker,
            entry_date=entry_date,
            expiry_date=expiry_date,
            dte_target=self.dte_target,
            frequency=self.frequency,
            entry_spot=entry_spot,
            exit_spot=exit_spot,
            body_strike=body_strike,
            spot_move_pct=self.spot_db.calculate_spot_move_pct(ticker, entry_date, expiry_date),
            realized_volatility=self.spot_db.calculate_realized_volatility(ticker, entry_date, expiry_date),
            has_body_call=body_call is not None and body_call.bid > 0 and body_call.ask > 0 and body_call.mid > 0,
            has_body_put=body_put is not None and body_put.bid > 0 and body_put.ask > 0 and body_put.mid > 0,
            n_surface_quotes=len(quote_rows),
            processing_time=elapsed(),
        )
        return metadata, quote_rows
