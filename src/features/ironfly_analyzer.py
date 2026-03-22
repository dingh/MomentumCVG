"""
Iron butterfly history builder: enumerate all wing candidates per entry date.

Unlike ``StraddleHistoryBuilder``, which records one row per (ticker, date),
this module records **one row per (ticker, date, wing_width)**.  Every
symmetric wing pair that passes the (deliberately permissive) quality filters
is stored with full entry economics and realized P&L at expiry.

The resulting parquet is the raw material for delta-sweep analysis:

    >>> df = pd.read_parquet('cache/ironfly_history_monthly_2018_2024.parquet')
    >>> tradeable = df[df['is_tradeable']]
    >>> # Best avg_wing_delta per year
    >>> tradeable.groupby(tradeable['entry_date'].dt.year)['avg_wing_delta'].describe()

Typical usage (via ``precompute_ironfly_history.py``):

    >>> builder = IronFlyHistoryBuilder(
    ...     data_root='C:/ORATS/data/ORATS_Adjusted',
    ...     spot_db=spot_db,
    ...     dte_target=30,
    ...     frequency='monthly',
    ... )
    >>> rows = builder.process_single_entry('AAPL', date(2023, 1, 6))
    >>> # Returns a list — multiple rows for each wing width candidate
"""

import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.orats_provider import ORATSDataProvider
from ..data.spot_price_db import SpotPriceDB
from ..strategy.builders import IronButterflyBuilder, IronButterflyCandidate
from ..core.models import OptionLeg, OptionStrategy, StrategyType, Position


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: build a minimal failure row
# ---------------------------------------------------------------------------

def _failure_row(
    ticker: str,
    entry_date: date,
    dte_target: int,
    frequency: str,
    failure_reason: str,
    processing_time: float,
) -> Dict:
    """Return a single-row failure dict with all candidate fields set to None."""
    return {
        # Identity
        'ticker': ticker,
        'entry_date': entry_date,
        'dte_category': frequency,
        'dte_target': dte_target,
        'dte_actual': None,
        'expiry_date': None,
        'entry_spot': None,
        'body_strike': None,
        # Candidate geometry
        'wing_width': None,
        'call_wing_strike': None,
        'put_wing_strike': None,
        'avg_wing_delta': None,
        # Entry economics
        'net_credit': None,
        'credit_to_width': None,
        'total_spread': None,
        'spread_cost_ratio': None,
        # Entry greeks
        'net_delta': None,
        'net_gamma': None,
        'net_vega': None,
        'net_theta': None,
        # Short call leg
        'sc_bid': None,
        'sc_ask': None,
        'sc_iv': None,
        'sc_delta': None,
        # Short put leg
        'sp_bid': None,
        'sp_ask': None,
        'sp_iv': None,
        'sp_delta': None,
        # Long call leg
        'lc_bid': None,
        'lc_ask': None,
        'lc_iv': None,
        'lc_delta': None,
        # Long put leg
        'lp_bid': None,
        'lp_ask': None,
        'lp_iv': None,
        'lp_delta': None,
        # Exit / P&L
        'exit_spot': None,
        'exit_value': None,
        'pnl': None,
        'return_pct_on_width': None,
        'return_pct_on_credit': None,
        'annualized_return_on_width': None,
        'spot_move_pct': None,
        'days_held': None,
        # Status
        'is_tradeable': False,
        'failure_reason': failure_reason,
        'processing_time': processing_time,
    }


def _candidate_row(
    ticker: str,
    entry_date: date,
    expiry_date: date,
    dte_target: int,
    frequency: str,
    entry_spot: float,
    body_strike: Decimal,
    candidate: IronButterflyCandidate,
    exit_spot: Decimal,
    spot_move_pct: Optional[float],
    processing_time: float,
) -> Dict:
    """
    Build a fully populated output row for one wing candidate.

    P&L is computed by assembling a temporary ``OptionStrategy`` from the
    candidate legs so that ``calculate_payoff()`` can be called cleanly.
    """
    # Assemble temporary strategy from candidate legs (no validation needed —
    # candidate was already validated by enumerate_candidates)
    strategy = OptionStrategy(
        ticker=ticker,
        strategy_type=StrategyType.IRON_BUTTERFLY,
        legs=(
            OptionLeg(option=candidate.long_put,   quantity=1),
            OptionLeg(option=candidate.short_put,  quantity=-1),
            OptionLeg(option=candidate.short_call, quantity=-1),
            OptionLeg(option=candidate.long_call,  quantity=1),
        ),
        trade_date=entry_date,
    )

    # Intrinsic value at expiry
    exit_value = strategy.calculate_payoff({expiry_date: exit_spot})

    # P&L: intrinsic minus the signed entry cost.
    # strategy.net_premium is negative for credit strategies, so
    # pnl = exit_value - net_premium = exit_value + |credit|
    position = Position(
        ticker=ticker,
        entry_date=entry_date,
        strategy=strategy,
        quantity=1.0,
        entry_cost=strategy.net_premium,
        exit_date=expiry_date,
        exit_value=exit_value,
        metadata={},
    )

    pnl = float(position.pnl)
    wing_width_f = float(candidate.wing_width)
    net_credit_f = float(candidate.net_credit)
    days_held = (expiry_date - entry_date).days

    return_pct_on_width  = (pnl / wing_width_f  * 100) if wing_width_f  > 0 else None
    return_pct_on_credit = (pnl / abs(net_credit_f) * 100) if net_credit_f != 0 else None
    annualized = (return_pct_on_width * 365 / days_held) if (return_pct_on_width is not None and days_held > 0) else None

    sc = candidate.short_call
    sp = candidate.short_put
    lc = candidate.long_call
    lp = candidate.long_put

    return {
        # Identity
        'ticker': ticker,
        'entry_date': entry_date,
        'dte_category': frequency,
        'dte_target': dte_target,
        'dte_actual': days_held,
        'expiry_date': expiry_date,
        'entry_spot': float(entry_spot),
        'body_strike': float(body_strike),
        # Candidate geometry
        'wing_width': float(candidate.wing_width),
        'call_wing_strike': float(candidate.call_wing_strike),
        'put_wing_strike': float(candidate.put_wing_strike),
        'avg_wing_delta': candidate.avg_wing_delta,
        # Entry economics
        'net_credit': float(candidate.net_credit),
        'credit_to_width': candidate.credit_to_width,
        'total_spread': float(candidate.total_spread),
        'spread_cost_ratio': candidate.spread_cost_ratio,
        # Entry greeks
        'net_delta': candidate.net_delta,
        'net_gamma': candidate.net_gamma,
        'net_vega': candidate.net_vega,
        'net_theta': candidate.net_theta,
        # Short call leg
        'sc_bid': float(sc.bid),
        'sc_ask': float(sc.ask),
        'sc_iv': sc.iv,
        'sc_delta': sc.delta,
        # Short put leg
        'sp_bid': float(sp.bid),
        'sp_ask': float(sp.ask),
        'sp_iv': sp.iv,
        'sp_delta': sp.delta,
        # Long call leg
        'lc_bid': float(lc.bid),
        'lc_ask': float(lc.ask),
        'lc_iv': lc.iv,
        'lc_delta': lc.delta,
        # Long put leg
        'lp_bid': float(lp.bid),
        'lp_ask': float(lp.ask),
        'lp_iv': lp.iv,
        'lp_delta': lp.delta,
        # Exit / P&L
        'exit_spot': float(exit_spot),
        'exit_value': float(exit_value),
        'pnl': pnl,
        'return_pct_on_width': return_pct_on_width,
        'return_pct_on_credit': return_pct_on_credit,
        'annualized_return_on_width': annualized,
        'spot_move_pct': spot_move_pct * 100 if spot_move_pct is not None else None,
        'days_held': days_held,
        # Status
        'is_tradeable': True,
        'failure_reason': None,
        'processing_time': processing_time,
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class IronFlyHistoryBuilder:
    """
    Build a historical database of iron butterfly wing candidates.

    For every (ticker, entry_date) pair, ALL symmetric wing pairs that pass
    the quality filters are recorded — one row per wing width.  This lets
    analysis scripts sweep across wing deltas without re-processing raw data.

    Mirrors the interface of ``StraddleHistoryBuilder`` so the precompute
    script pattern is identical.

    Args:
        data_root:            Path to ORATS data directory.
        spot_db:              Pre-loaded ``SpotPriceDB`` for RV and exit spots.
        dte_target:           Target days to expiry (30 for monthly, 7 for weekly).
        max_spread_pct:       Maximum bid-ask spread / mid per leg.
                              Default 0.99 (permissive — keeps all tradeable wings).
        min_yield_on_capital: Minimum net_credit / wing_width.
                              Default 0.0 (permissive — no yield floor).
        min_volume:           Minimum option volume (passed to ORATSDataProvider).
        min_oi:               Minimum open interest (passed to ORATSDataProvider).
        frequency:            ``'monthly'`` or ``'weekly'`` — controls expiry
                              selection logic in ``_find_best_expiry()``.

    Example::

        >>> spot_db = SpotPriceDB.load('cache/spot_prices_adjusted.parquet')
        >>> builder = IronFlyHistoryBuilder(
        ...     data_root='C:/ORATS/data/ORATS_Adjusted',
        ...     spot_db=spot_db,
        ...     dte_target=30,
        ...     frequency='monthly',
        ... )
        >>> rows = builder.process_single_entry('AAPL', date(2023, 1, 6))
        >>> len(rows)         # One row per wing width candidate
        8
        >>> rows[0]['wing_width']
        5.0
    """

    def __init__(
        self,
        data_root: str,
        spot_db: SpotPriceDB,
        dte_target: int = 30,
        max_spread_pct: float = 0.99,
        min_yield_on_capital: float = 0.0,
        min_volume: int = 0,
        min_oi: int = 0,
        frequency: str = 'monthly',
    ):
        if frequency not in ('monthly', 'weekly'):
            raise ValueError(f"frequency must be 'monthly' or 'weekly', got {frequency!r}")

        self.data_root = data_root
        self.spot_db = spot_db
        self.dte_target = dte_target
        self.max_spread_pct = max_spread_pct
        self.min_yield_on_capital = min_yield_on_capital
        self.min_volume = min_volume
        self.min_oi = min_oi
        self.frequency = frequency

        # Lazily initialised per worker (avoids issues with joblib forks)
        self.provider: Optional[ORATSDataProvider] = None
        self.builder: Optional[IronButterflyBuilder] = None

    # ------------------------------------------------------------------
    # Worker initialisation
    # ------------------------------------------------------------------

    def _init_worker_components(self) -> None:
        """
        Lazily initialise ``ORATSDataProvider`` and ``IronButterflyBuilder``.

        Called once per worker process on the first ``process_single_entry``
        invocation.  Safe to call multiple times (idempotent).
        """
        if self.provider is None:
            self.provider = ORATSDataProvider(
                data_root=self.data_root,
                min_volume=self.min_volume,
                min_open_interest=self.min_oi,
                max_spread_pct=self.max_spread_pct,
            )
            # wing_delta is required by IronButterflyBuilder but irrelevant
            # here — enumerate_candidates() ignores it entirely.
            self.builder = IronButterflyBuilder(
                wing_delta=0.15,
                max_spread_pct=self.max_spread_pct,
                min_yield_on_capital=self.min_yield_on_capital,
            )
            logger.info(f"Worker initialised with data_root={self.data_root}")

    # ------------------------------------------------------------------
    # Expiry selection (mirrors StraddleHistoryBuilder._find_best_expiry)
    # ------------------------------------------------------------------

    def _find_best_expiry(
        self,
        ticker: str,
        trade_date: date,
        target_dte: int,
        tolerance_days: int = 4,
    ) -> Optional[date]:
        """
        Find the best expiry date for the given entry date.

        Monthly logic (``dte_target >= 28``):
            Target the first Friday (or Thursday) of the next calendar month.
            Falls back to any expiry within ~30 DTE ± ``tolerance_days`` if
            no Friday/Thursday exists in the target month.

        Weekly logic (``dte_target < 28``):
            Find the nearest Friday within ±``tolerance_days`` of ``target_dte``;
            fall back to Thursday, then any day.

        Args:
            ticker:         Stock ticker.
            trade_date:     Entry date.
            target_dte:     Target days to expiry.
            tolerance_days: Allowed deviation from target DTE.

        Returns:
            Best expiry date, or ``None`` if none found within tolerance.
        """
        try:
            expiries = self.provider.get_available_expiries(ticker, trade_date)
            if not expiries:
                return None

            if target_dte >= 28:  # Monthly
                next_month = trade_date.month + 1 if trade_date.month < 12 else 1
                next_year  = trade_date.year if trade_date.month < 12 else trade_date.year + 1

                target_month_expiries = [
                    exp for exp in expiries
                    if exp.year == next_year
                    and exp.month == next_month
                    and exp.weekday() in (3, 4)   # Thursday or Friday
                    and exp >= trade_date
                ]

                if not target_month_expiries:
                    logger.warning(
                        f"{ticker} on {trade_date}: "
                        f"No Fri/Thu expiry in {next_year}-{next_month:02d}"
                    )
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

                target_month_expiries.sort()
                best_expiry = target_month_expiries[0]

                dte = (best_expiry - trade_date).days
                if dte < 20 or dte > 45:
                    logger.warning(
                        f"{ticker} on {trade_date}: "
                        f"First Fri of {next_year}-{next_month:02d} is {dte} DTE (unusual)"
                    )

                return best_expiry

            else:  # Weekly
                expiry_dtes = [
                    (exp, (exp - trade_date).days) for exp in expiries
                ]
                valid = [
                    (exp, dte) for exp, dte in expiry_dtes
                    if dte > 0 and abs(dte - target_dte) <= tolerance_days
                ]
                if not valid:
                    return None

                friday_expiries   = [(e, d) for e, d in valid if e.weekday() == 4]
                thursday_expiries = [(e, d) for e, d in valid if e.weekday() == 3]

                if friday_expiries:
                    return min(friday_expiries,   key=lambda x: abs(x[1] - target_dte))[0]
                if thursday_expiries:
                    return min(thursday_expiries, key=lambda x: abs(x[1] - target_dte))[0]

                logger.warning(
                    f"{ticker} on {trade_date}: No Fri/Thu expiry in range, using any day"
                )
                return min(valid, key=lambda x: abs(x[1] - target_dte))[0]

        except Exception as exc:
            logger.error(
                f"Error finding expiry for {ticker} on {trade_date}: {exc}"
            )
            return None

    # ------------------------------------------------------------------
    # Core processing method
    # ------------------------------------------------------------------

    def process_single_entry(
        self,
        ticker: str,
        entry_date: date,
    ) -> List[Dict]:
        """
        Process all wing candidates for one (ticker, entry_date) pair.

        Returns a **list** of dicts:
        - On success: one dict per valid ``IronButterflyCandidate``.
        - On failure: a single dict with ``is_tradeable=False`` and
          ``failure_reason`` set.

        All numeric fields in the failure row are ``None`` so that downstream
        ``pd.DataFrame`` construction handles mixed types cleanly.

        Steps
        -----
        1.  Look up entry spot price.
        2.  Find best expiry via ``_find_best_expiry()``.
        3.  Fetch option chain for that expiry.
        4.  Locate ATM body strike.
        5.  Validate body call and put (non-None, mid > 0).
        6.  Call ``enumerate_candidates()`` — permissive filters, returns all.
        7.  Look up exit spot at expiry for P&L calculation.
        8.  Build output row for each candidate.

        Args:
            ticker:     Stock ticker symbol.
            entry_date: Trade entry date (Friday).

        Returns:
            List of row dicts (one per candidate, or one failure row).

        Example::

            >>> rows = builder.process_single_entry('AAPL', date(2023, 1, 6))
            >>> df = pd.DataFrame(rows)
            >>> df[df['is_tradeable']][['wing_width', 'avg_wing_delta', 'pnl']]
        """
        start_time = datetime.now()
        self._init_worker_components()

        def elapsed() -> float:
            return (datetime.now() - start_time).total_seconds()

        def fail(reason: str) -> List[Dict]:
            return [_failure_row(ticker, entry_date, self.dte_target,
                                 self.frequency, reason, elapsed())]

        try:
            # 1. Entry spot
            entry_spot = self.provider.get_spot_price(ticker, entry_date)
            if entry_spot is None:
                return fail('no_spot_price')

            # 2. Find expiry
            expiry_date = self._find_best_expiry(ticker, entry_date, self.dte_target)
            if expiry_date is None:
                return fail('no_expiry_found')

            # 3. Option chain
            chain = self.provider.get_option_chain(
                ticker=ticker,
                trade_date=entry_date,
                expiry_date=expiry_date,
            )
            if not chain:
                return fail('no_options_at_entry')

            # 4. ATM body strike
            body_strike = self.builder._find_atm_strike(chain, entry_spot)

            # 5. Validate body legs
            short_call = self.builder._get_option_at_strike(chain, body_strike, 'call')
            short_put  = self.builder._get_option_at_strike(chain, body_strike, 'put')

            if short_call is None:
                return fail('no_body_call')
            if short_put is None:
                return fail('no_body_put')
            if short_call.mid <= 0 or short_put.mid <= 0:
                return fail('invalid_body_mid')

            # 6. Enumerate all valid wing candidates (permissive filters)
            candidates = self.builder.enumerate_candidates(
                option_chain=chain,
                body_strike=body_strike,
                short_call=short_call,
                short_put=short_put,
            )
            if not candidates:
                return fail('no_candidates')

            # 7. Exit spot for P&L
            exit_spot_raw = self.provider.get_spot_price(ticker, expiry_date)
            if exit_spot_raw is None:
                return fail('no_spot_at_expiry')
            exit_spot = Decimal(str(exit_spot_raw))

            # Optional spot-move from SpotPriceDB (may return None — non-fatal)
            spot_move_pct = self.spot_db.calculate_spot_move_pct(
                ticker, entry_date, expiry_date
            )

            # 8. Build one output row per candidate
            rows: List[Dict] = []
            for cand in candidates:
                try:
                    row = _candidate_row(
                        ticker=ticker,
                        entry_date=entry_date,
                        expiry_date=expiry_date,
                        dte_target=self.dte_target,
                        frequency=self.frequency,
                        entry_spot=entry_spot,
                        body_strike=body_strike,
                        candidate=cand,
                        exit_spot=exit_spot,
                        spot_move_pct=spot_move_pct,
                        processing_time=elapsed(),
                    )
                    rows.append(row)
                except Exception as exc:
                    # Single-candidate failure: log and skip, don't abort the batch
                    logger.warning(
                        f"Candidate row failed for {ticker} on {entry_date} "
                        f"wing_width={cand.wing_width}: {exc}"
                    )

            if not rows:
                # All candidates failed in _candidate_row (highly unusual)
                return fail('all_candidates_failed')

            return rows

        except ValueError as exc:
            error_msg = str(exc)
            if 'No data found' in error_msg or 'not found' in error_msg.lower():
                reason = 'data_missing'
            else:
                reason = f'value_error_{error_msg[:50]}'
            return fail(reason)

        except Exception as exc:
            logger.error(
                f"Unexpected error processing {ticker} on {entry_date}: "
                f"{type(exc).__name__}: {exc}",
                exc_info=True,
            )
            return fail(f'error_{type(exc).__name__}_{str(exc)[:100]}')
