"""
Unit tests for build_straddle_from_surface.

Each test uses a minimal synthetic OptionSurfaceDB built from hand-crafted
DataFrames so that every expected value can be derived by hand and verified
independently of real market data.

Synthetic surface parameters (used throughout unless overridden):
    body_strike  = 100.00
    call: bid=2.00  ask=2.40  mid=2.20  delta=+0.50  iv=0.20
    put:  bid=1.80  ask=2.20  mid=2.00  delta=-0.50  iv=0.20

Mid straddle premium = 2.20 + 2.00 = 4.20
Cross straddle debit  = 2.40 (call ask) + 2.20 (put ask) = 4.60   (long)
Cross straddle credit = 2.00 (call bid) + 1.80 (put bid) = 3.80   (short)
"""

import pytest
import pandas as pd
from datetime import date
from decimal import Decimal

from src.backtest.option_surface import (
    OptionSurfaceDB,
    FillAssumption,
    build_straddle_from_surface,
)


# =============================================================================
# Synthetic surface helpers
# =============================================================================

TICKER     = "TEST"
ENTRY_DATE = date(2024, 1, 5)
EXPIRY_DATE = date(2024, 1, 12)
BODY_STRIKE = 100.0
ENTRY_SPOT  = 100.0
EXIT_SPOT   = 102.0   # spot moved +2 from body


def _make_surface_db(
    call_bid="2.00", call_ask="2.40",
    put_bid="1.80",  put_ask="2.20",
    exit_spot=EXIT_SPOT,
) -> OptionSurfaceDB:
    """Build a minimal OptionSurfaceDB with one valid surface row."""
    call_mid = (float(call_bid) + float(call_ask)) / 2
    put_mid  = (float(put_bid)  + float(put_ask))  / 2

    meta_df = pd.DataFrame([{
        "ticker":          TICKER,
        "entry_date":      pd.Timestamp(ENTRY_DATE),
        "expiry_date":     pd.Timestamp(EXPIRY_DATE),
        "surface_valid":   True,
        "failure_reason":  None,
        "entry_spot":      ENTRY_SPOT,
        "body_strike":     BODY_STRIKE,
        "exit_spot":       exit_spot,
        "spot_move_pct":   (exit_spot - ENTRY_SPOT) / ENTRY_SPOT * 100,
        "realized_volatility": 0.18,
    }])

    quotes_df = pd.DataFrame([
        # ATM call (body)
        dict(ticker=TICKER, entry_date=pd.Timestamp(ENTRY_DATE),
             expiry_date=pd.Timestamp(EXPIRY_DATE),
             side="call", strike=BODY_STRIKE,
             bid=float(call_bid), ask=float(call_ask), mid=call_mid,
             iv=0.20, delta=0.50, gamma=0.05, vega=0.10, theta=-0.03,
             volume=1000, open_interest=5000,
             abs_delta=0.50, spread_pct=(float(call_ask)-float(call_bid))/call_mid,
             is_body=True, is_otm=False),
        # ATM put (body)
        dict(ticker=TICKER, entry_date=pd.Timestamp(ENTRY_DATE),
             expiry_date=pd.Timestamp(EXPIRY_DATE),
             side="put", strike=BODY_STRIKE,
             bid=float(put_bid), ask=float(put_ask), mid=put_mid,
             iv=0.20, delta=-0.50, gamma=0.05, vega=0.10, theta=-0.03,
             volume=900, open_interest=4500,
             abs_delta=0.50, spread_pct=(float(put_ask)-float(put_bid))/put_mid,
             is_body=True, is_otm=False),
    ])

    return OptionSurfaceDB(meta_df, quotes_df)


# =============================================================================
# Long straddle — entry economics
# =============================================================================

class TestLongStraddleMidFill:
    """Long straddle filled at mid — the arithmetic is simple and fully checkable."""

    def setup_method(self):
        self.db  = _make_surface_db()
        self.fly = build_straddle_from_surface(
            self.db, TICKER, ENTRY_DATE, direction="long",
            fill=FillAssumption.mid(),
        )

    def test_strategy_name(self):
        assert self.fly.strategy_name == "long_straddle"

    def test_entry_cost_is_positive(self):
        # Long straddle is a debit — trader pays premium out.
        assert self.fly.entry_cost > 0

    def test_entry_cost_equals_sum_of_mids(self):
        # call mid=2.20, put mid=2.00 → total=4.20
        assert float(self.fly.entry_cost) == pytest.approx(4.20)

    def test_net_credit_is_negative(self):
        # net_credit = -entry_cost → negative for a debit structure
        assert self.fly.net_credit < 0

    def test_max_loss_equals_entry_cost(self):
        # Max loss for a long straddle = what you paid
        assert self.fly.max_loss_per_share == self.fly.entry_cost

    def test_return_on_max_loss_is_none(self):
        # Unlimited upside → ROC ratio is not meaningful
        assert self.fly.return_on_max_loss is None

    def test_spread_cost_is_zero_at_mid(self):
        # Mid fill by definition has no spread friction
        assert float(self.fly.spread_cost) == pytest.approx(0.0)

    def test_leg_spread_to_credit_ratio_is_none(self):
        # Property returns None when net_credit <= 0 (debit strategy)
        assert self.fly.leg_spread_to_credit_ratio is None

    def test_total_leg_spread(self):
        # call spread=0.40, put spread=0.40 → total=0.80 (2 long legs, |qty|=1 each)
        assert float(self.fly.total_leg_spread) == pytest.approx(0.80)

    def test_diagnostics_contain_direction(self):
        assert self.fly.diagnostics["direction"] == "long"


class TestLongStraddleCrossFill:
    """Long straddle filled at ask (cross fill) — extra debit vs mid."""

    def setup_method(self):
        self.db  = _make_surface_db()
        self.fly = build_straddle_from_surface(
            self.db, TICKER, ENTRY_DATE, direction="long",
            fill=FillAssumption.cross(),
        )

    def test_entry_cost_at_ask(self):
        # Buy call at ask=2.40, buy put at ask=2.20 → total=4.60
        assert float(self.fly.entry_cost) == pytest.approx(4.60)

    def test_spread_cost_positive(self):
        # Paid ask instead of mid — extra cost per leg = (ask - mid):
        # call: 2.40 - 2.20 = 0.20;  put: 2.20 - 2.00 = 0.20  → total = 0.40
        assert float(self.fly.spread_cost) == pytest.approx(0.40)

    def test_spread_cost_ratio_normalised_against_mid_debit(self):
        # spread_cost_ratio = |spread_cost| / |entry_cost_mid| = 0.40 / 4.20 ≈ 9.52 %
        assert self.fly.spread_cost_ratio == pytest.approx(0.40 / 4.20, rel=1e-4)


# =============================================================================
# Short straddle — entry economics
# =============================================================================

class TestShortStraddleMidFill:
    """Short straddle — sells both legs, receives credit."""

    def setup_method(self):
        self.db  = _make_surface_db()
        self.fly = build_straddle_from_surface(
            self.db, TICKER, ENTRY_DATE, direction="short",
            fill=FillAssumption.mid(),
        )

    def test_strategy_name(self):
        assert self.fly.strategy_name == "short_straddle"

    def test_entry_cost_is_negative(self):
        # Short straddle receives premium — cash flows in
        assert self.fly.entry_cost < 0

    def test_entry_cost_equals_negative_sum_of_mids(self):
        # sell call at mid=2.20 + sell put at mid=2.00 → credit=4.20
        assert float(self.fly.entry_cost) == pytest.approx(-4.20)

    def test_net_credit_equals_sum_of_mids(self):
        assert float(self.fly.net_credit) == pytest.approx(4.20)

    def test_max_loss_is_none(self):
        # Short straddle has theoretically unlimited downside
        assert self.fly.max_loss_per_share is None

    def test_return_on_max_loss_is_none(self):
        assert self.fly.return_on_max_loss is None

    def test_leg_spread_to_credit_ratio(self):
        # total_leg_spread=0.80, net_credit=4.20 → 0.80/4.20 ≈ 19.05 %
        assert self.fly.leg_spread_to_credit_ratio == pytest.approx(0.80 / 4.20, rel=1e-4)

    def test_spread_cost_is_zero_at_mid(self):
        assert float(self.fly.spread_cost) == pytest.approx(0.0)


class TestShortStraddleCrossFill:
    """Short straddle sold at bid — worst-case credit."""

    def setup_method(self):
        self.db  = _make_surface_db()
        self.fly = build_straddle_from_surface(
            self.db, TICKER, ENTRY_DATE, direction="short",
            fill=FillAssumption.cross(),
        )

    def test_entry_cost_at_bid(self):
        # Sell call at bid=2.00, sell put at bid=1.80 → credit=3.80
        assert float(self.fly.entry_cost) == pytest.approx(-3.80)

    def test_net_credit_at_bid(self):
        assert float(self.fly.net_credit) == pytest.approx(3.80)

    def test_spread_cost_positive(self):
        # Got bid instead of mid — extra cost per leg = (mid - bid):
        # call: 2.20 - 2.00 = 0.20;  put: 2.00 - 1.80 = 0.20  → total = 0.40
        assert float(self.fly.spread_cost) == pytest.approx(0.40)

    def test_spread_cost_ratio_normalised_against_net_credit(self):
        # 0.40 / 3.80 ≈ 10.53 %  (net_credit at cross fill is 3.80)
        assert self.fly.spread_cost_ratio == pytest.approx(0.40 / 3.80, rel=1e-4)


# =============================================================================
# Settle at expiry — P&L verification
# =============================================================================

class TestStraddleSettle:
    """Verify P&L payoff at expiry against hand-calculated values.

    Long straddle payoff at expiry:
        exit_value = max(spot - strike, 0) + max(strike - spot, 0)
        pnl        = exit_value - entry_cost

    Short straddle payoff (mirror):
        exit_value = -(max(spot - strike, 0) + max(strike - spot, 0))
        pnl        = exit_value - entry_cost  (entry_cost is negative)
    """

    def setup_method(self):
        self.db = _make_surface_db()

    def test_long_straddle_settle_at_body_strike(self):
        # Spot = strike → both legs expire worthless → pnl = -entry_cost (lose all premium)
        straddle = build_straddle_from_surface(
            self.db, TICKER, ENTRY_DATE, direction="long", fill=FillAssumption.mid()
        )
        pos = straddle.settle(Decimal("100.00"))
        # exit_value = 0; pnl = 0 - 4.20 = -4.20
        assert float(pos.exit_value) == pytest.approx(0.0)
        assert float(pos.pnl) == pytest.approx(-4.20)

    def test_long_straddle_settle_call_side(self):
        # Spot = 106 → call intrinsic = 6, put = 0 → exit_value = 6 → pnl = 6 - 4.20 = 1.80
        straddle = build_straddle_from_surface(
            self.db, TICKER, ENTRY_DATE, direction="long", fill=FillAssumption.mid()
        )
        pos = straddle.settle(Decimal("106.00"))
        assert float(pos.exit_value) == pytest.approx(6.0)
        assert float(pos.pnl) == pytest.approx(1.80)

    def test_long_straddle_settle_put_side(self):
        # Spot = 94 → put intrinsic = 6, call = 0 → exit_value = 6 → pnl = 6 - 4.20 = 1.80
        straddle = build_straddle_from_surface(
            self.db, TICKER, ENTRY_DATE, direction="long", fill=FillAssumption.mid()
        )
        pos = straddle.settle(Decimal("94.00"))
        assert float(pos.exit_value) == pytest.approx(6.0)
        assert float(pos.pnl) == pytest.approx(1.80)

    def test_short_straddle_settle_at_body_strike(self):
        # Both legs expire worthless — collect full credit
        straddle = build_straddle_from_surface(
            self.db, TICKER, ENTRY_DATE, direction="short", fill=FillAssumption.mid()
        )
        pos = straddle.settle(Decimal("100.00"))
        # exit_value = 0; entry_cost = -4.20; pnl = 0 - (-4.20) = +4.20
        assert float(pos.exit_value) == pytest.approx(0.0)
        assert float(pos.pnl) == pytest.approx(4.20)

    def test_short_straddle_settle_call_side(self):
        # Spot = 106 → short call intrinsic loss = -6, short put = 0
        # exit_value = -6; pnl = -6 - (-4.20) = -1.80
        straddle = build_straddle_from_surface(
            self.db, TICKER, ENTRY_DATE, direction="short", fill=FillAssumption.mid()
        )
        pos = straddle.settle(Decimal("106.00"))
        assert float(pos.exit_value) == pytest.approx(-6.0)
        assert float(pos.pnl) == pytest.approx(-1.80)


# =============================================================================
# Liquidity filter
# =============================================================================

class TestMaxLegSpreadFilter:
    """max_leg_spread_pct drops illiquid body quotes before construction."""

    def test_tight_filter_raises_when_spread_too_wide(self):
        # call spread_pct = 0.40/2.20 ≈ 18.2 %, put ≈ 20.0 %
        # Setting threshold below that should raise
        db = _make_surface_db()
        with pytest.raises(ValueError, match="Missing tradeable body"):
            build_straddle_from_surface(
                db, TICKER, ENTRY_DATE, direction="long",
                fill=FillAssumption.mid(),
                max_leg_spread_pct=0.10,   # 10 % — tighter than either leg
            )

    def test_wide_filter_passes(self):
        db = _make_surface_db()
        result = build_straddle_from_surface(
            db, TICKER, ENTRY_DATE, direction="long",
            fill=FillAssumption.mid(),
            max_leg_spread_pct=0.50,   # 50 % — wider than both legs
        )
        assert result is not None

    def test_invalid_direction_raises(self):
        db = _make_surface_db()
        with pytest.raises(ValueError, match="direction must be"):
            build_straddle_from_surface(db, TICKER, ENTRY_DATE, direction="neutral")
