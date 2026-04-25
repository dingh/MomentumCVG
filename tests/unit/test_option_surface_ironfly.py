"""
Unit tests for build_ironfly_from_surface.

Synthetic surface layout
------------------------
Body strike = 100

  Leg               Side   Strike   bid    ask    mid    abs_delta  is_body  is_otm
  ─────────────────────────────────────────────────────────────────────────────────
  ATM call (body)   call   100     3.00   3.40   3.20   0.50       True     False
  ATM put  (body)   put    100     2.80   3.20   3.00   0.50       True     False
  OTM call (wing)   call   105     1.00   1.20   1.10   0.25       False    True
  OTM put  (wing)   put     95     0.90   1.10   1.00   0.25       False    True
  Far OTM call      call   110     0.70   0.90   0.80   0.15       False    True
  (used only in asymmetric tests)

Iron fly legs (short body / long wings), mid fill:
  +1 long put  @  95  mid=1.00   → +1.00
  -1 short put @ 100  mid=3.00   → -3.00
  -1 short call@ 100  mid=3.20   → -3.20
  +1 long call @ 105  mid=1.10   → +1.10
  ──────────────────────────────────
  entry_cost (mid)   = 1.00 - 3.00 - 3.20 + 1.10 = -4.10  (credit)
  net_credit         = 4.10
  wing_width         = max(105-100, 100-95) = 5
  max_loss_per_share = 5 - 4.10 = 0.90
  return_on_max_loss = 4.10 / 0.90 ≈ 4.5556
  total_leg_spread   = 0.20 + 0.40 + 0.40 + 0.20 = 1.20
  leg_spread/credit  = 1.20 / 4.10 ≈ 0.29268

Cross fill (buy at ask, sell at bid):
  +1 long put  @  95  ask=1.10   → +1.10
  -1 short put @ 100  bid=2.80   → -2.80
  -1 short call@ 100  bid=3.00   → -3.00
  +1 long call @ 105  ask=1.20   → +1.20
  entry_cost (cross) = 1.10 - 2.80 - 3.00 + 1.20 = -3.50
  net_credit (cross) = 3.50
  spread_cost        = -3.50 - (-4.10) = 0.60
  spread_cost_ratio  = 0.60 / 3.50 ≈ 0.17143
"""

import pytest
import pandas as pd
from datetime import date
from decimal import Decimal

from src.backtest.option_surface import (
    OptionSurfaceDB,
    FillAssumption,
    build_ironfly_from_surface,
)

# =============================================================================
# Constants
# =============================================================================

TICKER      = "TEST"
ENTRY_DATE  = date(2024, 1, 5)
EXPIRY_DATE = date(2024, 2, 2)
BODY_STRIKE = 100.0
ENTRY_SPOT  = 100.0


# =============================================================================
# Synthetic surface helper
# =============================================================================

def _quote_row(
    side: str, strike: float, bid: float, ask: float,
    delta: float, abs_delta: float,
    is_body: bool = False, is_otm: bool = False,
) -> dict:
    mid = (bid + ask) / 2
    return dict(
        ticker=TICKER,
        entry_date=pd.Timestamp(ENTRY_DATE),
        expiry_date=pd.Timestamp(EXPIRY_DATE),
        side=side,
        strike=float(strike),
        bid=bid, ask=ask, mid=mid,
        iv=0.22, delta=delta, gamma=0.04, vega=0.09, theta=-0.02,
        volume=500, open_interest=2000,
        abs_delta=abs_delta,
        spread_pct=(ask - bid) / mid if mid > 0 else 0.0,
        is_body=is_body,
        is_otm=is_otm,
    )


def _make_ironfly_db(include_far_otm_wings: bool = False) -> OptionSurfaceDB:
    """Symmetric 5-point wings (call at 105, put at 95); wing delta ≈ 0.25.

    When *include_far_otm_wings* is True, two additional deep-OTM legs are added to
    produce asymmetric wing distances for testing max() vs min():
      - long call @ 110  abs_delta=0.15  (10 pts from body)
      - long put  @  97  abs_delta=0.14  ( 3 pts from body)
    Both have abs_delta ≤ 0.15, so they are eligible under _choose_below_nearest
    with wing_target_delta=0.15, while the nearer wings (abs_delta=0.25) are excluded.
    """
    meta_df = pd.DataFrame([{
        "ticker":              TICKER,
        "entry_date":          pd.Timestamp(ENTRY_DATE),
        "expiry_date":         pd.Timestamp(EXPIRY_DATE),
        "surface_valid":       True,
        "failure_reason":      None,
        "entry_spot":          ENTRY_SPOT,
        "body_strike":         BODY_STRIKE,
        "exit_spot":           ENTRY_SPOT,
        "spot_move_pct":       0.0,
        "realized_volatility": 0.20,
    }])

    rows = [
        _quote_row("call", 100, bid=3.00, ask=3.40, delta=+0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        _quote_row("put",  100, bid=2.80, ask=3.20, delta=-0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        _quote_row("call", 105, bid=1.00, ask=1.20, delta=+0.25, abs_delta=0.25, is_body=False, is_otm=True),
        _quote_row("put",   95, bid=0.90, ask=1.10, delta=-0.25, abs_delta=0.25, is_body=False, is_otm=True),
    ]
    if include_far_otm_wings:
        rows.extend([
            _quote_row("call", 110, bid=0.70, ask=0.90, delta=+0.15, abs_delta=0.15, is_body=False, is_otm=True),
            _quote_row("put",   97, bid=0.40, ask=0.60, delta=-0.14, abs_delta=0.14, is_body=False, is_otm=True),
        ])

    return OptionSurfaceDB(meta_df, pd.DataFrame(rows))


# =============================================================================
# Happy path — mid fill, symmetric wings
# =============================================================================

class TestIronFlyMidFill:
    """Symmetric 5-point wings, mid fill: full hand-calculated verification."""

    def setup_method(self):
        self.db  = _make_ironfly_db()
        self.fly = build_ironfly_from_surface(
            self.db, TICKER, ENTRY_DATE,
            wing_target_delta=0.25,
            fill=FillAssumption.mid(),
        )

    def test_strategy_name(self):
        assert self.fly.strategy_name == "iron_fly"

    def test_entry_cost_is_credit(self):
        assert self.fly.entry_cost < 0

    def test_entry_cost_value(self):
        # 1.00 - 3.00 - 3.20 + 1.10 = -4.10
        assert float(self.fly.entry_cost) == pytest.approx(-4.10)

    def test_net_credit(self):
        assert float(self.fly.net_credit) == pytest.approx(4.10)

    def test_max_loss_per_share(self):
        # wing_width=5, net_credit=4.10 → max_loss=0.90
        assert float(self.fly.max_loss_per_share) == pytest.approx(0.90)

    def test_return_on_max_loss(self):
        # 4.10 / 0.90 ≈ 4.5556
        assert self.fly.return_on_max_loss == pytest.approx(4.10 / 0.90, rel=1e-4)

    def test_spread_cost_zero_at_mid(self):
        assert float(self.fly.spread_cost) == pytest.approx(0.0)

    def test_total_leg_spread(self):
        # |qty|×spread per leg: 0.20 + 0.40 + 0.40 + 0.20 = 1.20
        assert float(self.fly.total_leg_spread) == pytest.approx(1.20)

    def test_leg_spread_to_credit_ratio(self):
        # 1.20 / 4.10
        assert self.fly.leg_spread_to_credit_ratio == pytest.approx(1.20 / 4.10, rel=1e-4)

    def test_wing_width_in_diagnostics(self):
        assert self.fly.diagnostics["wing_width"] == pytest.approx(5.0)

    def test_long_call_strike_in_diagnostics(self):
        assert self.fly.diagnostics["long_call_strike"] == pytest.approx(105.0)

    def test_long_put_strike_in_diagnostics(self):
        assert self.fly.diagnostics["long_put_strike"] == pytest.approx(95.0)

    def test_four_legs(self):
        assert len(self.fly.strategy.legs) == 4

    def test_leg_signs(self):
        quantities = sorted(leg.quantity for leg in self.fly.strategy.legs)
        assert quantities == [-1, -1, 1, 1]

    def test_both_short_legs_at_body_strike(self):
        short_strikes = {
            leg.option.strike
            for leg in self.fly.strategy.legs
            if leg.quantity < 0
        }
        assert short_strikes == {Decimal("100")}


# =============================================================================
# Cross fill
# =============================================================================

class TestIronFlyCrossFill:
    """Cross fill: buy at ask, sell at bid — entry is worse than mid."""

    def setup_method(self):
        self.db  = _make_ironfly_db()
        self.fly = build_ironfly_from_surface(
            self.db, TICKER, ENTRY_DATE,
            wing_target_delta=0.25,
            fill=FillAssumption.cross(),
        )

    def test_entry_cost_at_cross(self):
        # Buy put-wing at ask 1.10, sell put-body at bid 2.80,
        # sell call-body at bid 3.00, buy call-wing at ask 1.20
        # 1.10 - 2.80 - 3.00 + 1.20 = -3.50
        assert float(self.fly.entry_cost) == pytest.approx(-3.50)

    def test_net_credit_at_cross(self):
        assert float(self.fly.net_credit) == pytest.approx(3.50)

    def test_spread_cost_positive(self):
        # -3.50 - (-4.10) = 0.60
        assert float(self.fly.spread_cost) == pytest.approx(0.60)

    def test_spread_cost_ratio(self):
        # 0.60 / 3.50 ≈ 0.17143
        assert self.fly.spread_cost_ratio == pytest.approx(0.60 / 3.50, rel=1e-4)

    def test_total_leg_spread_fill_agnostic(self):
        # total_leg_spread is computed from bid/ask, not fill prices → same as mid
        assert float(self.fly.total_leg_spread) == pytest.approx(1.20)


# =============================================================================
# Asymmetric wings — verifies max() is used (not min) for wing_width
# =============================================================================

class TestIronFlyAsymmetricWings:
    """Call wing at 110 (10 pts), put wing at 95 (5 pts) → max_width = 10."""

    def setup_method(self):
        self.db  = _make_ironfly_db(include_far_otm_wings=True)
        self.fly = build_ironfly_from_surface(
            self.db, TICKER, ENTRY_DATE,
            wing_target_delta=0.15,
            fill=FillAssumption.mid(),
        )

    def test_call_wing_at_110(self):
        assert self.fly.diagnostics["long_call_strike"] == pytest.approx(110.0)

    def test_put_wing_at_97(self):
        # put@95 has abs_delta=0.25 > 0.15 threshold → excluded; put@97 (abs_delta=0.14) selected
        assert self.fly.diagnostics["long_put_strike"] == pytest.approx(97.0)

    def test_wing_width_uses_max_not_min(self):
        # max(110-100, 100-97) = max(10, 3) = 10
        # If min() were used by mistake the width would be 3.
        assert self.fly.diagnostics["wing_width"] == pytest.approx(10.0)

    def test_max_loss_based_on_wider_wing(self):
        # entry_cost mid: +0.50 (put@97) - 3.00 (put body) - 3.20 (call body) + 0.80 (call@110)
        # = 0.50 - 3.00 - 3.20 + 0.80 = -4.90 → net_credit = 4.90
        # max_loss = 10 - 4.90 = 5.10
        assert float(self.fly.net_credit) == pytest.approx(4.90)
        assert float(self.fly.max_loss_per_share) == pytest.approx(5.10)


# =============================================================================
# Settle at expiry — P&L verification
# =============================================================================

class TestIronFlySettle:
    """P&L payoff at expiry, verified by intrinsic value arithmetic.

    Iron fly structure (long put @ 95 / short put @ 100 / short call @ 100 / long call @ 105):

      Spot ≤ 95   : put-wing intrinsic cancels short-put partially; net = max_loss region
      95 < Spot < 100 : put side; short-call = 0; short-put = -(100-spot); long-put = 0
      Spot = 100  : all at/OTM → exit_value=0 → keep full credit (max profit)
      100 < Spot < 105: call side; short-call = -(spot-100); long-call = 0
      Spot ≥ 105  : call-wing offsets; net = max_loss region
    """

    def setup_method(self):
        self.db  = _make_ironfly_db()
        self.fly = build_ironfly_from_surface(
            self.db, TICKER, ENTRY_DATE,
            wing_target_delta=0.25,
            fill=FillAssumption.mid(),
        )

    def test_settle_at_body_strike_max_profit(self):
        # All legs expire at/OTM → exit_value = 0 → pnl = 0 - (-4.10) = +4.10
        pos = self.fly.settle(Decimal("100"))
        assert float(pos.exit_value) == pytest.approx(0.0)
        assert float(pos.pnl) == pytest.approx(4.10)

    def test_settle_between_body_and_call_wing(self):
        # spot=103: short call intrinsic = -(103-100) = -3; rest = 0
        # exit_value = -3 → pnl = -3 - (-4.10) = +1.10
        pos = self.fly.settle(Decimal("103"))
        assert float(pos.exit_value) == pytest.approx(-3.0)
        assert float(pos.pnl) == pytest.approx(1.10)

    def test_settle_beyond_call_wing_max_loss(self):
        # spot=107: short call = -(107-100) = -7; long call = +(107-105) = +2
        # exit_value = -5 → pnl = -5 - (-4.10) = -0.90 = -(max_loss_per_share)
        pos = self.fly.settle(Decimal("107"))
        assert float(pos.exit_value) == pytest.approx(-5.0)
        assert float(pos.pnl) == pytest.approx(-0.90)

    def test_settle_between_body_and_put_wing(self):
        # spot=97: short put intrinsic = -(100-97) = -3; rest = 0
        # exit_value = -3 → pnl = -3 - (-4.10) = +1.10
        pos = self.fly.settle(Decimal("97"))
        assert float(pos.exit_value) == pytest.approx(-3.0)
        assert float(pos.pnl) == pytest.approx(1.10)

    def test_settle_beyond_put_wing_max_loss(self):
        # spot=93: short put = -(100-93) = -7; long put = +(95-93) = +2
        # exit_value = -5 → pnl = -5 - (-4.10) = -0.90
        pos = self.fly.settle(Decimal("93"))
        assert float(pos.exit_value) == pytest.approx(-5.0)
        assert float(pos.pnl) == pytest.approx(-0.90)


# =============================================================================
# Liquidity filters and guard clauses
# =============================================================================

class TestIronFlyFilters:

    def test_max_leg_spread_pct_drops_illiquid_wings(self):
        # OTM call spread_pct = 0.20/1.10 ≈ 18.2 %; 10 % threshold filters all wings.
        # _choose_below_nearest then raises because the filtered df is empty.
        db = _make_ironfly_db()
        with pytest.raises(ValueError, match="No quotes with abs_delta"):
            build_ironfly_from_surface(
                db, TICKER, ENTRY_DATE,
                wing_target_delta=0.25,
                max_leg_spread_pct=0.10,
            )

    def test_max_leg_spread_pct_wide_passes(self):
        db = _make_ironfly_db()
        result = build_ironfly_from_surface(
            db, TICKER, ENTRY_DATE,
            wing_target_delta=0.25,
            max_leg_spread_pct=0.50,
        )
        assert result is not None

    def test_max_spread_cost_ratio_raises_when_exceeded(self):
        # Cross fill ratio ≈ 17.1 %; threshold of 5 % should raise
        db = _make_ironfly_db()
        with pytest.raises(ValueError, match="spread_cost_ratio"):
            build_ironfly_from_surface(
                db, TICKER, ENTRY_DATE,
                wing_target_delta=0.25,
                fill=FillAssumption.cross(),
                max_spread_cost_ratio=0.05,
            )

    def test_max_spread_cost_ratio_passes_when_met(self):
        # Mid fill has spread_cost_ratio = 0.0, passes any positive threshold
        db = _make_ironfly_db()
        result = build_ironfly_from_surface(
            db, TICKER, ENTRY_DATE,
            wing_target_delta=0.25,
            fill=FillAssumption.mid(),
            max_spread_cost_ratio=0.05,
        )
        assert result is not None

    def test_missing_ticker_raises_key_error(self):
        db = _make_ironfly_db()
        with pytest.raises(KeyError):
            build_ironfly_from_surface(db, "NOPE", ENTRY_DATE, wing_target_delta=0.25)

    def test_missing_date_raises_key_error(self):
        db = _make_ironfly_db()
        with pytest.raises(KeyError):
            build_ironfly_from_surface(db, TICKER, date(2025, 1, 1), wing_target_delta=0.25)


# =============================================================================
# Below-nearest wing selection semantics
# =============================================================================

def _make_ironfly_db_above_below_target() -> OptionSurfaceDB:
    """Surface with two OTM options per side: one with abs_delta just above the target
    and one below.  Used to verify that _choose_below_nearest rejects the above-target
    option even when it is 'nearer' to the target by raw delta distance.

    Call side:
      @103  abs_delta=0.28  (above target=0.25, raw distance=0.03 — closer by distance)
      @107  abs_delta=0.20  (below target=0.25, raw distance=0.05 — farther by distance)
    Put side (symmetric):
      @97   abs_delta=0.28
      @93   abs_delta=0.20
    """
    meta_df = pd.DataFrame([{
        "ticker":              TICKER,
        "entry_date":          pd.Timestamp(ENTRY_DATE),
        "expiry_date":         pd.Timestamp(EXPIRY_DATE),
        "surface_valid":       True,
        "failure_reason":      None,
        "entry_spot":          ENTRY_SPOT,
        "body_strike":         BODY_STRIKE,
        "exit_spot":           ENTRY_SPOT,
        "spot_move_pct":       0.0,
        "realized_volatility": 0.20,
    }])
    rows = [
        _quote_row("call", 100, bid=3.00, ask=3.40, delta=+0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        _quote_row("put",  100, bid=2.80, ask=3.20, delta=-0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        # Above-target wings (closer to money, 'nearer' by raw delta distance — must NOT be selected)
        _quote_row("call", 103, bid=1.70, ask=1.90, delta=+0.28, abs_delta=0.28, is_body=False, is_otm=True),
        _quote_row("put",   97, bid=1.60, ask=1.80, delta=-0.28, abs_delta=0.28, is_body=False, is_otm=True),
        # Below-target wings (further OTM, abs_delta < target — must be selected)
        _quote_row("call", 107, bid=0.80, ask=1.00, delta=+0.20, abs_delta=0.20, is_body=False, is_otm=True),
        _quote_row("put",   93, bid=0.50, ask=0.70, delta=-0.20, abs_delta=0.20, is_body=False, is_otm=True),
    ]
    return OptionSurfaceDB(meta_df, pd.DataFrame(rows))


def _make_ironfly_db_no_wing_below_target() -> OptionSurfaceDB:
    """Surface where the only OTM wings have abs_delta above any reasonable target.
    Used to verify _choose_below_nearest raises when the threshold is not met."""
    meta_df = pd.DataFrame([{
        "ticker":              TICKER,
        "entry_date":          pd.Timestamp(ENTRY_DATE),
        "expiry_date":         pd.Timestamp(EXPIRY_DATE),
        "surface_valid":       True,
        "failure_reason":      None,
        "entry_spot":          ENTRY_SPOT,
        "body_strike":         BODY_STRIKE,
        "exit_spot":           ENTRY_SPOT,
        "spot_move_pct":       0.0,
        "realized_volatility": 0.20,
    }])
    rows = [
        _quote_row("call", 100, bid=3.00, ask=3.40, delta=+0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        _quote_row("put",  100, bid=2.80, ask=3.20, delta=-0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        # High-delta OTM wings — both exceed target=0.25
        _quote_row("call", 103, bid=2.00, ask=2.20, delta=+0.38, abs_delta=0.38, is_body=False, is_otm=True),
        _quote_row("put",   97, bid=1.90, ask=2.10, delta=-0.38, abs_delta=0.38, is_body=False, is_otm=True),
    ]
    return OptionSurfaceDB(meta_df, pd.DataFrame(rows))


class TestIronFlyBelowNearestSemantics:
    """Verify that wings are selected as the highest abs_delta <= target (a ceiling),
    not simply the closest abs_delta to the target (which can overshoot the ceiling).

    Key difference from _choose_nearest:
      _choose_nearest       — picks the smallest |abs_delta - target|, may select a wing
                              that is closer to the money than intended.
      _choose_below_nearest — filters abs_delta <= target first, then picks the highest
                              (closest to threshold from below).  Options above the
                              target are never selected.
    """

    def test_wing_above_delta_ceiling_not_selected(self):
        """Given call candidates @103 (abs_delta=0.28, distance=0.03) and
        @107 (abs_delta=0.20, distance=0.05) with target=0.25:
        _choose_nearest would pick @103 (smaller raw distance);
        _choose_below_nearest must pick @107 (only option with abs_delta <= 0.25)."""
        db  = _make_ironfly_db_above_below_target()
        fly = build_ironfly_from_surface(
            db, TICKER, ENTRY_DATE,
            wing_target_delta=0.25,
            fill=FillAssumption.mid(),
        )
        assert fly.diagnostics["long_call_strike"] == pytest.approx(107.0)
        assert fly.diagnostics["long_put_strike"]  == pytest.approx(93.0)
        assert fly.diagnostics["actual_call_abs_delta"] == pytest.approx(0.20)
        assert fly.diagnostics["actual_put_abs_delta"]  == pytest.approx(0.20)

    def test_no_wing_below_target_raises(self):
        """When every OTM wing has abs_delta > wing_target_delta, ValueError is raised
        with a message that includes the threshold value."""
        db = _make_ironfly_db_no_wing_below_target()
        with pytest.raises(ValueError, match="No quotes with abs_delta"):
            build_ironfly_from_surface(
                db, TICKER, ENTRY_DATE,
                wing_target_delta=0.25,
            )


# =============================================================================
# Below-nearest wing selection semantics
# =============================================================================

def _make_ironfly_db_above_below_target() -> OptionSurfaceDB:
    """Surface with two OTM options per side: one with abs_delta just above the target
    and one below.  Used to verify that _choose_below_nearest rejects the above-target
    option even when it is 'nearer' to the target by raw delta distance.

    Call side:
      @103  abs_delta=0.28  (above target=0.25, raw distance=0.03 — closer by distance)
      @107  abs_delta=0.20  (below target=0.25, raw distance=0.05 — farther by distance)
    Put side (symmetric):
      @97   abs_delta=0.28
      @93   abs_delta=0.20
    """
    meta_df = pd.DataFrame([{
        "ticker":              TICKER,
        "entry_date":          pd.Timestamp(ENTRY_DATE),
        "expiry_date":         pd.Timestamp(EXPIRY_DATE),
        "surface_valid":       True,
        "failure_reason":      None,
        "entry_spot":          ENTRY_SPOT,
        "body_strike":         BODY_STRIKE,
        "exit_spot":           ENTRY_SPOT,
        "spot_move_pct":       0.0,
        "realized_volatility": 0.20,
    }])
    rows = [
        _quote_row("call", 100, bid=3.00, ask=3.40, delta=+0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        _quote_row("put",  100, bid=2.80, ask=3.20, delta=-0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        # Above-target wings (closer to money, 'nearer' by raw delta distance — must NOT be selected)
        _quote_row("call", 103, bid=1.70, ask=1.90, delta=+0.28, abs_delta=0.28, is_body=False, is_otm=True),
        _quote_row("put",   97, bid=1.60, ask=1.80, delta=-0.28, abs_delta=0.28, is_body=False, is_otm=True),
        # Below-target wings (further OTM, abs_delta < target — must be selected)
        _quote_row("call", 107, bid=0.80, ask=1.00, delta=+0.20, abs_delta=0.20, is_body=False, is_otm=True),
        _quote_row("put",   93, bid=0.50, ask=0.70, delta=-0.20, abs_delta=0.20, is_body=False, is_otm=True),
    ]
    return OptionSurfaceDB(meta_df, pd.DataFrame(rows))


def _make_ironfly_db_no_wing_below_target() -> OptionSurfaceDB:
    """Surface where the only OTM wings have abs_delta above any reasonable target.
    Used to verify _choose_below_nearest raises when the threshold is not met."""
    meta_df = pd.DataFrame([{
        "ticker":              TICKER,
        "entry_date":          pd.Timestamp(ENTRY_DATE),
        "expiry_date":         pd.Timestamp(EXPIRY_DATE),
        "surface_valid":       True,
        "failure_reason":      None,
        "entry_spot":          ENTRY_SPOT,
        "body_strike":         BODY_STRIKE,
        "exit_spot":           ENTRY_SPOT,
        "spot_move_pct":       0.0,
        "realized_volatility": 0.20,
    }])
    rows = [
        _quote_row("call", 100, bid=3.00, ask=3.40, delta=+0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        _quote_row("put",  100, bid=2.80, ask=3.20, delta=-0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        # High-delta OTM wings — both exceed target=0.25
        _quote_row("call", 103, bid=2.00, ask=2.20, delta=+0.38, abs_delta=0.38, is_body=False, is_otm=True),
        _quote_row("put",   97, bid=1.90, ask=2.10, delta=-0.38, abs_delta=0.38, is_body=False, is_otm=True),
    ]
    return OptionSurfaceDB(meta_df, pd.DataFrame(rows))


class TestIronFlyBelowNearestSemantics:
    """Verify that wings are selected as the highest abs_delta ≤ target (a ceiling),
    not simply the closest abs_delta to the target (which can overshoot the ceiling).

    Key difference from _choose_nearest:
      _choose_nearest  — picks the smallest |abs_delta - target|, may select a wing
                          that is closer to the money than intended.
      _choose_below_nearest — filters abs_delta ≤ target first, then picks the
                          highest (closest to threshold from below).  Options above
                          the target are never selected.
    """

    def test_wing_above_delta_ceiling_not_selected(self):
        """Given call candidates @103 (abs_delta=0.28, distance=0.03) and
        @107 (abs_delta=0.20, distance=0.05) with target=0.25:
        _choose_nearest would pick @103 (smaller raw distance);
        _choose_below_nearest must pick @107 (only option with abs_delta ≤ 0.25)."""
        db  = _make_ironfly_db_above_below_target()
        fly = build_ironfly_from_surface(
            db, TICKER, ENTRY_DATE,
            wing_target_delta=0.25,
            fill=FillAssumption.mid(),
        )
        assert fly.diagnostics["long_call_strike"] == pytest.approx(107.0)
        assert fly.diagnostics["long_put_strike"]  == pytest.approx(93.0)
        assert fly.diagnostics["actual_call_abs_delta"] == pytest.approx(0.20)
        assert fly.diagnostics["actual_put_abs_delta"]  == pytest.approx(0.20)

    def test_no_wing_below_target_raises(self):
        """When every OTM wing has abs_delta > wing_target_delta, ValueError is raised
        with a message that includes the threshold value."""
        db = _make_ironfly_db_no_wing_below_target()
        with pytest.raises(ValueError, match="No quotes with abs_delta"):
            build_ironfly_from_surface(
                db, TICKER, ENTRY_DATE,
                wing_target_delta=0.25,
            )
