"""
Unit tests for build_ironcondor_from_surface.

Synthetic surface layout
------------------------
Body strike = 100 (is_body=True, also a short-leg candidate for the condor)

  Leg               Side   Strike   bid    ask    mid    abs_delta  is_body  is_otm
  ─────────────────────────────────────────────────────────────────────────────────
  ATM call          call   100     3.00   3.40   3.20   0.50       True     False
  ATM put           put    100     2.80   3.20   3.00   0.50       True     False
  Near OTM call     call   105     1.40   1.60   1.50   0.30       False    True
  Near OTM put      put     95     1.50   1.70   1.60   0.30       False    True
  Far OTM call      call   110     0.60   0.80   0.70   0.15       False    True
  Far OTM put       put     90     0.70   0.90   0.80   0.15       False    True

Default condor: short_delta=0.30, long_delta=0.15
  _choose_nearest(calls, 0.30) → 105 (abs_delta distance=0, exact)
  _choose_nearest(puts,  0.30) → 95  (exact)
  long calls > 105:  only 110 → long call at 110
  long puts  < 95:   only 90  → long put at 90

Iron condor legs (mid fill):
  +1 long put  @  90  mid=0.80   → +0.80
  -1 short put @  95  mid=1.60   → -1.60
  -1 short call@ 105  mid=1.50   → -1.50
  +1 long call @ 110  mid=0.70   → +0.70
  ──────────────────────────────────
  entry_cost (mid)   = 0.80 - 1.60 - 1.50 + 0.70 = -1.60  (credit)
  net_credit         = 1.60
  call_spread_width  = 110 - 105 = 5
  put_spread_width   = 95  - 90  = 5
  max_width          = 5
  max_loss_per_share = 5 - 1.60 = 3.40
  return_on_max_loss = 1.60 / 3.40 ≈ 0.47059
  total_leg_spread   = 0.20 + 0.20 + 0.20 + 0.20 = 0.80
  leg_spread/credit  = 0.80 / 1.60 = 0.50

Cross fill (buy at ask, sell at bid):
  +1 long put  @  90  ask=0.90   → +0.90
  -1 short put @  95  bid=1.50   → -1.50
  -1 short call@ 105  bid=1.40   → -1.40
  +1 long call @ 110  ask=0.80   → +0.80
  entry_cost (cross) = 0.90 - 1.50 - 1.40 + 0.80 = -1.20
  net_credit (cross) = 1.20
  spread_cost        = -1.20 - (-1.60) = 0.40
  spread_cost_ratio  = 0.40 / 1.20 ≈ 0.33333
"""

import pytest
import pandas as pd
from datetime import date
from decimal import Decimal

from src.backtest.option_surface import (
    OptionSurfaceDB,
    FillAssumption,
    build_ironcondor_from_surface,
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
# Synthetic surface helpers
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
        side=side, strike=float(strike),
        bid=bid, ask=ask, mid=mid,
        iv=0.22, delta=delta, gamma=0.03, vega=0.08, theta=-0.02,
        volume=400, open_interest=1500,
        abs_delta=abs_delta,
        spread_pct=(ask - bid) / mid if mid > 0 else 0.0,
        is_body=is_body, is_otm=is_otm,
    )


def _meta_row() -> dict:
    return {
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
    }


def _make_condor_db() -> OptionSurfaceDB:
    """Full six-quote surface (body + two OTM levels per side)."""
    rows = [
        _quote_row("call", 100, bid=3.00, ask=3.40, delta=+0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        _quote_row("put",  100, bid=2.80, ask=3.20, delta=-0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        _quote_row("call", 105, bid=1.40, ask=1.60, delta=+0.30, abs_delta=0.30, is_body=False, is_otm=True),
        _quote_row("put",   95, bid=1.50, ask=1.70, delta=-0.30, abs_delta=0.30, is_body=False, is_otm=True),
        _quote_row("call", 110, bid=0.60, ask=0.80, delta=+0.15, abs_delta=0.15, is_body=False, is_otm=True),
        _quote_row("put",   90, bid=0.70, ask=0.90, delta=-0.15, abs_delta=0.15, is_body=False, is_otm=True),
    ]
    return OptionSurfaceDB(pd.DataFrame([_meta_row()]), pd.DataFrame(rows))


def _make_minimal_condor_db() -> OptionSurfaceDB:
    """Only body + one OTM level per side — no further-OTM wing candidates."""
    rows = [
        _quote_row("call", 100, bid=3.00, ask=3.40, delta=+0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        _quote_row("put",  100, bid=2.80, ask=3.20, delta=-0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        _quote_row("call", 105, bid=1.40, ask=1.60, delta=+0.30, abs_delta=0.30, is_body=False, is_otm=True),
        _quote_row("put",   95, bid=1.50, ask=1.70, delta=-0.30, abs_delta=0.30, is_body=False, is_otm=True),
    ]
    return OptionSurfaceDB(pd.DataFrame([_meta_row()]), pd.DataFrame(rows))


# =============================================================================
# Happy path — mid fill
# =============================================================================

class TestIronCondorMidFill:
    """Symmetric 5-point spreads, mid fill: full hand-calculated verification."""

    def setup_method(self):
        self.db     = _make_condor_db()
        self.condor = build_ironcondor_from_surface(
            self.db, TICKER, ENTRY_DATE,
            short_delta_target=0.30,
            long_delta_target=0.15,
            fill=FillAssumption.mid(),
        )

    def test_strategy_name(self):
        assert self.condor.strategy_name == "iron_condor"

    def test_entry_cost_is_credit(self):
        assert self.condor.entry_cost < 0

    def test_entry_cost_value(self):
        # 0.80 - 1.60 - 1.50 + 0.70 = -1.60
        assert float(self.condor.entry_cost) == pytest.approx(-1.60)

    def test_net_credit(self):
        assert float(self.condor.net_credit) == pytest.approx(1.60)

    def test_max_loss_per_share(self):
        # max_width=5, net_credit=1.60 → max_loss=3.40
        assert float(self.condor.max_loss_per_share) == pytest.approx(3.40)

    def test_return_on_max_loss(self):
        # 1.60 / 3.40 ≈ 0.47059
        assert self.condor.return_on_max_loss == pytest.approx(1.60 / 3.40, rel=1e-4)

    def test_spread_cost_zero_at_mid(self):
        assert float(self.condor.spread_cost) == pytest.approx(0.0)

    def test_total_leg_spread(self):
        # each leg spread = 0.20 × 4 legs = 0.80
        assert float(self.condor.total_leg_spread) == pytest.approx(0.80)

    def test_leg_spread_to_credit_ratio(self):
        # 0.80 / 1.60 = 0.50
        assert self.condor.leg_spread_to_credit_ratio == pytest.approx(0.50)

    def test_four_legs(self):
        assert len(self.condor.strategy.legs) == 4

    def test_leg_signs(self):
        quantities = sorted(leg.quantity for leg in self.condor.strategy.legs)
        assert quantities == [-1, -1, 1, 1]

    def test_short_call_strike_in_diagnostics(self):
        assert self.condor.diagnostics["short_call_strike"] == pytest.approx(105.0)

    def test_short_put_strike_in_diagnostics(self):
        assert self.condor.diagnostics["short_put_strike"] == pytest.approx(95.0)

    def test_long_call_strike_in_diagnostics(self):
        assert self.condor.diagnostics["long_call_strike"] == pytest.approx(110.0)

    def test_long_put_strike_in_diagnostics(self):
        assert self.condor.diagnostics["long_put_strike"] == pytest.approx(90.0)

    def test_call_spread_width_in_diagnostics(self):
        assert self.condor.diagnostics["call_spread_width"] == pytest.approx(5.0)

    def test_put_spread_width_in_diagnostics(self):
        assert self.condor.diagnostics["put_spread_width"] == pytest.approx(5.0)


# =============================================================================
# Cross fill
# =============================================================================

class TestIronCondorCrossFill:
    """Cross fill degrades credit vs mid."""

    def setup_method(self):
        self.db     = _make_condor_db()
        self.condor = build_ironcondor_from_surface(
            self.db, TICKER, ENTRY_DATE,
            short_delta_target=0.30,
            long_delta_target=0.15,
            fill=FillAssumption.cross(),
        )

    def test_entry_cost_at_cross(self):
        # 0.90 - 1.50 - 1.40 + 0.80 = -1.20
        assert float(self.condor.entry_cost) == pytest.approx(-1.20)

    def test_net_credit_at_cross(self):
        assert float(self.condor.net_credit) == pytest.approx(1.20)

    def test_spread_cost(self):
        # -1.20 - (-1.60) = 0.40
        assert float(self.condor.spread_cost) == pytest.approx(0.40)

    def test_spread_cost_ratio(self):
        # 0.40 / 1.20 ≈ 0.33333
        assert self.condor.spread_cost_ratio == pytest.approx(0.40 / 1.20, rel=1e-4)

    def test_total_leg_spread_fill_agnostic(self):
        # total_leg_spread is always computed from bid/ask, not fill prices
        assert float(self.condor.total_leg_spread) == pytest.approx(0.80)


# =============================================================================
# Wing selection — delta targeting
# =============================================================================

class TestIronCondorWingSelection:
    """Verify that delta targeting correctly resolves short and long legs."""

    def test_short_legs_at_nearest_delta_to_target(self):
        # ATM call has abs_delta=0.50; near OTM call at 105 has 0.30.
        # short_delta_target=0.30 → distance(105)=0 vs distance(100)=0.20 → picks 105.
        db     = _make_condor_db()
        condor = build_ironcondor_from_surface(
            db, TICKER, ENTRY_DATE,
            short_delta_target=0.30, long_delta_target=0.15,
        )
        assert condor.diagnostics["short_call_strike"] == pytest.approx(105.0)
        assert condor.diagnostics["short_put_strike"]  == pytest.approx(95.0)

    def test_body_strike_selected_when_short_delta_targets_atm(self):
        # short_delta_target=0.50 → ATM (abs_delta=0.50) wins over 105 (0.30)
        db     = _make_condor_db()
        condor = build_ironcondor_from_surface(
            db, TICKER, ENTRY_DATE,
            short_delta_target=0.50, long_delta_target=0.15,
        )
        assert condor.diagnostics["short_call_strike"] == pytest.approx(100.0)

    def test_per_leg_delta_override_changes_one_side(self):
        # Override only the long call delta to 0.20 — should still find 110
        # (abs_delta=0.15, distance=0.05) vs no other option → 110 selected.
        db     = _make_condor_db()
        condor = build_ironcondor_from_surface(
            db, TICKER, ENTRY_DATE,
            short_delta_target=0.30, long_delta_target=0.15,
            long_call_delta_target=0.20,
        )
        # Only 110 (abs_delta=0.15) is available beyond 105 → must still pick 110
        assert condor.diagnostics["long_call_strike"] == pytest.approx(110.0)


# =============================================================================
# Settle at expiry — P&L verification
# =============================================================================

class TestIronCondorSettle:
    """
    Payoff diagram for the condor (short put@95 / long put@90 / short call@105 / long call@110):

      Spot < 90    : max loss region (put spread fully in-the-money)
      90 ≤ spot ≤ 95: between long and short put (partial loss, put side)
      95 < spot < 105: inside the tent → keep full credit
      105 ≤ spot ≤ 110: between short and long call (partial loss, call side)
      spot > 110   : max loss region (call spread fully in-the-money)
    """

    def setup_method(self):
        self.db     = _make_condor_db()
        self.condor = build_ironcondor_from_surface(
            self.db, TICKER, ENTRY_DATE,
            short_delta_target=0.30, long_delta_target=0.15,
            fill=FillAssumption.mid(),
        )

    def test_settle_inside_tent_max_profit(self):
        # spot=100: all legs OTM → exit_value=0 → pnl=+1.60
        pos = self.condor.settle(Decimal("100"))
        assert float(pos.exit_value) == pytest.approx(0.0)
        assert float(pos.pnl) == pytest.approx(1.60)

    def test_settle_between_short_and_long_call(self):
        # spot=107: short call = -(107-105)=-2; rest=0
        # exit_value=-2; pnl=-2-(-1.60)=-0.40
        pos = self.condor.settle(Decimal("107"))
        assert float(pos.exit_value) == pytest.approx(-2.0)
        assert float(pos.pnl) == pytest.approx(-0.40)

    def test_settle_beyond_long_call_max_loss(self):
        # spot=113: short call=-(113-105)=-8; long call=+(113-110)=+3
        # exit_value=-5; pnl=-5-(-1.60)=-3.40
        pos = self.condor.settle(Decimal("113"))
        assert float(pos.exit_value) == pytest.approx(-5.0)
        assert float(pos.pnl) == pytest.approx(-3.40)

    def test_settle_between_short_and_long_put(self):
        # spot=93: short put=-(95-93)=-2; rest=0
        # Wait: short put qty=-1, payoff = -1 * max(95-93,0) = -2
        # exit_value=-2; pnl=-2-(-1.60)=-0.40
        pos = self.condor.settle(Decimal("93"))
        assert float(pos.exit_value) == pytest.approx(-2.0)
        assert float(pos.pnl) == pytest.approx(-0.40)

    def test_settle_beyond_long_put_max_loss(self):
        # spot=87: short put=-(95-87)=-8; long put=+(90-87)=+3
        # exit_value=-5; pnl=-5-(-1.60)=-3.40
        pos = self.condor.settle(Decimal("87"))
        assert float(pos.exit_value) == pytest.approx(-5.0)
        assert float(pos.pnl) == pytest.approx(-3.40)

    def test_max_loss_matches_settle(self):
        # Confirm that max_loss_per_share equals the realised loss at max-loss spot
        pos = self.condor.settle(Decimal("113"))
        assert float(self.condor.max_loss_per_share) == pytest.approx(abs(float(pos.pnl)), rel=1e-4)


# =============================================================================
# Guard clauses and filters
# =============================================================================

class TestIronCondorFilters:

    def test_no_further_otm_call_raises(self):
        # Minimal surface has only one OTM call at 105.
        # short_delta_target=0.30 → short call at 105; no further OTM call → raises.
        db = _make_minimal_condor_db()
        with pytest.raises(ValueError, match="No further-OTM call wing"):
            build_ironcondor_from_surface(
                db, TICKER, ENTRY_DATE,
                short_delta_target=0.30, long_delta_target=0.15,
            )

    def test_no_further_otm_put_raises(self):
        # Asymmetric minimal surface: include far OTM call but not far OTM put.
        rows = [
            _quote_row("call", 100, bid=3.00, ask=3.40, delta=+0.50, abs_delta=0.50, is_body=True,  is_otm=False),
            _quote_row("put",  100, bid=2.80, ask=3.20, delta=-0.50, abs_delta=0.50, is_body=True,  is_otm=False),
            _quote_row("call", 105, bid=1.40, ask=1.60, delta=+0.30, abs_delta=0.30, is_body=False, is_otm=True),
            _quote_row("put",   95, bid=1.50, ask=1.70, delta=-0.30, abs_delta=0.30, is_body=False, is_otm=True),
            _quote_row("call", 110, bid=0.60, ask=0.80, delta=+0.15, abs_delta=0.15, is_body=False, is_otm=True),
            # No far OTM put
        ]
        db = OptionSurfaceDB(pd.DataFrame([_meta_row()]), pd.DataFrame(rows))
        with pytest.raises(ValueError, match="No further-OTM put wing"):
            build_ironcondor_from_surface(
                db, TICKER, ENTRY_DATE,
                short_delta_target=0.30, long_delta_target=0.15,
            )

    def test_max_leg_spread_pct_drops_illiquid_wings(self):
        # OTM call spread_pct = 0.20/1.50 ≈ 13.3 %; threshold 5 % → no candidates
        db = _make_condor_db()
        with pytest.raises(ValueError, match="No eligible quotes"):
            build_ironcondor_from_surface(
                db, TICKER, ENTRY_DATE,
                short_delta_target=0.30, long_delta_target=0.15,
                max_leg_spread_pct=0.05,
            )

    def test_max_spread_cost_ratio_raises_when_exceeded(self):
        # Cross fill ratio ≈ 33.3 %; threshold 10 % should raise
        db = _make_condor_db()
        with pytest.raises(ValueError, match="spread_cost_ratio"):
            build_ironcondor_from_surface(
                db, TICKER, ENTRY_DATE,
                short_delta_target=0.30, long_delta_target=0.15,
                fill=FillAssumption.cross(),
                max_spread_cost_ratio=0.10,
            )

    def test_max_spread_cost_ratio_passes_for_mid_fill(self):
        # Mid fill has spread_cost=0 → always passes
        db = _make_condor_db()
        result = build_ironcondor_from_surface(
            db, TICKER, ENTRY_DATE,
            short_delta_target=0.30, long_delta_target=0.15,
            fill=FillAssumption.mid(),
            max_spread_cost_ratio=0.01,
        )
        assert result is not None

    def test_missing_ticker_raises(self):
        db = _make_condor_db()
        with pytest.raises(KeyError):
            build_ironcondor_from_surface(
                db, "NOPE", ENTRY_DATE,
                short_delta_target=0.30, long_delta_target=0.15,
            )


# =============================================================================
# Below-nearest long-leg selection semantics
# =============================================================================

def _make_condor_db_above_below_target() -> OptionSurfaceDB:
    """Condor surface with two long-leg candidates per side: one with abs_delta just
    above the long_delta_target and one below.

    Short legs: call@105 (abs_delta=0.30), put@95 (abs_delta=0.30)
    Long leg candidates beyond the short strikes:
      call@108  abs_delta=0.17  (above target=0.15, raw distance=0.02 — 'nearer')
      call@112  abs_delta=0.12  (below target=0.15, raw distance=0.03)
      put@92    abs_delta=0.17  (above target)
      put@88    abs_delta=0.12  (below target)

    _choose_nearest would pick @108/@92 (distance 0.02 < 0.03).
    _choose_below_nearest must pick @112/@88 (highest abs_delta <= 0.15).
    """
    rows = [
        _quote_row("call", 100, bid=3.00, ask=3.40, delta=+0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        _quote_row("put",  100, bid=2.80, ask=3.20, delta=-0.50, abs_delta=0.50, is_body=True,  is_otm=False),
        _quote_row("call", 105, bid=1.40, ask=1.60, delta=+0.30, abs_delta=0.30, is_body=False, is_otm=True),
        _quote_row("put",   95, bid=1.50, ask=1.70, delta=-0.30, abs_delta=0.30, is_body=False, is_otm=True),
        # Above-target long legs (nearer by raw distance — must NOT be selected)
        _quote_row("call", 108, bid=0.80, ask=1.00, delta=+0.17, abs_delta=0.17, is_body=False, is_otm=True),
        _quote_row("put",   92, bid=0.70, ask=0.90, delta=-0.17, abs_delta=0.17, is_body=False, is_otm=True),
        # Below-target long legs (farther by raw distance — must be selected)
        _quote_row("call", 112, bid=0.40, ask=0.60, delta=+0.12, abs_delta=0.12, is_body=False, is_otm=True),
        _quote_row("put",   88, bid=0.30, ask=0.50, delta=-0.12, abs_delta=0.12, is_body=False, is_otm=True),
    ]
    return OptionSurfaceDB(pd.DataFrame([_meta_row()]), pd.DataFrame(rows))


class TestIronCondorBelowNearestSemantics:
    """Verify that long-leg selection uses abs_delta <= long_delta_target as a ceiling.

    Short legs are placed with _choose_nearest (unchanged — the target is a centre, not
    a cap).  Long legs use _choose_below_nearest: only candidates with abs_delta <=
    long_delta_target are eligible, and the one closest to the threshold from below
    is chosen.  A long leg with abs_delta above the target is never selected, even if
    it is 'nearer' by raw delta distance.
    """

    def test_long_leg_above_delta_threshold_not_selected(self):
        """Given long call candidates @108 (abs_delta=0.17, above target=0.15,
        distance=0.02) and @112 (abs_delta=0.12, below, distance=0.03):
        _choose_below_nearest must pick @112, not @108."""
        db     = _make_condor_db_above_below_target()
        condor = build_ironcondor_from_surface(
            db, TICKER, ENTRY_DATE,
            short_delta_target=0.30, long_delta_target=0.15,
        )
        assert condor.diagnostics["long_call_strike"] == pytest.approx(112.0)
        assert condor.diagnostics["long_put_strike"]  == pytest.approx(88.0)
        assert condor.diagnostics["actual_long_call_abs_delta"] == pytest.approx(0.12)
        assert condor.diagnostics["actual_long_put_abs_delta"]  == pytest.approx(0.12)

    def test_long_leg_raises_when_no_candidate_below_threshold(self):
        """When all further-OTM call candidates have abs_delta > long_delta_target,
        ValueError is raised with a message that includes the threshold value."""
        db = _make_condor_db()
        # long_delta_target=0.05 is below all available candidates (abs_delta=0.15)
        with pytest.raises(ValueError, match="No quotes with abs_delta"):
            build_ironcondor_from_surface(
                db, TICKER, ENTRY_DATE,
                short_delta_target=0.30, long_delta_target=0.05,
            )
