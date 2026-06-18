"""
Contract: S5 — step5_select_and_size SELECT + SIZE + SIMULATE.

Scope (Sprint 003):
- Phase 1 (SELECT): per-side cap + rank — Deliverable 2 / Phase 2.
- Phase 2 (SIZE): Tier A (conceptual) + Tier B (integer_lots) — Phase 3.
- Phase 3 (SIMULATE): S7 settle, M1–M3, pnl_total, capital_at_risk_dollars — Phase 4.

See docs/surface_engine_data_contract.md § S5 and
docs/surface_engine_portfolio_metrics_design.md § S5.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.backtest.option_surface import FillAssumption

from src.backtest.pipeline import (
    EXCLUSION_EARNINGS,
    EXCLUSION_INVALID_MAX_LOSS,
    EXCLUSION_MAX_LOSS_EXCEEDS_FAIR_SHARE,
    EXCLUSION_MAX_NAMES_CAP,
    EXCLUSION_NO_SHORT_CREDIT,
    EXCLUSION_NO_STRUCTURE,
    EXCLUSION_PREMIUM_EXCEEDS_FAIR_SHARE,
    step5_select_and_size,
)
from tests.contract.conftest import make_contract_config


# ---------------------------------------------------------------------------
# Synthetic S4-output builder (step5 is a pure function on S4 output, so we
# hand-build the post-S3+S4 frame for fully auditable cap / rank cases).
# ---------------------------------------------------------------------------

def _s4_row(
    ticker: str,
    direction: str,
    signal_rank_pct: float,
    *,
    structure_ok: bool = True,
    had_earnings_nearby: bool = False,
    max_loss_per_share: float = 2.0,
    instrument_type: str = "short_ironfly",
    net_credit_per_share: float = 2.0,
    entry_cost_per_share: float = 8.0,
    body_credit_per_share: float | None = None,
    exit_spot: float = 100.0,
    assembly=None,
) -> dict:
    row = dict(
        ticker=ticker,
        direction=direction,
        signal_score=signal_rank_pct,
        signal_rank_pct=signal_rank_pct,
        cvg_score=1.0,
        cvg_rank_pct=1.0,
        structure_ok=structure_ok,
        had_earnings_nearby=had_earnings_nearby,
        instrument_type=instrument_type,
        max_loss_per_share=max_loss_per_share,
        net_credit_per_share=net_credit_per_share,
        entry_cost_per_share=entry_cost_per_share,
        exit_spot=exit_spot,
    )
    if body_credit_per_share is not None:
        row["body_credit_per_share"] = body_credit_per_share
    if assembly is not None:
        row["_assembly"] = assembly
    return row


def _mock_assembly(pnl_per_share: float):
    """Minimal S7 stand-in returning a fixed per-share settle PnL."""

    class _Assembly:
        def settle(self, exit_spot=None, exit_date=None):
            return SimpleNamespace(pnl=pnl_per_share)

    return _Assembly()


def _s4_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Schema / annotation
# ---------------------------------------------------------------------------

def test_adds_inclusion_columns_on_every_row():
    structures = _s4_frame(
        [
            _s4_row("L1", "long", 0.95),
            _s4_row("S1", "short", 0.05),
        ]
    )
    out = step5_select_and_size(None, structures, make_contract_config())
    assert "included_in_portfolio" in out.columns
    assert "exclusion_reason" in out.columns
    assert len(out) == len(structures)
    # Every row is annotated (no NaN inclusion flags).
    assert out["included_in_portfolio"].notna().all()


def test_preserves_input_columns():
    structures = _s4_frame([_s4_row("L1", "long", 0.95)])
    out = step5_select_and_size(None, structures, make_contract_config())
    for col in structures.columns:
        assert col in out.columns


def test_pure_function_does_not_mutate_input():
    structures = _s4_frame([_s4_row("L1", "long", 0.95), _s4_row("S1", "short", 0.05)])
    snapshot = structures.copy(deep=True)
    step5_select_and_size(None, structures, make_contract_config())
    pd.testing.assert_frame_equal(structures, snapshot)


def test_empty_structures_returns_empty_with_columns():
    out = step5_select_and_size(None, pd.DataFrame(), make_contract_config())
    assert out.empty
    assert "included_in_portfolio" in out.columns
    assert "exclusion_reason" in out.columns
    assert "quantity" in out.columns
    assert "sizing_mode" in out.columns


def _long_premium_spend(row: pd.Series) -> float:
    """Dollar premium from quantity in share-equivalent units (Tier A and Tier B)."""
    return abs(float(row["quantity"])) * float(row["entry_cost_per_share"])


def _short_credit_collected(row: pd.Series) -> float:
    return abs(float(row["quantity"])) * float(row["net_credit_per_share"])


def _short_max_loss_deployed(row: pd.Series) -> float:
    return abs(float(row["quantity"])) * float(row["max_loss_per_share"])


def _tier_b_integer_lots_config(**overrides):
    defaults = dict(
        sizing_mode="integer_lots",
        tier_a_mode=None,
        tier_a_short_budget=None,
        tier_a_long_budget=None,
        tier_b_short_max_loss_budget=10_000.0,
        contract_multiplier=100.0,
    )
    defaults.update(overrides)
    return make_contract_config(**defaults)


# ---------------------------------------------------------------------------
# Eligibility (reads S3/S4 flags; does not re-run upstream filters)
# ---------------------------------------------------------------------------

def test_eligible_rows_included():
    structures = _s4_frame(
        [
            _s4_row("L1", "long", 0.95),
            _s4_row("S1", "short", 0.05),
        ]
    )
    out = step5_select_and_size(None, structures, make_contract_config())
    for tkr in ("L1", "S1"):
        row = out[out["ticker"] == tkr].iloc[0]
        assert bool(row["included_in_portfolio"]) is True
        assert row["exclusion_reason"] is None


def test_no_tradeable_structure_excluded():
    structures = _s4_frame(
        [
            _s4_row("L1", "long", 0.95),
            _s4_row("BAD", "short", 0.05, structure_ok=False, max_loss_per_share=float("nan")),
        ]
    )
    out = step5_select_and_size(None, structures, make_contract_config())
    bad = out[out["ticker"] == "BAD"].iloc[0]
    assert bool(bad["included_in_portfolio"]) is False
    assert bad["exclusion_reason"] == EXCLUSION_NO_STRUCTURE


def test_earnings_exclusion():
    structures = _s4_frame(
        [
            _s4_row("L1", "long", 0.95),
            _s4_row("S1", "short", 0.05, had_earnings_nearby=True),
        ]
    )
    out = step5_select_and_size(None, structures, make_contract_config())
    short = out[out["ticker"] == "S1"].iloc[0]
    assert bool(short["included_in_portfolio"]) is False
    assert short["exclusion_reason"] == EXCLUSION_EARNINGS


def test_no_structure_takes_priority_over_earnings():
    # A row that is both structure_ok==False AND earnings-flagged → no_tradeable_structure.
    structures = _s4_frame(
        [
            _s4_row(
                "BAD", "short", 0.05,
                structure_ok=False, had_earnings_nearby=True,
                max_loss_per_share=float("nan"),
            ),
        ]
    )
    out = step5_select_and_size(None, structures, make_contract_config())
    bad = out.iloc[0]
    assert bad["exclusion_reason"] == EXCLUSION_NO_STRUCTURE


# ---------------------------------------------------------------------------
# Per-side cap + rank (decision 003: independent long / short pools)
# ---------------------------------------------------------------------------

def test_per_side_cap_honored_independently():
    # 3 longs + 3 shorts, cap = 2 per side ⇒ 4 included (cap is per side, not global).
    structures = _s4_frame(
        [
            _s4_row("L_hi", "long", 0.90),
            _s4_row("L_mid", "long", 0.80),
            _s4_row("L_lo", "long", 0.70),
            _s4_row("S_lo", "short", 0.10),
            _s4_row("S_mid", "short", 0.20),
            _s4_row("S_hi", "short", 0.30),
        ]
    )
    cfg = make_contract_config(max_names_per_side=2)
    out = step5_select_and_size(None, structures, cfg)

    included = set(out.loc[out["included_in_portfolio"], "ticker"])
    assert included == {"L_hi", "L_mid", "S_lo", "S_mid"}
    assert int(out["included_in_portfolio"].sum()) == 4


def test_overflow_marked_max_names_cap():
    structures = _s4_frame(
        [
            _s4_row("L_hi", "long", 0.90),
            _s4_row("L_mid", "long", 0.80),
            _s4_row("L_lo", "long", 0.70),
            _s4_row("S_lo", "short", 0.10),
            _s4_row("S_mid", "short", 0.20),
            _s4_row("S_hi", "short", 0.30),
        ]
    )
    cfg = make_contract_config(max_names_per_side=2)
    out = step5_select_and_size(None, structures, cfg)

    # Long overflow = lowest rank; short overflow = highest rank.
    assert out.loc[out["ticker"] == "L_lo", "exclusion_reason"].iloc[0] == EXCLUSION_MAX_NAMES_CAP
    assert out.loc[out["ticker"] == "S_hi", "exclusion_reason"].iloc[0] == EXCLUSION_MAX_NAMES_CAP


def test_rank_direction_long_descending_short_ascending():
    # cap = 1 ⇒ keep best signal per side: long = highest rank, short = lowest rank.
    structures = _s4_frame(
        [
            _s4_row("L_hi", "long", 0.90),
            _s4_row("L_lo", "long", 0.60),
            _s4_row("S_lo", "short", 0.10),
            _s4_row("S_hi", "short", 0.40),
        ]
    )
    cfg = make_contract_config(max_names_per_side=1)
    out = step5_select_and_size(None, structures, cfg)

    included = set(out.loc[out["included_in_portfolio"], "ticker"])
    assert included == {"L_hi", "S_lo"}


def test_excluded_rows_not_counted_against_cap():
    # An earnings-excluded long should not consume a long slot; the next eligible
    # long is still included under cap = 1.
    structures = _s4_frame(
        [
            _s4_row("L_top_earn", "long", 0.95, had_earnings_nearby=True),
            _s4_row("L_next", "long", 0.85),
        ]
    )
    cfg = make_contract_config(max_names_per_side=1)
    out = step5_select_and_size(None, structures, cfg)

    assert out.loc[out["ticker"] == "L_top_earn", "exclusion_reason"].iloc[0] == EXCLUSION_EARNINGS
    assert bool(out.loc[out["ticker"] == "L_next", "included_in_portfolio"].iloc[0]) is True


# ---------------------------------------------------------------------------
# invalid_max_loss (sizing-eligibility reject at the selection boundary)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [0.0, -1.0, None, float("nan")])
def test_invalid_max_loss_excludes_selected_row(bad_value):
    structures = _s4_frame([_s4_row("S1", "short", 0.05, max_loss_per_share=bad_value)])
    out = step5_select_and_size(None, structures, make_contract_config())
    row = out.iloc[0]
    assert bool(row["included_in_portfolio"]) is False
    assert row["exclusion_reason"] == EXCLUSION_INVALID_MAX_LOSS


def test_long_straddle_positive_premium_not_invalid():
    # Long straddle carries max_loss_per_share = premium paid (> 0) ⇒ stays included.
    structures = _s4_frame(
        [_s4_row("L1", "long", 0.95, instrument_type="long_straddle", max_loss_per_share=4.0)]
    )
    out = step5_select_and_size(None, structures, make_contract_config())
    row = out.iloc[0]
    assert bool(row["included_in_portfolio"]) is True
    assert row["exclusion_reason"] is None


# ---------------------------------------------------------------------------
# Diagnostics toggle
# ---------------------------------------------------------------------------

def test_include_diagnostics_false_drops_excluded():
    structures = _s4_frame(
        [
            _s4_row("L1", "long", 0.95),
            _s4_row("BAD", "short", 0.05, structure_ok=False, max_loss_per_share=float("nan")),
        ]
    )
    cfg = make_contract_config(include_diagnostics=False)
    out = step5_select_and_size(None, structures, cfg)
    assert set(out["ticker"]) == {"L1"}
    assert out["included_in_portfolio"].all()


def test_include_diagnostics_true_keeps_excluded():
    structures = _s4_frame(
        [
            _s4_row("L1", "long", 0.95),
            _s4_row("BAD", "short", 0.05, structure_ok=False, max_loss_per_share=float("nan")),
        ]
    )
    cfg = make_contract_config(include_diagnostics=True)
    out = step5_select_and_size(None, structures, cfg)
    assert set(out["ticker"]) == {"L1", "BAD"}


# ---------------------------------------------------------------------------
# S5 Phase 2 — SIZE (Tier A conceptual + Tier B integer_lots)
# ---------------------------------------------------------------------------

def test_tier_a_equal_premium_divides_side_budget_by_included_count():
    # 2 shorts, tier_a_short_budget=10_000, credit=$2/sh → per-name $5_000 → 2_500 units each.
    structures = _s4_frame(
        [
            _s4_row("S1", "short", 0.05, net_credit_per_share=2.0),
            _s4_row("S2", "short", 0.10, net_credit_per_share=2.0),
        ]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_premium",
        tier_a_short_budget=10_000.0,
        max_names_per_side=2,
    )
    out = step5_select_and_size(None, structures, cfg)
    for tkr in ("S1", "S2"):
        row = out[out["ticker"] == tkr].iloc[0]
        assert bool(row["included_in_portfolio"]) is True
        assert row["quantity"] == pytest.approx(-2_500.0)
    assert out["sizing_mode"].eq("conceptual").all()


def test_tier_a_does_not_apply_contract_multiplier():
    # Same geometry as above; if ×100 were applied, abs(quantity) would be 25 not 2_500.
    structures = _s4_frame(
        [_s4_row("S1", "short", 0.05, net_credit_per_share=2.0)]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_premium",
        tier_a_short_budget=10_000.0,
        contract_multiplier=100.0,
        max_names_per_side=1,
    )
    out = step5_select_and_size(None, structures, cfg)
    row = out.iloc[0]
    assert abs(float(row["quantity"])) == pytest.approx(5_000.0)


def test_tier_a_equal_max_loss_sizes_short_by_max_loss_denominator():
    structures = _s4_frame(
        [_s4_row("S1", "short", 0.05, max_loss_per_share=4.0, net_credit_per_share=2.0)]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_max_loss",
        tier_a_short_budget=8_000.0,
        max_names_per_side=1,
    )
    out = step5_select_and_size(None, structures, cfg)
    row = out.iloc[0]
    # 8_000 / 1 name / 4.0 max_loss_per_share = 2_000 units (short → negative sign).
    assert row["quantity"] == pytest.approx(-2_000.0)


def test_tier_a_equal_max_loss_long_quantity_from_collected_short_credit():
    # Short: 8_000 / 4.0 max_loss = 2_000 units; credit = 2_000 × $2 = $4_000.
    # Long: 2 names → $4_000 / 2 / $8 premium = 250 units each.
    structures = _s4_frame(
        [
            _s4_row("S1", "short", 0.05, max_loss_per_share=4.0, net_credit_per_share=2.0),
            _s4_row("L1", "long", 0.95, entry_cost_per_share=8.0, max_loss_per_share=8.0),
            _s4_row("L2", "long", 0.85, entry_cost_per_share=8.0, max_loss_per_share=8.0),
        ]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_max_loss",
        tier_a_short_budget=8_000.0,
        tier_a_long_budget=99_999.0,  # ignored when shorts fund longs
        max_names_per_side=2,
    )
    out = step5_select_and_size(None, structures, cfg)
    short_row = out[out["ticker"] == "S1"].iloc[0]
    collected = abs(float(short_row["quantity"])) * float(short_row["net_credit_per_share"])
    assert collected == pytest.approx(4_000.0)
    for tkr in ("L1", "L2"):
        row = out[out["ticker"] == tkr].iloc[0]
        expected = (collected / 2) / 8.0
        assert float(row["quantity"]) == pytest.approx(expected)


def test_tier_a_equal_max_loss_fallback_to_long_budget_without_shorts():
    # No shorts → long_budget = tier_a_long_budget (edge rule).
    structures = _s4_frame(
        [
            _s4_row("L1", "long", 0.95, entry_cost_per_share=10.0, max_loss_per_share=10.0),
            _s4_row("L2", "long", 0.85, entry_cost_per_share=10.0, max_loss_per_share=10.0),
        ]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_max_loss",
        tier_a_short_budget=8_000.0,
        tier_a_long_budget=5_000.0,
        max_names_per_side=2,
    )
    out = step5_select_and_size(None, structures, cfg)
    for tkr in ("L1", "L2"):
        row = out[out["ticker"] == tkr].iloc[0]
        # 5_000 / 2 names / $10 premium = 250 units.
        assert float(row["quantity"]) == pytest.approx(250.0)


def test_tier_a_equal_max_loss_fallback_when_short_credit_zero():
    # Shorts present but collect $0 credit → fall back to tier_a_long_budget.
    structures = _s4_frame(
        [
            _s4_row("S1", "short", 0.05, max_loss_per_share=4.0, net_credit_per_share=0.0),
            _s4_row("L1", "long", 0.95, entry_cost_per_share=10.0, max_loss_per_share=10.0),
        ]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_max_loss",
        tier_a_short_budget=8_000.0,
        tier_a_long_budget=3_000.0,
        max_names_per_side=1,
    )
    out = step5_select_and_size(None, structures, cfg)
    long_row = out[out["ticker"] == "L1"].iloc[0]
    # 3_000 / 1 / $10 = 300 units (not financed from zero short credit).
    assert float(long_row["quantity"]) == pytest.approx(300.0)


def test_tier_a_equal_premium_long_side_divides_budget_by_included_count():
    # 2 longs, tier_a_long_budget=10_000, premium=$8/sh → 5_000 / 8 = 625 units each.
    structures = _s4_frame(
        [
            _s4_row("L1", "long", 0.95, entry_cost_per_share=8.0, max_loss_per_share=8.0),
            _s4_row("L2", "long", 0.85, entry_cost_per_share=8.0, max_loss_per_share=8.0),
        ]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_premium",
        tier_a_short_budget=10_000.0,
        tier_a_long_budget=10_000.0,
        max_names_per_side=2,
    )
    out = step5_select_and_size(None, structures, cfg)
    for tkr in ("L1", "L2"):
        row = out[out["ticker"] == tkr].iloc[0]
        assert float(row["quantity"]) == pytest.approx(625.0)


def test_tier_b_short_sizes_from_total_max_loss_fair_share():
    # 1 short, tier_b_short_max_loss_budget=600, max_loss=$2/sh → floor(600/200)=3 contracts.
    structures = _s4_frame(
        [_s4_row("S1", "short", 0.05, max_loss_per_share=2.0)]
    )
    cfg = _tier_b_integer_lots_config(tier_b_short_max_loss_budget=600.0, max_names_per_side=1)
    out = step5_select_and_size(None, structures, cfg)
    row = out.iloc[0]
    assert row["quantity"] == -300.0
    assert _short_max_loss_deployed(row) <= cfg.tier_b_short_max_loss_budget


def test_tier_b_quantity_is_share_equivalent_units():
    """Tier B quantity = contracts × 100; quantity/100 is always integer."""
    structures = _s4_frame(
        [
            _s4_row("S1", "short", 0.05, max_loss_per_share=2.0, net_credit_per_share=5.0),
            _s4_row(
                "L1", "long", 0.95,
                entry_cost_per_share=8.0,
                max_loss_per_share=8.0,
            ),
        ]
    )
    cfg = _tier_b_integer_lots_config(tier_b_short_max_loss_budget=1_000.0)
    out = step5_select_and_size(None, structures, cfg)
    mult = cfg.contract_multiplier
    for _, row in out[out["included_in_portfolio"]].iterrows():
        qty = float(row["quantity"])
        assert qty % mult == 0
        assert int(abs(qty) / mult) >= 1


def test_tier_b_short_quantity_is_integer_floor():
    # floor(550/200)=2, not ceil(2.75)=3.
    structures = _s4_frame(
        [_s4_row("S1", "short", 0.05, max_loss_per_share=2.0)]
    )
    cfg = _tier_b_integer_lots_config(tier_b_short_max_loss_budget=550.0, max_names_per_side=1)
    out = step5_select_and_size(None, structures, cfg)
    assert out.iloc[0]["quantity"] == -200.0


def test_tier_b_quantity_sign_long_positive_short_negative():
    # Short budget $1,000 → 5 contracts; credit = 5 × $5 × 100 = $2,500 finances 3 long lots.
    structures = _s4_frame(
        [
            _s4_row(
                "L1", "long", 0.95,
                instrument_type="long_straddle",
                max_loss_per_share=8.0,
                entry_cost_per_share=8.0,
            ),
            _s4_row(
                "S1", "short", 0.05,
                max_loss_per_share=2.0,
                net_credit_per_share=5.0,
            ),
        ]
    )
    cfg = _tier_b_integer_lots_config(tier_b_short_max_loss_budget=1_000.0)
    out = step5_select_and_size(None, structures, cfg)
    long_row = out[out["ticker"] == "L1"].iloc[0]
    short_row = out[out["ticker"] == "S1"].iloc[0]
    assert float(short_row["quantity"]) == -500.0
    assert float(long_row["quantity"]) == 300.0
    assert _long_premium_spend(long_row) <= _short_credit_collected(short_row)


def test_select_excluded_rows_do_not_receive_active_sizing():
    structures = _s4_frame(
        [
            _s4_row("S_best", "short", 0.05),
            _s4_row("S_overflow", "short", 0.30),
        ]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_premium",
        tier_a_short_budget=10_000.0,
        max_names_per_side=1,
        include_diagnostics=True,
    )
    out = step5_select_and_size(None, structures, cfg)
    overflow = out[out["ticker"] == "S_overflow"].iloc[0]
    assert bool(overflow["included_in_portfolio"]) is False
    assert overflow["exclusion_reason"] == EXCLUSION_MAX_NAMES_CAP
    assert pd.isna(overflow["quantity"])


def test_deployable_capital_does_not_cap_tier_b_long_budget():
    # Long budget = collected short credit only; deployable_capital is ignored in S5 Tier B.
    structures = _s4_frame(
        [
            _s4_row("S1", "short", 0.05, max_loss_per_share=2.0, net_credit_per_share=2.0),
            _s4_row("S2", "short", 0.10, max_loss_per_share=2.0, net_credit_per_share=2.0),
            _s4_row(
                "L1", "long", 0.95,
                instrument_type="long_straddle",
                entry_cost_per_share=5.0,
                max_loss_per_share=5.0,
            ),
        ]
    )
    cfg = _tier_b_integer_lots_config(
        tier_b_short_max_loss_budget=4_000.0,
        deployable_capital=600.0,
        max_names_per_side=2,
        include_diagnostics=True,
    )
    out = step5_select_and_size(None, structures, cfg)
    shorts = out[out["direction"] == "short"]
    assert shorts["included_in_portfolio"].all()
    total_credit = sum(_short_credit_collected(row) for _, row in shorts.iterrows())
    assert total_credit == pytest.approx(4_000.0)

    long_row = out[out["ticker"] == "L1"].iloc[0]
    assert bool(long_row["included_in_portfolio"]) is True
    # fair_share = $4,000 → floor(4000/500) = 8 contracts, not capped at $600.
    assert float(long_row["quantity"]) == 800.0
    assert _long_premium_spend(long_row) <= total_credit


def test_tier_b_longs_financed_from_short_credit():
    # 1 short: budget $1,000 → 5 contracts × $5 × 100 = $2,500 credit; 2 longs at $4/sh.
    structures = _s4_frame(
        [
            _s4_row("S1", "short", 0.05, max_loss_per_share=2.0, net_credit_per_share=5.0),
            _s4_row(
                "L1", "long", 0.95,
                instrument_type="long_straddle",
                entry_cost_per_share=4.0,
                max_loss_per_share=4.0,
            ),
            _s4_row(
                "L2", "long", 0.85,
                instrument_type="long_straddle",
                entry_cost_per_share=4.0,
                max_loss_per_share=4.0,
            ),
        ]
    )
    cfg = _tier_b_integer_lots_config(
        tier_b_short_max_loss_budget=1_000.0,
        max_names_per_side=2,
    )
    out = step5_select_and_size(None, structures, cfg)
    short_row = out[out["ticker"] == "S1"].iloc[0]
    credit = _short_credit_collected(short_row)
    longs = out[(out["direction"] == "long") & out["included_in_portfolio"]]
    total_long_spend = sum(_long_premium_spend(row) for _, row in longs.iterrows())
    assert total_long_spend <= credit
    assert len(longs) == 2
    assert (longs["quantity"] == 300.0).all()


def test_tier_b_max_loss_exceeds_fair_share_excludes_short():
    # Budget $6,000; 3 shorts → initial fair_share $2,000; S_expensive needs $2,500/contract.
    structures = _s4_frame(
        [
            _s4_row("S_a", "short", 0.05, max_loss_per_share=2.0, net_credit_per_share=2.0),
            _s4_row("S_b", "short", 0.10, max_loss_per_share=2.0, net_credit_per_share=2.0),
            _s4_row(
                "S_expensive", "short", 0.30,
                max_loss_per_share=25.0,
                net_credit_per_share=10.0,
            ),
        ]
    )
    cfg = _tier_b_integer_lots_config(
        tier_b_short_max_loss_budget=6_000.0,
        max_names_per_side=3,
        include_diagnostics=True,
    )
    out = step5_select_and_size(None, structures, cfg)
    expensive = out[out["ticker"] == "S_expensive"].iloc[0]
    assert bool(expensive["included_in_portfolio"]) is False
    assert expensive["exclusion_reason"] == EXCLUSION_MAX_LOSS_EXCEEDS_FAIR_SHARE
    survivors = out[(out["direction"] == "short") & out["included_in_portfolio"]]
    assert len(survivors) == 2
    total_risk = sum(_short_max_loss_deployed(row) for _, row in survivors.iterrows())
    assert total_risk <= cfg.tier_b_short_max_loss_budget


def test_tier_b_fair_share_drops_worst_offender_one_at_a_time():
    # Budget $6,000 across 3 shorts: $200, $2,500, $4,000 per contract.
    # Bulk filter would keep only S_cheap (fair_share $2,000). Worst-first drops
    # S_worst first → fair_share rises to $3,000 → S_mid ($2,500) also fits.
    structures = _s4_frame(
        [
            _s4_row("S_cheap", "short", 0.05, max_loss_per_share=2.0),
            _s4_row("S_mid", "short", 0.10, max_loss_per_share=25.0),
            _s4_row("S_worst", "short", 0.30, max_loss_per_share=40.0),
        ]
    )
    cfg = _tier_b_integer_lots_config(
        tier_b_short_max_loss_budget=6_000.0,
        max_names_per_side=3,
        include_diagnostics=True,
    )
    out = step5_select_and_size(None, structures, cfg)
    included = set(out.loc[out["included_in_portfolio"], "ticker"])
    assert included == {"S_cheap", "S_mid"}
    worst = out[out["ticker"] == "S_worst"].iloc[0]
    assert bool(worst["included_in_portfolio"]) is False
    assert worst["exclusion_reason"] == EXCLUSION_MAX_LOSS_EXCEEDS_FAIR_SHARE


def test_tier_b_premium_exceeds_fair_share_excludes_long():
    # Collected credit = $10,000 (10 short contracts × $10/sh × 100).
    # 5 longs → initial fair_share $2,000; L_expensive needs $3,000/contract → dropped.
    structures = _s4_frame(
        [
            _s4_row(
                "S1", "short", 0.05,
                max_loss_per_share=2.0,
                net_credit_per_share=10.0,
            ),
            _s4_row("L_a", "long", 0.95, entry_cost_per_share=12.0, max_loss_per_share=12.0),
            _s4_row("L_b", "long", 0.90, entry_cost_per_share=15.0, max_loss_per_share=15.0),
            _s4_row("L_c", "long", 0.85, entry_cost_per_share=18.0, max_loss_per_share=18.0),
            _s4_row("L_d", "long", 0.80, entry_cost_per_share=20.0, max_loss_per_share=20.0),
            _s4_row(
                "L_expensive", "long", 0.75,
                entry_cost_per_share=30.0,
                max_loss_per_share=30.0,
            ),
        ]
    )
    cfg = _tier_b_integer_lots_config(
        tier_b_short_max_loss_budget=2_500.0,
        max_names_per_side=5,
        include_diagnostics=True,
    )
    out = step5_select_and_size(None, structures, cfg)
    expensive = out[out["ticker"] == "L_expensive"].iloc[0]
    assert bool(expensive["included_in_portfolio"]) is False
    assert expensive["exclusion_reason"] == EXCLUSION_PREMIUM_EXCEEDS_FAIR_SHARE
    survivors = out[(out["direction"] == "long") & out["included_in_portfolio"]]
    assert len(survivors) == 4


def test_tier_b_short_only_when_no_long_fits():
    structures = _s4_frame(
        [
            _s4_row("S1", "short", 0.05, max_loss_per_share=2.0, net_credit_per_share=2.0),
            _s4_row(
                "L_costly", "long", 0.95,
                entry_cost_per_share=50.0,
                max_loss_per_share=50.0,
            ),
        ]
    )
    cfg = _tier_b_integer_lots_config(
        tier_b_short_max_loss_budget=400.0,
        include_diagnostics=True,
    )
    out = step5_select_and_size(None, structures, cfg)
    # Short: budget $400 → 2 contracts; credit $400; long needs $5,000/contract → dropped.
    short_row = out[out["ticker"] == "S1"].iloc[0]
    long_row = out[out["ticker"] == "L_costly"].iloc[0]
    assert bool(short_row["included_in_portfolio"]) is True
    assert bool(long_row["included_in_portfolio"]) is False
    assert long_row["exclusion_reason"] == EXCLUSION_PREMIUM_EXCEEDS_FAIR_SHARE


def test_tier_b_no_short_credit_excludes_longs():
    structures = _s4_frame(
        [_s4_row("L1", "long", 0.95, entry_cost_per_share=8.0, max_loss_per_share=8.0)]
    )
    cfg = _tier_b_integer_lots_config(
        tier_b_short_max_loss_budget=400.0,
        include_diagnostics=True,
    )
    out = step5_select_and_size(None, structures, cfg)
    row = out.iloc[0]
    assert bool(row["included_in_portfolio"]) is False
    assert row["exclusion_reason"] == EXCLUSION_NO_SHORT_CREDIT


def test_tier_b_long_only_ignores_short_max_loss_budget():
    """tier_b_short_max_loss_budget is required on config but unused when no shorts."""
    structures = _s4_frame(
        [_s4_row("L1", "long", 0.95, entry_cost_per_share=8.0, max_loss_per_share=8.0)]
    )
    cfg = _tier_b_integer_lots_config(tier_b_short_max_loss_budget=50_000.0)
    out = step5_select_and_size(None, structures, cfg)
    assert out.iloc[0]["exclusion_reason"] == EXCLUSION_NO_SHORT_CREDIT
    assert pd.isna(out.iloc[0]["quantity"])


def test_tier_b_short_budget_below_one_contract_max_loss_excluded():
    # Budget $199 < 1-contract max-loss $200 → cannot fit even one lot.
    structures = _s4_frame(
        [_s4_row("S1", "short", 0.05, max_loss_per_share=2.0)]
    )
    cfg = _tier_b_integer_lots_config(tier_b_short_max_loss_budget=199.0, max_names_per_side=1)
    out = step5_select_and_size(None, structures, cfg)
    row = out.iloc[0]
    assert bool(row["included_in_portfolio"]) is False
    assert row["exclusion_reason"] == EXCLUSION_MAX_LOSS_EXCEEDS_FAIR_SHARE
    assert pd.isna(row["quantity"])


def test_tier_b_deployable_capital_none_matches_ignored_when_set():
    """Long budget = collected credit regardless of deployable_capital (ADR 004)."""
    structures = _s4_frame(
        [
            _s4_row("S1", "short", 0.05, max_loss_per_share=2.0, net_credit_per_share=5.0),
            _s4_row("L1", "long", 0.95, entry_cost_per_share=8.0, max_loss_per_share=8.0),
        ]
    )
    base = dict(tier_b_short_max_loss_budget=1_000.0)
    out_none = step5_select_and_size(
        None, structures, _tier_b_integer_lots_config(deployable_capital=None, **base)
    )
    out_set = step5_select_and_size(
        None, structures, _tier_b_integer_lots_config(deployable_capital=600.0, **base)
    )
    for tkr in ("S1", "L1"):
        assert out_none[out_none["ticker"] == tkr]["quantity"].iloc[0] == (
            out_set[out_set["ticker"] == tkr]["quantity"].iloc[0]
        )


def test_tier_b_max_loss_budget_per_trade_not_echoed_on_rows():
    """Tier B sizing does not use or echo max_loss_budget_per_trade on trade log rows."""
    structures = _s4_frame(
        [_s4_row("S1", "short", 0.05, max_loss_per_share=2.0, net_credit_per_share=2.0)]
    )
    cfg = _tier_b_integer_lots_config(
        tier_b_short_max_loss_budget=600.0,
        max_loss_budget_per_trade=999.0,
    )
    out = step5_select_and_size(None, structures, cfg)
    assert out["max_loss_budget_per_trade"].isna().all()
    # Sizing follows tier_b_short_max_loss_budget, not max_loss_budget_per_trade.
    assert out.iloc[0]["quantity"] == -300.0


# ---------------------------------------------------------------------------
# S5 Phase 3 — SIMULATE (settle + returns)
# ---------------------------------------------------------------------------

SIMULATE_OUTPUT_COLS = [
    "pnl_per_share",
    "pnl_total",
    "capital_at_risk_dollars",
    "return_on_premium",
    "return_on_max_loss",
    "return_on_atm_straddle",
    "fill_label",
]


def test_simulate_adds_output_columns():
    structures = _s4_frame(
        [
            _s4_row(
                "L1", "long", 0.95,
                instrument_type="long_straddle",
                max_loss_per_share=8.0,
                assembly=_mock_assembly(1.0),
            ),
        ]
    )
    out = step5_select_and_size(None, structures, make_contract_config())
    for col in SIMULATE_OUTPUT_COLS:
        assert col in out.columns


def test_tier_a_long_pnl_total_is_abs_quantity_times_pnl_per_share():
    # quantity = +1_250; pnl_per_share = +2.0 → pnl_total = abs(qty) × pnl = 2_500.
    structures = _s4_frame(
        [
            _s4_row(
                "L1", "long", 0.95,
                instrument_type="long_straddle",
                entry_cost_per_share=8.0,
                max_loss_per_share=8.0,
                body_credit_per_share=8.0,
                assembly=_mock_assembly(2.0),
            ),
        ]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_premium",
        tier_a_long_budget=10_000.0,
        contract_multiplier=100.0,
        max_names_per_side=1,
    )
    out = step5_select_and_size(None, structures, cfg)
    row = out.iloc[0]
    assert float(row["quantity"]) == pytest.approx(1_250.0)
    assert float(row["pnl_per_share"]) == pytest.approx(2.0)
    assert float(row["pnl_total"]) == pytest.approx(2_500.0)
    # If ×100 were applied again, pnl_total would be 250_000.
    assert float(row["pnl_total"]) != pytest.approx(250_000.0)


def test_tier_b_short_pnl_total_uses_abs_quantity_not_sign():
    # 3 contracts → quantity = -300 (sign = short); pnl_per_share = +1.0 → +300 profit.
    structures = _s4_frame(
        [
            _s4_row(
                "S1", "short", 0.05,
                instrument_type="iron_fly",
                max_loss_per_share=2.0,
                net_credit_per_share=2.0,
                body_credit_per_share=3.0,
                assembly=_mock_assembly(1.0),
            ),
        ]
    )
    cfg = _tier_b_integer_lots_config(tier_b_short_max_loss_budget=600.0, max_names_per_side=1)
    out = step5_select_and_size(None, structures, cfg)
    row = out.iloc[0]
    assert float(row["quantity"]) == -300.0
    assert float(row["pnl_per_share"]) == pytest.approx(1.0)
    assert float(row["pnl_total"]) == pytest.approx(300.0)
    # Signed qty × pnl would wrongly yield -300; extra ×100 would be 30_000.
    assert float(row["pnl_total"]) != pytest.approx(-300.0)
    assert float(row["pnl_total"]) != pytest.approx(30_000.0)


def test_tier_b_short_losing_pnl_total_is_negative():
    # Losing short: abs(-300) × (-1.0) = -300 (not +300 from sign cancellation).
    structures = _s4_frame(
        [
            _s4_row(
                "S1", "short", 0.05,
                instrument_type="iron_fly",
                max_loss_per_share=2.0,
                net_credit_per_share=2.0,
                body_credit_per_share=3.0,
                assembly=_mock_assembly(-1.0),
            ),
        ]
    )
    cfg = _tier_b_integer_lots_config(tier_b_short_max_loss_budget=600.0, max_names_per_side=1)
    out = step5_select_and_size(None, structures, cfg)
    row = out.iloc[0]
    assert float(row["pnl_total"]) == pytest.approx(-300.0)


def test_simulate_excluded_max_names_cap_has_nan_outputs():
    structures = _s4_frame(
        [
            _s4_row("S_best", "short", 0.05, assembly=_mock_assembly(1.0)),
            _s4_row("S_overflow", "short", 0.30, assembly=_mock_assembly(99.0)),
        ]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_premium",
        tier_a_short_budget=10_000.0,
        max_names_per_side=1,
        include_diagnostics=True,
    )
    out = step5_select_and_size(None, structures, cfg)
    overflow = out[out["ticker"] == "S_overflow"].iloc[0]
    assert overflow["exclusion_reason"] == EXCLUSION_MAX_NAMES_CAP
    assert pd.isna(overflow["pnl_per_share"])
    assert pd.isna(overflow["pnl_total"])
    assert pd.isna(overflow["capital_at_risk_dollars"])
    assert pd.isna(overflow["return_on_premium"])
    assert pd.isna(overflow["fill_label"])


def test_simulate_excluded_no_short_credit_has_nan_outputs():
    structures = _s4_frame(
        [
            _s4_row(
                "L1", "long", 0.95,
                instrument_type="long_straddle",
                entry_cost_per_share=8.0,
                max_loss_per_share=8.0,
                assembly=_mock_assembly(5.0),
            ),
        ]
    )
    cfg = _tier_b_integer_lots_config(include_diagnostics=True)
    out = step5_select_and_size(None, structures, cfg)
    row = out.iloc[0]
    assert row["exclusion_reason"] == EXCLUSION_NO_SHORT_CREDIT
    assert pd.isna(row["pnl_per_share"])
    assert pd.isna(row["pnl_total"])
    assert pd.isna(row["return_on_max_loss"])


def test_capital_at_risk_long_uses_premium_paid():
    # quantity = 500; premium = $8/sh → CAR = 4_000.
    structures = _s4_frame(
        [
            _s4_row(
                "L1", "long", 0.95,
                instrument_type="long_straddle",
                entry_cost_per_share=8.0,
                max_loss_per_share=8.0,
                body_credit_per_share=8.0,
                assembly=_mock_assembly(0.0),
            ),
        ]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_premium",
        tier_a_long_budget=4_000.0,
        max_names_per_side=1,
    )
    out = step5_select_and_size(None, structures, cfg)
    row = out.iloc[0]
    assert float(row["quantity"]) == pytest.approx(500.0)
    assert float(row["capital_at_risk_dollars"]) == pytest.approx(4_000.0)


def test_capital_at_risk_short_defined_risk_uses_max_loss_per_share():
    # quantity = -300 (3 × 100); max_loss = $2/sh → CAR = 600.
    structures = _s4_frame(
        [
            _s4_row(
                "S1", "short", 0.05,
                instrument_type="iron_fly",
                max_loss_per_share=2.0,
                net_credit_per_share=2.0,
                body_credit_per_share=3.0,
                assembly=_mock_assembly(0.0),
            ),
        ]
    )
    cfg = _tier_b_integer_lots_config(tier_b_short_max_loss_budget=600.0, max_names_per_side=1)
    out = step5_select_and_size(None, structures, cfg)
    row = out.iloc[0]
    assert float(row["capital_at_risk_dollars"]) == pytest.approx(600.0)


def test_capital_at_risk_long_and_short_in_same_cycle():
    # Long:  budget $8_000 / $8 premium = 1_000 units → CAR = 1_000 × 8 = 8_000.
    # Short: budget $6_000 / $3 credit = 2_000 units (qty −2_000) → CAR = 2_000 × 2 max_loss = 4_000.
    structures = _s4_frame(
        [
            _s4_row(
                "L1", "long", 0.95,
                instrument_type="long_straddle",
                entry_cost_per_share=8.0,
                max_loss_per_share=8.0,
                body_credit_per_share=8.0,
                assembly=_mock_assembly(0.0),
            ),
            _s4_row(
                "S1", "short", 0.05,
                instrument_type="iron_fly",
                max_loss_per_share=2.0,
                net_credit_per_share=3.0,
                body_credit_per_share=3.0,
                assembly=_mock_assembly(0.0),
            ),
        ]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_premium",
        tier_a_short_budget=6_000.0,
        tier_a_long_budget=8_000.0,
        max_names_per_side=1,
    )
    out = step5_select_and_size(None, structures, cfg)
    long_row = out[out["ticker"] == "L1"].iloc[0]
    short_row = out[out["ticker"] == "S1"].iloc[0]
    assert float(long_row["quantity"]) == pytest.approx(1_000.0)
    assert float(long_row["capital_at_risk_dollars"]) == pytest.approx(8_000.0)
    assert float(short_row["quantity"]) == pytest.approx(-2_000.0)
    assert float(short_row["capital_at_risk_dollars"]) == pytest.approx(4_000.0)


def test_m1_m2_m3_hand_calculated_short_iron_fly():
    # pnl = +1; net credit (M1) = 2; max_loss (M2) = 5; ATM body (M3) = 3.
    structures = _s4_frame(
        [
            _s4_row(
                "S1", "short", 0.05,
                instrument_type="iron_fly",
                max_loss_per_share=5.0,
                net_credit_per_share=2.0,
                body_credit_per_share=3.0,
                assembly=_mock_assembly(1.0),
            ),
        ]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_premium",
        tier_a_short_budget=10_000.0,
        max_names_per_side=1,
    )
    out = step5_select_and_size(None, structures, cfg)
    row = out.iloc[0]
    assert float(row["return_on_premium"]) == pytest.approx(0.5)
    assert float(row["return_on_max_loss"]) == pytest.approx(0.2)
    assert float(row["return_on_atm_straddle"]) == pytest.approx(1.0 / 3.0)


def test_m1_m3_equal_on_long_straddle_m2_is_nan():
    # pnl = +4; premium paid = 8 → M1 = M3 = 0.5; M2 undefined for long straddle.
    structures = _s4_frame(
        [
            _s4_row(
                "L1", "long", 0.95,
                instrument_type="long_straddle",
                entry_cost_per_share=8.0,
                max_loss_per_share=8.0,
                body_credit_per_share=8.0,
                assembly=_mock_assembly(4.0),
            ),
        ]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_premium",
        tier_a_long_budget=8_000.0,
        max_names_per_side=1,
    )
    out = step5_select_and_size(None, structures, cfg)
    row = out.iloc[0]
    assert float(row["return_on_premium"]) == pytest.approx(0.5)
    assert float(row["return_on_atm_straddle"]) == pytest.approx(0.5)
    assert pd.isna(row["return_on_max_loss"])


def test_m1_nan_when_structure_premium_non_positive():
    # equal_max_loss sizes shorts by max_loss, not net credit — row stays included with credit=0.
    structures = _s4_frame(
        [
            _s4_row(
                "S1", "short", 0.05,
                instrument_type="iron_fly",
                max_loss_per_share=5.0,
                net_credit_per_share=0.0,
                body_credit_per_share=3.0,
                assembly=_mock_assembly(1.0),
            ),
        ]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_max_loss",
        tier_a_short_budget=10_000.0,
        max_names_per_side=1,
    )
    out = step5_select_and_size(None, structures, cfg)
    row = out.iloc[0]
    assert bool(row["included_in_portfolio"]) is True
    assert pd.isna(row["return_on_premium"])
    assert float(row["return_on_atm_straddle"]) == pytest.approx(1.0 / 3.0)


def test_m3_nan_when_atm_straddle_premium_non_positive():
    structures = _s4_frame(
        [
            _s4_row(
                "S1", "short", 0.05,
                instrument_type="iron_fly",
                max_loss_per_share=5.0,
                net_credit_per_share=2.0,
                body_credit_per_share=0.0,
                assembly=_mock_assembly(1.0),
            ),
        ]
    )
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_premium",
        tier_a_short_budget=10_000.0,
        max_names_per_side=1,
    )
    out = step5_select_and_size(None, structures, cfg)
    row = out.iloc[0]
    assert float(row["return_on_premium"]) == pytest.approx(0.5)
    assert pd.isna(row["return_on_atm_straddle"])


def test_fill_label_on_included_rows_from_config():
    structures = _s4_frame(
        [
            _s4_row(
                "S1", "short", 0.05,
                instrument_type="iron_fly",
                assembly=_mock_assembly(1.0),
            ),
        ]
    )
    cfg = make_contract_config(fill=FillAssumption.cross())
    out = step5_select_and_size(None, structures, cfg)
    assert out.iloc[0]["fill_label"] == "cross"


def test_fill_label_nan_on_excluded_rows():
    structures = _s4_frame(
        [
            _s4_row("S_best", "short", 0.05, assembly=_mock_assembly(1.0)),
            _s4_row("S_overflow", "short", 0.30, assembly=_mock_assembly(1.0)),
        ]
    )
    cfg = make_contract_config(
        fill=FillAssumption.mid(),
        max_names_per_side=1,
        include_diagnostics=True,
    )
    out = step5_select_and_size(None, structures, cfg)
    included = out[out["ticker"] == "S_best"].iloc[0]
    excluded = out[out["ticker"] == "S_overflow"].iloc[0]
    assert included["fill_label"] == "mid"
    assert pd.isna(excluded["fill_label"])
