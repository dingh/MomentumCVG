"""Synthetic Sprint 009 D0 checks. Does not read official artifacts."""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.backtest.sprint009_d0_body_wing_readiness import (
    FORBIDDEN_REPORT_KEYS,
    MATCHED_COLUMNS,
    build_matched_rows,
    classify_short_calendar,
    reconstructed_leg_economics,
    signed_entry_cash,
    unsigned_intrinsic,
    window_label,
)

DAY = date(2024, 1, 5)
DEV = date(2021, 6, 4)


def _legs(quantity: float = -2.0, *, ask_bump: float = 0.0, payoff_bump: float = 0.0, swap_wings: bool = False, exit_spot: float = 95.0) -> pd.DataFrame:
    specs = [
        (0, "put", 1, 90.0, 0.40, 0.60),
        (1, "put", -1, 100.0, 2.00, 2.20),
        (2, "call", -1, 100.0, 2.10, 2.30),
        (3, "call", 1, 112.0, 0.30, 0.50),
    ]
    if swap_wings:
        specs[0] = (0, "call", 1, 90.0, 0.40, 0.60)
    rows = []
    for index, option_type, unit, strike, bid, ask in specs:
        ask = ask + ask_bump
        fill = ask if unit > 0 else bid
        entry = fill * abs(unit) if unit > 0 else -fill * abs(unit)
        intrinsic = max(strike - exit_spot, 0.0) if option_type == "put" else max(exit_spot - strike, 0.0)
        payoff = intrinsic * unit + payoff_bump
        pnl_per = payoff - entry
        rows.append(
            {
                "trade_date": DAY,
                "ticker": "AAA",
                "direction": "short",
                "expiry_date": date(2024, 1, 12),
                "option_type": option_type,
                "strike": strike,
                "leg_index": index,
                "unit_quantity": unit,
                "portfolio_quantity": abs(quantity) * unit,
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2,
                "fill_price": fill,
                "entry_cash_per_unit": entry,
                "expiry_payoff_per_unit": payoff,
                "pnl_per_unit": pnl_per,
                "pnl_total_leg": abs(quantity) * pnl_per,
                "exit_spot": exit_spot,
                "included_in_portfolio": True,
                "fill_label": "cross",
            }
        )
    return pd.DataFrame(rows)


def _trade(quantity: float = -2.0, pnl: float | None = None, legs: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = legs if legs is not None else _legs(quantity)
    official = float(frame["pnl_total_leg"].sum()) if pnl is None else pnl
    return pd.DataFrame(
        [
            {
                "trade_date": DAY,
                "ticker": "AAA",
                "direction": "short",
                "included_in_portfolio": True,
                "instrument_type": "iron_fly",
                "expiry_date": date(2024, 1, 12),
                "entry_spot": 101.0,
                "exit_spot": 95.0,
                "body_strike": 100.0,
                "quantity": quantity,
                "capital_at_risk_dollars": 250.0,
                "pnl_total": official,
                "fill_label": "cross",
                "long_put_strike": 90.0,
                "long_call_strike": 112.0,
            }
        ]
    )


def _mid_trade(quantity: float = -4.0) -> pd.DataFrame:
    frame = _trade(quantity)
    frame["fill_label"] = "mid"
    return frame


def _mid_legs(cross_legs: pd.DataFrame) -> pd.DataFrame:
    frame = cross_legs.copy()
    frame["fill_label"] = "mid"
    return frame


def test_cash_signs_and_independent_settlement() -> None:
    sold = signed_entry_cash(2.0, -1)
    bought = signed_entry_cash(0.5, 1)
    assert sold == -2.0
    assert bought == 0.5
    assert unsigned_intrinsic("put", 90.0, 100.0) == 0.0
    assert unsigned_intrinsic("put", 110.0, 100.0) == 10.0
    rebuilt = reconstructed_leg_economics(
        option_type="put",
        strike=100.0,
        unit_quantity=-1,
        bid=2.0,
        ask=2.2,
        exit_spot=100.0,
        quantity_magnitude=2.0,
    )
    assert rebuilt["entry_cash_per_unit"] < 0
    assert rebuilt["fill_price"] == 2.0
    assert rebuilt["expiry_payoff_per_unit"] == 0.0
    assert rebuilt["pnl_total_leg"] == 2.0 * (0.0 - rebuilt["entry_cash_per_unit"])


def test_quantity_is_share_equivalent_not_times_100() -> None:
    legs = _legs(-2.0)
    wrong = abs(-2.0) * 100 * -1
    assert float(legs.loc[legs["leg_index"] == 1, "portfolio_quantity"].iloc[0]) == -2.0
    assert wrong != -2.0
    matched, _, issues = build_matched_rows(_trade(-2.0, legs=legs), legs, _mid_trade(), _mid_legs(legs))
    assert matched.iloc[0]["Q"] == 2.0
    assert not any("portfolio_quantity" in item for item in issues)


def test_swapped_wing_keeps_row() -> None:
    legs = _legs(swap_wings=True)
    matched, _, issues = build_matched_rows(_trade(legs=legs), legs, _mid_trade(), _mid_legs(legs))
    assert len(matched) == 1
    assert any("option_type" in item for item in issues)


def test_reconciliation_failure_is_a_gate_input() -> None:
    legs = _legs()
    trade = _trade(pnl=999.0, legs=legs)
    matched, _, issues = build_matched_rows(trade, legs, _mid_trade(), _mid_legs(legs))
    assert len(matched) == 1
    assert abs(matched.iloc[0]["residual_legs_vs_official"]) > 0.01
    assert any("official P&L" in item for item in issues)


def test_missing_and_extra_mid_keys_are_not_inner_joined_away() -> None:
    legs = _legs()
    extra = _mid_trade()
    extra["ticker"] = "BBB"
    extra_legs = _mid_legs(legs)
    extra_legs["ticker"] = "BBB"
    matched, pairing, _issues = build_matched_rows(
        _trade(legs=legs),
        legs,
        pd.concat([extra], ignore_index=True),
        extra_legs,
    )
    assert len(matched) == 1
    assert matched.iloc[0]["ticker"] == "AAA"
    assert bool(matched.iloc[0]["pairing_ok"]) is False
    assert pairing["unmatched_mid"][0][1] == "BBB"
    assert matched.iloc[0]["quantity_mid_abs"] is None


def test_quote_mismatch_keeps_cross_row() -> None:
    legs = _legs()
    mid_legs = _mid_legs(legs)
    mid_legs.loc[mid_legs["leg_index"] == 0, "bid"] = 9.0
    matched, pairing, _issues = build_matched_rows(_trade(legs=legs), legs, _mid_trade(), mid_legs)
    assert len(matched) == 1
    assert bool(matched.iloc[0]["pairing_ok"]) is False
    assert "AAA" not in {key[1] for key in pairing["unmatched_mid"]}


def test_long_only_and_missing_funnel() -> None:
    matched = pd.DataFrame(columns=list(MATCHED_COLUMNS))
    status = pd.DataFrame(
        [
            {"trade_date": DEV, "status": "traded", "reason": "ok"},
            {"trade_date": DAY, "status": "traded", "reason": "ok"},
        ]
    )
    funnel = pd.DataFrame(
        [
            {
                "trade_date": DEV,
                "n_included": 1,
                "n_included_long": 1,
                "n_included_short": 0,
                "date_status": "traded",
                "date_reason": "ok",
            }
        ]
    )
    calendar, issues = classify_short_calendar(status, funnel, matched)
    by_day = {row.trade_date: row for row in calendar.itertuples()}
    assert by_day[DEV].short_book_class == "verified_zero_short"
    assert by_day[DAY].short_book_class == "blocked"
    assert "null" not in by_day[DAY].blocker_reason
    assert "missing funnel" in by_day[DAY].blocker_reason
    assert any(DAY.isoformat() in item or str(DAY) in item for item in issues)


def test_null_short_count_is_blocked_not_cash() -> None:
    status = pd.DataFrame([{"trade_date": DEV, "status": "traded", "reason": "ok"}])
    funnel = pd.DataFrame(
        [
            {
                "trade_date": DEV,
                "n_included": 0,
                "n_included_long": 0,
                "n_included_short": float("nan"),
                "date_status": "traded",
                "date_reason": "ok",
            }
        ]
    )
    calendar, _issues = classify_short_calendar(status, funnel, pd.DataFrame(columns=list(MATCHED_COLUMNS)))
    assert calendar.iloc[0]["short_book_class"] == "blocked"
    assert calendar.iloc[0]["blocker_reason"] == "null n_included_short"


def test_times_100_quantity_fails_and_row_is_kept() -> None:
    legs = _legs(-2.0)
    legs["portfolio_quantity"] = legs["portfolio_quantity"] * 100
    matched, _, issues = build_matched_rows(_trade(-2.0, legs=legs), legs, _mid_trade(), _mid_legs(legs))
    assert len(matched) == 1
    assert any("portfolio_quantity" in item for item in issues)


def test_non_body_strike_and_duplicate_leg_keep_row() -> None:
    legs = _legs()
    trade = _trade(legs=legs)
    trade["body_strike"] = 99.0
    matched, _, issues = build_matched_rows(trade, legs, _mid_trade(), _mid_legs(legs))
    assert len(matched) == 1
    assert any("not body_strike" in item for item in issues)
    dup = pd.concat([legs, legs.iloc[[0]]], ignore_index=True)
    matched_dup, _, dup_issues = build_matched_rows(_trade(legs=legs), dup, _mid_trade(), _mid_legs(legs))
    assert len(matched_dup) == 1
    assert any("duplicate leg key" in item or "expected 4 legs" in item for item in dup_issues)


def test_paired_settlement_mismatch_keeps_cross_row() -> None:
    legs = _legs()
    mid_legs = _mid_legs(legs)
    mid_legs.loc[mid_legs["leg_index"] == 1, "expiry_payoff_per_unit"] = 99.0
    matched, pairing, _issues = build_matched_rows(_trade(legs=legs), legs, _mid_trade(), mid_legs)
    assert len(matched) == 1
    assert matched.iloc[0]["ticker"] == "AAA"
    assert bool(matched.iloc[0]["pairing_ok"]) is False
    assert "expiry_payoff" in matched.iloc[0]["pairing_reason"]
    assert pairing["unmatched_mid"] == []


def test_saved_row_contract_and_later_period_storage() -> None:
    legs = _legs()
    matched, _, issues = build_matched_rows(_trade(legs=legs), legs, _mid_trade(), _mid_legs(legs))
    assert not issues
    row = matched.iloc[0]
    for column in MATCHED_COLUMNS:
        assert column in matched.columns
    assert row["window_label"] == "later_period"
    assert window_label(DEV) == "development"
    assert row["Q"] == 2.0
    assert row["entry_spot"] == 101.0
    assert row["capital_at_risk_dollars"] == 250.0
    assert row["put_wing_entry_cash_per_unit"] > 0
    assert row["body_put_entry_cash_per_unit"] < 0
    assert row["pnl_mid_at_cross_q"] is not None
    assert row["residual_legs_vs_official"] == 0.0 or abs(row["residual_legs_vs_official"]) < 1e-6
    assert not set(FORBIDDEN_REPORT_KEYS) & set(matched.columns)
